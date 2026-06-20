"""
Tests for the bundle pack/install/bind_config/uninstall/list pipeline.
"""

import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from SRToolkit.bundle._manifest import BundleManifest, _sha256
from SRToolkit.bundle._pack import pack
from SRToolkit.bundle._relocate import _relocate_class_paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_files(root: Path, files: dict) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


SIMPLE_CONFIG = {
    "approach_class": "alice.pkg.approach.MyApproach",
    "constraint_class": "SRToolkit.utils.grammar.constraints.MaxDepth",
    "limit": 5,
}

SIMPLE_FILES = {"approach.py": "class MyApproach:\n    pass\n"}


# ---------------------------------------------------------------------------
# BundleManifest
# ---------------------------------------------------------------------------


class TestBundleManifest:
    def test_import_prefix_slug(self):
        m = BundleManifest(name="my-approach", version="1.2.3")
        assert m.import_prefix == "srtk_bundles.my_approach_1_2_3"

    def test_to_from_dict_roundtrip(self):
        m = BundleManifest(
            name="foo",
            version="0.1.0",
            author="alice",
            python_deps=["torch>=2.0"],
            files={"src/a.py": "abc123"},
        )
        assert BundleManifest.from_dict(m.to_dict()) == m

    def test_no_entry_config_field(self):
        m = BundleManifest(name="x", version="0.1")
        assert "entry_config" not in m.to_dict()

    def test_verify_passes(self, tmp_path):
        p = tmp_path / "src" / "a.py"
        p.parent.mkdir()
        p.write_text("class X: pass")
        m = BundleManifest(name="x", version="0.1", files={"src/a.py": _sha256(p)})
        m.verify(tmp_path)  # must not raise

    def test_verify_missing_file_raises(self, tmp_path):
        m = BundleManifest(name="x", version="0.1", files={"src/missing.py": "abc"})
        with pytest.raises(ValueError, match="missing"):
            m.verify(tmp_path)

    def test_verify_checksum_mismatch_raises(self, tmp_path):
        p = tmp_path / "src" / "a.py"
        p.parent.mkdir()
        p.write_text("class X: pass")
        m = BundleManifest(name="x", version="0.1", files={"src/a.py": "wrong_hash"})
        with pytest.raises(ValueError, match="Checksum"):
            m.verify(tmp_path)


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


class TestPack:
    def _src_files(self, tmp_path: Path) -> list[Path]:
        for name, content in SIMPLE_FILES.items():
            (tmp_path / name).write_text(content)
        return [tmp_path / name for name in SIMPLE_FILES]

    def test_creates_zip(self, tmp_path):
        files = self._src_files(tmp_path)
        out = tmp_path / "my.srtk"
        pack(files, out, name="foo", version="0.1.0", author="alice")
        assert out.exists()
        assert zipfile.is_zipfile(out)

    def test_zip_contents(self, tmp_path):
        files = self._src_files(tmp_path)
        out = tmp_path / "my.srtk"
        pack(files, out, name="foo", version="0.1.0")
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "manifest.json" in names
        assert "src/approach.py" in names

    def test_no_config_in_zip(self, tmp_path):
        files = self._src_files(tmp_path)
        out = tmp_path / "my.srtk"
        pack(files, out, name="foo", version="0.1.0")
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
        assert "config.json" not in names

    def test_duplicate_basenames_raise(self, tmp_path):
        a = tmp_path / "dir_a" / "ops.py"
        b = tmp_path / "dir_b" / "ops.py"
        a.parent.mkdir()
        a.write_text("class A: pass")
        b.parent.mkdir()
        b.write_text("class B: pass")
        with pytest.raises(ValueError, match="Duplicate basename"):
            pack([a, b], tmp_path / "out.srtk", name="foo", version="0.1.0")

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pack([tmp_path / "ghost.py"], tmp_path / "out.srtk", name="x", version="0.1")

    def test_non_py_file_raises(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Only .py files"):
            pack([f], tmp_path / "out.srtk", name="x", version="0.1")

    def test_configs_annotated_and_written(self, tmp_path):
        files = self._src_files(tmp_path)
        cfg = tmp_path / "settings.json"
        cfg.write_text(json.dumps({"approach_class": "alice.MyApproach"}))
        pack(files, tmp_path / "out.srtk", name="foo", version="0.2.0", configs=[cfg])
        annotated = json.loads((tmp_path / "settings.srtk.json").read_text())
        assert annotated["_bundle"] == "foo"
        assert annotated["_version"] == "0.2.0"
        assert annotated["approach_class"] == "alice.MyApproach"
        assert json.loads(cfg.read_text()) == {"approach_class": "alice.MyApproach"}

    def test_configs_already_annotated_raises(self, tmp_path):
        files = self._src_files(tmp_path)
        cfg = tmp_path / "settings.json"
        cfg.write_text(json.dumps({"_bundle": "foo", "approach_class": "alice.MyApproach"}))
        with pytest.raises(ValueError, match="already contains"):
            pack(files, tmp_path / "out.srtk", name="foo", version="0.2.0", configs=[cfg])


class TestInspectWithoutInstall:
    """read_manifest / bundle_class_index inspect a .srtk without installing it."""

    def _bundle(self, tmp_path: Path) -> Path:
        f = tmp_path / "approach.py"
        f.write_text("class MyApproach:\n    pass\n\nclass Helper:\n    pass\n")
        out = tmp_path / "b.srtk"
        pack([f], out, name="my-bundle", version="2.0.1")
        return out

    def test_read_manifest_returns_metadata(self, tmp_path):
        from SRToolkit.bundle import read_manifest

        m = read_manifest(self._bundle(tmp_path))
        assert m.name == "my-bundle"
        assert m.version == "2.0.1"
        assert m.import_prefix == "srtk_bundles.my_bundle_2_0_1"

    def test_read_manifest_missing_file_raises(self, tmp_path):
        from SRToolkit.bundle import read_manifest

        with pytest.raises(FileNotFoundError):
            read_manifest(tmp_path / "nope.srtk")

    def test_read_manifest_not_a_bundle_raises(self, tmp_path):
        from SRToolkit.bundle import read_manifest

        bogus = tmp_path / "bogus.srtk"
        with zipfile.ZipFile(bogus, "w") as zf:
            zf.writestr("readme.txt", "no manifest here")
        with pytest.raises(ValueError, match="not a valid .srtk"):
            read_manifest(bogus)

    def test_bundle_class_index_lists_defined_classes(self, tmp_path):
        from SRToolkit.bundle._relocate import bundle_class_index

        index = bundle_class_index(self._bundle(tmp_path))
        assert index == {"MyApproach": ["approach"], "Helper": ["approach"]}


# ---------------------------------------------------------------------------
# install / bind_config / uninstall / list_installed
# ---------------------------------------------------------------------------


class TestInstallUninstall:
    def _make_bundle(self, tmp_path: Path) -> Path:
        approach = tmp_path / "approach.py"
        approach.write_text(SIMPLE_FILES["approach.py"])
        srtk = tmp_path / "foo.srtk"
        pack([approach], srtk, name="test-bundle", version="0.1.0", author="tester")
        return srtk

    def test_full_roundtrip(self, tmp_path, monkeypatch):
        from SRToolkit.bundle import _store, bind_config, install, list_installed, uninstall

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

        srtk = self._make_bundle(tmp_path)

        with patch("SRToolkit.bundle._install._confirm", return_value=True):
            install(srtk)

        entries = list_installed()
        assert any(e["name"] == "test-bundle" and e["version"] == "0.1.0" for e in entries)

        # bind_config with explicit name
        config = bind_config(SIMPLE_CONFIG, "test-bundle", version="0.1.0")
        assert config["approach_class"].startswith("srtk_bundles.")
        assert config["constraint_class"] == "SRToolkit.utils.grammar.constraints.MaxDepth"

        # bind_config reads _bundle/_version from config when not passed explicitly
        annotated = {"_bundle": "test-bundle", "_version": "0.1.0", **SIMPLE_CONFIG}
        config2 = bind_config(annotated)
        assert config2["approach_class"] == config["approach_class"]
        assert "_bundle" not in config2
        assert "_version" not in config2

        uninstall("test-bundle", version="0.1.0")
        assert not any(e["name"] == "test-bundle" for e in list_installed())

    def test_repack_preserves_deps_and_min_version(self, tmp_path, monkeypatch):
        """_repack rebuilds an installed bundle, carrying python_deps + srtk_min_version."""
        from SRToolkit.bundle import _store, install, list_installed
        from SRToolkit.bundle._install import _repack

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

        approach = tmp_path / "approach.py"
        approach.write_text(SIMPLE_FILES["approach.py"])
        srtk = tmp_path / "foo.srtk"
        pack(
            [approach],
            srtk,
            name="dep-bundle",
            version="0.1.0",
            author="tester",
            python_deps=["numpy>=1.0"],
            srtk_min_version="1.2.3",
        )
        with patch("SRToolkit.bundle._install._confirm", return_value=True):
            install(srtk)

        # The index now carries the two fields (so they survive install).
        entry = next(e for e in list_installed() if e["name"] == "dep-bundle")
        assert entry["python_deps"] == ["numpy>=1.0"]
        assert entry["srtk_min_version"] == "1.2.3"

        out = tmp_path / "out" / "dep-bundle.srtk"
        _repack("dep-bundle", "0.1.0", out)
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["python_deps"] == ["numpy>=1.0"]
        assert manifest["srtk_min_version"] == "1.2.3"
        assert "src/approach.py" in manifest["files"]

    def test_repack_unknown_bundle_raises(self, tmp_path, monkeypatch):
        from SRToolkit.bundle import _store
        from SRToolkit.bundle._install import _repack

        monkeypatch.setattr(_store, "bundles_root", lambda: tmp_path / "bundles")
        with pytest.raises(KeyError):
            _repack("nope", None, tmp_path / "x.srtk")

    def test_install_cancelled_on_trust_prompt(self, tmp_path, monkeypatch):
        from SRToolkit.bundle import _store, install, list_installed

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

        srtk = self._make_bundle(tmp_path)

        with patch("SRToolkit.bundle._install._confirm", return_value=False):
            install(srtk)

        assert list_installed() == []

    def test_checksum_mismatch_raises(self, tmp_path, monkeypatch):
        from SRToolkit.bundle import _store, install

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

        srtk = self._make_bundle(tmp_path)

        import shutil as _shutil

        tmp_corrupt = tmp_path / "corrupt.srtk"
        _shutil.copy(srtk, tmp_corrupt)
        with zipfile.ZipFile(tmp_corrupt, "a") as zf:
            zf.writestr("src/approach.py", "class MyApproach:\n    pass  # tampered\n")

        with pytest.raises(ValueError, match="Checksum"):
            install(tmp_corrupt)

    def test_missing_dep_suggests_pip(self, tmp_path, monkeypatch, capsys):
        from SRToolkit.bundle import _store, install

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)

        approach = tmp_path / "approach.py"
        approach.write_text(SIMPLE_FILES["approach.py"])
        srtk = tmp_path / "foo_deps.srtk"
        pack([approach], srtk, name="dep-bundle", version="0.1.0", python_deps=["nonexistent-pkg-xyz"])

        with patch("SRToolkit.bundle._install._confirm", return_value=False):
            install(srtk)

        captured = capsys.readouterr()
        assert "nonexistent-pkg-xyz" in captured.out
        assert "pip install" in captured.out


# ---------------------------------------------------------------------------
# _store.lookup: latest version uses semver ordering not lexicographic
# ---------------------------------------------------------------------------


class TestStoreLookup:
    def test_latest_version_semver_order(self, tmp_path, monkeypatch):
        from SRToolkit.bundle import _store

        bundle_root = tmp_path / "bundles"
        monkeypatch.setattr(_store, "bundles_root", lambda: bundle_root)
        bundle_root.mkdir(parents=True)

        for v in ("0.9.0", "0.10.0", "0.2.0"):
            m = BundleManifest(name="ver-test", version=v)
            path = _store.bundle_path(m.name, m.version)
            path.mkdir(parents=True)
            _store.register(m, path)

        entry = _store.lookup("ver-test")
        assert entry["version"] == "0.10.0"


# ---------------------------------------------------------------------------
# _relocate_class_paths — low-level rewriter (spot-check)
# ---------------------------------------------------------------------------


class TestRelocateIntegration:
    def test_relocate_with_flat_src(self, tmp_path):
        (tmp_path / "constraints.py").write_text("class MyC:\n    pass\n")
        config = {
            "constraint_class": "alice.constraints.MyC",
            "other_class": "SRToolkit.utils.MaxDepth",
        }
        out = _relocate_class_paths(config, "B", tmp_path)
        assert out["constraint_class"] == "B.constraints.MyC"
        assert out["other_class"] == "SRToolkit.utils.MaxDepth"
