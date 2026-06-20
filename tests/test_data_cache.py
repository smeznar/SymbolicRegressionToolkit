"""
Tests for SRToolkit.dataset.data_cache.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from SRToolkit.dataset import data_cache
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOL_LIST = ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "ln", "C"]


def _simple_dataset_config(name: str = "test_ds", benchmark: str = "test_bench", version: str = "1.0.0") -> dict:
    """Return a minimal format_version=2 dataset config for use in tests."""
    return {
        "format_version": 2,
        "dataset_name": name,
        "benchmark": benchmark,
        "version": version,
        "symbol_library": SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 2).to_dict(),
        "ranking_function": "rmse",
        "max_evaluations": 100,
        "success_threshold": 1e-6,
        "original_equation": "y = x0 + x1",
        "seed": 0,
        "dataset_metadata": None,
        "kwargs": {},
        "samplers": [
            {
                "sampler_class": "SRToolkit.dataset.sampling.UniformSampling",
                "min_value": 0,
                "max_value": 1,
                "uses_positive": True,
                "uses_negative": True,
            },
            {
                "sampler_class": "SRToolkit.dataset.sampling.UniformSampling",
                "min_value": 0,
                "max_value": 1,
                "uses_positive": True,
                "uses_negative": True,
            },
        ],
        "ground_truth": ["X_0", "+", "X_1"],
        "data_source": {"source_class": "SRToolkit.dataset.data_source.SampleSource", "n_samples": 20, "seed": 0},
    }


# ---------------------------------------------------------------------------
# data_root
# ---------------------------------------------------------------------------


class TestDataRoot:
    def test_returns_path(self):
        p = data_cache.data_root()
        assert isinstance(p, Path)
        assert p.name == "data"
        assert "SRToolkit" in str(p)


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------


class TestDatasetCachePath:
    def test_basic_path(self):
        p = data_cache.dataset_path("feynman", "1.0.0", "I.16.6")
        assert p.suffix == ".npz"
        assert p.stem == "I.16.6"
        assert "1_0_0" in str(p)
        assert "feynman" in str(p)

    def test_version_slug_replaces_dots(self):
        p = data_cache.dataset_path("bench", "2.3.1", "ds")
        assert "2_3_1" in str(p)

    def test_version_slug_replaces_hyphens(self):
        p = data_cache.dataset_path("bench", "1-0-0", "ds")
        assert "1_0_0" in str(p)


# ---------------------------------------------------------------------------
# resolve — SampleSource
# ---------------------------------------------------------------------------


class TestResolveSample:
    def test_creates_npz_and_sidecar(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        p = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        assert p.exists()
        assert p.suffix == ".npz"
        meta_path = p.with_suffix("").with_suffix(".meta.json")
        assert meta_path.exists()

    def test_cache_hit_no_regen(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        p1 = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        mtime1 = p1.stat().st_mtime
        import time

        time.sleep(0.01)
        p2 = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        assert p1 == p2
        assert p2.stat().st_mtime == mtime1  # not regenerated

    def test_force_regenerates(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        p1 = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        mtime1 = p1.stat().st_mtime
        import time

        time.sleep(0.05)
        p2 = data_cache.resolve("test_bench", "1.0.0", "test_ds", config, force=True)
        assert p2.stat().st_mtime > mtime1

    def test_changed_source_warns(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        config_changed = {**config, "data_source": {**config["data_source"], "n_samples": 9999}}
        with pytest.warns(UserWarning, match="refresh"):
            data_cache.resolve("test_bench", "1.0.0", "test_ds", config_changed)

    def test_no_config_no_cache_raises_fnf(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            data_cache.resolve("test_bench", "1.0.0", "nonexistent")

    def test_no_config_cache_exists_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        p = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        p2 = data_cache.resolve("test_bench", "1.0.0", "test_ds")
        assert p == p2

    def test_data_has_correct_shape(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        p = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        data = np.load(str(p))
        assert "X" in data
        assert data["X"].shape[1] == 2
        assert data["X"].shape[0] == 20  # n_samples


# ---------------------------------------------------------------------------
# resolve — no config with no cache
# ---------------------------------------------------------------------------


class TestResolveNoConfig:
    def test_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            data_cache.resolve("bench", "1.0.0", "ds")


# ---------------------------------------------------------------------------
# import_archive
# ---------------------------------------------------------------------------


class TestImportArchive:
    def _make_archive(self, tmp_path: Path, files: dict) -> Path:
        archive = tmp_path / "test.zip"
        with zipfile.ZipFile(str(archive), "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return archive

    def test_extracts_data_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        # Create a fake npz in memory
        import io

        buf = io.BytesIO()
        np.savez(buf, X=np.array([[1.0, 2.0]]))
        buf.seek(0)
        archive = self._make_archive(
            tmp_path,
            {
                "data/myds.npz": buf.read(),
                "benchmark.json": "{}",
            },
        )
        data_cache.import_archive(archive, "mybench", "1.0.0")
        version_dir = tmp_path / "mybench" / "1_0_0"
        assert (version_dir / "myds.npz").exists()

    def test_ignores_non_data_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        archive = self._make_archive(
            tmp_path,
            {
                "benchmark.json": "{}",
                "manifest.txt": "info",
            },
        )
        data_cache.import_archive(archive, "mybench", "1.0.0")
        version_dir = tmp_path / "mybench" / "1_0_0"
        # version_dir may not even exist if no data/ files
        if version_dir.exists():
            assert not (version_dir / "benchmark.json").exists()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListCached:
    def test_empty_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        assert data_cache.list() == []

    def test_lists_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = _simple_dataset_config()
        data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        entries = data_cache.list()
        assert len(entries) == 1
        e = entries[0]
        assert e["benchmark"] == "test_bench"
        assert e["key"] == "test_ds"
        assert "size_bytes" in e


# ---------------------------------------------------------------------------
# gc
# ---------------------------------------------------------------------------


class TestGC:
    def _make_two_versions(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        cfg1 = _simple_dataset_config(version="1.0.0")
        cfg2 = _simple_dataset_config(version="2.0.0")
        data_cache.resolve("test_bench", "1.0.0", "test_ds", cfg1)
        data_cache.resolve("test_bench", "2.0.0", "test_ds", cfg2)

    def test_gc_removes_old_version(self, tmp_path, monkeypatch):
        self._make_two_versions(tmp_path, monkeypatch)
        removed = data_cache.gc(keep_latest=True)
        assert len(removed) == 1
        # Remaining should be version 2.0.0
        remaining = data_cache.list()
        assert all(e["version"].startswith("2") for e in remaining)

    def test_gc_keep_latest_false_removes_all(self, tmp_path, monkeypatch):
        self._make_two_versions(tmp_path, monkeypatch)
        removed = data_cache.gc(keep_latest=False)
        assert len(removed) == 2
        assert data_cache.list() == []


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    def _populate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        for version in ("1.0.0", "2.0.0"):
            for key in ("I.16.6", "I.16.7"):
                cfg = _simple_dataset_config(name=key, benchmark="feynman", version=version)
                data_cache.resolve("feynman", version, key, cfg)

    def test_remove_single_dataset(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch)
        removed = data_cache.remove("feynman", "1.0.0", "I.16.6")
        # .npz + .meta.json sidecar (SampleSource is hashed)
        names = sorted(p.name for p in removed)
        assert "I.16.6.npz" in names
        remaining = {(e["version"], e["key"]) for e in data_cache.list()}
        assert ("1.0.0", "I.16.6") not in remaining
        assert ("1.0.0", "I.16.7") in remaining

    def test_remove_version_dir(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch)
        data_cache.remove("feynman", "1.0.0")
        remaining = {e["version"] for e in data_cache.list()}
        assert remaining == {"2.0.0"}

    def test_remove_whole_benchmark(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch)
        data_cache.remove("feynman")
        assert data_cache.list() == []

    def test_remove_missing_is_noop(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch)
        assert data_cache.remove("nope") == []
        assert data_cache.remove("feynman", "9.9.9") == []
        assert data_cache.remove("feynman", "1.0.0", "ghost") == []
        # nothing was deleted by the no-ops
        assert len(data_cache.list()) == 4

    def test_meta_path_handles_dotted_keys(self, tmp_path):
        p = data_cache.dataset_path("feynman", "1.0.0", "I.16.6")
        assert data_cache._meta_path(p).name == "I.16.6.meta.json"
        # distinct keys must not collide on the same sidecar
        p2 = data_cache.dataset_path("feynman", "1.0.0", "I.16.7")
        assert data_cache._meta_path(p).name != data_cache._meta_path(p2).name


# ---------------------------------------------------------------------------
# SR_dataset.refresh raises ValueError when data_source is None
# ---------------------------------------------------------------------------


class TestSRDatasetRefresh:
    def test_refresh_null_source_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sl = SymbolLibrary.default_symbols(2)
        ds = SR_dataset(X, sl, benchmark="test", version="1.0.0")
        with pytest.raises(ValueError, match="data_source is null"):
            ds.refresh()

    def test_refresh_null_benchmark_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sl = SymbolLibrary.default_symbols(2)
        ds = SR_dataset(X, sl, benchmark=None, version="1.0.0")
        ds.data_source = SampleSource(n_samples=2, seed=0)
        with pytest.raises(ValueError):
            ds.refresh()


# ---------------------------------------------------------------------------
# SR_benchmark round-trip: to_archive / from_archive (zip)
# ---------------------------------------------------------------------------


class TestSRBenchmarkArchiveRoundtrip:
    def test_to_archive_and_from_zip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        from SRToolkit.dataset.sampling import UniformSampling
        from SRToolkit.dataset.sr_benchmark import SR_benchmark

        sl = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 2)
        bm = SR_benchmark("mybench", version="1.0.0")
        bm.add_dataset(
            sl,
            None,
            dataset_name="ds1",
            ranking_function="rmse",
            ground_truth=["X_0", "+", "X_1"],
            original_equation="y = x0 + x1",
            samplers=[UniformSampling(0, 1), UniformSampling(0, 1)],
            data_source=SampleSource(n_samples=10, seed=0),
        )

        archive_path = tmp_path / "mybench.zip"
        bm.to_archive(archive_path)
        assert archive_path.exists()
        assert zipfile.is_zipfile(str(archive_path))

        # Verify zip contents
        with zipfile.ZipFile(str(archive_path)) as zf:
            names = zf.namelist()
        assert "benchmark.json" in names
        assert "data/ds1.npz" in names

        # Reconstruct from zip
        bm2 = SR_benchmark.from_archive(str(archive_path))
        assert bm2.benchmark_name == "mybench"
        assert "ds1" in bm2.datasets

    def test_from_dict_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        from SRToolkit.dataset.sampling import UniformSampling
        from SRToolkit.dataset.sr_benchmark import SR_benchmark

        sl = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, 2)
        bm = SR_benchmark("bench2", version="1.0.0")
        bm.add_dataset(
            sl,
            None,
            dataset_name="ds_a",
            ranking_function="rmse",
            ground_truth=["X_0", "*", "X_1"],
            original_equation="y = x0 * x1",
            samplers=[UniformSampling(0, 1), UniformSampling(0, 1)],
            data_source=SampleSource(n_samples=5, seed=1),
        )

        d = bm.to_dict()
        bm2 = SR_benchmark.from_dict(d)
        assert bm2.benchmark_name == "bench2"
        assert bm2.version == "1.0.0"
        assert "ds_a" in bm2.datasets
        # Lazy materialisation via create_dataset
        ds = bm2.create_dataset("ds_a")
        assert ds.X.shape[1] == 2


# ---------------------------------------------------------------------------
# FallbackSource
# ---------------------------------------------------------------------------
class TestFallbackSource:
    def test_roundtrip_and_volatility(self):
        from SRToolkit.dataset.data_source import (
            DataSource,
            FallbackSource,
            SampleSource,
            UrlSource,
        )

        fs = FallbackSource([UrlSource("http://x/a.zip"), SampleSource(n_samples=5, seed=1)])
        restored = DataSource.from_dict(fs.to_dict())
        assert isinstance(restored, FallbackSource)
        assert [type(s).__name__ for s in restored.sources] == ["UrlSource", "SampleSource"]
        assert restored.to_dict() == fs.to_dict()
        # Volatile because the SampleSource fallback is volatile; a URL-only chain is not.
        assert fs.is_volatile is True
        assert FallbackSource([UrlSource("http://x/a.zip")]).is_volatile is False

    def test_rejects_empty(self):
        from SRToolkit.dataset.data_source import FallbackSource

        with pytest.raises(ValueError, match="at least one source"):
            FallbackSource([])

    def test_materialize_falls_through_to_next(self, tmp_path):
        from SRToolkit.dataset.data_source import DataSource, FallbackSource

        calls = []

        class _Boom(DataSource):
            def to_dict(self):
                return {"source_class": "t._Boom"}

            @classmethod
            def from_dict(cls, d):
                return cls()

            def materialize(self, cache_path, config):
                calls.append("boom")
                raise RuntimeError("nope")

        class _Ok(DataSource):
            def to_dict(self):
                return {"source_class": "t._Ok"}

            @classmethod
            def from_dict(cls, d):
                return cls()

            def materialize(self, cache_path, config):
                calls.append("ok")
                Path(cache_path).write_text("data")

        target = tmp_path / "out.npz"
        with pytest.warns(UserWarning, match="trying the next fallback"):
            FallbackSource([_Boom(), _Ok()]).materialize(target, {})
        assert calls == ["boom", "ok"]
        assert target.exists()

    def test_materialize_raises_when_all_fail(self, tmp_path):
        from SRToolkit.dataset.data_source import DataSource, FallbackSource

        class _Boom(DataSource):
            def to_dict(self):
                return {"source_class": "t._Boom"}

            @classmethod
            def from_dict(cls, d):
                return cls()

            def materialize(self, cache_path, config):
                raise RuntimeError("nope")

        with pytest.warns(UserWarning):
            with pytest.raises(RuntimeError, match="All 2 sources failed"):
                FallbackSource([_Boom(), _Boom()]).materialize(tmp_path / "out.npz", {})


def test_builtin_benchmarks_use_fallback_source():
    """Built-ins carry a FallbackSource (download → sample); force_generate pins to sampling."""
    from SRToolkit.dataset import Feynman

    f = Feynman()
    src = f.datasets["I.16.6"]["data_source"]
    assert src["source_class"].endswith("FallbackSource")
    kinds = [s["source_class"].split(".")[-1] for s in src["sources"]]
    assert kinds == ["UrlSource", "SampleSource"]

    fg = Feynman(force_generate=True)
    assert fg.datasets["I.16.6"]["data_source"]["source_class"].endswith("SampleSource")
