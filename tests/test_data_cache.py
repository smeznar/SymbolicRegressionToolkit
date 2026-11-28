"""
Tests for SRToolkit.dataset.data_cache.
"""

import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest

from SRToolkit.dataset import data_cache
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.free_constants import Constant

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYMBOL_LIST = ["+", "-", "*", "/", "u-", "sqrt", "sin", "cos", "exp", "ln", "C"]


def _simple_dataset_config(name: str = "test_ds", benchmark: str = "test_bench", version: str = "1.0.0") -> dict:
    """Return a minimal format_version=3 dataset config for use in tests."""
    return {
        "format_version": 3,
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
                "mirror": True,
            },
            {
                "sampler_class": "SRToolkit.dataset.sampling.UniformSampling",
                "min_value": 0,
                "max_value": 1,
                "mirror": True,
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

    # Reason: clause (SampleSource.materialize Raises: a free constant without a fixed value in
    # ground_truth_constants) + regression (issue #4: the Cython backend used to read past the empty
    # constant array and return garbage y).
    @pytest.mark.parametrize(
        "ground_truth, specs, named",
        [
            (["X_0", "*", "C"], None, "C"),
            (["X_0", "*", "C_1"], None, "C_1"),
            (["X_0", "*", "C_1"], [Constant("C_1", value=1.0).to_dict()], "C_1"),
            (["X_0", "*", "C_1", "+", "C"], [Constant("C_1", value=3.0, fixed=True).to_dict()], "C"),
        ],
        ids=["bare", "labelled", "spec-not-fixed", "fixed-plus-bare"],
    )
    def test_free_constant_raises(self, tmp_path, monkeypatch, ground_truth, specs, named):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = {**_simple_dataset_config(), "ground_truth": ground_truth, "ground_truth_constants": specs}
        with pytest.raises(ValueError, match=rf"free constant \({named}\) without a fixed value"):
            data_cache.resolve("test_bench", "1.0.0", "test_ds", config)

    # Reason: clause (SampleSource.materialize Raises: ground_truth_constants that do not fit the ground
    # truth); the message names the call and the argument.
    def test_specs_that_do_not_fit_raise(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        specs = [Constant("C_9", value=1.0, fixed=True).to_dict()]
        config = {**_simple_dataset_config(), "ground_truth": ["X_0", "*", "C_1"], "ground_truth_constants": specs}
        with pytest.raises(
            ValueError, match=r"^\[SampleSource\.materialize\] Dataset '.+' 'ground_truth_constants': .*'C_9'"
        ):
            data_cache.resolve("test_bench", "1.0.0", "test_ds", config)

    def test_free_constant_allowed_for_bed(self, tmp_path, monkeypatch):
        # BED samples the constants itself and needs no y, so a free constant is fine.
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        config = {**_simple_dataset_config(), "ground_truth": ["X_0", "*", "C"], "ranking_function": "bed"}
        path = data_cache.resolve("test_bench", "1.0.0", "test_ds", config)
        with np.load(path) as data:
            assert "X" in data
            assert "y" not in data


# ---------------------------------------------------------------------------
# resolve — no config with no cache
# ---------------------------------------------------------------------------


class TestResolveNoConfig:
    def test_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        with pytest.raises(FileNotFoundError):
            data_cache.resolve("bench", "1.0.0", "ds")


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
# SR_benchmark round-trip through to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSRBenchmarkDictRoundtrip:
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


# ---------------------------------------------------------------------------
# UrlSource
# ---------------------------------------------------------------------------
def _url_array_ds(name, benchmark, offset):
    """Dataset from arrays; X lies outside any sampler range, so it can only come from the archive."""
    X = np.arange(8.0).reshape(4, 2) + offset
    return SR_dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=X.sum(axis=1),
        dataset_name=name,
        benchmark=benchmark,
        version="1.0.0",
    )


class TestUrlSource:
    def test_reads_own_data_from_share_archive(self, tmp_path, monkeypatch, http_dir):
        from SRToolkit import share
        from SRToolkit.approaches import ProGED
        from SRToolkit.dataset.data_source import UrlSource
        from SRToolkit.experiments import ExperimentGrid

        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "sender")
        served, base = http_dir
        a, a2, c = _url_array_ds("a", "b1", 10.0), _url_array_ds("a2", "b1", 20.0), _url_array_ds("c", "b2", 40.0)
        grid = ExperimentGrid([a, a2, c], [ProGED()], num_experiments=1, results_dir=str(tmp_path / "run"))
        share.save(grid, served / "run.zip")
        cfg = a.to_dict()  # in the sender's cache, so the recipient's stays empty
        cfg["data_source"] = UrlSource(f"{base}/run.zip").to_dict()
        recipient = tmp_path / "recipient"
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: recipient)
        # A sidecar left from the recipient's own earlier a2 data, which the download replaces.
        stale_meta = data_cache._meta_path(data_cache.dataset_path("b1", "1.0.0", "a2"))
        stale_meta.parent.mkdir(parents=True)
        stale_meta.write_text('{"source_hash": "stale"}')

        loaded = SR_dataset.from_dict(cfg)

        np.testing.assert_array_equal(loaded.X, a.X)
        assert not stale_meta.exists()
        # The archive's whole part for this dataset's own benchmark and version is taken, and nothing else.
        assert sorted(recipient.rglob("*.npz")) == sorted(
            [data_cache.dataset_path("b1", "1.0.0", "a"), data_cache.dataset_path("b1", "1.0.0", "a2")]
        )
        np.testing.assert_array_equal(np.load(data_cache.dataset_path("b1", "1.0.0", "a2"))["X"], a2.X)

    @pytest.mark.parametrize(
        "members",
        [{"eq1.npz": None}, {"benchmark.json": "{}", "data/": "", "data/eq1.npz": None}],
        ids=["flat", "hosted_data_prefix"],
    )
    def test_reads_flat_and_hosted_archives(self, tmp_path, monkeypatch, http_dir, members):
        """Regression: ``<key>.npz`` at the root, or under ``data/`` next to a ``benchmark.json`` (the layout of
        the built-in benchmarks' hosted archives, also read by SRToolkit 1.6.0)."""
        import io

        from SRToolkit.dataset.data_source import UrlSource

        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        served, base = http_dir
        ds = _url_array_ds("eq1", "flat_bench", 10.0)
        buf = io.BytesIO()
        np.savez(buf, X=ds.X, y=ds.y)
        with zipfile.ZipFile(served / "flat.zip", "w") as zf:
            for name, content in members.items():
                zf.writestr(name, buf.getvalue() if content is None else content)
        cfg = ds.to_dict()
        cfg["data_source"] = UrlSource(f"{base}/flat.zip").to_dict()
        data_cache.remove("flat_bench")  # to_dict wrote the arrays to the cache; force the download
        from SRToolkit.dataset import data_source

        calls = []

        def recording_urlopen(url, **kwargs):
            calls.append(kwargs)
            return urlopen(url, **kwargs)

        monkeypatch.setattr(data_source, "urlopen", recording_urlopen)

        loaded = SR_dataset.from_dict(cfg)

        assert calls == [{"timeout": 60}]

        np.testing.assert_array_equal(loaded.X, ds.X)
        np.testing.assert_array_equal(loaded.y, ds.y)

    @pytest.mark.parametrize(
        "members, exc, match",
        [
            ({"readme.txt": "no data"}, RuntimeError, "eq1"),
            ({"eq1.npz": None, "../evil.npz": None}, ValueError, "evil.npz"),
        ],
        ids=["no_file_for_key", "member_outside_version_dir"],
    )
    def test_materialize_rejects(self, tmp_path, monkeypatch, http_dir, members, exc, match):
        import io

        from SRToolkit.dataset.data_source import UrlSource

        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        served, base = http_dir
        buf = io.BytesIO()
        np.savez(buf, X=np.ones((2, 2)))
        with zipfile.ZipFile(served / "bad.zip", "w") as zf:
            for name, content in members.items():
                zf.writestr(name, buf.getvalue() if content is None else content)

        with pytest.raises(exc, match=match):
            UrlSource(f"{base}/bad.zip").materialize(data_cache.dataset_path("b", "1.0.0", "eq1"), {})

        assert list(tmp_path.rglob("*.npz")) == []  # nothing written, also not the valid member
