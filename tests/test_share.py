"""Tests for SRToolkit.share (save / load of datasets, benchmarks and grids as one archive).

Written from the docstrings of ``share.save`` / ``share.load`` before the implementation.

Isolation: every test runs with the data cache at ``tmp_path/sender`` and the bundle store at
``tmp_path/store/srtk_bundles``; the ``to_recipient`` fixture switches the cache to an empty
``tmp_path/recipient`` to play the other machine. Shipped arrays are chosen outside the samplers'
range (and next to unreachable URLs), so equality with them proves the split was not regenerated or
downloaded.
"""

import inspect
import io
import json
import os
import stat
import sys
import urllib.error
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from SRToolkit import share
from SRToolkit.approaches import Approach, ProGED
from SRToolkit.bundle import _store, bind_config, install, list_installed, pack
from SRToolkit.dataset import data_cache
from SRToolkit.dataset.benchmark import Benchmark
from SRToolkit.dataset.data_source import DataSource, FallbackSource, SampleSource, UrlSource
from SRToolkit.dataset.data_split import DataSplit
from SRToolkit.dataset.dataset import Dataset
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.experiments import ExperimentGrid
from SRToolkit.utils.symbol_library import SymbolLibrary

_DEAD_URL = "http://127.0.0.1:9/unreachable.zip"  # port 9 (discard): never serves a file
_BUNDLE = "custom_ap_bundle"


# --------------------------------------------------------------------------- #
# Fixtures and builders
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(data_cache, "data_root", lambda: tmp_path / "sender")
    # Installed bundles import as ``srtk_bundles.*`` from the store's parent, which install adds to sys.path.
    monkeypatch.setattr(_store, "bundles_root", lambda: tmp_path / "store" / "srtk_bundles")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")


@pytest.fixture
def to_recipient(tmp_path, monkeypatch):
    """Call to switch the data cache to an empty directory (the recipient's machine)."""

    def switch():
        monkeypatch.setattr(data_cache, "data_root", lambda: tmp_path / "recipient")

    return switch


@pytest.fixture
def custom(tmp_path):
    """An importable module ``share_custom`` with a custom approach and a custom data source.

    Importing it touches ``marker``. ``srtk(version)`` packs it as bundle ``custom_ap_bundle``;
    ``forget()`` makes it unimportable (a fresh machine) and deletes the marker.
    """
    mod_dir = tmp_path / "custom_mod"
    mod_dir.mkdir()
    marker = tmp_path / "imported.marker"
    (mod_dir / "share_custom.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).touch()\n"
        "from SRToolkit.approaches import ApproachConfig, Approach\n"
        "from SRToolkit.dataset.data_source import DataSource\n\n"
        "class CustomAp(Approach):\n"
        "    def __init__(self):\n"
        "        super().__init__(ApproachConfig(name='CustomAp'))\n"
        "    def search(self, evaluator, seed=None):\n"
        "        pass\n"
        "    @classmethod\n"
        "    def from_config(cls, config):\n"
        "        return cls()\n\n"
        "class CustomSource(DataSource):\n"
        "    def to_dict(self):\n"
        "        return {'source_class': type(self).__module__ + '.' + type(self).__qualname__}\n"
        "    @classmethod\n"
        "    def from_dict(cls, d):\n"
        "        return cls()\n"
        "    def materialize(self, cache_path, config):\n"
        "        raise RuntimeError('CustomSource has no data')\n"
    )
    sys.path.insert(0, str(mod_dir))
    import share_custom

    def srtk(version="1.2.0", name=_BUNDLE):
        out = tmp_path / "packed" / f"{name}-{version}-src.srtk"  # not the archive's naming
        out.parent.mkdir(exist_ok=True)
        pack(files=[inspect.getsourcefile(share_custom)], out_path=out, name=name, version=version)
        return out

    def forget():
        if str(mod_dir) in sys.path:
            sys.path.remove(str(mod_dir))
        sys.modules.pop("share_custom", None)
        marker.unlink(missing_ok=True)

    yield SimpleNamespace(
        CustomAp=share_custom.CustomAp,
        CustomSource=share_custom.CustomSource,
        ap_path="share_custom.CustomAp",
        marker=marker,
        srtk=srtk,
        forget=forget,
    )
    forget()
    for name in [m for m in sys.modules if m.startswith("srtk_bundles")]:
        sys.modules.pop(name)


def _install(path):
    with patch("SRToolkit.bundle._install._confirm", return_value=True):
        install(path)


def _installed():
    return {(e["name"], e["version"]) for e in list_installed()}


def _samplers():
    return [UniformSampling(0, 1), UniformSampling(0, 1)]


def _array_ds(name="eq1", benchmark="b", offset=10.0, **kwargs):
    """Dataset from arrays (``data_source`` None). X lies outside the samplers' [0, 1) range."""
    X = np.arange(8.0).reshape(4, 2) + offset
    return Dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=X.sum(axis=1),
        samplers=_samplers(),
        dataset_name=name,
        benchmark=benchmark,
        version="1.0.0",
        **kwargs,
    )


def _full_ds():
    """RMSE dataset b/1.0.0/eq1: seeded train source, test split and aux split from arrays."""
    ds = _array_ds(
        test_X=np.array([[20.0, 21.0], [22.0, 23.0]]),
        test_y=np.array([41.0, 45.0]),
        aux_data={"grid": {"X": np.array([[30.0, 31.0]])}},
    )
    ds.data_source = SampleSource(n_samples=4, seed=0)
    return ds


def _bed_ds():
    X = np.arange(8.0).reshape(4, 2) + 10.0
    return Dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ranking_function="bed",
        ground_truth=["X_0", "+", "X_1"],
        samplers=_samplers(),
        dataset_name="bed1",
        benchmark="b",
        version="1.0.0",
    )


def _seeded_ds(name="s1", benchmark="b"):
    return Dataset.from_samplers(
        ground_truth=["X_0", "+", "X_1"],
        samplers=_samplers(),
        n_samples=5,
        seed=1,
        dataset_name=name,
        benchmark=benchmark,
        version="1.0.0",
    )


def _seeded_benchmark():
    bm = Benchmark("b", version="1.0.0")
    bm.add_dataset(
        SymbolLibrary.default_symbols(2),
        None,
        dataset_name="s1",
        ground_truth=["X_0", "+", "X_1"],
        samplers=_samplers(),
        data_source=SampleSource(n_samples=5, seed=1),
    )
    return bm


def _benchmark():
    """Benchmark ``b``: an array entry ``e1`` and an instance entry ``e2`` with test and aux splits."""
    bm = Benchmark("b", version="1.0.0")
    X = np.arange(6.0).reshape(3, 2) + 50.0
    bm.add_dataset(
        SymbolLibrary.default_symbols(2), (X, X.sum(axis=1)), dataset_name="e1", ground_truth=["X_0", "+", "X_1"]
    )
    e2 = _full_ds()
    e2.dataset_name = "e2"
    bm.add_dataset_instance("e2", e2)
    return bm


def _grid(tmp_path, datasets=None, approaches=None):
    return ExperimentGrid(
        datasets if datasets is not None else [_array_ds("a", "b1"), _array_ds("c", "b2", offset=40.0)],
        approaches if approaches is not None else [ProGED()],
        num_experiments=1,
        results_dir=str(tmp_path / "run"),
    )


def _member(benchmark, key, version="1.0.0"):
    """Archive member of a shipped split: its path relative to the data cache, under ``data/``."""
    rel = data_cache.dataset_path(benchmark, version, key).relative_to(data_cache.data_root())
    return "data/" + rel.as_posix()


def _names(path):
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def _data_members(path):
    return {n for n in _names(path) if n.startswith("data/")}


def _config(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("config.json"))


def _manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


def _rewrite_zip(src, dst, extra):
    """Copy ``src`` to ``dst`` with the members of ``extra`` ({name: bytes}) appended."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            zout.writestr(info, zin.read(info))
        for name, content in extra.items():
            zout.writestr(name, content)


class _no_warning:
    """Context manager asserting that no warning is raised inside it."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings(record=True)
        self._log = self._cm.__enter__()
        warnings.simplefilter("always")

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        assert [str(w.message) for w in self._log] == []


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
class TestSaveRoundTrip:
    @pytest.mark.parametrize("make", [_full_ds, _bed_ds], ids=["rmse_test_aux", "bed"])
    def test_round_trip_ships_every_split(self, tmp_path, to_recipient, make):
        ds = make()
        out = share.save(ds, tmp_path / "ds.zip")
        to_recipient()

        loaded = share.load(out)

        assert isinstance(loaded, Dataset)
        np.testing.assert_array_equal(loaded.X, ds.X)
        np.testing.assert_array_equal(loaded.y, ds.y)  # None for BED
        assert loaded.to_dict() == ds.to_dict()  # sources kept, settings unchanged

    def test_round_trip_ships_test_and_aux_splits(self, tmp_path, to_recipient):
        ds = _full_ds()
        out = share.save(ds, tmp_path / "ds.zip")
        to_recipient()

        loaded = share.load(out)

        np.testing.assert_array_equal(loaded.test_X, [[20.0, 21.0], [22.0, 23.0]])
        np.testing.assert_array_equal(loaded.test_y, [41.0, 45.0])
        np.testing.assert_array_equal(loaded.aux_data["grid"].X, [[30.0, 31.0]])

    def test_benchmark_round_trip_ships_test_and_aux(self, tmp_path, to_recipient):
        bm = _benchmark()
        out = share.save(bm, tmp_path / "bm.zip")
        to_recipient()

        loaded = share.load(out)

        assert isinstance(loaded, Benchmark)
        assert sorted(loaded.list_datasets(verbose=False)) == ["e1", "e2"]
        np.testing.assert_array_equal(loaded.create_dataset("e1").X, np.arange(6.0).reshape(3, 2) + 50.0)
        e2 = loaded.create_dataset("e2")
        np.testing.assert_array_equal(e2.X, np.arange(8.0).reshape(4, 2) + 10.0)
        np.testing.assert_array_equal(e2.test_X, [[20.0, 21.0], [22.0, 23.0]])
        np.testing.assert_array_equal(e2.aux_data["grid"].X, [[30.0, 31.0]])

    @pytest.mark.parametrize("results", [False, True], ids=["no_results", "results"])
    def test_grid_round_trip(self, tmp_path, to_recipient, results):
        grid = _grid(tmp_path)
        result_file = tmp_path / "run" / "a" / "ProGED" / "exp_0.json"
        result_file.parent.mkdir(parents=True)
        result_file.write_text('{"from": "sender"}')
        out = share.save(grid, tmp_path / "grid.zip", results=results)
        to_recipient()

        loaded = share.load(out, results_dir=tmp_path / "recipient_run")

        assert isinstance(loaded, ExperimentGrid)
        assert loaded.to_dict() == grid.to_dict()
        assert Path(loaded.results_dir) == tmp_path / "recipient_run"
        np.testing.assert_array_equal(Dataset.from_dict(loaded.datasets["c"]).X, np.arange(8.0).reshape(4, 2) + 40.0)
        assert ("results/a/ProGED/exp_0.json" in _names(out)) is results
        assert (tmp_path / "recipient_run" / "a" / "ProGED" / "exp_0.json").exists() is results

    def test_archive_layout(self, tmp_path):
        ds = _full_ds()
        out = share.save(ds, tmp_path / "ds.zip")

        assert {"manifest.json", "config.json", "README.md"} <= _names(out)
        assert _data_members(out) == {_member("b", "eq1"), _member("b", "eq1/test"), _member("b", "eq1/aux/grid")}
        assert _config(out) == ds.to_dict()
        manifest = _manifest(out)
        assert (manifest["format_version"], manifest["kind"]) == (1, "dataset")


class TestSaveData:
    @pytest.mark.parametrize(
        "make_source, shipped",
        [
            (lambda c: None, True),
            (lambda c: SampleSource(n_samples=4), True),
            (lambda c: SampleSource(n_samples=4, seed=3), False),
            (lambda c: UrlSource(_DEAD_URL), False),
            (lambda c: FallbackSource([UrlSource(_DEAD_URL)]), False),
            (lambda c: FallbackSource([SampleSource(n_samples=4, seed=1)]), False),
            (lambda c: FallbackSource([SampleSource(n_samples=4)]), True),
            (lambda c: FallbackSource([FallbackSource([SampleSource(n_samples=4, seed=1)])]), False),
            (lambda c: FallbackSource([FallbackSource([SampleSource(n_samples=4)])]), True),
            (lambda c: c.CustomSource(), True),
            (lambda c: FallbackSource([c.CustomSource()]), True),
        ],
        ids=[
            "null",
            "seedless_sample",
            "seeded_sample",
            "url",
            "fallback_url",
            "fallback_seeded",
            "fallback_seedless",
            "nested_fallback_seeded",
            "nested_fallback_seedless",
            "custom",
            "fallback_custom",
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:\[share.save\] These custom classes")  # not covered by a bundle
    def test_auto_ships_only_unreproducible_splits(self, tmp_path, custom, make_source, shipped):
        ds = _array_ds()
        ds.data_source = make_source(custom)

        out = share.save(ds, tmp_path / "ds.zip", data="auto")

        assert (_member("b", "eq1") in _data_members(out)) is shipped

    @pytest.mark.parametrize("sourced", ["aux", "test"], ids=["test_shipped", "aux_shipped"])
    def test_auto_decides_each_split_on_its_own(self, tmp_path, sourced):
        ds = _full_ds()  # train: seeded source; test and aux: arrays only
        seeded = DataSplit(arrays={"X": np.zeros((5, 2))}, data_source=SampleSource(n_samples=5, seed=2))
        if sourced == "aux":
            ds.aux_data["grid"] = seeded
        else:
            ds.test = seeded

        out = share.save(ds, tmp_path / "ds.zip", data="auto")

        shipped = "eq1/test" if sourced == "aux" else "eq1/aux/grid"
        assert _data_members(out) == {_member("b", shipped)}

    def test_referenced_split_regenerated_on_load(self, tmp_path, to_recipient):
        ds = _seeded_ds()
        out = share.save(ds, tmp_path / "ds.zip", data="auto")
        to_recipient()

        loaded = share.load(out)

        np.testing.assert_array_equal(loaded.X, ds.X)

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_none_ships_nothing_and_warns_about_unreachable_splits(self, tmp_path):
        ds = _full_ds()  # test split has no data_source

        with pytest.warns(UserWarning, match="eq1"):
            out = share.save(ds, tmp_path / "ds.zip", data="none")

        assert _data_members(out) == set()

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_none_does_not_warn_when_every_split_has_a_source(self, tmp_path):
        out = share.save(_seeded_ds(), tmp_path / "ds.zip", data="none")

        assert _data_members(out) == set()

    def test_unshipped_missing_data_is_not_materialised(self, tmp_path):
        bm = _benchmark()
        data_cache.remove("b", "1.0.0", "e1")  # null source, gone from the cache

        with pytest.warns(UserWarning, match="e1"):
            out = share.save(bm, tmp_path / "bm.zip", data="none")

        assert out.is_file()

    def test_config_dataset_missing_from_cache_is_materialised(self, tmp_path):
        ds = _seeded_ds()
        grid = _grid(tmp_path, datasets=[ds])
        data_cache.remove("b")  # the grid holds only the config; its data must be regenerated

        out = share.save(grid, tmp_path / "g.zip")

        with zipfile.ZipFile(out) as zf, np.load(io.BytesIO(zf.read(_member("b", "s1")))) as npz:
            np.testing.assert_array_equal(npz["X"], ds.X)

    def test_same_dataset_twice_ships_one_copy(self, tmp_path):
        bm = Benchmark("b", version="1.0.0")
        bm.add_dataset_instance("k1", _array_ds())
        bm.add_dataset_instance("k2", _array_ds())

        out = share.save(bm, tmp_path / "bm.zip")

        assert _data_members(out) == {_member("b", "eq1")}

    def test_same_dataset_key_with_different_data_raises(self, tmp_path):
        bm = Benchmark("b", version="1.0.0")
        bm.add_dataset_instance("k1", _array_ds())
        bm.add_dataset_instance("k2", _array_ds(offset=99.0))

        with pytest.raises(ValueError, match="eq1"):
            share.save(bm, tmp_path / "bm.zip")


class TestSaveCode:
    @pytest.mark.parametrize(
        "make, annotated, enclosing",
        [
            (
                lambda tmp, c: _grid(tmp, approaches=[c.CustomAp()]),
                lambda cfg: cfg["approaches"][0],
                lambda cfg: cfg,
            ),
            (
                lambda tmp, c: _custom_source_ds(c),
                lambda cfg: cfg["data_source"],
                lambda cfg: cfg,
            ),
        ],
        ids=["grid_custom_approach", "dataset_custom_source"],
    )
    @pytest.mark.filterwarnings("error::UserWarning")
    def test_supplied_bundle_shipped_and_annotates_config(self, tmp_path, custom, make, annotated, enclosing):
        out = share.save(make(tmp_path, custom), tmp_path / "x.zip", bundles=[custom.srtk("1.2.0")])

        assert {n for n in _names(out) if n.startswith("bundles/")} == {f"bundles/{_BUNDLE}_1.2.0.srtk"}
        cfg = _config(out)
        assert (annotated(cfg)["_bundle"], annotated(cfg)["_version"]) == (_BUNDLE, "1.2.0")
        assert "_bundle" not in enclosing(cfg)

    @pytest.mark.parametrize(
        "supplied, shipped_version",
        [(None, "1.0.0"), ("2.0.0", "2.0.0")],
        ids=["installed_repacked", "supplied_wins"],
    )
    @pytest.mark.filterwarnings("error::UserWarning")
    def test_installed_bundle_repacked_unless_supplied(self, tmp_path, custom, supplied, shipped_version):
        _install(custom.srtk("1.0.0"))
        bound = bind_config({"approach_class": custom.ap_path}, _BUNDLE, "1.0.0")
        approach = Approach.from_config_dict(bound)
        bundles = [custom.srtk(supplied)] if supplied else []

        out = share.save(_grid(tmp_path, approaches=[approach]), tmp_path / "g.zip", bundles=bundles)

        assert {n for n in _names(out) if n.startswith("bundles/")} == {f"bundles/{_BUNDLE}_{shipped_version}.srtk"}
        assert _config(out)["approaches"][0]["_version"] == shipped_version

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_repacks_the_installed_version_the_object_uses(self, tmp_path, custom):
        _install(custom.srtk("1.0.0"))
        _install(custom.srtk("2.0.0"))
        bound = bind_config({"approach_class": custom.ap_path}, _BUNDLE, "1.0.0")
        grid = _grid(tmp_path, approaches=[Approach.from_config_dict(bound)])

        out = share.save(grid, tmp_path / "g.zip")

        with zipfile.ZipFile(out) as zf, zipfile.ZipFile(io.BytesIO(zf.read(f"bundles/{_BUNDLE}_1.0.0.srtk"))) as srtk:
            assert json.loads(srtk.read("manifest.json"))["version"] == "1.0.0"

    def test_uncovered_class_does_not_stop_covered_ones(self, tmp_path, custom, request):
        loose_dir = tmp_path / "loose"
        loose_dir.mkdir()
        (loose_dir / "a_loose.py").write_text(
            "from SRToolkit.dataset.data_source import DataSource\n\n"
            "class LooseSource(DataSource):\n"
            "    def to_dict(self):\n"
            "        return {'source_class': 'a_loose.LooseSource'}\n"
            "    @classmethod\n"
            "    def from_dict(cls, d):\n"
            "        return cls()\n"
            "    def materialize(self, cache_path, config):\n"
            "        raise RuntimeError('no data')\n"
        )
        sys.path.insert(0, str(loose_dir))  # _isolated restores sys.path
        request.addfinalizer(lambda: sys.modules.pop("a_loose", None))
        import a_loose

        ds = _array_ds()
        ds.data_source = a_loose.LooseSource()  # sorts before share_custom.CustomAp and is uncovered
        grid = _grid(tmp_path, datasets=[ds], approaches=[custom.CustomAp()])

        with pytest.warns(UserWarning, match="a_loose.LooseSource"):
            out = share.save(grid, tmp_path / "g.zip", data="none", bundles=[custom.srtk()])

        assert f"bundles/{_BUNDLE}_1.2.0.srtk" in _names(out)

    @pytest.mark.parametrize("kind", ["benchmark", "grid"])
    def test_loaded_object_saves_again_with_its_installed_bundle(self, tmp_path, custom, kind):
        ds = _custom_source_ds(custom)
        if kind == "benchmark":
            obj = Benchmark("b", version="1.0.0")
            obj.add_dataset_instance("e", ds)
        else:
            # The grid holds only the config, so put the arrays in the cache for the custom source.
            cached = data_cache.dataset_path("b", "1.0.0", "eq1")
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cached, X=ds.X, y=ds.y)
            obj = _grid(tmp_path, datasets=[ds])
        first = share.save(obj, tmp_path / "first.zip", bundles=[custom.srtk()])
        custom.forget()
        with patch("SRToolkit.bundle._install._confirm", return_value=True):
            loaded = share.load(
                first, install_bundles=True, **({"results_dir": tmp_path / "r"} if kind == "grid" else {})
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            second = share.save(loaded, tmp_path / "second.zip")

        assert f"bundles/{_BUNDLE}_1.2.0.srtk" in _names(second)
        cfg = _config(second)
        dataset = cfg["datasets"]["e"] if kind == "benchmark" else cfg["datasets"]["eq1"]
        assert (dataset["data_source"]["_bundle"], dataset["data_source"]["_version"]) == (_BUNDLE, "1.2.0")

    def test_config_bundle_without_version_uses_latest_installed(self, tmp_path, custom):
        _install(custom.srtk("1.2.0"))
        entry = _custom_source_ds(custom).to_dict()
        entry["data_source"]["_bundle"] = _BUNDLE  # as a packed config may carry it, with no _version
        bm = Benchmark.from_dict({"benchmark_name": "b", "version": "1.0.0", "datasets": {"e": entry}})
        custom.forget()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = share.save(bm, tmp_path / "bm.zip", data="none")

        assert f"bundles/{_BUNDLE}_1.2.0.srtk" in _names(out)

    def test_stale_annotation_of_an_uncovered_class_is_dropped(self, tmp_path, custom):
        entry = _custom_source_ds(custom).to_dict()
        entry["data_source"].update(_bundle="gone_bundle", _version="9.9.9")  # not installed, not supplied
        bm = Benchmark.from_dict({"benchmark_name": "b", "version": "1.0.0", "datasets": {"e": entry}})

        with pytest.warns(UserWarning, match="share_custom.CustomSource"):
            out = share.save(bm, tmp_path / "bm.zip", data="none")

        source = _config(out)["datasets"]["e"]["data_source"]
        assert "_bundle" not in source and "_version" not in source

    @pytest.mark.parametrize("strict", [False, True])
    def test_class_nested_in_another_bundles_config(self, tmp_path, strict):
        bundles = []
        for module, cls in [("outer_mod", "Outer"), ("inner_mod", "Inner")]:
            src = tmp_path / f"{module}.py"
            src.write_text(f"class {cls}:\n    pass\n")
            bundles.append(tmp_path / f"{module}.srtk")
            pack(files=[src], out_path=bundles[-1], name=module, version="1.0.0")
        ds = _array_ds()
        ds.data_source = _NestingSource()  # outer_mod.Outer's config nests inner_mod.Inner, from another bundle

        context = pytest.raises(ValueError) if strict else pytest.warns(UserWarning)
        with context as caught:
            share.save(ds, tmp_path / "ds.zip", data="none", bundles=bundles, strict=strict)

        messages = [str(caught.value)] if strict else [str(w.message) for w in caught]
        assert any("another bundle" in m and "inner_mod.Inner" in m and "outer_mod.Outer" not in m for m in messages)

    def test_uncovered_class_warns_and_ships_no_bundle(self, tmp_path, custom):
        with pytest.warns(UserWarning, match=custom.ap_path):
            out = share.save(_grid(tmp_path, approaches=[custom.CustomAp()]), tmp_path / "g.zip")

        assert not any(n.startswith("bundles/") for n in _names(out))

    def test_unused_supplied_bundle_warns_and_is_not_shipped(self, tmp_path, custom):
        decoy_src = tmp_path / "decoy.py"
        decoy_src.write_text("class Unrelated:\n    pass\n")
        decoy = tmp_path / "decoy.srtk"
        pack(files=[decoy_src], out_path=decoy, name="decoy", version="0.1.0")
        grid = _grid(tmp_path, approaches=[custom.CustomAp()])

        with pytest.warns(UserWarning, match="decoy"):
            out = share.save(grid, tmp_path / "g.zip", bundles=[custom.srtk(), decoy])

        assert not any("decoy" in n for n in _names(out))


class _NestingSource(DataSource):
    """A data source whose config names a class from one module and nests a class from another."""

    def to_dict(self):
        return {"source_class": "outer_mod.Outer", "parts": [{"thing_class": "inner_mod.Inner"}]}

    @classmethod
    def from_dict(cls, d):
        return cls()

    def materialize(self, cache_path, config):
        raise RuntimeError("no data")


def _custom_source_ds(custom):
    ds = _array_ds()
    ds.data_source = custom.CustomSource()
    return ds


def _py_file(tmp, name="approach.py"):
    path = tmp / name
    path.write_text("x = 1\n")
    return path


def _missing_null_entry(tmp, custom):
    bm = _benchmark()
    data_cache.remove("b", "1.0.0", "e1")
    return bm, {}


def _same_bundle_twice(tmp, custom):
    first = custom.srtk("1.0.0")
    second = tmp / "copy_of_bundle.srtk"
    second.write_bytes(first.read_bytes())
    return _grid(tmp, approaches=[custom.CustomAp()]), {"bundles": [first, second]}


class TestSaveContract:
    @pytest.mark.parametrize(
        "make, exc, match",
        [
            (lambda tmp, c: ({"a": 1}, {}), TypeError, "dict"),
            (lambda tmp, c: (_array_ds(), {"data": "some"}), ValueError, "data"),
            (lambda tmp, c: (_array_ds(), {"results": True}), ValueError, "results"),
            (lambda tmp, c: (_benchmark(), {"results": True}), ValueError, "results"),
            (lambda tmp, c: (_array_ds(benchmark=None), {}), ValueError, "benchmark"),
            (
                lambda tmp, c: (_grid(tmp, approaches=[c.CustomAp()]), {"bundles": [_py_file(tmp)]}),
                ValueError,
                "approach.py",
            ),
            (
                lambda tmp, c: (_grid(tmp, approaches=[c.CustomAp()]), {"bundles": [tmp / "missing.srtk"]}),
                ValueError,
                "missing.srtk",
            ),
            (
                lambda tmp, c: (
                    _grid(tmp, approaches=[c.CustomAp()]),
                    {"bundles": [c.srtk("1.0.0", "bundle_one"), c.srtk("1.0.0", "bundle_two")]},
                ),
                ValueError,
                "CustomAp",
            ),
            (_missing_null_entry, FileNotFoundError, "e1"),
            (_same_bundle_twice, ValueError, "are all bundle"),
        ],
        ids=[
            "not_shareable",
            "bad_data",
            "results_for_dataset",
            "results_for_benchmark",
            "no_benchmark",
            "bundle_not_srtk",
            "bundle_missing",
            "two_bundles_same_class",
            "null_split_missing_from_cache",
            "same_bundle_name_and_version_twice",
        ],
    )
    def test_save_rejects(self, tmp_path, custom, make, exc, match):
        obj, kwargs = make(tmp_path, custom)

        with pytest.raises(exc, match=match):
            share.save(obj, tmp_path / "out.zip", **kwargs)

    @pytest.mark.parametrize(
        "relpath, as_str, expect_warning",
        [
            ("a/b/x.zip", True, lambda: _no_warning()),
            ("x.tar", False, lambda: pytest.warns(UserWarning, match=r"\.zip")),
        ],
        ids=["str_creates_parents", "non_zip_suffix_warns"],
    )
    def test_save_path(self, tmp_path, relpath, as_str, expect_warning):
        path = tmp_path / relpath

        with expect_warning():
            ret = share.save(_array_ds(), str(path) if as_str else path)

        assert ret == path and isinstance(ret, Path)
        assert zipfile.is_zipfile(ret)

    @pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")
    def test_archive_gets_the_mode_of_a_new_file(self, tmp_path):
        out = share.save(_array_ds(), tmp_path / "x.zip")
        normal = tmp_path / "normal.txt"
        normal.write_text("x")

        assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(normal.stat().st_mode)

    def test_save_replaces_existing_file(self, tmp_path):
        path = tmp_path / "x.zip"
        path.write_bytes(b"old")

        share.save(_array_ds(), path)

        assert _config(path) == _array_ds().to_dict()

    @pytest.mark.parametrize("existing", [{}, {"g.zip": b"keep me"}], ids=["no_file", "existing_file"])
    def test_failed_save_writes_nothing(self, tmp_path, custom, existing):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        for name, content in existing.items():
            (out_dir / name).write_bytes(content)

        with pytest.raises(ValueError, match=custom.ap_path):
            share.save(_grid(tmp_path, approaches=[custom.CustomAp()]), out_dir / "g.zip", strict=True)

        assert {p.name: p.read_bytes() for p in out_dir.iterdir()} == existing

    def test_failed_write_leaves_no_partial_file(self, tmp_path):
        out_dir = tmp_path / "out"
        (out_dir / "g.zip").mkdir(parents=True)  # the final move onto ``path`` fails
        (out_dir / "g.zip" / "inner.txt").write_text("keep")

        with pytest.raises(OSError):
            share.save(_array_ds(), out_dir / "g.zip")

        assert [p.name for p in out_dir.iterdir()] == ["g.zip"]
        assert (out_dir / "g.zip" / "inner.txt").read_text() == "keep"

    def test_save_does_not_mutate_obj(self, tmp_path):
        grid = _grid(tmp_path)
        before = (grid.to_dict(), grid.results_dir)

        share.save(grid, tmp_path / "g.zip", data="auto")

        assert (grid.to_dict(), grid.results_dir) == before


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_load_from_url(self, http_dir, to_recipient):
        served, base = http_dir
        ds = _full_ds()
        share.save(ds, served / "ds.zip")
        to_recipient()

        loaded = share.load(f"{base}/ds.zip")

        np.testing.assert_array_equal(loaded.X, ds.X)

    def test_load_from_https_url(self, tmp_path, to_recipient):
        ds = _full_ds()
        payload = share.save(ds, tmp_path / "ds.zip").read_bytes()
        to_recipient()

        with patch("SRToolkit.share.urlopen", return_value=io.BytesIO(payload)) as fake:
            loaded = share.load("https://example.org/ds.zip")

        fake.assert_called_once_with("https://example.org/ds.zip", timeout=60)
        np.testing.assert_array_equal(loaded.X, ds.X)

    @pytest.mark.parametrize("via_url", [False, True], ids=["path", "url"])
    def test_results_dir_default(self, tmp_path, http_dir, via_url):
        served, base = http_dir
        archive = (served if via_url else tmp_path / "share") / "run.zip"
        share.save(_grid(tmp_path), archive)
        expected = (tmp_path / "cwd" / "run") if via_url else (tmp_path / "share" / "run")

        loaded = share.load(f"{base}/run.zip" if via_url else archive)

        assert Path(loaded.results_dir).resolve() == expected.resolve()

    @pytest.mark.parametrize(
        "make, load_kwargs, use",
        [
            (lambda tmp: _seeded_benchmark(), {}, lambda obj: obj.create_dataset("s1")),
            (
                lambda tmp: _grid(tmp, datasets=[_seeded_ds()]),
                {"results_dir": "r"},
                lambda obj: Dataset.from_dict(obj.datasets["s1"]),
            ),
        ],
        ids=["benchmark", "grid"],
    )
    def test_benchmark_and_grid_stay_lazy(self, tmp_path, to_recipient, make, load_kwargs, use):
        out = share.save(make(tmp_path), tmp_path / "x.zip", data="auto")
        to_recipient()
        cached = data_cache.dataset_path("b", "1.0.0", "s1")

        loaded = share.load(out, **load_kwargs)
        assert not cached.exists()
        use(loaded)
        assert cached.exists()

    @pytest.mark.parametrize(
        "preinstalled, install_bundles, confirm",
        [(None, False, True), ("1.0.0", False, True), (None, True, False)],
        ids=["not_installed", "other_version_installed", "install_declined"],
    )
    def test_missing_bundle_warns_and_runs_nothing(self, tmp_path, custom, preinstalled, install_bundles, confirm):
        _install(custom.srtk(preinstalled)) if preinstalled else None
        grid = _grid(tmp_path, approaches=[custom.CustomAp()])
        out = share.save(grid, tmp_path / "g.zip", bundles=[custom.srtk("1.2.0")])
        custom.forget()

        with patch("SRToolkit.bundle._install._confirm", return_value=confirm):
            with pytest.warns(UserWarning, match=r"custom_ap_bundle(.|\n)*install_bundles=True"):
                share.load(out, install_bundles=install_bundles, results_dir=tmp_path / "r")

        assert (_BUNDLE, "1.2.0") not in _installed()
        assert not custom.marker.exists()

    def test_install_bundles_installs_and_binds(self, tmp_path, custom, recwarn):
        out = share.save(_grid(tmp_path, approaches=[custom.CustomAp()]), tmp_path / "g.zip", bundles=[custom.srtk()])
        custom.forget()

        with patch("SRToolkit.bundle._install._confirm", return_value=True):
            loaded = share.load(out, install_bundles=True, results_dir=tmp_path / "r")

        assert (_BUNDLE, "1.2.0") in _installed()
        assert type(Approach.from_config_dict(loaded.approach_configs[0])).__module__.startswith("srtk_bundles.")
        assert not [w for w in recwarn if "install_bundles=True" in str(w.message)]

    def test_installed_bundle_is_skipped(self, tmp_path, custom, recwarn):
        _install(custom.srtk("1.2.0"))
        out = share.save(_grid(tmp_path, approaches=[custom.CustomAp()]), tmp_path / "g.zip", bundles=[custom.srtk()])
        custom.forget()

        with patch("SRToolkit.bundle._install._confirm", side_effect=AssertionError("install prompted again")):
            share.load(out, install_bundles=True, results_dir=tmp_path / "r")

        assert not [w for w in recwarn if "install_bundles=True" in str(w.message)]

    @pytest.mark.parametrize(
        "bundles, warning, exc, match",
        [
            (lambda c: [c.srtk()], _BUNDLE, KeyError, "not installed"),
            (lambda c: [], "share_custom.CustomSource", ImportError, "CustomSource"),
        ],
        ids=["bundle_not_installed", "uncovered"],
    )
    @pytest.mark.filterwarnings(r"ignore:\[share.save\] These custom classes:UserWarning")
    def test_dataset_with_unavailable_custom_class_raises_on_load(self, tmp_path, custom, bundles, warning, exc, match):
        out = share.save(_custom_source_ds(custom), tmp_path / "ds.zip", bundles=bundles(custom))
        custom.forget()

        with pytest.warns(UserWarning, match=warning), pytest.raises(exc, match=match):
            share.load(out)

    def test_uncovered_class_warns_on_load(self, tmp_path, custom):
        with pytest.warns(UserWarning):
            out = share.save(_grid(tmp_path, approaches=[custom.CustomAp()]), tmp_path / "g.zip")

        # The class is still importable here, so only the archive's "uncovered" list can warn.
        with pytest.warns(UserWarning, match=custom.ap_path):
            share.load(out, results_dir=tmp_path / "r")

    def test_shipped_results_do_not_overwrite_existing(self, tmp_path, recwarn):
        grid = _grid(tmp_path)
        sender = tmp_path / "run" / "a" / "ProGED"
        sender.mkdir(parents=True)
        (sender / "exp_0.json").write_text('{"from": "sender"}')
        (sender / "exp_1.json").write_text('{"from": "sender"}')
        saved = share.save(grid, tmp_path / "g.zip", results=True)
        out = tmp_path / "with_dirs.zip"  # zip tools often add directory entries
        _rewrite_zip(saved, out, {"results/": b"", "results/a/": b"", "results/a/ProGED/": b""})
        recipient = tmp_path / "recipient_run" / "a" / "ProGED"
        recipient.mkdir(parents=True)
        (recipient / "exp_0.json").write_text('{"from": "recipient"}')

        share.load(out, results_dir=tmp_path / "recipient_run")

        kept = [str(w.message) for w in recwarn if "Kept" in str(w.message)]
        assert len(kept) == 1 and kept[0].endswith("\n  - " + str(Path("a", "ProGED", "exp_0.json")))  # only exp_0
        assert (recipient / "exp_0.json").read_text() == '{"from": "recipient"}'
        assert (recipient / "exp_1.json").read_text() == '{"from": "sender"}'

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_identical_cache_file_gives_no_warning(self, tmp_path, to_recipient):
        ds = _full_ds()
        ds.aux_data["grid"] = DataSplit(arrays={"X": np.array([[np.nan, 31.0]])})  # NaN equals NaN here
        out = share.save(ds, tmp_path / "ds.zip")
        to_recipient()
        share.load(out)

        loaded = share.load(out)  # second load finds the same arrays in the cache

        np.testing.assert_array_equal(loaded.X, ds.X)

    def test_differing_cache_file_replaced_without_stale_cache_warning(self, tmp_path, to_recipient):
        ds = _full_ds()
        out = share.save(ds, tmp_path / "ds.zip")
        to_recipient()
        stale = ds.to_dict()
        stale["data_source"] = SampleSource(n_samples=4, seed=99).to_dict()
        Dataset.from_dict(stale)  # recipient's own earlier materialisation of b/1.0.0/eq1

        with pytest.warns(UserWarning, match="eq1"):
            share.load(out)

        with _no_warning():
            reloaded = Dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(reloaded.X, ds.X)

    def test_changed_source_after_load_gives_stale_cache_warning(self, tmp_path, to_recipient):
        ds = _full_ds()
        out = share.save(ds, tmp_path / "ds.zip")
        to_recipient()
        share.load(out)
        changed = ds.to_dict()
        changed["data_source"] = SampleSource(n_samples=4, seed=5).to_dict()

        with pytest.warns(UserWarning, match="changed since the cache was built"):
            Dataset.from_dict(changed)


def _text(path, content):
    path.write_text(content)
    return path


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def _saved(make_obj):
    return lambda tmp, url: share.save(make_obj(), tmp / "saved.zip")


def _grid_without_suffix(tmp, url):
    with pytest.warns(UserWarning, match=r"\.zip"):
        return share.save(_grid(tmp), tmp / "run")


def _grid_url_without_suffix(tmp, url):
    with pytest.warns(UserWarning, match=r"\.zip"):
        share.save(_grid(tmp), tmp / "served" / "run")  # the http_dir fixture serves tmp/served
    return f"{url}/run"


class TestLoadContract:
    @pytest.mark.parametrize(
        "make, kwargs, exc, match",
        [
            (lambda tmp, url: tmp / "nope.zip", {}, FileNotFoundError, "nope.zip"),
            (lambda tmp, url: f"{url}/nope.zip", {}, urllib.error.URLError, "404"),
            (lambda tmp, url: _zip(tmp / "x.srtk", {"manifest.json": "{}"}), {}, ValueError, "install"),
            (lambda tmp, url: _text(tmp / "x.zip", "not a zip"), {}, ValueError, "zip"),
            (lambda tmp, url: _zip(tmp / "x.zip", {"readme.txt": "hi"}), {}, ValueError, "manifest.json"),
            (
                lambda tmp, url: _zip(tmp / "x.zip", {"dataset.json": "{}", "data/eq1.npz": b""}),
                {},
                ValueError,
                "manifest.json",
            ),
            (
                lambda tmp, url: _zip(
                    tmp / "x.zip",
                    {"manifest.json": json.dumps({"format_version": 2, "kind": "dataset"}), "config.json": "{}"},
                ),
                {},
                ValueError,
                "format_version",
            ),
            (
                lambda tmp, url: _zip(
                    tmp / "x.zip", {"manifest.json": json.dumps({"kind": "dataset"}), "config.json": "{}"}
                ),
                {},
                ValueError,
                "format_version",
            ),
            (
                lambda tmp, url: _zip(
                    tmp / "x.zip",
                    {"manifest.json": json.dumps({"format_version": 1, "kind": "bundle"}), "config.json": "{}"},
                ),
                {},
                ValueError,
                "kind",
            ),
            (_saved(_array_ds), {"results_dir": "r"}, ValueError, "results_dir"),
            (_saved(_benchmark), {"results_dir": "r"}, ValueError, "results_dir"),
            (_grid_without_suffix, {}, ValueError, "results_dir"),
            (_grid_url_without_suffix, {}, ValueError, "results_dir"),
        ],
        ids=[
            "missing_path",
            "url_not_found",
            "srtk_bundle",
            "not_a_zip",
            "no_manifest",
            "1_6_0_layout",
            "newer_format_version",
            "no_format_version",
            "unknown_kind",
            "results_dir_for_dataset",
            "results_dir_for_benchmark",
            "grid_archive_without_suffix",
            "grid_url_without_suffix",
        ],
    )
    def test_load_rejects(self, tmp_path, http_dir, make, kwargs, exc, match):
        source = make(tmp_path, http_dir[1])  # the served directory is empty

        with pytest.raises(exc, match=match):
            share.load(source, **kwargs)

    def test_unsafe_member_rejected_before_copying(self, tmp_path, to_recipient):
        good = share.save(_full_ds(), tmp_path / "good.zip")
        bad = tmp_path / "bad.zip"
        _rewrite_zip(good, bad, {"data/b/1_0_0/../../../evil.npz": b"x"})  # escapes to tmp_path
        to_recipient()

        with pytest.raises(ValueError, match="outside"):
            share.load(bad)

        assert list((tmp_path / "recipient").rglob("*.npz")) == []
        assert not (tmp_path / "evil.npz").exists()
