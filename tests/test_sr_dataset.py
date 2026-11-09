"""Tests for SRToolkit.dataset.sr_dataset."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from SRToolkit.approaches.sr_approach import ApproachConfig, SR_approach
from SRToolkit.dataset import data_cache
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.data_split import DataSplit
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.utils.expression_constraints import (
    MaxConstants,
    MaxTokens,
    MaxTreeDepth,
    NoNesting,
    OutputBounds,
    RequiredVariables,
)
from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.grammar.constraints import MaxDepth
from SRToolkit.utils.grammar.grammar import Grammar, Rule
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary


@pytest.fixture
def simple_dataset():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return SR_dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=np.array([3.0, 7.0, 11.0]),
        max_evaluations=10000,
        original_equation="z = x + y",
        success_threshold=1e-6,
    )


@pytest.fixture
def serialisable_dataset(tmp_path, monkeypatch):
    """Dataset with benchmark/version/data_source set, using a patched cache root."""
    monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    samplers = [UniformSampling(0, 5), UniformSampling(0, 5)]
    dataset = SR_dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=np.array([3.0, 7.0, 11.0], dtype=float),
        max_evaluations=10000,
        original_equation="z = x + y",
        success_threshold=1e-6,
        dataset_name="test_ds",
        samplers=samplers,
        benchmark="test_bench",
        version="1.0.0",
    )
    dataset.data_source = SampleSource(n_samples=3, seed=0)
    return dataset


class TestSRDatasetInit:
    def test_basic_init(self, simple_dataset):
        assert simple_dataset.X.shape == (3, 2)
        assert simple_dataset.ranking_function == "rmse"
        assert simple_dataset.success_threshold == 1e-6

    @pytest.mark.parametrize(
        "symbols, expected", [(["+", "C"], True), (["+", "*"], False)], ids=["with-C", "without-C"]
    )
    def test_contains_constants(self, symbols, expected):
        lib = SymbolLibrary.from_symbol_list(symbols, num_variables=1)
        assert SR_dataset(np.zeros((2, 1)), lib, y=np.zeros(2)).contains_constants is expected

    def test_str_bed_describes_bed(self):
        dataset = SR_dataset(
            np.zeros((3, 2)),
            SymbolLibrary.default_symbols(2),
            ranking_function="bed",
            ground_truth=["X_0", "+", "C"],
            success_threshold=0.5,
        )
        s = str(dataset)
        assert "BED ranking function" in s
        assert "behavioral expression distance is less than 0.5" in s
        assert "root mean squared error" not in s

    def test_new_fields_default_none(self, simple_dataset):
        assert simple_dataset.benchmark is None
        assert simple_dataset.version is None
        assert simple_dataset.data_source is None


def _bed_dataset(**kwargs):
    """BED dataset whose samplers draw from |x| in [10, 11], far from X in [0, 1)."""
    return SR_dataset(
        np.random.default_rng(0).random((20, 2)),
        SymbolLibrary.default_symbols(2),
        ranking_function="bed",
        ground_truth=["X_0", "+", "C"],
        success_threshold=1.0,
        samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
        num_consts_sampled=4,
        **kwargs,
    )


class TestSRDatasetEvaluator:
    def test_bed_draws_points_from_dataset_samplers(self):
        bed_X = _bed_dataset(num_points_sampled=8).create_evaluator(seed=0).ranking.bed_X
        assert bed_X.shape == (8, 2)
        assert np.all(np.abs(bed_X) >= 10)

    def test_bed_precomputed_bed_gt_scores_on_X_not_samplers(self):
        X = np.random.default_rng(0).random((20, 2))
        gt = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=4, seed=0)
        dataset = _bed_dataset(bed_gt=gt)
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, dataset.X)

    def test_bed_fixed_per_dataset_across_run_seeds(self):
        dataset = _bed_dataset(num_points_sampled=8, seed=4)
        a, b = dataset.create_evaluator(seed=1), dataset.create_evaluator(seed=2)
        np.testing.assert_array_equal(a.ranking.bed_X, b.ranking.bed_X)
        assert a.evaluate_expr(["X_0", "*", "C"]) == b.evaluate_expr(["X_0", "*", "C"])

    @pytest.mark.parametrize("dataset_seed, expected", [(None, 0), (4, 4)], ids=["no-seed-zero", "dataset-seed"])
    def test_bed_ranking_seed_is_dataset_seed(self, dataset_seed, expected):
        assert _bed_dataset(seed=dataset_seed).create_evaluator(seed=1).ranking.seed == expected

    def test_bed_resample_uses_run_seed(self):
        dataset = _bed_dataset(num_points_sampled=8, bed_resample=True)
        a, b = dataset.create_evaluator(seed=1), dataset.create_evaluator(seed=2)
        assert not np.array_equal(a.ranking.bed_X, b.ranking.bed_X)

    def test_bed_without_ground_truth_uses_y_not_samplers(self):
        X = np.random.default_rng(0).random((20, 2))
        dataset = SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            ranking_function="bed",
            y=X[:, 0],
            success_threshold=1.0,
            samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
            num_points_sampled=-1,
        )
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, X)

    def test_bed_num_points_minus_one_uses_all_of_X(self):
        dataset = _bed_dataset(num_points_sampled=-1)
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, dataset.X)

    def test_create_evaluator(self, simple_dataset):
        evaluator = simple_dataset.create_evaluator()
        error = evaluator.evaluate_expr(["X_0", "+", "X_1"])
        assert error < 1e-6

    def test_evaluator_inherits_settings(self, simple_dataset):
        evaluator = simple_dataset.create_evaluator()
        assert evaluator.max_evaluations == 10000
        assert evaluator.success_threshold == 1e-6

    def test_evaluator_metadata_and_seed(self):
        ds = SR_dataset(_X2, _lib2c(), y=_Y, seed=7, dataset_name="ds", dataset_metadata={"citation": "x"})
        evaluator = ds.create_evaluator(metadata={"key": "val"})
        assert evaluator.metadata == {"key": "val", "dataset_name": "ds", "dataset_metadata": {"citation": "x"}}
        assert evaluator.seed == 7
        assert ds.create_evaluator(seed=2).seed == 2


class TestSRDatasetSerialization:
    def test_to_dict_requires_benchmark(self, simple_dataset):
        with pytest.raises(ValueError, match="benchmark"):
            simple_dataset.to_dict()

    def test_to_dict_requires_version(self):
        X = np.array([[1.0, 2.0]])
        ds = SR_dataset(X, SymbolLibrary.default_symbols(2), benchmark="b", version=None)
        with pytest.raises(ValueError, match="version"):
            ds.to_dict()

    def test_to_dict_includes_benchmark_version(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        assert d["benchmark"] == "test_bench"
        assert d["version"] == "1.0.0"

    def test_to_dict_pure_no_filesystem_side_effects(self, serialisable_dataset, tmp_path):
        before = list(tmp_path.iterdir())
        serialisable_dataset.to_dict()
        after = list(tmp_path.iterdir())
        assert before == after  # no files written


class TestSRDatasetRefresh:
    def test_refresh_null_source_raises(self, simple_dataset):
        simple_dataset.benchmark = "b"
        simple_dataset.version = "1.0.0"
        simple_dataset.data_source = None
        with pytest.raises(ValueError, match="data_source is null"):
            simple_dataset.refresh()

    def test_refresh_reruns_sampling(self, serialisable_dataset):
        serialisable_dataset.refresh()
        # After refresh the cache should be regenerated; X should be reloaded
        assert serialisable_dataset.X.shape[1] == 2


class TestSRDatasetResample:
    def test_resample_returns_correct_shape(self, serialisable_dataset):
        X_new, y_new = serialisable_dataset.resample(50)
        assert X_new.shape == (50, 2)
        assert y_new.shape == (50,)

    def test_resample_no_samplers_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="samplers"):
            simple_dataset.resample(10)

    def test_resample_free_constant_raises(self):
        # A free constant ("C") has no value to evaluate against during generation.
        # This must raise clearly rather than pass an empty constant array (which the
        # compiled Cython backend would read out of bounds, silently producing garbage).
        with pytest.raises(ValueError, match="free constant"):
            SR_dataset.from_samplers(
                ground_truth=["X_0", "*", "C"],
                samplers=[UniformSampling(0.5, 5.0)],
                n_samples=16,
                seed=1,
            )


# ── Data splits (train / test / aux) ──────────────────────────────────────────


class TestDataSplits:
    def _dataset_with_extras(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            ground_truth=["X_0", "+", "X_1"],
            y=np.array([3.0, 7.0, 11.0]),
            test_X=np.array([[7.0, 8.0], [9.0, 10.0]]),
            test_y=np.array([15.0, 19.0]),
            aux_data={"grid": {"X": np.array([[0.0, 0.0]])}},
            dataset_name="ds",
            benchmark="bench",
            version="1.0.0",
        )

    def test_x_setter_updates_the_data(self, simple_dataset):
        simple_dataset.X = np.zeros((3, 2))
        assert simple_dataset.X.sum() == 0.0

    def test_y_setter_none_drops_target(self, simple_dataset):
        simple_dataset.y = None
        assert simple_dataset.y is None

    def test_test_and_aux_accessors(self):
        ds = self._dataset_with_extras()
        assert isinstance(ds.test, DataSplit)
        assert ds.test_X is ds.test.X
        assert ds.test_y.shape == (2,)
        assert set(ds.aux_data) == {"grid"}
        assert isinstance(ds.aux_data["grid"], DataSplit)
        assert ds.aux_data["grid"].X.shape == (1, 2)

    def test_no_test_split_returns_none(self, simple_dataset):
        assert simple_dataset.test is None
        assert simple_dataset.test_X is None
        assert simple_dataset.test_y is None
        assert simple_dataset.aux_data == {}

    def test_ndarray_ground_truth_rejected(self):
        with pytest.raises(ValueError, match="ndarray ground_truth"):
            SR_dataset(np.zeros((2, 1)), SymbolLibrary.default_symbols(1), ground_truth=np.zeros((4, 4)))

    @pytest.mark.parametrize("bad_key", ["", ".", "..", "a/b", "a\\b", "a:b", "a?b"])
    def test_invalid_aux_key_rejected(self, bad_key):
        with pytest.raises(ValueError, match="aux keys become file names"):
            SR_dataset(
                np.zeros((2, 1)),
                SymbolLibrary.default_symbols(1),
                y=np.zeros(2),
                aux_data={bad_key: {"X": np.zeros((1, 1))}},
            )

    @pytest.mark.parametrize("key", ["train", "test", "a__b", "grid 2"])
    def test_aux_keys_only_need_to_be_valid_file_names(self, key):
        ds = SR_dataset(
            np.zeros((2, 1)), SymbolLibrary.default_symbols(1), y=np.zeros(2), aux_data={key: {"X": np.ones((1, 1))}}
        )
        assert ds.aux_data[key].X.sum() == 1.0

    def test_splits_cached_in_dataset_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        self._dataset_with_extras().to_dict()
        for key in ("ds", "ds/test", "ds/aux/grid"):
            assert data_cache.dataset_path("bench", "1.0.0", key).exists()

    def test_dataset_names_with_separators_do_not_collide(self, tmp_path, monkeypatch):
        # A dataset named "ds__test" used to share its file with the test data of "ds".
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        other = SR_dataset(
            np.full((2, 2), 5.0),
            SymbolLibrary.default_symbols(2),
            y=np.zeros(2),
            dataset_name="ds__test",
            benchmark="bench",
            version="1.0.0",
        )
        ds.to_dict()
        other.to_dict()
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test_X, ds.test_X)

    def test_aux_value_can_be_a_data_split(self):
        split = DataSplit({"X": np.ones((2, 2))})
        ds = SR_dataset(np.zeros((2, 2)), SymbolLibrary.default_symbols(2), y=np.zeros(2), aux_data={"grid": split})
        assert ds.aux_data["grid"] is split

    def test_aux_split_with_source_regenerated_not_stored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        far = DataSplit({"X": np.zeros((5, 2))}, data_source=SampleSource(n_samples=5, seed=3))
        ds = SR_dataset(
            np.zeros((3, 2)),
            SymbolLibrary.default_symbols(2),
            y=np.zeros(3),
            ground_truth=["X_0", "+", "X_1"],
            samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
            aux_data={"far": far},
            dataset_name="ds",
            benchmark="bench",
            version="1.0.0",
        )
        d = ds.to_dict()
        assert not data_cache.dataset_path("bench", "1.0.0", "ds/aux/far").exists()
        restored = SR_dataset.from_dict(d)
        assert restored.aux_data["far"].X.shape == (5, 2)
        assert np.all(np.abs(restored.aux_data["far"].X) >= 10)
        assert restored.aux_data["far"].data_source is not None

    def test_aux_entry_added_later_is_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.aux_data["weights"] = DataSplit({"w": np.arange(3.0)})
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.aux_data["weights"].arrays["w"], np.arange(3.0))

    def test_invalid_aux_key_added_later_rejected_on_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.aux_data["a/b"] = DataSplit({"X": np.zeros((1, 2))})
        with pytest.raises(ValueError, match="aux keys become file names"):
            ds.to_dict()

    def test_extra_test_arrays_survive_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.test.arrays["weights"] = np.array([1.0, 2.0])
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test.arrays["weights"], np.array([1.0, 2.0]))

    def test_refresh_regenerates_test_split(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 5), UniformSampling(0, 5)],
            n_samples=10,
            n_test=4,
            seed=0,
            benchmark="bench",
            version="1.0.0",
            dataset_name="gen",
        )
        expected = ds.test_X.copy()
        ds.test.arrays["X"] = np.zeros((4, 2))
        ds.refresh()
        np.testing.assert_array_equal(ds.test_X, expected)

    def test_cache_remove_deletes_split_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        self._dataset_with_extras().to_dict()
        assert [e["key"] for e in data_cache.list()] == ["ds"]
        data_cache.remove("bench", "1.0.0", "ds")
        assert not data_cache.dataset_path("bench", "1.0.0", "ds/test").exists()
        assert not (data_cache.dataset_path("bench", "1.0.0", "ds").parent / "ds").exists()

    def test_from_samplers_n_test_creates_test_split(self):
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 5), UniformSampling(0, 5)],
            n_samples=40,
            n_test=12,
            seed=0,
        )
        assert ds.X.shape == (40, 2)
        assert ds.test_X.shape == (12, 2)
        assert ds.test_y.shape == (12,)
        # The test split carries its own source, so it is regenerated (not persisted).
        assert ds.test.data_source is not None
        # Seeded at seed + 1.
        same_seed = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"], [UniformSampling(0, 5), UniformSampling(0, 5)], n_samples=12, seed=1
        )
        np.testing.assert_array_equal(ds.test_X, same_seed.X)
        unseeded = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"], [UniformSampling(0, 5), UniformSampling(0, 5)], n_samples=4, n_test=3
        )
        assert unseeded.test_X.shape == (3, 2)

    def test_to_dict_from_dict_round_trips_test_and_aux(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test_X, ds.test_X)
        np.testing.assert_array_equal(restored.test_y, ds.test_y)
        np.testing.assert_array_equal(restored.aux_data["grid"].X, ds.aux_data["grid"].X)

    def test_constant_bounds_survive_json_round_trip(self, tmp_path, monkeypatch):
        # JSON turns the tuple into a list; expressions with constants must still be fitted.
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        X = np.random.default_rng(0).random((10, 2))
        ds = SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            y=2 * X[:, 0] + X[:, 1],
            benchmark="b",
            version="1",
            dataset_name="cb",
            constant_bounds=(-3, 3),
        )
        restored = SR_dataset.from_dict(json.loads(json.dumps(ds.to_dict())))
        assert restored.create_evaluator().evaluate_expr(["C", "*", "X_0", "+", "X_1"]) < 1e-6

    def test_node_ground_truth_serialised_with_dataset_library(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        lib = SymbolLibrary.from_symbol_list(["+", "*", "sin", "C"], num_variables=2)
        gt = tokens_to_tree(["sin", "(", "X_0", ")", "*", "X_1"], lib)
        ds = SR_dataset(np.zeros((3, 2)), lib, y=np.zeros(3), ground_truth=gt, benchmark="b", version="1")
        assert ds.to_dict()["ground_truth"] == gt.to_list(symbol_library=lib)

    def test_to_archive_from_archive_round_trips_test(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        archive_path = tmp_path / "ds.zip"
        ds.to_archive(archive_path)
        restored = SR_dataset.from_archive(archive_path)
        np.testing.assert_array_equal(restored.test_X, ds.test_X)
        np.testing.assert_array_equal(restored.aux_data["grid"].X, ds.aux_data["grid"].X)


# ── Background knowledge, objective and format 4 (stage C) ────────────────────

_GOLDEN = Path(__file__).parent / "data" / "v3_cache"
# Golden v3 cache: generated on 2026-11-04 with the pre-stage-C data_cache.py / data_source.py
# (unchanged from commit 9bab9d0) by calling
#   data_cache.resolve("golden_bench", "1.0.0", "ds", config)
# with data_root patched to tests/data/v3_cache/cache and `config` = config_v3.json. It is what a
# user's cache written by an older version looks like: the .meta.json holds the old drift hash.

_NINE_CONSTANTS = ["C", "*", "X_0", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C"]


def _grammar():
    return Grammar(Rule.from_line("E -> E '+' F [0.4] | F [0.6]") + Rule.from_line("F -> 'X_0' [1.0]"))


def _lib2c():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _dicts(items):
    return [c.to_dict() for c in items]


def _v3_config():
    return json.loads((_GOLDEN / "config_v3.json").read_text())


_X2 = np.zeros((3, 2))
_Y = np.zeros(3)


class TestBackgroundKnowledgeArguments:
    def test_flat_arguments_build_background_knowledge(self):
        lib = _lib2c()
        s0, s1 = UniformSampling(2, 3), UniformSampling(0, 1)
        variables = {"X_0": VariableInfo("x", "position")}
        ds = SR_dataset(
            _X2,
            lib,
            y=_Y,
            samplers={"X_1": s1, "X_0": s0},
            variables=variables,
            target=VariableInfo("F"),
            description="Toy",
            constant_bounds=(-3, 3),
            advisory_constraints=[MaxTreeDepth(4)],
            reference_grammar=_grammar(),
            method="L-BFGS-B",
        )
        expected = BackgroundKnowledge(
            lib,
            samplers=[s0, s1],
            constant_bounds=(-3, 3),
            constraints=[MaxTreeDepth(4)],
            reference_grammar=_grammar(),
            variables=variables,
            target=VariableInfo("F"),
            description="Toy",
        )
        assert ds.background_knowledge.to_dict() == expected.to_dict()
        assert ds.symbol_library is lib
        assert ds.samplers[0] is s0 and ds.samplers[1] is s1
        assert ds.constant_bounds == (-3.0, 3.0)
        assert ds.kwargs == {"method": "L-BFGS-B"}

    @pytest.mark.parametrize(
        "name",
        [
            "background_knowledge",
            "symbol_library",
            "samplers",
            "constant_bounds",
            "required_constraints",
            "original_equation",
        ],
    )
    def test_bk_properties_read_only(self, simple_dataset, name):
        with pytest.raises(AttributeError):
            setattr(simple_dataset, name, None)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: SR_dataset(_X2, _lib2c(), y=_Y, max_constants=True),
            lambda: SR_dataset(_X2, _lib2c(), y=_Y, max_constants=2.0),
            lambda: SR_dataset(_X2, _lib2c(), y=_Y, max_constants=3, required_constraints=[MaxConstants(2)]),
            lambda: SR_dataset(_X2, _lib2c(), y=_Y, max_expr_length=5, required_constraints=[MaxTokens(4)]),
            lambda: SR_dataset(_X2, _lib2c(), y=_Y, samplers=[UniformSampling(0, 1)] * 3),
            lambda: SR_dataset(np.zeros((3, 3)), _lib2c(), y=_Y),
        ],
        ids=[
            "max-constants-bool",
            "max-constants-float",
            "max-constants-disagrees",
            "max-expr-length-disagrees",
            "sampler-count",
            "columns-differ-from-variables",
        ],
    )
    def test_init_value_errors(self, make):
        with pytest.raises(ValueError):
            make()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"required_constraints": MaxConstants(3)},
            {"required_constraints": [MaxDepth(3)]},
            {"required_constraints": ["MaxConstants"]},
            {"variables": {"X_0": "x"}},
            {"advisory_constraints": MaxTreeDepth(4)},
            {"reference_grammar": "g"},
        ],
        ids=["lone-constraint", "grammar-constraint", "string-item", "variable-not-info", "lone-hint", "grammar-str"],
    )
    def test_init_type_errors(self, kwargs):
        with pytest.raises(TypeError):
            SR_dataset(_X2, _lib2c(), y=_Y, **kwargs)


class TestRequiredConstraints:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, [MaxConstants(8)]),
            ({"max_expr_length": 20}, [MaxConstants(8), MaxTokens(20)]),
            ({"max_constants": -1}, []),
            ({"max_constants": np.int64(3)}, [MaxConstants(3)]),
            (
                {"required_constraints": [MaxTreeDepth(4)], "max_expr_length": 5},
                [MaxConstants(8), MaxTokens(5), MaxTreeDepth(4)],
            ),
            ({"required_constraints": [MaxConstants(2)]}, [MaxConstants(2)]),
            ({"required_constraints": [MaxConstants(2)], "max_constants": 2}, [MaxConstants(2)]),
            ({"max_expr_length": -1}, [MaxConstants(8)]),
            ({"required_constraints": [MaxConstants(2)], "max_constants": -1}, [MaxConstants(2)]),
        ],
        ids=[
            "default",
            "max-expr-length",
            "no-limit",
            "numpy-int",
            "limits-first",
            "listed-replaces-default",
            "agreeing-setting",
            "negative-length",
            "negative-keeps-listed",
        ],
    )
    def test_required_constraints_assembly(self, kwargs, expected):
        ds = SR_dataset(_X2, _lib2c(), y=_Y, **kwargs)
        assert _dicts(ds.required_constraints) == _dicts(expected)

    def test_required_constraints_accepts_any_iterable(self):
        c = MaxTreeDepth(4)
        ds = SR_dataset(_X2, _lib2c(), y=_Y, required_constraints=(x for x in [c]))
        assert isinstance(ds.required_constraints, tuple)
        assert ds.required_constraints[-1].to_dict() == c.to_dict()


class TestDescribe:
    def test_describe_lists_required_constraints(self):
        lib = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=1)
        ds = SR_dataset(np.zeros((3, 1)), lib, y=_Y, description="Toy")
        assert ds.describe() == (
            "Task: Toy\n"
            "Symbols: +, *, C, X_0\n"
            "Free constants are fitted in [-5.0, 5.0].\n"
            "Enforced constraints (expressions that break them are rejected):\n"
            "  - at most 8 free constants"
        )

    def test_priors_split_required_and_advisory(self):
        hints = [MaxTreeDepth(4), NoNesting(["sin", "cos"]), NoNesting(["exp", "ln"])]
        ds = SR_dataset(
            np.zeros((3, 1)), SymbolLibrary.default_symbols(1), y=_Y, advisory_constraints=hints, max_expr_length=20
        )
        assert ds.priors() == {
            "required": [MaxConstants(8).summary(), MaxTokens(20).summary()],
            "advisory": [c.summary() for c in hints],
        }

    def test_describe_independent_of_answer(self):
        # b's equation is derived from its names: "F = m*X_1"; neither answer may leak into describe.
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        names = dict(variables={"X_0": VariableInfo("m")}, target=VariableInfo("F"))
        a = SR_dataset(
            X,
            _lib2c(),
            y=X.sum(axis=1),
            ground_truth=["X_0", "+", "X_1"],
            original_equation="z = x + y",
            success_threshold=1e-6,
            **names,
        )
        b = SR_dataset(
            X,
            _lib2c(),
            y=X.prod(axis=1),
            ground_truth=["X_0", "*", "X_1"],
            success_threshold=0.5,
            **names,
        )
        assert a.describe() == b.describe()
        assert "z = x + y" not in a.describe()
        assert "F = m*X_1" not in b.describe()

    def test_str_layout(self):
        ds = SR_dataset(
            _X2,
            _lib2c(),
            y=_Y,
            original_equation="z = x + y",
            success_threshold=1e-6,
            description="Toy",
            method="L-BFGS-B",
        )
        s = str(ds)
        lines = s.split("\n")
        assert "z = x + y" in lines[0]
        assert s.index("z = x + y") < s.index("RMSE") < s.index("L-BFGS-B") < s.index(ds.describe())
        assert s.index(ds.describe()) + len(ds.describe()) < len(s) - len(lines[-1])
        assert "dataset_metadata" in lines[-1]


class TestCreateEvaluatorConstraints:
    def test_ranking_settings_are_what_the_evaluator_gets(self):
        X = np.random.default_rng(0).random((5, 2))
        samplers = [UniformSampling(0, 1), UniformSampling(0, 1)]
        ds = SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            y=X.sum(axis=1),
            ground_truth=["X_0", "+", "X_1"],
            samplers=samplers,
            constant_bounds=(-2, 2),
            seed=3,
            method="L-BFGS-B",
        )
        settings = ds.ranking_settings()
        assert settings["constant_bounds"] == (-2.0, 2.0)
        assert settings["ranking_seed"] == 3
        assert settings["method"] == "L-BFGS-B"
        assert _dicts(settings["samplers"]) == _dicts(samplers)
        assert "max_constants" not in settings and "max_expr_length" not in settings
        ev_settings = ds.create_evaluator().kwargs
        assert all(ev_settings[k] == v for k, v in settings.items() if k != "samplers")

    def test_create_evaluator_passes_constraint_lists(self):
        hints = [MaxTreeDepth(3)]
        ds = SR_dataset(
            _X2,
            SymbolLibrary.default_symbols(2),
            y=_Y,
            advisory_constraints=hints,
            required_constraints=[RequiredVariables(["X_0"])],
            max_expr_length=30,
        )
        ev = ds.create_evaluator()
        ds.create_evaluator()
        assert _dicts(ev.required_constraints) == _dicts(ds.required_constraints)
        assert _dicts(ev.advisory_constraints) == _dicts(hints)
        assert all(c.symbol_library is ds.symbol_library for c in ds.required_constraints)
        assert all(c.symbol_library is ds.symbol_library for c in ds.background_knowledge.constraints)
        assert all(e is not d for e, d in zip(ev.required_constraints, ds.required_constraints))

    @pytest.mark.parametrize(
        "kwargs, expr, status, detail",
        [
            ({}, _NINE_CONSTANTS, "rejected", "MaxConstants: 9 constants, limit 8"),
            ({"max_constants": -1}, _NINE_CONSTANTS, "ok", None),
            (
                {"required_constraints": [MaxConstants(1)]},
                ["C", "*", "X_0", "+", "C"],
                "rejected",
                "MaxConstants: 2 constants, limit 1",
            ),
        ],
        ids=["default-limit", "no-limit", "listed-limit"],
    )
    def test_create_evaluator_adds_no_limits(self, kwargs, expr, status, detail):
        X = np.random.default_rng(0).random((10, 1))
        ds = SR_dataset(X, SymbolLibrary.default_symbols(1), y=3 * X[:, 0], **kwargs)
        ev = ds.create_evaluator()
        ev.evaluate_expr(expr)
        assert ev.models[-1].status == status
        assert ev.models[-1].status_detail == detail

    def test_create_evaluator_records_hints(self):
        X = np.random.default_rng(0).random((10, 2))
        ds = SR_dataset(X, SymbolLibrary.default_symbols(2), y=X.sum(axis=1), advisory_constraints=[MaxTreeDepth(1)])
        ev = ds.create_evaluator()
        assert ev.evaluate_expr(["X_0", "+", "X_1"]) < 1e-9
        violations = ev.models[-1].advisory_violations
        assert [(v.constraint, v.stage) for v in violations] == [("MaxTreeDepth", "pre")]

    @pytest.mark.parametrize(
        "make, has_samplers, bounds",
        [
            (lambda: _bed_dataset(), True, (-5.0, 5.0)),
            (lambda: _bed_dataset(constant_bounds=(-1, 1)), True, (-1.0, 1.0)),
            (lambda: _bed_dataset(num_points_sampled=-1), True, (-5.0, 5.0)),
            (
                lambda: _bed_dataset(
                    bed_gt=create_behavior_matrix(
                        ["X_0", "+", "C"], np.random.default_rng(0).random((20, 2)), num_consts_sampled=4, seed=0
                    )
                ),
                True,
                (-5.0, 5.0),
            ),
            (
                lambda: SR_dataset(
                    np.random.default_rng(0).random((20, 2)),
                    SymbolLibrary.default_symbols(2),
                    y=np.zeros(20),
                    samplers=[UniformSampling(0, 1), UniformSampling(0, 1)],
                ),
                True,
                (-5.0, 5.0),
            ),
            (
                lambda: SR_dataset(
                    np.random.default_rng(0).random((20, 2)),
                    SymbolLibrary.default_symbols(2),
                    y=np.zeros(20),
                    ground_truth=["X_0", "+", "X_1"],
                ),
                False,
                (-5.0, 5.0),
            ),
        ],
        ids=["all-conditions", "bounds-set", "num-points-minus-one", "bed-gt", "no-ground-truth", "no-samplers"],
    )
    def test_create_evaluator_ranking_settings(self, make, has_samplers, bounds):
        ds = make()
        ev = ds.create_evaluator(seed=0)
        assert tuple(ev.kwargs["constant_bounds"]) == bounds
        assert ("samplers" in ev.kwargs) is has_samplers
        if has_samplers:
            assert _dicts(ev.kwargs["samplers"]) == _dicts(ds.samplers)


class TestFormat4:
    def test_to_dict_v4_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = SR_dataset(
            _X2, _lib2c(), y=_Y, benchmark="b", version="1", constant_bounds=(-3, 3), max_constants=8, seed=3,
            dataset_metadata={"citation": "x"},
        )  # fmt: skip
        d = ds.to_dict()
        assert set(d) == {
            "format_version", "dataset_name", "benchmark", "version", "ranking_function", "max_evaluations",
            "success_threshold", "original_equation", "seed", "dataset_metadata", "kwargs", "ground_truth",
            "data_source", "test", "aux", "background_knowledge", "required_constraints",
        }  # fmt: skip
        assert d["format_version"] == 4
        assert d["required_constraints"] == [MaxConstants(8).to_dict()]
        assert d["background_knowledge"] == ds.background_knowledge.to_dict()
        assert d["kwargs"] == {}
        assert (d["seed"], d["dataset_metadata"]) == (3, {"citation": "x"})
        assert d["ground_truth"] is None

    def test_v4_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = SR_dataset(
            _X2,
            _lib2c(),
            y=_Y,
            samplers={"X_1": UniformSampling(0, 1), "X_0": UniformSampling(2, 3)},
            constant_bounds=(-2, 2),
            advisory_constraints=[MaxTreeDepth(4)],
            reference_grammar=_grammar(),
            description="Toy",
            variables={"X_1": VariableInfo("v", "speed")},
            target=VariableInfo("F"),
            required_constraints=[RequiredVariables(["X_0"])],
            max_expr_length=12,
            method="L-BFGS-B",
            benchmark="b",
            version="1",
            ranking_function="bed",
            ground_truth=["X_0", "+", "C"],
            max_evaluations=500,
            original_equation="F = x + c",
            success_threshold=0.5,
            seed=3,
            dataset_metadata={"citation": "x"},
        )
        d = json.loads(json.dumps(ds.to_dict()))
        assert d["kwargs"] == {"method": "L-BFGS-B"}
        restored = SR_dataset.from_dict(d)
        assert restored.to_dict() == d
        assert restored.background_knowledge.to_dict() == ds.background_knowledge.to_dict()
        path = tmp_path / "ds.json"
        path.write_text(json.dumps(d))
        assert SR_dataset.from_dict(path).to_dict() == d

    def test_migrates_v3_without_samplers(self, tmp_path, monkeypatch):
        # Array data saved in an older version: no samplers, no data_source, arrays in the cache.
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        X = np.arange(6.0).reshape(3, 2)
        SR_dataset(X, _lib2c(), y=X.sum(axis=1), benchmark="b", version="1", dataset_name="arr").to_dict()
        v3 = {
            **_v3_config(),
            "benchmark": "b",
            "version": "1",
            "dataset_name": "arr",
            "symbol_library": _lib2c().to_dict(),
            "samplers": None,
            "data_source": None,
        }
        ds = SR_dataset.from_dict(v3)
        assert ds.samplers is None
        np.testing.assert_array_equal(ds.X, X)

    @pytest.mark.parametrize(
        "format_version, kwargs, bounds, required, kept_kwargs",
        [
            (
                3,
                {"constant_bounds": [-3, 3], "max_constants": 4, "max_expr_length": 20, "method": "L-BFGS-B"},
                (-3.0, 3.0),
                [MaxConstants(4), MaxTokens(20)],
                {"method": "L-BFGS-B"},
            ),
            (3, {}, (-5.0, 5.0), [MaxConstants(8)], {}),
            (3, {"max_constants": -1}, (-5.0, 5.0), [], {}),
            (2, {}, (-5.0, 5.0), [MaxConstants(8)], {}),
        ],
        ids=["v3-all-settings", "v3-defaults", "v3-no-limit", "v2"],
    )
    def test_migrates_v2_v3(self, tmp_path, monkeypatch, format_version, kwargs, bounds, required, kept_kwargs):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        d = {**_v3_config(), "format_version": format_version, "kwargs": kwargs}
        if format_version == 2:
            del d["test"], d["aux"]
        ds = SR_dataset.from_dict(d)
        assert ds.constant_bounds == bounds
        assert _dicts(ds.required_constraints) == _dicts(required)
        assert ds.kwargs == kept_kwargs
        assert _dicts(ds.samplers) == d["samplers"]
        assert ds.symbol_library.to_dict() == d["symbol_library"]
        out = ds.to_dict()
        assert out["format_version"] == 4
        assert "samplers" not in out and "symbol_library" not in out

    @pytest.mark.parametrize(
        "make",
        [
            lambda v4: {**v4, "format_version": 1},
            lambda v4: {**v4, "format_version": 5},
            lambda v4: {**v4, "format_version": 99},
            lambda v4: "dataset.zip",
            lambda v4: {**_v3_config(), "samplers": _v3_config()["samplers"][:1]},
        ],
        ids=["v1", "v5", "v99", "zip-path", "v3-sampler-count"],
    )
    def test_from_dict_errors(self, serialisable_dataset, make):
        with pytest.raises(ValueError):
            SR_dataset.from_dict(make(serialisable_dataset.to_dict()))

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_v3_cache_has_no_drift_after_migration(self, tmp_path, monkeypatch):
        shutil.copytree(_GOLDEN / "cache", tmp_path / "cache")
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        cached_X = np.load(tmp_path / "cache" / "golden_bench" / "1_0_0" / "ds.npz")["X"]
        ds = SR_dataset.from_dict(_v3_config())
        v4 = ds.to_dict()
        restored = SR_dataset.from_dict(v4)
        data_cache.resolve("golden_bench", "1.0.0", "ds", v4)
        np.testing.assert_array_equal(ds.X, cached_X)
        np.testing.assert_array_equal(restored.X, cached_X)

    def test_v4_sampler_change_warns_refresh(self, tmp_path, monkeypatch):
        shutil.copytree(_GOLDEN / "cache", tmp_path / "cache")
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")
        v4 = SR_dataset.from_dict(_v3_config()).to_dict()
        v4["background_knowledge"]["samplers"][0]["max_value"] = 2
        with pytest.warns(UserWarning, match="refresh"):
            SR_dataset.from_dict(v4)


class TestFromSamplersStageC:
    def test_from_samplers_mapping_regenerates(self, tmp_path, monkeypatch):
        # A custom function ("sq" is not a default symbol) checks that regeneration on load uses the
        # dataset's own symbol library.
        lib = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        lib.add_symbol("sq", "fn", 5, "np.square({})")
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "a")
        ds = SR_dataset.from_samplers(
            ["sq", "(", "X_0", ")", "+", "X_1"],
            {"X_1": UniformSampling(10, 11, uses_negative=False), "X_0": UniformSampling(0, 1, uses_negative=False)},
            symbol_library=lib,
            n_samples=20,
            n_test=4,
            seed=0,
            benchmark="b",
            version="1",
            dataset_name="m",
        )
        assert np.all((ds.X[:, 0] >= 0) & (ds.X[:, 0] <= 1))
        assert np.all((ds.X[:, 1] >= 10) & (ds.X[:, 1] <= 11))
        np.testing.assert_allclose(ds.y, ds.X[:, 0] ** 2 + ds.X[:, 1])
        d = ds.to_dict()
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "b")
        restored = SR_dataset.from_dict(d)
        np.testing.assert_array_equal(restored.X, ds.X)
        np.testing.assert_allclose(restored.y, ds.y)
        np.testing.assert_array_equal(restored.test_X, ds.test_X)

    @pytest.mark.parametrize(
        "original_equation, expected", [(None, "z = x+X_1"), ("z = x + y", "z = x + y")], ids=["auto", "given"]
    )
    def test_from_samplers_forwards_arguments(self, original_equation, expected):
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 1), UniformSampling(0, 1)],
            n_samples=5,
            n_test=3,
            seed=0,
            ranking_function="bed",
            original_equation=original_equation,
            success_threshold=0.1,
            max_evaluations=50,
            dataset_name="fwd",
            dataset_metadata={"citation": "x"},
            variables={"X_0": VariableInfo("x")},
            target=VariableInfo("z"),
            description="Toy",
            method="L-BFGS-B",
        )
        assert ds.original_equation == expected
        assert (ds.ranking_function, ds.success_threshold, ds.max_evaluations) == ("bed", 0.1, 50)
        assert ds.test_X.shape == (3, 2) and ds.test_y is None
        assert (ds.dataset_name, ds.dataset_metadata) == ("fwd", {"citation": "x"})
        assert ds.background_knowledge.variables == {"X_0": VariableInfo("x")}
        assert (ds.background_knowledge.target, ds.background_knowledge.description) == (VariableInfo("z"), "Toy")
        assert ds.kwargs == {"method": "L-BFGS-B"}

    def test_from_samplers_uses_flat_arguments_and_limits(self):
        lib = SymbolLibrary.default_symbols(2)
        samplers = [UniformSampling(0, 1), UniformSampling(0, 1)]
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            samplers,
            symbol_library=lib,
            description="Toy",
            advisory_constraints=[MaxTreeDepth(3)],
            n_samples=15,
            seed=0,
            max_constants=2,
            max_expr_length=9,
            required_constraints=[MaxTreeDepth(5)],
        )
        expected = BackgroundKnowledge(lib, samplers=samplers, description="Toy", constraints=[MaxTreeDepth(3)])
        assert ds.background_knowledge.to_dict() == expected.to_dict()
        assert ds.X.shape == (15, 2)
        assert _dicts(ds.required_constraints) == _dicts([MaxConstants(2), MaxTokens(9), MaxTreeDepth(5)])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"samplers": []},
            {"samplers": [UniformSampling(0, 1)] * 2, "symbol_library": SymbolLibrary.default_symbols(3)},
        ],
        ids=["no-samplers", "empty-samplers", "count-mismatch"],
    )
    def test_from_samplers_errors(self, kwargs):
        with pytest.raises(ValueError):
            SR_dataset.from_samplers(["X_0"], n_samples=5, seed=0, **kwargs)


# ── evaluate_approach ─────────────────────────────────────────────────────────


# ── original_equation: given or derived from ground_truth and names ────────────

_LIB_EQ = SymbolLibrary.default_symbols(2)
_GT_EQ = ["X_0", "*", "X_1"]
_M = {"X_0": VariableInfo("m")}


def _eq_dataset(**kwargs):
    return SR_dataset(np.zeros((3, 2)), _LIB_EQ, y=np.zeros(3), **kwargs)


class TestOriginalEquation:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            (dict(ground_truth=_GT_EQ, variables=_M, target=VariableInfo("F")), "F = m*X_1"),
            (
                dict(
                    ground_truth=_GT_EQ,
                    variables={"X_0": VariableInfo("", "mass"), "X_1": VariableInfo("a")},
                    target=VariableInfo("F"),
                ),
                "F = X_0*a",
            ),
            (dict(ground_truth=_GT_EQ, variables=_M), "y = m*X_1"),
            (dict(ground_truth=_GT_EQ, variables=_M, target=VariableInfo("", "force")), "y = m*X_1"),
            (dict(ground_truth=_GT_EQ), "y = X_0*X_1"),
            # Whole-token replacement: names that are other tokens must not be substituted again.
            (
                dict(ground_truth=_GT_EQ, variables={"X_0": VariableInfo("X_1"), "X_1": VariableInfo("X_0")}),
                "y = X_1*X_0",
            ),
            # With the library to_list gives X_0 ^2; without it ( X_0 ) ^2.
            (
                dict(
                    ground_truth=tokens_to_tree(["X_0", "^2"], _LIB_EQ),
                    variables={"X_0": VariableInfo("v")},
                    target=VariableInfo("E"),
                ),
                "E = v^2",
            ),
            # A token list is used exactly as given (redundant parentheses kept).
            (dict(ground_truth=["(", "X_0", ")", "*", "X_1"], variables=_M), "y = (m)*X_1"),
            (dict(ground_truth=["u-", "X_0", "*", "X_1"], variables=_M), "y = -m*X_1"),
            # The default left side skips names taken by variables.
            (
                dict(ground_truth=_GT_EQ, variables={"X_0": VariableInfo("y"), "X_1": VariableInfo("y_1")}),
                "y_2 = y*y_1",
            ),
            (dict(ground_truth=_GT_EQ, variables=_M, target=VariableInfo("F"), original_equation="F = m a"), "F = m a"),
            (dict(ground_truth=_GT_EQ, variables=_M, original_equation=""), ""),
            (dict(), None),
        ],
        ids=[
            "spec-example",
            "empty-var-name",
            "no-target",
            "empty-target-name",
            "no-names",
            "swapped-names",
            "node-gt",
            "token-list-verbatim",
            "unary-minus",
            "default-target-avoids-names",
            "given-wins",
            "empty-given-wins",
            "no-gt",
        ],
    )
    def test_original_equation_resolution(self, kwargs, expected):
        assert _eq_dataset(**kwargs).original_equation == expected

    @pytest.mark.parametrize(
        "given, saved, value",
        [("F = m a", "F = m a", "F = m a"), (None, None, "F = m*X_1")],
        ids=["given", "derived"],
    )
    def test_to_dict_saves_only_given_equation(self, tmp_path, monkeypatch, given, saved, value):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = _eq_dataset(
            ground_truth=_GT_EQ,
            variables=_M,
            target=VariableInfo("F"),
            original_equation=given,
            benchmark="b",
            version="1",
        )
        d = ds.to_dict()
        assert d["original_equation"] == saved
        restored = SR_dataset.from_dict(d)
        assert restored.original_equation == value
        assert restored.to_dict() == d

    def test_str_omits_equation_line_when_none(self):
        # Metamorphic: without an equation, str is the same text minus the equation line.
        with_equation = str(_eq_dataset(original_equation="F = m a"))
        assert with_equation.startswith("Dataset for target expression F = m a.\n")
        assert str(_eq_dataset()) == with_equation.split("\n", 1)[1]

    def test_str_shows_derived_equation(self):
        ds = _eq_dataset(ground_truth=_GT_EQ, variables=_M, target=VariableInfo("F"))
        assert "F = m*X_1" in str(ds).split("\n")[0]


class _SeedRecorder(SR_approach):
    """Records the seed of every search; evaluates one expression so each run has a result."""

    def __init__(self):
        super().__init__(ApproachConfig(name="recorder"))
        self.seeds = []

    def prepare(self):
        pass

    def search(self, sr_evaluator, seed=None):
        self.seeds.append(seed)
        sr_evaluator.evaluate_expr(["X_0"])


@pytest.mark.parametrize(
    "initial_seed, dataset_seed, expected",
    [(10, None, [10, 11, 12]), (None, 4, [4, 5, 6]), (None, None, [None, None, None])],
    ids=["initial-seed", "dataset-seed", "unseeded"],
)
def test_evaluate_approach_seeds_experiments_from_base(initial_seed, dataset_seed, expected):
    # regression: experiment i got base + i + 1, so the base seed was never used (ExperimentGrid
    # gives job i the seed initial_seed + i).
    ds = SR_dataset(
        np.array([[1.0], [2.0]]), SymbolLibrary.default_symbols(1), y=np.array([1.0, 2.0]), seed=dataset_seed
    )
    approach = _SeedRecorder()
    results = ds.evaluate_approach(approach, num_experiments=3, initial_seed=initial_seed)
    assert approach.seeds == expected
    assert len(results.results) == 3


# ── Approach access (stage D) ─────────────────────────────────────────────────


class _AdaptRecorder(SR_approach):
    """Logs the framework's calls and what ``adapt`` received; ``pre_check`` raises unless set up."""

    def __init__(self, scope="experiment"):
        super().__init__(ApproachConfig(name="recorder"))
        self._scope = scope
        self.events = []
        self.adapt_args = []
        self.loaded_from = []

    @property
    def adaptation_scope(self):
        return self._scope

    def prepare(self):
        self.events.append("prepare")

    def adapt(self, X, background_knowledge, required_constraints):
        self.events.append("adapt")
        required = tuple(required_constraints)
        checked = [c.pre_check(["X_0"]).passed for c in required]
        self.adapt_args.append((X, background_knowledge, required, checked))

    def search(self, sr_evaluator, seed=None):
        self.events.append("search")
        sr_evaluator.evaluate_expr(["X_0"])

    def save_adapted_state(self, path):
        self.events.append("save")
        Path(path).write_text("state")

    def load_adapted_state(self, path):
        self.events.append("load")
        self.loaded_from.append(path)


def _task_dataset(**kwargs):
    """Dataset whose required list is [MaxConstants(8), MaxTokens(20), MaxTreeDepth(5)], with a hint."""
    X = np.random.default_rng(0).random((10, 2))
    defaults = dict(
        y=X[:, 0],
        required_constraints=[MaxTreeDepth(5)],
        max_expr_length=20,
        advisory_constraints=[MaxTreeDepth(3)],
        description="Toy task",
        variables={"X_0": VariableInfo("x", "position")},
        target=VariableInfo("F", "force"),
    )
    defaults.update(kwargs)
    return SR_dataset(X, _lib2c(), **defaults)


_TASK_REQUIRED = [MaxConstants(8), MaxTokens(20), MaxTreeDepth(5)]


class TestRequiredConstraintsSetUp:
    @pytest.mark.parametrize(
        "kwargs, expected",
        [({}, _TASK_REQUIRED), ({"max_constants": -1, "max_expr_length": -1, "required_constraints": []}, [])],
        ids=["limits-first", "none"],
    )
    def test_required_constraints_are_set_up_copies(self, kwargs, expected):
        given = MaxTreeDepth(5)
        kwargs.setdefault("required_constraints", [given])
        ds = _task_dataset(**kwargs)
        stored = ds.required_constraints
        assert isinstance(stored, tuple)
        assert _dicts(stored) == _dicts(expected)
        assert all(c.symbol_library is ds.symbol_library for c in stored)
        assert all(c is not given for c in stored)
        assert given.symbol_library is None

    @pytest.mark.parametrize(
        "constraint, match",
        [(RequiredVariables("X_9"), "X_9"), (OutputBounds(np.zeros((2, 3)), lower=0.0), None)],
        ids=["unknown-variable", "points-wrong-columns"],
    )
    def test_misfit_required_constraint_rejected(self, constraint, match):
        with pytest.raises(ValueError, match=match):
            SR_dataset(_X2, _lib2c(), y=_Y, required_constraints=[constraint])


class TestEvaluateApproachAdapt:
    @pytest.mark.parametrize("scope", ["experiment", "once"])
    def test_evaluate_approach_passes_task_to_adapt(self, scope):
        ds = _task_dataset()
        approach = _AdaptRecorder(scope)
        ds.evaluate_approach(approach, num_experiments=1, verbose=False)
        [(X, bk, required, checked)] = approach.adapt_args
        np.testing.assert_array_equal(X, ds.X)
        assert bk is ds.background_knowledge
        assert _dicts(required) == _dicts(_TASK_REQUIRED)
        assert checked == [True, True, True]
        assert len(required) == len(ds.required_constraints)
        assert all(a is b for a, b in zip(required, ds.required_constraints))

    @pytest.mark.parametrize(
        "scope, with_path, expected",
        [
            ("never", False, ["prepare", "search", "prepare", "search"]),
            ("once", False, ["prepare", "adapt", "search", "prepare", "search"]),
            ("experiment", True, ["prepare", "adapt", "search", "prepare", "adapt", "search"]),
        ],
        ids=["never", "once-without-path", "experiment-not-saved"],
    )
    def test_evaluate_approach_adapts_per_scope(self, tmp_path, scope, with_path, expected):
        ds = _task_dataset()
        approach = _AdaptRecorder(scope)
        path = str(tmp_path / "state") if with_path else None
        ds.evaluate_approach(approach, num_experiments=2, verbose=False, adaptation_path=path)
        ds.evaluate_approach(approach, num_experiments=2, verbose=False, adaptation_path=path)
        assert approach.events == expected * 2
        assert not (tmp_path / "state").exists()

    def test_evaluate_approach_once_reuses_saved_state_as_is(self, tmp_path):
        path = str(tmp_path / "missing" / "dirs" / "state")  # parent directories are created
        first = _AdaptRecorder("once")
        _task_dataset().evaluate_approach(first, num_experiments=2, verbose=False, adaptation_path=path)
        assert (first.events.count("adapt"), first.events.count("save")) == (1, 1)
        assert Path(path).read_text() == "state"

        second = _AdaptRecorder("once")
        changed = _task_dataset(max_constants=3)
        changed.evaluate_approach(second, num_experiments=2, verbose=False, adaptation_path=path)
        assert (second.events.count("adapt"), second.loaded_from) == (0, [path])


@pytest.mark.parametrize(
    "make",
    [
        lambda: SR_dataset(_X2, _lib2c(), y=_Y),
        lambda: _task_dataset(
            samplers=[UniformSampling(0, 1), UniformSampling(2, 3)],
            constant_bounds=(-2, 2),
            reference_grammar=_grammar(),
        ),
    ],
    ids=["minimal", "rich"],
)
def test_evaluator_describes_task_like_dataset(make):
    ds = make()
    ev = ds.create_evaluator()
    assert ev.background_knowledge.to_dict() == ds.background_knowledge.to_dict()
    assert ev.describe() == ds.describe()
    assert ev.priors() == ds.priors()
