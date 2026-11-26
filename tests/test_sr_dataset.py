"""Tests for SRToolkit.dataset.sr_dataset."""

import json
import shutil
import signal
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pytest
from strategies import (
    EventLog,
    ScriptedApproach,
    SignalAfter,
    StatelessSteppedApproach,
    SteppedApproach,
    result_answers,
    result_dict,
)

from SRToolkit.approaches import ProGED
from SRToolkit.approaches.sr_approach import ApproachConfig, SR_approach
from SRToolkit.dataset import data_cache
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.data_split import DataSplit
from SRToolkit.dataset.sampling import LogUniformSampling, UniformSampling
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.callbacks import SR_callback, TimeLimitCallback
from SRToolkit.evaluation.model_store import ModelStore
from SRToolkit.evaluation.results import SR_results
from SRToolkit.utils import units as u
from SRToolkit.utils.expression_constraints import (
    ConsistentUnits,
    MaxConstants,
    MaxTokens,
    MaxTreeDepth,
    NoNesting,
    OutputBounds,
    RequiredVariables,
)
from SRToolkit.utils.expression_simplifier import DefaultRules
from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.grammar.constraints import MaxDepth
from SRToolkit.utils.grammar.grammar import Grammar, Rule
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.units import Unit

_FIX2 = Constant("C_0", value=2.0, fixed=True)
_SPEC_C0 = Constant("C_0", prior=UniformSampling(0, 1))
_SPEC_C1 = Constant("C_1", value=1.0)


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

    @pytest.mark.parametrize("const_symbol, expected", [("C", True), (None, False)], ids=["with-C", "without-C"])
    def test_contains_constants(self, const_symbol, expected):
        lib = SymbolLibrary.from_symbol_list(["+"], num_variables=1, const_symbol=const_symbol)
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
        assert a.evaluate_expr(["X_0", "*", "C"]).error == b.evaluate_expr(["X_0", "*", "C"]).error

    @pytest.mark.parametrize("dataset_seed, expected", [(None, 0), (4, 4)], ids=["no-seed-zero", "dataset-seed"])
    def test_bed_ranking_seed_is_dataset_seed(self, dataset_seed, expected):
        assert _bed_dataset(seed=dataset_seed).create_evaluator(seed=1).ranking.seed == expected

    @pytest.mark.parametrize("resample, same", [(False, True), (True, False)], ids=["fixed", "resample"])
    def test_rmse_score_same_across_run_seeds_unless_resampled(self, resample, same):
        # One optimizer iteration on a multimodal fit leaves the score dependent on the start.
        X = np.random.default_rng(0).random((10, 1))
        dataset = SR_dataset(
            X, SymbolLibrary.default_symbols(1), y=np.sin(7 * X[:, 0]), seed=4, ranking_resample=resample, max_iter=1
        )
        expr = ["sin", "(", "C", "*", "X_0", ")"]
        a, b = dataset.create_evaluator(seed=1), dataset.create_evaluator(seed=2)
        assert (a.evaluate_expr(expr).error == b.evaluate_expr(expr).error) is same

    @pytest.mark.parametrize(
        "dataset_seed, kwargs, expected",
        [
            (4, {}, 4),
            (None, {}, 0),
            (4, {"ranking_resample": True}, None),
            (4, {"ranking_seed": 7}, 7),
            (None, {"ranking_seed": 7}, 7),
        ],
        ids=["dataset-seed", "no-seed-zero", "resample-none", "explicit-wins", "explicit-wins-no-seed"],
    )
    def test_ranking_settings_ranking_seed(self, dataset_seed, kwargs, expected):
        X = np.random.default_rng(0).random((5, 1))
        dataset = SR_dataset(X, SymbolLibrary.default_symbols(1), y=X[:, 0], seed=dataset_seed, **kwargs)
        assert dataset.ranking_settings().get("ranking_seed") == expected

    def test_ranking_resample_uses_run_seed(self):
        dataset = _bed_dataset(num_points_sampled=8, ranking_resample=True)
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
        error = evaluator.evaluate_expr(["X_0", "+", "X_1"]).error
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

    # Reason: clause (resample/from_samplers Raises: "a free constant without a fixed value in
    # ground_truth_constants"). The message names the constants that have no fixed value.
    @pytest.mark.parametrize(
        "ground_truth, specs, named",
        [
            (["X_0", "*", "C"], None, "C"),
            (tokens_to_tree(["X_0", "*", "C_1"], SymbolLibrary.default_symbols(1)), None, "C_1"),
            (["X_0", "*", "C_0"], [Constant("C_0", value=1.0)], "C_0"),
            (["C_0", "*", "X_0", "+", "C"], [_FIX2], "C"),
        ],
        ids=["bare", "tree-with-label", "spec-not-fixed", "fixed-plus-bare"],
    )
    def test_resample_free_constant_raises(self, ground_truth, specs, named):
        # A free constant has no value to evaluate against during generation (issue #4: the compiled
        # backend used to read past an empty constant array and produce garbage).
        with pytest.raises(ValueError, match=rf"free constant \({named}\) without a fixed value"):
            SR_dataset.from_samplers(
                ground_truth=ground_truth,
                samplers=[UniformSampling(0.5, 5.0)],
                n_samples=16,
                seed=1,
                ground_truth_constants=specs,
            )


class TestGroundTruthConstants:
    # Reason: clause (attribute: "the specs as a tuple sorted by label, or None"; stored as given,
    # an empty iterable gives None). The specs are compared by to_dict.
    @pytest.mark.parametrize(
        "make_specs, expected",
        [
            (lambda: [_SPEC_C1, _SPEC_C0], (_SPEC_C0, _SPEC_C1)),
            (lambda: (c for c in [_SPEC_C1, _SPEC_C0]), (_SPEC_C0, _SPEC_C1)),
            (lambda: [], None),
            (lambda: None, None),
        ],
        ids=["unsorted-list", "generator", "empty", "none"],
    )
    def test_stored_sorted_or_none(self, make_specs, expected):
        ds = SR_dataset(
            _X2, _lib2c(), y=_Y, ground_truth=["C_0", "*", "X_0", "+", "C_1"], ground_truth_constants=make_specs()
        )
        # Same objects as given (stored as given, not resolved against the default).
        assert ds.ground_truth_constants == expected

    # Reason: clause (Raises: "given without ground_truth or does not fit it", see resolve_constants).
    # The message names the user's argument, ground_truth_constants, never resolve_constants'
    # free_constants, and says what is wrong.
    @pytest.mark.parametrize(
        "ground_truth, specs, reason",
        [
            (None, [_FIX2], "needs a 'ground_truth'"),
            (["C_0", "*", "X_0"], [Constant("C_9", value=1.0)], r"\['C_9'\]"),
            (["C_0", "*", "X_0"], [_FIX2, Constant("C_0", value=1.0)], "share the label 'C_0'"),
            (["C_0", "*", "X_0"], [Constant(prior=UniformSampling(0, 1))], "needs a label"),
            (["C_0", "*", "X_0"], [Constant("C_0", value=9.0)], "outside 'bounds'"),
        ],
        ids=["no-ground-truth", "unknown-label", "duplicate-label", "unlabelled", "value-outside-default-bounds"],
    )
    def test_errors(self, ground_truth, specs, reason):
        with pytest.raises(ValueError, match=reason) as info:
            SR_dataset(_X2, _lib2c(), y=_Y, ground_truth=ground_truth, ground_truth_constants=specs)
        assert "'ground_truth_constants'" in str(info.value)
        assert "free_constants" not in str(info.value)

    # Reason: clause (Constant: "a token the symbol library registers as a symbol (e.g. a named
    # constant C_1) is never a labelled constant"). With C_1 a named constant of value 3, a spec for it
    # does not fit, and y = x + 3 needs no spec.
    def test_named_constant_shaped_like_a_label(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "3.0")
        with pytest.raises(ValueError, match=r"\['C_1'\]"):
            SR_dataset(
                np.zeros((3, 1)), lib, y=np.zeros(3), ground_truth=["X_0", "+", "C_1"],
                ground_truth_constants=[Constant("C_1", value=1.0, fixed=True)],
            )  # fmt: skip
        ds = SR_dataset.from_samplers(["X_0", "+", "C_1"], [UniformSampling(0, 1)], symbol_library=lib, n_samples=5)
        np.testing.assert_allclose(ds.y, ds.X[:, 0] + 3.0)

    # Reason: clause (ranking_settings "plus ground_truth_constants if it has them"; RMSE generates y
    # with the fixed values). y = 2x + 2, so C*X_0 + C fits exactly with parameters (2, 2).
    def test_ranking_settings_carry_them_to_the_evaluator(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        ds = SR_dataset(
            X, SymbolLibrary.default_symbols(1), ground_truth=["C_0", "*", "X_0", "+", "C_0"],
            ground_truth_constants=[_FIX2],
        )  # fmt: skip
        assert ds.ranking_settings()["ground_truth_constants"] == ds.ground_truth_constants
        ev = ds.create_evaluator(seed=0)
        assert ev.evaluate_expr(["C", "*", "X_0", "+", "C"]).error == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(ev.models[-1].parameters, [2.0, 2.0], atol=1e-5)

    # Reason: clause (from_samplers / resample compute y with the fixed values; to_dict writes the specs
    # and from_dict regenerates y from them). y = 2.5x + 2.5, one value at both occurrences.
    def test_y_uses_fixed_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "a")
        ds = SR_dataset.from_samplers(
            ["C_0", "*", "X_0", "+", "C_0"],
            [UniformSampling(1, 2)],
            n_samples=10,
            seed=0,
            ground_truth_constants=[Constant("C_0", value=2.5, fixed=True)],
            benchmark="b",
            version="1",
        )
        np.testing.assert_allclose(ds.y, 2.5 * ds.X[:, 0] + 2.5)
        X_new, y_new = ds.resample(5, seed=1)
        np.testing.assert_allclose(y_new, 2.5 * X_new[:, 0] + 2.5)
        d = ds.to_dict()
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "b")
        restored = SR_dataset.from_dict(d)
        np.testing.assert_allclose(restored.y, 2.5 * restored.X[:, 0] + 2.5)
        np.testing.assert_array_equal(restored.X, ds.X)

    # Reason: boundary (D14: format-4 files saved before the key existed still load).
    def test_format4_without_key_loads_as_none(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        del d["ground_truth_constants"]
        assert SR_dataset.from_dict(d).ground_truth_constants is None


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

    def test_zero_budget_rejected(self):
        with pytest.raises(ValueError, match="max_evaluations"):
            SR_dataset(np.zeros((2, 1)), SymbolLibrary.default_symbols(1), max_evaluations=0)

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
        assert restored.create_evaluator().evaluate_expr(["C", "*", "X_0", "+", "X_1"]).error < 1e-6

    def test_node_ground_truth_serialised_with_dataset_library(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        lib = SymbolLibrary.from_symbol_list(["+", "*", "sin", "C"], num_variables=2)
        gt = tokens_to_tree(["sin", "(", "X_0", ")", "*", "X_1"], lib)
        ds = SR_dataset(np.zeros((3, 2)), lib, y=np.zeros(3), ground_truth=gt, benchmark="b", version="1")
        assert ds.to_dict()["ground_truth"] == gt.to_list(symbol_library=lib)


# ── Background knowledge, objective and format 4 (stage C) ────────────────────

_GOLDEN = Path(__file__).parent / "data" / "v3_cache"
# Golden v3 cache: generated on 2026-11-04 with the pre-stage-C data_cache.py / data_source.py
# (unchanged from commit 9bab9d0) by calling
#   data_cache.resolve("golden_bench", "1.0.0", "ds", config)
# with data_root patched to tests/data/v3_cache/cache and `config` = config_v3.json. It is what a
# user's cache written by an older version looks like: the .meta.json holds the old drift hash.
# On 2026-11-14 the samplers in config_v3.json were converted from the old sign flags to `mirror` and
# the .meta.json hash recomputed, since old-format sampler dicts are no longer read.

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
            "default_constant",
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
            (
                {"required_constraints": [ConsistentUnits({"X_0": u.m}, target_unit=u.s)]},
                ["X_0"],
                "rejected",
                "ConsistentUnits: result has unit m, target s",
            ),
        ],
        ids=["default-limit", "no-limit", "listed-limit", "consistent-units"],
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
        assert ev.evaluate_expr(["X_0", "+", "X_1"]).error < 1e-9
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
        assert "ground_truth_constants" not in ev.kwargs
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
            "ground_truth_constants", "data_source", "test", "aux", "background_knowledge", "required_constraints",
        }  # fmt: skip
        assert d["format_version"] == 4
        assert d["required_constraints"] == [MaxConstants(8).to_dict()]
        assert d["background_knowledge"] == ds.background_knowledge.to_dict()
        assert d["kwargs"] == {}
        assert (d["seed"], d["dataset_metadata"]) == (3, {"citation": "x"})
        assert d["ground_truth"] is None
        assert d["ground_truth_constants"] is None

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
            ground_truth=["X_0", "+", "C_0", "+", "C"],
            ground_truth_constants=[Constant("C_0", prior=UniformSampling(1, 2), bounds=(0, float("inf")))],
            default_constant=Constant(prior=LogUniformSampling(0.1, 10), bounds=(0, float("inf"))),
            max_evaluations=500,
            original_equation="F = x + c",
            success_threshold=0.5,
            seed=3,
            dataset_metadata={"citation": "x"},
        )
        d = json.loads(json.dumps(ds.to_dict()))
        assert d["kwargs"] == {"method": "L-BFGS-B"}
        assert d["ground_truth_constants"] == [ds.ground_truth_constants[0].to_dict()]
        restored = SR_dataset.from_dict(d)
        assert restored.to_dict() == d
        assert restored.background_knowledge.to_dict() == ds.background_knowledge.to_dict()
        path = tmp_path / "ds.json"
        path.write_text(json.dumps(d))
        assert SR_dataset.from_dict(path).to_dict() == d

    # Reason: clause (RankingSettings simplify_*: stored as dataset kwargs, the rules saved with their
    # to_dict, and read by the evaluator): rules without "powers" keep X_0*X_0 after a JSON round trip.
    def test_simplify_rules_round_trip_and_reach_evaluator(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        rules = DefaultRules(exclude=["powers"])
        ds = SR_dataset(
            _X2, SymbolLibrary.default_symbols(2), y=_Y, benchmark="b", version="1",
            simplify_rules=rules, simplify_max_iterations=5, simplify_fold_aware=False, fold_constants="keep",
        )  # fmt: skip
        d = json.loads(json.dumps(ds.to_dict()))
        assert d["kwargs"] == {
            "simplify_rules": rules.to_dict(), "simplify_max_iterations": 5, "simplify_fold_aware": False,
            "fold_constants": "keep",
        }  # fmt: skip
        restored = SR_dataset.from_dict(d)
        assert restored.to_dict() == d
        se = restored.create_evaluator()
        se.evaluate_expr(["X_0", "*", "X_0"], simplify_expr=True)
        assert se.models[0].expr == ["X_0", "*", "X_0"]
        se = restored.create_evaluator()
        se.evaluate_expr(["C", "*", "C", "*", "X_1"])  # folded with fold_constants
        assert se.models[0].expr == ["C", "*", "X_1"]

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
        # The golden file predates unit rules; loading fills in the predefined ones.
        assert ds.symbol_library.to_dict() == _lib2c().to_dict()
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
            {"X_1": UniformSampling(10, 11), "X_0": UniformSampling(0, 1)},
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
        self.adapt_args = []
        self.loaded_from = []

    @property
    def adaptation_scope(self):
        return self._scope

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
            ("never", False, ["search", "search"]),
            ("once", False, ["adapt", "search", "search"]),
            ("experiment", True, ["adapt", "search", "adapt", "search"]),
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

    def test_evaluate_approach_once_adapts_or_loads_through_path(self, tmp_path):
        # caller wiring: "once" goes through adapt_or_load with adaptation_path (pinned in TestAdaptOrLoad)
        path = str(tmp_path / "state")
        first = _AdaptRecorder("once")
        _task_dataset().evaluate_approach(first, num_experiments=2, verbose=False, adaptation_path=path)
        assert first.events == ["adapt", "save", "search", "search"]

        second = _AdaptRecorder("once")
        _task_dataset().evaluate_approach(second, num_experiments=2, verbose=False, adaptation_path=path)
        assert second.events == ["load", "search", "search"]
        assert second.loaded_from == [path]


class _AdaptOnly(SR_approach):
    """Implements ``adapt`` but keeps the base ``save_adapted_state`` / ``load_adapted_state``."""

    def __init__(self):
        super().__init__(ApproachConfig(name="adapt-only"))
        self.adapted = 0

    def adapt(self, X, background_knowledge, required_constraints):
        self.adapted += 1

    def search(self, sr_evaluator, seed=None):
        pass


class TestAdaptOrLoad:
    @pytest.mark.parametrize(
        "rel_path, existing, force, events, content",
        [
            (None, False, False, ["adapt"], None),
            ("a/b/state", False, False, ["adapt", "save"], "state"),
            ("state", True, False, ["load"], "OLD"),
            ("state", True, True, ["adapt", "save"], "state"),
        ],
        ids=["no-path", "missing-file", "existing-file", "existing-force"],
    )
    def test_adapt_or_load_calls_per_path_state(self, tmp_path, rel_path, existing, force, events, content):
        # scope "never": adapt_or_load ignores adaptation_scope
        approach = _AdaptRecorder("never")
        path = None if rel_path is None else str(tmp_path / rel_path)
        if existing:
            Path(path).write_text("OLD")
        _task_dataset().adapt_or_load(approach, path, force=force)
        assert approach.events == events
        assert approach.loaded_from == ([path] if "load" in events else [])
        if path is None:
            assert list(tmp_path.iterdir()) == []
        else:
            assert Path(path).read_text() == content

    def test_adapt_or_load_saves_bare_file_name_in_working_directory(self, tmp_path, monkeypatch):
        # boundary: a path with no parent directory (nothing to create)
        monkeypatch.chdir(tmp_path)
        approach = _AdaptRecorder("never")
        _task_dataset().adapt_or_load(approach, "state")
        assert approach.events == ["adapt", "save"]
        assert (tmp_path / "state").read_text() == "state"

    @pytest.mark.parametrize(
        "existing, adapted",
        [(True, 0), (False, 1)],
        ids=["existing-file-load", "missing-file-save"],
    )
    def test_adapt_or_load_without_state_io_raises_when_path_set(self, tmp_path, existing, adapted):
        # the base methods raise with no message: whether adapt ran tells load (0) from save (1)
        approach = _AdaptOnly()
        path = str(tmp_path / "state")
        if existing:
            Path(path).write_text("OLD")
        with pytest.raises(NotImplementedError):
            _task_dataset().adapt_or_load(approach, path)
        assert approach.adapted == adapted

    def test_adapt_or_load_without_state_io_no_path_adapts(self):
        approach = _AdaptOnly()
        _task_dataset().adapt_or_load(approach, None)
        assert approach.adapted == 1


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


# ── Named constants and units: not lost on the way to the evaluator or through a save ──

_PI = {"pi": VariableInfo("π", unit=Unit())}
_UNIT_INFOS = dict(
    variables={"X_0": VariableInfo("x", unit=u.m)}, target=VariableInfo("y", unit=u.s), named_constants=_PI
)


@pytest.mark.parametrize(
    "make",
    [
        lambda lib: SR_dataset(_X2, lib, y=_Y, benchmark="b", version="1", **_UNIT_INFOS),
        lambda lib: SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 1), UniformSampling(0, 1)],
            symbol_library=lib,
            n_samples=5,
            seed=0,
            benchmark="b",
            version="1",
            **_UNIT_INFOS,
        ),
    ],
    ids=["constructor", "from_samplers"],
)
def test_constants_and_units_survive(make, tmp_path, monkeypatch):
    monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
    ds = make(SymbolLibrary.from_symbol_list(["+", "*", "C", "pi"], num_variables=2))
    restored = SR_dataset.from_dict(json.loads(json.dumps(ds.to_dict())))
    bks = [ds.background_knowledge, ds.create_evaluator().background_knowledge, restored.background_knowledge]
    assert [bk.named_constants for bk in bks] == [_PI] * 3
    assert [(bk.variables["X_0"].unit, bk.target.unit) for bk in bks] == [(u.m, u.s)] * 3


class TestDefaultConstant:
    # Reason: clause (default_constant "as for BackgroundKnowledge"; the default_constant property,
    # complete; ranking_settings() includes it; create_evaluator uses it; ground_truth_constants are
    # checked against it: value 9 is outside the uniform default's (-2, 3) but inside (0, inf)).
    @pytest.mark.parametrize(
        "make",
        [
            lambda: SR_dataset(
                _X2,
                _lib2c(),
                y=_Y,
                ground_truth=["C_0", "*", "X_0"],
                ground_truth_constants=[Constant("C_0", value=9.0)],
                constant_bounds=(-2, 3),
                default_constant=Constant(bounds=(0, float("inf"))),
                benchmark="b",
                version="1",
            ),
            lambda: SR_dataset.from_samplers(
                ["X_0"],
                [UniformSampling(0, 1)],
                n_samples=5,
                constant_bounds=(-2, 3),
                default_constant=Constant(bounds=(0, float("inf"))),
                benchmark="b",
                version="1",
            ),
        ],
        ids=["init", "from_samplers"],
    )
    def test_default_constant_reaches_dataset(self, make, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = make()
        expected = Constant(prior=UniformSampling(-2.0, 3.0), bounds=(0, float("inf"))).to_dict()
        assert ds.default_constant.to_dict() == expected
        assert ds.background_knowledge.default_constant.to_dict() == expected
        assert ds.ranking_settings()["default_constant"].to_dict() == expected
        assert ds.create_evaluator().default_constant.to_dict() == expected
        assert "default_constant" not in ds.to_dict()["kwargs"]


# ── Checkpoints in evaluate_approach (stage 2b) ───────────────────────────────


def _ckpt_dataset(scale=2.0, seed=None):
    X = np.array([[1.0], [2.0], [3.0]])
    return SR_dataset(X, SymbolLibrary.default_symbols(1), y=scale * X[:, 0], dataset_name="ds", seed=seed)


def _without_wall_time(results):
    return [result_dict(r) for r in results.results]


def _files(root):
    return sorted(p.name for p in root.rglob("*") if p.is_file())


def _journal_mode(path):
    with sqlite3.connect(path) as con:
        return con.execute("PRAGMA journal_mode").fetchone()[0]


class _PerSeed(SR_callback):
    # Counts the evaluation events of each experiment and records how each one ended.
    def __init__(self):
        self.counts = {}
        self.ended = []

    def on_experiment_start(self, event):
        self.current = event.seed
        self.counts.setdefault(event.seed, 0)

    def on_expr_evaluated(self, event):
        self.counts[self.current] += 1

    def on_experiment_end(self, event, results):
        self.ended.append((event.seed, results.stop_reason))


def _evaluate(approach, checkpoint_dir=None, num_experiments=3, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _ckpt_dataset().evaluate_approach(
            approach,
            num_experiments=num_experiments,
            initial_seed=10,
            verbose=False,
            checkpoint_dir=checkpoint_dir,
            **kwargs,
        )


class TestEvaluateApproachCheckpoints:
    def test_create_evaluator_passes_checkpoint_and_resume(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _ckpt_dataset().create_evaluator(seed=0, checkpoint=path)
        se.evaluate_expr(["X_0"])
        se.save_snapshot(b"state")
        se.close()
        with pytest.raises(FileExistsError):
            _ckpt_dataset().create_evaluator(seed=0, checkpoint=path)
        se = _ckpt_dataset().create_evaluator(seed=0, checkpoint=path, resume=True, snapshot_interval=60.0)
        assert (se.resumed_from.state, list(se.trace), se.snapshot_interval) == (b"state", [0], 60.0)

    @pytest.mark.parametrize("kwargs, expected", [({}, 300.0), ({"snapshot_interval": None}, None)])
    def test_evaluate_approach_passes_snapshot_interval(self, tmp_path, kwargs, expected):
        approach = SteppedApproach(stop_after_asks=1)
        _evaluate(approach, tmp_path, num_experiments=1, **kwargs)
        assert approach.evaluator.snapshot_interval == expected

    def test_finished_experiments_delete_checkpoints(self, tmp_path):
        root = tmp_path / "ck" / "sub"  # created if missing
        results = _evaluate(ScriptedApproach(), root, num_experiments=2)
        assert root.is_dir() and _files(root) == []
        assert _without_wall_time(results) == _without_wall_time(_evaluate(ScriptedApproach(), num_experiments=2))

    def test_keep_checkpoint_keeps_compacted_files(self, tmp_path):
        results = _evaluate(ScriptedApproach(), tmp_path, num_experiments=2, keep_checkpoint=True)
        path = tmp_path / "ds" / "scripted" / "exp_10.sqlite"
        assert _files(tmp_path) == ["exp_10.sqlite", "exp_11.sqlite"]  # no -wal or -shm
        assert _journal_mode(path) == "delete"
        assert SR_results.from_checkpoint(path)[0].trace == results[0].trace
        se = _ckpt_dataset().create_evaluator(seed=10, checkpoint=path, resume=True)
        assert _journal_mode(path) == "wal"  # resuming switches back to WAL
        se.close()

    def test_interrupt_returns_partial_results_and_keeps_checkpoints(self, tmp_path):
        approach, recorder, results = ScriptedApproach(kill_seed=11), _PerSeed(), SR_results()
        with pytest.warns(UserWarning, match="exp_11.sqlite"):
            returned = _ckpt_dataset().evaluate_approach(
                approach,
                3,
                initial_seed=10,
                results=results,
                callbacks=recorder,
                verbose=False,
                checkpoint_dir=tmp_path,
            )
        assert returned is results  # appended to
        assert approach.seeds == [10, 11]  # experiment 12 is not run
        assert [r.stop_reason for r in results.results] == ["returned", "interrupted"]
        assert results[1].trace == [0, 1, 0]
        assert recorder.ended == [(10, "returned"), (11, "interrupted")]
        assert _files(tmp_path) == ["exp_10.sqlite", "exp_11.sqlite"]  # all kept

    def test_rerun_with_resume_equals_uninterrupted_run(self, tmp_path):
        # Interrupted in the second step of experiment 11; that step is completed, so 4 calls are saved.
        _evaluate(SteppedApproach(stop_after_asks=3), tmp_path, callbacks=SignalAfter(3, seed=11))
        recorder = _PerSeed()
        results = _evaluate(SteppedApproach(stop_after_asks=3), tmp_path, resume=True, callbacks=recorder)
        assert _without_wall_time(results) == _without_wall_time(_evaluate(SteppedApproach(stop_after_asks=3)))
        assert recorder.counts == {11: 2, 12: 6}  # 10 is read from its checkpoint; 11 continues from its snapshot
        assert _files(tmp_path) == []

    @pytest.mark.parametrize(
        "make",
        [ScriptedApproach, lambda: StatelessSteppedApproach(stop_after_asks=3)],
        ids=["search-only", "stateless"],
    )
    def test_rerun_without_snapshots_restarts_the_interrupted_experiment(self, tmp_path, make):
        first = _evaluate(make(), tmp_path, callbacks=SignalAfter(2, seed=11))
        recorder = _PerSeed()
        with pytest.warns(UserWarning, match="exp_11.sqlite"):
            results = _ckpt_dataset().evaluate_approach(
                make(), 3, initial_seed=10, verbose=False, checkpoint_dir=tmp_path, resume=True, callbacks=recorder
            )
        assert list(recorder.counts) == [11, 12]  # 10 is read from its checkpoint, 11 starts afresh
        assert _without_wall_time(results) == _without_wall_time(_evaluate(make()))
        assert results[0].wall_time == first[0].wall_time  # saved in the checkpoint

    def test_search_exception_keeps_checkpoints(self, tmp_path):
        with pytest.raises(RuntimeError, match="boom"):
            _evaluate(ScriptedApproach(raise_seed=11), tmp_path)
        assert _files(tmp_path) == ["exp_10.sqlite", "exp_11.sqlite"]
        saved = SR_results.from_checkpoint(tmp_path / "ds" / "scripted" / "exp_11.sqlite")[0]
        assert (saved.trace, saved.stop_reason) == ([0, 1, 0], "interrupted")  # committed and closed

    # Reason: clause (keep_all_models=False for every experiment; with keep_checkpoint the compacted
    # checkpoint holds all models, at the indices of the incumbents).
    def test_evaluate_approach_compact_results(self, tmp_path):
        # top_k=1: the smallest allowed; each experiment evaluates 5 distinct models.
        results = _evaluate(ScriptedApproach(), tmp_path, 2, keep_checkpoint=True, keep_all_models=False, top_k=1)
        full = _evaluate(ScriptedApproach(), None, 2, top_k=1)
        assert [r.summary is not None for r in results] == [True, True]
        assert [r.all_models for r in results] == [r.top_models(1) for r in full]
        np.testing.assert_equal([result_answers(r, 1) for r in results], [result_answers(r, 1) for r in full])
        store = results[1].open_checkpoint()
        try:
            assert list(store.trace) == full[1].trace
            incumbents = results[1].summary.incumbents
            assert [store.models[i] for _, i, _ in incumbents] == [m for _, _, m in incumbents]
        finally:
            store.close()

    # Reason: clause (keep_all_models applies to experiments read from checkpoints too).
    def test_evaluate_approach_compact_results_read_from_checkpoints(self, tmp_path):
        _evaluate(ScriptedApproach(kill_seed=11), tmp_path, 2)  # experiment 10 finished, 11 interrupted
        results = _evaluate(ScriptedApproach(), tmp_path, 2, resume=True, keep_all_models=False, top_k=1)
        full = _evaluate(ScriptedApproach(), None, 2, top_k=1)
        assert [r.summary is not None for r in results] == [True, True]
        np.testing.assert_equal(result_answers(results[0], 1), result_answers(full[0], 1))

    # Reason: clause (a result refers to its checkpoint exactly when the file remains: kept or
    # interrupted; never without a checkpoint_dir or once it is deleted). A callback sees the
    # reference only if the file is sure to stay (kept, or this experiment interrupted).
    @pytest.mark.parametrize(
        "approach, use_dir, keep, expected, seen",
        [
            (ScriptedApproach, False, False, [None, None], [None, None]),
            (ScriptedApproach, True, False, [None, None], [None, None]),
            (ScriptedApproach, True, True, ["exp_10.sqlite", "exp_11.sqlite"], ["exp_10.sqlite", "exp_11.sqlite"]),
            (
                lambda: ScriptedApproach(kill_seed=11),
                True,
                False,
                ["exp_10.sqlite", "exp_11.sqlite"],
                [None, "exp_11.sqlite"],
            ),
        ],
        ids=["in-memory", "deleted", "kept", "interrupted"],
    )
    def test_evaluate_approach_checkpoint_reference(self, tmp_path, approach, use_dir, keep, expected, seen):
        EventLog.references.clear()
        results = _evaluate(approach(), tmp_path if use_dir else None, 2, keep_checkpoint=keep, callbacks=EventLog())
        root = tmp_path / "ds" / "scripted"
        assert [r.checkpoint and r.checkpoint["path"] for r in results] == [
            name and str(root / name) for name in expected
        ]
        assert [r and r["path"] for r in EventLog.references] == [name and str(root / name) for name in seen]

    @pytest.mark.parametrize("use_dir", [False, True], ids=["in-memory", "checkpoint-dir"])
    def test_signal_handlers_only_with_checkpoint_dir(self, tmp_path, use_dir):
        approach = ScriptedApproach()
        _evaluate(approach, tmp_path if use_dir else None, num_experiments=1)
        assert (approach.handlers[0] is signal.getsignal(signal.SIGINT)) is not use_dir

    @pytest.mark.parametrize(
        "make, prepare, kwargs, error, match",
        [
            (ScriptedApproach, lambda path: None, {"initial_seed": None}, ValueError, "seed"),
            (
                ScriptedApproach,
                lambda path: _evaluate(ScriptedApproach(), path, keep_checkpoint=True),
                {},
                FileExistsError,
                None,
            ),
            (  # continuing the interrupted experiment 11 with other data
                SteppedApproach,
                lambda path: _evaluate(SteppedApproach(stop_after_asks=3), path, callbacks=SignalAfter(3, seed=11)),
                {"resume": True, "dataset": _ckpt_dataset(scale=3.0)},
                ValueError,
                r"\by\b",
            ),
            (ScriptedApproach, lambda path: None, {"keep_all_models": False, "top_k": 0}, ValueError, "top_k"),
        ],
        ids=["unseeded", "existing-checkpoint", "other-settings", "compact-top-k-0"],
    )
    def test_evaluate_approach_checkpoint_errors(self, tmp_path, make, prepare, kwargs, error, match):
        prepare(tmp_path)
        # Reading a checkpoint in WAL mode may add -wal / -shm files; the checkpoints themselves stay unchanged.
        before = {p: p.read_bytes() for p in tmp_path.rglob("*.sqlite")}
        args = {"initial_seed": 10, **kwargs}  # a copy: the parameter dict is shared between runs
        dataset = args.pop("dataset", _ckpt_dataset())
        approach = make()
        with pytest.raises(error, match=match):
            dataset.evaluate_approach(approach, 3, verbose=False, checkpoint_dir=tmp_path, **args)
        assert {p: p.read_bytes() for p in tmp_path.rglob("*.sqlite")} == before
        assert getattr(approach, "seeds", []) == []  # nothing was run


# --------------------------------------------------------------------------- #
# extend_run (checkpoint stage 6)
# --------------------------------------------------------------------------- #
def _limited(max_evaluations, success_threshold=-1.0, scale=2.0, **settings):
    X = np.array([[1.0], [2.0], [3.0]])
    return SR_dataset(
        X,
        SymbolLibrary.default_symbols(1),
        y=scale * X[:, 0],
        dataset_name="ds",
        max_evaluations=max_evaluations,
        success_threshold=success_threshold,
        initialization="mean",  # fitted constants do not depend on the random state
        **settings,
    )


def _first_run(tmp_path, approach, dataset, **kwargs):
    # One kept experiment seeded 10; returns its checkpoint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dataset.evaluate_approach(
            approach, 1, initial_seed=10, verbose=False, checkpoint_dir=tmp_path, keep_checkpoint=True, **kwargs
        )
    return tmp_path / "ds" / approach.name / "exp_10.sqlite"


def _extend(dataset, approach, path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return dataset.extend_run(approach, path, **kwargs)[0]


def _run_dict(result):
    return {**result_dict(result), "checkpoint": None}


def _straight(dataset, approach):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return dataset.evaluate_approach(approach, 1, initial_seed=10, verbose=False)[0]


class _Tick(SR_callback):
    # Advances the fake clock of TimeLimitCallback by one second per evaluation.
    now = [0.0]

    def on_expr_evaluated(self, event):
        _Tick.now[0] += 1.0


class _ExperimentEvents(SR_callback):
    # Records each experiment start, the experiment ids of the evaluations and how each experiment ended.
    def __init__(self):
        self.starts, self.ends, self.ids = [], [], set()

    def on_experiment_start(self, event):
        self.starts.append(
            (event.dataset_name, event.approach_name, event.max_evaluations, event.success_threshold, event.seed)
        )

    def on_expr_evaluated(self, event):
        self.ids.add(event.experiment_id)

    def on_experiment_end(self, event, results):
        self.ends.append((results.evaluation_calls, results.stop_reason))


class _Unsaveable(SR_callback):
    def to_dict(self):
        return {**super().to_dict(), "limit": {1}}


class _AdaptingStepped(SteppedApproach):
    # A SteppedApproach that adapts with the given scope and logs adapt / save / load and each ask.
    def __init__(self, scope="experiment", **kwargs):
        super().__init__(**kwargs)
        self._scope = scope
        self.events = []
        self.adapt_args = []

    @property
    def adaptation_scope(self):
        return self._scope

    def adapt(self, X, background_knowledge, required_constraints):
        self.events.append("adapt")
        self.adapt_args.append((X.tolist(), [c.to_dict() for c in required_constraints]))

    def save_adapted_state(self, path):
        self.events.append("save")
        Path(path).write_text("state")

    def load_adapted_state(self, path):
        self.events.append("load")

    def ask(self, sr_evaluator):
        self.events.append("ask")
        return super().ask(sr_evaluator)


class _FailingStepped(SteppedApproach):
    # A SteppedApproach whose ask raises once the search state reaches ``fail_at`` asks.
    def __init__(self, fail_at, **kwargs):
        super().__init__(**kwargs)
        self.fail_at = fail_at

    def ask(self, sr_evaluator):
        if self.asks == self.fail_at:
            raise RuntimeError("boom")
        return super().ask(sr_evaluator)


def _plain_file(path):
    other = path.with_name("notes.sqlite")
    other.write_text("not a checkpoint")
    return other


class TestExtendRun:
    # Reason: clause (an extended run equals a run that had the new limits from the start). Batches of
    # two with an odd budget, or a threshold met in a batch, cut the last step: the extension continues
    # from the snapshot before it (or, without one, from the seed). The new segment starts at the calls
    # of that snapshot (``kept``): ProGED evaluates one expression per step, so its last step is complete
    # (6); batches of two cut by a budget of 5 keep 4, or 0 without a snapshot; seeded 10, the threshold
    # 1.0 is first met by call 3 (X_0 + C), the first of the second batch, so 2 are kept.
    @pytest.mark.parametrize(
        "make, first, new, interval, kept",
        [
            (lambda: ProGED(), (6, -1.0), {"max_evaluations": 12}, 300.0, 6),
            (lambda: SteppedApproach(), (5, -1.0), {"max_evaluations": 10}, 1e-9, 4),
            (lambda: SteppedApproach(), (5, -1.0), {"max_evaluations": 10}, None, 0),
            (lambda: SteppedApproach(), (12, 1.0), {"success_threshold": 0.5}, 1e-9, 2),
        ],
        ids=["ProGED-budget", "cut-budget", "cut-without-snapshot", "cut-threshold"],
    )
    def test_extension_equals_straight_run(self, tmp_path, make, first, new, interval, kept):
        path = _first_run(tmp_path, make(), _limited(*first), snapshot_interval=interval)
        stopped = SR_results.from_checkpoint(path)[0]
        assert stopped.stop_reason == ("max_evaluations" if "max_evaluations" in new else "success_threshold")
        limits = (new.get("max_evaluations", first[0]), new.get("success_threshold", first[1]))
        extended = _extend(_limited(*first), make(), path, **new)
        straight = _straight(_limited(*limits), make())
        assert _run_dict(extended) == _run_dict(straight)
        assert extended.extended and not straight.extended
        segment = extended.limits[1]
        assert (segment["max_evaluations"], segment["success_threshold"]) == limits
        assert (segment["previous_stop_reason"], segment["callbacks"]) == (stopped.stop_reason, [])
        assert segment["calls"] == kept
        assert extended.checkpoint["path"] == str(path) and _journal_mode(path) == "delete"  # compacted

    # Reason: clause (an extension of a run that already meets its limits stops at once, without a call,
    # with "success_threshold" checked first; an approach that stopped the run itself stops it again at
    # its first ask; the new segment records no callbacks). "both": every expression has an error below
    # 100, so the first run stops after one call, and the budget 1 is then met too.
    @pytest.mark.parametrize(
        "make, first, new, reason",
        [
            (lambda: ProGED(), (6, -1.0), {}, "max_evaluations"),
            (lambda: SteppedApproach(batch=1), (20, 1.0), {}, "success_threshold"),
            (lambda: SteppedApproach(batch=1), (20, 100.0), {"max_evaluations": 1}, "success_threshold"),
            (
                lambda: SteppedApproach(batch=1, stop_after_asks=3, stop_with="exhausted"),
                (-1, -1.0),
                {"max_evaluations": 10},
                "exhausted",
            ),
        ],
        ids=["budget", "threshold", "both", "stops-itself"],
    )
    def test_run_meeting_its_limits_stops_at_once(self, tmp_path, make, first, new, reason):
        path = _first_run(tmp_path, make(), _limited(*first))
        stopped = SR_results.from_checkpoint(path)[0]
        counter = _PerSeed()
        extended = _extend(_limited(*first), make(), path, callbacks=counter, **new)
        assert (extended.trace, extended.stop_reason) == (stopped.trace, reason)
        assert counter.counts == {10: 0}
        assert (len(extended.limits), extended.limits[1]["callbacks"]) == (2, None)

    # Reason: clause (an extension that changes nothing adds no segment and is a plain resume; so a
    # repeated extension, or one after a segment that made no call, adds none). The first run's threshold
    # is an int: regression (code review), it hashed differently once read back as a float, so an
    # extension with no new limits raised.
    def test_extension_changing_nothing_adds_no_segment(self, tmp_path):
        path = _first_run(tmp_path, ProGED(), _limited(6, -1))
        assert _extend(_limited(6), ProGED(), path).limits == SR_results.from_checkpoint(path)[0].limits
        once = _extend(_limited(6), ProGED(), path, max_evaluations=12)
        twice = _extend(_limited(6), ProGED(), path, max_evaluations=12)
        assert (len(twice.limits), twice.trace, twice.stop_reason) == (2, once.trace, "max_evaluations")
        counter = _PerSeed()
        _extend(_limited(6), ProGED(), path, callbacks=counter)  # stops at once: a segment without callbacks
        assert len(_extend(_limited(6), ProGED(), path).limits) == 3

    # Reason: clause (interrupted runs are extended too; an interrupted extension continues with the
    # new limits, here through extend_run again).
    def test_interrupted_extension_continues(self, tmp_path):
        path = _first_run(tmp_path, SteppedApproach(), _limited(6), callbacks=SignalAfter(3))
        assert SR_results.from_checkpoint(path)[0].stop_reason == "interrupted"
        with pytest.warns(UserWarning, match="interrupted"):
            interrupted = _limited(6).extend_run(SteppedApproach(), path, max_evaluations=10, callbacks=SignalAfter(1))[
                0
            ]
        assert interrupted.stop_reason == "interrupted"
        extended = _extend(_limited(6), SteppedApproach(), path)  # None keeps the new budget
        assert _run_dict(extended) == _run_dict(_straight(_limited(10), SteppedApproach()))
        assert [s["previous_stop_reason"] for s in extended.limits] == [None, "interrupted", "interrupted"]

    # Reason: clause (a TimeLimitCallback with a longer limit counts on from the time saved with the
    # snapshot; callbacks are not checked at the start, so one whose limit is already used up stops the
    # run after one call).
    def test_longer_time_limit_continues_from_saved_time(self, tmp_path, monkeypatch):
        from SRToolkit.evaluation import callbacks

        _Tick.now[0] = 0.0
        monkeypatch.setattr(callbacks.time, "monotonic", lambda: _Tick.now[0])
        cbs = [_Tick(), TimeLimitCallback(3)]
        path = _first_run(tmp_path, SteppedApproach(batch=1), _limited(-1), callbacks=cbs, snapshot_interval=1e-9)
        assert SR_results.from_checkpoint(path)[0].evaluation_calls == 3
        _Tick.now[0] = 100.0  # time between the sessions does not count
        extended = _extend(_limited(-1), SteppedApproach(batch=1), path, callbacks=[_Tick(), TimeLimitCallback(5)])
        assert (extended.evaluation_calls, extended.stop_reason) == (5, "time_limit")
        assert [s["callbacks"][1]["time_limit"] for s in extended.limits] == [3, 5]
        # the saved elapsed time is 5 >= 5: the first call (one more second) stops the run
        again = _extend(
            _limited(-1), SteppedApproach(batch=1), path, max_evaluations=100, callbacks=[_Tick(), TimeLimitCallback(5)]
        )
        assert (again.evaluation_calls, again.stop_reason) == (6, "time_limit")

    # Reason: clause (callbacks get on_experiment_start with the new limits and the run's seed, and
    # on_experiment_end with the result; the evaluations carry the experiment id; top_k and
    # keep_all_models are as for get_results).
    def test_extension_reports_new_limits_to_callbacks(self, tmp_path):
        path = _first_run(tmp_path, ProGED(), _limited(6))
        events = _ExperimentEvents()
        extended = _extend(
            _limited(6),
            ProGED(),
            path,
            max_evaluations=12,
            success_threshold=-2.0,
            callbacks=events,
            top_k=1,
            keep_all_models=False,
        )
        assert events.starts == [("ds", "ProGED", 12, -2.0, 10)]
        assert (events.ends, events.ids) == ([(12, "max_evaluations")], {"ds_ProGED_10"})
        assert (len(extended.all_models), extended.summary is not None) == (1, True)

    # Reason: clause (the approach adapts first, as in evaluate_approach: "experiment" adapts again,
    # "once" loads the state saved at adaptation_path). A run of 6 calls in batches of two extended to 8
    # makes one ask.
    @pytest.mark.parametrize(
        "scope, expected, adapts",
        [("experiment", ["adapt", "ask"], 1), ("once", ["load", "ask"], 0)],
        ids=["experiment", "once"],
    )
    def test_extension_adapts_before_running(self, tmp_path, scope, expected, adapts):
        state = str(tmp_path / "adapted")
        path = _first_run(tmp_path, _AdaptingStepped(scope), _limited(6), adaptation_path=state)
        approach, dataset = _AdaptingStepped(scope), _limited(6)
        _extend(dataset, approach, path, max_evaluations=8, adaptation_path=state)
        assert approach.events == expected
        assert (
            approach.adapt_args == [(dataset.X.tolist(), [c.to_dict() for c in dataset.required_constraints])] * adapts
        )

    # Regression (dev-check): a callback whose config cannot be saved raised only after the approach had
    # adapted (and, for "once", saved its state).
    def test_unsaveable_callback_raises_before_adapting(self, tmp_path):
        path = _first_run(tmp_path, _AdaptingStepped(), _limited(6))
        approach = _AdaptingStepped()
        with pytest.raises(TypeError, match="set"):
            _limited(6).extend_run(approach, path, callbacks=_Unsaveable())
        assert approach.events == []

    # Reason: clause (after a failure the checkpoint keeps the new limits, and extend_run with None limits
    # continues the run); implied (snapshot_interval applies to the extension: with 1e-9 a snapshot is
    # due before every step, so the run of 6 calls extended to 12 keeps its snapshot at 10 calls when the
    # sixth ask raises).
    def test_failed_extension_continues(self, tmp_path):
        path = _first_run(tmp_path, SteppedApproach(), _limited(6))
        with pytest.raises(RuntimeError, match="boom"):
            _extend(_limited(6), _FailingStepped(5), path, max_evaluations=12, snapshot_interval=1e-9)
        store = ModelStore(path, "read")
        assert (store.snapshot.num_calls, len(store.get_header("limits"))) == (10, 2)
        store.close()
        extended = _extend(_limited(6), SteppedApproach(), path)
        assert _run_dict(extended) == _run_dict(_straight(_limited(12), SteppedApproach()))

    # Reason: boundary (an extension does not shorten a run: a budget below the calls the run made raises,
    # one equal to them is accepted). Batches of two with a budget of 5 make 5 calls, the snapshot keeps 4,
    # so the budget 5 (with a new callback, so that the extension is not a no-op) redoes the fifth call
    # and stops there.
    def test_budget_cannot_shorten_the_run(self, tmp_path):
        path = _first_run(tmp_path, SteppedApproach(), _limited(5), snapshot_interval=1e-9)
        stopped = SR_results.from_checkpoint(path)[0]
        with pytest.raises(ValueError, match="below the 5 calls"):
            _limited(5).extend_run(SteppedApproach(), path, max_evaluations=4)
        counter = _PerSeed()
        extended = _extend(_limited(5), SteppedApproach(), path, max_evaluations=5, callbacks=counter)
        assert (extended.trace, extended.stop_reason, counter.counts) == (stopped.trace, "max_evaluations", {10: 1})

    # Reason: clause (validation happens before anything is changed).
    @pytest.mark.parametrize(
        "approach, dataset, target, kwargs, error, match",
        [
            (StatelessSteppedApproach(), _limited(6), None, {}, ValueError, "snapshot"),
            (SteppedApproach(), _limited(6), None, {"max_evaluations": 4}, ValueError, "below the 6 calls"),
            (SteppedApproach(), _limited(6), None, {"max_evaluations": 0}, ValueError, "positive"),
            (SteppedApproach(), _limited(6), None, {"max_evaluations": True}, TypeError, "int"),
            (SteppedApproach(), _limited(6), None, {"success_threshold": "0.1"}, TypeError, "success_threshold"),
            (SteppedApproach(), _limited(6), None, {"success_threshold": True}, TypeError, "success_threshold"),
            (SteppedApproach(), _limited(6, scale=3.0), None, {}, ValueError, "other settings than this dataset's: y"),
            (SteppedApproach(), _limited(6, tol=1e-3), None, {}, ValueError, "other settings than this dataset's: tol"),
            (SteppedApproach(), _limited(6), None, {"keep_all_models": False, "top_k": 0}, ValueError, "top_k"),
            (SteppedApproach(), _limited(6), None, {"callbacks": _Unsaveable()}, TypeError, "set"),
            (SteppedApproach(), _limited(6), _plain_file, {}, ValueError, "checkpoint"),
            (
                SteppedApproach(),
                _limited(6),
                lambda p: p.with_name("missing.sqlite"),
                {},
                FileNotFoundError,
                "missing.sqlite",
            ),
        ],
        ids=[
            "no-snapshots",
            "below-calls",
            "zero",
            "bool",
            "threshold-str",
            "threshold-bool",
            "other-data",
            "extra-setting",
            "top_k",
            "unsaveable-callback",
            "not-a-checkpoint",
            "missing-file",
        ],
    )
    def test_invalid_extension_changes_nothing(self, tmp_path, approach, dataset, target, kwargs, error, match):
        path = _first_run(tmp_path, SteppedApproach(), _limited(6))
        path = target(path) if target else path
        before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
        with pytest.raises(error, match=match):
            dataset.extend_run(approach, path, **kwargs)
        assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
