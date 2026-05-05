import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from SRToolkit.evaluation.callbacks import (
    CallbackDispatcher,
    EarlyStoppingCallback,
    SRCallbacks,
)
from SRToolkit.evaluation.sr_evaluator import ResultAugmenter, SR_evaluator, SR_results
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EvalResult, ModelResult

# ── Shared data ────────────────────────────────────────────────────────────────

_X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
_y = np.array([3.0, 0.0, 3.0, 11.0])  # y = 2*X_1 - X_0; optimal: ["C","*","X_1","-","X_0"]

_X_BED = np.random.default_rng(0).random((20, 2)) - 0.5


# ── Concrete helpers ──────────────────────────────────────────────────────────


class _SimpleCb(SRCallbacks):
    pass


class _TestAugmenter(ResultAugmenter):
    def write_results(self, results: EvalResult) -> None:
        results.add_augmentation(self.name, {"value": 42}, self._type)

    def to_dict(self, base_path: str, name: str) -> dict:
        return {"name": self.name}


class _FailingAugmenter(ResultAugmenter):
    def write_results(self, results: EvalResult) -> None:
        raise RuntimeError("intentional failure")

    def to_dict(self, base_path: str, name: str) -> dict:
        return {}


def _make_se(**kwargs):
    defaults = dict(seed=42, success_threshold=-1)
    defaults.update(kwargs)
    return SR_evaluator(_X, _y, **defaults)


def _single_result(approach_name="", top_k=5):
    se = _make_se()
    se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
    return se.get_results(approach_name=approach_name, top_k=top_k)


def _two_result(approach_name="", top_k=5):
    se = _make_se()
    se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
    se.evaluate_expr(["X_0", "+", "X_1"])
    return se.get_results(approach_name=approach_name, top_k=top_k)


# ── ResultAugmenter ────────────────────────────────────────────────────────────


class TestResultAugmenterSubclass:
    def test_type_attribute_set_to_qualified_name(self):
        assert hasattr(_TestAugmenter, "_type")
        assert _TestAugmenter._type.endswith("_TestAugmenter")
        assert "." in _TestAugmenter._type


class TestResultAugmenterInit:
    def test_name_stored(self):
        aug = _TestAugmenter("my_aug")
        assert aug.name == "my_aug"


class TestResultAugmenterFormatEvalResult:
    def test_formats_keys_excluding_type(self):
        result = ResultAugmenter.format_eval_result({"_type": "T", "key": "val"})
        assert "key: val" in result
        assert "_type" not in result

    def test_empty_after_excluding_type(self):
        result = ResultAugmenter.format_eval_result({"_type": "T"})
        assert result == ""


class TestResultAugmenterFormatModelResult:
    def test_formats_as_comma_joined_kv(self):
        result = ResultAugmenter.format_model_result({"_type": "T", "a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result

    def test_excludes_type_key(self):
        result = ResultAugmenter.format_model_result({"_type": "T", "x": 9})
        assert "_type" not in result


class TestResultAugmenterFromDict:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ResultAugmenter.from_dict({})


# ── SR_evaluator.__init__ — RMSE ──────────────────────────────────────────────


class TestSREvaluatorInitRmse:
    def test_stores_X_y_seed(self):
        se = SR_evaluator(_X, _y, seed=7)
        assert se.X is _X
        assert se.y is _y
        assert se.seed == 7

    def test_default_symbol_library(self):
        se = _make_se()
        assert se.symbol_library is not None
        assert isinstance(se.symbol_library, SymbolLibrary)

    def test_custom_symbol_library(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        se = SR_evaluator(_X, _y, symbol_library=sl, seed=0)
        assert se.symbol_library is sl

    def test_rmse_requires_y(self):
        with pytest.raises(ValueError, match="Target values must be provided"):
            SR_evaluator(_X, ranking_function="rmse")

    def test_rmse_default_success_threshold_is_1e7(self):
        se = SR_evaluator(_X, _y, seed=0, success_threshold=None)
        assert se.success_threshold == 1e-7

    def test_rmse_explicit_success_threshold_preserved(self):
        se = SR_evaluator(_X, _y, seed=0, success_threshold=0.05)
        assert se.success_threshold == 0.05

    def test_invalid_ranking_function_warns_and_falls_back(self):
        with pytest.warns(UserWarning, match="not supported"):
            se = SR_evaluator(_X, _y, seed=0, ranking_function="invalid")
        assert se.ranking_function == "rmse"

    def test_num_points_sampled_minus_one_uses_x_row_count(self):
        se = SR_evaluator(_X, _y, seed=0, num_points_sampled=-1)
        assert se.bed_evaluation_parameters["num_points_sampled"] == _X.shape[0]

    def test_callbacks_dispatcher_with_early_stopping_created(self):
        se = _make_se()
        assert isinstance(se._callbacks, CallbackDispatcher)
        assert any(isinstance(cb, EarlyStoppingCallback) for cb in se._callbacks.get_callbacks())


# ── SR_evaluator.__init__ — BED ───────────────────────────────────────────────


class TestSREvaluatorInitBed:
    def test_bed_requires_ground_truth(self):
        with pytest.raises(ValueError, match="Ground truth must be provided"):
            SR_evaluator(_X_BED, ranking_function="bed")

    def test_bed_list_ground_truth_creates_behavior_matrix(self):
        se = SR_evaluator(_X_BED, ground_truth=["C", "+", "X_0"], ranking_function="bed", seed=0)
        assert se.gt_behavior is not None
        assert isinstance(se.gt_behavior, np.ndarray)

    def test_bed_ndarray_ground_truth_used_directly(self):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            _X_BED,
            ground_truth=gt,
            ranking_function="bed",
            seed=0,
            num_points_sampled=_X_BED.shape[0],
            num_consts_sampled=4,
            bed_X=_X_BED,
        )
        assert se.gt_behavior is gt

    def test_bed_invalid_ground_truth_type_raises(self):
        with pytest.raises(ValueError):
            SR_evaluator(_X_BED, ground_truth="not_valid", ranking_function="bed", seed=0)

    def test_bed_domain_bounds_samples_bed_X(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            domain_bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            seed=0,
        )
        bed_X = se.bed_evaluation_parameters["bed_X"]
        assert bed_X is not None
        assert bed_X.shape[1] == _X_BED.shape[1]
        assert np.all(bed_X >= -1.0) and np.all(bed_X <= 1.0)

    def test_bed_no_domain_bounds_subsamples_X(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            num_points_sampled=10,
        )
        bed_X = se.bed_evaluation_parameters["bed_X"]
        assert bed_X.shape == (10, _X_BED.shape[1])
        for row in bed_X:
            assert any(np.allclose(row, x_row) for x_row in _X_BED)

    def test_bed_auto_success_threshold_computed(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=None,
        )
        assert se.success_threshold is not None
        assert se.success_threshold > 0

    def test_bed_explicit_success_threshold_preserved(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=0.5,
        )
        assert se.success_threshold == 0.5


# ── SR_evaluator.set_callbacks ────────────────────────────────────────────────


class TestSetCallbacks:
    def test_add_single_callback_appends_to_dispatcher(self):
        se = _make_se()
        initial_count = len(se._callbacks.get_callbacks())
        se.register_callbacks(_SimpleCb())
        assert len(se._callbacks.get_callbacks()) == initial_count + 1

    def test_set_dispatcher_preserves_existing_callbacks(self):
        se = _make_se()
        new_dispatcher = CallbackDispatcher()
        se.register_callbacks(new_dispatcher)
        assert any(isinstance(cb, EarlyStoppingCallback) for cb in se._callbacks.get_callbacks())

    def test_set_dispatcher_when_current_is_plain_srcallback(self):
        se = _make_se()
        se._callbacks = _SimpleCb()
        new_dispatcher = CallbackDispatcher()
        se.register_callbacks(new_dispatcher)
        assert se._callbacks is new_dispatcher

    def test_set_srcallback_when_current_is_srcallback(self):
        se = _make_se()
        se._callbacks = _SimpleCb()
        se.register_callbacks(_SimpleCb())
        assert isinstance(se._callbacks, CallbackDispatcher)
        assert len(se._callbacks.get_callbacks()) == 2

    def test_set_srcallback_when_current_is_none(self):
        se = _make_se()
        se._callbacks = None
        se.register_callbacks(_SimpleCb())
        assert isinstance(se._callbacks, CallbackDispatcher)


# ── SR_evaluator.evaluate_expr — RMSE ─────────────────────────────────────────


class TestEvaluateExprRmse:
    def test_basic_evaluation_returns_low_rmse(self):
        se = _make_se()
        rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert rmse < 1e-6

    def test_increments_total_evaluations_every_call(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"])  # cache hit
        assert se.total_evaluations == 2

    def test_cache_hit_returns_same_result(self):
        se = _make_se()
        r1 = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        r2 = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert r1 == r2
        assert len(se.models) == 1

    def test_cache_hit_still_fires_on_expr_evaluated(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = None
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0"])  # cache hit
        cb.on_expr_evaluated.assert_called_once()

    def test_invalid_expression_returns_nan(self):
        se = _make_se()
        assert np.isnan(se.evaluate_expr(["C", "*", "X_1", "X_0"]))

    def test_invalid_expression_added_to_invalid_list(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"])
        assert "C*X_1X_0" in se.invalid

    def test_should_stop_returns_nan_with_warning(self):
        se = _make_se()
        se.should_stop = True
        with pytest.warns(UserWarning):
            result = se.evaluate_expr(["X_0"])
        assert np.isnan(result)
        assert se.total_evaluations == 1

    def test_max_evaluations_triggers_stop(self):
        se = _make_se(max_evaluations=1, success_threshold=-1)
        se.evaluate_expr(["X_0"])
        with pytest.warns(UserWarning):
            result = se.evaluate_expr(["X_1"])
        assert np.isnan(result)

    def test_first_evaluation_fires_best_expression_callback(self):
        se = _make_se()
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = None
        cb.on_best_expression.return_value = None
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0", "+", "X_1"])
        cb.on_best_expression.assert_called_once()

    def test_worse_evaluation_does_not_fire_best_expression_callback(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])  # near-zero — sets the bar
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = None
        cb.on_best_expression.return_value = None
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0", "+", "X_1"])  # worse error
        cb.on_best_expression.assert_not_called()

    def test_callback_returning_false_sets_should_stop(self):
        se = _make_se()
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = False
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0"])
        assert se.should_stop is True

    def test_simplify_expr_stores_simplified_key(self):
        se = _make_se()
        se.evaluate_expr(
            ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"],
            simplify_expr=True,
        )
        assert "C+X_1" in se.models

    def test_simplify_expr_failure_warns_and_continues(self):
        se = _make_se()
        with patch("SRToolkit.evaluation.sr_evaluator.simplify", side_effect=Exception("bad")):
            with pytest.warns(UserWarning, match="Unable to simplify"):
                result = se.evaluate_expr(["X_0"], simplify_expr=True)
        assert not np.isnan(result)

    def test_simplify_failure_with_node_input_warns(self):
        se = _make_se()
        node = Node("*", left=Node("C"), right=Node("X_0"))
        with patch("SRToolkit.evaluation.sr_evaluator.simplify", side_effect=Exception("bad")):
            with pytest.warns(UserWarning, match="Unable to simplify"):
                result = se.evaluate_expr(node, simplify_expr=True)
        assert np.isfinite(result)

    def test_node_input_accepted(self):
        se = _make_se()
        node = Node("*", left=Node("C"), right=Node("X_0"))
        result = se.evaluate_expr(node)
        assert np.isfinite(result)

    def test_cache_hit_with_verbose_covers_log(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"], verbose=1)  # cache hit

    def test_verbose_1_covers_parameter_logging(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"], verbose=1)  # with params
        se.evaluate_expr(["X_0", "+", "X_1"], verbose=1)  # no params

    def test_verbose_1_covers_invalid_expr_logging(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"], verbose=1)

    def test_evaluate_expr_unsupported_ranking_function_raises(self):
        # Covers line 616: ranking_function bypasses __init__ check
        se = _make_se()
        se.ranking_function = "invalid"
        with pytest.raises(ValueError):
            se.evaluate_expr(["X_0"])


# ── SR_evaluator.evaluate_expr — BED ──────────────────────────────────────────


class TestEvaluateExprBed:
    def setup_method(self):
        self.se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=42,
            success_threshold=-1,
        )

    def test_bed_evaluation_returns_finite_float(self):
        result = self.se.evaluate_expr(["C", "+", "X_1"])
        assert np.isfinite(result)

    def test_bed_equivalent_expression_lower_error(self):
        err_match = self.se.evaluate_expr(["C", "+", "X_0"])
        err_other = self.se.evaluate_expr(["C", "+", "X_1"])
        assert err_match < err_other

    def test_bed_invalid_expression_returns_nan_and_appends_invalid(self):
        result = self.se.evaluate_expr(["C", "*", "X_1", "X_0"])
        assert np.isnan(result)
        assert "C*X_1X_0" in self.se.invalid

    def test_bed_verbose_1_covers_logging(self):
        self.se.evaluate_expr(["C", "+", "X_0"], verbose=1)

    def test_bed_invalid_verbose_covers_exception_log(self):
        self.se.evaluate_expr(["C", "*", "X_1", "X_0"], verbose=1)

    def test_bed_callback_stop_sets_should_stop(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = False
        self.se.register_callbacks(cb)
        self.se.evaluate_expr(["C", "+", "X_0"])
        assert self.se.should_stop is True


# ── SR_evaluator.get_results ──────────────────────────────────────────────────


class TestGetResults:
    def test_basic_fields(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        r = se.get_results()[0]
        assert r.num_evaluated == 1
        assert r.evaluation_calls == 1
        assert r.best_expr == "C*X_1-X_0"
        assert r.min_error < 1e-6

    def test_top_k_limits_top_models_not_all_models(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        r = se.get_results(top_k=1)[0]
        assert len(r.top_models) == 1
        assert len(r.all_models) == 2

    def test_top_k_negative_includes_all(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        r = se.get_results(top_k=-1)[0]
        assert len(r.top_models) == 2

    def test_top_k_exceeds_total_capped(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        r = se.get_results(top_k=100)[0]
        assert len(r.top_models) == 1

    def test_appends_to_existing_sr_results(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        existing = SR_results()
        se.get_results(results=existing)
        assert len(existing) == 1

    def test_metadata_dataset_name_extracted(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "test_ds", "fold": 3}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name == "test_ds"
        assert r.metadata == {"fold": 3}

    def test_metadata_only_dataset_name_gives_none_remaining(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "only"}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name == "only"
        assert r.metadata is None


# ── SR_evaluator.to_dict / from_dict ─────────────────────────────────────────


class TestSREvaluatorSerialization:
    def test_to_dict_creates_base_path_if_missing(self, tmp_path):
        se = _make_se()
        subdir = str(tmp_path / "new_subdir")
        d = se.to_dict(subdir, "test")
        assert os.path.isdir(subdir)
        assert os.path.isfile(d["X"])

    def test_to_dict_saves_X_and_y_as_npy(self, tmp_path):
        se = _make_se()
        d = se.to_dict(str(tmp_path), "test")
        assert os.path.isfile(d["X"])
        assert os.path.isfile(d["y"])

    def test_to_dict_y_none_stored_as_none(self, tmp_path):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
        )
        d = se.to_dict(str(tmp_path), "test")
        assert d["y"] is None

    def test_to_dict_list_ground_truth_stored_directly(self, tmp_path):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
        )
        d = se.to_dict(str(tmp_path), "test")
        assert isinstance(d["ground_truth"], list)

    def test_to_dict_node_ground_truth_converted_to_list(self, tmp_path):
        gt_node = Node("+", left=Node("C"), right=Node("X_0"))
        se = SR_evaluator(
            _X_BED,
            ground_truth=gt_node,
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
        )
        d = se.to_dict(str(tmp_path), "test")
        assert isinstance(d["ground_truth"], list)

    def test_to_dict_ndarray_ground_truth_saved_as_npy(self, tmp_path):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            _X_BED,
            ground_truth=gt,
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
            num_points_sampled=_X_BED.shape[0],
            num_consts_sampled=4,
            bed_X=_X_BED,
        )
        d = se.to_dict(str(tmp_path), "test")
        assert isinstance(d["ground_truth"], str)
        assert d["ground_truth"].endswith(".npy")

    def test_from_dict_round_trip_bed_list_ground_truth(self, tmp_path):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
        )
        d = se.to_dict(str(tmp_path), "bed")
        restored = SR_evaluator.from_dict(d)
        assert restored.y is None
        assert restored.ranking_function == "bed"
        np.testing.assert_array_equal(restored.X, se.X)

    def test_from_dict_round_trip_bed_ndarray_ground_truth(self, tmp_path):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            _X_BED,
            ground_truth=gt,
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
            num_points_sampled=_X_BED.shape[0],
            num_consts_sampled=4,
            bed_X=_X_BED,
        )
        d = se.to_dict(str(tmp_path), "bed_ndarray")
        restored = SR_evaluator.from_dict(d)
        assert restored.ranking_function == "bed"
        np.testing.assert_array_equal(restored.X, se.X)

    def test_from_dict_round_trip_rmse(self, tmp_path):
        se = _make_se()
        d = se.to_dict(str(tmp_path), "test")
        restored = SR_evaluator.from_dict(d)
        np.testing.assert_array_equal(restored.X, se.X)
        np.testing.assert_array_equal(restored.y, se.y)
        assert restored.max_evaluations == se.max_evaluations
        assert restored.ranking_function == se.ranking_function
        assert restored.success_threshold == se.success_threshold

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        se = _make_se()
        d = se.to_dict(str(tmp_path), "test")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="Unsupported format_version"):
            SR_evaluator.from_dict(d)

    def test_from_dict_missing_file_raises(self, tmp_path):
        se = _make_se()
        d = se.to_dict(str(tmp_path), "test")
        os.remove(d["X"])
        with pytest.raises(ValueError):
            SR_evaluator.from_dict(d)


# ── ModelResult / EvalResult augmentation ────────────────────────────────────


class TestModelResultAugmentation:
    def test_add_augmentation_stores_data_and_type(self):
        mr = ModelResult(expr=["X_0"], error=0.5)
        mr.add_augmentation("test", {"value": 42}, "T")
        assert "test" in mr.augmentations
        assert mr.augmentations["test"]["value"] == 42
        assert mr.augmentations["test"]["_type"] == "T"

    def test_add_augmentation_collision_renames_key(self):
        mr = ModelResult(expr=["X_0"], error=0.5)
        mr.add_augmentation("test", {"a": 1}, "T")
        mr.add_augmentation("test", {"b": 2}, "T")
        assert "test" in mr.augmentations
        assert "test_1" in mr.augmentations


class TestEvalResultAugmentation:
    def test_add_augmentation_stores_data(self):
        er = EvalResult(
            min_error=0.1,
            best_expr="X_0",
            num_evaluated=1,
            evaluation_calls=1,
            top_models=[],
            all_models=[],
            approach_name="",
            success=False,
        )
        er.add_augmentation("test", {"value": 42}, "T")
        assert "test" in er.augmentations
        assert er.augmentations["test"]["value"] == 42
        assert er.augmentations["test"]["_type"] == "T"


# ── SR_results.__init__ ────────────────────────────────────────────────────────


class TestSRResultsInit:
    def test_empty_results_list(self):
        assert SR_results().results == []


# ── SR_results.add_results ────────────────────────────────────────────────────


class TestSRResultsAddResults:
    def test_best_is_lowest_error(self):
        se = _make_se()
        se.evaluate_expr(["X_0", "+", "X_1"])  # worse
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])  # better
        r = se.get_results()[0]
        assert r.min_error < 1e-6

    def test_success_true_when_below_threshold(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=1.0)
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert se.get_results()[0].success is True

    def test_success_false_when_above_threshold(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=1e-20)
        se.evaluate_expr(["X_0", "+", "X_1"])
        assert se.get_results()[0].success is False

    def test_success_false_when_threshold_none(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=-1)
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        # success_threshold=-1 (< 0) means threshold check never passes
        r = se.get_results()[0]
        assert r.success is False

    def test_dataset_name_extracted_from_metadata(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "ds", "extra": 1}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name == "ds"
        assert r.metadata == {"extra": 1}

    def test_metadata_without_dataset_name_fully_preserved(self):
        se = SR_evaluator(_X, _y, metadata={"source": "feynman"}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name is None
        assert r.metadata == {"source": "feynman"}

    def test_only_dataset_name_in_metadata_gives_none_remaining(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "only"}, seed=0)
        se.evaluate_expr(["X_0"])
        assert se.get_results()[0].metadata is None


# ── SR_results.print_results ──────────────────────────────────────────────────


class TestSRResultsPrintResults:
    def test_prints_header_for_all_experiments(self, capsys):
        results = _single_result()
        results.print_results()
        assert "=== Experiment 1/1 ===" in capsys.readouterr().out

    def test_prints_specific_experiment_no_header(self, capsys):
        results = _single_result()
        results.print_results(experiment_number=0)
        out = capsys.readouterr().out
        assert "===" not in out
        assert "Best expression" in out

    def test_out_of_bounds_experiment_number_raises(self):
        results = _single_result()
        with pytest.raises(AssertionError):
            results.print_results(experiment_number=99)

    def test_prints_dataset_and_approach_when_set(self, capsys):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "MyDS"}, seed=0)
        se.evaluate_expr(["X_0"])
        se.get_results(approach_name="MyApproach").print_results()
        out = capsys.readouterr().out
        assert "MyDS" in out
        assert "MyApproach" in out

    def test_prints_metadata_block(self, capsys):
        se = SR_evaluator(_X, _y, metadata={"source": "feynman"}, seed=0)
        se.evaluate_expr(["X_0"])
        se.get_results().print_results()
        assert "source: feynman" in capsys.readouterr().out

    def test_detailed_shows_models_section(self, capsys):
        results = _single_result()
        results.print_results(detailed=True)
        assert "Models:" in capsys.readouterr().out

    def test_model_scope_best_shows_one_model(self, capsys):
        results = _two_result(top_k=2)
        results.print_results(detailed=True, model_scope="best")
        out = capsys.readouterr().out
        # Only the best model expression should appear in the Models section
        lines = [line for line in out.splitlines() if line.startswith("  ") and "error=" in line]
        assert len(lines) == 1

    def test_model_scope_all_shows_all_models(self, capsys):
        results = _two_result(top_k=1)
        results.print_results(detailed=True, model_scope="all")
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.startswith("  ") and "error=" in line]
        assert len(lines) == 2

    def test_eval_augmentation_printed_fallback(self, capsys):
        # augmentation with unresolvable _type → k/v fallback printing
        results = _single_result()
        results[0].add_augmentation("mykey", {"info": "hello"}, "NonExistent.Class")
        results.print_results()
        out = capsys.readouterr().out
        assert "mykey" in out
        assert "hello" in out

    def test_model_augmentation_printed_fallback(self, capsys):
        results = _single_result()
        results[0].top_models[0].add_augmentation("aug", {"x": 5}, "NonExistent.Class")
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "x=5" in out

    def test_model_augmentation_only_type_not_printed(self, capsys):
        # When augmentation data has only _type, parts=[] so nothing printed for that aug
        results = _single_result()
        results[0].top_models[0].add_augmentation("empty_aug", {}, "NonExistent.Class")
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "empty_aug" not in out

    def test_augmentations_filter(self, capsys):
        results = _single_result()
        results[0].add_augmentation("keep", {"val": 1}, "NonExistent.Class")
        results[0].add_augmentation("skip", {"val": 2}, "NonExistent.Class")
        results.print_results(augmentations=["keep"])
        out = capsys.readouterr().out
        assert "keep" in out
        assert "skip" not in out

    def test_model_augmentation_resolved_prints_line(self, capsys):
        from SRToolkit.evaluation.result_augmentation import RMSE as RMSEAugmenter

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([RMSEAugmenter(se)])
        results.print_results(detailed=True)
        capsys.readouterr()  # discard

    def test_model_augmentation_filter_applied(self, capsys):
        # Covers line 984: augmentations_filter is applied to model-level augmentation keys
        from SRToolkit.evaluation.result_augmentation import RMSE as RMSEAugmenter

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([RMSEAugmenter(se)])
        results.print_results(detailed=True, augmentations=["RMSE"])
        out = capsys.readouterr().out
        assert "RMSE" in out

    def test_augmentation_resolved_to_result_augmenter_class(self, capsys):
        # Use a real augmenter whose _type is importable
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.print_results()
        out = capsys.readouterr().out
        assert "ExpressionToLatex" in out

    def test_model_augmentation_resolved_to_result_augmenter_class(self, capsys):
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "ExpressionToLatex" in out


# ── SR_results.augment ────────────────────────────────────────────────────────


class TestSRResultsAugment:
    def test_augment_all_experiments(self):
        se1 = _make_se()
        se1.evaluate_expr(["X_0"])
        se2 = _make_se()
        se2.evaluate_expr(["X_1"])
        combined = se1.get_results() + se2.get_results()
        combined.augment([_TestAugmenter("aug")])
        assert "aug" in combined[0].augmentations
        assert "aug" in combined[1].augmentations

    def test_augment_specific_experiment(self):
        combined = _single_result() + _single_result()
        combined.augment([_TestAugmenter("aug")], experiment_number=0)
        assert "aug" in combined[0].augmentations
        assert "aug" not in combined[1].augmentations

    def test_augment_single_augmenter_not_in_list(self):
        results = _single_result()
        results.augment(_TestAugmenter("aug"))
        assert "aug" in results[0].augmentations

    def test_augment_specific_experiment_exception_warns(self):
        results = _single_result()
        with pytest.warns(UserWarning, match="intentional failure"):
            results.augment([_FailingAugmenter("bad")], experiment_number=0)

    def test_augment_exception_warns_and_continues(self):
        combined = _single_result() + _single_result()
        with pytest.warns(UserWarning, match="intentional failure"):
            combined.augment([_FailingAugmenter("bad"), _TestAugmenter("good")])
        assert "good" in combined[0].augmentations
        assert "good" in combined[1].augmentations


# ── SR_results container ops ──────────────────────────────────────────────────


class TestSRResultsContainerOps:
    def test_len(self):
        assert len(_single_result()) == 1

    def test_getitem_returns_eval_result(self):
        assert isinstance(_single_result()[0], EvalResult)

    def test_getitem_out_of_bounds_raises(self):
        with pytest.raises(AssertionError):
            _single_result()[99]

    def test_add_creates_new_object_without_mutating_operands(self):
        r1 = _single_result()
        r2 = _single_result()
        r3 = r1 + r2
        assert len(r1) == 1
        assert len(r3) == 2
        assert r3 is not r1

    def test_iadd_modifies_in_place(self):
        r1 = _single_result()
        r2 = _single_result()
        r1 += r2
        assert len(r1) == 2


# ── SR_results.save / load ────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
class TestSRResultsSaveLoad:
    def test_save_to_explicit_json_file(self, tmp_path):
        results = _single_result()
        path = str(tmp_path / "out.json")
        results.save(path)
        assert os.path.isfile(path)

    def test_save_to_existing_directory(self, tmp_path):
        results = _single_result()
        results.save(str(tmp_path))
        assert os.path.isfile(tmp_path / "results.json")

    def test_save_extensionless_path_treated_as_directory(self, tmp_path):
        results = _single_result()
        target = str(tmp_path / "mydir")
        results.save(target)
        assert os.path.isfile(os.path.join(target, "results.json"))

    def test_save_invalid_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file extension"):
            _single_result().save(str(tmp_path / "out.txt"))

    def test_save_creates_nested_directories(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c")
        _single_result().save(path)
        assert os.path.isfile(os.path.join(path, "results.json"))

    def test_load_from_file(self, tmp_path):
        results = _single_result()
        path = str(tmp_path / "r.json")
        results.save(path)
        loaded = SR_results.load(path)
        assert len(loaded) == 1

    def test_load_from_directory(self, tmp_path):
        _single_result().save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert len(loaded) == 1

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SR_results.load(str(tmp_path / "missing.json"))

    def test_load_invalid_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file extension"):
            SR_results.load(str(tmp_path / "out.txt"))

    def test_load_invalid_format_version_raises(self, tmp_path):
        _single_result().save(str(tmp_path))
        path = str(tmp_path / "results.json")
        with open(path) as f:
            data = json.load(f)
        data["format_version"] = 99
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Unsupported format_version"):
            SR_results.load(str(tmp_path))

    def test_save_load_round_trip(self, tmp_path):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        results = se.get_results(top_k=2)
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert loaded[0].best_expr == results[0].best_expr
        assert abs(loaded[0].min_error - results[0].min_error) < 1e-12
        assert loaded[0].num_evaluated == results[0].num_evaluated
        assert len(loaded[0].top_models) == len(results[0].top_models)
        np.testing.assert_array_almost_equal(loaded[0].top_models[0].parameters, results[0].top_models[0].parameters)

    def test_save_load_multiple_experiments(self, tmp_path):
        r1 = _single_result()
        r2 = _single_result()
        combined = r1 + r2
        combined.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert len(loaded) == 2
        assert loaded[0].best_expr == combined[0].best_expr
        assert loaded[1].best_expr == combined[1].best_expr

    def test_save_load_with_metadata(self, tmp_path):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "ds", "fold": 3}, seed=42)
        se.evaluate_expr(["X_0"])
        results = se.get_results(approach_name="Ap")
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert loaded[0].dataset_name == "ds"
        assert loaded[0].metadata == {"fold": 3}
        assert loaded[0].approach_name == "Ap"

    def test_save_load_with_augmentations(self, tmp_path):
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert "ExpressionToLatex" in loaded[0].augmentations
        orig = results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]
        assert loaded[0].augmentations["ExpressionToLatex"]["best_expr_latex"] == orig

    def test_save_load_rmse_augmentation_round_trip(self, tmp_path):
        from SRToolkit.evaluation.result_augmentation import RMSE as RMSEAugmenter

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        results = se.get_results(top_k=2)
        results.augment([RMSEAugmenter(se)])
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        orig_aug = results[0].augmentations["RMSE"]
        rest_aug = loaded[0].augmentations["RMSE"]
        assert rest_aug["_type"] == "SRToolkit.evaluation.result_augmentation.RMSE"
        assert abs(rest_aug["min_error"] - orig_aug["min_error"]) < 1e-12
        # model-level parameters survive the JSON round-trip
        orig_params = results[0].top_models[0].augmentations["RMSE"]["parameters"]
        rest_params = loaded[0].top_models[0].augmentations["RMSE"]["parameters"]
        np.testing.assert_array_almost_equal(rest_params, orig_params)
