import json
import logging
import math
import os
import warnings
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import infix_expressions

from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation.callbacks import (
    CallbackDispatcher,
    EarlyStoppingCallback,
    SRCallbacks,
)
from SRToolkit.evaluation.ranking import RankingFunction, RankingResult
from SRToolkit.evaluation.sr_evaluator import ResultAugmenter, SR_evaluator, SR_results
from SRToolkit.utils.expression_constraints import (
    ExpressionConstraint,
    MaxConstants,
    MaxTokens,
    MaxTreeDepth,
    OutputBounds,
    RequiredVariables,
)
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, tokens_to_tree
from SRToolkit.utils.grammar import MaxDepth, MaxNodes
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, EvalResult, ModelResult

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


_LOGGER = "SRToolkit.evaluation.sr_evaluator"


def _log_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


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

    def test_stores_arguments_and_seeds_global_rng(self):
        # "If given, the constructor also calls np.random.seed(seed)": the next global draw is the
        # first draw after np.random.seed(3).
        ground_truth = ["C", "*", "X_1", "-", "X_0"]
        se = SR_evaluator(_X, _y, seed=3, ground_truth=ground_truth, max_constants=3)
        drawn = np.random.random()
        np.random.seed(3)
        assert drawn == np.random.random()
        assert se.ground_truth is ground_truth
        assert se.kwargs == {"max_constants": 3}

    def test_default_symbol_library(self):
        se = _make_se()
        assert se.symbol_library is not None
        assert isinstance(se.symbol_library, SymbolLibrary)

    def test_custom_symbol_library(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        se = SR_evaluator(_X, _y, symbol_library=sl, seed=0)
        assert se.symbol_library is sl

    def test_rmse_requires_y_or_ground_truth(self):
        with pytest.raises(ValueError, match="either target values y or a ground truth"):
            SR_evaluator(_X, ranking_function="rmse")

    def test_rmse_derives_y_from_ground_truth(self):
        se = SR_evaluator(_X, ground_truth=["X_0", "+", "X_1"], ranking_function="rmse", seed=0)
        assert se.evaluate_expr(["X_0", "+", "X_1"]) == pytest.approx(0.0)

    def test_rmse_ground_truth_with_free_constant_rejected(self):
        with pytest.raises(ValueError, match="free constant"):
            SR_evaluator(_X, ground_truth=["X_0", "+", "C"], ranking_function="rmse", seed=0)

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
        se = SR_evaluator(_X_BED, ground_truth=["C", "+", "X_0"], ranking_function="bed", seed=0, num_points_sampled=-1)
        assert se.ranking.bed_X.shape[0] == _X_BED.shape[0]

    def test_callbacks_dispatcher_with_early_stopping_created(self):
        se = _make_se()
        assert isinstance(se._callbacks, CallbackDispatcher)
        assert any(isinstance(cb, EarlyStoppingCallback) for cb in se._callbacks.get_callbacks())


# ── SR_evaluator.__init__ — BED ───────────────────────────────────────────────


class TestSREvaluatorInitBed:
    def test_bed_requires_ground_truth_bed_gt_or_y(self):
        with pytest.raises(ValueError, match="a ground truth .* a precomputed bed_gt .* or target values y"):
            SR_evaluator(_X_BED, ranking_function="bed")

    def test_bed_list_ground_truth_creates_behavior_matrix(self):
        se = SR_evaluator(_X_BED, ground_truth=["C", "+", "X_0"], ranking_function="bed", seed=0)
        assert se.ranking.gt_behavior is not None
        assert isinstance(se.ranking.gt_behavior, np.ndarray)

    def test_bed_ndarray_ground_truth_rejected(self):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        with pytest.raises(ValueError, match="list of tokens or a Node object"):
            SR_evaluator(
                _X_BED,
                ground_truth=gt,
                ranking_function="bed",
                seed=0,
                num_points_sampled=_X_BED.shape[0],
                num_consts_sampled=4,
            )

    def test_bed_invalid_ground_truth_type_raises(self):
        with pytest.raises(ValueError):
            SR_evaluator(_X_BED, ground_truth="not_valid", ranking_function="bed", seed=0)

    def test_bed_samplers_generate_bed_X(self):
        samplers = [UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)]
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            samplers=samplers,
            num_points_sampled=15,
            seed=0,
        )
        bed_X = se.ranking.bed_X
        assert bed_X.shape == (15, _X_BED.shape[1])
        assert np.all(np.abs(bed_X) <= 1.0)

    def test_bed_samplers_seeded_reproducible(self):
        def build():
            return SR_evaluator(
                _X_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
                num_points_sampled=15,
                seed=7,
            )

        assert np.array_equal(build().ranking.bed_X, build().ranking.bed_X)

    def test_bed_samplers_reject_use_all_points(self):
        samplers = [UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)]
        with pytest.raises(ValueError, match="not valid together with samplers"):
            SR_evaluator(
                _X_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                samplers=samplers,
                num_points_sampled=-1,
                seed=0,
            )

    def test_bed_subsamples_X_without_replacement(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            num_points_sampled=10,
        )
        bed_X = se.ranking.bed_X
        assert bed_X.shape == (10, _X_BED.shape[1])
        for row in bed_X:
            assert any(np.allclose(row, x_row) for x_row in _X_BED)
        # Without replacement: all sampled rows are distinct.
        assert len({tuple(row) for row in bed_X}) == 10

    def test_bed_num_points_minus_one_uses_all_X(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            seed=0,
            num_points_sampled=-1,
        )
        assert se.ranking.bed_X.shape == _X_BED.shape

    def test_bed_precomputed_bed_gt_used(self):
        points = _X_BED[:8]
        gt = create_behavior_matrix(["X_0", "+", "C"], points, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            points,
            ground_truth=["X_0", "+", "C"],
            ranking_function="bed",
            bed_gt=gt,
            num_consts_sampled=4,
            seed=0,
        )
        np.testing.assert_array_equal(se.ranking.gt_behavior, gt)

    def test_bed_gt_subsampled_jointly_with_X(self):
        # The matrix is tied to the rows of X, so subsampling keeps the same rows of both.
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        se = SR_evaluator(_X_BED, ranking_function="bed", bed_gt=gt, num_consts_sampled=4, num_points_sampled=5, seed=0)
        assert se.ranking.bed_X.shape == (5, _X_BED.shape[1])
        rows = [int(np.flatnonzero((_X_BED == row).all(axis=1))[0]) for row in se.ranking.bed_X]
        np.testing.assert_array_equal(se.ranking.gt_behavior, gt[rows])

    def test_bed_gt_column_mismatch_rejected(self):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        with pytest.raises(ValueError, match="4 columns but num_consts_sampled is 32"):
            SR_evaluator(_X_BED, ranking_function="bed", bed_gt=gt, seed=0)

    def test_bed_gt_one_dimensional_rejected(self):
        with pytest.raises(ValueError, match="2-D behaviour matrix"):
            SR_evaluator(_X_BED, ranking_function="bed", bed_gt=np.zeros(_X_BED.shape[0]), seed=0)

    def test_bed_gt_with_samplers_ignored(self):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED, num_consts_sampled=4, seed=0)
        with pytest.warns(UserWarning, match="samplers are ignored"):
            se = SR_evaluator(
                _X_BED,
                ranking_function="bed",
                bed_gt=gt,
                num_consts_sampled=4,
                samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
                seed=0,
            )
        np.testing.assert_array_equal(se.ranking.bed_X, _X_BED)

    def test_bed_gt_row_mismatch_rejected(self):
        gt = create_behavior_matrix(["X_0", "+", "C"], _X_BED[:5], num_consts_sampled=4, seed=0)
        with pytest.raises(ValueError, match="X it was computed on"):
            SR_evaluator(
                _X_BED[:8],
                ground_truth=["X_0", "+", "C"],
                ranking_function="bed",
                bed_gt=gt,
                seed=0,
            )

    def test_bed_gt_without_ground_truth_works(self):
        points = _X_BED[:8]
        gt = create_behavior_matrix(["X_0", "+", "C"], points, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            points,
            ranking_function="bed",
            bed_gt=gt,
            num_consts_sampled=4,
            success_threshold=-1,
            seed=0,
        )
        assert se.ranking.ground_truth is None
        np.testing.assert_array_equal(se.ranking.gt_behavior, gt)
        # Candidates can still be scored against the precomputed behaviour.
        assert se.evaluate_expr(["X_0", "+", "C"]) < 1

    def test_bed_gt_without_ground_truth_has_no_default_threshold(self):
        points = _X_BED[:8]
        gt = create_behavior_matrix(["X_0", "+", "C"], points, num_consts_sampled=4, seed=0)
        se = SR_evaluator(
            points,
            ranking_function="bed",
            bed_gt=gt,
            num_consts_sampled=4,
            success_threshold=None,
            seed=0,
        )
        assert se.success_threshold is None

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

    def test_bed_auto_success_threshold_reproducible_with_seed(self):
        def build():
            return SR_evaluator(_X_BED, ground_truth=["C", "+", "X_0"], ranking_function="bed", seed=3)

        assert build().success_threshold == build().success_threshold

    def test_bed_y_used_as_deterministic_behavior(self):
        y = _X_BED[:, 0]
        se = SR_evaluator(_X_BED, y, ranking_function="bed", num_points_sampled=-1, success_threshold=-1, seed=0)
        np.testing.assert_array_equal(se.ranking.gt_behavior, np.repeat(y[:, None], 32, axis=1))
        assert se.evaluate_expr(["X_0"]) == 0.0
        assert se.evaluate_expr(["X_1"]) > 0.0

    def test_bed_y_subsampled_jointly_with_X(self):
        y = _X_BED[:, 0] * 2.0
        se = SR_evaluator(_X_BED, y, ranking_function="bed", num_points_sampled=6, success_threshold=-1, seed=0)
        np.testing.assert_allclose(se.ranking.gt_behavior[:, 0], se.ranking.bed_X[:, 0] * 2.0)

    def test_bed_ground_truth_takes_precedence_over_y(self):
        y = np.zeros(_X_BED.shape[0])
        se = SR_evaluator(_X_BED, y, ground_truth=["X_0"], ranking_function="bed", num_points_sampled=-1, seed=0)
        np.testing.assert_array_equal(se.ranking.gt_behavior[:, 0], _X_BED[:, 0])

    def test_bed_y_with_samplers_ignored(self):
        with pytest.warns(UserWarning, match="samplers are ignored"):
            se = SR_evaluator(
                _X_BED,
                _X_BED[:, 0],
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
                seed=0,
            )
        np.testing.assert_array_equal(se.ranking.bed_X, _X_BED)

    def test_bed_y_has_no_default_threshold(self):
        se = SR_evaluator(_X_BED, _X_BED[:, 0], ranking_function="bed", seed=0)
        assert se.success_threshold is None

    def test_bed_sampler_count_must_match_variables(self):
        with pytest.raises(ValueError, match="1 samplers for 2 variables"):
            SR_evaluator(
                _X_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0)],
                seed=0,
            )

    def test_bed_ranking_seed_overrides_evaluator_seed(self):
        def build(seed):
            return SR_evaluator(
                _X_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                num_points_sampled=10,
                success_threshold=-1,
                seed=seed,
                ranking_seed=5,
            )

        a, b = build(1), build(2)
        np.testing.assert_array_equal(a.ranking.bed_X, b.ranking.bed_X)
        assert a.evaluate_expr(["X_0", "*", "C"]) == b.evaluate_expr(["X_0", "*", "C"])


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

    def test_invalid_expression_returns_inf(self):
        se = _make_se()
        assert se.evaluate_expr(["C", "*", "X_1", "X_0"]) == np.inf
        assert se.models[0].status == "invalid"

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
        assert se.total_evaluations == 0
        assert se.trace == []

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
        assert [m.expr for m in se.models] == [["C", "+", "X_1"]]

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

    @pytest.mark.parametrize(
        "make_kwargs, expr, status",
        [
            (lambda: {}, ["X_0", "X_0"], "invalid"),
            (lambda: {}, ["C"] + ["+", "C"] * 8, "rejected"),  # 9 constants, default limit 8
            (
                lambda: {"symbol_library": _LIB2(), "required_constraints": [OutputBounds([[1.0, 1.0]], lower=100.0)]},
                ["X_0"],
                "rejected",
            ),
            pytest.param(
                lambda: {"required_constraints": [_Boom()]},
                ["X_0"],
                "error",
                marks=pytest.mark.filterwarnings("ignore:.*_Boom.pre_check raised"),
            ),
            (lambda: {}, ["X_0"], "ok"),
        ],
        ids=["invalid", "rejected-pre-check", "rejected-post-check", "check-raised", "ok"],
    )
    def test_verbose_controls_debug_log_of_new_expression(self, caplog, make_kwargs, expr, status):
        # verbose=0: no logging; verbose=1: one DEBUG record with the final status, error and detail.
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        _make_se(**make_kwargs()).evaluate_expr(expr, verbose=0)
        assert _log_messages(caplog) == []
        se = _make_se(**make_kwargs())
        se.evaluate_expr(expr, verbose=1)
        model = se.models[0]
        (message,) = _log_messages(caplog)
        assert model.status == status
        assert "".join(expr) in message
        assert status in message
        assert str(model.error) in message
        assert str(model.status_detail or "") in message

    def test_verbose_1_logs_cache_hit(self, caplog):
        caplog.set_level(logging.DEBUG, logger=_LOGGER)
        se = _make_se()
        se.evaluate_expr(["X_0", "+", "X_1"])
        se.evaluate_expr(["X_0", "+", "X_1"], verbose=0)
        assert _log_messages(caplog) == []
        se.evaluate_expr(["X_0", "+", "X_1"], verbose=1)
        (message,) = _log_messages(caplog)
        assert "X_0+X_1" in message

    def test_success_threshold_stops_evaluation(self):
        se = _make_se(success_threshold=1e-6)
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])  # exact fit: error below the threshold
        assert se.should_stop is True
        with pytest.warns(UserWarning, match="success_threshold"):
            assert np.isnan(se.evaluate_expr(["X_0"]))
        assert se.total_evaluations == 1

    def test_ranking_function_is_read_only(self):
        # ranking_function now reflects the active RankingFunction instance and cannot be reassigned;
        # the old string-dispatch branch in evaluate_expr no longer exists.
        se = _make_se()
        with pytest.raises(AttributeError):
            se.ranking_function = "invalid"


# ── SR_evaluator.evaluate_expr — BED ──────────────────────────────────────────


class TestEvaluationTrace:
    def test_trace_records_misses_and_cache_hits(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_1"])
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"])
        assert se.trace == [0, 1, 0, 0]
        assert [m.expr for m in se.models] == [["X_0"], ["X_1"]]
        assert se.total_evaluations == 4

    def test_invalid_expressions_enter_trace(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"])
        se.evaluate_expr(["C", "*", "X_1", "X_0"])
        assert se.trace == [0, 0]
        assert se.models[0].error == np.inf
        assert se.models[0].status == "invalid"

    def test_calls_after_stop_not_recorded(self):
        se = _make_se(max_evaluations=2)
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_1"])
        with pytest.warns(UserWarning):
            se.evaluate_expr(["X_0"])
        assert se.trace == [0, 1]
        assert se.total_evaluations == 2

    def test_event_evaluation_number_matches_trace_length(self):
        se = _make_se()
        numbers = []

        class _Recorder(SRCallbacks):
            def on_expr_evaluated(self, event):
                numbers.append((event.evaluation_number, len(se.trace)))

        se.register_callbacks(_Recorder())
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_1"])
        assert numbers == [(1, 1), (2, 2), (3, 3)]

    def test_get_results_carries_trace(self):
        se = _make_se()
        se.evaluate_expr(["X_0", "+", "X_1"])
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        r = se.get_results(top_k=1)[0]
        assert r.trace == [0, 1, 0]
        assert r.evaluation_calls == 3
        assert r.num_evaluated == 2
        assert r.best_so_far() == [(1, 0), (2, 1)]
        assert r.calls_to_threshold(threshold=1e-6) == 2

    def test_get_results_copies_trace(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        se.evaluate_expr(["X_1"])
        assert r.trace == [0]
        assert r.num_evaluated == 1

    def test_trace_survives_save_load(self, tmp_path):
        se = _make_se(success_threshold=1e-6)
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])  # success: stops evaluation
        results = se.get_results(top_k=1)
        path = str(tmp_path / "r.json")
        results.save(path)
        loaded = SR_results.load(path)[0]
        assert loaded.trace == [0, 0, 1]
        assert loaded.calls_to_threshold() == 3
        assert loaded.success_threshold == 1e-6
        assert loaded.top_k == 1
        assert loaded.success is True
        assert loaded.best_expr == "C*X_1-X_0"


class _FixedRanking(RankingFunction):
    """Returns a fixed RankingResult, or raises if it is an exception."""

    name = "fixed"

    def __init__(self, outcome):
        self.outcome = outcome

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        pass

    def score(self, expr):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


# Each case: expression, status under RMSE, status under BED. BED tolerates NaN for some sampled
# constants and gives inf only when a point has no finite value, so its "undefined" set differs.
_STATUS_CASES = [
    (["C", "*", "X_1", "-", "X_0"], "ok", "ok"),
    (["C", "*", "X_1", "X_0"], "invalid", "invalid"),
    (["X_0", "+", "foo"], "invalid", "invalid"),
    (["(", "X_0", "+", "X_1"], "invalid", "invalid"),
    (["X_0", "(", ")"], "invalid", "invalid"),
    (["sqrt", "(", "X_0", "-", "100", ")"], "undefined", "undefined"),
    (["log", "(", "X_0", "-", "X_0", ")"], "undefined", "undefined"),
    (["X_0", "/", "(", "X_0", "-", "X_0", ")"], "undefined", "undefined"),
    (["C"] + ["+", "C"] * 8, "rejected", "rejected"),
]


class TestEvaluationStatus:
    @pytest.mark.parametrize("backend", ["stack", "stack_py", "codegen"])
    @pytest.mark.parametrize("ranking", ["rmse", "bed"])
    def test_classification(self, backend, ranking):
        if ranking == "rmse":
            se = _make_se(backend=backend)
        else:
            se = SR_evaluator(
                _X_BED, ground_truth=["C", "*", "X_0"], ranking_function="bed", seed=0, success_threshold=-1
            )
        for expr, rmse_status, bed_status in _STATUS_CASES:
            expected = rmse_status if ranking == "rmse" else bed_status
            error = se.evaluate_expr(expr)
            model = se.models[-1]
            assert model.status == expected, "".join(expr)
            if expected == "ok":
                assert np.isfinite(error) and model.status_detail is None
            else:
                assert error == np.inf and model.error == np.inf
                assert model.status_detail

    def test_status_details(self):
        se = _make_se()
        se.evaluate_expr(["(", "X_0"])
        se.evaluate_expr(["C"] + ["+", "C"] * 8)
        se.evaluate_expr(["log", "(", "X_0", "-", "X_0", ")"])
        details = [m.status_detail for m in se.models]
        assert "unclosed '('" in details[0]
        assert details[1] == "MaxConstants: 9 constants, limit 8"
        assert details[2] == "rmse score is inf"

    def test_max_expr_length_rejects(self):
        se = _make_se(max_expr_length=3)
        assert se.evaluate_expr(["X_0", "+", "X_1", "+", "X_0"]) == np.inf
        assert se.models[0].status == "rejected"
        assert se.models[0].status_detail == "MaxTokens: 5 tokens, limit 3"
        assert np.isfinite(se.evaluate_expr(["X_0", "+", "X_1"]))

    def test_bed_enforces_max_constants(self):
        se = SR_evaluator(
            _X_BED,
            ground_truth=["C", "*", "X_0"],
            ranking_function="bed",
            seed=0,
            success_threshold=-1,
            max_constants=1,
        )
        se.evaluate_expr(["C", "*", "X_0", "+", "C"])
        assert se.models[0].status == "rejected"

    def test_ranking_exception_gives_error_with_warning(self):
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(RuntimeError("boom")), success_threshold=-1)
        with pytest.warns(UserWarning, match="boom"):
            assert se.evaluate_expr(["X_0"]) == np.inf
        assert se.models[0].status == "error"
        assert se.models[0].status_detail == "RuntimeError: boom"
        assert se.models[0].expr == ["X_0"]
        assert se.invalid == []

    def test_ranking_status_is_respected(self):
        outcome = RankingResult(score=0.3, status="rejected", status_detail="not physical")
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(outcome), success_threshold=-1)
        assert se.evaluate_expr(["X_0"]) == np.inf
        assert (se.models[0].status, se.models[0].status_detail) == ("rejected", "not physical")

    def test_ranking_rejection_without_detail(self):
        outcome = RankingResult(score=np.inf, status="rejected")
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(outcome), success_threshold=-1)
        se.evaluate_expr(["X_0"])
        assert (se.models[0].status, se.models[0].status_detail) == ("rejected", "rejected by the ranking")

    def test_non_finite_score_is_undefined(self):
        outcome = RankingResult(score=np.nan)
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(outcome), success_threshold=-1)
        assert se.evaluate_expr(["X_0"]) == np.inf
        assert (se.models[0].status, se.models[0].status_detail) == ("undefined", "fixed score is nan")

    @pytest.mark.parametrize("status", ["ok", "invalid", "error", "weird"])
    def test_disallowed_ranking_status_gives_error(self, status):
        outcome = RankingResult(score=0.3, status=status)  # type: ignore[arg-type]
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(outcome), success_threshold=-1)
        with pytest.warns(UserWarning, match="status it may not set"):
            se.evaluate_expr(["X_0"])
        assert se.models[0].status == "error"
        assert se.models[0].error == np.inf
        assert se.models[0].expr == ["X_0"]
        assert se.models[0].status_detail  # every status other than "ok" says why

    def test_ranking_undefined_with_finite_score(self):
        outcome = RankingResult(score=0.3, status="undefined", status_detail="no finite behaviour")
        se = SR_evaluator(_X, _y, ranking_function=_FixedRanking(outcome), success_threshold=-1)
        assert se.evaluate_expr(["X_0"]) == np.inf
        assert (se.models[0].status, se.models[0].status_detail) == ("undefined", "no finite behaviour")

    @pytest.mark.parametrize(
        "operation",
        [
            lambda: np.log(np.zeros(1)),
            lambda: np.sqrt(-np.ones(1)),
            lambda: np.exp(np.full(1, 1000.0)),
        ],
        ids=["divide", "invalid", "overflow"],
    )
    @pytest.mark.parametrize("verbose, warns", [(0, False), (1, False), (2, True)], ids=["0", "1", "2"])
    def test_numpy_warnings_surface_only_at_verbose_2(self, operation, verbose, warns):
        # Documented for evaluate_expr(verbose=2): "also surfaces NumPy warnings during evaluation".
        class _WarningRanking(_FixedRanking):
            def score(self, expr):
                operation()
                return RankingResult(score=0.1)

        se = SR_evaluator(_X, _y, ranking_function=_WarningRanking(None), success_threshold=-1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            se.evaluate_expr(["X_0"], verbose=verbose)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught) is warns

    def test_invalid_property_lists_only_invalid(self):
        se = _make_se()
        se.evaluate_expr(["X_0", "X_1"])
        se.evaluate_expr(["log", "(", "X_0", "-", "X_0", ")"])
        se.evaluate_expr(["X_1", "X_0"])
        assert se.invalid == ["X_0X_1", "X_1X_0"]

    def test_node_input_is_classified(self):
        se = _make_se()
        tree = tokens_to_tree(["log", "(", "X_0", "-", "X_0", ")"], se.symbol_library)
        se.evaluate_expr(tree)
        assert se.models[0].status == "undefined"

    def test_failed_models_never_become_best(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = True
        cb.on_best_expression.return_value = True
        se = _make_se()
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0", "X_1"])
        se.evaluate_expr(["X_0"])
        events = [c.args[0] for c in cb.on_expr_evaluated.call_args_list]
        assert [(e.status, e.is_new_best) for e in events] == [("invalid", False), ("ok", True)]
        assert cb.on_best_expression.call_count == 1

    def test_cache_hit_event_carries_status(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = True
        se = _make_se()
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0", "X_1"])
        se.evaluate_expr(["X_0", "X_1"])
        assert cb.on_expr_evaluated.call_args_list[-1].args[0].status == "invalid"

    def test_rmse_records_optimizer_outcome(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_1", "-", "X_0"])
        assert set(se.models[0].ranking_data) == {"converged", "evaluations", "iterations"}
        assert se.models[1].ranking_data == {}

    def test_get_metric_nan_for_undefined(self):
        se = _make_se()
        expr = ["log", "(", "X_0", "-", "X_0", ")"]
        se.evaluate_expr(expr)
        assert np.isnan(se.get_metric(expr, "r2"))

    def test_results_carry_statuses(self, tmp_path):
        se = _make_se()
        se.evaluate_expr(["X_0", "X_1"])
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "X_1"])
        result = se.get_results()[0]
        assert result.best_expr == "C*X_1-X_0"
        assert result.status_counts(per="call")["invalid"] == 2
        assert result.status_counts() == {"ok": 1, "invalid": 1, "rejected": 0, "undefined": 0, "error": 0}
        path = tmp_path / "result.json"
        path.write_text(json.dumps(result.to_dict()))
        loaded = EvalResult.from_dict(json.loads(path.read_text()))
        assert [m.status for m in loaded.all_models] == ["invalid", "ok"]
        assert loaded.all_models[0].error == np.inf
        assert loaded.all_models[0].status_detail == se.models[0].status_detail


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

    def test_bed_invalid_expression_returns_inf_and_appends_invalid(self):
        result = self.se.evaluate_expr(["C", "*", "X_1", "X_0"])
        assert result == np.inf
        assert "C*X_1X_0" in self.se.invalid

    def test_bed_callback_stop_sets_should_stop(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = False
        self.se.register_callbacks(cb)
        self.se.evaluate_expr(["C", "+", "X_0"])
        assert self.se.should_stop is True


# ── SR_evaluator.get_metric ───────────────────────────────────────────────────


class TestGetMetric:
    _BEST = ["C", "*", "X_1", "-", "X_0"]  # exact fit for _y

    def test_builtin_r2_for_exact_fit(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        assert se.get_metric(self._BEST, "r2") > 0.999

    def test_builtin_mse_matches_squared_rmse(self):
        se = _make_se()
        rmse = se.evaluate_expr(self._BEST)
        assert se.get_metric(self._BEST, "mse") == pytest.approx(rmse**2, abs=1e-9)

    def test_all_builtin_metrics_callable_by_name(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        for name in ("mse", "mae", "r2", "nrmse"):
            assert np.isfinite(se.get_metric(self._BEST, name))

    def test_default_metric_is_r2(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        assert se.get_metric(self._BEST) == se.get_metric(self._BEST, "r2")

    def test_node_input_accepted(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        tree = tokens_to_tree(self._BEST, se.symbol_library)
        assert se.get_metric(tree, "r2") > 0.999

    def test_custom_callable_receives_predict_X_y(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        seen = {}

        def metric(predict, X, y):
            seen["pred"] = predict(X)
            seen["X"] = X
            seen["y"] = y
            return float(np.max(np.abs(predict(X) - y)))

        result = se.get_metric(self._BEST, metric)
        assert result < 1e-6
        assert seen["X"] is se.X and seen["y"] is se.y
        assert seen["pred"].shape == _y.shape

    def test_custom_callable_can_probe_off_dataset_points(self):
        # The predict callable enables evaluating the model away from X.
        se = _make_se()
        se.evaluate_expr(self._BEST)

        def reaches_far_point(predict, X, y):
            return float(predict(np.array([[0.0, 100.0]]))[0])  # 2*100 - 0 = 200

        assert se.get_metric(self._BEST, reaches_far_point) == pytest.approx(200.0, abs=1e-6)

    def test_unevaluated_expression_raises(self):
        se = _make_se()
        with pytest.raises(ValueError, match="has not been evaluated"):
            se.get_metric(["X_0", "+", "X_1"], "r2")

    def test_unknown_string_metric_raises(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        with pytest.raises(ValueError, match="Unknown metric"):
            se.get_metric(self._BEST, "not_a_metric")

    def test_non_string_non_callable_raises(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        with pytest.raises(TypeError):
            se.get_metric(self._BEST, 123)

    def test_invalid_expression_returns_nan(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"])  # invalid
        assert np.isnan(se.get_metric(["C", "*", "X_1", "X_0"], "r2"))

    def test_bed_mode_returns_nan(self):
        se = SR_evaluator(_X_BED, ground_truth=["C", "+", "X_0"], ranking_function="bed", seed=0)
        se.evaluate_expr(["C", "+", "X_0"])
        assert np.isnan(se.get_metric(["C", "+", "X_0"], "r2"))

    def test_constant_free_expression(self):
        se = _make_se()
        expr = ["X_1", "-", "X_0"]  # no free constants
        se.evaluate_expr(expr)
        assert np.isfinite(se.get_metric(expr, "mae"))

    def test_does_not_count_toward_budget(self):
        se = _make_se()
        se.evaluate_expr(self._BEST)
        before = se.total_evaluations
        se.get_metric(self._BEST, "r2")
        assert se.total_evaluations == before


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
        assert len(r.top_models()) == 1
        assert len(r.all_models) == 2

    def test_top_k_negative_includes_all(self):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        r = se.get_results(top_k=-1)[0]
        assert len(r.top_models()) == 2

    def test_top_k_exceeds_total_capped(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        r = se.get_results(top_k=100)[0]
        assert len(r.top_models()) == 1

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
        er = EvalResult(all_models=[], trace=[], approach_name="")
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
        with pytest.raises(IndexError):
            results.print_results(experiment_number=99)

    def test_negative_experiment_number_prints_from_end(self, capsys):
        results = _single_result()
        results.print_results(experiment_number=-1)
        assert "Best expression" in capsys.readouterr().out

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
        results[0].top_models()[0].add_augmentation("aug", {"x": 5}, "NonExistent.Class")
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "x=5" in out

    def test_model_augmentation_only_type_not_printed(self, capsys):
        # When augmentation data has only _type, parts=[] so nothing printed for that aug
        results = _single_result()
        results[0].top_models()[0].add_augmentation("empty_aug", {}, "NonExistent.Class")
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
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.print_results(detailed=True)
        capsys.readouterr()  # discard

    def test_model_augmentation_filter_applied(self, capsys):
        # Covers line 984: augmentations_filter is applied to model-level augmentation keys
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.print_results(detailed=True, augmentations=["HoldoutMetric"])
        out = capsys.readouterr().out
        assert "HoldoutMetric" in out

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
    def test_for_loop_visits_every_result(self):
        combined = _single_result() + _single_result()
        seen = [r for r in combined]
        assert len(seen) == 2

    def test_iteration_matches_indexing(self):
        combined = _single_result() + _single_result()
        for i, r in enumerate(combined):
            assert r is combined[i]

    def test_iteration_yields_eval_results(self):
        combined = _single_result() + _single_result()
        assert all(isinstance(r, EvalResult) for r in combined)

    def test_iteration_over_empty_results(self):
        assert list(SR_results()) == []

    def test_can_iterate_more_than_once(self):
        combined = _single_result() + _single_result()
        assert len(list(combined)) == len(list(combined)) == 2

    def test_len(self):
        assert len(_single_result()) == 1

    def test_getitem_returns_eval_result(self):
        assert isinstance(_single_result()[0], EvalResult)

    def test_getitem_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            _single_result()[99]

    def test_getitem_negative_index(self):
        results = _single_result() + _single_result()
        assert results[-1] is results.results[1]
        assert results[-2] is results.results[0]

    def test_getitem_negative_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            _single_result()[-2]

    def test_getitem_non_int_raises(self):
        with pytest.raises(TypeError):
            _single_result()["0"]

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
        assert len(loaded[0].top_models()) == len(results[0].top_models())
        np.testing.assert_array_almost_equal(
            loaded[0].top_models()[0].parameters, results[0].top_models()[0].parameters
        )

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

    def test_save_load_holdout_augmentation_round_trip(self, tmp_path):
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        results = se.get_results(top_k=2)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        orig_aug = results[0].augmentations["HoldoutMetric"]
        rest_aug = loaded[0].augmentations["HoldoutMetric"]
        assert rest_aug["_type"] == "SRToolkit.evaluation.result_augmentation.HoldoutMetric"
        assert abs(rest_aug["best_value"] - orig_aug["best_value"]) < 1e-12
        # model-level values survive the JSON round-trip
        orig_value = results[0].top_models()[0].augmentations["HoldoutMetric"]["value"]
        rest_value = loaded[0].top_models()[0].augmentations["HoldoutMetric"]["value"]
        assert abs(rest_value - orig_value) < 1e-12


# ── Constraints ────────────────────────────────────────────────────────────────────────────────
# Expected values follow the module docs "Evaluation statuses" and "Constraints". Constraint
# objects are built inside the tests (or in lambdas), never at import time.

_INF = float("inf")
_LIB1 = lambda: SymbolLibrary.default_symbols(1)  # noqa: E731
_LIB2 = lambda: SymbolLibrary.default_symbols(2)  # noqa: E731
_X1 = _X[:, :1]
_NINE_CONSTANTS = ["C"] + ["+", "C"] * 8
_STD_Y = math.sqrt(16.6875)  # best constant fit to _y: mean 4.25, squared deviations sum 66.75, /4


def _violation(constraint, stage, message, amount):
    return CheckResult(
        passed=False, message=message, amount=amount, implemented=True, constraint=constraint, stage=stage
    )


def _raised(constraint, stage):
    # The recorded result of a check that raised RuntimeError("boom").
    return CheckResult(
        passed=False,
        message="raised RuntimeError: boom",
        amount=_INF,
        constraint=constraint,
        stage=stage,
        error="RuntimeError: boom",
    )


class _Boom(ExpressionConstraint):
    def pre_check(self, expr):
        raise RuntimeError("boom")


class _BoomPost(ExpressionConstraint):
    def post_check(self, model):
        raise RuntimeError("boom")


class _ReturnsBool(ExpressionConstraint):
    def pre_check(self, expr):
        return True


class _Spy(ExpressionConstraint):
    """Records what its checks receive; the lists are shared with the evaluator's shallow copy."""

    def __init__(self):
        self.pre = []
        self.post = []

    def pre_check(self, expr):
        self.pre.append(expr)
        return CheckResult()

    def post_check(self, model):
        self.post.append(model)
        return CheckResult()


class _Unimpl(ExpressionConstraint):
    def pre_check(self, expr):
        return CheckResult(passed=False, implemented=False)

    def post_check(self, model):
        return CheckResult(passed=False, implemented=False)


class _CountingRanking(RankingFunction):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        pass

    def score(self, expr):
        self.calls += 1
        return RankingResult(score=0.5)


class _Recorder(SRCallbacks):
    def __init__(self):
        self.events = []

    def on_expr_evaluated(self, event):
        self.events.append(event)


def _fit_ranking():
    """Scores every expression 0.1 with fitted constant 2.0 and some ranking data."""
    return _FixedRanking(RankingResult(score=0.1, parameters=np.array([2.0]), extra={"k": 1}))


class TestConstraintSetup:
    @pytest.mark.parametrize(
        "make_kwargs, expected",
        [
            (lambda: {}, [("MaxConstants", 8)]),
            (lambda: {"max_constants": 3, "max_expr_length": 20}, [("MaxConstants", 3), ("MaxTokens", 20)]),
            (lambda: {"max_constants": -1}, []),
            (lambda: {"max_constants": -1, "max_expr_length": 0}, [("MaxTokens", 0)]),
            (lambda: {"max_constants": np.int64(2)}, [("MaxConstants", 2)]),
            (lambda: {"required_constraints": []}, [("MaxConstants", 8)]),
            (
                lambda: {"required_constraints": [MaxTreeDepth(4)], "max_expr_length": 5},
                [("MaxConstants", 8), ("MaxTokens", 5), ("MaxTreeDepth", 4)],
            ),
            (lambda: {"required_constraints": [MaxConstants(2)]}, [("MaxConstants", 2)]),
            (lambda: {"required_constraints": [MaxConstants(2)], "max_constants": 2}, [("MaxConstants", 2)]),
            (lambda: {"required_constraints": None}, [("MaxConstants", 8)]),
            (lambda: {"required_constraints": [MaxTokens(4)]}, [("MaxConstants", 8), ("MaxTokens", 4)]),
            (
                lambda: {"required_constraints": [MaxTokens(0)], "max_expr_length": 0},
                [("MaxConstants", 8), ("MaxTokens", 0)],
            ),
        ],
        ids=[
            "default",
            "both",
            "disabled",
            "zero-is-a-limit",
            "numpy-int",
            "empty-list-keeps-default",
            "settings-before-list",
            "list-replaces-default",
            "agreeing-setting",
            "none-is-empty-list",
            "list-MaxTokens-default-setting",
            "agreeing-zero-limit",
        ],
    )
    def test_required_constraints_from_settings_and_list(self, make_kwargs, expected):
        se = _make_se(symbol_library=_LIB2(), **make_kwargs())
        assert isinstance(se.required_constraints, tuple)
        assert [(type(c).__name__, c.limit) for c in se.required_constraints] == expected
        assert all(c.symbol_library is se.symbol_library for c in se.required_constraints)
        assert se.parameter_estimator.estimation_settings["max_constants"] == -1
        assert se.parameter_estimator.estimation_settings["max_expr_length"] == -1

    def test_constraints_stored_as_set_up_shallow_copies(self):
        given = OutputBounds([[1.0, 1.0]], lower=0.0)
        advisory = (c for c in [MaxTreeDepth(3), MaxTokens(5)])
        se = _make_se(symbol_library=_LIB2(), required_constraints=[given], advisory_constraints=advisory)
        _, stored = se.required_constraints  # MaxConstants(8) from the settings comes first
        assert stored is not given and type(stored) is OutputBounds
        assert stored.points is given.points
        assert stored.symbol_library is se.symbol_library
        assert given.symbol_library is None
        assert isinstance(se.advisory_constraints, tuple)
        assert [(type(c).__name__, c.limit) for c in se.advisory_constraints] == [("MaxTreeDepth", 3), ("MaxTokens", 5)]
        assert _make_se(symbol_library=_LIB2(), advisory_constraints=None).advisory_constraints == ()
        assert _make_se(symbol_library=_LIB2()).advisory_constraints == ()

    def test_same_instance_serves_evaluators_with_different_libraries(self):
        shared = RequiredVariables()
        se1 = SR_evaluator(_X1, _y, symbol_library=_LIB1(), required_constraints=[shared])
        se3 = SR_evaluator(
            np.column_stack([_X, _X[:, 0]]),
            _y,
            symbol_library=SymbolLibrary.default_symbols(3),
            required_constraints=[shared],
        )
        assert se1.check_constraints(["X_0"]) == ([], [])
        assert se3.check_constraints(["X_0"]) == ([_violation("RequiredVariables", "pre", "missing X_1, X_2", 2.0)], [])

    @pytest.mark.parametrize(
        "make_kwargs, error, match",
        [
            (lambda: {"required_constraints": [MaxDepth(3)]}, TypeError, "MaxDepth"),
            (lambda: {"advisory_constraints": [MaxNodes(3)]}, TypeError, "MaxNodes"),
            (lambda: {"required_constraints": MaxTokens(3)}, TypeError, "MaxTokens"),
            (lambda: {"required_constraints": "MaxTokens"}, TypeError, "M"),
            (lambda: {"required_constraints": [MaxDepth(3)], "max_constants": 3}, TypeError, "MaxDepth"),
            (lambda: {"required_constraints": [MaxConstants(2)], "max_constants": 3}, ValueError, "max_constants"),
            (lambda: {"required_constraints": [MaxTokens(4)], "max_expr_length": 5}, ValueError, "max_expr_length"),
            (lambda: {"required_constraints": [MaxConstants(3)], "max_constants": 8}, ValueError, "max_constants"),
            (lambda: {"max_constants": True}, ValueError, "max_constants"),
            (lambda: {"max_constants": 2.0}, ValueError, "max_constants"),
            (lambda: {"max_constants": None}, ValueError, "max_constants"),
            (lambda: {"max_expr_length": "5"}, ValueError, "max_expr_length"),
            (lambda: {"required_constraints": [OutputBounds(np.zeros((1, 3)), lower=0.0)]}, ValueError, "3"),
            (lambda: {"advisory_constraints": [RequiredVariables(["X0"])]}, ValueError, "X0"),
        ],
        ids=[
            "grammar-constraint",
            "grammar-constraint-advisory",
            "lone-constraint",
            "string",
            "type-checked-before-values",
            "max_constants-disagrees-with-list",
            "max_expr_length-disagrees-with-list",
            "explicit-default-max_constants-disagrees",
            "bool-limit",
            "float-limit",
            "none-limit",
            "str-limit",
            "setup-error-propagates",
            "advisory-setup-error-propagates",
        ],
    )
    def test_constructor_error_contract(self, make_kwargs, error, match):
        kwargs = make_kwargs()
        with pytest.raises(error, match=match):
            _make_se(symbol_library=_LIB2(), **kwargs)

    @pytest.mark.parametrize(
        "settings, make_list, expected",
        [
            ({}, lambda: None, [("MaxConstants", 8)]),
            (
                {"max_constants": 3, "max_expr_length": 20},
                lambda: [MaxTreeDepth(4)],
                [("MaxConstants", 3), ("MaxTokens", 20), ("MaxTreeDepth", 4)],
            ),
            (
                # A negative setting adds no limit and does not clash with a listed one.
                {"max_constants": -1, "max_expr_length": -1},
                lambda: [MaxConstants(3), MaxTokens(4)],
                [("MaxConstants", 3), ("MaxTokens", 4)],
            ),
        ],
        ids=["default-settings", "given-settings", "negative-settings-keep-listed"],
    )
    def test_required_constraints_round_trip_with_same_settings(self, settings, make_list, expected):
        first = _make_se(symbol_library=_LIB2(), required_constraints=make_list(), **settings)
        again = _make_se(symbol_library=_LIB2(), required_constraints=first.required_constraints, **settings)
        assert [(type(c).__name__, c.limit) for c in first.required_constraints] == expected
        assert [(type(c).__name__, c.limit) for c in again.required_constraints] == expected

    @pytest.mark.parametrize(
        "make_kwargs",
        [
            lambda: {"max_constants": 12},
            lambda: {"max_constants": -1},
            lambda: {"required_constraints": [MaxConstants(12)]},
        ],
        ids=["max_constants-12", "max_constants-disabled", "explicit-MaxConstants-12"],
    )
    def test_estimator_limits_disabled_for_scoring(self, make_kwargs):
        # Regression: the estimator's own default limit (8) must not cut a 9-constant expression.
        se = _make_se(symbol_library=_LIB2(), **make_kwargs())
        error = se.evaluate_expr(_NINE_CONSTANTS)
        assert se.models[0].status == "ok"
        assert error == pytest.approx(_STD_Y, rel=1e-3)


class TestCheckConstraints:
    def _se(self):
        return SR_evaluator(
            _X1,
            _y,
            symbol_library=_LIB1(),
            required_constraints=[MaxConstants(1), MaxTokens(3), OutputBounds([[1.0]], lower=0.0)],
            advisory_constraints=[MaxTreeDepth(2), RequiredVariables()],
        )

    def test_check_constraints_splits_required_and_advisory(self):
        # C * sin(X_0 + C): 2 constants, 8 tokens, height 4. OutputBounds has no pre-check.
        expr = ["C", "*", "sin", "(", "X_0", "+", "C", ")"]
        assert self._se().check_constraints(expr) == (
            [
                _violation("MaxConstants", "pre", "2 constants, limit 1", 1.0),
                _violation("MaxTokens", "pre", "8 tokens, limit 3", 5.0),
            ],
            [_violation("MaxTreeDepth", "pre", "depth 4, limit 2", 2.0)],
        )

    def test_check_constraints_same_class_in_both_lists(self):
        se = SR_evaluator(
            _X1, _y, symbol_library=_LIB1(), required_constraints=[MaxTokens(5)], advisory_constraints=[MaxTokens(2)]
        )
        assert se.check_constraints(["X_0", "+", "C"]) == (
            [],
            [_violation("MaxTokens", "pre", "3 tokens, limit 2", 1.0)],
        )

    def test_check_constraints_has_no_side_effects(self):
        se = SR_evaluator(_X1, _y, symbol_library=_LIB1(), required_constraints=[MaxConstants(0)])
        recorder = _Recorder()
        se.register_callbacks(recorder)
        assert se.check_constraints(["C"]) == ([_violation("MaxConstants", "pre", "1 constants, limit 0", 1.0)], [])
        assert (se.total_evaluations, se.trace, se.models, recorder.events) == (0, [], [], [])

    @pytest.mark.parametrize(
        "make, make_expr, messages",
        [
            (lambda: MaxTokens(3), lambda sl: ["(", "X_0", "+", "C", ")"], ["5 tokens, limit 3"]),
            (lambda: MaxTokens(3), lambda sl: tokens_to_tree(["(", "X_0", "+", "C", ")"], sl), []),
            (lambda: MaxConstants(1), lambda sl: ["C", "+", "C"], ["2 constants, limit 1"]),
            (lambda: MaxTokens(5), lambda sl: tokens_to_tree(["X_0", "+", "X_0", "*", "X_0"], sl), []),
        ],
        ids=["list-as-given", "tree-written-out", "not-simplified", "tree-minimal-brackets"],
    )
    def test_check_constraints_input_form(self, make, make_expr, messages):
        se = SR_evaluator(_X1, _y, symbol_library=_LIB1(), required_constraints=[make()])
        required, _ = se.check_constraints(make_expr(se.symbol_library))
        assert [r.message for r in required] == messages

    @pytest.mark.parametrize(
        "make_kwargs, expr, error, match",
        [
            (lambda: {"max_constants": -1}, ["X_0", "X_0"], ExpressionParseError, None),
            (lambda: {"required_constraints": [MaxTokens(10)]}, ["X_0", "X_0"], ExpressionParseError, None),
            (lambda: {}, "X_0", TypeError, "str"),
        ],
        ids=["parsed-with-no-constraints", "parsed-for-token-only", "wrong-type"],
    )
    def test_check_constraints_error_contract(self, make_kwargs, expr, error, match):
        se = SR_evaluator(_X1, _y, symbol_library=_LIB1(), **make_kwargs())
        with pytest.raises(error, match=match):
            se.check_constraints(expr)

    def test_check_constraints_returns_raising_check(self):
        # The raising check is reported in either list, and the checks after it still run.
        se = SR_evaluator(
            _X1,
            _y,
            symbol_library=_LIB1(),
            required_constraints=[_Boom(), MaxTokens(0)],
            advisory_constraints=[_Boom()],
            max_constants=-1,
        )
        with pytest.warns(UserWarning, match=r"_Boom\.pre_check raised RuntimeError: boom") as record:
            required, advisory = se.check_constraints(["X_0"])
        assert len([w for w in record if issubclass(w.category, UserWarning)]) == 2
        assert (required, advisory) == (
            [_raised("_Boom", "pre"), _violation("MaxTokens", "pre", "1 tokens, limit 0", 1.0)],
            [_raised("_Boom", "pre")],
        )

    @settings(max_examples=60, deadline=None)
    @given(tokens=infix_expressions(), as_tree=st.booleans())
    def test_check_constraints_required_empty_iff_not_rejected(self, tokens, as_tree):
        library = _LIB2()
        se = SR_evaluator(
            _X,
            _y,
            symbol_library=library,
            ranking_function=_FixedRanking(RankingResult(score=0.0)),
            success_threshold=-1,
            required_constraints=[MaxConstants(1), MaxTreeDepth(3), MaxTokens(9)],
        )
        expr = tokens_to_tree(tokens, library) if as_tree else tokens
        failed, _ = se.check_constraints(expr)
        se.evaluate_expr(expr)
        expected = ("rejected", f"{failed[0].constraint}: {failed[0].message}") if failed else ("ok", None)
        assert (se.models[0].status, se.models[0].status_detail) == expected


class TestConstraintEvaluation:
    @pytest.mark.parametrize(
        "make_required, detail",
        [
            (lambda: [MaxTreeDepth(2), MaxConstants(0)], "MaxTreeDepth: depth 3, limit 2"),
            (lambda: [MaxConstants(0), MaxTreeDepth(2)], "MaxConstants: 1 constants, limit 0"),
        ],
        ids=["depth-first", "constants-first"],
    )
    def test_first_failed_required_pre_check_decides(self, make_required, detail):
        se = SR_evaluator(
            _X1, _y, symbol_library=_LIB1(), ranking_function=_CountingRanking(), required_constraints=make_required()
        )
        assert se.evaluate_expr(["C", "*", "sin", "(", "X_0", ")"]) == _INF
        model = se.models[0]
        assert (model.status, model.error, model.status_detail) == ("rejected", _INF, detail)
        assert se.ranking.calls == 0
        assert se.total_evaluations == 1

    def test_required_post_check_rejects_and_keeps_fit(self):
        # Fitted C = 2 gives 2 * (-1) - 0 = -2 at the point (0, -1), below the lower bound 0.
        se = SR_evaluator(
            _X,
            _y,
            symbol_library=_LIB2(),
            ranking_function=_fit_ranking(),
            success_threshold=-1,
            required_constraints=[OutputBounds([[0.0, -1.0]], lower=0.0)],
        )
        assert se.evaluate_expr(["C", "*", "X_1", "-", "X_0"]) == _INF
        model = se.models[0]
        assert (model.status, model.error) == ("rejected", _INF)
        assert model.status_detail == "OutputBounds: output outside [0.0, inf] at 1 of 1 points"
        np.testing.assert_array_equal(model.parameters, [2.0])
        assert model.ranking_data == {"k": 1}

    def test_post_checks_run_only_for_ok_models(self):
        spy = _Spy()
        se = _make_se(symbol_library=_LIB2(), required_constraints=[spy])
        se.evaluate_expr(["log", "(", "X_0", "-", "X_0", ")"])
        assert (se.models[0].status, spy.post) == ("undefined", [])
        se.evaluate_expr(["X_0"])
        assert len(spy.post) == 1

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        "ranking, expr, status, expected",
        [
            (
                lambda: "rmse",
                ["X_0", "-", "X_1"],
                "ok",
                [
                    _violation("MaxTreeDepth", "pre", "depth 2, limit 1", 1.0),
                    _violation("OutputBounds", "post", "output outside [0.5, inf] at 1 of 1 points", 0.5),
                ],
            ),
            (
                lambda: "rmse",
                ["X_0", "-", "X_1", "-", "X_1"],
                "rejected",
                [_violation("MaxTreeDepth", "pre", "depth 3, limit 1", 2.0)],
            ),
            (
                lambda: "rmse",
                ["C", "*", "C", "*", "X_1"],
                "rejected",
                [_violation("MaxTreeDepth", "pre", "depth 3, limit 1", 2.0)],
            ),
            (
                lambda: "rmse",
                ["log", "(", "X_0", "-", "X_0", ")"],
                "undefined",
                [_violation("MaxTreeDepth", "pre", "depth 3, limit 1", 2.0)],
            ),
            (
                lambda: _FixedRanking(RuntimeError("boom")),
                ["X_0", "+", "X_1"],
                "error",
                [_violation("MaxTreeDepth", "pre", "depth 2, limit 1", 1.0)],
            ),
            (lambda: "rmse", ["C", "C"], "invalid", []),
        ],
        ids=[
            "ok-pre-then-post",
            "rejected-by-post-no-advisory-post",
            "rejected-by-pre",
            "undefined",
            "error",
            "invalid-before-constraints",
        ],
    )
    def test_advisory_violations_recorded_by_stage(self, ranking, expr, status, expected):
        # At the point (1, 1): X_0 - X_1 = 0 passes the required lower bound 0 but not the advisory
        # 0.5; X_0 - X_1 - X_1 = -1 fails the required bound (and would fail the advisory one).
        point = [[1.0, 1.0]]
        se = SR_evaluator(
            _X,
            _y,
            symbol_library=_LIB2(),
            ranking_function=ranking(),
            seed=42,
            success_threshold=-1,
            required_constraints=[MaxConstants(1), OutputBounds(point, lower=0.0)],
            advisory_constraints=[MaxTreeDepth(1), OutputBounds(point, lower=0.5)],
        )
        se.evaluate_expr(expr)
        assert se.models[0].status == status
        assert se.models[0].advisory_violations == expected

    @pytest.mark.parametrize(
        "expr",
        [["X_0", "+", "X_0"], ["C", "*", "X_1", "-", "X_0"], ["log", "(", "X_0", "-", "X_0", ")"]],
        ids=["constant-free", "fitted", "undefined"],
    )
    def test_advisory_constraints_never_change_error(self, expr):
        plain = _make_se(symbol_library=_LIB2())
        advised = _make_se(
            symbol_library=_LIB2(),
            advisory_constraints=[MaxTreeDepth(1), RequiredVariables(["X_1"]), OutputBounds([[1.0, 1.0]], lower=100.0)],
        )
        plain.evaluate_expr(expr)
        advised.evaluate_expr(expr)
        assert advised.models[0].advisory_violations != []
        assert (advised.models[0].error, advised.models[0].status) == (plain.models[0].error, plain.models[0].status)

    @pytest.mark.parametrize(
        "make_expr, simplify, expected",
        [
            (lambda sl: ["X_0", "+", "C"], False, ["X_0", "+", "C"]),
            (lambda sl: tokens_to_tree(["(", "X_0", "+", "C", ")"], sl), False, ["X_0", "+", "C"]),
            (
                lambda sl: ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"],
                True,
                ["C", "+", "X_1"],
            ),
        ],
        ids=["list", "tree-written-out", "simplified"],
    )
    def test_checks_receive_stored_expression(self, make_expr, simplify, expected):
        required, advisory = _Spy(), _Spy()
        se = _make_se(symbol_library=_LIB2(), required_constraints=[required], advisory_constraints=[advisory])
        se.evaluate_expr(make_expr(se.symbol_library), simplify_expr=simplify)
        model = se.models[0]
        assert model.expr == expected
        assert (required.pre, advisory.pre) == ([expected], [expected])
        assert [m.expr for m in required.post] == [expected]
        assert [m.expr for m in advisory.post] == [expected]
        np.testing.assert_array_equal(required.post[0].parameters, model.parameters)

    @pytest.mark.parametrize(
        "make, detail, parameters, ranking_data",
        [
            (_Boom, "_Boom.pre_check raised RuntimeError: boom", None, {}),
            (_BoomPost, "_BoomPost.post_check raised RuntimeError: boom", [2.0], {"k": 1}),
            (
                _ReturnsBool,
                "_ReturnsBool.pre_check raised TypeError: pre_check returned bool, expected CheckResult",
                None,
                {},
            ),
        ],
        ids=["pre_check", "post_check", "returns-non-CheckResult"],
    )
    def test_required_check_exception_gives_error_status(self, make, detail, parameters, ranking_data):
        # A pre-check that raises stops before scoring; a post-check that raises keeps the fit. The
        # advisory pre-checks ran before and stay recorded, as for every parsed expression.
        se = SR_evaluator(
            _X,
            _y,
            symbol_library=_LIB2(),
            ranking_function=_fit_ranking(),
            success_threshold=-1,
            required_constraints=[make()],
            advisory_constraints=[MaxTreeDepth(1)],
        )
        recorder = _Recorder()
        se.register_callbacks(recorder)
        check = detail.split(" ")[0]  # "<Class>.<check>"
        with pytest.warns(UserWarning, match=check.replace(".", r".*")):
            assert se.evaluate_expr(["C", "*", "X_1", "-", "X_0"]) == _INF
        model = se.models[0]
        assert (model.status, model.error, model.status_detail) == ("error", _INF, detail)
        assert model.expr == ["C", "*", "X_1", "-", "X_0"]
        assert model.advisory_violations == [_violation("MaxTreeDepth", "pre", "depth 3, limit 1", 2.0)]
        assert (None if model.parameters is None else model.parameters.tolist()) == parameters
        assert model.ranking_data == ranking_data
        assert (se.total_evaluations, len(recorder.events)) == (1, 1)
        assert se.evaluate_expr(["C", "*", "X_1", "-", "X_0"]) == _INF
        assert (se.total_evaluations, len(se.models)) == (2, 1)

    @pytest.mark.parametrize("make, check", [(_Boom, "pre_check"), (_BoomPost, "post_check")], ids=["pre", "post"])
    def test_advisory_check_exception_is_recorded_with_error(self, make, check):
        # The raising check is recorded with error set; the advisory constraints after it are still
        # checked. X_0 + X_0 has height 2 and is 2 at the point (1, 1).
        plain = _make_se(symbol_library=_LIB2())
        advised = _make_se(
            symbol_library=_LIB2(),
            advisory_constraints=[make(), MaxTreeDepth(1), OutputBounds([[1.0, 1.0]], lower=100.0)],
        )
        plain.evaluate_expr(["X_0", "+", "X_0"])
        with pytest.warns(UserWarning, match=f"{make.__name__}.*{check}"):
            advised.evaluate_expr(["X_0", "+", "X_0"])
        model = advised.models[0]
        assert (model.status, model.error) == ("ok", plain.models[0].error)
        raised = _raised(make.__name__, check.split("_")[0])
        assert model.advisory_violations == [
            *([raised] if check == "pre_check" else []),
            _violation("MaxTreeDepth", "pre", "depth 2, limit 1", 1.0),
            *([raised] if check == "post_check" else []),
            _violation("OutputBounds", "post", "output outside [100.0, inf] at 1 of 1 points", 98.0),
        ]

    def test_unimplemented_results_are_ignored(self):
        se = _make_se(symbol_library=_LIB2(), required_constraints=[_Unimpl()], advisory_constraints=[_Unimpl()])
        se.evaluate_expr(["X_0"])
        assert (se.models[0].status, se.models[0].advisory_violations) == ("ok", [])
        assert se.check_constraints(["X_0"]) == ([], [])

    @pytest.mark.parametrize(
        "expr, status, error, detail",
        [
            (["C", "*", "X_1", "-", "X_0"], "ok", 0.0, None),
            (["X_0"], "ok", math.sqrt(22), None),  # residuals 2, -8, -2, 4
            (["X_1", "-", "X_0"], "ok", math.sqrt(29.25), None),  # residual = X_1
            (["C"] + ["+", "C"] * 7, "ok", _STD_Y, None),  # 8 constants: at the default limit
            (_NINE_CONSTANTS, "rejected", _INF, "MaxConstants: 9 constants, limit 8"),
            (["X_0"] + ["+", "X_0"] * 50, "ok", math.sqrt(87997), None),  # 101 tokens: no default limit
            (["X_0", "X_0"], "invalid", _INF, ANY),
            (["log", "(", "X_0", "-", "X_0", ")"], "undefined", _INF, ANY),
        ],
        ids=["exact-fit", "X_0", "X_1-X_0", "eight-constants", "nine-constants", "long", "invalid", "undefined"],
    )
    def test_default_settings_regression(self, expr, status, error, detail):
        se = _make_se()
        assert se.evaluate_expr(expr) == pytest.approx(error, rel=1e-3, abs=1e-6)
        model = se.models[0]
        assert (model.status, model.status_detail, model.advisory_violations) == (status, detail, [])
