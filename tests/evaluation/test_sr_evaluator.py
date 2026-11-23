import json
import logging
import math
import os
import random
import re
import shutil
import signal
import sqlite3
import threading
import warnings
from contextlib import nullcontext
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import SignalAfter, SteppedApproach, infix_expressions, result_answers, result_dict, send_signal

from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation import model_store
from SRToolkit.evaluation.callbacks import (
    CallbackDispatcher,
    SRCallbacks,
)
from SRToolkit.evaluation.model_store import ModelStore
from SRToolkit.evaluation.ranking import RankingFunction, RankingResult, RMSERanking
from SRToolkit.evaluation.results import SR_results
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.expression_constraints import (
    ExpressionConstraint,
    MaxConstants,
    MaxTokens,
    MaxTreeDepth,
    OutputBounds,
    RequiredVariables,
)
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, tokens_to_tree
from SRToolkit.utils.free_constants import Constant, canonicalise, resolve_constants, uniform_default
from SRToolkit.utils.grammar import Grammar, MaxDepth, MaxNodes
from SRToolkit.utils.measures import bed, create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, EvalResult

# ── Shared data ────────────────────────────────────────────────────────────────

_X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
_y = np.array([3.0, 0.0, 3.0, 11.0])  # y = 2*X_1 - X_0; optimal: ["C","*","X_1","-","X_0"]

_X_BED = np.random.default_rng(0).random((20, 2)) - 0.5
_LIB_BED = SymbolLibrary.default_symbols(2)


# ── Concrete helpers ──────────────────────────────────────────────────────────


class _SimpleCb(SRCallbacks):
    pass


_LOGGER = "SRToolkit.evaluation.sr_evaluator"


def _log_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


def _make_se(**kwargs):
    defaults = dict(seed=42, success_threshold=-1)
    defaults.update(kwargs)
    return SR_evaluator(_X, _y, **defaults)


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
        # The message names the offending tokens.
        with pytest.raises(ValueError, match=r"contains a free constant \(C, C_1\)"):
            SR_evaluator(_X, ground_truth=["X_0", "+", "C_1", "+", "C"], ranking_function="rmse", seed=0)

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

    def test_no_callbacks_by_default(self):
        se = _make_se()
        assert se._callbacks.get_callbacks() == []
        assert se.stop_reason is None and se.should_stop is False


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
            symbol_library=_LIB_BED,
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
                symbol_library=_LIB_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
                num_points_sampled=15,
                seed=7,
            )

        assert np.array_equal(build().ranking.bed_X, build().ranking.bed_X)

    def test_bed_use_all_points_wins_over_samplers(self):
        se = SR_evaluator(
            _X_BED,
            symbol_library=_LIB_BED,
            ground_truth=["C", "+", "X_0"],
            ranking_function="bed",
            samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
            num_points_sampled=-1,
            seed=0,
        )
        np.testing.assert_array_equal(se.ranking.bed_X, _X_BED)

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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            se = SR_evaluator(
                _X_BED,
                symbol_library=_LIB_BED,
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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            se = SR_evaluator(
                _X_BED,
                _X_BED[:, 0],
                symbol_library=_LIB_BED,
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0), UniformSampling(0.0, 1.0)],
                seed=0,
            )
        np.testing.assert_array_equal(se.ranking.bed_X, _X_BED)

    def test_bed_y_has_no_default_threshold(self):
        se = SR_evaluator(_X_BED, _X_BED[:, 0], ranking_function="bed", seed=0)
        assert se.success_threshold is None

    def test_bed_sampler_count_must_match_variables(self):
        with pytest.raises(ValueError, match="1 samplers"):
            SR_evaluator(
                _X_BED,
                symbol_library=_LIB_BED,
                ground_truth=["C", "+", "X_0"],
                ranking_function="bed",
                samplers=[UniformSampling(0.0, 1.0)],
                seed=0,
            )

    @pytest.mark.parametrize(
        "settings, expected",
        [({"ranking_seed": 5}, 5), ({"ranking_seed": None}, 1), ({}, 1)],
        ids=["ranking-seed", "ranking-seed-none", "no-ranking-seed"],
    )
    def test_ranking_setup_gets_ranking_seed_else_evaluator_seed(self, settings, expected):
        # Custom rankings get the per-dataset seed without reading ranking_seed themselves.
        se = SR_evaluator(_X, _y, ranking_function=_SeedRecordingRanking(), seed=1, **settings)
        assert se.ranking.seed == expected

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
    def test_single_callback_is_added(self):
        se = _make_se()
        cb = _SimpleCb()
        se.register_callbacks(cb)
        assert se._callbacks.get_callbacks() == [cb]

    def test_dispatcher_callbacks_are_added_and_dispatcher_unchanged(self):
        se = _make_se()
        first, second, third = _SimpleCb(), _SimpleCb(), _SimpleCb()
        se.register_callbacks(first)
        dispatcher = CallbackDispatcher([second, third])
        se.register_callbacks(dispatcher)
        assert se._callbacks.get_callbacks() == [first, second, third]
        assert dispatcher.get_callbacks() == [second, third]

    def test_other_types_are_rejected(self):
        with pytest.raises(TypeError, match="list"):
            _make_se().register_callbacks([_SimpleCb()])

    def test_none_adds_nothing(self):
        se = _make_se()
        se.register_callbacks(None)
        assert se._callbacks.get_callbacks() == []


# ── SR_evaluator stopping ─────────────────────────────────────────────────────


class TestStopReason:
    def test_threshold_records_success_threshold(self):
        se = _make_se(success_threshold=1e-6)
        se.evaluate_expr(["X_0"])
        assert se.stop_reason is None
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert se.stop_reason == "success_threshold"

    def test_budget_on_new_model_records_max_evaluations(self):
        se = _make_se(max_evaluations=2, success_threshold=-1)
        se.evaluate_expr(["X_0"])
        assert se.stop_reason is None
        se.evaluate_expr(["X_1"])
        assert se.stop_reason == "max_evaluations"

    def test_zero_budget_is_rejected(self):
        with pytest.raises(ValueError, match="max_evaluations"):
            _make_se(max_evaluations=0)

    @pytest.mark.parametrize("budget", [None, 10.0, True])
    def test_non_int_budget_is_rejected(self, budget):
        with pytest.raises(TypeError, match="max_evaluations"):
            _make_se(max_evaluations=budget)

    def test_budget_on_cache_hit_records_max_evaluations(self):
        se = _make_se(max_evaluations=2, success_threshold=-1)
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["X_0"])
        assert se.stop_reason == "max_evaluations"

    def test_threshold_wins_on_last_budgeted_call(self):
        se = _make_se(max_evaluations=2, success_threshold=1e-6)
        se.evaluate_expr(["X_0"])
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert se.stop_reason == "success_threshold"

    def test_stop_keeps_first_reason_and_later_calls_warn(self):
        se = _make_se()
        se.stop("grammar exhausted")
        se.stop("other")
        assert se.should_stop is True and se.stop_reason == "grammar exhausted"
        with pytest.warns(UserWarning, match="grammar exhausted"):
            assert np.isnan(se.evaluate_expr(["X_0"]))
        assert se.total_evaluations == 0

    @pytest.mark.parametrize("reason", ["", None, 3])
    def test_stop_rejects_non_string_or_empty_reason(self, reason):
        with pytest.raises(ValueError, match="reason"):
            _make_se().stop(reason)

    def test_should_stop_is_read_only(self):
        with pytest.raises(AttributeError):
            _make_se().should_stop = True

    def test_callback_reason_is_recorded_after_evaluators_own(self):
        se = _make_se(max_evaluations=1, success_threshold=-1)
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = "time_limit"
        cb.on_best_expression.return_value = None
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0"])
        assert se.stop_reason == "max_evaluations"

    def test_callback_reason_is_recorded(self):
        se = _make_se()
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = None
        cb.on_best_expression.return_value = "plateau"
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0"])
        assert se.stop_reason == "plateau"

    def test_callback_reason_on_cache_hit_is_recorded(self):
        se = _make_se(success_threshold=-1)
        se.evaluate_expr(["X_0"])
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = "plateau"
        se.register_callbacks(cb)
        se.evaluate_expr(["X_0"])
        assert se.stop_reason == "plateau"

    def test_error_equal_to_threshold_does_not_stop(self):
        se = _make_se(success_threshold=np.inf)
        assert se.evaluate_expr(["C", "*", "X_1", "X_0"]) == np.inf  # invalid: error inf
        assert se.stop_reason is None

    def test_get_results_records_reason_or_returned(self):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        assert se.get_results()[0].stop_reason == "returned"
        se.stop("done")
        assert se.get_results()[0].stop_reason == "done"


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
        se.stop("test")
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
        assert se.stop_reason == "callback:MagicMock"

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


class _SeedRecordingRanking(RankingFunction):
    """Records the seed its setup receives; scores every expression 0."""

    name = "seed_recording"

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        self.seed = seed

    def score(self, expr, free_constants=None):
        return RankingResult(score=0.0)


class _OddAmount(ExpressionConstraint):
    """Fails every expression with an amount that cannot be saved."""

    def pre_check(self, expr):
        return CheckResult(passed=False, message="odd", amount=object())


class _FixedRanking(RankingFunction):
    """Returns a fixed RankingResult, or raises if it is an exception."""

    name = "fixed"

    def __init__(self, outcome):
        self.outcome = outcome

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        pass

    def score(self, expr, free_constants=None):
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

    # Reason: clause (module docs, status 3: a model that cannot be saved is stored with status
    # "error", empty ranking_data and no parameters, with a warning; it never counts as a success).
    @pytest.mark.parametrize(
        "parameters, extra, advisory, path",
        [
            (np.array([1.5]), {"fit": object()}, [], r"ranking_data\['fit'\]"),
            (np.array([1 + 2j]), {"iterations": 3}, [], "parameters"),
            (np.array([1.5]), {"iterations": 3}, [_OddAmount()], r"advisory_violations\[0\]"),
        ],
        ids=["extra", "complex-parameters", "advisory-amount"],
    )
    def test_unsaveable_model_gives_error_with_warning(self, parameters, extra, advisory, path):
        outcome = RankingResult(score=0.0, parameters=parameters, extra=extra)
        se = SR_evaluator(
            _X, _y, ranking_function=_FixedRanking(outcome), success_threshold=1.0, advisory_constraints=advisory
        )
        with pytest.warns(UserWarning, match="cannot be saved"):
            assert se.evaluate_expr(["X_0"]) == np.inf
        model = se.models[0]
        assert (model.status, model.error, model.ranking_data, model.parameters) == ("error", np.inf, {}, None)
        assert model.advisory_violations == []
        assert model.status_detail.startswith("the model cannot be saved: ")
        assert re.search(path, model.status_detail)
        assert se.stop_reason is None  # the score 0.0 below the threshold does not count

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
            def score(self, expr, free_constants=None):
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

    def score(self, expr, free_constants=None):
        self.calls += 1
        return RankingResult(score=0.5)


# Exact ties in top_models: X_0*X_1 and X_1*X_0 (same error, same length); X_0 and X_0+X_0-X_0
# (same error, lengths 1 and 5). "X_0 +" is invalid (error inf). Two cache hits.
_TIE_CALLS = [
    ["X_0"],
    ["X_1"],
    ["X_0"],
    ["X_0", "+"],
    ["X_0", "*", "X_1"],
    ["X_1", "*", "X_0"],
    ["X_0", "+", "X_0", "-", "X_0"],
    ["C"],
    ["X_1"],
    ["C", "*", "X_1", "-", "X_0"],
]


class TestCompactResults:
    # Reason: clause (keep_all_models=False keeps the top_k models and a RunSummary; the result
    # answers as the full one, including ties, a failed model and top_k above the finite models).
    @pytest.mark.parametrize("top_k", [1, 3, 20])
    def test_get_results_compact_answers_like_full(self, top_k):
        se = _make_se(seed=0, success_threshold=1e-3)
        for expr in _TIE_CALLS:
            se.evaluate_expr(expr)
        full = se.get_results("A", top_k)[0]
        compact = se.get_results("A", top_k, keep_all_models=False)[0]
        assert compact.summary is not None
        assert (compact.all_models, compact.trace) == (full.top_models(top_k), [])
        assert (compact.stop_reason, compact.success_threshold, compact.checkpoint) == (full.stop_reason, 1e-3, None)
        # Errors: X_1 2.35 < C 4.09 < X_0 4.69 (length 1), X_0*X_1 31.7 (length 3), the last 6e-9.
        assert [m.expr for m in compact.pareto_front()] == [["X_1"], ["C", "*", "X_1", "-", "X_0"]]
        budgets = range(-1, len(_TIE_CALLS) + 3)
        np.testing.assert_equal(result_answers(compact, top_k, budgets), result_answers(full, top_k, budgets))

    # Reason: clause (Raises: ValueError if keep_all_models is False and top_k <= 0).
    @pytest.mark.parametrize("top_k", [0, -1])
    def test_get_results_compact_rejects_top_k_not_positive(self, top_k):
        se = _make_se()
        se.evaluate_expr(["X_0"])
        with pytest.raises(ValueError, match="top_k"):
            se.get_results(top_k=top_k, keep_all_models=False)


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


# ── Background knowledge ──────────────────────────────────────────────────────


def _lib_pmc():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _task_parts(**kwargs):
    # The parts of a task's background knowledge, as SR_evaluator takes them (flat).
    parts = dict(
        advisory_constraints=[MaxTreeDepth(4)],
        description="Toy task",
        variables={"X_0": VariableInfo("x", "position")},
        target=VariableInfo("F"),
    )
    parts.update(kwargs)
    return parts


def _bk_from_parts(lib, advisory_constraints=(), constant_bounds=(-5.0, 5.0), **parts):
    return BackgroundKnowledge(lib, constraints=advisory_constraints, constant_bounds=constant_bounds, **parts)


class TestBackgroundKnowledge:
    def test_background_knowledge_built_from_arguments(self):
        lib = _lib_pmc()
        hint = MaxTreeDepth(4)
        grammar = Grammar.from_symbol_library(lib)
        samplers = [UniformSampling(0, 1), UniformSampling(2, 3)]
        parts = _task_parts(advisory_constraints=[hint], reference_grammar=grammar)
        se = SR_evaluator(_X, _y, symbol_library=lib, constant_bounds=(-2, 2), samplers=samplers, **parts)
        bk = se.background_knowledge
        expected = _bk_from_parts(lib, constant_bounds=(-2, 2), samplers=samplers, **parts)
        assert bk.to_dict() == expected.to_dict()
        assert bk.symbol_library is lib
        assert se.advisory_constraints is bk.constraints
        assert bk.constraints[0] is not hint and bk.constraints[0].symbol_library is lib
        assert hint.symbol_library is None
        with pytest.raises(AttributeError):
            se.background_knowledge = expected

    def test_minimal_background_knowledge_defaults(self):
        se = SR_evaluator(_X, _y)
        bk = se.background_knowledge
        assert bk.symbol_library is se.symbol_library
        assert bk.constant_bounds == (-5.0, 5.0)
        assert bk.constraints == ()
        assert bk.samplers is None
        assert (bk.description, bk.variables, bk.target) == ("", {}, None)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"constant_bounds": (2, -2)}, None),
            ({"variables": {"X_9": VariableInfo("z")}}, "X_9"),
            ({"samplers": [UniformSampling(0, 1)]}, None),
            ({"default_constant": Constant("C_0")}, "default_constant"),
        ],
        ids=["bad-bounds", "unknown-variable", "samplers-count", "default-with-label"],
    )
    def test_background_knowledge_argument_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            SR_evaluator(_X, _y, symbol_library=_lib_pmc(), **kwargs)

    # Reason: clause (default_constant setting -> background knowledge -> evaluator.default_constant,
    # completed from constant_bounds -> the ranking's default_constant setting -> EvalResult.default_constant;
    # kwargs keep what was passed). y = 50 x: C reaches 50 only under the default's bounds (0, inf);
    # under the uniform default on (-2, 3) the RMSE would be at least 47 * rms(x).
    def test_default_constant_setting_reaches_fit_and_results(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        template = Constant(bounds=(0, float("inf")))
        se = SR_evaluator(
            X,
            50.0 * X[:, 0],
            symbol_library=SymbolLibrary.default_symbols(1),
            constant_bounds=(-2, 3),
            default_constant=template,
            seed=0,
        )
        assert se.evaluate_expr(["C", "*", "X_0"]) < 1e-4
        expected = Constant(prior=UniformSampling(-2.0, 3.0), bounds=(0, float("inf"))).to_dict()
        assert se.default_constant.to_dict() == expected
        assert se.background_knowledge.default_constant.to_dict() == expected
        assert se.get_results()[0].default_constant.to_dict() == expected
        assert se.kwargs["default_constant"] is template

    def test_describe_lists_settings_limits_as_enforced(self):
        lib = _lib_pmc()
        se = SR_evaluator(
            _X, _y, symbol_library=lib, required_constraints=[MaxTreeDepth(3)], max_expr_length=10, **_task_parts()
        )
        expected = _bk_from_parts(lib, **_task_parts())
        assert se.describe() == expected.describe([MaxConstants(8), MaxTokens(10), MaxTreeDepth(3)])

    def test_describe_and_priors_ignore_answer_ranking_and_budget(self):
        lib = _lib_pmc()
        a = SR_evaluator(
            _X, _y, symbol_library=lib, ground_truth=["X_0", "+", "X_1"], success_threshold=1e-6, **_task_parts()
        )
        b = SR_evaluator(
            _X,
            symbol_library=lib,
            **_task_parts(),
            ranking_function="bed",
            ground_truth=["X_0", "*", "X_1"],
            success_threshold=0.5,
            max_evaluations=7,
            seed=0,
        )
        assert a.describe() == b.describe()
        assert a.priors() == b.priors()

    def test_priors_are_constraint_summaries(self):
        se = SR_evaluator(_X, _y, symbol_library=_lib_pmc(), required_constraints=[MaxTreeDepth(3)], **_task_parts())
        assert se.priors() == {
            "required": [MaxConstants(8).summary(), MaxTreeDepth(3).summary()],
            "advisory": [MaxTreeDepth(4).summary()],
        }


class TestGetModel:
    def test_get_model_returns_cached_result_without_billing(self):
        se = SR_evaluator(_X, _y, symbol_library=_LIB2())
        assert se.get_model(["X_0"]) is None
        se.evaluate_expr(["X_0"])
        for _ in range(3):
            assert se.get_model(["X_0"]).to_dict() == se.models[0].to_dict()
        assert se.total_evaluations == 1 and se.trace == [0]

    def test_get_model_accepts_node(self):
        lib = _LIB2()
        se = SR_evaluator(_X, _y, symbol_library=lib)
        se.evaluate_expr(["X_0", "+", "X_1"])
        assert se.get_model(tokens_to_tree(["X_0", "+", "X_1"], lib)).to_dict() == se.models[0].to_dict()

    def test_get_model_does_not_simplify(self):
        se = SR_evaluator(_X, _y, symbol_library=_LIB2())
        se.evaluate_expr(["X_0", "+", "X_0"], simplify_expr=True)
        assert se.get_model(["X_0", "+", "X_0"]) is None


# ── Free-constant specs (constants rework, stage 1 step 5) ────────────────────

_X3 = np.array([[1.0], [2.0], [3.0]])
_y3 = 2 * _X3[:, 0]
_E = ["C_0", "*", "X_0"]


def _fix(label, value):
    return Constant(label=label, value=value, fixed=True)


def _spec_se(**kwargs):
    return SR_evaluator(
        _X3, _y3, symbol_library=SymbolLibrary.default_symbols(1), seed=0, success_threshold=-1, **kwargs
    )


class _RecordingRanking(RankingFunction):
    """A custom ranking that records the specs it is given."""

    name = "recording"

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        self.received = []

    def score(self, expr, free_constants=None):
        self.received.append(free_constants)
        return RankingResult(score=1.0)


class TestFreeConstants:
    # Reason: clause (evaluate_expr Raises: a spec that cannot be saved raises TypeError; nothing is
    # recorded or counted).
    def test_unsaveable_spec_raises_and_records_nothing(self):
        class _Unsaveable(UniformSampling):
            def to_dict(self):
                return {**super().to_dict(), "extra": object()}

        se = SR_evaluator(_X, _y)
        with pytest.raises(TypeError):
            se.evaluate_expr(["C_0", "*", "X_0"], [Constant(label="C_0", prior=_Unsaveable(0, 1))])
        assert (se.total_evaluations, len(se.models)) == (0, 0)

    # Reason: clause (evaluate_expr: same tokens with different specs are a different candidate,
    # specs stored as ModelResult.free_constants and passed to the ranking; get_model by tokens + specs).
    def test_same_tokens_with_different_specs_are_separate_models(self):
        se = _spec_se()
        spec = _fix("C_0", 1.0)
        se.evaluate_expr(_E)
        se.evaluate_expr(_E, [spec])
        assert len(se.models) == 2
        assert se.trace == [0, 1]
        assert se.models[0].error < 1e-6  # C_0 fitted to 2
        # C_0 fixed to 1: residual 2x - x = x, RMSE = sqrt((1 + 4 + 9) / 3)
        assert se.models[1].error == pytest.approx(math.sqrt(14 / 3))
        np.testing.assert_array_equal(se.models[1].parameters, [1.0])
        assert [c.to_dict() for c in se.models[1].free_constants] == [spec.to_dict()]
        assert se.models[0].free_constants is None
        assert se.get_model(_E).to_dict() == se.models[0].to_dict()
        assert se.get_model(_E, [_fix("C_0", 1.0)]).to_dict() == se.models[1].to_dict()

    # Reason: clause (get_metric uses the model of the given specs; ValueError for specs not evaluated).
    def test_get_metric_uses_the_model_of_the_given_specs(self):
        se = _spec_se()
        se.evaluate_expr(_E)
        se.evaluate_expr(_E, [_fix("C_0", 1.0)])
        assert se.get_metric(_E, "mae", [_fix("C_0", 1.0)]) == pytest.approx(2.0)  # mean |2x - x|
        assert se.get_metric(_E, "mae") < 1e-6
        assert se.get_model(_E, [_fix("C_0", 5.0)]) is None
        with pytest.raises(ValueError, match="evaluated"):
            se.get_metric(_E, "r2", [_fix("C_0", 5.0)])

    # Reason: clause (specs in any order, None or empty means none, stored sorted by label, matched by
    # value) + implied (the caller's list is not mutated).
    @pytest.mark.parametrize(
        "first, second, stored",
        [
            (
                lambda: [Constant("C_1", bounds=(-1.0, 1.0)), Constant("C_0", prior=UniformSampling(0.0, 2.0))],
                lambda: [Constant("C_0", prior=UniformSampling(0.0, 2.0)), Constant("C_1", bounds=(-1.0, 1.0))],
                ("C_0", "C_1"),
            ),
            (lambda: None, lambda: [], None),
        ],
        ids=["reordered", "empty-is-none"],
    )
    def test_equivalent_specs_share_one_entry(self, first, second, stored):
        se = _spec_se()
        expr = ["C_0", "*", "X_0", "+", "C_1"]
        first_specs = first()
        given_order = None if first_specs is None else [c.label for c in first_specs]
        se.evaluate_expr(expr, first_specs)
        se.evaluate_expr(expr, second())
        assert len(se.models) == 1
        assert se.trace == [0, 0]
        labels = se.models[0].free_constants
        assert (None if labels is None else tuple(c.label for c in labels)) == stored
        assert (None if first_specs is None else [c.label for c in first_specs]) == given_order
        assert se.get_model(expr, second()).to_dict() == se.models[0].to_dict()

    # Reason: clause (error contract: specs that do not fit, or specs with simplify_expr=True, raise
    # ValueError and nothing is recorded or counted; check_constraints checks them the same way).
    @pytest.mark.parametrize(
        "call, match",
        [
            (lambda se: se.evaluate_expr(_E, [_fix("C_9", 1.0)]), "C_9"),
            (lambda se: se.evaluate_expr(_E, [_fix("C_0", 1.0), _fix("C_0", 2.0)]), "C_0"),
            (lambda se: se.evaluate_expr(_E, [Constant(value=1.0, fixed=True)]), "label"),
            (lambda se: se.evaluate_expr(_E, [_fix("C_0", 1.0)], simplify_expr=True), "simplif"),
            (lambda se: se.check_constraints(_E, [_fix("C_9", 1.0)]), "C_9"),
        ],
        ids=["unknown-label", "duplicate-label", "no-label", "simplify", "check_constraints"],
    )
    def test_invalid_specs_raise_and_record_nothing(self, call, match):
        se = _spec_se()
        with pytest.raises(ValueError, match=match):
            call(se)
        assert se.models == []
        assert se.trace == []
        assert se.total_evaluations == 0

    # Reason: implied (specs are checked against the evaluator's symbol library, not the default one):
    # with the free constant renamed to K, K_0 is a labelled constant that a spec can address.
    def test_specs_are_checked_with_the_evaluators_library(self):
        se = SR_evaluator(
            _X3, _y3, symbol_library=SymbolLibrary.default_symbols(1, const_symbol="K"), seed=0, success_threshold=-1
        )
        expr = ["K_0", "*", "X_0"]
        assert se.check_constraints(expr, [_fix("K_0", 1.0)]) == ([], [])
        assert se.evaluate_expr(expr, [_fix("K_0", 1.0)]) == pytest.approx(math.sqrt(14 / 3))

    # Reason: boundary (the ValueError applies to a well-formed expr only; a malformed one is
    # "invalid", stored with its specs and keyed by them).
    def test_malformed_expression_with_specs_is_invalid_and_keyed_by_specs(self):
        se = _spec_se()
        expr = ["C_0", "*", "X_0", "X_0"]
        spec = _fix("C_0", 1.0)
        assert se.evaluate_expr(expr, [spec]) == math.inf
        assert se.models[0].status == "invalid"
        assert [c.to_dict() for c in se.models[0].free_constants] == [spec.to_dict()]
        se.evaluate_expr(expr)
        assert len(se.models) == 2

    # Reason: clause (the ranking's score receives the specs as free_constants: None without specs,
    # the stored tuple with them).
    def test_custom_ranking_receives_specs(self):
        ranking = _RecordingRanking()
        se = _spec_se(ranking_function=ranking)
        se.evaluate_expr(_E)
        se.evaluate_expr(_E, [_fix("C_0", 1.0)])
        assert ranking.received[0] is None
        assert [c.to_dict() for c in ranking.received[1]] == [_fix("C_0", 1.0).to_dict()]

    # Reason: clause (BED receives the specs; default_constant reaches EvalResult and survives
    # save/load) + plan test "resampling from a saved result". C_1 - C_1 makes the score independent
    # of the default's draws: every column is x + 3 against x, so BED = 3.
    def test_bed_score_with_specs_is_reproduced_from_saved_result(self, tmp_path):
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        lib = SymbolLibrary.default_symbols(1)
        se = SR_evaluator(
            X,
            ground_truth=["X_0"],
            ranking_function="bed",
            symbol_library=lib,
            seed=0,
            success_threshold=-1,
            constant_bounds=(-2.0, 4.0),
            num_points_sampled=-1,
        )
        expr = ["X_0", "+", "C_0", "+", "(", "C_1", "-", "C_1", ")"]
        spec = _fix("C_0", 3.0)
        assert se.evaluate_expr(expr, [spec]) == pytest.approx(3.0, abs=1e-9)
        se.get_results().save(str(tmp_path))
        result = SR_results.load(str(tmp_path))[0]
        model = result.all_models[0]
        default = uniform_default((-2.0, 4.0))
        assert result.default_constant.to_dict() == default.to_dict()
        assert se.default_constant.to_dict() == default.to_dict()
        resolved = resolve_constants(model.expr, model.free_constants, result.default_constant, lib)
        expected_c0 = Constant("C_0", prior=default.prior, bounds=(-2.0, 4.0), value=3.0, fixed=True)
        assert [c.to_dict() for c in resolved] == [expected_c0.to_dict(), default.to_dict()]
        rebed = bed(
            model.expr,
            ["X_0"],
            X,
            symbol_library=lib,
            seed=1,
            free_constants1=model.free_constants,
            default_constant=result.default_constant,
        )
        assert rebed == pytest.approx(model.error, abs=1e-9)


_y3_affine = 2 * _X3[:, 0] + 2  # fitted by C_0*X_0 + C_0 with C_0 = 2


def _canon_se(lib=None, **kwargs):
    lib = SymbolLibrary.default_symbols(1) if lib is None else lib
    return SR_evaluator(_X3, _y3_affine, symbol_library=lib, seed=0, success_threshold=-1, **kwargs)


class TestCanonicalisation:
    # Reason: clause (with canonicalise_constants=True the canonical form is cached and stored: C_1*X_0+C_1
    # after C_4*X_0+C_4 is a cache hit) + implied (canonicalised with the evaluator's library; input not mutated).
    @pytest.mark.parametrize(
        "const, lib",
        [
            ("C", lambda: SymbolLibrary.default_symbols(1)),
            ("K", lambda: SymbolLibrary.default_symbols(1, const_symbol="K")),
        ],
        ids=["C", "K"],
    )
    def test_relabelled_expressions_are_one_canonical_model(self, const, lib):
        se = _canon_se(lib())
        first = f"{const}_4 * X_0 + {const}_4".split()
        error = se.evaluate_expr(first, canonicalise_constants=True)
        assert first == f"{const}_4 * X_0 + {const}_4".split()
        assert se.evaluate_expr(f"{const}_1 * X_0 + {const}_1".split(), canonicalise_constants=True) == error
        assert len(se.models) == 1
        assert se.trace == [0, 0]
        assert se.models[0].expr == f"{const}_0 * X_0 + {const}_0".split()
        assert error < 1e-6
        np.testing.assert_allclose(se.models[0].parameters, [2.0], rtol=1e-6)

    # Reason: clause (canonicalise_constants defaults to False: relabelled expressions are separate
    # models stored as written) + clause (get_model / get_metric match as given, without canonicalising).
    def test_default_keeps_labels_and_lookups_match_as_given(self):
        se = _canon_se()
        se.evaluate_expr(["C_4", "*", "X_0", "+", "C_4"])
        se.evaluate_expr(["C_1", "*", "X_0", "+", "C_1"])
        assert [m.expr for m in se.models] == [["C_4", "*", "X_0", "+", "C_4"], ["C_1", "*", "X_0", "+", "C_1"]]
        assert se.get_model(["C_0", "*", "X_0", "+", "C_0"]) is None
        canonical = _canon_se()
        canonical.evaluate_expr(["C_4", "*", "X_0", "+", "C_4"], canonicalise_constants=True)
        assert canonical.get_model(["C_4", "*", "X_0", "+", "C_4"]) is None
        assert canonical.get_model(["C_0", "*", "X_0", "+", "C_0"]).to_dict() == canonical.models[0].to_dict()
        assert canonical.get_metric(["C_0", "*", "X_0", "+", "C_0"], "mae") < 1e-6

    # Reason: clause (get_model / get_metric with canonicalise_constants=True find a model under another
    # labelling, without billing) + implied (with the evaluator's library).
    @pytest.mark.parametrize(
        "const, lib",
        [
            ("C", lambda: SymbolLibrary.default_symbols(1)),
            ("K", lambda: SymbolLibrary.default_symbols(1, const_symbol="K")),
        ],
        ids=["C", "K"],
    )
    def test_lookups_canonicalise_with_the_flag(self, const, lib):
        se = _canon_se(lib())
        se.evaluate_expr(f"{const}_0 * X_0 + {const}_0".split(), canonicalise_constants=True)
        relabelled = f"{const}_3 * X_0 + {const}_3".split()
        assert se.get_model(relabelled, canonicalise_constants=True).to_dict() == se.models[0].to_dict()
        assert se.get_metric(relabelled, "mae", canonicalise_constants=True) < 1e-6
        assert se.total_evaluations == 1

    # Reason: clause (canonicalise_constants is ignored with free_constants: labels with specs are kept).
    def test_canonicalise_is_ignored_with_specs(self):
        se = _canon_se()
        expr = ["C_4", "*", "X_0", "+", "C_4"]
        spec = _fix("C_4", 1.0)
        se.evaluate_expr(expr, [spec], canonicalise_constants=True)
        assert se.models[0].expr == expr
        assert se.get_model(expr, [spec], canonicalise_constants=True).to_dict() == se.models[0].to_dict()
        assert se.get_metric(expr, "mae", [spec], canonicalise_constants=True) > 0

    # Reason: clause (the canonical form is scored, and a fit is seeded by its tokens): with max_iter=0
    # the parameters are the seeded starting point, so they match those of the canonical form.
    def test_fit_starts_from_the_canonical_forms_seed(self):
        relabelled = _canon_se(max_iter=0)
        relabelled.evaluate_expr(["C_4", "*", "X_0", "+", "C_4"], canonicalise_constants=True)
        canonical = _canon_se(max_iter=0)
        canonical.evaluate_expr(["C_0", "*", "X_0", "+", "C_0"])
        np.testing.assert_array_equal(relabelled.models[0].parameters, canonical.models[0].parameters)

    # Reason: clause (canonicalised after simplifying): simplify keeps
    # the shared labels C_5 and C_2, the stored form must be canonical (a fixed point of canonicalise).
    def test_simplified_expression_is_canonicalised(self):
        lib = SymbolLibrary.default_symbols(2)
        se = SR_evaluator(_X, _y, symbol_library=lib, seed=0, success_threshold=-1)
        se.evaluate_expr("C_5 * X_0 + C_5 + C_2 * X_1 + C_2".split(), simplify_expr=True, canonicalise_constants=True)
        stored = se.models[0].expr
        assert canonicalise(stored, lib) == stored
        assert sorted(t for t in stored if t.startswith("C")) == ["C_0", "C_0", "C_1", "C_1"]

    # Reason: boundary (a malformed expression is canonicalised token-wise, still "invalid").
    def test_malformed_expression_is_canonicalised_and_invalid(self):
        se = _canon_se()
        assert se.evaluate_expr(["C_3", "*", "X_0", "X_0", "C_3"], canonicalise_constants=True) == math.inf
        assert se.models[0].status == "invalid"
        assert se.models[0].expr == ["C_0", "*", "X_0", "X_0", "C_0"]
        se.evaluate_expr(["C_5", "*", "X_0", "X_0", "C_5"], canonicalise_constants=True)
        assert len(se.models) == 1


class TestCacheKey:
    # Reason: regression (#cache-key: the cache was keyed on "".join(tokens), so token lists that
    # join to the same string shared one entry).
    @pytest.mark.parametrize(
        "lib, X, y, first, second, second_error, second_status",
        [
            (
                lambda: SymbolLibrary.default_symbols(2),
                _X,
                _X[:, 0] * _X[:, 1],
                ["X_1", "*", "X_0"],
                ["X_1", "*", "X", "_0"],
                math.inf,
                "invalid",
            ),
            (
                lambda: _u_library(),
                np.array([[1.0], [2.0]]),
                np.array([-1.0, -2.0]),
                ["u", "-", "X_0"],
                ["u-", "X_0"],
                0.0,
                "ok",  # first: residual 10, RMSE 10
            ),
        ],
        ids=["typo", "custom-lib"],
    )
    def test_cache_key_distinguishes_token_lists_with_equal_join(
        self, lib, X, y, first, second, second_error, second_status
    ):
        se = SR_evaluator(X, y, symbol_library=lib(), seed=0, success_threshold=-1)
        se.evaluate_expr(first)
        assert se.evaluate_expr(second) == pytest.approx(second_error)
        assert len(se.models) == 2
        assert se.models[1].status == second_status


def _u_library():
    """``u`` is a named constant 10, so ``u - X_0`` and ``u- X_0`` (unary minus) both join to "u-X_0"."""
    lib = SymbolLibrary.from_symbol_list(["-", "u-"], 1)
    lib.add_symbol("u", "lit", 5, "10.0")
    return lib


# ── Checkpoints and snapshots (stages 2a, 2c) ─────────────────────────────────

_CX = np.array([[1.0], [2.0], [3.0]])
_CY = 2 * _CX[:, 0]
# Models 0 (X_0), 1 (C*X_0, error ~0), a hit, 2 (invalid), 3 (X_0+C), a hit: trace [0, 1, 0, 2, 3, 1].
_CALLS = [["X_0"], ["C", "*", "X_0"], ["X_0"], ["X_0", "X_0"], ["X_0", "+", "C"], ["C", "*", "X_0"]]
_SCORED = []  # one entry per expression scored by a _CountingRMSE


class _CountingRMSE(RMSERanking):
    def score(self, expr, free_constants=None):
        _SCORED.append(expr)
        return super().score(expr, free_constants)


def _ckpt_se(checkpoint=None, **kwargs):
    # "mean" initialisation keeps fitted constants independent of the random state.
    args = dict(seed=0, success_threshold=-1, ranking_function=_CountingRMSE(), initialization="mean")
    args.update(kwargs)
    return SR_evaluator(_CX, _CY, checkpoint=checkpoint, **args)


def _evaluate(se, calls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # calls after a stop
        for expr in calls:
            se.evaluate_expr(expr)


def _snapshot(se):
    return [m.to_dict() for m in se.models], list(se.trace), se.invalid


def _saved_run(path, calls=_CALLS, **kwargs):
    se = _ckpt_se(path, **kwargs)
    _evaluate(se, calls)
    se.close()


def _straight(calls, **kwargs):
    se = _ckpt_se(**kwargs)
    _evaluate(se, calls)
    return se


class _StateLog(SRCallbacks):
    def __init__(self, state=None):
        self.log = []
        self.seen = 0
        self.state = state

    def on_expr_evaluated(self, event):
        self.seen += 1
        self.log.append("eval")

    def on_best_expression(self, event):
        self.log.append("best")

    def get_state(self):
        return {"seen": self.seen} if self.state is None else self.state

    def set_state(self, state):
        self.log.append(("set_state", state))
        self.seen = state["seen"]


def _stepped(path=None, resume=False, signal_after=None, approach=None, **kwargs):
    # One session of a SteppedApproach (batches of 2) run through SR_approach.run.
    se = _ckpt_se(path, resume=resume, **kwargs)
    approach = approach or SteppedApproach()
    if signal_after is not None:
        se.register_callbacks(SignalAfter(signal_after))
    with se.stop_on_signals():
        approach.run(se, seed=0)
    return se, approach


class TestCheckpoints:
    def setup_method(self):
        _SCORED.clear()

    @pytest.mark.parametrize("signal_after", [3, 4], ids=["mid-step", "end-of-step"])
    def test_resumed_run_equals_uninterrupted_run(self, tmp_path, signal_after):
        straight, _ = _stepped(max_evaluations=10)
        path = tmp_path / "run.sqlite"
        first, _ = _stepped(path, signal_after=signal_after, max_evaluations=10)
        assert (first.stop_reason, first.total_evaluations) == ("interrupted", 4)  # the step was completed
        first.close()
        np.random.seed(123)  # the resumed run must not depend on the random states left behind
        random.seed(123)
        se, approach = _stepped(path, resume=True, max_evaluations=10)
        assert approach.started is False  # continued from the snapshot
        assert _snapshot(se) == _snapshot(straight)
        assert result_dict(se.get_results()[0]) == result_dict(straight.get_results()[0])

    def test_crash_resumes_from_last_snapshot(self, tmp_path, monkeypatch):
        # S3: models committed after the snapshot stay readable, but a resume cuts back to the snapshot.
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        _evaluate(se, _CALLS[:2])
        se.save_snapshot(b"state")
        now[0] = 30.0
        _evaluate(se, _CALLS[2:4])  # a commit is due at the third call, the fourth stays pending
        crash = tmp_path / "crash"
        crash.mkdir()
        for suffix in ("", "-wal", "-shm"):  # the files as a crash would leave them
            if os.path.exists(f"{path}{suffix}"):
                shutil.copy(f"{path}{suffix}", crash / f"run.sqlite{suffix}")
        se.close()
        assert SR_results.from_checkpoint(crash / "run.sqlite")[0].trace == [0, 1, 0]
        resumed = _ckpt_se(crash / "run.sqlite", resume=True)
        assert (resumed.resumed_from.state, resumed.resumed_from.num_calls) == (b"state", 2)
        assert (list(resumed.trace), resumed.total_evaluations, resumed.stop_reason) == ([0, 1], 2, None)
        assert SR_results.from_checkpoint(crash / "run.sqlite")[0].trace == [0, 1]  # rewound and committed

    def test_resume_without_snapshot_starts_empty(self, tmp_path):
        path = tmp_path / "run.sqlite"
        _saved_run(path, _CALLS[:2])
        se = _ckpt_se(path, resume=True)
        assert (se.resumed_from, se.models, list(se.trace)) == (None, [], [])

    @pytest.mark.parametrize(
        "kwargs, approach, reason",
        [
            ({"max_evaluations": 4}, lambda: SteppedApproach(), "max_evaluations"),
            ({"success_threshold": 100.0}, lambda: SteppedApproach(), "success_threshold"),
            ({}, lambda: SteppedApproach(stop_after_asks=2, stop_with="done"), "done"),
            ({}, lambda: SteppedApproach(stop_after_asks=2), "returned"),  # saved by get_results
        ],
        ids=["budget", "threshold", "approach-stop", "returned"],
    )
    def test_resumed_finished_run_stays_stopped(self, tmp_path, kwargs, approach, reason):
        path = tmp_path / "run.sqlite"
        first, _ = _stepped(path, approach=approach(), **kwargs)
        expected = first.get_results()[0].to_dict()
        assert expected["stop_reason"] == reason
        first.close()
        se, _ = _stepped(path, resume=True, approach=approach(), **kwargs)
        assert se.stop_reason == reason
        assert se.get_results()[0].to_dict() == expected  # nothing more was evaluated

    def test_wall_time_ends_at_the_stop_and_adds_up_across_sessions(self, tmp_path, monkeypatch):
        now = [100.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])  # the same module as the evaluator's
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        now[0] = 110.0
        se.stop("interrupted")
        now[0] = 150.0  # after the stop: not counted
        assert (se.wall_time, se.get_results()[0].wall_time) == (10.0, 10.0)
        se.close()
        now[0] = 200.0
        se = _ckpt_se(path, resume=True)
        now[0] = 205.0
        assert se.wall_time == 15.0  # still running: until now, plus the first session
        se.stop("done")
        se.close()
        assert SR_results.from_checkpoint(path)[0].wall_time == 15.0

    def test_snapshot_saves_wall_time(self, tmp_path, monkeypatch):
        now = [100.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        now[0] = 105.0
        se.save_snapshot(b"state")
        se.close()  # as after a crash: not stopped, no results taken
        assert SR_results.from_checkpoint(path)[0].wall_time == 5.0

    def test_resume_restores_best_error(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        _evaluate(se, [["X_0", "+", "C"], ["X_0"]])  # best error sqrt(2/3)
        se.save_snapshot(b"state")
        se.close()
        se = _ckpt_se(path, resume=True)
        cb = _StateLog()
        se.register_callbacks(cb)
        se.evaluate_expr(["C"])  # error sqrt(8/3): not a new best
        first = list(cb.log)
        se.evaluate_expr(["C", "*", "X_0"])  # error ~0
        assert (first, sorted(cb.log[len(first) :])) == (["eval"], ["best", "eval"])

    def test_save_snapshot_records_state_and_callbacks(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        cb = _StateLog()
        se.register_callbacks(cb)
        _evaluate(se, _CALLS[:2])
        se.save_snapshot(b"state")
        snap = ModelStore(path, "read").snapshot
        assert (snap.state, snap.num_calls, snap.num_models, snap.callbacks) == (
            b"state",
            2,
            2,
            CallbackDispatcher([cb]).get_state(),
        )

    def test_snapshots_need_a_checkpoint(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        in_memory, on_disk = _ckpt_se(), _ckpt_se(tmp_path / "run.sqlite")
        _evaluate(in_memory, _CALLS[:1])
        _evaluate(on_disk, _CALLS[:1])
        now[0] = 299.0
        assert (in_memory.snapshot_due, on_disk.snapshot_due) == (False, False)
        now[0] = 300.0  # the default interval
        assert (in_memory.snapshot_due, on_disk.snapshot_due) == (False, True)
        assert (in_memory.save_snapshot(b"state"), in_memory.resumed_from) == (None, None)

    def test_snapshot_interval(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        every_10s = _ckpt_se(tmp_path / "a.sqlite", snapshot_interval=10)
        _evaluate(every_10s, _CALLS[:1])
        now[0] = 10.0
        assert every_10s.snapshot_due
        never, _ = _stepped(tmp_path / "b.sqlite", snapshot_interval=None, max_evaluations=4)
        now[0] = 1e9
        assert not never.snapshot_due
        assert ModelStore(tmp_path / "b.sqlite", "read").snapshot.num_calls == 4  # taken when the loop ended

    @pytest.mark.parametrize("interval", [0, -1.0, float("nan")])
    def test_snapshot_interval_must_be_positive(self, interval):
        with pytest.raises(ValueError, match="snapshot_interval"):
            _ckpt_se(snapshot_interval=interval)

    @pytest.mark.parametrize("resumed", [False, True], ids=["new-run", "resumed-run"])
    def test_snapshot_before_the_first_event_saves_the_callback_states(self, tmp_path, resumed):
        # A new run saves the callbacks' current states; a resumed run, whose callbacks have not
        # been restored yet, keeps the states it resumed with.
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        se.register_callbacks(_StateLog())
        if resumed:
            _evaluate(se, _CALLS[:2])
            se.save_snapshot(b"state")  # seen = 2
            se.close()
            se = _ckpt_se(path, resume=True)
            se.register_callbacks(_StateLog())  # seen = 0 until restored
        se.save_snapshot(b"again")
        callbacks = ModelStore(path, "read").snapshot.callbacks
        assert [c["state"] for c in callbacks] == [{"seen": 2 if resumed else 0}]

    @pytest.mark.parametrize("snapshot", [False, True], ids=["no-snapshot", "calls-after-snapshot"])
    def test_resumed_finished_run_is_kept_whole(self, tmp_path, snapshot):
        # A finished run is not cut back to its snapshot, nor emptied without one (a search approach).
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        _evaluate(se, _CALLS[:2])
        if snapshot:
            se.save_snapshot(b"state")
        _evaluate(se, _CALLS[2:4])
        se.stop("done")
        se.close()
        se = _ckpt_se(path, resume=True)
        assert (se.stop_reason, se.total_evaluations) == ("done", 4)
        se.close()
        assert SR_results.from_checkpoint(path)[0].stop_reason == "done"

    def test_callback_state_restored_from_snapshot(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        se.register_callbacks(_StateLog())
        _evaluate(se, _CALLS[:2])
        se.save_snapshot(b"state")  # seen = 2
        _evaluate(se, _CALLS[2:4])
        se.close()
        se = _ckpt_se(path, resume=True)
        saved, extra = _StateLog(), _StateLog()  # the second has no saved state
        se.register_callbacks(saved)
        se.register_callbacks(extra)
        se.evaluate_expr(["X_0"])
        assert (saved.log, extra.log) == ([("set_state", {"seen": 2}), "eval"], ["eval"])

    @pytest.mark.parametrize(
        "changed, name",
        [
            ({"X": _CX + 1}, r"\bX\b"),
            ({"y": _CY + 1}, r"\by\b"),
            ({"ground_truth": ["C", "*", "X_0"]}, "ground_truth"),
            ({"ranking_function": _CountingRMSE()}, "ranking_function"),
            ({"max_iter": 50}, "max_iter"),
            ({"required_constraints": [MaxTreeDepth(10)]}, "required_constraints"),
            ({"advisory_constraints": [MaxTreeDepth(10)]}, "advisory_constraints"),
            ({"symbol_library": SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=1)}, "symbol_library"),
            ({"constant_bounds": (-1.0, 1.0)}, "default_constant"),
            ({"seed": 1}, "seed"),
            ({"max_evaluations": 7}, "max_evaluations"),  # S6: no unmarked extended runs
            ({"success_threshold": 0.5}, "success_threshold"),
        ],
        ids=[
            "X",
            "y",
            "ground_truth",
            "ranking",
            "setting",
            "required",
            "advisory",
            "symbols",
            "default_constant",
            "seed",
            "max_evaluations",
            "success_threshold",
        ],
    )
    def test_changed_fingerprint_part_raises(self, tmp_path, changed, name):
        path = tmp_path / "run.sqlite"
        base = {"X": _CX, "y": _CY, "seed": 0}
        se = SR_evaluator(**base, checkpoint=path)
        se.evaluate_expr(["X_0"])
        se.close()
        with pytest.raises(ValueError, match=name):
            SR_evaluator(**{**base, **changed}, checkpoint=path, resume=True)
        SR_evaluator(**base, checkpoint=path, resume=True).close()  # the file is kept

    @pytest.mark.parametrize(
        "make_file, kwargs, error, match",
        [
            (None, {"checkpoint": None}, ValueError, "checkpoint"),
            (None, {}, FileNotFoundError, None),
            (lambda path: path.write_text("keep"), {}, ValueError, "checkpoint"),
        ],
        ids=["no-checkpoint", "missing-file", "not-a-checkpoint"],
    )
    def test_resume_errors(self, tmp_path, make_file, kwargs, error, match):
        path = tmp_path / "run.sqlite"
        if make_file is not None:
            make_file(path)
        with pytest.raises(error, match=match):
            _ckpt_se(**{"checkpoint": path, **kwargs}, resume=True)

    @pytest.mark.parametrize(
        "kwargs, name",
        [({"metadata": {"f": object()}}, "metadata"), ({"extra_setting": object()}, "extra_setting")],
        ids=["metadata", "setting"],
    )
    def test_unsaveable_metadata_or_setting_with_checkpoint_raises(self, tmp_path, kwargs, name):
        path = tmp_path / "run.sqlite"
        with pytest.raises(TypeError, match=name):  # the message names what cannot be saved
            _ckpt_se(path, **kwargs)
        assert sorted(p.name for p in tmp_path.iterdir()) == []  # no file, -wal or -shm
        _ckpt_se(**kwargs).close()  # accepted without a checkpoint

    def test_unsaveable_callback_state_raises_at_snapshot(self, tmp_path):
        se = _ckpt_se(tmp_path / "run.sqlite")
        se.register_callbacks(_StateLog(state={"f": object()}))
        se.evaluate_expr(["X_0"])
        with pytest.raises(TypeError):
            se.save_snapshot(b"state")

    def test_close_leaves_a_complete_checkpoint(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        _evaluate(se, _CALLS[:3])
        expected = se.get_results()[0].to_dict()
        se.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.sqlite"]
        # as get_results gave, but the run never stopped, so it reads as unfinished
        loaded = SR_results.from_checkpoint(path)[0].to_dict()
        assert loaded.pop("checkpoint")["path"] == str(path)
        assert expected.pop("checkpoint") is None
        assert loaded == {**expected, "stop_reason": "interrupted"}
        assert se.close() is None

    @pytest.mark.parametrize(
        "use",
        [
            lambda se: len(se.models),
            lambda se: list(se.trace),
            lambda se: se.evaluate_expr(["X_0"]),
            lambda se: se.get_model(["X_0"]),
            lambda se: se.get_metric(["X_0"]),
            lambda se: se.get_results(),
        ],
        ids=["models", "trace", "evaluate_expr", "get_model", "get_metric", "get_results"],
    )
    def test_use_after_close_raises(self, tmp_path, use):
        se = _ckpt_se(tmp_path / "run.sqlite")
        se.evaluate_expr(["X_0"])
        se.close()
        with pytest.raises(sqlite3.ProgrammingError):
            use(se)


class TestSteps:
    @pytest.mark.parametrize(
        "end, raises",
        [(lambda: None, nullcontext()), (lambda: _raise_boom(), pytest.raises(RuntimeError, match="boom"))],
        ids=["normal-end", "exception"],
    )
    def test_step_defers_first_signal_to_block_end(self, tmp_path, end, raises):
        se = _ckpt_se(tmp_path / "run.sqlite")
        with raises, se.stop_on_signals():
            with se.step():
                se.evaluate_expr(["X_0"])
                send_signal(signal.SIGINT)
                assert se.should_stop is False
                se.evaluate_expr(["X_0", "+", "C"])  # still counted
                end()
        assert (se.stop_reason, se.total_evaluations) == ("interrupted", 2)

    def test_signal_outside_a_step_waits_for_the_end_of_the_next_step(self, tmp_path):
        se = _ckpt_se(tmp_path / "run.sqlite")
        with se.stop_on_signals():
            send_signal(signal.SIGINT)
            with se.step():
                se.evaluate_expr(["X_0"])
                assert se.should_stop is False
            assert se.stop_reason == "interrupted"
        assert se.total_evaluations == 1

    @pytest.mark.parametrize(
        "kwargs, stop",
        [({"max_evaluations": 1}, lambda se: None), ({}, lambda se: se.stop("done"))],
        ids=["budget", "stop"],
    )
    def test_step_does_not_defer_other_stops(self, kwargs, stop):
        se = _ckpt_se(**kwargs)
        with se.step():
            se.evaluate_expr(["X_0"])
            stop(se)
            assert se.should_stop is True
            with pytest.warns(UserWarning):
                assert np.isnan(se.evaluate_expr(["X_0", "+", "C"]))
        assert se.total_evaluations == 1

    def test_second_signal_inside_step_raises(self, tmp_path):
        se = _ckpt_se(tmp_path / "run.sqlite")
        with pytest.raises(KeyboardInterrupt), se.stop_on_signals(), se.step():
            send_signal(signal.SIGINT)
            send_signal(signal.SIGINT)
        assert se.stop_reason == "interrupted"


def _raise_boom():
    raise RuntimeError("boom")


# ── Stopping on signals (stage 2b) ─────────────────────────────────────────────


def _sentinel_handler(signum, frame):
    raise AssertionError("the previous handler was called")


class TestStopOnSignals:
    @pytest.mark.parametrize(
        "sig, safe_point",
        [
            (signal.SIGINT, lambda se: se.should_stop),
            (signal.SIGINT, lambda se: se.evaluate_expr(["C", "*", "X_0"])),  # nan, not counted
            (signal.SIGINT, lambda se: None),  # the end of the block
            (signal.SIGTERM, lambda se: se.should_stop),
        ],
        ids=["should_stop", "evaluate_expr", "end-of-block", "sigterm"],
    )
    def test_first_signal_stops_at_next_safe_point(self, tmp_path, sig, safe_point):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # a call after the stop
            with se.stop_on_signals():
                se.evaluate_expr(["X_0"])
                send_signal(sig)
                assert se.stop_reason is None  # the handler only sets a flag
                safe_point(se)
        assert (se.stop_reason, se.total_evaluations) == ("interrupted", 1)
        saved = SR_results.from_checkpoint(path)[0]  # the writer is still open: the stop committed
        assert (saved.trace, saved.stop_reason) == ([0], "interrupted")

    @pytest.mark.parametrize(
        "between", [lambda se: None, lambda se: se.should_stop], ids=["back-to-back", "after-a-safe-point"]
    )
    def test_second_signal_raises_keyboard_interrupt(self, tmp_path, between):
        path = tmp_path / "run.sqlite"
        se = _ckpt_se(path)
        with pytest.raises(KeyboardInterrupt):
            with se.stop_on_signals():
                se.evaluate_expr(["X_0"])
                send_signal(signal.SIGINT)
                between(se)
                send_signal(signal.SIGINT)
        assert se.stop_reason == "interrupted"  # the pending stop is applied on the way out
        assert SR_results.from_checkpoint(path)[0].stop_reason == "interrupted"

    def test_signal_after_a_stop_keeps_the_first_reason(self, tmp_path):
        se = _ckpt_se(tmp_path / "run.sqlite")
        with pytest.raises(KeyboardInterrupt):
            with se.stop_on_signals():
                se.stop("done")
                send_signal(signal.SIGINT)
                assert (se.should_stop, se.stop_reason) == (True, "done")
                send_signal(signal.SIGINT)
        assert se.stop_reason == "done"

    def _raise(self):
        raise RuntimeError("boom")

    @pytest.mark.parametrize(
        "body, raises",
        [
            (lambda self: None, nullcontext()),
            (lambda self: self._raise(), pytest.raises(RuntimeError, match="boom")),
            (lambda self: (send_signal(signal.SIGINT), send_signal(signal.SIGINT)), pytest.raises(KeyboardInterrupt)),
        ],
        ids=["normal-exit", "exception", "second-signal"],
    )
    def test_handlers_restored_after_block(self, tmp_path, body, raises):
        se = _ckpt_se(tmp_path / "run.sqlite")
        old = {sig: signal.signal(sig, _sentinel_handler) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with raises:
                with se.stop_on_signals():
                    installed = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
                    body(self)
            assert _sentinel_handler not in installed
            assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == (
                _sentinel_handler,
                _sentinel_handler,
            )
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)

    def test_block_in_worker_thread_runs_without_handlers(self):
        se = _ckpt_se()
        before = signal.getsignal(signal.SIGINT)
        errors = []

        def work():
            try:
                with se.stop_on_signals():
                    se.evaluate_expr(["X_0"])
            except Exception as e:  # reported below
                errors.append(e)

        worker = threading.Thread(target=work)
        worker.start()
        worker.join()
        assert (errors, list(se.trace), signal.getsignal(signal.SIGINT)) == ([], [0], before)
