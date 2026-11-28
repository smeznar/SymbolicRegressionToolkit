import dataclasses
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.evaluation.ranking import BEDRanking, RankingFunction, RankingResult, RMSERanking
from SRToolkit.evaluation.result_augmentation import (
    ExpressionSimplifier,
    ExpressionToLatex,
    HoldoutMetric,
)
from SRToolkit.evaluation.results import SR_results
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.expression import Expression
from SRToolkit.utils.expression_simplifier import DefaultRules, simplify
from SRToolkit.utils.free_constants import Constant, uniform_default
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EvalResult, ModelResult, RunSummary

# ── Shared data ────────────────────────────────────────────────────────────────

_X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
_y = np.array([3.0, 0.0, 3.0, 11.0])  # 2*X_1 - X_0

_X_BED = np.random.default_rng(0).random((20, 2)) - 0.5

_sl = SymbolLibrary.default_symbols(num_variables=2)

_BEST_EXPR = ["C", "*", "X_1", "-", "X_0"]
_OTHER_EXPR = ["X_0", "+", "X_1"]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _rmse_se(**kwargs):
    defaults = dict(seed=42, success_threshold=-1)
    defaults.update(kwargs)
    return SR_evaluator(_X, _y, **defaults)


def _bed_se(**kwargs):
    defaults = dict(
        seed=42,
        success_threshold=-1,
        ranking_function="bed",
        ground_truth=["C", "+", "X_0"],
    )
    defaults.update(kwargs)
    return SR_evaluator(_X_BED, **defaults)


def _one_model_result(se=None):
    """One expression; top_models=[best], all_models=[best]."""
    if se is None:
        se = _rmse_se()
    se.evaluate_expr(_BEST_EXPR)
    return se, se.get_results(top_k=1)[0]


def _two_model_result(se=None):
    """Two expressions; top_models=[best], all_models=[best, other]."""
    if se is None:
        se = _rmse_se()
    se.evaluate_expr(_BEST_EXPR)
    se.evaluate_expr(_OTHER_EXPR)
    return se, se.get_results(top_k=1)[0]


# ── ExpressionToLatex.__init__ ────────────────────────────────────────────────


class TestExpressionToLatexInit:
    def test_stores_symbol_library_scope_verbose_name(self):
        aug = ExpressionToLatex(_sl, scope="best", verbose=True, name="myLatex")
        assert aug.symbol_library is _sl
        assert aug.scope == "best"
        assert aug.verbose is True
        assert aug.name == "myLatex"

    def test_defaults(self):
        aug = ExpressionToLatex(_sl)
        assert aug.scope == "top"
        assert aug.verbose is False
        assert aug.name == "ExpressionToLatex"

    def test_invalid_scope_raises(self):
        with pytest.raises(Exception, match="scope"):
            ExpressionToLatex(_sl, scope="invalid")


# ── ExpressionToLatex.write_results ──────────────────────────────────────────


class TestExpressionToLatexWriteResults:
    def test_best_scope_writes_best_expr_latex_to_eval_result(self):
        _, result = _one_model_result()
        ExpressionToLatex(_sl, scope="best").write_results(result)
        assert "best_expr_latex" in result.augmentations["ExpressionToLatex"]

    def test_latex_value_on_perfect_expr(self):
        _, result = _one_model_result()
        ExpressionToLatex(_sl).write_results(result)
        latex = result.augmentations["ExpressionToLatex"]["best_expr_latex"]
        assert latex == r"$C_{0} \cdot X_{1} - X_{0}$"

    def test_best_scope_does_not_write_to_models(self):
        _, result = _one_model_result()
        ExpressionToLatex(_sl, scope="best").write_results(result)
        assert "ExpressionToLatex" not in result.top_models()[0].augmentations

    def test_top_scope_writes_to_top_models(self):
        _, result = _two_model_result()
        ExpressionToLatex(_sl, scope="top").write_results(result)
        assert "ExpressionToLatex" in result.top_models()[0].augmentations
        assert "expr_latex" in result.top_models()[0].augmentations["ExpressionToLatex"]

    def test_top_scope_does_not_write_to_non_top_models(self):
        _, result = _two_model_result()
        ExpressionToLatex(_sl, scope="top").write_results(result)
        # all_models[1] is _OTHER_EXPR, which is not in top_models
        assert "ExpressionToLatex" not in result.all_models[1].augmentations

    def test_all_scope_writes_to_all_models(self):
        _, result = _two_model_result()
        ExpressionToLatex(_sl, scope="all").write_results(result)
        assert "ExpressionToLatex" in result.all_models[1].augmentations

    def test_best_expr_failure_verbose_warns(self):
        _, result = _one_model_result()
        aug = ExpressionToLatex(_sl, scope="best", verbose=True)
        with patch("SRToolkit.evaluation.result_augmentation.tokens_to_tree", side_effect=ValueError("bad")):
            with pytest.warns(UserWarning, match="best expression"):
                aug.write_results(result)

    def test_best_expr_failure_silent(self):
        _, result = _one_model_result()
        aug = ExpressionToLatex(_sl, scope="best", verbose=False)
        with patch("SRToolkit.evaluation.result_augmentation.tokens_to_tree", side_effect=ValueError("bad")):
            with patch("SRToolkit.evaluation.result_augmentation.warnings.warn") as mock_warn:
                aug.write_results(result)
        mock_warn.assert_not_called()

    def test_model_conversion_failure_verbose_warns(self):
        _, result = _one_model_result()
        aug = ExpressionToLatex(_sl, scope="top", verbose=True)
        mock_tree = MagicMock()
        mock_tree.to_latex.return_value = "$X_0$"
        with patch(
            "SRToolkit.evaluation.result_augmentation.tokens_to_tree",
            side_effect=[mock_tree, ValueError("model fail")],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_model_conversion_failure_silent(self):
        _, result = _one_model_result()
        aug = ExpressionToLatex(_sl, scope="top", verbose=False)
        mock_tree = MagicMock()
        mock_tree.to_latex.return_value = "$X_0$"
        with patch(
            "SRToolkit.evaluation.result_augmentation.tokens_to_tree",
            side_effect=[mock_tree, ValueError("model fail")],
        ):
            with patch("SRToolkit.evaluation.result_augmentation.warnings.warn") as mock_warn:
                aug.write_results(result)
        mock_warn.assert_not_called()

    def test_all_scope_model_failure_verbose_warns(self):
        # Exception in the all_models loop with verbose=True
        _, result = _two_model_result()
        aug = ExpressionToLatex(_sl, scope="all", verbose=True)
        mock_tree = MagicMock()
        mock_tree.to_latex.return_value = "$X_0$"
        # calls: best_expr, all_models[0], all_models[1] (fails)
        with patch(
            "SRToolkit.evaluation.result_augmentation.tokens_to_tree",
            side_effect=[mock_tree, mock_tree, ValueError("all fail")],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_all_scope_annotates_each_model_once(self):
        _, result = _two_model_result()
        ExpressionToLatex(_sl, scope="all").write_results(result)
        for model in result.all_models:
            assert list(model.augmentations) == ["ExpressionToLatex"]


# ── ExpressionToLatex.format_eval_result ─────────────────────────────────────


class TestExpressionToLatexFormatEvalResult:
    def test_returns_string_when_latex_present(self):
        out = ExpressionToLatex.format_eval_result({"best_expr_latex": "$x$"})
        assert "$x$" in out
        assert out != ""

    def test_returns_empty_when_key_absent(self):
        assert ExpressionToLatex.format_eval_result({}) == ""

    def test_returns_empty_when_latex_is_empty_string(self):
        assert ExpressionToLatex.format_eval_result({"best_expr_latex": ""}) == ""


# ── ExpressionToLatex.format_model_result ────────────────────────────────────


class TestExpressionToLatexFormatModelResult:
    def test_returns_string_when_latex_present(self):
        out = ExpressionToLatex.format_model_result({"expr_latex": "$x$"})
        assert "$x$" in out
        assert out != ""

    def test_returns_empty_when_key_absent(self):
        assert ExpressionToLatex.format_model_result({}) == ""

    def test_returns_empty_when_latex_is_empty_string(self):
        assert ExpressionToLatex.format_model_result({"expr_latex": ""}) == ""


# ── ExpressionToLatex.to_dict / from_dict ────────────────────────────────────


class TestExpressionToLatexSerialization:
    def test_to_dict_has_expected_keys(self):
        aug = ExpressionToLatex(_sl, scope="best", verbose=True, name="L")
        d = aug.to_dict("", "")
        assert d["format_version"] == 1
        assert d["type"] == "ExpressionToLatex"
        assert d["scope"] == "best"
        assert d["verbose"] is True
        assert d["name"] == "L"

    def test_from_dict_round_trip(self):
        aug = ExpressionToLatex(_sl, scope="all", verbose=True, name="myL")
        d = aug.to_dict("", "")
        restored = ExpressionToLatex.from_dict(d)
        assert restored.scope == "all"
        assert restored.verbose is True
        assert restored.name == "myL"

    def test_from_dict_invalid_format_version_raises(self):
        aug = ExpressionToLatex(_sl)
        d = aug.to_dict("", "")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            ExpressionToLatex.from_dict(d)


# ── ExpressionSimplifier.__init__ ─────────────────────────────────────────────


class TestExpressionSimplifierInit:
    def test_stores_attributes(self):
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True, name="Simp")
        assert aug.symbol_library is _sl
        assert aug.scope == "all"
        assert aug.verbose is True
        assert aug.name == "Simp"

    def test_defaults(self):
        aug = ExpressionSimplifier(_sl)
        assert aug.scope == "top"
        assert aug.verbose is False
        assert aug.name == "ExpressionSimplifier"

    # Reason: clause (Raises: TypeError for rules that are not SimplificationRules or None; ValueError
    # for an unknown scope, limits that are not non-negative ints (bool rejected), max_time not
    # None or positive (bool, nan rejected)). The message names the argument.
    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"scope": "invalid"}, ValueError),
            ({"rules": "d"}, TypeError),
            ({"max_iterations": "30"}, ValueError),
            ({"max_iterations": -1}, ValueError),
            ({"max_iterations": True}, ValueError),
            ({"max_nodes": -1}, ValueError),
            ({"max_time": 0}, ValueError),
            ({"max_time": True}, ValueError),
            ({"max_time": float("nan")}, ValueError),
        ],
        ids=[
            "scope", "rules", "iterations-string", "iterations-negative", "iterations-bool", "nodes-negative",
            "time-zero", "time-bool", "time-nan",
        ],
    )  # fmt: skip
    def test_invalid_arguments_raise(self, kwargs, error):
        with pytest.raises(error, match=next(iter(kwargs))):
            ExpressionSimplifier(_sl, **kwargs)

    # Reason: clause (Raises: ValueError for a fold that is neither None nor a fold mode, TypeError for
    # a fold_aware that is neither a bool nor None).
    @pytest.mark.parametrize(
        "kwargs, error, match",
        [({"fold": "all"}, ValueError, "^fold must"), ({"fold_aware": "yes"}, TypeError, "^fold_aware must")],
        ids=["fold", "fold-aware"],
    )
    def test_invalid_fold_arguments_raise(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            ExpressionSimplifier(_sl, **kwargs)


# ── ExpressionSimplifier.write_results ───────────────────────────────────────


class TestExpressionSimplifierWriteResults:
    def test_best_scope_writes_simplified_best_expr(self):
        _, result = _one_model_result()
        ExpressionSimplifier(_sl, scope="best").write_results(result)
        aug = result.augmentations["ExpressionSimplifier"]
        assert "simplified_best_expr" in aug
        assert isinstance(aug["simplified_best_expr"], str)
        assert len(aug["simplified_best_expr"]) > 0

    def test_best_scope_does_not_write_to_models(self):
        _, result = _one_model_result()
        ExpressionSimplifier(_sl, scope="best").write_results(result)
        assert "ExpressionSimplifier" not in result.top_models()[0].augmentations

    def test_top_scope_writes_simplified_expr_to_top_models(self):
        _, result = _one_model_result()
        ExpressionSimplifier(_sl, scope="top").write_results(result)
        assert "ExpressionSimplifier" in result.top_models()[0].augmentations

    def test_all_scope_writes_to_all_models(self):
        _, result = _two_model_result()
        ExpressionSimplifier(_sl, scope="all").write_results(result)
        assert "ExpressionSimplifier" in result.all_models[1].augmentations

    def test_simplification_failure_verbose_warns(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="best", verbose=True)
        with patch("SRToolkit.evaluation.result_augmentation.simplify", side_effect=ValueError("bad")):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_simplification_failure_silent(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="best", verbose=False)
        with patch("SRToolkit.evaluation.result_augmentation.simplify", side_effect=ValueError("bad")):
            with patch("SRToolkit.evaluation.result_augmentation.warnings.warn") as mock_warn:
                aug.write_results(result)
        mock_warn.assert_not_called()

    def test_model_simplification_failure_verbose_warns(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[(["X_0"], None), ValueError("model fail")],
        ):
            with pytest.warns(UserWarning, match="model fail"):
                aug.write_results(result)
        assert result.augmentations["ExpressionSimplifier"]["simplified_best_expr"] == "X_0"

    def test_model_simplification_failure_silent(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top", verbose=False)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[(["X_0"], None), ValueError("model fail")],
        ):
            with patch("SRToolkit.evaluation.result_augmentation.warnings.warn") as mock_warn:
                aug.write_results(result)
        mock_warn.assert_not_called()
        assert result.augmentations["ExpressionSimplifier"]["simplified_best_expr"] == "X_0"

    def test_best_expr_invalid_return_warns_verbose(self):
        # simplify returns a pair whose expression is not a token list: nothing is stored, a warning
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="best", verbose=True)
        with patch("SRToolkit.evaluation.result_augmentation.simplify", return_value=(42, None)):
            with pytest.warns(UserWarning, match="Unable to simplify"):
                aug.write_results(result)
        assert "simplified_best_expr" not in result.augmentations["ExpressionSimplifier"]

    def test_top_model_invalid_return_warns_verbose(self):
        # The top model's simplified expression is not a token list: warning, nothing stored for it
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[(["X_0"], None), (42, None)],
        ):
            with pytest.warns(UserWarning, match="Unable to simplify"):
                aug.write_results(result)
        assert result.augmentations["ExpressionSimplifier"]["simplified_best_expr"] == "X_0"
        assert "simplified_expr" not in result.top_models()[0].augmentations["ExpressionSimplifier"]

    def test_all_model_invalid_return_warns_verbose(self):
        # The second model's simplified expression is not a token list: warning, nothing stored for it
        _, result = _two_model_result()
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[(["X_0"], None), (["X_0"], None), (42, None)],
        ):
            with pytest.warns(UserWarning, match="Unable to simplify"):
                aug.write_results(result)
        assert result.all_models[0].augmentations["ExpressionSimplifier"]["simplified_expr"] == "X_0"
        assert "simplified_expr" not in result.all_models[1].augmentations["ExpressionSimplifier"]

    # Reason: boundary ("A result without models gets an entry without simplified_best_expr").
    def test_result_without_models_gets_entry_without_expression(self):
        result = _rmse_se().get_results()[0]
        ExpressionSimplifier(_sl, scope="top", verbose=True).write_results(result)
        assert result.augmentations["ExpressionSimplifier"] == {"_type": ExpressionSimplifier._type}

    # Reason: clause (symbol_library, rules, max_iterations: "As in simplify"; the entry is added with
    # the augmenter's type, per ResultAugmenter): X_0*X_0 is X_0^2 by default, stays with a library
    # without ^2, without the "powers" group or with no steps.
    @pytest.mark.parametrize(
        "library, kwargs, expected",
        [
            (_sl, {"rules": DefaultRules(exclude=["powers"])}, "X_0*X_0"),
            (_sl, {"max_iterations": 0}, "X_0*X_0"),
            (SymbolLibrary.from_symbol_list(["+", "-", "*", "/"], num_variables=2), {}, "X_0*X_0"),
            (_sl, {}, "X_0^2"),
        ],
        ids=["exclude-powers", "no-steps", "library-without-^2", "default"],
    )
    def test_write_results_uses_rules_and_limits(self, library, kwargs, expected):
        se = _rmse_se()
        se.evaluate_expr(["X_0", "*", "X_0"])
        result = se.get_results(top_k=1)[0]
        ExpressionSimplifier(library, scope="top", **kwargs).write_results(result)
        kind = ExpressionSimplifier._type
        assert result.augmentations["ExpressionSimplifier"] == {"_type": kind, "simplified_best_expr": expected}
        assert result.top_models()[0].augmentations["ExpressionSimplifier"] == {
            "_type": kind,
            "simplified_expr": expected,
        }

    # Reason: clause (fold, fold_aware: "As in simplify"; with folding each entry has "folded": True,
    # and the format functions say so).
    def test_write_results_folds(self):
        se = _rmse_se()
        se.evaluate_expr(["C", "*", "C", "*", "X_0"])
        result = se.get_results(top_k=1)[0]
        ExpressionSimplifier(_sl, scope="top", fold="keep", fold_aware=False).write_results(result)
        data = result.augmentations["ExpressionSimplifier"]
        assert (data["simplified_best_expr"], data["folded"]) == ("C*X_0", True)
        assert result.top_models()[0].augmentations["ExpressionSimplifier"]["folded"] is True
        assert ExpressionSimplifier.format_eval_result(data) == "Simplified (folded): C*X_0"
        assert ExpressionSimplifier.format_model_result({"simplified_expr": "C", "folded": True}) == (
            "Simplified (folded): C"
        )
        ExpressionSimplifier(_sl, scope="best", fold="none", name="Plain").write_results(result)
        assert "folded" not in result.augmentations["Plain"]
        assert ExpressionSimplifier.format_eval_result({"simplified_best_expr": "C*X_0"}) == "Simplified: C*X_0"
        assert ExpressionSimplifier.format_model_result({"simplified_expr": "C*X_0"}) == "Simplified: C*X_0"

    # Reason: clause ("each with its model's specs"): C_1 fixed at 2 makes C_1-2 zero; without specs
    # C_1 is free and nothing combines.
    @pytest.mark.parametrize(
        "specs, expected",
        [([Constant("C_1", value=2.0, fixed=True)], "0"), (None, "C_1-2")],
        ids=["fixed", "no-specs"],
    )
    def test_write_results_uses_model_specs(self, specs, expected):
        se = _rmse_se()
        se.evaluate_expr(Expression(["C_1", "-", "2"], free_constants=specs))
        result = se.get_results(top_k=1)[0]
        ExpressionSimplifier(_sl, scope="top").write_results(result)
        assert result.augmentations["ExpressionSimplifier"]["simplified_best_expr"] == expected
        assert result.top_models()[0].augmentations["ExpressionSimplifier"]["simplified_expr"] == expected


# ── ExpressionSimplifier.format_eval_result ───────────────────────────────────


class TestExpressionSimplifierFormatEvalResult:
    def test_returns_string_when_simplified_present(self):
        out = ExpressionSimplifier.format_eval_result({"simplified_best_expr": "x+1"})
        assert "x+1" in out

    def test_returns_empty_when_key_absent(self):
        assert ExpressionSimplifier.format_eval_result({}) == ""

    def test_returns_empty_when_value_is_empty_string(self):
        assert ExpressionSimplifier.format_eval_result({"simplified_best_expr": ""}) == ""


# ── ExpressionSimplifier.format_model_result ──────────────────────────────────


class TestExpressionSimplifierFormatModelResult:
    def test_returns_string_when_simplified_present(self):
        out = ExpressionSimplifier.format_model_result({"simplified_expr": "x+1"})
        assert "x+1" in out

    def test_returns_empty_when_key_absent(self):
        assert ExpressionSimplifier.format_model_result({}) == ""

    def test_returns_empty_when_value_is_empty_string(self):
        assert ExpressionSimplifier.format_model_result({"simplified_expr": ""}) == ""


# ── ExpressionSimplifier.to_dict / from_dict ──────────────────────────────────


class TestExpressionSimplifierSerialization:
    def test_to_dict_has_expected_keys(self):
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True, name="S")
        d = aug.to_dict("", "")
        assert d["format_version"] == 2
        assert d["type"] == "ExpressionSimplifier"
        assert d["scope"] == "all"
        assert d["verbose"] is True
        assert d["name"] == "S"
        assert d["rules"] == DefaultRules().to_dict()
        assert (d["max_iterations"], d["max_nodes"], d["max_time"]) == (None, None, None)  # None: simplify's defaults
        assert (d["fold"], d["fold_aware"]) == (None, None)

    def test_from_dict_round_trip(self):
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True, name="myS")
        d = aug.to_dict("", "")
        restored = ExpressionSimplifier.from_dict(d)
        assert restored.scope == "all"
        assert restored.verbose is True
        assert restored.name == "myS"

    # Reason: clause (from_dict recreates the augmenter from to_dict): library, rules and limits
    # survive a JSON round trip; the restored augmenter with a library without ^2 and rules without
    # "powers" leaves X_0*X_0 as it is.
    def test_to_dict_from_dict_round_trip(self):
        lib = SymbolLibrary.default_symbols(2)
        lib.remove_symbol("^2")
        aug = ExpressionSimplifier(
            lib, scope="top", rules=DefaultRules(exclude=["powers"]), max_iterations=5, max_nodes=500, max_time=0.5,
            fold="keep", fold_aware=False,
        )  # fmt: skip
        d = json.loads(json.dumps(aug.to_dict("", "")))
        assert (d["max_iterations"], d["max_nodes"], d["max_time"], d["fold"], d["fold_aware"]) == (
            5, 500, 0.5, "keep", False
        )  # fmt: skip
        restored = ExpressionSimplifier.from_dict(d)
        assert restored.to_dict("", "") == d
        assert "^2" not in restored.symbol_library
        se = _rmse_se()
        se.evaluate_expr(["X_0", "*", "X_0"])
        result = se.get_results(top_k=1)[0]
        with patch("SRToolkit.evaluation.result_augmentation.simplify", wraps=simplify) as spy:
            restored.write_results(result)
        assert result.augmentations["ExpressionSimplifier"]["simplified_best_expr"] == "X_0*X_0"
        # the limits reach simplify (max_time cannot be observed from the output)
        assert {
            k: spy.call_args.kwargs[k] for k in ("max_iterations", "max_nodes", "max_time", "fold", "fold_aware")
        } == {
            "max_iterations": 5,
            "max_nodes": 500,
            "max_time": 0.5,
            "fold": "keep",
            "fold_aware": False,
        }

    # Reason: clause (from_dict: version 1, saved before the rules and limits, gets the defaults).
    # The format-1 layout is the to_dict of the previous release.
    def test_from_dict_format_1_gets_defaults(self):
        d = {
            "format_version": 1,
            "type": "ExpressionSimplifier",
            "name": "old",
            "symbol_library": _sl.to_dict(),
            "scope": "best",
            "verbose": False,
        }
        restored = ExpressionSimplifier.from_dict(d)
        assert (restored.name, restored.scope, restored.verbose) == ("old", "best", False)
        assert restored.to_dict("", "")["symbol_library"] == _sl.to_dict()
        assert restored.rules.to_dict() == DefaultRules().to_dict()
        assert (restored.max_iterations, restored.max_nodes, restored.max_time) == (None, None, None)
        assert (restored.fold, restored.fold_aware) == (None, None)

    def test_from_dict_invalid_format_version_raises(self):
        d = ExpressionSimplifier(_sl).to_dict("", "")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            ExpressionSimplifier.from_dict(d)


# ── HoldoutMetric ─────────────────────────────────────────────────────────────

# Held-out data drawn from the same function as the training set (y = 2*X_1 - X_0).
_X_TEST = np.array([[2.0, 3.0], [4.0, 1.0], [0.0, 5.0], [6.0, 2.0]])
_y_TEST = 2.0 * _X_TEST[:, 1] - _X_TEST[:, 0]


class TestHoldoutMetricInit:
    def test_stores_scope_and_name(self):
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST), metric="rmse", scope="best", background_knowledge=BackgroundKnowledge(_sl), name="myH"
        )
        assert aug.scope == "best"
        assert aug.name == "myH"

    def test_defaults(self):
        aug = HoldoutMetric((_X_TEST, _y_TEST), background_knowledge=BackgroundKnowledge(_sl))
        assert aug.scope == "top"
        assert aug.name == "HoldoutMetric"

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="scope"):
            HoldoutMetric((_X_TEST, _y_TEST), scope="invalid", background_knowledge=BackgroundKnowledge(_sl))

    def test_invalid_data_type_raises(self):
        with pytest.raises(TypeError, match="DataSplit"):
            HoldoutMetric("not_data", background_knowledge=BackgroundKnowledge(_sl))

    def test_invalid_metric_type_raises(self):
        with pytest.raises(TypeError, match="metric must be"):
            HoldoutMetric((_X_TEST, _y_TEST), metric=123, background_knowledge=BackgroundKnowledge(_sl))

    def test_accepts_data_split(self):
        from SRToolkit.dataset.data_split import DataSplit

        aug = HoldoutMetric(DataSplit({"X": _X_TEST, "y": _y_TEST}), background_knowledge=BackgroundKnowledge(_sl))
        np.testing.assert_array_equal(aug._X, _X_TEST)
        np.testing.assert_array_equal(aug._y, _y_TEST)

    # Reason: clause (Raises TypeError if "settings is not a Settings or background_knowledge is not a
    # BackgroundKnowledge"): a dict of sections, and a SymbolLibrary where the removed symbol_library
    # argument used to go.
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"settings": {"minimize": {"max_iter": 5}}}, "settings"),
            ({"background_knowledge": _sl}, "background_knowledge"),
        ],
        ids=["settings-dict", "library-as-background-knowledge"],
    )
    def test_settings_and_background_knowledge_types_are_checked(self, kwargs, match):
        with pytest.raises(TypeError, match=match):
            HoldoutMetric((_X_TEST, _y_TEST), **kwargs)

    # Reason: clause (Raises ValueError "if a built-in or callable metric is given held-out data without
    # y").
    @pytest.mark.parametrize("metric", ["rmse", lambda predict, X, y: 0.0], ids=["builtin", "callable"])
    def test_metric_without_held_out_y_raises(self, metric):
        with pytest.raises(ValueError, match="requires held-out targets y"):
            HoldoutMetric((_X_TEST,), metric=metric)

    # Reason: clause ("Expressions are compiled with its symbol_library"). C_1 is the named constant 50
    # only in the background knowledge's library, so y = X_0 + 50 is matched exactly (RMSE 0); compiled
    # with the default library C_1 would be a free constant fitted within (-5, 5) (RMSE 45).
    def test_compiles_with_background_knowledge_library(self):
        lib = SymbolLibrary.default_symbols(2)
        lib.add_symbol("C_1", "lit", 5, "50.0")
        bk = BackgroundKnowledge(lib)
        result = _result_of(ModelResult(["X_0", "+", "C_1"], 0.0))
        aug = HoldoutMetric((_X_TEST, _X_TEST[:, 0] + 50.0), scope="all", background_knowledge=bk)
        aug.write_results(result)
        assert aug.symbol_library is lib
        assert _values(result)[0] == pytest.approx(0.0, abs=1e-9)


class TestHoldoutMetricReuseParams:
    def test_best_scope_writes_best_value_to_eval_result(self):
        se, result = _one_model_result()
        HoldoutMetric(
            (_X_TEST, _y_TEST), metric="rmse", scope="best", background_knowledge=BackgroundKnowledge(_sl)
        ).write_results(result)
        aug = result.augmentations["HoldoutMetric"]
        assert "best_value" in aug
        assert aug["metric"] == "rmse"
        assert aug["best_value"] < 1e-6  # perfect expr generalises

    def test_best_scope_does_not_write_to_models(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="best", background_knowledge=BackgroundKnowledge(_sl)).write_results(
            result
        )
        assert "HoldoutMetric" not in result.top_models()[0].augmentations

    def test_top_scope_writes_value_to_top_models(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="top", background_knowledge=BackgroundKnowledge(_sl)).write_results(
            result
        )
        assert result.top_models()[0].augmentations["HoldoutMetric"]["value"] < 1e-6

    def test_all_scope_writes_to_all_models(self):
        se, result = _two_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl)).write_results(
            result
        )
        assert "HoldoutMetric" in result.all_models[1].augmentations

    def test_reuses_train_params_no_refit(self):
        # The reported value must equal scoring with the EvalResult's *stored* (train-fitted)
        # parameters — i.e. no refit on the held-out data.
        from SRToolkit.evaluation.sr_evaluator import evaluate_metric

        se, result = _one_model_result()
        HoldoutMetric(
            (_X_TEST, _y_TEST), metric="rmse", scope="top", background_knowledge=BackgroundKnowledge(_sl)
        ).write_results(result)
        model = result.top_models()[0]
        expected = evaluate_metric(model.expr, model.parameters, _X_TEST, _y_TEST, "rmse", _sl)
        assert model.augmentations["HoldoutMetric"]["value"] == pytest.approx(expected)

    def test_r2_metric_on_holdout(self):
        se, result = _one_model_result()
        HoldoutMetric(
            (_X_TEST, _y_TEST), metric="r2", scope="best", background_knowledge=BackgroundKnowledge(_sl)
        ).write_results(result)
        assert result.augmentations["HoldoutMetric"]["best_value"] == pytest.approx(1.0, abs=1e-4)

    def test_invalid_expression_scores_nan_without_aborting(self):
        # An unparseable expression (recorded invalid, parameters=None) evaluated *first* must score
        # NaN and not stop the models after it from being augmented.
        se = _rmse_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"])
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl)).write_results(
            result
        )
        values = {"".join(m.expr): m.augmentations["HoldoutMetric"]["value"] for m in result.all_models}
        assert np.isnan(values["C*X_1X_0"])
        assert values["C*X_1-X_0"] < 1e-6

    def test_symbol_library_defaults_when_none(self):
        aug = HoldoutMetric((_X_TEST, _y_TEST))
        assert isinstance(aug.symbol_library, SymbolLibrary)

    def test_model_rejected_by_search_scores_nan(self):
        # Over max_constants in the search: status "rejected", no parameters. It must not be refit
        # and scored on the holdout as if it were a candidate answer.
        se = _rmse_se(settings=Settings(evaluator={"max_constants": 1}))
        se.evaluate_expr(["C", "*", "X_1", "+", "C", "*", "X_0"])
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=2)[0]
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl)).write_results(
            result
        )
        values = {"".join(m.expr): m.augmentations["HoldoutMetric"]["value"] for m in result.all_models}
        assert np.isnan(values["C*X_1+C*X_0"])
        assert values["C*X_1-X_0"] < 1e-6

    def test_constant_free_model_not_refit(self):
        se, result = _one_model_result()
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=2)[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl))
        with patch.object(aug, "_fit_parameters") as fit:
            aug.write_results(result)
        fit.assert_not_called()

    def test_constant_free_model_without_parameters_not_fit(self):
        # BED results carry parameters=None even for constant-free expressions; there is nothing to fit.
        se = _bed_se()
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="best", background_knowledge=BackgroundKnowledge(_sl))
        with patch.object(aug, "_fit_parameters") as fit:
            aug.write_results(result)
        fit.assert_not_called()
        assert np.isfinite(result.augmentations["HoldoutMetric"]["best_value"])

    def test_each_model_scored_once(self):
        # top_models() returns the ModelResult objects in all_models.
        se, result = _two_model_result()
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl))
        with patch.object(aug, "_value", wraps=aug._value) as value:
            aug.write_results(result)
        assert value.call_count == len(result.all_models)

    # Reason: clause (fits use the background knowledge's default_constant). BED results carry no parameters, so X_0 + C is fitted on y = X_0 + 50: under the
    # uniform default C stops at 5 and the residual is 45 everywhere (RMSE 45); under (0, inf) it fits.
    @pytest.mark.parametrize(
        "settings, expected",
        [
            ({}, 45.0),
            ({"default_constant": Constant(prior=UniformSampling(1, 100), bounds=(0, float("inf")))}, 0.0),
        ],
        ids=["uniform-default", "default-constant"],
    )
    def test_holdout_fit_uses_default_constant(self, settings, expected):
        se = _bed_se()
        se.evaluate_expr(["X_0", "+", "C"])
        result = se.get_results(top_k=1)[0]
        bk = BackgroundKnowledge(_sl, **settings)
        HoldoutMetric((_X_TEST, _X_TEST[:, 0] + 50.0), scope="best", background_knowledge=bk).write_results(result)
        assert result.augmentations["HoldoutMetric"]["best_value"] == pytest.approx(expected, abs=1e-5)


# 9 free constants: over the evaluator's default limit of 8.
_NINE_CONSTANTS = ["C", "*", "X_1", "-", "X_0"] + ["+", "C"] * 8


class TestHoldoutMetricLimits:
    def test_ranking_metric_gives_nan_for_rejected_model(self):
        # Regression: the ranking branch scored every model, so a model the search rejected (here
        # for 9 constants, over the evaluator's default limit of 8) got a finite holdout score.
        se = _rmse_se(symbol_library=_sl)
        se.evaluate_expr(_NINE_CONSTANTS)
        result = se.get_results(top_k=1)[0]
        assert result.all_models[0].status == "rejected"
        HoldoutMetric(
            (_X_TEST, _y_TEST), metric=RMSERanking(), scope="all", background_knowledge=BackgroundKnowledge(_sl)
        ).write_results(result)
        assert np.isnan(result.all_models[0].augmentations["HoldoutMetric"]["value"])


class TestHoldoutMetricRefitConstants:
    def test_refit_constants_ignores_train_parameters(self):
        se, result = _one_model_result()
        model = result.top_models()[0]
        model.parameters = np.array([5.0])  # deliberately wrong (the true constant is 2)
        reuse = HoldoutMetric((_X_TEST, _y_TEST), scope="best", background_knowledge=BackgroundKnowledge(_sl))
        refit = HoldoutMetric(
            (_X_TEST, _y_TEST), scope="best", background_knowledge=BackgroundKnowledge(_sl), refit_constants=True
        )
        assert reuse._value(model) > 1.0
        assert refit._value(model) < 1e-6

    def test_refit_constants_keeps_rejected_models_nan(self):
        se = _rmse_se(settings=Settings(evaluator={"max_constants": 1}))
        se.evaluate_expr(["C", "*", "X_1", "+", "C", "*", "X_0"])
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST), scope="best", background_knowledge=BackgroundKnowledge(_sl), refit_constants=True
        )
        aug.write_results(result)
        assert np.isnan(result.augmentations["HoldoutMetric"]["best_value"])

    def test_refit_constants_recorded_in_augmentation(self):
        se, result = _one_model_result()
        HoldoutMetric(
            (_X_TEST, _y_TEST), scope="top", background_knowledge=BackgroundKnowledge(_sl), refit_constants=True
        ).write_results(result)
        assert result.augmentations["HoldoutMetric"]["refit_constants"] is True
        assert result.top_models()[0].augmentations["HoldoutMetric"]["refit_constants"] is True

    def test_refit_seed_is_holdout_seed_not_ranking_seed(self, fit_starts):
        se, result = _one_model_result()
        fits_before = len(fit_starts)
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST),
            scope="best",
            background_knowledge=BackgroundKnowledge(_sl),
            settings=Settings(evaluator={"ranking_seed": 5}),
            seed=7,
            refit_constants=True,
        )
        aug.write_results(result)
        assert len(fit_starts) == fits_before + 1
        ParameterEstimator(_X_TEST, _y_TEST, _sl, seed=7).estimate_parameters(_BEST_EXPR)
        np.testing.assert_array_equal(fit_starts[-2], fit_starts[-1])

    def test_ranking_metric_seed_is_ranking_seed(self, fit_starts):
        se, result = _one_model_result()
        fits_before = len(fit_starts)
        HoldoutMetric(
            (_X_TEST, _y_TEST),
            metric=RMSERanking(),
            scope="best",
            background_knowledge=BackgroundKnowledge(_sl),
            settings=Settings(evaluator={"ranking_seed": 5}),
            seed=7,
        ).write_results(result)
        assert len(fit_starts) == fits_before + 1
        ParameterEstimator(_X_TEST, _y_TEST, _sl, seed=5).estimate_parameters(_BEST_EXPR)
        np.testing.assert_array_equal(fit_starts[-2], fit_starts[-1])

    # Reason: clause (settings are used "for fitting constants on the held-out data (estimation, minimize, ...)").
    # Differential: the refit equals a ParameterEstimator with the same settings, seed and default spec;
    # with max_iter=0 the constant stays at its start, far from the exact fit of the default settings.
    def test_refit_uses_minimize_settings(self):
        settings = Settings(minimize={"max_iter": 0})
        model = ModelResult(_BEST_EXPR, 0.0, parameters=np.array([2.0]))
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST),
            background_knowledge=BackgroundKnowledge(_sl),
            settings=settings,
            refit_constants=True,
        )
        expected, _ = ParameterEstimator(
            _X_TEST, _y_TEST, _sl, 0, settings, uniform_default((-5, 5))
        ).estimate_parameters(_BEST_EXPR)
        assert expected > 1e-3
        assert aug._value(model) == pytest.approx(expected, rel=1e-9)

    def test_refit_constants_with_ranking_function_raises(self):
        with pytest.raises(ValueError, match="refit_constants applies to string/callable metrics only"):
            HoldoutMetric(
                (_X_TEST, _y_TEST),
                metric=RMSERanking(),
                background_knowledge=BackgroundKnowledge(_sl),
                refit_constants=True,
            )


class TestHoldoutMetricRefit:
    def test_refit_ranking_writes_finite_value(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric(
            (_X_BED,),
            metric=BEDRanking(),
            scope="best",
            background_knowledge=BackgroundKnowledge(_sl),
            ground_truth=["C", "+", "X_0"],
            settings=Settings(bed={"num_consts_sampled": 8, "num_points_sampled": _X_BED.shape[0]}),
        )
        aug.write_results(result)
        assert np.isfinite(result.augmentations["HoldoutMetric"]["best_value"])
        assert result.augmentations["HoldoutMetric"]["metric"] == "bed"

    def test_refit_rmse_without_symbol_library_generates_targets(self):
        # RMSERanking.setup builds y from the ground truth through the symbol library, so a None
        # library must be resolved before setup.
        aug = HoldoutMetric((_X_TEST,), metric=RMSERanking(), ground_truth=["X_0", "+", "X_1"])
        assert aug._value(ModelResult(expr=["X_0", "+", "X_1"], error=0.0)) < 1e-6


class TestHoldoutMetricFormat:
    def test_format_eval_result(self):
        out = HoldoutMetric.format_eval_result({"best_value": 0.05, "metric": "rmse"})
        assert "0.05" in out and "rmse" in out

    def test_format_eval_result_empty(self):
        assert HoldoutMetric.format_eval_result({}) == ""

    def test_format_model_result(self):
        out = HoldoutMetric.format_model_result({"value": 0.1, "metric": "rmse"})
        assert "0.1" in out and "rmse" in out

    def test_format_model_result_empty(self):
        assert HoldoutMetric.format_model_result({}) == ""

    def test_format_marks_refit_values(self):
        data = {"value": 0.1, "best_value": 0.1, "metric": "rmse", "refit_constants": True}
        assert "rmse (refit)" in HoldoutMetric.format_model_result(data)
        assert "rmse (refit)" in HoldoutMetric.format_eval_result(data)


class TestHoldoutMetricSerialization:
    def test_to_dict_from_dict_round_trip(self, tmp_path):
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST), metric="r2", scope="all", background_knowledge=BackgroundKnowledge(_sl), name="myH"
        )
        d = aug.to_dict(str(tmp_path), "t")
        assert d["type"] == "HoldoutMetric"
        assert d["metric"] == "r2"
        assert d["scope"] == "all"
        restored = HoldoutMetric.from_dict(d)
        assert restored.scope == "all"
        assert restored.name == "myH"
        assert restored.symbol_library.to_dict() == _sl.to_dict()  # the background knowledge's library
        np.testing.assert_array_equal(restored._X, _X_TEST)
        np.testing.assert_array_equal(restored._y, _y_TEST)

    def test_refit_constants_round_trip(self, tmp_path):
        aug = HoldoutMetric((_X_TEST, _y_TEST), background_knowledge=BackgroundKnowledge(_sl), refit_constants=True)
        assert HoldoutMetric.from_dict(aug.to_dict(str(tmp_path), "t")).refit_constants is True

    def test_seed_and_settings_round_trip(self, tmp_path):
        settings = Settings(minimize={"max_iter": 50})
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST), background_knowledge=BackgroundKnowledge(_sl), settings=settings, seed=7
        )
        restored = HoldoutMetric.from_dict(json.loads(json.dumps(aug.to_dict(str(tmp_path), "t"))))
        assert restored.seed == 7
        assert restored.settings == settings

    # Reason: clause (to_dict keeps only default_constant of the background knowledge) + clause
    # (from_dict recreates the augmenter): the restored augmenter refits under the same default.
    def test_default_constant_round_trip(self, tmp_path):
        default = Constant(prior=UniformSampling(1, 100), bounds=(0, float("inf")))
        bk = BackgroundKnowledge(_sl, samplers=[UniformSampling(0, 1)] * 2, default_constant=default)
        aug = HoldoutMetric((_X_TEST, _X_TEST[:, 0] + 50.0), scope="best", background_knowledge=bk)
        restored = HoldoutMetric.from_dict(json.loads(json.dumps(aug.to_dict(str(tmp_path), "t"))))
        assert restored.background_knowledge.default_constant.to_dict() == default.to_dict()
        se = _bed_se()
        se.evaluate_expr(["X_0", "+", "C"])
        result = se.get_results(top_k=1)[0]
        aug.write_results(result)
        original = result.augmentations["HoldoutMetric"]["best_value"]
        restored.write_results(result)
        assert original < 1e-5
        assert result.augmentations["HoldoutMetric"]["best_value"] == original

    def test_to_dict_stores_absolute_data_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = HoldoutMetric((_X_TEST, _y_TEST), background_knowledge=BackgroundKnowledge(_sl)).to_dict(
            "relative_dir", "t"
        )
        assert os.path.isabs(d["data"])

    def test_loaded_results_scored_once_per_expression(self, tmp_path):
        # After SR_results.load, top_models() still returns the objects in all_models.
        se = _rmse_se()
        se.evaluate_expr(_BEST_EXPR)
        se.evaluate_expr(_OTHER_EXPR)
        se.get_results(top_k=1).save(str(tmp_path / "r.json"))
        result = SR_results.load(str(tmp_path / "r.json"))[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl))
        with patch.object(aug, "_value", wraps=aug._value) as value:
            aug.write_results(result)
        assert value.call_count == 2

    def test_callable_metric_not_serialisable(self, tmp_path):
        aug = HoldoutMetric(
            (_X_TEST, _y_TEST), metric=lambda predict, X, y: 0.0, background_knowledge=BackgroundKnowledge(_sl)
        )
        with pytest.raises(NotImplementedError):
            aug.to_dict(str(tmp_path), "t")

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        d = HoldoutMetric((_X_TEST, _y_TEST), metric="rmse", background_knowledge=BackgroundKnowledge(_sl)).to_dict(
            str(tmp_path), "t"
        )
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            HoldoutMetric.from_dict(d)


# ── HoldoutMetric with free-constant specs ────────────────────────────────────

_X3 = np.array([[1.0], [2.0], [3.0]])
_E = ["C_0", "*", "X_0"]
_sl1 = SymbolLibrary.default_symbols(1)


def _fix(label, value):
    return Constant(label=label, value=value, fixed=True)


def _result_of(*models):
    return EvalResult(all_models=list(models), trace=list(range(len(models))), approach_name="")


def _values(result, name="HoldoutMetric"):
    return [m.augmentations[name]["value"] for m in result.all_models]


class _RecordingRanking(RankingFunction):
    name = "recording"

    def setup(self, X, y, ground_truth, ground_truth_constants, background_knowledge, seed, settings):
        self.received = []

    def score(self, expr):
        self.received.append(expr.free_constants)
        return RankingResult(score=7.0)


class _TaskRecordingRanking(RankingFunction):
    name = "task_recording"

    def setup(self, X, y, ground_truth, ground_truth_constants, background_knowledge, seed, settings):
        self.args = (ground_truth, ground_truth_constants, background_knowledge, seed, settings)

    def score(self, expr):
        return RankingResult(score=0.0)


class TestHoldoutMetricRankingSetup:
    # Reason: clause (a RankingFunction metric's setup gets ground_truth, ground_truth_constants, all of
    # the background knowledge and the settings; the seed when no ranking_seed is set).
    def test_ranking_metric_setup_gets_task_facts_and_settings(self):
        gt, specs = ["X_0", "+", "C_0"], [_fix("C_0", 1.0)]
        bk = BackgroundKnowledge(_sl, samplers=[UniformSampling(0, 1)] * 2)
        settings = Settings(bed={"num_points_sampled": 3})
        aug = HoldoutMetric(
            (_X_TEST,),
            metric=_TaskRecordingRanking(),
            ground_truth=gt,
            ground_truth_constants=specs,
            background_knowledge=bk,
            settings=settings,
            seed=4,
        )
        received = aug.metric.args
        assert received[0] is gt and received[1] is specs and received[2] is bk and received[4] is settings
        assert received[3] == 4


class TestHoldoutMetricSpecs:
    # Reason: clause ("Fits use each model's specs") + implied (the memo tells the same tokens with
    # different specs apart). y = 2x: C_0 fixed to 1 gives mean |x| = 2, fixed to 4 gives mean |2x| = 4.
    @pytest.mark.parametrize(
        "refit, parameters",
        [(True, np.array([2.0])), (False, None)],
        ids=["refit", "fit-missing-params"],
    )
    def test_fits_use_each_models_specs(self, refit, parameters):
        result = _result_of(
            ModelResult(_E, 0.0, parameters=parameters, free_constants=(_fix("C_0", 1.0),)),
            ModelResult(_E, 0.0, parameters=parameters, free_constants=(_fix("C_0", 4.0),)),
        )
        aug = HoldoutMetric(
            (_X3, 2 * _X3[:, 0]),
            metric="mae",
            scope="all",
            background_knowledge=BackgroundKnowledge(_sl1),
            refit_constants=refit,
        )
        aug.write_results(result)
        np.testing.assert_allclose(_values(result), [2.0, 4.0])

    # Reason: clause (a RankingFunction metric scores each model with its specs). Fixed C_0 = 3 against
    # the ground truth x: every column is x + 3, so BED = 3.
    def test_ranking_metric_receives_specs(self):
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        result = _result_of(ModelResult(["X_0", "+", "C_0"], 0.0, free_constants=(_fix("C_0", 3.0),)))
        aug = HoldoutMetric(
            (X,),
            metric=BEDRanking(),
            scope="all",
            background_knowledge=BackgroundKnowledge(_sl1),
            ground_truth=["X_0"],
            settings=Settings(bed={"num_points_sampled": -1}),
        )
        aug.write_results(result)
        assert _values(result)[0] == pytest.approx(3.0, abs=1e-9)

    # Reason: clause (a ranking metric scores each model with its specs) + implied
    # (free_constants=() counts as no specs: it shares the memo entry of the spec-free model, so the
    # ranking is called twice, with None and with the spec).
    def test_ranking_metric_gets_each_models_specs(self):
        spec = _fix("C_0", 1.0)
        result = _result_of(
            ModelResult(_E, 0.0),
            ModelResult(_E, 0.0, free_constants=(spec,)),
            ModelResult(_E, 0.0, free_constants=()),
        )
        aug = HoldoutMetric(
            (_X3, 2 * _X3[:, 0]),
            metric=_RecordingRanking(),
            scope="all",
            background_knowledge=BackgroundKnowledge(_sl1),
        )
        aug.write_results(result)
        assert _values(result) == [7.0, 7.0, 7.0]
        received = aug.metric.received
        assert len(received) == 2 and received[0] == ()
        assert [c.to_dict() for c in received[1]] == [spec.to_dict()]

    # Reason: regression (#cache-key: the write_results memo was keyed on "".join(tokens)). ``u`` is the
    # named constant 10, so "u - X_0" and "u- X_0" (unary minus) both join to "u-X_0":
    # mean |10 - x| = 8.5 and mean |-x| = 1.5 on x = 1, 2.
    def test_memo_distinguishes_token_lists_with_equal_join(self):
        lib = SymbolLibrary.from_symbol_list(["-", "u-"], 1)
        lib.add_symbol("u", "lit", 5, "10.0")
        result = _result_of(ModelResult(["u", "-", "X_0"], 0.0), ModelResult(["u-", "X_0"], 0.0))
        aug = HoldoutMetric(
            (np.array([[1.0], [2.0]]), np.zeros(2)),
            metric="mae",
            scope="all",
            background_knowledge=BackgroundKnowledge(lib),
        )
        aug.write_results(result)
        np.testing.assert_allclose(_values(result), [8.5, 1.5])


# ── Compact results ───────────────────────────────────────────────────────────


def _compact_result():
    """The two-model result as a compact one: its top model and a summary of the run."""
    _, full = _two_model_result()
    top = full.top_models(1)
    summary = RunSummary(
        num_evaluated=2,
        evaluation_calls=2,
        incumbents=[(1, 0, top[0])],
        status_counts=full.status_counts(),
        call_status_counts=full.status_counts(per="call"),
        pareto_front=top,
    )
    return dataclasses.replace(full, all_models=top, trace=[], summary=summary)


# Reason: clause (write_results raises with scope="all" on a result with a summary; nothing is added).
@pytest.mark.parametrize(
    "make",
    [
        lambda: ExpressionToLatex(_sl, scope="all"),
        lambda: ExpressionSimplifier(_sl, scope="all"),
        lambda: HoldoutMetric((_X_TEST, _y_TEST), scope="all", background_knowledge=BackgroundKnowledge(_sl)),
    ],
    ids=["latex", "simplifier", "holdout"],
)
def test_scope_all_on_compact_result_raises(make):
    result = _compact_result()
    with pytest.raises(ValueError, match="all"):
        make().write_results(result)
    assert result.augmentations == {}
    assert result.all_models[0].augmentations == {}
