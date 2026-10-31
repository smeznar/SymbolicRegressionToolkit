import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from SRToolkit.evaluation.ranking import BEDRanking, RMSERanking
from SRToolkit.evaluation.result_augmentation import (
    ExpressionSimplifier,
    ExpressionToLatex,
    HoldoutMetric,
)
from SRToolkit.evaluation.sr_evaluator import ModelResult, SR_evaluator, SR_results
from SRToolkit.utils import Node
from SRToolkit.utils.symbol_library import SymbolLibrary

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

    def test_invalid_scope_raises(self):
        with pytest.raises(Exception, match="scope"):
            ExpressionSimplifier(_sl, scope="invalid")


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

    def test_node_return_from_simplify_uses_to_list(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="best")
        mock_node = MagicMock(spec=Node)
        mock_node.to_list.return_value = ["X", "_", "0"]
        with patch("SRToolkit.evaluation.result_augmentation.simplify", return_value=mock_node):
            aug.write_results(result)
        mock_node.to_list.assert_called_once_with(_sl)
        simplified = result.augmentations["ExpressionSimplifier"]["simplified_best_expr"]
        assert simplified == "X_0"

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
            side_effect=[["X", "_", "0"], ValueError("model fail")],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_model_simplification_failure_silent(self):
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top", verbose=False)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X", "_", "0"], ValueError("model fail")],
        ):
            with patch("SRToolkit.evaluation.result_augmentation.warnings.warn") as mock_warn:
                aug.write_results(result)
        mock_warn.assert_not_called()

    def test_best_expr_invalid_return_warns_verbose(self):
        # Covers line 229: else raise when simplify returns neither list nor Node
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="best", verbose=True)
        with patch("SRToolkit.evaluation.result_augmentation.simplify", return_value=42):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_top_model_node_return_uses_to_list(self):
        # Covers lines 242-243: Node branch in the top_models loop
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top")
        mock_node = MagicMock(spec=Node)
        mock_node.to_list.return_value = ["X", "_", "1"]
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], mock_node],
        ):
            aug.write_results(result)
        mock_node.to_list.assert_called_once_with(_sl)
        assert result.top_models()[0].augmentations["ExpressionSimplifier"]["simplified_expr"] == "X_1"

    def test_top_model_invalid_return_warns_verbose(self):
        # Covers lines 244-245: else raise in top_models loop
        _, result = _one_model_result()
        aug = ExpressionSimplifier(_sl, scope="top", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], 42],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)

    def test_all_model_node_return_uses_to_list(self):
        # Node branch in the all_models loop
        _, result = _two_model_result()
        aug = ExpressionSimplifier(_sl, scope="all")
        mock_node = MagicMock(spec=Node)
        mock_node.to_list.return_value = ["X", "_", "1"]
        # calls: best_expr, all_models[0], all_models[1] (Node)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], ["X_0"], mock_node],
        ):
            aug.write_results(result)
        assert result.all_models[1].augmentations["ExpressionSimplifier"]["simplified_expr"] == "X_1"

    def test_all_model_invalid_return_warns_verbose(self):
        # Else raise and verbose warn in the all_models loop
        _, result = _two_model_result()
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], ["X_0"], 42],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)


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
        assert d["format_version"] == 1
        assert d["type"] == "ExpressionSimplifier"
        assert d["scope"] == "all"
        assert d["verbose"] is True
        assert d["name"] == "S"

    def test_from_dict_round_trip(self):
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True, name="myS")
        d = aug.to_dict("", "")
        restored = ExpressionSimplifier.from_dict(d)
        assert restored.scope == "all"
        assert restored.verbose is True
        assert restored.name == "myS"

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
        aug = HoldoutMetric((_X_TEST, _y_TEST), metric="rmse", scope="best", symbol_library=_sl, name="myH")
        assert aug.scope == "best"
        assert aug.name == "myH"

    def test_defaults(self):
        aug = HoldoutMetric((_X_TEST, _y_TEST), symbol_library=_sl)
        assert aug.scope == "top"
        assert aug.name == "HoldoutMetric"

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="scope"):
            HoldoutMetric((_X_TEST, _y_TEST), scope="invalid", symbol_library=_sl)

    def test_invalid_data_type_raises(self):
        with pytest.raises(TypeError, match="DataSplit"):
            HoldoutMetric("not_data", symbol_library=_sl)

    def test_invalid_metric_type_raises(self):
        with pytest.raises(TypeError, match="metric must be"):
            HoldoutMetric((_X_TEST, _y_TEST), metric=123, symbol_library=_sl)

    def test_accepts_data_split(self):
        from SRToolkit.dataset.data_split import DataSplit

        aug = HoldoutMetric(DataSplit({"X": _X_TEST, "y": _y_TEST}), symbol_library=_sl)
        np.testing.assert_array_equal(aug._X, _X_TEST)
        np.testing.assert_array_equal(aug._y, _y_TEST)


class TestHoldoutMetricReuseParams:
    def test_best_scope_writes_best_value_to_eval_result(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), metric="rmse", scope="best", symbol_library=_sl).write_results(result)
        aug = result.augmentations["HoldoutMetric"]
        assert "best_value" in aug
        assert aug["metric"] == "rmse"
        assert aug["best_value"] < 1e-6  # perfect expr generalises

    def test_best_scope_does_not_write_to_models(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="best", symbol_library=_sl).write_results(result)
        assert "HoldoutMetric" not in result.top_models()[0].augmentations

    def test_top_scope_writes_value_to_top_models(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="top", symbol_library=_sl).write_results(result)
        assert result.top_models()[0].augmentations["HoldoutMetric"]["value"] < 1e-6

    def test_all_scope_writes_to_all_models(self):
        se, result = _two_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl).write_results(result)
        assert "HoldoutMetric" in result.all_models[1].augmentations

    def test_reuses_train_params_no_refit(self):
        # The reported value must equal scoring with the EvalResult's *stored* (train-fitted)
        # parameters — i.e. no refit on the held-out data.
        from SRToolkit.evaluation.sr_evaluator import evaluate_metric

        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), metric="rmse", scope="top", symbol_library=_sl).write_results(result)
        model = result.top_models()[0]
        expected = evaluate_metric(model.expr, model.parameters, _X_TEST, _y_TEST, "rmse", _sl)
        assert model.augmentations["HoldoutMetric"]["value"] == pytest.approx(expected)

    def test_r2_metric_on_holdout(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), metric="r2", scope="best", symbol_library=_sl).write_results(result)
        assert result.augmentations["HoldoutMetric"]["best_value"] == pytest.approx(1.0, abs=1e-4)

    def test_invalid_expression_scores_nan_without_aborting(self):
        # An unparseable expression (recorded invalid, parameters=None) evaluated *first* must score
        # NaN and not stop the models after it from being augmented.
        se = _rmse_se()
        se.evaluate_expr(["C", "*", "X_1", "X_0"])
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl).write_results(result)
        values = {"".join(m.expr): m.augmentations["HoldoutMetric"]["value"] for m in result.all_models}
        assert np.isnan(values["C*X_1X_0"])
        assert values["C*X_1-X_0"] < 1e-6

    def test_symbol_library_defaults_when_none(self):
        aug = HoldoutMetric((_X_TEST, _y_TEST))
        assert isinstance(aug.symbol_library, SymbolLibrary)

    def test_model_rejected_by_search_scores_nan(self):
        # Over max_constants in the search: status "rejected", no parameters. It must not be refit
        # and scored on the holdout as if it were a candidate answer.
        se = _rmse_se(max_constants=1)
        se.evaluate_expr(["C", "*", "X_1", "+", "C", "*", "X_0"])
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=2)[0]
        HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl).write_results(result)
        values = {"".join(m.expr): m.augmentations["HoldoutMetric"]["value"] for m in result.all_models}
        assert np.isnan(values["C*X_1+C*X_0"])
        assert values["C*X_1-X_0"] < 1e-6

    def test_constant_free_model_not_refit(self):
        se, result = _one_model_result()
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=2)[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl)
        with patch.object(aug, "_fit_parameters") as fit:
            aug.write_results(result)
        fit.assert_not_called()

    def test_constant_free_model_without_parameters_not_fit(self):
        # BED results carry parameters=None even for constant-free expressions; there is nothing to fit.
        se = _bed_se()
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="best", symbol_library=_sl)
        with patch.object(aug, "_fit_parameters") as fit:
            aug.write_results(result)
        fit.assert_not_called()
        assert np.isfinite(result.augmentations["HoldoutMetric"]["best_value"])

    def test_each_model_scored_once(self):
        # top_models() returns the ModelResult objects in all_models.
        se, result = _two_model_result()
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl)
        with patch.object(aug, "_value", wraps=aug._value) as value:
            aug.write_results(result)
        assert value.call_count == len(result.all_models)

    def test_holdout_fit_uses_ranking_settings(self):
        # BED results carry no parameters, so constants are fit on the holdout, under the given
        # settings: with max_constants=0 an expression with a constant is rejected (NaN).
        se = _bed_se()
        se.evaluate_expr(["C", "+", "X_0"])
        result = se.get_results(top_k=1)[0]
        HoldoutMetric(
            (_X_TEST, _y_TEST), scope="best", symbol_library=_sl, ranking_settings={"max_constants": 0}
        ).write_results(result)
        assert np.isnan(result.augmentations["HoldoutMetric"]["best_value"])


class TestHoldoutMetricRefitConstants:
    def test_refit_constants_ignores_train_parameters(self):
        se, result = _one_model_result()
        model = result.top_models()[0]
        model.parameters = np.array([5.0])  # deliberately wrong (the true constant is 2)
        reuse = HoldoutMetric((_X_TEST, _y_TEST), scope="best", symbol_library=_sl)
        refit = HoldoutMetric((_X_TEST, _y_TEST), scope="best", symbol_library=_sl, refit_constants=True)
        assert reuse._value(model) > 1.0
        assert refit._value(model) < 1e-6

    def test_refit_constants_keeps_rejected_models_nan(self):
        se = _rmse_se(max_constants=1)
        se.evaluate_expr(["C", "*", "X_1", "+", "C", "*", "X_0"])
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="best", symbol_library=_sl, refit_constants=True)
        aug.write_results(result)
        assert np.isnan(result.augmentations["HoldoutMetric"]["best_value"])

    def test_refit_constants_recorded_in_augmentation(self):
        se, result = _one_model_result()
        HoldoutMetric((_X_TEST, _y_TEST), scope="top", symbol_library=_sl, refit_constants=True).write_results(result)
        assert result.augmentations["HoldoutMetric"]["refit_constants"] is True
        assert result.top_models()[0].augmentations["HoldoutMetric"]["refit_constants"] is True

    def test_refit_constants_with_ranking_function_raises(self):
        with pytest.raises(ValueError, match="refit_constants applies to string/callable metrics only"):
            HoldoutMetric((_X_TEST, _y_TEST), metric=RMSERanking(), symbol_library=_sl, refit_constants=True)


class TestHoldoutMetricRefit:
    def test_refit_ranking_writes_finite_value(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        aug = HoldoutMetric(
            (_X_BED,),
            metric=BEDRanking(),
            scope="best",
            symbol_library=_sl,
            ground_truth=["C", "+", "X_0"],
            ranking_settings={"num_consts_sampled": 8, "num_points_sampled": _X_BED.shape[0]},
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
        aug = HoldoutMetric((_X_TEST, _y_TEST), metric="r2", scope="all", symbol_library=_sl, name="myH")
        d = aug.to_dict(str(tmp_path), "t")
        assert d["type"] == "HoldoutMetric"
        assert d["metric"] == "r2"
        assert d["scope"] == "all"
        restored = HoldoutMetric.from_dict(d)
        assert restored.scope == "all"
        assert restored.name == "myH"
        np.testing.assert_array_equal(restored._X, _X_TEST)
        np.testing.assert_array_equal(restored._y, _y_TEST)

    def test_refit_constants_round_trip(self, tmp_path):
        aug = HoldoutMetric((_X_TEST, _y_TEST), symbol_library=_sl, refit_constants=True)
        assert HoldoutMetric.from_dict(aug.to_dict(str(tmp_path), "t")).refit_constants is True

    def test_seed_and_ranking_settings_round_trip(self, tmp_path):
        settings = {"constant_bounds": (-2, 2), "max_constants": 3}
        aug = HoldoutMetric((_X_TEST, _y_TEST), symbol_library=_sl, ranking_settings=settings, seed=7)
        restored = HoldoutMetric.from_dict(json.loads(json.dumps(aug.to_dict(str(tmp_path), "t"))))
        assert restored.seed == 7
        assert restored.ranking_settings["max_constants"] == 3
        assert list(restored.ranking_settings["constant_bounds"]) == [-2, 2]
        assert restored._fit_parameters(_BEST_EXPR)[0] < 1e-6  # the list bounds still fit

    def test_to_dict_is_json_safe_with_ranking_only_settings(self, tmp_path):
        from SRToolkit.dataset.sampling import UniformSampling

        settings = {"samplers": [UniformSampling(0, 1)], "max_constants": 3}
        aug = HoldoutMetric((_X_TEST, _y_TEST), symbol_library=_sl, ranking_settings=settings)
        d = json.loads(json.dumps(aug.to_dict(str(tmp_path), "t")))
        assert d["ranking_settings"] == {"max_constants": 3}

    def test_to_dict_stores_absolute_data_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = HoldoutMetric((_X_TEST, _y_TEST), symbol_library=_sl).to_dict("relative_dir", "t")
        assert os.path.isabs(d["data"])

    def test_loaded_results_scored_once_per_expression(self, tmp_path):
        # After SR_results.load, top_models() still returns the objects in all_models.
        se = _rmse_se()
        se.evaluate_expr(_BEST_EXPR)
        se.evaluate_expr(_OTHER_EXPR)
        se.get_results(top_k=1).save(str(tmp_path / "r.json"))
        result = SR_results.load(str(tmp_path / "r.json"))[0]
        aug = HoldoutMetric((_X_TEST, _y_TEST), scope="all", symbol_library=_sl)
        with patch.object(aug, "_value", wraps=aug._value) as value:
            aug.write_results(result)
        assert value.call_count == 2

    def test_callable_metric_not_serialisable(self, tmp_path):
        aug = HoldoutMetric((_X_TEST, _y_TEST), metric=lambda predict, X, y: 0.0, symbol_library=_sl)
        with pytest.raises(NotImplementedError):
            aug.to_dict(str(tmp_path), "t")

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        d = HoldoutMetric((_X_TEST, _y_TEST), metric="rmse", symbol_library=_sl).to_dict(str(tmp_path), "t")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            HoldoutMetric.from_dict(d)
