from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from SRToolkit.evaluation.result_augmentation import (
    BED,
    R2,
    RMSE,
    ExpressionSimplifier,
    ExpressionToLatex,
)
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
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
        assert "ExpressionToLatex" not in result.top_models[0].augmentations

    def test_top_scope_writes_to_top_models(self):
        _, result = _two_model_result()
        ExpressionToLatex(_sl, scope="top").write_results(result)
        assert "ExpressionToLatex" in result.top_models[0].augmentations
        assert "expr_latex" in result.top_models[0].augmentations["ExpressionToLatex"]

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
        # Covers lines 100-102: exception in the all_models loop with verbose=True
        _, result = _two_model_result()
        aug = ExpressionToLatex(_sl, scope="all", verbose=True)
        mock_tree = MagicMock()
        mock_tree.to_latex.return_value = "$X_0$"
        # calls: best_expr, top_models[0], all_models[0], all_models[1] (fails)
        with patch(
            "SRToolkit.evaluation.result_augmentation.tokens_to_tree",
            side_effect=[mock_tree, mock_tree, mock_tree, ValueError("all fail")],
        ):
            with pytest.warns(UserWarning):
                aug.write_results(result)


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
        assert "ExpressionSimplifier" not in result.top_models[0].augmentations

    def test_top_scope_writes_simplified_expr_to_top_models(self):
        _, result = _one_model_result()
        ExpressionSimplifier(_sl, scope="top").write_results(result)
        assert "ExpressionSimplifier" in result.top_models[0].augmentations

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
        assert result.top_models[0].augmentations["ExpressionSimplifier"]["simplified_expr"] == "X_1"

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
        # Covers lines 258-259: Node branch in the all_models loop
        _, result = _two_model_result()
        aug = ExpressionSimplifier(_sl, scope="all")
        mock_node = MagicMock(spec=Node)
        mock_node.to_list.return_value = ["X", "_", "1"]
        # calls: best_expr, top_models[0], all_models[0], all_models[1] (Node)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], ["X_0"], ["X_0"], mock_node],
        ):
            aug.write_results(result)
        assert result.all_models[1].augmentations["ExpressionSimplifier"]["simplified_expr"] == "X_1"

    def test_all_model_invalid_return_warns_verbose(self):
        # Covers lines 260-264: else raise and verbose warn in the all_models loop
        _, result = _two_model_result()
        aug = ExpressionSimplifier(_sl, scope="all", verbose=True)
        with patch(
            "SRToolkit.evaluation.result_augmentation.simplify",
            side_effect=[["X_0"], ["X_0"], ["X_0"], 42],
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


# ── RMSE.__init__ ─────────────────────────────────────────────────────────────


class TestRMSEInit:
    def test_stores_evaluator_scope_name(self):
        se = _rmse_se()
        aug = RMSE(se, scope="best", name="myRMSE")
        assert aug.evaluator is se
        assert aug.scope == "best"
        assert aug.name == "myRMSE"

    def test_defaults(self):
        se = _rmse_se()
        aug = RMSE(se)
        assert aug.scope == "top"
        assert aug.name == "RMSE"

    def test_invalid_scope_raises(self):
        with pytest.raises(Exception, match="scope"):
            RMSE(_rmse_se(), scope="invalid")

    def test_wrong_ranking_function_raises(self):
        se = _bed_se()
        with pytest.raises(Exception, match="rmse"):
            RMSE(se)

    def test_none_y_raises(self):
        mock_se = MagicMock()
        mock_se.ranking_function = "rmse"
        mock_se.y = None
        with pytest.raises(Exception, match="y"):
            RMSE(mock_se)


# ── RMSE.write_results ────────────────────────────────────────────────────────


class TestRMSEWriteResults:
    def test_writes_min_error_to_eval_result(self):
        se, result = _one_model_result()
        RMSE(se, scope="best").write_results(result)
        assert "min_error" in result.augmentations["RMSE"]
        assert result.augmentations["RMSE"]["min_error"] < 1e-6

    def test_best_scope_does_not_write_to_models(self):
        se, result = _one_model_result()
        RMSE(se, scope="best").write_results(result)
        assert "RMSE" not in result.top_models[0].augmentations

    def test_top_scope_writes_error_and_params_to_top_models(self):
        se, result = _one_model_result()
        RMSE(se, scope="top").write_results(result)
        model_aug = result.top_models[0].augmentations["RMSE"]
        assert "error" in model_aug
        assert "parameters" in model_aug

    def test_all_scope_writes_to_all_models(self):
        se, result = _two_model_result()
        RMSE(se, scope="all").write_results(result)
        assert "RMSE" in result.all_models[1].augmentations
        assert "error" in result.all_models[1].augmentations["RMSE"]


# ── RMSE.format_eval_result ───────────────────────────────────────────────────


class TestRMSEFormatEvalResult:
    def test_returns_string_when_min_error_present(self):
        out = RMSE.format_eval_result({"min_error": 0.05})
        assert "0.05" in out
        assert out != ""

    def test_returns_empty_when_min_error_absent(self):
        assert RMSE.format_eval_result({}) == ""


# ── RMSE.format_model_result ──────────────────────────────────────────────────


class TestRMSEFormatModelResult:
    def test_returns_rmse_with_params(self):
        out = RMSE.format_model_result({"error": 0.5, "parameters": np.array([2.0])})
        assert "RMSE=" in out
        assert "params=" in out

    def test_returns_rmse_without_params_when_none(self):
        out = RMSE.format_model_result({"error": 0.5, "parameters": None})
        assert "RMSE=" in out
        assert "params=" not in out

    def test_returns_rmse_without_params_when_key_absent(self):
        out = RMSE.format_model_result({"error": 0.5})
        assert "RMSE=" in out
        assert "params=" not in out


# ── RMSE.to_dict / from_dict ──────────────────────────────────────────────────


class TestRMSESerialization:
    def test_to_dict_has_expected_keys(self, tmp_path):
        se = _rmse_se()
        aug = RMSE(se, scope="best", name="R")
        d = aug.to_dict(str(tmp_path), "test")
        assert d["format_version"] == 1
        assert d["type"] == "RMSE"
        assert d["scope"] == "best"
        assert d["name"] == "R"
        assert "evaluator" in d

    def test_from_dict_round_trip(self, tmp_path):
        se = _rmse_se()
        aug = RMSE(se, scope="best", name="myR")
        d = aug.to_dict(str(tmp_path), "test")
        restored = RMSE.from_dict(d)
        assert restored.scope == "best"
        assert restored.name == "myR"
        np.testing.assert_array_equal(restored.evaluator.X, _X)

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        d = RMSE(_rmse_se()).to_dict(str(tmp_path), "test")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            RMSE.from_dict(d)


# ── BED.__init__ ──────────────────────────────────────────────────────────────


class TestBEDInit:
    def test_stores_evaluator_scope_name(self):
        se = _bed_se()
        aug = BED(se, scope="best", name="myBED")
        assert aug.evaluator is se
        assert aug.scope == "best"
        assert aug.name == "myBED"

    def test_defaults(self):
        aug = BED(_bed_se())
        assert aug.scope == "top"
        assert aug.name == "BED"

    def test_invalid_scope_raises(self):
        with pytest.raises(Exception, match="scope"):
            BED(_bed_se(), scope="invalid")

    def test_wrong_ranking_function_raises(self):
        with pytest.raises(Exception, match="bed"):
            BED(_rmse_se())


# ── BED.write_results ─────────────────────────────────────────────────────────


class TestBEDWriteResults:
    def test_writes_best_expr_bed_to_eval_result(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        BED(se, scope="best").write_results(result)
        assert "best_expr_bed" in result.augmentations["BED"]

    def test_best_scope_does_not_write_to_models(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        BED(se, scope="best").write_results(result)
        assert "BED" not in result.top_models[0].augmentations

    def test_top_scope_writes_bed_to_top_models(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        result = se.get_results(top_k=1)[0]
        BED(se, scope="top").write_results(result)
        assert "BED" in result.top_models[0].augmentations
        assert "bed" in result.top_models[0].augmentations["BED"]

    def test_all_scope_writes_to_all_models(self):
        se = _bed_se()
        se.evaluate_expr(_BEST_EXPR)
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=1)[0]
        BED(se, scope="all").write_results(result)
        assert "BED" in result.all_models[1].augmentations


# ── BED.format_eval_result ────────────────────────────────────────────────────


class TestBEDFormatEvalResult:
    def test_returns_string_when_bed_present(self):
        out = BED.format_eval_result({"best_expr_bed": 0.1})
        assert "0.1" in out
        assert out != ""

    def test_returns_empty_when_key_absent(self):
        assert BED.format_eval_result({}) == ""


# ── BED.format_model_result ───────────────────────────────────────────────────


class TestBEDFormatModelResult:
    def test_returns_string_when_bed_present(self):
        out = BED.format_model_result({"bed": 0.2})
        assert "0.2" in out
        assert out != ""

    def test_returns_empty_when_key_absent(self):
        assert BED.format_model_result({}) == ""

    def test_returns_empty_when_value_is_empty_string(self):
        assert BED.format_model_result({"bed": ""}) == ""


# ── BED.to_dict / from_dict ───────────────────────────────────────────────────


class TestBEDSerialization:
    def test_to_dict_has_expected_keys(self, tmp_path):
        se = _bed_se()
        aug = BED(se, scope="all", name="B")
        d = aug.to_dict(str(tmp_path), "test")
        assert d["format_version"] == 1
        assert d["type"] == "BED"
        assert d["scope"] == "all"
        assert d["name"] == "B"

    def test_from_dict_round_trip(self, tmp_path):
        se = _bed_se()
        aug = BED(se, scope="all", name="myB")
        d = aug.to_dict(str(tmp_path), "test")
        restored = BED.from_dict(d)
        assert restored.scope == "all"
        assert restored.name == "myB"

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        d = BED(_bed_se()).to_dict(str(tmp_path), "test")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            BED.from_dict(d)


# ── R2.__init__ ───────────────────────────────────────────────────────────────


class TestR2Init:
    def test_stores_evaluator_scope_name(self):
        se = _rmse_se()
        aug = R2(se, scope="best", name="myR2")
        assert aug.evaluator is se
        assert aug.scope == "best"
        assert aug.name == "myR2"

    def test_defaults(self):
        aug = R2(_rmse_se())
        assert aug.scope == "top"
        assert aug.name == "R2"

    def test_invalid_scope_raises(self):
        with pytest.raises(Exception, match="scope"):
            R2(_rmse_se(), scope="invalid")

    def test_wrong_ranking_function_raises(self):
        with pytest.raises(Exception, match="rmse"):
            R2(_bed_se())

    def test_none_y_raises(self):
        mock_se = MagicMock()
        mock_se.ranking_function = "rmse"
        mock_se.y = None
        with pytest.raises(Exception, match="y"):
            R2(mock_se)

    def test_ss_tot_computed(self):
        se = _rmse_se()
        aug = R2(se)
        expected = np.sum((_y - np.mean(_y)) ** 2)
        assert abs(aug.ss_tot - expected) < 1e-12


# ── R2._compute_r2 ────────────────────────────────────────────────────────────


class TestR2ComputeR2:
    def test_perfect_fit_returns_one(self):
        se, result = _one_model_result()
        aug = R2(se)
        r2_val = aug._compute_r2(result.top_models[0])
        assert r2_val == pytest.approx(1.0, abs=1e-4)

    def test_clamped_to_zero_when_ss_res_exceeds_ss_tot(self):
        # Near-constant y → small ss_tot; X_0 far from y → ss_res >> ss_tot
        X_bad = np.array([[100.0], [200.0], [300.0], [400.0]])
        y_bad = np.array([1.0, 1.1, 1.2, 1.3])
        se = SR_evaluator(X_bad, y_bad, seed=42)
        se.evaluate_expr(["X_0"])
        result = se.get_results(top_k=1)[0]
        aug = R2(se)
        r2_val = aug._compute_r2(result.top_models[0])
        assert r2_val == 0.0


# ── R2.write_results ──────────────────────────────────────────────────────────


class TestR2WriteResults:
    def test_writes_best_expr_r2_to_eval_result(self):
        se, result = _one_model_result()
        R2(se, scope="best").write_results(result)
        assert "best_expr_r^2" in result.augmentations["R2"]

    def test_perfect_fit_r2_near_one(self):
        se, result = _one_model_result()
        R2(se).write_results(result)
        assert result.augmentations["R2"]["best_expr_r^2"] == pytest.approx(1.0, abs=1e-4)

    def test_bad_expr_r2_less_than_one(self):
        se = _rmse_se()
        se.evaluate_expr(_OTHER_EXPR)
        result = se.get_results(top_k=1)[0]
        R2(se).write_results(result)
        assert result.augmentations["R2"]["best_expr_r^2"] < 1.0

    def test_best_scope_does_not_write_to_models(self):
        se, result = _one_model_result()
        R2(se, scope="best").write_results(result)
        assert "R2" not in result.top_models[0].augmentations

    def test_top_scope_writes_r2_and_params_to_top_models(self):
        se, result = _one_model_result()
        R2(se, scope="top").write_results(result)
        model_aug = result.top_models[0].augmentations["R2"]
        assert "r^2" in model_aug
        assert "parameters_r^2" in model_aug

    def test_all_scope_writes_to_all_models(self):
        se, result = _two_model_result()
        R2(se, scope="all").write_results(result)
        assert "R2" in result.all_models[1].augmentations


# ── R2.format_eval_result ─────────────────────────────────────────────────────


class TestR2FormatEvalResult:
    def test_returns_string_when_r2_present(self):
        out = R2.format_eval_result({"best_expr_r^2": 0.95})
        assert "0.95" in out
        assert out != ""

    def test_returns_empty_when_key_absent(self):
        assert R2.format_eval_result({}) == ""


# ── R2.format_model_result ────────────────────────────────────────────────────


class TestR2FormatModelResult:
    def test_returns_r2_with_params(self):
        out = R2.format_model_result({"r^2": 0.9, "parameters_r^2": np.array([1.0])})
        assert "R²=" in out
        assert "params=" in out

    def test_returns_r2_without_params_when_none(self):
        out = R2.format_model_result({"r^2": 0.9, "parameters_r^2": None})
        assert "R²=" in out
        assert "params=" not in out

    def test_returns_r2_without_params_when_key_absent(self):
        out = R2.format_model_result({"r^2": 0.9})
        assert "R²=" in out
        assert "params=" not in out


# ── R2.to_dict / from_dict ────────────────────────────────────────────────────


class TestR2Serialization:
    def test_to_dict_has_expected_keys(self, tmp_path):
        se = _rmse_se()
        aug = R2(se, scope="all", name="myR2")
        d = aug.to_dict(str(tmp_path), "test")
        assert d["format_version"] == 1
        assert d["type"] == "R2"
        assert d["scope"] == "all"
        assert d["name"] == "myR2"

    def test_from_dict_round_trip(self, tmp_path):
        se = _rmse_se()
        aug = R2(se, scope="all", name="myR2")
        d = aug.to_dict(str(tmp_path), "test")
        restored = R2.from_dict(d)
        assert restored.scope == "all"
        assert restored.name == "myR2"
        np.testing.assert_array_equal(restored.evaluator.X, _X)

    def test_from_dict_invalid_format_version_raises(self, tmp_path):
        d = R2(_rmse_se()).to_dict(str(tmp_path), "test")
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            R2.from_dict(d)
