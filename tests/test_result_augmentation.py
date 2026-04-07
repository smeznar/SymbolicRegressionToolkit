"""Tests for SRToolkit.evaluation.result_augmentation and SR_results.augment()."""

import numpy as np
import pytest

from SRToolkit.evaluation.result_augmentation import (
    R2,
    RMSE,
    ExpressionSimplifier,
    ExpressionToLatex,
)
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.symbol_library import SymbolLibrary


@pytest.fixture
def evaluator():
    X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
    y = np.array([3.0, 0.0, 3.0, 11.0])
    return SR_evaluator(X, y, seed=42, success_threshold=-1)


@pytest.fixture
def results(evaluator):
    evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
    evaluator.evaluate_expr(["C", "-", "X_0"])
    return evaluator.get_results(top_k=1)


class TestAugmentMethod:
    def test_augment_all_results(self, evaluator):
        evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = evaluator.get_results(top_k=1)

        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl)])

        assert "ExpressionToLatex" in results[0].augmentations
        assert "best_expr_latex" in results[0].augmentations["ExpressionToLatex"]

    def test_augment_specific_experiment(self, evaluator):
        """Multiple experiments, augment only one."""
        X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
        y = np.array([3.0, 0.0, 3.0, 11.0])

        se1 = SR_evaluator(X, y, seed=42)
        se1.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        r1 = se1.get_results()

        se2 = SR_evaluator(X, y, seed=42)
        se2.evaluate_expr(["X_0", "+", "X_1"])
        r2 = se2.get_results()

        combined = r1 + r2
        assert len(combined) == 2

        sl = SymbolLibrary.default_symbols(2)
        combined.augment([ExpressionToLatex(sl)], experiment_number=0)

        assert "ExpressionToLatex" in combined[0].augmentations
        assert "ExpressionToLatex" not in combined[1].augmentations

    def test_augment_out_of_bounds(self, results):
        sl = SymbolLibrary.default_symbols(2)
        with pytest.raises(AssertionError, match="out of bounds"):
            results.augment([ExpressionToLatex(sl)], experiment_number=99)


class TestExpressionToLatex:
    def test_latex_on_perfect_match(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl)])

        latex = results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]
        assert latex == "$C_{0} \\cdot X_{1} - X_{0}$"

    def test_scope_best(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl, scope="best")])

        # Should have best_expr_latex on EvalResult
        assert "best_expr_latex" in results[0].augmentations["ExpressionToLatex"]
        # But NOT on top models (scope="best" skips model augmentation)
        assert "ExpressionToLatex" not in results[0].top_models[0].augmentations
        assert "ExpressionToLatex" not in results[0].all_models[0].augmentations

    def test_scope_top(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl, scope="top")])

        assert "best_expr_latex" in results[0].augmentations["ExpressionToLatex"]
        assert "ExpressionToLatex" in results[0].top_models[0].augmentations
        # Model 0 will have the augmentation because of the shared reference, but model 1 won't'
        assert "ExpressionToLatex" not in results[0].all_models[1].augmentations

    def test_scope_all(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl, scope="all")])

        assert "best_expr_latex" in results[0].augmentations["ExpressionToLatex"]
        assert "ExpressionToLatex" in results[0].top_models[0].augmentations
        assert "ExpressionToLatex" in results[0].all_models[1].augmentations


class TestExpressionSimplifier:
    def test_simplify(self, evaluator):
        evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = evaluator.get_results(top_k=1)

        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionSimplifier(sl)])

        assert "ExpressionSimplifier" in results[0].augmentations
        simplified = results[0].augmentations["ExpressionSimplifier"]["simplified_best_expr"]
        assert isinstance(simplified, str)
        assert len(simplified) > 0


class TestR2Augmenter:
    def test_r2_on_perfect_match(self, evaluator):
        evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = evaluator.get_results(top_k=1)

        results.augment([R2(evaluator)])

        r2_data = results[0].augmentations["R2"]
        assert "best_expr_r^2" in r2_data
        assert 0.99 < r2_data["best_expr_r^2"] <= 1.0

    def test_r2_on_bad_expr(self, evaluator):
        evaluator.evaluate_expr(["X_0", "+", "X_1"])  # Not a perfect match
        results = evaluator.get_results(top_k=1)

        results.augment([R2(evaluator)])

        r2_data = results[0].augmentations["R2"]
        assert r2_data["best_expr_r^2"] < 1.0


class TestRMSEAugmenter:
    def test_rmse_on_perfect_match(self, evaluator):
        evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = evaluator.get_results(top_k=1)

        results.augment([RMSE(evaluator)])

        rmse_data = results[0].augmentations["RMSE"]
        assert "min_error" in rmse_data
        assert rmse_data["min_error"] < 1e-6


class TestAugmentErrorHandling:
    def test_broken_augmenter_doesnt_crash(self, results, caplog):
        """A failing augmenter should warn but not crash the whole augmentation."""

        class BrokenAugmenter:
            name = "Broken"

            def write_results(self, result):
                raise ValueError("intentional error")

        with pytest.warns(UserWarning, match="Error augmenting"):
            results.augment([BrokenAugmenter()])

    def test_multiple_augmenters(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl), ExpressionSimplifier(sl)])

        assert "ExpressionToLatex" in results[0].augmentations
        assert "ExpressionSimplifier" in results[0].augmentations

    def test_multiple_augmenters_same_type(self, results):
        sl = SymbolLibrary.default_symbols(2)
        results.augment([ExpressionToLatex(sl), ExpressionToLatex(sl)])

        assert "ExpressionToLatex" in results[0].augmentations
        assert "ExpressionToLatex_1" in results[0].augmentations
