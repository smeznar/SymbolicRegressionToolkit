"""Tests for SRToolkit.evaluation.sr_evaluator."""

import tempfile

import numpy as np
import pytest

from SRToolkit.evaluation.sr_evaluator import EvalResult, ModelResult, SR_evaluator


@pytest.fixture
def simple_evaluator():
    X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
    y = np.array([3.0, 0.0, 3.0, 11.0])
    return SR_evaluator(X, y, seed=42)


class TestSREvaluatorBasic:
    def test_evaluate_simple_expr(self, simple_evaluator):
        rmse = simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert rmse < 1e-6

    def test_evaluate_caches(self, simple_evaluator):
        rmse1 = simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        rmse2 = simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert rmse1 == rmse2
        assert len(simple_evaluator.models) == 1

    def test_evaluate_invalid_returns_nan(self, simple_evaluator):
        with np.errstate(invalid="ignore"):
            result = simple_evaluator.evaluate_expr(["C", "*", "X_1", "X_0"])
        assert np.isnan(result)
        assert "C*X_1X_0" in simple_evaluator.invalid

    def test_max_evaluations(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        se = SR_evaluator(X, y, max_evaluations=1, seed=42)
        se.evaluate_expr(["X_0"])
        with np.errstate(invalid="ignore"):
            result = se.evaluate_expr(["X_1"])
        assert np.isnan(result)

    def test_unknown_ranking_function_warns(self):
        X = np.array([[1.0]])
        y = np.array([1.0])
        with pytest.warns(UserWarning, match="not supported"):
            SR_evaluator(X, y, ranking_function="unknown")

    def test_rmse_requires_y(self):
        X = np.array([[1.0]])
        with pytest.raises(ValueError, match="Target values must be provided"):
            SR_evaluator(X, ranking_function="rmse")

    def test_bed_requires_ground_truth(self):
        X = np.array([[1.0]])
        with pytest.raises(ValueError, match="Ground truth must be provided"):
            SR_evaluator(X, ranking_function="bed")


class TestSREvaluatorGetResults:
    def test_get_results(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = simple_evaluator.get_results(top_k=1)

        assert len(results) == 1
        assert results[0].num_evaluated == 1
        assert results[0].evaluation_calls == 1
        assert results[0].best_expr == "C*X_1-X_0"
        assert results[0].min_error < 1e-6
        assert results[0].success

    def test_top_k(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        simple_evaluator.evaluate_expr(["X_0", "+", "X_1"])
        results = simple_evaluator.get_results(top_k=1)
        assert len(results[0].top_models) == 1
        assert len(results[0].all_models) == 2

    def test_get_results_metadata(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        se = SR_evaluator(X, y, metadata={"dataset_name": "test", "key": "val"}, seed=42)
        se.evaluate_expr(["X_0"])
        results = se.get_results()
        assert results[0].dataset_name == "test"
        assert results[0].metadata == {"key": "val"}


class TestSREvaluatorSerialization:
    def test_to_dict_round_trip(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        with tempfile.TemporaryDirectory() as tmpdir:
            d = simple_evaluator.to_dict(tmpdir, "test")
            restored = SR_evaluator.from_dict(d)

            assert restored.max_evaluations == simple_evaluator.max_evaluations
            assert restored.ranking_function == simple_evaluator.ranking_function
            assert restored.success_threshold == simple_evaluator.success_threshold
            np.testing.assert_array_equal(restored.X, simple_evaluator.X)
            np.testing.assert_array_equal(restored.y, simple_evaluator.y)

    def test_to_dict_bad_format_version(self, simple_evaluator):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = simple_evaluator.to_dict(tmpdir, "test")
            d["format_version"] = 99
            with pytest.raises(ValueError, match="Unsupported format_version"):
                SR_evaluator.from_dict(d)


class TestSRResults:
    def test_add_does_not_mutate(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        se1 = SR_evaluator(X, y, seed=42)
        se1.evaluate_expr(["X_0"])
        r1 = se1.get_results()

        se2 = SR_evaluator(X, y, seed=42)
        se2.evaluate_expr(["X_1"])
        r2 = se2.get_results()

        r1_len_before = len(r1)
        r3 = r1 + r2
        assert len(r1) == r1_len_before  # r1 not mutated
        assert len(r3) == 2
        assert r3 is not r1

    def test_iadd_mutates(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        se1 = SR_evaluator(X, y, seed=42)
        se1.evaluate_expr(["X_0"])
        r1 = se1.get_results()

        se2 = SR_evaluator(X, y, seed=42)
        se2.evaluate_expr(["X_1"])
        r2 = se2.get_results()

        r1 += r2
        assert len(r1) == 2

    def test_getitem(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["X_0"])
        results = simple_evaluator.get_results()
        result = results[0]
        assert isinstance(result, EvalResult)

    def test_getitem_out_of_bounds(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["X_0"])
        results = simple_evaluator.get_results()
        with pytest.raises(AssertionError):
            results[99]

    def test_len(self, simple_evaluator):
        simple_evaluator.evaluate_expr(["X_0"])
        results = simple_evaluator.get_results()
        assert len(results) == 1


class TestModelResult:
    def test_add_augmentation(self):
        mr = ModelResult(expr=["X_0"], error=0.5)
        mr.add_augmentation("test", {"value": 42}, "TestAugmenter")
        assert "test" in mr.augmentations
        assert mr.augmentations["test"]["value"] == 42
        assert mr.augmentations["test"]["_type"] == "TestAugmenter"

    def test_add_augmentation_collision(self):
        mr = ModelResult(expr=["X_0"], error=0.5)
        mr.add_augmentation("test", {"a": 1}, "TestAugmenter")
        mr.add_augmentation("test", {"b": 2}, "TestAugmenter")
        assert "test" in mr.augmentations
        assert "test_1" in mr.augmentations


class TestEvalResult:
    def test_add_augmentation(self):
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
        er.add_augmentation("test", {"value": 42}, "TestAugmenter")
        assert "test" in er.augmentations
