"""Tests for SRToolkit.utils.measures."""

import numpy as np
import pytest

from SRToolkit.utils.measures import bed, create_behavior_matrix, edit_distance, tree_edit_distance


class TestEditDistance:
    def test_identical_expressions(self):
        assert edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"]) == 0

    def test_different_expressions(self):
        assert edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"]) == 1


class TestTreeEditDistance:
    def test_identical(self):
        assert tree_edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"]) == 0

    def test_different(self):
        assert tree_edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"]) == 1


class TestCreateBehaviorMatrix:
    def test_shape(self):
        X = np.random.rand(10, 2)
        bm = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32)
        assert bm.shape == (10, 32)

    def test_deterministic_no_constants(self):
        X = np.random.rand(10, 2)
        bm1 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        bm2 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        np.testing.assert_array_equal(bm1, bm2)

    def test_constants_bounds_affect_output(self):
        X = np.random.rand(10, 2)
        bm_low = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=64, consts_bounds=(0, 1), seed=42)
        bm_high = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=64, consts_bounds=(1, 5), seed=42)
        assert np.mean(bm_low) < np.mean(bm_high)

    def test_invalid_expr_raises(self):
        X = np.random.rand(10, 2)
        with pytest.raises(Exception, match="Expression should be"):
            create_behavior_matrix(42, X)


class TestBED:
    def test_identical_expressions_low_distance(self):
        X = np.random.rand(20, 2)
        result = bed(["X_0", "+", "C"], ["X_0", "+", "C"], X, seed=42)
        assert result < 0.5

    def test_with_domain_bounds(self):
        result = bed(
            ["X_0", "+", "C"],
            ["X_1", "+", "C"],
            domain_bounds=[(0, 1), (0, 1)],
            seed=42,
        )
        assert result < 1.0

    def test_missing_X_and_domain_raises(self):
        with pytest.raises(Exception, match="domain_bounds parameter must be given"):
            bed(["X_0", "+", "C"], ["X_1", "+", "C"])

    def test_behavior_matrices(self):
        X = np.random.rand(10, 2)
        bm1 = create_behavior_matrix(["X_0", "+", "C"], X, seed=42)
        bm2 = create_behavior_matrix(["X_1", "+", "C"], X, seed=42)
        result = bed(bm1, bm2)
        assert isinstance(result, float)
        assert result >= 0

    def test_mismatched_behavior_matrix_shapes(self):
        bm1 = np.random.rand(10, 32)
        bm2 = np.random.rand(5, 32)
        with pytest.raises(AssertionError):
            bed(bm1, bm2)
