"""Tests for SRToolkit.evaluation.parameter_estimator."""

import numpy as np
import pytest

from SRToolkit.evaluation.parameter_estimator import ParameterEstimator


@pytest.fixture
def simple_pe():
    X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
    y = np.array([3.0, 0.0, 3.0, 11.0])
    return ParameterEstimator(X, y, seed=42)


class TestParameterEstimator:
    def test_estimate_with_constant(self, simple_pe):
        rmse, constants = simple_pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
        assert rmse < 1e-6
        assert len(constants) == 1
        assert 1.99 < constants[0] < 2.01

    def test_estimate_no_constants(self, simple_pe):
        rmse, constants = simple_pe.estimate_parameters(["X_0", "+", "X_1"])
        assert isinstance(rmse, float)
        assert len(constants) == 0
        assert rmse > 0

    def test_max_constants_limit(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        pe = ParameterEstimator(X, y, max_constants=1, seed=42)
        rmse, constants = pe.estimate_parameters(["C", "+", "C"])
        assert np.isnan(rmse)
        assert len(constants) == 0

    def test_max_expr_length_limit(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        pe = ParameterEstimator(X, y, max_expr_length=2, seed=42)
        rmse, constants = pe.estimate_parameters(["X_0", "+", "X_0", "+", "X_0"])
        assert np.isnan(rmse)

    def test_zero_max_constants_allows_no_const_expr(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        pe = ParameterEstimator(X, y, max_constants=0, seed=42)
        rmse, constants = pe.estimate_parameters(["X_0"])
        assert rmse == pytest.approx(0.0)

    def test_mean_initialization(self):
        X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
        y = np.array([3.0, 0.0, 3.0, 11.0])
        pe = ParameterEstimator(X, y, initialization="mean", seed=42)
        rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
        assert rmse < 1e-6

    def test_custom_constant_bounds(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([10.0, 20.0])
        pe = ParameterEstimator(X, y, constant_bounds=(5, 15), seed=42)
        rmse, constants = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-6
        assert 9.9 < constants[0] < 10.1
