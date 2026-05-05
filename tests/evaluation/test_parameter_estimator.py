from unittest.mock import patch

import numpy as np
import pytest

from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.symbol_library import SymbolLibrary

# ── Shared data ─────────────────────────────────────────────────────────────────
# Two-feature dataset; analytical solutions exist for the expressions used below.

_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
_y_exact = _X[:, 0] + _X[:, 1]  # y = X_0 + X_1  (no-constant test)
_y_linear = 2.0 * _X[:, 0]  # y = 2*X_0      (one-constant test, C should → 2)


# ── ParameterEstimator.__init__ ────────────────────────────────────────────────


class TestParameterEstimatorInit:
    def test_stores_X_y_seed(self):
        pe = ParameterEstimator(_X, _y_linear, seed=7)
        assert pe.X is _X
        assert pe.y is _y_linear
        assert pe.seed == 7

    def test_default_symbol_library_when_none(self):
        pe = ParameterEstimator(_X, _y_linear)
        assert pe.symbol_library is not None
        assert isinstance(pe.symbol_library, SymbolLibrary)

    def test_custom_symbol_library_stored(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        pe = ParameterEstimator(_X, _y_linear, symbol_library=sl)
        assert pe.symbol_library is sl

    def test_default_estimation_settings(self):
        pe = ParameterEstimator(_X, _y_linear)
        s = pe.estimation_settings
        assert s["method"] == "L-BFGS-B"
        assert s["tol"] == 1e-6
        assert s["gtol"] == 1e-3
        assert s["max_iter"] == 100
        assert s["constant_bounds"] == (-5, 5)
        assert s["initialization"] == "random"
        assert s["max_constants"] == 8
        assert s["max_expr_length"] == -1
        assert s["backend"] == "stack"

    def test_kwargs_override_known_settings(self):
        pe = ParameterEstimator(_X, _y_linear, method="Nelder-Mead", max_iter=50)
        assert pe.estimation_settings["method"] == "Nelder-Mead"
        assert pe.estimation_settings["max_iter"] == 50

    def test_unknown_kwargs_are_ignored(self):
        pe = ParameterEstimator(_X, _y_linear, not_a_real_key=99)
        assert "not_a_real_key" not in pe.estimation_settings

    def test_invalid_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="backend"):
            ParameterEstimator(_X, _y_linear, backend="numpy")

    def test_valid_backends_accepted(self):
        for backend in ("stack", "codegen", "stack_py"):
            ParameterEstimator(_X, _y_linear, backend=backend)


# ── ParameterEstimator.estimate_parameters ─────────────────────────────────────


class TestEstimateParametersNoConstants:
    def test_returns_near_zero_rmse(self):
        pe = ParameterEstimator(_X, _y_exact)
        rmse, _ = pe.estimate_parameters(["X_0", "+", "X_1"])
        assert rmse < 1e-10

    def test_returns_empty_parameters_array(self):
        pe = ParameterEstimator(_X, _y_exact)
        _, params = pe.estimate_parameters(["X_0", "+", "X_1"])
        assert isinstance(params, np.ndarray)
        assert params.size == 0


class TestEstimateParametersWithConstants:
    def test_one_constant_recovers_coefficient(self):
        pe = ParameterEstimator(_X, _y_linear, seed=0)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert len(params) == 1
        assert abs(params[0] - 2.0) < 0.01

    def test_node_input_branch(self):
        # Node tree for C * X_0: prefix order is ["*", "C", "X_0"]
        node = Node("*", left=Node("C"), right=Node("X_0"))
        pe = ParameterEstimator(_X, _y_linear, seed=0)
        rmse, params = pe.estimate_parameters(node)
        assert rmse < 1e-4
        assert abs(params[0] - 2.0) < 0.01


class TestEstimateParametersGuards:
    def test_max_constants_exceeded_returns_nan(self):
        pe = ParameterEstimator(_X, _y_linear, max_constants=0)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert np.isnan(rmse)
        assert params.size == 0

    def test_max_constants_at_limit_not_rejected(self):
        # strict < means exactly max_constants constants is allowed
        pe = ParameterEstimator(_X, _y_linear, max_constants=1, seed=0)
        rmse, _ = pe.estimate_parameters(["C", "*", "X_0"])
        assert not np.isnan(rmse)

    def test_max_constants_negative_not_filtered(self):
        # 0 <= -1 is False, so filter never triggers
        pe = ParameterEstimator(_X, _y_linear, max_constants=-1, seed=0)
        rmse, _ = pe.estimate_parameters(["C", "*", "X_0"])
        assert not np.isnan(rmse)

    def test_max_expr_length_exceeded_returns_nan(self):
        # ["C", "*", "X_0"] has length 3; set limit to 2
        pe = ParameterEstimator(_X, _y_linear, max_expr_length=2)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert np.isnan(rmse)
        assert params.size == 0

    def test_max_expr_length_at_limit_not_rejected(self):
        # strict < means length == max_expr_length is allowed
        pe = ParameterEstimator(_X, _y_linear, max_expr_length=3, seed=0)
        rmse, _ = pe.estimate_parameters(["C", "*", "X_0"])
        assert not np.isnan(rmse)

    def test_max_expr_length_negative_not_filtered(self):
        # 0 <= -1 is False, so filter never triggers
        pe = ParameterEstimator(_X, _y_linear, max_expr_length=-1, seed=0)
        rmse, _ = pe.estimate_parameters(["C", "*", "X_0"])
        assert not np.isnan(rmse)

    def test_zero_max_constants_with_zero_const_expression(self):
        # 0 <= 0 < 0 is False, so a no-constant expression must not be filtered
        pe = ParameterEstimator(_X, _y_exact, max_constants=0)
        rmse, params = pe.estimate_parameters(["X_0", "+", "X_1"])
        assert not np.isnan(rmse)
        assert params.size == 0


# ── ParameterEstimator._optimize_parameters (via estimate_parameters) ──────────


class TestOptimizeParameters:
    def test_invalid_constant_bounds_not_tuple_raises(self):
        pe = ParameterEstimator(_X, _y_linear, constant_bounds=5)
        with pytest.raises(ValueError, match="constant_bounds"):
            pe.estimate_parameters(["C", "*", "X_0"])

    def test_invalid_constant_bounds_wrong_length_raises(self):
        pe = ParameterEstimator(_X, _y_linear, constant_bounds=(1, 2, 3))
        with pytest.raises(ValueError, match="constant_bounds"):
            pe.estimate_parameters(["C", "*", "X_0"])

    def test_mean_initialization_converges(self):
        pe = ParameterEstimator(_X, _y_linear, initialization="mean")
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert abs(params[0] - 2.0) < 0.01

    def test_random_initialization_is_reproducible_with_seed(self):
        pe1 = ParameterEstimator(_X, _y_linear, seed=42)
        pe2 = ParameterEstimator(_X, _y_linear, seed=42)
        rmse1, params1 = pe1.estimate_parameters(["C", "*", "X_0"])
        rmse2, params2 = pe2.estimate_parameters(["C", "*", "X_0"])
        assert rmse1 == rmse2
        assert np.array_equal(params1, params2)

    def test_custom_constant_bounds_respected(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = 10.0 * X[:, 0]
        pe = ParameterEstimator(X, y, constant_bounds=(5, 15), seed=0)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert 9.9 < params[0] < 10.1

    def test_random_initialization_differs_across_seeds(self):
        x0_values = []

        from scipy.optimize import minimize as real_minimize

        def capture_x0(fn, x0, **kwargs):
            x0_values.append(x0.copy())
            return real_minimize(fn, x0, **kwargs)

        with patch("SRToolkit.evaluation.parameter_estimator.minimize", side_effect=capture_x0):
            ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(["C", "*", "X_0"])
            ParameterEstimator(_X, _y_linear, seed=99).estimate_parameters(["C", "*", "X_0"])

        assert not np.allclose(x0_values[0], x0_values[1])
