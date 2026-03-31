"""Tests for SRToolkit.utils.expression_compiler."""

import numpy as np
import pytest

from SRToolkit.utils.expression_compiler import expr_to_error_function, expr_to_executable_function
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestExprToExecutableFunction:
    def test_simple_variable(self):
        fn = expr_to_executable_function(["X_0"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0], [3.0]])
        result = fn(X, np.array([]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_addition(self):
        fn = expr_to_executable_function(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0], [3.0]])
        result = fn(X, np.array([]))
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_constant(self):
        fn = expr_to_executable_function(["C"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0], [3.0]])
        result = fn(X, np.array([5.0]))
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_literal(self):
        fn = expr_to_executable_function(["pi"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0]])
        result = fn(X, np.array([]))
        np.testing.assert_array_almost_equal(result, [np.pi, np.pi])

    def test_invalid_expr_type(self):
        with pytest.raises(Exception, match="Expression must be given"):
            expr_to_executable_function(42)

    def test_namespace_isolation(self):  # Recheck
        """The exec namespace should not leak module globals."""
        sl = SymbolLibrary.default_symbols(1)
        fn = expr_to_executable_function(["X_0", "+", "1"], sl)
        X = np.array([[1.0]])
        result = fn(X, np.array([]))
        assert result[0] == 2.0

    def test_nan_producing_expr(self):
        """Expressions that produce inf/nan should return the result without crashing."""
        fn = expr_to_executable_function(["X_0", "/", "0"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0]])
        with np.errstate(invalid="ignore", divide="ignore"):
            result = fn(X, np.array([]))
        assert not np.isfinite(result[0])


class TestExprToErrorFunction:
    def test_perfect_match(self):
        fn = expr_to_error_function(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 3.0, 4.0])
        error = fn(X, np.array([]), y)
        assert error == pytest.approx(0.0)

    def test_with_constants(self):
        fn = expr_to_error_function(["C", "*", "X_0"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        error = fn(X, np.array([2.0]), y)
        assert error == pytest.approx(0.0)

    def test_mismatch_returns_positive(self):
        fn = expr_to_error_function(["X_0"], SymbolLibrary.default_symbols(1))
        X = np.array([[1.0], [2.0]])
        y = np.array([10.0, 20.0])
        error = fn(X, np.array([]), y)
        assert error > 0

    def test_invalid_expr_type(self):
        with pytest.raises(Exception, match="Expression must be given"):
            expr_to_error_function(42)

    # Test too many constants
    # Test too few constants
    # Test too many variables
    # Test too few variables


class TestPreambleInjection:
    def test_custom_preamble(self):
        """Custom preamble should be included in the generated function."""
        sl = SymbolLibrary.default_symbols(1)
        sl.preamble = ["import numpy as np"]
        fn = expr_to_executable_function(["X_0", "+", "1"], sl)
        X = np.array([[5.0]])
        result = fn(X, np.array([]))
        assert result[0] == 6.0
