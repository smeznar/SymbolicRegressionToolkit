"""Tests for SRToolkit.utils.expression_simplifier."""

import pytest

from SRToolkit.utils.expression_simplifier import simplify
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestSimplify:
    def test_constant_collapsing(self):
        expr = ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
        result = simplify(expr)
        assert "".join(result) == "C+X_1"

    def test_already_simplified(self):
        expr = ["X_0", "+", "X_1"]
        result = simplify(expr, SymbolLibrary.default_symbols(2))
        assert "".join(result) == "X_0+X_1"

    def test_identity_simplification(self):
        expr = ["X_0", "*", "1"]
        result = simplify(expr, SymbolLibrary.default_symbols(1))
        assert "".join(result) == "X_0"

    def test_returns_list(self):
        expr = ["X_0", "+", "C"]
        result = simplify(expr, SymbolLibrary.default_symbols(1))
        assert isinstance(result, list)

    def test_constant_only_expression(self):
        """An expression with only constants should collapse to a single C."""
        expr = ["C", "+", "C"]
        result = simplify(expr, SymbolLibrary.default_symbols(0))
        assert "".join(result) == "C"

    def test_missing_symbols_raises(self):
        """Simplifying with a symbol library missing required symbols should raise."""
        sl = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=1)
        expr = ["X_0", "*", "X_0", "^2"]
        with pytest.raises(Exception, match="invalid symbols"):
            simplify(expr, sl)
