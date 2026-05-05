import pytest
from sympy import Float, Integer, sin
from sympy import symbols as sp_symbols

from SRToolkit.utils.expression_simplifier import (
    _check_tree,
    _denumerate_constants,
    _enumerate_constants,
    _sympy_to_number,
    _sympy_to_sr,
    simplify,
)
from SRToolkit.utils.expression_tree import Node, tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestSimplify:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)

    def test_docstring_example(self):
        expr = ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
        assert "".join(simplify(expr)) == "C+X_1"

    def test_list_in_list_out(self):
        result = simplify(["X_0", "+", "X_1"], self.sl)
        assert isinstance(result, list)

    def test_node_in_node_out(self):
        tree = tokens_to_tree(["X_0", "*", "X_1", "/", "X_0"], self.sl)
        result = simplify(tree, self.sl)
        assert isinstance(result, Node)
        assert result.symbol == "X_1"

    def test_constant_only_collapses_to_single_C(self):
        result = simplify(["C", "+", "C"], SymbolLibrary.default_symbols(num_variables=0))
        assert "".join(result) == "C"

    def test_identity_simplification(self):
        result = simplify(["X_0", "*", "1"], SymbolLibrary.default_symbols(num_variables=1))
        assert "".join(result) == "X_0"

    def test_no_const_in_library_uses_fallback(self):
        sl = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=1)
        result = simplify(["X_0", "+", "X_0"], sl)
        # X_0 + X_0 = 2 * X_0; result contains X_0 and a numeric coefficient
        assert "X_0" in "".join(result)

    def test_fn_with_mixed_constant_and_variable_arg(self):
        # sin(C + X_0): the non-Add/Mul/Pow else branch in _simplify_constants
        result = simplify(["sin", "(", "C", "+", "X_0", ")"])
        assert "sin" in "".join(result)
        assert "X_0" in "".join(result)

    def test_raises_on_invalid_symbol_in_result(self):
        # X_0 * X_0 * X_0 simplifies to X_0**3, which introduces "^" not in the library
        sl = SymbolLibrary.from_symbol_list(["*"], num_variables=1)
        with pytest.raises(Exception, match="invalid symbols"):
            simplify(["X_0", "*", "X_0", "*", "X_0"], sl)


class TestCheckTree:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=1)

    def test_valid_tree_returns_true(self):
        assert _check_tree(Node("X_0"), self.sl) is True

    def test_float_symbol_is_valid(self):
        assert _check_tree(Node("1.5"), self.sl) is True

    def test_invalid_root_returns_false(self):
        assert _check_tree(Node("unknown"), self.sl) is False

    def test_invalid_right_child_returns_false(self):
        node = Node("+", right=Node("unknown"), left=Node("X_0"))
        assert _check_tree(node, self.sl) is False

    def test_invalid_left_child_returns_false(self):
        node = Node("+", right=Node("X_0"), left=Node("unknown"))
        assert _check_tree(node, self.sl) is False


class TestSympyToNumber:
    def test_integer_value_returns_int(self):
        result = _sympy_to_number(Integer(3))
        assert result == 3
        assert isinstance(result, int)

    def test_float_value_returns_float(self):
        result = _sympy_to_number(Float(2.5))
        assert result == 2.5
        assert isinstance(result, float)


class TestSympyToSr:
    def test_number(self):
        node = _sympy_to_sr(Integer(2))
        assert node.symbol == "2"
        assert node.left is None and node.right is None

    def test_symbol(self):
        node = _sympy_to_sr(sp_symbols("X_0"))
        assert node.symbol == "X_0"

    def test_function(self):
        x = sp_symbols("x")
        node = _sympy_to_sr(sin(x))
        assert node.symbol == "sin"
        assert node.left.symbol == "x"

    def test_add_two_terms(self):
        x, y = sp_symbols("x y")
        node = _sympy_to_sr(x + y)
        assert node.symbol == "+"
        assert len(node) == 3

    def test_add_three_terms_builds_nested_tree(self):
        x, y, z = sp_symbols("x y z")
        node = _sympy_to_sr(x + y + z)
        assert node.symbol == "+"
        assert len(node) == 5

    def test_subtraction_detected(self):
        x, y = sp_symbols("x y")
        node = _sympy_to_sr(x - y)
        # Two-term sum where one is negative → detected as subtraction
        assert node.symbol in ("+", "-")
        assert len(node) == 3

    def test_multiply(self):
        x, y = sp_symbols("x y")
        node = _sympy_to_sr(x * y)
        assert node.symbol == "*"
        assert len(node) == 3

    def test_multiply_three_factors_builds_nested_tree(self):
        x, y, z = sp_symbols("x y z")
        node = _sympy_to_sr(x * y * z)
        assert node.symbol == "*"
        assert len(node) == 5

    def test_division(self):
        x, y = sp_symbols("x y")
        node = _sympy_to_sr(x / y)
        assert node.symbol == "/"
        assert len(node) == 3

    def test_pow(self):
        x = sp_symbols("x")
        node = _sympy_to_sr(x**2)
        assert node.symbol == "^"
        assert len(node) == 3


class TestEnumerateConstants:
    def test_single_constant(self):
        expr, constants = _enumerate_constants("C+X_0", "C")
        assert constants == ("C0",)
        assert "C0" in str(expr)

    def test_multiple_constants(self):
        expr, constants = _enumerate_constants("C*X_0+C", "C")
        assert constants == ("C0", "C1")
        assert "C0" in str(expr)
        assert "C1" in str(expr)


class TestDenumerateConstants:
    def test_removes_numeric_suffix(self):
        assert _denumerate_constants("C0*X_0 + C1", "C") == "C*X_0 + C"

    def test_no_constants_unchanged(self):
        assert _denumerate_constants("X_0 + X_1", "C") == "X_0 + X_1"
