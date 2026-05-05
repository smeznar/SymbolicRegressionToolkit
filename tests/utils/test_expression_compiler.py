import numpy as np
import pytest

from SRToolkit.utils import _eval_cython as _eval
from SRToolkit.utils.expression_compiler import (
    _build_partial_instructions,
    _collect_maximal_const_free,
    _compile_fallback_binary,
    _compile_fallback_unary,
    _expr_to_cython_callable,
    _expr_to_cython_error_callable,
    _expr_to_error_function,
    _expr_to_executable_function,
    _expr_to_python_callable,
    _expr_to_python_stack_error_callable,
    _prepare_cython_traversal,
    _prepare_cython_traversal_partial,
    _tree_to_function_rec,
    compile_expr,
    compile_expr_rmse,
)
from SRToolkit.utils.expression_tree import Node, tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sl(n_vars=2):
    return SymbolLibrary.default_symbols(num_variables=n_vars)


def _X(*rows):
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# _tree_to_function_rec
# ---------------------------------------------------------------------------


class TestTreeToFunctionRec:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])

    def test_var_leaf_returns_np_fn_directly(self):
        tree = Node("X_0")
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert code == []
        assert symbol == "X[:, 0]"
        assert vc == 0
        assert cc == 0

    def test_var_second_column(self):
        tree = Node("X_1")
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert symbol == "X[:, 1]"

    def test_lit_leaf_generates_np_full(self):
        tree = Node("pi")
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert len(code) == 1
        assert "np.full" in code[0]
        assert symbol == "y_0"
        assert vc == 1
        assert cc == 0

    def test_const_leaf_returns_c_index(self):
        tree = Node("C")
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert code == []
        assert "0" in symbol
        assert vc == 0
        assert cc == 1

    def test_multiple_consts_get_sequential_indices(self):
        # C + C  ->  two consts, indexed 0 and 1
        left = Node("C")
        right = Node("C")
        tree = Node("+", right=right, left=left)
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert cc == 2
        # both C[0] and C[1] should appear in the generated code
        full_code = " ".join(code)
        assert "C[0]" in full_code or "0" in full_code
        assert "C[1]" in full_code or "1" in full_code

    def test_float_leaf_returns_symbol_string(self):
        tree = Node("1.5")
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert code == []
        assert symbol == "1.5"

    def test_unknown_leaf_raises(self):
        tree = Node("??invalid??")
        with pytest.raises(Exception, match="invalid symbol"):
            _tree_to_function_rec(tree, self.sl)

    def test_unary_fn(self):
        tree = Node("sin", left=Node("X_0"))
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert len(code) == 1
        assert "np.sin" in code[0]
        assert "X[:, 0]" in code[0]

    def test_binary_op(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        code, symbol, vc, cc = _tree_to_function_rec(tree, self.sl)
        assert len(code) == 1
        assert "X[:, 0]" in code[0]
        assert "X[:, 1]" in code[0]

    def test_var_counter_increments_for_each_intermediate(self):
        # sin(sin(X_0)) — two intermediates
        inner = Node("sin", left=Node("X_0"))
        outer = Node("sin", left=inner)
        code, symbol, vc, cc = _tree_to_function_rec(outer, self.sl)
        assert vc == 2
        assert symbol == "y_1"


# ---------------------------------------------------------------------------
# _compile_fallback_unary
# ---------------------------------------------------------------------------


class TestCompileFallbackUnary:
    def setup_method(self):
        self.preamble = ["import numpy as np"]

    def test_basic_sin(self):
        fn = _compile_fallback_unary("np.sin({})", self.preamble)
        x = np.array([0.0, np.pi / 2])
        result = fn(x)
        assert result == pytest.approx(np.sin(x))

    def test_negation(self):
        fn = _compile_fallback_unary("-{}", self.preamble)
        x = np.array([1.0, 2.0])
        assert fn(x) == pytest.approx(-x)

    def test_custom_preamble_used(self):
        fn = _compile_fallback_unary("np.sqrt({})", ["import numpy as np"])
        x = np.array([4.0, 9.0])
        assert fn(x) == pytest.approx([2.0, 3.0])

    def test_empty_preamble(self):
        fn = _compile_fallback_unary("np.abs({})", [])
        x = np.array([-1.0, -2.0])
        assert fn(x) == pytest.approx([1.0, 2.0])


# ---------------------------------------------------------------------------
# _compile_fallback_binary
# ---------------------------------------------------------------------------


class TestCompileFallbackBinary:
    def setup_method(self):
        self.preamble = ["import numpy as np"]

    def test_basic_add(self):
        fn = _compile_fallback_binary("{} + {}", self.preamble)
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert fn(a, b) == pytest.approx([4.0, 6.0])

    def test_multiply(self):
        fn = _compile_fallback_binary("{} * {}", self.preamble)
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        assert fn(a, b) == pytest.approx([8.0, 15.0])

    def test_custom_preamble(self):
        fn = _compile_fallback_binary("np.power({}, {})", ["import numpy as np"])
        a = np.array([2.0])
        b = np.array([3.0])
        assert fn(a, b) == pytest.approx([8.0])

    def test_empty_preamble(self):
        fn = _compile_fallback_binary("{} - {}", [])
        a = np.array([5.0])
        b = np.array([3.0])
        assert fn(a, b) == pytest.approx([2.0])


# ---------------------------------------------------------------------------
# _collect_maximal_const_free
# ---------------------------------------------------------------------------


class TestCollectMaximalConstFree:
    def setup_method(self):
        self.sl = _sl(2)

    def test_float_node_returns_false_empty(self):
        node = Node("1.5")
        has_const, nodes = _collect_maximal_const_free(node, self.sl)
        assert has_const is False
        assert nodes == []

    def test_const_node_returns_true_empty(self):
        node = Node("C")
        has_const, nodes = _collect_maximal_const_free(node, self.sl)
        assert has_const is True
        assert nodes == []

    def test_var_leaf_returns_false_empty(self):
        node = Node("X_0")
        has_const, nodes = _collect_maximal_const_free(node, self.sl)
        assert has_const is False
        assert nodes == []

    def test_lit_leaf_returns_false_empty(self):
        node = Node("pi")
        has_const, nodes = _collect_maximal_const_free(node, self.sl)
        assert has_const is False
        assert nodes == []

    def test_expression_with_no_const_returns_false_empty(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is False
        assert nodes == []

    def test_x0_plus_c_left_is_maximal_const_free(self):
        # X_0 + C  ->  left child X_0 is const-free, right C has const
        x0 = Node("X_0")
        c = Node("C")
        tree = Node("+", left=x0, right=c)
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert x0 in nodes
        assert c not in nodes

    def test_c_plus_x0_right_is_maximal_const_free(self):
        # C + X_0  ->  right child X_0 is const-free, left C has const
        c = Node("C")
        x0 = Node("X_0")
        tree = Node("+", left=c, right=x0)
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert x0 in nodes
        assert c not in nodes

    def test_sin_c_unary_const_child_no_const_free_nodes(self):
        # sin(C)  ->  unary fn with const child: no const-free nodes
        c = Node("C")
        tree = Node("sin", left=c)
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert nodes == []

    def test_sin_x0_plus_c_subtree_identified(self):
        # sin(X_0) + C  ->  sin(X_0) is the maximal const-free subtree
        sin_x0 = Node("sin", left=Node("X_0"))
        c = Node("C")
        tree = Node("+", left=sin_x0, right=c)
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert sin_x0 in nodes

    def test_c_plus_c_both_const_no_const_free_nodes(self):
        # C + C  ->  both children are const: returns True but no const-free nodes
        tree = Node("+", left=Node("C"), right=Node("C"))
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert nodes == []

    def test_x0_plus_c_times_x1_both_vars_identified(self):
        # X_0 + (C * X_1)  ->  both X_0 and X_1 are maximal const-free
        x0 = Node("X_0")
        x1 = Node("X_1")
        c = Node("C")
        c_times_x1 = Node("*", left=c, right=x1)
        tree = Node("+", left=x0, right=c_times_x1)
        has_const, nodes = _collect_maximal_const_free(tree, self.sl)
        assert has_const is True
        assert x0 in nodes
        assert x1 in nodes


# ---------------------------------------------------------------------------
# _build_partial_instructions
# ---------------------------------------------------------------------------


class TestBuildPartialInstructions:
    def setup_method(self):
        self.sl = _sl(2)

    def _empty_lists(self):
        return [], [], [], [], [], [0]

    def test_cached_node_emits_cached(self):
        node = Node("X_0")
        cached_array = np.array([1.0, 2.0])
        const_free_ids = {id(node)}
        cached_arrays = {id(node): cached_array}
        tok, ari, ei, fv, po, cc = self._empty_lists()
        _build_partial_instructions(node, self.sl, const_free_ids, cached_arrays, tok, ari, ei, fv, po, cc)
        assert tok[0] == _eval.CACHED
        assert ari[0] == 0
        assert po[0] is cached_array

    def test_const_node_emits_const_and_increments_counter(self):
        node = Node("C")
        tok, ari, ei, fv, po, cc = self._empty_lists()
        _build_partial_instructions(node, self.sl, set(), {}, tok, ari, ei, fv, po, cc)
        assert tok[0] == _eval.CONST
        assert ei[0] == 0
        assert cc[0] == 1

    def test_fn_with_cython_id_emits_cython_id(self):
        # sin(C): sin has a positive cython_id
        c = Node("C")
        tree = Node("sin", left=c)
        tok, ari, ei, fv, po, cc = self._empty_lists()
        _build_partial_instructions(tree, self.sl, set(), {}, tok, ari, ei, fv, po, cc)
        # postfix: C, sin
        assert tok[0] == _eval.CONST
        assert tok[1] == self.sl.get_cython_id("sin")
        assert ari[1] == 1
        assert po[1] is None

    def test_fn_with_negative_cython_id_emits_python(self):
        sl = _sl(2)
        sl.add_symbol("myfn", "fn", 5, "np.sin({})")
        c = Node("C")
        tree = Node("myfn", left=c)
        tok, ari, ei, fv, po, cc = [], [], [], [], [], [0]
        _build_partial_instructions(tree, sl, set(), {}, tok, ari, ei, fv, po, cc)
        assert tok[1] == _eval.PYTHON
        assert callable(po[1])

    def test_op_with_cython_id_emits_cython_id(self):
        # C + C: + has a positive cython_id
        c1 = Node("C")
        c2 = Node("C")
        tree = Node("+", left=c1, right=c2)
        tok, ari, ei, fv, po, cc = self._empty_lists()
        _build_partial_instructions(tree, self.sl, set(), {}, tok, ari, ei, fv, po, cc)
        assert tok[2] == self.sl.get_cython_id("+")
        assert ari[2] == 2
        assert po[2] is None

    def test_op_with_negative_cython_id_emits_python(self):
        sl = _sl(2)
        sl.add_symbol("myop", "op", 0, "{} + {}")
        c1 = Node("C")
        c2 = Node("C")
        tree = Node("myop", left=c1, right=c2)
        tok, ari, ei, fv, po, cc = [], [], [], [], [], [0]
        _build_partial_instructions(tree, sl, set(), {}, tok, ari, ei, fv, po, cc)
        assert tok[2] == _eval.PYTHON
        assert callable(po[2])

    def test_var_not_in_const_free_ids_raises(self):
        node = Node("X_0")
        tok, ari, ei, fv, po, cc = self._empty_lists()
        with pytest.raises(ValueError, match="Unexpected token"):
            _build_partial_instructions(node, self.sl, set(), {}, tok, ari, ei, fv, po, cc)


# ---------------------------------------------------------------------------
# _prepare_cython_traversal
# ---------------------------------------------------------------------------


class TestPrepareCythonTraversal:
    def setup_method(self):
        self.sl = _sl(2)

    def test_var_token_emits_var_with_column_index(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["X_0"], self.sl)
        assert tok[0] == _eval.VAR
        assert ei[0] == 0

    def test_var_second_column(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["X_1"], self.sl)
        assert tok[0] == _eval.VAR
        assert ei[0] == 1

    def test_const_token_emits_const_sequential_indices(self):
        tree = tokens_to_tree(["X_0", "+", "C"], self.sl)
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, self.sl)
        const_positions = [i for i, t in enumerate(tok) if t == _eval.CONST]
        assert len(const_positions) == 1
        assert ei[const_positions[0]] == 0

    def test_multiple_consts_sequential(self):
        # Build C + C
        tree = Node("+", left=Node("C"), right=Node("C"))
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, self.sl)
        const_positions = [i for i, t in enumerate(tok) if t == _eval.CONST]
        assert len(const_positions) == 2
        indices = sorted(ei[p] for p in const_positions)
        assert indices == [0, 1]

    def test_lit_pi_emits_float(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["pi"], self.sl)
        assert tok[0] == _eval.FLOAT
        assert fv[0] == pytest.approx(np.pi)

    def test_lit_e_emits_float(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["e"], self.sl)
        assert tok[0] == _eval.FLOAT
        assert fv[0] == pytest.approx(np.e)

    def test_numeric_float_token_emits_float(self):
        tree = Node("1.5")
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, self.sl)
        assert tok[0] == _eval.FLOAT
        assert fv[0] == pytest.approx(1.5)

    def test_fn_with_cython_id_positive(self):
        tree = Node("sin", left=Node("X_0"))
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, self.sl)
        sin_idx = 1  # postfix: X_0, sin
        assert tok[sin_idx] == self.sl.get_cython_id("sin")
        assert ari[sin_idx] == 1
        assert po[sin_idx] is None

    def test_fn_with_negative_cython_id_emits_python(self):
        sl = _sl(2)
        sl.add_symbol("myfn", "fn", 5, "np.sin({})")
        tree = Node("myfn", left=Node("X_0"))
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, sl)
        assert tok[1] == _eval.PYTHON
        assert callable(po[1])

    def test_op_with_cython_id_positive(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["X_0", "+", "X_1"], self.sl)
        op_idx = 2  # postfix: X_0, X_1, +
        assert tok[op_idx] == self.sl.get_cython_id("+")
        assert ari[op_idx] == 2
        assert po[op_idx] is None

    def test_op_with_negative_cython_id_emits_python(self):
        sl = _sl(2)
        sl.add_symbol("myop", "op", 0, "{} + {}")
        tree = Node("myop", left=Node("X_0"), right=Node("X_1"))
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, sl)
        op_idx = 2
        assert tok[op_idx] == _eval.PYTHON
        assert callable(po[op_idx])

    def test_unknown_token_raises_value_error(self):
        tree = Node("??unknown??")
        with pytest.raises(ValueError, match="unknown type"):
            _prepare_cython_traversal(tree, self.sl)

    def test_list_input_works(self):
        tok, ari, ei, fv, po = _prepare_cython_traversal(["X_0", "+", "X_1"], self.sl)
        assert len(tok) == 3

    def test_node_input_works(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        tok, ari, ei, fv, po = _prepare_cython_traversal(tree, self.sl)
        assert len(tok) == 3


# ---------------------------------------------------------------------------
# _prepare_cython_traversal_partial
# ---------------------------------------------------------------------------


class TestPrepareCythonTraversalPartial:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0])

    def test_no_const_falls_through_to_regular(self):
        # Expression without C: has_const=False, falls through
        tok_partial, ari_partial, ei_partial, fv_partial, po_partial = _prepare_cython_traversal_partial(
            ["X_0", "+", "X_1"], self.sl, self.X
        )
        tok_regular, ari_regular, ei_regular, fv_regular, po_regular = _prepare_cython_traversal(
            ["X_0", "+", "X_1"], self.sl
        )
        np.testing.assert_array_equal(tok_partial, tok_regular)

    def test_pure_const_falls_through(self):
        # Expression is just ["C"]: has_const=True but const_free_nodes=[]
        tok_partial, _, _, _, _ = _prepare_cython_traversal_partial(["C"], self.sl, self.X)
        tok_regular, _, _, _, _ = _prepare_cython_traversal(["C"], self.sl)
        np.testing.assert_array_equal(tok_partial, tok_regular)

    def test_normal_case_x0_plus_c_caches_x0(self):
        # X_0 + C: X_0 subtree is pre-evaluated and replaced with CACHED
        tok, ari, ei, fv, po = _prepare_cython_traversal_partial(["X_0", "+", "C"], self.sl, self.X)
        # After caching, instructions should contain CACHED (not VAR)
        assert _eval.CACHED in tok

    def test_normal_case_result_is_correct(self):
        # Verify the cached partial traversal produces the same result as the regular one
        X = _X([2.0, 3.0], [4.0, 5.0])
        C = np.array([10.0])
        tok_p, ari_p, ei_p, fv_p, po_p = _prepare_cython_traversal_partial(["X_0", "+", "C"], self.sl, X)
        tok_r, ari_r, ei_r, fv_r, po_r = _prepare_cython_traversal(["X_0", "+", "C"], self.sl)
        from SRToolkit.utils import _eval_cython as ev

        result_partial = ev.execute(X, C, tok_p, ari_p, ei_p, fv_p, po_p)
        result_regular = ev.execute(X, C, tok_r, ari_r, ei_r, fv_r, po_r)
        np.testing.assert_allclose(result_partial, result_regular)


# ---------------------------------------------------------------------------
# _expr_to_cython_callable
# ---------------------------------------------------------------------------


class TestExprToCythonCallable:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])

    def test_basic_expression_correctness(self):
        f = _expr_to_cython_callable(["X_0", "+", "X_1"], self.sl)
        result = f(self.X, np.array([]))
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])

    def test_c_none_handled(self):
        f = _expr_to_cython_callable(["X_0", "+", "X_1"], self.sl)
        result = f(self.X, None)
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])

    def test_with_constant(self):
        f = _expr_to_cython_callable(["X_0", "+", "C"], self.sl)
        result = f(self.X, np.array([5.0]))
        np.testing.assert_allclose(result, [6.0, 8.0, 10.0])

    def test_node_input(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        f = _expr_to_cython_callable(tree, self.sl)
        result = f(self.X, None)
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])


# ---------------------------------------------------------------------------
# _expr_to_cython_error_callable
# ---------------------------------------------------------------------------


class TestExprToCythonErrorCallable:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [2.0, 3.0], [3.0, 4.0])

    def test_zero_rmse_perfect_prediction(self):
        f = _expr_to_cython_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([3.0, 5.0, 7.0])
        assert f(self.X, np.array([]), y) == pytest.approx(0.0)

    def test_nonzero_rmse(self):
        f = _expr_to_cython_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([0.0, 0.0, 0.0])
        assert f(self.X, np.array([]), y) > 0

    def test_x_provided_for_caching(self):
        f = _expr_to_cython_error_callable(["X_0", "+", "C"], self.sl, X=self.X)
        y = np.array([2.0, 3.0, 4.0])
        assert f(self.X, np.array([1.0]), y) == pytest.approx(0.0)

    def test_c_none_handled(self):
        f = _expr_to_cython_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([3.0, 5.0, 7.0])
        assert f(self.X, None, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _expr_to_python_callable
# ---------------------------------------------------------------------------


class TestExprToPythonCallable:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0])

    def test_basic_expression_correctness(self):
        f = _expr_to_python_callable(["X_0", "+", "X_1"], self.sl)
        result = f(self.X, np.array([]))
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_c_none_handled(self):
        f = _expr_to_python_callable(["X_0", "+", "X_1"], self.sl)
        result = f(self.X, None)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_with_constant(self):
        f = _expr_to_python_callable(["X_0", "+", "C"], self.sl)
        result = f(self.X, np.array([10.0]))
        np.testing.assert_allclose(result, [11.0, 13.0])

    def test_node_input(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        f = _expr_to_python_callable(tree, self.sl)
        result = f(self.X, None)
        np.testing.assert_allclose(result, [3.0, 7.0])


# ---------------------------------------------------------------------------
# _expr_to_python_stack_error_callable
# ---------------------------------------------------------------------------


class TestExprToPythonStackErrorCallable:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [2.0, 3.0], [3.0, 4.0])

    def test_zero_rmse(self):
        f = _expr_to_python_stack_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([3.0, 5.0, 7.0])
        assert f(self.X, np.array([]), y) == pytest.approx(0.0)

    def test_nonzero_rmse(self):
        f = _expr_to_python_stack_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([0.0, 0.0, 0.0])
        assert f(self.X, np.array([]), y) > 0

    def test_x_provided_for_caching(self):
        f = _expr_to_python_stack_error_callable(["X_0", "+", "C"], self.sl, X=self.X)
        y = np.array([2.0, 3.0, 4.0])
        assert f(self.X, np.array([1.0]), y) == pytest.approx(0.0)

    def test_c_none_handled(self):
        f = _expr_to_python_stack_error_callable(["X_0", "+", "X_1"], self.sl)
        y = np.array([3.0, 5.0, 7.0])
        assert f(self.X, None, y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _expr_to_executable_function (codegen backend)
# ---------------------------------------------------------------------------


class TestExprToExecutableFunction:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])

    def test_basic_arithmetic(self):
        f = _expr_to_executable_function(["X_0", "+", "X_1"], self.sl)
        result = f(self.X, np.array([]))
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])

    def test_standalone_literal_pi(self):
        f = _expr_to_executable_function(["pi"], self.sl)
        result = f(self.X, np.array([]))
        assert result.shape == (3,)
        np.testing.assert_allclose(result, np.full(3, np.pi))

    def test_constant_expression(self):
        f = _expr_to_executable_function(["C"], self.sl)
        result = f(self.X, np.array([3.0]))
        np.testing.assert_allclose(result, np.full(3, 3.0))

    def test_node_input(self):
        tree = tokens_to_tree(["X_0", "+", "X_1"], self.sl)
        f = _expr_to_executable_function(tree, self.sl)
        result = f(self.X, np.array([]))
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])

    def test_invalid_input_raises(self):
        with pytest.raises(Exception):
            _expr_to_executable_function(42, self.sl)

    def test_custom_preamble_used(self):
        sl = _sl(2)
        sl.preamble = ["import numpy as np"]
        f = _expr_to_executable_function(["X_0", "+", "X_1"], sl)
        result = f(self.X, np.array([]))
        np.testing.assert_allclose(result, [3.0, 7.0, 11.0])


# ---------------------------------------------------------------------------
# _expr_to_error_function (codegen backend)
# ---------------------------------------------------------------------------


class TestExprToErrorFunction:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])

    def test_zero_rmse(self):
        f = _expr_to_error_function(["X_0", "+", "X_1"], self.sl)
        y = np.array([3.0, 7.0, 11.0])
        assert float(f(self.X, np.array([]), y)) == pytest.approx(0.0)

    def test_nonzero_rmse(self):
        f = _expr_to_error_function(["X_0", "+", "X_1"], self.sl)
        y = np.array([0.0, 0.0, 0.0])
        assert float(f(self.X, np.array([]), y)) > 0

    def test_node_input(self):
        tree = tokens_to_tree(["X_0", "+", "X_1"], self.sl)
        f = _expr_to_error_function(tree, self.sl)
        y = np.array([3.0, 7.0, 11.0])
        assert float(f(self.X, np.array([]), y)) == pytest.approx(0.0)

    def test_invalid_input_raises(self):
        with pytest.raises(Exception):
            _expr_to_error_function(42, self.sl)


# ---------------------------------------------------------------------------
# compile_expr
# ---------------------------------------------------------------------------


class TestCompileExpr:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])

    def test_backend_stack(self):
        f = compile_expr(["X_0", "+", "X_1"], self.sl, backend="stack")
        np.testing.assert_allclose(f(self.X, None), [3.0, 7.0, 11.0])

    def test_backend_codegen(self):
        f = compile_expr(["X_0", "+", "X_1"], self.sl, backend="codegen")
        np.testing.assert_allclose(f(self.X, np.array([])), [3.0, 7.0, 11.0])

    def test_backend_stack_py(self):
        f = compile_expr(["X_0", "+", "X_1"], self.sl, backend="stack_py")
        np.testing.assert_allclose(f(self.X, None), [3.0, 7.0, 11.0])

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            compile_expr(["X_0"], self.sl, backend="unknown_backend")

    def test_node_input(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        f = compile_expr(tree, self.sl, backend="stack")
        np.testing.assert_allclose(f(self.X, None), [3.0, 7.0, 11.0])


# ---------------------------------------------------------------------------
# compile_expr_rmse
# ---------------------------------------------------------------------------


class TestCompileExprRmse:
    def setup_method(self):
        self.sl = _sl(2)
        self.X = _X([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
        self.y_exact = np.array([3.0, 7.0, 11.0])

    def test_backend_stack_zero_rmse(self):
        f = compile_expr_rmse(["X_0", "+", "X_1"], self.sl, backend="stack")
        assert f(self.X, np.array([]), self.y_exact) == pytest.approx(0.0)

    def test_backend_codegen_zero_rmse(self):
        f = compile_expr_rmse(["X_0", "+", "X_1"], self.sl, backend="codegen")
        assert float(f(self.X, np.array([]), self.y_exact)) == pytest.approx(0.0)

    def test_backend_stack_py_zero_rmse(self):
        f = compile_expr_rmse(["X_0", "+", "X_1"], self.sl, backend="stack_py")
        assert f(self.X, np.array([]), self.y_exact) == pytest.approx(0.0)

    def test_backend_stack_with_x_caching(self):
        f = compile_expr_rmse(["X_0", "+", "C"], self.sl, backend="stack", X=self.X)
        y = np.array([2.0, 4.0, 6.0])
        assert f(self.X, np.array([1.0]), y) == pytest.approx(0.0)

    def test_backend_stack_py_with_x_caching(self):
        f = compile_expr_rmse(["X_0", "+", "C"], self.sl, backend="stack_py", X=self.X)
        y = np.array([2.0, 4.0, 6.0])
        assert f(self.X, np.array([1.0]), y) == pytest.approx(0.0)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            compile_expr_rmse(["X_0"], self.sl, backend="bad_backend")

    def test_node_input(self):
        tree = Node("+", left=Node("X_0"), right=Node("X_1"))
        f = compile_expr_rmse(tree, self.sl, backend="stack")
        assert f(self.X, np.array([]), self.y_exact) == pytest.approx(0.0)
