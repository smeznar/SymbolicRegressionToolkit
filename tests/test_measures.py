import numpy as np
import pytest

from SRToolkit.utils.expression_tree import Node, tokens_to_tree
from SRToolkit.utils.measures import (
    _custom_wasserstein,
    _expr_to_zss,
    bed,
    create_behavior_matrix,
    edit_distance,
    tree_edit_distance,
)
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestEditDistance:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)

    def test_identical_expressions_return_zero(self):
        assert edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"], symbol_library=self.sl) == 0

    def test_one_token_different(self):
        assert edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"], symbol_library=self.sl) == 1

    def test_node_inputs(self):
        t1 = tokens_to_tree(["X_0", "+", "1"], self.sl)
        t2 = tokens_to_tree(["X_0", "-", "1"], self.sl)
        assert edit_distance(t1, t2, symbol_library=self.sl) == 1

    def test_mixed_node_and_list(self):
        t1 = tokens_to_tree(["X_0", "+", "1"], self.sl)
        assert edit_distance(t1, ["X_0", "+", "1"], symbol_library=self.sl) == 0

    def test_notation_parameter(self):
        # Both notations should give 0 for identical expressions
        assert edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"], notation="prefix", symbol_library=self.sl) == 0
        assert edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"], notation="postfix", symbol_library=self.sl) == 0

    def test_longer_expression_larger_distance(self):
        d1 = edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"], symbol_library=self.sl)
        d2 = edit_distance(["X_0", "+", "X_1", "+", "1"], ["X_0", "-", "X_1", "-", "1"], symbol_library=self.sl)
        assert d2 > d1


class TestExprToZss:
    def test_leaf_node_has_no_children(self):
        znode = _expr_to_zss(Node("X_0"))
        assert znode.label == "X_0"
        assert znode.children == []

    def test_unary_node_has_one_child(self):
        node = Node("sin", left=Node("X_0"))
        znode = _expr_to_zss(node)
        assert znode.label == "sin"
        assert len(znode.children) == 1
        assert znode.children[0].label == "X_0"

    def test_binary_node_has_two_children(self):
        node = Node("+", right=Node("X_0"), left=Node("1"))
        znode = _expr_to_zss(node)
        assert znode.label == "+"
        assert len(znode.children) == 2


class TestTreeEditDistance:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)

    def test_identical_expressions_return_zero(self):
        assert tree_edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"], symbol_library=self.sl) == 0

    def test_one_node_different(self):
        assert tree_edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"], symbol_library=self.sl) == 1

    def test_node_inputs(self):
        t1 = tokens_to_tree(["X_0", "+", "1"], self.sl)
        t2 = tokens_to_tree(["X_0", "-", "1"], self.sl)
        assert tree_edit_distance(t1, t2, symbol_library=self.sl) == 1

    def test_mixed_node_and_list(self):
        t1 = tokens_to_tree(["X_0", "+", "1"], self.sl)
        assert tree_edit_distance(t1, ["X_0", "+", "1"], symbol_library=self.sl) == 0


class TestCreateBehaviorMatrix:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)
        rng = np.random.default_rng(0)
        self.X = rng.random((10, 2)) - 0.5

    def test_output_shape(self):
        bm = create_behavior_matrix(["X_0", "+", "C"], self.X, num_consts_sampled=32, symbol_library=self.sl)
        assert bm.shape == (10, 32)

    def test_no_constant_columns_identical(self):
        bm = create_behavior_matrix(["X_0", "+", "X_1"], self.X, symbol_library=self.sl)
        assert np.all(bm == bm[:, [0]])

    def test_with_constant_columns_vary(self):
        bm = create_behavior_matrix(["X_0", "+", "C"], self.X, num_consts_sampled=32, symbol_library=self.sl, seed=0)
        assert not np.all(bm == bm[:, [0]])

    def test_consts_bounds_shifts_mean(self):
        bm_low = create_behavior_matrix(
            ["C"], self.X, num_consts_sampled=32, consts_bounds=(0, 1), symbol_library=self.sl, seed=0
        )
        bm_high = create_behavior_matrix(
            ["C"], self.X, num_consts_sampled=32, consts_bounds=(10, 20), symbol_library=self.sl, seed=0
        )
        assert np.mean(bm_low) < np.mean(bm_high)

    def test_seed_makes_reproducible(self):
        bm1 = create_behavior_matrix(["X_0", "+", "C"], self.X, num_consts_sampled=16, symbol_library=self.sl, seed=42)
        bm2 = create_behavior_matrix(["X_0", "+", "C"], self.X, num_consts_sampled=16, symbol_library=self.sl, seed=42)
        assert np.array_equal(bm1, bm2)

    def test_node_input_accepted(self):
        tree = tokens_to_tree(["X_0", "+", "X_1"], self.sl)
        bm = create_behavior_matrix(tree, self.X, symbol_library=self.sl)
        assert bm.shape[0] == 10

    def test_multiple_constants(self):
        # Two C tokens → LHS samples a 2-D constant vector per draw
        bm = create_behavior_matrix(
            ["C", "+", "C", "*", "X_0"], self.X, num_consts_sampled=16, symbol_library=self.sl, seed=0
        )
        assert bm.shape == (10, 16)

    def test_invalid_input_raises(self):
        with pytest.raises(Exception, match="list of strings"):
            create_behavior_matrix(42, self.X, symbol_library=self.sl)

    def test_none_symbol_library_uses_default(self):
        bm = create_behavior_matrix(["X_0", "+", "X_1"], self.X, symbol_library=None)
        assert bm.shape[0] == 10


class TestCustomWasserstein:
    def test_identical_arrays_return_zero(self):
        u = np.array([1.0, 2.0, 3.0])
        assert _custom_wasserstein(u, u) == 0.0

    def test_unit_shift_returns_one(self):
        u = np.array([0.0])
        v = np.array([1.0])
        assert _custom_wasserstein(u, v) == pytest.approx(1.0)

    def test_symmetry(self):
        rng = np.random.default_rng(7)
        u = rng.random(20)
        v = rng.random(20)
        assert _custom_wasserstein(u, v) == pytest.approx(_custom_wasserstein(v, u))

    def test_different_length_arrays(self):
        # CDF normalises by u.size and v.size separately; unequal lengths are a distinct path
        u = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 2.0])
        assert _custom_wasserstein(u, v) == pytest.approx(_custom_wasserstein(v, u))


class TestBed:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)
        rng = np.random.default_rng(1)
        self.X = rng.random((20, 2)) - 0.5

    def test_list_inputs_with_X_returns_float(self):
        result = bed(["X_0", "+", "C"], ["X_1", "+", "C"], self.X, symbol_library=self.sl, seed=0)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_identical_expressions_return_zero(self):
        result = bed(["X_0", "+", "X_1"], ["X_0", "+", "X_1"], self.X, symbol_library=self.sl)
        assert result == pytest.approx(0.0)

    def test_domain_bounds_replaces_X(self):
        result = bed(["X_0"], ["X_1"], domain_bounds=[(0, 1), (0, 1)], symbol_library=self.sl, seed=0)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_precomputed_matrices_accepted(self):
        bm1 = create_behavior_matrix(["X_0", "+", "C"], self.X, symbol_library=self.sl, seed=0)
        bm2 = create_behavior_matrix(["X_1", "+", "C"], self.X, symbol_library=self.sl, seed=0)
        result = bed(bm1, bm2, symbol_library=self.sl)
        assert isinstance(result, float)

    def test_node_inputs(self):
        t1 = tokens_to_tree(["X_0", "+", "C"], self.sl)
        t2 = tokens_to_tree(["X_1", "+", "C"], self.sl)
        result = bed(t1, t2, self.X, symbol_library=self.sl, seed=0)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_mixed_matrix_and_expression(self):
        bm1 = create_behavior_matrix(["X_0", "+", "C"], self.X, symbol_library=self.sl, seed=0)
        result = bed(bm1, ["X_1", "+", "C"], self.X, symbol_library=self.sl, seed=0)
        assert isinstance(result, float)

    def test_x_none_no_domain_bounds_raises(self):
        with pytest.raises(Exception, match="domain_bounds"):
            bed(["X_0"], ["X_1"], X=None, symbol_library=self.sl)

    def test_x_none_one_matrix_one_not_raises(self):
        bm = create_behavior_matrix(["X_0"], self.X, symbol_library=self.sl)
        with pytest.raises(Exception, match="behavior matrix"):
            bed(bm, ["X_1"], X=None, symbol_library=self.sl)

    def test_shape_mismatch_raises(self):
        bm1 = np.ones((10, 8))
        bm2 = np.ones((5, 8))
        with pytest.raises(ValueError, match="same number of rows"):
            bed(bm1, bm2, symbol_library=self.sl)

    def test_empty_matrix_raises(self):
        bm1 = np.ones((0, 8))
        bm2 = np.ones((0, 8))
        with pytest.raises(ValueError, match="at least one row"):
            bed(bm1, bm2, symbol_library=self.sl)

    def test_invalid_domain_bounds_raises(self):
        with pytest.raises(ValueError, match="lower bound"):
            bed(["X_0"], ["X_1"], domain_bounds=[(1, 0), (0, 1)], symbol_library=self.sl)

    def test_one_row_all_nonfinite_returns_inf(self):
        # Row 0: both finite → wasserstein = 0; Row 1: expr1 all-inf, expr2 finite → inf
        # mean([0, inf]) = inf
        bm1 = np.array([[1.0, 2.0], [np.inf, np.inf]])
        bm2 = np.array([[1.0, 2.0], [1.0, 2.0]])
        result = bed(bm1, bm2, symbol_library=self.sl)
        assert result == np.inf

    def test_both_rows_all_nonfinite_returns_zero(self):
        # Both rows entirely non-finite → both filtered to empty → wasserstein = 0 per row
        bm1 = np.array([[np.nan, np.nan], [np.inf, np.inf]])
        bm2 = np.array([[np.nan, np.nan], [np.nan, np.inf]])
        result = bed(bm1, bm2, symbol_library=self.sl)
        assert result == pytest.approx(0.0)
