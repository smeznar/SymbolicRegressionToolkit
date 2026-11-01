"""Specification tests for SRToolkit.utils.measures.

Expected values come from the docstrings (hand-computed, analytic, or from a trusted reference such
as scipy), never from the current output of the code. Every test states its reason: clause (a
documented behaviour), boundary (an edge of the documented input domain), or implied (a contract
users rely on without it being written). Known bugs are recorded as strict xfails, one per bug.
"""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import wasserstein_distance
from strategies import infix_expressions, matrix_pairs, regression_data

from SRToolkit.utils.expression_tree import ExpressionParseError, tokens_to_tree
from SRToolkit.utils.measures import (
    REGRESSION_METRICS,
    bed,
    create_behavior_matrix,
    edit_distance,
    mae,
    mse,
    nrmse,
    r2,
    rmse,
    tree_edit_distance,
)
from SRToolkit.utils.symbol_library import SymbolLibrary

FAST = settings(max_examples=25)


LIB = SymbolLibrary.default_symbols()
LIB30 = SymbolLibrary.default_symbols(num_variables=30)
NAN, INF = np.nan, np.inf
X3 = np.array([[0.0], [1.0], [3.0]])
X28 = np.arange(28.0).reshape(1, 28)


def tree(tokens):
    return tokens_to_tree(tokens, LIB)


def naive_levenshtein(a, b):
    """Reference Levenshtein distance (textbook dynamic programme)."""
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


def scipy_bed(m1, m2):
    """Reference BED: mean over rows of scipy's 1-D Wasserstein distance, after dropping nan."""
    return np.mean([wasserstein_distance(u[~np.isnan(u)], v[~np.isnan(v)]) for u, v in zip(m1, m2)])


def constant_predictor(values):
    values = np.asarray(values, dtype=float)
    return lambda X: values


# ============================================================================================
# edit_distance
# ============================================================================================


class TestEditDistance:
    @pytest.mark.parametrize(
        "a, b, notation, expected",
        [
            # postfix X_0 1 +  vs  X_0 1 -
            (["X_0", "+", "1"], ["X_0", "-", "1"], "postfix", 1),
            # postfix X_0 X_1 sin *  vs  X_1 sin X_0 *: delete and insert X_0 (substitutions alone cost 3)
            (["X_0", "*", "sin", "(", "X_1", ")"], ["sin", "(", "X_1", ")", "*", "X_0"], "postfix", 2),
            (["sin", "(", "X_0", "*", "C", ")"], ["sin", "(", "X_0", "*", "C", ")"], "postfix", 0),
            (["(", "(", "X_0", ")", ")", "+", "1"], ["X_0", "+", "1"], "infix", 0),
            (["(", "(", "X_0", ")", ")", "+", "1"], ["X_0", "+", "1"], "prefix", 0),
            (["(", "(", "X_0", ")", ")", "+", "1"], ["X_0", "+", "1"], "postfix", 0),
            # (X_0+X_1)*X_2 vs X_0*X_2: postfix/prefix delete X_1 and +; infix also ( and )
            (["(", "X_0", "+", "X_1", ")", "*", "X_2"], ["X_0", "*", "X_2"], "postfix", 2),
            (["(", "X_0", "+", "X_1", ")", "*", "X_2"], ["X_0", "*", "X_2"], "prefix", 2),
            (["(", "X_0", "+", "X_1", ")", "*", "X_2"], ["X_0", "*", "X_2"], "infix", 4),
            # only the parentheses needed to serialise count: X_0 + X_1 * X_2 has none (delete * and X_2)
            (["X_0", "+", "X_1", "*", "X_2"], ["X_0", "+", "X_1"], "infix", 2),
            # both Nodes are serialised in the comparison notation
            (tree(["X_0", "+", "1"]), tree(["X_0", "-", "1"]), "postfix", 1),
            # a Node is serialised with the symbol library (infix without it adds parentheses)
            (tree(["X_0", "+", "X_1", "*", "X_2"]), ["X_0", "+", "X_1", "*", "X_2"], "infix", 0),
            (["X_0", "+", "X_1", "*", "X_2"], tree(["X_0", "+", "X_1", "*", "X_2"]), "infix", 0),
        ],
        ids=[
            "substitution",
            "deletion-plus-insertion",
            "identical",
            "redundant-parentheses-infix",
            "redundant-parentheses-prefix",
            "redundant-parentheses-postfix",
            "notation-postfix",
            "notation-prefix",
            "notation-infix-counts-parentheses",
            "infix-counts-only-needed-parentheses",
            "node-vs-node",
            "node-first-infix",
            "node-second-infix",
        ],
    )
    def test_hand_computed_distances(self, a, b, notation, expected):
        # Reason: clause (unit costs; Node or list; parentheses ignored; comparison notation).
        assert edit_distance(a, b, notation=notation) == expected

    @FAST
    @given(a=infix_expressions(), b=infix_expressions(), notation=st.sampled_from(["infix", "prefix", "postfix"]))
    def test_equals_levenshtein_of_serialised_sequences(self, a, b, notation):
        # Reason: clause (Levenshtein distance of the serialised sequences; implies symmetry, 0 iff equal).
        expected = naive_levenshtein(tree(a).to_list(LIB, notation=notation), tree(b).to_list(LIB, notation=notation))
        assert edit_distance(a, b, notation=notation) == expected

    def test_unknown_notation_raises_value_error(self):
        # Reason: clause (Raises ValueError for an unknown notation).
        with pytest.raises(ValueError, match="notation") as exc:
            edit_distance(["X_0"], ["X_0"], notation="polish")
        assert not isinstance(exc.value, ExpressionParseError)


# ============================================================================================
# tree_edit_distance
# ============================================================================================


class TestTreeEditDistance:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (["X_0", "+", "1"], ["X_0", "-", "1"], 1),
            (["X_0", "+", "1"], ["X_0"], 2),
            (["X_0", "-", "X_1"], ["X_1", "-", "X_0"], 2),
            (["X_0", "+", "X_1"], ["X_1", "+", "X_0"], 2),
            (["(", "(", "X_0", ")", ")"], ["X_0"], 0),
            (tree(["X_0", "+", "1"]), ["X_0", "-", "1"], 1),
            (["X_0", "+", "1"], tree(["X_0", "-", "1"]), 1),
        ],
        ids=[
            "relabel",
            "delete-two-nodes",
            "ordered-children",
            "no-commutativity",
            "parentheses",
            "node-first",
            "node-second",
        ],
    )
    def test_hand_computed_distances(self, a, b, expected):
        # Reason: clause (unit costs; ordered children; parentheses play no role; Node or list).
        assert tree_edit_distance(a, b) == expected

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (["sin", "(", "X_0", ")"], ["cos", "(", "X_0", ")"], 1),
            (["exp", "(", "X_0", ")"], ["sqrt", "(", "X_0", ")"], 1),
            (["X_0"], ["sin", "(", "X_0", ")"], 1),
            (["(", "X_0", "+", "X_1", ")", "*", "X_2"], ["X_0", "*", "X_2"], 2),
        ],
        ids=["relabel-sin-cos", "relabel-exp-sqrt", "insert-sin", "delete-operator-and-leaf"],
    )
    def test_every_edit_costs_one(self, a, b, expected):
        # Reason: clause (each insertion, deletion or relabelling costs 1, for any label length).
        assert tree_edit_distance(a, b) == expected

    @FAST
    @given(a=infix_expressions(), b=infix_expressions())
    def test_is_symmetric_and_zero_iff_trees_equal(self, a, b):
        # Reason: clause (symmetric; 0 iff same symbols in the same ordered structure).
        d = tree_edit_distance(a, b)
        assert d == tree_edit_distance(b, a)
        assert (d == 0) == (tree(a).to_list(LIB, notation="prefix") == tree(b).to_list(LIB, notation="prefix"))


# ============================================================================================
# Contracts shared by edit_distance, tree_edit_distance, create_behavior_matrix and bed
# ============================================================================================

CALLS_BY_POSITION = {
    "edit_distance-expr1": (lambda e: edit_distance(e, ["X_0"]), "expr1"),
    "edit_distance-expr2": (lambda e: edit_distance(["X_0"], e), "expr2"),
    "tree_edit_distance-expr1": (lambda e: tree_edit_distance(e, ["X_0"]), "expr1"),
    "tree_edit_distance-expr2": (lambda e: tree_edit_distance(["X_0"], e), "expr2"),
    "create_behavior_matrix": (lambda e: create_behavior_matrix(e, np.zeros((2, 1))), "expr"),
    "bed-expr1": (lambda e: bed(e, ["X_0"], X3), "expr1"),
    "bed-expr2": (lambda e: bed(["X_0"], e, X3), "expr2"),
}

USES_X27 = {
    "edit_distance": (lambda sl: edit_distance(["X_27"], ["X_27"], symbol_library=sl), 0),
    "tree_edit_distance": (lambda sl: tree_edit_distance(["X_27"], ["X_27"], symbol_library=sl), 0),
    "create_behavior_matrix": (lambda sl: create_behavior_matrix(["X_27"], X28, symbol_library=sl)[0, 0], 27.0),
    "bed": (lambda sl: bed(["X_27"], ["X_27"], X28, symbol_library=sl), 0.0),
}


def call_with_explicit_library(call):
    return call(LIB30)


def call_in_library_context(call):
    with LIB30:
        return call(None)


def call_with_module_default(call):
    SymbolLibrary.set_default(LIB30)
    return call(None)


class TestSharedExpressionContracts:
    @pytest.mark.parametrize("call, name", list(CALLS_BY_POSITION.values()), ids=list(CALLS_BY_POSITION))
    @pytest.mark.parametrize(
        "bad", ["X_0+1", ("X_0", "+", "1"), ["X_0", "+", 1]], ids=["string", "tuple", "list-with-int"]
    )
    def test_unsupported_expression_type_raises_type_error(self, call, name, bad):
        # Reason: clause (TypeError for anything but a list of token strings or a Node).
        with pytest.raises(TypeError, match=name):
            call(bad)

    @pytest.mark.parametrize("call, name", list(CALLS_BY_POSITION.values()), ids=list(CALLS_BY_POSITION))
    @pytest.mark.parametrize("bad", [[], ["X_0", "+"], ["foo"]], ids=["empty", "malformed", "unknown-symbol"])
    def test_unparsable_expression_raises_parse_error(self, call, name, bad):
        # Reason: clause (ExpressionParseError for an empty, malformed or unknown-symbol token list).
        with pytest.raises(ExpressionParseError):
            call(bad)

    @pytest.mark.parametrize("call, expected", list(USES_X27.values()), ids=list(USES_X27))
    @pytest.mark.parametrize(
        "resolve",
        [call_with_explicit_library, call_in_library_context, call_with_module_default],
        ids=["explicit", "context", "module-default"],
    )
    def test_symbol_library_is_used(self, call, expected, resolve):
        # Reason: clause (explicit library, else active context, else set_default).
        assert resolve(call) == expected

    @pytest.mark.parametrize("call, expected", list(USES_X27.values()), ids=list(USES_X27))
    def test_builtin_default_library_lacks_x27(self, call, expected):
        # Reason: clause (last fallback is default_symbols(), which has X_0..X_24).
        with pytest.raises(ExpressionParseError):
            call(None)

    @pytest.mark.parametrize(
        "call",
        [
            lambda e: create_behavior_matrix(e, X3, seed=5),
            lambda e: bed(e, ["sin", "(", "X_0", ")"], X3, seed=5),
            lambda e: bed(e, ["sin", "(", "X_0", ")"], domain_bounds=[(0, 1)], seed=5),
        ],
        ids=["create_behavior_matrix", "bed", "bed-sampled-points"],
    )
    def test_node_input_equals_list_input(self, call):
        # Reason: clause (an expression is a token list or a Node tree).
        tokens = ["X_0", "*", "C", "+", "C"]
        np.testing.assert_array_equal(call(tree(tokens)), call(tokens))

    @pytest.mark.parametrize(
        "call, match",
        [
            (lambda: create_behavior_matrix(["X_0", "+", "X_1"], np.zeros((3, 1))), "X"),
            (lambda: bed(["X_0"], ["X_1"], np.zeros((3, 1))), "X"),
            (lambda: bed(["X_0"], ["X_1"], domain_bounds=[(0, 1)]), "domain_bounds"),
        ],
        ids=["create_behavior_matrix-X", "bed-X", "bed-domain_bounds"],
    )
    def test_missing_column_for_used_variable_raises_value_error(self, call, match):
        # Reason: clause (ValueError if X / domain_bounds lacks a column for a variable used).
        with pytest.raises(ValueError, match=match) as exc:
            call()
        assert not isinstance(exc.value, ExpressionParseError)


# ============================================================================================
# create_behavior_matrix
# ============================================================================================


class TestCreateBehaviorMatrix:
    @pytest.mark.parametrize("n, k", [(5, 3), (1, 1), (0, 4)], ids=["5x3", "1x1", "no-points"])
    def test_shape_is_points_by_constant_samples(self, n, k):
        # Reason: clause, boundary (shape (n_samples, num_consts_sampled); n_samples may be 0).
        X = np.zeros((n, 1))
        assert create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=k, seed=0).shape == (n, k)

    def test_constant_free_columns_equal_direct_evaluation(self):
        # Reason: clause (constant-free gives identical columns; X_i reads column i).
        X = np.random.default_rng(0).normal(size=(6, 3))
        B = create_behavior_matrix(["X_0", "-", "X_1"], X, num_consts_sampled=4)
        np.testing.assert_allclose(B, np.repeat((X[:, 0] - X[:, 1])[:, None], 4, axis=1))

    @pytest.mark.parametrize("lower, upper", [(0, 1), (1, 5)], ids=["bounds-0-1", "bounds-1-5"])
    def test_additive_constant_shifts_each_column_within_bounds(self, lower, upper):
        # Reason: clause (column j uses constant vector j; constants in [lower, upper)).
        X = np.array([[0.0], [1.0], [2.5]])
        B = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=16, consts_bounds=(lower, upper), seed=1)
        offsets = B - X
        np.testing.assert_allclose(offsets, np.broadcast_to(offsets[0], offsets.shape))
        assert np.all((offsets[0] >= lower) & (offsets[0] < upper))

    def test_each_constant_occurrence_is_a_separate_stratified_parameter(self):
        # Reason: clause (each C is a free parameter; one LHS sample of k points per dimension).
        # C + C * X_0: at X_0 = 0 the value is c_a, at X_0 = 1 it is c_a + c_b; 10 samples in [0, 10).
        B = create_behavior_matrix(
            ["C", "+", "C", "*", "X_0"], np.array([[0.0], [1.0]]), num_consts_sampled=10, consts_bounds=(0, 10), seed=4
        )
        c_a, c_b = B[0], B[1] - B[0]
        assert not np.allclose(c_a, c_b)
        np.testing.assert_array_equal(np.sort(np.floor(c_a)), np.arange(10))
        np.testing.assert_array_equal(np.sort(np.floor(c_b + 1e-9)), np.arange(10))

    def test_seed_controls_sampling(self):
        # Reason: clause (sampling is random unless seed is set).
        X = np.array([[1.0]])
        np.testing.assert_array_equal(
            create_behavior_matrix(["C"], X, seed=7), create_behavior_matrix(["C"], X, seed=7)
        )
        assert not np.array_equal(create_behavior_matrix(["C"], X, seed=0), create_behavior_matrix(["C"], X, seed=1))
        assert not np.array_equal(create_behavior_matrix(["C"], X), create_behavior_matrix(["C"], X))

    @pytest.mark.parametrize("backend", ["stack_py", "codegen"])
    def test_all_backends_give_the_same_matrix(self, backend):
        # Reason: clause (backend: all give the same values). Differential against the default backend.
        X = np.random.default_rng(3).random((6, 2))
        expr = ["C", "*", "sin", "(", "X_0", "*", "C", ")", "+", "X_1", "^2"]
        np.testing.assert_allclose(
            create_behavior_matrix(expr, X, num_consts_sampled=5, seed=2, backend=backend),
            create_behavior_matrix(expr, X, num_consts_sampled=5, seed=2),
            rtol=1e-12,
        )

    def test_non_finite_values_kept_silently_and_settings_restored(self):
        # Reason: clause (non-finite outputs kept; warnings suppressed; error settings restored),
        # under the strictest settings. Covers divide, invalid, overflow and underflow; bed too.
        X = np.array([[0.0, 1000.0], [-1.0, -1000.0]])
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            before = np.geterr()
            inv = create_behavior_matrix(["1", "/", "X_0"], X, num_consts_sampled=2)
            log = create_behavior_matrix(["ln", "(", "X_0", ")"], X, num_consts_sampled=2)
            ex = create_behavior_matrix(["exp", "(", "X_1", ")"], X, num_consts_sampled=2)
            bed(["1", "/", "X_0"], ["X_0"], X)
            assert np.geterr() == before
        np.testing.assert_array_equal(inv, [[INF, INF], [-1.0, -1.0]])
        np.testing.assert_array_equal(log, [[-INF, -INF], [NAN, NAN]])
        np.testing.assert_array_equal(ex, [[INF, INF], [0.0, 0.0]])

    @pytest.mark.parametrize(
        "expr, X, kwargs, match",
        [
            (["X_0", "+", "C"], np.zeros((2, 1)), {"num_consts_sampled": 0}, "num_consts_sampled"),
            (["X_0", "+", "C"], np.zeros((2, 1)), {"num_consts_sampled": -1}, "num_consts_sampled"),
            (["X_0", "+", "C"], np.zeros((2, 1)), {"consts_bounds": (1, 1)}, "consts_bounds"),
            (["X_0", "+", "C"], np.zeros((2, 1)), {"consts_bounds": (5, -5)}, "consts_bounds"),
            (["X_0"], np.zeros((2, 1)), {"num_consts_sampled": 0}, "num_consts_sampled"),
            (["X_0"], np.zeros((2, 1)), {"consts_bounds": (1, 1)}, "consts_bounds"),
            (["X_0"], np.array([0.0, 1.0]), {}, "X"),
            (["X_0"], np.zeros((2, 1)), {"backend": "fortran"}, "backend"),
        ],
        ids=[
            "zero-samples",
            "negative-samples",
            "empty-bounds",
            "reversed-bounds",
            "zero-samples-constant-free",
            "empty-bounds-constant-free",
            "one-dimensional-X",
            "unknown-backend",
        ],
    )
    def test_invalid_arguments_raise_value_error(self, expr, X, kwargs, match):
        # Reason: clause (Raises ValueError, message names the parameter).
        with pytest.raises(ValueError, match=match) as exc:
            create_behavior_matrix(expr, X, **kwargs)
        assert not isinstance(exc.value, ExpressionParseError)


# ============================================================================================
# bed
# ============================================================================================


class TestBedValues:
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (["X_0"], ["X_0", "+", "1"], 1.0),
            (["X_0"], ["2", "*", "X_0"], 4 / 3),
            (["X_0"], ["X_0"], 0.0),
        ],
        ids=["unit-shift", "doubling-mean-abs-x", "identical"],
    )
    def test_hand_computed_values_at_given_points(self, a, b, expected):
        # Reason: clause (mean over points of W1). Constant-free rows are point masses, so
        # W1 = |a(x) - b(x)|, averaged over x in {0, 1, 3}.
        assert bed(a, b, X3) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "m1, m2, expected",
        [
            ([[0.0, 2.0]], [[1.0, 1.0]], 1.0),
            ([[0.0, 2.0], [0.0, 0.0]], [[1.0, 1.0], [3.0, 3.0]], 2.0),
        ],
        ids=["one-row", "two-rows"],
    )
    def test_hand_computed_values_for_matrices(self, m1, m2, expected):
        # Reason: clause (precomputed matrices). W1([0,2],[1,1]) = 1; W1([0,0],[3,3]) = 3; mean 2.
        assert bed(np.array(m1), np.array(m2)) == pytest.approx(expected)

    @FAST
    @given(pair=matrix_pairs())
    def test_matches_scipy(self, pair):
        # Reason: clause (rows are empirical distributions compared with the 1-D Wasserstein distance).
        m1, m2 = pair
        assert bed(m1, m2) == pytest.approx(scipy_bed(m1, m2), rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize(
        "m1, m2, expected",
        [([[0.0, 2.0]], [[1.0]], 1.0), ([[0.0, 2.0], [0.0, 0.0]], [[1.0], [3.0]], 2.0)],
        ids=["one-row", "two-rows"],
    )
    def test_matrices_of_different_widths(self, m1, m2, expected):
        # Reason: clause (a matrix may have a different number of columns).
        assert bed(np.array(m1), np.array(m2)) == pytest.approx(expected)

    def test_wide_matrices_processed_in_chunks_match_scipy(self):
        # Reason: implied (the result does not depend on matrix size). k = 3000 forces chunks of
        # 2 rows: 7 rows give three full chunks and a partial one.
        rng = np.random.default_rng(0)
        m1, m2 = rng.normal(size=(7, 3000)), rng.normal(0.5, 2.0, size=(7, 3000))
        assert bed(m1, m2) == pytest.approx(scipy_bed(m1, m2), rel=1e-9)


class TestBedModes:
    def test_matrix_argument_computed_at_x(self):
        # Reason: clause (a matrix argument is assumed to be computed at X).
        X = np.linspace(-1, 1, 7).reshape(-1, 1)
        assert bed(["X_0", "+", "1"], create_behavior_matrix(["X_0"], X), X) == pytest.approx(1.0)
        assert bed(["X_0", "*", "C"], create_behavior_matrix(["X_0", "*", "C"], X, seed=2), X, seed=2) == 0.0

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((["X_0"], ["X_0", "+", "1"]), {"X": X3, "domain_bounds": [(100, 200)]}),
            ((["X_0"], ["X_0", "+", "1"]), {"X": X3, "domain_bounds": [(1, 1)]}),
            ((["X_0"], ["X_0", "+", "1"]), {"X": X3, "num_points_sampled": 0}),
            ((np.array([[0.0]]), np.array([[1.0]])), {"num_consts_sampled": 0}),
            ((np.array([[0.0]]), np.array([[1.0]])), {"consts_bounds": (1, 1)}),
        ],
        ids=[
            "domain-ignored-with-X",
            "invalid-domain-with-X",
            "zero-points-with-X",
            "zero-constant-samples-two-matrices",
            "empty-consts-interval-two-matrices",
        ],
    )
    def test_unused_parameters_are_not_validated(self, args, kwargs):
        # Reason: clause (parameters are validated only when used). Both cases are a unit shift.
        assert bed(*args, **kwargs) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "var, domain, n, low, high",
        [
            ("X_0", [(0, 1)], 4, 3 / 8, 5 / 8),
            ("X_0", [(2, 3)], 64, 2 + 63 / 128, 2 + 65 / 128),
            ("X_0", [(0, 1)], 1, 0.0, 1.0),
            ("X_0", [(0, 4)], 4, 1.5, 2.5),
            ("X_1", [(0, 1), (10, 20)], 4, 13.75, 16.25),
        ],
        ids=["unit-4-points", "shifted-64-points", "single-point", "width-4", "second-variable"],
    )
    def test_points_are_latin_hypercube_samples_of_domain(self, var, domain, n, low, high):
        # Reason: clause (num_points_sampled points drawn from domain_bounds by LHS).
        # bed(x, 2x) = mean |x| over the points; an LHS of n points in [l, u) has mean within
        # (u - l) / (2n) of the midpoint.
        value = bed([var], ["2", "*", var], domain_bounds=domain, num_points_sampled=n, seed=0)
        assert low <= value <= high

    @pytest.mark.parametrize(
        "bounds, k, low, high",
        [((1, 5), 4, 2.5, 3.5), ((0, 1), 4, 3 / 8, 5 / 8), ((1, 5), 1, 1.0, 5.0)],
        ids=["bounds-1-5", "bounds-0-1", "single-sample"],
    )
    def test_constants_follow_consts_bounds_and_num_consts_sampled(self, bounds, k, low, high):
        # Reason: clause (consts_bounds, num_consts_sampled). W1 from k LHS samples of C to the
        # point mass 0 is their mean, within (u - l) / (2k) of the midpoint.
        value = bed(["C"], ["0"], np.zeros((1, 1)), consts_bounds=bounds, num_consts_sampled=k, seed=0)
        assert low <= value <= high

    def test_seeding(self):
        # Reason: clause (with seed: deterministic, symmetric, bed(e, e) == 0; seed=None: random).
        a, b = ["C", "*", "X_0"], ["C", "+", "X_0"]
        kwargs = {"domain_bounds": [(-1, 1)], "num_points_sampled": 16, "num_consts_sampled": 8, "seed": 3}
        value = bed(a, b, **kwargs)
        assert value > 0
        assert bed(a, b, **kwargs) == value
        assert bed(b, a, **kwargs) == pytest.approx(value)
        assert bed(a, a, **kwargs) == 0.0
        assert bed(a, a, X3, seed=3) == 0.0
        assert bed(a, a, X3, num_consts_sampled=4) != bed(a, a, X3, num_consts_sampled=4)


class TestBedNonFiniteAndErrors:
    def test_nan_entries_are_dropped_before_wasserstein(self):
        # Reason: clause (steps 1 and 4). Integer values create ties, so CDF steps are exercised.
        rng = np.random.default_rng(1)
        m1 = rng.integers(0, 5, size=(6, 8)).astype(float)
        m2 = rng.integers(0, 7, size=(6, 8)).astype(float)
        m1[rng.random(m1.shape) < 0.3] = NAN
        m1[:, 0] = 0.0  # every row keeps at least one finite value
        assert bed(m1, m2) == pytest.approx(scipy_bed(m1, m2), rel=1e-12)

    @pytest.mark.parametrize(
        "m1, m2, expected",
        [
            ([[0.0, NAN, 2.0]], [[1.0, 1.0, 1.0]], 1.0),
            ([[NAN, 1.0], [0.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]], 0.0),
            ([[NAN, NAN], [0.0, 0.0]], [[NAN, NAN], [1.0, 1.0]], 0.5),
            ([[NAN, NAN]], [[1.0, 2.0]], INF),
            ([[NAN, NAN]], [[INF, INF]], INF),
            ([[INF, 1.0]], [[1.0, 2.0]], INF),
            ([[INF, INF]], [[INF, INF]], 0.0),
            ([[-INF, -INF]], [[-INF, -INF]], 0.0),
            ([[INF, INF]], [[-INF, -INF]], INF),
            ([[INF, INF]], [[INF, 1.0]], INF),
            ([[INF, INF], [0.0, 0.0]], [[INF, INF], [2.0, 2.0]], 1.0),
            ([[NAN, INF]], [[INF, INF]], 0.0),
            ([[INF, -INF]], [[INF, -INF]], INF),
        ],
        ids=[
            "nan-dropped",
            "nan-dropped-row-matches",
            "both-all-nan-counts-as-zero",
            "one-all-nan",
            "all-nan-vs-all-inf",
            "inf-vs-finite",
            "same-plus-inf",
            "same-minus-inf",
            "opposite-inf",
            "inf-vs-partly-inf",
            "same-inf-row-counts-as-zero",
            "nan-dropped-before-inf-rule",
            "mixed-infinities-in-both-rows",
        ],
    )
    def test_rows_follow_the_non_finite_steps(self, m1, m2, expected):
        # Reason: clause (Returns, steps 1-3; every point counts in the mean).
        assert bed(np.array(m1), np.array(m2)) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "args, kwargs, match",
        [
            ((["X_0"], ["X_0"]), {}, "domain_bounds"),
            ((np.zeros((3, 1)), ["X_0"]), {"domain_bounds": [(0, 1)]}, "X"),
            ((["X_0"], ["X_0"]), {"domain_bounds": [(1, 1)]}, "domain_bounds"),
            ((["X_0"], ["X_0"]), {"domain_bounds": [(0, 1)], "num_points_sampled": 0}, "num_points_sampled"),
            ((["C"], ["X_0"]), {"X": X3, "num_consts_sampled": 0}, "num_consts_sampled"),
            ((["C"], ["X_0"]), {"X": X3, "consts_bounds": (1, 1)}, "consts_bounds"),
            ((np.zeros((3, 2)), np.zeros((4, 2))), {}, "expr[12]"),
            ((np.zeros((4, 2)), ["X_0"]), {"X": X3}, "expr1"),
            ((np.zeros((3, 2)), np.zeros((3, 2))), {"X": np.zeros((4, 1))}, "expr1"),
            ((np.zeros((3, 2)), np.zeros((3, 2))), {"X": np.zeros(3)}, r"\bX\b"),
            ((np.zeros((0, 2)), np.zeros((0, 2))), {}, "expr[12]"),
            ((["X_0"], ["X_0"]), {"X": np.zeros(3)}, r"\bX\b"),
            ((["X_0"], ["X_0"]), {"X": np.zeros((0, 1))}, r"\bX\b|expr[12]"),
            ((["X_0"], ["X_0", "+", "1"]), {"X": X3, "num_consts_sampled": 0}, "num_consts_sampled"),
            ((["X_0"], ["X_0", "+", "1"]), {"X": X3, "consts_bounds": (1, 1)}, "consts_bounds"),
            ((np.zeros((3, 0)), np.zeros((3, 2))), {}, "expr1"),
            ((np.zeros(3), np.zeros((3, 1))), {}, "expr1"),
            ((["X_0"], ["X_0"]), {"X": X3, "backend": "fortran"}, "backend"),
        ],
        ids=[
            "no-X-no-domain",
            "no-X-one-matrix",
            "empty-domain-interval",
            "zero-points",
            "zero-constant-samples",
            "empty-consts-interval",
            "matrix-row-mismatch",
            "matrix-rows-differ-from-X",
            "two-matrices-rows-differ-from-X",
            "two-matrices-one-dimensional-X",
            "matrices-without-rows",
            "one-dimensional-X",
            "X-without-rows",
            "zero-constant-samples-no-constants",
            "empty-consts-interval-no-constants",
            "matrix-without-columns",
            "one-dimensional-matrix",
            "unknown-backend",
        ],
    )
    def test_invalid_arguments_raise_value_error(self, args, kwargs, match):
        # Reason: clause (Raises ValueError, message names the offending parameter).
        with pytest.raises(ValueError, match=match) as exc:
            bed(*args, **kwargs)
        assert not isinstance(exc.value, ExpressionParseError)


# ============================================================================================
# Regression metrics
# ============================================================================================

ALL_METRICS = [mse, rmse, mae, r2, nrmse]
METRIC_IDS = ["mse", "rmse", "mae", "r2", "nrmse"]
X4 = np.zeros((4, 1))
Y4 = np.array([1.0, 3.0, 5.0, 7.0])


class TestRegressionMetrics:
    # Errors [2, 0, 0, 0] against y = [1, 3, 5, 7] (given as a list): mse = 4/4 = 1, rmse = 1,
    # mae = 2/4 = 0.5; SS_tot = 9+1+1+9 = 20, SS_res = 4 -> r2 = 0.8; std(y) = sqrt(5) -> nrmse = 1/sqrt(5).
    @pytest.mark.parametrize(
        "metric, y_pred, y, expected",
        [
            (mse, [3.0, 3.0, 5.0, 7.0], [1, 3, 5, 7], 1.0),
            (rmse, [3.0, 3.0, 5.0, 7.0], [1, 3, 5, 7], 1.0),
            (mae, [3.0, 3.0, 5.0, 7.0], [1, 3, 5, 7], 0.5),
            (r2, [3.0, 3.0, 5.0, 7.0], [1, 3, 5, 7], 0.8),
            (nrmse, [3.0, 3.0, 5.0, 7.0], [1, 3, 5, 7], 1 / np.sqrt(5)),
            (mse, Y4, Y4, 0.0),
            (rmse, Y4, Y4, 0.0),
            (mae, Y4, Y4, 0.0),
            (r2, Y4, Y4, 1.0),
            (nrmse, Y4, Y4, 0.0),
            (r2, [4.0] * 4, Y4, 0.0),
            (nrmse, [4.0] * 4, Y4, 1.0),
        ],
        ids=[
            "mse-list-targets",
            "rmse-list-targets",
            "mae-list-targets",
            "r2-list-targets",
            "nrmse-list-targets",
            "mse-perfect-fit",
            "rmse-perfect-fit",
            "mae-perfect-fit",
            "r2-perfect-fit",
            "nrmse-perfect-fit",
            "r2-mean-predictor",
            "nrmse-mean-predictor",
        ],
    )
    def test_hand_computed_values(self, metric, y_pred, y, expected):
        # Reason: clause (documented formulas; Python float; y as any array-like).
        result = metric(constant_predictor(y_pred), X4, y)
        assert type(result) is float
        assert result == pytest.approx(expected)

    @FAST
    @given(data=regression_data())
    def test_match_numpy_formulas(self, data):
        # Reason: clause (documented formulas, on arbitrary finite data with non-constant y).
        X, y, y_pred = data
        predict = constant_predictor(y_pred)
        err = y_pred - y
        assert mse(predict, X, y) == pytest.approx(np.mean(err**2))
        assert rmse(predict, X, y) == pytest.approx(np.sqrt(np.mean(err**2)))
        assert mae(predict, X, y) == pytest.approx(np.mean(np.abs(err)))
        assert r2(predict, X, y) == pytest.approx(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2), rel=1e-9, abs=1e-9)
        assert nrmse(predict, X, y) == pytest.approx(np.sqrt(np.mean(err**2)) / np.std(y), rel=1e-9)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=METRIC_IDS)
    def test_predict_is_called_with_x(self, metric):
        # Reason: clause (predict is a PredictFn applied to X).
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        seen = []

        def predict(arg):
            seen.append(arg)
            return arg[:, 1] / 10

        metric(predict, X, np.array([1.0, 2.0, 4.0]))
        assert len(seen) >= 1
        assert all(np.array_equal(arg, X) for arg in seen)

    @pytest.mark.parametrize(
        "metric, y",
        [
            (mse, Y4),
            (rmse, Y4),
            (mae, Y4),
            (r2, Y4),
            (nrmse, Y4),
            (r2, np.full(4, 2.0)),
            (nrmse, np.full(4, 2.0)),
        ],
        ids=["mse", "rmse", "mae", "r2", "nrmse", "r2-constant-y", "nrmse-constant-y"],
    )
    def test_exceptions_from_predict_propagate(self, metric, y):
        # Reason: clause (exceptions raised by predict propagate).
        def predict(X):
            raise ZeroDivisionError("boom")

        with pytest.raises(ZeroDivisionError, match="boom"):
            metric(predict, X4, y)

    @pytest.mark.parametrize(
        "y_pred, y, expected",
        [
            ([], [], (NAN, NAN, NAN)),
            ([NAN, 1.0], [1.0, 1.0], (NAN, NAN, NAN)),
            ([1.0, 1.0], [NAN, 1.0], (NAN, NAN, NAN)),
            ([INF, 1.0], [INF, 1.0], (NAN, NAN, NAN)),
            ([NAN, INF], [1.0, 1.0], (NAN, NAN, NAN)),
            ([INF, 1.0], [1.0, 1.0], (INF, INF, INF)),
            ([-INF, 1.0], [1.0, 1.0], (INF, INF, INF)),
            ([1.0, 1.0], [INF, 1.0], (INF, INF, INF)),
            # errors [2e200, 0]: the square overflows (mse, rmse inf); mae = 1e200
            ([1e200, 0.0], [-1e200, 0.0], (INF, INF, 1e200)),
        ],
        ids=[
            "empty",
            "nan-pred",
            "nan-target",
            "inf-minus-inf",
            "nan-beats-inf",
            "inf-pred",
            "minus-inf-pred",
            "inf-target",
            "squared-overflow",
        ],
    )
    def test_error_metrics_special_values(self, y_pred, y, expected):
        # Reason: clause (mse, rmse, mae Returns: nan / inf rules).
        predict, X, y = constant_predictor(y_pred), np.zeros((len(y), 1)), np.array(y, dtype=float)
        np.testing.assert_array_equal((mse(predict, X, y), rmse(predict, X, y), mae(predict, X, y)), expected)

    @pytest.mark.parametrize(
        "y_pred, y, expected",
        [
            ([], [], NAN),
            ([1.0, 2.0], [1.0, INF], NAN),
            ([1.0, 2.0], [NAN, 2.0], NAN),
            ([NAN, 2.0], [1.0, 2.0], NAN),
            ([NAN, INF], [1.0, 2.0], NAN),
            ([NAN, 2.0], [2.0, 2.0], NAN),
            ([INF, 2.0], [1.0, 2.0], -INF),
            ([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 1.0),
            ([2.0, 2.0, 2.5], [2.0, 2.0, 2.0], -INF),
            ([INF, 2.0], [2.0, 2.0], -INF),
            ([3.0], [3.0], 1.0),
            ([4.0], [3.0], -INF),
            ([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 1.0),
            ([0.1, 0.1, 0.2], [0.1, 0.1, 0.1], -INF),
        ],
        ids=[
            "empty",
            "inf-target",
            "nan-target",
            "nan-pred",
            "nan-beats-inf-pred",
            "nan-pred-constant-y",
            "inf-pred",
            "constant-y-exact",
            "constant-y-inexact",
            "constant-y-inf-pred",
            "single-sample-exact",
            "single-sample-inexact",
            "inexact-mean-constant-y-exact",
            "inexact-mean-constant-y-inexact",
        ],
    )
    def test_r2_special_values(self, y_pred, y, expected):
        # Reason: clause (r2 special values, in order of precedence; constant means exactly equal).
        result = r2(constant_predictor(y_pred), np.zeros((len(y), 1)), np.array(y, dtype=float))
        assert result == pytest.approx(expected, nan_ok=True)

    @pytest.mark.parametrize(
        "y_pred, y, expected",
        [
            ([], [], NAN),
            ([1.0, 2.0], [1.0, INF], NAN),
            ([1.0, 2.0], [NAN, 2.0], NAN),
            ([NAN, 2.0], [1.0, 2.0], NAN),
            ([NAN, INF], [1.0, 2.0], NAN),
            ([NAN, 2.0], [2.0, 2.0], NAN),
            ([INF, 2.0], [1.0, 2.0], INF),
            ([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], 0.0),
            ([2.0, 2.0, 2.5], [2.0, 2.0, 2.0], INF),
            ([INF, 2.0], [2.0, 2.0], INF),
            ([3.0], [3.0], 0.0),
            ([4.0], [3.0], INF),
            ([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], 0.0),
            ([0.1, 0.1, 0.2], [0.1, 0.1, 0.1], INF),
        ],
        ids=[
            "empty",
            "inf-target",
            "nan-target",
            "nan-pred",
            "nan-beats-inf-pred",
            "nan-pred-constant-y",
            "inf-pred",
            "constant-y-exact",
            "constant-y-inexact",
            "constant-y-inf-pred",
            "single-sample-exact",
            "single-sample-inexact",
            "inexact-mean-constant-y-exact",
            "inexact-mean-constant-y-inexact",
        ],
    )
    def test_nrmse_special_values(self, y_pred, y, expected):
        # Reason: clause (nrmse special values, in order of precedence; constant means exactly equal).
        result = nrmse(constant_predictor(y_pred), np.zeros((len(y), 1)), np.array(y, dtype=float))
        assert result == pytest.approx(expected, nan_ok=True)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=METRIC_IDS)
    @pytest.mark.parametrize(
        "y_pred, y",
        [([NAN, 1.0], [1.0, 2.0]), ([INF, 1.0], [1.0, 2.0]), ([2.0, 3.0], [2.0, 2.0])],
        ids=["nan-pred", "inf-pred", "constant-y"],
    )
    def test_metrics_are_silent(self, metric, y_pred, y):
        # Reason: clause (warnings suppressed, error settings restored; special values are the only signal).
        X = np.zeros((len(y), 1))
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            before = np.geterr()
            metric(constant_predictor(y_pred), X, np.array(y))
            assert np.geterr() == before

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=METRIC_IDS)
    @pytest.mark.parametrize("y_pred, y", [([], []), ([INF, 1.0], [INF, 2.0])], ids=["empty", "inf-minus-inf"])
    def test_metrics_are_silent_on_degenerate_input(self, metric, y_pred, y):
        # Reason: clause (as above, for empty input and inf - inf).
        X = np.zeros((len(y), 1))
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            metric(constant_predictor(y_pred), X, np.array(y, dtype=float))

    @pytest.mark.parametrize(
        "metric, y_pred, y, match",
        [
            (mse, np.ones((4, 1)), np.ones(4), "predict"),
            (rmse, np.ones((4, 1)), np.ones(4), "predict"),
            (mae, np.ones((4, 1)), np.ones(4), "predict"),
            (r2, np.ones((4, 1)), np.ones(4), "predict"),
            (nrmse, np.ones((4, 1)), np.ones(4), "predict"),
            (mse, np.ones(3), np.ones(4), "predict"),
            (mse, np.float64(1.0), np.ones(4), "predict"),
            (mse, np.ones(4), np.ones((4, 1)), r"\by\b"),
            (mse, np.ones(4), np.ones(3), r"\by\b"),
            (mse, np.ones(3), np.ones(3), "predict"),
        ],
        ids=[
            "mse-column-predictions",
            "rmse-column-predictions",
            "mae-column-predictions",
            "r2-column-predictions",
            "nrmse-column-predictions",
            "short-predictions",
            "scalar-prediction",
            "column-targets",
            "short-targets",
            "both-differ-from-X",
        ],
    )
    def test_shape_mismatch_raises_value_error(self, metric, y_pred, y, match):
        # Reason: clause (ValueError if predict(X) or y is not 1-D of length n_samples).
        # (n, 1) - (n,) would otherwise broadcast silently to (n, n).
        with pytest.raises(ValueError, match=match):
            metric(lambda X: y_pred, X4, y)


class TestRegressionMetricsRegistry:
    def test_maps_names_to_builtin_metrics(self):
        # Reason: clause (keys mse, rmse, mae, r2, nrmse).
        assert dict(REGRESSION_METRICS) == {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "nrmse": nrmse}

    def test_cannot_be_modified(self):
        # Reason: clause (the mapping cannot be modified).
        with pytest.raises(TypeError):
            REGRESSION_METRICS["custom"] = mse
        with pytest.raises(TypeError):
            del REGRESSION_METRICS["mse"]
