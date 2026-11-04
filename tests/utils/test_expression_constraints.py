"""Spec tests for SRToolkit.utils.expression_constraints.

Expected values are derived from the docstrings (messages, amounts, saved forms), never from the
implementation. Grammar-counterpart property tests drive a derivation with Hypothesis-drawn
choices, so they use no global RNG.
"""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from strategies import infix_expressions

from SRToolkit.utils.expression_constraints import (
    ExpressionConstraint,
    MaxConstants,
    MaxSymbolCount,
    MaxTokens,
    MaxTreeDepth,
    MaxTreeSize,
    NoNesting,
    OutputBounds,
    RequiredVariables,
)
from SRToolkit.utils.expression_tree import ExpressionParseError, tokens_to_tree
from SRToolkit.utils.grammar import MaxDepth, MaxNodes
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, ModelResult

INF = float("inf")
NAN = float("nan")
PASS = CheckResult()

BAD_LIMITS = [-1, True, False, np.bool_(True), 2.0, np.float64(2), "3", None]
LIMIT_CLASSES = [
    pytest.param(MaxConstants, id="MaxConstants"),
    pytest.param(MaxTokens, id="MaxTokens"),
    pytest.param(MaxTreeDepth, id="MaxTreeDepth"),
    pytest.param(MaxTreeSize, id="MaxTreeSize"),
    pytest.param(lambda limit: MaxSymbolCount("X_0", limit), id="MaxSymbolCount"),
]


def lib(n):
    return SymbolLibrary.default_symbols(n)


def fail(message, amount):
    return CheckResult(passed=False, message=message, amount=amount)


def set_up(constraint, library):
    constraint.setup(library)
    return constraint


def assert_result(result, expected):
    """Equal field by field, and a failed built-in check reports its amount as a float."""
    assert result == expected
    assert type(result.amount) is float


def lib_with_extra_const():
    library = lib(1)
    library.add_symbol("K", "const", 5, "C[{}]")
    return library


def lib_vars_y_then_x():
    library = SymbolLibrary()
    library.add_symbol("y", "var", 5, "X[:, 0]")
    library.add_symbol("x", "var", 5, "X[:, 1]")
    library.add_symbol("C", "const", 5, "C[{}]")
    return library


def without(d, key):
    return {k: v for k, v in d.items() if k != key}


# ── User-defined constraints (module level, so from_dict can import them by path) ────────────


class _Forbid(ExpressionConstraint):
    """The ForbidSymbol example of the module docs; needs no library for token lists."""

    def __init__(self, symbol):
        self.symbol = symbol

    def pre_check(self, expr):
        tokens = expr if isinstance(expr, list) else expr.to_list(self.symbol_library)
        n = tokens.count(self.symbol)
        if n == 0:
            return CheckResult()
        return CheckResult(passed=False, message=f"uses {self.symbol!r}", amount=n)

    def to_dict(self):
        return {**super().to_dict(), "symbol": self.symbol}

    @classmethod
    def from_dict(cls, d):
        return cls(d["symbol"])


class _NoOverride(ExpressionConstraint):
    """Implements pre_check only; does not override post_check or from_dict."""

    def pre_check(self, expr):
        return CheckResult()


def _model(expr, parameters=None):
    return ModelResult(expr=list(expr), error=0.1, parameters=parameters)


# ── Protocol: unimplemented checks, setup ─────────────────────────────────────────────────────


class TestProtocol:
    @pytest.mark.parametrize(
        "make, method, arg, message",
        [
            (ExpressionConstraint, "pre_check", ["X_0"], "ExpressionConstraint implements no pre_check"),
            (ExpressionConstraint, "post_check", _model(["X_0"]), "ExpressionConstraint implements no post_check"),
            (_NoOverride, "post_check", _model(["X_0"]), "_NoOverride implements no post_check"),
            (
                lambda: OutputBounds([[0.0]], lower=0.0),
                "pre_check",
                ["X_0"],
                "OutputBounds implements no pre_check",
            ),
            (lambda: MaxConstants(1), "post_check", _model(["X_0"]), "MaxConstants implements no post_check"),
            (
                lambda: set_up(OutputBounds([[0.0]], lower=0.0), lib(1)),
                "pre_check",
                ["X_0"],
                "OutputBounds implements no pre_check",
            ),
            (
                lambda: set_up(MaxConstants(1), lib(1)),
                "post_check",
                _model(["X_0"]),
                "MaxConstants implements no post_check",
            ),
        ],
        ids=[
            "base-pre",
            "base-post",
            "subclass-post",
            "OutputBounds-pre-no-setup",
            "MaxConstants-post-no-setup",
            "OutputBounds-pre-after-setup",
            "MaxConstants-post-after-setup",
        ],
    )
    def test_unimplemented_checks_report_not_implemented(self, make, method, arg, message):
        result = getattr(make(), method)(arg)
        assert result == CheckResult(passed=True, message=message, implemented=False)

    def test_setup_stores_and_replaces_library(self):
        assert _Forbid("sin").symbol_library is None
        constraint = RequiredVariables()
        library1, library3 = lib(1), lib(3)
        constraint.setup(library1)
        assert constraint.symbol_library is library1
        constraint.setup(library3)
        assert constraint.symbol_library is library3
        assert_result(constraint.pre_check(["X_0"]), fail("missing X_1, X_2", 2.0))

    @pytest.mark.parametrize(
        "make, method, arg",
        [
            (lambda: MaxConstants(1), "pre_check", ["X_0"]),
            (lambda: MaxTokens(1), "pre_check", ["X_0"]),
            (lambda: MaxTreeDepth(1), "pre_check", ["X_0"]),
            (lambda: MaxTreeSize(1), "pre_check", ["X_0"]),
            (lambda: RequiredVariables(["X_0"]), "pre_check", ["X_0"]),
            (lambda: MaxSymbolCount("X_0", 1), "pre_check", ["X_0"]),
            (lambda: NoNesting("sin"), "pre_check", ["X_0"]),
            (lambda: OutputBounds([[0.0]], lower=0.0), "post_check", _model(["X_0"])),
            (lambda: ExpressionConstraint.from_dict(MaxTokens(3).to_dict()), "pre_check", ["X_0"]),
        ],
        ids=[
            "MaxConstants",
            "MaxTokens",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "MaxSymbolCount",
            "NoNesting",
            "OutputBounds",
            "rebuilt-by-from_dict",
        ],
    )
    def test_builtin_checks_raise_before_setup(self, make, method, arg):
        constraint = make()
        with pytest.raises(RuntimeError, match=type(constraint).__name__) as raised:
            getattr(constraint, method)(arg)
        # NotImplementedError subclasses RuntimeError; it must not count as the documented error.
        assert not isinstance(raised.value, NotImplementedError)

    def test_custom_constraint_works_without_setup(self):
        result = _Forbid("sin").pre_check(["sin", "(", "X_0", ")"])
        assert result == CheckResult(passed=False, message="uses 'sin'", amount=1)


# ── Shared contracts of the built-in pre-checks ──────────────────────────────────────────────


class TestSharedContracts:
    @pytest.mark.parametrize("make", LIMIT_CLASSES)
    @pytest.mark.parametrize("limit", BAD_LIMITS, ids=repr)
    def test_limit_constructors_reject_invalid_limits(self, request, make, limit):
        # The message names the constraint class (the parameter id) and the limit.
        name = request.node.callspec.id.split("-")[-1]
        with pytest.raises(ValueError, match=rf"(?s)(?=.*{name})(?=.*limit)"):
            make(limit)

    @pytest.mark.parametrize("make", LIMIT_CLASSES)
    @pytest.mark.parametrize("limit", [0, 5, np.int64(5), np.int32(5)], ids=["0", "5", "int64", "int32"])
    def test_limit_constructors_store_int(self, make, limit):
        constraint = make(limit)
        assert constraint.limit == limit
        assert type(constraint.limit) is int

    @pytest.mark.parametrize(
        "make, tokens, expected",
        [
            (lambda: MaxConstants(0), ["C", "C"], fail("2 constants, limit 0", 2.0)),
            (lambda: MaxTokens(1), ["X_0", "X_0"], fail("2 tokens, limit 1", 1.0)),
            (lambda: RequiredVariables(["X_0"]), ["X_0", "("], PASS),
            (lambda: MaxSymbolCount("X_0", 1), ["X_0", "X_0", ")"], fail("'X_0' occurs 2 times, limit 1", 1.0)),
        ],
        ids=["MaxConstants", "MaxTokens", "RequiredVariables", "MaxSymbolCount"],
    )
    def test_token_constraints_do_not_parse_malformed_lists(self, make, tokens, expected):
        assert_result(set_up(make(), lib(1)).pre_check(tokens), expected)

    @pytest.mark.parametrize(
        "make",
        [lambda: MaxTreeDepth(5), lambda: MaxTreeSize(5), lambda: NoNesting("sin")],
        ids=["MaxTreeDepth", "MaxTreeSize", "NoNesting"],
    )
    def test_tree_constraints_raise_parse_error_on_malformed_lists(self, make):
        constraint = set_up(make(), lib(1))
        with pytest.raises(ExpressionParseError):
            constraint.pre_check(["X_0", "X_0"])

    @pytest.mark.parametrize(
        "make",
        [
            lambda: MaxConstants(0),
            lambda: MaxTokens(1),
            lambda: MaxTreeDepth(1),
            lambda: MaxTreeSize(1),
            lambda: RequiredVariables(["X_0"]),
            lambda: MaxSymbolCount("X_0", 0),
            lambda: NoNesting(["sin", "+"]),
        ],
        ids=[
            "MaxConstants",
            "MaxTokens",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "MaxSymbolCount",
            "NoNesting",
        ],
    )
    def test_pre_check_does_not_modify_expr(self, make):
        library = lib(1)
        constraint = set_up(make(), library)
        tokens = ["sin", "(", "X_0", "+", "C", ")"]
        tree = tokens_to_tree(list(tokens), library)
        constraint.pre_check(tokens)
        constraint.pre_check(tree)
        assert tokens == ["sin", "(", "X_0", "+", "C", ")"]
        assert tree.to_list(library) == ["sin", "(", "X_0", "+", "C", ")"]

    def test_post_check_does_not_modify_model(self):
        constraint = set_up(OutputBounds([[-1.0], [2.0]], lower=0.0), lib(1))
        model = _model(["C", "*", "X_0"], parameters=np.array([2.0]))
        constraint.post_check(model)
        assert model.expr == ["C", "*", "X_0"]
        np.testing.assert_array_equal(model.parameters, [2.0])

    @settings(max_examples=60)
    @given(tokens=infix_expressions())
    @pytest.mark.parametrize(
        "make",
        [
            lambda: MaxConstants(1),
            lambda: MaxTreeDepth(3),
            lambda: MaxTreeSize(5),
            lambda: RequiredVariables(),
            lambda: NoNesting(["sin", "cos"]),
            lambda: MaxSymbolCount("X_0", 1),
        ],
        ids=["MaxConstants", "MaxTreeDepth", "MaxTreeSize", "RequiredVariables", "NoNesting", "MaxSymbolCount"],
    )
    def test_tree_and_token_list_give_same_result(self, make, tokens):
        library = lib(2)
        constraint = set_up(make(), library)
        assert constraint.pre_check(tokens) == constraint.pre_check(tokens_to_tree(tokens, library))

    @settings(max_examples=60)
    @given(tokens=infix_expressions())
    @pytest.mark.parametrize(
        "make, count, template",
        [
            (MaxConstants, lambda t: t.count("C"), "{n} constants, limit {limit}"),
            (MaxTokens, len, "{n} tokens, limit {limit}"),
            (MaxTreeSize, lambda t: len([s for s in t if s not in ("(", ")")]), "{n} nodes, limit {limit}"),
        ],
        ids=["MaxConstants", "MaxTokens", "MaxTreeSize"],
    )
    def test_counts_match_naive_counts(self, make, count, template, tokens):
        # infix_expressions brackets every subtree, so every non-bracket token is one tree node.
        n = count(tokens)
        library = lib(2)
        assert set_up(make(n), library).pre_check(tokens) == PASS
        if n > 0:
            expected = fail(template.format(n=n, limit=n - 1), 1.0)
            assert_result(set_up(make(n - 1), library).pre_check(tokens), expected)


# ── Per-constraint pre-check tables ──────────────────────────────────────────────────────────


class TestMaxConstants:
    @pytest.mark.parametrize(
        "limit, tokens, make_lib, expected",
        [
            (1, ["C", "*", "X_0", "+", "C"], lambda: lib(1), fail("2 constants, limit 1", 1.0)),
            (2, ["C", "*", "X_0", "+", "C"], lambda: lib(1), PASS),
            (0, ["C"], lambda: lib(1), fail("1 constants, limit 0", 1.0)),
            (0, ["pi", "*", "X_0", "+", "2"], lambda: lib(1), PASS),
            (1, ["K", "*", "C"], lib_with_extra_const, fail("2 constants, limit 1", 1.0)),
        ],
        ids=["over", "at-limit", "zero-limit", "literals-not-counted", "library-decides"],
    )
    def test_max_constants_pre_check_table(self, limit, tokens, make_lib, expected):
        assert_result(set_up(MaxConstants(limit), make_lib()).pre_check(tokens), expected)


class TestMaxTokens:
    @pytest.mark.parametrize(
        "limit, tokens, expected",
        [
            (3, ["(", "X_0", "+", "C", ")"], fail("5 tokens, limit 3", 2.0)),
            (3, ["X_0", "+", "C"], PASS),
            (0, ["X_0"], fail("1 tokens, limit 0", 1.0)),
        ],
        ids=["brackets-counted", "at-limit", "zero-limit"],
    )
    def test_max_tokens_pre_check_table(self, limit, tokens, expected):
        assert_result(set_up(MaxTokens(limit), lib(1)).pre_check(tokens), expected)

    @pytest.mark.parametrize(
        "limit, tokens, expected",
        [
            (3, ["(", "X_0", "+", "C", ")"], PASS),
            (6, ["(", "X_0", "+", "C", ")", "*", "X_0"], fail("7 tokens, limit 6", 1.0)),
            (5, ["X_0", "+", "C", "*", "X_0"], PASS),
        ],
        ids=["redundant-brackets-dropped", "needed-brackets-kept", "precedence-needs-no-brackets"],
    )
    def test_max_tokens_counts_tree_as_written_by_to_list(self, limit, tokens, expected):
        library = lib(1)
        assert_result(set_up(MaxTokens(limit), library).pre_check(tokens_to_tree(tokens, library)), expected)


class TestMaxTreeDepth:
    @pytest.mark.parametrize(
        "limit, tokens, expected",
        [
            (2, ["sin", "(", "X_0", "+", "C", ")"], fail("depth 3, limit 2", 1.0)),
            (2, ["X_0", "+", "C"], PASS),
            (0, ["X_0"], fail("depth 1, limit 0", 1.0)),
            (1, ["(", "(", "X_0", ")", ")"], PASS),
            (1, ["X_0", "+", "X_0", "+", "X_0"], fail("depth 3, limit 1", 2.0)),
        ],
        ids=["over", "at-limit", "leaf-is-height-1", "brackets-add-nothing", "binary-chain"],
    )
    def test_max_tree_depth_pre_check_table(self, limit, tokens, expected):
        assert_result(set_up(MaxTreeDepth(limit), lib(1)).pre_check(tokens), expected)


class TestMaxTreeSize:
    @pytest.mark.parametrize(
        "limit, tokens, expected",
        [
            (3, ["sin", "(", "X_0", "+", "C", ")"], fail("4 nodes, limit 3", 1.0)),
            (3, ["(", "X_0", "+", "C", ")"], PASS),
            (1, ["2", "*", "pi"], fail("3 nodes, limit 1", 2.0)),
            (0, ["X_0"], fail("1 nodes, limit 0", 1.0)),
        ],
        ids=["over", "brackets-not-nodes", "literals-and-numbers-are-nodes", "zero-limit"],
    )
    def test_max_tree_size_pre_check_table(self, limit, tokens, expected):
        assert_result(set_up(MaxTreeSize(limit), lib(1)).pre_check(tokens), expected)


class TestRequiredVariables:
    @pytest.mark.parametrize(
        "variables, expected",
        [
            (None, None),
            ("X_0", ("X_0",)),
            (["X_1", "X_0", "X_1"], ("X_1", "X_0")),
            ([], ()),
        ],
        ids=["none", "single-string", "duplicates-dropped", "empty"],
    )
    def test_required_variables_constructor_normalises(self, variables, expected):
        assert RequiredVariables(variables).variables == expected

    @pytest.mark.parametrize(
        "variables, make_lib, tokens, as_tree, expected",
        [
            (None, lambda: lib(3), ["X_0", "+", "X_2"], False, fail("missing X_1", 1.0)),
            (None, lambda: lib(3), ["C"], False, fail("missing X_0, X_1, X_2", 3.0)),
            (["X_2", "X_0"], lambda: lib(3), ["X_1"], False, fail("missing X_2, X_0", 2.0)),
            ([], lambda: lib(3), ["C"], False, PASS),
            (None, lib_vars_y_then_x, ["C"], False, fail("missing y, x", 2.0)),
            (None, lambda: lib(3), ["X_0", "+", "X_2"], True, fail("missing X_1", 1.0)),
        ],
        ids=[
            "all-library-vars",
            "several-missing-in-library-order",
            "order-of-variables-not-sorted",
            "empty-never-violated",
            "library-insertion-order",
            "tree-input",
        ],
    )
    def test_required_variables_pre_check_table(self, variables, make_lib, tokens, as_tree, expected):
        library = make_lib()
        expr = tokens_to_tree(tokens, library) if as_tree else tokens
        assert_result(set_up(RequiredVariables(variables), library).pre_check(expr), expected)

    @pytest.mark.parametrize(
        "variables, named",
        [(["X0"], "X0"), (["X_0", "sin"], "sin"), (["X_5"], "X_5")],
        ids=["typo", "not-a-variable", "variable-of-another-library"],
    )
    def test_required_variables_setup_rejects_unknown_tokens(self, variables, named):
        constraint = RequiredVariables(variables)
        with pytest.raises(ValueError, match=rf"(?s)(?=.*RequiredVariables)(?=.*{named})(?=.*X_0, X_1, X_2)"):
            constraint.setup(lib(3))
        assert constraint.symbol_library is None


class TestMaxSymbolCount:
    @pytest.mark.parametrize(
        "symbol, limit, tokens, expected",
        [
            ("X_0", 1, ["X_0", "*", "X_0"], fail("'X_0' occurs 2 times, limit 1", 1.0)),
            ("X_0", 2, ["X_0", "*", "X_0"], PASS),
            ("X_0", np.int64(1), ["X_0", "*", "X_0"], fail("'X_0' occurs 2 times, limit 1", 1.0)),
            ("sin", 0, ["X_0", "+", "C"], PASS),
            ("sin", 0, ["sin", "(", "X_0", ")"], fail("'sin' occurs 1 times, limit 0", 1.0)),
            ("(", 0, ["(", "(", "X_0", ")", ")"], fail("'(' occurs 2 times, limit 0", 2.0)),
            (
                ["sin", "cos"],
                1,
                ["sin", "(", "X_0", ")", "+", "cos", "(", "X_0", ")"],
                fail("'cos', 'sin' occur 2 times together, limit 1", 1.0),
            ),
            (("sin", "cos"), 2, ["sin", "(", "X_0", ")", "+", "cos", "(", "X_0", ")"], PASS),
            (
                ["sin", "cos", "sin"],
                0,
                ["cos", "(", "X_0", ")"],
                fail("'cos', 'sin' occur 1 times together, limit 0", 1.0),
            ),
            (["sin"], 0, ["sin", "(", "X_0", ")"], fail("'sin' occurs 1 times, limit 0", 1.0)),
        ],
        ids=[
            "over",
            "at-limit",
            "numpy-limit",
            "absent",
            "zero-limit",
            "bracket-list-as-given",
            "group-counted-together",
            "group-at-limit",
            "group-duplicates-dropped",
            "group-of-one-like-single",
        ],
    )
    def test_max_symbol_count_pre_check_table(self, symbol, limit, tokens, expected):
        assert_result(set_up(MaxSymbolCount(symbol, limit), lib(1)).pre_check(tokens), expected)

    @pytest.mark.parametrize(
        "tokens, expected",
        [
            (["(", "X_0", ")"], PASS),
            (["(", "X_0", "+", "C", ")", "*", "X_0"], fail("'(' occurs 1 times, limit 0", 1.0)),
            (["X_0", "+", "C", "*", "X_0"], PASS),
        ],
        ids=["redundant-brackets-dropped", "needed-brackets-kept", "precedence-needs-no-brackets"],
    )
    def test_max_symbol_count_counts_tree_as_written_by_to_list(self, tokens, expected):
        library = lib(1)
        constraint = set_up(MaxSymbolCount("(", 0), library)
        assert_result(constraint.pre_check(tokens_to_tree(tokens, library)), expected)

    @pytest.mark.parametrize(
        "symbols, expected",
        [("sin", frozenset({"sin"})), (["sin", "cos", "sin"], frozenset({"sin", "cos"}))],
        ids=["single-string", "iterable-deduplicated"],
    )
    def test_max_symbol_count_constructor(self, symbols, expected):
        assert MaxSymbolCount(symbols, 1).symbols == expected

    def test_max_symbol_count_rejects_empty_group(self):
        with pytest.raises(ValueError, match="MaxSymbolCount.*empty"):
            MaxSymbolCount([], 1)


class TestNoNesting:
    @pytest.mark.parametrize(
        "group, tokens, expected",
        [
            (["sin", "cos"], ["sin", "(", "cos", "(", "X_0", ")", ")"], fail("'cos' nested inside 'sin'", 1.0)),
            (["sin", "cos"], ["sin", "(", "X_0", ")", "+", "cos", "(", "X_0", ")"], PASS),
            ("sin", ["sin", "(", "sin", "(", "sin", "(", "X_0", ")", ")", ")"], fail("'sin' nested inside 'sin'", 2.0)),
            ("sin", ["sin", "(", "cos", "(", "X_0", ")", ")"], PASS),
            (
                ["sin", "cos"],
                ["sin", "(", "X_0", "+", "cos", "(", "X_0", ")", ")"],
                fail("'cos' nested inside 'sin'", 1.0),
            ),
            ("-", ["X_0", "-", "X_0", "+", "X_0", "-", "X_0"], fail("'-' nested inside '-'", 1.0)),
            ("-", ["X_0", "-", "(", "X_0", "+", "X_0", ")"], PASS),
            ("-", ["X_0", "-", "(", "X_0", "-", "X_0", ")"], fail("'-' nested inside '-'", 1.0)),
            (
                ["-", "sin"],
                ["sin", "(", "X_0", ")", "-", "sin", "(", "sin", "(", "X_0", ")", ")"],
                fail("'sin' nested inside '-'", 3.0),
            ),
            (
                ["sin", "cos"],
                ["sin", "(", "cos", "(", "X_0", ")", "*", "sin", "(", "X_0", ")", ")"],
                fail("'cos' nested inside 'sin'", 2.0),
            ),
            (
                ["sin", "cos"],
                ["cos", "(", "sin", "(", "X_0", ")", ")", "+", "sin", "(", "cos", "(", "X_0", ")", ")"],
                fail("'sin' nested inside 'cos'", 2.0),
            ),
        ],
        ids=[
            "cross-nesting",
            "siblings-pass",
            "self-nesting-amount-2",
            "outside-group-passes",
            "deep-not-only-direct-child",
            "parsed-left-assoc",
            "bracketed-other-operator",
            "nested-only-in-right-operand",
            "outer-is-first-in-pre-order",
            "inner-is-first-below-outer",
            "left-operand-before-right",
        ],
    )
    def test_no_nesting_pre_check_table(self, group, tokens, expected):
        assert_result(set_up(NoNesting(group), lib(1)).pre_check(tokens), expected)

    def test_no_nesting_tree_input(self):
        library = lib(1)
        tree = tokens_to_tree(["sin", "(", "cos", "(", "X_0", ")", ")"], library)
        assert_result(
            set_up(NoNesting(["sin", "cos"]), library).pre_check(tree), fail("'cos' nested inside 'sin'", 1.0)
        )

    @pytest.mark.parametrize(
        "symbols, expected",
        [("sin", frozenset({"sin"})), (["sin", "cos", "sin"], frozenset({"sin", "cos"}))],
        ids=["single-string", "iterable-deduplicated"],
    )
    def test_no_nesting_constructor(self, symbols, expected):
        assert NoNesting(symbols).symbols == expected

    def test_no_nesting_rejects_empty_group(self):
        with pytest.raises(ValueError, match="NoNesting.*empty"):
            NoNesting([])


# ── OutputBounds ─────────────────────────────────────────────────────────────────────────────

_P = [[-1.0], [0.0], [2.0]]


class TestOutputBounds:
    @pytest.mark.parametrize(
        "kwargs, lower, upper",
        [
            ({"lower": 0, "upper": INF}, 0.0, None),
            ({"lower": -INF, "upper": 1}, None, 1.0),
            ({"lower": 1.0, "upper": 1.0}, 1.0, 1.0),
        ],
        ids=["int-bound-inf-upper", "minus-inf-lower", "equal-bounds"],
    )
    def test_output_bounds_constructor_normalises(self, kwargs, lower, upper):
        constraint = OutputBounds([[0], [1]], **kwargs)
        assert constraint.points.shape == (2, 1)
        assert np.issubdtype(constraint.points.dtype, np.floating)
        assert (constraint.lower, constraint.upper) == (lower, upper)
        assert all(type(b) is float for b in (constraint.lower, constraint.upper) if b is not None)

    @pytest.mark.parametrize(
        "points, kwargs",
        [
            ([1.0, 2.0], {"lower": 0.0}),
            (np.zeros((0, 1)), {"lower": 0.0}),
            ([[NAN]], {"lower": 0.0}),
            ([[INF]], {"lower": 0.0}),
            ([[0.0]], {"lower": NAN}),
            ([[0.0]], {"upper": NAN}),
            ([[0.0]], {}),
            ([[0.0]], {"lower": -INF}),
            ([[0.0]], {"lower": -INF, "upper": INF}),
            ([[0.0]], {"lower": 2.0, "upper": 1.0}),
            ([[0.0], [1.0]], {"lower": [0.0, 1.0, 2.0]}),
            ([[0.0], [1.0]], {"upper": [[1.0], [2.0]]}),
            ([[0.0], [1.0]], {"lower": [0.0, NAN]}),
            ([[0.0], [1.0]], {"lower": [-INF, -INF], "upper": [INF, INF]}),
            ([[0.0], [1.0]], {"lower": [0.0, 3.0], "upper": 2.0}),
        ],
        ids=[
            "1-d-points",
            "no-rows",
            "nan-point",
            "inf-point",
            "nan-lower",
            "nan-upper",
            "no-bounds",
            "only-infinite-lower",
            "both-infinite",
            "lower-above-upper",
            "array-wrong-length",
            "array-not-1-d",
            "array-with-nan",
            "arrays-infinite-everywhere",
            "array-above-scalar-at-one-point",
        ],
    )
    def test_output_bounds_constructor_rejects_invalid(self, points, kwargs):
        with pytest.raises(ValueError, match=r"^OutputBounds: "):
            OutputBounds(points, **kwargs)

    def test_output_bounds_array_bounds_normalised(self):
        constraint = OutputBounds([[0.0], [1.0]], lower=[0, -INF], upper=[1, INF])
        assert constraint.lower.dtype == np.float64 and constraint.lower.tolist() == [0.0, -INF]
        assert constraint.upper.dtype == np.float64 and constraint.upper.tolist() == [1.0, INF]
        assert OutputBounds([[0.0], [1.0]], lower=[-INF, -INF], upper=1.0).lower is None
        assert OutputBounds([[0.0], [1.0]], lower=0.0, upper=[INF, INF]).upper is None
        assert OutputBounds([[0.0], [1.0]], upper=[1, 2]).upper.dtype == np.float64

    def test_output_bounds_setup_checks_column_count(self):
        with pytest.raises(ValueError, match=r"(?s)(?=.*OutputBounds)(?=.*2)(?=.*1)"):
            OutputBounds(np.zeros((3, 2)), lower=0.0).setup(lib(1))
        library2 = lib(2)
        constraint = OutputBounds(np.zeros((3, 2)), lower=0.0)
        constraint.setup(library2)
        assert constraint.symbol_library is library2

    def test_output_bounds_evaluates_in_set_up_library(self):
        # Variables y, x of the library map to columns 0 and 1; the default library knows neither.
        constraint = set_up(OutputBounds([[0.0, 5.0]], upper=1.0), lib_vars_y_then_x())
        result = constraint.post_check(_model(["x"]))
        assert_result(result, fail("output outside [-inf, 1.0] at 1 of 1 points", 4.0))

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        "kwargs, expr, parameters, expected",
        [
            ({"upper": 1.0}, ["X_0"], None, fail("output outside [-inf, 1.0] at 1 of 3 points", 1.0)),
            ({"lower": -1.0, "upper": 2.0}, ["X_0"], None, PASS),
            ({"lower": -1.0, "upper": 2.0}, ["X_0"], np.array([]), PASS),
            (
                {"lower": 0.0, "upper": 1.0},
                ["C", "*", "X_0"],
                np.array([2.0]),
                fail("output outside [0.0, 1.0] at 2 of 3 points", 3.0),
            ),
            (
                {"lower": -10.0},
                ["log", "(", "X_0", ")"],
                None,
                fail("output outside [-10.0, inf] at 2 of 3 points", INF),
            ),
            ({"lower": 0.0}, ["C", "*", "X_0"], None, fail("no fitted parameters", INF)),
            (
                {"lower": 0.0},
                ["C", "*", "X_0", "+", "C"],
                np.array([1.0]),
                fail("1 fitted parameters, expression has 2 constants", INF),
            ),
            (
                {"lower": -1.0, "upper": 2.0},
                ["X_0"],
                np.array([1.0]),
                fail("1 fitted parameters, expression has 0 constants", INF),
            ),
            (
                {"lower": [-INF, -10.0, -10.0]},
                ["log", "(", "X_0", ")"],
                None,
                fail("output outside the bounds at 2 of 3 points", INF),
            ),
        ],
        ids=[
            "upper-only-constant-free",
            "bounds-inclusive",
            "empty-parameters-constant-free",
            "largest-distance-both-sides",
            "non-finite-output",
            "no-fitted-parameters",
            "too-few-parameters",
            "parameters-for-constant-free",
            "non-finite-output-array-bounds",
        ],
    )
    def test_output_bounds_post_check_table(self, kwargs, expr, parameters, expected):
        constraint = set_up(OutputBounds(_P, **kwargs), lib(1))
        assert_result(constraint.post_check(_model(expr, parameters)), expected)

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"lower": [-2.0, 0.0, 2.0], "upper": [0.0, 0.0, 2.0]}, PASS),
            ({"lower": [-0.5, 0.0, 2.5]}, fail("output outside the bounds at 2 of 3 points", 0.5)),
            ({"lower": -5.0, "upper": [-1.5, 1.0, 1.0]}, fail("output outside the bounds at 2 of 3 points", 1.0)),
            ({"lower": [-INF, 1.0, -INF]}, fail("output outside the bounds at 1 of 3 points", 1.0)),
            ({"upper": [INF, INF, 0.0]}, fail("output outside the bounds at 1 of 3 points", 2.0)),
            ({"lower": [-INF, -INF, -INF], "upper": 1.0}, fail("output outside [-inf, 1.0] at 1 of 3 points", 1.0)),
        ],
        ids=[
            "inside-per-point",
            "array-lower",
            "scalar-lower-array-upper",
            "infinite-entries-unbounded",
            "array-upper",
            "array-infinite-everywhere-scalar-message",
        ],
    )
    def test_output_bounds_per_point_bounds(self, kwargs, expected):
        # Output of X_0 at _P: -1, 0, 2.
        constraint = set_up(OutputBounds(_P, **kwargs), lib(1))
        assert_result(constraint.post_check(_model(["X_0"])), expected)


# ── Serialisation ────────────────────────────────────────────────────────────────────────────

_EC = "SRToolkit.utils.expression_constraints"


class TestSerialisation:
    @pytest.mark.parametrize(
        "make, expected",
        [
            (lambda: set_up(MaxConstants(4), lib(1)), {"constraint_class": f"{_EC}.MaxConstants", "limit": 4}),
            (lambda: MaxTreeDepth(np.int64(3)), {"constraint_class": f"{_EC}.MaxTreeDepth", "limit": 3}),
            (lambda: RequiredVariables("X_0"), {"constraint_class": f"{_EC}.RequiredVariables", "variables": ["X_0"]}),
            (lambda: RequiredVariables(), {"constraint_class": f"{_EC}.RequiredVariables", "variables": None}),
            (
                lambda: MaxSymbolCount("sin", 2),
                {"constraint_class": f"{_EC}.MaxSymbolCount", "symbols": ["sin"], "limit": 2},
            ),
            (lambda: NoNesting(["sin", "cos"]), {"constraint_class": f"{_EC}.NoNesting", "symbols": ["cos", "sin"]}),
        ],
        ids=[
            "set-up-no-library-key",
            "numpy-limit",
            "single-variable",
            "all-variables",
            "MaxSymbolCount",
            "sorted-symbols",
        ],
    )
    def test_to_dict_exact_form(self, make, expected):
        d = make().to_dict()
        assert d == expected
        assert all(type(v) is int for k, v in d.items() if k == "limit")

    @pytest.mark.parametrize(
        "make",
        [
            lambda: MaxConstants(4),
            lambda: MaxTokens(20),
            lambda: MaxTreeDepth(3),
            lambda: MaxTreeSize(9),
            lambda: RequiredVariables(["X_1", "X_0"]),
            lambda: MaxSymbolCount(["sin", "cos"], 2),
            lambda: NoNesting(["sin", "cos"]),
            lambda: OutputBounds(np.array([[0.0, 1.0], [2.0, 3.0]]), lower=-1.0, upper=None),
            lambda: _Forbid("cos"),
        ],
        ids=[
            "MaxConstants",
            "MaxTokens",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "MaxSymbolCount",
            "NoNesting",
            "OutputBounds",
            "user-defined",
        ],
    )
    def test_json_round_trip_via_base_loader(self, make):
        original = make()
        saved = json.dumps(original.to_dict(), sort_keys=True)
        rebuilt = ExpressionConstraint.from_dict(json.loads(saved))
        assert type(rebuilt) is type(original)
        assert json.dumps(rebuilt.to_dict(), sort_keys=True) == saved
        assert rebuilt.symbol_library is None

    def test_output_bounds_round_trip_keeps_array_bounds(self):
        original = OutputBounds([[0.0], [1.0]], lower=[0.0, -INF], upper=5.0)
        rebuilt = ExpressionConstraint.from_dict(json.loads(json.dumps(original.to_dict())))
        np.testing.assert_array_equal(rebuilt.lower, original.lower)
        assert rebuilt.upper == 5.0

    def test_output_bounds_round_trip_keeps_points(self):
        original = OutputBounds(np.array([[0.0, 1.0], [2.0, 3.0]]), lower=-1.0)
        rebuilt = ExpressionConstraint.from_dict(json.loads(json.dumps(original.to_dict())))
        np.testing.assert_array_equal(rebuilt.points, original.points)
        assert rebuilt.points.dtype == original.points.dtype
        assert (rebuilt.lower, rebuilt.upper) == (-1.0, None)

    @pytest.mark.parametrize(
        "load, make_dict, error, match",
        [
            (ExpressionConstraint.from_dict, lambda: {}, KeyError, "constraint_class"),
            (
                ExpressionConstraint.from_dict,
                lambda: {"constraint_class": "no_such_module_xyz.Foo"},
                ImportError,
                "Foo",
            ),
            (ExpressionConstraint.from_dict, lambda: {"constraint_class": f"{_EC}.Nope"}, ImportError, "Nope"),
            (ExpressionConstraint.from_dict, lambda: MaxNodes(3).to_dict(), TypeError, "MaxNodes"),
            (ExpressionConstraint.from_dict, lambda: MaxDepth(3).to_dict(), TypeError, "MaxDepth"),
            (ExpressionConstraint.from_dict, lambda: _NoOverride().to_dict(), NotImplementedError, None),
            (
                ExpressionConstraint.from_dict,
                lambda: {"constraint_class": f"{_EC}.ExpressionConstraint"},
                NotImplementedError,
                None,
            ),
            (_NoOverride.from_dict, lambda: _NoOverride().to_dict(), NotImplementedError, None),
        ],
        ids=[
            "missing-class-key",
            "unknown-module",
            "unknown-class",
            "grammar-MaxNodes",
            "grammar-MaxDepth",
            "named-class-no-override",
            "named-base-class",
            "subclass-no-override",
        ],
    )
    def test_from_dict_error_contract(self, load, make_dict, error, match):
        d = make_dict()
        with pytest.raises(error, match=match):
            load(d)

    @pytest.mark.parametrize(
        "load, make_dict, expected",
        [
            (MaxConstants.from_dict, lambda: {"limit": 3}, {"limit": 3}),
            (MaxTokens.from_dict, lambda: {"limit": 3}, {"limit": 3}),
            (MaxTreeDepth.from_dict, lambda: {"limit": 3}, {"limit": 3}),
            (MaxTreeSize.from_dict, lambda: {"limit": 3}, {"limit": 3}),
            (RequiredVariables.from_dict, lambda: {"variables": ["X_1"]}, {"variables": ("X_1",)}),
            (
                MaxSymbolCount.from_dict,
                lambda: {"symbols": ["sin"], "limit": 2},
                {"symbols": frozenset({"sin"}), "limit": 2},
            ),
            (NoNesting.from_dict, lambda: {"symbols": ["sin"]}, {"symbols": frozenset({"sin"})}),
            (
                OutputBounds.from_dict,
                lambda: without(OutputBounds([[0.0]], lower=0.0).to_dict(), "constraint_class"),
                {"lower": 0.0, "upper": None},
            ),
        ],
        ids=[
            "MaxConstants",
            "MaxTokens",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "MaxSymbolCount",
            "NoNesting",
            "OutputBounds",
        ],
    )
    def test_subclass_from_dict_ignores_unused_keys(self, load, make_dict, expected):
        rebuilt = load({**make_dict(), "constraint_class": "elsewhere.Other", "extra": 1})
        assert {k: getattr(rebuilt, k) for k in expected} == expected

    @pytest.mark.parametrize(
        "load, make_dict, error, match",
        [
            (MaxConstants.from_dict, lambda: {}, KeyError, "limit"),
            (MaxTokens.from_dict, lambda: {}, KeyError, "limit"),
            (MaxTreeDepth.from_dict, lambda: {}, KeyError, "limit"),
            (MaxTreeSize.from_dict, lambda: {}, KeyError, "limit"),
            (MaxSymbolCount.from_dict, lambda: {"limit": 1}, KeyError, "symbols"),
            (NoNesting.from_dict, lambda: {}, KeyError, "symbols"),
            (RequiredVariables.from_dict, lambda: {}, KeyError, "variables"),
            (
                OutputBounds.from_dict,
                lambda: without(OutputBounds([[0.0]], lower=0.0).to_dict(), "upper"),
                KeyError,
                "upper",
            ),
            (MaxConstants.from_dict, lambda: {"limit": -1}, ValueError, "limit"),
            (NoNesting.from_dict, lambda: {"symbols": []}, ValueError, "empty"),
            (
                OutputBounds.from_dict,
                lambda: {**OutputBounds([[0.0]], lower=0.0).to_dict(), "lower": NAN},
                ValueError,
                None,
            ),
        ],
        ids=[
            "MaxConstants-missing",
            "MaxTokens-missing",
            "MaxTreeDepth-missing",
            "MaxTreeSize-missing",
            "MaxSymbolCount-missing-symbols",
            "NoNesting-missing",
            "RequiredVariables-missing",
            "OutputBounds-missing-upper",
            "invalid-limit",
            "empty-group",
            "nan-bound",
        ],
    )
    def test_subclass_from_dict_error_contract(self, load, make_dict, error, match):
        d = make_dict()
        with pytest.raises(error, match=match):
            load(d)
