"""Spec tests for SRToolkit.utils.expression_constraints.

Expected values are derived from the docstrings (messages, amounts, saved forms), never from the
implementation. Grammar-counterpart property tests drive a derivation with Hypothesis-drawn
choices, so they use no global RNG.
"""

import json
import sys
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from strategies import infix_expressions

from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.utils import units as u
from SRToolkit.utils.expression_constraints import (
    ConsistentUnits,
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
from SRToolkit.utils.units import Unit, UnitRule

u.m, S = u.m, u.s
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
                lambda: set_up(ConsistentUnits({"X_0": u.m}, u.m), lib(1)),
                "post_check",
                _model(["X_0"]),
                "ConsistentUnits implements no post_check",
            ),
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
            "ConsistentUnits-post",
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
            (lambda: ConsistentUnits({"X_0": u.m}, u.m), "pre_check", ["X_0"]),
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
            "ConsistentUnits",
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
        [
            lambda: MaxTreeDepth(5),
            lambda: MaxTreeSize(5),
            lambda: NoNesting("sin"),
            lambda: ConsistentUnits({"X_0": u.m}, u.m),
        ],
        ids=["MaxTreeDepth", "MaxTreeSize", "NoNesting", "ConsistentUnits"],
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
            lambda: ConsistentUnits({"X_0": u.m}, u.m),
        ],
        ids=[
            "MaxConstants",
            "MaxTokens",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "MaxSymbolCount",
            "NoNesting",
            "ConsistentUnits",
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
            lambda: ConsistentUnits({"X_0": u.m, "X_1": u.s}, Unit()),
        ],
        ids=[
            "MaxConstants",
            "MaxTreeDepth",
            "MaxTreeSize",
            "RequiredVariables",
            "NoNesting",
            "MaxSymbolCount",
            "ConsistentUnits",
        ],
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


# ── ConsistentUnits ──────────────────────────────────────────────────────────────────────────


def _units_lib():
    """lib(2) plus custom symbols with explicit unit rules, a literal ``c`` and an unruled op ``%``."""
    library = lib(2)
    library.add_symbol("abs", "fn", 1000, "np.abs({})", unit_rule=UnitRule.SAME)
    library.add_symbol("cbrt", "fn", 1000, "np.cbrt({})", unit_rule=UnitRule.power(Fraction(1, 3)))
    library.add_symbol("isqrt", "fn", 1000, "1 / np.sqrt({})", unit_rule=UnitRule.power(Fraction(-1, 2)))
    # Binds tighter than "*", so the message's subexpression depends on the library's precedences.
    library.add_symbol("max", "op", 500, "np.maximum({}, {})", unit_rule=UnitRule.SAME)
    library.add_symbol("%", "op", 200, "np.mod({}, {})")
    library.add_symbol("c", "lit", 1000, "299792458.0")
    return library


def _units_lib_sqrt_dimensionless():
    library = _units_lib()
    library.add_symbol("sqrt", "fn", 1000, "np.sqrt({})", unit_rule=UnitRule.DIMENSIONLESS)
    return library


def _lit_lib():
    library = lib(2)
    library.add_symbol("c", "lit", 1000, "299792458.0")
    return library


class TestConsistentUnits:
    # X_0: m, X_1: s, literal c: m s^-1; the rules come from the library.
    @pytest.mark.parametrize(
        "make_lib, tokens, target, expected",
        [
            (_units_lib, ["X_0", "+", "X_0"], u.m, PASS),
            (_units_lib, ["X_0", "-", "X_1"], u.m, fail("X_0-X_1: operands have units m and s", 1.0)),
            (_units_lib, ["X_0", "*", "X_0", "/", "X_1"], Unit({"m": 2, "s": -1}), PASS),
            (_units_lib, ["X_0", "^", "2"], Unit({"m": 2}), PASS),
            (_units_lib, ["X_0", "^", "0.5"], Unit({"m": Fraction(1, 2)}), PASS),
            (_units_lib, ["X_1", "^", "0.1"], Unit({"s": Fraction(1, 10)}), PASS),
            (_units_lib, ["X_0", "^", "1e-3"], Unit({"m": Fraction(1, 1000)}), PASS),
            (_units_lib, ["X_1", "^", "0"], Unit(), PASS),
            (_units_lib, ["X_0", "^", "X_1"], Unit(), fail("X_0^X_1: base has unit m", 1.0)),
            (_units_lib, ["2", "^", "X_1"], Unit(), fail("2^X_1: exponent has unit s", 1.0)),
            (_units_lib, ["X_0", "^", "1e400"], Unit(), fail("X_0^1e400: base has unit m", 1.0)),
            (_units_lib, ["X_0", "^", "inf"], Unit(), fail("X_0^inf: base has unit m", 1.0)),
            (_units_lib, ["(", "X_0", "/", "X_0", ")", "^", "C"], Unit(), PASS),
            (
                _units_lib,
                ["sqrt", "(", "X_0", ")", "*", "cbrt", "(", "X_1", ")"],
                Unit({"m": Fraction(1, 2), "s": Fraction(1, 3)}),
                PASS,
            ),
            (_units_lib, ["(", "X_0", ")", "^3", "*", "(", "X_1", ")", "^-1"], Unit({"m": 3, "s": -1}), PASS),
            (_units_lib, ["abs", "(", "floor", "(", "ceil", "(", "u-", "X_0", ")", ")", ")"], u.m, PASS),
            (_units_lib, ["ln", "(", "X_0", "/", "X_0", ")"], Unit(), PASS),
            (_units_lib, ["sin", "(", "X_0", ")"], Unit(), fail("sin(X_0): argument has unit m", 1.0)),
            (_units_lib, ["X_0", "%", "C"], Unit(), fail("X_0%C: left operand has unit m", 1.0)),
            (_units_lib, ["C", "%", "X_1"], Unit(), fail("C%X_1: right operand has unit s", 1.0)),
            (_units_lib, ["X_0", "max", "X_1"], u.m, fail("X_0maxX_1: operands have units m and s", 1.0)),
            (
                _units_lib,
                ["sin", "(", "(", "X_0", "max", "X_0", ")", "*", "X_1", ")"],
                Unit(),
                fail("sin(X_0maxX_0*X_1): argument has unit m s", 1.0),
            ),
            (
                _units_lib,
                ["(", "X_0", "max", "X_0", ")", "*", "X_1", "-", "X_0"],
                Unit(),
                fail("X_0maxX_0*X_1-X_0: operands have units m s and m", 1.0),
            ),
            (
                _units_lib,
                ["(", "X_0", "+", "X_1", ")", "^", "2"],
                Unit({"m": 2}),
                fail("X_0+X_1: operands have units m and s", 1.0),
            ),
            (_units_lib, ["isqrt", "(", "X_0", ")"], Unit({"m": Fraction(-1, 2)}), PASS),
            (
                _units_lib_sqrt_dimensionless,
                ["sqrt", "(", "X_0", ")"],
                Unit(),
                fail("sqrt(X_0): argument has unit m", 1.0),
            ),
            (_units_lib, ["c", "*", "X_1", "+", "pi", "*", "2", "*", "C", "*", "X_0"], u.m, PASS),
            (_units_lib, ["C"], u.m, fail("result has unit 1, target m", 1.0)),
            (_units_lib, ["X_0"], u.s, fail("result has unit m, target s", 1.0)),
            (_units_lib, ["sin", "(", "X_0", ")", "+", "X_1"], Unit(), fail("sin(X_0): argument has unit m", 1.0)),
            (_units_lib, ["X_0", "+", "sin", "(", "X_1", ")"], u.m, fail("sin(X_1): argument has unit s", 1.0)),
        ],
        ids=[
            "same-equal",
            "same-differ",
            "multiply-divide",
            "exponent-integer",
            "exponent-decimal-half",
            "exponent-exact-decimal",
            "exponent-scientific",
            "exponent-zero",
            "exponent-not-number-base-before-exponent",
            "exponent-not-number-exponent-conflict",
            "exponent-not-finite",
            "exponent-inf-not-a-number",
            "exponent-not-number-dimensionless",
            "power-sqrt-custom-cbrt",
            "power-postfix-tokens",
            "same-unary-chain-custom-abs",
            "dimensionless-fn-dimensionless-arg",
            "dimensionless-fn",
            "unruled-op-left",
            "unruled-op-right",
            "custom-op-same",
            "message-uses-library-precedence",
            "same-message-uses-library-precedence",
            "number-exponent-compound-base",
            "custom-fn-power",
            "builtin-rule-overridden",
            "leaves-literal-number-const",
            "const-cannot-repair",
            "root-mismatch",
            "post-order-left-subtree-first",
            "post-order-subtrees-before-node",
        ],
    )
    def test_consistent_units_pre_check_table(self, make_lib, tokens, target, expected):
        constraint = ConsistentUnits({"X_0": u.m, "X_1": u.s}, target, literal_units={"c": u.m / u.s})
        assert_result(set_up(constraint, make_lib()).pre_check(tokens), expected)

    def test_consistent_units_exponent_token_float_accepts_but_fraction_may_not(self):
        # Regression: on Python 3.10, Fraction rejects "1_0" although float() accepts it; the check
        # must then treat it as no number exponent instead of raising.
        constraint = set_up(ConsistentUnits({"X_0": u.m, "X_1": u.s}, u.m**10), lib(2))
        result = constraint.pre_check(["X_0", "^", "1_0"])
        if sys.version_info >= (3, 11):
            assert result.passed
        else:
            assert_result(result, fail("X_0^1_0: base has unit m", 1.0))

    def test_consistent_units_reads_rules_at_each_pre_check(self):
        library = lib(1)
        constraint = set_up(ConsistentUnits({"X_0": u.m}, u.m), library)
        assert_result(constraint.pre_check(["sin", "(", "X_0", ")"]), fail("sin(X_0): argument has unit m", 1.0))
        library.add_symbol("sin", "fn", 1000, "np.sin({})", unit_rule=UnitRule.SAME)
        assert constraint.pre_check(["sin", "(", "X_0", ")"]) == PASS

    @settings(max_examples=60)
    @given(tokens=infix_expressions())
    def test_consistent_units_dimensionless_task_accepts_everything(self, tokens):
        constraint = set_up(ConsistentUnits({"X_0": Unit(), "X_1": Unit()}, Unit()), lib(2))
        assert constraint.pre_check(tokens) == PASS

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"variable_units": {"X_0": "m"}, "target_unit": u.m},
            {"variable_units": {"X_0": {"m": 1}}, "target_unit": u.m},
            {"variable_units": {"X_0": u.m}, "target_unit": "m"},
            {"variable_units": {"X_0": u.m}, "target_unit": None},
            {"variable_units": {"X_0": u.m}, "target_unit": u.m, "literal_units": {"c": 1}},
        ],
        ids=["variable-str", "variable-dict", "target-str", "target-none", "literal-int"],
    )
    def test_consistent_units_constructor_rejects_non_units(self, kwargs):
        with pytest.raises(TypeError):
            ConsistentUnits(**kwargs)

    def test_consistent_units_constructor_stores_copies(self):
        given_units = {"X_0": u.m}
        constraint = ConsistentUnits(given_units, u.s)
        given_units["X_1"] = u.s
        assert constraint.variable_units == {"X_0": u.m}
        assert type(constraint.variable_units) is dict
        assert constraint.target_unit == u.s
        assert constraint.literal_units == {}

    def test_consistent_units_from_background_knowledge(self):
        bk = BackgroundKnowledge(
            _lit_lib(),
            variables={"X_0": VariableInfo("x", unit=u.m), "X_1": VariableInfo(unit=u.s)},
            target=VariableInfo("v", unit=u.m / u.s),
            constants={"c": VariableInfo("c", unit=u.m / u.s), "pi": VariableInfo("π")},
        )
        constraint = ConsistentUnits.from_background_knowledge(bk)
        assert constraint.variable_units == {"X_0": u.m, "X_1": u.s}
        assert constraint.target_unit == u.m / u.s
        assert constraint.literal_units == {"c": u.m / u.s}
        assert constraint.symbol_library is None

    @pytest.mark.parametrize(
        "variables, target, match",
        [
            ({"X_0": VariableInfo(unit=u.m)}, VariableInfo(unit=u.s), "X_1"),
            ({"X_0": VariableInfo(unit=u.m), "X_1": VariableInfo("t")}, VariableInfo(unit=u.s), "X_1"),
            ({"X_0": VariableInfo(unit=u.m), "X_1": VariableInfo(unit=u.s)}, None, "target"),
            ({"X_0": VariableInfo(unit=u.m), "X_1": VariableInfo(unit=u.s)}, VariableInfo("v"), "target"),
            ({"X_0": VariableInfo(unit=u.m)}, None, r"(?s)(?=.*X_1)(?=.*target)"),
        ],
        ids=["variable-missing", "variable-unit-none", "target-none", "target-unit-none", "both-named"],
    )
    def test_consistent_units_from_background_knowledge_errors(self, variables, target, match):
        bk = BackgroundKnowledge(lib(2), variables=variables, target=target)
        with pytest.raises(ValueError, match=match):
            ConsistentUnits.from_background_knowledge(bk)

    @pytest.mark.parametrize(
        "make_lib, variable_units, literal_units, match",
        [
            (lambda: lib(2), {"X_0": u.m}, None, "X_1"),
            (lambda: lib(3), {"X_0": u.m}, None, r"(?s)(?=.*X_1)(?=.*X_2)"),
            (lambda: lib(2), {"X_0": u.m, "X_1": u.s, "X_9": u.m}, None, "X_9"),
            (lambda: lib(2), {"X_0": u.m, "X_1": u.s, "C": u.m}, None, "C"),
            (lambda: lib(2), {"X_0": u.m, "X_1": u.s}, {"X_0": u.m}, "X_0"),
            (lambda: lib(2), {"X_0": u.m, "X_1": u.s}, {"c": u.m}, "c"),
            (lambda: lib(2), {"X_0": u.m}, {"c": u.m}, r"(?s)(?=.*X_1)(?=.*c\b)"),
        ],
        ids=[
            "variable-without-unit",
            "all-variables-without-unit-named",
            "unknown-variable",
            "const-as-variable",
            "variable-as-literal",
            "unknown-literal",
            "every-problem-listed",
        ],
    )
    def test_consistent_units_setup_errors(self, make_lib, variable_units, literal_units, match):
        constraint = ConsistentUnits(variable_units, u.m, literal_units)
        with pytest.raises(ValueError, match=match):
            constraint.setup(make_lib())

    def test_consistent_units_round_trip_keeps_units(self):
        original = ConsistentUnits(
            {"X_0": u.m ** Fraction(1, 2), "X_1": Unit({"bit": 1}, custom_units=["bit"])}, u.m / u.s, {"c": u.m / u.s}
        )
        rebuilt = ExpressionConstraint.from_dict(json.loads(json.dumps(original.to_dict())))
        assert rebuilt.variable_units == {"X_0": u.m ** Fraction(1, 2), "X_1": Unit({"bit": 1}, custom_units=["bit"])}
        assert rebuilt.target_unit == u.m / u.s
        assert rebuilt.literal_units == {"c": u.m / u.s}
        assert rebuilt.symbol_library is None


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
            (
                lambda: ConsistentUnits({"X_0": u.m, "X_1": u.s}, u.m / u.s),
                {
                    "constraint_class": f"{_EC}.ConsistentUnits",
                    "variable_units": {"X_0": {"m": "1"}, "X_1": {"s": "1"}},
                    "target_unit": {"m": "1", "s": "-1"},
                    "literal_units": {},
                },
            ),
        ],
        ids=[
            "set-up-no-library-key",
            "numpy-limit",
            "single-variable",
            "all-variables",
            "MaxSymbolCount",
            "sorted-symbols",
            "ConsistentUnits",
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
            lambda: ConsistentUnits(
                {"X_0": u.m ** Fraction(1, 2)}, Unit({"bit": 1}, custom_units=["bit"]), {"pi": u.m}
            ),
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
            "ConsistentUnits",
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
            (
                ConsistentUnits.from_dict,
                lambda: without(ConsistentUnits({"X_0": u.m}, u.s).to_dict(), "constraint_class"),
                {"variable_units": {"X_0": u.m}, "target_unit": u.s, "literal_units": {}},
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
            "ConsistentUnits",
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
            (
                ConsistentUnits.from_dict,
                lambda: without(ConsistentUnits({"X_0": u.m}, u.s).to_dict(), "variable_units"),
                KeyError,
                "variable_units",
            ),
            (
                ConsistentUnits.from_dict,
                lambda: without(ConsistentUnits({"X_0": u.m}, u.s).to_dict(), "target_unit"),
                KeyError,
                "target_unit",
            ),
            (
                ConsistentUnits.from_dict,
                lambda: without(ConsistentUnits({"X_0": u.m}, u.s).to_dict(), "literal_units"),
                KeyError,
                "literal_units",
            ),
            (NoNesting.from_dict, lambda: {"symbols": []}, ValueError, "empty"),
            (
                ConsistentUnits.from_dict,
                lambda: {**ConsistentUnits({"X_0": u.m}, u.s).to_dict(), "target_unit": {"m": 0.5}},
                ValueError,
                None,
            ),
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
            "ConsistentUnits-missing-variable-units",
            "ConsistentUnits-missing-target-unit",
            "ConsistentUnits-missing-literal-units",
            "invalid-limit",
            "empty-group",
            "ConsistentUnits-invalid-unit",
            "nan-bound",
        ],
    )
    def test_subclass_from_dict_error_contract(self, load, make_dict, error, match):
        d = make_dict()
        with pytest.raises(error, match=match):
            load(d)


# ── describe / summary: the text and record forms of a constraint ─────────────────────────────


@pytest.mark.parametrize(
    "constraint, text, record",
    [
        (MaxConstants(8), "at most 8 free constants", {"kind": "max_constants", "limit": 8}),
        (MaxTokens(30), "at most 30 tokens", {"kind": "max_tokens", "limit": 30}),
        (MaxTreeDepth(4), "expression tree at most 4 levels deep", {"kind": "max_tree_depth", "limit": 4}),
        (MaxTreeSize(15), "at most 15 nodes in the expression tree", {"kind": "max_tree_size", "limit": 15}),
        (
            RequiredVariables(["X_1", "X_0"]),
            "uses each of X_1, X_0",
            {"kind": "required_variables", "variables": ["X_1", "X_0"]},
        ),
        (RequiredVariables(), "uses every variable", {"kind": "required_variables", "variables": None}),
        (MaxSymbolCount("sin", 2), "sin at most 2 times", {"kind": "max_symbol_count", "symbols": ["sin"], "limit": 2}),
        (
            MaxSymbolCount(["sin", "cos"], 2),
            "cos and sin together at most 2 times",
            {"kind": "max_symbol_count", "symbols": ["cos", "sin"], "limit": 2},
        ),
        (
            NoNesting(["sin", "cos"]),
            "no cos or sin inside another cos or sin",
            {"kind": "no_nesting", "symbols": ["cos", "sin"]},
        ),
        (NoNesting("exp"), "no exp inside another exp", {"kind": "no_nesting", "symbols": ["exp"]}),
        (
            OutputBounds(np.zeros((50, 1)), lower=0),
            "output >= 0.0 at 50 given points",
            {"kind": "output_bounds", "lower": 0.0, "upper": None, "num_points": 50},
        ),
        (
            OutputBounds(np.zeros((3, 1)), upper=1.5),
            "output <= 1.5 at 3 given points",
            {"kind": "output_bounds", "lower": None, "upper": 1.5, "num_points": 3},
        ),
        (
            OutputBounds(np.zeros((3, 1)), lower=-1, upper=1),
            "output within [-1.0, 1.0] at 3 given points",
            {"kind": "output_bounds", "lower": -1.0, "upper": 1.0, "num_points": 3},
        ),
        (
            OutputBounds(np.zeros((2, 1)), lower=[0.0, 1.0]),
            "output >= per-point bounds at 2 given points",
            {"kind": "output_bounds", "lower": "per point", "upper": None, "num_points": 2},
        ),
        (
            ConsistentUnits({"X_0": u.m}, u.m / u.s),
            "dimensionally consistent, with unit m s^-1",
            {
                "kind": "consistent_units",
                "variable_units": {"X_0": {"m": "1"}},
                "target_unit": {"m": "1", "s": "-1"},
                "literal_units": {},
            },
        ),
        (
            ConsistentUnits({"X_0": u.m}, Unit()),
            "dimensionally consistent, with unit 1",
            {"kind": "consistent_units", "variable_units": {"X_0": {"m": "1"}}, "target_unit": {}, "literal_units": {}},
        ),
    ],
    ids=[
        "max-constants",
        "max-tokens",
        "max-tree-depth",
        "max-tree-size",
        "required-variables",
        "required-all",
        "symbol-count-one",
        "symbol-count-group",
        "no-nesting-group",
        "no-nesting-one",
        "bounds-lower",
        "bounds-upper",
        "bounds-both",
        "bounds-per-point",
        "consistent-units",
        "consistent-units-dimensionless",
    ],
)
def test_builtin_describe_and_summary(constraint, text, record):
    assert constraint.describe() == text
    assert constraint.summary() == record
    json.dumps(constraint.summary())


def test_default_describe_and_summary_from_to_dict():
    # A custom constraint without overrides: rendered from its to_dict, kind from the class name.
    assert _Forbid("sin").describe() == "_Forbid(symbol='sin')"
    assert _Forbid("sin").summary() == {"kind": "_forbid", "symbol": "sin"}
    assert _NoOverride().describe() == "_NoOverride()"
    assert _NoOverride().summary() == {"kind": "_no_override"}
