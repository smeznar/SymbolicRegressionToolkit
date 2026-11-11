"""
Tests for SRToolkit/utils/grammar/constraints.py.

Covers: AncestorInfo, ExpansionContext, GrammarConstraint base
class, MaxDepth, MaxNodes, MaxOccurrences, NoNested, and DimensionalConsistency (through
derivations: what each slot allows, child requirements, dead ends, and a ConsistentUnits oracle).
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from SRToolkit.utils import units as u
from SRToolkit.utils.expression_constraints import ConsistentUnits
from SRToolkit.utils.grammar import (
    AncestorInfo,
    DimensionalConsistency,
    ExpansionContext,
    Grammar,
    GrammarConstraint,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    NoNested,
    ParseTreeNode,
    Rule,
)
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.units import Unit, UnitRule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sl(*extra_symbols: str, n_vars: int = 1) -> SymbolLibrary:
    """Build a minimal SymbolLibrary with optional extra symbols."""
    return SymbolLibrary.from_symbol_list(list(extra_symbols), n_vars)


def _make_slot(
    nonterminal: str = "E",
    local=None,
    nonterminals: frozenset[str] | None = None,
    steps: int = 0,
    frontier_size: int = 1,
) -> ExpansionContext:
    if nonterminals is None:
        nonterminals = frozenset({"E", "F"})
    return ExpansionContext(
        nonterminal=nonterminal,
        local=local,
        steps=steps,
        parent_rule=None,
        child_index=None,
        ancestors=(),
        partial_tree=ParseTreeNode("E", None),
        nonterminals=nonterminals,
        frontier_size=frontier_size,
    )


# ---------------------------------------------------------------------------
# AncestorInfo
# ---------------------------------------------------------------------------


class TestAncestorInfo:
    def test_field_access(self):
        r = Rule("E", ["E", "+", "F"])
        a = AncestorInfo("E", r, 0)
        assert a.nonterminal == "E"
        assert a.rule is r
        assert a.child_index == 0

    def test_frozen_nonterminal(self):
        r = Rule("E", ["x"])
        a = AncestorInfo("E", r, 0)
        with pytest.raises((AttributeError, TypeError)):
            a.nonterminal = "F"  # type: ignore[misc]

    def test_equality_same_values(self):
        r = Rule("E", ["x"])
        a1 = AncestorInfo("E", r, 0)
        a2 = AncestorInfo("E", r, 0)
        assert a1 == a2

    def test_equality_different_child_index(self):
        r = Rule("E", ["x"])
        a1 = AncestorInfo("E", r, 0)
        a2 = AncestorInfo("E", r, 1)
        assert a1 != a2

    def test_not_hashable_because_rule_is_not_hashable(self):
        r = Rule("E", ["x"])
        a = AncestorInfo("E", r, 0)
        with pytest.raises(TypeError):
            hash(a)


# ---------------------------------------------------------------------------
# ExpansionContext
# ---------------------------------------------------------------------------


class TestExpansionContext:
    def test_all_fields_stored(self):
        root = ParseTreeNode("E", None)
        r = Rule("E", ["x"])
        slot = ExpansionContext(
            nonterminal="E",
            local=42,
            steps=3,
            parent_rule=r,
            child_index=1,
            ancestors=(),
            partial_tree=root,
            nonterminals=frozenset({"E"}),
            frontier_size=5,
        )
        assert slot.nonterminal == "E"
        assert slot.local == 42
        assert slot.steps == 3
        assert slot.parent_rule is r
        assert slot.child_index == 1
        assert slot.ancestors == ()
        assert slot.partial_tree is root
        assert slot.nonterminals == frozenset({"E"})
        assert slot.frontier_size == 5

    def test_frontier_size_defaults_to_one(self):
        slot = ExpansionContext(
            nonterminal="E",
            local=None,
            steps=0,
            parent_rule=None,
            child_index=None,
            ancestors=(),
            partial_tree=ParseTreeNode("E", None),
            nonterminals=frozenset({"E"}),
        )
        assert slot.frontier_size == 1

    def test_frozen(self):
        slot = _make_slot()
        with pytest.raises((AttributeError, TypeError)):
            slot.steps = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GrammarConstraint base class
# ---------------------------------------------------------------------------


class TestConstraintBase:
    def test_initial_local_is_none(self):
        assert GrammarConstraint().initial_local("E") is None

    def test_initial_global_is_none(self):
        assert GrammarConstraint().initial_global() is None

    def test_allows_returns_true(self):
        c = GrammarConstraint()
        slot = _make_slot(local=None)
        assert c.allows(slot, Rule("E", ["x"]), None) is True

    def test_update_propagates_local_to_nt_children(self):
        c = GrammarConstraint()
        nts = frozenset({"E", "F"})
        slot = _make_slot(local=99, nonterminals=nts)
        rule = Rule("E", ["E", "+", "F"])
        child_locals, new_global = c.update(slot, rule, "g")
        assert child_locals == [99, 99]
        assert new_global == "g"

    def test_update_terminal_only_returns_empty_list(self):
        c = GrammarConstraint()
        slot = _make_slot(local=7, nonterminals=frozenset({"E"}))
        child_locals, new_global = c.update(slot, Rule("E", ["x", "+", "y"]), None)
        assert child_locals == []

    def test_update_global_unchanged(self):
        c = GrammarConstraint()
        slot = _make_slot(local=None, nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["x"]), "original")
        assert new_global == "original"

    def test_nonterminals_is_none_by_default(self):
        assert GrammarConstraint.nonterminals is None

    def test_rule_names_is_none_by_default(self):
        assert GrammarConstraint.rule_names is None


# ---------------------------------------------------------------------------
# MaxDepth
# ---------------------------------------------------------------------------


class TestMaxDepth:
    def test_initial_local_equals_limit(self):
        assert MaxDepth(5).initial_local("E") == 5

    def test_initial_local_zero(self):
        assert MaxDepth(0).initial_local("E") == 0

    def test_allows_at_budget_zero_rejects_nt_rule(self):
        c = MaxDepth(0)
        slot = _make_slot(local=0, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), None) is False

    def test_allows_at_budget_zero_accepts_terminal_rule(self):
        c = MaxDepth(0)
        slot = _make_slot(local=0, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["x"]), None) is True

    def test_allows_at_budget_positive_always_true(self):
        c = MaxDepth(3)
        slot = _make_slot(local=3, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), None) is True

    def test_allows_at_budget_one_allows_nt_rule(self):
        c = MaxDepth(1)
        slot = _make_slot(local=1, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), None) is True

    def test_allows_negative_budget_rejects_nt_rule(self):
        c = MaxDepth(0)
        slot = _make_slot(local=-1, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), None) is False

    def test_update_decrements_by_one_per_nt_child(self):
        c = MaxDepth(5)
        nts = frozenset({"E"})
        slot = _make_slot(local=5, nonterminals=nts)
        child_locals, _ = c.update(slot, Rule("E", ["E", "+", "E"]), None)
        assert child_locals == [4, 4]

    def test_update_terminal_only_returns_empty(self):
        c = MaxDepth(5)
        slot = _make_slot(local=5, nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["x"]), None)
        assert child_locals == []

    def test_update_global_stays_none(self):
        c = MaxDepth(5)
        slot = _make_slot(local=5, nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["x"]), None)
        assert new_global is None

    def test_integration_depth_respected(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxDepth(2))
        for _ in range(20):
            tokens = g.start_derivation("E").generate(limit=200)
            assert set(tokens).issubset({"x", "+"})


# ---------------------------------------------------------------------------
# MaxNodes
# ---------------------------------------------------------------------------


class TestMaxNodes:
    def test_initial_global_is_zero(self):
        assert MaxNodes(10).initial_global() == 0

    def test_allows_at_exact_limit(self):
        c = MaxNodes(3)
        # global_=2, frontier_size=1, n_children=0 → min_total = 2+1+0 = 3 ≤ 3 → allowed
        slot = _make_slot(local=None, frontier_size=1, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["x"]), 2) is True

    def test_allows_rejected_one_over_limit(self):
        c = MaxNodes(3)
        # global_=2, frontier_size=1, n_children=1 → min_total = 2+1+1 = 4 > 3 → rejected
        slot = _make_slot(local=None, frontier_size=1, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), 2) is False

    def test_allows_frontier_size_affects_budget(self):
        c = MaxNodes(5)
        # global_=0, frontier_size=3, n_children=0 → min_total = 0+3+0 = 3 ≤ 5 → allowed
        slot = _make_slot(local=None, frontier_size=3, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["x"]), 0) is True

    def test_allows_frontier_size_causes_rejection(self):
        c = MaxNodes(3)
        # global_=0, frontier_size=3, n_children=1 → min_total = 0+3+1 = 4 > 3 → rejected
        slot = _make_slot(local=None, frontier_size=3, nonterminals=frozenset({"E"}))
        assert c.allows(slot, Rule("E", ["E", "+", "E"]), 0) is False

    def test_update_increments_global_by_one(self):
        c = MaxNodes(10)
        slot = _make_slot(local=None, nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["x"]), 4)
        assert new_global == 5

    def test_update_returns_none_per_nt_child(self):
        c = MaxNodes(10)
        slot = _make_slot(local=None, nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["E", "+", "E"]), 0)
        assert child_locals == [None, None]

    def test_integration_stays_within_limit(self):
        limit = 5
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxNodes(limit))
        c = g._constraints[0]
        for _ in range(20):
            d = g.start_derivation("E")
            d.generate(limit=200)
            assert d.global_state(c) <= limit


# ---------------------------------------------------------------------------
# MaxOccurrences
# ---------------------------------------------------------------------------


class TestMaxOccurrences:
    def test_initial_global_is_zero(self):
        assert MaxOccurrences("x", 3).initial_global() == 0

    def test_allows_when_below_limit(self):
        c = MaxOccurrences("x", 2)
        slot = _make_slot()
        assert c.allows(slot, Rule("E", ["x"]), 1) is True

    def test_allows_when_at_limit_with_symbol_rejected(self):
        c = MaxOccurrences("x", 2)
        slot = _make_slot()
        assert c.allows(slot, Rule("E", ["x"]), 2) is False

    def test_allows_when_at_limit_without_symbol_accepted(self):
        c = MaxOccurrences("x", 2)
        slot = _make_slot()
        assert c.allows(slot, Rule("E", ["y"]), 2) is True

    def test_allows_when_above_limit_with_symbol_rejected(self):
        c = MaxOccurrences("x", 1)
        slot = _make_slot()
        assert c.allows(slot, Rule("E", ["x"]), 5) is False

    def test_update_increments_by_rhs_count(self):
        c = MaxOccurrences("x", 10)
        slot = _make_slot(nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["x", "+", "x"]), 0)
        assert new_global == 2

    def test_update_zero_occurrences_in_rhs(self):
        c = MaxOccurrences("x", 10)
        slot = _make_slot(nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["y"]), 3)
        assert new_global == 3

    def test_update_returns_none_per_nt_child(self):
        c = MaxOccurrences("x", 10)
        slot = _make_slot(nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["E", "+", "E"]), 0)
        assert child_locals == [None, None]

    @pytest.mark.parametrize(
        "limit, count, rhs, allowed",
        [
            (1, 0, ["x", "*", "x"], False),
            (2, 0, ["x", "*", "x"], True),
            (2, 1, ["x", "*", "x"], False),
            (1, 0, ["x"], True),
            (1, 1, ["y"], True),
        ],
        ids=["two-at-once-over-limit", "two-at-once-fits", "two-at-once-after-one", "single", "symbol-absent"],
    )
    def test_max_occurrences_allows_counts_symbols_in_rule(self, limit, count, rhs, allowed):
        # Regression: a rule is allowed only if count + rhs.count(symbol) <= limit, so a rule
        # adding several occurrences at once cannot push the count over the limit.
        c = MaxOccurrences("x", limit)
        assert c.allows(_make_slot(), Rule("E", rhs), count) is allowed

    def test_integration_never_exceeds_limit(self):
        g = Grammar(
            [
                Rule("E", ["E", "+", "E"]),
                Rule("E", ["x"]),
                Rule("E", ["y"]),
            ]
        )
        g.add_constraint(MaxOccurrences("x", 1))
        g.add_constraint(MaxDepth(4))
        for _ in range(30):
            tokens = g.start_derivation("E").generate(limit=500)
            assert tokens.count("x") <= 1


# ---------------------------------------------------------------------------
# NoNested
# ---------------------------------------------------------------------------


class TestNoNested:
    def test_single_string_becomes_singleton_frozenset(self):
        c = NoNested("sin")
        assert c.symbols == frozenset({"sin"})

    def test_list_becomes_frozenset(self):
        c = NoNested(["sin", "cos"])
        assert c.symbols == frozenset({"sin", "cos"})

    def test_initial_local_is_false(self):
        assert NoNested("sin").initial_local("E") is False

    def test_allows_when_local_false_all_rules_pass(self):
        c = NoNested("sin")
        slot = _make_slot(local=False)
        assert c.allows(slot, Rule("E", ["sin", "(", "E", ")"]), None) is True

    def test_allows_when_local_true_group_symbol_rejected(self):
        c = NoNested("sin")
        slot = _make_slot(local=True)
        assert c.allows(slot, Rule("E", ["sin", "(", "E", ")"]), None) is False

    def test_allows_when_local_true_non_group_symbol_accepted(self):
        c = NoNested("sin")
        slot = _make_slot(local=True)
        assert c.allows(slot, Rule("E", ["x"]), None) is True

    def test_update_sets_true_when_group_symbol_in_rhs(self):
        c = NoNested("sin")
        slot = _make_slot(local=False, nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["sin", "E"]), None)
        assert all(v is True for v in child_locals)

    def test_update_propagates_true_even_without_group_symbol(self):
        """Once under=True it stays True for all children, even if group symbol absent."""
        c = NoNested("sin")
        slot = _make_slot(local=True, nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["E", "+", "E"]), None)
        assert all(v is True for v in child_locals)

    def test_update_stays_false_when_no_group_symbol(self):
        c = NoNested("sin")
        slot = _make_slot(local=False, nonterminals=frozenset({"E"}))
        child_locals, _ = c.update(slot, Rule("E", ["E", "+", "E"]), None)
        assert all(v is False for v in child_locals)

    def test_cross_nesting_rejected(self):
        """sin inside cos (or cos inside sin) is rejected when both in the group."""
        g = Grammar(
            [
                Rule("E", ["sin", "(", "E", ")"]),
                Rule("E", ["cos", "(", "E", ")"]),
                Rule("E", ["x"]),
            ]
        )
        g.add_constraint(NoNested(["sin", "cos"]))
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # sin(E)
        opts = d.options()
        assert all("sin" not in r.rhs and "cos" not in r.rhs for r in opts)

    def test_integration_no_nesting_in_generated_expressions(self):
        g = Grammar(
            [
                Rule("E", ["sin", "(", "E", ")"]),
                Rule("E", ["E", "+", "E"]),
                Rule("E", ["x"]),
            ]
        )
        g.add_constraint(NoNested("sin"))
        g.add_constraint(MaxDepth(4))
        for _ in range(20):
            tokens = g.start_derivation("E").generate(limit=500)
            # sin(sin(...)) must never occur — find each "sin" and check the
            # token two positions ahead (past the opening paren) is not "sin".
            for i, t in enumerate(tokens):
                if t == "sin" and i + 2 < len(tokens):
                    assert tokens[i + 2] != "sin"


# ---------------------------------------------------------------------------
# DimensionalConsistency
# ---------------------------------------------------------------------------
# Library: X_0 [m], X_1 [s], literal c [m s^-1]; custom fns z (power(0)) and inv (power(-1)).
# Expected values are derived from the class docstring's slot rules and child table.


def _units_lib() -> SymbolLibrary:
    sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "^", "u-", "sin", "sqrt", "^2", "^-1", "C", "pi"], 2)
    sl.add_symbol("c", "lit", 1000, "299792458.0")
    sl.add_symbol("z", "fn", 1000, "np.ones_like({})", unit_rule=UnitRule.power(0))
    sl.add_symbol("inv", "fn", 1000, "1 / ({})", unit_rule=UnitRule.power(-1))
    return sl


_UNITS = {"X_0": u.m, "X_1": u.s}
_LITERALS = {"c": u.m / u.s}
_LEAF_KEYS = {"X_0", "X_1", "C", "pi", "c", "2", "F:X_0", "F:X_1"}
_ALL_LEAVES = {"X_0", "X_1", "C", "pi", "c", "2"}


def _g1_rules() -> dict[str, Rule]:
    """An ambiguous one-non-terminal grammar, used only for slot-level checks."""
    return {
        "X_0": Rule("E", ["X_0"]),
        "X_1": Rule("E", ["X_1"]),
        "C": Rule("E", ["C"]),
        "pi": Rule("E", ["pi"]),
        "c": Rule("E", ["c"]),
        "2": Rule("E", ["2"]),
        "sin": Rule("E", ["sin", "(", "E", ")"]),
        "^": Rule("E", ["E", "^", "E"]),
        "z": Rule("E", ["z", "(", "E", ")"]),
        "inv": Rule("E", ["inv", "(", "E", ")"]),
        "sqrt": Rule("E", ["sqrt", "(", "E", ")"]),
        "*": Rule("E", ["E", "*", "E"]),
        "/": Rule("E", ["E", "/", "E"]),
        "+": Rule("E", ["E", "+", "E"]),
        "()": Rule("E", ["(", "E", ")"]),
        "neg": Rule("E", ["u-", "E"]),
        "F": Rule("E", ["F"]),
        "F:X_0": Rule("F", ["X_0"]),
        "F:X_1": Rule("F", ["X_1"]),
    }


def _allowed_after(target: Unit, steps: list[str], constraint: DimensionalConsistency | None = None) -> set[str]:
    """Keys of the G1 rules allowed at the next slot after applying ``steps`` (keys of G1 rules)."""
    rules = _g1_rules()
    g = Grammar(list(rules.values()))
    g.add_constraint(constraint or DimensionalConsistency(_UNITS, target, _units_lib(), _LITERALS))
    d = g.start_derivation("E")
    for key in steps:
        d.apply(rules[key])
    options = d.options()
    return {key for key, rule in rules.items() if rule in options}


def _generated(grammar: Grammar, seed: int, attempts: int, **generate_kwargs) -> list[list[str]]:
    np.random.seed(seed)
    results = [grammar.generate_one(**generate_kwargs) for _ in range(attempts)]
    return [r for r in results if r is not None]


def _passes_consistent_units(tokens: list[str], units: dict, target: Unit, sl: SymbolLibrary) -> bool:
    check = ConsistentUnits(units, target)
    check.setup(sl)
    return check.pre_check(tokens).passed


class TestDCInit:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"variable_units": {"X_0": {"m": 1}, "X_1": u.s}, "target_unit": u.m},
            {"variable_units": _UNITS, "target_unit": {"m": 1}},
            {"variable_units": _UNITS, "target_unit": None},
            {"variable_units": _UNITS, "target_unit": u.m, "literal_units": {"c": "m"}},
        ],
        ids=["variable-dict", "target-dict", "target-none", "literal-str"],
    )
    def test_dc_rejects_non_units(self, kwargs):
        with pytest.raises(TypeError):
            DimensionalConsistency(symbol_library=_units_lib(), **kwargs)

    @pytest.mark.parametrize(
        "variable_units, literal_units, match",
        [
            ({"X_0": u.m}, None, "X_1"),
            ({"X_0": u.m, "X_1": u.s, "C": u.m}, None, "C"),
            (_UNITS, {"X_0": u.m}, "X_0"),
            ({"X_0": u.m, "C": u.m}, {"X_0": u.m}, r"(?s)(?=.*X_1)(?=.*\bC\b)(?=.*X_0)"),
        ],
        ids=["variable-without-unit", "non-variable-key", "non-literal-key", "every-problem-listed"],
    )
    def test_dc_value_errors_list_every_problem(self, variable_units, literal_units, match):
        with pytest.raises(ValueError, match=match):
            DimensionalConsistency(variable_units, u.m, _units_lib(), literal_units)

    def test_dc_stores_copies(self):
        units = dict(_UNITS)
        sl = _units_lib()
        dc = DimensionalConsistency(units, u.m, sl)
        units["X_2"] = u.kg
        assert dc.variable_units == _UNITS
        assert type(dc.variable_units) is dict
        assert dc.target_unit == u.m
        assert dc.literal_units == {}
        assert dc.symbol_library is sl


class TestDCSlots:
    def test_start_slot_with_dimensional_target(self):
        # P = m: wrong-unit leaves and dimensionless-result rules (^, sin, z) are rejected.
        assert _allowed_after(u.m, []) == {"X_0", "F", "inv", "sqrt", "neg", "*", "/", "+", "()"}

    def test_start_slot_with_dimensionless_target(self):
        assert _allowed_after(Unit(), []) == {
            "C", "pi", "2", "sin", "^", "z", "inv", "sqrt", "neg", "*", "/", "+", "()", "F",
        }  # fmt: skip

    @pytest.mark.parametrize("op", ["*", "/"])
    def test_any_slot_accepts_every_rule(self, op):
        # The first child of MULTIPLY and DIVIDE may take any unit.
        assert _allowed_after(u.m, [op]) == {k for k in _g1_rules() if not k.startswith("F:")}

    @pytest.mark.parametrize(
        "target, steps, expected",
        [
            (u.m, ["+"], {"X_0"}),
            (u.m, ["+", "X_0"], {"X_0"}),
            (u.m, ["*", "+", "X_1"], {"X_1"}),
            (u.m * u.s, ["*", "X_0"], {"X_1"}),
            (u.m, ["*", "X_1"], {"c"}),
            (u.m, ["*", "*", "X_0"], _ALL_LEAVES),
            (u.m / u.s, ["/", "X_0"], {"X_1"}),
            (Unit(), ["/", "c"], {"c"}),
            (u.m, ["*", "/", "X_1"], _ALL_LEAVES),
            (u.s ** Fraction(1, 2), ["sqrt"], {"X_1"}),
            (u.s**-1, ["inv"], {"X_1"}),
            (Unit(), ["z"], _ALL_LEAVES),
            (Unit(), ["sin"], {"C", "pi", "2"}),
            (u.m, ["*", "^"], {"C", "pi", "2"}),
            (u.m, ["*", "^", "C"], {"C", "pi", "2"}),
            (u.m, ["()"], {"X_0"}),
            (u.m, ["F"], {"F:X_0"}),
            (u.m, ["neg"], {"X_0"}),
        ],
        ids=[
            "same-first-is-P",
            "same-second-is-U",
            "same-second-under-any",
            "multiply-second-P-over-U",
            "multiply-second-needs-literal",
            "multiply-under-any",
            "divide-second-U-over-P",
            "divide-dimensionless-target",
            "divide-under-any",
            "power-half",
            "power-minus-one",
            "power-zero-child-any",
            "dimensionless-child",
            "exponent-base-dimensionless-even-under-any",
            "exponent-exponent-dimensionless",
            "pass-through-paren",
            "pass-through-chain",
            "same-unary-child-is-P",
        ],
    )
    def test_child_requirement_table(self, target, steps, expected):
        assert _allowed_after(target, steps) & _LEAF_KEYS == expected


class TestDCDerivations:
    def test_dead_end_raises_in_generate(self):
        to_product, f_x1 = Rule("E", ["F", "*", "F"]), Rule("F", ["X_1"])
        g = Grammar([to_product, Rule("F", ["X_0"]), f_x1])
        g.add_constraint(DimensionalConsistency(_UNITS, u.m, _units_lib()))
        d = g.start_derivation("E")
        d.apply(to_product)
        d.apply(f_x1)  # the second factor would need m s^-1: no rule fits
        assert d.options() == []
        with pytest.raises(RuntimeError):
            d.generate()

    def test_impossible_target_generate_one_returns_none(self):
        # Bug note §6: no variable provides kg.
        sl = _units_lib()
        g = Grammar.from_symbol_library(sl)
        g.add_constraint(DimensionalConsistency(_UNITS, u.kg, sl))
        g.add_constraint(MaxDepth(5))
        np.random.seed(0)
        assert g.generate_one(max_steps=200, max_retries=20) is None

    @pytest.mark.parametrize(
        "rules",
        [
            [
                Rule("E", ["E", "*", "F"]),
                Rule("E", ["F"]),
                Rule("F", ["X_0"]),
                Rule("F", ["X_1"]),
                Rule("F", ["R", "^-1"]),
                Rule("R", ["X_1"]),
            ],
            [Rule("E", ["E", "/", "F"]), Rule("E", ["F"]), Rule("F", ["X_0"]), Rule("F", ["X_1"])],
        ],
        ids=["multiplicative", "division"],
    )
    def test_bug_note_grammars_generate_only_target_unit(self, rules):
        # Bug note §6: with target m, X_1*X_1, X_0/X_1 or X_0/X_0 must never appear.
        sl = _units_lib()
        g = Grammar(rules, start="E")
        g.add_constraint(DimensionalConsistency(_UNITS, u.m, sl))
        g.add_constraint(MaxDepth(6))
        generated = _generated(g, seed=0, attempts=30, max_steps=200, max_retries=20)
        assert generated
        assert all(_passes_consistent_units(t, _UNITS, u.m, sl) for t in generated)

    @settings(max_examples=25, deadline=None)
    @given(
        unit_0=st.sampled_from([u.m, u.s, u.m / u.s, Unit(), u.kg]),
        unit_1=st.sampled_from([u.m, u.s, u.m / u.s, Unit(), u.kg]),
        a=st.sampled_from([-1, 0, 1, 2]),
        b=st.sampled_from([-1, 0, 1, 2]),
        seed=st.integers(0, 2**31 - 1),
    )
    def test_generated_expressions_pass_consistent_units(self, unit_0, unit_1, a, b, seed):
        # Oracle: everything the from_symbol_library grammar generates passes ConsistentUnits.
        sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "/", "^", "sin", "sqrt", "^2", "^-1", "C", "pi"], 2)
        units = {"X_0": unit_0, "X_1": unit_1}
        target = unit_0**a * unit_1**b  # reachable from the variables
        g = Grammar.from_symbol_library(sl)
        g.add_constraint(DimensionalConsistency(units, target, sl))
        g.add_constraint(MaxDepth(15))
        generated = _generated(g, seed=seed, attempts=5, max_steps=300, max_retries=50)
        assert all(_passes_consistent_units(t, units, target, sl) for t in generated)
        # Compound targets rarely succeed (see the class's Warning), so the guard against a vacuous
        # pass covers only targets that are a variable's unit or dimensionless.
        if target in (unit_0, unit_1, Unit()):
            assert generated

    @pytest.mark.parametrize(
        "rhs, match",
        [
            (["E", "+", "C"], r"E\W+\+\W+C"),
            (["foo"], "foo"),
            (["*", "E"], r"\*\W+E"),
            (["E", "E"], r"E\W+E"),
            (["X_0", "E"], r"X_0\W+E"),
        ],
        ids=[
            "two-terminals",
            "unknown-terminal",
            "op-with-one-non-terminal",
            "no-terminal-two-children",
            "leaf-with-child",
        ],
    )
    def test_unsupported_rule_shape_raises_when_candidate(self, rhs, match):
        g = Grammar([Rule("E", ["X_0"]), Rule("E", rhs)])
        g.add_constraint(DimensionalConsistency(_UNITS, u.m, _units_lib()))
        with pytest.raises(ValueError, match=match):
            g.start_derivation("E").options()

    def test_ambiguous_grammar_raises_when_a_finished_child_reparses_with_a_conflict(self):
        # The tree (X_0 - X_0) * X_1 is consistent, but its tokens X_0 - X_0 * X_1 parse as
        # X_0 - (X_0 * X_1): m against m s.
        r = {
            "+": Rule("E", ["E", "+", "E"]),
            "-": Rule("E", ["E", "-", "E"]),
            "*": Rule("E", ["E", "*", "E"]),
            "X_0": Rule("E", ["X_0"]),
            "X_1": Rule("E", ["X_1"]),
        }
        g = Grammar(list(r.values()))
        g.add_constraint(DimensionalConsistency(_UNITS, u.m * u.s, _units_lib()))
        d = g.start_derivation("E")
        for key in ["+", "*", "-", "X_0", "X_0", "X_1"]:
            d.apply(r[key])
        with pytest.raises(ValueError, match="ambiguous"):
            d.options()

    def test_unsupported_rule_not_checked_until_candidate(self):
        leaf = Rule("E", ["X_0"])
        g = Grammar([leaf, Rule("Z", ["foo"])])
        g.add_constraint(DimensionalConsistency(_UNITS, u.m, _units_lib()))
        assert g.start_derivation("E").options() == [leaf]


class TestDCSerialization:
    def test_dc_round_trip(self):
        sl = _units_lib()
        original = DimensionalConsistency(_UNITS, u.m, sl, _LITERALS)
        d = json.loads(json.dumps(original.to_dict()))
        assert d == {
            "constraint_class": "SRToolkit.utils.grammar.constraints.DimensionalConsistency",
            "variable_units": {"X_0": {"m": "1"}, "X_1": {"s": "1"}},
            "target_unit": {"m": "1"},
            "symbol_library": sl.to_dict(),
            "literal_units": {"c": {"m": "1", "s": "-1"}},
        }
        rebuilt = GrammarConstraint.from_dict({**d, "extra": 1, "allow_unit_polymorphic_constants": True})
        assert isinstance(rebuilt, DimensionalConsistency)
        assert (rebuilt.variable_units, rebuilt.target_unit, rebuilt.literal_units) == (_UNITS, u.m, _LITERALS)
        assert rebuilt.symbol_library.to_dict() == sl.to_dict()
        assert _allowed_after(u.m, [], rebuilt) == _allowed_after(u.m, [], original)

    @pytest.mark.parametrize(
        "change, error, match",
        [
            (
                lambda d: d.pop("symbol_library"),
                ValueError,
                "earlier version",
            ),
            (lambda d: d.update(target_unit={"m": 0.5}), ValueError, None),
            (lambda d: d["symbol_library"].update(format_version=2), ValueError, None),
            (lambda d: d.pop("target_unit"), KeyError, "target_unit"),
            (lambda d: d.pop("literal_units"), KeyError, "literal_units"),
        ],
        ids=["saved-by-earlier-version", "invalid-unit", "invalid-library", "missing-target", "missing-literals"],
    )
    def test_dc_from_dict_errors(self, change, error, match):
        d = DimensionalConsistency(_UNITS, u.m, _units_lib()).to_dict()
        change(d)
        with pytest.raises(error, match=match):
            DimensionalConsistency.from_dict(d)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestConstraintBaseSerialization:
    def test_to_dict_stores_class_path(self):
        c = MaxDepth(5)
        d = c.to_dict()
        assert "constraint_class" in d
        assert "MaxDepth" in d["constraint_class"]

    def test_base_from_dict_missing_class_key_raises(self):
        with pytest.raises(KeyError):
            GrammarConstraint.from_dict({})


class TestMaxDepthSerialization:
    def test_roundtrip(self):
        c = MaxDepth(7)
        d = c.to_dict()
        assert d["limit"] == 7
        c2 = MaxDepth.from_dict(d)
        assert c2.limit == 7

    def test_constraint_class_key(self):
        assert "MaxDepth" in MaxDepth(1).to_dict()["constraint_class"]


class TestMaxNodesSerialization:
    def test_roundtrip(self):
        c = MaxNodes(15)
        d = c.to_dict()
        assert d["limit"] == 15
        c2 = MaxNodes.from_dict(d)
        assert c2.limit == 15

    def test_constraint_class_key(self):
        assert "MaxNodes" in MaxNodes(1).to_dict()["constraint_class"]


class TestMaxOccurrencesSerialization:
    def test_roundtrip(self):
        c = MaxOccurrences("sin", 3)
        d = c.to_dict()
        assert d["symbol"] == "sin"
        assert d["limit"] == 3
        c2 = MaxOccurrences.from_dict(d)
        assert c2.symbol == "sin"
        assert c2.limit == 3

    def test_constraint_class_key(self):
        assert "MaxOccurrences" in MaxOccurrences("x", 1).to_dict()["constraint_class"]


class TestNoNestedSerialization:
    def test_roundtrip_list(self):
        c = NoNested(["sin", "cos"])
        d = c.to_dict()
        assert set(d["symbols"]) == {"sin", "cos"}
        c2 = NoNested.from_dict(d)
        assert c2.symbols == frozenset({"sin", "cos"})

    def test_roundtrip_single_string(self):
        c = NoNested("sin")
        d = c.to_dict()
        c2 = NoNested.from_dict(d)
        assert c2.symbols == frozenset({"sin"})

    def test_constraint_class_key(self):
        assert "NoNested" in NoNested("x").to_dict()["constraint_class"]


class TestConstraintFromDict:
    def test_dispatch_max_depth(self):
        d = MaxDepth(4).to_dict()
        c = GrammarConstraint.from_dict(d)
        assert isinstance(c, MaxDepth)
        assert c.limit == 4

    def test_dispatch_max_nodes(self):
        c = GrammarConstraint.from_dict(MaxNodes(20).to_dict())
        assert isinstance(c, MaxNodes)
        assert c.limit == 20

    def test_dispatch_max_occurrences(self):
        c = GrammarConstraint.from_dict(MaxOccurrences("cos", 2).to_dict())
        assert isinstance(c, MaxOccurrences)
        assert c.symbol == "cos"

    def test_dispatch_no_nested(self):
        c = GrammarConstraint.from_dict(NoNested(["sin"]).to_dict())
        assert isinstance(c, NoNested)

    def test_dispatch_dimensional_consistency(self):
        original = DimensionalConsistency({"X_0": u.m}, u.m, _sl(n_vars=1))
        c = GrammarConstraint.from_dict(original.to_dict())
        assert isinstance(c, DimensionalConsistency)
        assert c.target_unit == u.m

    def test_dispatch_user_defined(self):
        """GrammarConstraint.from_dict resolves classes via importlib, including user-defined ones."""

        class _AlwaysAllow(GrammarConstraint):
            def to_dict(self):
                return {**super().to_dict(), "extra": 42}

            @classmethod
            def from_dict(cls, d):
                return cls()

        # Patch the class into a reachable module so importlib can find it
        import sys

        mod = sys.modules[__name__]
        mod._AlwaysAllow = _AlwaysAllow
        _AlwaysAllow.__module__ = __name__
        _AlwaysAllow.__qualname__ = "_AlwaysAllow"

        d = _AlwaysAllow().to_dict()
        c = GrammarConstraint.from_dict(d)
        assert isinstance(c, _AlwaysAllow)

    def test_subclass_without_override_raises(self):
        class _NoFromDict(GrammarConstraint):
            pass

        with pytest.raises(NotImplementedError, match="_NoFromDict"):
            _NoFromDict.from_dict({})
