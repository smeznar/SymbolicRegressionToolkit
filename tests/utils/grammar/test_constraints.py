"""
Tests for SRToolkit/utils/grammar/constraints.py.

Covers: _to_unit, _units_equal, AncestorInfo, ExpansionContext, Constraint base
class, MaxDepth, MaxNodes, MaxOccurrences, NoNested, and DimensionalConsistency
(all internal helpers: _classify_rule named and fallback paths, _fn_child_unit,
_leaf_compatible, _child_units, _is_additive, allows, update).
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from SRToolkit.utils.grammar import (
    AncestorInfo,
    Constraint,
    DimensionalConsistency,
    ExpansionContext,
    Grammar,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    NoNested,
    ParseTreeNode,
    Rule,
)
from SRToolkit.utils.grammar.constraints import (
    TRANSCENDENTAL_FNS,
    _to_unit,
    _units_equal,
)
from SRToolkit.utils.symbol_library import SymbolLibrary

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


def _dc(
    variable_units: dict | None = None,
    target_unit: dict | None = None,
    constant_units: dict | None = None,
    symbol_library: SymbolLibrary | None = None,
    allow_unit_polymorphic_constants: bool = False,
) -> DimensionalConsistency:
    return DimensionalConsistency(
        variable_units=variable_units or {},
        target_unit=target_unit or {},
        constant_units=constant_units,
        symbol_library=symbol_library,
        allow_unit_polymorphic_constants=allow_unit_polymorphic_constants,
    )


# ---------------------------------------------------------------------------
# _to_unit
# ---------------------------------------------------------------------------


class TestToUnit:
    def test_nonzero_dimensions_preserved(self):
        result = _to_unit({"m": 1, "s": -1})
        assert result == {"m": Fraction(1), "s": Fraction(-1)}

    def test_zero_exponent_dropped(self):
        result = _to_unit({"m": 1, "s": 0})
        assert "s" not in result
        assert result == {"m": Fraction(1)}

    def test_all_zero_gives_empty(self):
        assert _to_unit({"m": 0, "s": 0}) == {}

    def test_empty_input_stays_empty(self):
        assert _to_unit({}) == {}

    def test_fractional_exponent_preserved(self):
        result = _to_unit({"m": Fraction(1, 2)})
        assert result == {"m": Fraction(1, 2)}

    def test_int_exponent_becomes_fraction(self):
        result = _to_unit({"kg": 2})
        assert isinstance(result["kg"], Fraction)


# ---------------------------------------------------------------------------
# _units_equal
# ---------------------------------------------------------------------------


class TestUnitsEqual:
    def test_equal_units(self):
        assert _units_equal({"m": 1, "s": -1}, {"m": 1, "s": -1})

    def test_different_values_not_equal(self):
        assert not _units_equal({"m": 1}, {"m": 2})

    def test_different_dims_not_equal(self):
        assert not _units_equal({"m": 1}, {"s": 1})

    def test_zero_exp_normalized_before_comparison(self):
        assert _units_equal({"m": 1, "s": 0}, {"m": 1})

    def test_both_empty_equal(self):
        assert _units_equal({}, {})

    def test_one_empty_other_nonempty(self):
        assert not _units_equal({}, {"m": 1})

    def test_fractional_exponents(self):
        assert _units_equal({"m": Fraction(1, 2)}, {"m": Fraction(1, 2)})


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
# Constraint base class
# ---------------------------------------------------------------------------


class TestConstraintBase:
    def test_initial_local_is_none(self):
        assert Constraint().initial_local("E") is None

    def test_initial_global_is_none(self):
        assert Constraint().initial_global() is None

    def test_allows_returns_true(self):
        c = Constraint()
        slot = _make_slot(local=None)
        assert c.allows(slot, Rule("E", ["x"]), None) is True

    def test_update_propagates_local_to_nt_children(self):
        c = Constraint()
        nts = frozenset({"E", "F"})
        slot = _make_slot(local=99, nonterminals=nts)
        rule = Rule("E", ["E", "+", "F"])
        child_locals, new_global = c.update(slot, rule, "g")
        assert child_locals == [99, 99]
        assert new_global == "g"

    def test_update_terminal_only_returns_empty_list(self):
        c = Constraint()
        slot = _make_slot(local=7, nonterminals=frozenset({"E"}))
        child_locals, new_global = c.update(slot, Rule("E", ["x", "+", "y"]), None)
        assert child_locals == []

    def test_update_global_unchanged(self):
        c = Constraint()
        slot = _make_slot(local=None, nonterminals=frozenset({"E"}))
        _, new_global = c.update(slot, Rule("E", ["x"]), "original")
        assert new_global == "original"

    def test_nonterminals_is_none_by_default(self):
        assert Constraint.nonterminals is None

    def test_rule_names_is_none_by_default(self):
        assert Constraint.rule_names is None


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
# DimensionalConsistency._classify_rule — named paths
# ---------------------------------------------------------------------------


class TestDCClassifyRuleNamed:
    def _classify(self, rule: Rule, nts: frozenset[str] | None = None):
        if nts is None:
            nts = frozenset({"E", "F", "B", "T", "R", "P", "V", "C"})
        return _dc()._classify_rule(rule, nts)

    def test_e_add_is_binary(self):
        assert self._classify(Rule("E", ["E", "+", "F"], name="E_add_+")) == ("binary", "+")

    def test_f_mul_is_binary(self):
        assert self._classify(Rule("F", ["F", "*", "B"], name="F_mul_*")) == ("binary", "*")

    def test_b_pow_is_binary(self):
        assert self._classify(Rule("B", ["B", "^", "T"], name="B_pow_^")) == ("binary", "^")

    def test_e_to_f_is_chain(self):
        assert self._classify(Rule("E", ["F"], name="E_to_F")) == ("chain", None)

    def test_f_to_b_is_chain(self):
        assert self._classify(Rule("F", ["B"], name="F_to_B")) == ("chain", None)

    def test_b_to_t_is_chain(self):
        assert self._classify(Rule("B", ["T"], name="B_to_T")) == ("chain", None)

    def test_t_to_r_is_chain(self):
        assert self._classify(Rule("T", ["R"], name="T_to_R")) == ("chain", None)

    def test_t_to_k_is_chain(self):
        assert self._classify(Rule("T", ["K"], name="T_to_K")) == ("chain", None)

    def test_t_to_v_is_chain(self):
        assert self._classify(Rule("T", ["V"], name="T_to_V")) == ("chain", None)

    def test_r_to_p_is_chain(self):
        assert self._classify(Rule("R", ["P"], name="R_to_P")) == ("chain", None)

    def test_r_fn_sin_is_function(self):
        assert self._classify(Rule("R", ["sin", "(", "P", ")"], name="R_fn_sin")) == ("function", "sin")

    def test_r_fn_sqrt_is_function(self):
        assert self._classify(Rule("R", ["sqrt", "(", "P", ")"], name="R_fn_sqrt")) == ("function", "sqrt")

    def test_p_postfix_is_postfix(self):
        assert self._classify(Rule("P", ["T", "^2"], name="P_^2")) == ("postfix", "^2")

    def test_v_leaf(self):
        kind, info = self._classify(Rule("V", ["x"], name="V_x"))
        assert kind == "leaf"
        assert "x" in info

    def test_k_leaf(self):
        kind, info = self._classify(Rule("K", ["C_const"], name="K_C_const"))
        assert kind == "leaf"

    def test_unrecognized_name_falls_to_fallback_terminal(self):
        # Name doesn't match any known prefix → fallback; nt=0 → leaf
        kind, _ = self._classify(Rule("E", ["x"], name="MyCustomRule"))
        assert kind == "leaf"


# ---------------------------------------------------------------------------
# DimensionalConsistency._classify_rule — fallback paths
# ---------------------------------------------------------------------------


class TestDCClassifyRuleFallback:
    def _classify(self, rule: Rule, nts: frozenset[str] | None = None, sl=None):
        if nts is None:
            nts = frozenset({"E", "F"})
        return _dc(symbol_library=sl)._classify_rule(rule, nts)

    def test_nt0_is_leaf(self):
        kind, terms = self._classify(Rule("E", ["x", "y"]))
        assert kind == "leaf"
        assert "x" in terms and "y" in terms

    def test_nt1_no_terms_is_chain(self):
        assert self._classify(Rule("E", ["F"])) == ("chain", None)

    def test_nt2_one_term_is_binary(self):
        assert self._classify(Rule("E", ["E", "+", "F"])) == ("binary", "+")

    def test_nt1_transcendental_fn_is_function(self):
        kind, fn = self._classify(Rule("E", ["sin", "F"]))
        assert kind == "function"
        assert fn == "sin"

    def test_nt1_unit_preserving_fn_is_function(self):
        kind, fn = self._classify(Rule("E", ["abs", "F"]))
        assert kind == "function"
        assert fn == "abs"

    def test_nt1_sqrt_fn_is_function(self):
        kind, fn = self._classify(Rule("E", ["sqrt", "F"]))
        assert kind == "function"
        assert fn == "sqrt"

    def test_nt1_cbrt_fn_is_function(self):
        kind, fn = self._classify(Rule("E", ["cbrt", "F"]))
        assert kind == "function"
        assert fn == "cbrt"

    def test_nt1_caret_term_is_postfix(self):
        kind, info = self._classify(Rule("E", ["F", "^2"]))
        assert kind == "postfix"
        assert info == "^2"

    def test_nt1_multiple_terms_one_caret_is_postfix(self):
        kind, info = self._classify(Rule("E", ["x", "F", "^2"]))
        assert kind == "postfix"
        assert info == "^2"

    def test_nt1_sl_fn_type_is_function(self):
        sl = _sl("sin", "+")
        kind, fn = self._classify(Rule("E", ["sin", "F"]), sl=sl)
        assert kind == "function"
        assert fn == "sin"

    def test_nt1_unknown_term_is_unknown(self):
        kind, _ = self._classify(Rule("E", ["F", "someop"]))
        assert kind == "unknown"

    def test_nt2_no_terms_is_unknown(self):
        kind, _ = self._classify(Rule("E", ["E", "F"]))
        assert kind == "unknown"

    def test_nt1_multiple_terms_no_caret_is_unknown(self):
        kind, _ = self._classify(Rule("E", ["a", "F", "b"]))
        assert kind == "unknown"

    def test_nt2_two_terms_is_unknown(self):
        kind, _ = self._classify(Rule("E", ["E", "+", "-", "F"]))
        assert kind == "unknown"

    def test_parens_excluded_from_terms(self):
        # Rule E -> ( F ) — after filtering parens, nt=1 and terms=[] → chain
        kind, _ = self._classify(Rule("E", ["(", "F", ")"]))
        assert kind == "chain"


# ---------------------------------------------------------------------------
# DimensionalConsistency._is_additive
# ---------------------------------------------------------------------------


class TestDCIsAdditive:
    def test_plus_with_sl_is_additive(self):
        sl = _sl("+", "*")
        dc = _dc(symbol_library=sl)
        assert dc._is_additive("+") is True

    def test_minus_with_sl_is_additive(self):
        sl = _sl("+", "-")
        dc = _dc(symbol_library=sl)
        assert dc._is_additive("-") is True

    def test_mul_with_sl_not_additive(self):
        sl = _sl("+", "*")
        dc = _dc(symbol_library=sl)
        assert dc._is_additive("*") is False

    def test_div_with_sl_not_additive(self):
        sl = _sl("+", "/")
        dc = _dc(symbol_library=sl)
        assert dc._is_additive("/") is False

    def test_unknown_op_with_sl_not_additive(self):
        sl = _sl("+")
        dc = _dc(symbol_library=sl)
        assert dc._is_additive("??") is False

    def test_plus_without_sl_is_additive(self):
        assert _dc()._is_additive("+") is True

    def test_minus_without_sl_is_additive(self):
        assert _dc()._is_additive("-") is True

    def test_mul_without_sl_not_additive(self):
        assert _dc()._is_additive("*") is False

    def test_caret_without_sl_not_additive(self):
        assert _dc()._is_additive("^") is False


# ---------------------------------------------------------------------------
# DimensionalConsistency._fn_child_unit
# ---------------------------------------------------------------------------


class TestDCFnChildUnit:
    def test_transcendental_always_returns_dimensionless(self):
        dc = _dc()
        for fn in list(TRANSCENDENTAL_FNS)[:5]:
            assert dc._fn_child_unit(fn, {"m": Fraction(1)}) == {}

    def test_transcendental_with_none_parent_still_dimensionless(self):
        assert _dc()._fn_child_unit("sin", None) == {}

    def test_sqrt_with_none_parent_returns_none(self):
        assert _dc()._fn_child_unit("sqrt", None) is None

    def test_sqrt_doubles_exponents(self):
        result = _dc()._fn_child_unit("sqrt", {"m": Fraction(1, 2)})
        assert result == {"m": Fraction(1)}

    def test_sqrt_with_multidim_unit(self):
        result = _dc()._fn_child_unit("sqrt", {"m": Fraction(1), "s": Fraction(-2)})
        assert result == {"m": Fraction(2), "s": Fraction(-4)}

    def test_cbrt_with_none_parent_returns_none(self):
        assert _dc()._fn_child_unit("cbrt", None) is None

    def test_cbrt_triples_exponents(self):
        result = _dc()._fn_child_unit("cbrt", {"m": Fraction(1, 3)})
        assert result == {"m": Fraction(1)}

    def test_unit_preserving_returns_same_unit(self):
        unit = {"m": Fraction(1), "s": Fraction(-1)}
        result = _dc()._fn_child_unit("abs", unit)
        assert result == unit

    def test_unit_preserving_with_none(self):
        assert _dc()._fn_child_unit("abs", None) is None

    def test_unknown_fn_returns_none(self):
        assert _dc()._fn_child_unit("myfn", {"m": Fraction(1)}) is None


# ---------------------------------------------------------------------------
# DimensionalConsistency._leaf_compatible
# ---------------------------------------------------------------------------


class TestDCLeafCompatible:
    def test_var_unit_matches(self):
        dc = _dc(variable_units={"v": {"m": 1, "s": -1}})
        assert dc._leaf_compatible(["v"], {"m": Fraction(1), "s": Fraction(-1)}) is True

    def test_var_unit_mismatches(self):
        dc = _dc(variable_units={"v": {"m": 1, "s": -1}})
        assert dc._leaf_compatible(["v"], {"s": Fraction(1)}) is False

    def test_const_units_match(self):
        dc = _dc(constant_units={"g": {"m": 1, "s": -2}})
        assert dc._leaf_compatible(["g"], {"m": Fraction(1), "s": Fraction(-2)}) is True

    def test_const_units_mismatch(self):
        dc = _dc(constant_units={"g": {"m": 1, "s": -2}})
        assert dc._leaf_compatible(["g"], {"m": Fraction(1)}) is False

    def test_sl_const_allow_poly_false_dimensionless_required(self):
        sl = _sl("C")
        dc = _dc(symbol_library=sl, allow_unit_polymorphic_constants=False)
        assert dc._leaf_compatible(["C"], {}) is True

    def test_sl_const_allow_poly_false_nondimensionless_required(self):
        sl = _sl("C")
        dc = _dc(symbol_library=sl, allow_unit_polymorphic_constants=False)
        assert dc._leaf_compatible(["C"], {"m": Fraction(1)}) is False

    def test_sl_const_allow_poly_true_any_required(self):
        sl = _sl("C")
        dc = _dc(symbol_library=sl, allow_unit_polymorphic_constants=True)
        assert dc._leaf_compatible(["C"], {"m": Fraction(1), "s": Fraction(-2)}) is True

    def test_sl_lit_dimensionless_required(self):
        sl = _sl("pi")
        dc = _dc(symbol_library=sl)
        assert dc._leaf_compatible(["pi"], {}) is True

    def test_sl_lit_nondimensionless_required(self):
        sl = _sl("pi")
        dc = _dc(symbol_library=sl)
        assert dc._leaf_compatible(["pi"], {"m": Fraction(1)}) is False

    def test_sl_var_always_true(self):
        sl = _sl(n_vars=1)
        dc = _dc(symbol_library=sl)
        assert dc._leaf_compatible(["X_0"], {"m": Fraction(5)}) is True

    def test_unknown_token_conservatively_true(self):
        dc = _dc()
        assert dc._leaf_compatible(["unknown_token"], {"m": Fraction(1)}) is True


# ---------------------------------------------------------------------------
# DimensionalConsistency._child_units
# ---------------------------------------------------------------------------


class TestDCChildUnits:
    NTS = frozenset({"E", "F"})

    def test_terminal_only_returns_empty(self):
        dc = _dc()
        assert dc._child_units(Rule("E", ["x"]), {"m": Fraction(1)}, self.NTS) == []

    def test_chain_propagates_required(self):
        dc = _dc()
        required = {"m": Fraction(1)}
        result = dc._child_units(Rule("E", ["F"]), required, self.NTS)
        assert result == [required]

    def test_additive_binary_both_get_required(self):
        dc = _dc()
        required = {"m": Fraction(1), "s": Fraction(-1)}
        result = dc._child_units(Rule("E", ["E", "+", "F"]), required, self.NTS)
        assert result == [required, required]

    def test_multiplicative_binary_both_get_none(self):
        dc = _dc()
        result = dc._child_units(Rule("E", ["E", "*", "F"]), {"m": Fraction(1)}, self.NTS)
        assert result == [None, None]

    def test_function_transcendental_child_gets_dimensionless(self):
        dc = _dc()
        result = dc._child_units(Rule("E", ["sin", "F"]), {}, self.NTS)
        assert result == [{}]

    def test_function_sqrt_child_gets_doubled(self):
        dc = _dc()
        required = {"m": Fraction(1, 2)}
        result = dc._child_units(Rule("R", ["sqrt", "F"], name="R_fn_sqrt"), required, self.NTS)
        assert result == [{"m": Fraction(1)}]

    def test_postfix_named_inverses_exponent(self):
        dc = _dc()
        required = {"m": Fraction(1)}
        rule = Rule("P", ["F", "^2"], name="P_^2")
        result = dc._child_units(rule, required, self.NTS)
        assert result == [{"m": Fraction(1, 2)}]

    def test_postfix_zero_exponent_returns_none(self):
        dc = _dc()
        rule = Rule("P", ["F", "^0"], name="P_^0")
        result = dc._child_units(rule, {"m": Fraction(1)}, self.NTS)
        assert result == [None]

    def test_postfix_unparseable_exponent_returns_none(self):
        dc = _dc()
        rule = Rule("P", ["F", "^abc"], name="P_^abc")
        result = dc._child_units(rule, {"m": Fraction(1)}, self.NTS)
        assert result == [None]

    def test_postfix_required_none_returns_none(self):
        dc = _dc()
        rule = Rule("P", ["F", "^2"], name="P_^2")
        assert dc._child_units(rule, None, self.NTS) == [None]

    def test_unknown_kind_returns_none_per_child(self):
        dc = _dc()
        # nt=2, no terms → unknown; 2 NT children
        result = dc._child_units(Rule("E", ["E", "F"]), {"m": Fraction(1)}, self.NTS)
        assert result == [None, None]


# ---------------------------------------------------------------------------
# DimensionalConsistency.allows
# ---------------------------------------------------------------------------


class TestDCAllows:
    NTS = frozenset({"E", "F"})

    def test_slot_local_none_always_true(self):
        dc = _dc(variable_units={"t": {"s": 1}}, target_unit={"m": 1})
        slot = _make_slot(local=None, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("F", ["t"]), None) is True

    def test_transcendental_fn_with_dimensionless_required_true(self):
        dc = _dc()
        slot = _make_slot(local={}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("E", ["sin", "F"], name="R_fn_sin"), None) is True

    def test_transcendental_fn_with_nondimensionless_required_false(self):
        dc = _dc()
        slot = _make_slot(local={"m": Fraction(1)}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("E", ["sin", "F"], name="R_fn_sin"), None) is False

    def test_non_transcendental_fn_always_true(self):
        dc = _dc()
        slot = _make_slot(local={"m": Fraction(1)}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("E", ["abs", "F"], name="R_fn_abs"), None) is True

    def test_leaf_compatible_variable_true(self):
        dc = _dc(variable_units={"v": {"m": 1, "s": -1}})
        slot = _make_slot(local={"m": Fraction(1), "s": Fraction(-1)}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("F", ["v"]), None) is True

    def test_leaf_incompatible_variable_false(self):
        dc = _dc(variable_units={"v": {"m": 1, "s": -1}})
        slot = _make_slot(local={"s": Fraction(1)}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("F", ["v"]), None) is False

    def test_chain_rule_always_true(self):
        dc = _dc(variable_units={}, target_unit={"m": 1})
        slot = _make_slot(local={"m": Fraction(1)}, nonterminals=self.NTS)
        assert dc.allows(slot, Rule("E", ["F"], name="E_to_F"), None) is True


# ---------------------------------------------------------------------------
# DimensionalConsistency.update
# ---------------------------------------------------------------------------


class TestDCUpdate:
    NTS = frozenset({"E", "F"})

    def test_chain_propagates_required(self):
        dc = _dc()
        required = {"m": Fraction(1)}
        slot = _make_slot(local=required, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["F"], name="E_to_F"), None)
        assert child_locals == [required]

    def test_additive_binary_gives_required_to_both(self):
        dc = _dc()
        required = {"m": Fraction(1), "s": Fraction(-1)}
        slot = _make_slot(local=required, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["E", "+", "F"]), None)
        assert child_locals == [required, required]

    def test_multiplicative_binary_gives_none_to_both(self):
        dc = _dc()
        slot = _make_slot(local={"m": Fraction(1)}, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["E", "*", "F"]), None)
        assert child_locals == [None, None]

    def test_function_sin_gives_dimensionless_to_child(self):
        dc = _dc()
        slot = _make_slot(local={}, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["sin", "F"]), None)
        assert child_locals == [{}]

    def test_terminal_rule_gives_empty_list(self):
        dc = _dc()
        slot = _make_slot(local={"m": Fraction(1)}, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["x"]), None)
        assert child_locals == []

    def test_global_always_returns_none(self):
        dc = _dc()
        slot = _make_slot(local={}, nonterminals=self.NTS)
        _, new_global = dc.update(slot, Rule("E", ["F"], name="E_to_F"), None)
        assert new_global is None

    def test_slot_local_none_propagates_none(self):
        dc = _dc()
        slot = _make_slot(local=None, nonterminals=self.NTS)
        child_locals, _ = dc.update(slot, Rule("E", ["E", "+", "F"]), None)
        assert child_locals == [None, None]


# ---------------------------------------------------------------------------
# DimensionalConsistency integration
# ---------------------------------------------------------------------------


class TestDCIntegration:
    def test_full_derivation_respects_target_unit(self):
        g = Grammar()
        g.add_rule(Rule("E", ["E", "+", "F"], weight=0.5, name="E_add_+"))
        g.add_rule(Rule("E", ["F"], weight=0.5, name="E_to_F"))
        g.add_rule(Rule("F", ["v"], weight=0.5, name="F_v"))
        g.add_rule(Rule("F", ["t"], weight=0.5, name="F_t"))
        dc = DimensionalConsistency(
            variable_units={"v": {"m": 1, "s": -1}, "t": {"s": 1}},
            target_unit={"m": 1, "s": -1},
        )
        g.add_constraint(dc)
        for _ in range(20):
            tokens = g.start_derivation("E").generate(limit=100)
            assert "t" not in tokens

    def test_allow_unit_polymorphic_constants(self):
        g = Grammar()
        g.add_rule(Rule("E", ["F"], name="E_to_F"))
        g.add_rule(Rule("F", ["C"], name="F_C"))
        g.add_rule(Rule("F", ["v"], name="F_v"))
        sl = _sl("C")
        dc = DimensionalConsistency(
            variable_units={"v": {"m": 1}},
            target_unit={"m": 1},
            symbol_library=sl,
            allow_unit_polymorphic_constants=True,
        )
        g.add_constraint(dc)
        for _ in range(10):
            tokens = g.start_derivation("E").generate(limit=50)
            assert tokens in [["C"], ["v"]]

    def test_constant_units_declared_constant_behaves_like_variable(self):
        g = Grammar()
        g.add_rule(Rule("E", ["F"], name="E_to_F"))
        g.add_rule(Rule("F", ["g"], name="F_g"))  # g = 9.81 m/s²
        g.add_rule(Rule("F", ["t"], name="F_t"))  # t in seconds
        dc = DimensionalConsistency(
            variable_units={"t": {"s": 1}},
            target_unit={"m": 1, "s": -2},
            constant_units={"g": {"m": 1, "s": -2}},
        )
        g.add_constraint(dc)
        for _ in range(10):
            tokens = g.start_derivation("E").generate(limit=50)
            assert tokens == ["g"]

    def test_multiple_constraints_compose(self):
        """MaxDepth and DimensionalConsistency together restrict generation correctly."""
        g = Grammar()
        g.add_rule(Rule("E", ["E", "+", "F"], name="E_add_+"))
        g.add_rule(Rule("E", ["F"], name="E_to_F"))
        g.add_rule(Rule("F", ["v"], name="F_v"))
        g.add_rule(Rule("F", ["t"], name="F_t"))
        dc = DimensionalConsistency(
            variable_units={"v": {"m": 1, "s": -1}, "t": {"s": 1}},
            target_unit={"m": 1, "s": -1},
        )
        g.add_constraint(dc)
        g.add_constraint(MaxDepth(10))
        for _ in range(20):
            tokens = g.start_derivation("E").generate(limit=500)
            assert "t" not in tokens
            assert all(tok in {"v", "+"} for tok in tokens)


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
            Constraint.from_dict({})


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


class TestDimensionalConsistencySerialization:
    def test_roundtrip(self):
        c = DimensionalConsistency(
            variable_units={"v": {"m": 1, "s": -1}, "t": {"s": 1}},
            target_unit={"m": 1, "s": -1},
            constant_units={"g": {"m": 1, "s": -2}},
            allow_unit_polymorphic_constants=True,
        )
        d = c.to_dict()
        c2 = DimensionalConsistency.from_dict(d)
        from fractions import Fraction

        assert c2._var_units == {"v": {"m": Fraction(1), "s": Fraction(-1)}, "t": {"s": Fraction(1)}}
        assert c2._target == {"m": Fraction(1), "s": Fraction(-1)}
        assert c2._const_units == {"g": {"m": Fraction(1), "s": Fraction(-2)}}
        assert c2._allow_poly_const is True
        assert c2._sl is None

    def test_roundtrip_no_constant_units(self):
        c = DimensionalConsistency(
            variable_units={"x": {"m": 1}},
            target_unit={"m": 1},
        )
        d = c.to_dict()
        c2 = DimensionalConsistency.from_dict(d)
        assert c2._const_units == {}

    def test_fractions_serialized_as_strings(self):
        c = DimensionalConsistency(
            variable_units={"x": {"m": 1}},
            target_unit={"m": 1},
        )
        d = c.to_dict()
        assert d["target_unit"]["m"] == "1"

    def test_constraint_class_key(self):
        c = DimensionalConsistency(variable_units={}, target_unit={})
        assert "DimensionalConsistency" in c.to_dict()["constraint_class"]


class TestConstraintFromDict:
    def test_dispatch_max_depth(self):
        d = MaxDepth(4).to_dict()
        c = Constraint.from_dict(d)
        assert isinstance(c, MaxDepth)
        assert c.limit == 4

    def test_dispatch_max_nodes(self):
        c = Constraint.from_dict(MaxNodes(20).to_dict())
        assert isinstance(c, MaxNodes)
        assert c.limit == 20

    def test_dispatch_max_occurrences(self):
        c = Constraint.from_dict(MaxOccurrences("cos", 2).to_dict())
        assert isinstance(c, MaxOccurrences)
        assert c.symbol == "cos"

    def test_dispatch_no_nested(self):
        c = Constraint.from_dict(NoNested(["sin"]).to_dict())
        assert isinstance(c, NoNested)

    def test_dispatch_dimensional_consistency(self):
        original = DimensionalConsistency(variable_units={"x": {"m": 1}}, target_unit={"m": 1})
        c = Constraint.from_dict(original.to_dict())
        assert isinstance(c, DimensionalConsistency)

    def test_dispatch_user_defined(self):
        """Constraint.from_dict resolves classes via importlib, including user-defined ones."""

        class _AlwaysAllow(Constraint):
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
        c = Constraint.from_dict(d)
        assert isinstance(c, _AlwaysAllow)

    def test_subclass_without_override_raises(self):
        class _NoFromDict(Constraint):
            pass

        with pytest.raises(NotImplementedError, match="_NoFromDict"):
            _NoFromDict.from_dict({})
