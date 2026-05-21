"""
Tests for SRToolkit/utils/grammar/grammar.py.

Coverage target: 100% of Rule, ParseTreeNode, ParseTree, and Grammar
(including all branches of Grammar.from_symbol_library).
"""

from __future__ import annotations

import pytest

from SRToolkit.utils.grammar import (
    Grammar,
    MaxDepth,
    MaxOccurrences,
    ParseTree,
    ParseTreeNode,
    Rule,
)
from SRToolkit.utils.grammar.derivation import Derivation
from SRToolkit.utils.symbol_library import SymbolLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _leaf(symbol: str) -> ParseTreeNode:
    """Terminal leaf node (no rule applied, no children)."""
    return ParseTreeNode(symbol, None)


def _simple_grammar() -> Grammar:
    """E -> E + E | x  (flat, left-recursive)"""
    return Grammar(
        [
            Rule("E", ["E", "+", "E"], name="E_add_+"),
            Rule("E", ["x"], name="E_x"),
        ]
    )


def _two_level_grammar() -> Grammar:
    """E -> E + F | F;  F -> x | y"""
    return Grammar(
        [
            Rule("E", ["E", "+", "F"], name="E_add_+"),
            Rule("E", ["F"], name="E_to_F"),
            Rule("F", ["x"], name="F_x"),
            Rule("F", ["y"], name="F_y"),
        ]
    )


def _sl(*symbols: str, n_vars: int = 1) -> SymbolLibrary:
    return SymbolLibrary.from_symbol_list(list(symbols), n_vars)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


class TestRule:
    def test_default_weight_is_one(self):
        assert Rule("E", ["x"]).weight == 1.0

    def test_default_name_is_none(self):
        assert Rule("E", ["x"]).name is None

    def test_lhs_and_rhs(self):
        r = Rule("S", ["a", "b", "c"])
        assert r.lhs == "S"
        assert r.rhs == ["a", "b", "c"]

    def test_custom_weight_and_name(self):
        r = Rule("E", ["x"], weight=0.4, name="leaf")
        assert r.weight == pytest.approx(0.4)
        assert r.name == "leaf"

    def test_empty_rhs_is_valid(self):
        """Epsilon production is representable."""
        r = Rule("E", [])
        assert r.rhs == []

    def test_equality_same_fields(self):
        assert Rule("E", ["x"]) == Rule("E", ["x"])

    def test_inequality_different_weight(self):
        assert Rule("E", ["x"], weight=1.0) != Rule("E", ["x"], weight=0.5)

    def test_inequality_different_lhs(self):
        assert Rule("E", ["x"]) != Rule("F", ["x"])

    def test_inequality_different_rhs(self):
        assert Rule("E", ["x"]) != Rule("E", ["y"])

    def test_inequality_different_name(self):
        """Rule is a dataclass: all fields including name participate in __eq__."""
        assert Rule("E", ["x"], name="a") != Rule("E", ["x"], name="b")

    def test_rule_is_mutable(self):
        """Dataclasses are mutable by default; rhs list can be mutated in place."""
        r = Rule("E", ["x"])
        r.rhs.append("y")
        assert r.rhs == ["x", "y"]


# ---------------------------------------------------------------------------
# ParseTreeNode
# ---------------------------------------------------------------------------


class TestParseTreeNode:
    def test_terminal_leaf(self):
        node = _leaf("x")
        assert node.symbol == "x"
        assert node.rule_applied is None
        assert node.children == []

    def test_internal_node_stores_rule(self):
        r = Rule("E", ["x"])
        child = _leaf("x")
        node = ParseTreeNode("E", r, [child])
        assert node.symbol == "E"
        assert node.rule_applied is r
        assert node.children == [child]

    def test_default_children_is_independent_per_instance(self):
        """field(default_factory=list) means each instance gets its own list."""
        n1 = ParseTreeNode("E", None)
        n2 = ParseTreeNode("E", None)
        n1.children.append(_leaf("x"))
        assert n2.children == []

    def test_children_can_be_provided(self):
        child = _leaf("x")
        node = ParseTreeNode("E", None, [child])
        assert node.children == [child]


# ---------------------------------------------------------------------------
# ParseTree
# ---------------------------------------------------------------------------


class TestParseTree:
    # --- to_token_list ---

    def test_single_terminal_root(self):
        """A leaf node with no children returns its own symbol."""
        assert ParseTree(_leaf("x")).to_token_list() == ["x"]

    def test_flat_children_all_terminals(self):
        r = Rule("E", ["a", "+", "b"])
        root = ParseTreeNode("E", r, [_leaf("a"), _leaf("+"), _leaf("b")])
        assert ParseTree(root).to_token_list() == ["a", "+", "b"]

    def test_nested_left_right_order(self):
        """Left subtree tokens precede right subtree tokens."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        r_y = Rule("E", ["y"])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_y, [_leaf("y")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert ParseTree(root).to_token_list() == ["x", "+", "y"]

    def test_deeply_nested(self):
        """(x + y) + z → tokens in left-to-right surface order."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x, r_y, r_z = Rule("E", ["x"]), Rule("E", ["y"]), Rule("E", ["z"])
        leaf_x = ParseTreeNode("E", r_x, [_leaf("x")])
        leaf_y = ParseTreeNode("E", r_y, [_leaf("y")])
        leaf_z = ParseTreeNode("E", r_z, [_leaf("z")])
        inner = ParseTreeNode("E", r_add, [leaf_x, _leaf("+"), leaf_y])
        root = ParseTreeNode("E", r_add, [inner, _leaf("+"), leaf_z])
        assert ParseTree(root).to_token_list() == ["x", "+", "y", "+", "z"]

    def test_function_application(self):
        """sin(x) produces tokens in surface order including parentheses."""
        r_fn = Rule("E", ["sin", "(", "E", ")"])
        r_x = Rule("E", ["x"])
        inner = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_fn, [_leaf("sin"), _leaf("("), inner, _leaf(")")])
        assert ParseTree(root).to_token_list() == ["sin", "(", "x", ")"]

    # --- productions_used ---

    def test_single_production(self):
        r = Rule("E", ["x"])
        root = ParseTreeNode("E", r, [_leaf("x")])
        assert ParseTree(root).productions_used() == [r]

    def test_leaf_only_has_no_productions(self):
        """A pure terminal leaf (rule_applied=None) contributes no productions."""
        assert ParseTree(_leaf("x")).productions_used() == []

    def test_preorder_traversal_order(self):
        """Root rule first, then left subtree, then right subtree."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        r_y = Rule("E", ["y"])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_y, [_leaf("y")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert ParseTree(root).productions_used() == [r_add, r_x, r_y]

    def test_terminal_siblings_not_included(self):
        """Terminal children appear in token list but not in productions_used."""
        r = Rule("E", ["x", "+", "y"])
        root = ParseTreeNode("E", r, [_leaf("x"), _leaf("+"), _leaf("y")])
        assert ParseTree(root).productions_used() == [r]

    def test_three_levels_preorder(self):
        """Productions at three nesting levels come out in correct pre-order."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        # Build ((x + x) + x)
        leaf1 = ParseTreeNode("E", r_x, [_leaf("x")])
        leaf2 = ParseTreeNode("E", r_x, [_leaf("x")])
        leaf3 = ParseTreeNode("E", r_x, [_leaf("x")])
        inner = ParseTreeNode("E", r_add, [leaf1, _leaf("+"), leaf2])
        root = ParseTreeNode("E", r_add, [inner, _leaf("+"), leaf3])
        # Pre-order: root add, inner add, x, x, x
        assert ParseTree(root).productions_used() == [r_add, r_add, r_x, r_x, r_x]


# ---------------------------------------------------------------------------
# Grammar.__init__ and initial state
# ---------------------------------------------------------------------------


class TestGrammarInit:
    def test_empty_grammar(self):
        g = Grammar()
        assert g.nonterminals == set()
        assert g.start is None

    def test_with_initial_rules(self):
        rules = [Rule("E", ["x"]), Rule("E", ["y"])]
        g = Grammar(rules)
        assert g.rules_for("E") == rules

    def test_with_start(self):
        g = Grammar(start="E")
        assert g.start == "E"

    def test_rules_none_equivalent_to_empty(self):
        g = Grammar(rules=None)
        assert g.nonterminals == set()

    def test_initial_rules_applied_in_insertion_order(self):
        r1, r2 = Rule("E", ["x"]), Rule("E", ["y"])
        g = Grammar([r1, r2])
        assert g.rules_for("E") == [r1, r2]


# ---------------------------------------------------------------------------
# Grammar.nonterminals
# ---------------------------------------------------------------------------


class TestGrammarNonterminals:
    def test_empty_grammar_empty_set(self):
        assert Grammar().nonterminals == set()

    def test_lhs_symbols_only(self):
        g = Grammar([Rule("E", ["F", "x"]), Rule("F", ["y"])])
        assert g.nonterminals == {"E", "F"}

    def test_rhs_only_symbols_excluded(self):
        """'x' appears only in rhs and must not appear in nonterminals."""
        g = Grammar([Rule("E", ["x"])])
        assert "x" not in g.nonterminals

    def test_returns_set_type(self):
        g = Grammar([Rule("E", ["x"]), Rule("F", ["y"])])
        assert isinstance(g.nonterminals, set)

    def test_grows_as_rules_added(self):
        g = Grammar()
        assert g.nonterminals == set()
        g.add_rule(Rule("E", ["x"]))
        assert g.nonterminals == {"E"}
        g.add_rule(Rule("F", ["y"]))
        assert g.nonterminals == {"E", "F"}


# ---------------------------------------------------------------------------
# Grammar.add_rule
# ---------------------------------------------------------------------------


class TestGrammarAddRule:
    def test_adds_nonterminal(self):
        g = Grammar()
        g.add_rule(Rule("E", ["x"]))
        assert "E" in g.nonterminals

    def test_insertion_order_preserved(self):
        g = Grammar()
        r1, r2 = Rule("E", ["x"]), Rule("E", ["y"])
        g.add_rule(r1)
        g.add_rule(r2)
        assert g.rules_for("E") == [r1, r2]

    def test_multiple_nonterminals(self):
        g = Grammar()
        g.add_rule(Rule("E", ["F"]))
        g.add_rule(Rule("F", ["x"]))
        assert g.nonterminals == {"E", "F"}

    def test_duplicate_rule_is_kept(self):
        """Grammar does not deduplicate; both entries are stored."""
        r = Rule("E", ["x"])
        g = Grammar()
        g.add_rule(r)
        g.add_rule(r)
        assert len(g.rules_for("E")) == 2


# ---------------------------------------------------------------------------
# Grammar.rules_for
# ---------------------------------------------------------------------------


class TestGrammarRulesFor:
    def test_known_nonterminal_returns_rules(self):
        r = Rule("E", ["x"])
        g = Grammar([r])
        assert g.rules_for("E") == [r]

    def test_unknown_nonterminal_returns_empty_list(self):
        g = Grammar([Rule("E", ["x"])])
        assert g.rules_for("Z") == []

    def test_multiple_rules_in_insertion_order(self):
        r1, r2 = Rule("E", ["x"]), Rule("E", ["y"])
        g = Grammar([r1, r2])
        assert g.rules_for("E") == [r1, r2]

    def test_returns_copy_not_internal_list(self):
        """Mutating the returned list must not affect the grammar."""
        g = Grammar([Rule("E", ["x"])])
        g.rules_for("E").clear()
        assert len(g.rules_for("E")) == 1


# ---------------------------------------------------------------------------
# Grammar.is_pcfg
# ---------------------------------------------------------------------------


class TestGrammarIsPCFG:
    def test_all_default_weights_is_cfg(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        assert g.is_pcfg() is False

    def test_any_nondefault_weight_is_pcfg(self):
        g = Grammar([Rule("E", ["x"], weight=0.4), Rule("E", ["y"], weight=0.6)])
        assert g.is_pcfg() is True

    def test_empty_grammar_not_pcfg(self):
        assert Grammar().is_pcfg() is False

    def test_single_rule_with_nondefault_weight(self):
        assert Grammar([Rule("E", ["x"], weight=2.0)]).is_pcfg() is True

    def test_mixed_only_one_nondefault_suffices(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"], weight=0.7)])
        assert g.is_pcfg() is True


# ---------------------------------------------------------------------------
# Grammar.validate
# ---------------------------------------------------------------------------


class TestGrammarValidate:
    # --- round-trip: derivation-generated trees always verify ---

    def test_simple_valid_tree(self):
        r = Rule("E", ["x"])
        g = Grammar([r])
        root = ParseTreeNode("E", r, [_leaf("x")])
        assert g.validate(ParseTree(root)) is True

    def test_terminal_leaf_root(self):
        """Single terminal root with no rule applied is a valid leaf."""
        g = Grammar([Rule("E", ["x"])])
        assert g.validate(ParseTree(_leaf("x"))) is True

    def test_nonterminal_leaf_root_returns_false(self):
        """A non-terminal symbol as an unexpanded leaf must be rejected."""
        g = Grammar([Rule("E", ["x"])])
        assert g.validate(ParseTree(ParseTreeNode("E", None))) is False

    def test_nested_valid_tree(self):
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.validate(ParseTree(root)) is True

    def test_derivation_roundtrip_no_constraints(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        for _ in range(10):
            d = g.start_derivation("E")
            d.generate()
            assert g.validate(d.to_parse_tree()) is True

    def test_derivation_roundtrip_with_constraints(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"]), Rule("E", ["y"])])
        g.add_constraint(MaxDepth(3))
        g.add_constraint(MaxOccurrences("x", 2))
        for _ in range(15):
            d = g.start_derivation("E")
            d.generate(limit=500)
            assert g.validate(d.to_parse_tree()) is True

    # --- foreign / unknown rules ---

    def test_foreign_rule_returns_false(self):
        r_known = Rule("E", ["x"])
        r_foreign = Rule("E", ["y"])
        g = Grammar([r_known])
        root = ParseTreeNode("E", r_foreign, [_leaf("y")])
        assert g.validate(ParseTree(root)) is False

    def test_foreign_rule_in_nested_subtree_returns_false(self):
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        r_foreign = Rule("E", ["y"])
        g = Grammar([r_add, r_x])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_foreign, [_leaf("y")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.validate(ParseTree(root)) is False

    def test_different_weight_is_foreign(self):
        r_grammar = Rule("E", ["x"], weight=1.0)
        r_tree = Rule("E", ["x"], weight=0.5)
        g = Grammar([r_grammar])
        root = ParseTreeNode("E", r_tree, [_leaf("x")])
        assert g.validate(ParseTree(root)) is False

    # --- structural errors ---

    def test_wrong_child_count_returns_false(self):
        """Rule says 3 RHS symbols but node has only 1 child."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        root = ParseTreeNode("E", r_add, [_leaf("x")])
        assert g.validate(ParseTree(root)) is False

    def test_wrong_child_symbol_returns_false(self):
        """Child symbol does not match the corresponding RHS entry."""
        r = Rule("E", ["x"])
        g = Grammar([r])
        root = ParseTreeNode("E", r, [_leaf("y")])
        assert g.validate(ParseTree(root)) is False

    def test_unexpanded_nonterminal_leaf_returns_false(self):
        """An internal NT node with rule_applied=None (unexpanded) must be rejected."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        unexpanded = ParseTreeNode("E", None)
        root = ParseTreeNode("E", r_add, [unexpanded, _leaf("+"), unexpanded])
        assert g.validate(ParseTree(root)) is False

    def test_root_symbol_not_a_nonterminal_returns_false(self):
        """rule_applied on a root whose symbol is not in the grammar."""
        g = Grammar([Rule("E", ["x"])])
        r_bad = Rule("Z", ["x"])
        root = ParseTreeNode("Z", r_bad, [_leaf("x")])
        assert g.validate(ParseTree(root)) is False

    # --- constraint violations ---

    def test_constraint_violation_returns_false(self):
        r_add = Rule("E", ["E", "+", "E"], name="E_add_+")
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        g.add_constraint(MaxDepth(0))
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.validate(ParseTree(root)) is False

    def test_no_constraints_does_not_short_circuit(self):
        """Without constraints, structural and grammar checks still run."""
        r_known = Rule("E", ["x"])
        r_foreign = Rule("E", ["y"])
        g = Grammar([r_known])
        root = ParseTreeNode("E", r_foreign, [_leaf("y")])
        assert g.validate(ParseTree(root)) is False

    # --- require_start ---

    def test_require_start_accepts_matching_root(self):
        r = Rule("E", ["x"])
        g = Grammar([r], start="E")
        root = ParseTreeNode("E", r, [_leaf("x")])
        assert g.validate(ParseTree(root), require_start=True) is True

    def test_require_start_rejects_non_start_root(self):
        r_e = Rule("E", ["F"])
        r_f = Rule("F", ["x"])
        g = Grammar([r_e, r_f], start="E")
        root = ParseTreeNode("F", r_f, [_leaf("x")])
        assert g.validate(ParseTree(root), require_start=True) is False

    def test_require_start_raises_when_no_start_symbol(self):
        r = Rule("E", ["x"])
        g = Grammar([r])
        root = ParseTreeNode("E", r, [_leaf("x")])
        with pytest.raises(ValueError, match="no start symbol"):
            g.validate(ParseTree(root), require_start=True)


# ---------------------------------------------------------------------------
# Grammar.start_derivation
# ---------------------------------------------------------------------------


class TestGrammarStartDerivation:
    def test_explicit_start_returns_derivation(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert isinstance(d, Derivation)
        assert not d.complete

    def test_uses_grammar_start_attribute(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        d = g.start_derivation()
        assert not d.complete

    def test_no_start_anywhere_raises(self):
        g = Grammar([Rule("E", ["x"])])
        with pytest.raises(ValueError, match="[Ss]tart"):
            g.start_derivation()

    def test_unknown_start_symbol_raises(self):
        g = Grammar([Rule("E", ["x"])])
        with pytest.raises(ValueError, match="not a non-terminal"):
            g.start_derivation("Z")

    def test_generates_complete_expression(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        assert g.start_derivation().generate() == ["x"]

    def test_explicit_start_overrides_grammar_start(self):
        """Explicit argument takes precedence over self.start."""
        g = Grammar([Rule("E", ["x"]), Rule("F", ["y"])], start="E")
        assert g.start_derivation("F").generate() == ["y"]

    def test_derivation_starts_incomplete(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert not d.complete
        d.generate()
        assert d.complete


# ---------------------------------------------------------------------------
# Grammar.generate_one
# ---------------------------------------------------------------------------


class TestGrammarGenerateOne:
    def test_returns_token_list(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        assert g.generate_one() == ["x"]

    def test_returns_none_when_all_attempts_fail(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        result = g.generate_one(max_steps=0, max_retries=5)
        assert result is None

    def test_retries_succeed_eventually(self):
        """A grammar that terminates in ≤10 steps should succeed within retries."""
        g = Grammar([Rule("E", ["x"])], start="E")
        result = g.generate_one(max_steps=10, max_retries=3)
        assert result == ["x"]

    def test_unlimited_steps_never_raises(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        result = g.generate_one(max_steps=-1)
        assert result == ["x"]

    def test_unknown_start_raises_value_error(self):
        g = Grammar([Rule("E", ["x"])])
        with pytest.raises(ValueError):
            g.generate_one(start="Z")


# ---------------------------------------------------------------------------
# Grammar.from_symbol_library — hierarchy and branch coverage
# ---------------------------------------------------------------------------


class TestGrammarFromSymbolLibrary:
    # --- return type and basic properties ---

    def test_returns_grammar_instance(self):
        g = Grammar.from_symbol_library(_sl("+"))
        assert isinstance(g, Grammar)

    def test_default_start_is_E(self):
        g = Grammar.from_symbol_library(_sl("+"))
        assert g.start == "E"

    def test_custom_start_stored(self):
        g = Grammar.from_symbol_library(_sl("+"), start="Root")
        assert g.start == "Root"

    def test_is_pcfg(self):
        g = Grammar.from_symbol_library(_sl("+", "*", "sin"))
        assert g.is_pcfg() is True

    def test_none_symbol_library_uses_active(self):
        """Passing None falls back to SymbolLibrary.get_active()."""
        sl = _sl("+")
        with sl:
            g = Grammar.from_symbol_library(None)
        assert isinstance(g, Grammar)

    # --- E level (additive operators) ---

    def test_additive_op_produces_E_add_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+"))._rules}
        assert "E_add_+" in names

    def test_multiple_additive_ops(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "-"))._rules}
        assert "E_add_+" in names
        assert "E_add_-" in names

    def test_no_additive_ops_start_is_F(self):
        """Without additive operators the start NT is F (no E level generated)."""
        g = Grammar.from_symbol_library(_sl("*"))
        assert g.start == "F"
        assert g.rules_for("E") == []

    # --- F level (multiplicative operators) ---

    def test_multiplicative_op_produces_F_mul_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("*"))._rules}
        assert "F_mul_*" in names

    def test_multiple_multiplicative_ops(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("*", "/"))._rules}
        assert "F_mul_*" in names
        assert "F_mul_/" in names

    def test_no_multiplicative_ops_no_F_level(self):
        """Without multiplicative operators no F non-terminal is generated."""
        g = Grammar.from_symbol_library(_sl("+"))
        assert g.rules_for("F") == []

    # --- B level (power operators) ---

    def test_power_op_produces_B_pow_rule(self):
        """'^' is a precedence-2 OP and triggers the B_ops branch."""
        g = Grammar.from_symbol_library(_sl("^"))
        names = {r.name for r in g.rules_for("B")}
        assert "B_pow_^" in names
        assert "B_to_T" in names

    def test_no_power_ops_no_B_level(self):
        """Without power operators no B non-terminal is generated."""
        g = Grammar.from_symbol_library(_sl("+"))
        assert g.rules_for("B") == []

    # --- T level (leaf dispatcher) ---

    def test_T_to_V_present_when_variables_exist(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_V" in names

    def test_T_to_V_absent_without_variables(self):
        g = Grammar.from_symbol_library(_sl("C", n_vars=0))
        t_names = {r.name for r in g.rules_for("T")}
        assert "T_to_V" not in t_names
        assert "T_to_K" in t_names

    def test_T_to_K_dominant_without_variables(self):
        """K weight should be 0.7 when variables are absent (mirrors V's usual share)."""
        g = Grammar.from_symbol_library(_sl("C", n_vars=0))
        rules = {r.name: r for r in g.rules_for("T")}
        assert rules["T_to_K"].weight == pytest.approx(0.7)
        assert rules["T_to_R"].weight == pytest.approx(0.3)

    def test_T_to_K_present_when_constants_exist(self):
        """When CONST or LIT symbols are present, T -> K is added."""
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "C")).rules_for("T")}
        assert "T_to_K" in names

    def test_T_to_K_absent_without_constants(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_K" not in names

    def test_T_to_R_always_present(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_R" in names

    # --- K level (constants and literals) ---

    def test_only_const_K_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "C")).rules_for("K")}
        assert "K_C" in names

    def test_only_lit_K_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "pi")).rules_for("K")}
        assert "K_pi" in names

    def test_both_lit_and_const_K_rules(self):
        g = Grammar.from_symbol_library(_sl("+", "C", "pi"))
        names = {r.name for r in g.rules_for("K")}
        assert "K_C" in names
        assert "K_pi" in names

    def test_no_constants_no_K_rules(self):
        g = Grammar.from_symbol_library(_sl("+"))
        assert g.rules_for("K") == []

    # --- R level (prefix functions) ---

    def test_function_produces_R_fn_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("sin")).rules_for("R")}
        assert "R_fn_sin" in names

    def test_R_paren_always_present(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("R")}
        assert "R_paren" in names

    def test_postfix_produces_R_postfix_rule(self):
        """Postfix functions appear directly at R level as R_postfix_{sym}."""
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "^2")).rules_for("R")}
        assert "R_postfix_^2" in names

    def test_no_postfix_rule_absent(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("R")}
        assert not any(n and n.startswith("R_postfix_") for n in names)

    def test_no_R_fns_with_postfix_has_R_postfix_and_paren(self):
        """No prefix functions but postfix present: R_postfix and R_paren only."""
        g = Grammar.from_symbol_library(_sl("+", "^2"))
        r_names = {r.name for r in g.rules_for("R")}
        assert "R_postfix_^2" in r_names
        assert "R_paren" in r_names
        assert not any(n and n.startswith("R_fn_") for n in r_names)

    def test_no_R_fns_no_P_fns_only_R_paren(self):
        g = Grammar.from_symbol_library(_sl("+"))
        r_names = {r.name for r in g.rules_for("R")}
        assert r_names == {"R_paren"}

    # --- P non-terminal (removed — postfix rules now live at R level) ---

    def test_no_P_nonterminal(self):
        """P is no longer a non-terminal; postfix rules are added directly to R."""
        g = Grammar.from_symbol_library(_sl("+", "^2"))
        assert "P" not in g.nonterminals
        assert g.rules_for("P") == []

    # --- V level (variables) ---

    def test_variable_produces_V_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("V")}
        assert any(n and n.startswith("V_") for n in names)

    def test_multiple_variables(self):
        g = Grammar.from_symbol_library(_sl("+", n_vars=3))
        assert len(g.rules_for("V")) == 3

    # --- no terminal symbols ---

    def test_no_terminals_raises(self):
        """A library with only operators and no leaf symbols must raise."""
        with pytest.raises(ValueError, match="no variables, constants, or literals"):
            Grammar.from_symbol_library(_sl("+", n_vars=0))

    # --- end-to-end generation ---

    def test_can_generate_expression(self):
        sl = _sl("+", "*", "sin")
        g = Grammar.from_symbol_library(sl)
        tokens = g.start_derivation().generate(limit=1000)
        assert isinstance(tokens, list)
        assert len(tokens) > 0


# ---------------------------------------------------------------------------
# Grammar.from_nltk
# ---------------------------------------------------------------------------


class TestGrammarFromNLTK:
    def test_basic_parsing(self):
        g = Grammar.from_grammar_string("E -> E '+' F | F\nF -> 'x'", start="E")
        assert sorted(g.nonterminals) == ["E", "F"]
        assert g.rules_for("F")[0].rhs == ["x"]

    def test_missing_start_raises(self):
        with pytest.raises(ValueError, match="[Nn]o start symbol"):
            Grammar.from_grammar_string("E -> F\nF -> 'x'")

    def test_explicit_start(self):
        g = Grammar.from_grammar_string("E -> F\nF -> 'x'", start="F")
        assert g.start == "F"

    def test_weight_parsed(self):
        g = Grammar.from_grammar_string("E -> 'x' [0.4] | 'y' [0.6]", start="E")
        rules = g.rules_for("E")
        assert rules[0].weight == pytest.approx(0.4)
        assert rules[1].weight == pytest.approx(0.6)

    def test_unweighted_defaults_to_one(self):
        g = Grammar.from_grammar_string("E -> 'x' | 'y'", start="E")
        assert all(r.weight == 1.0 for r in g.rules_for("E"))

    def test_start_comment_parsed(self):
        g = Grammar.from_grammar_string("# start: E\nE -> 'x'")
        assert g.start == "E"
        assert len(g.rules_for("E")) == 1

    def test_explicit_start_overrides_comment(self):
        g = Grammar.from_grammar_string("# start: E\nE -> 'x'\nF -> 'y'", start="F")
        assert g.start == "F"

    def test_other_comment_lines_ignored(self):
        text = "# this is a comment\nE -> 'x'"
        g = Grammar.from_grammar_string(text, start="E")
        assert len(g.rules_for("E")) == 1

    def test_blank_lines_ignored(self):
        g = Grammar.from_grammar_string("\nE -> 'x'\n\nF -> 'y'\n", start="E")
        assert g.nonterminals == {"E", "F"}

    def test_lines_without_arrow_raises(self):
        with pytest.raises(ValueError, match="Expected '->'"):
            Grammar.from_grammar_string("just some text\nE -> 'x'", start="E")

    def test_unquoted_token_is_nonterminal(self):
        """Unquoted tokens in rhs are stored as-is and become nonterminals if they have rules."""
        g = Grammar.from_grammar_string("E -> F\nF -> 'x'", start="E")
        assert g.rules_for("E")[0].rhs == ["F"]
        assert "F" in g.nonterminals

    def test_quoted_terminal_stripped(self):
        """Single quotes are stripped from terminal tokens."""
        g = Grammar.from_grammar_string("E -> 'hello world'", start="E")
        # 'hello world' is a single quoted token → one symbol with a space in it
        assert g.rules_for("E")[0].rhs == ["hello world"]

    def test_multiple_rules_per_line(self):
        g = Grammar.from_grammar_string("E -> 'x' | 'y' | 'z'", start="E")
        assert len(g.rules_for("E")) == 3

    def test_names_are_none(self):
        g = Grammar.from_grammar_string("E -> 'x'", start="E")
        assert g.rules_for("E")[0].name is None

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="No rules parsed"):
            Grammar.from_grammar_string("", start="E")

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            Grammar.from_grammar_string("E -> 'x' [not_a_float]", start="E")

    def test_multiline_same_lhs(self):
        """The same LHS can appear on multiple lines; all rules are collected."""
        g = Grammar.from_grammar_string("E -> 'x'\nE -> 'y'", start="E")
        assert len(g.rules_for("E")) == 2

    def test_trailing_pipe_ignored(self):
        """A trailing '|' produces an empty alternative that is silently skipped."""
        g = Grammar.from_grammar_string("E -> 'x' | 'y' |", start="E")
        assert len(g.rules_for("E")) == 2

    def test_leading_pipe_ignored(self):
        """A leading '|' also produces an empty alternative that is silently skipped."""
        g = Grammar.from_grammar_string("E -> | 'x'", start="E")
        assert len(g.rules_for("E")) == 1


# ---------------------------------------------------------------------------
# Grammar.to_nltk
# ---------------------------------------------------------------------------


class TestGrammarToNLTK:
    def test_terminals_quoted(self):
        g = Grammar([Rule("E", ["+"])])
        assert "'+'" in g.to_grammar_string()

    def test_nonterminals_unquoted(self):
        g = Grammar([Rule("E", ["F"]), Rule("F", ["x"])])
        line = [ln for ln in g.to_grammar_string().splitlines() if ln.startswith("E")][0]
        assert "F" in line
        assert "'F'" not in line

    def test_cfg_no_weights(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        output = g.to_grammar_string()
        assert "[" not in output

    def test_pcfg_weights_present(self):
        g = Grammar([Rule("E", ["x"], weight=0.4), Rule("E", ["y"], weight=0.6)])
        output = g.to_grammar_string()
        assert "[0.4]" in output
        assert "[0.6]" in output

    def test_multiple_rules_same_lhs_one_line(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        lines = g.to_grammar_string().splitlines()
        e_lines = [ln for ln in lines if ln.startswith("E")]
        assert len(e_lines) == 1
        assert "|" in e_lines[0]

    def test_empty_grammar_returns_empty_string(self):
        assert Grammar().to_grammar_string() == ""

    def test_start_header_emitted_when_start_set(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        assert g.to_grammar_string().startswith("# start: E")

    def test_no_start_header_when_start_is_none(self):
        g = Grammar([Rule("E", ["x"])])
        assert not g.to_grammar_string().startswith("#")

    def test_rhs_token_order_preserved(self):
        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("F", ["x"])])
        e_line = [ln for ln in g.to_grammar_string().splitlines() if ln.startswith("E")][0]
        assert e_line == "E -> E '+' F"

    def test_roundtrip_cfg(self):
        """Round-trip via to/from_grammar_string without passing start explicitly."""
        original = Grammar(
            [
                Rule("E", ["E", "+", "F"], name="add"),
                Rule("E", ["F"], name="E_to_F"),
                Rule("F", ["x"]),
            ],
            start="E",
        )
        restored = Grammar.from_grammar_string(original.to_grammar_string())
        assert restored.start == original.start
        assert restored.nonterminals == original.nonterminals
        for nt in original.nonterminals:
            orig_rules = [(r.lhs, tuple(r.rhs), r.weight) for r in original.rules_for(nt)]
            rest_rules = [(r.lhs, tuple(r.rhs), r.weight) for r in restored.rules_for(nt)]
            assert orig_rules == rest_rules

    def test_roundtrip_pcfg(self):
        original = Grammar(
            [
                Rule("E", ["x"], weight=0.4),
                Rule("E", ["y"], weight=0.6),
            ],
            start="E",
        )
        restored = Grammar.from_grammar_string(original.to_grammar_string())
        assert restored.start == "E"
        for ro, rr in zip(original.rules_for("E"), restored.rules_for("E")):
            assert ro.weight == pytest.approx(rr.weight)

    def test_arrow_format(self):
        g = Grammar([Rule("E", ["x"])])
        assert " -> " in g.to_grammar_string()


# ---------------------------------------------------------------------------
# Rule.from_line
# ---------------------------------------------------------------------------


class TestRuleFromLine:
    def test_basic(self):
        rules = Rule.from_line("E -> 'x'")
        assert len(rules) == 1
        assert rules[0] == Rule("E", ["x"])

    def test_multiple_alternatives(self):
        rules = Rule.from_line("E -> E '+' F [0.4] | F [0.6]")
        assert len(rules) == 2
        assert rules[0].weight == pytest.approx(0.4)
        assert rules[1].weight == pytest.approx(0.6)

    def test_empty_lhs_raises(self):
        with pytest.raises(ValueError, match="Empty left-hand side"):
            Rule.from_line("-> 'x'")

    def test_no_arrow_raises(self):
        with pytest.raises(ValueError, match="Expected '->'"):
            Rule.from_line("E 'x'")

    def test_no_alternatives_raises(self):
        with pytest.raises(ValueError, match="No alternatives"):
            Rule.from_line("E -> ")


# ---------------------------------------------------------------------------
# Serialization: Rule
# ---------------------------------------------------------------------------


class TestRuleSerialization:
    def test_to_dict_roundtrip(self):
        r = Rule("E", ["E", "+", "F"], weight=0.4, name="E_add")
        d = r.to_dict()
        assert d == {"lhs": "E", "rhs": ["E", "+", "F"], "weight": 0.4, "name": "E_add"}
        r2 = Rule.from_dict(d)
        assert r2.lhs == r.lhs
        assert r2.rhs == r.rhs
        assert r2.weight == r.weight
        assert r2.name == r.name

    def test_to_dict_defaults(self):
        r = Rule("E", ["x"])
        d = r.to_dict()
        assert d["weight"] == 1.0
        assert d["name"] is None

    def test_from_dict_defaults(self):
        r = Rule.from_dict({"lhs": "E", "rhs": ["x"]})
        assert r.weight == 1.0
        assert r.name is None

    def test_from_dict_with_name(self):
        r = Rule.from_dict({"lhs": "F", "rhs": ["sin", "(", "E", ")"], "weight": 0.3, "name": "fn_sin"})
        assert r.lhs == "F"
        assert r.rhs == ["sin", "(", "E", ")"]
        assert r.weight == pytest.approx(0.3)
        assert r.name == "fn_sin"


# ---------------------------------------------------------------------------
# Serialization: Grammar.to_dict / Grammar.from_dict
# ---------------------------------------------------------------------------


class TestGrammarDictSerialization:
    def test_to_dict_keys(self):
        g = Grammar([Rule("E", ["x"], name="E_x")], start="E")
        d = g.to_dict()
        assert set(d.keys()) == {"start", "rules", "constraints"}
        assert d["start"] == "E"
        assert len(d["rules"]) == 1
        assert d["constraints"] == []

    def test_roundtrip_no_constraints(self):
        g = Grammar(
            [
                Rule("E", ["E", "+", "F"], weight=0.4, name="E_add"),
                Rule("E", ["F"], weight=0.6, name="E_to_F"),
                Rule("F", ["x"], name="F_x"),
            ],
            start="E",
        )
        g2 = Grammar.from_dict(g.to_dict())
        assert g2.start == "E"
        assert len(g2._rules) == 3
        for r1, r2 in zip(g._rules, g2._rules):
            assert r1.lhs == r2.lhs
            assert r1.rhs == r2.rhs
            assert r1.weight == pytest.approx(r2.weight)
            assert r1.name == r2.name

    def test_roundtrip_with_constraints(self):
        from SRToolkit.utils.grammar import MaxDepth, MaxNodes, MaxOccurrences, NoNested

        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])], start="E")
        g.add_constraint(MaxDepth(3))
        g.add_constraint(MaxNodes(10))
        g.add_constraint(MaxOccurrences("x", 2))
        g.add_constraint(NoNested(["sin", "cos"]))
        g2 = Grammar.from_dict(g.to_dict())
        assert len(g2._constraints) == 4
        types = [type(c).__name__ for c in g2._constraints]
        assert types == ["MaxDepth", "MaxNodes", "MaxOccurrences", "NoNested"]

    def test_roundtrip_preserves_start_none(self):
        g = Grammar([Rule("E", ["x"])])
        assert g.start is None
        g2 = Grammar.from_dict(g.to_dict())
        assert g2.start is None

    def test_from_dict_empty_grammar(self):
        g = Grammar.from_dict({"start": None, "rules": [], "constraints": []})
        assert g._rules == []
        assert g._constraints == []
