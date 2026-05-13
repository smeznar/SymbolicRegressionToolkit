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
    def test_valid_tree_returns_true(self):
        r = Rule("E", ["x"])
        g = Grammar([r])
        root = ParseTreeNode("E", r, [_leaf("x")])
        assert g.validate(ParseTree(root)) is True

    def test_tree_with_foreign_rule_returns_false(self):
        r_known = Rule("E", ["x"])
        r_foreign = Rule("E", ["y"])
        g = Grammar([r_known])
        root = ParseTreeNode("E", r_foreign, [_leaf("y")])
        assert g.validate(ParseTree(root)) is False

    def test_same_lhs_rhs_different_weight_is_foreign(self):
        """weight is part of the identity key (lhs, tuple(rhs), weight)."""
        r_grammar = Rule("E", ["x"], weight=1.0)
        r_tree = Rule("E", ["x"], weight=0.5)
        g = Grammar([r_grammar])
        root = ParseTreeNode("E", r_tree, [_leaf("x")])
        assert g.validate(ParseTree(root)) is False

    def test_same_lhs_rhs_weight_different_name_is_valid(self):
        """Name is NOT part of the key; only (lhs, rhs, weight) matter."""
        r_grammar = Rule("E", ["x"], name="grammar_name")
        r_tree = Rule("E", ["x"], name="tree_name")
        g = Grammar([r_grammar])
        root = ParseTreeNode("E", r_tree, [_leaf("x")])
        assert g.validate(ParseTree(root)) is True

    def test_nested_valid_tree(self):
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.validate(ParseTree(root)) is True

    def test_nested_tree_one_foreign_rule_returns_false(self):
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        r_foreign = Rule("E", ["y"])
        g = Grammar([r_add, r_x])
        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_foreign, [_leaf("y")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.validate(ParseTree(root)) is False

    def test_leaf_only_tree_vacuously_true(self):
        """productions_used() returns [] for a leaf → all([]) == True."""
        g = Grammar([Rule("E", ["x"])])
        assert g.validate(ParseTree(_leaf("x"))) is True

    def test_rule_from_different_grammar_same_signature_is_valid(self):
        """Two separately constructed rules with identical (lhs, rhs, weight) are interchangeable."""
        r1 = Rule("E", ["x"], weight=1.0)
        r2 = Rule("E", ["x"], weight=1.0)
        g = Grammar([r1])
        root = ParseTreeNode("E", r2, [_leaf("x")])
        assert g.validate(ParseTree(root)) is True


# ---------------------------------------------------------------------------
# Grammar.check_constraints
# ---------------------------------------------------------------------------


class TestGrammarCheckConstraints:
    def test_no_constraints_always_true(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        assert g.check_constraints(d.to_parse_tree()) is True

    def test_constraint_consistent_tree_returns_true(self):
        g = _simple_grammar()
        g.add_constraint(MaxDepth(5))
        d = g.start_derivation("E")
        d.generate()
        assert g.check_constraints(d.to_parse_tree()) is True

    def test_constraint_violating_tree_returns_false(self):
        """A tree with depth 1 violates MaxDepth(0)."""
        r_add = Rule("E", ["E", "+", "E"], name="E_add_+")
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        g.add_constraint(MaxDepth(0))

        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.check_constraints(ParseTree(root)) is False

    def test_round_trip_with_multiple_constraints(self):
        """Any generated parse tree should satisfy the constraints that guided it."""
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"]), Rule("E", ["y"])])
        g.add_constraint(MaxDepth(3))
        g.add_constraint(MaxOccurrences("x", 2))
        for _ in range(15):
            d = g.start_derivation("E")
            d.generate(limit=500)
            assert g.check_constraints(d.to_parse_tree()) is True


# ---------------------------------------------------------------------------
# Grammar.verify
# ---------------------------------------------------------------------------


class TestGrammarVerify:
    def test_valid_and_constraint_consistent_returns_true(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        assert g.verify(d.to_parse_tree()) is True

    def test_invalid_structure_short_circuits(self):
        """validate() returns False → verify() returns False without calling check_constraints."""
        r_foreign = Rule("E", ["z"])
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxDepth(5))
        root = ParseTreeNode("E", r_foreign, [_leaf("z")])
        assert g.verify(ParseTree(root)) is False

    def test_valid_structure_constraint_violation(self):
        """Structure is valid (rules are known) but MaxDepth(0) rejects depth-1 tree."""
        r_add = Rule("E", ["E", "+", "E"], name="E_add_+")
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        g.add_constraint(MaxDepth(0))

        left = ParseTreeNode("E", r_x, [_leaf("x")])
        right = ParseTreeNode("E", r_x, [_leaf("x")])
        root = ParseTreeNode("E", r_add, [left, _leaf("+"), right])
        assert g.verify(ParseTree(root)) is False

    def test_verify_equals_validate_and_check(self):
        """verify is exactly the conjunction of validate and check_constraints."""
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxDepth(4))
        for _ in range(10):
            d = g.start_derivation("E")
            d.generate()
            pt = d.to_parse_tree()
            assert g.verify(pt) == (g.validate(pt) and g.check_constraints(pt))


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

    def test_no_additive_ops_only_E_to_F(self):
        """Without additive operators only E -> F is generated."""
        g = Grammar.from_symbol_library(_sl("*"))
        e_rules = g.rules_for("E")
        assert len(e_rules) == 1
        assert e_rules[0].name == "E_to_F"

    # --- F level (multiplicative operators) ---

    def test_multiplicative_op_produces_F_mul_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("*"))._rules}
        assert "F_mul_*" in names

    def test_multiple_multiplicative_ops(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("*", "/"))._rules}
        assert "F_mul_*" in names
        assert "F_mul_/" in names

    def test_no_multiplicative_ops_only_F_to_B(self):
        g = Grammar.from_symbol_library(_sl("+"))
        f_rules = g.rules_for("F")
        assert len(f_rules) == 1
        assert f_rules[0].name == "F_to_B"

    # --- B level (power operators) ---

    def test_power_op_produces_B_pow_rule(self):
        """'^' is a precedence-2 OP and triggers the B_ops branch."""
        g = Grammar.from_symbol_library(_sl("^"))
        names = {r.name for r in g.rules_for("B")}
        assert "B_pow_^" in names
        assert "B_to_T" in names

    def test_no_power_ops_only_B_to_T(self):
        g = Grammar.from_symbol_library(_sl("+"))
        b_rules = g.rules_for("B")
        assert len(b_rules) == 1
        assert b_rules[0].name == "B_to_T"

    # --- T level (leaf dispatcher) ---

    def test_T_to_V_always_present(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_V" in names

    def test_T_to_C_present_when_constants_exist(self):
        """When CONST or LIT symbols are present, T -> C is added."""
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "C")).rules_for("T")}
        assert "T_to_C" in names

    def test_T_to_C_absent_without_constants(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_C" not in names

    def test_T_to_R_always_present(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("T")}
        assert "T_to_R" in names

    # --- C level (constants and literals) ---

    def test_only_const_C_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "C")).rules_for("C")}
        assert "C_C" in names

    def test_only_lit_C_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "pi")).rules_for("C")}
        assert "C_pi" in names

    def test_both_lit_and_const_C_rules(self):
        g = Grammar.from_symbol_library(_sl("+", "C", "pi"))
        names = {r.name for r in g.rules_for("C")}
        assert "C_C" in names
        assert "C_pi" in names

    def test_no_constants_no_C_rules(self):
        g = Grammar.from_symbol_library(_sl("+"))
        assert g.rules_for("C") == []

    # --- R level (prefix functions) ---

    def test_function_produces_R_fn_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("sin")).rules_for("R")}
        assert "R_fn_sin" in names

    def test_R_paren_always_present(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("R")}
        assert "R_paren" in names

    def test_R_to_P_present_when_postfix_exists(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "^2")).rules_for("R")}
        assert "R_to_P" in names

    def test_R_to_P_absent_without_postfix(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("R")}
        assert "R_to_P" not in names

    def test_no_R_fns_with_P_fns_has_R_to_P(self):
        """No prefix functions but postfix present: R -> P and R -> (E) only."""
        g = Grammar.from_symbol_library(_sl("+", "^2"))
        r_names = {r.name for r in g.rules_for("R")}
        assert "R_to_P" in r_names
        assert "R_paren" in r_names
        assert not any(n and n.startswith("R_fn_") for n in r_names)

    def test_no_R_fns_no_P_fns_only_R_paren(self):
        g = Grammar.from_symbol_library(_sl("+"))
        r_names = {r.name for r in g.rules_for("R")}
        assert r_names == {"R_paren"}

    # --- P level (postfix functions) ---

    def test_postfix_produces_P_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+", "^2")).rules_for("P")}
        assert "P_^2" in names

    def test_no_postfix_no_P_rules(self):
        assert Grammar.from_symbol_library(_sl("+")).rules_for("P") == []

    # --- V level (variables) ---

    def test_variable_produces_V_rule(self):
        names = {r.name for r in Grammar.from_symbol_library(_sl("+")).rules_for("V")}
        assert any(n and n.startswith("V_") for n in names)

    def test_multiple_variables(self):
        g = Grammar.from_symbol_library(_sl("+", n_vars=3))
        assert len(g.rules_for("V")) == 3

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
        g = Grammar.from_grammar_string("E -> E '+' F | F\nF -> 'x'")
        assert sorted(g.nonterminals) == ["E", "F"]
        assert g.rules_for("F")[0].rhs == ["x"]

    def test_start_defaults_to_first_lhs(self):
        g = Grammar.from_grammar_string("E -> F\nF -> 'x'")
        assert g.start == "E"

    def test_explicit_start_overrides(self):
        g = Grammar.from_grammar_string("E -> F\nF -> 'x'", start="F")
        assert g.start == "F"

    def test_weight_parsed(self):
        g = Grammar.from_grammar_string("E -> 'x' [0.4] | 'y' [0.6]")
        rules = g.rules_for("E")
        assert rules[0].weight == pytest.approx(0.4)
        assert rules[1].weight == pytest.approx(0.6)

    def test_unweighted_defaults_to_one(self):
        g = Grammar.from_grammar_string("E -> 'x' | 'y'")
        assert all(r.weight == 1.0 for r in g.rules_for("E"))

    def test_comment_lines_ignored(self):
        text = "# this is a comment\nE -> 'x'"
        g = Grammar.from_grammar_string(text)
        assert len(g.rules_for("E")) == 1

    def test_blank_lines_ignored(self):
        g = Grammar.from_grammar_string("\nE -> 'x'\n\nF -> 'y'\n")
        assert g.nonterminals == {"E", "F"}

    def test_lines_without_arrow_ignored(self):
        g = Grammar.from_grammar_string("just some text\nE -> 'x'")
        assert len(g.rules_for("E")) == 1

    def test_unquoted_token_is_nonterminal(self):
        """Unquoted tokens in rhs are stored as-is and become nonterminals if they have rules."""
        g = Grammar.from_grammar_string("E -> F\nF -> 'x'")
        assert g.rules_for("E")[0].rhs == ["F"]
        assert "F" in g.nonterminals

    def test_quoted_terminal_stripped(self):
        """Single quotes are stripped from terminal tokens."""
        g = Grammar.from_grammar_string("E -> 'hello world'")
        # 'hello world' is a single quoted token → one symbol with a space in it
        assert g.rules_for("E")[0].rhs == ["hello world"]

    def test_multiple_rules_per_line(self):
        g = Grammar.from_grammar_string("E -> 'x' | 'y' | 'z'")
        assert len(g.rules_for("E")) == 3

    def test_names_are_none(self):
        g = Grammar.from_grammar_string("E -> 'x'")
        assert g.rules_for("E")[0].name is None

    def test_empty_text_returns_empty_grammar(self):
        g = Grammar.from_grammar_string("")
        assert g.nonterminals == set()
        assert g.start is None

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            Grammar.from_grammar_string("E -> 'x' [not_a_float]")

    def test_multiline_same_lhs(self):
        """The same LHS can appear on multiple lines; all rules are collected."""
        g = Grammar.from_grammar_string("E -> 'x'\nE -> 'y'")
        assert len(g.rules_for("E")) == 2

    def test_trailing_pipe_ignored(self):
        """A trailing '|' produces an empty alternative that is silently skipped."""
        g = Grammar.from_grammar_string("E -> 'x' | 'y' |")
        assert len(g.rules_for("E")) == 2

    def test_leading_pipe_ignored(self):
        """A leading '|' also produces an empty alternative that is silently skipped."""
        g = Grammar.from_grammar_string("E -> | 'x'")
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

    def test_rhs_token_order_preserved(self):
        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("F", ["x"])])
        e_line = [ln for ln in g.to_grammar_string().splitlines() if ln.startswith("E")][0]
        assert e_line == "E -> E '+' F"

    def test_roundtrip_cfg(self):
        """from_nltk(to_nltk(g)) reconstructs equivalent rules (names lost, weights kept)."""
        original = Grammar(
            [
                Rule("E", ["E", "+", "F"], name="add"),
                Rule("E", ["F"], name="E_to_F"),
                Rule("F", ["x"]),
            ]
        )
        restored = Grammar.from_grammar_string(original.to_grammar_string())
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
            ]
        )
        restored = Grammar.from_grammar_string(original.to_grammar_string())
        for ro, rr in zip(original.rules_for("E"), restored.rules_for("E")):
            assert ro.weight == pytest.approx(rr.weight)

    def test_arrow_format(self):
        g = Grammar([Rule("E", ["x"])])
        assert " -> " in g.to_grammar_string()
