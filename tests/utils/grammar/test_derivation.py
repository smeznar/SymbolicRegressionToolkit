"""
Tests for SRToolkit/utils/grammar/derivation.py.

Covers: Derivation (complete, options, apply, sample, generate,
to_token_list, to_parse_tree, local_stack, global_state, _slot_for),
_scope_miss, and _Frame initialisation via the engine.

Intentionally avoids duplicating constraint-behaviour tests that already
live in tests/utils/test_constraints.py; the focus here is on the
*engine mechanics* of the derivation loop.
"""

from __future__ import annotations

import pytest

from SRToolkit.utils.grammar import (
    AncestorInfo,
    Constraint,
    Grammar,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    ParseTree,
    Rule,
)
from SRToolkit.utils.grammar.derivation import _scope_miss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_grammar() -> Grammar:
    """E -> E + E | x  (flat, self-recursive)"""
    return Grammar(
        [
            Rule("E", ["E", "+", "E"], name="E_add"),
            Rule("E", ["x"], name="E_x"),
        ]
    )


def _two_level_grammar() -> Grammar:
    """E -> E + F | F;  F -> x | y"""
    return Grammar(
        [
            Rule("E", ["E", "+", "F"], name="E_add"),
            Rule("E", ["F"], name="E_to_F"),
            Rule("F", ["x"], name="F_x"),
            Rule("F", ["y"], name="F_y"),
        ]
    )


class _RejectAll(Constraint):
    def allows(self, slot, rule, global_):
        return False

    def update(self, slot, rule, global_):
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        return [None] * n, None


class _CountCalls(Constraint):
    """Counts how many times allows() is called."""

    def __init__(self):
        self.call_count = 0

    def allows(self, slot, rule, global_):
        self.call_count += 1
        return True

    def update(self, slot, rule, global_):
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        return [None] * n, None


# ---------------------------------------------------------------------------
# Derivation.complete
# ---------------------------------------------------------------------------


class TestComplete:
    def test_initially_false(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert d.complete is False

    def test_true_after_terminal_rule(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.apply(d.options()[0])
        assert d.complete is True

    def test_false_while_nt_children_remain(self):
        g = _two_level_grammar()
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[1])  # E -> F — one open slot left
        assert d.complete is False

    def test_complete_iff_frames_empty(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert len(d._frames) == 1
        d.apply(d.options()[0])
        assert len(d._frames) == 0
        assert d.complete


# ---------------------------------------------------------------------------
# Derivation.options
# ---------------------------------------------------------------------------


class TestOptions:
    def test_raises_when_complete(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        with pytest.raises(RuntimeError, match="complete"):
            d.options()

    def test_all_rules_returned_without_constraints(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"]), Rule("E", ["z"])])
        d = g.start_derivation("E")
        assert len(d.options()) == 3

    def test_constraint_filters_candidates(self):
        class RejectX(Constraint):
            def allows(self, slot, rule, global_):
                return "x" not in rule.rhs

            def update(self, slot, rule, global_):
                n = sum(1 for s in rule.rhs if s in slot.nonterminals)
                return [None] * n, None

        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        g.add_constraint(RejectX())
        opts = g.start_derivation("E").options()
        assert len(opts) == 1
        assert opts[0].rhs == ["y"]

    def test_returns_empty_list_when_all_filtered(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(_RejectAll())
        assert g.start_derivation("E").options() == []

    def test_early_exit_skips_later_constraints(self):
        """Once candidates is empty, subsequent constraints are not queried."""
        counter = _CountCalls()
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(_RejectAll())  # empties the list
        g.add_constraint(counter)  # should never be reached
        g.start_derivation("E").options()
        assert counter.call_count == 0

    def test_options_target_leftmost_nonterminal(self):
        """After E -> E + F the leftmost open slot is E, not F."""
        g = _two_level_grammar()
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # E -> E + F
        assert all(r.lhs == "E" for r in d.options())

    def test_scope_miss_lets_rule_through_unfiltered(self):
        """A constraint scoped to 'F' must not reject rules at an 'E' slot."""

        class RejectAtF(Constraint):
            nonterminals = frozenset({"F"})

            def allows(self, slot, rule, global_):
                return False  # always reject when in scope

            def update(self, slot, rule, global_):
                n = sum(1 for s in rule.rhs if s in slot.nonterminals)
                return [None] * n, None

        g = _two_level_grammar()
        g.add_constraint(RejectAtF())
        opts = g.start_derivation("E").options()
        assert len(opts) == 2  # both E rules survive — scope missed


# ---------------------------------------------------------------------------
# Derivation.apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_raises_when_complete(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        with pytest.raises(RuntimeError, match="complete"):
            d.apply(Rule("E", ["x"]))

    def test_raises_on_wrong_lhs(self):
        g = Grammar([Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        with pytest.raises(ValueError, match="lhs"):
            d.apply(g.rules_for("F")[0])

    def test_terminal_only_rhs_completes_derivation(self):
        g = Grammar([Rule("E", ["x", "+", "y"])])
        d = g.start_derivation("E")
        d.apply(d.options()[0])
        assert d.complete

    def test_sets_rule_applied_on_current_node(self):
        r = Rule("E", ["x"])
        g = Grammar([r])
        d = g.start_derivation("E")
        d.apply(r)
        assert d._root.rule_applied is r

    def test_children_appended_to_current_node(self):
        r = Rule("E", ["a", "+", "b"])
        g = Grammar([r])
        d = g.start_derivation("E")
        d.apply(r)
        assert [c.symbol for c in d._root.children] == ["a", "+", "b"]

    def test_steps_incremented(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert d._steps == 0
        d.apply(d.options()[0])
        assert d._steps == 1

    def test_each_apply_increments_steps(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])
        assert d._steps == 1
        d.apply(g.rules_for("E")[1])
        assert d._steps == 2
        d.apply(g.rules_for("E")[1])
        assert d._steps == 3

    def test_leftmost_nt_child_on_top_of_stack(self):
        """After E -> E + F, _frames[-1] is the leftmost E child."""
        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])
        assert d._frames[-1].nonterminal == "E"
        assert d._frames[-2].nonterminal == "F"

    def test_nt_child_indices_assigned_in_order(self):
        """NT children are numbered 0, 1, ... across rhs positions."""
        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])
        # _frames[-1] = E child (index 0), _frames[-2] = F child (index 1)
        assert d._frames[-1].child_index == 0
        assert d._frames[-2].child_index == 1

    def test_parent_rule_set_on_child_frames(self):
        r_add = Rule("E", ["E", "+", "F"])
        g = Grammar([r_add, Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        d.apply(r_add)
        for frame in d._frames:
            assert frame.parent_rule is r_add

    def test_constraint_update_receives_pre_apply_frontier_size(self):
        """slot.frontier_size inside update() is the count BEFORE new frames are pushed."""
        captured = []

        class CaptureFrontierSize(Constraint):
            def update(self, slot, rule, global_):
                captured.append(slot.frontier_size)
                n = sum(1 for s in rule.rhs if s in slot.nonterminals)
                return [None] * n, global_

        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("E", ["x"]), Rule("F", ["y"])])
        g.add_constraint(CaptureFrontierSize())
        d = g.start_derivation("E")
        # Before apply: 1 open slot
        d.apply(g.rules_for("E")[0])
        assert captured[0] == 1  # pre-apply size was 1

    def test_wrong_child_local_count_raises(self):
        class BadUpdate(Constraint):
            def update(self, slot, rule, global_):
                return [None] * 99, None  # always wrong

        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(BadUpdate())
        d = g.start_derivation("E")
        with pytest.raises(RuntimeError, match="child locals"):
            d.apply(g.rules_for("E")[0])

    def test_terminal_children_do_not_create_frames(self):
        """Only NT symbols in rhs produce new frames."""
        g = Grammar([Rule("E", ["x", "+", "y"])])
        d = g.start_derivation("E")
        assert len(d._frames) == 1
        d.apply(d.options()[0])
        assert len(d._frames) == 0  # terminals added no frames

    def test_multiple_nt_children_all_enqueued(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # E -> E + E
        assert len(d._frames) == 2

    def test_global_state_updated_through_constraint(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxNodes(10))
        c = g._constraints[0]
        d = g.start_derivation("E")
        assert d.global_state(c) == 0
        d.apply(g.rules_for("E")[0])
        assert d.global_state(c) == 1


# ---------------------------------------------------------------------------
# Derivation.sample
# ---------------------------------------------------------------------------


class TestSample:
    def test_raises_when_complete(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        with pytest.raises(RuntimeError):
            d.sample()

    def test_raises_when_no_candidates(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(_RejectAll())
        with pytest.raises(RuntimeError, match="No valid rules"):
            g.start_derivation("E").sample()

    def test_single_candidate_always_applied(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.sample()
        assert d.complete

    def test_increments_steps(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.sample()
        assert d._steps == 1

    def test_uniform_sampling_covers_all_options(self):
        """With equal weights both terminals should appear across many samples."""
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        seen = set()
        for _ in range(50):
            seen.add(tuple(g.start_derivation("E").generate()))
            if seen == {("x",), ("y",)}:
                break
        assert seen == {("x",), ("y",)}

    def test_pcfg_heavy_rule_dominates(self):
        """Rule with weight 0.99 should be sampled almost exclusively."""
        g = Grammar([Rule("E", ["x"], weight=0.01), Rule("E", ["y"], weight=0.99)])
        y_count = sum(1 for _ in range(100) if g.start_derivation("E").generate() == ["y"])
        assert y_count >= 90

    def test_applied_rule_is_from_options(self):
        """The rule chosen by sample() must be one returned by options()."""
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        for _ in range(10):
            d = g.start_derivation("E")
            opts = d.options()
            d.sample()
            pt = d.to_parse_tree()
            assert pt.productions_used()[0] in opts


# ---------------------------------------------------------------------------
# Derivation.generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_single_terminal_grammar(self):
        g = Grammar([Rule("E", ["x"])])
        assert g.start_derivation("E").generate() == ["x"]

    def test_returns_list_of_strings(self):
        g = Grammar([Rule("E", ["x"])])
        result = g.start_derivation("E").generate()
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_completes_derivation(self):
        g = _simple_grammar()
        g.add_constraint(MaxDepth(4))
        d = g.start_derivation("E")
        d.generate(limit=500)
        assert d.complete

    def test_limit_zero_raises(self):
        """limit=0 should raise immediately for any non-trivial grammar."""
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        with pytest.raises(RuntimeError, match="did not complete"):
            g.start_derivation("E").generate(limit=0)

    def test_limit_exactly_enough(self):
        """A grammar that requires exactly 1 step completes with limit=1."""
        g = Grammar([Rule("E", ["x"])])
        assert g.start_derivation("E").generate(limit=1) == ["x"]

    def test_limit_exceeded_raises(self):
        """A grammar that needs more steps than the limit raises."""
        # Grammar with no terminal rule can never complete
        g2 = Grammar([Rule("E", ["E", "+", "E"])])
        with pytest.raises(RuntimeError, match="did not complete"):
            g2.start_derivation("E").generate(limit=2)

    def test_tokens_match_grammar(self):
        # _simple_grammar (E -> E + E | x) — E always has a terminal rule.
        g = _simple_grammar()
        g.add_constraint(MaxDepth(4))
        allowed = {"x", "+"}
        for _ in range(10):
            tokens = g.start_derivation("E").generate(limit=200)
            assert set(tokens).issubset(allowed)


# ---------------------------------------------------------------------------
# Derivation.to_token_list
# ---------------------------------------------------------------------------


class TestToTokenList:
    def test_raises_when_incomplete(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        with pytest.raises(RuntimeError, match="not yet complete"):
            d.to_token_list()

    def test_single_terminal(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        assert d.to_token_list() == ["x"]

    def test_multi_token_expression(self):
        g = Grammar(
            [
                Rule("E", ["E", "+", "F"]),
                Rule("E", ["F"]),
                Rule("F", ["x"]),
            ]
        )
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # E -> E + F
        d.apply(g.rules_for("E")[1])  # E -> F (leftmost)
        d.apply(g.rules_for("F")[0])  # F -> x
        d.apply(g.rules_for("F")[0])  # F -> x (remaining)
        assert d.to_token_list() == ["x", "+", "x"]

    def test_consistent_with_parse_tree(self):
        g = _simple_grammar()
        g.add_constraint(MaxDepth(3))
        d = g.start_derivation("E")
        d.generate(limit=200)
        assert d.to_token_list() == d.to_parse_tree().to_token_list()


# ---------------------------------------------------------------------------
# Derivation.to_parse_tree
# ---------------------------------------------------------------------------


class TestToParseTree:
    def test_raises_when_incomplete(self):
        g = Grammar([Rule("E", ["x"])])
        with pytest.raises(RuntimeError, match="not yet complete"):
            g.start_derivation("E").to_parse_tree()

    def test_returns_parse_tree(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.generate()
        assert isinstance(d.to_parse_tree(), ParseTree)

    def test_parse_tree_root_symbol(self):
        g = Grammar([Rule("E", ["x"])], start="E")
        d = g.start_derivation()
        d.generate()
        assert d.to_parse_tree().root.symbol == "E"

    def test_parse_tree_validates_against_grammar(self):
        g = _simple_grammar()
        g.add_constraint(MaxDepth(3))
        d = g.start_derivation("E")
        d.generate(limit=200)
        assert g.validate(d.to_parse_tree()) is True

    def test_grammar_verify_round_trip(self):
        # Use _simple_grammar (E -> E + E | x) — E always has a terminal rule,
        # so MaxDepth(3) can never starve the derivation.
        g = _simple_grammar()
        g.add_constraint(MaxDepth(3))
        for _ in range(10):
            d = g.start_derivation("E")
            d.generate(limit=200)
            assert g.validate(d.to_parse_tree()) is True


# ---------------------------------------------------------------------------
# local_stack and global_state
# ---------------------------------------------------------------------------


class TestLocalStack:
    def test_single_slot_returns_one_entry(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxDepth(5))
        c = g._constraints[0]
        d = g.start_derivation("E")
        assert d.local_stack(c) == [5]

    def test_leftmost_slot_is_first(self):
        """local_stack[0] is the leftmost (current) slot's value."""

        # Use an asymmetric constraint to distinguish left vs right child.
        class AsymDepth(Constraint):
            def initial_local(self, start):
                return 10

            def update(self, slot, rule, global_):
                nt_syms = [s for s in rule.rhs if s in slot.nonterminals]
                if len(nt_syms) == 2:
                    return [slot.local - 1, slot.local - 2], None
                return [slot.local - 1] * len(nt_syms), None

        g = Grammar([Rule("E", ["E", "+", "F"]), Rule("E", ["x"]), Rule("F", ["y"])])
        c = AsymDepth()
        g.add_constraint(c)
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # E -> E + F
        stack = d.local_stack(c)
        assert stack[0] == 9  # E (leftmost, local-1)
        assert stack[1] == 8  # F (local-2)

    def test_stack_length_matches_open_slots(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxDepth(5))
        c = g._constraints[0]
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # opens two slots
        assert len(d.local_stack(c)) == 2

    def test_independent_across_derivations(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxDepth(5))
        c = g._constraints[0]
        d1 = g.start_derivation("E")
        d2 = g.start_derivation("E")
        d1.apply(g.rules_for("E")[0])  # d1 now has two open slots at depth 4
        assert d1.local_stack(c) == [4, 4]
        assert d2.local_stack(c) == [5]  # d2 untouched

    def test_depth_decrements_cascade(self):
        """Expanding leftmost slot again pushes its children at local-1."""
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxDepth(3))
        c = g._constraints[0]
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # root E(3) → [E(2), E(2)]
        d.apply(g.rules_for("E")[0])  # leftmost E(2) → [E(1), E(1)]
        stack = d.local_stack(c)
        assert stack == [1, 1, 2]  # leftmost first, outer-right last


class TestGlobalState:
    def test_initial_global_state(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxNodes(10))
        c = g._constraints[0]
        assert g.start_derivation("E").global_state(c) == 0

    def test_increments_on_apply(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(MaxNodes(10))
        c = g._constraints[0]
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])
        assert d.global_state(c) == 1

    def test_independent_across_derivations(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxNodes(10))
        c = g._constraints[0]
        d1 = g.start_derivation("E")
        d2 = g.start_derivation("E")
        d1.apply(d1.options()[0])
        assert d1.global_state(c) == 1
        assert d2.global_state(c) == 0


# ---------------------------------------------------------------------------
# Frontier size in ExpansionContext
# ---------------------------------------------------------------------------


class TestFrontierSize:
    def test_frontier_size_in_options_reflects_open_slots(self):
        """slot.frontier_size passed to allows() equals the current open-slot count."""
        observed = []

        class ObserveFrontierSize(Constraint):
            def allows(self, slot, rule, global_):
                observed.append(slot.frontier_size)
                return True

            def update(self, slot, rule, global_):
                n = sum(1 for s in rule.rhs if s in slot.nonterminals)
                return [None] * n, global_

        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(ObserveFrontierSize())
        d = g.start_derivation("E")
        d.options()  # one open slot
        assert observed[-1] == 1
        d.apply(g.rules_for("E")[0])  # now two open slots
        d.options()
        assert observed[-1] == 2

    def test_frontier_size_in_update_is_pre_apply(self):
        """update() sees the frontier size from BEFORE new frames were pushed."""
        observed_in_update = []

        class ObserveUpdate(Constraint):
            def update(self, slot, rule, global_):
                observed_in_update.append(slot.frontier_size)
                n = sum(1 for s in rule.rhs if s in slot.nonterminals)
                return [None] * n, global_

        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        g.add_constraint(ObserveUpdate())
        d = g.start_derivation("E")
        d.apply(g.rules_for("E")[0])  # pre-apply frontier = 1
        assert observed_in_update[0] == 1

        d.apply(g.rules_for("E")[1])  # pre-apply frontier = 2 (after first apply)
        assert observed_in_update[1] == 2


# ---------------------------------------------------------------------------
# Ancestor chain
# ---------------------------------------------------------------------------


class TestAncestorChain:
    def test_root_slot_has_empty_ancestors(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert d._frames[-1].ancestors == ()

    def test_child_slot_has_one_ancestor(self):
        r_add = Rule("E", ["E", "+", "F"])
        g = Grammar([r_add, Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        d.apply(r_add)
        e_frame = d._frames[-1]  # leftmost child (E)
        assert len(e_frame.ancestors) == 1
        anc = e_frame.ancestors[0]
        assert anc.nonterminal == "E"  # parent was E
        assert anc.rule is r_add
        assert anc.child_index == 0  # first NT child

    def test_right_child_ancestor_index(self):
        r_add = Rule("E", ["E", "+", "F"])
        g = Grammar([r_add, Rule("E", ["x"]), Rule("F", ["y"])])
        d = g.start_derivation("E")
        d.apply(r_add)
        f_frame = d._frames[-2]  # second (F) child
        assert f_frame.ancestors[0].child_index == 1

    def test_ancestors_accumulate_with_depth(self):
        r_add = Rule("E", ["E", "+", "E"], name="add")
        g = Grammar([r_add, Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.apply(r_add)  # depth 1
        d.apply(r_add)  # depth 2 on leftmost E
        leftmost = d._frames[-1]
        assert len(leftmost.ancestors) == 2

    def test_ancestor_info_fields(self):
        """AncestorInfo captures nonterminal, rule, and nt-child index."""
        r_add = Rule("E", ["E", "+", "E"])
        g = Grammar([r_add, Rule("E", ["x"])])
        d = g.start_derivation("E")
        d.apply(r_add)
        anc: AncestorInfo = d._frames[-1].ancestors[0]
        assert isinstance(anc, AncestorInfo)
        assert anc.nonterminal == "E"
        assert anc.rule is r_add
        assert isinstance(anc.child_index, int)


# ---------------------------------------------------------------------------
# Depth-first leftmost ordering
# ---------------------------------------------------------------------------


class TestFrontierOrdering:
    def test_left_branch_fully_expanded_before_right(self):
        """After E -> E + E -> (E + E) + E, the deepest left slot is current."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        d = g.start_derivation("E")
        d.apply(r_add)  # root E → [E_left, E_right]
        d.apply(r_add)  # E_left → [E_ll, E_lr]
        # leftmost is E_ll; E_lr and E_right are pending
        assert d._frames[-1].nonterminal == "E"
        assert len(d._frames) == 3

    def test_complete_leftmost_derivation(self):
        """Full manual derivation produces correct token order."""
        r_add = Rule("E", ["E", "+", "E"])
        r_x = Rule("E", ["x"])
        g = Grammar([r_add, r_x])
        d = g.start_derivation("E")
        # Build (x + x) + x deterministically
        d.apply(r_add)  # E -> E + E
        d.apply(r_add)  # leftmost E -> E + E
        d.apply(r_x)  # innermost left E -> x
        d.apply(r_x)  # inner right E -> x
        d.apply(r_x)  # outer right E -> x
        assert d.complete
        assert d.to_token_list() == ["x", "+", "x", "+", "x"]

    def test_frame_count_after_branching(self):
        g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert len(d._frames) == 1
        d.apply(g.rules_for("E")[0])
        assert len(d._frames) == 2
        d.apply(g.rules_for("E")[1])  # close left branch
        assert len(d._frames) == 1
        d.apply(g.rules_for("E")[1])  # close right branch
        assert len(d._frames) == 0


# ---------------------------------------------------------------------------
# _scope_miss
# ---------------------------------------------------------------------------


class TestScopeMiss:
    def _c(self, nonterminals=None, rule_names=None) -> Constraint:
        c = Constraint()
        c.nonterminals = nonterminals
        c.rule_names = rule_names
        return c

    def test_no_restrictions_never_misses(self):
        c = self._c()
        assert _scope_miss(c, "E", Rule("E", ["x"])) is False
        assert _scope_miss(c, "F", Rule("E", ["x"])) is False

    def test_nt_restriction_hit(self):
        c = self._c(nonterminals=frozenset({"E"}))
        assert _scope_miss(c, "E", Rule("E", ["x"])) is False

    def test_nt_restriction_miss(self):
        c = self._c(nonterminals=frozenset({"E"}))
        assert _scope_miss(c, "F", Rule("E", ["x"])) is True

    def test_rule_name_restriction_hit(self):
        c = self._c(rule_names=frozenset({"add"}))
        assert _scope_miss(c, "E", Rule("E", ["x"], name="add")) is False

    def test_rule_name_restriction_miss(self):
        c = self._c(rule_names=frozenset({"add"}))
        assert _scope_miss(c, "E", Rule("E", ["x"], name="mul")) is True

    def test_rule_name_none_not_in_set(self):
        """A nameless rule does not match a frozenset of names."""
        c = self._c(rule_names=frozenset({"add"}))
        assert _scope_miss(c, "E", Rule("E", ["x"])) is True  # name=None ∉ {"add"}

    def test_nt_miss_short_circuits_rule_name_check(self):
        """When the NT check triggers a miss the rule_names check is never reached."""
        c = self._c(nonterminals=frozenset({"E"}), rule_names=frozenset({"add"}))
        # nt="F" misses the NT scope → True regardless of rule_name
        assert _scope_miss(c, "F", Rule("E", ["x"], name="add")) is True

    def test_both_in_scope_returns_false(self):
        c = self._c(nonterminals=frozenset({"E"}), rule_names=frozenset({"add"}))
        assert _scope_miss(c, "E", Rule("E", ["x"], name="add")) is False

    def test_nt_hit_rule_name_miss(self):
        c = self._c(nonterminals=frozenset({"E"}), rule_names=frozenset({"add"}))
        assert _scope_miss(c, "E", Rule("E", ["x"], name="mul")) is True


# ---------------------------------------------------------------------------
# Constraint state initialisation in __init__
# ---------------------------------------------------------------------------


class TestDerivationInit:
    def test_no_constraints_no_globals(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert d._globals == {}

    def test_constraints_initialised(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxDepth(5))
        g.add_constraint(MaxNodes(10))
        d = g.start_derivation("E")
        assert len(d._globals) == 2

    def test_initial_local_set_on_first_frame(self):
        g = Grammar([Rule("E", ["x"])])
        g.add_constraint(MaxDepth(7))
        c = g._constraints[0]
        d = g.start_derivation("E")
        cid = id(c)
        assert d._frames[-1].locals[cid] == 7

    def test_nonterminals_frozen_at_construction(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert isinstance(d._nonterminals, frozenset)
        assert d._nonterminals == frozenset({"E"})

    def test_root_node_has_start_symbol(self):
        g = Grammar([Rule("E", ["x"])])
        d = g.start_derivation("E")
        assert d._root.symbol == "E"
        assert d._root.rule_applied is None  # not yet applied


# ---------------------------------------------------------------------------
# Multiple constraints — all must pass
# ---------------------------------------------------------------------------


class TestMultipleConstraints:
    def test_both_must_pass(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"]), Rule("E", ["z"])])
        g.add_constraint(MaxOccurrences("x", 0))
        g.add_constraint(MaxOccurrences("y", 0))
        d = g.start_derivation("E")
        opts = d.options()
        assert len(opts) == 1
        assert opts[0].rhs == ["z"]
