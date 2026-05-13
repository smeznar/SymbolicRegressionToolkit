"""
Stateful leftmost derivation for CFG/PCFG grammars.

Provides [Derivation][SRToolkit.utils.grammar.derivation.Derivation].  Obtain a
derivation by calling
[Grammar.start_derivation][SRToolkit.utils.grammar.Grammar.start_derivation]
rather than constructing [Derivation][SRToolkit.utils.grammar.derivation.Derivation]
directly.

The engine threads per-slot **local** state and per-derivation **global** state
through every registered [Constraint][SRToolkit.utils.grammar.constraints.Constraint].  See
[SRToolkit.utils.grammar.constraints][] for the full protocol documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .constraints import AncestorInfo, Constraint, ExpansionContext
from .grammar import Grammar, ParseTree, ParseTreeNode, Rule


@dataclass
class _Frame:
    """
    Per-slot state for one open non-terminal in the derivation frontier.

    Bundled together so that the engine works on a single list of frames
    rather than several parallel arrays.  The list is stored with the current
    leftmost slot at the *end* (classic top-of-stack), which makes push/pop
    O(1) amortised.
    """

    nonterminal: str
    node: ParseTreeNode
    ancestors: tuple[AncestorInfo, ...]
    parent_rule: Optional[Rule]
    child_index: Optional[int]
    locals: dict[int, Any] = field(default_factory=dict)


class Derivation:
    """
    Stateful leftmost derivation over a [Grammar][SRToolkit.utils.grammar.Grammar].

    A derivation begins at a start non-terminal and proceeds by repeatedly
    choosing a rule to apply to the leftmost unexpanded non-terminal.
    [options][SRToolkit.utils.grammar.derivation.Derivation.options] returns candidate
    rules filtered by all registered constraints;
    [apply][SRToolkit.utils.grammar.derivation.Derivation.apply] advances the
    derivation by one production.

    Per-slot local state and per-derivation global state for each registered
    constraint are maintained internally; constraint instances carry only
    construction-time configuration and are safe to share across parallel
    derivations.

    Obtain a [Derivation][SRToolkit.utils.grammar.derivation.Derivation] via
    [Grammar.start_derivation][SRToolkit.utils.grammar.Grammar.start_derivation].

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar()
        >>> g.add_rule(Rule("E", ["x"]))
        >>> d = g.start_derivation("E")
        >>> d.complete
        False
        >>> opts = d.options()
        >>> len(opts)
        1
        >>> d.apply(opts[0])
        >>> d.complete
        True
        >>> d.to_token_list()
        ['x']
    """

    def __init__(self, grammar: Grammar, start: str) -> None:
        self._grammar = grammar
        self._nonterminals: frozenset[str] = frozenset(grammar.nonterminals)
        self._steps: int = 0
        self._root: ParseTreeNode = ParseTreeNode(start, None)

        initial_frame = _Frame(
            nonterminal=start,
            node=self._root,
            ancestors=(),
            parent_rule=None,
            child_index=None,
        )
        self._globals: dict[int, Any] = {}
        for c in grammar._constraints:
            cid = id(c)
            initial_frame.locals[cid] = c.initial_local(start)
            self._globals[cid] = c.initial_global()
        # _frames[-1] is the current leftmost open slot.
        self._frames: list[_Frame] = [initial_frame]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def complete(self) -> bool:
        """
        ``True`` when no unexpanded non-terminals remain.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> d = g.start_derivation("E")
            >>> d.complete
            False
            >>> d.apply(d.options()[0])
            >>> d.complete
            True
        """
        return len(self._frames) == 0

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def local_stack(self, constraint: Constraint) -> list[Any]:
        """
        Return the local state stack for ``constraint`` across the open frontier,
        leftmost slot first.

        Args:
            constraint: A constraint previously registered on the grammar.

        Returns:
            One entry per open non-terminal in left-to-right order.
        """
        cid = id(constraint)
        return [f.locals[cid] for f in reversed(self._frames)]

    def global_state(self, constraint: Constraint) -> Any:
        """
        Return the per-derivation global state for ``constraint``.

        Args:
            constraint: A constraint previously registered on the grammar.

        Returns:
            The current global value owned by this derivation for ``constraint``.
        """
        return self._globals[id(constraint)]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def options(self) -> list[Rule]:
        """
        Return candidate rules for the current leftmost unexpanded
        non-terminal, filtered by every registered constraint's
        [allows][SRToolkit.utils.grammar.constraints.Constraint.allows].

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.add_rule(Rule("E", ["y"]))
            >>> d = g.start_derivation("E")
            >>> len(d.options())
            2

        Returns:
            List of [Rule][SRToolkit.utils.grammar.Rule] objects that every
            constraint accepts.

        Raises:
            RuntimeError: If the derivation is already complete.
        """
        if self.complete:
            raise RuntimeError("Derivation is already complete; no options remain.")

        top = self._frames[-1]
        candidates = self._grammar.rules_for(top.nonterminal)

        for c in self._grammar._constraints:
            cid = id(c)
            slot = self._slot_for(top, top.locals[cid])
            global_ = self._globals[cid]
            surviving: list[Rule] = []
            for rule in candidates:
                if _scope_miss(c, top.nonterminal, rule):
                    surviving.append(rule)
                    continue
                if c.allows(slot, rule, global_):
                    surviving.append(rule)
            candidates = surviving
            if not candidates:
                break

        return candidates

    def apply(self, rule: Rule) -> None:
        """
        Apply ``rule`` to the current leftmost unexpanded non-terminal.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["E", "+", "F"]))
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.add_rule(Rule("F", ["y"]))
            >>> d = g.start_derivation("E")
            >>> d.apply(g.rules_for("E")[0])   # E -> E + F
            >>> d.apply(g.rules_for("E")[1])   # E -> x
            >>> d.apply(g.rules_for("F")[0])   # F -> y
            >>> d.to_token_list()
            ['x', '+', 'y']

        Args:
            rule: A rule whose ``lhs`` matches the current leftmost
                non-terminal.

        Raises:
            RuntimeError: If the derivation is already complete.
            ValueError: If ``rule.lhs`` does not match the current
                non-terminal.
        """
        if self.complete:
            raise RuntimeError("Derivation is already complete.")

        top = self._frames[-1]
        if rule.lhs != top.nonterminal:
            raise ValueError(f"Rule lhs '{rule.lhs}' does not match current non-terminal '{top.nonterminal}'.")

        # Build child parse-tree nodes and the new frames for each NT child.
        child_frames: list[_Frame] = []
        nt_child_index = 0
        for sym in rule.rhs:
            child_node = ParseTreeNode(sym, None)
            top.node.children.append(child_node)
            if sym in self._nonterminals:
                frame = AncestorInfo(top.nonterminal, rule, nt_child_index)
                child_frames.append(
                    _Frame(
                        nonterminal=sym,
                        node=child_node,
                        ancestors=top.ancestors + (frame,),
                        parent_rule=rule,
                        child_index=nt_child_index,
                    )
                )
                nt_child_index += 1

        top.node.rule_applied = rule
        n_nt_children = len(child_frames)

        # Thread state through every constraint using the pre-apply slot view.
        pre_frontier_size = len(self._frames)
        for c in self._grammar._constraints:
            cid = id(c)
            slot = self._slot_for(top, top.locals[cid], frontier_size=pre_frontier_size)
            child_locals, new_global = c.update(slot, rule, self._globals[cid])
            if len(child_locals) != n_nt_children:
                raise RuntimeError(
                    f"Constraint {c!r}.update() returned {len(child_locals)} child locals "
                    f"but rule '{rule.lhs} -> {rule.rhs}' has {n_nt_children} "
                    f"non-terminal children."
                )
            for i, cf in enumerate(child_frames):
                cf.locals[cid] = child_locals[i]
            self._globals[cid] = new_global

        # Pop the top frame and push children so that the leftmost child ends
        # up at the end of the list (i.e. on top of the stack).
        self._frames.pop()
        for cf in reversed(child_frames):
            self._frames.append(cf)
        self._steps += 1

    def sample(self) -> None:
        """
        Apply one rule chosen proportionally by weight (PCFG) or uniformly
        (CFG) from the surviving candidates.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> d = g.start_derivation("E")
            >>> d.sample()
            >>> d.complete
            True

        Raises:
            RuntimeError: If the derivation is already complete.
            RuntimeError: If all candidate rules are filtered out by
                [allows][SRToolkit.utils.grammar.constraints.Constraint.allows].
        """
        candidates = self.options()
        if not candidates:
            raise RuntimeError(
                "No valid rules available for the current non-terminal after applying constraint filters."
            )

        weights = [float(r.weight) for r in candidates]
        total = sum(weights)
        probs = [w / total for w in weights]
        self.apply(candidates[np.random.choice(len(candidates), p=probs)])

    def generate(self, limit: int = 1000) -> list[str]:
        """
        Run the derivation to completion and return the token list.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.start_derivation("E").generate()
            ['x']

        Args:
            limit: Maximum number of rule applications. A negative value means
                unlimited. Default ``1000``.

        Returns:
            Flat list of terminal tokens in left-to-right order.

        Raises:
            RuntimeError: If the derivation does not complete within
                ``limit`` steps (only when ``limit >= 0``).
        """
        steps = 0
        while not self.complete:
            if steps >= limit >= 0:
                raise RuntimeError(f"Derivation did not complete within {limit} rule applications.")
            self.sample()
            steps += 1
        return self.to_token_list()

    def to_token_list(self) -> list[str]:
        """
        Return the completed expression as a flat token list.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> d = g.start_derivation("E")
            >>> d.apply(d.options()[0])
            >>> d.to_token_list()
            ['x']

        Returns:
            Flat list of terminal tokens in left-to-right order.

        Raises:
            RuntimeError: If the derivation is not yet complete.
        """
        if not self.complete:
            raise RuntimeError("Derivation is not yet complete.")
        return self.to_parse_tree().to_token_list()

    def to_parse_tree(self) -> ParseTree:
        """
        Return the completed derivation as a
        [ParseTree][SRToolkit.utils.grammar.ParseTree].

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> d = g.start_derivation("E")
            >>> d.apply(d.options()[0])
            >>> isinstance(d.to_parse_tree(), ParseTree)
            True

        Returns:
            The [ParseTree][SRToolkit.utils.grammar.ParseTree] rooted at the
            start symbol.

        Raises:
            RuntimeError: If the derivation is not yet complete.
        """
        if not self.complete:
            raise RuntimeError("Derivation is not yet complete.")
        return ParseTree(self._root)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slot_for(
        self,
        frame: _Frame,
        local: Any,
        frontier_size: int | None = None,
    ) -> ExpansionContext:
        return ExpansionContext(
            nonterminal=frame.nonterminal,
            local=local,
            steps=self._steps,
            parent_rule=frame.parent_rule,
            child_index=frame.child_index,
            ancestors=frame.ancestors,
            partial_tree=self._root,
            nonterminals=self._nonterminals,
            frontier_size=len(self._frames) if frontier_size is None else frontier_size,
        )


def _scope_miss(constraint: Constraint, nt: str, rule: Rule) -> bool:
    if constraint.nonterminals is not None and nt not in constraint.nonterminals:
        return True
    if constraint.rule_names is not None and rule.name not in constraint.rule_names:
        return True
    return False
