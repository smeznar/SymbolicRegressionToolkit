"""
Context-free and probabilistic context-free grammar (CFG/PCFG) representation.

Provides [Rule][SRToolkit.utils.grammar.Rule], [ParseTreeNode][SRToolkit.utils.grammar.ParseTreeNode],
[ParseTree][SRToolkit.utils.grammar.ParseTree], and [Grammar][SRToolkit.utils.grammar.Grammar].
For stateful expression generation see [Derivation][SRToolkit.utils.grammar.derivation.Derivation].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..symbol_library import SymbolLibrary
from ..types import CONST, FN, FN_POSTFIX, FN_PREFIX, LIT, OP, OP_ADDITIVE, OP_MULTIPLICATIVE, OP_POWER, VAR

if TYPE_CHECKING:
    from .constraints import Constraint
    from .derivation import Derivation


# Hybrid naming scheme: standard precedences get short readable names; custom
# precedences fall back to L_{prec}.  Rule-infix labels keep names like
# E_add_+, F_mul_*, B_pow_^ for the standard levels.
_STANDARD_LEVEL_NAMES: dict[int, str] = {
    OP_ADDITIVE: "E",
    OP_MULTIPLICATIVE: "F",
    OP_POWER: "B",
}
_LEVEL_RULE_INFIXES: dict[int, str] = {
    OP_ADDITIVE: "add",
    OP_MULTIPLICATIVE: "mul",
    OP_POWER: "pow",
}
_LEVEL_REC_WEIGHTS: dict[int, float] = {
    OP_ADDITIVE: 0.3,
    OP_MULTIPLICATIVE: 0.4,
    OP_POWER: 0.05,
}


def _level_name(prec: int) -> str:
    return _STANDARD_LEVEL_NAMES.get(prec, f"L_{prec}")


@dataclass
class Rule:
    """
    A single production rule in a grammar.

    A rule is a CFG production when weights remain 1 for all rules in the grammar;
    when weights differ across rules, the grammar is treated as a PCFG and
    productions are sampled proportionally. Weights are unnormalised — the
    grammar normalises them at sampling time.

    Examples:
        >>> r = Rule("E", ["E", "+", "F"], weight=0.4, name="E_add_+")
        >>> r.lhs
        'E'
        >>> r.rhs
        ['E', '+', 'F']
        >>> r.weight
        0.4
        >>> r.name
        'E_add_+'
        >>> Rule("E", ["F"]).weight
        1.0

    Attributes:
        lhs: The non-terminal being expanded, e.g. ``"E"``.
        rhs: Ordered sequence of symbols the non-terminal expands to. Each
            element is either a terminal token (e.g. ``"+"`` or ``"sin"``) or
            the name of another non-terminal. A symbol is treated as a
            non-terminal if and only if it appears as the ``lhs`` of at least
            one rule in the grammar.
        weight: Unnormalised sampling weight.  Defaults to ``1.0``, which
            produces uniform sampling within a group when all rules share the
            same weight.
        name: Optional stable identifier for this rule. Used by constraints
            for scoping and identification.  ``None`` by default.
    """

    lhs: str
    rhs: list[str]
    weight: float = 1.0
    name: Optional[str] = None

    def to_dict(self) -> dict:
        return {"lhs": self.lhs, "rhs": self.rhs, "weight": self.weight, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(lhs=d["lhs"], rhs=d["rhs"], weight=d.get("weight", 1.0), name=d.get("name"))

    @classmethod
    def from_line(cls, line: str) -> list["Rule"]:
        """
        Parse one NLTK production line into a list of [Rule][SRToolkit.utils.grammar.Rule] objects.

        The line must have the form::

            LHS -> RHS_1 | RHS_2 | ...

        where each alternative is a space-separated sequence of symbols.
        Terminals are enclosed in single quotes; unquoted tokens are non-terminals.
        An optional weight in square brackets at the end of an alternative is parsed
        as a ``float``; alternatives without a weight default to ``1.0``.

        Examples:
            >>> Rule.from_line("E -> E '+' F [0.4] | F [0.6]")
            [Rule(lhs='E', rhs=['E', '+', 'F'], weight=0.4, name=None), Rule(lhs='E', rhs=['F'], weight=0.6, name=None)]
            >>> Rule.from_line("F -> 'x'")
            [Rule(lhs='F', rhs=['x'], weight=1.0, name=None)]

        Args:
            line: A single non-empty, non-comment production line.

        Returns:
            List of [Rule][SRToolkit.utils.grammar.Rule] objects, one per alternative.

        Raises:
            ValueError: If ``line`` contains no ``->``.
            ValueError: If the left-hand side is empty.
            ValueError: If a weight token cannot be converted to ``float``.
            ValueError: If no alternatives are parsed from the right-hand side.
        """
        line = line.strip()
        if "->" not in line:
            raise ValueError(f"Expected '->' in grammar line: {line!r}")
        lhs, rhs_part = line.split("->", 1)
        lhs = lhs.strip()
        if not lhs:
            raise ValueError(f"Empty left-hand side in grammar line: {line!r}")
        rules = []
        for alt in rhs_part.split("|"):
            alt = alt.strip()
            if not alt:
                continue
            weight = 1.0
            m = re.search(r"\[([^\]]+)\]\s*$", alt)
            if m:
                try:
                    weight = float(m.group(1))
                except ValueError:
                    raise ValueError(f"Invalid weight {m.group(1)!r} in grammar line: {line!r}")
                alt = alt[: m.start()].strip()
            tokens = [t[1:-1] if t.startswith("'") and t.endswith("'") else t for t in re.findall(r"'[^']*'|\S+", alt)]
            if tokens:
                rules.append(cls(lhs, tokens, weight=weight))
        if not rules:
            raise ValueError(f"No alternatives parsed from grammar line: {line!r}")
        return rules


@dataclass
class ParseTreeNode:
    """
    A node in a derivation parse tree.

    Terminal leaves have ``rule_applied=None`` and ``children=[]``.
    Internal nodes store the rule that expanded the non-terminal at this position.

    Attributes:
        symbol: The grammar symbol at this node (terminal or non-terminal name).
        rule_applied: The [Rule][SRToolkit.utils.grammar.Rule] used to expand this node.
            ``None`` for terminal leaves.
        children: Ordered child nodes corresponding to the symbols in
            ``rule_applied.rhs``.
    """

    symbol: str
    rule_applied: Optional[Rule]
    children: list[ParseTreeNode] = field(default_factory=list)


class ParseTree:
    """
    Full derivation history of an expression under a grammar.

    Unlike [Node][SRToolkit.utils.expression_tree.Node], which holds only terminal
    symbols for expression evaluation, a [ParseTree][SRToolkit.utils.grammar.ParseTree]
    retains every non-terminal and every production applied during the derivation.

    Examples:
        >>> r = Rule("E", ["x"])
        >>> leaf = ParseTreeNode("x", None)
        >>> root = ParseTreeNode("E", r, [leaf])
        >>> pt = ParseTree(root)
        >>> pt.to_token_list()
        ['x']
        >>> pt.productions_used()
        [Rule(lhs='E', rhs=['x'], weight=1.0, name=None)]

    Attributes:
        root: The root node of the parse tree.
    """

    def __init__(self, root: ParseTreeNode) -> None:
        self.root = root

    def to_token_list(self) -> list[str]:
        """
        Collect all terminal tokens in left-to-right order.

        Examples:
            >>> leaf1 = ParseTreeNode("a", None)
            >>> leaf2 = ParseTreeNode("+", None)
            >>> leaf3 = ParseTreeNode("b", None)
            >>> root = ParseTreeNode("E", Rule("E", ["a", "+", "b"]), [leaf1, leaf2, leaf3])
            >>> ParseTree(root).to_token_list()
            ['a', '+', 'b']

        Returns:
            Flat list of terminal tokens in left-to-right order.
        """
        tokens: list[str] = []
        self._collect_tokens(self.root, tokens)
        return tokens

    def _collect_tokens(self, node: ParseTreeNode, tokens: list[str]) -> None:
        if not node.children:
            tokens.append(node.symbol)
        else:
            for child in node.children:
                self._collect_tokens(child, tokens)

    def productions_used(self) -> list[Rule]:
        """
        Return all rules applied in the derivation, in pre-order.

        Examples:
            >>> leaf = ParseTreeNode("x", None)
            >>> r = Rule("E", ["x"])
            >>> root = ParseTreeNode("E", r, [leaf])
            >>> ParseTree(root).productions_used()
            [Rule(lhs='E', rhs=['x'], weight=1.0, name=None)]

        Returns:
            List of [Rule][SRToolkit.utils.grammar.Rule] objects in pre-order traversal order.
        """
        rules: list[Rule] = []
        self._collect_rules(self.root, rules)
        return rules

    def _collect_rules(self, node: ParseTreeNode, rules: list[Rule]) -> None:
        if node.rule_applied is not None:
            rules.append(node.rule_applied)
        for child in node.children:
            self._collect_rules(child, rules)


class Grammar:
    """
    A context-free grammar (CFG) or probabilistic context-free grammar (PCFG).

    Rules are added via [add_rule][SRToolkit.utils.grammar.Grammar.add_rule]. The set of
    non-terminals is derived automatically: a symbol is a non-terminal if and only
    if it appears as the ``lhs`` of at least one rule.

    A grammar is a CFG when every rule carries the default weight (``1.0``), making
    sampling uniform within each group. It is a PCFG when any rule has a weight
    that differs from ``1.0``.

    Constraints are registered via
    [add_constraint][SRToolkit.utils.grammar.Grammar.add_constraint]. During a
    derivation, [options][SRToolkit.utils.grammar.derivation.Derivation.options]
    returns only rules that every constraint's
    [allows][SRToolkit.utils.grammar.constraints.Constraint.allows] accepts.

    Examples:
        >>> g = Grammar([
        ...     Rule("E", ["E", "+", "F"], weight=0.4),
        ...     Rule("E", ["F"], weight=0.6),
        ...     Rule("F", ["x"]),
        ... ])
        >>> "E" in g.nonterminals
        True
        >>> g.is_pcfg()
        True
        >>> len(g.rules_for("E"))
        2
    """

    def __init__(self, rules: Optional[list[Rule]] = None, start: Optional[str] = None) -> None:
        """
        Args:
            rules: Optional list of [Rule][SRToolkit.utils.grammar.Rule] objects to add at
                construction time. Equivalent to calling
                [add_rule][SRToolkit.utils.grammar.Grammar.add_rule] for each entry.
            start: Default start non-terminal used by
                [start_derivation][SRToolkit.utils.grammar.Grammar.start_derivation]
                when no ``start`` argument is given.
        """
        self.start: Optional[str] = start
        self._rules: list[Rule] = []
        self._rules_by_lhs: dict[str, list[Rule]] = {}
        self._constraints: list = []
        for rule in rules or []:
            self.add_rule(rule)

    @property
    def nonterminals(self) -> set[str]:
        """
        Set of all non-terminal symbols in the grammar.

        A symbol is a non-terminal if and only if it appears as the ``lhs`` of
        at least one rule.

        Examples:
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["F"]))
            >>> g.add_rule(Rule("F", ["x"]))
            >>> g.nonterminals == {"E", "F"}
            True
        """
        return set(self._rules_by_lhs.keys())

    def add_rule(self, rule: Rule) -> None:
        """
        Add a production rule to the grammar.

        Examples:
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> len(g.rules_for("E"))
            1

        Args:
            rule: The [Rule][SRToolkit.utils.grammar.Rule] to register.
        """
        self._rules.append(rule)
        self._rules_by_lhs.setdefault(rule.lhs, []).append(rule)

    def add_constraint(self, constraint: "Constraint") -> None:
        """
        Register a [constraint][SRToolkit.utils.grammar.constraints.Constraint] applied at each derivation step.

        A rule is offered as an option only when every registered constraint's
        [allows][SRToolkit.utils.grammar.constraints.Constraint.allows] returns ``True``
        for it.

        Examples:
            >>> from SRToolkit.utils.grammar import MaxDepth
            >>> g = Grammar([Rule("E", ["E", "+", "E"]), Rule("E", ["x"])])
            >>> g.add_constraint(MaxDepth(0))
            >>> d = g.start_derivation("E")
            >>> [r.rhs for r in d.options()]
            [['x']]

        Args:
            constraint: A [Constraint][SRToolkit.utils.grammar.constraints.Constraint]
                instance — typically a built-in such as
                [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth],
                [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes],
                [MaxOccurrences][SRToolkit.utils.grammar.constraints.MaxOccurrences],
                [NoNested][SRToolkit.utils.grammar.constraints.NoNested], or
                [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency],
                or a user-defined subclass.
        """
        self._constraints.append(constraint)

    def rules_for(self, nonterminal: str) -> list[Rule]:
        """
        Return all rules whose ``lhs`` matches ``nonterminal``.

        Examples:
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["F"]))
            >>> g.add_rule(Rule("E", ["x"]))
            >>> len(g.rules_for("E"))
            2
            >>> g.rules_for("Z")
            []

        Args:
            nonterminal: Non-terminal to look up.

        Returns:
            List of matching [Rule][SRToolkit.utils.grammar.Rule] objects in insertion order.
        """
        return list(self._rules_by_lhs.get(nonterminal, []))

    def is_pcfg(self) -> bool:
        """
        Return ``True`` if any rule deviates from the default weight of ``1.0``.

        Examples:
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.add_rule(Rule("E", ["y"]))
            >>> g.is_pcfg()
            False
            >>> g.add_rule(Rule("F", ["z"], weight=0.3))
            >>> g.is_pcfg()
            True

        Returns:
            ``True`` if at least one rule has a weight other than ``1.0``.
        """
        return any(r.weight != 1.0 for r in self._rules)

    def validate(self, parse_tree: ParseTree, require_start: bool = False) -> bool:
        """
        Validate that ``parse_tree`` is structurally valid, uses only productions
        from this grammar, and satisfies every registered constraint.

        By default the root of the tree may be **any non-terminal** in the grammar,
        not just ``self.start``.  This lets you validate sub-trees (e.g. an
        ``F``-rooted fragment) in addition to full expressions.  Pass
        ``require_start=True`` to additionally enforce that the root symbol equals
        ``self.start`` (useful when validating complete, top-level expressions).

        The check replays the parse tree through a fresh
        [Derivation][SRToolkit.utils.grammar.derivation.Derivation] rooted at
        ``parse_tree.root.symbol``, walking nodes in leftmost order (mirroring the
        derivation frontier).  At each internal node it checks that:

        - The applied rule is permitted at the current frontier position — meaning
          it exists in the grammar **and** every registered constraint accepts it.
        - The number of children equals ``len(rule.rhs)``.
        - Each ``children[i].symbol`` equals ``rule.rhs[i]``.

        Terminal leaves are accepted when their symbol is not a non-terminal in
        this grammar.  A non-terminal symbol appearing as a leaf (unexpanded) is
        rejected.

        Examples:
            >>> g = Grammar(start="E")
            >>> r = Rule("E", ["x"])
            >>> g.add_rule(r)
            >>> leaf = ParseTreeNode("x", None)
            >>> root = ParseTreeNode("E", r, [leaf])
            >>> g.validate(ParseTree(root))
            True
            >>> foreign = Rule("E", ["y"])
            >>> root2 = ParseTreeNode("E", foreign, [ParseTreeNode("y", None)])
            >>> g.validate(ParseTree(root2))
            False
            >>> g.add_rule(Rule("F", ["x"]))
            >>> r_f = Rule("F", ["x"])
            >>> g.add_rule(r_f)
            >>> f_root = ParseTreeNode("F", r_f, [ParseTreeNode("x", None)])
            >>> g.validate(ParseTree(f_root))
            True
            >>> g.validate(ParseTree(f_root), require_start=True)
            False

        Args:
            parse_tree: The [ParseTree][SRToolkit.utils.grammar.ParseTree] to validate.
            require_start: When ``True``, return ``False`` if the root symbol does
                not equal ``self.start``.  Raises ``ValueError`` if ``self.start``
                is ``None`` and ``require_start`` is ``True``.

        Returns:
            ``True`` if the tree is structurally consistent, every production exists
            in this grammar, and all constraints would have permitted the derivation
            (and the root matches ``self.start`` when ``require_start=True``).

        Raises:
            ValueError: If ``require_start=True`` but ``self.start`` is ``None``.
        """
        if require_start:
            if self.start is None:
                raise ValueError("require_start=True but this grammar has no start symbol.")
            if parse_tree.root.symbol != self.start:
                return False
        root = parse_tree.root
        if root.rule_applied is None:
            return len(root.children) == 0 and root.symbol not in self.nonterminals
        try:
            d = self.start_derivation(root.symbol)
        except ValueError:
            return False
        stack = [root]
        while stack:
            node = stack.pop()
            if node.rule_applied is None:
                if node.symbol in self.nonterminals:
                    return False
                continue
            if d.complete:
                return False
            if node.rule_applied not in d.options():
                return False
            if len(node.children) != len(node.rule_applied.rhs):
                return False
            for child, expected in zip(node.children, node.rule_applied.rhs):
                if child.symbol != expected:
                    return False
            d.apply(node.rule_applied)
            for child in reversed(node.children):
                if child.symbol in self.nonterminals:
                    stack.append(child)
        return d.complete

    def start_derivation(self, start: Optional[str] = None) -> "Derivation":
        """
        Begin a new derivation.

        Examples:
            >>> g = Grammar(start="E")
            >>> g.add_rule(Rule("E", ["x"]))
            >>> d = g.start_derivation()
            >>> d.complete
            False

        Args:
            start: Start non-terminal.  Defaults to
                ``self.start`` when
                ``None``.

        Returns:
            A [Derivation][SRToolkit.utils.grammar.derivation.Derivation] at the first expansion step.

        Raises:
            ValueError: If the resolved start symbol is not a non-terminal in
                this grammar.
        """
        from .derivation import Derivation

        s = self.start if start is None else start

        if s is None:
            raise ValueError(
                "No start symbol. Either add it thorough the constructor or as parameter "
                "in the Grammar.start_derivation method."
            )
        if s not in self.nonterminals:
            raise ValueError(f"Start symbol '{s}' is not a non-terminal in this grammar.")

        return Derivation(self, s)

    def generate_one(
        self,
        start: Optional[str] = None,
        max_steps: int = 1000,
        max_retries: int = 10,
    ) -> Optional[list[str]]:
        """
        Generate a single expression by sampling the grammar to completion.

        Convenience wrapper around
        [start_derivation][SRToolkit.utils.grammar.grammar.Grammar.start_derivation]
        and [Derivation.generate][SRToolkit.utils.grammar.derivation.Derivation.generate].

        Each attempt runs a fresh derivation from scratch.  When a derivation
        exceeds ``max_steps`` rule applications (e.g. because random sampling
        kept choosing recursive rules), the attempt is discarded and a new one
        starts.  ``None`` is returned only when every attempt fails, signalling
        that the caller should either relax the grammar, add more liberal
        constraints, or increase ``max_steps`` / ``max_retries``.

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar(start="E")
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.generate_one()
            ['x']
            >>> g.generate_one(max_steps=0, max_retries=1) is None
            True

        Args:
            start: Start non-terminal. Defaults to ``self.start`` when ``None``.
            max_steps: Maximum rule applications per attempt. A negative number
                means unlimited (no retry logic applies).
            max_retries: Maximum number of fresh attempts before returning
                ``None``. Must be at least ``1``.

        Returns:
            A list of terminal tokens in left-to-right order, or ``None`` if
            every attempt exceeded ``max_steps``.

        Raises:
            ValueError: If the resolved start symbol is not a non-terminal in
                this grammar.
        """
        for _ in range(max(1, max_retries)):
            try:
                return self.start_derivation(start).generate(limit=max_steps)
            except RuntimeError:
                continue
        return None

    def to_dict(self) -> dict:
        """
        Serialise this grammar to a JSON-safe dictionary.

        Constraints are serialised via their own ``to_dict`` method.  User-defined
        constraint subclasses must implement ``to_dict``/``from_dict`` to survive the
        round-trip; built-in constraints are fully supported.

        Returns:
            Dictionary with keys ``start``, ``rules``, and ``constraints``.
        """
        return {
            "start": self.start,
            "rules": [r.to_dict() for r in self._rules],
            "constraints": [c.to_dict() for c in self._constraints],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Grammar":
        """
        Reconstruct a [Grammar][SRToolkit.utils.grammar.Grammar] from a dictionary
        produced by [to_dict][SRToolkit.utils.grammar.Grammar.to_dict].

        If ``d`` carries ``_bundle`` metadata (added by ``pack(..., configs=[...])``),
        every ``*_class`` path — including nested ``constraint_class`` paths — is
        rewritten to point at the installed bundle before reconstruction.

        Args:
            d: Dictionary with keys ``start``, ``rules``, and ``constraints``.

        Returns:
            A new [Grammar][SRToolkit.utils.grammar.Grammar] with all rules and
            constraints registered.
        """
        from ...bundle._relocate import _auto_bind
        from .constraints import Constraint

        d = _auto_bind(d)
        rules = [Rule.from_dict(r) for r in d.get("rules", [])]
        g = cls(rules, start=d.get("start"))
        for cd in d.get("constraints", []):
            g.add_constraint(Constraint.from_dict(cd))
        return g

    @classmethod
    def from_symbol_library(
        cls,
        symbol_library: Optional[SymbolLibrary] = None,
        start: Optional[str] = None,
    ) -> "Grammar":
        """
        Build a PCFG from a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary]
        using a generic operator-precedence non-terminal hierarchy.

        One non-terminal is created per unique ``OP`` precedence level present in
        the library.  Known precedences map to readable names; custom precedences
        fall back to ``L_{p}``:

        - ``OP_ADDITIVE`` → ``E``
        - ``OP_MULTIPLICATIVE`` → ``F``
        - ``OP_POWER`` → ``B``
        - custom precedence *p* → ``L_{p}``

        The chain runs from the lowest-precedence level (the grammar start) down
        to ``T``, ``K``, ``R``, and ``V``.  Only levels that have at least one
        operator are generated; there are no empty pass-through non-terminals.

        Heuristic weights are calibrated against empirical expression distributions
        (e.g. Wikipedia mathematical formulae): E recursion ~30 %, F recursion ~40 %,
        B recursion ~5 %, custom-level recursion ~40 %.

        Examples:
            >>> sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "sin", "^2"], 2)
            >>> g = Grammar.from_symbol_library(sl)
            >>> "E" in g.nonterminals and "V" in g.nonterminals
            True
            >>> g.is_pcfg()
            True

        Args:
            symbol_library: Token vocabulary.  Falls back to the active library
                from the context manager when ``None``.
            start: Override the name of the lowest-precedence non-terminal (and
                ``g.start``).  When ``None`` (default) the name is auto-inferred
                from the precedence constants (``"E"`` for standard libraries).

        Returns:
            A [Grammar][SRToolkit.utils.grammar.Grammar] with heuristic PCFG weights.

        Raises:
            ValueError: If the symbol library contains no variables, constants, or
                literals, making it impossible to generate any terminal expression.
        """
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_active()

        symbols = list(symbol_library.symbols.values())

        # Group operators by precedence level
        op_groups: dict[int, list[str]] = {}
        for s in symbols:
            if s["type"] == OP:
                op_groups.setdefault(s["precedence"], []).append(s["symbol"])
        sorted_precs = sorted(op_groups)

        R_fns = [s["symbol"] for s in symbols if s["type"] == FN and s["precedence"] == FN_PREFIX]
        P_fns = [s["symbol"] for s in symbols if s["type"] == FN and s["precedence"] == FN_POSTFIX]
        variables = [s["symbol"] for s in symbols if s["type"] == VAR]
        consts = [s["symbol"] for s in symbols if s["type"] == CONST]
        lits = [s["symbol"] for s in symbols if s["type"] == LIT]

        if not variables and not consts and not lits:
            raise ValueError(
                "Symbol library has no variables, constants, or literals; "
                "the grammar cannot generate any terminal expression."
            )

        # Collect terminal names so NT names can be made unique against them.
        all_terminals: set[str] = {sym for ops_list in op_groups.values() for sym in ops_list}
        all_terminals.update(R_fns + P_fns + variables + consts + lits)

        # If the caller explicitly chose a start name that is also a terminal symbol,
        # that is an unresolvable conflict — raise immediately.
        if start is not None and start in all_terminals:
            raise ValueError(
                f"start={start!r} is also a terminal symbol in the SymbolLibrary; "
                f"choose a different start symbol (e.g. {start + '_'!r})."
            )

        def _unique_nt(base: str, taken: set[str]) -> str:
            name = base
            while name in taken:
                name += "_"
            return name

        level_nts: dict[int, str] = {}
        for prec in sorted_precs:
            level_nts[prec] = _unique_nt(_level_name(prec), all_terminals)

        t_nt = _unique_nt("T", all_terminals)
        k_nt = _unique_nt("K", all_terminals)
        r_nt = _unique_nt("R", all_terminals)
        v_nt = _unique_nt("V", all_terminals)

        # top_nt is the topmost level of the hierarchy; functions recurse back here.
        top_nt = level_nts[sorted_precs[0]] if sorted_precs else t_nt

        g = cls(start=start if start is not None else top_nt)

        # Build operator-precedence chain.  Level i expands to itself OP level i+1;
        # the highest level chains to T.
        for i, prec in enumerate(sorted_precs):
            nt = level_nts[prec]
            next_nt = level_nts[sorted_precs[i + 1]] if i + 1 < len(sorted_precs) else t_nt
            ops = op_groups[prec]
            weight = _LEVEL_REC_WEIGHTS.get(prec, 0.4)
            infix = _LEVEL_RULE_INFIXES.get(prec, "op")

            for sym in ops:
                # Power is right-associative (a^b^c = a^(b^c)), so recurse on the right.
                if prec == OP_POWER:
                    g.add_rule(Rule(nt, [next_nt, sym, nt], weight=weight / len(ops), name=f"{nt}_{infix}_{sym}"))
                else:
                    g.add_rule(Rule(nt, [nt, sym, next_nt], weight=weight / len(ops), name=f"{nt}_{infix}_{sym}"))
            g.add_rule(Rule(nt, [next_nt], weight=1.0 - weight, name=f"{nt}_to_{next_nt}"))

        # T level — leaf dispatcher; K and V branches added only when their symbols exist.
        if variables and (consts or lits):
            g.add_rule(Rule(t_nt, [r_nt], weight=0.2, name=f"{t_nt}_to_{r_nt}"))
            g.add_rule(Rule(t_nt, [k_nt], weight=0.2, name=f"{t_nt}_to_{k_nt}"))
            g.add_rule(Rule(t_nt, [v_nt], weight=0.6, name=f"{t_nt}_to_{v_nt}"))
        elif consts or lits:
            g.add_rule(Rule(t_nt, [r_nt], weight=0.3, name=f"{t_nt}_to_{r_nt}"))
            g.add_rule(Rule(t_nt, [k_nt], weight=0.7, name=f"{t_nt}_to_{k_nt}"))
        else:
            g.add_rule(Rule(t_nt, [r_nt], weight=0.3, name=f"{t_nt}_to_{r_nt}"))
            g.add_rule(Rule(t_nt, [v_nt], weight=0.7, name=f"{t_nt}_to_{v_nt}"))

        # K level — constants and literals
        if lits and consts:
            for sym in lits:
                g.add_rule(Rule(k_nt, [sym], weight=0.2 / len(lits), name=f"{k_nt}_{sym}"))
            for sym in consts:
                g.add_rule(Rule(k_nt, [sym], weight=0.8 / len(consts), name=f"{k_nt}_{sym}"))
        elif lits:
            for sym in lits:
                g.add_rule(Rule(k_nt, [sym], weight=1.0 / len(lits), name=f"{k_nt}_{sym}"))
        elif consts:
            for sym in consts:
                g.add_rule(Rule(k_nt, [sym], weight=1.0 / len(consts), name=f"{k_nt}_{sym}"))

        # R level — prefix functions, postfix functions (top_nt sym), and parens.
        # Postfix rules live here directly (no separate P non-terminal).
        if R_fns:
            for sym in R_fns:
                g.add_rule(Rule(r_nt, [sym, "(", top_nt, ")"], weight=0.4 / len(R_fns), name=f"{r_nt}_fn_{sym}"))
            if P_fns:
                for sym in P_fns:
                    g.add_rule(Rule(r_nt, [top_nt, sym], weight=0.05 / len(P_fns), name=f"{r_nt}_postfix_{sym}"))
                g.add_rule(Rule(r_nt, ["(", top_nt, ")"], weight=0.55, name=f"{r_nt}_paren"))
            else:
                g.add_rule(Rule(r_nt, ["(", top_nt, ")"], weight=0.6, name=f"{r_nt}_paren"))
        else:
            if P_fns:
                for sym in P_fns:
                    g.add_rule(Rule(r_nt, [top_nt, sym], weight=0.05 / len(P_fns), name=f"{r_nt}_postfix_{sym}"))
                g.add_rule(Rule(r_nt, ["(", top_nt, ")"], weight=0.95, name=f"{r_nt}_paren"))
            else:
                g.add_rule(Rule(r_nt, ["(", top_nt, ")"], weight=1.0, name=f"{r_nt}_paren"))

        # V level — variables
        if variables:
            for sym in variables:
                g.add_rule(Rule(v_nt, [sym], weight=1.0 / len(variables), name=f"{v_nt}_{sym}"))

        return g

    @classmethod
    def from_grammar_string(cls, text: str, start: Optional[str] = None) -> "Grammar":
        """
        Construct a [Grammar][SRToolkit.utils.grammar.Grammar] from a string in
        NLTK production-rule notation.

        Each non-empty, non-comment line must have the form::

            LHS -> RHS_1 | RHS_2 | ...

        where each alternative is a space-separated sequence of symbols.
        Terminals are enclosed in single quotes (e.g. ``'+'``); unquoted
        tokens are treated as non-terminals.  An optional weight in square
        brackets at the end of an alternative is parsed as a
        ``float`` (e.g. ``E '+' F [0.4]``); alternatives without a weight
        default to ``1.0``.

        The start symbol may be embedded as a comment header emitted by
        [to_grammar_string][SRToolkit.utils.grammar.Grammar.to_grammar_string]::

            # start: E

        The explicit ``start`` parameter takes precedence over this comment.
        When ``start`` is ``None``, only the first ``# start:`` comment encountered
        is used; subsequent ones are ignored along with all other comment lines.

        Rule names are not preserved in NLTK notation and are set to ``None``
        on all returned [Rule][SRToolkit.utils.grammar.Rule] objects.

        Examples:
            >>> g = Grammar.from_grammar_string("E -> E '+' F | F\\nF -> 'x'", start="E")
            >>> sorted(g.nonterminals)
            ['E', 'F']
            >>> g.rules_for("F")[0].rhs
            ['x']
            >>> g.start
            'E'

            >>> g2 = Grammar.from_grammar_string(
            ...     "# start: E\\nE -> E '+' F [0.4] | F [0.6]\\nF -> 'x' [1.0]"
            ... )
            >>> g2.start
            'E'
            >>> g2.is_pcfg()
            True

        Args:
            text: Grammar specification in NLTK notation, optionally with a
                ``# start: <symbol>`` header line.
            start: Start non-terminal stored on the returned grammar.  Takes
                precedence over a ``# start:`` comment in ``text``.  Required
                when neither the parameter nor the comment is present.

        Returns:
            A new [Grammar][SRToolkit.utils.grammar.Grammar] populated with
            the parsed rules.

        Raises:
            ValueError: If a content line contains no ``->``.
            ValueError: If a weight token cannot be converted to ``float``.
            ValueError: If ``text`` contains no parseable rules.
            ValueError: If the start symbol cannot be determined from either
                the parameter or the ``# start:`` comment.
        """
        resolved_start = start
        rules: list[Rule] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if resolved_start is None and line.startswith("# start:"):
                    resolved_start = line[len("# start:") :].strip()
                continue
            rules.extend(Rule.from_line(line))
        if not rules:
            raise ValueError("No rules parsed from grammar string.")
        if resolved_start is None:
            raise ValueError(
                "No start symbol found. Either pass start=<nonterminal> or include a '# start: <symbol>' line."
            )
        return cls(rules, start=resolved_start)

    def to_grammar_string(self) -> str:
        """
        Serialise this grammar to a string in NLTK production-rule notation.

        All rules sharing the same left-hand side are written on a single
        line, separated by `` | ``.  Symbols that are non-terminals (i.e.
        appear as the ``lhs`` of at least one rule) are written unquoted;
        all other symbols are enclosed in single quotes.  When
        [is_pcfg][SRToolkit.utils.grammar.Grammar.is_pcfg] returns ``True``,
        the probability of each alternative (its weight divided by the sum of
        weights for that non-terminal) is appended in square brackets.

        Rule names and registered constraints are not included in the output.
        When ``self.start`` is set, a ``# start: <symbol>`` comment is
        prepended so that
        [from_grammar_string][SRToolkit.utils.grammar.Grammar.from_grammar_string]
        can reconstruct the grammar without requiring an explicit ``start``
        argument.

        Examples:
            >>> g = Grammar([
            ...     Rule("E", ["E", "+", "F"]),
            ...     Rule("E", ["F"]),
            ...     Rule("F", ["x"]),
            ... ], start="E")
            >>> print(g.to_grammar_string())
            # start: E
            E -> E '+' F | F
            F -> 'x'

            >>> gp = Grammar([
            ...     Rule("E", ["x"], weight=1.0),
            ...     Rule("E", ["y"], weight=3.0),
            ... ])
            >>> print(gp.to_grammar_string())
            E -> 'x' [0.25] | 'y' [0.75]

        Returns:
            Multi-line string in NLTK production-rule notation, optionally
            preceded by a ``# start:`` header.
        """
        nts = self.nonterminals
        pcfg = self.is_pcfg()
        lines: list[str] = []
        if self.start is not None:
            lines.append(f"# start: {self.start}")
        for lhs, rules in self._rules_by_lhs.items():
            parts: list[str] = []
            if pcfg:
                total = sum(r.weight for r in rules)
            for rule in rules:
                rhs_tokens = [sym if sym in nts else f"'{sym}'" for sym in rule.rhs]
                alt = " ".join(rhs_tokens)
                if pcfg:
                    alt += f" [{rule.weight / total}]"
                parts.append(alt)
            lines.append(f"{lhs} -> {' | '.join(parts)}")
        return "\n".join(lines)
