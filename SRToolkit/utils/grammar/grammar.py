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
from ..types import CONST, FN, LIT, OP, VAR

if TYPE_CHECKING:
    from .constraints import Constraint
    from .derivation import Derivation


@dataclass
class Rule:
    """
    A single production rule in a grammar.

    A rule is a CFG production when all rules in its group share the same weight
    (uniform sampling); when weights differ across rules for the same non-terminal,
    the grammar is treated as a PCFG and productions are sampled proportionally.
    Weights are unnormalised — the grammar normalises them at sampling time.

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
        rhs: Ordered sequence of symbols the non-terminal expands to.  Each
            element is either a terminal token (e.g. ``"+"`` or ``"sin"``) or
            the name of another non-terminal.  A symbol is treated as a
            non-terminal if and only if it appears as the ``lhs`` of at least
            one rule in the grammar.
        weight: Unnormalised sampling weight.  Defaults to ``1.0``, which
            produces uniform sampling within a group when all rules share the
            same weight.
        name: Optional stable identifier for this rule.  Used by constraints
            and reweighters for scoping and classification.  ``None`` by
            default.
    """

    lhs: str
    rhs: list[str]
    weight: float = 1.0
    name: Optional[str] = None


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

    Rules are added via [add_rule][SRToolkit.utils.grammar.Grammar.add_rule].  The set of
    non-terminals is derived automatically: a symbol is a non-terminal if and only
    if it appears as the ``lhs`` of at least one rule.

    A grammar is a CFG when every rule carries the default weight (``1.0``), making
    sampling uniform within each group.  It is a PCFG when any rule has a weight
    that differs from ``1.0``.

    Constraints are registered via
    [add_constraint][SRToolkit.utils.grammar.Grammar.add_constraint].  During a
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
        Register a constraint applied at each derivation step.

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

    def validate(self, parse_tree: ParseTree) -> bool:
        """
        Check that every production used in ``parse_tree`` exists in this grammar.

        Rules are compared by value (``lhs``, ``rhs``, ``weight``), not by identity,
        so parse trees built from separately constructed but equivalent rules are
        accepted.

        Examples:
            >>> g = Grammar()
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

        Args:
            parse_tree: The [ParseTree][SRToolkit.utils.grammar.ParseTree] to validate.

        Returns:
            ``True`` if every applied rule exists in this grammar.
        """
        known = {(r.lhs, tuple(r.rhs), r.weight) for r in self._rules}
        return all((r.lhs, tuple(r.rhs), r.weight) in known for r in parse_tree.productions_used())

    def check_constraints(self, parse_tree: ParseTree) -> bool:
        """
        Verify that every production in a completed parse tree would have been
        permitted by the registered constraints.

        Replays the parse tree through a fresh
        [Derivation][SRToolkit.utils.grammar.derivation.Derivation], checking
        [options][SRToolkit.utils.grammar.derivation.Derivation.options] at each
        step.  Returns ``False`` as soon as any constraint would have rejected
        the applied rule, ``True`` if every constraint accepts every production.

        Examples:
            >>> from SRToolkit.utils.grammar import MaxDepth
            >>> g = Grammar()
            >>> g.add_rule(Rule("E", ["E", "+", "E"], name="add"))
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.add_constraint(MaxDepth(3))
            >>> d = g.start_derivation("E")
            >>> tokens = d.generate()
            >>> g.check_constraints(d.to_parse_tree())
            True

        Args:
            parse_tree: A completed
                [ParseTree][SRToolkit.utils.grammar.ParseTree] to validate.

        Returns:
            ``True`` if all constraints accept every production used.
        """
        if not self._constraints:
            return True

        d = self.start_derivation(parse_tree.root.symbol)
        for rule in parse_tree.productions_used():
            if rule not in d.options():
                return False
            d.apply(rule)
        return True

    def verify(self, parse_tree: ParseTree) -> bool:
        """
        Check that ``parse_tree`` is both structurally valid and constraint-consistent.

        Equivalent to ``self.validate(parse_tree) and self.check_constraints(parse_tree)``.
        ``validate`` is evaluated first; if it returns ``False``, constraint checking is
        skipped.

        Examples:
            >>> g = Grammar()
            >>> r = Rule("E", ["x"])
            >>> g.add_rule(r)
            >>> leaf = ParseTreeNode("x", None)
            >>> root = ParseTreeNode("E", r, [leaf])
            >>> g.verify(ParseTree(root))
            True

        Args:
            parse_tree: The [ParseTree][SRToolkit.utils.grammar.ParseTree] to verify.

        Returns:
            ``True`` if every applied rule exists in this grammar and all constraints
            would have permitted the derivation.
        """
        return self.validate(parse_tree) and self.check_constraints(parse_tree)

    def start_derivation(self, start: Optional[str] = None) -> "Derivation":
        """
        Begin a new leftmost derivation.

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

    def generate_one(self, start: Optional[str] = None, max_steps: int = 1000) -> list[str]:
        """
        Generate a single expression by sampling the grammar to completion.

        Convenience wrapper around
        [start_derivation][SRToolkit.utils.grammar.grammar.Grammar.start_derivation]
        and [Derivation.generate][SRToolkit.utils.grammar.derivation.Derivation.generate].

        Examples:
            >>> from SRToolkit.utils.grammar import Grammar, Rule
            >>> g = Grammar(start="E")
            >>> g.add_rule(Rule("E", ["x"]))
            >>> g.generate_one()
            ['x']

        Args:
            start: Start non-terminal. Defaults to ``self.start`` when ``None``.
            max_steps: Maximum number of rule applications. ``-1`` means unlimited.

        Returns:
            Flat list of terminal tokens in left-to-right order.

        Raises:
            ValueError: If the resolved start symbol is not a non-terminal in
                this grammar.
            RuntimeError: If the derivation does not complete within
                ``max_steps`` rule applications.
        """
        return self.start_derivation(start).generate(limit=max_steps)

    @classmethod
    def from_symbol_library(
        cls,
        symbol_library: Optional[SymbolLibrary] = None,
        start: str = "E",
    ) -> "Grammar":
        """
        Build a PCFG from a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary]
        using the standard operator-precedence non-terminal hierarchy.

        The hierarchy and heuristic weights mirror those of
        [create_generic_pcfg][SRToolkit.utils.expression_generator.create_generic_pcfg]:

        - ``E`` — additive operators (precedence 0)
        - ``F`` — multiplicative operators (precedence 1)
        - ``B`` — power operators (precedence 2)
        - ``T`` — leaf dispatcher: ``R`` (function/paren), ``C`` (constant), or ``V`` (variable)
        - ``C`` — named literals (``lit``) and free constants (``const``)
        - ``R`` — unary functions (precedence 5) and parenthesised sub-expressions
        - ``P`` — postfix functions (precedence -1)
        - ``V`` — variables

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
            start: Default start non-terminal stored on the returned grammar and
                used by [start_derivation][SRToolkit.utils.grammar.Grammar.start_derivation]
                when called without arguments.  Default ``"E"``.

        Returns:
            A [Grammar][SRToolkit.utils.grammar.Grammar] with heuristic PCFG weights.
        """
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_active()

        symbols = list(symbol_library.symbols.values())
        E_ops = [s["symbol"] for s in symbols if s["type"] == OP and s["precedence"] == 0]
        F_ops = [s["symbol"] for s in symbols if s["type"] == OP and s["precedence"] == 1]
        B_ops = [s["symbol"] for s in symbols if s["type"] == OP and s["precedence"] == 2]
        R_fns = [s["symbol"] for s in symbols if s["type"] == FN and s["precedence"] == 5]
        P_fns = [s["symbol"] for s in symbols if s["type"] == FN and s["precedence"] == -1]
        variables = [s["symbol"] for s in symbols if s["type"] == VAR]
        consts = [s["symbol"] for s in symbols if s["type"] == CONST]
        lits = [s["symbol"] for s in symbols if s["type"] == LIT]

        g = cls(start=start)

        # E level — additive
        if E_ops:
            for sym in E_ops:
                g.add_rule(Rule("E", ["E", sym, "F"], weight=0.4 / len(E_ops), name=f"E_add_{sym}"))
            g.add_rule(Rule("E", ["F"], weight=0.6, name="E_to_F"))
        else:
            g.add_rule(Rule("E", ["F"], weight=1.0, name="E_to_F"))

        # F level — multiplicative
        if F_ops:
            for sym in F_ops:
                g.add_rule(Rule("F", ["F", sym, "B"], weight=0.4 / len(F_ops), name=f"F_mul_{sym}"))
            g.add_rule(Rule("F", ["B"], weight=0.6, name="F_to_B"))
        else:
            g.add_rule(Rule("F", ["B"], weight=1.0, name="F_to_B"))

        # B level — power
        if B_ops:
            for sym in B_ops:
                g.add_rule(Rule("B", ["B", sym, "T"], weight=0.05 / len(B_ops), name=f"B_pow_{sym}"))
            g.add_rule(Rule("B", ["T"], weight=0.95, name="B_to_T"))
        else:
            g.add_rule(Rule("B", ["T"], weight=1.0, name="B_to_T"))

        # T level — leaf dispatcher
        if consts or lits:
            g.add_rule(Rule("T", ["R"], weight=0.2, name="T_to_R"))
            g.add_rule(Rule("T", ["C"], weight=0.2, name="T_to_C"))
            g.add_rule(Rule("T", ["V"], weight=0.6, name="T_to_V"))
        else:
            g.add_rule(Rule("T", ["R"], weight=0.3, name="T_to_R"))
            g.add_rule(Rule("T", ["V"], weight=0.7, name="T_to_V"))

        # C level — constants and literals
        if lits and consts:
            for sym in lits:
                g.add_rule(Rule("C", [sym], weight=0.2 / len(lits), name=f"C_{sym}"))
            for sym in consts:
                g.add_rule(Rule("C", [sym], weight=0.8 / len(consts), name=f"C_{sym}"))
        elif lits:
            for sym in lits:
                g.add_rule(Rule("C", [sym], weight=1.0 / len(lits), name=f"C_{sym}"))
        elif consts:
            for sym in consts:
                g.add_rule(Rule("C", [sym], weight=1.0 / len(consts), name=f"C_{sym}"))

        # R level — functions and parenthesised expressions
        if R_fns:
            for sym in R_fns:
                g.add_rule(Rule("R", [sym, "(", "E", ")"], weight=0.4 / len(R_fns), name=f"R_fn_{sym}"))
            if P_fns:
                g.add_rule(Rule("R", ["P"], weight=0.15, name="R_to_P"))
                g.add_rule(Rule("R", ["(", "E", ")"], weight=0.45, name="R_paren"))
            else:
                g.add_rule(Rule("R", ["(", "E", ")"], weight=0.6, name="R_paren"))
        else:
            if P_fns:
                g.add_rule(Rule("R", ["P"], weight=0.15, name="R_to_P"))
                g.add_rule(Rule("R", ["(", "E", ")"], weight=0.85, name="R_paren"))
            else:
                g.add_rule(Rule("R", ["(", "E", ")"], weight=1.0, name="R_paren"))

        # P level — postfix functions
        if P_fns:
            total = sum(1.0 / abs(float(sym[1:])) for sym in P_fns)
            for sym in P_fns:
                g.add_rule(Rule("P", ["(", "E", ")", sym], weight=(1.0 / abs(float(sym[1:]))) / total, name=f"P_{sym}"))

        # V level — variables
        if variables:
            for sym in variables:
                g.add_rule(Rule("V", [sym], weight=1.0 / len(variables), name=f"V_{sym}"))

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
        default to ``1.0``.  Lines beginning with ``#`` are ignored.

        Rule names are not preserved in NLTK notation and are set to ``None``
        on all returned [Rule][SRToolkit.utils.grammar.Rule] objects.

        Examples:
            >>> g = Grammar.from_grammar_string("E -> E '+' F | F\\nF -> 'x'")
            >>> sorted(g.nonterminals)
            ['E', 'F']
            >>> g.rules_for("F")[0].rhs
            ['x']
            >>> g.start
            'E'

            >>> g2 = Grammar.from_grammar_string(
            ...     "E -> E '+' F [0.4] | F [0.6]\\nF -> 'x' [1.0]"
            ... )
            >>> g2.is_pcfg()
            True
            >>> g2.rules_for("E")[0].weight
            0.4

        Args:
            text: Grammar specification in NLTK notation.
            start: Start non-terminal stored on the returned grammar.  When
                ``None`` (default), the LHS of the first production in
                ``text`` is used.

        Returns:
            A new [Grammar][SRToolkit.utils.grammar.Grammar] populated with
            the parsed rules.

        Raises:
            ValueError: If a weight token cannot be converted to ``float``.
        """
        rules: list[Rule] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                continue
            lhs, rhs_part = line.split("->", 1)
            lhs = lhs.strip()
            for alt in rhs_part.split("|"):
                alt = alt.strip()
                if not alt:
                    continue
                weight = 1.0
                m = re.search(r"\[([^\]]+)\]\s*$", alt)
                if m:
                    weight = float(m.group(1))
                    alt = alt[: m.start()].strip()
                tokens = [
                    t[1:-1] if t.startswith("'") and t.endswith("'") else t for t in re.findall(r"'[^']*'|\S+", alt)
                ]
                if tokens:
                    rules.append(Rule(lhs, tokens, weight=weight))
        if start is None and rules:
            start = rules[0].lhs
        return cls(rules, start=start)

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
        The round-trip ``Grammar.from_grammar_string(g.to_grammar_string())`` reconstructs an
        equivalent grammar (same rules and probabilities) with ``name=None`` on every
        rule and no constraints registered.

        Examples:
            >>> g = Grammar([
            ...     Rule("E", ["E", "+", "F"]),
            ...     Rule("E", ["F"]),
            ...     Rule("F", ["x"]),
            ... ])
            >>> print(g.to_grammar_string())
            E -> E '+' F | F
            F -> 'x'

            >>> gp = Grammar([
            ...     Rule("E", ["x"], weight=1.0),
            ...     Rule("E", ["y"], weight=3.0),
            ... ])
            >>> print(gp.to_grammar_string())
            E -> 'x' [0.25] | 'y' [0.75]

        Returns:
            Multi-line string in NLTK production-rule notation.
        """
        nts = self.nonterminals
        pcfg = self.is_pcfg()
        lines: list[str] = []
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
