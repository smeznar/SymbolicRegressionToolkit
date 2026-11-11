"""Constraint protocol for grammar-guided expression generation.

A [GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint] is a hard filter: at each
derivation step its [allows][SRToolkit.utils.grammar.constraints.GrammarConstraint.allows] method
decides which candidate rules survive.

All constraints operate on an [ExpansionContext][SRToolkit.utils.grammar.constraints.ExpansionContext]
view that exposes the current derivation position together with the
constraint's own per-slot local state.

Per-derivation state comes in two flavours:

- **Local** — per-slot, inherited from parent to children.  The engine
  maintains a stack parallel to the open frontier; each constraint sees only its
  own local value for the current slot.
- **Global** — per-derivation scalar, lives on the engine alongside the
  frontier.  Suitable for counters and accumulators.

Both flavours are owned by the engine; the [Derivation][SRToolkit.utils.grammar.derivation.Derivation]
initialises them at [Grammar.start_derivation][SRToolkit.utils.grammar.Grammar.start_derivation]
and threads them through every [allows][SRToolkit.utils.grammar.constraints.GrammarConstraint.allows]
/ [update][SRToolkit.utils.grammar.constraints.GrammarConstraint.update] call. GrammarConstraint instances
carry only construction-time configuration and are safe to share across
parallel derivations.

## Built-in constraints

- [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth]: limit on the derivation depth
- [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes]: limit on the number of rule
  applications
- [MaxOccurrences][SRToolkit.utils.grammar.constraints.MaxOccurrences]: limit on how often a
  terminal occurs
- [NoNested][SRToolkit.utils.grammar.constraints.NoNested]: no symbol of a group nested inside one
  of the group
- [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency]: units
  consistent and equal to the target's

"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Generic, Iterable, Mapping, Optional, TypeVar, Union

from ...bundle._relocate import _auto_bind
from ..expression_constraints import (
    _DIMENSIONLESS,
    _check_unit_types,
    _check_units_fit,
    _expression_unit,
    _leaf_unit,
    _UnitConflict,
)
from ..expression_tree import is_float, tokens_to_tree
from ..symbol_library import SymbolLibrary
from ..types import CONST, LIT, VAR
from ..units import Unit, UnitRule
from .grammar import ParseTree, ParseTreeNode, Rule

L = TypeVar("L")
G = TypeVar("G")


# ---------------------------------------------------------------------------
# AncestorInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AncestorInfo:
    """One entry in an [ExpansionContext][SRToolkit.utils.grammar.constraints.ExpansionContext]'s ancestor chain.

    Attributes:
        nonterminal: The non-terminal symbol at this ancestor position.
        rule: The rule applied at this position.
        child_index: The index within ``rule.rhs`` that leads toward the
            current slot, counting only non-terminal positions.
    """

    nonterminal: str
    rule: Rule
    child_index: int


# ---------------------------------------------------------------------------
# ExpansionContext
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionContext(Generic[L]):
    """Read-only view of the current derivation position passed to constraints.

    The engine constructs an [ExpansionContext][SRToolkit.utils.grammar.constraints.ExpansionContext] on demand
    for each ``(constraint, candidate_rule)`` pair evaluated in
    [Derivation.options][SRToolkit.utils.grammar.derivation.Derivation.options] and for
    each ``(constraint, selected_rule)`` pair in
    [Derivation.apply][SRToolkit.utils.grammar.derivation.Derivation.apply].

    Attributes:
        nonterminal: The non-terminal symbol being expanded at this position.
        local: This constraint's per-slot inherited state.
        steps: Number of rule applications made so far in the derivation
            (i.e. how many times ``apply()`` has been called).
        parent_rule: The rule whose application created this slot (``None`` for
            the start symbol).
        child_index: Index of this slot in ``parent_rule.rhs`` counting only
            non-terminal positions (``None`` for the start symbol).
        ancestors: Ancestor chain from the root to this slot's immediate
            parent, in root-first order.
        partial_tree: Root of the partially-built parse tree.  Read-only by
            contract — constraints must not mutate it.
        nonterminals: Frozen set of all non-terminal symbols in the grammar.
        siblings: [ParseTreeNode][SRToolkit.utils.grammar.grammar.ParseTreeNode]s of the
            non-terminal children of ``parent_rule`` that come before this slot, in order (``()``
            for the first child and the start symbol), the same objects as in ``partial_tree``. They
            are complete, because the derivation expands the leftmost slot first, so a constraint
            can read what they derived.
        frontier_size: Number of open (unexpanded) non-terminal slots in the
            current derivation frontier, including this slot.  Useful for
            budget constraints such as
            [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes] that need to
            account for the minimum number of rule applications still required
            to complete the derivation.
    """

    nonterminal: str
    local: L
    steps: int
    parent_rule: Optional[Rule]
    child_index: Optional[int]
    ancestors: tuple[AncestorInfo, ...]
    partial_tree: ParseTreeNode
    nonterminals: frozenset[str]
    frontier_size: int = 1
    siblings: tuple[ParseTreeNode, ...] = ()


# ---------------------------------------------------------------------------
# GrammarConstraint protocol
# ---------------------------------------------------------------------------


class GrammarConstraint(Generic[L, G]):
    """Base class for derivation constraints (hard filters).

    Subclass and override the methods you need. Constraints carry only
    construction-time configuration; all per-derivation state is managed by the
    engine and threaded through method arguments.

    **Scoping** — set ``nonterminals`` and/or ``rule_names`` to restrict
    [allows][SRToolkit.utils.grammar.constraints.GrammarConstraint.allows] to a subset of slots
    or rules. A scope miss is treated as implicit acceptance.
    [update][SRToolkit.utils.grammar.constraints.GrammarConstraint.update] is *always* called
    regardless of scope so that global counters stay accurate.

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        >>> class RejectY(GrammarConstraint):
        ...     def allows(self, slot, rule, global_): return rule.rhs != ["y"]
        >>> g.add_constraint(RejectY())
        >>> d = g.start_derivation("E")
        >>> [r.rhs for r in d.options()]
        [['x']]
    """

    #: If set, ``allows`` only activates when ``slot.nonterminal`` is in this
    #: set.  ``None`` means "any non-terminal".
    nonterminals: Optional[frozenset[str]] = None

    #: If set, ``allows`` is only called when ``rule.name`` is in this set.
    #: ``None`` means "any rule name".
    rule_names: Optional[frozenset[str]] = None

    def to_dict(self) -> dict:
        """Serialise this constraint to a JSON-safe dictionary.

        The base implementation stores only the fully-qualified class path under
        ``constraint_class``. Subclasses should call ``super().to_dict()`` and
        add their own constructor arguments so that ``from_dict`` can reconstruct
        the instance faithfully. See
        [from_dict][SRToolkit.utils.grammar.constraints.GrammarConstraint.from_dict]
        for an example.

        Returns:
            Dictionary with at least the key ``constraint_class``.
        """
        return {"constraint_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}

    @classmethod
    def from_dict(cls, d: dict) -> "GrammarConstraint":
        """Reconstruct a constraint from a dictionary produced by ``to_dict``.

        When called on the base [GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint]
        class, dispatches to the correct subclass via the ``constraint_class`` key using
        `importlib` — both built-in and user-defined subclasses are supported.
        When called on a concrete subclass, the subclass must override this method.

        The dictionary must contain at minimum the key ``constraint_class``, whose value
        is the fully-qualified class path (e.g. ``"mymodule.MyConstraint"``).  Any
        additional keys are forwarded to the subclass override.

        To make a custom subclass serialisable, override both ``to_dict`` and
        ``from_dict``::

            class MyConstraint(GrammarConstraint):
                def __init__(self, threshold: float) -> None:
                    self.threshold = threshold

                def to_dict(self) -> dict:
                    return {**super().to_dict(), "threshold": self.threshold}

                @classmethod
                def from_dict(cls, d: dict) -> "MyConstraint":
                    return cls(d["threshold"])

        Examples:
            >>> from SRToolkit.utils.grammar import GrammarConstraint, MaxDepth
            >>> c = GrammarConstraint.from_dict(MaxDepth(5).to_dict())
            >>> isinstance(c, MaxDepth) and c.limit == 5
            True

        Args:
            d: Dictionary previously returned by ``to_dict``.

        Returns:
            A reconstructed [GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint] instance.

        Raises:
            KeyError: If ``constraint_class`` is missing from ``d`` (dispatch path).
            ImportError: If the module cannot be imported (dispatch path).
            AttributeError: If the class cannot be found in the module (dispatch path).
            NotImplementedError: If called on a subclass that has not overridden this method.
        """
        if cls is GrammarConstraint:
            d = _auto_bind(d)
            class_path = d["constraint_class"]
            module_path, cls_name = class_path.rsplit(".", 1)
            try:
                resolved = getattr(importlib.import_module(module_path), cls_name)
            except (ImportError, AttributeError):
                raise ImportError(
                    f"Cannot import constraint class {class_path!r}. "
                    "If this is a bundle class, install the bundle first. "
                    "If the config has no '_bundle' key, call bind_config(config) manually."
                ) from None
            return resolved.from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")

    def initial_local(self, start: str) -> L:
        """Return the initial local state for the constraint.

        Args:
            start: The non-terminal the derivation starts from.

        Returns:
            The local state of the start slot (``None`` by default).
        """
        return None  # type: ignore[return-value]

    def initial_global(self) -> G:
        """Return the initial global state for the constraint."""
        return None  # type: ignore[return-value]

    def allows(self, slot: ExpansionContext[L], rule: Rule, global_: G) -> bool:
        """Return ``True`` if ``rule`` may be applied at ``slot``.

        Called only when the slot's non-terminal and rule name are within scope.

        Args:
            slot: Current derivation position with this constraint's local state.
            rule: Candidate production rule.
            global_: Current per-derivation global state.

        Returns:
            ``True`` to keep the rule in the candidate set.
        """
        return True

    def update(self, slot: ExpansionContext[L], rule: Rule, global_: G) -> tuple[list[L], G]:
        """Called after ``rule`` is applied at ``slot``.

        Returns per-child local states (one per non-terminal in ``rule.rhs``,
        in order) and the new global state. May be used to update global
        counters by returning a new ``global_`` value.

        Args:
            slot: Derivation position immediately before the rule was applied.
            rule: The rule that was applied.
            global_: Global state before this application.

        Returns:
            ``(child_locals, new_global)`` where ``len(child_locals)`` equals
            the number of non-terminals in ``rule.rhs``.
        """
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        return [slot.local] * n, global_


# ---------------------------------------------------------------------------
# Built-in constraints
# ---------------------------------------------------------------------------


class MaxDepth(GrammarConstraint[int, None]):
    """Hard limit on derivation depth.

    Local state is the remaining depth budget at each slot.  A rule is
    rejected when its application would require at least one recursive
    non-terminal child but the budget has reached zero.

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([
        ...     Rule("E", ["E", "+", "E"]),
        ...     Rule("E", ["x"]),
        ... ])
        >>> g.add_constraint(MaxDepth(0))
        >>> d = g.start_derivation("E")
        >>> all(r.rhs == ["x"] for r in d.options())
        True

    Args:
        limit: Maximum nesting depth (number of non-terminal expansion levels).
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def to_dict(self) -> dict:
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> "MaxDepth":
        return cls(d["limit"])

    def initial_local(self, start: str) -> int:
        return self.limit

    def allows(self, slot: ExpansionContext[int], rule: Rule, global_: None) -> bool:
        if slot.local <= 0:
            has_nt = any(s in slot.nonterminals for s in rule.rhs)
            return not has_nt
        return True

    def update(self, slot: ExpansionContext[int], rule: Rule, global_: None) -> tuple[list[int], None]:
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        return [slot.local - 1] * n, None


class MaxNodes(GrammarConstraint[None, int]):
    """Hard limit on the total number of rule applications in a derivation.

    Global state is a running count of applications so far.  A rule is
    rejected when applying it would push the count over the limit taking
    into account the non-terminal children it introduces (each of which
    will require at least one more application).

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([
        ...     Rule("E", ["E", "+", "E"]),
        ...     Rule("E", ["x"]),
        ... ])
        >>> g.add_constraint(MaxNodes(3))
        >>> d = g.start_derivation("E")
        >>> # At node-count 3 only the terminal rule survives
        >>> tokens = d.generate()
        >>> len(tokens) <= 5
        True

    Args:
        limit: Maximum number of rule applications.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def to_dict(self) -> dict:
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> "MaxNodes":
        return cls(d["limit"])

    def initial_global(self) -> int:
        return 0

    def allows(self, slot: ExpansionContext[None], rule: Rule, global_: int) -> bool:
        n_children = sum(1 for s in rule.rhs if s in slot.nonterminals)
        # After this application, the remaining open slots will be:
        # (frontier_size - 1) existing slots + n_children new slots.
        # Each requires at least one application, so the minimum total is:
        # global_ + 1 (this) + (frontier_size - 1 + n_children) remaining.
        min_total = global_ + slot.frontier_size + n_children
        return min_total <= self.limit

    def update(self, slot: ExpansionContext[None], rule: Rule, global_: int) -> tuple[list[None], int]:
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        return [None] * n, global_ + 1


class MaxOccurrences(GrammarConstraint[None, int]):
    """Hard limit on how many times a specific terminal symbol may appear.

    Global state counts occurrences committed so far. A rule is rejected if applying it would
    push the count over the limit, i.e. if the count plus the occurrences of the symbol in its
    ``rhs`` exceeds the limit (so ``E -> x * x`` is rejected under a limit of 1).

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([
        ...     Rule("E", ["E", "+", "E"]),
        ...     Rule("E", ["x"]),
        ...     Rule("E", ["y"]),
        ... ])
        >>> g.add_constraint(MaxOccurrences("x", 1))
        >>> d = g.start_derivation("E")
        >>> tokens = d.generate(limit=200)
        >>> tokens.count("x") <= 1
        True

    Args:
        symbol: Terminal token to track.
        limit: Maximum allowed occurrences.
    """

    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    def to_dict(self) -> dict:
        return {**super().to_dict(), "symbol": self.symbol, "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> "MaxOccurrences":
        return cls(d["symbol"], d["limit"])

    def initial_global(self) -> int:
        return 0

    def allows(self, slot: ExpansionContext[None], rule: Rule, global_: int) -> bool:
        return global_ + rule.rhs.count(self.symbol) <= self.limit

    def update(self, slot: ExpansionContext[None], rule: Rule, global_: int) -> tuple[list[None], int]:
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        count = rule.rhs.count(self.symbol)
        return [None] * n, global_ + count


class NoNested(GrammarConstraint[bool, None]):
    """Prevent any symbol in a group from appearing nested inside a symbol of the group.

    The same symbol counts: ``NoNested("sin")`` forbids ``sin(sin(x))``.

    Local state is a boolean "currently under any symbol in the group".  Any
    rule whose ``rhs`` contains a group symbol is rejected while the local
    state is ``True``.  Children inherit ``True`` once such a rule is applied.

    Pass a single string to forbid self-nesting only; pass multiple symbols to
    forbid cross-nesting within the group (e.g. ``NoNested(TRIG_FNS)`` prevents
    ``sin(cos(x))`` as well as ``sin(sin(x))``).

    Warning:
        This constraint propagates the "inside" flag to **all** non-terminal
        children of any rule whose ``rhs`` contains a group symbol.  It works
        correctly when each group symbol appears in a rule with exactly one
        non-terminal child (typical prefix-function rules such as
        ``E -> sin ( E )``) and for infix operators where every child should be
        blocked (e.g. ``E -> E + E``).  It does **not** work correctly for
        mixed rules that combine a function call with additional non-terminal
        siblings in a single production (e.g. ``E -> sin ( E ) + F``): the
        sibling ``F`` will be incorrectly treated as inside the group.  If your
        grammar uses such rules, implement a custom
        [GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint] instead.

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([
        ...     Rule("E", ["sin", "(", "E", ")"]),
        ...     Rule("E", ["cos", "(", "E", ")"]),
        ...     Rule("E", ["x"]),
        ... ])
        >>> g.add_constraint(NoNested(["sin", "cos"]))
        >>> d = g.start_derivation("E")
        >>> d.apply(g.rules_for("E")[0])   # apply sin(E)
        >>> # Now inside sin; both sin and cos rules should be gone
        >>> opts = d.options()
        >>> all("sin" not in r.rhs and "cos" not in r.rhs for r in opts)
        True

    Args:
        symbols: A single terminal token or an iterable of terminal tokens
            that form the nesting group.
    """

    def __init__(self, symbols: str | Iterable[str]) -> None:
        if isinstance(symbols, str):
            self.symbols: frozenset[str] = frozenset({symbols})
        else:
            self.symbols = frozenset(symbols)

    def to_dict(self) -> dict:
        return {**super().to_dict(), "symbols": sorted(self.symbols)}

    @classmethod
    def from_dict(cls, d: dict) -> "NoNested":
        return cls(d["symbols"])

    def initial_local(self, start: str) -> bool:
        return False

    def allows(self, slot: ExpansionContext[bool], rule: Rule, global_: None) -> bool:
        if slot.local:
            return self.symbols.isdisjoint(rule.rhs)
        return True

    def update(self, slot: ExpansionContext[bool], rule: Rule, global_: None) -> tuple[list[bool], None]:
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        under = slot.local or not self.symbols.isdisjoint(rule.rhs)
        return [under] * n, None


# ---------------------------------------------------------------------------
# DimensionalConsistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Second:
    # Local state of a second child: its required unit follows from the finished first sibling.
    rule: UnitRule
    parent: Optional[Unit]


class DimensionalConsistency(GrammarConstraint[Any, None]):
    """Generate only expressions whose units are consistent and equal to the target's.

    Prunes a grammar's derivations so that every expression it completes passes
    [ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits] with the same units:
    variables and named constants (``"lit"`` symbols) have the given units; free constants, numbers
    and literals without a unit are dimensionless. How an operator or function combines units is
    its [UnitRule][SRToolkit.utils.units.UnitRule] in ``symbol_library``. This holds for grammars
    whose parse tree is the tree of the emitted expression, as with
    [Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library]; with an
    ambiguous grammar such as ``E -> E + E | E * E`` the emitted tokens can parse to a different
    tree.

    **Supported grammars.** A rule's role comes from its terminals, brackets ignored:

    - one ``"op"`` or ``"fn"`` terminal, with as many non-terminals as its unit rule accepts: that
      unit rule;
    - no terminals and one non-terminal (``E -> F``, ``R -> ( E )``): the requirement passes on;
    - one leaf terminal (a variable, literal, ``"const"`` symbol, or a number: any token ``float()``
      accepts) and no non-terminals: a leaf, with the unit above.

    Any other rule, e.g. ``E -> E + C`` or one with a terminal that is not a library symbol, bracket
    or number, raises ``ValueError`` naming the rule when it is a candidate in a derivation;
    [Grammar.generate_one][SRToolkit.utils.grammar.Grammar.generate_one] does not catch it.

    **Required units.** Each open slot has a required unit, or "any": the slot may take any unit, but
    its subtree must still have no conflict as ``ConsistentUnits`` counts them. The start slot
    requires ``target_unit``. A slot with required unit ``P`` rejects a leaf whose unit is not ``P``,
    and a rule whose result is dimensionless (``EXPONENT``, ``DIMENSIONLESS``, ``power(0)``) if ``P``
    is not dimensionless. Every other rule fits; any rule fits an any slot. The children of a rule
    with unit rule ``r`` must have these units, the second one decided when it is expanded, from the
    finished first child's unit ``U`` (the derivation is leftmost):

    | ``r`` | first child | second child |
    |---|---|---|
    | ``SAME`` | ``P`` | ``U`` |
    | ``MULTIPLY`` | any | ``P / U`` |
    | ``DIVIDE`` | any | ``U / P`` |
    | ``power(k)`` | ``P ** (1/k)``; any for ``power(0)`` | |
    | ``DIMENSIONLESS`` | dimensionless | dimensionless |
    | ``EXPONENT`` | dimensionless | dimensionless |

    Where ``P`` is any, a child given as ``P``, ``P / U``, ``U / P`` or ``P ** (1/k)`` is any too.
    ``EXPONENT`` (the ``^`` operator) is stricter here than in ``ConsistentUnits``: a dimensional
    base raised to a number (``X_0 ^ 2``) is never generated, even though ``ConsistentUnits``
    accepts it; postfix powers such as ``^2`` cover it.

    **Dead ends.** A required unit can turn out impossible once the first child is finished (e.g.
    the second factor would need ``m^3`` but the depth limit leaves only a leaf). The slot then has
    no allowed rule:
    [Derivation.generate][SRToolkit.utils.grammar.derivation.Derivation.generate] raises
    ``RuntimeError``, and ``generate_one`` starts again and returns ``None`` if every attempt
    fails. With tight units or depth limits, raise its ``max_retries``.

    Warning:
        Generation is slow for target units that are products of several dimensions. The first
        factor of a product is chosen without knowing whether the second can still reach the
        required unit, so most derivations end in a dead end. With
        ``Grammar.from_symbol_library`` and variables ``X_0`` in m and ``X_1`` in s, about a third
        of the derivations succeed for target ``m``, about one in seven for a dimensionless target,
        1–2 % for ``m s^-1``, and 0.3 % or less for ``m^2 s`` and for ``m^-2`` when both variables are
        in m. Every
        generated expression is correct; only the success rate is low. Raise ``max_retries`` or
        give the grammar fewer choices.

    Args:
        variable_units: The unit of every variable of ``symbol_library``, keyed by token.
        target_unit: The unit of the generated expressions.
        symbol_library: The library the grammar's terminals come from: their types and unit rules.
        literal_units: Units of named constants, keyed by ``"lit"`` token; literals left out are
            dimensionless. ``None`` means none.

    Raises:
        TypeError: If a unit is not a [Unit][SRToolkit.utils.units.Unit].
        ValueError: If a variable of ``symbol_library`` has no unit, or a key of ``variable_units``
            or ``literal_units`` is not a ``"var"`` or ``"lit"`` symbol of it; the message lists
            every problem.

    Attributes:
        variable_units: A copy of ``variable_units`` as a dict.
        target_unit: The target unit.
        symbol_library: The symbol library, as given.
        literal_units: The literal units as a dict; ``{}`` if none were given.

    Examples:
        >>> import numpy as np
        >>> from SRToolkit.utils import ConsistentUnits, SymbolLibrary
        >>> from SRToolkit.utils import units as u
        >>> from SRToolkit.utils.grammar import Grammar, MaxDepth
        >>> sl = SymbolLibrary.from_symbol_list(["+", "*", "/", "sin", "C"], num_variables=2)
        >>> units = {"X_0": u.m, "X_1": u.s}
        >>> g = Grammar.from_symbol_library(sl)
        >>> g.add_constraint(DimensionalConsistency(units, u.m / u.s, sl))
        >>> g.add_constraint(MaxDepth(8))
        >>> np.random.seed(0)
        >>> expr = g.generate_one(max_retries=50)
        >>> check = ConsistentUnits(units, u.m / u.s)
        >>> check.setup(sl)
        >>> check.pre_check(expr).passed
        True
    """

    def __init__(
        self,
        variable_units: Mapping[str, Unit],
        target_unit: Unit,
        symbol_library: SymbolLibrary,
        literal_units: Optional[Mapping[str, Unit]] = None,
    ) -> None:
        _check_unit_types("DimensionalConsistency", variable_units, target_unit, literal_units)
        self.variable_units = dict(variable_units)
        self.target_unit = target_unit
        self.symbol_library = symbol_library
        self.literal_units = dict(literal_units or {})
        self._sibling_units: dict[tuple[str, ...], Unit] = {}
        _check_units_fit("DimensionalConsistency", symbol_library, self.variable_units, self.literal_units)

    def to_dict(self) -> dict:
        """Return the saved form of this constraint.

        Returns:
            ``{"constraint_class": ..., "variable_units": ..., "target_unit": ..., "symbol_library":
            ..., "literal_units": ...}``: each unit as [Unit.to_dict][SRToolkit.utils.units.Unit.to_dict],
            ``variable_units`` and ``literal_units`` as ``{token: unit dict}`` (``{}`` if empty), and
            the library as [SymbolLibrary.to_dict][SRToolkit.utils.symbol_library.SymbolLibrary.to_dict].
        """
        return {
            **super().to_dict(),
            "variable_units": {t: v.to_dict() for t, v in self.variable_units.items()},
            "target_unit": self.target_unit.to_dict(),
            "symbol_library": self.symbol_library.to_dict(),
            "literal_units": {t: v.to_dict() for t, v in self.literal_units.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DimensionalConsistency":
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary produced by ``to_dict``.

        Returns:
            The rebuilt constraint.

        Raises:
            ValueError: If ``d`` has no ``"symbol_library"`` (a file saved by an earlier version of
                this class, which cannot be loaded), or a saved unit or the library is invalid.
            KeyError: If another key is missing.
        """
        if "symbol_library" not in d:
            raise ValueError(
                "DimensionalConsistency.from_dict: no 'symbol_library'; the constraint was saved by an earlier "
                "version, whose files cannot be loaded."
            )
        return cls(
            {t: Unit.from_dict(v) for t, v in d["variable_units"].items()},
            Unit.from_dict(d["target_unit"]),
            SymbolLibrary.from_dict(d["symbol_library"]),
            {t: Unit.from_dict(v) for t, v in d["literal_units"].items()},
        )

    def initial_local(self, start: str) -> Any:
        return self.target_unit

    def allows(self, slot: ExpansionContext[Any], rule: Rule, global_: None) -> bool:
        """Return whether ``rule`` fits the slot's required unit (see the class docstring).

        Returns:
            ``True`` if the rule fits.

        Raises:
            ValueError: If ``rule`` has a shape the constraint does not support (see "Supported
                grammars" in the class docstring), or if the grammar is ambiguous: a finished
                child's tokens parse to a tree with a unit conflict.
        """
        required = self._required(slot)
        shape = self._shape(rule, slot.nonterminals)
        if required is None or shape is None:
            return True
        if isinstance(shape, Unit):  # a leaf
            return shape == required
        if shape in (UnitRule.EXPONENT, UnitRule.DIMENSIONLESS) or shape.exponent == 0:
            return required == _DIMENSIONLESS
        return True

    def update(self, slot: ExpansionContext[Any], rule: Rule, global_: None) -> tuple[list[Any], None]:
        required = self._required(slot)
        shape = self._shape(rule, slot.nonterminals)
        if isinstance(shape, Unit):  # a leaf
            return [], None
        if shape is None:  # pass-through
            return [required], None
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        if shape == UnitRule.SAME:  # the second operand needs the first one's unit; no parent needed
            return ([required] if n == 1 else [required, _Second(shape, None)]), None
        if shape in (UnitRule.MULTIPLY, UnitRule.DIVIDE):
            return [None, _Second(shape, required)], None
        if shape.exponent is not None:  # power(k)
            if shape.exponent == 0 or required is None:
                return [None], None
            return [required ** (1 / shape.exponent)], None
        return [_DIMENSIONLESS] * n, None  # DIMENSIONLESS, EXPONENT

    def _required(self, slot: ExpansionContext[Any]) -> Optional[Unit]:
        # The unit the slot must have (None: any). A second child's requirement depends on the unit U
        # of its finished first sibling.
        local = slot.local
        if not isinstance(local, _Second):
            return local
        tokens = tuple(ParseTree(slot.siblings[0]).to_token_list())
        u = self._sibling_units.get(tokens)
        if u is None:
            u = self._tokens_unit(tokens)
            if len(self._sibling_units) >= 4096:  # a cache, not a record: keep it bounded
                self._sibling_units.clear()
            self._sibling_units[tokens] = u
        if local.rule == UnitRule.SAME:
            return u
        if local.parent is None:
            return None
        return local.parent / u if local.rule == UnitRule.MULTIPLY else u / local.parent

    def _tokens_unit(self, tokens: tuple[str, ...]) -> Unit:
        # Unit of a finished subtree's tokens; computed once per distinct subtree (allows runs per candidate).
        try:
            u = _expression_unit(
                tokens_to_tree(list(tokens), self.symbol_library),
                self.symbol_library,
                self.variable_units,
                self.literal_units,
            )
        except _UnitConflict:
            # Impossible if the emitted tokens parse to the derived tree (the documented precondition).
            raise ValueError(
                f"DimensionalConsistency: the finished subtree {' '.join(tokens)!r} parses to an expression with a "
                "unit conflict; the grammar is ambiguous (its emitted tokens parse to another tree)."
            ) from None
        return u

    def _shape(self, rule: Rule, nonterminals: frozenset[str]) -> Union[None, Unit, UnitRule]:
        # None for a pass-through rule, the leaf's Unit for a leaf, or the UnitRule of the rule's
        # operator or function; ValueError for an unsupported shape.
        terminals = [s for s in rule.rhs if s not in nonterminals and s not in ("(", ")")]
        n = sum(1 for s in rule.rhs if s in nonterminals)
        if not terminals and n == 1:
            return None
        if len(terminals) == 1:
            token = terminals[0]
            kind = self.symbol_library.get_type(token)
            if n == 0 and (kind in (VAR, LIT, CONST) or is_float(token)):
                return _leaf_unit(token, self.variable_units, self.literal_units)
            unit_rule = self.symbol_library.get_unit_rule(token)
            if unit_rule is not None and n in unit_rule.arities:
                return unit_rule
        raise ValueError(
            f"DimensionalConsistency: unsupported rule {rule.lhs} -> {' '.join(rule.rhs)}; a rule needs one "
            "operator or function terminal with as many non-terminals as its unit rule accepts, one leaf "
            "terminal, or no terminals and one non-terminal."
        )
