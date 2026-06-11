"""
Constraint protocol for grammar-guided expression generation.

A [Constraint][SRToolkit.utils.grammar.constraints.Constraint] is a hard filter: at each
derivation step its [allows][SRToolkit.utils.grammar.constraints.Constraint.allows] method
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
and threads them through every [allows][SRToolkit.utils.grammar.constraints.Constraint.allows]
/ [update][SRToolkit.utils.grammar.constraints.Constraint.update] call. Constraint instances
carry only construction-time configuration and are safe to share across
parallel derivations.

Built-in constraints
--------------------
- [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth]
- [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes]
- [MaxOccurrences][SRToolkit.utils.grammar.constraints.MaxOccurrences]
- [NoNested][SRToolkit.utils.grammar.constraints.NoNested]
- [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency]

"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, Iterable, Optional, TypeVar, cast

from ...bundle._relocate import _auto_bind
from ..symbol_library import SymbolLibrary
from ..types import CONST, FN, LIT, OP, OP_ADDITIVE, VAR
from .grammar import ParseTreeNode, Rule

L = TypeVar("L")
G = TypeVar("G")


# ---------------------------------------------------------------------------
# AncestorInfo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AncestorInfo:
    """
    One entry in an [ExpansionContext][SRToolkit.utils.grammar.constraints.ExpansionContext]'s ancestor chain.

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
    """
    Read-only view of the current derivation position passed to constraints.

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


# ---------------------------------------------------------------------------
# Constraint protocol
# ---------------------------------------------------------------------------


class Constraint(Generic[L, G]):
    """
    Base class for derivation constraints (hard filters).

    Subclass and override the methods you need. Constraints carry only
    construction-time configuration; all per-derivation state is managed by the
    engine and threaded through method arguments.

    **Scoping** — set ``nonterminals`` and/or ``rule_names`` to restrict
    [allows][SRToolkit.utils.grammar.constraints.Constraint.allows] to a subset of slots
    or rules. A scope miss is treated as implicit acceptance.
    [update][SRToolkit.utils.grammar.constraints.Constraint.update] is *always* called
    regardless of scope so that global counters stay accurate.

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])])
        >>> class RejectY(Constraint):
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
        """
        Serialise this constraint to a JSON-safe dictionary.

        The base implementation stores only the fully-qualified class path under
        ``constraint_class``. Subclasses should call ``super().to_dict()`` and
        add their own constructor arguments so that ``from_dict`` can reconstruct
        the instance faithfully. See
        [from_dict][SRToolkit.utils.grammar.constraints.Constraint.from_dict]
        for an example.

        Returns:
            Dictionary with at least the key ``constraint_class``.
        """
        return {"constraint_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        """
        Reconstruct a constraint from a dictionary produced by ``to_dict``.

        When called on the base [Constraint][SRToolkit.utils.grammar.constraints.Constraint]
        class, dispatches to the correct subclass via the ``constraint_class`` key using
        `importlib` — both built-in and user-defined subclasses are supported.
        When called on a concrete subclass, the subclass must override this method.

        The dictionary must contain at minimum the key ``constraint_class``, whose value
        is the fully-qualified class path (e.g. ``"mymodule.MyConstraint"``).  Any
        additional keys are forwarded to the subclass override.

        To make a custom subclass serialisable, override both ``to_dict`` and
        ``from_dict``::

            class MyConstraint(Constraint):
                def __init__(self, threshold: float) -> None:
                    self.threshold = threshold

                def to_dict(self) -> dict:
                    return {**super().to_dict(), "threshold": self.threshold}

                @classmethod
                def from_dict(cls, d: dict) -> "MyConstraint":
                    return cls(d["threshold"])

        Examples:
            >>> from SRToolkit.utils.grammar import Constraint, MaxDepth
            >>> c = Constraint.from_dict(MaxDepth(5).to_dict())
            >>> isinstance(c, MaxDepth) and c.limit == 5
            True

        Args:
            d: Dictionary previously returned by ``to_dict``.

        Returns:
            A reconstructed [Constraint][SRToolkit.utils.grammar.constraints.Constraint] instance.

        Raises:
            KeyError: If ``constraint_class`` is missing from ``d`` (dispatch path).
            ImportError: If the module cannot be imported (dispatch path).
            AttributeError: If the class cannot be found in the module (dispatch path).
            NotImplementedError: If called on a subclass that has not overridden this method.
        """
        if cls is Constraint:
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
        """Return the initial local state for the constraint."""
        return None  # type: ignore[return-value]

    def initial_global(self) -> G:
        """Return the initial global state for the constraint."""
        return None  # type: ignore[return-value]

    def allows(self, slot: ExpansionContext[L], rule: Rule, global_: G) -> bool:
        """
        Return ``True`` if ``rule`` may be applied at ``slot``.

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
        """
        Called after ``rule`` is applied at ``slot``.

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


class MaxDepth(Constraint[int, None]):
    """
    Hard limit on derivation depth.

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


class MaxNodes(Constraint[None, int]):
    """
    Hard limit on the total number of rule applications in a derivation.

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


class MaxOccurrences(Constraint[None, int]):
    """
    Hard limit on how many times a specific terminal symbol may appear.

    Global state counts occurrences committed so far.  A rule whose ``rhs``
    contains the tracked symbol is rejected once the count equals the limit.

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
        if global_ >= self.limit:
            return self.symbol not in rule.rhs
        return True

    def update(self, slot: ExpansionContext[None], rule: Rule, global_: int) -> tuple[list[None], int]:
        n = sum(1 for s in rule.rhs if s in slot.nonterminals)
        count = rule.rhs.count(self.symbol)
        return [None] * n, global_ + count


class NoNested(Constraint[bool, None]):
    """
    Prevent any symbol in a group from appearing nested inside any other symbol
    in the same group.

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
        [Constraint][SRToolkit.utils.grammar.constraints.Constraint] instead.

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

TRANSCENDENTAL_FNS: frozenset[str] = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "exp",
        "exp2",
        "log",
        "log2",
        "log10",
        "ln",
    }
)
"""
Functions that both require dimensionless input and produce dimensionless output.
Used by [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency].
"""

UNIT_PRESERVING_FNS: frozenset[str] = frozenset(
    {
        "abs",
        "fabs",
        "floor",
        "ceil",
        "round",
        "sign",
        "sgn",
        "neg",
    }
)
"""
Functions whose output carries the same unit as their input.
Used by [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency].
"""

SQRT_FNS: frozenset[str] = frozenset({"sqrt"})
"""
Square-root functions: output unit = input unit raised to the power ½.
Used by [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency].
"""

CBRT_FNS: frozenset[str] = frozenset({"cbrt"})
"""
Cube-root functions: output unit = input unit raised to the power ⅓.
Used by [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency].
"""

Unit = dict[str, Fraction]


def _to_unit(raw: dict) -> Unit:
    return {dim: Fraction(exp) for dim, exp in raw.items() if Fraction(exp) != 0}


def _units_equal(a: Unit, b: Unit) -> bool:
    return _to_unit(a) == _to_unit(b)


class DimensionalConsistency(Constraint[Optional[Unit], None]):
    """
    Stateful constraint that enforces dimensional analysis during expression
    generation.

    Local state at each slot is the required unit (``Optional[Unit]``).
    ``None`` means "any unit is acceptable here" — used for the children of
    multiplicative operators, whose individual units are underdetermined.

    **Unit representation**: units are ``dict[str, Fraction]`` mapping base
    dimension names (e.g. ``"m"``, ``"s"``, ``"kg"``) to rational exponents.
    Dimensionless is ``{}`` (empty dict).

    **Free constants** (``const`` type): treated as dimensionless by default.
    Set ``allow_unit_polymorphic_constants=True`` to let them absorb any
    required unit.

    **Named physical constants** (``lit`` type, e.g. ``"g"``, ``"c"``):
    declared via ``constant_units``; checked like variables.

    **Undeclared variables or literals**: conservatively accepted.

    **Rule classification**: uses ``rule.name`` when available (see
    [Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library]
    for the naming scheme).  Falls back to ``rhs``-shape heuristics for
    unnamed rules.

    Examples:
        >>> from SRToolkit.utils.grammar import Grammar, Rule
        >>> from SRToolkit.utils.grammar import DimensionalConsistency
        >>> from fractions import Fraction
        >>> g = Grammar()
        >>> g.add_rule(Rule("E", ["E", "+", "F"], weight=0.6, name="E_add_+"))
        >>> g.add_rule(Rule("E", ["F"], weight=0.4, name="E_to_F"))
        >>> g.add_rule(Rule("F", ["v"], weight=0.5, name="F_v"))
        >>> g.add_rule(Rule("F", ["t"], weight=0.5, name="F_t"))
        >>> dc = DimensionalConsistency(
        ...     variable_units={"v": {"m": 1, "s": -1}, "t": {"s": 1}},
        ...     target_unit={"m": 1, "s": -1},
        ... )
        >>> g.add_constraint(dc)
        >>> d = g.start_derivation("E")
        >>> all(r.rhs != ["t"] for r in d.options())
        True

    Args:
        variable_units: Mapping from variable token to its unit.
        target_unit: Required unit of the generated expression.
        constant_units: Units for named physical constants (``lit`` type).
        symbol_library: Used to classify tokens by type and precedence.
        allow_unit_polymorphic_constants: If ``True``, free constants absorb
            whatever unit their slot requires.  Default ``False``.
    """

    def __init__(
        self,
        variable_units: dict[str, dict],
        target_unit: dict,
        constant_units: Optional[dict[str, dict]] = None,
        symbol_library: Optional[SymbolLibrary] = None,
        allow_unit_polymorphic_constants: bool = False,
    ) -> None:
        self._var_units: dict[str, Unit] = {k: _to_unit(v) for k, v in variable_units.items()}
        self._target: Unit = _to_unit(target_unit)
        self._const_units: dict[str, Unit] = {k: _to_unit(v) for k, v in (constant_units or {}).items()}
        self._sl = symbol_library
        self._allow_poly_const = allow_unit_polymorphic_constants

    @staticmethod
    def _unit_to_dict(unit: Unit) -> dict[str, str]:
        return {dim: str(exp) for dim, exp in unit.items()}

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "variable_units": {k: self._unit_to_dict(v) for k, v in self._var_units.items()},
            "target_unit": self._unit_to_dict(self._target),
            "constant_units": {k: self._unit_to_dict(v) for k, v in self._const_units.items()} or None,
            "allow_unit_polymorphic_constants": self._allow_poly_const,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DimensionalConsistency":
        return cls(
            variable_units=d["variable_units"],
            target_unit=d["target_unit"],
            constant_units=d.get("constant_units"),
            allow_unit_polymorphic_constants=d.get("allow_unit_polymorphic_constants", False),
        )

    def initial_local(self, start: str) -> Optional[Unit]:
        return self._target

    def allows(self, slot: ExpansionContext[Optional[Unit]], rule: Rule, global_: None) -> bool:
        if slot.local is None:
            return True
        return self._is_compatible(rule, slot.local, slot.nonterminals)

    def update(
        self,
        slot: ExpansionContext[Optional[Unit]],
        rule: Rule,
        global_: None,
    ) -> tuple[list[Optional[Unit]], None]:
        child_units = self._child_units(rule, slot.local, slot.nonterminals)
        return child_units, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_compatible(self, rule: Rule, required: Unit, nonterminals: frozenset[str]) -> bool:
        kind, info = self._classify_rule(rule, nonterminals)

        if kind == "leaf":
            return self._leaf_compatible(cast(list[str], info), required)

        if kind == "function":
            fn: str = cast(str, info)
            if fn in TRANSCENDENTAL_FNS:
                return _units_equal(required, {})
            return True

        return True

    def _leaf_compatible(self, terminals: list[str], required: Unit) -> bool:
        for t in terminals:
            if t in self._var_units:
                return _units_equal(self._var_units[t], required)
            if t in self._const_units:
                return _units_equal(self._const_units[t], required)
            if self._sl and t in self._sl.symbols:
                sym_type = self._sl.symbols[t]["type"]
                if sym_type == CONST:
                    return self._allow_poly_const or _units_equal(required, {})
                if sym_type == LIT:
                    return _units_equal(required, {})
                if sym_type == VAR:
                    return True
        return True

    def _child_units(self, rule: Rule, required: Optional[Unit], nonterminals: frozenset[str]) -> list[Optional[Unit]]:
        nt_count = sum(1 for s in rule.rhs if s in nonterminals)
        if nt_count == 0:
            return []

        kind, info = self._classify_rule(rule, nonterminals)

        if kind == "chain":
            return [required]

        if kind == "binary":
            if self._is_additive(cast(str, info)):
                return [required, required]
            return [None, None]

        if kind == "function":
            return [self._fn_child_unit(cast(str, info), required)]

        if kind == "postfix":
            postfix: str = cast(str, info)
            if postfix.startswith("^") and required is not None:
                try:
                    exp = Fraction(postfix[1:])
                    return [{dim: Fraction(v) / exp for dim, v in required.items()}]
                except (ValueError, ZeroDivisionError):
                    pass
            return [None]

        return [None] * nt_count

    def _classify_rule(self, rule: Rule, nonterminals: frozenset[str]) -> tuple[str, object]:
        # Fast path: use stable rule name when available
        if rule.name:
            name = rule.name
            if name.startswith("E_add_") or name.startswith("F_mul_") or name.startswith("B_pow_"):
                op = name.split("_", 2)[-1]
                return "binary", op
            if name in ("E_to_F", "F_to_B", "B_to_T", "T_to_R", "T_to_K", "T_to_V", "R_to_P"):
                return "chain", None
            if name.startswith("R_fn_"):
                return "function", name[5:]
            if name.startswith("P_"):
                return "postfix", name[2:]
            if name.startswith("V_") or name.startswith("K_"):
                terms = [s for s in rule.rhs if s not in nonterminals and s not in ("(", ")")]
                return "leaf", terms

        # Fallback: infer from rhs shape
        nt_count = sum(1 for s in rule.rhs if s in nonterminals)
        terms = [s for s in rule.rhs if s not in nonterminals and s not in ("(", ")")]

        if nt_count == 0:
            return "leaf", terms
        if nt_count == 1 and not terms:
            return "chain", None
        if nt_count == 2 and len(terms) == 1:
            return "binary", terms[0]
        if nt_count == 1 and len(terms) == 1:
            t = terms[0]
            fn_type = self._sl.symbols.get(t, {}).get("type") if self._sl else None
            if fn_type == FN or t in TRANSCENDENTAL_FNS or t in UNIT_PRESERVING_FNS or t in SQRT_FNS or t in CBRT_FNS:
                return "function", t
            if t.startswith("^"):
                return "postfix", t
        if nt_count == 1 and len(terms) > 1:
            for t in terms:
                if t.startswith("^"):
                    return "postfix", t

        return "unknown", None

    def _fn_child_unit(self, fn: str, parent_unit: Optional[Unit]) -> Optional[Unit]:
        if fn in TRANSCENDENTAL_FNS:
            return {}
        if fn in SQRT_FNS:
            if parent_unit is None:
                return None
            return {dim: Fraction(v) * 2 for dim, v in parent_unit.items()}
        if fn in CBRT_FNS:
            if parent_unit is None:
                return None
            return {dim: Fraction(v) * 3 for dim, v in parent_unit.items()}
        if fn in UNIT_PRESERVING_FNS:
            return parent_unit
        return None

    def _is_additive(self, op: str) -> bool:
        if self._sl:
            info = self._sl.symbols.get(op, {})
            return info.get("type") == OP and info.get("precedence") == OP_ADDITIVE
        return op in {"+", "-"}
