"""What a symbolic regression task tells the approach besides the data.

[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge] collects what a
dataset deliberately gives the approach that solves it:

- the symbols expressions may use (the symbol library),
- the input distribution (one [Sampler][SRToolkit.dataset.sampling.Sampler] per variable),
- the range free constants are fitted in (``constant_bounds``),
- advisory [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]s:
  hints about the search space that an approach may use or ignore,
- an optional reference [Grammar][SRToolkit.utils.grammar.grammar.Grammar] (rule weights a
  grammar-based approach can start from),
- a description of the task, of each variable, of named constants (literals) and of the target
  ([VariableInfo][SRToolkit.dataset.background_knowledge.VariableInfo]: name, description and
  physical [Unit][SRToolkit.utils.units.Unit]).

Priors about the answer's form are advisory constraints: each has a text form
([describe][SRToolkit.utils.expression_constraints.ExpressionConstraint.describe]) and a record form
([summary][SRToolkit.utils.expression_constraints.ExpressionConstraint.summary]), and
[priors][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.priors] lists the records. A prior
that cannot be checked is a constraint that implements neither check: the evaluator skips it, and it
is still described. Anything else goes into ``description``.

[SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] builds one from its arguments
(``symbol_library=``, ``samplers=``, ``advisory_constraints=``, ``variables=``, ``constants=``,
...).

Units are facts about the task: stating them does not constrain anything. To check expressions
against them, build a
[ConsistentUnits][SRToolkit.utils.expression_constraints.ConsistentUnits] constraint with
[from_background_knowledge][SRToolkit.utils.expression_constraints.ConsistentUnits.from_background_knowledge]
and register it as an advisory or a required constraint.

## What it does not hold

The answer stays out: the ground truth, the human-readable equation, the success threshold and
anything derived from them live on [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset], not
here.
[describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe] renders only
background knowledge, so its text can go into a prompt without leaking the answer (unlike
``str(dataset)``, which starts with the equation).

Required constraints are not background knowledge either: they are part of the objective and live
on the dataset (``SR_dataset.required_constraints``). Advisory and required constraints are the
same classes; the list a constraint is in decides its role (see
[expression_constraints][SRToolkit.utils.expression_constraints]).
"""

import logging
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from SRToolkit.dataset.sampling import Sampler
from SRToolkit.utils.expression_constraints import ExpressionConstraint, _set_up_copy
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.units import Unit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariableInfo:
    """What the task says about one variable, named constant or the target.

    All fields are optional; an empty ``VariableInfo()`` says nothing. Instances compare equal
    when all fields are equal, and are hashable.

    Args:
        name: Human-readable name, e.g. ``"v"`` or ``"velocity"``. This is not the token
            the expression uses (that is the key in ``BackgroundKnowledge.variables`` or
            ``BackgroundKnowledge.constants``).
        description: Free text, e.g. ``"velocity of the particle in the lab frame"``.
        unit: The physical unit, e.g. ``u.m / u.s``, or ``None`` if not stated.

    Raises:
        TypeError: If ``name`` or ``description`` is not a ``str``, or ``unit`` is not a
            [Unit][SRToolkit.utils.units.Unit] or ``None``.

    Attributes:
        name: The name, as given.
        description: The description, as given.
        unit: The unit, as given.

    Examples:
        >>> from SRToolkit.utils import units as u
        >>> v = VariableInfo("v", "velocity of the particle", u.m / u.s)
        >>> v.to_dict()
        {'name': 'v', 'description': 'velocity of the particle', 'unit': {'m': '1', 's': '-1'}}
    """

    name: str = ""
    description: str = ""
    unit: Optional[Unit] = None

    def __post_init__(self) -> None:
        for attr in ("name", "description"):
            if not isinstance(getattr(self, attr), str):
                raise TypeError(f"[VariableInfo] '{attr}' must be a str, got {type(getattr(self, attr)).__name__}.")
        if self.unit is not None and not isinstance(self.unit, Unit):
            raise TypeError(f"[VariableInfo] 'unit' must be a Unit or None, got {type(self.unit).__name__}.")

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary of this variable's information.

        Returns:
            ``{"name": name, "description": description, "unit": unit}``, with the unit as its
            [to_dict][SRToolkit.utils.units.Unit.to_dict], or ``None``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit.to_dict() if self.unit is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VariableInfo":
        """Rebuild a ``VariableInfo`` from the output of ``to_dict``.

        Missing keys take their defaults; other keys are ignored. ``unit`` is read with
        [Unit.from_dict][SRToolkit.utils.units.Unit.from_dict]; a missing or ``None`` unit gives
        ``None``.

        Args:
            d: A dictionary produced by
                [to_dict][SRToolkit.dataset.background_knowledge.VariableInfo.to_dict].

        Returns:
            A ``VariableInfo`` equal to the one that was saved.

        Raises:
            TypeError: As for the constructor.
            ValueError: If the saved unit is invalid (see
                [Unit.from_dict][SRToolkit.utils.units.Unit.from_dict]).
        """
        unit = d.get("unit")
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            unit=Unit.from_dict(unit) if unit is not None else None,
        )

    def describe(self) -> str:
        """Return a one-line description: ``"<name>: <description> [<unit>]"``.

        If only one of ``name`` and ``description`` is non-empty, it is written alone, without
        ``": "``. The unit, written with ``str``, is appended in brackets after a space, or alone if
        both are empty; it is left out if ``None``.

        Returns:
            The description, or ``""`` if all fields are empty.

        Examples:
            >>> from SRToolkit.utils import units as u
            >>> VariableInfo("v", "velocity", u.m / u.s).describe()
            'v: velocity [m s^-1]'
            >>> VariableInfo("v").describe()
            'v'
        """
        text = ": ".join(part for part in (self.name, self.description) if part)
        if self.unit is None:
            return text
        return f"{text} [{self.unit}]" if text else f"[{self.unit}]"


@dataclass(frozen=True, eq=False)
class BackgroundKnowledge:
    """Everything a dataset tells the approach besides the data, without the answer.

    See the module docstring for what belongs here and what does not.

    Args:
        symbol_library: The tokens expressions may use.
        samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per variable of
            ``symbol_library``, as a mapping ``{token: sampler}`` or a list in the library's
            variable order (the column order of ``X``). ``None`` if not given.
        constant_bounds: ``(lower, upper)`` range free constants are fitted in: finite, with
            ``lower < upper``. Defaults to ``(-5, 5)``.
        constraints: Advisory constraints (hints).
        reference_grammar: A grammar an approach may start from. Its terminal tokens must be
            symbols of ``symbol_library``, brackets or numbers.
        description: Free text, as a domain expert would write it: what was measured, under which
            conditions, and what is known about the phenomenon (signs, limiting cases,
            monotonicity, symmetries). Must not name the law or equation being discovered.
        variables: Information per variable, keyed by the variable's token in ``symbol_library``
            (e.g. ``"X_0"``). Variables may be left out.
        target: Information about the target, or ``None``.
        constants: Information per named constant, keyed by the constant's token, which must be a
            symbol of type ``"lit"`` in ``symbol_library`` (e.g. ``"c"`` for the speed of light).
            Constants may be left out.

    Raises:
        TypeError: If an argument has the wrong type, e.g. a lone constraint or a string as
            ``constraints``, or ``bool`` as a bound.
        ValueError: If ``samplers`` does not give exactly one sampler per variable (the message
            names missing or unknown tokens); ``constant_bounds`` is not finite with
            ``lower < upper``; a key of ``variables`` is not a variable of ``symbol_library``, or a
            key of ``constants`` is not one of its ``"lit"`` symbols; a
            constraint does not fit ``symbol_library`` (raised by its ``setup``); or
            ``reference_grammar`` has a terminal token that is not a symbol, bracket or number (the
            message names it).

    Attributes:
        symbol_library: The symbol library, as given.
        samplers: The samplers as a tuple in the library's variable order, or ``None``.
        constant_bounds: ``(lower, upper)`` as a tuple of two ``float``s.
        constraints: The advisory constraints, as a tuple.
        reference_grammar: The grammar, as given, or ``None``.
        description: The description, as given.
        variables: A copy of ``variables`` as a dict from token to ``VariableInfo``, in the
            library's variable order.
        target: The target information, or ``None``.
        constants: A copy of ``constants`` as a dict from token to ``VariableInfo``, in the
            library's order of ``"lit"`` symbols.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> from SRToolkit.utils.expression_constraints import MaxTreeDepth
        >>> bk = BackgroundKnowledge(
        ...     SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2),
        ...     description=(
        ...         "Voltage measured across metal conductors of different resistance, each carrying "
        ...         "a steady current, at room temperature. Without current there is no voltage; the "
        ...         "voltage grows with the current and with the resistance."
        ...     ),
        ...     variables={"X_0": VariableInfo("I", "current"), "X_1": VariableInfo("R", "resistance")},
        ...     target=VariableInfo("U", "voltage"),
        ...     constraints=[MaxTreeDepth(4)])
        >>> bk.constant_bounds
        (-5.0, 5.0)
        >>> bk.variables["X_0"].name
        'I'
    """

    symbol_library: SymbolLibrary
    samplers: Optional[Union[Mapping[str, Sampler], Sequence[Sampler]]] = None
    constant_bounds: Tuple[float, float] = (-5.0, 5.0)
    constraints: Iterable[ExpressionConstraint] = ()
    reference_grammar: Optional[Grammar] = None
    description: str = ""
    variables: Mapping[str, VariableInfo] = field(default_factory=dict)
    target: Optional[VariableInfo] = None
    constants: Mapping[str, VariableInfo] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol_library, SymbolLibrary):
            raise TypeError(
                f"[BackgroundKnowledge] 'symbol_library' must be a SymbolLibrary, got "
                f"{type(self.symbol_library).__name__}."
            )
        var_tokens = list(self.symbol_library.get_symbols_of_type("var"))

        # Samplers: one per variable, stored in the library's variable order.
        if self.samplers is not None:
            object.__setattr__(self, "samplers", self._normalise_samplers(self.samplers, var_tokens))

        # Constant bounds: two finite reals, lower < upper.
        bounds = self.constant_bounds
        try:
            pair = tuple(bounds)
        except TypeError:
            raise ValueError(
                f"[BackgroundKnowledge] 'constant_bounds' must be a (lower, upper) pair, got {bounds!r}."
            ) from None
        if len(pair) != 2:
            raise ValueError(f"[BackgroundKnowledge] 'constant_bounds' must be a (lower, upper) pair, got {bounds!r}.")
        if not all(isinstance(b, numbers.Real) and not isinstance(b, bool) for b in pair):
            raise TypeError(
                f"[BackgroundKnowledge] 'constant_bounds' must hold two real numbers (not bool), got {bounds!r}."
            )
        lower, upper = float(pair[0]), float(pair[1])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(
                f"[BackgroundKnowledge] 'constant_bounds' must be finite with lower < upper, got ({lower!r}, {upper!r})."
            )
        object.__setattr__(self, "constant_bounds", (lower, upper))

        # Advisory constraints.
        constraints = self.constraints
        try:
            constraints = tuple(constraints)
        except TypeError:
            raise TypeError(
                f"[BackgroundKnowledge] 'constraints' must be an iterable of ExpressionConstraints, got {type(constraints).__name__}."
            ) from None
        for c in constraints:
            if not isinstance(c, ExpressionConstraint):
                raise TypeError(
                    f"[BackgroundKnowledge] every constraint must be an ExpressionConstraint, got {type(c).__name__}."
                )
        object.__setattr__(self, "constraints", tuple(_set_up_copy(c, self.symbol_library) for c in constraints))

        if self.reference_grammar is not None and not isinstance(self.reference_grammar, Grammar):
            raise TypeError(
                f"[BackgroundKnowledge] 'reference_grammar' must be a Grammar or None, got {type(self.reference_grammar).__name__}."
            )
        if self.reference_grammar is not None:
            unknown_tokens = self.reference_grammar.unknown_tokens(self.symbol_library)
            if unknown_tokens:
                raise ValueError(
                    f"[BackgroundKnowledge] 'reference_grammar' has terminal tokens that are not symbols of "
                    f"'symbol_library', brackets or numbers: {sorted(unknown_tokens)}."
                )
        if not isinstance(self.description, str):
            raise TypeError(
                f"[BackgroundKnowledge] 'description' must be a str, got {type(self.description).__name__}."
            )

        # Variables: keyed by var token, stored in the library's variable order.
        variables = self.variables
        if not isinstance(variables, Mapping):
            raise TypeError("[BackgroundKnowledge] 'variables' must be a mapping from token to VariableInfo.")
        unknown = [k for k in variables if k not in var_tokens]
        if unknown:
            raise ValueError(
                f"[BackgroundKnowledge] 'variables' has keys that are not variables of the symbol library: {unknown}."
            )
        for k, v in variables.items():
            if not isinstance(v, VariableInfo):
                raise TypeError(
                    f"[BackgroundKnowledge] 'variables[{k!r}]' must be a VariableInfo, got {type(v).__name__}."
                )
        object.__setattr__(self, "variables", {t: variables[t] for t in var_tokens if t in variables})

        # Constants: keyed by lit token, stored in the library's order of lit symbols.
        constants = self.constants
        if not isinstance(constants, Mapping):
            raise TypeError("[BackgroundKnowledge] 'constants' must be a mapping from token to VariableInfo.")
        lit_tokens = list(self.symbol_library.get_symbols_of_type("lit"))
        unknown = [k for k in constants if k not in lit_tokens]
        if unknown:
            raise ValueError(
                f"[BackgroundKnowledge] 'constants' has keys that are not literals ('lit' symbols) of the symbol "
                f"library: {unknown}."
            )
        for k, v in constants.items():
            if not isinstance(v, VariableInfo):
                raise TypeError(
                    f"[BackgroundKnowledge] 'constants[{k!r}]' must be a VariableInfo, got {type(v).__name__}."
                )
        object.__setattr__(self, "constants", {t: constants[t] for t in lit_tokens if t in constants})

        if self.target is not None and not isinstance(self.target, VariableInfo):
            raise TypeError(
                f"[BackgroundKnowledge] 'target' must be a VariableInfo or None, got {type(self.target).__name__}."
            )

    @staticmethod
    def _normalise_samplers(
        samplers: Union[Mapping[str, Sampler], Sequence[Sampler]], var_tokens: List[str]
    ) -> Tuple[Sampler, ...]:
        """Return the samplers as a tuple in ``var_tokens`` order, validating coverage and types."""
        if isinstance(samplers, Mapping):
            missing = [t for t in var_tokens if t not in samplers]
            unknown = [t for t in samplers if t not in var_tokens]
            if missing or unknown:
                raise ValueError(
                    f"[BackgroundKnowledge] 'samplers' must give one sampler per variable; missing: {missing}, unknown: {unknown}."
                )
            ordered = [samplers[t] for t in var_tokens]
        else:
            ordered = list(samplers)
            if len(ordered) != len(var_tokens):
                raise ValueError(
                    f"[BackgroundKnowledge] 'samplers' has {len(ordered)} samplers but the symbol library has "
                    f"{len(var_tokens)} variables {var_tokens}."
                )
        for s in ordered:
            if not isinstance(s, Sampler):
                raise TypeError(f"[BackgroundKnowledge] every sampler must be a Sampler, got {type(s).__name__}.")
        return tuple(ordered)

    def sampling_grammar(self) -> Grammar:
        """Return the grammar to sample expressions from.

        ``reference_grammar`` if it is set (logged at ``INFO`` level), otherwise a generic grammar
        for ``symbol_library``
        ([Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library]).

        Returns:
            The grammar.
        """
        if self.reference_grammar is not None:
            logger.info("Sampling from the reference grammar of the background knowledge.")
            return self.reference_grammar
        return Grammar.from_symbol_library(self.symbol_library)

    def describe(self, required_constraints: Iterable[ExpressionConstraint] = ()) -> str:
        r"""Return a plain-text description of the task for people and LLM prompts.

        Contains nothing about the answer. Lines, in this order, each left out when its field is
        empty or ``None``:

        - ``Task: <description>``
        - ``Target: <target.describe()>`` (also left out if that is ``""``)
        - ``Variables:`` then one line ``  <token>: <VariableInfo.describe()>`` per entry of
          ``variables``, in the library's variable order (just ``  <token>`` if it describes to
          ``""``)
        - ``Constants:`` then one line ``  <token>: <VariableInfo.describe()>`` per entry of
          ``constants``, in the library's order of ``"lit"`` symbols (just ``  <token>`` if it
          describes to ``""``)
        - ``Symbols: <str(symbol_library)>`` (always present)
        - ``Free constants are fitted in [<lower>, <upper>].``, bounds written with ``str`` of a
          ``float`` (only if the library has a ``"const"`` symbol)
        - ``Input distribution:`` then ``  <token>: <sampler>`` per variable (only with samplers)
        - ``Enforced constraints (expressions that break them are rejected):`` then
          ``  - <constraint>`` per required constraint
        - ``Hints (advisory, may be ignored):`` then ``  - <constraint>`` per advisory constraint
        - ``Reference grammar: given``

        A constraint is written as its
        [describe][SRToolkit.utils.expression_constraints.ExpressionConstraint.describe], e.g.
        ``expression tree at most 4 levels deep``. A sampler is written as its
        [describe][SRToolkit.dataset.sampling.Sampler.describe], e.g. ``uniform on [1, 5]``.

        Args:
            required_constraints: The dataset's required constraints, listed apart from the
                advisory ones. Empty when called on its own;
                [SR_dataset.describe][SRToolkit.dataset.sr_dataset.SR_dataset.describe] passes the
                dataset's.

        Returns:
            The description, lines separated by ``"\n"``, without a trailing newline.

        Examples:
            >>> from SRToolkit.utils import SymbolLibrary
            >>> bk = BackgroundKnowledge(SymbolLibrary.from_symbol_list(["+", "*"], num_variables=1),
            ...     description="Toy task", variables={"X_0": VariableInfo("x", "position")})
            >>> print(bk.describe())
            Task: Toy task
            Variables:
              X_0: x: position
            Symbols: +, *, X_0
        """
        lines: List[str] = []
        if self.description:
            lines.append(f"Task: {self.description}")
        if self.target is not None and self.target.describe():
            lines.append(f"Target: {self.target.describe()}")
        if self.variables:
            lines.append("Variables:")
            for token, info in self.variables.items():
                text = info.describe()
                lines.append(f"  {token}: {text}" if text else f"  {token}")
        if self.constants:
            lines.append("Constants:")
            for token, info in self.constants.items():
                text = info.describe()
                lines.append(f"  {token}: {text}" if text else f"  {token}")
        lines.append(f"Symbols: {self.symbol_library}")
        if self.symbol_library.get_symbols_of_type("const"):
            lower, upper = self.constant_bounds
            lines.append(f"Free constants are fitted in [{lower}, {upper}].")
        if self.samplers is not None and len(self.samplers) > 0:
            lines.append("Input distribution:")
            samplers = cast(Tuple[Sampler, ...], self.samplers)  # normalised to a tuple in __post_init__
            for token, sampler in zip(self.symbol_library.get_symbols_of_type("var"), samplers):
                lines.append(f"  {token}: {sampler.describe()}")
        required = tuple(required_constraints)
        if required:
            lines.append("Enforced constraints (expressions that break them are rejected):")
            lines.extend(f"  - {c.describe()}" for c in required)
        if self.constraints:
            lines.append("Hints (advisory, may be ignored):")
            lines.extend(f"  - {c.describe()}" for c in self.constraints)
        if self.reference_grammar is not None:
            lines.append("Reference grammar: given")
        return "\n".join(lines)

    def priors(self) -> List[Dict[str, Any]]:
        """Return the advisory constraints as records, the structured form of the hints.

        One [summary][SRToolkit.utils.expression_constraints.ExpressionConstraint.summary] per
        advisory constraint, in order. Constraints of the same kind are all listed, not merged.

        Returns:
            The records; ``[]`` without advisory constraints.

        Examples:
            >>> from SRToolkit.utils import SymbolLibrary
            >>> from SRToolkit.utils.expression_constraints import MaxTreeDepth, NoNesting
            >>> bk = BackgroundKnowledge(SymbolLibrary.default_symbols(1),
            ...     constraints=[MaxTreeDepth(4), NoNesting(["exp", "ln"])])
            >>> bk.priors()
            [{'kind': 'max_tree_depth', 'limit': 4}, {'kind': 'no_nesting', 'symbols': ['exp', 'ln']}]
        """
        return [c.summary() for c in self.constraints]

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary of this background knowledge.

        Returns:
            A dictionary with ``"format_version": 1`` and one key per field, named as the field:
            ``symbol_library``, ``reference_grammar`` (or ``None``) and each sampler and constraint
            as their own ``to_dict``; ``samplers`` as a list in the library's variable order (or
            ``None``); ``constant_bounds`` as ``[lower, upper]``; ``variables`` as
            ``{token: VariableInfo.to_dict()}`` in the library's variable order; ``constants``
            likewise in the library's order of ``"lit"`` symbols (``{}`` if empty); ``target`` as its ``to_dict``
            (or ``None``); ``description`` as given.
        """
        return {
            "format_version": 1,
            "symbol_library": self.symbol_library.to_dict(),
            "samplers": (
                [s.to_dict() for s in cast(Tuple[Sampler, ...], self.samplers)] if self.samplers is not None else None
            ),
            "constant_bounds": list(self.constant_bounds),
            "constraints": [c.to_dict() for c in self.constraints],
            "reference_grammar": self.reference_grammar.to_dict() if self.reference_grammar is not None else None,
            "description": self.description,
            "variables": {token: info.to_dict() for token, info in self.variables.items()},
            "target": self.target.to_dict() if self.target is not None else None,
            "constants": {token: info.to_dict() for token, info in self.constants.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundKnowledge":
        """Rebuild background knowledge from the output of ``to_dict``.

        Missing keys other than ``symbol_library`` take their defaults (a missing
        ``format_version`` is read as ``1``); unknown keys are ignored.

        Args:
            d: A dictionary produced by
                [to_dict][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.to_dict].

        Returns:
            Background knowledge whose ``to_dict`` equals ``d`` with missing keys filled in, nested
            ones included (e.g. a variable's ``unit``).

        Raises:
            KeyError: If ``symbol_library`` is missing.
            ValueError: If ``format_version`` is not ``1``, or as for the constructor.
            TypeError: As for the constructor.
            ImportError: If a saved sampler or constraint class cannot be imported.
        """
        fmt = d.get("format_version", 1)
        if fmt != 1:
            raise ValueError(f"[BackgroundKnowledge.from_dict] Unsupported format_version: {fmt!r}. Expected 1.")
        library = SymbolLibrary.from_dict(d["symbol_library"])
        samplers = d.get("samplers")
        grammar = d.get("reference_grammar")
        target = d.get("target")
        return cls(
            symbol_library=library,
            samplers=[Sampler.from_dict(s) for s in samplers] if samplers is not None else None,
            constant_bounds=tuple(d.get("constant_bounds", (-5.0, 5.0))),  # type: ignore[arg-type]
            constraints=[ExpressionConstraint.from_dict(c) for c in d.get("constraints") or ()],
            reference_grammar=Grammar.from_dict(grammar) if grammar is not None else None,
            description=d.get("description", ""),
            variables={k: VariableInfo.from_dict(v) for k, v in (d.get("variables") or {}).items()},
            target=VariableInfo.from_dict(target) if target is not None else None,
            constants={k: VariableInfo.from_dict(v) for k, v in (d.get("constants") or {}).items()},
        )
