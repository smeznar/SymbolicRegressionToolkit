"""Constraints that a candidate expression must, or should, satisfy.

An [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint] checks a
candidate expression at one or both of two points of its evaluation:

- [pre_check][SRToolkit.utils.expression_constraints.ExpressionConstraint.pre_check] looks at the
  expression itself, before it is scored: its length, its symbols, its structure. It reads no data.
- [post_check][SRToolkit.utils.expression_constraints.ExpressionConstraint.post_check] looks at the
  scored model, a [ModelResult][SRToolkit.utils.types.ModelResult] with the fitted parameters,
  e.g. to require that the fitted model's output stays non-negative.

A constraint implements one of the checks or both. A check it does not implement keeps the base
class's default, which returns a [CheckResult][SRToolkit.utils.types.CheckResult] with
``implemented=False``; the evaluator skips it. Each check returns whether the expression passed, a
short message, and an ``amount`` saying how far outside the constraint it is (e.g. ``3`` for three
tokens too many). Amounts of different constraints are in different units and are not comparable.

## Data and setup

A constraint receives the data it owns at construction (limits, units, evaluation points), and
saves it with [to_dict][SRToolkit.utils.expression_constraints.ExpressionConstraint.to_dict]. The
context it shares with the evaluator, the symbol library, comes through
[setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup] and is not saved. The
[SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] calls ``setup`` in its constructor,
on its own shallow copies of the constraints, so the same instance can be given to evaluators with
different symbol libraries. Every built-in constraint needs the library: to use one on its own, call
``setup`` first, or its checks raise ``RuntimeError``. A custom constraint that does not need the
library can be used without ``setup``.

## Required and advisory constraints

The same constraint can play two roles; the evaluator list it is registered in decides which:

- **Required** constraints belong to the objective (``required_constraints``). An expression that
  fails one gets the status ``"rejected"`` and the error ``+inf``. Use them for restrictions every
  approach must respect, e.g. to compare approaches on the same search space.
- **Advisory** constraints describe the search space the task suggests (``advisory_constraints``).
  An approach may use them or ignore them. The evaluator records each failed advisory check in
  ``ModelResult.advisory_violations`` but does not change the score.

Penalties are not part of the evaluator: a custom
[RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] can hold constraints and combine
their ``amount``s into its score.

## Input forms of pre_check

``pre_check`` accepts a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree, and each
constraint converts it to the form it needs: constraints that look at the tree parse a token list
with [tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree]; constraints that count tokens
write a tree out with [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list]. A tree has no
unique token list: ``to_list`` places only the brackets that are needed, so a tree can have fewer
tokens than a token list with redundant brackets for the same expression. The evaluator always
passes the token list it stores as ``ModelResult.expr``, so a saved result re-checks to the same
verdict.

The expression must be well-formed. For a malformed token list, constraints that parse it raise
[ExpressionParseError][SRToolkit.utils.expression_tree.ExpressionParseError], while constraints
that only look at tokens give a result without noticing. The evaluator parses every expression
before checking it, so this cannot happen there.

## Error messages

The ``ValueError`` and ``RuntimeError`` messages of the built-in constraints start with the
constraint's class name (e.g. ``"MaxTokens: the limit must be a non-negative int, got -1."``), so
an error raised while a dataset or an evaluator is built says which constraint caused it.

## Built-in constraints

| Constraint | Check | Fails when |
|---|---|---|
| [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants] | pre | more free constants (tokens of type ``"const"``) than the limit |
| [MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] | pre | more infix tokens, brackets included, than the limit |
| [MaxTreeDepth][SRToolkit.utils.expression_constraints.MaxTreeDepth] | pre | the tree is higher than the limit (a single leaf has height 1) |
| [MaxTreeSize][SRToolkit.utils.expression_constraints.MaxTreeSize] | pre | the tree has more nodes than the limit |
| [RequiredVariables][SRToolkit.utils.expression_constraints.RequiredVariables] | pre | a required variable does not occur |
| [MaxSymbolCount][SRToolkit.utils.expression_constraints.MaxSymbolCount] | pre | a symbol, or a group of symbols together, occurs more often than the limit |
| [NoNesting][SRToolkit.utils.expression_constraints.NoNesting] | pre | a symbol of a group occurs below a symbol of the group, the same symbol included |
| [OutputBounds][SRToolkit.utils.expression_constraints.OutputBounds] | post | the fitted model's output leaves the bounds at one of the given points |

Expression constraints are separate from the
[GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint]s that prune a grammar's
derivations: an expression constraint checks a finished expression and is part of the task, a
grammar constraint is a tool of an approach that generates expressions from a grammar.

## Writing your own constraint

Subclass [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint] and
implement ``pre_check``, ``post_check`` or both. To save it with a dataset or an experiment, also
override ``to_dict`` and ``from_dict``, as the built-in constraints do: see
[MaxSymbolCount][SRToolkit.utils.expression_constraints.MaxSymbolCount] for plain arguments and
[OutputBounds][SRToolkit.utils.expression_constraints.OutputBounds] for an array, saved through
[serialization][SRToolkit.utils.serialization]:

```python
from SRToolkit.utils.expression_constraints import ExpressionConstraint
from SRToolkit.utils.types import CheckResult


class ForbidSymbol(ExpressionConstraint):
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def pre_check(self, expr):
        tokens = expr if isinstance(expr, list) else expr.to_list(self.symbol_library)  # set by setup
        n = tokens.count(self.symbol)
        if n == 0:
            return CheckResult()
        return CheckResult(passed=False, message=f"uses {self.symbol!r}", amount=n)

    def to_dict(self):
        return {**super().to_dict(), "symbol": self.symbol}

    @classmethod
    def from_dict(cls, d):
        return cls(d["symbol"])
```

The saved form is ``{"constraint_class": "<module>.<ClassName>", ...}``, where the remaining keys are
the constructor arguments; the symbol library is not saved.
[from_dict][SRToolkit.utils.expression_constraints.ExpressionConstraint.from_dict] imports the class
by that path, so user-defined constraints load without being registered anywhere. The class must
therefore be importable by that path: define it at module level in an importable module, not inside of
``__main__``, a notebook or a function.
"""

from __future__ import annotations

import importlib
import re
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ..bundle._relocate import _auto_bind
from .expression_compiler import compile_expr
from .expression_tree import Node, _as_tokens, _as_tree, _count_constants
from .serialization import _from_json_safe, _to_json_safe
from .symbol_library import SymbolLibrary
from .types import VAR, CheckResult, ModelResult


def _predict(model: ModelResult, points: np.ndarray, symbol_library: SymbolLibrary) -> Union[np.ndarray, str]:
    # Output of the fitted model at ``points``: compiles model.expr and evaluates it with
    # model.parameters, with NumPy warnings suppressed. Returns a message instead of an array if
    # the expression has free constants but model.parameters is None ("no fitted parameters") or
    # has the wrong length ("<n> fitted parameters, expression has <m> constants"). Shared by
    # post-check constraints.
    n_constants = _count_constants(model.expr, symbol_library)
    if model.parameters is None:
        if n_constants > 0:
            return "no fitted parameters"
        constants = None
    else:
        constants = np.asarray(model.parameters, dtype=float).ravel()
        if constants.size != n_constants:
            return f"{constants.size} fitted parameters, expression has {n_constants} constants"
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        output = compile_expr(model.expr, symbol_library)(points, constants)
        return np.broadcast_to(np.asarray(output, dtype=float), (points.shape[0],))


def _require_setup(constraint: "ExpressionConstraint") -> SymbolLibrary:
    # Returns constraint.symbol_library, or raises RuntimeError naming the class if setup has not
    # been called. Used by every built-in check.
    if constraint.symbol_library is None:
        raise RuntimeError(f"{type(constraint).__name__}: call setup(symbol_library) before checking expressions.")
    return constraint.symbol_library


def _resolve_constraint_class(d: dict) -> Tuple[Type["ExpressionConstraint"], dict]:
    # Used by ExpressionConstraint.from_dict: binds bundle paths (_auto_bind), imports the class
    # named by d["constraint_class"] and checks that it is an ExpressionConstraint subclass.
    # Returns the class and the bound dictionary.
    # KeyError if the key is missing, ImportError (naming the class path) if the class cannot be
    # imported, TypeError (naming the class path) if it is not an ExpressionConstraint.
    d = _auto_bind(d)
    class_path = d["constraint_class"]
    module_path, _, cls_name = class_path.rpartition(".")
    try:
        resolved = getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError, ValueError):
        raise ImportError(
            f"Cannot import constraint class {class_path!r}. "
            "If this is a bundle class, install the bundle first. "
            "If the config has no '_bundle' key, call bind_config(config) manually."
        ) from None
    if not (isinstance(resolved, type) and issubclass(resolved, ExpressionConstraint)):
        raise TypeError(f"{class_path!r} is not an ExpressionConstraint and cannot be loaded as one.")
    return resolved, d


def _bound(value: Any, unbounded: float, name: str, n_points: int) -> Optional[Union[float, np.ndarray]]:
    # OutputBounds bound: None stays None; a scalar becomes a float, an array a float array of one
    # entry per point; a bound equal to ``unbounded`` (+-inf) everywhere becomes None. NaN, or an
    # array of the wrong shape, raises ValueError.
    if value is None:
        return None
    bound = np.asarray(value, dtype=float)
    if bound.ndim == 0:
        scalar = float(bound)
        if np.isnan(scalar):
            raise ValueError(f"OutputBounds: {name} must not be NaN.")
        return None if scalar == unbounded else scalar
    if bound.shape != (n_points,):
        raise ValueError(
            f"OutputBounds: {name} must be a scalar or a 1-D array with one entry per point ({n_points}), "
            f"got shape {bound.shape}."
        )
    if np.any(np.isnan(bound)):
        raise ValueError(f"OutputBounds: {name} must not contain NaN.")
    return None if np.all(bound == unbounded) else bound


def _validate_limit(limit: int, name: str, allow_negative: bool = False) -> int:
    # Returns ``limit`` as an int if it is an int or numpy integer (bool rejected) and, unless
    # ``allow_negative``, non-negative; else raises ValueError naming ``name`` (the constraint class
    # or setting). Shared by the built-in limits and, with allow_negative=True (a negative value
    # disables the limit), by SR_evaluator and ParameterEstimator for max_constants /
    # max_expr_length.
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)):
        kind = "an int (a negative value disables the limit)" if allow_negative else "a non-negative int"
        raise ValueError(f"{name}: the limit must be {kind}, got {limit!r}.")
    if limit < 0 and not allow_negative:
        raise ValueError(f"{name}: the limit must be a non-negative int, got {limit!r}.")
    return int(limit)


class ExpressionConstraint:
    """Base class for constraints on candidate expressions.

    Subclasses take the data they own as constructor arguments and implement
    [pre_check][SRToolkit.utils.expression_constraints.ExpressionConstraint.pre_check],
    [post_check][SRToolkit.utils.expression_constraints.ExpressionConstraint.post_check] or both.
    The base class has no constructor arguments, and a subclass constructor need not call
    ``super().__init__()``. See [expression_constraints][SRToolkit.utils.expression_constraints]
    for the roles a constraint can play and how to write one.

    Attributes:
        symbol_library: The symbol library given to
            [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup], or ``None``
            before setup (a class attribute, so it exists on every instance).

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxConstants(1)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["C", "*", "X_0", "+", "C"])
        CheckResult(passed=False, message='2 constants, limit 1', amount=1.0, implemented=True, constraint='', stage='',
        error=None)
        >>> c.post_check(ModelResult(expr=["C", "*", "X_0"], error=0.1)).implemented
        False
    """

    symbol_library: Optional[SymbolLibrary] = None

    def setup(self, symbol_library: SymbolLibrary) -> None:
        """Give the constraint the symbol library its expressions are written in.

        The evaluator calls it in its constructor. The base implementation stores the library in
        ``symbol_library`` and never raises; calling it again replaces the library. Subclasses that
        override it, e.g. to validate their data against the library, call
        ``super().setup(symbol_library)``. An exception raised here propagates out of the
        evaluator's constructor.

        Args:
            symbol_library: The symbol library.
        """
        self.symbol_library = symbol_library

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Check an expression before it is scored.

        Must not modify ``expr``. The base implementation does not check anything.

        Args:
            expr: A well-formed expression, as an infix token list or a
                [Node][SRToolkit.utils.expression_tree.Node] tree (see "Input forms of pre_check"
                in [expression_constraints][SRToolkit.utils.expression_constraints]).

        Returns:
            The outcome. The base implementation returns ``CheckResult(implemented=False,
            message="<ClassName> implements no pre_check")``. A check a constraint does not
            implement never raises, whether or not ``setup`` was called. A passing check of a
            built-in constraint returns ``CheckResult()`` (``message=""``, ``amount=0.0``).

        Raises:
            RuntimeError: If the constraint needs the symbol library and
                [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup] has not
                been called (every built-in constraint needs it).
            ExpressionParseError: If ``expr`` is a malformed token list and the constraint parses
                it.
        """
        return CheckResult(implemented=False, message=f"{type(self).__name__} implements no pre_check")

    def post_check(self, model: ModelResult) -> CheckResult:
        """Check a scored model.

        The evaluator calls it only for models with status ``"ok"``. ``model.parameters`` holds the
        fitted constants, or ``None`` if the ranking fits none (e.g. BED); a check that needs them
        then fails. Must not modify ``model``. The base implementation does not check anything.

        Args:
            model: The scored model: its ``expr``, ``parameters``, ``error``, ``ranking_data`` and
                ``status``.

        Returns:
            The outcome. The base implementation returns ``CheckResult(implemented=False,
            message="<ClassName> implements no post_check")``. A check a constraint does not
            implement never raises, whether or not ``setup`` was called. A passing check of a
            built-in constraint returns ``CheckResult()`` (``message=""``, ``amount=0.0``).

        Raises:
            RuntimeError: If the constraint needs the symbol library and
                [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup] has not
                been called (every built-in constraint needs it).
        """
        return CheckResult(implemented=False, message=f"{type(self).__name__} implements no post_check")

    def describe(self) -> str:
        """Return a short phrase saying what the constraint requires, for people and LLM prompts.

        [BackgroundKnowledge.describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe]
        lists it under "Enforced constraints" or "Hints". The base implementation renders
        ``ClassName(key=repr(value), ...)`` from
        [to_dict][SRToolkit.utils.expression_constraints.ExpressionConstraint.to_dict] without
        ``"constraint_class"``; the built-ins return a phrase such as ``"at most 8 free
        constants"``. Does not need ``setup``.

        Returns:
            The phrase.

        Examples:
            >>> MaxTreeDepth(4).describe()
            'expression tree at most 4 levels deep'
        """
        d = self.to_dict()
        args = ", ".join(f"{k}={v!r}" for k, v in d.items() if k != "constraint_class")
        return f"{type(self).__name__}({args})"

    def summary(self) -> dict:
        """Return what the constraint requires as a plain record, the structured form of ``describe``.

        A record is a JSON-safe dict with a ``"kind"`` key naming the constraint in snake_case and
        its parameters under readable keys, e.g. ``{"kind": "max_tree_depth", "limit": 4}``. Unlike
        ``to_dict`` it is for reading, not for rebuilding: it has no class path and leaves out
        bulky data such as evaluation points. The base implementation returns ``"kind"`` (the class
        name in snake_case) and the ``to_dict`` entries without ``"constraint_class"``. Does not
        need ``setup``.

        Returns:
            The record.

        Examples:
            >>> NoNesting(["sin", "cos"]).summary()
            {'kind': 'no_nesting', 'symbols': ['cos', 'sin']}
        """
        d = {k: v for k, v in self.to_dict().items() if k != "constraint_class"}
        kind = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", type(self).__name__).lower()  # MaxTreeDepth -> max_tree_depth
        return {"kind": kind, **d}

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary from which ``from_dict`` can rebuild this constraint.

        The base implementation stores only the class path under ``"constraint_class"``, which is
        enough for a constraint without constructor arguments. Subclasses with arguments call
        ``super().to_dict()`` and add them. The symbol library is not saved.

        Returns:
            A dictionary with the key ``"constraint_class"`` (the fully qualified class path) and
            the constructor arguments.

        Examples:
            >>> MaxConstants(4).to_dict()
            {'constraint_class': 'SRToolkit.utils.expression_constraints.MaxConstants', 'limit': 4}
        """
        return {"constraint_class": f"{type(self).__module__}.{type(self).__qualname__}"}

    @classmethod
    def from_dict(cls, d: dict) -> ExpressionConstraint:
        """Rebuild a constraint from the output of ``to_dict``.

        Called on the base class, it imports the class named by ``d["constraint_class"]`` and calls
        that class's ``from_dict``, so built-in and user-defined constraints load the same way.
        Called on a subclass, the subclass must override it. The rebuilt constraint is not set up.

        Args:
            d: A dictionary produced by ``to_dict``, possibly after a JSON round trip.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"constraint_class"`` (called on the base class).
            ImportError: If the named class cannot be imported (called on the base class).
            TypeError: If the named class is not an
                [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint],
                e.g. the grammar-only [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes]
                (called on the base class).
            NotImplementedError: If called on a subclass that does not override it, or if the named
                class does not override it.

        Examples:
            >>> c = ExpressionConstraint.from_dict(MaxTokens(20).to_dict())
            >>> type(c).__name__, c.limit
            ('MaxTokens', 20)
        """
        if cls is ExpressionConstraint:
            resolved, d = _resolve_constraint_class(d)
            if resolved is ExpressionConstraint:
                raise NotImplementedError("ExpressionConstraint.from_dict cannot rebuild the base class.")
            return resolved.from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")


class MaxConstants(ExpressionConstraint):
    """Limit the number of free constants in an expression.

    Counts the tokens of type ``"const"`` (e.g. ``C``); each occurrence is a separate free
    parameter. Literals such as ``pi`` and numbers such as ``2`` are not counted. Only looks at
    tokens; does not parse. This is the check behind the evaluator's ``max_constants`` setting.

    Args:
        limit: The largest number of free constants allowed. ``0`` forbids free constants.

    Raises:
        ValueError: If ``limit`` is not a non-negative ``int`` or numpy integer (``bool`` is
            rejected).

    Attributes:
        limit: The largest number of free constants allowed, as an ``int``.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxConstants(1)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["C", "*", "X_0", "+", "C"]).message
        '2 constants, limit 1'
        >>> c.pre_check(["C", "*", "X_0"]).passed
        True
    """

    def __init__(self, limit: int) -> None:
        self.limit = _validate_limit(limit, "MaxConstants")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if the expression has more than ``limit`` free constants.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if there are at most ``limit`` free constants; otherwise failed with the message
            ``"<n> constants, limit <limit>"`` and ``amount = float(n - limit)``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
        """
        symbol_library = _require_setup(self)
        n = _count_constants(_as_tokens(expr, symbol_library), symbol_library)
        if n <= self.limit:
            return CheckResult()
        return CheckResult(passed=False, message=f"{n} constants, limit {self.limit}", amount=float(n - self.limit))

    def describe(self) -> str:
        """Return ``"at most <limit> free constants"``.

        Returns:
            The phrase.
        """
        return f"at most {self.limit} free constants"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "limit": limit}``.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> MaxConstants:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"limit"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"limit"``.
            ValueError: If the limit is invalid.
        """
        return cls(d["limit"])


class MaxTokens(ExpressionConstraint):
    """Limit the length of an expression in infix tokens, brackets included.

    A token list is counted as given; a tree is first written out with
    [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list], which adds only the brackets that
    are needed. Only looks at tokens; does not parse. This is the check behind the evaluator's
    ``max_expr_length`` setting.

    Args:
        limit: The largest number of tokens allowed. ``0`` rejects every expression.

    Raises:
        ValueError: If ``limit`` is not a non-negative ``int`` or numpy integer (``bool`` is
            rejected).

    Attributes:
        limit: The largest number of tokens allowed, as an ``int``.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxTokens(3)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["(", "X_0", "+", "C", ")"]).message
        '5 tokens, limit 3'
        >>> c.pre_check(["X_0", "+", "C"]).passed
        True
    """

    def __init__(self, limit: int) -> None:
        self.limit = _validate_limit(limit, "MaxTokens")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if the expression has more than ``limit`` tokens.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if there are at most ``limit`` tokens; otherwise failed with the message
            ``"<n> tokens, limit <limit>"`` and ``amount = float(n - limit)``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
        """
        symbol_library = _require_setup(self)
        n = len(_as_tokens(expr, symbol_library))
        if n <= self.limit:
            return CheckResult()
        return CheckResult(passed=False, message=f"{n} tokens, limit {self.limit}", amount=float(n - self.limit))

    def describe(self) -> str:
        """Return ``"at most <limit> tokens"``.

        Returns:
            The phrase.
        """
        return f"at most {self.limit} tokens"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "limit": limit}``.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> MaxTokens:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"limit"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"limit"``.
            ValueError: If the limit is invalid.
        """
        return cls(d["limit"])


class MaxTreeDepth(ExpressionConstraint):
    """Limit the height of the expression tree.

    The height is that of [Node.height][SRToolkit.utils.expression_tree.Node.height]: a single leaf
    has height 1, and ``X_0 + C`` has height 2; brackets add nothing. Unlike the grammar constraint
    [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth], it does not depend on how a grammar
    derives the expression.

    Args:
        limit: The largest height allowed. ``0`` rejects every expression.

    Raises:
        ValueError: If ``limit`` is not a non-negative ``int`` or numpy integer (``bool`` is
            rejected).

    Attributes:
        limit: The largest height allowed, as an ``int``.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxTreeDepth(2)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["sin", "(", "X_0", "+", "C", ")"]).message
        'depth 3, limit 2'
        >>> c.pre_check(["X_0", "+", "C"]).passed
        True
    """

    def __init__(self, limit: int) -> None:
        self.limit = _validate_limit(limit, "MaxTreeDepth")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if the expression tree is higher than ``limit``.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if the height is at most ``limit``; otherwise failed with the message
            ``"depth <h>, limit <limit>"`` and ``amount = float(h - limit)``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
            ExpressionParseError: If ``expr`` is a malformed token list.
        """
        symbol_library = _require_setup(self)
        n = _as_tree(expr, symbol_library).height()
        if n <= self.limit:
            return CheckResult()
        return CheckResult(passed=False, message=f"depth {n}, limit {self.limit}", amount=float(n - self.limit))

    def describe(self) -> str:
        """Return ``"expression tree at most <limit> levels deep"``.

        Returns:
            The phrase.
        """
        return f"expression tree at most {self.limit} levels deep"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "limit": limit}``.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> MaxTreeDepth:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"limit"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"limit"``.
            ValueError: If the limit is invalid.
        """
        return cls(d["limit"])


class MaxTreeSize(ExpressionConstraint):
    """Limit the number of nodes in the expression tree.

    Every operator, function, variable, constant, literal and number is one node; brackets are not
    nodes. Unlike the grammar constraint [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes],
    which counts rule applications, it does not depend on how a grammar derives the expression.

    Args:
        limit: The largest number of nodes allowed. ``0`` rejects every expression.

    Raises:
        ValueError: If ``limit`` is not a non-negative ``int`` or numpy integer (``bool`` is
            rejected).

    Attributes:
        limit: The largest number of nodes allowed, as an ``int``.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxTreeSize(3)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["sin", "(", "X_0", "+", "C", ")"]).message
        '4 nodes, limit 3'
        >>> c.pre_check(["(", "X_0", "+", "C", ")"]).passed
        True
    """

    def __init__(self, limit: int) -> None:
        self.limit = _validate_limit(limit, "MaxTreeSize")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if the expression tree has more than ``limit`` nodes.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if there are at most ``limit`` nodes; otherwise failed with the message
            ``"<n> nodes, limit <limit>"`` and ``amount = float(n - limit)``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
            ExpressionParseError: If ``expr`` is a malformed token list.
        """
        symbol_library = _require_setup(self)
        n = len(_as_tree(expr, symbol_library))
        if n <= self.limit:
            return CheckResult()
        return CheckResult(passed=False, message=f"{n} nodes, limit {self.limit}", amount=float(n - self.limit))

    def describe(self) -> str:
        """Return ``"at most <limit> nodes in the expression tree"``.

        Returns:
            The phrase.
        """
        return f"at most {self.limit} nodes in the expression tree"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "limit": limit}``.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> MaxTreeSize:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"limit"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"limit"``.
            ValueError: If the limit is invalid.
        """
        return cls(d["limit"])


class RequiredVariables(ExpressionConstraint):
    """Require that given variables occur in the expression.

    Only checks which tokens occur; does not parse.

    Args:
        variables: The variable tokens that must occur. A single string is one variable.
            Duplicates are dropped, keeping the first occurrence; an empty sequence is never
            violated. ``None`` requires every symbol of type ``"var"`` in the symbol library given
            to [setup][SRToolkit.utils.expression_constraints.RequiredVariables.setup], which also
            checks that every listed token is such a variable.

    Attributes:
        variables: The required variable tokens as a tuple without duplicates, or ``None`` for all
            variables of the library.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = RequiredVariables()
        >>> c.setup(SymbolLibrary.default_symbols(3))
        >>> c.pre_check(["X_0", "+", "X_2"]).message
        'missing X_1'
        >>> c = RequiredVariables(["X_0", "X_2"])
        >>> c.setup(SymbolLibrary.default_symbols(3))
        >>> c.pre_check(["X_0", "+", "X_2"]).passed
        True
    """

    def __init__(self, variables: Optional[Union[str, Sequence[str]]] = None) -> None:
        if variables is None:
            self.variables: Optional[Tuple[str, ...]] = None
        elif isinstance(variables, str):
            self.variables = (variables,)
        else:
            self.variables = tuple(dict.fromkeys(variables))

    def setup(self, symbol_library: SymbolLibrary) -> None:
        """Store the symbol library and check that every listed token is one of its variables.

        Args:
            symbol_library: The symbol library.

        Raises:
            ValueError: If a token in ``variables`` is not a symbol of type ``"var"`` in
                ``symbol_library`` (e.g. a typo, or a function such as ``"sin"``). The message names
                the tokens and the library's variables.
        """
        library_variables = symbol_library.get_symbols_of_type(VAR)
        unknown = [v for v in (self.variables or ()) if v not in library_variables]
        if unknown:
            raise ValueError(
                f"RequiredVariables: {', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} not "
                f"a variable of the symbol library (its variables: {', '.join(library_variables)})."
            )
        super().setup(symbol_library)

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if a required variable does not occur in the expression.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if every required variable occurs; otherwise failed with the message
            ``"missing <tokens>"``, the missing tokens joined by ``", "`` in the order of
            ``variables`` (or the order the variables were added to the library, for
            ``variables=None``), and ``amount`` = the number of missing variables, as a ``float``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
        """
        symbol_library = _require_setup(self)
        present = set(_as_tokens(expr, symbol_library))
        if self.variables is not None:
            required: Sequence[str] = self.variables
        else:
            required = symbol_library.get_symbols_of_type(VAR)
        missing = [v for v in required if v not in present]
        if not missing:
            return CheckResult()
        return CheckResult(passed=False, message="missing " + ", ".join(missing), amount=float(len(missing)))

    def describe(self) -> str:
        """Return ``"uses each of <v1>, <v2>, ..."``, or ``"uses every variable"`` for all of them.

        Returns:
            The phrase.
        """
        if self.variables is None:
            return "uses every variable"
        return "uses each of " + ", ".join(self.variables)

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "variables": [...] or None}``.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "variables": None if self.variables is None else list(self.variables)}

    @classmethod
    def from_dict(cls, d: dict) -> RequiredVariables:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"variables"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"variables"``.
        """
        return cls(d["variables"])


class MaxSymbolCount(ExpressionConstraint):
    """Limit how often a symbol, or a group of symbols together, occurs in an expression.

    Counts the tokens that belong to ``symbols``: a token list as given, a tree after writing it out
    with [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list]. For a group, the occurrences
    of all its symbols are added up: ``MaxSymbolCount(["sin", "cos", "exp"], 2)`` allows at most two
    of these functions in total. To limit several symbols separately (e.g. at most one ``sin`` and
    at most one ``cos``), create one ``MaxSymbolCount`` per symbol. Tracking a bracket counts
    brackets, which a tree does not determine uniquely. Only looks at tokens; does not parse.

    Args:
        symbols: The token to count, e.g. ``"sin"`` or ``"X_0"``, or an iterable of tokens whose
            occurrences are counted together. Not checked against the library.
        limit: The largest number of occurrences allowed. ``0`` forbids the symbols.

    Raises:
        ValueError: If the group is empty, or if ``limit`` is not a non-negative ``int`` or numpy
            integer (``bool`` is rejected).

    Attributes:
        symbols: The counted tokens, as a frozenset.
        limit: The largest number of occurrences allowed, as an ``int``.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = MaxSymbolCount("X_0", 1)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> c.pre_check(["X_0", "*", "X_0"]).message
        "'X_0' occurs 2 times, limit 1"
        >>> trig = MaxSymbolCount(["sin", "cos"], 1)
        >>> trig.setup(SymbolLibrary.default_symbols(1))
        >>> trig.pre_check(["sin", "(", "X_0", ")", "+", "cos", "(", "X_0", ")"]).message
        "'cos', 'sin' occur 2 times together, limit 1"
    """

    def __init__(self, symbols: Union[str, Sequence[str]], limit: int) -> None:
        self.symbols: frozenset = frozenset({symbols}) if isinstance(symbols, str) else frozenset(symbols)
        if not self.symbols:
            raise ValueError("MaxSymbolCount: the group of symbols is empty.")
        self.limit = _validate_limit(limit, "MaxSymbolCount")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if the symbols occur more than ``limit`` times in total.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if there are at most ``limit`` occurrences; otherwise failed with
            ``amount = float(n - limit)`` and the message ``"<symbol> occurs <n> times, limit
            <limit>"`` for a single symbol, or ``"<symbols> occur <n> times together, limit
            <limit>"`` for a group, the symbols in ``repr`` form, sorted and joined by ``", "``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
        """
        symbol_library = _require_setup(self)
        n = sum(1 for token in _as_tokens(expr, symbol_library) if token in self.symbols)
        if n <= self.limit:
            return CheckResult()
        names = ", ".join(repr(symbol) for symbol in sorted(self.symbols))
        verb = "occurs" if len(self.symbols) == 1 else "occur"
        together = "" if len(self.symbols) == 1 else " together"
        return CheckResult(
            passed=False,
            message=f"{names} {verb} {n} times{together}, limit {self.limit}",
            amount=float(n - self.limit),
        )

    def describe(self) -> str:
        """Return ``"<s> at most <limit> times"``, or ``"<s1> and <s2> together at most <limit> times"``.

        Symbols are listed sorted.

        Returns:
            The phrase.
        """
        symbols = sorted(self.symbols)
        if len(symbols) == 1:
            return f"{symbols[0]} at most {self.limit} times"
        return f"{' and '.join(symbols)} together at most {self.limit} times"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "symbols": [...], "limit": limit}``, the symbols sorted.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "symbols": sorted(self.symbols), "limit": self.limit}

    @classmethod
    def from_dict(cls, d: dict) -> MaxSymbolCount:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the keys ``"symbols"`` and ``"limit"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If a key is missing.
            ValueError: If the group is empty or the limit is invalid.
        """
        return cls(d["symbols"], d["limit"])


class NoNesting(ExpressionConstraint):
    """Forbid a symbol of a group to occur below a symbol of the group in the expression tree.

    ``sin(X_0 + cos(X_0))`` nests ``cos`` in ``sin``; with a single-symbol group, ``sin(sin(X_0))``
    nests ``sin`` in itself. Siblings are fine: ``sin(X_0) + cos(X_0)`` passes. To forbid only
    self-nesting of several symbols (``sin(sin(x))`` and ``cos(cos(x))``, but allow ``sin(cos(x))``),
    create one ``NoNesting`` per symbol.

    Args:
        symbols: A single token, or an iterable of tokens, forming the group. Not checked against the
            library.

    Raises:
        ValueError: If the group is empty.

    Attributes:
        symbols: The group, as a frozenset.

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> trig = NoNesting(["sin", "cos"])
        >>> trig.setup(SymbolLibrary.default_symbols(1))
        >>> trig.pre_check(["sin", "(", "cos", "(", "X_0", ")", ")"]).message
        "'cos' nested inside 'sin'"
        >>> trig.pre_check(["sin", "(", "X_0", ")", "+", "cos", "(", "X_0", ")"]).passed
        True
    """

    def __init__(self, symbols: Union[str, Sequence[str]]) -> None:
        self.symbols: frozenset = frozenset({symbols}) if isinstance(symbols, str) else frozenset(symbols)
        if not self.symbols:
            raise ValueError("NoNesting: the group of symbols is empty.")

    def pre_check(self, expr: Union[List[str], Node]) -> CheckResult:
        """Fail if a group symbol occurs below a group symbol in the expression tree.

        Args:
            expr: A well-formed expression, as a token list or a tree.

        Returns:
            Passed if no group symbol has a group symbol below it. Otherwise failed with the message
            ``"<inner> nested inside <outer>"`` (both in ``repr`` form) and ``amount`` = the number
            of group-symbol nodes that have a group symbol above them, as a ``float``
            (``sin(sin(sin(X_0)))`` gives ``2.0``). With several nested symbols, the message names
            the first in pre-order (node, then its left operand or function argument, then its right
            operand, operands in the order they are written) as ``inner``, and the nearest group
            symbol above it as ``outer``.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
            ExpressionParseError: If ``expr`` is a malformed token list.
        """
        symbol_library = _require_setup(self)
        nested: List[Tuple[str, str]] = []
        self._collect_nested(_as_tree(expr, symbol_library), None, nested)
        if not nested:
            return CheckResult()
        inner, outer = nested[0]
        return CheckResult(passed=False, message=f"{inner!r} nested inside {outer!r}", amount=float(len(nested)))

    def _collect_nested(self, node: Optional[Node], outer: Optional[str], nested: List[Tuple[str, str]]) -> None:
        # Pre-order walk (node, left, right); ``outer`` is the nearest group symbol above ``node``.
        # Appends (inner, outer) for every group symbol that has a group symbol above it.
        if node is None:
            return
        if node.symbol in self.symbols:
            if outer is not None:
                nested.append((node.symbol, outer))
            outer = node.symbol
        self._collect_nested(node.left, outer, nested)
        self._collect_nested(node.right, outer, nested)

    def describe(self) -> str:
        """Return ``"no <group> inside another <group>"``, the group's symbols sorted and joined with ``or``.

        Returns:
            The phrase.
        """
        group = " or ".join(sorted(self.symbols))
        return f"no {group} inside another {group}"

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "symbols": [...]}``, the symbols sorted.

        Returns:
            The saved form of this constraint.
        """
        return {**super().to_dict(), "symbols": sorted(self.symbols)}

    @classmethod
    def from_dict(cls, d: dict) -> NoNesting:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the key ``"symbols"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If ``d`` has no ``"symbols"``.
            ValueError: If the group is empty.
        """
        return cls(d["symbols"])


class OutputBounds(ExpressionConstraint):
    """Require the fitted model's output to stay within bounds at given points.

    A post-check: the model is evaluated with its fitted parameters at ``points``, and every output
    must lie in ``[lower, upper]`` (bounds included). A bound is either one value for all points or
    an array with one value per point, e.g. to keep the output between two known curves. A non-finite output (``nan``, ``±inf``) counts
    as outside. The points are the constraint's own data, e.g. a grid over the input domain; they
    are not the training data, and the targets are never read.

    If the expression has free constants and the model has no fitted parameters (e.g. under BED),
    the check fails. Whether such a constraint suits a dataset is up to its author. Evaluating the
    model emits no NumPy warnings.

    Args:
        points: Inputs of shape ``(n_points, n_variables)``, with columns in the order of ``X``;
            anything ``np.asarray`` turns into such an array.
        lower: Smallest allowed output: a number, a 1-D array with one entry per point, or ``None``
            for no lower bound. ``-inf`` (or an entry ``-inf``) means no bound (at that point).
        upper: Largest allowed output, in the same forms; ``inf`` means no bound.

    Raises:
        ValueError: If ``points`` is not a 2-D array with at least one row or contains a NaN or
            infinite value; if a bound is NaN or contains NaN, or is an array whose shape is not
            ``(n_points,)``; if neither bound limits anything at any point (both ``None`` or
            infinite everywhere); or if ``lower > upper`` at some point.

    Attributes:
        points: The evaluation points, as a 2-D float array.
        lower: The lower bound as a ``float``, a 1-D float array, or ``None`` (a bound that is
            ``-inf`` everywhere is stored as ``None``).
        upper: The upper bound in the same forms (``inf`` everywhere is stored as ``None``).

    Examples:
        >>> from SRToolkit.utils import SymbolLibrary
        >>> c = OutputBounds(np.array([[-1.0], [0.0], [2.0]]), lower=0.0)
        >>> c.setup(SymbolLibrary.default_symbols(1))
        >>> result = c.post_check(ModelResult(expr=["C", "*", "X_0"], error=0.1, parameters=np.array([2.0])))
        >>> result.message, result.amount
        ('output outside [0.0, inf] at 1 of 3 points', 2.0)
        >>> c.pre_check(["C", "*", "X_0"]).implemented
        False
    """

    def __init__(
        self,
        points: np.ndarray,
        lower: Optional[Union[float, np.ndarray]] = None,
        upper: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError(
                f"OutputBounds: points must be a 2-D array with at least one row, got shape {points.shape}."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("OutputBounds: points must not contain NaN or infinite values.")
        self.points = points
        self.lower = _bound(lower, -np.inf, "lower", points.shape[0])
        self.upper = _bound(upper, np.inf, "upper", points.shape[0])
        if self.lower is None and self.upper is None:
            raise ValueError("OutputBounds: at least one of lower and upper must limit the output.")
        if self.lower is not None and self.upper is not None:
            n_crossed = int(np.count_nonzero(np.asarray(self.lower) > np.asarray(self.upper)))
            if n_crossed:
                raise ValueError(f"OutputBounds: lower is greater than upper at {n_crossed} point(s).")

    def setup(self, symbol_library: SymbolLibrary) -> None:
        """Store the symbol library and check that ``points`` has one column per variable.

        Args:
            symbol_library: The symbol library.

        Raises:
            ValueError: If the number of columns of ``points`` differs from the number of symbols of
                type ``"var"`` in ``symbol_library``; the message gives both numbers.
        """
        n_variables = len(symbol_library.get_symbols_of_type(VAR))
        if self.points.shape[1] != n_variables:
            raise ValueError(
                f"OutputBounds: points have {self.points.shape[1]} columns, but the symbol library has "
                f"{n_variables} variables."
            )
        super().setup(symbol_library)

    def post_check(self, model: ModelResult) -> CheckResult:
        """Fail if the fitted model's output leaves ``[lower, upper]`` at one of the points.

        Args:
            model: The scored model; its ``expr`` and ``parameters`` are used.

        Returns:
            Passed if every output lies in its bounds; a constant-free expression needs no
            parameters. Otherwise failed with ``amount`` = the largest distance of an output from
            its interval (``inf`` if an output is not finite) and the message
            ``"output outside [<lower>, <upper>] at <k> of <n> points"`` if neither stored bound
            is an array (an array infinite everywhere is stored as ``None``; the bounds formatted with ``str(float(...))``, a missing bound shown as ``-inf``
            / ``inf``), else ``"output outside the bounds at <k> of <n> points"``. If the expression has free constants and ``model.parameters`` is ``None``,
            failed with the message ``"no fitted parameters"``. If ``model.parameters`` is not
            ``None`` and its length differs from the number of free constants (also for a
            constant-free expression, where ``None`` and an empty array pass), failed with
            ``"<n> fitted parameters, expression has <m> constants"``. ``amount = inf`` in both
            cases.

        Raises:
            RuntimeError: If [setup][SRToolkit.utils.expression_constraints.ExpressionConstraint.setup]
                has not been called.
        """
        symbol_library = _require_setup(self)
        output = _predict(model, self.points, symbol_library)
        if isinstance(output, str):
            return CheckResult(passed=False, message=output, amount=float("inf"))
        lower = -np.inf if self.lower is None else self.lower
        upper = np.inf if self.upper is None else self.upper
        finite = np.isfinite(output)
        outside = ~finite | (output < lower) | (output > upper)
        n_outside = int(np.count_nonzero(outside))
        if n_outside == 0:
            return CheckResult()
        if not np.all(finite):
            amount = float("inf")
        else:
            amount = float(np.max(np.maximum(lower - output, output - upper)[outside]))
        where = f"at {n_outside} of {output.shape[0]} points"
        if np.ndim(lower) == 0 and np.ndim(upper) == 0:
            message = f"output outside [{float(lower)}, {float(upper)}] {where}"
        else:
            message = f"output outside the bounds {where}"
        return CheckResult(passed=False, message=message, amount=amount)

    def describe(self) -> str:
        """Return ``"output >= <lower> at <n> given points"``, or ``<= <upper>``, or ``within [<lower>, <upper>]``.

        A bound given per point is written ``per-point bounds``.

        Returns:
            The phrase.
        """
        lower, upper = (str(b) if isinstance(b, float) else "per-point bounds" for b in (self.lower, self.upper))
        where = f"at {len(self.points)} given points"
        if self.upper is None:
            return f"output >= {lower} {where}"
        if self.lower is None:
            return f"output <= {upper} {where}"
        return f"output within [{lower}, {upper}] {where}"

    def summary(self) -> dict:
        """Return ``{"kind": "output_bounds", "lower": ..., "upper": ..., "num_points": n}``.

        A bound is a ``float``, ``None`` (no bound) or ``"per point"``; the points are left out.

        Returns:
            The record.
        """
        lower, upper = (b if b is None or isinstance(b, float) else "per point" for b in (self.lower, self.upper))
        return {"kind": "output_bounds", "lower": lower, "upper": upper, "num_points": len(self.points)}

    def to_dict(self) -> dict:
        """Return ``{"constraint_class": ..., "points": ..., "lower": ..., "upper": ...}``.

        ``points``, and ``lower`` / ``upper`` when they are arrays, are saved as array markers of
        [serialization][SRToolkit.utils.serialization], so they keep their dtype and shape.

        Returns:
            The saved form of this constraint.
        """
        return {
            **super().to_dict(),
            "points": _to_json_safe(self.points, "points"),
            "lower": _to_json_safe(self.lower, "lower"),
            "upper": _to_json_safe(self.upper, "upper"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> OutputBounds:
        """Rebuild the constraint from the output of ``to_dict``; other keys are ignored.

        Args:
            d: A dictionary with the keys ``"points"``, ``"lower"`` and ``"upper"``.

        Returns:
            The rebuilt constraint.

        Raises:
            KeyError: If a key is missing.
            ValueError: If the values are invalid.
        """
        return cls(_from_json_safe(d["points"]), _from_json_safe(d["lower"]), _from_json_safe(d["upper"]))
