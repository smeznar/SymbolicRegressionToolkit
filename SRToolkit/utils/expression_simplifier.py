"""Algebraic simplification of expressions with e-graphs.

[simplify][SRToolkit.utils.expression_simplifier.simplify] puts an expression into an e-graph
([egglog](https://egglog-python.readthedocs.io/)), applies rewrite rules that add equivalent forms
of its parts, and returns the cheapest equivalent expression found.

**Cost.** The cost of an expression is the number of its written-out tokens (operators,
functions, variables, constants and numbers); brackets are not counted. The input is one of the
candidates, so the output is never longer than it. Forms of equal cost are not ordered: ``3*X_0``
may come back as ``X_0*3``.

**Constants.** Simplification keeps the hypothesis: every free constant and label of the input
that is not cancelled stays in the output, e.g. ``C*X_0 + C*X_0`` becomes ``(C+C)*X_0``. Each bare
``C`` is a parameter of its own, while every occurrence of a label is the same parameter, so
``C_1 - C_1`` becomes ``0`` and ``C_1 + C_2 - C_1`` becomes ``C_2``. If a rule copies a bare ``C``,
its copies share a fresh label, so they stay one parameter: the lowest ``C_k`` (with the library's
constant symbol) that is not in the input and not a symbol of the library.

**Fixed values** are numbers, labelled constants with a fixed spec and every ``"lit"`` symbol of
the library, with the value of its ``python_str`` (a custom ``pi`` of ``3.0`` makes ``2*pi`` into
``6``). A fixed value on its own is
written as it was (``pi``, ``C_1``). Fixed values combined by ``+``, ``-``, ``*``, ``/``, ``sqrt``,
``exp``, ``^2`` or ``^3`` become one number (``C + 2 + 3`` becomes ``5+C``), unless the result is
not finite or undefined: ``1/0``, ``0/0``, ``sqrt(-1)`` and ``exp(1000)`` stay as written. A whole
number is written as an integer (``6``, ``-2``), any other at full precision
(``0.3333333333333333``). Combined values are computed in floating point by egglog (``exp(x)`` as
``e**x``), so they can differ from NumPy's in the last digits.

**Rules.** The rules are a [SimplificationRules][SRToolkit.utils.expression_simplifier.SimplificationRules]
object; [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules] is used if none is
given. Rules such as ``X_0/X_0 -> 1`` and ``exp(ln(X_0)) -> X_0`` ignore the points where the left
side is undefined. A rule set gives one *step* of the simplification: the default step saturates
[RELATIONS][SRToolkit.utils.expression_simplifier.RELATIONS], then the simplifying groups
(``"identities"``, ``"inverses"``, ``"fixed_arithmetic"``), then those together with ``"division"``,
then the relations again, and applies each other group once. Custom rules are written over
[EggExpr][SRToolkit.utils.expression_simplifier.EggExpr] and may use the relations
[is_fixed][SRToolkit.utils.expression_simplifier.is_fixed],
[no_free_const][SRToolkit.utils.expression_simplifier.no_free_const] and
[no_const][SRToolkit.utils.expression_simplifier.no_const] after saturating ``RELATIONS``.
"""

# egglog constructors are declared with "..." bodies.
# mypy: disable-error-code="empty-body"
from __future__ import annotations

import math
import numbers
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, cast

import egglog as egg

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.utils.expression_constraints import _validate_limit
from SRToolkit.utils.expression_tree import Node, _as_tree, parameter_labels
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.symbol_library import _DEFAULT_SYMBOLS, SymbolLibrary


class ConstTree(egg.Expr):
    """What a constant [EggExpr.c][SRToolkit.utils.expression_simplifier.EggExpr.c] is made of.

    Every node costs one token. The default rules only use leaves; ``node1`` and ``node2`` are for
    custom rules and are written out as their tokens.
    """

    @egg.method(cost=100)
    @classmethod
    def leaf(cls, id: egg.i64Like, lb: egg.f64Like, rb: egg.f64Like) -> ConstTree:
        """A constant of the input (``id >= 0``) or a value computed by a rule (``id == -1``).

        ``lb`` and ``rb`` are its bounds; ``lb == rb`` means the value is fixed to ``lb``. A constant
        that is not fixed has the ``bounds`` of its spec, or ``(-inf, inf)`` without them; the
        default rules do not use these bounds.
        """
        ...

    @egg.method(cost=100)
    @classmethod
    def node1(cls, name: egg.StringLike, t: ConstTree) -> ConstTree:
        """The ``"fn"`` token ``name`` applied to ``t``."""
        ...

    @egg.method(cost=100)
    @classmethod
    def node2(cls, name: egg.StringLike, t1: ConstTree, t2: ConstTree) -> ConstTree:
        """The ``"op"`` token ``name`` applied to ``t1`` and ``t2``."""
        ...


class EggExpr(egg.Expr):
    """An expression in the e-graph; each node costs one token (``c`` costs what its tree costs)."""

    @egg.method(cost=100)
    @classmethod
    def var(cls, name: egg.StringLike) -> EggExpr:
        """The ``"var"`` token ``name``."""
        ...

    @egg.method(cost=0)
    @classmethod
    def c(cls, t: ConstTree) -> EggExpr:
        """A constant made of ``t``."""
        ...

    @egg.method(cost=100)
    @classmethod
    def op(cls, name: egg.StringLike, a: EggExpr, b: EggExpr) -> EggExpr:
        """The ``"op"`` token ``name`` applied to ``a`` and ``b``."""
        ...

    @egg.method(cost=100)
    @classmethod
    def fn(cls, name: egg.StringLike, a: EggExpr) -> EggExpr:
        """The ``"fn"`` token ``name`` applied to ``a``."""
        ...


is_fixed = egg.relation("is_fixed", ConstTree)
"""Holds for a [ConstTree][SRToolkit.utils.expression_simplifier.ConstTree] whose leaves are all fixed."""
no_free_const = egg.relation("no_free_const", EggExpr)
"""Holds for a [EggExpr][SRToolkit.utils.expression_simplifier.EggExpr] without constants that are not fixed."""
no_const = egg.relation("no_const", EggExpr)
"""Holds for a [EggExpr][SRToolkit.utils.expression_simplifier.EggExpr] without constants."""


class SimplificationRules(ABC):
    """The rewrite rules of [simplify][SRToolkit.utils.expression_simplifier.simplify].

    Rules are written over [EggExpr][SRToolkit.utils.expression_simplifier.EggExpr] and
    [ConstTree][SRToolkit.utils.expression_simplifier.ConstTree] (see the module docs for the relations a
    rule may use). A subclass implements ``build``, ``to_dict`` and ``from_dict``; its class must
    be importable for [from_dict][SRToolkit.utils.expression_simplifier.SimplificationRules.from_dict].
    """

    @abstractmethod
    def build(self, symbol_library: SymbolLibrary) -> egg.Schedule:
        """Return the egglog schedule of one simplification step for expressions over ``symbol_library``.

        [simplify][SRToolkit.utils.expression_simplifier.simplify] runs it repeatedly. It must not
        write tokens that ``symbol_library`` lacks.

        Args:
            symbol_library: The library of the expressions to simplify.

        Returns:
            The schedule of one step.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the rules to a JSON-compatible dictionary.

        Returns:
            The settings of the rules and ``"rules_class"``, the fully-qualified class path.
        """

    def __repr__(self) -> str:
        """Return ``ClassName(key=value, ...)`` from ``to_dict`` without ``"rules_class"``.

        Examples:
            >>> DefaultRules(exclude=["powers"])
            DefaultRules(exclude=['powers'])
        """
        args = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "rules_class")
        return f"{type(self).__name__}({args})"

    @classmethod
    def from_dict(cls, d: dict) -> SimplificationRules:
        """Reconstruct rules from the output of ``to_dict``.

        On the base class, dispatches to the class named by ``"rules_class"``; a subclass must
        override it.

        Args:
            d: A dictionary produced by
                [to_dict][SRToolkit.utils.expression_simplifier.SimplificationRules.to_dict].

        Returns:
            The rules.

        Raises:
            KeyError: If ``"rules_class"`` is missing (dispatch path).
            ImportError: If the class cannot be imported (dispatch path).
            TypeError: If the class is not a subclass of this one (dispatch path).
            NotImplementedError: If called on a subclass that does not override it.
        """
        if cls is SimplificationRules:
            d = _auto_bind(d)
            return _import_class(d["rules_class"], "rules", SimplificationRules).from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")


_t1, _t2 = egg.vars_("t1 t2", ConstTree)
_a, _b, _c = egg.vars_("a b c", EggExpr)
_f1, _f2 = egg.vars_("f1 f2", egg.f64)
_i1, _i2 = egg.vars_("i1 i2", egg.i64)
_s = egg.var("s", egg.String)


def _value(v: egg.f64Like) -> EggExpr:
    return EggExpr.c(ConstTree.leaf(-1, v, v))


_ZERO, _ONE = ConstTree.leaf(_i1, 0.0, 0.0), ConstTree.leaf(_i1, 1.0, 1.0)
_FIXED1, _FIXED2 = ConstTree.leaf(_i1, _f1, _f1), ConstTree.leaf(_i2, _f2, _f2)

RELATIONS = egg.ruleset(
    egg.rule(_FIXED1).then(is_fixed(_FIXED1)),
    egg.rule(ConstTree.node1(_s, _t1), is_fixed(_t1)).then(is_fixed(ConstTree.node1(_s, _t1))),
    egg.rule(ConstTree.node2(_s, _t1, _t2), is_fixed(_t1), is_fixed(_t2)).then(is_fixed(ConstTree.node2(_s, _t1, _t2))),
    egg.rule(EggExpr.c(_t1), is_fixed(_t1)).then(no_free_const(EggExpr.c(_t1))),
    egg.rule(EggExpr.var(_s)).then(no_free_const(EggExpr.var(_s)), no_const(EggExpr.var(_s))),
    egg.rule(EggExpr.op(_s, _a, _b), no_free_const(_a), no_free_const(_b)).then(no_free_const(EggExpr.op(_s, _a, _b))),
    egg.rule(EggExpr.fn(_s, _a), no_free_const(_a)).then(no_free_const(EggExpr.fn(_s, _a))),
    egg.rule(EggExpr.op(_s, _a, _b), no_const(_a), no_const(_b)).then(no_const(EggExpr.op(_s, _a, _b))),
    egg.rule(EggExpr.fn(_s, _a), no_const(_a)).then(no_const(EggExpr.fn(_s, _a))),
    name="relations",
)
"""The egglog ruleset computing [is_fixed][SRToolkit.utils.expression_simplifier.is_fixed],
[no_free_const][SRToolkit.utils.expression_simplifier.no_free_const] and
[no_const][SRToolkit.utils.expression_simplifier.no_const]; saturate it before rules that use them."""


def _op(name: str, x: EggExpr, y: EggExpr) -> EggExpr:
    return EggExpr.op(name, x, y)


def _fn(name: str, x: EggExpr) -> EggExpr:
    return EggExpr.fn(name, x)


def _fixed_op(name: str, value: egg.f64, *conditions: Any) -> tuple[tuple[str, ...], Any]:
    # name applied to two fixed values becomes their combined value, if it is finite.
    rewritten = egg.rewrite(_op(name, EggExpr.c(_FIXED1), EggExpr.c(_FIXED2)), subsume=True)
    return (name,), rewritten.to(_value(value), value < math.inf, value > -math.inf, *conditions)


def _fixed_fn(name: str, value: egg.f64, *conditions: Any) -> tuple[tuple[str, ...], Any]:
    # name applied to a fixed value becomes the result, if it is finite.
    rewritten = egg.rewrite(_fn(name, EggExpr.c(_FIXED1)), subsume=True)
    return (name,), rewritten.to(_value(value), value < math.inf, value > -math.inf, *conditions)


# Rule groups of DefaultRules: name -> (phase, ((tokens the rule relies on, rule), ...)).
# Phases: "simplify" is saturated in each step, then "simplify" and "divide" together (so the
# zeros "simplify" finds are known before a/a or 0/a can fire), and "explore" runs once per step.
_GROUPS: dict[str, tuple[str, tuple[tuple[tuple[str, ...], Any], ...]]] = {
    "identities": (
        "simplify",
        (
            (("-",), egg.rewrite(_op("-", _a, _a), subsume=True).to(_value(0.0))),
            (("+",), egg.rewrite(_op("+", _a, EggExpr.c(_ZERO)), subsume=True).to(_a)),
            (("+",), egg.rewrite(_op("+", EggExpr.c(_ZERO), _a), subsume=True).to(_a)),
            (("-",), egg.rewrite(_op("-", _a, EggExpr.c(_ZERO)), subsume=True).to(_a)),
            (("*",), egg.rewrite(_op("*", _a, EggExpr.c(_ZERO)), subsume=True).to(_value(0.0))),
            (("*",), egg.rewrite(_op("*", EggExpr.c(_ZERO), _a), subsume=True).to(_value(0.0))),
            (("*",), egg.rewrite(_op("*", _a, EggExpr.c(_ONE)), subsume=True).to(_a)),
            (("*",), egg.rewrite(_op("*", EggExpr.c(_ONE), _a), subsume=True).to(_a)),
            (("/",), egg.rewrite(_op("/", _a, EggExpr.c(_ONE)), subsume=True).to(_a)),
            (("sin",), egg.rewrite(_fn("sin", EggExpr.c(_ZERO)), subsume=True).to(_value(0.0))),
            (("cos",), egg.rewrite(_fn("cos", EggExpr.c(_ZERO)), subsume=True).to(_value(1.0))),
            (("exp",), egg.rewrite(_fn("exp", EggExpr.c(_ZERO)), subsume=True).to(_value(1.0))),
            (("sqrt",), egg.rewrite(_fn("sqrt", EggExpr.c(_ZERO)), subsume=True).to(_value(0.0))),
            (("^2",), egg.rewrite(_fn("^2", EggExpr.c(_ZERO)), subsume=True).to(_value(0.0))),
            (("^3",), egg.rewrite(_fn("^3", EggExpr.c(_ZERO)), subsume=True).to(_value(0.0))),
            (("ln",), egg.rewrite(_fn("ln", EggExpr.c(_ONE)), subsume=True).to(_value(0.0))),
            (("sqrt",), egg.rewrite(_fn("sqrt", EggExpr.c(_ONE)), subsume=True).to(_value(1.0))),
            (("^2",), egg.rewrite(_fn("^2", EggExpr.c(_ONE)), subsume=True).to(_value(1.0))),
            (("^3",), egg.rewrite(_fn("^3", EggExpr.c(_ONE)), subsume=True).to(_value(1.0))),
            # Every fixed zero is the computed zero, so the guards of the division group can tell zeros apart.
            ((), egg.rewrite(_ZERO).to(ConstTree.leaf(-1, 0.0, 0.0))),
        ),
    ),
    "division": (
        "divide",
        (
            (("/",), egg.rewrite(_op("/", _a, _a), subsume=True).to(_value(1.0), egg.ne(_a).to(_value(0.0)))),
            (
                ("/",),
                egg.rewrite(_op("/", EggExpr.c(_ZERO), _a), subsume=True).to(_value(0.0), egg.ne(_a).to(_value(0.0))),
            ),
        ),
    ),
    "inverses": (
        "simplify",
        (
            (("^2", "sqrt"), egg.rewrite(_fn("^2", _fn("sqrt", _a))).to(_a)),
            (("ln", "exp"), egg.rewrite(_fn("ln", _fn("exp", _a))).to(_a)),
            (("ln", "exp"), egg.rewrite(_fn("exp", _fn("ln", _a))).to(_a)),
        ),
    ),
    "fixed_arithmetic": (
        "simplify",
        (
            _fixed_op("+", _f1 + _f2),
            _fixed_op("-", _f1 - _f2),
            _fixed_op("*", _f1 * _f2),
            _fixed_op("/", _f1 / _f2, egg.ne(_f2).to(egg.f64(0.0))),
            _fixed_fn("sqrt", _f1 ** egg.f64(0.5), _f1 >= 0.0),
            _fixed_fn("exp", egg.f64(math.e) ** _f1),
            _fixed_fn("^2", _f1 * _f1),
            _fixed_fn("^3", _f1 * _f1 * _f1),
        ),
    ),
    "like_terms": (
        "explore",
        (
            (("+", "*"), egg.rewrite(_op("+", _a, _a)).to(_op("*", _value(2.0), _a), no_const(_a))),
            (
                ("+", "*"),
                egg.rewrite(_op("+", _op("*", EggExpr.c(_t1), _a), _a)).to(
                    _op("*", _op("+", EggExpr.c(_t1), _value(1.0)), _a), no_const(_a)
                ),
            ),
            (
                ("-", "*"),
                egg.rewrite(_op("-", _op("*", EggExpr.c(_t1), _a), _a)).to(
                    _op("*", _op("-", EggExpr.c(_t1), _value(1.0)), _a), no_const(_a)
                ),
            ),
            (
                ("-", "*"),
                egg.rewrite(_op("-", _a, _op("*", EggExpr.c(_t1), _a))).to(
                    _op("*", _op("-", _value(1.0), EggExpr.c(_t1)), _a), no_const(_a)
                ),
            ),
        ),
    ),
    "commutativity": (
        "explore",
        (
            (("+",), egg.rewrite(_op("+", _a, _b)).to(_op("+", _b, _a))),
            (("*",), egg.rewrite(_op("*", _a, _b)).to(_op("*", _b, _a))),
        ),
    ),
    "associativity": (
        "explore",
        (
            (("+",), egg.rewrite(_op("+", _op("+", _a, _b), _c)).to(_op("+", _op("+", _a, _c), _b))),
            (("*",), egg.rewrite(_op("*", _op("*", _a, _b), _c)).to(_op("*", _op("*", _a, _c), _b))),
            (("+", "-"), egg.birewrite(_op("+", _a, _op("-", _b, _c))).to(_op("-", _op("+", _a, _b), _c))),
            (("+", "-"), egg.birewrite(_op("-", _a, _op("+", _b, _c))).to(_op("-", _op("-", _a, _b), _c))),
            (("+", "-"), egg.birewrite(_op("-", _a, _op("-", _b, _c))).to(_op("+", _op("-", _a, _b), _c))),
            (("*", "/"), egg.birewrite(_op("*", _a, _op("/", _b, _c))).to(_op("/", _op("*", _a, _b), _c))),
            (("*", "/"), egg.birewrite(_op("/", _a, _op("*", _b, _c))).to(_op("/", _op("/", _a, _b), _c))),
            (("*", "/"), egg.birewrite(_op("/", _a, _op("/", _b, _c))).to(_op("*", _op("/", _a, _b), _c))),
        ),
    ),
    "distributivity": (
        "explore",
        (
            (
                ("*", "+"),
                egg.birewrite(_op("*", _op("+", _a, _b), _c)).to(_op("+", _op("*", _a, _c), _op("*", _b, _c))),
            ),
            (
                ("*", "-"),
                egg.birewrite(_op("*", _op("-", _a, _b), _c)).to(_op("-", _op("*", _a, _c), _op("*", _b, _c))),
            ),
            (
                ("/", "+"),
                egg.birewrite(_op("/", _op("+", _a, _b), _c)).to(_op("+", _op("/", _a, _c), _op("/", _b, _c))),
            ),
            (
                ("/", "-"),
                egg.birewrite(_op("/", _op("-", _a, _b), _c)).to(_op("-", _op("/", _a, _c), _op("/", _b, _c))),
            ),
            (
                ("+", "/", "*"),
                egg.rewrite(_op("+", _op("/", _a, _b), _op("/", _a, _c))).to(
                    _op("*", _a, _op("+", _op("/", _value(1.0), _b), _op("/", _value(1.0), _c))), no_const(_a)
                ),
            ),
            (
                ("-", "/", "*"),
                egg.rewrite(_op("-", _op("/", _a, _b), _op("/", _a, _c))).to(
                    _op("*", _a, _op("-", _op("/", _value(1.0), _b), _op("/", _value(1.0), _c))), no_const(_a)
                ),
            ),
        ),
    ),
    "powers": (
        "explore",
        (
            (("*", "^2"), egg.birewrite(_op("*", _fn("^2", _a), _fn("^2", _b))).to(_fn("^2", _op("*", _a, _b)))),
            (("/", "^2"), egg.birewrite(_op("/", _fn("^2", _a), _fn("^2", _b))).to(_fn("^2", _op("/", _a, _b)))),
            (("*", "^3"), egg.birewrite(_op("*", _fn("^3", _a), _fn("^3", _b))).to(_fn("^3", _op("*", _a, _b)))),
            (("/", "^3"), egg.birewrite(_op("/", _fn("^3", _a), _fn("^3", _b))).to(_fn("^3", _op("/", _a, _b)))),
            (("*", "^2"), egg.birewrite(_op("*", _a, _a)).to(_fn("^2", _a))),
            (("*", "^2", "^3"), egg.birewrite(_op("*", _fn("^2", _a), _a)).to(_fn("^3", _a))),
        ),
    ),
    "exp_ln_sqrt": (
        "explore",
        (
            (
                ("*", "exp", "+"),
                egg.birewrite(_op("*", _fn("exp", _a), _fn("exp", _b))).to(_fn("exp", _op("+", _a, _b))),
            ),
            (
                ("/", "exp", "-"),
                egg.birewrite(_op("/", _fn("exp", _a), _fn("exp", _b))).to(_fn("exp", _op("-", _a, _b))),
            ),
            (
                ("*", "sqrt"),
                egg.birewrite(_op("*", _fn("sqrt", _a), _fn("sqrt", _b))).to(_fn("sqrt", _op("*", _a, _b))),
            ),
            (
                ("/", "sqrt"),
                egg.birewrite(_op("/", _fn("sqrt", _a), _fn("sqrt", _b))).to(_fn("sqrt", _op("/", _a, _b))),
            ),
            (("ln", "*", "+"), egg.birewrite(_fn("ln", _op("*", _a, _b))).to(_op("+", _fn("ln", _a), _fn("ln", _b)))),
            (("ln", "/", "-"), egg.birewrite(_fn("ln", _op("/", _a, _b))).to(_op("-", _fn("ln", _a), _fn("ln", _b)))),
        ),
    ),
}

_DEFAULTS = {symbol.token: symbol for symbol in _DEFAULT_SYMBOLS}


class DefaultRules(SimplificationRules):
    """The default rules, in named groups.

    - ``"identities"``: ``a - a -> 0``, ``a + 0``, ``0 + a``, ``a - 0``, ``a*1``, ``1*a``,
      ``a/1 -> a``, ``a*0``, ``0*a -> 0``, and ``sin 0 = 0``, ``cos 0 = 1``, ``exp 0 = 1``, ``sqrt 0 = 0``, ``0^2 = 0^3 = 0``, ``ln 1 = 0``,
      ``sqrt 1 = 1^2 = 1^3 = 1``.
    - ``"division"``: ``a/a -> 1`` and ``0/a -> 0``, for ``a`` not found to be ``0`` (so ``0/0``
      stays). Applied after the other simplifying groups, so that the zeros they find are known.
    - ``"inverses"``: ``sqrt(a)^2``, ``ln(exp(a))`` and ``exp(ln(a))`` to ``a``.
    - ``"fixed_arithmetic"``: fixed values combined by ``+``, ``-``, ``*``, ``/``, ``sqrt``,
      ``exp``, ``^2``, ``^3`` into one number, if the result is finite (no division by ``0``, no
      ``sqrt`` of a negative).
    - ``"like_terms"``: ``a + a -> 2*a``, ``c*a + a -> (c + 1)*a``, ``c*a - a -> (c - 1)*a``,
      ``a - c*a -> (1 - c)*a``, for ``a`` without constants.
    - ``"commutativity"``, ``"associativity"`` (with mixed ``+``/``-`` and ``*``/``/``),
      ``"distributivity"``, ``"powers"`` (``^2`` and ``^3`` of products and quotients,
      ``a*a = a^2``, ``a^2*a = a^3``), ``"exp_ln_sqrt"`` (``exp``, ``sqrt`` of sums and products,
      ``ln`` of products and quotients).

    A rule is used only if every operator and function it relies on is in the symbol library with
    the type and ``python_str`` of the predefined symbol (so a custom ``ln`` with another meaning
    turns its rules off). Tokens without rules are left as they are; their arguments are still
    simplified.

    Args:
        exclude: Names of groups to leave out.

    Raises:
        ValueError: If ``exclude`` names an unknown group.

    Examples:
        >>> DefaultRules(exclude=["exp_ln_sqrt"]).to_dict()
        {'rules_class': 'SRToolkit.utils.expression_simplifier.DefaultRules', 'exclude': ['exp_ln_sqrt']}
    """

    def __init__(self, exclude: Sequence[str] = ()) -> None:
        unknown = sorted(set(exclude) - set(_GROUPS))
        if unknown:
            raise ValueError(f"Unknown rule groups {unknown}; the groups are {list(_GROUPS)}.")
        self.exclude = tuple(exclude)

    def build(self, symbol_library: SymbolLibrary) -> egg.Schedule:
        """Return the schedule of one step for ``symbol_library`` (see the module docs).

        Args:
            symbol_library: Decides which rules are used.

        Returns:
            The schedule of one step.
        """
        # The predefined tokens (+, sin, ln, ...) that this library has with their predefined
        # meaning: same type and same python_str. A custom "ln" that computes log10, or a library
        # without "^2", leaves that token out, and with it every rule that relies on it.
        usable = frozenset(
            token
            for token, default in _DEFAULTS.items()
            if (symbol := symbol_library.symbols.get(token)) is not None
            and (symbol.type, symbol.python_str) == (default.type, default.python_str)
        )
        # Sort the rules of the groups that are not excluded into the two phases of a step. Each
        # rule lists the tokens it relies on (it matches or writes them); keep it only if all of
        # them are usable, so a rule never writes a token the library lacks.
        phases: dict[str, list] = {"simplify": [], "divide": [], "explore": []}
        for name, (phase, rules) in _GROUPS.items():
            if name not in self.exclude:
                phases[phase] += [r for tokens, r in rules if usable.issuperset(tokens)]
        # One step of the search (simplify runs it repeatedly):
        return (
            # 1. Compute the relations (is_fixed, no_free_const, no_const) that guard some rules.
            RELATIONS.saturate()
            # 2. Apply the simplifying rules (identities, inverses, fixed arithmetic) until nothing
            #    changes; they only shrink the expression, so this ends quickly.
            + egg.ruleset(*phases["simplify"]).saturate()
            # 3. Add the division identities (a/a -> 1, 0/a -> 0, for a not 0) and saturate again.
            #    They come after step 2 so that every zero step 2 can find is known: "is not zero"
            #    can only be checked against zeros already found.
            + egg.ruleset(*phases["simplify"], *phases["divide"]).saturate()
            # 4. Recompute the relations for the forms steps 2 and 3 added.
            + RELATIONS.saturate()
            # 5. Apply the exploring rules (commutativity, associativity, distributivity, ...)
            #    once. They add equivalent forms, often larger ones, so saturating them would let
            #    the e-graph grow without end; simplify's limits bound the steps instead.
            + egg.ruleset(*phases["explore"])
        )

    def to_dict(self) -> dict:
        """Serialize the rules to a JSON-compatible dictionary.

        Returns:
            ``{"rules_class": ..., "exclude": [...]}``.
        """
        return {"rules_class": "SRToolkit.utils.expression_simplifier.DefaultRules", "exclude": list(self.exclude)}

    @classmethod
    def from_dict(cls, d: dict) -> DefaultRules:
        """Reconstruct the rules from the output of ``to_dict``.

        Args:
            d: A dictionary produced by
                [to_dict][SRToolkit.utils.expression_simplifier.DefaultRules.to_dict].

        Returns:
            The rules.

        Raises:
            KeyError: If ``d`` has no ``"exclude"``.
            ValueError: If ``"exclude"`` names an unknown group.
        """
        return cls(exclude=d["exclude"])


def _check_settings(rules: Any, max_iterations: Any, max_nodes: Any, max_time: Any, prefix: str = "") -> None:
    # The checks of the simplification settings of SR_evaluator (prefix "simplify_") and the
    # ExpressionSimplifier augmenter: TypeError for rules, ValueError for the limits. None means
    # simplify's default and is not checked.
    if rules is not None and not isinstance(rules, SimplificationRules):
        raise TypeError(f"{prefix}rules must be a SimplificationRules or None, got {rules!r}.")
    if max_iterations is not None:
        _validate_limit(max_iterations, f"{prefix}max_iterations")
    if max_nodes is not None:
        _validate_limit(max_nodes, f"{prefix}max_nodes")
    if max_time is not None and (
        isinstance(max_time, bool) or not isinstance(max_time, numbers.Real) or not float(max_time) > 0
    ):
        raise ValueError(f"{prefix}max_time must be a positive number or None, got {max_time!r}.")


def simplify(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
    free_constants: Sequence[Constant] | None = None,
    rules: SimplificationRules | None = None,
    max_iterations: int = 30,
    max_nodes: int = 10_000,
    max_time: float | None = None,
) -> tuple[list[str] | Node, tuple[Constant, ...] | None]:
    """Simplify an expression algebraically.

    The rules are applied in steps (see the module docs for the cost, constants and fixed values).
    After a step that finds a cheaper expression, the search continues from a new e-graph holding
    only that expression. It stops when a step changes nothing and finds nothing cheaper, after
    ``max_iterations`` steps, once the e-graph has grown to more than ``max_nodes`` entries without
    finding anything cheaper, or once ``max_time`` has passed; the cheapest expression found is
    returned. The limits are checked between steps, so a step may overshoot them, and they
    are not validated. ``max_iterations`` and ``max_nodes`` count work, so the result does not
    depend on the machine; with ``max_time`` it can, so leave it ``None`` for reproducible results.
    Simplifying the result again may simplify it further.

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library of the expression. ``None`` uses the active or default library.
        free_constants: Specs of labelled constants of ``expr``, in any order. A fixed one is used
            as its value; the bounds of the others are recorded but not used by the default rules.
        rules: The rewrite rules; ``None`` uses [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules].
        max_iterations: Maximum number of steps; ``0`` returns ``expr`` unchanged (as a tree; a
            token list may gain redundant brackets).
        max_nodes: The search stops once the e-graph has grown to more than ``max_nodes``
            entries since the last cheaper expression was found.
        max_time: Seconds after which the search stops; ``None`` for no time limit.

    Returns:
        The simplified expression, in the form of ``expr`` (token list or
        [Node][SRToolkit.utils.expression_tree.Node]), and the specs of ``free_constants`` whose
        label it still contains, in the given order (``()`` if none, ``None`` if
        ``free_constants`` is ``None``).

    Raises:
        TypeError: If ``expr`` is neither a token list nor a [Node][SRToolkit.utils.expression_tree.Node].
        ExpressionParseError: If a token list cannot be parsed.
        ValueError: If a spec in ``free_constants`` has no label, two share one, or its label is not
            a labelled constant of ``expr``.

    Examples:
        >>> print("".join(simplify(["C", "*", "X_0", "+", "X_1", "*", "X_0", "/", "X_1", "*", "C"])[0]))
        (C+C)*X_0
        >>> from SRToolkit.utils import Constant
        >>> fixed = Constant(label="C_1", value=2.0, fixed=True)
        >>> expr, specs = simplify(["C_1", "*", "X_0", "+", "C_1", "*", "3"], free_constants=[fixed])
        >>> "".join(expr), specs == (fixed,)
        ('C_1*X_0+6', True)
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    if rules is None:
        rules = DefaultRules()
    tree = _as_tree(expr, symbol_library)

    specs = {spec.label: spec for spec in free_constants or ()}
    labels = set(parameter_labels(tree, symbol_library))
    if None in specs or len(specs) != len(free_constants or ()) or not labels.issuperset(specs):
        raise ValueError(
            f"Every spec in 'free_constants' needs a distinct label of a labelled constant of the expression, "
            f"got {[spec.label for spec in free_constants or ()]}."
        )

    tokens: list[str] = []  # leaf id -> token
    labelled: dict[str, EggExpr] = {}  # label -> its constant

    def to_num(node: Node) -> EggExpr:
        if node.left is not None and node.right is not None:
            return EggExpr.op(node.symbol, to_num(node.left), to_num(node.right))
        if node.left is not None:
            return EggExpr.fn(node.symbol, to_num(node.left))
        symbol = symbol_library.get(node.symbol)
        if symbol is not None and symbol.type == "var":
            return EggExpr.var(node.symbol)
        if node.symbol in labelled:
            return labelled[node.symbol]
        tokens.append(node.symbol)
        if symbol is not None and symbol.type == "lit":
            value = float(eval(symbol.python_str, symbol_library.module_namespace()))
            bounds = (value, value)
        elif symbol is None or symbol.type != "const":  # a number
            bounds = (float(node.symbol), float(node.symbol))
        elif (spec := specs.get(node.symbol)) is not None and spec.fixed:
            assert spec.value is not None
            bounds = (spec.value, spec.value)
        else:
            bounds = spec.bounds if spec is not None and spec.bounds is not None else (-math.inf, math.inf)
        term = EggExpr.c(ConstTree.leaf(len(tokens) - 1, *bounds))
        if symbol is not None and symbol.type == "const" and node.symbol != symbol_library.constant_symbol:
            labelled[node.symbol] = term
        return term

    best = to_num(tree)
    start = time.monotonic()
    schedule = rules.build(symbol_library)

    def new_egraph(term: Any) -> tuple[egg.EGraph, Any]:
        # An e-graph holding term and the computed zero, which the "is not zero" guards of the
        # default rules (a/a, 0/a) compare against; a guard cannot compare with a missing term.
        egraph = egg.EGraph()
        egraph.register(_value(0.0))
        return egraph, egraph.let("expr", term)

    egraph, handle = new_egraph(best)
    best_cost = math.inf
    for _ in range(max_iterations):
        report = egraph.run(schedule)
        candidate, cost = egraph.extract(handle, include_cost=True)
        if cost < best_cost:
            best, best_cost = candidate, cost
            egraph, handle = new_egraph(best)
        elif not report.updated or sum(size for _, size in egraph.all_function_sizes()) > max_nodes:
            break
        if max_time is not None and time.monotonic() - start > max_time:
            break

    copies: dict[int, list[Node]] = {}  # leaf id of a bare constant -> its nodes in the result

    def to_node(term: Any) -> Node:
        fn, args = egg.get_callable_fn(term), cast(Any, egg.get_callable_args(term))
        if fn == EggExpr.var:
            return Node(args[0].value)
        if fn == EggExpr.op or fn == ConstTree.node2:
            return Node(args[0].value, left=to_node(args[1]), right=to_node(args[2]))
        if fn == EggExpr.fn or fn == ConstTree.node1:
            return Node(args[0].value, left=to_node(args[1]))
        if fn == EggExpr.c:
            return to_node(args[0])
        leaf_id = args[0].value
        if leaf_id >= 0:
            node = Node(tokens[leaf_id])
            if tokens[leaf_id] == symbol_library.constant_symbol:
                copies.setdefault(leaf_id, []).append(node)
            return node
        value = args[1].value
        return Node(str(int(value)) if value.is_integer() else repr(value))

    result = to_node(best)
    # A bare constant that a rule copied is one parameter, so its copies share a new label: the
    # lowest C_k not used in the input and not a symbol of the library.
    k = 0
    for nodes in copies.values():
        if len(nodes) > 1:
            while (label := f"{symbol_library.constant_symbol}_{k}") in labels or label in symbol_library.symbols:
                k += 1
            k += 1
            for node in nodes:
                node.symbol = label
    kept = None
    if free_constants is not None:
        remaining = set(parameter_labels(result, symbol_library))
        kept = tuple(spec for spec in free_constants if spec.label in remaining)
    if isinstance(expr, Node):
        return result, kept
    return result.to_list(symbol_library=symbol_library), kept
