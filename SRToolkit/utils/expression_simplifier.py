"""Algebraic simplification and constant folding of expressions with e-graphs.

The e-graph simplifier was created by Martin Perčinić ([mpercinic](https://github.com/mpercinic))
and adapted for SRToolkit by Sebastian Mežnar.

[simplify][SRToolkit.utils.expression_simplifier.simplify] puts an expression into an e-graph
([egglog](https://egglog-python.readthedocs.io/)), applies rewrite rules that add equivalent forms
of its parts, and returns the cheapest equivalent expression found.

**Cost.** The cost of an expression is the number of its written-out tokens (operators,
functions, variables, constants and numbers); brackets are not counted. The input is one of the
candidates, so the output is never longer than it. Forms of equal cost are not ordered: ``3*X_0``
may come back as ``X_0*3``. With folding and ``fold_aware=True``, the e-graph is searched once more
with a cost that counts each part folding would merge as one token (ties: the written-out token
count). That cost cannot see everything that keeps a part from folding (e.g. a label used
elsewhere), so its result is used only if it is shorter than the token-count result once both are
folded.

**Constants.** Simplification keeps the hypothesis: every free constant and label of the input
that is not cancelled stays in the output, e.g. ``C*X_0 + C*X_0`` becomes ``(C+C)*X_0``. Each bare
``C`` is a parameter of its own, while every occurrence of a label is the same parameter, so
``C_1 - C_1`` becomes ``0`` and ``C_1 + C_2 - C_1`` becomes ``C_2``. If a rule copies a bare ``C``,
its copies share a fresh label, so they stay one parameter: the lowest ``C_k`` (with the library's
constant symbol) that is not in the input and not a symbol of the library.

**Folding.** [fold][SRToolkit.utils.expression_simplifier.fold] replaces each *constant part* of
an expression, a subexpression without variables that contains a free constant, with one new bare
``C``: ``C*C*X_0`` and ``sin(C + 2)*X_0`` both become ``C*X_0``. The expression gets shorter and has
fewer parameters, but it is a different hypothesis. The new ``C`` gets the default spec, so it ranges
over different values than the part did: with bounds ``(-5, 5)``, ``2*C`` reaches ``[-10, 10]`` and
``sqrt(C)`` is never negative, while ``C`` stays in ``[-5, 5]``. Fitting and BED can therefore give a
different score.

A part that contains a label also used elsewhere is not folded, since that would split one parameter
into two (``sin(C_1)*X_0 + cos(C_1)*X_1`` stays). The ``mode`` decides which constants may fold at
all.

Folding works best after simplification, which first gathers constants: ``C*X_0 + C*X_0`` folds only
once [simplify][SRToolkit.utils.expression_simplifier.simplify] has made it ``(C+C)*X_0``, so use
``simplify(fold=...)`` rather than ``fold`` alone.

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
from typing import Any, Literal, cast

import egglog as egg

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.utils.expression import Expression, _as_expression
from SRToolkit.utils.expression_constraints import _validate_limit
from SRToolkit.utils.expression_tree import Node, _parameter_labels
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
            # One way only: the left side is defined only for non-negative (or positive) arguments,
            # where both sides are equal; the other way would rewrite sqrt(a*b) or ln(a*b) of two
            # negative values (e.g. sqrt(X_0*X_0) -> sqrt(X_0)^2 -> X_0).
            # TODO: look at these four rules again later (one-way since the sqrt(X_0*X_0) -> X_0 fix;
            # check whether they still find the simplifications they were meant for, and whether other
            # rules have the same domain problem).
            (
                ("*", "sqrt"),
                egg.rewrite(_op("*", _fn("sqrt", _a), _fn("sqrt", _b))).to(_fn("sqrt", _op("*", _a, _b))),
            ),
            (
                ("/", "sqrt"),
                egg.rewrite(_op("/", _fn("sqrt", _a), _fn("sqrt", _b))).to(_fn("sqrt", _op("/", _a, _b))),
            ),
            (("ln", "*", "+"), egg.rewrite(_op("+", _fn("ln", _a), _fn("ln", _b))).to(_fn("ln", _op("*", _a, _b)))),
            (("ln", "/", "-"), egg.rewrite(_op("-", _fn("ln", _a), _fn("ln", _b))).to(_fn("ln", _op("/", _a, _b)))),
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
      ``a*a = a^2``, ``a^2*a = a^3``), ``"exp_ln_sqrt"`` (``exp`` of sums and differences, and
      ``sqrt(a)*sqrt(b) -> sqrt(a*b)``, ``ln(a) + ln(b) -> ln(a*b)`` and their ``/`` and ``-``
      forms in that direction only, where they hold wherever the left side is defined).

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


_FOLD_MODES = ("none", "default", "keep")  # the modes of fold; see its docstring


def _check_fold_mode(mode: Any, name: str) -> None:
    if not isinstance(mode, str) or mode not in _FOLD_MODES:
        raise ValueError(f"{name} must be one of {list(_FOLD_MODES)}, got {mode!r}.")


def _check_settings(
    rules: Any, max_iterations: Any, max_nodes: Any, max_time: Any, fold_aware: Any, prefix: str = ""
) -> None:
    # The checks of the simplification settings of Evaluator (prefix "simplify_") and the
    # ExpressionSimplifier augmenter: TypeError for rules and fold_aware, ValueError for the limits.
    # None means simplify's default and is not checked.
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
    if fold_aware is not None and not isinstance(fold_aware, bool):
        raise TypeError(f"{prefix}fold_aware must be a bool or None, got {fold_aware!r}.")


def fold(
    expr: list[str] | Expression,
    symbol_library: SymbolLibrary | None = None,
    mode: Literal["none", "default", "keep"] = "keep",
) -> Expression:
    """Replace the constant parts of an expression with single constants.

    A constant part is a subexpression without variables that contains a free constant (one without a
    fixed spec). Each largest one becomes a bare ``C`` (the library's constant symbol), together with
    any numbers, ``"lit"`` symbols and fixed constants in it: ``C*C*X_0`` and ``sin(C + 2)*X_0`` both
    become ``C*X_0``. A part stays as it is if
    it contains a label also used elsewhere in the expression, or a constant that ``mode`` keeps.

    ``fold`` computes no values and does no algebra;
    [simplify][SRToolkit.utils.expression_simplifier.simplify] does both and gathers constants
    first, so ``simplify(fold=...)`` usually folds more. The result is a different hypothesis (see
    the module docs).

    Args:
        expr: The expression, as a token list or an
            [Expression][SRToolkit.utils.expression.Expression] carrying the specs of its labelled
            constants.
        symbol_library: Symbol library of the expression. ``None`` uses the expression's own
            library, or the active or default one for a token list.
        mode: Which constants may fold.

            - ``"none"``: nothing is folded; the expression comes back as given, unparsed.
            - ``"default"``: every constant may fold; the specs of folded labels are dropped.
            - ``"keep"``: labelled constants with a spec (fixed or not) never fold and keep their
              specs. Without specs the same as ``"default"``.

    Returns:
        The folded expression in the library of ``expr``, with the specs of ``expr`` whose label
        it still contains, in the given order. An ``Expression`` that does not change is returned
        itself.

    Raises:
        TypeError: If ``expr`` is neither a list or tuple of token strings nor an ``Expression``.
        ExpressionParseError: If the tokens cannot be parsed (not with ``mode="none"``, which
            returns the expression unparsed).
        ValueError: If ``mode`` is unknown.

    Examples:
        >>> str(fold(["C", "*", "C", "*", "X_0", "+", "sin", "(", "C", "+", "2", ")"]))
        'C*X_0+C'
        >>> from SRToolkit.utils import Constant, Expression
        >>> g = Constant(label="C_1", value=9.81, fixed=True)
        >>> expr = Expression(["C_1", "*", "C", "*", "X_0"], free_constants=[g])
        >>> fold(expr) is expr
        True
        >>> folded = fold(expr, mode="default")
        >>> str(folded), folded.free_constants
        ('C*X_0', ())
    """
    _check_fold_mode(mode, "mode")
    expr = _as_expression(expr, symbol_library)
    if mode == "none":
        return expr
    tree, kept, changed = _fold_tree(expr.tree, expr.symbol_library, expr.free_constants, mode)
    if not changed:  # not written out: to_list may add brackets
        return expr
    return Expression.from_tree(tree, expr.symbol_library, kept)


def _fold_tree(
    tree: Node, symbol_library: SymbolLibrary, free_constants: tuple[Constant, ...], mode: str
) -> tuple[Node, tuple[Constant, ...], bool]:
    # fold on a tree, for a mode other than "none": the folded tree (new nodes; tree is not
    # changed), the specs whose label it still contains, and whether anything folded.
    constant = symbol_library.constant_symbol
    kinds: dict[str, str | None] = {}  # symbol -> its type in the library (None for a number)

    # Bottom-up, for each node: (has a variable, has a bare constant, the labels inside with their
    # counts, or None if there are none).
    info: dict[int, tuple[bool, bool, dict[str, int] | None]] = {}

    def scan(node: Node) -> tuple[bool, bool, dict[str, int] | None]:
        if node.symbol not in kinds:
            symbol = symbol_library.get(node.symbol)
            kinds[node.symbol] = symbol.type if symbol is not None else None
        kind = kinds[node.symbol]
        has_var = kind == "var"
        bare = kind == "const" and node.symbol == constant
        labels = {node.symbol: 1} if kind == "const" and not bare else None
        for child in (node.left, node.right):
            if child is not None:
                child_var, child_bare, inside = scan(child)
                has_var, bare = has_var or child_var, bare or child_bare
                if inside and labels is None:
                    labels = inside  # shared with the child; copied before it is changed
                elif inside and labels is not None:
                    labels = {**labels}
                    for label, n in inside.items():
                        labels[label] = labels.get(label, 0) + n
        info[id(node)] = (has_var, bare, labels)
        return info[id(node)]

    total = scan(tree)[2] or {}
    specs = {spec.label: spec for spec in free_constants}
    pinned = set(specs) if mode == "keep" else set()
    free = {label for label in total if not (label in specs and specs[label].fixed)}
    folded: set[str] = set()  # the labels folded away (all their occurrences)
    changed = False

    def rebuild(node: Node) -> Node:
        nonlocal changed
        has_var, bare, labels = info[id(node)]
        labels = labels or {}
        if (
            node.left is not None
            and not has_var
            and (bare or not free.isdisjoint(labels))
            and pinned.isdisjoint(labels)
            and all(total[label] == n for label, n in labels.items() if label in free)
        ):
            folded.update(label for label, n in labels.items() if total[label] == n)
            changed = True
            return Node(cast(str, constant))
        return Node(
            node.symbol,
            left=rebuild(node.left) if node.left is not None else None,
            right=rebuild(node.right) if node.right is not None else None,
        )

    result = rebuild(tree)
    return result, tuple(spec for spec in free_constants if spec.label not in folded), changed


def simplify(
    expr: list[str] | Expression,
    symbol_library: SymbolLibrary | None = None,
    rules: SimplificationRules | None = None,
    max_iterations: int = 30,
    max_nodes: int = 10_000,
    max_time: float | None = None,
    fold: Literal["none", "default", "keep"] = "none",
    fold_aware: bool = True,
) -> Expression:
    """Simplify an expression algebraically, and fold its constants if asked.

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
        expr: The expression, as a token list or an
            [Expression][SRToolkit.utils.expression.Expression] carrying the specs of its labelled
            constants. A fixed spec is used as its value; the bounds of the others are recorded but
            not used by the default rules.
        symbol_library: Symbol library of the expression. ``None`` uses the expression's own
            library, or the active or default one for a token list.
        rules: The rewrite rules; ``None`` uses [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules].
        max_iterations: Maximum number of steps; ``0`` applies no rules (the result is still
            folded; a token list may gain redundant brackets).
        max_nodes: The search stops once the e-graph has grown to more than ``max_nodes``
            entries since the last cheaper expression was found.
        max_time: Seconds after which the search stops; ``None`` for no time limit.
        fold: Fold the result with [fold][SRToolkit.utils.expression_simplifier.fold] in this
            mode (``"none"``, ``"default"`` or ``"keep"``, see its ``mode``); ``"none"`` keeps the
            hypothesis.
        fold_aware: With folding, also look for the form that is shortest after folding (see the
            module docs): never longer than with ``False``, often shorter, but slower (up to about
            ten times on hard expressions). ``False`` chooses by token count and then folds.
            Ignored with ``fold="none"``.

    Returns:
        The simplified expression in the library of ``expr``, with the specs of ``expr`` whose
        label it still contains, in the given order. An ``Expression`` that does not change is
        returned itself.

    Raises:
        TypeError: If ``expr`` is neither a list or tuple of token strings nor an ``Expression``.
        ExpressionParseError: If the tokens cannot be parsed.
        ValueError: If ``fold`` is unknown.

    Examples:
        >>> print(simplify(["C", "*", "X_0", "+", "X_1", "*", "X_0", "/", "X_1", "*", "C"]))
        (C+C)*X_0
        >>> print(simplify(["C", "*", "X_0", "+", "X_1", "*", "X_0", "/", "X_1", "*", "C"], fold="keep"))
        C*X_0
        >>> from SRToolkit.utils import Constant, Expression
        >>> fixed = Constant(label="C_1", value=2.0, fixed=True)
        >>> simplified = simplify(Expression(["C_1", "*", "X_0", "+", "C_1", "*", "3"], free_constants=[fixed]))
        >>> str(simplified), simplified.free_constants == (fixed,)
        ('C_1*X_0+6', True)
    """
    _check_fold_mode(fold, "fold")
    if rules is None:
        rules = DefaultRules()
    expr = _as_expression(expr, symbol_library)
    symbol_library = expr.symbol_library
    tree = expr.tree
    free_constants = expr.free_constants
    labels = set(expr.parameter_labels)
    specs = {spec.label: spec for spec in free_constants}

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
    found: tuple[egg.EGraph, Any] | None = None  # the e-graph in which best was found, and its handle
    explored = False  # whether rules ran on the current e-graph without finding anything cheaper
    best_cost = math.inf
    for _ in range(max_iterations):
        report = egraph.run(schedule)
        candidate, cost = egraph.extract(handle, include_cost=True)
        explored = True
        if cost < best_cost:
            best, best_cost = candidate, cost
            found = egraph, handle
            egraph, handle = new_egraph(best)
            explored = False
        elif not report.updated or sum(size for _, size in egraph.all_function_sizes()) > max_nodes:
            break
        if max_time is not None and time.monotonic() - start > max_time:
            break

    def to_node(term: Any, copies: dict[int, list[Node]]) -> Node:
        # The term as a tree; copies collects the nodes of each bare constant (leaf id -> nodes).
        fn, args = egg.get_callable_fn(term), cast(Any, egg.get_callable_args(term))
        if fn == EggExpr.var:
            return Node(args[0].value)
        if fn == EggExpr.op or fn == ConstTree.node2:
            return Node(args[0].value, left=to_node(args[1], copies), right=to_node(args[2], copies))
        if fn == EggExpr.fn or fn == ConstTree.node1:
            return Node(args[0].value, left=to_node(args[1], copies))
        if fn == EggExpr.c:
            return to_node(args[0], copies)
        leaf_id = args[0].value
        if leaf_id >= 0:
            node = Node(tokens[leaf_id])
            if tokens[leaf_id] == symbol_library.constant_symbol:
                copies.setdefault(leaf_id, []).append(node)
            return node
        value = args[1].value
        return Node(str(int(value)) if value.is_integer() else repr(value))

    def write_out(term: Any) -> tuple[Node, tuple[Constant, ...]]:
        # The term as a tree with the specs of the labels it contains, folded if asked.
        copies: dict[int, list[Node]] = {}
        result = to_node(term, copies)
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
        remaining = set(_parameter_labels(result, symbol_library)) if free_constants else set()
        kept = tuple(spec for spec in free_constants if spec.label in remaining)
        if fold == "none":
            return result, kept
        folded, kept, _ = _fold_tree(result, symbol_library, kept, fold)
        return folded, kept

    result, kept = write_out(best)
    if fold != "none" and fold_aware:
        # The forms that are cheapest after folding, from the e-graph in which best was found and
        # from the last one if rules ran on it (both hold best). The cost cannot see what keeps a
        # part from folding (labels used elsewhere, bare constants a rule copied, values the rules
        # cannot compute), so a fold-aware form is kept only if it is shorter once folded.
        pinned = frozenset(i for i, token in enumerate(tokens) if token in specs) if fold == "keep" else frozenset()

        def fold_cost(egraph: egg.EGraph, expr: egg.BaseExpr, children_costs: list[Any]) -> Any:
            # (tokens after folding, written-out tokens, not absorbable); an absorbable part has no
            # variable and no pinned leaf (leaf ids of constants that must not fold) and folds to
            # one token.
            fn = egg.get_callable_fn(expr)
            if fn is None:  # a primitive argument: a name, id or bound
                return (0, 0, False)
            kind = cast(Any, fn).__egg_ref__.method_name
            if kind == "leaf":
                return (1, 1, bool(pinned) and cast(Any, egg.get_callable_args(expr))[0].value in pinned)
            if kind == "var":
                return (1, 1, True)
            if kind == "c":
                return children_costs[0]
            parts = children_costs[1:]  # op, fn, node1, node2: the first argument is the name
            tokens = 1 + sum(part[1] for part in parts)
            if not any(part[2] for part in parts):
                return (1, tokens, False)
            return (1 + sum(part[0] for part in parts), tokens, True)

        for graph, graph_handle in [g for g in (found, (egraph, handle) if explored else None) if g is not None]:
            aware, aware_kept = write_out(graph.extract(graph_handle, cost_model=fold_cost))
            if len(aware) < len(result):
                result, kept = aware, aware_kept
    tokens = result.to_list(symbol_library=symbol_library)
    if tokens == list(expr.tokens) and kept == free_constants:
        return expr
    return Expression(tokens, symbol_library, kept)
