"""The token vocabulary of symbolic expressions.

A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] maps each token to a
[Symbol][SRToolkit.utils.symbol_library.Symbol] holding its properties: type, precedence, NumPy
and LaTeX templates, Cython dispatch ID and [UnitRule][SRToolkit.utils.units.UnitRule].
[SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]
lists the predefined tokens.

**Symbol types.**

- ``"op"``: binary operator (e.g. ``+``, ``*``).
- ``"fn"``: unary function (e.g. ``sin``, ``sqrt``), prefix or postfix (``^2``).
- ``"lit"``: named constant, a symbol with a fixed value (e.g. ``pi``, ``e``). Elsewhere in the
  toolkit "named constant" always means a ``"lit"`` symbol.
- ``"const"``: the free constant, whose value is fitted during parameter estimation.
- ``"var"``: input variable, a column of the data array ``X``.

**Free, labelled and named constants.** A library has at most one free constant, a ``"const"``
symbol ``s`` (``C`` by default). Each occurrence of ``s`` in an expression is a separate
parameter. The library also knows the *labelled constants* ``<s>_<n>``, ``n`` one or more digits
(``C_0``, ``C_17``): all occurrences of one labelled constant are one parameter. The label is the
whole token, so ``C_1`` and ``C_01`` are different parameters. Labelled constants are not entries
of ``symbols``; [get][SRToolkit.utils.symbol_library.SymbolLibrary.get] returns the symbol of
``s`` for them. Free constants, bare or labelled, are parameters of an expression; a known value
that belongs to the task and has a name of its own (e.g. ``g``) is a named constant (``"lit"``)
instead.

**Active library.** Functions that take an optional library use the active one: the innermost
``with library:`` block, else the default set by
[set_default][SRToolkit.utils.symbol_library.SymbolLibrary.set_default] (see
[get_active][SRToolkit.utils.symbol_library.SymbolLibrary.get_active] and
[get_or_default][SRToolkit.utils.symbol_library.SymbolLibrary.get_or_default]).
"""

import importlib
import re
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from SRToolkit.utils.types import (
    FN_POSTFIX,
    FN_PREFIX,
    LEAF,
    OP_ADDITIVE,
    OP_MULTIPLICATIVE,
    OP_POWER,
    VALID_SYMBOL_TYPES,
)
from SRToolkit.utils.units import UnitRule

# Unit rules of the predefined symbols, keyed by (token, type); see default_symbols.
_PREDEFINED_UNIT_RULES: dict[tuple, UnitRule] = {
    **{(t, "op"): UnitRule.SAME for t in ("+", "-")},
    ("*", "op"): UnitRule.MULTIPLY,
    ("/", "op"): UnitRule.DIVIDE,
    ("^", "op"): UnitRule.EXPONENT,
    **{(t, "fn"): UnitRule.SAME for t in ("u-", "floor", "ceil")},
    ("sqrt", "fn"): UnitRule.power(Fraction(1, 2)),
    **{(f"^{k}", "fn"): UnitRule.power(k) for k in (-1, 2, 3, 4, 5)},
}


@dataclass(frozen=True)
class Symbol:
    r"""Immutable record of one token of a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].

    Usually built by [SymbolLibrary.add_symbol][SRToolkit.utils.symbol_library.SymbolLibrary.add_symbol]
    and returned by [SymbolLibrary.get][SRToolkit.utils.symbol_library.SymbolLibrary.get].

    Attributes:
        token: The token string.
        type: ``"op"``, ``"fn"``, ``"lit"``, ``"const"`` or ``"var"`` (see the
            [module docstring][SRToolkit.utils.symbol_library]).
        precedence: For ``"op"``, the binding strength, ``>= 0``: ``OP_ADDITIVE`` (100),
            ``OP_MULTIPLICATIVE`` (200), ``OP_POWER`` (300) or a custom value. For ``"fn"``, a
            positive value (``FN_PREFIX``, 1000) makes a prefix function and zero or a negative
            value (``FN_POSTFIX``, -1000) a postfix one. ``LEAF`` (1000) for the other types.
        python_str: Python expression template used in compiled callables, e.g. ``"np.sin({})"``.
            ``{}`` (or ``{0}``, ``{1}``) placeholders take the arguments as source text, so a
            repeated or embedded argument should be bracketed. It may use NumPy as ``np`` and the
            library's ``modules``.
        latex_str: LaTeX ``str.format`` template (literal braces doubled) with ``{}`` placeholders
            for the arguments. ``""`` generates a default, shown here rendered: ``a \text{tok} b``
            for an ``"op"``, ``\text{tok} a`` for a ``"fn"``, ``tok_{n}`` for the ``"const"``
            (``n`` the constant's number; ``{tok}_{n}`` if ``tok`` contains ``_``), and
            ``\text{tok}`` otherwise.
        cython_id: Dispatch ID of the Cython evaluator; ``-1`` means no C implementation
            (evaluated through a Python callable).
        unit_rule: How an ``"op"`` or ``"fn"`` combines the physical units of its arguments
            (see [UnitRule][SRToolkit.utils.units.UnitRule]). ``None`` gives the rule of the
            predefined symbol with the same token and type (see
            [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]),
            otherwise ``UnitRule.DIMENSIONLESS``. Always ``None`` for the other types.

    Raises:
        ValueError: If ``type`` is invalid, an ``"op"`` has a negative ``precedence``, or
            ``unit_rule`` is given for a type other than ``"op"`` and ``"fn"`` or does not accept
            the symbol's number of arguments (2 for ``"op"``, 1 for ``"fn"``).
        TypeError: If ``unit_rule`` is not a [UnitRule][SRToolkit.utils.units.UnitRule] or ``None``.

    Examples:
        >>> s = SymbolLibrary.default_symbols(1).get("sqrt")
        >>> s.type, s.precedence, s.python_str, s.cython_id
        ('fn', 1000, 'np.sqrt({})', 14)
        >>> s.unit_rule
        UnitRule.power(Fraction(1, 2))
    """

    token: str
    type: str
    precedence: int
    python_str: str
    latex_str: str = ""
    cython_id: int = -1
    unit_rule: UnitRule | None = None

    def __post_init__(self) -> None:
        if self.type not in VALID_SYMBOL_TYPES:
            raise ValueError(f"Invalid symbol type '{self.type}'. Must be one of: {sorted(VALID_SYMBOL_TYPES)}")
        if self.type == "op" and self.precedence < 0:
            raise ValueError(
                f"Operator '{self.token}' has precedence {self.precedence}; operators need a precedence >= 0."
            )

        if not self.latex_str:
            if self.type == "op":
                latex_str = r"{} \text{{" + self.token + r"}} {}"
            elif self.type == "fn":
                latex_str = r"\text{{" + self.token + r"}} {}"
            elif self.type == "const":
                latex_str = ("{{" + self.token + "}}" if "_" in self.token else self.token) + "_{{{}}}"
            else:
                latex_str = r"\text{{" + self.token + r"}}"
            object.__setattr__(self, "latex_str", latex_str)

        rule = self.unit_rule
        if rule is not None and not isinstance(rule, UnitRule):
            raise TypeError(f"Symbol '{self.token}': unit_rule must be a UnitRule or None, got {type(rule).__name__}.")
        if self.type in ("op", "fn"):
            arity = 2 if self.type == "op" else 1
            if rule is None:
                rule = _PREDEFINED_UNIT_RULES.get((self.token, self.type), UnitRule.DIMENSIONLESS)
            elif arity not in rule.arities:
                raise ValueError(f"Symbol '{self.token}': {rule!r} does not accept {arity} argument(s).")
            object.__setattr__(self, "unit_rule", rule)
        elif rule is not None:
            raise ValueError(f"Symbol '{self.token}' of type '{self.type}' cannot have a unit rule.")

    def to_dict(self) -> dict:
        """Serialize the symbol to a JSON-safe dictionary.

        Returns:
            The keys ``symbol`` (the token), ``type``, ``precedence``, ``np_fn`` (``python_str``), ``latex_str``,
            ``cython_id`` and, for ``"op"`` and ``"fn"``, ``unit_rule`` as
            [UnitRule.to_dict][SRToolkit.utils.units.UnitRule.to_dict].
        """
        d: dict[str, Any] = {
            "symbol": self.token,
            "type": self.type,
            "precedence": self.precedence,
            "np_fn": self.python_str,
            "latex_str": self.latex_str,
            "cython_id": self.cython_id,
        }
        if self.unit_rule is not None:
            d["unit_rule"] = self.unit_rule.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Symbol":
        """Reconstruct a [Symbol][SRToolkit.utils.symbol_library.Symbol] from its dictionary form.

        Args:
            d: Dictionary with the keys of [to_dict][SRToolkit.utils.symbol_library.Symbol.to_dict].
                ``cython_id`` may be missing (``-1``) and ``unit_rule`` missing or ``None`` (the
                default rule).

        Returns:
            The reconstructed symbol.

        Raises:
            KeyError: If any other key is missing.
            ValueError: If the symbol is invalid (see [Symbol][SRToolkit.utils.symbol_library.Symbol])
                or so is the saved unit rule (see
                [UnitRule.from_dict][SRToolkit.utils.units.UnitRule.from_dict]).
        """
        rule = d.get("unit_rule")
        return Symbol(
            d["symbol"],
            d["type"],
            d["precedence"],
            d["np_fn"],
            d["latex_str"],
            d.get("cython_id", -1),
            None if rule is None else UnitRule.from_dict(rule),
        )


# The predefined symbols, without the free constant and the variables; see default_symbols.
_DEFAULT_SYMBOLS: tuple[Symbol, ...] = (
    Symbol("+", "op", OP_ADDITIVE, "{} + {}", r"{} + {}", 0),
    Symbol("-", "op", OP_ADDITIVE, "{} - {}", r"{} - {}", 1),
    Symbol("*", "op", OP_MULTIPLICATIVE, "{} * {}", r"{} \cdot {}", 2),
    Symbol("/", "op", OP_MULTIPLICATIVE, "{} / {}", r"\frac{{{}}}{{{}}}", 3),
    Symbol("^", "op", OP_POWER, "np.power({},{})", r"{}^{{{}}}", 4),
    Symbol("u-", "fn", FN_PREFIX, "-{}", r"- {}", 25),
    Symbol("sqrt", "fn", FN_PREFIX, "np.sqrt({})", r"\sqrt {{{}}}", 14),
    Symbol("sin", "fn", FN_PREFIX, "np.sin({})", r"\sin {}", 10),
    Symbol("cos", "fn", FN_PREFIX, "np.cos({})", r"\cos {}", 11),
    Symbol("exp", "fn", FN_PREFIX, "np.exp({})", r"e^{{{}}}", 13),
    Symbol("tan", "fn", FN_PREFIX, "np.tan({})", r"\tan {}", 12),
    Symbol("arcsin", "fn", FN_PREFIX, "np.arcsin({})", r"\arcsin {}", 17),
    Symbol("arccos", "fn", FN_PREFIX, "np.arccos({})", r"\arccos {}", 18),
    Symbol("arctan", "fn", FN_PREFIX, "np.arctan({})", r"\arctan {}", 19),
    Symbol("sinh", "fn", FN_PREFIX, "np.sinh({})", r"\sinh {}", 20),
    Symbol("cosh", "fn", FN_PREFIX, "np.cosh({})", r"\cosh {}", 21),
    Symbol("tanh", "fn", FN_PREFIX, "np.tanh({})", r"\tanh {}", 22),
    Symbol("floor", "fn", FN_PREFIX, "np.floor({})", r"\lfloor {} \rfloor", 23),
    Symbol("ceil", "fn", FN_PREFIX, "np.ceil({})", r"\lceil {} \rceil", 24),
    Symbol("ln", "fn", FN_PREFIX, "np.log({})", r"\ln {}", 15),
    Symbol("log", "fn", FN_PREFIX, "np.log10({})", r"\log_{{10}} {}", 16),
    Symbol("^-1", "fn", FN_POSTFIX, "1/{}", r"{}^{{-1}}", 26),
    Symbol("^2", "fn", FN_POSTFIX, "{}**2", r"{}^2", 27),
    Symbol("^3", "fn", FN_POSTFIX, "{}**3", r"{}^3", 28),
    Symbol("^4", "fn", FN_POSTFIX, "{}**4", r"{}^4", 29),
    Symbol("^5", "fn", FN_POSTFIX, "{}**5", r"{}^5", 30),
    Symbol("pi", "lit", LEAF, "np.pi", r"\pi"),
    Symbol("e", "lit", LEAF, "np.e", r"e"),
)

_active_sl: ContextVar["SymbolLibrary"] = ContextVar("active_symbol_library")
_default_sl: "SymbolLibrary | None" = None


class SymbolLibrary:
    r"""Registry of the tokens of symbolic expressions and their properties.

    The toolkit uses it to parse, compile, display and generate expressions. Symbols can be selected from the predefined set (see
    [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]) and added with
    [add_symbol][SRToolkit.utils.symbol_library.SymbolLibrary.add_symbol]. A library can be used as
    a context manager to make it the active library (see the
    [module docstring][SRToolkit.utils.symbol_library]).

    Args:
        symbols: Predefined tokens to include. ``None`` adds none besides the variables and the
            free constant.
        num_variables: Number of variables, ``X_0`` through ``X_{num_variables-1}``, reading the
            columns of ``X`` in order.
        modules: Modules that ``python_str`` templates of custom symbols may use, as alias → module
            path (e.g. ``{"sp": "scipy.special"}``). They are imported here. NumPy is always
            available as ``np``, an alias that is reserved.
        const_symbol: Token of the free constant; ``None`` for a library without one.

    Attributes:
        symbols: Mapping from token to its [Symbol][SRToolkit.utils.symbol_library.Symbol], in
            insertion order. Change it only through ``add_symbol`` and ``remove_symbol``.
        modules: Mapping from alias to module path (a copy of the argument; ``{}`` for ``None``).
        num_variables: Column of ``X`` given to the next ``"var"`` added without ``python_str``.
            Raised by one on every ``"var"`` added, never lowered.

    Raises:
        ValueError: If ``symbols`` contains a token that is not predefined, or ``modules`` uses
            the alias ``"np"``.
        ImportError: If a module in ``modules`` cannot be imported.

    Examples:
        >>> library = SymbolLibrary(["+", "*", "sin"], num_variables=2)
        >>> library.symbols.keys()
        dict_keys(['+', '*', 'sin', 'C', 'X_0', 'X_1'])
        >>> library = SymbolLibrary(["+"], 1, modules={"sp": "scipy.special"})
        >>> library.add_symbol("gamma", "fn", FN_PREFIX, "sp.gamma({})", r"\Gamma({})")
        >>> with SymbolLibrary.default_symbols(num_variables=2):
        ...     SymbolLibrary.get_active().get("X_1").python_str
        'X[:, 1]'
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        num_variables: int = 0,
        modules: dict[str, str] | None = None,
        const_symbol: str | None = "C",
    ) -> None:
        self.modules = dict(modules or {})
        if "np" in self.modules:
            raise ValueError("The alias 'np' is reserved for NumPy and cannot be used in modules.")
        self.module_namespace()

        self._constant_symbol: str | None = None
        if symbols is None and num_variables == 0:
            self.symbols: dict[str, Symbol] = dict()
            self.num_variables = 0
            if const_symbol is not None:
                self._add_constant(const_symbol)
        else:
            selected = SymbolLibrary.from_symbol_list(symbols or [], num_variables, const_symbol)
            self.symbols = selected.symbols
            self._constant_symbol = selected._constant_symbol
            self.num_variables = num_variables

    def _add_constant(self, name: str) -> None:
        self.add_symbol(name, "const", LEAF, "np.full(X.shape[0], C[{}])")

    def add_symbol(
        self,
        symbol: str,
        symbol_type: str,
        precedence: int,
        python_str: str,
        latex_str: str | None = None,
        cython_id: int = -1,
        unit_rule: UnitRule | None = None,
    ):
        r"""Add a token to the library, replacing any symbol with the same token.

        The arguments are the fields of [Symbol][SRToolkit.utils.symbol_library.Symbol], which
        documents them. Predefined symbols (see
        [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]) are
        faster than custom ones: they have C implementations for the Cython evaluator, while
        custom symbols are evaluated through Python callables.

        Args:
            symbol: Token to register.
            symbol_type: ``"op"``, ``"fn"``, ``"lit"``, ``"const"`` or ``"var"``.
            precedence: Precedence (see [Symbol][SRToolkit.utils.symbol_library.Symbol]).
            python_str: Python expression template (see [Symbol][SRToolkit.utils.symbol_library.Symbol]).
                For a ``"var"``, ``None`` or ``""`` gives ``"X[:, i]"`` with ``i`` the library's
                ``num_variables``.
            latex_str: LaTeX template; ``None`` or ``""`` generates the default for the type.
            cython_id: Dispatch ID of the Cython evaluator; ``-1`` evaluates through a Python
                callable. To keep a predefined symbol's acceleration, pass
                ``SymbolLibrary.default_symbols()[token].cython_id``.
            unit_rule: Unit rule of an ``"op"`` or ``"fn"``, e.g. ``UnitRule.power(Fraction(1, 3))``
                for a cube root. ``None`` gives the predefined rule for the token and type,
                otherwise ``UnitRule.DIMENSIONLESS``, so give a custom symbol that accepts
                physical units its rule.

        Raises:
            ValueError: If the symbol is invalid (see [Symbol][SRToolkit.utils.symbol_library.Symbol]),
                or ``symbol_type`` is ``"const"`` and the library has a free constant with another
                token, or ``symbol`` is the free constant and ``symbol_type`` is not ``"const"``.
            TypeError: If ``unit_rule`` is not a [UnitRule][SRToolkit.utils.units.UnitRule] or
                ``None``.

        Examples:
            >>> library = SymbolLibrary(["+"], num_variables=1)
            >>> library.add_symbol("sat", "fn", FN_PREFIX, "({0}) / (1 + ({0}))", r"\frac{{{0}}}{{1 + {0}}}")
            >>> from SRToolkit.utils import compile_expr
            >>> f = compile_expr(["sat", "(", "X_0", ")"], library)
            >>> f(np.array([[1.0], [3.0]]), None)
            array([0.5 , 0.75])
        """
        if symbol_type == "var" and not python_str:
            python_str = f"X[:, {self.num_variables}]"
        new = Symbol(symbol, symbol_type, precedence, python_str, latex_str or "", cython_id, unit_rule)
        if symbol_type == "const" and self.constant_symbol not in (None, symbol):
            raise ValueError(
                f"Cannot add free constant '{symbol}': the library already has '{self.constant_symbol}', "
                "and a library has at most one. Use labelled constants (C_0, C_1, ...) for shared parameters."
            )
        if symbol_type != "const" and symbol == self.constant_symbol:
            raise ValueError(
                f"Cannot redefine '{symbol}' as '{symbol_type}': it is the library's free constant. Call "
                f"remove_symbol('{symbol}') first, or build the library with const_symbol=None or another name."
            )
        if symbol_type == "var":
            self.num_variables += 1
        self.symbols[symbol] = new
        if symbol_type == "const":
            self._constant_symbol = symbol

    def remove_symbol(self, symbol: str):
        """Remove a token from the library.

        Examples:
            >>> library = SymbolLibrary(const_symbol=None)
            >>> library.add_symbol("x", "var", 0, "x")
            >>> len(library.symbols)
            1
            >>> library.remove_symbol("x")
            >>> len(library.symbols)
            0

        Args:
            symbol: Token string to remove.

        Raises:
            KeyError: If ``symbol`` is not present in the library.
        """
        del self.symbols[symbol]
        if symbol == self._constant_symbol:
            self._constant_symbol = None

    @property
    def constant_symbol(self) -> str | None:
        """The token of the free constant, or ``None`` if the library has none.

        Examples:
            >>> SymbolLibrary.default_symbols().constant_symbol
            'C'
        """
        return self._constant_symbol

    def get(self, token: str) -> Symbol | None:
        """Return the [Symbol][SRToolkit.utils.symbol_library.Symbol] of a token.

        Args:
            token: Token to look up.

        Returns:
            The symbol registered for ``token``; otherwise, for a labelled constant of the free
            constant (see the [module docstring][SRToolkit.utils.symbol_library]), the free
            constant's symbol; otherwise ``None`` (brackets, numbers, unknown names). A registered
            token wins, so a ``"lit"`` named ``C_1`` is returned as itself.

        Examples:
            >>> library = SymbolLibrary.default_symbols(1)
            >>> library.get("+").precedence
            100
            >>> library.get("C_3").token
            'C'
            >>> print(library.get("("))
            None
        """
        symbol = self.symbols.get(token)
        if symbol is not None or not isinstance(token, str):
            return symbol
        base, _, label = token.rpartition("_")
        symbol = self.symbols.get(base)
        if symbol is not None and symbol.type == "const" and label.isascii() and label.isdigit():
            return symbol
        return None

    def __getitem__(self, token: str) -> Symbol:
        """Return the [Symbol][SRToolkit.utils.symbol_library.Symbol] of a token, or raise if there is none.

        Args:
            token: Token to look up.

        Returns:
            The symbol that [get][SRToolkit.utils.symbol_library.SymbolLibrary.get] returns.

        Raises:
            KeyError: If [get][SRToolkit.utils.symbol_library.SymbolLibrary.get] returns ``None``.

        Examples:
            >>> SymbolLibrary.default_symbols(1)["C_3"].type
            'const'
        """
        symbol = self.get(token)
        if symbol is None:
            raise KeyError(token)
        return symbol

    def __contains__(self, token: object) -> bool:
        """Return whether [get][SRToolkit.utils.symbol_library.SymbolLibrary.get] finds a symbol for ``token``.

        Examples:
            >>> library = SymbolLibrary.default_symbols(1)
            >>> "X_0" in library, "C_3" in library, "(" in library
            (True, True, False)
        """
        return isinstance(token, str) and self.get(token) is not None

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered tokens, in insertion order.

        Examples:
            >>> list(SymbolLibrary(["+"], 1))
            ['+', 'C', 'X_0']
        """
        return iter(self.symbols)

    def get_symbols_of_type(self, symbol_type: str) -> list[str]:
        """Return the tokens of all symbols of a type, in insertion order.

        Args:
            symbol_type: ``"op"``, ``"fn"``, ``"lit"``, ``"const"`` or ``"var"``.

        Returns:
            The matching tokens; empty if there are none.

        Examples:
            >>> SymbolLibrary(["+", "sin"], num_variables=2).get_symbols_of_type("var")
            ['X_0', 'X_1']
        """
        return [token for token, s in self.symbols.items() if s.type == symbol_type]

    def module_namespace(self) -> dict[str, Any]:
        """Return the global names available to ``python_str`` templates.

        Returns:
            A new dict mapping ``"np"`` to NumPy and each alias in ``modules`` to its imported
            module.

        Raises:
            ImportError: If a module in ``modules`` cannot be imported.

        Examples:
            >>> sorted(SymbolLibrary(modules={"m": "math"}).module_namespace())
            ['m', 'np']
        """
        return {"np": np, **{alias: importlib.import_module(path) for alias, path in self.modules.items()}}

    def symbols2index(self) -> dict[str, int]:
        """Return a mapping from each token to its index in insertion order.

        Examples:
            >>> library = SymbolLibrary(const_symbol=None)
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("y", "var", 0, "y")
            >>> print(library.symbols2index())
            {'x': 0, 'y': 1}
            >>> library.remove_symbol("x")
            >>> print(library.symbols2index())
            {'y': 0}

        Returns:
            Dict mapping each token string to its zero-based position in the library.
        """
        return {s: i for i, s in enumerate(self.symbols.keys())}

    @staticmethod
    def from_symbol_list(
        symbols: list[str], num_variables: int = 25, const_symbol: str | None = "C"
    ) -> "SymbolLibrary":
        """Create a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] containing only the specified subset of default symbols.

        The supported token names are those defined in [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        The free constant is always included (listing it is allowed); ``const_symbol=None`` leaves
        it out.

        Examples:
            >>> library = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=2)
            >>> library.symbols.keys()
            dict_keys(['+', '*', 'C', 'X_0', 'X_1'])
            >>> SymbolLibrary.from_symbol_list(["+"], num_variables=1, const_symbol=None).symbols.keys()
            dict_keys(['+', 'X_0'])

        Args:
            symbols: Token strings to include.
            num_variables: Number of variable tokens (``X_0`` through ``X_{num_variables-1}``).
                Default is ``25``.
            const_symbol: Name of the free constant, as in
                [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].

        Returns:
            A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] restricted to the requested symbols, the variables
            and the free constant.

        Raises:
            ValueError: If ``symbols`` contains a name that is not in the default set (``C`` is
                not when ``const_symbol`` names another constant or is ``None``), such as a
                whole expression like ``"x/(a + x)"``. Expressions are built from tokens
                (``/``, ``+``, ``C``, variables), their shape can be restricted with a custom
                [Grammar][SRToolkit.utils.grammar.Grammar], and custom tokens are added with
                [add_symbol][SRToolkit.utils.symbol_library.SymbolLibrary.add_symbol].
        """
        sl = SymbolLibrary.default_symbols(num_variables, const_symbol)
        unknown = [s for s in symbols if s not in sl.symbols]
        if unknown:
            if any(re.fullmatch(r"X_\d+", s) for s in unknown):
                hint = f"Only variables X_0..X_{num_variables - 1} exist; increase num_variables."
            elif "C" in unknown:
                hint = f"The free constant is named by const_symbol (here {const_symbol!r})."
            else:
                hint = (
                    "Build expressions such as 'X/(C + X)' from tokens ('/', '+', 'C', variables), "
                    "restrict their shape with a custom Grammar, or add custom tokens with add_symbol."
                )
            raise ValueError(
                f"Unknown symbols {unknown}: only default tokens are accepted "
                f"(see SymbolLibrary.default_symbols). {hint}"
            )

        symbols = symbols + [f"X_{i}" for i in range(num_variables)] + ([const_symbol] if const_symbol else [])

        all_symbols = list(sl.symbols.keys())
        for symbol in all_symbols:
            if symbol not in symbols:
                sl.remove_symbol(symbol)

        return sl

    @staticmethod
    def default_symbols(num_variables: int = 25, const_symbol: str | None = "C") -> "SymbolLibrary":
        """Return a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] with all predefined symbols.

        The predefined tokens:

        - **Operators** (``"op"``): ``+``, ``-``, ``*``, ``/``, ``^``.
        - **Functions** (``"fn"``): ``u-`` (negation), ``sqrt``, ``sin``, ``cos``, ``exp``, ``tan``,
          ``arcsin``, ``arccos``, ``arctan``, ``sinh``, ``cosh``, ``tanh``, ``floor``, ``ceil``,
          ``ln``, ``log`` (base 10), and the postfix ``^-1``, ``^2``, ``^3``, ``^4``, ``^5``.
        - **Named constants** (``"lit"``): ``pi``, ``e``.
        - **Free constant** (``"const"``): ``const_symbol``.
        - **Variables** (``"var"``): ``X_0`` through ``X_{num_variables-1}``, reading the columns
          of ``X`` in order.

        [Unit rules][SRToolkit.utils.units.UnitRule]: ``+``, ``-``, ``u-``, ``floor``, ``ceil``:
        ``SAME``; ``*``: ``MULTIPLY``; ``/``: ``DIVIDE``; ``^``: ``EXPONENT``; ``sqrt``:
        ``power(1/2)``; ``^-1``, ``^2`` ... ``^5``: ``power(-1)``, ``power(2)`` ... ``power(5)``;
        all other functions: ``DIMENSIONLESS``.

        Args:
            num_variables: Number of variables.
            const_symbol: Token of the free constant; ``None`` for a library without one.

        Returns:
            A new library with the symbols listed above.

        Examples:
            >>> len(SymbolLibrary.default_symbols(num_variables=2))
            31
        """
        sl = SymbolLibrary(const_symbol=None)
        sl.symbols = {s.token: s for s in _DEFAULT_SYMBOLS}
        if const_symbol is not None:
            sl._add_constant(const_symbol)
        for i in range(num_variables):
            sl.add_symbol(f"X_{i}", "var", LEAF, f"X[:, {i}]", f"X_{{{i}}}")
        return sl

    def __enter__(self) -> "SymbolLibrary":
        self._token = _active_sl.set(self)
        return self

    def __exit__(self, *args) -> None:
        _active_sl.reset(self._token)

    @staticmethod
    def get_active() -> "SymbolLibrary":
        """Return the currently active [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].

        Checks, in order: (1) the context manager stack, (2) the module-level default set via
        [set_default][SRToolkit.utils.symbol_library.SymbolLibrary.set_default].

        Returns:
            The active [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] instance.

        Raises:
            RuntimeError: If no library is active and no default has been set.
        """
        try:
            return _active_sl.get()
        except LookupError:
            if _default_sl is not None:
                return _default_sl
            raise RuntimeError(
                "No active SymbolLibrary. Either pass one explicitly, use "
                "'with SymbolLibrary(...) as sl:', or call SymbolLibrary.set_default(sl)."
            )

    @staticmethod
    def get_or_default() -> "SymbolLibrary":
        """Return the active [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary], falling back to
        [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols] when nothing is active.

        Checks, in order: (1) the context manager stack, (2) the module-level default set via
        [set_default][SRToolkit.utils.symbol_library.SymbolLibrary.set_default], (3) a freshly
        constructed default library.

        Returns:
            The active or default [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] instance.
        """
        try:
            return _active_sl.get()
        except LookupError:
            if _default_sl is not None:
                return _default_sl
            return SymbolLibrary.default_symbols()

    @staticmethod
    def set_default(sl: "SymbolLibrary | None") -> None:
        """Set (or clear) a module-level default [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].

        The default is used as a fallback by [get_active][SRToolkit.utils.symbol_library.SymbolLibrary.get_active]
        and [get_or_default][SRToolkit.utils.symbol_library.SymbolLibrary.get_or_default] when no context manager
        is active. It is module-global (not per-thread or per-task) and intended for scripts and
        notebooks where a single library is used throughout a session.

        Pass ``None`` to clear the default.

        Args:
            sl: Library to set as the module-level default, or ``None`` to clear it.
        """
        global _default_sl
        _default_sl = sl

    def to_dict(self) -> dict:
        """Serialize the library to a JSON-safe dictionary.

        Returns:
            A dictionary for [from_dict][SRToolkit.utils.symbol_library.SymbolLibrary.from_dict]
            with the keys ``format_version`` (``1``), ``type`` (``"SymbolLibrary"``), ``symbols``
            (token → [Symbol.to_dict][SRToolkit.utils.symbol_library.Symbol.to_dict], in insertion
            order), ``modules`` and ``num_variables``.

        Examples:
            >>> d = SymbolLibrary.from_symbol_list(["+"], num_variables=1).to_dict()
            >>> d["symbols"]["+"]["unit_rule"]
            {'kind': 'same'}
        """
        return {
            "format_version": 1,
            "type": "SymbolLibrary",
            "symbols": {token: s.to_dict() for token, s in self.symbols.items()},
            "modules": dict(self.modules),
            "num_variables": self.num_variables,
        }

    @staticmethod
    def from_dict(d: dict) -> "SymbolLibrary":
        """Reconstruct a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] from its dictionary form.

        A missing ``format_version`` means ``1``. A dictionary saved before ``modules`` existed has
        a ``"preamble"`` of import lines instead; it is read only if ``modules`` is missing or
        ``None``, and may contain only ``import numpy as np``, which is dropped (NumPy is always
        available).

        Args:
            d: Dictionary with the keys of
                [to_dict][SRToolkit.utils.symbol_library.SymbolLibrary.to_dict].

        Returns:
            The reconstructed library.

        Raises:
            KeyError: If ``symbols`` or ``num_variables`` is missing.
            ValueError: If ``format_version`` is not ``1``, the library has more than one
                ``"const"`` symbol, a symbol is invalid (see
                [Symbol.from_dict][SRToolkit.utils.symbol_library.Symbol.from_dict]), ``modules``
                uses the alias ``"np"``, or a ``"preamble"`` line is not ``import numpy as np``.
            ImportError: If a module in ``modules`` cannot be imported.
        """
        if d.get("format_version", 1) != 1:
            raise ValueError(
                f"[SymbolLibrary.from_dict] Unsupported format_version: {d.get('format_version')!r}. Expected 1."
            )
        modules = d.get("modules")
        if modules is None:
            other = [line for line in d.get("preamble", []) if line.strip() != "import numpy as np"]
            if other:
                raise ValueError(
                    f"[SymbolLibrary.from_dict] Cannot convert preamble lines {other} to modules; only "
                    "'import numpy as np' is supported (other modules never worked with preamble)."
                )
            modules = {}
        symbols = {token: Symbol.from_dict(entry) for token, entry in d["symbols"].items()}
        consts = [token for token, s in symbols.items() if s.type == "const"]
        if len(consts) > 1:
            raise ValueError(f"[SymbolLibrary.from_dict] At most one free constant is allowed, got {consts}.")
        sl = SymbolLibrary(modules=modules, const_symbol=None)
        sl.symbols = symbols
        sl._constant_symbol = consts[0] if consts else None
        sl.num_variables = d["num_variables"]
        return sl

    def __len__(self) -> int:
        """Return the number of symbols currently in the library.

        Examples:
             >>> library = SymbolLibrary.default_symbols(5)
             >>> len(library)
             34
             >>> library.add_symbol("a", "lit", 5, "a", "a")
             >>> len(library)
             35

        Returns:
            Number of tokens registered in the library.
        """
        return len(self.symbols)

    def __str__(self) -> str:
        r"""Return a comma-separated string of all registered token strings.

        Examples:
            >>> library = SymbolLibrary(const_symbol=None)
            >>> library.add_symbol("x", "var", 0, "x", "x")
            >>> str(library)
            'x'
            >>> library.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
            >>> str(library)
            'x, sin'

        Returns:
            All token names joined by ``", "``, in insertion order.
        """
        return ", ".join(self.symbols.keys())

    def __copy__(self) -> "SymbolLibrary":
        """Return a copy of the library that can be changed independently of the original.

        Returns:
            A new [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] with its own
            ``symbols`` and ``modules`` (the [Symbol][SRToolkit.utils.symbol_library.Symbol]
            records are immutable and shared).

        Examples:
            >>> import copy
            >>> library = SymbolLibrary(["+"], num_variables=1)
            >>> new = copy.copy(library)
            >>> new.add_symbol("sin", "fn", FN_PREFIX, "np.sin({})")
            >>> str(library), str(new)
            ('+, C, X_0', '+, C, X_0, sin')
        """
        sl = SymbolLibrary(modules=self.modules, const_symbol=None)
        sl.symbols = dict(self.symbols)
        sl._constant_symbol = self._constant_symbol
        sl.num_variables = self.num_variables
        return sl

    def __deepcopy__(self, memo=None) -> "SymbolLibrary":
        return self.__copy__()
