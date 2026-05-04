"""
The [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] class for managing the token vocabulary used in symbolic
regression expressions.
"""

import copy
from typing import Any, Dict, List, Optional

from SRToolkit.utils.types import VALID_SYMBOL_TYPES


class SymbolLibrary:
    def __init__(
        self, symbols: Optional[List[str]] = None, num_variables: int = 0, preamble: Optional[List[str]] = None
    ) -> None:
        """
        A registry of tokens and their properties, used throughout the toolkit to parse,
        compile, and generate symbolic expressions.

        By default, the library uses NumPy for operator and function evaluation. To use a
        different backend, pass the required import statements via ``preamble``.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x", "x")
            >>> library.get_type("x")
            'var'
            >>> library.get_precedence("x")
            0
            >>> library.get_np_fn("x")
            'x'
            >>> library.remove_symbol("x")
            >>> library = SymbolLibrary.default_symbols()
            >>> # You can also initialize the library with a list of symbols (listed in SymbolLibrary.default_symbols)
            >>> # and the number of variables.
            >>> library2 = SymbolLibrary(["+", "*", "sin"], num_variables=2)
            >>> len(library2)
            5

        Args:
            symbols: Symbols to pre-populate from the default set. ``None`` with
                ``num_variables=0`` produces an empty library; with ``num_variables > 0``,
                only variable tokens are added. Symbols not present in the default set are
                silently ignored. See [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols] for the supported names.
            num_variables: Number of variable tokens to add, labeled ``X_0`` through
                ``X_{num_variables-1}``. Default is ``0``.
            preamble: Import statements prepended to compiled expression functions.
                Defaults to ``["import numpy as np"]``.

        Attributes:
            symbols: Mapping from token string to its property dict (type, precedence,
                NumPy function string, LaTeX template).
        """
        if preamble is None:
            self.preamble = ["import numpy as np"]
        else:
            self.preamble = preamble

        if symbols is None and num_variables == 0:
            self.symbols: Dict[str, Any] = dict()
            self.num_variables = 0
        else:
            if symbols is None:
                symbols = []

            self.symbols = SymbolLibrary.from_symbol_list(symbols, num_variables).symbols
            self.num_variables = num_variables

    def add_symbol(
        self,
        symbol: str,
        symbol_type: str,
        precedence: int,
        np_fn: str,
        latex_str: Optional[str] = None,
        cython_id: int = -1,
    ):
        r"""
        Add a token to the library with its associated type, precedence, NumPy function
        string, and LaTeX template.

        Symbol types:

        - ``"op"``: binary operator (e.g. ``+``, ``*``).
        - ``"fn"``: unary function (e.g. ``sin``, ``sqrt``).
        - ``"lit"``: literal with a fixed value (e.g. ``pi``, ``e``).
        - ``"const"``: free constant whose value is optimised during parameter estimation
          (e.g. ``C``). Using a single ``"const"`` token is recommended; multiple tokens
          increase complexity and reduce readability.
        - ``"var"``: input variable whose values are read from the data array ``X``.

        If ``latex_str`` is omitted, a default template is generated based on the symbol type:
        ``"{} \text{symb} {}"`` for operators (e.g. ``+`` → ``{} \text{+} {}``),
        ``"\text{symb} {}"`` for functions (e.g. ``sin`` → ``\text{sin} {}``),
        and ``"\text{symb}"`` for all other types.

        Note:
            For best performance, we recommend selecting symbols from the predefined set via the
            constructor or [from_symbol_list][SRToolkit.utils.symbol_library.SymbolLibrary.from_symbol_list]
            rather than adding custom ones. Predefined symbols have C implementations and
            benefit from Cython acceleration; custom symbols always fall back to Python callables.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
            >>> library.add_symbol("C", "const", 5, "C[{}]", r"c_{}")
            >>> library.add_symbol("X_0", "var", 5, "X[:, 0]", r"X_0")
            >>> library.add_symbol("pi", "lit", 5, "np.pi", r"\pi")

        Args:
            symbol: Token string to register.
            symbol_type: One of ``"op"``, ``"fn"``, ``"lit"``, ``"const"``, or ``"var"``.
            precedence: Operator precedence, used for infix reconstruction and PCFG generation.
            np_fn: Python/NumPy expression string used in compiled callables
                (e.g. ``"np.sin({})"``). For ``"var"`` type, passing ``None``
                or ``""`` auto-generates ``"X[:, i]"`` where ``i`` is the current
                ``num_variables`` count at the time of the call.
            latex_str: LaTeX template string with ``{}`` placeholders for operands.
                Auto-generated if omitted.
            cython_id: Integer dispatch ID used by the Cython-based evaluator.
                ``-1`` (default) means the symbol has no C implementation and will
                fall back to a Python callable during Cython evaluation. Custom symbols
                should leave this at ``-1``; to redefine a built-in symbol while
                preserving its Cython acceleration, look up its ID in the source of
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

        Raises:
            ValueError: If ``symbol_type`` is not one of the valid types.
        """
        if symbol_type not in VALID_SYMBOL_TYPES:
            raise ValueError(f"Invalid symbol type '{symbol_type}'. Must be one of: {sorted(VALID_SYMBOL_TYPES)}")

        if latex_str is None:
            if symbol_type == "op":
                latex_str = r"{} \text{{" + symbol + r"}} {}"
            elif symbol_type == "fn":
                latex_str = r"\text{{" + symbol + r"}} {}"
            else:
                latex_str = r"\text{{" + symbol + r"}}"

        if symbol_type == "var" and (np_fn is None or np_fn == ""):
            np_fn = "X[:, {}]".format(self.num_variables)

        if symbol_type == "var":
            self.num_variables += 1

        self.symbols[symbol] = {
            "symbol": symbol,
            "type": symbol_type,
            "precedence": precedence,
            "np_fn": np_fn,
            "latex_str": latex_str,
            "cython_id": cython_id,
        }

    def remove_symbol(self, symbol: str):
        """
        Remove a token from the library.

        Examples:
            >>> library = SymbolLibrary()
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

    def get_type(self, symbol: str) -> str:
        """
        Return the type of a symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_type("x")
            'var'

        Args:
            symbol: Token to look up.

        Returns:
            The type string (``"op"``, ``"fn"``, ``"lit"``, ``"const"``, or ``"var"``) if the symbol is in the library, otherwise an empty string.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["type"]
        else:
            return ""

    def get_precedence(self, symbol: str) -> int:
        """
        Return the precedence of a symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_precedence("x")
            0

        Args:
            symbol: Token to look up.

        Returns:
            The precedence value if the symbol is in the library, otherwise ``-1``.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["precedence"]
        else:
            return -1

    def get_np_fn(self, symbol: str) -> str:
        """
        Return the NumPy function string for a symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_np_fn("x")
            'x'

        Args:
            symbol: Token to look up.

        Returns:
            The NumPy function string if the symbol is in the library, otherwise an empty string.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["np_fn"]
        else:
            return ""

    def get_cython_id(self, symbol: str) -> int:
        """
        Return the Cython stack machine dispatch ID for a symbol.

        Returns ``-1`` for symbols without a C implementation (they will fall
        back to a Python callable during Cython evaluation).

        Examples:
            >>> library = SymbolLibrary.default_symbols()
            >>> library.get_cython_id("+")
            0
            >>> library.get_cython_id("sin")
            10
            >>> library.get_cython_id("unknown_symbol")
            -1

        Args:
            symbol: Token to look up.

        Returns:
            Integer dispatch ID, or ``-1`` if the symbol is not in the library
            or has no Cython implementation.
        """
        if symbol in self.symbols:
            return self.symbols[symbol].get("cython_id", -1)
        return -1

    def get_latex_str(self, symbol: str) -> str:
        """
        Return the LaTeX template string for a symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x", "test")
            >>> library.get_latex_str("x")
            'test'

        Args:
            symbol: Token to look up.

        Returns:
            The LaTeX template string if the symbol is in the library, otherwise an empty string.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["latex_str"]
        else:
            return ""

    def get_symbols_of_type(self, symbol_type: str) -> List[str]:
        """
        Return all symbols of a given type.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("y", "var", 0, "y")
            >>> library.get_symbols_of_type("var")
            ['x', 'y']

        Args:
            symbol_type: Type to filter by. One of ``"op"``, ``"fn"``, ``"var"``,
                ``"const"``, ``"lit"``.

        Returns:
            List of token strings matching the requested type. Returns an empty list
            if no symbols match or if ``symbol_type`` is not recognised.
        """
        symbols = list()
        for symbol in self.symbols.keys():
            if self.get_type(symbol) == symbol_type:
                symbols.append(symbol)

        return symbols

    def symbols2index(self) -> Dict[str, int]:
        """
        Return a mapping from each token to its index in insertion order.

        Examples:
            >>> library = SymbolLibrary()
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
    def from_symbol_list(symbols: List[str], num_variables: int = 25) -> "SymbolLibrary":
        """
        Create a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] containing only the specified subset of default symbols.

        The supported token names are those defined in [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

        Examples:
            >>> library = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)
            >>> len(library.symbols)
            5

        Args:
            symbols: Token strings to include. Symbols not in the default set are silently ignored.
            num_variables: Number of variable tokens (``X_0`` through ``X_{num_variables-1}``).
                Default is ``25``.

        Returns:
            A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] restricted to the requested symbols and variables.
        """
        variables = [f"X_{i}" for i in range(num_variables)]
        symbols = symbols + variables

        sl = SymbolLibrary.default_symbols(num_variables)

        all_symbols = list(sl.symbols.keys())
        for symbol in all_symbols:
            if symbol not in symbols:
                sl.remove_symbol(symbol)

        return sl

    @staticmethod
    def default_symbols(num_variables: int = 25) -> "SymbolLibrary":
        """
        Return a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] pre-populated with standard mathematical symbols.

        Supported tokens:

        - **Operators** (``"op"``): ``+``, ``-``, ``*``, ``/``, ``^``
        - **Functions** (``"fn"``): ``u-``, ``sqrt``, ``sin``, ``cos``, ``exp``, ``tan``,
          ``arcsin``, ``arccos``, ``arctan``, ``sinh``, ``cosh``, ``tanh``, ``floor``,
          ``ceil``, ``ln``, ``log``, ``^-1``, ``^2``, ``^3``, ``^4``, ``^5``
        - **Literals** (``"lit"``): ``pi``, ``e``
        - **Free constant** (``"const"``): ``C``
        - **Variables** (``"var"``): ``X_0`` through ``X_{num_variables-1}``,
          mapped to columns of the input array in order.

        Examples:
            >>> library = SymbolLibrary.default_symbols()
            >>> len(library)
            54

        Args:
            num_variables: Number of variable tokens to include. Default is ``25``.

        Returns:
            A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] populated with the symbols listed above.
        """
        sl = SymbolLibrary()
        sl.add_symbol(
            "+",
            symbol_type="op",
            precedence=0,
            np_fn="{} + {}",
            latex_str=r"{} + {}",
            cython_id=0,
        )
        sl.add_symbol(
            "-",
            symbol_type="op",
            precedence=0,
            np_fn="{} - {}",
            latex_str=r"{} - {}",
            cython_id=1,
        )
        sl.add_symbol(
            "*",
            symbol_type="op",
            precedence=1,
            np_fn="{} * {}",
            latex_str=r"{} \cdot {}",
            cython_id=2,
        )
        sl.add_symbol(
            "/",
            symbol_type="op",
            precedence=1,
            np_fn="{} / {}",
            latex_str=r"\frac{{{}}}{{{}}}",
            cython_id=3,
        )
        sl.add_symbol(
            "^",
            symbol_type="op",
            precedence=2,
            np_fn="np.power({},{})",
            latex_str=r"{}^{{{}}}",
            cython_id=4,
        )
        sl.add_symbol("u-", symbol_type="fn", precedence=5, np_fn="-{}", latex_str=r"- {}", cython_id=25)
        sl.add_symbol(
            "sqrt",
            symbol_type="fn",
            precedence=5,
            np_fn="np.sqrt({})",
            latex_str=r"\sqrt {{{}}}",
            cython_id=14,
        )
        sl.add_symbol(
            "sin",
            symbol_type="fn",
            precedence=5,
            np_fn="np.sin({})",
            latex_str=r"\sin {}",
            cython_id=10,
        )
        sl.add_symbol(
            "cos",
            symbol_type="fn",
            precedence=5,
            np_fn="np.cos({})",
            latex_str=r"\cos {}",
            cython_id=11,
        )
        sl.add_symbol(
            "exp",
            symbol_type="fn",
            precedence=5,
            np_fn="np.exp({})",
            latex_str=r"e^{{{}}}",
            cython_id=13,
        )
        sl.add_symbol(
            "tan",
            symbol_type="fn",
            precedence=5,
            np_fn="np.tan({})",
            latex_str=r"\tan {}",
            cython_id=12,
        )
        sl.add_symbol(
            "arcsin",
            symbol_type="fn",
            precedence=5,
            np_fn="np.arcsin({})",
            latex_str=r"\arcsin {}",
            cython_id=17,
        )
        sl.add_symbol(
            "arccos",
            symbol_type="fn",
            precedence=5,
            np_fn="np.arccos({})",
            latex_str=r"\arccos {}",
            cython_id=18,
        )
        sl.add_symbol(
            "arctan",
            symbol_type="fn",
            precedence=5,
            np_fn="np.arctan({})",
            latex_str=r"\arctan {}",
            cython_id=19,
        )
        sl.add_symbol(
            "sinh",
            symbol_type="fn",
            precedence=5,
            np_fn="np.sinh({})",
            latex_str=r"\sinh {}",
            cython_id=20,
        )
        sl.add_symbol(
            "cosh",
            symbol_type="fn",
            precedence=5,
            np_fn="np.cosh({})",
            latex_str=r"\cosh {}",
            cython_id=21,
        )
        sl.add_symbol(
            "tanh",
            symbol_type="fn",
            precedence=5,
            np_fn="np.tanh({})",
            latex_str=r"\tanh {}",
            cython_id=22,
        )
        sl.add_symbol(
            "floor",
            symbol_type="fn",
            precedence=5,
            np_fn="np.floor({})",
            latex_str=r"\lfloor {} \rfloor",
            cython_id=23,
        )
        sl.add_symbol(
            "ceil",
            symbol_type="fn",
            precedence=5,
            np_fn="np.ceil({})",
            latex_str=r"\lceil {} \rceil",
            cython_id=24,
        )
        sl.add_symbol(
            "ln",
            symbol_type="fn",
            precedence=5,
            np_fn="np.log({})",
            latex_str=r"\ln {}",
            cython_id=15,
        )
        sl.add_symbol(
            "log",
            symbol_type="fn",
            precedence=5,
            np_fn="np.log10({})",
            latex_str=r"\log_{{10}} {}",
            cython_id=16,
        )
        sl.add_symbol(
            "^-1",
            symbol_type="fn",
            precedence=-1,
            np_fn="1/{}",
            latex_str=r"{}^{{-1}}",
            cython_id=26,
        )
        sl.add_symbol("^2", symbol_type="fn", precedence=-1, np_fn="{}**2", latex_str=r"{}^2", cython_id=27)
        sl.add_symbol("^3", symbol_type="fn", precedence=-1, np_fn="{}**3", latex_str=r"{}^3", cython_id=28)
        sl.add_symbol("^4", symbol_type="fn", precedence=-1, np_fn="{}**4", latex_str=r"{}^4", cython_id=29)
        sl.add_symbol("^5", symbol_type="fn", precedence=-1, np_fn="{}**5", latex_str=r"{}^5", cython_id=30)
        sl.add_symbol(
            "pi",
            symbol_type="lit",
            precedence=5,
            np_fn="np.pi",
            latex_str=r"\pi",
        )
        sl.add_symbol(
            "e",
            symbol_type="lit",
            precedence=5,
            np_fn="np.e",
            latex_str=r"e",
        )
        sl.add_symbol(
            "C",
            symbol_type="const",
            precedence=5,
            np_fn="np.full(X.shape[0], C[{}])",
            latex_str=r"C_{{{}}}",
        )

        if num_variables > 0:
            for i in range(num_variables):
                sl.add_symbol(f"X_{i}", "var", 5, "X[:, {}]".format(i), "X_{{{}}}".format(i))

        return sl

    def to_dict(self) -> dict:
        """
        Serialize the library to a JSON-safe dictionary.

        Examples:
            >>> library = SymbolLibrary.from_symbol_list(["+"], num_variables=1)
            >>> d = library.to_dict()
            >>> d["format_version"]
            1
            >>> d["num_variables"]
            1

        Returns:
            A dictionary suitable for passing to [from_dict][SRToolkit.utils.symbol_library.SymbolLibrary.from_dict].
        """
        return {
            "format_version": 1,
            "type": "SymbolLibrary",
            "symbols": self.symbols,
            "preamble": self.preamble,
            "num_variables": self.num_variables,
        }

    @staticmethod
    def from_dict(d: dict) -> "SymbolLibrary":
        """
        Reconstruct a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] from a dictionary produced by [to_dict][SRToolkit.utils.symbol_library.SymbolLibrary.to_dict].

        Args:
            d: Dictionary representation of the library, as produced by
                [to_dict][SRToolkit.utils.symbol_library.SymbolLibrary.to_dict].

        Returns:
            The reconstructed [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].

        Raises:
            ValueError: If ``d["format_version"]`` is not ``1``.
        """
        if d.get("format_version", 1) != 1:
            raise ValueError(
                f"[SymbolLibrary.from_dict] Unsupported format_version: {d.get('format_version')!r}. Expected 1."
            )
        sl = SymbolLibrary()
        sl.symbols = d["symbols"]
        sl.preamble = d["preamble"]
        sl.num_variables = d["num_variables"]
        return sl

    def __len__(self) -> int:
        """
        Return the number of symbols currently in the library.

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
        r"""
        Return a comma-separated string of all registered token strings.

        Examples:
            >>> library = SymbolLibrary()
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
        r"""
        Return a copy of the library with independent copies of all attributes.

        Examples:
            >>> old_symbols = SymbolLibrary()
            >>> old_symbols.add_symbol("x", "var", 0, "x", "x")
            >>> print(old_symbols)
            x
            >>> new_symbols = copy.copy(old_symbols)
            >>> new_symbols.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
            >>> print(old_symbols)
            x
            >>> print(new_symbols)
            x, sin

        Returns:
            A new [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] instance with deep-copied symbols and preamble.
        """
        sl = SymbolLibrary()
        sl.symbols = copy.deepcopy(self.symbols)
        sl.preamble = copy.deepcopy(self.preamble)
        sl.num_variables = self.num_variables
        return sl

    def __deepcopy__(self, memo=None) -> "SymbolLibrary":
        return self.__copy__()
