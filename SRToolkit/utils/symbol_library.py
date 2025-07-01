"""
This module contains the SymbolLibrary class, which is used for managing symbols and their properties.
"""
import copy
from typing import List, Dict


class SymbolLibrary:
    def __init__(self):
        """
        Initializes an instance of the SymbolLibrary class. This class is used for managing symbols and their
        properties for other functionality in this package.

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

        Attributes:
            symbols : dict
                A dictionary mapping symbols to their properties (type, precedence, numpy function).

        Methods:
            add_symbol(symbol, symbol_type, precedence, np_fn):
                Adds a symbol to the library.
            remove_symbol(symbol):
                Removes a symbol from the library.
            get_type(symbol):
                Retrieves the type of a symbol from the library.
            get_precedence(symbol):
                Returns the precedence of the given symbol.
            get_np_fn(symbol):
                Returns the numpy function corresponding to the given symbol.
            default_symbols():
                Returns a SymbolLibrary with the default symbols.
        """
        self.symbols = dict()
        self.num_variables = 0

    def __len__(self) -> int:
        """
        Returns the number of symbols currently stored in the SymbolLibrary.

        Examples:
             >>> library = SymbolLibrary.default_symbols(5)
             >>> len(library)
             34
             >>> library.add_symbol("a", "lit", 5, "a", "a")
             >>> len(library)
             35

        Returns
            Number of symbols currently stored in the SymbolLibrary.
        """
        return len(self.symbols)

    def __str__(self) -> str:
        r"""
        Returns a string representation of the SymbolLibrary instance.

        This method provides a comma-separated string of all the symbol keys
        currently stored in the SymbolLibrary.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x", "x")
            >>> str(library)
            'x'
            >>> library.add_symbol("sin", "fn", 5, "{} = np.sin({})", r"\sin {}")
            >>> str(library)
            'x, sin'

        Returns:
            A string containing all symbols in the library, separated by commas.
        """
        return ", ".join(self.symbols.keys())

    def __copy__(self) -> "SymbolLibrary":
        r"""
        Creates a copy of the SymbolLibrary instance.

        Examples:
            >>> old_symbols = SymbolLibrary()
            >>> old_symbols.add_symbol("x", "var", 0, "x", "x")
            >>> print(old_symbols)
            x
            >>> new_symbols = copy.copy(old_symbols)
            >>> new_symbols.add_symbol("sin", "fn", 5, "{} = np.sin({})", r"\sin {}")
            >>> print(old_symbols)
            x
            >>> print(new_symbols)
            x, sin

        Returns:
            A copy of the SymbolLibrary instance.
        """
        sl = SymbolLibrary()
        sl.symbols = copy.deepcopy(self.symbols)
        return sl

    def add_symbol(self, symbol: str, symbol_type: str, precedence: int, np_fn: str, latex_str: str = None):
        r"""
        Adds a symbol to the library. A symbol should have a type, precedence, a numpy function, and a LaTeX template associated with it.
        Type "op" should be used for symbols operating on two operands, "fn" for symbols operating on one operand,
        "lit" for constants with a known value (such as pi or e), "const" for constants/parameters without a value that
        need to be optimized, and "var" for variables whose values are provided as input data.

        We recommend you use a single token of "const" type as using multiple might lead to more work, errors, and less
        readability.

        If the argument 'latex_str' is ommited, a default LaTeX template will be generated for the symbol. In case of symbol 'symb', the default template
        will be '{} \text{symb} {}' for an operator,'\text{symb} {}' for a function, and '\text{symb}' otherwise.

        For example, look at the default_symbols function for the SymbolLibrary class.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
            >>> library.add_symbol("C", "const", 5, "C[{}]", r"c_{}")
            >>> library.add_symbol("X_0", "var", 5, "X[:, 0]", r"X_0")
            >>> library.add_symbol("pi", "lit", 5, "np.pi", r"\pi")

        Args:
            symbol: The symbol to be added to the library.
            symbol_type: The type of the symbol, one of "op" (operator), "fn" (function), "lit" (literal), "const" (constant), or "var" (variable).
            precedence: The precedence of the symbol, used to determine the order of operations.
            np_fn: A string representing the numpy function associated with this symbol.
            latex_str: A string that represents how the symbol is written in LaTeX
        """
        if latex_str is None:
            if symbol_type == "var":
                latex_str = f"{{}} \text{{{symbol}}} {{}}"
            elif symbol_type == "fn":
                latex_str = f"\text{{{symbol}}} {{}}"
            else:
                latex_str = f"\text{{{symbol}}}"

        if symbol_type == "var" and (np_fn is None or np_fn == ""):
            np_fn = "X[:, {}]".format(self.num_variables)

        if symbol_type == "var":
            self.num_variables += 1

        self.symbols[symbol] = {
            "symbol": symbol,
            "type": symbol_type,
            "precedence": precedence,
            "np_fn": np_fn,
            "latex_str": latex_str
        }

    def remove_symbol(self, symbol: str):
        """
        Removes a symbol from the library.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> len(library.symbols)
            1
            >>> library.remove_symbol("x")
            >>> len(library.symbols)
            0

        Args:
            symbol: The symbol to be removed from the library.

        Raises:
            KeyError: If the symbol does not exist in the library.
        """
        del self.symbols[symbol]

    def get_type(self, symbol: str) -> str:
        """
        Retrieves the type of a symbol from the library.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_type("x")
            'var'

        Args:
            symbol: The symbol whose type is to be retrieved.

        Returns:
            The type of the symbol if it exists in the library, otherwise an empty string.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["type"]
        else:
            return ""

    def get_precedence(self, symbol: str) -> int:
        """
        Retrieves the precedence of the given symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_precedence("x")
            0

        Args:
            symbol: The symbol whose precedence is to be retrieved.

        Returns:
            The precedence of the symbol if it exists in the library, otherwise -1.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["precedence"]
        else:
            return -1

    def get_np_fn(self, symbol: str) -> str:
        """
        Returns the numpy function corresponding to the given symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.get_np_fn("x")
            'x'

        Args:
            symbol: The symbol to look up.

        Returns:
            The numpy function corresponding to the given symbol, or an empty string if the symbol was not found.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["np_fn"]
        else:
            return ""

    def get_latex_str(self, symbol):
        """
        Returns the LaTeX template for the corresponding symbol.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x", "test")
            >>> library.get_latex_str("x")
            'test'

        Args:
            symbol: The symbol to look up.

        Returns:
            The LaTeX template for the corresponding symbol, or an empty string if the symbol was not found.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["latex_str"]
        else:
            return ""

    def get_symbols_of_type(self, symbol_type: str) -> List[str]:
        """
        Returns a list of symbols with the requested type ("op", "fn", "var", "const", "lit").

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("y", "var", 0, "y")
            >>> library.get_symbols_of_type("var")
            ['x', 'y']

        Args:
            symbol_type: Type of symbols you want to get.

        Returns:
            A list of symbols with the requested type
        """
        symbols = list()
        for symbol in self.symbols.keys():
            if self.get_type(symbol) == symbol_type:
                symbols.append(symbol)

        return symbols

    def symbols2index(self) -> Dict[str, int]:
        """
        Generates a dictionary mapping symbols to their indices in the symbol list.

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
            A dictionary mapping symbols to their indices in the symbol list.
        """
        return {s:i for i, s in enumerate(self.symbols.keys())}

    @staticmethod
    def from_symbol_list(symbols: List[str], num_variables=25):
        """
        Creates an instance of SymbolLibrary from a list of symbols and number of variables. The list of currently
        supported symbols (by default) can be seen in the SymbolLibrary.default_symbols() function.

        Examples:
            >>> library = SymbolLibrary().from_symbol_list(["+", "*", "C"], num_variables=2)
            >>> len(library.symbols)
            5

        Args:
            symbols: List of symbols you want.
            num_variables: Number of variables you want.

        Returns:
            An instance of SymbolLibrary
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
        Creates a SymbolLibrary instance populated with default mathematical symbols.

        This method adds a set of predefined symbols to a SymbolLibrary instance,
        representing common mathematical operations, functions, constants, and optional
        variables. The symbols include basic arithmetic operations, trigonometric and
        exponential functions, and mathematical constants like pi and e.

        If num_variables is greater than 0, it adds variables labeled 'X_0' to 'X_{num_variables-1}', each
         associated with a column in a data array X.

        By default, we currently support the following symbols: "+", "-", "*", "/", "^", "u-" (unary minus), "sqrt",
        "sin", "cos", "exp", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "floor", "ceil", "ln", "log",
        "^-1", "^2", "^3", "^4", "^5", "pi", "e", "C" (unknown constant).

        Notes: The variables in the default_symbols function are added in the predefined order,
        which is the same order as the columns in the data array X.

        Examples:
            >>> library = SymbolLibrary.default_symbols()
            >>> len(library.symbols)
            54

        Args:
            num_variables: The number of variables to add to the library (default is 25).

        Returns:
            A SymbolLibrary instance populated with default mathematical symbols.
        """
        sl = SymbolLibrary()
        sl.add_symbol("+", symbol_type="op", precedence=0, np_fn="{} = {} + {}", latex_str=r"{} + {}")
        sl.add_symbol("-", symbol_type="op", precedence=0, np_fn="{} = {} - {}", latex_str=r"{} - {}")
        sl.add_symbol("*", symbol_type="op", precedence=1, np_fn="{} = {} * {}", latex_str=r"{} \cdot {}")
        sl.add_symbol("/", symbol_type="op", precedence=1, np_fn="{} = {} / {}", latex_str=r"\frac{{{}}}{{{}}}")
        sl.add_symbol("^", symbol_type="op", precedence=2, np_fn="{} = np.power({},{})", latex_str=r"{}^{{{}}}")
        sl.add_symbol("u-", symbol_type="fn", precedence=5, np_fn="{} = -{}", latex_str=r"- {}")
        sl.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})", latex_str=r"\sqrt {{{}}}")
        sl.add_symbol("sin", symbol_type="fn", precedence=5, np_fn="{} = np.sin({})", latex_str=r"\sin {}")
        sl.add_symbol("cos", symbol_type="fn", precedence=5, np_fn="{} = np.cos({})", latex_str=r"\cos {}")
        sl.add_symbol("exp", symbol_type="fn", precedence=5, np_fn="{} = np.exp({})", latex_str=r"e^{{{}}}")
        sl.add_symbol("tan", symbol_type="fn", precedence=5, np_fn="{} = np.tan({})", latex_str=r"\tan {}")
        sl.add_symbol("arcsin", symbol_type="fn", precedence=5, np_fn="{} = np.arcsin({})", latex_str=r"\arcsin {}")
        sl.add_symbol("arccos", symbol_type="fn", precedence=5, np_fn="{} = np.arccos({})", latex_str=r"\arccos {}")
        sl.add_symbol("arctan", symbol_type="fn", precedence=5, np_fn="{} = np.arctan({})", latex_str=r"\arctan {}")
        sl.add_symbol("sinh", symbol_type="fn", precedence=5, np_fn="{} = np.sinh({})", latex_str=r"\sinh {}")
        sl.add_symbol("cosh", symbol_type="fn", precedence=5, np_fn="{} = np.cosh({})", latex_str=r"\cosh {}")
        sl.add_symbol("tanh", symbol_type="fn", precedence=5, np_fn="{} = np.tanh({})", latex_str=r"\tanh {}")
        sl.add_symbol("floor", symbol_type="fn", precedence=5, np_fn="{} = np.floor({})", latex_str=r"\lfloor {} \rfloor")
        sl.add_symbol("ceil", symbol_type="fn", precedence=5, np_fn="{} = np.ceil({})", latex_str=r"\lceil {} \rceil")
        sl.add_symbol("ln", symbol_type="fn", precedence=5, np_fn="{} = np.log({})", latex_str=r"\ln {}")
        sl.add_symbol("log", symbol_type="fn", precedence=5, np_fn="{} = np.log10({})", latex_str=r"\log_{{10}} {}")
        sl.add_symbol("^-1", symbol_type="fn", precedence=-1, np_fn="{} = 1/{}", latex_str=r"{}^{{-1}}")
        sl.add_symbol("^2", symbol_type="fn", precedence=-1, np_fn="{} = {}**2", latex_str=r"{}^2")
        sl.add_symbol("^3", symbol_type="fn", precedence=-1, np_fn="{} = {}**3", latex_str=r"{}^3")
        sl.add_symbol("^4", symbol_type="fn", precedence=-1, np_fn="{} = {}**4", latex_str=r"{}^4")
        sl.add_symbol("^5", symbol_type="fn", precedence=-1, np_fn="{} = {}**5", latex_str=r"{}^5")
        sl.add_symbol("pi", symbol_type="lit", precedence=5, np_fn="np.full(X.shape[0], np.pi)", latex_str=r"\pi")
        sl.add_symbol("e", symbol_type="lit", precedence=5, np_fn="np.full(X.shape[0], np.e)", latex_str=r"e")
        sl.add_symbol("C", symbol_type="const", precedence=5, np_fn="np.full(X.shape[0], C[{}])", latex_str=r"C_{{{}}}")

        if num_variables > 0:
            for i in range(num_variables):
                sl.add_symbol(f"X_{i}", "var", 5, "X[:, {}]".format(i), "X_{{{}}}".format(i))

        return sl