"""
This module contains the SymbolLibrary class, which is used for managing symbols and their properties.
"""

class SymbolLibrary:
    def __init__(self):
        """
        Initializes an instance of the SymbolLibrary class. This class is used for managing symbols and their
        properties for other functionality in this package.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
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

    def add_symbol(self, symbol: str, symbol_type: str, precedence: int, np_fn: str):
        """
        Adds a symbol to the library. A symbol should have a type, precedence, and numpy function associated with it.
        Type "op" should be used for symbols operating on two operands, "fn" for symbols operating on one operand,
        "lit" for constants with a known value (such as pi or e), "const" for constants/parameters without a value that
        need to be optimized, and "var" for variables whose values are provided as input data.

        For example, look at the default_symbols function for the SymbolLibrary class.

        Examples:
            >>> library = SymbolLibrary()
            >>> library.add_symbol("x", "var", 0, "x")
            >>> library.add_symbol("sin", "fn", 5, "np.sin({})")
            >>> library.add_symbol("C", "const", 5, "C[{}]")
            >>> library.add_symbol("X", "var", 5, "X[:, 0]")
            >>> library.add_symbol("pi", "lit", 5, "np.pi")

        Args:
            symbol: The symbol to be added to the library.
            symbol_type: The type of the symbol, one of "op" (operator), "fn" (function), "lit" (literal), "const" (constant), or "var" (variable).
            precedence: The precedence of the symbol, used to determine the order of operations.
            np_fn: A string representing the numpy function associated with this symbol.
        """
        self.symbols[symbol] = {
            "symbol": symbol,
            "type": symbol_type,
            "precedence": precedence,
            "np_fn": np_fn,
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

        Note: The variables in the default_symbols function are added in the predefined order,
        which is the same order as the columns in the data array X.

        Examples:
            >>> library = SymbolLibrary.default_symbols()
            >>> len(library.symbols)
            44

        Args:
            num_variables: The number of variables to add to the library (default is 25).

        Returns:
            A SymbolLibrary instance populated with default mathematical symbols.
        """
        sl = SymbolLibrary()
        sl.add_symbol("+", symbol_type="op", precedence=0, np_fn="{} = {} + {}")
        sl.add_symbol("-", symbol_type="op", precedence=0, np_fn="{} = {} - {}")
        sl.add_symbol("*", symbol_type="op", precedence=1, np_fn="{} = {} * {}")
        sl.add_symbol("/", symbol_type="op", precedence=1, np_fn="{} = {} / {}")
        sl.add_symbol("^", symbol_type="op", precedence=2, np_fn="{} = np.pow({},{})")
        sl.add_symbol("u-", symbol_type="fn", precedence=5, np_fn="{} = -{}")
        sl.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})")
        sl.add_symbol("sin", symbol_type="fn", precedence=5, np_fn="{} = np.sin({})")
        sl.add_symbol("cos", symbol_type="fn", precedence=5, np_fn="{} = np.cos({})")
        sl.add_symbol("exp", symbol_type="fn", precedence=5, np_fn="{} = np.exp({})")
        sl.add_symbol("log", symbol_type="fn", precedence=5, np_fn="{} = np.log({})")
        sl.add_symbol("^-1", symbol_type="fn", precedence=-1, np_fn="{} = 1/{}")
        sl.add_symbol("^2", symbol_type="fn", precedence=-1, np_fn="{} = {}**2")
        sl.add_symbol("^3", symbol_type="fn", precedence=-1, np_fn="{} = {}**3")
        sl.add_symbol("^4", symbol_type="fn", precedence=-1, np_fn="{} = {}**4")
        sl.add_symbol("^5", symbol_type="fn", precedence=-1, np_fn="{} = {}**5")
        sl.add_symbol("pi", symbol_type="lit", precedence=5, np_fn="np.pi")
        sl.add_symbol("e", symbol_type="lit", precedence=5, np_fn="np.e")
        sl.add_symbol("C", symbol_type="const", precedence=5, np_fn="C[{}]")

        if num_variables > 0:
            for i in range(num_variables):
                sl.add_symbol(f"X_{i}", "var", 5, "X[:, {}]".format(i))

        return sl