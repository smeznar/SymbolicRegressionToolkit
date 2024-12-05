class SymbolLibrary:
    def __init__(self):
        """
        Initializes an instance of the SymbolLibrary class. This class is used for managing symbols and their
        properties for other functionality in this package.
        """
        self.symbols = dict()

    def add_symbol(self, symbol: str, symbol_type: str, precedence: int, np_fn: str):
        """
        Adds a symbol to the library. A symbol should have a type, precedence, and numpy function associated with it.
        Type "op" should be used for symbols operating on two operands, "fn" for symbols operating on one operand,
        "lit" for constants with a known value (such as pi or e), "const" for constants/parameters without a value that
        need to be optimized, and "var" for variables whose values are provided as input data.

        For example, look at the default_symbols function for the SymbolLibrary class.

        Parameters
        ----------
        symbol : str
            The symbol to be added to the library.
        symbol_type : str
            The type of the symbol, one of "op" (operator), "fn" (function), "lit" (literal), "const" (constant), or
            "var" (variable).
        precedence : int
            The precedence of the symbol, used to determine the order of operations.
        np_fn : str
            A string representing the numpy function associated with this symbol.
        """
        self.symbols[symbol] = {"symbol": symbol, "type": symbol_type, "precedence": precedence, "np_fn": np_fn}

    def remove_symbol(self, symbol: str):
        """
        Removes a symbol from the library.

        Parameters
        ----------
        symbol : str
            The symbol to be removed from the library.

        Raises
        ------
        KeyError
            If the symbol does not exist in the library.
        """
        del self.symbols[symbol]

    def get_type(self, symbol: str) -> str:
        """
        Retrieves the type of a symbol from the library.

        Parameters
        ----------
        symbol : str
            The symbol whose type is to be retrieved.

        Returns
        -------
        str
            The type of the symbol if it exists in the library, otherwise an empty string.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["type"]
        else:
            return ""

    def get_precedence(self, symbol: str) -> int:
        """
        Returns the precedence of the given symbol.

        Parameters
        ----------
        symbol : str
            The symbol whose precedence is to be retrieved.

        Returns
        -------
        int
            The precedence of the symbol if it exists in the library, otherwise -1.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["precedence"]
        else:
            return -1

    def get_np_fn(self, symbol: str) -> str:
        """
        Returns the numpy function corresponding to the given symbol.

        Parameters
        ----------
        symbol : str
            The symbol to look up.

        Returns
        -------
        str
            The numpy function corresponding to the given symbol, or an empty string if the symbol was not found.
        """
        if symbol in self.symbols:
            return self.symbols[symbol]["np_fn"]
        else:
            return ""


    @staticmethod
    def default_symbols(add_variables: bool=True) -> "SymbolLibrary":
        """
        Creates a SymbolLibrary instance populated with default mathematical symbols.

        This method adds a set of predefined symbols to a SymbolLibrary instance,
        representing common mathematical operations, functions, constants, and optional
        variables. The symbols include basic arithmetic operations, trigonometric and
        exponential functions, and mathematical constants like pi and e.

        Args:
            add_variables (bool): If True, adds variables labeled 'A' to 'Z', along
                                  with 'Č', 'Š', 'Ž', each associated with a column
                                  in a data array X. By default, this is set to True.
                                  Character 'C' is excluded from this list, as it is
                                  reserved for constants.

        Returns:
            SymbolLibrary: An instance of SymbolLibrary with the default symbols.
        """
        sl = SymbolLibrary()
        sl.add_symbol("+",    symbol_type="op", precedence=0, np_fn="{} = {} + {}")
        sl.add_symbol("-",    symbol_type="op", precedence=0, np_fn="{} = {} - {}")
        sl.add_symbol("*",    symbol_type="op", precedence=1, np_fn="{} = {} * {}")
        sl.add_symbol("/",    symbol_type="op", precedence=1, np_fn="{} = {} / {}")
        sl.add_symbol("^",    symbol_type="op", precedence=2, np_fn="{} = np.pow({},{})")
        sl.add_symbol("u-",   symbol_type="fn", precedence=5, np_fn="{} = -{}")
        sl.add_symbol("sqrt", symbol_type="fn", precedence=5, np_fn="{} = np.sqrt({})")
        sl.add_symbol("sin",  symbol_type="fn", precedence=5, np_fn="{} = np.sin({})")
        sl.add_symbol("cos",  symbol_type="fn", precedence=5, np_fn="{} = np.cos({})")
        sl.add_symbol("exp",  symbol_type="fn", precedence=5, np_fn="{} = np.exp({})")
        sl.add_symbol("log",  symbol_type="fn", precedence=5, np_fn="{} = np.log({})")
        sl.add_symbol("^-1",  symbol_type="fn", precedence=-1, np_fn="{} = 1/{}")
        sl.add_symbol("^2",   symbol_type="fn", precedence=-1, np_fn="{} = {}**2")
        sl.add_symbol("^3",   symbol_type="fn", precedence=-1, np_fn="{} = {}**3")
        sl.add_symbol("^4",   symbol_type="fn", precedence=-1, np_fn="{} = {}**4")
        sl.add_symbol("^5",   symbol_type="fn", precedence=-1, np_fn="{} = {}**5")
        sl.add_symbol("pi",   symbol_type="lit", precedence=5, np_fn="np.pi")
        sl.add_symbol("e",    symbol_type="lit", precedence=5, np_fn="np.e")
        sl.add_symbol("C",    symbol_type="const", precedence=5, np_fn="C[{}]")

        if add_variables:
            for i, char in enumerate('ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'):
                sl.add_symbol(char, "var", 5, "X[:, {}]".format(i))

        return sl