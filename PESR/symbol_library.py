class SymbolLibrary:
    def __init__(self):
        self.symbols = dict()

    def add_symbol(self, symbol, symbol_type, precedence, np_fn):
        self.symbols[symbol] = {"symbol": symbol, "type": symbol_type, "precedence": precedence, "np_fn": np_fn}

    def remove_symbol(self, symbol):
        del self.symbols[symbol]

    def get_type(self, symbol: str):
        if symbol in self.symbols:
            return self.symbols[symbol]["type"]
        else:
            return ""

    def get_precedence(self, symbol: str):
        if symbol in self.symbols:
            return self.symbols[symbol]["precedence"]
        else:
            return -1

    def get_np_fn(self, symbol: str):
        if symbol in self.symbols:
            return self.symbols[symbol]["np_fn"]
        else:
            return ""


    @staticmethod
    def default_symbols(add_variables=True):
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