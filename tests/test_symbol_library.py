"""Tests for SRToolkit.utils.symbol_library."""

import copy

import pytest

from SRToolkit.utils.symbol_library import SymbolLibrary


class TestSymbolLibraryInit:
    def test_empty_init(self):
        sl = SymbolLibrary()
        assert len(sl) == 0
        assert sl.num_variables == 0
        assert sl.preamble == ["import numpy as np"]

    def test_init_with_symbols(self):
        sl = SymbolLibrary(["+", "*", "sin"], num_variables=2)
        assert len(sl) == 5  # 3 operators + 2 variables
        assert sl.num_variables == 2

    def test_custom_preamble(self):
        preamble = ["import numpy as np", "import scipy.special as sp"]
        sl = SymbolLibrary(preamble=preamble)
        assert sl.preamble == preamble


class TestSymbolLibraryAddRemove:
    def test_add_symbol(self):
        sl = SymbolLibrary()
        sl.add_symbol("x", "var", 0, "X[:, 0]", "x")
        assert sl.get_type("x") == "var"
        assert sl.get_precedence("x") == 0
        assert sl.get_np_fn("x") == "X[:, 0]"
        assert sl.get_latex_str("x") == "x"

    def test_add_var_increments_counter(self):
        sl = SymbolLibrary()
        assert sl.num_variables == 0
        sl.add_symbol("X_0", "var", 5, "X[:, 0]")
        assert sl.num_variables == 1
        sl.add_symbol("X_1", "var", 5, "X[:, 1]")
        assert sl.num_variables == 2

    def test_remove_symbol(self):
        sl = SymbolLibrary()
        sl.add_symbol("x", "var", 0, "x")
        assert "x" in sl.symbols
        sl.remove_symbol("x")
        assert "x" not in sl.symbols

    def test_remove_nonexistent_raises(self):
        sl = SymbolLibrary()
        with pytest.raises(Exception):
            sl.remove_symbol("nonexistent")

    def test_get_type_nonexistent(self):
        sl = SymbolLibrary()
        assert sl.get_type("nonexistent") == ""


class TestSymbolLibraryDefaultSymbols:
    def test_default_symbols_has_expected_types(self):
        sl = SymbolLibrary.default_symbols(2)
        operators = sl.get_symbols_of_type("op")
        functions = sl.get_symbols_of_type("fn")
        variables = sl.get_symbols_of_type("var")
        constants = sl.get_symbols_of_type("const")
        literals = sl.get_symbols_of_type("lit")

        assert "+" in operators
        assert "*" in operators
        assert "sin" in functions
        assert "X_0" in variables
        assert "X_1" in variables
        assert "C" in constants
        assert "pi" in literals

    def test_default_symbols_variable_count(self):
        sl = SymbolLibrary.default_symbols(5)
        variables = sl.get_symbols_of_type("var")
        assert len(variables) == 5
        assert all(f"X_{i}" in variables for i in range(5))


class TestSymbolLibrarySerialization:
    def test_to_dict_round_trip(self):
        sl = SymbolLibrary.default_symbols(3)
        sl.add_symbol("custom_fn", "fn", 5, "np.custom({})")
        d = sl.to_dict()
        restored = SymbolLibrary.from_dict(d)

        assert restored.symbols == sl.symbols
        assert restored.preamble == sl.preamble
        assert restored.num_variables == sl.num_variables

    def test_to_dict_format_version(self):
        sl = SymbolLibrary.default_symbols(2)
        d = sl.to_dict()
        assert d["format_version"] == 1
        assert d["type"] == "SymbolLibrary"

    def test_from_dict_bad_format_version(self):
        sl = SymbolLibrary.default_symbols(2)
        d = sl.to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError, match="Unsupported format_version"):
            SymbolLibrary.from_dict(d)


class TestSymbolLibraryCopy:
    def test_copy_independence(self):
        sl = SymbolLibrary()
        sl.add_symbol("x", "var", 0, "x")
        sl_copy = copy.copy(sl)

        sl_copy.add_symbol("sin", "fn", 5, "np.sin({})")

        assert "sin" not in sl.symbols
        assert "sin" in sl_copy.symbols
        assert sl.num_variables == sl_copy.num_variables

    def test_copy_preamble_independence(self):
        sl = SymbolLibrary(preamble=["import numpy as np"])
        sl_copy = copy.copy(sl)

        sl_copy.preamble.append("import scipy as sp")

        assert len(sl.preamble) == 1
        assert len(sl_copy.preamble) == 2

    def test_copy_is_not_same_object(self):
        sl = SymbolLibrary.default_symbols(2)
        sl_copy = copy.copy(sl)
        assert sl is not sl_copy
        assert sl.symbols is not sl_copy.symbols
        assert sl.preamble is not sl_copy.preamble


class TestSymbolLibraryStr:
    def test_str_representation(self):
        sl = SymbolLibrary()
        sl.add_symbol("x", "var", 0, "x")
        assert str(sl) == "x"
        sl.add_symbol("sin", "fn", 5, "np.sin({})")
        assert str(sl) == "x, sin"


class TestSymbolLibraryFromSymbolList:
    def test_from_symbol_list(self):
        sl = SymbolLibrary.from_symbol_list(["+", "*", "sin"], num_variables=2)
        assert sl.get_type("+") == "op"
        assert sl.get_type("sin") == "fn"
        assert sl.get_type("X_0") == "var"
        assert len(sl.get_symbols_of_type("var")) == 2
