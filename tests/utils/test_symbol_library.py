import copy

import pytest

from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestSymbolLibraryInit:
    def test_empty_init(self):
        lib = SymbolLibrary()
        assert lib.symbols == {}
        assert lib.num_variables == 0
        assert lib.preamble == ["import numpy as np"]

    def test_custom_preamble(self):
        lib = SymbolLibrary(preamble=["import math"])
        assert lib.preamble == ["import math"]

    def test_with_symbols_and_variables(self):
        lib = SymbolLibrary(["+", "*"], num_variables=2)
        assert "+" in lib.symbols
        assert "*" in lib.symbols
        assert "X_0" in lib.symbols
        assert "X_1" in lib.symbols
        assert lib.num_variables == 2

    def test_only_num_variables(self):
        lib = SymbolLibrary(num_variables=2)
        assert "X_0" in lib.symbols
        assert "X_1" in lib.symbols
        assert lib.num_variables == 2
        assert all(lib.symbols[s]["type"] == "var" for s in lib.symbols)

    def test_unknown_symbols_silently_dropped(self):
        lib = SymbolLibrary(["not_a_real_symbol"], num_variables=0)
        assert len(lib.symbols) == 0

    def test_empty_list_produces_empty_library(self):
        lib = SymbolLibrary([])
        assert lib.symbols == {}
        assert lib.num_variables == 0


class TestSymbolLibraryAddSymbol:
    def test_invalid_type_raises(self):
        lib = SymbolLibrary()
        with pytest.raises(ValueError, match="Invalid symbol type"):
            lib.add_symbol("x", "invalid", 0, "x")

    def test_auto_latex_op(self):
        lib = SymbolLibrary()
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.symbols["+"]["latex_str"] == r"{} \text{{+}} {}"

    def test_auto_latex_fn(self):
        lib = SymbolLibrary()
        lib.add_symbol("sin", "fn", 5, "np.sin({})")
        assert lib.symbols["sin"]["latex_str"] == r"\text{{sin}} {}"

    def test_auto_latex_other(self):
        lib = SymbolLibrary()
        lib.add_symbol("pi", "lit", 5, "np.pi")
        assert lib.symbols["pi"]["latex_str"] == r"\text{{pi}}"

    def test_explicit_latex_stored(self):
        lib = SymbolLibrary()
        lib.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
        s = lib.symbols["sin"]
        assert s["symbol"] == "sin"
        assert s["type"] == "fn"
        assert s["precedence"] == 5
        assert s["np_fn"] == "np.sin({})"
        assert s["latex_str"] == r"\sin {}"

    def test_var_auto_np_fn(self):
        lib = SymbolLibrary()
        lib.add_symbol("X_0", "var", 5, "")
        assert lib.symbols["X_0"]["np_fn"] == "X[:, 0]"
        lib.add_symbol("X_1", "var", 5, None)
        assert lib.symbols["X_1"]["np_fn"] == "X[:, 1]"

    def test_var_increments_num_variables(self):
        lib = SymbolLibrary()
        assert lib.num_variables == 0
        lib.add_symbol("X_0", "var", 5, "X[:, 0]")
        assert lib.num_variables == 1
        lib.add_symbol("X_1", "var", 5, "X[:, 1]")
        assert lib.num_variables == 2

    def test_duplicate_token_overwrites(self):
        lib = SymbolLibrary()
        lib.add_symbol("+", "op", 0, "first")
        lib.add_symbol("+", "op", 1, "second")
        assert lib.symbols["+"]["np_fn"] == "second"
        assert lib.symbols["+"]["precedence"] == 1
        assert len(lib) == 1


class TestSymbolLibraryRemoveSymbol:
    def test_removes_existing_symbol(self):
        lib = SymbolLibrary()
        lib.add_symbol("x", "var", 0, "x")
        assert "x" in lib.symbols
        lib.remove_symbol("x")
        assert "x" not in lib.symbols

    def test_missing_symbol_raises(self):
        lib = SymbolLibrary()
        with pytest.raises(KeyError):
            lib.remove_symbol("nonexistent")


class TestSymbolLibraryGetType:
    def test_returns_type_for_known_symbol(self):
        lib = SymbolLibrary()
        lib.add_symbol("x", "var", 0, "x")
        assert lib.get_type("x") == "var"

    def test_returns_empty_string_for_unknown_symbol(self):
        lib = SymbolLibrary()
        assert lib.get_type("unknown") == ""


class TestSymbolLibraryGetPrecedence:
    def test_returns_precedence_for_known_symbol(self):
        lib = SymbolLibrary()
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.get_precedence("+") == 0

    def test_returns_minus_one_for_unknown_symbol(self):
        lib = SymbolLibrary()
        assert lib.get_precedence("unknown") == -1


class TestSymbolLibraryGetNpFn:
    def test_returns_np_fn_for_known_symbol(self):
        lib = SymbolLibrary()
        lib.add_symbol("sin", "fn", 5, "np.sin({})")
        assert lib.get_np_fn("sin") == "np.sin({})"

    def test_returns_empty_string_for_unknown_symbol(self):
        lib = SymbolLibrary()
        assert lib.get_np_fn("unknown") == ""


class TestSymbolLibraryGetLatexStr:
    def test_returns_latex_str_for_known_symbol(self):
        lib = SymbolLibrary()
        lib.add_symbol("sin", "fn", 5, "np.sin({})", r"\sin {}")
        assert lib.get_latex_str("sin") == r"\sin {}"

    def test_returns_empty_string_for_unknown_symbol(self):
        lib = SymbolLibrary()
        assert lib.get_latex_str("unknown") == ""


class TestSymbolLibraryGetSymbolsOfType:
    def test_returns_matching_symbols(self):
        lib = SymbolLibrary()
        lib.add_symbol("X_0", "var", 5, "X[:, 0]")
        lib.add_symbol("X_1", "var", 5, "X[:, 1]")
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.get_symbols_of_type("var") == ["X_0", "X_1"]
        assert lib.get_symbols_of_type("op") == ["+"]

    def test_returns_empty_list_when_no_match(self):
        lib = SymbolLibrary()
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.get_symbols_of_type("fn") == []


class TestSymbolLibrarySymbols2Index:
    def test_returns_correct_index_mapping(self):
        lib = SymbolLibrary()
        lib.add_symbol("x", "var", 0, "x")
        lib.add_symbol("y", "var", 0, "y")
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.symbols2index() == {"x": 0, "y": 1, "+": 2}
        lib.remove_symbol("x")
        assert lib.symbols2index() == {"y": 0, "+": 1}

    def test_empty_library_returns_empty_dict(self):
        lib = SymbolLibrary()
        assert lib.symbols2index() == {}


class TestSymbolLibraryFromSymbolList:
    def test_filters_to_requested_symbols(self):
        lib = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=0)
        assert "+" in lib.symbols
        assert "*" in lib.symbols
        assert "sin" not in lib.symbols
        assert "cos" not in lib.symbols

    def test_variables_added_by_num_variables(self):
        lib = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        assert "X_0" in lib.symbols
        assert "X_1" in lib.symbols
        assert "X_2" not in lib.symbols

    def test_variable_in_symbols_list_not_duplicated(self):
        lib = SymbolLibrary.from_symbol_list(["X_0"], num_variables=1)
        assert list(lib.symbols.keys()).count("X_0") == 1

    def test_default_num_variables_is_25(self):
        lib = SymbolLibrary.from_symbol_list([])
        assert len(lib.get_symbols_of_type("var")) == 25


class TestSymbolLibraryDefaultSymbols:
    def test_default_symbol_count(self):
        lib = SymbolLibrary.default_symbols()
        assert len(lib) == 54  # 5 ops + 21 fns + 2 lits + 1 const + 25 vars

    def test_num_variables_controls_variable_count(self):
        lib = SymbolLibrary.default_symbols(num_variables=0)
        assert lib.get_symbols_of_type("var") == []
        lib3 = SymbolLibrary.default_symbols(num_variables=3)
        assert lib3.get_symbols_of_type("var") == ["X_0", "X_1", "X_2"]


class TestSymbolLibraryToDict:
    def test_serializes_all_fields(self):
        lib = SymbolLibrary(["+"], num_variables=1)
        d = lib.to_dict()
        assert d["format_version"] == 1
        assert d["type"] == "SymbolLibrary"
        assert d["symbols"] == lib.symbols
        assert d["preamble"] == lib.preamble
        assert d["num_variables"] == 1


class TestSymbolLibraryFromDict:
    def test_round_trip(self):
        original = SymbolLibrary(["+", "sin"], num_variables=2)
        restored = SymbolLibrary.from_dict(original.to_dict())
        assert restored.symbols == original.symbols
        assert restored.preamble == original.preamble
        assert restored.num_variables == original.num_variables

    def test_unsupported_format_version_raises(self):
        d = SymbolLibrary(["+"], num_variables=1).to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            SymbolLibrary.from_dict(d)

    def test_missing_format_version_defaults_to_1(self):
        d = SymbolLibrary(["+"], num_variables=1).to_dict()
        del d["format_version"]
        lib = SymbolLibrary.from_dict(d)
        assert "+" in lib.symbols


class TestSymbolLibraryLen:
    def test_returns_symbol_count(self):
        lib = SymbolLibrary()
        assert len(lib) == 0
        lib.add_symbol("x", "var", 0, "x")
        assert len(lib) == 1
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert len(lib) == 2
        lib.remove_symbol("x")
        assert len(lib) == 1


class TestSymbolLibraryStr:
    def test_returns_comma_separated_tokens(self):
        lib = SymbolLibrary()
        lib.add_symbol("x", "var", 0, "x")
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert str(lib) == "x, +"

    def test_empty_library_returns_empty_string(self):
        assert str(SymbolLibrary()) == ""


class TestSymbolLibraryCopy:
    def test_copy_is_independent(self):
        original = SymbolLibrary(["+"], num_variables=1)
        copied = copy.copy(original)
        copied.add_symbol("sin", "fn", 5, "np.sin({})")
        assert "sin" not in original.symbols
        assert "sin" in copied.symbols


class TestSymbolLibraryDeepcopy:
    def test_deepcopy_is_independent(self):
        original = SymbolLibrary(["+"], num_variables=1)
        deep = copy.deepcopy(original)
        deep.add_symbol("sin", "fn", 5, "np.sin({})")
        assert "sin" not in original.symbols
        assert "sin" in deep.symbols


class TestSymbolLibraryContextManager:
    def test_basic_context_manager(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        with sl:
            tree_implicit = tokens_to_tree(["X_0", "+", "X_1"])
        tree_explicit = tokens_to_tree(["X_0", "+", "X_1"], sl)
        assert len(tree_implicit) == len(tree_explicit)
        assert tree_implicit.symbol == tree_explicit.symbol

    def test_nested_context_managers(self):
        outer = SymbolLibrary.default_symbols(num_variables=2)
        inner = SymbolLibrary.default_symbols(num_variables=5)
        with outer:
            assert SymbolLibrary.get_active() is outer
            with inner:
                assert SymbolLibrary.get_active() is inner
            assert SymbolLibrary.get_active() is outer

    def test_no_active_context_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            SymbolLibrary.get_active()

    def test_explicit_parameter_takes_precedence_over_context(self):
        context_sl = SymbolLibrary.default_symbols(num_variables=1)
        explicit_sl = SymbolLibrary.default_symbols(num_variables=3)
        with context_sl:
            tree = tokens_to_tree(["X_0", "+", "X_2"], explicit_sl)
        assert tree.symbol == "+"

    def test_exception_inside_with_still_resets_context(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        with pytest.raises(ValueError):
            with sl:
                raise ValueError("deliberate")
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            SymbolLibrary.get_active()

    def test_context_manager_without_as_clause(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        with sl:
            assert SymbolLibrary.get_active() is sl
        with pytest.raises(RuntimeError):
            SymbolLibrary.get_active()


class TestSymbolLibrarySetDefault:
    def setup_method(self):
        # Always start each test with no default set.
        SymbolLibrary.set_default(None)

    def teardown_method(self):
        SymbolLibrary.set_default(None)

    def test_set_default_returned_by_get_active(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        SymbolLibrary.set_default(sl)
        assert SymbolLibrary.get_active() is sl

    def test_set_default_returned_by_get_or_default(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        SymbolLibrary.set_default(sl)
        assert SymbolLibrary.get_or_default() is sl

    def test_context_manager_overrides_default(self):
        default_sl = SymbolLibrary.default_symbols(num_variables=1)
        context_sl = SymbolLibrary.default_symbols(num_variables=3)
        SymbolLibrary.set_default(default_sl)
        with context_sl:
            assert SymbolLibrary.get_active() is context_sl
        assert SymbolLibrary.get_active() is default_sl

    def test_set_default_none_clears_default(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        SymbolLibrary.set_default(sl)
        SymbolLibrary.set_default(None)
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            SymbolLibrary.get_active()

    def test_get_or_default_falls_back_to_default_symbols_when_nothing_set(self):
        result = SymbolLibrary.get_or_default()
        assert isinstance(result, SymbolLibrary)
        assert len(result) > 0  # default_symbols() is non-empty

    def test_no_active_context_or_default_still_raises_for_get_active(self):
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            SymbolLibrary.get_active()
