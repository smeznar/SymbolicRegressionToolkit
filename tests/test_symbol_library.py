"""Tests for SRToolkit.utils.symbol_library — SymbolLibrary."""

import copy

import pytest

from SRToolkit.utils.symbol_library import SymbolLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty() -> SymbolLibrary:
    return SymbolLibrary()


def _mixed() -> SymbolLibrary:
    """Library with one op, one fn, one var — used across multiple groups."""
    sl = SymbolLibrary()
    sl.add_symbol("+", "op", 0, "{} = {} + {}", r"{} + {}")
    sl.add_symbol("sin", "fn", 5, "{} = np.sin({})", r"\sin {}")
    sl.add_symbol("X_0", "var", 5, "X[:, 0]", r"X_{0}")
    return sl


# ---------------------------------------------------------------------------
# Group 1 — __init__ / construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_no_args_empty_symbols(self):
        sl = SymbolLibrary()
        assert sl.symbols == {}

    def test_no_args_num_variables_zero(self):
        sl = SymbolLibrary()
        assert sl.num_variables == 0

    def test_no_args_default_preamble(self):
        sl = SymbolLibrary()
        assert sl.preamble == ["import numpy as np"]

    def test_preamble_none_explicit_gives_default(self):
        sl = SymbolLibrary(preamble=None)
        assert sl.preamble == ["import numpy as np"]

    def test_custom_preamble_stored(self):
        sl = SymbolLibrary(preamble=["import torch"])
        assert sl.preamble == ["import torch"]

    def test_num_variables_only_adds_var_tokens(self):
        sl = SymbolLibrary(num_variables=3)
        assert set(sl.symbols.keys()) == {"X_0", "X_1", "X_2"}
        assert sl.num_variables == 3

    def test_symbols_and_num_variables(self):
        sl = SymbolLibrary(["+", "sin"], num_variables=2)
        assert set(sl.symbols.keys()) == {"+", "sin", "X_0", "X_1"}
        assert len(sl) == 4

    def test_symbols_only_no_num_variables(self):
        sl = SymbolLibrary(["+"])
        assert set(sl.symbols.keys()) == {"+"}

    def test_empty_list_not_none_still_gives_empty_library(self):
        # symbols=[] is NOT None, so it goes through from_symbol_list — but
        # with no non-variable symbols and num_variables=0 the result is empty.
        sl = SymbolLibrary(symbols=[], num_variables=0)
        assert sl.symbols == {}

    def test_unknown_symbol_name_silently_dropped(self):
        sl = SymbolLibrary(["NOT_A_REAL_TOKEN"], num_variables=0)
        assert sl.symbols == {}


# ---------------------------------------------------------------------------
# Group 2 — add_symbol
# ---------------------------------------------------------------------------


class TestAddSymbol:
    @pytest.mark.parametrize("sym_type", ["op", "fn", "lit", "const", "var"])
    def test_add_all_valid_types_stores_entry(self, sym_type):
        sl = _empty()
        sl.add_symbol("tok", sym_type, 1, "fn_str", "latex_str")
        assert "tok" in sl.symbols
        entry = sl.symbols["tok"]
        assert entry["type"] == sym_type
        assert entry["precedence"] == 1
        assert entry["np_fn"] == "fn_str"
        assert entry["latex_str"] == "latex_str"

    def test_add_var_increments_num_variables(self):
        sl = _empty()
        assert sl.num_variables == 0
        sl.add_symbol("X_0", "var", 5, "X[:, 0]")
        assert sl.num_variables == 1
        sl.add_symbol("X_1", "var", 5, "X[:, 1]")
        assert sl.num_variables == 2

    def test_add_non_var_does_not_increment_num_variables(self):
        sl = _empty()
        sl.add_symbol("+", "op", 0, "{} + {}")
        assert sl.num_variables == 0

    def test_add_var_empty_np_fn_auto_assigned(self):
        sl = _empty()
        sl.add_symbol("X_0", "var", 5, "")
        assert sl.symbols["X_0"]["np_fn"] == "X[:, 0]"

    def test_add_var_none_np_fn_auto_assigned(self):
        sl = _empty()
        sl.add_symbol("X_0", "var", 5, None)
        assert sl.symbols["X_0"]["np_fn"] == "X[:, 0]"

    def test_add_var_auto_np_fn_uses_pre_increment_index(self):
        sl = _empty()
        sl.add_symbol("X_0", "var", 5, "")  # index 0
        sl.add_symbol("X_1", "var", 5, "")  # index 1
        assert sl.symbols["X_0"]["np_fn"] == "X[:, 0]"
        assert sl.symbols["X_1"]["np_fn"] == "X[:, 1]"

    def test_add_op_auto_latex(self):
        sl = _empty()
        sl.add_symbol("+", "op", 0, "fn")
        latex = sl.symbols["+"]["latex_str"]
        assert "\text{+}" in latex
        assert latex.count("{}") == 2

    def test_add_fn_auto_latex_has_one_placeholder(self):
        sl = _empty()
        sl.add_symbol("sin", "fn", 5, "fn")
        latex = sl.symbols["sin"]["latex_str"]
        # auto-latex uses a regular f-string, so \t is a tab character
        assert "\text{sin}" in latex
        assert "{}" in latex

    @pytest.mark.parametrize("sym_type", ["lit", "const"])
    def test_add_lit_const_auto_latex_no_placeholders(self, sym_type):
        sl = _empty()
        sl.add_symbol("pi", sym_type, 5, "fn")
        latex = sl.symbols["pi"]["latex_str"]
        # auto-latex uses a regular f-string, so \t is a tab character
        assert "\text{pi}" in latex

    @pytest.mark.parametrize("bad_type", ["VAR", "operator", "function", "", "Fn", "OP"])
    def test_invalid_symbol_type_raises(self, bad_type):
        sl = _empty()
        with pytest.raises(ValueError, match="Invalid symbol type"):
            sl.add_symbol("tok", bad_type, 0, "fn")

    def test_add_duplicate_token_overwrites(self):
        sl = _empty()
        sl.add_symbol("+", "op", 0, "first")
        sl.add_symbol("+", "op", 1, "second")
        assert sl.symbols["+"]["np_fn"] == "second"
        assert sl.symbols["+"]["precedence"] == 1
        assert len(sl) == 1


# ---------------------------------------------------------------------------
# Group 3 — remove_symbol
# ---------------------------------------------------------------------------


class TestRemoveSymbol:
    def test_remove_existing_symbol(self):
        sl = _mixed()
        sl.remove_symbol("+")
        assert "+" not in sl.symbols
        assert len(sl) == 2

    def test_remove_missing_symbol_raises_key_error(self):
        sl = _empty()
        with pytest.raises(KeyError):
            sl.remove_symbol("nonexistent")


# ---------------------------------------------------------------------------
# Group 4 — getter methods
# ---------------------------------------------------------------------------


class TestGetters:
    def setup_method(self):
        self.sl = _mixed()

    def test_get_type_present(self):
        assert self.sl.get_type("+") == "op"
        assert self.sl.get_type("sin") == "fn"
        assert self.sl.get_type("X_0") == "var"

    def test_get_type_missing(self):
        assert self.sl.get_type("MISSING") == ""

    def test_get_precedence_present(self):
        assert self.sl.get_precedence("+") == 0
        assert self.sl.get_precedence("sin") == 5

    def test_get_precedence_missing(self):
        assert self.sl.get_precedence("MISSING") == -1

    def test_get_np_fn_present(self):
        assert self.sl.get_np_fn("+") == "{} = {} + {}"

    def test_get_np_fn_missing(self):
        assert self.sl.get_np_fn("MISSING") == ""

    def test_get_latex_str_present(self):
        assert self.sl.get_latex_str("+") == r"{} + {}"

    def test_get_latex_str_missing(self):
        assert self.sl.get_latex_str("MISSING") == ""


# ---------------------------------------------------------------------------
# Group 5 — get_symbols_of_type
# ---------------------------------------------------------------------------


class TestGetSymbolsOfType:
    def setup_method(self):
        self.sl = _mixed()

    def test_get_ops(self):
        assert self.sl.get_symbols_of_type("op") == ["+"]

    def test_get_fns(self):
        assert self.sl.get_symbols_of_type("fn") == ["sin"]

    def test_get_vars(self):
        assert self.sl.get_symbols_of_type("var") == ["X_0"]

    def test_type_with_no_matches_returns_empty(self):
        assert self.sl.get_symbols_of_type("lit") == []

    def test_empty_library_returns_empty(self):
        assert _empty().get_symbols_of_type("op") == []


# ---------------------------------------------------------------------------
# Group 6 — symbols2index
# ---------------------------------------------------------------------------


class TestSymbols2Index:
    def test_empty_library(self):
        assert _empty().symbols2index() == {}

    def test_length_matches(self):
        sl = _mixed()
        idx = sl.symbols2index()
        assert len(idx) == len(sl)

    def test_values_are_zero_based_contiguous(self):
        sl = _mixed()
        idx = sl.symbols2index()
        assert sorted(idx.values()) == list(range(len(sl)))

    def test_insertion_order_preserved(self):
        sl = _empty()
        sl.add_symbol("a", "lit", 5, "a", "a")
        sl.add_symbol("b", "lit", 5, "b", "b")
        sl.add_symbol("c", "lit", 5, "c", "c")
        idx = sl.symbols2index()
        assert idx["a"] == 0
        assert idx["b"] == 1
        assert idx["c"] == 2

    def test_after_removal_reindexed(self):
        sl = _mixed()
        sl.remove_symbol("+")
        idx = sl.symbols2index()
        assert 0 in idx.values()
        assert max(idx.values()) == len(sl) - 1


# ---------------------------------------------------------------------------
# Group 7 — from_symbol_list
# ---------------------------------------------------------------------------


class TestFromSymbolList:
    def test_known_names_and_variables(self):
        sl = SymbolLibrary.from_symbol_list(["+", "sin"], num_variables=2)
        assert set(sl.symbols.keys()) == {"+", "sin", "X_0", "X_1"}

    def test_empty_list_zero_variables(self):
        sl = SymbolLibrary.from_symbol_list([], num_variables=0)
        assert sl.symbols == {}

    def test_variable_in_symbols_list_not_duplicated(self):
        sl = SymbolLibrary.from_symbol_list(["X_0"], num_variables=1)
        var_tokens = [k for k in sl.symbols if k.startswith("X_")]
        assert var_tokens.count("X_0") == 1

    def test_default_num_variables_is_25(self):
        sl = SymbolLibrary.from_symbol_list([])
        var_tokens = [k for k in sl.symbols if k.startswith("X_")]
        assert len(var_tokens) == 25

    def test_unknown_symbol_silently_omitted(self):
        sl = SymbolLibrary.from_symbol_list(["NOT_REAL"], num_variables=0)
        assert "NOT_REAL" not in sl.symbols


# ---------------------------------------------------------------------------
# Group 8 — default_symbols
# ---------------------------------------------------------------------------


class TestDefaultSymbols:
    def test_default_25_variables(self):
        sl = SymbolLibrary.default_symbols()
        var_tokens = sl.get_symbols_of_type("var")
        assert len(var_tokens) == 25

    def test_zero_variables(self):
        sl = SymbolLibrary.default_symbols(num_variables=0)
        assert sl.get_symbols_of_type("var") == []

    def test_non_var_symbols_present_with_zero_variables(self):
        sl = SymbolLibrary.default_symbols(num_variables=0)
        assert sl.get_type("+") == "op"
        assert sl.get_type("sin") == "fn"

    def test_custom_variable_count(self):
        sl = SymbolLibrary.default_symbols(num_variables=3)
        var_tokens = sl.get_symbols_of_type("var")
        assert set(var_tokens) == {"X_0", "X_1", "X_2"}

    def test_no_extra_variables_beyond_count(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        assert "X_2" not in sl.symbols

    @pytest.mark.parametrize(
        "sym,expected_type",
        [
            ("+", "op"),
            ("-", "op"),
            ("*", "op"),
            ("/", "op"),
            ("^", "op"),
            ("sin", "fn"),
            ("cos", "fn"),
            ("sqrt", "fn"),
            ("ln", "fn"),
            ("pi", "lit"),
            ("e", "lit"),
            ("C", "const"),
        ],
    )
    def test_spot_check_types(self, sym, expected_type):
        sl = SymbolLibrary.default_symbols(num_variables=0)
        assert sl.get_type(sym) == expected_type

    def test_num_variables_attribute_matches_arg(self):
        sl = SymbolLibrary.default_symbols(num_variables=7)
        assert sl.num_variables == 7


# ---------------------------------------------------------------------------
# Group 9 — to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_empty_library(self):
        d = _empty().to_dict()
        assert d["format_version"] == 1
        assert d["type"] == "SymbolLibrary"
        assert d["symbols"] == {}
        assert d["num_variables"] == 0

    def test_to_dict_populated(self):
        sl = _mixed()
        d = sl.to_dict()
        assert "+" in d["symbols"]
        assert d["num_variables"] == 1

    def test_to_dict_preamble_preserved(self):
        sl = SymbolLibrary(preamble=["import torch"])
        d = sl.to_dict()
        assert d["preamble"] == ["import torch"]

    def test_from_dict_reconstructs_symbols(self):
        sl = _mixed()
        sl2 = SymbolLibrary.from_dict(sl.to_dict())
        assert sl2.symbols == sl.symbols

    def test_from_dict_reconstructs_preamble_and_num_variables(self):
        sl = _mixed()
        sl2 = SymbolLibrary.from_dict(sl.to_dict())
        assert sl2.preamble == sl.preamble
        assert sl2.num_variables == sl.num_variables

    def test_full_roundtrip(self):
        sl = _mixed()
        sl2 = SymbolLibrary.from_dict(sl.to_dict())
        assert sl2.symbols == sl.symbols
        assert sl2.preamble == sl.preamble
        assert sl2.num_variables == sl.num_variables

    def test_from_dict_invalid_format_version_raises(self):
        d = _empty().to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError, match="format_version"):
            SymbolLibrary.from_dict(d)

    def test_from_dict_missing_format_version_defaults_to_1(self):
        d = _empty().to_dict()
        del d["format_version"]
        sl = SymbolLibrary.from_dict(d)
        assert sl.symbols == {}

    def test_custom_preamble_survives_roundtrip(self):
        sl = SymbolLibrary(preamble=["import torch", "import sympy"])
        sl2 = SymbolLibrary.from_dict(sl.to_dict())
        assert sl2.preamble == ["import torch", "import sympy"]

    def test_from_dict_symbols_aliasing(self):
        # from_dict assigns d["symbols"] directly — mutations bleed through.
        # This test documents (and pins) that behaviour.
        sl = _mixed()
        d = sl.to_dict()
        sl2 = SymbolLibrary.from_dict(d)
        sl2.symbols["+"]["precedence"] = 99
        assert d["symbols"]["+"]["precedence"] == 99  # aliased, not copied


# ---------------------------------------------------------------------------
# Group 10 — dunder methods (__len__, __str__, __copy__, __deepcopy__)
# ---------------------------------------------------------------------------


class TestDunders:
    def test_len_empty(self):
        assert len(_empty()) == 0

    def test_len_after_additions(self):
        sl = _mixed()
        assert len(sl) == 3

    def test_len_after_removal(self):
        sl = _mixed()
        sl.remove_symbol("+")
        assert len(sl) == 2

    def test_str_empty(self):
        assert str(_empty()) == ""

    def test_str_single_symbol(self):
        sl = _empty()
        sl.add_symbol("a", "lit", 5, "a", "a")
        assert str(sl) == "a"

    def test_str_multiple_symbols(self):
        sl = _empty()
        sl.add_symbol("a", "lit", 5, "a", "a")
        sl.add_symbol("b", "lit", 5, "b", "b")
        assert str(sl) == "a, b"

    def test_str_insertion_order(self):
        sl = _mixed()
        tokens = str(sl).split(", ")
        assert tokens == ["+", "sin", "X_0"]

    def test_copy_returns_new_object(self):
        sl = _mixed()
        sl2 = copy.copy(sl)
        assert sl2 is not sl

    def test_copy_symbols_dict_is_independent(self):
        sl = _mixed()
        sl2 = copy.copy(sl)
        sl2.symbols["+"]["precedence"] = 99
        assert sl.symbols["+"]["precedence"] == 0

    def test_copy_preamble_is_independent(self):
        sl = _mixed()
        sl2 = copy.copy(sl)
        sl2.preamble.append("extra")
        assert "extra" not in sl.preamble

    def test_copy_num_variables_matches(self):
        sl = _mixed()
        sl2 = copy.copy(sl)
        assert sl2.num_variables == sl.num_variables

    def test_copy_adding_symbol_does_not_affect_original(self):
        sl = _mixed()
        sl2 = copy.copy(sl)
        sl2.add_symbol("new", "lit", 5, "new", "new")
        assert "new" not in sl.symbols

    def test_deepcopy_returns_independent_copy(self):
        sl = _mixed()
        sl2 = copy.deepcopy(sl)
        assert sl2 is not sl
        assert sl2.symbols == sl.symbols
        sl2.symbols["+"]["precedence"] = 99
        assert sl.symbols["+"]["precedence"] == 0
