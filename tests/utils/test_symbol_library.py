import copy
import dataclasses
import json
import math
from fractions import Fraction

import numpy as np
import pytest

from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.symbol_library import Symbol, SymbolLibrary
from SRToolkit.utils.types import FN_POSTFIX, FN_PREFIX, LEAF, OP_ADDITIVE, OP_MULTIPLICATIVE, OP_POWER
from SRToolkit.utils.units import UnitRule

# ── Symbol ────────────────────────────────────────────────────────────────────


def _sym_dict(**changes):
    """Dictionary form of a custom fn symbol, with keys changed (value None deletes the key)."""
    d = {
        "symbol": "cbrt",
        "type": "fn",
        "precedence": FN_PREFIX,
        "np_fn": "np.cbrt({})",
        "latex_str": r"\sqrt[3]{{{}}}",
        "cython_id": -1,
        "unit_rule": {"kind": "power", "exponent": "1/3"},
    }
    d.update(changes)
    return {k: v for k, v in d.items() if v is not None or k == "unit_rule"}


def _without(key):
    d = _sym_dict()
    del d[key]
    return d


class TestSymbol:
    @pytest.mark.parametrize(
        "fields, args, expected",
        [
            (("mod", "op", OP_MULTIPLICATIVE, "np.mod({}, {})"), ("a", "b"), r"a \text{mod} b"),
            (("sat", "fn", FN_PREFIX, "{0} / (1 + {0})"), ("x",), r"\text{sat} x"),
            (("C", "const", LEAF, "C[{}]"), (17,), "C_{17}"),
            (("K_a", "const", LEAF, "K_a[{}]"), (17,), "{K_a}_{17}"),
            (("g", "lit", LEAF, "9.81"), (), r"\text{g}"),
            (("x", "var", LEAF, "X[:, 0]"), (), r"\text{x}"),
            (("sin", "fn", FN_PREFIX, "np.sin({})", r"\sin {}"), ("x",), r"\sin x"),
        ],
        ids=["op", "fn", "const", "const-with-underscore", "lit", "var", "explicit"],
    )
    def test_default_latex_template_by_type(self, fields, args, expected):
        assert Symbol(*fields).latex_str.format(*args) == expected

    @pytest.mark.parametrize(
        "token, symbol_type, rule, expected",
        [
            ("sqrt", "fn", None, UnitRule.power(Fraction(1, 2))),
            ("+", "op", None, UnitRule.SAME),
            ("sqrt", "op", None, UnitRule.DIMENSIONLESS),
            ("abs", "fn", None, UnitRule.DIMENSIONLESS),
            ("^6", "fn", None, UnitRule.DIMENSIONLESS),
            ("max", "op", UnitRule.SAME, UnitRule.SAME),
            ("sqrt", "fn", UnitRule.DIMENSIONLESS, UnitRule.DIMENSIONLESS),
            ("X_0", "var", None, None),
            ("C", "const", None, None),
            ("pi", "lit", None, None),
        ],
        ids=[
            "predefined-fn", "predefined-op", "type-differs", "not-predefined", "postfix-not-predefined",
            "explicit", "override", "var", "const", "lit",
        ],
    )  # fmt: skip
    def test_unit_rule_resolution(self, token, symbol_type, rule, expected):
        assert Symbol(token, symbol_type, 5, "f({})", unit_rule=rule).unit_rule == expected

    @pytest.mark.parametrize(
        "symbol_type, precedence, rule, error, match",
        [
            ("bogus", 5, None, ValueError, "type"),
            ("op", -1, None, ValueError, "precedence"),
            ("var", LEAF, UnitRule.SAME, ValueError, "unit"),
            ("const", LEAF, UnitRule.DIMENSIONLESS, ValueError, "unit"),
            ("lit", LEAF, UnitRule.SAME, ValueError, "unit"),
            ("op", 5, UnitRule.power(2), ValueError, "accept"),
            ("fn", 5, UnitRule.MULTIPLY, ValueError, "accept"),
            ("fn", 5, UnitRule.DIVIDE, ValueError, "accept"),
            ("fn", 5, UnitRule.EXPONENT, ValueError, "accept"),
            ("fn", 5, "same", TypeError, "UnitRule"),
            ("op", 5, {"kind": "same"}, TypeError, "UnitRule"),
        ],
        ids=[
            "bad-type", "op-negative-precedence", "rule-on-var", "rule-on-const", "rule-on-lit", "op-power",
            "fn-multiply", "fn-divide", "fn-exponent", "rule-str", "rule-dict",
        ],
    )  # fmt: skip
    def test_invalid_symbol_raises(self, symbol_type, precedence, rule, error, match):
        with pytest.raises(error, match=match):
            Symbol("t", symbol_type, precedence, "f({})", unit_rule=rule)

    @pytest.mark.parametrize(
        "symbol_type, precedence", [("op", 0), ("fn", 0), ("fn", FN_POSTFIX)], ids=["op-0", "fn-0", "fn-postfix"]
    )
    def test_precedence_boundaries_accepted(self, symbol_type, precedence):
        assert Symbol("t", symbol_type, precedence, "f({})").precedence == precedence

    def test_symbol_is_immutable(self):
        s = Symbol("sat", "fn", FN_PREFIX, "{0} / (1 + {0})")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.python_str = "np.sin({})"

    @pytest.mark.parametrize(
        "make, expected",
        [
            (
                lambda: Symbol(
                    "cbrt", "fn", FN_PREFIX, "np.cbrt({})", r"\sqrt[3]{{{}}}", -1, UnitRule.power(Fraction(1, 3))
                ),
                {
                    "symbol": "cbrt",
                    "type": "fn",
                    "precedence": FN_PREFIX,
                    "np_fn": "np.cbrt({})",
                    "latex_str": r"\sqrt[3]{{{}}}",
                    "cython_id": -1,
                    "unit_rule": UnitRule.power(Fraction(1, 3)).to_dict(),
                },
            ),
            (
                lambda: Symbol("x", "var", LEAF, "X[:, 0]", "x", 7),
                {
                    "symbol": "x",
                    "type": "var",
                    "precedence": LEAF,
                    "np_fn": "X[:, 0]",
                    "latex_str": "x",
                    "cython_id": 7,
                },
            ),
        ],
        ids=["fn-with-rule", "var-without-rule"],
    )
    def test_to_dict_fields(self, make, expected):
        assert make().to_dict() == expected

    @pytest.mark.parametrize(
        "make",
        [
            lambda: Symbol("max", "op", OP_ADDITIVE, "np.maximum({}, {})", r"\max({}, {})", -1, UnitRule.SAME),
            lambda: Symbol(
                "cbrt", "fn", FN_PREFIX, "np.cbrt({})", r"\sqrt[3]{{{}}}", -1, UnitRule.power(Fraction(1, 3))
            ),
            lambda: Symbol("x", "var", LEAF, "X[:, 0]", "x"),
            lambda: Symbol("K", "const", LEAF, "K[{}]", "k_{{{}}}"),
            lambda: Symbol("g", "lit", LEAF, "9.81", "g", 3),
        ],
        ids=["op", "fn", "var", "const", "lit"],
    )
    def test_round_trip_through_json(self, make):
        symbol = make()
        assert Symbol.from_dict(json.loads(json.dumps(symbol.to_dict()))) == symbol

    @pytest.mark.parametrize(
        "d, field, expected",
        [
            (_without("cython_id"), "cython_id", -1),
            (
                _sym_dict(symbol="sqrt", np_fn="np.sqrt({})", unit_rule=None),
                "unit_rule",
                UnitRule.power(Fraction(1, 2)),
            ),
            (_without("unit_rule"), "unit_rule", UnitRule.DIMENSIONLESS),
        ],
        ids=["cython-id-missing", "rule-none-predefined", "rule-missing-custom"],
    )
    def test_from_dict_optional_keys(self, d, field, expected):
        assert getattr(Symbol.from_dict(d), field) == expected

    @pytest.mark.parametrize(
        "d, error",
        [
            (_without("symbol"), KeyError),
            (_without("type"), KeyError),
            (_without("precedence"), KeyError),
            (_without("np_fn"), KeyError),
            (_without("latex_str"), KeyError),
            (_sym_dict(type="bogus"), ValueError),
            (_sym_dict(unit_rule={"kind": "nope"}), ValueError),
            (_sym_dict(type="op", unit_rule={"kind": "power", "exponent": "2"}), ValueError),
        ],
        ids=[
            "no-symbol", "no-type", "no-precedence", "no-np_fn", "no-latex_str", "bad-type", "unknown-rule-kind",
            "op-with-power",
        ],
    )  # fmt: skip
    def test_from_dict_invalid_raises(self, d, error):
        with pytest.raises(error):
            Symbol.from_dict(d)


# ── SymbolLibrary construction ────────────────────────────────────────────────


class TestSymbolLibraryInit:
    @pytest.mark.parametrize(
        "make, tokens",
        [
            (lambda: SymbolLibrary(["+", "*", "sin"], num_variables=2), ["+", "*", "sin", "C", "X_0", "X_1"]),
            (lambda: SymbolLibrary(num_variables=2), ["C", "X_0", "X_1"]),
            (lambda: SymbolLibrary(), ["C"]),
            (lambda: SymbolLibrary([], const_symbol=None), []),
        ],
        ids=["list", "none", "empty-default", "empty-no-const"],
    )
    def test_constructor_tokens_in_order(self, make, tokens):
        assert list(make().symbols) == tokens

    @pytest.mark.parametrize("token", ["+", "sqrt", "pi"])
    @pytest.mark.parametrize(
        "make",
        [lambda s: SymbolLibrary(s, 1), lambda s: SymbolLibrary.from_symbol_list(s, 1)],
        ids=["constructor", "from_symbol_list"],
    )
    def test_selected_symbols_equal_predefined(self, make, token):
        assert make(["+", "sqrt", "pi"]).get(token) == SymbolLibrary.default_symbols().get(token)

    def test_variables_read_columns_in_order(self):
        lib = SymbolLibrary(num_variables=2)
        assert (lib["X_1"].type, lib["X_1"].python_str, lib.num_variables) == ("var", "X[:, 1]", 2)

    @pytest.mark.parametrize(
        "make, error, match",
        [
            (lambda: SymbolLibrary(["not_a_real_symbol"]), ValueError, "not_a_real_symbol"),
            (lambda: SymbolLibrary(modules={"np": "numpy"}), ValueError, "np"),
            (lambda: SymbolLibrary(modules={"z": "no_such_module_xyz"}), ImportError, "no_such_module_xyz"),
        ],
        ids=["unknown-token", "np-alias", "unimportable-module"],
    )
    def test_constructor_invalid_raises(self, make, error, match):
        with pytest.raises(error, match=match):
            make()


_CONSTRUCTORS = {
    "init": lambda const_symbol: SymbolLibrary(["+"], 1, const_symbol=const_symbol),
    "from_symbol_list": lambda const_symbol: SymbolLibrary.from_symbol_list(["+"], 1, const_symbol=const_symbol),
    "default_symbols": lambda const_symbol: SymbolLibrary.default_symbols(1, const_symbol=const_symbol),
}


class TestOneFreeConstant:
    @pytest.mark.parametrize("constructor", list(_CONSTRUCTORS))
    @pytest.mark.parametrize(
        "const_symbol, consts", [("C", ["C"]), ("K", ["K"]), (None, [])], ids=["default", "renamed", "none"]
    )
    def test_const_symbol_names_the_only_constant(self, constructor, const_symbol, consts):
        lib = _CONSTRUCTORS[constructor](const_symbol)
        assert lib.constant_symbol == const_symbol
        assert lib.get_symbols_of_type("const") == consts

    def test_listing_the_constant_is_allowed(self):
        lib = SymbolLibrary.from_symbol_list(["C", "+"], num_variables=1)
        assert lib.get_symbols_of_type("const") == ["C"]

    def test_adding_a_constant_to_a_library_without_one(self):
        lib = SymbolLibrary(const_symbol=None)
        lib.add_symbol("K", "const", 1000, "K[{}]")
        assert lib.constant_symbol == "K"

    def test_constant_symbol_is_none_after_removal(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.remove_symbol("C")
        assert lib.constant_symbol is None

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: SymbolLibrary().add_symbol("K", "const", 1000, "K[{}]"), "at most one"),
            (lambda: SymbolLibrary.from_symbol_list(["C"], 1, const_symbol="K"), r"Unknown symbols \['C'\]"),
            (lambda: SymbolLibrary.from_symbol_list(["C"], 1, const_symbol=None), r"Unknown symbols \['C'\]"),
            (lambda: SymbolLibrary.from_dict(_dict_with_two_constants()), "At most one free constant"),
            (lambda: SymbolLibrary().add_symbol("C", "lit", LEAF, "1.0"), "free constant"),
        ],
        ids=["add-second", "listed-C-renamed", "listed-C-none", "from-dict-two", "redefine-as-lit"],
    )
    def test_second_constant_raises(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()


def _dict_with_two_constants():
    d = SymbolLibrary(["+"], 1).to_dict()
    d["symbols"]["K"] = {**d["symbols"]["C"], "symbol": "K"}
    return d


# ── add_symbol ────────────────────────────────────────────────────────────────


class TestSymbolLibraryAddSymbol:
    @pytest.mark.parametrize(
        "make, args, expected, consts",
        [
            (
                lambda: SymbolLibrary(const_symbol=None),
                ("x", "var", LEAF, None),
                Symbol("x", "var", LEAF, "X[:, 0]"),
                [],
            ),
            (
                SymbolLibrary,
                ("sat", "fn", FN_PREFIX, "({0}) / (1 + ({0}))"),
                Symbol("sat", "fn", FN_PREFIX, "({0}) / (1 + ({0}))"),
                ["C"],
            ),
            (
                SymbolLibrary,
                ("C", "const", LEAF, "C[{}]", r"c_{{{}}}"),
                Symbol("C", "const", LEAF, "C[{}]", r"c_{{{}}}"),
                ["C"],
            ),
            (
                SymbolLibrary,
                ("X_0", "var", LEAF, "X[:, 0]"),
                Symbol("X_0", "var", LEAF, "X[:, 0]"),
                ["C"],
            ),
            (
                SymbolLibrary,
                ("g", "lit", LEAF, "9.81", ""),
                Symbol("g", "lit", LEAF, "9.81"),
                ["C"],
            ),
            (
                SymbolLibrary,
                ("sqrt", "fn", FN_PREFIX, "np.sqrt({})", r"\sqrt{{{}}}", 14),
                Symbol("sqrt", "fn", FN_PREFIX, "np.sqrt({})", r"\sqrt{{{}}}", 14, UnitRule.power(Fraction(1, 2))),
                ["C"],
            ),
            (
                SymbolLibrary,
                ("sqrt", "fn", FN_PREFIX, "np.abs({})", r"\sqrt{{{}}}"),
                Symbol("sqrt", "fn", FN_PREFIX, "np.abs({})", r"\sqrt{{{}}}", -1, UnitRule.power(Fraction(1, 2))),
                ["C"],
            ),
        ],
        ids=[
            "var-auto-python_str", "fn-latex-none", "const-replaces-C", "X_0", "lit-latex-empty",
            "predefined-with-id", "predefined-without-id",
        ],
    )  # fmt: skip
    def test_add_symbol_stores_record(self, make, args, expected, consts):
        lib = make()
        lib.add_symbol(*args)
        assert lib.get(args[0]) == expected
        assert lib.get_symbols_of_type("const") == consts

    def test_add_symbol_replaces_same_token(self):
        lib = SymbolLibrary(["+"], 1)
        lib.add_symbol("+", "op", OP_ADDITIVE, "np.add({}, {})")
        assert (len(lib), lib["+"].python_str) == (3, "np.add({}, {})")

    def test_var_counter(self):
        lib = SymbolLibrary(const_symbol=None)
        lib.add_symbol("x", "var", LEAF, None)
        assert (lib["x"].python_str, lib.num_variables) == ("X[:, 0]", 1)
        lib.add_symbol("y", "var", LEAF, "y")
        assert (lib["y"].python_str, lib.num_variables) == ("y", 2)
        lib.add_symbol("x", "var", LEAF, "")
        assert (lib["x"].python_str, lib.num_variables) == ("X[:, 2]", 3)
        lib.remove_symbol("y")
        assert lib.num_variables == 3

    @pytest.mark.parametrize(
        "args",
        [
            ("K", "const", LEAF, "K[{}]"),
            ("z", "var", LEAF, None, None, -1, UnitRule.SAME),
            ("t", "bogus", 5, "t"),
            ("C", "lit", LEAF, "1.0"),
        ],
        ids=["second-const", "var-with-rule", "bad-type", "redefine-constant"],
    )
    def test_failed_add_leaves_library_unchanged(self, args):
        lib = SymbolLibrary(["+"], 1)
        with pytest.raises(ValueError):
            lib.add_symbol(*args)
        assert (str(lib), lib.num_variables, lib.constant_symbol) == ("+, C, X_0", 1, "C")


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


# ── Lookup ────────────────────────────────────────────────────────────────────


class TestSymbolLibraryGet:
    @pytest.mark.parametrize(
        "label, base",
        [("C_3", "C"), ("C_017", "C"), ("K_2", "K"), ("K_a_3", "K_a")],
        ids=["C", "leading-zeros", "renamed-constant", "name-with-underscore"],
    )
    def test_labelled_constant_resolves(self, label, base):
        lib = SymbolLibrary.default_symbols(1, const_symbol=base)
        assert lib.get(label) == lib.get(base)
        # Labels are recognised, not registered.
        assert lib.get_symbols_of_type("const") == [base]
        assert label not in lib.symbols

    @pytest.mark.parametrize(
        "token",
        [
            "(", ")", "1.5", "nope", "C_", "C_a", "C_1.5", "C_-1", "C_١", "pi_1", "X_0_1", "D_1",
            "C_1\n", "C_+1", "C_ 1", "C_1_0", "C_²", 2.0, 0,
        ],
    )  # fmt: skip
    def test_get_returns_none(self, token):
        assert SymbolLibrary.default_symbols(1).get(token) is None

    def test_get_label_without_free_constant_is_none(self):
        assert SymbolLibrary.default_symbols(1, const_symbol=None).get("C_1") is None

    def test_registered_lit_wins_over_label(self):
        lib = SymbolLibrary.from_symbol_list(["+", "C"], num_variables=1)
        lib.add_symbol("C_1", "lit", LEAF, "1.5")
        assert (lib.get("C_1").type, lib.get("C_2").type) == ("lit", "const")

    @pytest.mark.parametrize("token, expected", [("+", "+"), ("C_3", "C")], ids=["registered", "label"])
    def test_getitem_matches_get(self, token, expected):
        lib = SymbolLibrary.default_symbols(1)
        assert lib[token] == lib.get(token)
        assert lib[token].token == expected

    @pytest.mark.parametrize(
        "const_symbol, token",
        [("C", "("), ("C", "nope"), (None, "C_1")],
        ids=["bracket", "unknown", "label-without-constant"],
    )
    def test_getitem_raises_keyerror(self, const_symbol, token):
        lib = SymbolLibrary.default_symbols(1, const_symbol=const_symbol)
        with pytest.raises(KeyError):
            lib[token]

    @pytest.mark.parametrize(
        "token, expected",
        [("X_0", True), ("C_3", True), ("(", False), ("nope", False), (2.0, False), (0, False)],
        ids=["registered", "label", "bracket", "unknown", "float", "int"],
    )
    def test_contains_matches_get(self, token, expected):
        assert (token in SymbolLibrary.default_symbols(1)) is expected

    def test_iterates_registered_tokens_in_order(self):
        assert list(SymbolLibrary(["+"], 1)) == ["+", "C", "X_0"]


class TestSymbolLibraryModuleNamespace:
    def test_module_namespace(self):
        lib = SymbolLibrary(modules={"m": "math"})
        ns = lib.module_namespace()
        assert ns == {"np": np, "m": math}
        ns["extra"] = 1
        assert lib.module_namespace() == {"np": np, "m": math}
        assert SymbolLibrary().modules == {}

    def test_modules_are_a_copy_of_the_argument(self):
        modules = {"m": "math"}
        lib = SymbolLibrary(modules=modules)
        modules["z"] = "no_such_module_xyz"
        assert lib.modules == {"m": "math"}


def _xy_library():
    lib = SymbolLibrary(const_symbol=None)
    lib.add_symbol("x", "var", LEAF, None)
    lib.add_symbol("+", "op", OP_ADDITIVE, "{} + {}")
    lib.add_symbol("y", "var", LEAF, None)
    return lib


class TestSymbolLibraryGetSymbolsOfType:
    @pytest.mark.parametrize(
        "symbol_type, expected", [("var", ["x", "y"]), ("op", ["+"]), ("fn", [])], ids=["var", "op", "none"]
    )
    def test_symbols_of_type(self, symbol_type, expected):
        assert _xy_library().get_symbols_of_type(symbol_type) == expected


class TestSymbolLibrarySymbols2Index:
    def test_returns_correct_index_mapping(self):
        lib = SymbolLibrary(const_symbol=None)
        lib.add_symbol("x", "var", 0, "x")
        lib.add_symbol("y", "var", 0, "y")
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert lib.symbols2index() == {"x": 0, "y": 1, "+": 2}
        lib.remove_symbol("x")
        assert lib.symbols2index() == {"y": 0, "+": 1}

    def test_empty_library_returns_empty_dict(self):
        lib = SymbolLibrary(const_symbol=None)
        assert lib.symbols2index() == {}


# ── Predefined symbols ────────────────────────────────────────────────────────

_OPS = ["+", "-", "*", "/", "^"]
_PREFIX_FNS = [
    "u-", "sqrt", "sin", "cos", "exp", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "floor", "ceil",
    "ln", "log",
]  # fmt: skip
_POSTFIX_FNS = ["^-1", "^2", "^3", "^4", "^5"]

_X = np.array([0.2, 0.5, 0.9])
_Y = np.array([0.3, 0.7, 1.1])


class TestSymbolLibraryDefaultSymbols:
    @pytest.mark.parametrize(
        "symbol_type, tokens",
        [
            ("op", set(_OPS)),
            ("fn", set(_PREFIX_FNS + _POSTFIX_FNS)),
            ("lit", {"pi", "e"}),
            ("const", {"C"}),
            ("var", {"X_0", "X_1", "X_2"}),
        ],
    )
    def test_default_tokens_per_type(self, symbol_type, tokens):
        assert set(SymbolLibrary.default_symbols(3).get_symbols_of_type(symbol_type)) == tokens

    @pytest.mark.parametrize(
        "make, count",
        [
            (SymbolLibrary.default_symbols, 54),
            (lambda: SymbolLibrary.default_symbols(2), 31),
            (lambda: SymbolLibrary.default_symbols(0, const_symbol=None), 28),
        ],
        ids=["defaults", "two-vars", "no-vars-no-const"],
    )
    def test_default_symbol_count(self, make, count):
        # 5 ops + 21 fns + 2 lits + 1 const (if any) + num_variables vars (25 by default)
        assert len(make()) == count

    def test_default_unit_rules(self):
        lib = SymbolLibrary.default_symbols(1)
        rules = {s: lib[s].unit_rule for s in lib.get_symbols_of_type("op") + lib.get_symbols_of_type("fn")}
        same, dimensionless = UnitRule.SAME, UnitRule.DIMENSIONLESS
        assert rules == {
            "+": same,
            "-": same,
            "*": UnitRule.MULTIPLY,
            "/": UnitRule.DIVIDE,
            "^": UnitRule.EXPONENT,
            "u-": same,
            "sqrt": UnitRule.power(Fraction(1, 2)),
            "sin": dimensionless,
            "cos": dimensionless,
            "exp": dimensionless,
            "tan": dimensionless,
            "arcsin": dimensionless,
            "arccos": dimensionless,
            "arctan": dimensionless,
            "sinh": dimensionless,
            "cosh": dimensionless,
            "tanh": dimensionless,
            "floor": same,
            "ceil": same,
            "ln": dimensionless,
            "log": dimensionless,
            "^-1": UnitRule.power(-1),
            "^2": UnitRule.power(2),
            "^3": UnitRule.power(3),
            "^4": UnitRule.power(4),
            "^5": UnitRule.power(5),
        }

    @pytest.mark.parametrize(
        "tokens, precedence",
        [
            (["+", "-"], OP_ADDITIVE),
            (["*", "/"], OP_MULTIPLICATIVE),
            (["^"], OP_POWER),
            (_PREFIX_FNS, FN_PREFIX),
            (_POSTFIX_FNS, FN_POSTFIX),
            (["pi", "e", "C", "X_0"], LEAF),
        ],
        ids=["additive", "multiplicative", "power", "prefix", "postfix", "leaves"],
    )
    def test_default_precedences(self, tokens, precedence):
        lib = SymbolLibrary.default_symbols(1)
        assert {t: lib[t].precedence for t in tokens} == dict.fromkeys(tokens, precedence)

    @pytest.mark.parametrize(
        "token, args, expected",
        [
            ("+", ("x", "y"), _X + _Y),
            ("-", ("x", "y"), _X - _Y),
            ("*", ("x", "y"), _X * _Y),
            ("/", ("x", "y"), _X / _Y),
            ("^", ("x", "y"), _X**_Y),
            ("u-", ("x",), -_X),
            ("sqrt", ("x",), np.sqrt(_X)),
            ("sin", ("x",), np.sin(_X)),
            ("cos", ("x",), np.cos(_X)),
            ("exp", ("x",), np.exp(_X)),
            ("tan", ("x",), np.tan(_X)),
            ("arcsin", ("x",), np.arcsin(_X)),
            ("arccos", ("x",), np.arccos(_X)),
            ("arctan", ("x",), np.arctan(_X)),
            ("sinh", ("x",), np.sinh(_X)),
            ("cosh", ("x",), np.cosh(_X)),
            ("tanh", ("x",), np.tanh(_X)),
            ("floor", ("y",), np.floor(_Y)),
            ("ceil", ("y",), np.ceil(_Y)),
            ("ln", ("x",), np.log(_X)),
            ("log", ("x",), np.log10(_X)),
            ("^-1", ("x",), 1 / _X),
            ("^2", ("x",), _X**2),
            ("^3", ("x",), _X**3),
            ("^4", ("x",), _X**4),
            ("^5", ("x",), _X**5),
            ("pi", (), np.pi),
            ("e", (), np.e),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_default_python_str_semantics(self, token, args, expected):
        # floor/ceil are checked on _Y, whose values straddle an integer.
        lib = SymbolLibrary.default_symbols(1)
        result = eval(lib[token].python_str.format(*args), lib.module_namespace(), {"x": _X, "y": _Y})
        np.testing.assert_allclose(result, expected)


class TestSymbolLibraryFromSymbolList:
    @pytest.mark.parametrize(
        "make, tokens",
        [
            (lambda: SymbolLibrary.from_symbol_list(["+", "*"], num_variables=2), ["+", "*", "C", "X_0", "X_1"]),
            (lambda: SymbolLibrary.from_symbol_list(["+"], num_variables=1, const_symbol=None), ["+", "X_0"]),
        ],
        ids=["with-const", "without-const"],
    )
    def test_contains_exactly_the_requested_symbols(self, make, tokens):
        assert list(make().symbols) == tokens

    def test_variable_in_symbols_list_not_duplicated(self):
        lib = SymbolLibrary.from_symbol_list(["X_0"], num_variables=1)
        assert list(lib.symbols.keys()).count("X_0") == 1

    def test_expression_as_symbol_raises(self):
        # Issue #7: a whole expression is not a token and used to be dropped silently.
        with pytest.raises(ValueError, match=r"x/\(a \+ x\)"):
            SymbolLibrary.from_symbol_list(["+", "x/(a + x)"], num_variables=1)

    def test_variable_beyond_num_variables_raises(self):
        with pytest.raises(ValueError, match="X_3.*increase num_variables"):
            SymbolLibrary.from_symbol_list(["+", "X_3"], num_variables=1)

    def test_default_num_variables_is_25(self):
        lib = SymbolLibrary.from_symbol_list([])
        assert len(lib.get_symbols_of_type("var")) == 25


# ── Serialisation ─────────────────────────────────────────────────────────────


def _legacy_dict(**top):
    """A library dictionary as saved before modules, cython IDs and unit rules existed."""
    d = {
        "type": "SymbolLibrary",
        "symbols": {
            "+": {"symbol": "+", "type": "op", "precedence": 100, "np_fn": "{} + {}", "latex_str": "{} + {}"},
            "sqrt": {"symbol": "sqrt", "type": "fn", "precedence": 1000, "np_fn": "np.sqrt({})", "latex_str": ""},
            "X_0": {"symbol": "X_0", "type": "var", "precedence": 1000, "np_fn": "X[:, 0]", "latex_str": "X_0"},
        },
        "preamble": ["import numpy as np"],
        "num_variables": 1,
    }
    d.update(top)
    return d


def _current_dict(**top):
    d = SymbolLibrary(["+", "sqrt"], 1).to_dict()
    d.update(top)
    return d


def _drop(d, key):
    del d[key]
    return d


def _with_bad_symbol_type():
    d = _current_dict()
    d["symbols"]["+"]["type"] = "bogus"
    return d


class TestSymbolLibrarySerialisation:
    def test_library_to_dict(self):
        lib = SymbolLibrary(["+"], 1, modules={"m": "math"})
        d = lib.to_dict()
        assert d == {
            "format_version": 1,
            "type": "SymbolLibrary",
            "symbols": {"+": lib["+"].to_dict(), "C": lib["C"].to_dict(), "X_0": lib["X_0"].to_dict()},
            "modules": {"m": "math"},
            "num_variables": 1,
        }
        assert list(d["symbols"]) == ["+", "C", "X_0"]
        assert json.loads(json.dumps(d)) == d

    def test_library_round_trip(self):
        lib = SymbolLibrary(["+", "sqrt"], 2, modules={"m": "math"})
        lib.add_symbol("cbrt", "fn", FN_PREFIX, "np.cbrt({})", unit_rule=UnitRule.power(Fraction(1, 3)))
        lib.add_symbol("g", "lit", LEAF, "9.81")
        lib.add_symbol("z", "var", LEAF, None)
        lib.remove_symbol("X_0")
        d = json.loads(json.dumps(lib.to_dict()))
        before = copy.deepcopy(d)
        restored = SymbolLibrary.from_dict(d)
        assert d == before
        assert restored.symbols == lib.symbols
        assert list(restored.symbols) == list(lib.symbols)
        assert (restored.modules, restored.num_variables) == ({"m": "math"}, 3)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: _drop(_current_dict(), "format_version"),
            _legacy_dict,
            lambda: _legacy_dict(modules=None),
            lambda: _legacy_dict(preamble=["  import numpy as np  "]),
            lambda: _legacy_dict(modules={}, preamble=["import os"]),
            lambda: _drop(_legacy_dict(), "preamble"),
        ],
        ids=[
            "no-format-version", "legacy-preamble", "modules-none", "preamble-whitespace", "preamble-ignored",
            "no-modules-no-preamble",
        ],
    )  # fmt: skip
    def test_from_dict_compat(self, make):
        lib = SymbolLibrary.from_dict(make())
        assert lib.modules == {}
        assert lib["sqrt"].unit_rule == UnitRule.power(Fraction(1, 2))
        assert lib["+"].unit_rule == UnitRule.SAME

    @pytest.mark.parametrize(
        "make, error, match",
        [
            (lambda: _current_dict(format_version=99), ValueError, "format_version"),
            (_dict_with_two_constants, ValueError, "At most one free constant"),
            (_with_bad_symbol_type, ValueError, "type"),
            (lambda: _current_dict(modules={"np": "numpy"}), ValueError, "np"),
            (lambda: _legacy_dict(preamble=["import numpy as np", "import math"]), ValueError, "import math"),
            (lambda: _current_dict(modules={"z": "no_such_module_xyz"}), ImportError, "no_such_module_xyz"),
            (lambda: _drop(_current_dict(), "symbols"), KeyError, "symbols"),
            (lambda: _drop(_current_dict(), "num_variables"), KeyError, "num_variables"),
        ],
        ids=[
            "format-version", "two-constants", "bad-symbol", "np-alias", "preamble-line", "unimportable-module",
            "no-symbols", "no-num_variables",
        ],
    )  # fmt: skip
    def test_from_dict_invalid_raises(self, make, error, match):
        with pytest.raises(error, match=match):
            SymbolLibrary.from_dict(make())


# ── Dunder methods ────────────────────────────────────────────────────────────


class TestSymbolLibraryLen:
    def test_returns_symbol_count(self):
        lib = SymbolLibrary(const_symbol=None)
        assert len(lib) == 0
        lib.add_symbol("x", "var", 0, "x")
        assert len(lib) == 1
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert len(lib) == 2
        lib.remove_symbol("x")
        assert len(lib) == 1


class TestSymbolLibraryStr:
    def test_returns_comma_separated_tokens(self):
        lib = SymbolLibrary(const_symbol=None)
        lib.add_symbol("x", "var", 0, "x")
        lib.add_symbol("+", "op", 0, "{} + {}")
        assert str(lib) == "x, +"

    def test_empty_library_returns_empty_string(self):
        assert str(SymbolLibrary(const_symbol=None)) == ""


class TestSymbolLibraryCopy:
    def test_copy_is_independent(self):
        original = SymbolLibrary(["+"], num_variables=1, modules={"m": "math"})
        copied = copy.copy(original)
        assert (copied.symbols, copied.modules, copied.num_variables) == (original.symbols, {"m": "math"}, 1)
        copied.add_symbol("sin", "fn", FN_PREFIX, "np.sin({})")
        copied.add_symbol("z", "var", LEAF, None)
        copied.modules["s"] = "statistics"
        assert (str(original), original.modules, original.num_variables) == ("+, C, X_0", {"m": "math"}, 1)
        assert (str(copied), copied.num_variables) == ("+, C, X_0, sin, z", 2)


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
