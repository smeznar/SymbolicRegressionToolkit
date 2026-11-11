"""Spec tests for SRToolkit.utils.units.

Expected values are derived from the docstrings and the SI definitions of the derived units, never
from the implementation.
"""

import copy
import json
import pickle
from fractions import Fraction as Fr

import pytest
from hypothesis import given
from hypothesis import strategies as st
from strategies import units

from SRToolkit.utils import units as u
from SRToolkit.utils.units import Unit, UnitRule

EXPONENTS = st.fractions(min_value=-3, max_value=3, max_denominator=4)
BIT = Unit({"bit": 1}, custom_units=["bit"])

# ── Unit: constructor and predefined units ───────────────────────────────────────────────────


class TestUnitConstructor:
    @pytest.mark.parametrize(
        "units_, custom, expected",
        [
            ({"m": 1, "s": -1}, (), {"m": Fr(1), "s": Fr(-1)}),
            ({"J": 1}, (), {"kg": Fr(1), "m": Fr(2), "s": Fr(-2)}),
            ({"N": 1, "m": 1}, (), {"kg": Fr(1), "m": Fr(2), "s": Fr(-2)}),
            ({"N": 1, "kg": -1}, (), {"m": Fr(1), "s": Fr(-2)}),
            ({"m": Fr(1, 2)}, (), {"m": Fr(1, 2)}),
            ({"m": 1, "s": 0}, (), {"m": Fr(1)}),
            ({"bit": 1, "s": -1}, ["bit"], {"bit": Fr(1), "s": Fr(-1)}),
            ({"bit": 1}, "bit", {"bit": Fr(1)}),
            ({"J": 1}, ["J"], {"kg": Fr(1), "m": Fr(2), "s": Fr(-2)}),
            ({"m": 1}, ["bit"], {"m": Fr(1)}),
            (None, (), {}),
            ({}, (), {}),
        ],
        ids=[
            "base",
            "derived-expands",
            "derived-and-base-add",
            "cancelled-dimension-dropped",
            "fraction",
            "zero-dropped",
            "custom-confirmed",
            "custom-single-string",
            "predefined-name-in-custom-stays-predefined",
            "unused-custom",
            "none",
            "empty",
        ],
    )
    def test_unit_constructor_table(self, units_, custom, expected):
        assert dict(Unit(units_, custom_units=custom).dimensions) == expected

    @pytest.mark.parametrize(
        "units_, custom, error, match",
        [
            ({"meter": 1}, (), ValueError, r"(?s)(?=.*meter)(?=.*kg)(?=.*Ohm)"),
            ({"bit": 1}, (), ValueError, r"(?s)(?=.*bit)(?=.*kg)(?=.*Ohm)"),
            ({"M": 1}, (), ValueError, r"(?s)(?=.*'M')(?=.*kg)(?=.*Ohm)"),
            ({}, ["1bit"], ValueError, "1bit"),
            ({}, ["a-b"], ValueError, "a-b"),
            ({}, [""], ValueError, None),
            ({}, [1, "a b"], ValueError, "a b"),
            ({"m": 0.5}, (), TypeError, None),
            ({"m": 2.0}, (), TypeError, None),
            ({"m": "2"}, (), TypeError, None),
        ],
        ids=[
            "unknown-name",
            "unconfirmed-custom",
            "wrong-case",
            "custom-leading-digit",
            "custom-hyphen",
            "custom-empty",
            "custom-mixed-types",
            "float-exponent",
            "integral-float-exponent",
            "str-exponent",
        ],
    )
    def test_unit_constructor_error_contract(self, units_, custom, error, match):
        with pytest.raises(error, match=match):
            Unit(units_, custom_units=custom)

    # Derived units from their SI definitions: N = kg m s^-2, J = N m, W = J/s, Pa = N/m^2,
    # Hz = 1/s, C = A s, V = W/A, Ohm = V/A, F = C/V, T = V s/m^2.
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("m", {"m": 1}),
            ("kg", {"kg": 1}),
            ("s", {"s": 1}),
            ("A", {"A": 1}),
            ("K", {"K": 1}),
            ("mol", {"mol": 1}),
            ("cd", {"cd": 1}),
            ("N", {"kg": 1, "m": 1, "s": -2}),
            ("J", {"kg": 1, "m": 2, "s": -2}),
            ("W", {"kg": 1, "m": 2, "s": -3}),
            ("Pa", {"kg": 1, "m": -1, "s": -2}),
            ("Hz", {"s": -1}),
            ("C", {"A": 1, "s": 1}),
            ("V", {"A": -1, "kg": 1, "m": 2, "s": -3}),
            ("Ohm", {"A": -2, "kg": 1, "m": 2, "s": -3}),
            ("F", {"A": 2, "kg": -1, "m": -2, "s": 4}),
            ("T", {"A": -1, "kg": 1, "s": -2}),
        ],
        ids=["m", "kg", "s", "A", "K", "mol", "cd", "N", "J", "W", "Pa", "Hz", "C", "V", "Ohm", "F", "T"],
    )
    def test_predefined_units(self, name, expected):
        constant = getattr(u, name)
        assert constant == Unit({name: 1})
        assert dict(constant.dimensions) == expected


# ── Unit: value semantics and arithmetic ─────────────────────────────────────────────────────


class TestUnitValue:
    def test_dimensions_are_name_ordered_and_read_only(self):
        unit = Unit({"s": 1, "m": 1, "kg": 1, "A": 1})
        assert list(unit.dimensions) == ["A", "kg", "m", "s"]
        assert all(type(e) is Fr for e in unit.dimensions.values())
        with pytest.raises(TypeError):
            unit.dimensions["m"] = Fr(2)
        v = u.m
        v *= u.s
        v /= u.s**2
        v **= 2
        assert u.m == Unit({"m": 1}) and u.s == Unit({"s": 1})

    @pytest.mark.parametrize(
        "make, attribute",
        [
            (lambda: u.m, "_dims"),
            (lambda: u.m, "foo"),
            (lambda: UnitRule.SAME, "kind"),
            (lambda: UnitRule.power(2), "exponent"),
        ],
        ids=["unit-private", "unit-new", "rule-kind", "rule-exponent"],
    )
    def test_assigning_an_attribute_raises(self, make, attribute):
        value = make()
        with pytest.raises(AttributeError):
            setattr(value, attribute, None)
        assert value == make()

    @pytest.mark.parametrize(
        "value",
        [u.m / u.s**2, BIT / u.s, UnitRule.power(Fr(1, 2)), UnitRule.SAME],
        ids=["unit", "custom-unit", "power-rule", "constant-rule"],
    )
    def test_units_and_rules_survive_pickle_and_deepcopy(self, value):
        assert copy.deepcopy(value) == value
        assert pickle.loads(pickle.dumps(value)) == value

    @pytest.mark.parametrize(
        "make, expected",
        [
            (lambda: u.m * u.m, {"m": Fr(2)}),
            (lambda: u.m / u.m, {}),
            (lambda: u.m ** Fr(1, 2), {"m": Fr(1, 2)}),
            (lambda: (u.m / u.s) ** -2, {"m": Fr(-2), "s": Fr(2)}),
            (lambda: u.m**0, {}),
            (lambda: Unit() ** 5, {}),
            (lambda: u.kg * u.m**2 / u.s**2, {"kg": Fr(1), "m": Fr(2), "s": Fr(-2)}),
        ],
        ids=["mul-same", "div-cancels", "pow-fraction", "pow-negative", "pow-zero", "dimensionless-pow", "energy"],
    )
    def test_unit_arithmetic_table(self, make, expected):
        assert dict(make().dimensions) == expected

    @pytest.mark.parametrize("exponent", [0.5, 2.0], ids=["0.5", "2.0"])
    def test_pow_rejects_float(self, exponent):
        with pytest.raises(TypeError):
            u.m**exponent

    @given(a=units(custom=True), b=units(custom=True), n=EXPONENTS, k=EXPONENTS)
    def test_unit_group_laws(self, a, b, n, k):
        assert a * b == b * a
        assert hash(a * b) == hash(b * a)
        assert (a * b) / b == a
        assert a**n * a**k == a ** (n + k)
        assert (a**n) ** k == a ** (n * k)
        assert a / a == Unit()

    def test_equality_ignores_construction_path(self):
        built = [
            Unit({"m": 1, "s": -1}),
            u.m / u.s,
            Unit({"s": -1, "m": 1}),
            Unit.from_dict({"s": -1, "m": 1}),
            Unit({"Hz": 1, "m": 1}),
        ]
        assert all(b == built[0] for b in built)
        assert len({hash(b) for b in built}) == 1
        assert u.N * u.m == u.J
        assert u.m / u.s != u.m
        assert Unit() == Unit({})
        assert u.m != "m"
        assert Unit() != {}


# ── Unit: text and saved forms ───────────────────────────────────────────────────────────────


class TestUnitForms:
    @pytest.mark.parametrize(
        "make, text",
        [
            (lambda: Unit(), "1"),
            (lambda: Unit({"s": -2, "kg": 1, "m": 2}), "kg m^2 s^-2"),
            (lambda: u.J, "kg m^2 s^-2"),
            (lambda: u.m ** Fr(-1, 2), "m^(-1/2)"),
            (lambda: u.m ** Fr(4, 2), "m^2"),
            (lambda: Unit({"s": 1, "mol": 1, "m": 1, "kg": 1, "cd": 1, "K": 1, "A": 1}), "A K cd kg m mol s"),
            (lambda: BIT / u.s, "bit s^-1"),
        ],
        ids=[
            "dimensionless",
            "name-order",
            "derived-as-base",
            "negative-fraction",
            "integral-fraction",
            "uppercase-first",
            "custom",
        ],
    )
    def test_str_and_repr(self, make, text):
        unit = make()
        assert str(unit) == text
        assert repr(unit) == f"Unit('{text}')"

    @pytest.mark.parametrize(
        "make, expected",
        [
            (lambda: Unit(), {}),
            (lambda: u.m / u.s, {"m": "1", "s": "-1"}),
            (lambda: u.m ** Fr(1, 2), {"m": "1/2"}),
            (lambda: u.m * u.A, {"A": "1", "m": "1"}),
            (lambda: u.W, {"kg": "1", "m": "2", "s": "-3"}),
            (lambda: BIT, {"bit": "1"}),
        ],
        ids=["dimensionless", "velocity", "fraction", "uppercase-first", "derived-as-base", "custom"],
    )
    def test_to_dict_exact_form(self, make, expected):
        d = make().to_dict()
        assert d == expected
        assert list(d) == list(expected)

    @given(unit=units(custom=True))
    def test_dict_json_round_trip(self, unit):
        assert Unit.from_dict(json.loads(json.dumps(unit.to_dict()))) == unit

    @pytest.mark.parametrize(
        "d, expected",
        [
            ({"m": 2}, {"m": Fr(2)}),
            ({"m": Fr(1, 2)}, {"m": Fr(1, 2)}),
            ({"m": "+2"}, {"m": Fr(2)}),
            ({"m": "-2"}, {"m": Fr(-2)}),
            ({"m": "1/2"}, {"m": Fr(1, 2)}),
            ({"m": "0.5"}, {"m": Fr(1, 2)}),
            ({"m": 0}, {}),
            ({"m": "0"}, {}),
            ({"bit": "1"}, {"bit": Fr(1)}),
            ({"J": 1}, {"kg": Fr(1), "m": Fr(2), "s": Fr(-2)}),
            ({}, {}),
        ],
        ids=[
            "int",
            "fraction",
            "plus-str",
            "minus-str",
            "fraction-str",
            "decimal-str",
            "zero",
            "zero-str",
            "custom-unconfirmed",
            "derived-expands",
            "empty",
        ],
    )
    def test_from_dict_exponent_forms(self, d, expected):
        assert dict(Unit.from_dict(d).dimensions) == expected

    @pytest.mark.parametrize(
        "d",
        [{"1bit": 1}, {"m": 0.5}, {"m": 1.0}, {"m": "(-1/2)"}, {"m": "x"}, {"m": "1/0"}],
        ids=["bad-name", "float", "integral-float", "bracketed-str", "word-str", "zero-denominator"],
    )
    def test_from_dict_error_contract(self, d):
        with pytest.raises(ValueError):
            Unit.from_dict(d)


# ── UnitRule ─────────────────────────────────────────────────────────────────────────────────


class TestUnitRule:
    @pytest.mark.parametrize(
        "rule, kind, exponent, arities",
        [
            (UnitRule.SAME, "same", None, (1, 2)),
            (UnitRule.MULTIPLY, "multiply", None, (2,)),
            (UnitRule.DIVIDE, "divide", None, (2,)),
            (UnitRule.EXPONENT, "exponent", None, (2,)),
            (UnitRule.DIMENSIONLESS, "dimensionless", None, (1, 2)),
            (UnitRule.power(2), "power", Fr(2), (1,)),
            (UnitRule.power(Fr(-1, 2)), "power", Fr(-1, 2), (1,)),
            (UnitRule.power(0), "power", Fr(0), (1,)),
        ],
        ids=["SAME", "MULTIPLY", "DIVIDE", "EXPONENT", "DIMENSIONLESS", "power-int", "power-fraction", "power-zero"],
    )
    def test_unit_rule_attributes(self, rule, kind, exponent, arities):
        assert (rule.kind, rule.exponent, rule.arities) == (kind, exponent, arities)
        assert exponent is None or type(rule.exponent) is Fr

    def test_unit_rule_equality_and_hash(self):
        assert UnitRule.power(2) == UnitRule.power(Fr(4, 2))
        assert hash(UnitRule.power(2)) == hash(UnitRule.power(Fr(4, 2)))
        assert {UnitRule.power(2): "sq"}[UnitRule.power(Fr(4, 2))] == "sq"
        assert UnitRule.power(1) != UnitRule.SAME
        assert UnitRule.power(2) != UnitRule.power(3)
        assert UnitRule.SAME != UnitRule.DIMENSIONLESS

    @pytest.mark.parametrize("k", [0.5, 2.0], ids=["0.5", "2.0"])
    def test_power_rejects_float(self, k):
        with pytest.raises(TypeError):
            UnitRule.power(k)

    @pytest.mark.parametrize(
        "rule, text",
        [
            (UnitRule.SAME, "UnitRule.SAME"),
            (UnitRule.MULTIPLY, "UnitRule.MULTIPLY"),
            (UnitRule.DIVIDE, "UnitRule.DIVIDE"),
            (UnitRule.EXPONENT, "UnitRule.EXPONENT"),
            (UnitRule.DIMENSIONLESS, "UnitRule.DIMENSIONLESS"),
            (UnitRule.power(2), "UnitRule.power(2)"),
            (UnitRule.power(-1), "UnitRule.power(-1)"),
            (UnitRule.power(Fr(1, 2)), "UnitRule.power(Fraction(1, 2))"),
            (UnitRule.power(Fr(-1, 3)), "UnitRule.power(Fraction(-1, 3))"),
        ],
        ids=[
            "SAME",
            "MULTIPLY",
            "DIVIDE",
            "EXPONENT",
            "DIMENSIONLESS",
            "power-2",
            "power-minus-1",
            "power-half",
            "power-minus-third",
        ],
    )
    def test_unit_rule_repr(self, rule, text):
        assert repr(rule) == text
        assert eval(text, {"UnitRule": UnitRule, "Fraction": Fr}) == rule

    @pytest.mark.parametrize(
        "rule, expected",
        [
            (UnitRule.SAME, {"kind": "same"}),
            (UnitRule.MULTIPLY, {"kind": "multiply"}),
            (UnitRule.DIVIDE, {"kind": "divide"}),
            (UnitRule.EXPONENT, {"kind": "exponent"}),
            (UnitRule.DIMENSIONLESS, {"kind": "dimensionless"}),
            (UnitRule.power(Fr(1, 2)), {"kind": "power", "exponent": "1/2"}),
            (UnitRule.power(-1), {"kind": "power", "exponent": "-1"}),
        ],
        ids=["SAME", "MULTIPLY", "DIVIDE", "EXPONENT", "DIMENSIONLESS", "power-half", "power-minus-1"],
    )
    def test_unit_rule_to_dict_and_round_trip(self, rule, expected):
        assert rule.to_dict() == expected
        assert UnitRule.from_dict(json.loads(json.dumps(rule.to_dict()))) == rule

    @pytest.mark.parametrize(
        "d, expected",
        [
            ({"kind": "power", "exponent": "0.5"}, UnitRule.power(Fr(1, 2))),
            ({"kind": "power", "exponent": 2}, UnitRule.power(2)),
            ({"kind": "same", "exponent": "2", "x": 1}, UnitRule.SAME),
        ],
        ids=["decimal-str-exponent", "int-exponent", "other-keys-ignored"],
    )
    def test_unit_rule_from_dict_forms(self, d, expected):
        assert UnitRule.from_dict(d) == expected

    @pytest.mark.parametrize(
        "d",
        [
            {},
            {"kind": "Same"},
            {"kind": "nope"},
            {"kind": "power"},
            {"kind": "power", "exponent": "(-1/2)"},
            {"kind": "power", "exponent": 0.5},
        ],
        ids=["no-kind", "wrong-case", "unknown-kind", "power-no-exponent", "power-bracketed-str", "power-float"],
    )
    def test_unit_rule_from_dict_error_contract(self, d):
        with pytest.raises(ValueError):
            UnitRule.from_dict(d)
