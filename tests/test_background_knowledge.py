"""Tests for SRToolkit.dataset.background_knowledge (spec-first; see the module docstrings)."""

import dataclasses
import json
import logging

import numpy as np
import pytest

from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTreeDepth, NoNesting, RequiredVariables
from SRToolkit.utils.grammar.constraints import MaxDepth
from SRToolkit.utils.grammar.grammar import Grammar, Rule
from SRToolkit.utils.symbol_library import SymbolLibrary


def _lib(num_variables=2, symbols=("+", "*", "C")):
    return SymbolLibrary.from_symbol_list(list(symbols), num_variables=num_variables)


def _grammar():
    return Grammar(Rule.from_line("E -> E '+' F [0.4] | F [0.6]") + Rule.from_line("F -> 'X_0' [1.0]"))


# ── VariableInfo ──────────────────────────────────────────────────────────────


class TestVariableInfo:
    @pytest.mark.parametrize(
        "d, expected",
        [
            ({"name": "v", "description": "speed of the particle"}, VariableInfo("v", "speed of the particle")),
            ({}, VariableInfo()),
            ({"name": "v", "unit": "m"}, VariableInfo("v")),
        ],
        ids=["saved", "missing-keys-default", "unknown-key-ignored"],
    )
    def test_variable_info_dict_round_trip(self, d, expected):
        assert VariableInfo.from_dict(d) == expected
        assert VariableInfo("v", "speed of the particle").to_dict() == {
            "name": "v",
            "description": "speed of the particle",
        }

    @pytest.mark.parametrize(
        "name, description, expected",
        [("v", "speed", "v: speed"), ("v", "", "v"), ("", "speed", "speed"), ("", "", "")],
        ids=["both", "name-only", "description-only", "empty"],
    )
    def test_variable_info_describe(self, name, description, expected):
        assert VariableInfo(name, description).describe() == expected

    def test_variable_info_equality_and_hash(self):
        assert VariableInfo("a", "b") == VariableInfo("a", "b")
        assert hash(VariableInfo("a", "b")) == hash(VariableInfo("a", "b"))
        assert {VariableInfo("a", "b"): 1}[VariableInfo("a", "b")] == 1
        assert VariableInfo("a", "b") != VariableInfo("a", "c")

    @pytest.mark.parametrize(
        "make",
        [
            lambda: VariableInfo(1),
            lambda: VariableInfo("v", None),
            lambda: VariableInfo(b"x"),
            lambda: VariableInfo.from_dict({"name": 3}),
        ],
        ids=["int-name", "none-description", "bytes-name", "from_dict"],
    )
    def test_variable_info_rejects_non_str(self, make):
        with pytest.raises(TypeError):
            make()


@pytest.mark.parametrize(
    "obj, field",
    [(VariableInfo("v"), "name"), (BackgroundKnowledge(_lib()), "description")],
    ids=["VariableInfo", "BackgroundKnowledge"],
)
def test_instances_are_frozen(obj, field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, "x")


# ── BackgroundKnowledge: construction ─────────────────────────────────────────


class TestBackgroundKnowledgeInit:
    def test_fields_are_normalised(self):
        s0, s1 = UniformSampling(0, 1), UniformSampling(2, 3)
        bk = BackgroundKnowledge(
            _lib(),
            samplers={"X_1": s1, "X_0": s0},
            constant_bounds=(-1, np.int64(2)),
            constraints=(c for c in [MaxTreeDepth(4)]),
        )
        assert isinstance(bk.samplers, tuple)
        assert bk.samplers[0] is s0 and bk.samplers[1] is s1
        assert bk.constant_bounds == (-1.0, 2.0)
        assert type(bk.constant_bounds[0]) is float and type(bk.constant_bounds[1]) is float
        assert isinstance(bk.constraints, tuple) and len(bk.constraints) == 1
        assert BackgroundKnowledge(_lib()).constant_bounds == (-5.0, 5.0)

    def test_zero_variable_library_with_empty_samplers(self):
        assert BackgroundKnowledge(_lib(num_variables=0), samplers=[]).samplers == ()

    def test_variables_are_a_copy(self):
        variables = {"X_0": VariableInfo("x")}
        bk = BackgroundKnowledge(_lib(), variables=variables)
        variables["X_1"] = VariableInfo("y")
        assert bk.variables == {"X_0": VariableInfo("x")}

    def test_constraints_are_set_up_copies(self):
        given = MaxTreeDepth(4)
        lib = _lib()
        bk = BackgroundKnowledge(lib, constraints=[given])
        (stored,) = bk.constraints
        assert stored is not given and stored.to_dict() == given.to_dict()
        assert stored.symbol_library is lib
        assert given.symbol_library is None
        assert stored.pre_check(["X_0"]).passed

    def test_constraint_that_does_not_fit_library_is_rejected(self):
        with pytest.raises(ValueError, match="X_9"):
            BackgroundKnowledge(_lib(), constraints=[RequiredVariables("X_9")])

    def test_grammar_with_unknown_token_rejected(self):
        with pytest.raises(ValueError, match="Z"):
            BackgroundKnowledge(_lib(), reference_grammar=Grammar(Rule.from_line("E -> 'Z' [1.0]")))

    def test_grammar_with_symbols_brackets_and_numbers_accepted(self):
        grammar = Grammar(Rule.from_line("E -> '(' 'X_0' '+' '2.5' ')' [1.0]"), start="E")
        assert BackgroundKnowledge(_lib(), reference_grammar=grammar).reference_grammar is grammar

    def test_sampling_grammar_is_reference_grammar(self, caplog):
        grammar = Grammar(Rule.from_line("E -> 'X_0' [1.0]"), start="E")
        bk = BackgroundKnowledge(_lib(), reference_grammar=grammar)
        with caplog.at_level(logging.INFO, logger="SRToolkit.dataset.background_knowledge"):
            assert bk.sampling_grammar() is grammar
        assert [r.levelno for r in caplog.records if r.name == "SRToolkit.dataset.background_knowledge"] == [
            logging.INFO
        ]

    def test_sampling_grammar_defaults_to_library_grammar(self):
        lib = _lib()
        assert BackgroundKnowledge(lib).sampling_grammar().to_dict() == Grammar.from_symbol_library(lib).to_dict()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"symbol_library": "x"}, None),
            ({"samplers": [UniformSampling(0, 1), "s"]}, None),
            ({"constraints": MaxTreeDepth(4)}, None),
            ({"constraints": "ab"}, None),
            ({"constraints": 5}, None),
            ({"constraints": [MaxDepth(3)]}, None),
            ({"reference_grammar": "g"}, None),
            ({"variables": {"X_0": "x"}}, None),
            ({"target": "U"}, None),
            ({"variables": [("X_0", VariableInfo())]}, None),
            ({"description": 3}, None),
            ({"constant_bounds": (True, 5)}, None),
            ({"constant_bounds": ("a", 5)}, None),
        ],
        ids=[
            "library-str",
            "sampler-str",
            "lone-constraint",
            "constraints-str",
            "constraints-not-iterable",
            "grammar-constraint",
            "grammar-str",
            "variable-not-info",
            "target-str",
            "variables-not-mapping",
            "description-int",
            "bound-bool",
            "bound-str",
        ],
    )
    def test_constructor_type_errors(self, kwargs, match):
        args = {"symbol_library": _lib(), **kwargs}
        with pytest.raises(TypeError, match=match):
            BackgroundKnowledge(**args)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"samplers": [UniformSampling(0, 1)]}, None),
            ({"samplers": {"X_0": UniformSampling(0, 1)}}, "X_1"),
            (
                {
                    "samplers": {
                        "X_0": UniformSampling(0, 1),
                        "X_1": UniformSampling(0, 1),
                        "X_9": UniformSampling(0, 1),
                    }
                },
                "X_9",
            ),
            ({"constant_bounds": (5, -5)}, None),
            ({"constant_bounds": (1, 1)}, None),
            ({"constant_bounds": (float("nan"), 1)}, None),
            ({"constant_bounds": (float("-inf"), 1)}, None),
            ({"constant_bounds": (1, 2, 3)}, None),
            ({"constant_bounds": 5}, None),
            ({"variables": {"X_9": VariableInfo("x")}}, "X_9"),
            ({"variables": {"+": VariableInfo("plus")}}, None),
        ],
        ids=[
            "too-few-samplers",
            "mapping-missing",
            "mapping-unknown",
            "bounds-reversed",
            "bounds-equal",
            "bound-nan",
            "bound-inf",
            "bounds-not-pair",
            "bounds-not-iterable",
            "variable-unknown",
            "variable-not-var",
        ],
    )
    def test_constructor_value_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BackgroundKnowledge(_lib(), **kwargs)


# ── BackgroundKnowledge: describe ─────────────────────────────────────────────


class TestBackgroundKnowledgeDescribe:
    def test_describe_full_layout(self):
        bk = BackgroundKnowledge(
            _lib(),
            samplers=[UniformSampling(0.0, 5.0), UniformSampling(1.0, 2.0)],
            constant_bounds=(-2, 3),
            constraints=[MaxTreeDepth(4)],
            reference_grammar=_grammar(),
            description="Voltage measured across metal conductors of different resistance, each carrying a steady current, at room temperature. Without current there is no voltage; the voltage grows with the current and with the resistance.",
            variables={"X_1": VariableInfo("R", "resistance"), "X_0": VariableInfo("I")},
            target=VariableInfo("U", "voltage"),
        )
        assert bk.describe([MaxConstants(2)]) == (
            "Task: Voltage measured across metal conductors of different resistance, each carrying a steady current, at room temperature. Without current there is no voltage; the voltage grows with the current and with the resistance.\n"
            "Target: U: voltage\n"
            "Variables:\n"
            "  X_0: I\n"
            "  X_1: R: resistance\n"
            "Symbols: +, *, C, X_0, X_1\n"
            "Free constants are fitted in [-2.0, 3.0].\n"
            "Input distribution:\n"
            "  X_0: uniform on [-5, 5]\n"
            "  X_1: uniform on [-2, -1] ∪ [1, 2]\n"
            "Enforced constraints (expressions that break them are rejected):\n"
            "  - at most 2 free constants\n"
            "Hints (advisory, may be ignored):\n"
            "  - expression tree at most 4 levels deep\n"
            "Reference grammar: given"
        )

    def test_priors_are_hint_summaries_in_order(self):
        # Several constraints of one class are all kept: no merging by kind.
        hints = [MaxTreeDepth(4), NoNesting(["sin", "cos"]), NoNesting(["exp", "ln"])]
        bk = BackgroundKnowledge(_lib(), constraints=hints)
        assert bk.priors() == [c.summary() for c in hints]
        assert BackgroundKnowledge(_lib()).priors() == []

    @pytest.mark.parametrize(
        "bk, expected",
        [
            (lambda: BackgroundKnowledge(_lib(1, ("+", "*"))), "Symbols: +, *, X_0"),
            (lambda: BackgroundKnowledge(_lib(1)), "Symbols: +, *, C, X_0\nFree constants are fitted in [-5.0, 5.0]."),
            (
                lambda: BackgroundKnowledge(
                    _lib(1, ("+", "*")), variables={"X_0": VariableInfo()}, target=VariableInfo()
                ),
                "Variables:\n  X_0\nSymbols: +, *, X_0",
            ),
            (
                lambda: BackgroundKnowledge(_lib(1, ("+", "*")), samplers=[UniformSampling(0.0, 1.0)]),
                "Symbols: +, *, X_0\nInput distribution:\n  X_0: uniform on [-1, 1]",
            ),
            (lambda: BackgroundKnowledge(_lib(0, ("+", "*")), samplers=[]), "Symbols: +, *"),
        ],
        ids=["library-only", "constants-default-bounds", "empty-infos", "one-sampler", "no-variables"],
    )
    def test_describe_omits_empty_sections(self, bk, expected):
        assert bk().describe() == expected


# ── BackgroundKnowledge: serialisation ────────────────────────────────────────


class TestBackgroundKnowledgeSerialisation:
    def test_to_dict_layout(self):
        lib = _lib()
        samplers = [UniformSampling(0.0, 5.0), UniformSampling(1.0, 2.0)]
        bk = BackgroundKnowledge(
            lib,
            samplers=samplers,
            constant_bounds=(-2, 3),
            constraints=[MaxTreeDepth(4)],
            description="Voltage measured across metal conductors of different resistance, each carrying a steady current, at room temperature. Without current there is no voltage; the voltage grows with the current and with the resistance.",
            variables={"X_0": VariableInfo("I", "current")},
            target=VariableInfo("U", "voltage"),
        )
        d = bk.to_dict()
        assert d == {
            "format_version": 1,
            "symbol_library": lib.to_dict(),
            "samplers": [samplers[0].to_dict(), samplers[1].to_dict()],
            "constant_bounds": [-2.0, 3.0],
            "constraints": [MaxTreeDepth(4).to_dict()],
            "reference_grammar": None,
            "description": "Voltage measured across metal conductors of different resistance, each carrying a steady current, at room temperature. Without current there is no voltage; the voltage grows with the current and with the resistance.",
            "variables": {"X_0": {"name": "I", "description": "current"}},
            "target": {"name": "U", "description": "voltage"},
        }
        json.dumps(d)

    def test_from_dict_round_trip(self):
        bk = BackgroundKnowledge(
            _lib(),
            samplers=[UniformSampling(0.0, 5.0), UniformSampling(1.0, 2.0)],
            constant_bounds=(-2, 3),
            constraints=[MaxTreeDepth(4)],
            reference_grammar=_grammar(),
            description="Voltage measured across metal conductors of different resistance, each carrying a steady current, at room temperature. Without current there is no voltage; the voltage grows with the current and with the resistance.",
            variables={"X_0": VariableInfo("I", "current")},
            target=VariableInfo("U", "voltage"),
        )
        d = json.loads(json.dumps(bk.to_dict()))
        restored = BackgroundKnowledge.from_dict(d)
        assert restored.reference_grammar.to_dict() == bk.reference_grammar.to_dict()
        assert restored.to_dict() == d

    def test_from_dict_fills_defaults(self):
        # No "format_version" key either: it is read as 1.
        lib = _lib()
        loaded = BackgroundKnowledge.from_dict({"symbol_library": lib.to_dict(), "extra": 1})
        assert loaded.to_dict() == BackgroundKnowledge(lib).to_dict()

    @pytest.mark.parametrize(
        "change, exc",
        [
            (lambda d: d.pop("symbol_library"), KeyError),
            (lambda d: d.update(format_version=2), ValueError),
            (lambda d: d["samplers"][0].update(sampler_class="nope.Sampler"), ImportError),
            (lambda d: d["constraints"][0].update(constraint_class="nope.Constraint"), ImportError),
            (lambda d: d.update(constant_bounds=[5, -5]), ValueError),
        ],
        ids=["no-library", "format-version", "sampler-class", "constraint-class", "bad-bounds"],
    )
    def test_from_dict_errors(self, change, exc):
        d = BackgroundKnowledge(
            _lib(),
            samplers=[UniformSampling(0, 1), UniformSampling(0, 1)],
            constraints=[MaxTreeDepth(4)],
        ).to_dict()
        change(d)
        with pytest.raises(exc):
            BackgroundKnowledge.from_dict(d)
