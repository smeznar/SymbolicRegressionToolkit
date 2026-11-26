"""Spec tests for SRToolkit.utils.expression_simplifier (e-graph simplification)."""

import json

import numpy as np
import pytest
from egglog import i64, rewrite, ruleset, var, vars_
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import labelled_infix_expressions

from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_simplifier import (
    RELATIONS,
    ConstTree,
    DefaultRules,
    EggExpr,
    SimplificationRules,
    fold,
    no_const,
    simplify,
)
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, parameter_labels, tokens_to_tree
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.symbol_library import SymbolLibrary

_SL = SymbolLibrary.default_symbols(2)
_X = np.random.default_rng(0).uniform(-3.0, 3.0, size=(20, 2))


def _t(s):
    """Space-separated infix string to a token list."""
    return s.split()


def _s(expr, **kwargs):
    """Simplified expression (default library) joined into one string."""
    return "".join(simplify(_t(expr), _SL, **kwargs)[0])


def _tree(tokens, sl=_SL):
    """Tokens written out from their tree, so redundant brackets do not matter."""
    return "".join(tokens_to_tree(tokens, sl).to_list(sl))


def _cost(tokens):
    return len([t for t in tokens if t not in ("(", ")")])


def _values(tokens, label_values):
    """Evaluate ``tokens`` on ``_X``, each label read from ``label_values``."""
    params = np.array([label_values[label] for label in parameter_labels(tokens, _SL)], dtype=float)
    with np.errstate(all="ignore"):
        return np.broadcast_to(compile_expr(tokens, _SL)(_X, params), (_X.shape[0],))


class _CancelWithoutConst(SimplificationRules):
    """Custom rules using RELATIONS: ``a - a -> 0`` only for ``a`` without constants."""

    def build(self, symbol_library):
        (a,) = vars_("a", EggExpr)
        return RELATIONS.saturate() + ruleset(
            rewrite(EggExpr.op("-", a, a)).to(EggExpr.c(ConstTree.leaf(-1, 0.0, 0.0)), no_const(a))
        )

    def to_dict(self):
        return {"rules_class": f"{type(self).__module__}.{type(self).__qualname__}"}


class _OnlyCancel(SimplificationRules):
    """Custom rules with a single rule ``a - a -> 0``; does not override ``from_dict``."""

    def build(self, symbol_library):
        (a,) = vars_("a", EggExpr)
        return ruleset(rewrite(EggExpr.op("-", a, a)).to(EggExpr.c(ConstTree.leaf(-1, 0.0, 0.0))))

    def to_dict(self):
        return {"rules_class": f"{type(self).__module__}.{type(self).__qualname__}"}


class _Tuned(_OnlyCancel):
    """Custom rules with two settings, for ``__repr__``."""

    def to_dict(self):
        return {"rules_class": "tests._Tuned", "depth": 2, "name": "x"}


class _DropUnitBounded(_OnlyCancel):
    """Custom rules that read a constant's bounds: ``c*a -> a`` for a constant with bounds (0, 1)."""

    def build(self, symbol_library):
        (a,) = vars_("a", EggExpr)
        i = var("i", i64)
        return ruleset(rewrite(EggExpr.op("*", EggExpr.c(ConstTree.leaf(i, 0.0, 1.0)), a)).to(a))


def _lib(change):
    sl = SymbolLibrary.default_symbols(1)
    change(sl)
    return sl


class TestSimplifyRules:
    # Reason: clause (DefaultRules groups "identities", "inverses", "powers", "like_terms"; the
    # cheapest equivalent is returned). Expected values by hand.
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("X_0 * X_0 / X_0", {"X_0"}),
            ("X_0 - X_0", {"0"}),
            ("sin ( X_0 ) / sin ( X_0 )", {"1"}),
            ("X_0 + 0", {"X_0"}),
            ("X_0 * 1", {"X_0"}),
            ("X_0 / 1", {"X_0"}),
            ("X_0 * 0", {"0"}),
            ("X_0 * cos ( 0 )", {"X_0"}),
            ("ln ( 1 )", {"0"}),
            ("sin ( 0 )", {"0"}),
            ("ln ( exp ( X_0 ) )", {"X_0"}),
            ("exp ( ln ( X_0 ) )", {"X_0"}),
            ("sqrt ( X_0 ) ^2", {"X_0"}),
            ("X_0 * X_0", {"X_0^2"}),
            ("X_0 * X_0 * X_0", {"X_0^3"}),
            ("X_0 + X_0 + X_0", {"3*X_0", "X_0*3"}),
            ("0 / X_0", {"0"}),
            ("0 / C", {"0"}),
            ("0 / ( X_0 + X_1 )", {"0"}),
        ],
        ids=[
            "x*x/x", "x-x", "sin/sin", "+0", "*1", "/1", "*0", "*cos0", "ln1", "sin0", "ln-exp",
            "exp-ln", "sqrt^2", "x*x", "x*x*x", "x+x+x", "0/x", "0/c", "0/sum",
        ],
    )  # fmt: skip
    def test_simplify_algebra_cases(self, expr, expected):
        assert _s(expr) in expected

    # Reason: clause ("Each bare C is a value of its own"): bare constants never cancel or merge.
    @pytest.mark.parametrize(
        "expr, n_bare, allowed",
        [
            ("C - C", 2, None),
            ("( C + C ) - ( C + C )", 4, None),
            ("C * C * X_0", 2, None),
            ("C + C", 2, {"C+C"}),
            ("C * X_0 - C * X_0", 2, {"(C-C)*X_0", "X_0*(C-C)"}),
        ],
        ids=["c-c", "sum-minus-sum", "c*c*x", "c+c", "cx-cx"],
    )
    def test_bare_constants_are_separate_values(self, expr, n_bare, allowed):
        out = simplify(_t(expr), _SL)[0]
        assert out.count("C") == n_bare
        assert "0" not in out
        assert allowed is None or "".join(out) in allowed

    # Reason: clause ("every occurrence of a label is the same value, so C_1 - C_1 becomes 0").
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("C_1 - C_1", "0"),
            ("C_1 / C_1", "1"),
            ("C_1 + C_2 - C_1", "C_2"),
            ("C_1 * C_2 / C_1", "C_2"),
        ],
        ids=["c1-c1", "c1/c1", "c1+c2-c1", "c1*c2/c1"],
    )
    def test_labels_are_one_value(self, expr, expected):
        assert _s(expr) == expected

    # Reason: clause ("Fixed values combined ... become one number (C + 2 + 3 becomes 5+C)"; a free
    # constant is kept beside them; a cancelled part leaves only the number).
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("C + 2 + 3", {"C+5", "5+C"}),
            ("C * 2 * 3", {"C*6", "6*C"}),
            ("C_1 * X_0 - C_1 * X_0 + 2 * 3", {"6"}),
        ],
        ids=["sum", "product", "cancelled-part"],
    )
    def test_fixed_values_combine_beside_free_constant(self, expr, expected):
        assert _s(expr) in expected

    # Reason: clause (combined "unless the result is not finite or undefined": 1/0, 0/0, sqrt(-1)
    # and exp(1000) stay; the finite inner 0-1 still becomes -1). Regression: 0/0 became 1 (a/a)
    # or 0 (0/a), also when the zero is found by another rule (X_0-X_0).
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("exp ( 1000 )", "exp(1000)"),
            ("1 / 0", "1/0"),
            ("sqrt ( 0 - 1 )", "sqrt(-1)"),
            ("0 / 0", "0/0"),
            ("( X_0 - X_0 ) / ( X_0 - X_0 )", "0/0"),
        ],
        ids=["overflow", "division-by-zero", "sqrt-negative", "zero-by-zero", "found-zero-by-zero"],
    )
    def test_non_finite_combinations_stay(self, expr, expected):
        assert _s(expr) == expected

    # Reason: clause ("every lit symbol of the library, with the value of its python_str"): k = 2.5.
    def test_custom_lit_is_fixed_value(self):
        sl = _lib(lambda sl: sl.add_symbol("k", "lit", sl["pi"].precedence, "2.5"))
        assert simplify(_t("2 * k"), sl)[0] == ["5"]

    # Reason: clause (fixed values, pi and e combined by *, /, sqrt, exp, ^2, ^3; settled: whole
    # numbers as integer tokens, else repr at full precision). Values: 2*pi, sqrt(4)=2, exp(0)=1,
    # e^2, 2/4, 3^3=27.
    @pytest.mark.parametrize(
        "expr, value",
        [
            ("2 * pi", 6.283185307179586),
            ("sqrt ( 4 )", 2.0),
            ("exp ( 0 )", 1.0),
            ("e ^2", 7.38905609893065),
            ("2 / 4", 0.5),
            ("3 ^3", 27.0),
        ],
        ids=["2pi", "sqrt4", "exp0", "e^2", "2/4", "3^3"],
    )
    def test_fixed_values_combine_into_numbers(self, expr, value):
        out = simplify(_t(expr), _SL)[0]
        assert len(out) == 1
        assert float(out[0]) == pytest.approx(value, rel=1e-12)

    # Reason: clause ("Tokens without rules are left as they are, their arguments are still simplified").
    @pytest.mark.parametrize(
        "expr, expected",
        [("sin ( 2 )", "sin(2)"), ("tan ( X_0 * X_0 / X_0 )", "tan(X_0)")],
        ids=["sin2-no-rule", "tan-arg-simplified"],
    )
    def test_tokens_without_rules_are_kept(self, expr, expected):
        assert _s(expr) == expected


class TestSimplifySpecs:
    # Reason: clause ("a fixed one is used as its value"; specs returned only for labels still
    # contained). C_1 = 2 fixed: 2*3 = 6; free: nothing to combine.
    def test_fixed_spec_is_used_as_value(self):
        fixed = Constant("C_1", value=2.0, fixed=True)
        assert simplify(_t("C_1 * 3"), _SL, [fixed]) == (["6"], ())
        free = Constant("C_1", value=2.0)
        out, specs = simplify(_t("C_1 * 3"), _SL, [free])
        assert "".join(out) in {"C_1*3", "3*C_1"}
        assert specs == (free,)

    # Reason: clause (Returns: "the specs of free_constants whose label it still contains").
    def test_returns_specs_of_remaining_labels(self):
        s1, s2 = Constant("C_1", value=1.0), Constant("C_2", value=2.0)
        out, specs = simplify(_t("C_1 + C_2 - C_1"), _SL, [s1, s2])
        assert out == ["C_2"]
        assert specs == (s2,)
        # labels are those of the given library: with the constant renamed to K, K_2 keeps its spec
        k1, k2 = Constant("K_1", value=1.0), Constant("K_2", value=2.0)
        renamed = SymbolLibrary.default_symbols(1, const_symbol="K")
        assert simplify(_t("K_1 + K_2 - K_1"), renamed, [k1, k2]) == (["K_2"], (k2,))

    # Reason: boundary (free_constants None -> None; empty -> empty tuple).
    def test_specs_none_or_empty(self):
        assert simplify(["X_0"], _SL, None)[1] is None
        assert simplify(["X_0"], _SL, [])[1] == ()

    # Reason: clause (Raises ValueError: spec without label, two share one, label not a labelled
    # constant of expr).
    @pytest.mark.parametrize(
        "specs, match",
        [
            ([Constant(value=1.0)], "label"),
            ([Constant("C_1", value=1.0), Constant("C_1", value=2.0)], "C_1"),
            ([Constant("C_9", value=1.0)], "C_9"),
            ([Constant("X_0", value=1.0)], "X_0"),
        ],
        ids=["no-label", "duplicate", "not-in-expr", "variable-label"],
    )
    def test_invalid_specs_raise(self, specs, match):
        with pytest.raises(ValueError, match=match):
            simplify(_t("C_1 * X_0"), _SL, specs)


class TestSimplifyInterface:
    # Reason: clause (Raises TypeError: expr neither a token list nor a Node).
    @pytest.mark.parametrize("expr", ["X_0+X_1", 42, None], ids=["string", "int", "none"])
    def test_wrong_expr_type_raises(self, expr):
        with pytest.raises(TypeError):
            simplify(expr, _SL)

    # Reason: clause (Raises ExpressionParseError: a token list cannot be parsed).
    @pytest.mark.parametrize("expr", [["X_0", "+"], ["(", "X_0"]], ids=["missing-operand", "open-bracket"])
    def test_unparseable_tokens_raise(self, expr):
        with pytest.raises(ExpressionParseError):
            simplify(expr, _SL)

    # Reason: clause (Returns: "in the form of expr (token list or Node)").
    def test_output_has_form_of_input(self):
        assert simplify(_t("X_0 * X_1 / X_0"), _SL)[0] == ["X_1"]
        out = simplify(tokens_to_tree(_t("X_0 * X_1 / X_0"), _SL), _SL)[0]
        assert isinstance(out, Node)
        assert out.to_list(_SL) == ["X_1"]

    # Reason: implied (the input is not mutated).
    def test_input_not_mutated(self):
        tokens = _t("X_0 * X_1 / X_0")
        simplify(tokens, _SL)
        assert tokens == _t("X_0 * X_1 / X_0")
        tree = tokens_to_tree(tokens, _SL)
        written = tree.to_list(_SL)
        simplify(tree, _SL)
        assert tree.to_list(_SL) == written

    # Reason: clause (symbol_library None "uses the active or default library"): the active library
    # has no ^2, so X_0*X_0 is not written as X_0^2 as with the default library.
    def test_none_library_uses_active(self):
        with SymbolLibrary.from_symbol_list(["+", "-", "*", "/"], num_variables=1):
            assert "".join(simplify(_t("X_0 * X_0"))[0]) == "X_0*X_0"


class TestSimplifyProperties:
    # Reason: implied (the output parses with the library) + clause (no inf/nan number is written:
    # only finite results are combined) + clause (settled: cost = written-out
    # tokens without brackets, the output is never longer than the input).
    @settings(max_examples=25, deadline=None)
    @given(labelled_infix_expressions())
    def test_output_valid_and_not_longer(self, expr):
        out = simplify(expr, _SL)[0]
        tokens_to_tree(out, _SL)
        assert _cost(out) <= _cost(expr)
        assert not {"inf", "-inf", "nan"} & set(out)

    # Reason: clause ("returns the cheapest equivalent expression"): equal values wherever the input
    # is defined, for labels as free values and as fixed specs (same values by name).
    @settings(max_examples=25, deadline=None)
    @given(labelled_infix_expressions().filter(lambda e: "C" not in e), st.booleans())
    def test_output_numerically_equal(self, expr, fix):
        values = {"C_0": 0.7, "C_1": 1.3, "C_7": -2.1}
        specs = [Constant(label, value=values[label], fixed=True) for label in set(parameter_labels(expr, _SL))]
        out = simplify(expr, _SL, specs if fix else None)[0]
        before, after = _values(expr, values), _values(out, values)
        defined = np.isfinite(before)
        np.testing.assert_allclose(after[defined], before[defined], rtol=1e-7, atol=1e-9)


class TestRuleAvailability:
    # Reason: clause ("A rule is used only if every token it relies on is in the symbol library with
    # the type and python_str of the predefined symbol").
    @pytest.mark.parametrize(
        "change, expr, expected",
        [
            (lambda sl: sl.remove_symbol("^2"), "X_0 * X_0", {"X_0 * X_0"}),
            (lambda sl: sl.add_symbol("^2", "fn", sl["^2"].precedence, "{}**3"), "X_0 * X_0", {"X_0 * X_0"}),
            (lambda sl: sl.add_symbol("ln", "fn", sl["ln"].precedence, "np.log10({})"), "ln ( exp ( X_0 ) )",
             {"ln ( exp ( X_0 ) )"}),
            (lambda sl: sl.add_symbol("sqrt", "fn", sl["sqrt"].precedence, "np.cbrt({})"), "sqrt ( X_0 ) ^2",
             {"sqrt ( X_0 ) ^2"}),
        ],
        ids=["no-^2", "custom-^2", "custom-ln", "custom-sqrt"],
    )  # fmt: skip
    def test_rules_off_for_missing_or_redefined_tokens(self, change, expr, expected):
        sl = _lib(change)
        assert _tree(simplify(_t(expr), sl)[0], sl) in {_tree(_t(e), sl) for e in expected}

    # Reason: clause (DefaultRules(exclude=...) leaves the named groups out).
    @pytest.mark.parametrize(
        "group, expr, expected",
        [
            ("powers", "X_0 * X_0", {"X_0*X_0"}),
            ("inverses", "ln ( exp ( X_0 ) )", {"ln(exp(X_0))"}),
            ("fixed_arithmetic", "2 * 3", {"2*3", "3*2"}),
        ],
        ids=["powers", "inverses", "fixed_arithmetic"],
    )
    def test_default_rules_exclude_group(self, group, expr, expected):
        assert _s(expr, rules=DefaultRules(exclude=[group])) in expected

    # Reason: clause (DefaultRules Raises ValueError: exclude names an unknown group).
    def test_default_rules_unknown_group_raises(self):
        with pytest.raises(ValueError, match="nope"):
            DefaultRules(exclude=["nope"])

    # Reason: clause (to_dict / from_dict round trip, also through JSON and the base-class dispatch).
    def test_default_rules_round_trip(self):
        d = DefaultRules(exclude=["powers", "inverses"]).to_dict()
        assert d == {
            "rules_class": "SRToolkit.utils.expression_simplifier.DefaultRules",
            "exclude": ["powers", "inverses"],
        }
        restored = SimplificationRules.from_dict(json.loads(json.dumps(d)))
        assert type(restored) is DefaultRules
        assert restored.to_dict() == d
        assert DefaultRules.from_dict(d).to_dict() == d

    # Reason: clause (from_dict Raises: KeyError, ImportError, TypeError on dispatch,
    # NotImplementedError on a subclass without override; settled: DefaultRules without exclude
    # raises KeyError).
    @pytest.mark.parametrize(
        "call, error",
        [
            (lambda: SimplificationRules.from_dict({}), KeyError),
            (lambda: SimplificationRules.from_dict({"rules_class": "no_such_module.Rules"}), ImportError),
            (lambda: SimplificationRules.from_dict({"rules_class": "collections.OrderedDict"}), TypeError),
            (lambda: _OnlyCancel.from_dict(_OnlyCancel().to_dict()), NotImplementedError),
            (lambda: DefaultRules.from_dict({"rules_class": "SRToolkit.utils.expression_simplifier.DefaultRules"}),
             KeyError),
        ],
        ids=["no-class", "not-importable", "not-subclass", "no-override", "no-exclude"],
    )  # fmt: skip
    def test_rules_from_dict_errors(self, call, error):
        with pytest.raises(error):
            call()

    # Reason: clause (__repr__: "ClassName(key=value, ...) from to_dict without rules_class"; settled:
    # DefaultRules shows the excluded groups).
    def test_repr(self):
        assert repr(DefaultRules(exclude=["powers"])) == "DefaultRules(exclude=['powers'])"
        assert repr(_Tuned()) == "_Tuned(depth=2, name='x')"  # "ClassName(key=value, ...)"

    # Reason: clause (rules: "The rewrite rules"): only the given rules are used.
    def test_custom_rules_are_used(self):
        assert _s("X_0 - X_0", rules=_OnlyCancel()) == "0"
        out = simplify(_t("X_0 * X_0 / X_0"), _SL, rules=_OnlyCancel())[0]
        assert _tree(out) == _tree(_t("X_0 * X_0 / X_0"))

    # Reason: clause (custom rules "may use the relations ... after saturating RELATIONS"): the rule
    # guarded by no_const fires for X_0 - X_0, not for C_1 - C_1.
    def test_custom_rules_can_use_relations(self):
        assert _s("X_0 - X_0", rules=_CancelWithoutConst()) == "0"
        assert _s("C_1 - C_1", rules=_CancelWithoutConst()) == "C_1-C_1"

    # Reason: clause (ConstTree.leaf: "A constant that is not fixed has the bounds of its spec, or
    # (-inf, inf) without them"): a custom rule matching bounds (0, 1) fires only for such a spec.
    def test_custom_rules_see_spec_bounds(self):
        expr = _t("C_1 * X_0")
        unit = Constant("C_1", bounds=(0.0, 1.0))
        assert simplify(expr, _SL, [unit], rules=_DropUnitBounded()) == (["X_0"], ())
        assert simplify(expr, _SL, [Constant("C_1", bounds=(0.0, 2.0))], rules=_DropUnitBounded())[0] == expr
        assert simplify(expr, _SL, None, rules=_DropUnitBounded())[0] == expr


class TestLimits:
    # Reason: boundary (settled: max_iterations=0 returns the input as written) + clause (the search
    # stops at max_nodes / max_time; "the cheapest expression found is returned").
    def test_limits_stop_search(self):
        expr = _t("X_0 * X_0 / X_0")
        assert _tree(simplify(expr, _SL, max_iterations=0)[0]) == _tree(expr)
        big = _t("X_0 * X_1 / X_0 + sin ( X_0 ) / sin ( X_0 ) + X_1 * X_1")
        by_nodes = simplify(big, _SL, max_nodes=0)[0]
        by_time = simplify(big, _SL, max_time=1e-9)[0]
        assert _cost(by_nodes) <= _cost(big) and _cost(by_time) <= _cost(big)
        np.testing.assert_allclose(_values(by_nodes, {}), _values(big, {}), rtol=1e-12)
        np.testing.assert_allclose(_values(by_time, {}), _values(big, {}), rtol=1e-12)
        # An expression that needs several steps: max_nodes=0 stops before the full result (which is
        # X_1^2), a generous max_time does not.
        steps = _t("( ( X_0 * X_1 ) / X_1 ) * ( ( X_1 * X_0 ) / X_0 ) - X_0 * X_1 + X_1 * X_1 * X_1 / X_1")
        full = simplify(steps, _SL)[0]
        assert full == ["X_1", "^2"]
        assert _cost(simplify(steps, _SL, max_nodes=0)[0]) > _cost(full)
        assert simplify(steps, _SL, max_time=1000.0)[0] == full

    # Reason: clause ("max_iterations and max_nodes count work, so the result does not depend on the
    # machine"): repeated calls with work limits give the same result.
    @pytest.mark.parametrize("kwargs", [{"max_nodes": 300}, {"max_iterations": 2}], ids=["nodes", "iterations"])
    def test_deterministic_with_work_limits(self, kwargs):
        expr = _t("( X_0 + X_1 ) * ( X_0 - X_1 ) / ( X_0 + X_1 ) + C * X_0 * X_0 + sin ( X_1 ) ^2")
        first = simplify(expr, _SL, **kwargs)[0]
        assert simplify(expr, _SL, **kwargs)[0] == first
        assert simplify(expr, _SL, **kwargs)[0] == first


class TestCopiedConstants:
    # Reason: clause ("If a rule copies a bare C, its copies share a fresh label ... the lowest C_k that
    # is not in the input"). By hand: C1 + C2*(1 - C3) = C1 + C2 - C2*C3, so C2 is copied, e.g.
    # (C+C_0)-C*C_0; the parameter count stays that of the input. Orders of equal cost are accepted.
    # Two copied constants get the two lowest free labels.
    @pytest.mark.parametrize(
        "expr, labels, n_params",
        [
            ("C + C * ( X_0 / X_0 - ( ( X_0 + C ) - X_0 ) )", ["C_0", "C_0"], 3),
            ("C_0 * X_1 + C + C * ( X_0 / X_0 - ( ( X_0 + C ) - X_0 ) )", ["C_0", "C_1", "C_1"], 4),
            (
                "C_0 * X_1 + C_1 * X_0 + C + C * ( X_0 / X_0 - ( ( X_0 + C ) - X_0 ) )",
                ["C_0", "C_1", "C_2", "C_2"],
                5,
            ),
            (
                "C + C * ( X_0 / X_0 - ( ( X_0 + C ) - X_0 ) ) + C * ( X_1 / X_1 - ( ( X_1 + C ) - X_1 ) )",
                ["C_0", "C_0", "C_1", "C_1"],
                5,
            ),
        ],
        ids=["c0-free", "c0-taken", "c0-c1-taken", "two-copies"],
    )
    def test_copied_bare_constant_gets_shared_label(self, expr, labels, n_params):
        out = simplify(_t(expr), _SL)[0]
        assert sorted(t for t in out if t.startswith("C_")) == labels
        assert len(parameter_labels(out, _SL)) == n_params


def _f(expr, **kwargs):
    """Folded expression (default library) joined into one string."""
    return "".join(fold(_t(expr), _SL, **kwargs)[0])


class TestFold:
    # Reason: clause (fold: each largest part with an operator or function at its top, no variable
    # and a free constant becomes the bare constant; numbers and "lit" symbols in it fold with it).
    @pytest.mark.parametrize(
        "expr, expected",
        [
            ("C * C * X_0", "C*X_0"),
            ("sin ( C + 2 ) * X_0", "C*X_0"),
            ("C_1 / C_2 + X_0", "C+X_0"),
            ("pi * C", "C"),
            ("C * X_0 + C * C", "C*X_0+C"),
        ],
        ids=["product", "function-with-number", "labels", "lit", "two-parts"],
    )
    def test_folds_constant_parts(self, expr, expected):
        assert _f(expr) == expected

    # Reason: clause (not folded: parts with a variable, single tokens, parts without a free
    # constant); boundary: the token list comes back as given, without added brackets.
    @pytest.mark.parametrize(
        "expr",
        ["C * X_0", "sin ( X_0 ) + C", "C_1", "2 * pi", "( C * X_0 ) + X_1"],
        ids=["with-variable", "variable-in-function", "single-label", "no-free-constant", "brackets-kept"],
    )
    def test_leaves_other_parts(self, expr):
        assert fold(_t(expr), _SL)[0] == _t(expr)

    # Reason: clause ("no label that also occurs outside it"): a label shared with another part
    # blocks folding; a label only inside the part folds with it.
    @pytest.mark.parametrize(
        "expr, expected",
        [("sin ( C_1 ) * X_0 + cos ( C_1 ) * X_1", "sin(C_1)*X_0+cos(C_1)*X_1"), ("C_1 * C_1 * X_0", "C*X_0")],
        ids=["shared", "only-inside"],
    )
    def test_shared_labels(self, expr, expected):
        assert _f(expr) == expected

    # Reason: clause (fold mode "keep": labelled constants with a spec, fixed or not, never fold and
    # keep their specs; others still fold. "default": every free constant may fold, a fixed spec
    # like a number, the specs of folded labels are dropped).
    @pytest.mark.parametrize(
        "spec", [Constant("C_1", bounds=(0.0, 1.0)), Constant("C_1", value=9.81, fixed=True)], ids=["free", "fixed"]
    )
    def test_modes_with_specs(self, spec):
        expr = _t("C_1 * C * X_0 + C_2 * C * X_1")
        out, specs = fold(expr, _SL, [spec], mode="keep")
        assert _tree(out) == "(C_1*C)*X_0+C*X_1"
        assert specs == (spec,)
        out, specs = fold(expr, _SL, [spec], mode="default")
        assert "".join(out) == "C*X_0+C*X_1"
        assert specs == ()

    # Reason: clause ("none": nothing is folded; Returns: specs None for None).
    def test_none_folds_nothing(self):
        spec = Constant("C_1", value=1.0)
        assert fold(_t("C_1 * C * X_0"), _SL, [spec], mode="none") == (_t("C_1 * C * X_0"), (spec,))
        assert fold(_t("C * C"), _SL, mode="none") == (_t("C * C"), None)

    # Reason: clause (Raises ValueError: unknown mode; spec without label, duplicate or not in expr).
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "all"},
            {"mode": None},
            {"free_constants": [Constant(value=1.0)]},
            {"free_constants": [Constant("C_1", value=1.0), Constant("C_1", value=2.0)]},
            {"free_constants": [Constant("C_9", value=1.0)]},
        ],
        ids=["mode", "mode-none-value", "no-label", "duplicate", "not-in-expr"],
    )
    def test_invalid_arguments_raise(self, kwargs):
        match = r"^mode must be one of \['none', 'default', 'keep'\]" if "mode" in kwargs else "label"
        with pytest.raises(ValueError, match=match):
            fold(_t("C_1 * C"), _SL, **kwargs)

    # Reason: clause (symbol_library: the constants of the given library; None uses the active or
    # default library). With the constant K, K*K folds and the spec'd K_1 is kept.
    def test_symbol_library(self):
        lib = SymbolLibrary.default_symbols(2, const_symbol="K")
        spec = Constant("K_1", bounds=(0.0, 1.0))
        assert fold(_t("K_1 * X_0 + K * K"), lib, [spec]) == (_t("K_1 * X_0 + K"), (spec,))
        assert fold(["C", "*", "C"]) == (["C"], None)

    # Reason: regression (a fixed label used elsewhere blocked folding, but it is not a parameter, so
    # folding splits nothing); its spec stays while the label occurs.
    def test_shared_fixed_label_folds(self):
        g = Constant("C_1", value=2.0, fixed=True)
        out = fold(_t("C_1 * C * X_0 + C_1 * X_1"), _SL, [g], mode="default")
        assert out == (_t("C * X_0 + C_1 * X_1"), (g,))

    # Reason: clause (fold mode "default": a fixed spec folds like a number, so a part needs a free
    # constant to fold).
    def test_fixed_constant_alone_does_not_fold(self):
        g = Constant("C_1", value=9.81, fixed=True)
        assert fold(_t("C_1 * 2 + X_0"), _SL, [g], mode="default") == (_t("C_1 * 2 + X_0"), (g,))

    # Reason: clause (Raises ExpressionParseError: a symbol neither in the library nor a number);
    # regression: an unknown symbol in a tree was folded away like a number.
    @pytest.mark.parametrize(
        "tree",
        [Node("*", left=Node("C"), right=Node("X_7")), Node("foo", left=Node("C"))],
        ids=["unknown-variable", "unknown-function"],
    )
    def test_unknown_symbol_raises(self, tree):
        with pytest.raises(ExpressionParseError, match="Unknown symbol"):
            fold(tree, SymbolLibrary.default_symbols(1))

    # Reason: clause (Returns: "in the form of expr"); implied (the input is not mutated).
    def test_form_of_input(self):
        tree = tokens_to_tree(_t("C * C * X_0"), _SL)
        before = tree.to_list(_SL)
        out = fold(tree, _SL)[0]
        assert isinstance(out, Node)
        assert out.to_list(_SL) == ["C", "*", "X_0"]
        assert tree.to_list(_SL) == before

    # Reason: implied (folding never adds parameters, and the result is a valid expression).
    @settings(max_examples=50, deadline=None)
    @given(labelled_infix_expressions())
    def test_parameter_count_never_grows(self, expr):
        out = fold(expr, _SL)[0]
        assert len(parameter_labels(out, _SL)) <= len(parameter_labels(expr, _SL))
        tokens_to_tree(out, _SL)


class TestSimplifyFold:
    # Reason: clause (simplify fold: the result is folded after the algebra; the algebra gathers
    # the constants first). The second case is what the SymPy simplifier gave.
    @pytest.mark.parametrize(
        "expr, expected",
        [("C * X_0 + X_1 * X_0 / X_1 * C", {"C*X_0", "X_0*C"}), ("C + C * C + X_0 * X_1 / X_0", {"C+X_1", "X_1+C"})],
        ids=["gathered", "sympy-case"],
    )
    @pytest.mark.parametrize("aware", [True, False])
    def test_simplify_folds(self, expr, expected, aware):
        assert _s(expr, fold="keep", fold_aware=aware) in expected

    # Reason: clause (fold_aware: "choose the result by its length after folding"; False chooses by
    # token count and then folds). From notes/egraphs-main/data/40k.pkl: only reordering folds C.
    def test_fold_aware_finds_shorter(self):
        assert _s("( C / sin ( X_0 ) ) * C", fold="keep") == "C/sin(X_0)"
        assert _tree(simplify(_t("( C / sin ( X_0 ) ) * C"), _SL, fold="keep", fold_aware=False)[0]) == (
            "(C/sin(X_0))*C"
        )

    # Reason: clause (fold_aware "Ignored with fold='none'"): on this expression the two extractions
    # differ, so a fold-aware extraction without folding would show.
    def test_fold_aware_ignored_without_fold(self):
        expr = _t("( C / sin ( X_0 ) ) * C")
        assert simplify(expr, _SL) == simplify(expr, _SL, fold_aware=False)

    # Reason: clause (simplify folds with the given library).
    def test_folds_with_library(self):
        lib = SymbolLibrary.default_symbols(2, const_symbol="K")
        assert "".join(simplify(_t("K * K * X_0"), lib, fold="keep")[0]) in {"K*X_0", "X_0*K"}

    # Reason: clause (simplify fold="default": specs of folded labels are dropped).
    def test_default_mode_drops_specs(self):
        out, specs = simplify(_t("C_1 * C * X_0"), _SL, [Constant("C_1", bounds=(0.0, 1.0))], fold="default")
        assert "".join(out) in {"C*X_0", "X_0*C"}
        assert specs == ()

    # Reason: clause (fold_aware "never longer than with False, often shorter"; module docs: the
    # output is never longer than the input). Token counts by hand; the last three are regressions
    # (labels the cost cannot see made the fold-aware form longer than the input).
    @pytest.mark.parametrize(
        "expr, mode, spec, length",
        [
            ("( sin ( C ) / ( C / ( ( C_1 * 1 ) - X_0 ) ) )", "default", False, 5),
            ("C + ( ( ( 2 - C_1 ) + sin ( 1 ) ) + ( C + ( C - C ) ) )", "keep", True, 3),
            ("exp ( ( X_1 - C_1 ) - exp ( cos ( 2 ) ) )", "default", True, 4),
            ("exp ( sin ( 2 ) - C_1 / C ) * 1", "keep", True, 7),
            ("exp ( X_1 - 2 ) / ( C_1 * X_0 - C_1 ) + C", "default", False, 12),
            ("( 2 - ( X_1 * C - ( X_1 + C ) ) ) / ( C + cos ( 2 ) * C )", "keep", False, 11),
            ("cos ( exp ( C * 2 ) ) / ( ( ( C_1 - C ) / C ) - 1 )", "keep", True, 9),
        ],
        ids=["gathers", "keeps-spec", "drops-spec", "tie", "shared-label", "copied-constant", "spec-and-copies"],
    )
    def test_fold_aware_length(self, expr, mode, spec, length):
        specs = [Constant("C_1", bounds=(0.0, 1.0))] if spec else None
        aware = simplify(_t(expr), _SL, specs, fold=mode)[0]
        plain = simplify(_t(expr), _SL, specs, fold=mode, fold_aware=False)[0]
        assert _cost(aware) == length
        assert _cost(aware) <= min(_cost(plain), _cost(_t(expr)))

    # Reason: regression (the fold-aware search ran only on the last e-graph, which after a cheaper
    # step holds only that form, so one more step could give a longer result).
    def test_more_steps_never_longer(self):
        expr = _t("C / ( C * X_0 ) + X_1 * X_0 / X_1")
        lengths = [_cost(simplify(expr, _SL, fold="keep", max_iterations=i)[0]) for i in range(1, 8)]
        assert lengths == sorted(lengths, reverse=True)
        assert lengths[-1] == 5  # C/X_0+X_0

    # Reason: regression (on a tie the fold-aware form was kept, adding a label C_0 for nothing).
    def test_tie_keeps_token_count_form(self):
        expr = _t("C * ( X_0 + X_1 ) + C * X_0")
        assert simplify(expr, _SL, fold="keep") == simplify(expr, _SL, fold="keep", fold_aware=False)

    # Reason: clause (fold_aware "Ignored with fold='none'"; fold="none" keeps the hypothesis).
    def test_no_fold_by_default(self):
        expr = "C * X_0 + X_1 * X_0 / X_1 * C"
        assert _s(expr) == _s(expr, fold_aware=False) == "(C+C)*X_0"

    # Reason: clause ("keep": a spec'd label does not fold and keeps its spec, through simplify).
    def test_simplify_keeps_spec(self):
        spec = Constant("C_1", bounds=(0.0, 1.0))
        out, specs = simplify(_t("C_1 * C * C * X_0"), _SL, [spec], fold="keep")
        assert "C_1" in out and len(parameter_labels(out, _SL)) == 2
        assert specs == (spec,)

    # Reason: clause (Raises ValueError: fold unknown).
    def test_unknown_fold_raises(self):
        with pytest.raises(ValueError, match="^fold must"):
            simplify(["X_0"], _SL, fold="yes")
