"""Spec tests for SRToolkit.utils.free_constants (Constant, uniform_default, resolve_constants, canonicalise)."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from strategies import infix_expressions, labelled_infix_expressions

from SRToolkit.dataset import UniformSampling
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_tree import parameter_labels, tokens_to_tree
from SRToolkit.utils.free_constants import Constant, _spec_key, canonicalise, resolve_constants, uniform_default
from SRToolkit.utils.symbol_library import SymbolLibrary

INF = float("inf")
NAN = float("nan")


# ---------------------------------------------------------------------------------------------------
# Constant: construction and validation
# ---------------------------------------------------------------------------------------------------


class TestConstant:
    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"label": 5}, TypeError, "label"),
            ({"prior": "uniform"}, TypeError, "prior"),
            ({"prior": {"sampler_class": "SRToolkit.dataset.sampling.UniformSampling"}}, TypeError, "prior"),
            ({"bounds": 5}, TypeError, "bounds"),
            ({"bounds": np.array(3.0)}, TypeError, "bounds"),
            ({"bounds": (0.0,)}, TypeError, "bounds"),
            ({"bounds": (0, 1, 2)}, TypeError, "bounds"),
            ({"bounds": "ab"}, TypeError, "bounds"),
            ({"bounds": ("a", 1)}, TypeError, "bounds"),
            ({"bounds": (True, 2.0)}, TypeError, "bounds"),
            ({"value": "1"}, TypeError, "value"),
            ({"value": True}, TypeError, "value"),
            ({"value": 1 + 2j}, TypeError, "value"),
            ({"value": 1.0, "fixed": 1}, TypeError, "fixed"),
            ({"fixed": None}, TypeError, "fixed"),
            ({"bounds": (NAN, 1.0)}, ValueError, "bounds"),
            ({"bounds": (1.0, 1.0)}, ValueError, "bounds"),
            ({"bounds": (2.0, 1.0)}, ValueError, "bounds"),
            ({"bounds": (INF, INF)}, ValueError, "bounds"),
            ({"value": NAN}, ValueError, "value"),
            ({"value": INF}, ValueError, "value"),
            ({"value": INF, "fixed": True}, ValueError, "value"),
            ({"fixed": True}, ValueError, "value"),
            ({"bounds": (0.0, 1.0), "value": 1.5}, ValueError, "value|bounds"),
            ({"bounds": (0.0, 1.0), "value": -0.1}, ValueError, "value|bounds"),
        ],
        ids=[
            "label_int",
            "prior_str",
            "prior_dict",
            "bounds_scalar",
            "bounds_0d_array",
            "bounds_len1",
            "bounds_len3",
            "bounds_str",
            "bounds_non_numeric",
            "bounds_bool",
            "value_str",
            "value_bool",
            "value_complex",
            "fixed_int",
            "fixed_none",
            "bounds_nan",
            "bounds_equal",
            "bounds_unordered",
            "bounds_inf_inf",
            "value_nan",
            "value_inf",
            "value_inf_fixed",
            "fixed_without_value",
            "value_above_bounds",
            "value_below_bounds",
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, exc, match):
        with pytest.raises(exc, match=match):
            Constant(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, bounds, value",
        [
            ({"bounds": (-INF, INF)}, (-INF, INF), None),
            ({"bounds": (0, INF)}, (0.0, INF), None),
            ({"bounds": (0, 1), "value": 1}, (0.0, 1.0), 1.0),
            ({"bounds": (0, 1), "value": 0}, (0.0, 1.0), 0.0),
            ({"bounds": (0, 1), "value": 5, "fixed": True}, (0.0, 1.0), 5.0),
            ({"value": 5}, None, 5.0),
            ({"label": "anything", "value": 5}, None, 5.0),
        ],
        ids=[
            "unbounded",
            "half_infinite",
            "value_on_upper",
            "value_on_lower",
            "fixed_outside_own_bounds",
            "free_value_without_bounds",
            "label_not_checked_here",
        ],
    )
    def test_accepts_domain_edges(self, kwargs, bounds, value):
        c = Constant(**kwargs)
        assert c.bounds == bounds
        assert c.value == value

    @pytest.mark.parametrize(
        "bounds",
        [[0, 2], (np.float32(0.0), np.int64(2)), np.array([0, 2])],
        ids=["list", "numpy_scalars", "numpy_array"],
    )
    def test_normalises_bounds_to_tuple_of_floats(self, bounds):
        c = Constant(bounds=bounds)
        assert c.bounds == (0.0, 2.0)
        assert type(c.bounds) is tuple
        assert [type(b) for b in c.bounds] == [float, float]

    @pytest.mark.parametrize("value", [2, np.float64(2.0)], ids=["int", "numpy_float"])
    def test_normalises_value_to_float(self, value):
        c = Constant(value=value)
        assert c.value == 2.0
        assert type(c.value) is float


# ---------------------------------------------------------------------------------------------------
# Constant.to_dict / Constant.from_dict
# ---------------------------------------------------------------------------------------------------


class TestConstantSerialization:
    def test_to_dict_layout(self):
        prior = UniformSampling(0.0, 2.0)
        d = Constant(label="C_1", prior=prior, bounds=(0, INF), value=1).to_dict()
        assert d == {"label": "C_1", "prior": prior.to_dict(), "bounds": [0.0, INF], "value": 1.0, "fixed": False}
        assert type(d["bounds"]) is list

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"label": "C_0", "prior": UniformSampling(0.0, 2.0), "bounds": (-INF, INF), "value": 0.5},
            {"label": "C_3", "value": 9.81, "fixed": True},
        ],
        ids=["template_all_none", "labelled_prior_inf_bounds", "fixed_value"],
    )
    def test_round_trip_through_json(self, kwargs):
        spec = Constant(**kwargs)
        d = json.loads(json.dumps(spec.to_dict()))
        assert Constant.from_dict(d).to_dict() == spec.to_dict()

    def test_from_dict_ignores_unknown_keys(self):
        spec = Constant(label="C_0", bounds=(0.0, 1.0), value=0.5)
        assert Constant.from_dict({**spec.to_dict(), "units": "m"}).to_dict() == spec.to_dict()

    @pytest.mark.parametrize(
        "drop, override, exc",
        [
            ("label", {}, KeyError),
            ("prior", {}, KeyError),
            ("bounds", {}, KeyError),
            ("value", {}, KeyError),
            ("fixed", {}, KeyError),
            (None, {"prior": {"sampler_class": "no.such.Sampler"}}, ImportError),
            (None, {"fixed": "yes"}, TypeError),
            (None, {"bounds": [1.0, 0.0]}, ValueError),
        ],
        ids=[
            "missing_label",
            "missing_prior",
            "missing_bounds",
            "missing_value",
            "missing_fixed",
            "unknown_sampler_class",
            "fixed_wrong_type",
            "bounds_unordered",
        ],
    )
    def test_from_dict_rejects_invalid_dicts(self, drop, override, exc):
        d = {"label": "C_0", "prior": None, "bounds": [0.0, 1.0], "value": 0.5, "fixed": False, **override}
        d.pop(drop, None)
        with pytest.raises(exc):
            Constant.from_dict(d)


# ---------------------------------------------------------------------------------------------------
# uniform_default
# ---------------------------------------------------------------------------------------------------


class TestUniformDefault:
    def test_fields_and_prior(self):
        c = uniform_default([-2, 3])
        assert (c.label, c.value, c.fixed) == (None, None, False)
        assert c.bounds == (-2.0, 3.0)
        assert [type(b) for b in c.bounds] == [float, float]
        assert isinstance(c.prior, UniformSampling)
        # unmirrored uniform: ppf(u) = -2 + u * 5
        np.testing.assert_allclose(c.prior.ppf(np.array([0.0, 0.25, 0.5])), [-2.0, -0.75, 0.5])

    def test_returns_new_prior_each_call(self):
        assert uniform_default((-1.0, 1.0)).prior is not uniform_default((-1.0, 1.0)).prior

    @pytest.mark.parametrize(
        "constant_bounds, exc",
        [
            (5, TypeError),
            ((1.0,), TypeError),
            ((1, 2, 3), TypeError),
            (("a", 1), TypeError),
            ((True, 2), TypeError),
            ((-INF, 1.0), ValueError),
            ((0.0, INF), ValueError),
            ((NAN, 1.0), ValueError),
            ((1.0, 1.0), ValueError),
            ((2.0, 1.0), ValueError),
        ],
        ids=[
            "scalar",
            "len1",
            "len3",
            "non_numeric",
            "bool",
            "lower_inf",
            "upper_inf",
            "nan",
            "equal",
            "unordered",
        ],
    )
    def test_rejects_invalid_bounds(self, constant_bounds, exc):
        with pytest.raises(exc, match="bounds"):
            uniform_default(constant_bounds)


# ---------------------------------------------------------------------------------------------------
# resolve_constants
# ---------------------------------------------------------------------------------------------------


def _lit_c1_library():
    """Default library where ``C_1`` is a registered named constant, not a labelled constant (D17)."""
    sl = SymbolLibrary.default_symbols(1)
    sl.add_symbol("C_1", "lit", 5, "1.0")
    return sl


class TestResolveConstants:
    @pytest.mark.parametrize("specs", [None, ()], ids=["none", "empty"])
    def test_without_specs_gives_default_itself(self, specs):
        default = uniform_default((-5.0, 5.0))
        res = resolve_constants(["C", "*", "X_0", "+", "C_1"], specs, default)
        assert len(res) == 2
        assert [s is default for s in res] == [True, True]

    def test_expression_without_constants_is_empty(self):
        assert resolve_constants(["X_0", "+", "pi"], None, uniform_default((-5.0, 5.0))) == ()

    def test_maps_specs_to_parameter_slots(self):
        default = uniform_default((-5.0, 5.0))
        expr = ["C_2", "*", "X_0", "+", "C", "*", "C_0", "+", "C_2", "+", "C_7"]
        specs = [Constant(label="C_0", value=1.0), Constant(label="C_2", value=3.0, fixed=True)]
        res = resolve_constants(expr, specs, default)
        # parameters by first appearance: C_2, bare C, C_0, C_7 (C_7 has no spec)
        assert [s.label for s in res] == ["C_2", None, "C_0", None]
        assert [s.value for s in res] == [3.0, None, 1.0, None]
        assert [s.fixed for s in res] == [True, False, False, False]
        assert res[1] is default
        assert res[3] is default

    def test_fills_missing_prior_and_bounds_from_default(self):
        default = uniform_default((-5.0, 5.0))
        own_prior = UniformSampling(0.0, 1.0)
        specs = [
            Constant(label="C_0"),
            Constant(label="C_1", prior=own_prior),
            Constant(label="C_2", bounds=(0.0, INF)),
        ]
        c0, c1, c2 = resolve_constants(["C_0", "+", "C_1", "+", "C_2"], specs, default)
        assert c0.prior is default.prior
        assert c0.bounds == (-5.0, 5.0)
        assert c1.prior is own_prior
        assert c1.bounds == (-5.0, 5.0)
        assert c2.prior is default.prior
        assert c2.bounds == (0.0, INF)

    @pytest.mark.parametrize(
        "value, fixed",
        [(9.81, True), (5.0, False), (-5.0, False)],
        ids=["fixed_outside", "free_on_upper", "free_on_lower"],
    )
    def test_accepts_values_against_inherited_bounds(self, value, fixed):
        spec = Constant(label="C_0", value=value, fixed=fixed)
        (res,) = resolve_constants(["C_0", "*", "X_0"], [spec], uniform_default((-5.0, 5.0)))
        assert res.value == value
        assert res.bounds == (-5.0, 5.0)

    def test_node_input_equals_token_input(self):
        default = uniform_default((-5.0, 5.0))
        expr = ["C", "*", "X_0", "+", "C_1", "*", "C_1"]
        specs = [Constant(label="C_1", value=2.0)]
        sl = SymbolLibrary.default_symbols(1)
        assert resolve_constants(tokens_to_tree(expr, sl), specs, default, sl) == resolve_constants(
            expr, specs, default, sl
        )

    @pytest.mark.parametrize(
        "expr, specs, default, symbol_library, exc, match",
        [
            (
                ["C_0"],
                None,
                dict(label="C_0", prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0)),
                None,
                ValueError,
                "default",
            ),
            (["C_0"], None, dict(bounds=(-1.0, 1.0)), None, ValueError, "default"),
            (["C_0"], None, dict(prior=UniformSampling(-1.0, 1.0)), None, ValueError, "default"),
            (
                ["C_0"],
                None,
                dict(prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0), value=0.5),
                None,
                ValueError,
                "default_constant",
            ),
            (
                ["C_0"],
                None,
                dict(prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0), value=0.5, fixed=True),
                None,
                ValueError,
                "default_constant",
            ),
            (["C_0"], [dict(value=1.0)], None, None, ValueError, "label"),
            (["C_0"], [dict(label="C_0"), dict(label="C_0", value=1.0)], None, None, ValueError, "C_0"),
            (["C_0"], [dict(label="C_9")], None, None, ValueError, "C_9"),
            (["C", "+", "X_0"], [dict(label="C")], None, None, ValueError, "label"),
            (["C", "+", "X_0"], [dict(label="X_0")], None, None, ValueError, "X_0"),
            (["C_0"], [dict(label="C_0", value=6.0)], None, None, ValueError, "value|bounds"),
            (["C_1", "+", "C_0"], [dict(label="C_1")], None, _lit_c1_library(), ValueError, "C_1"),
            ("C+X_0", None, None, None, TypeError, "expr"),
            (5, None, None, None, TypeError, "expr"),
        ],
        ids=[
            "default_has_label",
            "default_without_prior",
            "default_without_bounds",
            "default_with_value",
            "default_fixed",
            "spec_without_label",
            "duplicate_label",
            "label_not_in_expr",
            "bare_C_label",
            "variable_label",
            "free_value_outside_inherited_bounds",
            "lit_symbol_is_not_labelled_constant",
            "expr_str",
            "expr_int",
        ],
    )
    def test_rejects_invalid_input(self, expr, specs, default, symbol_library, exc, match):
        # specs and default are given as Constant kwargs; default None means uniform_default((-5, 5))
        specs = None if specs is None else [Constant(**kw) for kw in specs]
        default = uniform_default((-5.0, 5.0)) if default is None else Constant(**default)
        with pytest.raises(exc, match=match):
            resolve_constants(expr, specs, default, symbol_library)


# ---------------------------------------------------------------------------------------------------
# _spec_key (feeds seeds and cache keys, D28)
# ---------------------------------------------------------------------------------------------------


BASE = dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 1.0), value=0.5)


def _base_spec():
    return Constant(**{**BASE, "prior": UniformSampling(0.0, 1.0)})


class TestSpecKey:
    @pytest.mark.parametrize("specs", [None, ()], ids=["none", "empty"])
    def test_empty(self, specs):
        assert _spec_key(specs) == ""

    def test_order_independent(self):
        a = _base_spec()
        b = Constant(label="C_1", value=2.0, fixed=True)
        assert _spec_key([a, b]) == _spec_key([b, a])

    def test_equal_for_separately_built_priors(self):
        assert _spec_key([_base_spec()]) == _spec_key([_base_spec()])

    @pytest.mark.parametrize("low, high", [(0, 1), (np.int64(0), np.int64(1))], ids=["int", "numpy_int"])
    def test_compares_prior_numbers_as_floats(self, low, high):
        spec = Constant(**{**BASE, "prior": UniformSampling(low, high)})
        assert _spec_key([spec]) == _spec_key([_base_spec()])

    @pytest.mark.parametrize(
        "variant",
        [
            [dict(label="C_1", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 1.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 2.0), bounds=(0.0, 1.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0, mirror=True), bounds=(0.0, 1.0), value=0.5)],
            [dict(label="C_0", bounds=(0.0, 1.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 2.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(-INF, 1.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), value=0.5)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 1.0), value=0.25)],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 1.0))],
            [dict(label="C_0", prior=UniformSampling(0.0, 1.0), bounds=(0.0, 1.0), value=0.5, fixed=True)],
            [BASE, dict(label="C_1")],
        ],
        ids=[
            "label",
            "prior_range",
            "prior_mirror",
            "prior_none",
            "bounds",
            "bounds_infinite",
            "bounds_none",
            "value",
            "value_none",
            "fixed",
            "extra_spec",
        ],
    )
    def test_changes_with_each_field(self, variant):
        # variant is a list of Constant kwargs differing from BASE in one field (or adding a spec)
        assert _spec_key([Constant(**kw) for kw in variant]) != _spec_key([_base_spec()])


# ---------------------------------------------------------------------------------------------------
# canonicalise
# ---------------------------------------------------------------------------------------------------

_X_CANON = np.random.default_rng(0).uniform(0.5, 2.0, size=(5, 2))


def _relabel(tokens):
    """Rename the labels of ``labelled_infix_expressions`` by a fixed bijection."""
    renaming = {"C_0": "C_7", "C_1": "C_0", "C_7": "C_1"}
    return [renaming.get(t, t) for t in tokens]


class TestCanonicalise:
    # Reason: clause (once -> bare, shared -> C_<k> with k from the parameter order, skipping
    # registered names, const symbol of the library) + boundary (empty, no constants, no const symbol,
    # malformed tokens) + implied (input not mutated).
    @pytest.mark.parametrize(
        "lib, expr, expected",
        [
            (None, "C_3 + X_0", "C + X_0"),
            (None, "C_2 * X_0 + C_9 + C_9 + C_2", "C_0 * X_0 + C_1 + C_1 + C_0"),
            # positions count bare and once-occurring constants: C_8 is parameter 2
            (None, "C_4 + C + C_8 * C_8", "C + C + C_2 * C_2"),
            # renaming is simultaneous: C_1 -> C_0 and C_0 -> C_1 do not collide
            (None, "C_1 * C_1 + C_0 * C_0", "C_0 * C_0 + C_1 * C_1"),
            (None, "X_0 + pi", "X_0 + pi"),
            (None, "", ""),
            (
                lambda: SymbolLibrary.default_symbols(1, const_symbol="K"),
                "K_3 * X_0 + K_3 + K_5",
                "K_0 * X_0 + K_0 + K",
            ),
            # D17: the named constant C_1 is not a parameter; C_5 is parameter 0
            (_lit_c1_library, "C_1 + C_5 * X_0 + C_5", "C_1 + C_0 * X_0 + C_0"),
            # D17: C_5 is parameter 1, but C_1 is registered, so it gets C_2
            (_lit_c1_library, "C_7 * C_7 + C_5 * X_0 + C_5", "C_0 * C_0 + C_2 * X_0 + C_2"),
            # labels are handed out per parameter, bare ones included: the bare C takes C_2
            (_lit_c1_library, "C_7 * C_7 + C + C_5 * C_5", "C_0 * C_0 + C + C_3 * C_3"),
            (lambda: SymbolLibrary.from_symbol_list(["+"], 1), "C_0 + C_0", "C_0 + C_0"),
            # tokens only: malformed input is rewritten without parsing
            (None, "C_3 * ( C_3", "C_0 * ( C_0"),
        ],
        ids=[
            "once",
            "renumber",
            "positions-count-bare",
            "swap",
            "no-constants",
            "empty",
            "const-symbol-K",
            "D17-registered-untouched",
            "D17-skip",
            "D17-skip-counts-bare",
            "no-const-symbol",
            "malformed",
        ],
    )
    def test_canonical_form(self, lib, expr, expected):
        tokens = expr.split()
        given_tokens = list(tokens)
        assert canonicalise(tokens, None if lib is None else lib()) == expected.split()
        assert tokens == given_tokens

    # Reason: clause (Node input gives the canonical form of its infix to_list) + implied (not mutated).
    def test_node_input(self):
        tree = tokens_to_tree(["C_5", "+", "C_5"], SymbolLibrary.default_symbols(1))
        assert canonicalise(tree) == ["C_0", "+", "C_0"]
        assert tree.to_list() == ["C_5", "+", "C_5"]

    # Reason: clause (Raises TypeError).
    @pytest.mark.parametrize("expr", ["C_0+C_0", None, [1, "+"]], ids=["str", "none", "non-str-token"])
    def test_rejects_non_expression(self, expr):
        with pytest.raises(TypeError, match="token strings"):
            canonicalise(expr)

    # Reason: implied (only names change): idempotent, same parameter count, same function of the
    # same parameter vector.
    @settings(max_examples=60)
    @given(expr=labelled_infix_expressions())
    def test_canonical_form_is_idempotent_and_computes_the_same_function(self, expr):
        canon = canonicalise(expr)
        assert canonicalise(canon) == canon
        n = len(parameter_labels(expr))
        assert len(parameter_labels(canon)) == n
        params = np.random.default_rng(1).uniform(0.5, 2.0, size=n)
        with np.errstate(all="ignore"):
            np.testing.assert_array_equal(compile_expr(canon)(_X_CANON, params), compile_expr(expr)(_X_CANON, params))

    # Reason: clause ("expressions that differ only in how their constants are written then have one form").
    @settings(max_examples=60)
    @given(expr=labelled_infix_expressions())
    def test_invariant_under_relabelling(self, expr):
        assert canonicalise(_relabel(expr)) == canonicalise(expr)

    # Reason: clause ("an expression with only bare constants is unchanged").
    @settings(max_examples=60)
    @given(expr=infix_expressions())
    def test_bare_only_expression_unchanged(self, expr):
        assert canonicalise(expr) == expr
