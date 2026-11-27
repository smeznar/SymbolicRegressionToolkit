import os
import subprocess
import sys

import numpy as np
import pytest

from SRToolkit.dataset.sampling import LogUniformSampling, Sampler, UniformSampling
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_tree import Node, tokens_to_tree
from SRToolkit.utils.free_constants import Constant, uniform_default
from SRToolkit.utils.symbol_library import SymbolLibrary

# ── Shared data ─────────────────────────────────────────────────────────────────
# Two-feature dataset; analytical solutions exist for the expressions used below.

_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
_y_exact = _X[:, 0] + _X[:, 1]  # y = X_0 + X_1  (no-constant test)
_y_linear = 2.0 * _X[:, 0]  # y = 2*X_0      (one-constant test, C should → 2)


# ── ParameterEstimator.__init__ ────────────────────────────────────────────────


class TestParameterEstimatorInit:
    def test_stores_X_y_seed(self):
        pe = ParameterEstimator(_X, _y_linear, seed=7)
        assert pe.X is _X
        assert pe.y is _y_linear
        assert pe.seed == 7

    def test_default_symbol_library_when_none(self):
        pe = ParameterEstimator(_X, _y_linear)
        assert pe.symbol_library is not None
        assert isinstance(pe.symbol_library, SymbolLibrary)

    def test_custom_symbol_library_stored(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        pe = ParameterEstimator(_X, _y_linear, symbol_library=sl)
        assert pe.symbol_library is sl

    def test_default_estimation_settings(self):
        pe = ParameterEstimator(_X, _y_linear)
        s = pe.estimation_settings
        assert s["method"] == "L-BFGS-B"
        assert s["tol"] == 1e-6
        assert s["gtol"] == 1e-3
        assert s["max_iter"] == 100
        assert s["constant_bounds"] == (-5, 5)
        assert s["initialization"] == "random"
        assert s["backend"] == "stack"

    def test_kwargs_override_known_settings(self):
        pe = ParameterEstimator(_X, _y_linear, method="Nelder-Mead", max_iter=50)
        assert pe.estimation_settings["method"] == "Nelder-Mead"
        assert pe.estimation_settings["max_iter"] == 50

    def test_unknown_kwargs_are_ignored(self):
        pe = ParameterEstimator(_X, _y_linear, not_a_real_key=99)
        assert "not_a_real_key" not in pe.estimation_settings

    def test_invalid_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="backend"):
            ParameterEstimator(_X, _y_linear, backend="numpy")

    def test_valid_backends_accepted(self):
        for backend in ("stack", "codegen", "stack_py"):
            ParameterEstimator(_X, _y_linear, backend=backend)

    def test_invalid_initialization_raises(self):
        with pytest.raises(ValueError, match="initialization"):
            ParameterEstimator(_X, _y_linear, initialization="bogus")

    @pytest.mark.parametrize("bounds", [5, (1, 2, 3), ("a", 1), (True, 2)], ids=["scalar", "triple", "string", "bool"])
    def test_constant_bounds_wrong_type_raises(self, bounds):
        with pytest.raises(TypeError, match="bounds"):
            ParameterEstimator(_X, _y_linear, constant_bounds=bounds)

    @pytest.mark.parametrize(
        "bounds", [(5, -5), (1, 1), (-np.inf, 5), (0, np.nan)], ids=["reversed", "equal", "inf", "nan"]
    )
    def test_constant_bounds_invalid_values_raise(self, bounds):
        with pytest.raises(ValueError, match="bounds"):
            ParameterEstimator(_X, _y_linear, constant_bounds=bounds)

    # Reason: clause (the default_constant setting must be a complete Constant template: TypeError for a
    # non-Constant, ValueError for a missing prior or bounds, a label, a value or fixed).
    @pytest.mark.parametrize(
        "default, exc, match",
        [
            ("x", TypeError, "default_constant.*got str"),
            (UniformSampling(0, 1), TypeError, "default_constant.*got UniformSampling"),
            (Constant(bounds=(0, 1)), ValueError, "default_constant"),
            (Constant(prior=UniformSampling(0, 1)), ValueError, "default_constant"),
            (Constant("C_0", prior=UniformSampling(0, 1), bounds=(0, 1)), ValueError, "default_constant"),
            (Constant(prior=UniformSampling(0, 1), bounds=(0, 1), value=0.5), ValueError, "default_constant"),
            (
                Constant(prior=UniformSampling(0, 1), bounds=(0, 1), value=0.5, fixed=True),
                ValueError,
                "default_constant",
            ),
        ],
        ids=["str", "prior-not-constant", "no-prior", "no-bounds", "label", "value", "fixed"],
    )
    def test_default_constant_setting_errors(self, default, exc, match):
        with pytest.raises(exc, match=match):
            ParameterEstimator(_X, _y_linear, default_constant=default)

    def test_default_constant_from_constant_bounds(self):
        # A JSON round trip turns the tuple into a list of ints; it still builds the default spec.
        pe = ParameterEstimator(_X, _y_linear, constant_bounds=[-2, 10])
        assert pe.default_constant.to_dict() == uniform_default((-2.0, 10.0)).to_dict()

    # Reason: clause (constant_bounds is not used, so not validated, when default_constant is given).
    def test_constant_bounds_ignored_with_default_constant(self):
        default = Constant(prior=UniformSampling(0, 1), bounds=(0, 1))
        pe = ParameterEstimator(_X, _y_linear, constant_bounds=(5, -5), default_constant=default)
        assert pe.default_constant is default

    # Reason: regression (an explicit None replaced the default, e.g. tol=None reached scipy as None;
    # None means the default for every setting).
    @pytest.mark.parametrize(
        "name", ["method", "tol", "gtol", "max_iter", "constant_bounds", "initialization", "backend"]
    )
    def test_none_setting_is_the_default(self, name):
        expr = ["C", "*", "X_0", "+", "C"]
        default = ParameterEstimator(_X, _y_linear, seed=0)
        given_none = ParameterEstimator(_X, _y_linear, seed=0, **{name: None})
        assert given_none.estimation_settings == default.estimation_settings
        rmse, params = given_none.estimate_parameters(expr)
        default_rmse, default_params = default.estimate_parameters(expr)
        assert rmse == default_rmse
        np.testing.assert_array_equal(params, default_params)


# ── ParameterEstimator.estimate_parameters ─────────────────────────────────────


class TestEstimateParametersNoConstants:
    def test_returns_near_zero_rmse(self):
        pe = ParameterEstimator(_X, _y_exact)
        rmse, _ = pe.estimate_parameters(["X_0", "+", "X_1"])
        assert rmse < 1e-10

    def test_returns_empty_parameters_array(self):
        pe = ParameterEstimator(_X, _y_exact)
        _, params = pe.estimate_parameters(["X_0", "+", "X_1"])
        assert isinstance(params, np.ndarray)
        assert params.size == 0


class TestEstimateParametersWithConstants:
    def test_one_constant_recovers_coefficient(self):
        pe = ParameterEstimator(_X, _y_linear, seed=0)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert len(params) == 1
        assert abs(params[0] - 2.0) < 0.01

    def test_node_input_branch(self):
        # Node tree for C * X_0: prefix order is ["*", "C", "X_0"]
        node = Node("*", left=Node("C"), right=Node("X_0"))
        pe = ParameterEstimator(_X, _y_linear, seed=0)
        rmse, params = pe.estimate_parameters(node)
        assert rmse < 1e-4
        assert abs(params[0] - 2.0) < 0.01


class TestLabelledConstants:
    @pytest.mark.parametrize("backend", ["stack", "stack_py", "codegen"])
    def test_label_is_fitted_as_one_parameter(self, backend):
        pe = ParameterEstimator(_X, 2.0 * _y_exact, SymbolLibrary.default_symbols(2), seed=0, backend=backend)
        error, params = pe.estimate_parameters(["C_0", "*", "X_0", "+", "C_0", "*", "X_1"])
        assert error < 1e-6
        np.testing.assert_allclose(params, [2.0], atol=1e-5)

    def test_parameters_follow_first_appearance(self):
        # Reason: implied (parameters are in parameter_labels order: [C_1, None]).
        # X_1 = X_0 + 1 on _X, so 5*X_1 + 3*X_0 = 8*X_0 + 5 identifies both constants.
        y = 5.0 * _X[:, 1] + 3.0 * _X[:, 0]
        pe = ParameterEstimator(_X, y, SymbolLibrary.default_symbols(2), seed=0)
        error, params = pe.estimate_parameters(["C_1", "*", "X_1", "+", "C", "*", "X_0"])
        assert error < 1e-6
        np.testing.assert_allclose(params, [5.0, 3.0], atol=1e-4)


# ── ParameterEstimator._optimize_parameters (via estimate_parameters) ──────────


_X_SQRT = np.linspace(0.5, 1.5, 50).reshape(-1, 1)
_y_SQRT = np.sqrt(2 - _X_SQRT[:, 0])  # sqrt(C - X_0) with C = 2; undefined for C < 1.5
_SQRT_EXPR = ["sqrt", "(", "C", "-", "X_0", ")"]


class TestEstimateParametersWithInfo:
    def test_converged_fit(self):
        # Fixed midpoint start C = 0: from about 1 in 10 random starts L-BFGS-B reports an abnormal
        # end even at the exact optimum (see test_exact_fit_can_report_not_converged).
        pe = ParameterEstimator(_X, _y_linear, initialization="mean")
        rmse, params, info = pe.estimate_parameters_with_info(["C", "*", "X_0"])
        assert rmse < 1e-6
        assert info["converged"] is True and "message" not in info
        assert info["evaluations"] >= info["iterations"] >= 1

    def test_exact_fit_can_report_not_converged(self):
        # RMSE has no gradient at a perfect fit, so the line search can fail right at the optimum.
        # The midpoint start C = -2 is one where this happens; the outcome is independent of seeding.
        pe = ParameterEstimator(_X, _y_linear, initialization="mean", constant_bounds=(-7, 3))
        rmse, params, info = pe.estimate_parameters_with_info(["C", "*", "X_0"])
        assert rmse < 1e-6
        assert params[0] == pytest.approx(2.0)
        assert info["converged"] is False
        assert "ABNORMAL" in info["message"]

    def test_no_optimizer_run_gives_empty_info(self):
        pe = ParameterEstimator(_X, _y_exact)
        assert pe.estimate_parameters_with_info(["X_0", "+", "X_1"])[2] == {}

    def test_estimate_parameters_matches_with_info(self):
        info_pe = ParameterEstimator(_X, _y_linear, seed=3)
        plain_pe = ParameterEstimator(_X, _y_linear, seed=3)
        rmse, params, _ = info_pe.estimate_parameters_with_info(["C", "*", "X_0", "+", "C"])
        rmse2, params2 = plain_pe.estimate_parameters(["C", "*", "X_0", "+", "C"])
        assert rmse == rmse2 and np.array_equal(params, params2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_aborted_fit_keeps_best_finite_value(self):
        # Starting at C = 4 the fit improves towards C = 2, then a step lands below 1.5, where the
        # expression is undefined, and L-BFGS-B aborts reporting NaN.
        pe = ParameterEstimator(
            _X_SQRT,
            _y_SQRT,
            symbol_library=SymbolLibrary.default_symbols(1),
            initialization="mean",
            constant_bounds=(-2, 10),
        )
        rmse, params, info = pe.estimate_parameters_with_info(_SQRT_EXPR)
        assert np.isfinite(rmse) and rmse < 0.5
        assert params[0] > 1.5
        expected = np.sqrt(np.mean((np.sqrt(params[0] - _X_SQRT[:, 0]) - _y_SQRT) ** 2))
        assert rmse == pytest.approx(expected)
        assert info["converged"] is False
        assert "message" in info

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_never_finite_returns_non_finite(self):
        # Midpoint start C = 0 lies where the expression is undefined; nothing finite is found.
        pe = ParameterEstimator(
            _X_SQRT, _y_SQRT, symbol_library=SymbolLibrary.default_symbols(1), initialization="mean"
        )
        rmse, params, info = pe.estimate_parameters_with_info(_SQRT_EXPR)
        assert not np.isfinite(rmse)
        assert not np.isfinite(pe.estimate_parameters(_SQRT_EXPR)[0])
        assert params.size == 1
        assert info["converged"] is False


class TestOptimizeParameters:
    def test_mean_initialization_converges(self):
        pe = ParameterEstimator(_X, _y_linear, initialization="mean")
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert abs(params[0] - 2.0) < 0.01

    def test_max_iter_limits_optimizer_iterations(self):
        # Unlimited, this fit takes 5 iterations. (scipy's L-BFGS-B runs one iteration at maxiter=0.)
        pe = ParameterEstimator(_X, _y_linear, seed=0, max_iter=2)
        assert pe.estimate_parameters_with_info(["C", "*", "X_0"])[2]["iterations"] <= 2

    def test_custom_constant_bounds_respected(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = 10.0 * X[:, 0]
        pe = ParameterEstimator(X, y, constant_bounds=(5, 15), seed=0)
        rmse, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert rmse < 1e-4
        assert 9.9 < params[0] < 10.1

    # Reason: clause ("the default_constant setting if given, else uniform_default(constant_bounds)").
    # y = 50 x: under the uniform default the best C in [-5, 5] is the bound 5; under bounds (0, inf) it is 50.
    @pytest.mark.parametrize(
        "settings, expected_c",
        [
            ({}, 5.0),
            ({"default_constant": Constant(prior=UniformSampling(1, 100), bounds=(0, float("inf")))}, 50.0),
        ],
        ids=["uniform-default", "default-constant"],
    )
    def test_default_constant_setting_replaces_uniform_default(self, settings, expected_c):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        pe = ParameterEstimator(X, 50.0 * X[:, 0], seed=0, **settings)
        _, params = pe.estimate_parameters(["C", "*", "X_0"])
        assert params[0] == pytest.approx(expected_c, rel=1e-3)

    def test_random_start_does_not_depend_on_earlier_fits(self, fit_starts):
        expr = ["C", "*", "X_0"]
        ParameterEstimator(_X, _y_linear, seed=42).estimate_parameters(expr)
        used = ParameterEstimator(_X, _y_linear, seed=42)
        used.estimate_parameters(["C", "+", "X_1"])
        used.estimate_parameters(expr)
        np.testing.assert_array_equal(fit_starts[0], fit_starts[2])

    @pytest.mark.parametrize(
        "seed_b, expr_b",
        [(99, ["C", "*", "X_0"]), (1, ["C", "*", "X_1"])],
        ids=["other-seed", "other-expression"],
    )
    def test_random_start_differs_by_seed_and_expression(self, fit_starts, seed_b, expr_b):
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(["C", "*", "X_0"])
        ParameterEstimator(_X, _y_linear, seed=seed_b).estimate_parameters(expr_b)
        assert not np.allclose(fit_starts[0], fit_starts[1])

    def test_random_start_without_seed_is_random(self, fit_starts):
        # Same estimator, same expression: each fit draws anew (not one draw per estimator).
        pe = ParameterEstimator(_X, _y_linear)
        pe.estimate_parameters(["C", "*", "X_0"])
        pe.estimate_parameters(["C", "*", "X_0"])
        assert not np.allclose(fit_starts[0], fit_starts[1])

    def test_random_start_of_tree_equals_its_tokens(self, fit_starts):
        # Infix order with brackets differs from the tree's prefix order (* + X_0 C C).
        tokens = ["(", "X_0", "+", "C", ")", "*", "C"]
        pe = ParameterEstimator(_X, _y_linear, seed=3)
        pe.estimate_parameters(tokens_to_tree(tokens, pe.symbol_library))
        pe.estimate_parameters(tokens)
        np.testing.assert_array_equal(fit_starts[0], fit_starts[1])

    @pytest.mark.parametrize("seed", [None, 1], ids=["no-seed", "seed"])
    def test_random_start_within_constant_bounds(self, fit_starts, seed):
        pe = ParameterEstimator(_X, _y_linear, seed=seed, constant_bounds=(10, 11))
        pe.estimate_parameters(["C", "*", "X_0", "+", "C", "*", "X_1", "+", "C"])
        assert fit_starts[0].shape == (3,)
        assert np.all((fit_starts[0] >= 10) & (fit_starts[0] <= 11))

    @pytest.mark.parametrize("seed", [None, 1, 2], ids=["no-seed", "seed-1", "seed-2"])
    def test_mean_start_is_midpoint_for_any_seed(self, fit_starts, seed):
        pe = ParameterEstimator(_X, _y_linear, seed=seed, initialization="mean", constant_bounds=(-2, 10))
        pe.estimate_parameters(["C", "*", "X_0", "+", "C"])
        np.testing.assert_array_equal(fit_starts[0], [4.0, 4.0])  # (-2 + 10) / 2

    # Reason: clause (initialization "mean" takes the prior's ppf(0.5); the default_constant setting gives
    # the prior; the default_constant attribute). LogUniform(0.1, 10): ppf(0.5) = 10 ** ((-1 + 1) / 2) = 1.
    def test_mean_start_follows_default_constant_prior(self, fit_starts):
        default = Constant(prior=LogUniformSampling(0.1, 10), bounds=(0, float("inf")))
        pe = ParameterEstimator(_X, _y_linear, initialization="mean", default_constant=default)
        pe.estimate_parameters(["C", "*", "X_0"])
        np.testing.assert_allclose(fit_starts[0], [1.0])
        assert pe.default_constant.to_dict() == default.to_dict()

    @pytest.mark.slow
    def test_random_start_same_across_processes(self):
        # The start, specs included in its seed, must not depend on Python's per-process string hashing.
        code = (
            "import numpy as np\n"
            "from SRToolkit.dataset.sampling import LogUniformSampling\n"
            "from SRToolkit.evaluation.parameter_estimator import ParameterEstimator\n"
            "from SRToolkit.utils.free_constants import Constant\n"
            "X = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
            "pe = ParameterEstimator(X, X[:, 0], seed=1, max_iter=1)\n"
            "spec = Constant(label='C_0', prior=LogUniformSampling(0.1, 10.0))\n"
            "print(repr(pe.estimate_parameters(['C_0', '*', 'X_0', '+', 'C'], [spec])[1].tolist()))\n"
        )
        outputs = [
            subprocess.run(
                [sys.executable, "-c", code],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            for hash_seed in ("1", "2")
        ]
        assert outputs[0] == outputs[1]


# ── free_constants: specs of labelled constants ───────────────────────────────


class _FixedDraw(Sampler):
    """A prior without ppf whose every draw is 0.7."""

    def __call__(self, sample_size, rng=None):
        return np.full(sample_size, 0.7)

    def to_dict(self):
        return {"sampler_class": "tests.evaluation.test_parameter_estimator._FixedDraw"}


class _NoPpfUniform(Sampler):
    """A prior without ppf drawing uniformly on [-1, 1] from ``rng`` (NumPy's global state if None)."""

    def __call__(self, sample_size, rng=None):
        return (np.random if rng is None else rng).uniform(-1.0, 1.0, sample_size)

    def to_dict(self):
        return {"sampler_class": "tests.evaluation.test_parameter_estimator._NoPpfUniform"}


_SIX_LABELS = ["C_0", "+", "C_1", "+", "C_2", "+", "C_3", "+", "C_4", "+", "C_5"]


def _six_specs(**fields):
    return [Constant(label=f"C_{i}", **fields) for i in range(6)]


class TestFreeConstantSpecs:
    @pytest.mark.parametrize("backend", ["stack", "stack_py", "codegen"])
    def test_all_fixed_is_forward_pass(self, fit_starts, backend):
        # Residual C_1 = 0.5 at every sample, so RMSE = 0.5.
        pe = ParameterEstimator(_X, _y_linear, seed=0, backend=backend)
        specs = [Constant(label="C_0", value=2.0, fixed=True), Constant(label="C_1", value=0.5, fixed=True)]
        rmse, params, info = pe.estimate_parameters_with_info(["C_0", "*", "X_0", "+", "C_1"], specs)
        assert rmse == pytest.approx(0.5)
        np.testing.assert_array_equal(params, [2.0, 0.5])
        assert info == {}
        assert fit_starts == []

    def test_mixed_keeps_fixed_value_and_fits_free(self):
        # Parameters [C, C_0]; C_0 = 50 lies outside the default bounds (-5, 5) and is used as is.
        pe = ParameterEstimator(_X, 50.0 * _X[:, 0] + 3.0, seed=0)
        rmse, params = pe.estimate_parameters(
            ["C", "+", "C_0", "*", "X_0"], [Constant(label="C_0", value=50.0, fixed=True)]
        )
        assert rmse < 1e-6
        assert params[1] == 50.0
        np.testing.assert_allclose(params, [3.0, 50.0], atol=1e-4)

    @pytest.mark.parametrize("initialization", ["random", "mean"])
    def test_value_is_first_start(self, fit_starts, initialization):
        pe = ParameterEstimator(_X, _y_linear, seed=0, initialization=initialization)
        pe.estimate_parameters(["C_0", "*", "X_0", "+", "C_1"], [Constant(label="C_0", value=1.5)])
        assert fit_starts[0][0] == 1.5

    @pytest.mark.parametrize(
        "bounds, coefficient, expected",
        [((0.0, 1.0), 2.0, 1.0), ((-np.inf, np.inf), 50.0, 50.0), ((0.0, np.inf), 50.0, 50.0)],
        ids=["clipped", "unbounded", "upper-unbounded"],
    )
    def test_bounds_limit_the_fit(self, bounds, coefficient, expected):
        # RMSE of C*X_0 against k*X_0 decreases towards C = k, so a bound below k is the optimum.
        pe = ParameterEstimator(_X, coefficient * _X[:, 0], seed=0)
        _, params = pe.estimate_parameters(["C_0", "*", "X_0"], [Constant(label="C_0", bounds=bounds)])
        assert bounds[0] <= params[0] <= bounds[1]
        np.testing.assert_allclose(params, [expected], atol=1e-3)

    @pytest.mark.parametrize("seed", [0, None], ids=["seed", "no-seed"])
    def test_random_start_outside_bounds_is_clipped(self, fit_starts, seed):
        pe = ParameterEstimator(_X, _y_linear, seed=seed, max_iter=0)
        pe.estimate_parameters(_SIX_LABELS, _six_specs(prior=UniformSampling(-100.0, 100.0), bounds=(10.0, 11.0)))
        assert np.all((fit_starts[0] >= 10.0) & (fit_starts[0] <= 11.0))
        # 6 draws from U(-100, 100): most fall outside [10, 11] and land on a bound.
        assert np.any(np.isin(fit_starts[0], [10.0, 11.0]))

    def test_mean_start_outside_bounds_is_clipped_to_nearest(self, fit_starts):
        # Median of U(-100, 100) is 0, nearest bound of (10, 11) is 10.
        pe = ParameterEstimator(_X, _y_linear, initialization="mean", max_iter=0)
        pe.estimate_parameters(
            ["C_0", "*", "X_0"], [Constant(label="C_0", prior=UniformSampling(-100.0, 100.0), bounds=(10.0, 11.0))]
        )
        np.testing.assert_array_equal(fit_starts[0], [10.0])

    def test_random_start_follows_prior(self, fit_starts):
        # log10 of a LogUniform(0.01, 100) draw is uniform on [-2, 2]: half the draws are below 1
        # (sd 0.046 for 120 draws); a uniform draw on the bounds would be below 1 with p = 1e-4.
        specs = _six_specs(prior=LogUniformSampling(0.01, 100.0), bounds=(0.01, 100.0))
        for seed in range(20):
            ParameterEstimator(_X, _y_linear, seed=seed, max_iter=0).estimate_parameters(_SIX_LABELS, specs)
        starts = np.concatenate(fit_starts)
        assert starts.size == 120
        assert 0.3 < np.mean(starts < 1.0) < 0.7

    def test_mean_start_is_prior_median(self, fit_starts):
        # ppf(0.5) of LogUniform(0.1, 10) = 10 ** ((-1 + 1) / 2) = 1; the midpoint would be 5.05.
        pe = ParameterEstimator(_X, _y_linear, initialization="mean", max_iter=0)
        pe.estimate_parameters(["C_0", "*", "X_0"], [Constant(label="C_0", prior=LogUniformSampling(0.1, 10.0))])
        np.testing.assert_allclose(fit_starts[0], [1.0], rtol=1e-12)

    def test_prior_without_ppf_raises_under_mean(self):
        pe = ParameterEstimator(_X, _y_linear, initialization="mean")
        with pytest.raises(NotImplementedError):
            pe.estimate_parameters(["C_0", "*", "X_0"], [Constant(label="C_0", prior=_FixedDraw())])

    @pytest.mark.parametrize(
        "initialization, spec",
        [
            ("random", Constant(label="C_0", prior=_FixedDraw())),
            ("mean", Constant(label="C_0", prior=_FixedDraw(), value=0.7)),
            ("mean", Constant(label="C_0", prior=_FixedDraw(), value=0.7, fixed=True)),
        ],
        ids=["random-draws", "mean-with-value", "mean-fixed"],
    )
    def test_prior_without_ppf_usable_when_not_needed(self, initialization, spec):
        # Every way the prior is handled starts (or stays) at 0.7, the exact optimum.
        pe = ParameterEstimator(_X, 0.7 * _X[:, 0], seed=0, initialization=initialization, max_iter=0)
        rmse, params = pe.estimate_parameters(["C_0", "*", "X_0"], [spec])
        np.testing.assert_array_equal(params, [0.7])
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_seed_with_specs_reproducible_and_order_independent(self, fit_starts):
        expr = ["C_0", "*", "X_0", "+", "C_1", "+", "C_2"]
        a = Constant(label="C_1", bounds=(0.0, 1.0))
        b = Constant(label="C_2", prior=_NoPpfUniform())  # drawn directly, must use the seeded generator
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, [a, b])
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, [a, b])
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, [b, a])
        np.testing.assert_array_equal(fit_starts[0], fit_starts[1])
        np.testing.assert_array_equal(fit_starts[0], fit_starts[2])

    @pytest.mark.parametrize(
        "other",
        [
            None,
            [Constant(label="C_1", value=2.0)],
            [Constant(label="C_1", value=1.0, prior=LogUniformSampling(0.1, 10.0))],
        ],
        ids=["no-specs", "other-value", "other-prior"],
    )
    def test_seed_depends_on_specs(self, fit_starts, other):
        # C_0 has no spec in any run, so only the seed can change its drawn start.
        expr = ["C_0", "*", "X_0", "+", "C_1"]
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, [Constant(label="C_1", value=1.0)])
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, other)
        assert fit_starts[0][0] != fit_starts[1][0]

    def test_empty_specs_same_seed_as_none(self, fit_starts):
        expr = ["C_0", "*", "X_0", "+", "C_1"]
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, [])
        ParameterEstimator(_X, _y_linear, seed=1).estimate_parameters(expr, None)
        np.testing.assert_array_equal(fit_starts[0], fit_starts[1])

    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_never_finite_returns_full_vector_with_fixed(self):
        # Parameters [C_1, C_0]; the midpoint start C_1 = 0 is undefined on all of X (as in
        # test_never_finite_returns_non_finite), so nothing finite is found.
        pe = ParameterEstimator(
            _X_SQRT, _y_SQRT, symbol_library=SymbolLibrary.default_symbols(1), initialization="mean"
        )
        expr = ["sqrt", "(", "C_1", "-", "X_0", ")", "*", "C_0"]
        rmse, params = pe.estimate_parameters(expr, [Constant(label="C_0", value=3.0, fixed=True)])
        assert not np.isfinite(rmse)
        assert params.size == 2
        assert params[1] == 3.0
        assert np.isfinite(params[0])  # the optimizer's final point, not a placeholder

    def test_registered_symbol_named_like_label_is_not_a_constant(self):
        # With this estimator's library C_1 is a named constant (2), so C_0 is the only parameter.
        sl = SymbolLibrary.default_symbols(2)
        sl.add_symbol("C_1", "lit", 5, "2.0")
        pe = ParameterEstimator(_X, 2.0 * _X[:, 0] + 3.0, symbol_library=sl)
        rmse, params = pe.estimate_parameters(
            ["C_1", "*", "X_0", "+", "C_0"], [Constant(label="C_0", value=3.0, fixed=True)]
        )
        assert rmse == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_array_equal(params, [3.0])

    def test_spec_for_unknown_label_raises(self):
        pe = ParameterEstimator(_X, _y_linear)
        with pytest.raises(ValueError, match="C_9"):
            pe.estimate_parameters(["C_0", "*", "X_0"], [Constant(label="C_9", value=1.0)])
