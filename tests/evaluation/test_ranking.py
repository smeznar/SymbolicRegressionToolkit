import numpy as np
import pytest

from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.evaluation.ranking import (
    BEDRanking,
    RankingFunction,
    RankingResult,
    RMSERanking,
    build_ranking,
)
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary


class _CustomRanking(RMSERanking):
    """A user-defined ranking, resolvable by its import path."""

    name = "custom"


class TestBuildRanking:
    def test_instance_passthrough(self):
        r = RMSERanking()
        assert build_ranking(r) is r

    def test_alias(self):
        assert isinstance(build_ranking("rmse"), RMSERanking)
        assert isinstance(build_ranking("BED"), BEDRanking)  # case-insensitive

    def test_import_path(self):
        path = f"{_CustomRanking.__module__}.{_CustomRanking.__qualname__}"
        assert isinstance(build_ranking(path), _CustomRanking)

    def test_unknown_bare_name_warns_and_falls_back(self):
        with pytest.warns(UserWarning, match="not supported"):
            r = build_ranking("does_not_exist")
        assert isinstance(r, RMSERanking)

    def test_unimportable_path_raises_importerror(self):
        with pytest.raises(ImportError, match="Cannot import ranking class"):
            build_ranking("nonexistent.module.Foo")

    def test_non_ranking_class_raises_typeerror(self):
        with pytest.raises(TypeError, match="not a subclass of RankingFunction"):
            build_ranking("builtins.dict")

    def test_non_string_non_ranking_raises_typeerror(self):
        with pytest.raises(TypeError):
            build_ranking(42)


class TestRankingSerialization:
    @pytest.mark.parametrize("ranking", [RMSERanking(), BEDRanking(), _CustomRanking()])
    def test_to_dict_from_dict_roundtrip(self, ranking):
        d = ranking.to_dict()
        assert d["ranking_class"] == f"{type(ranking).__module__}.{type(ranking).__qualname__}"
        restored = RankingFunction.from_dict(d)
        assert type(restored) is type(ranking)

    def test_from_dict_accepts_alias(self):
        assert isinstance(RankingFunction.from_dict({"ranking_class": "bed"}), BEDRanking)

    def test_from_dict_ignores_missing_bundle_metadata(self):
        # No "_bundle" key -> _auto_bind returns the dict unchanged.
        restored = RankingFunction.from_dict({"ranking_class": "rmse", "extra": "ignored"})
        assert isinstance(restored, RMSERanking)


class _CustomKwargRanking(RankingFunction):
    """A custom ranking that reads an arbitrary key from the settings dict."""

    name = "custom_kwarg"

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        self.my_param = settings.get("my_param")

    def score(self, expr, free_constants=None):
        return RankingResult(score=float(self.my_param))


class TestCustomRankingSettings:
    def test_custom_kwarg_flows_through_to_setup(self):
        # An unknown-to-RankingSettings kwarg still reaches a custom ranking's setup at runtime.
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = f"{_CustomKwargRanking.__module__}.{_CustomKwargRanking.__qualname__}"
        se = SR_evaluator(X, ranking_function=path, success_threshold=-1, my_param=0.5)
        assert se.ranking.my_param == 0.5
        assert se.evaluate_expr(["X_0"]).error == 0.5


class TestRMSERankingSetup:
    def test_fit_starts_from_setup_seed(self, fit_starts):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = 2.0 * X[:, 0]
        expr = ["C", "*", "X_0", "+", "C"]
        lib = SymbolLibrary.default_symbols(2)
        ranking = RMSERanking()
        ranking.setup(X, y, None, lib, 1, {})
        ranking.score(expr)
        ParameterEstimator(X, y, lib, seed=1).estimate_parameters(expr)
        np.testing.assert_array_equal(fit_starts[0], fit_starts[1])


class TestBEDRankingBackend:
    # Reason: clause (RankingSettings.backend applies to both rankings). An unknown backend only fails
    # when an expression is compiled, so the error shows the setting was passed on.
    X = np.random.default_rng(0).random((4, 1))

    def test_backend_is_used_for_the_ground_truth(self):
        ranking = BEDRanking()
        with pytest.raises(ValueError, match="backend"):
            ranking.setup(self.X, None, ["X_0", "+", "C"], SymbolLibrary.default_symbols(1), 0, {"backend": "fortran"})

    def test_backend_is_used_for_scored_expressions(self):
        ranking = BEDRanking()
        settings = {"backend": "fortran", "bed_gt": np.zeros((4, 32))}
        ranking.setup(self.X, None, None, SymbolLibrary.default_symbols(1), 0, settings)
        with pytest.raises(ValueError, match="backend"):
            ranking.score(["X_0"])


class TestBEDRankingSettings:
    # Reason: regression (setup let an explicit None override the BED defaults; None means the default
    # for every setting). 100 rows, so the default 64 sampled points is a real subsample.
    @pytest.mark.parametrize("name", ["num_consts_sampled", "num_points_sampled", "constant_bounds", "backend"])
    def test_none_setting_is_the_default(self, name):
        X = np.random.default_rng(0).random((100, 1))
        lib = SymbolLibrary.default_symbols(1)
        default, given_none = BEDRanking(), BEDRanking()
        default.setup(X, None, ["C", "*", "X_0"], lib, 0, {})
        given_none.setup(X, None, ["C", "*", "X_0"], lib, 0, {name: None})
        assert given_none.score(["C", "+", "X_0"]).score == default.score(["C", "+", "X_0"]).score


class _NoDefaults(RankingFunction):
    name = "no_defaults"

    def setup(self, X, y, ground_truth, symbol_library, seed, settings):
        pass

    def score(self, expr, free_constants=None):
        return RankingResult(score=0.0)


class TestDefaultSettings:
    # Reason: clause (default_settings: "the defaults of the settings this ranking reads"; "A new dict;
    # empty in the base class"; RMSE leaves out default_constant). Values from the RankingSettings docs.
    @pytest.mark.parametrize(
        "ranking, expected",
        [
            (_NoDefaults(), {}),
            (
                RMSERanking(),
                {
                    "method": "L-BFGS-B", "tol": 1e-6, "gtol": 1e-3, "max_iter": 100, "constant_bounds": (-5, 5),
                    "initialization": "random", "backend": "stack",
                },
            ),
            (
                BEDRanking(),
                {"num_consts_sampled": 32, "num_points_sampled": 64, "constant_bounds": (-5, 5), "backend": "stack"},
            ),
        ],
        ids=["base", "rmse", "bed"],
    )  # fmt: skip
    def test_default_settings(self, ranking, expected):
        first = ranking.default_settings()
        assert first == expected
        first["extra"] = 1
        assert ranking.default_settings() == expected

    # Reason: implied (the checkpoint fingerprint treats these defaults as "left out", so they must be
    # the ones the RMSE ranking's estimator actually uses).
    def test_rmse_default_settings_match_the_estimator(self):
        used = dict(ParameterEstimator(np.ones((3, 1)), np.ones(3)).estimation_settings)
        del used["default_constant"]
        assert RMSERanking().default_settings() == used


class TestRMSERankingSpecs:
    # Reason: clause (RankingFunction.score: the fit uses the given specs, bounds included). With y = 2x
    # and C_0 bounded to (0, 1), the best fit is at the bound: residual x, RMSE = sqrt((1 + 4 + 9) / 3).
    def test_rmse_score_fits_within_spec_bounds(self):
        X = np.array([[1.0], [2.0], [3.0]])
        ranking = RMSERanking()
        ranking.setup(X, 2 * X[:, 0], None, SymbolLibrary.default_symbols(1), 0, {})
        result = ranking.score(["C_0", "*", "X_0"], [Constant("C_0", bounds=(0.0, 1.0))])
        np.testing.assert_allclose(result.parameters, [1.0], atol=1e-6)
        assert result.score == pytest.approx(np.sqrt(14 / 3), rel=1e-6)


_GT_C0 = ["X_0", "+", "C_0"]
_X10 = np.linspace(0.0, 1.0, 10).reshape(-1, 1)


_DEFAULT_100 = {"default_constant": Constant(prior=UniformSampling(100, 101), bounds=(100, 101))}


def _fixed(value):
    return Constant("C_0", value=value, fixed=True)


class TestGroundTruthConstants:
    # Reason: clause (RMSERanking: y is generated from ground_truth "with the fixed values of the
    # ground_truth_constants setting"). y = 2x, so the best fit of C*X_0 is C = 2 with RMSE 0.
    def test_rmse_setup_generates_y_with_fixed_values(self):
        ranking = RMSERanking()
        settings = {"ground_truth_constants": [_fixed(2.0)]}
        ranking.setup(_X10, None, ["C_0", "*", "X_0"], SymbolLibrary.default_symbols(1), 0, settings)
        result = ranking.score(["C", "*", "X_0"])
        np.testing.assert_allclose(result.parameters, [2.0], atol=1e-6)
        assert result.score == pytest.approx(0.0, abs=1e-6)

    # Reason: clause (setup Raises: a ground-truth constant without a fixed value when y is generated;
    # a spec whose label is not a labelled constant of the ground truth, for RMSE and BED).
    @pytest.mark.parametrize(
        "ranking, ground_truth, specs, match",
        [
            (RMSERanking, ["X_0", "+", "C"], None, r"free constant \(C\) without a fixed value"),
            (
                RMSERanking,
                _GT_C0,
                [Constant("C_0", prior=UniformSampling(0, 1))],
                r"free constant \(C_0\) without a fixed value",
            ),
            (RMSERanking, _GT_C0, [Constant("C_1", value=1.0, fixed=True)], r"\['C_1'\] are not labelled constants"),
            (BEDRanking, _GT_C0, [Constant("C_1", value=1.0, fixed=True)], r"\['C_1'\] are not labelled constants"),
        ],
        ids=["rmse-bare", "rmse-not-fixed", "rmse-unknown-label", "bed-unknown-label"],
    )
    def test_setup_rejects_unusable_specs(self, ranking, ground_truth, specs, match):
        with pytest.raises(ValueError, match=match):
            ranking().setup(
                _X10, None, ground_truth, SymbolLibrary.default_symbols(1), 0, {"ground_truth_constants": specs}
            )

    # Reason: clause (BED: fixed ground-truth values are kept; the default threshold is the noise floor,
    # at least 1e-7). A fixed constant makes the behaviour deterministic: noise floor 0, threshold 1e-7.
    # The W1 distance between point masses is their difference, so a candidate fixed at 0 scores 9.81.
    def test_bed_fixed_ground_truth_constant(self):
        ranking = BEDRanking()
        settings = {"ground_truth_constants": [_fixed(9.81)], "num_points_sampled": -1}
        ranking.setup(_X10, None, _GT_C0, SymbolLibrary.default_symbols(1), 0, settings)
        assert ranking.default_success_threshold() == 1e-7
        assert ranking.score(_GT_C0, [_fixed(9.81)]).score < 1e-7
        assert ranking.score(_GT_C0, [_fixed(0.0)]).score == pytest.approx(9.81)

    # Reason: clause (BED: the ground truth's parameters follow the priors of its specs). With
    # C_0 ~ U(100, 101) and the candidate fixed at 100.5, BED is E|U - 0.5| = 1/4 for U ~ U(0, 1); 32
    # Latin Hypercube samples keep the sample mean within 0.02. Under the default prior it would be ~100.
    def test_bed_ground_truth_prior_shapes_behaviour(self):
        ranking = BEDRanking()
        settings = {
            "ground_truth_constants": [Constant("C_0", prior=UniformSampling(100, 101))],
            "num_points_sampled": -1,
            "num_consts_sampled": 32,
        }
        ranking.setup(_X10, None, _GT_C0, SymbolLibrary.default_symbols(1), 0, settings)
        assert ranking.score(_GT_C0, [_fixed(100.5)]).score == pytest.approx(0.25, abs=0.02)

    # Reason: clause (BED: parameters without a spec are drawn from the default_constant setting, on both
    # the ground-truth and the candidate side). The default C ~ U(100, 101) against a constant fixed at
    # 100.5 gives E|U - 0.5| = 1/4 as above; under the uniform default (-5, 5) it would be about 100.
    # Without the setting, the default is uniform on the constant_bounds setting: (100, 101) gives the same.
    @pytest.mark.parametrize(
        "default, ground_truth, gt_specs, candidate, candidate_specs",
        [
            (_DEFAULT_100, ["X_0", "+", "C"], None, _GT_C0, [_fixed(100.5)]),
            (_DEFAULT_100, _GT_C0, [_fixed(100.5)], ["X_0", "+", "C"], None),
            ({"constant_bounds": (100, 101)}, ["X_0", "+", "C"], None, _GT_C0, [_fixed(100.5)]),
        ],
        ids=["ground-truth-side", "candidate-side", "constant-bounds-fallback"],
    )
    def test_bed_default_constant_shapes_columns(self, default, ground_truth, gt_specs, candidate, candidate_specs):
        ranking = BEDRanking()
        settings = {
            **default,
            "ground_truth_constants": gt_specs,
            "num_points_sampled": -1,
            "num_consts_sampled": 32,
        }
        ranking.setup(_X10, None, ground_truth, SymbolLibrary.default_symbols(1), 0, settings)
        assert ranking.score(candidate, candidate_specs).score == pytest.approx(0.25, abs=0.02)

    # Reason: clause (default threshold: the noise floor of the ground truth's unfixed parameters, drawn
    # from the default spec, times 1.1, at least 1e-7). With C ~ U(0, 1e-9) every self-BED is at most
    # 1e-9, so 1.1e-9 < 1e-7 and the floor applies; under the uniform default it would be about 1.
    def test_bed_threshold_follows_default_constant(self):
        ranking = BEDRanking()
        settings = {
            "default_constant": Constant(prior=UniformSampling(0, 1e-9), bounds=(0, 1)),
            "num_points_sampled": -1,
        }
        ranking.setup(_X10, None, ["X_0", "+", "C"], SymbolLibrary.default_symbols(1), 0, settings)
        assert ranking.default_success_threshold() == 1e-7

    # Reason: default (neither default_constant nor constant_bounds is set: the ground truth's constants
    # are sampled from uniform_default((-5, 5)), exactly as create_behavior_matrix does with that range).
    def test_bed_without_settings_samples_constants_in_default_range(self):
        ranking = BEDRanking()
        sl = SymbolLibrary.default_symbols(1)
        ranking.setup(_X10, None, ["X_0", "+", "C"], sl, 0, {"num_points_sampled": -1})
        expected = create_behavior_matrix(["X_0", "+", "C"], ranking.bed_X, 32, (-5, 5), sl, 0)
        np.testing.assert_array_equal(ranking.gt_behavior, expected)

    # Reason: clause (setup: y is generated by evaluating ground_truth with the given symbol library). With
    # C_1 a named constant of value 3, y = x + 3 needs no spec and the best fit of X_0 + C is C = 3.
    def test_rmse_setup_generates_y_with_the_given_library(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "3.0")
        ranking = RMSERanking()
        ranking.setup(_X10, None, ["X_0", "+", "C_1"], lib, 0, {})
        result = ranking.score(["X_0", "+", "C"])
        np.testing.assert_allclose(result.parameters, [3.0], atol=1e-6)
