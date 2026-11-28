from dataclasses import dataclass

import numpy as np
import pytest

from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
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
from SRToolkit.utils.expression import Expression
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.settings import Settings, SettingsSection
from SRToolkit.utils.symbol_library import SymbolLibrary


def _setup(ranking, X, y, ground_truth, lib=None, seed=0, gt_specs=None, settings=None, **task):
    # setup with the background knowledge built from lib and the task facts in **task; the ground truth
    # becomes an Expression in its library with the specs gt_specs, as the evaluator passes it.
    bk = BackgroundKnowledge(SymbolLibrary.default_symbols(X.shape[1]) if lib is None else lib, **task)
    gt = None if ground_truth is None else Expression(ground_truth, bk.symbol_library, gt_specs)
    ranking.setup(X, y, gt, bk, seed, Settings() if settings is None else settings)
    return ranking


_GT_C0 = ["X_0", "+", "C_0"]
_X10 = np.linspace(0.0, 1.0, 10).reshape(-1, 1)


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


@dataclass(frozen=True, eq=False)
class _CustomSection(SettingsSection):
    name = "custom_kwarg"
    my_param: float = 1.0


class _CustomKwargRanking(RankingFunction):
    """A custom ranking that reads its own settings section."""

    name = "custom_kwarg"

    def setup(self, X, y, ground_truth, background_knowledge, seed, settings):
        self.my_param = settings.get(_CustomSection).my_param

    def score(self, expr):
        return RankingResult(score=float(self.my_param))


class TestCustomRankingSettings:
    # Reason: clause (setup: a ranking reads its own section with settings.get(MySection)).
    def test_custom_section_reaches_setup(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = f"{_CustomKwargRanking.__module__}.{_CustomKwargRanking.__qualname__}"
        se = SR_evaluator(
            X, ranking_function=path, success_threshold=-1, settings=Settings(custom_kwarg=_CustomSection(0.5))
        )
        assert se.evaluate_expr(["X_0"]).error == 0.5


class TestRMSERankingSetup:
    def test_fit_starts_from_setup_seed(self, fit_starts):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = 2.0 * X[:, 0]
        expr = ["C", "*", "X_0", "+", "C"]
        lib = SymbolLibrary.default_symbols(2)
        _setup(RMSERanking(), X, y, None, lib, seed=1).score(Expression(expr, lib))
        ParameterEstimator(X, y, lib, seed=1).estimate_parameters(expr)
        np.testing.assert_array_equal(fit_starts[0], fit_starts[1])

    # Reason: clause (RMSE: constants are fitted with the estimation and minimize settings and evaluator.backend).
    def test_estimator_gets_the_settings(self):
        settings = Settings(minimize={"max_iter": 3}, evaluator={"backend": "codegen"})
        ranking = _setup(RMSERanking(), _X10, _X10[:, 0], None, settings=settings)
        assert ranking.parameter_estimator.settings is settings


class TestBEDRankingSettings:
    # Reason: clause (BED reads the bed section and evaluator.backend).
    def test_bed_reads_its_section_and_backend(self):
        settings = Settings(bed={"num_points_sampled": 5, "num_consts_sampled": 4}, evaluator={"backend": "codegen"})
        ranking = _setup(BEDRanking(), _X10, None, ["X_0", "+", "C"], settings=settings)
        assert (ranking.backend, ranking.gt_behavior.shape) == ("codegen", (5, 4))

    # Reason: clause (bed_gt: "Rows are checked against X when the ranking is set up").
    def test_bed_gt_rows_must_match_X(self):
        settings = Settings(bed={"bed_gt": np.zeros((3, 32))})
        with pytest.raises(ValueError, match="rows"):
            _setup(BEDRanking(), _X10, None, None, settings=settings)


class TestRMSERankingSpecs:
    # Reason: clause (RankingFunction.score: the fit uses the given specs, bounds included). With y = 2x
    # and C_0 bounded to (0, 1), the best fit is at the bound: residual x, RMSE = sqrt((1 + 4 + 9) / 3).
    def test_rmse_score_fits_within_spec_bounds(self):
        X = np.array([[1.0], [2.0], [3.0]])
        ranking = _setup(RMSERanking(), X, 2 * X[:, 0], None)
        result = ranking.score(Expression(["C_0", "*", "X_0"], free_constants=[Constant("C_0", bounds=(0.0, 1.0))]))
        np.testing.assert_allclose(result.parameters, [1.0], atol=1e-6)
        assert result.score == pytest.approx(np.sqrt(14 / 3), rel=1e-6)


_DEFAULT_100 = {"default_constant": Constant(prior=UniformSampling(100, 101), bounds=(100, 101))}
_ALL_POINTS = Settings(bed={"num_points_sampled": -1})


def _fixed(value):
    return Constant("C_0", value=value, fixed=True)


class TestGroundTruthConstants:
    # Reason: clause (RMSERanking: y is generated from ground_truth "with the fixed values of the
    # ground_truth_constants setting"). y = 2x, so the best fit of C*X_0 is C = 2 with RMSE 0.
    def test_rmse_setup_generates_y_with_fixed_values(self):
        ranking = _setup(RMSERanking(), _X10, None, ["C_0", "*", "X_0"], gt_specs=[_fixed(2.0)])
        result = ranking.score(Expression(["C", "*", "X_0"]))
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
            (BEDRanking, _GT_C0, [Constant("C_0", value=9.0)], "free_constants"),
        ],
        ids=["rmse-bare", "rmse-not-fixed", "bed-free-value-outside-default-bounds"],
    )
    def test_setup_rejects_unusable_specs(self, ranking, ground_truth, specs, match):
        with pytest.raises(ValueError, match=match):
            _setup(ranking(), _X10, None, ground_truth, gt_specs=specs)

    # Reason: clause (BED: fixed ground-truth values are kept; the default threshold is the noise floor,
    # at least 1e-7). A fixed constant makes the behaviour deterministic: noise floor 0, threshold 1e-7.
    # The W1 distance between point masses is their difference, so a candidate fixed at 0 scores 9.81.
    def test_bed_fixed_ground_truth_constant(self):
        ranking = _setup(BEDRanking(), _X10, None, _GT_C0, gt_specs=[_fixed(9.81)], settings=_ALL_POINTS)
        assert ranking.default_success_threshold() == 1e-7
        assert ranking.score(Expression(_GT_C0, free_constants=[_fixed(9.81)])).score < 1e-7
        assert ranking.score(Expression(_GT_C0, free_constants=[_fixed(0.0)])).score == pytest.approx(9.81)

    # Reason: clause (BED: the ground truth's parameters follow the priors of its specs). With
    # C_0 ~ U(100, 101) and the candidate fixed at 100.5, BED is E|U - 0.5| = 1/4 for U ~ U(0, 1); 32
    # Latin Hypercube samples keep the sample mean within 0.02. Under the default prior it would be ~100.
    def test_bed_ground_truth_prior_shapes_behaviour(self):
        gt_specs = [Constant("C_0", prior=UniformSampling(100, 101))]
        ranking = _setup(BEDRanking(), _X10, None, _GT_C0, gt_specs=gt_specs, settings=_ALL_POINTS)
        assert ranking.score(Expression(_GT_C0, free_constants=[_fixed(100.5)])).score == pytest.approx(0.25, abs=0.02)

    # Reason: clause (BED: parameters without a spec are drawn from the background knowledge's
    # default_constant, on both the ground-truth and the candidate side). The default C ~ U(100, 101)
    # against a constant fixed at 100.5 gives E|U - 0.5| = 1/4 as above; under the uniform default
    # (-5, 5) it would be about 100. Without one, the default is uniform on constant_bounds:
    # (100, 101) gives the same.
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
        ranking = _setup(BEDRanking(), _X10, None, ground_truth, gt_specs=gt_specs, settings=_ALL_POINTS, **default)
        assert ranking.score(Expression(candidate, free_constants=candidate_specs)).score == pytest.approx(
            0.25, abs=0.02
        )

    # Reason: clause (default threshold: the noise floor of the ground truth's unfixed parameters, drawn
    # from the default spec, times 1.1, at least 1e-7). With C ~ U(0, 1e-9) every self-BED is at most
    # 1e-9, so 1.1e-9 < 1e-7 and the floor applies; under the uniform default it would be about 1.
    def test_bed_threshold_follows_default_constant(self):
        default = Constant(prior=UniformSampling(0, 1e-9), bounds=(0, 1))
        ranking = _setup(BEDRanking(), _X10, None, ["X_0", "+", "C"], settings=_ALL_POINTS, default_constant=default)
        assert ranking.default_success_threshold() == 1e-7

    # Reason: default (neither default_constant nor constant_bounds is given: the ground truth's constants
    # are sampled from uniform_default((-5, 5)), exactly as create_behavior_matrix does with that range).
    def test_bed_without_settings_samples_constants_in_default_range(self):
        sl = SymbolLibrary.default_symbols(1)
        ranking = _setup(BEDRanking(), _X10, None, ["X_0", "+", "C"], sl, settings=_ALL_POINTS)
        expected = create_behavior_matrix(["X_0", "+", "C"], ranking.bed_X, 32, (-5, 5), sl, 0)
        np.testing.assert_array_equal(ranking.gt_behavior, expected)

    # Reason: clause (BED uses the background knowledge's symbol_library). C_1 is the named constant 50
    # there, so the ground truth X_0 + C_1 is x + 50 in every column and the candidate X_0 is x: every
    # point is a shift by 50, BED 50. With another library C_1 would be a free constant drawn from
    # U(-5, 5), giving a BED of about 2.5.
    def test_bed_uses_the_background_knowledge_library(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "50.0")
        ranking = _setup(BEDRanking(), _X10, None, ["X_0", "+", "C_1"], lib, settings=_ALL_POINTS)
        assert ranking.score(Expression(["X_0"], lib)).score == pytest.approx(50.0, abs=1e-9)

    # Reason: implied (BEDRanking.score also takes a token list, as RMSERanking does, read in the
    # background knowledge's library): the same BED of 50 as for the Expression above.
    def test_bed_scores_a_token_list_in_its_library(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "50.0")
        ranking = _setup(BEDRanking(), _X10, None, ["X_0", "+", "C_1"], lib, settings=_ALL_POINTS)
        assert ranking.score(["X_0"]).score == pytest.approx(50.0, abs=1e-9)

    # Reason: clause (setup: y is generated by evaluating ground_truth with the given symbol library). With
    # C_1 a named constant of value 3, y = x + 3 needs no spec and the best fit of X_0 + C is C = 3.
    # Reason: regression (setup with a token-list ground truth crashed in RMSE and was read in the
    # default library by BED) + clause (a token list is read in the background knowledge's library).
    # C_1 is the named constant 50 there, so X_0 + 50 matches the ground truth exactly.
    @pytest.mark.parametrize("ranking", [RMSERanking, BEDRanking], ids=["rmse", "bed"])
    def test_setup_reads_a_token_list_ground_truth_in_its_library(self, ranking):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "50.0")
        scored = ranking()
        scored.setup(_X10, None, ["X_0", "+", "C_1"], BackgroundKnowledge(lib), 0, _ALL_POINTS)
        assert scored.score(Expression(["X_0", "+", "50"], lib)).score == pytest.approx(0.0, abs=1e-9)

    def test_rmse_setup_generates_y_with_the_given_library(self):
        lib = SymbolLibrary.default_symbols(1)
        lib.add_symbol("C_1", "lit", 5, "3.0")
        ranking = _setup(RMSERanking(), _X10, None, ["X_0", "+", "C_1"], lib)
        result = ranking.score(Expression(["X_0", "+", "C"], lib))
        np.testing.assert_allclose(result.parameters, [3.0], atol=1e-6)
