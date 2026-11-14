import numpy as np
import pytest

from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.evaluation.ranking import (
    BEDRanking,
    RankingFunction,
    RankingResult,
    RMSERanking,
    build_ranking,
)
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
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

    def test_non_ranking_class_raises_valueerror(self):
        with pytest.raises(ValueError, match="not a RankingFunction subclass"):
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

    def score(self, expr):
        return RankingResult(score=float(self.my_param))


class TestCustomRankingSettings:
    def test_custom_kwarg_flows_through_to_setup(self):
        # An unknown-to-RankingSettings kwarg still reaches a custom ranking's setup at runtime.
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        path = f"{_CustomKwargRanking.__module__}.{_CustomKwargRanking.__qualname__}"
        se = SR_evaluator(X, ranking_function=path, success_threshold=-1, my_param=0.5)
        assert se.ranking.my_param == 0.5
        assert se.evaluate_expr(["X_0"]) == 0.5


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
