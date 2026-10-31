import json

import numpy as np
import pytest

from SRToolkit.utils.types import (
    CONST,
    FN,
    FN_POSTFIX,
    FN_PREFIX,
    LEAF,
    LIT,
    OP,
    OP_ADDITIVE,
    OP_MULTIPLICATIVE,
    OP_POWER,
    STATUSES,
    VALID_SYMBOL_TYPES,
    VAR,
    EvalResult,
    ModelResult,
    RankingSettings,
)


class TestSymbolTypeConstants:
    def test_constant_values(self):
        assert VAR == "var"
        assert CONST == "const"
        assert FN == "fn"
        assert OP == "op"
        assert LIT == "lit"

    def test_valid_symbol_types_contains_all_constants(self):
        assert VALID_SYMBOL_TYPES == {VAR, CONST, FN, OP, LIT}

    def test_valid_symbol_types_is_set(self):
        assert isinstance(VALID_SYMBOL_TYPES, set)


class TestPrecedenceConstants:
    def test_operator_levels_ordered(self):
        assert OP_ADDITIVE < OP_MULTIPLICATIVE < OP_POWER

    def test_operator_levels_values(self):
        assert OP_ADDITIVE == 100
        assert OP_MULTIPLICATIVE == 200
        assert OP_POWER == 300

    def test_fn_prefix_above_all_ops(self):
        assert FN_PREFIX > OP_POWER

    def test_fn_postfix_below_zero(self):
        assert FN_POSTFIX < 0

    def test_fn_postfix_distinct_from_get_precedence_fallback(self):
        assert FN_POSTFIX != -1

    def test_leaf_above_all_ops(self):
        assert LEAF > OP_POWER

    def test_fn_prefix_equals_leaf(self):
        assert FN_PREFIX == LEAF


class TestRankingSettings:
    EXPECTED_FIELDS = {
        "method",
        "tol",
        "gtol",
        "max_iter",
        "constant_bounds",
        "initialization",
        "max_constants",
        "max_expr_length",
        "num_points_sampled",
        "samplers",
        "bed_gt",
        "ranking_seed",
        "bed_resample",
        "num_consts_sampled",
        "backend",
    }

    def test_all_fields_in_annotations(self):
        assert self.EXPECTED_FIELDS == set(RankingSettings.__annotations__.keys())

    def test_all_fields_optional(self):
        assert RankingSettings.__required_keys__ == frozenset()
        assert self.EXPECTED_FIELDS == set(RankingSettings.__optional_keys__)

    def test_dict_interface(self):
        settings: RankingSettings = {"method": "L-BFGS-B", "max_iter": 200}
        assert settings.get("method") == "L-BFGS-B"
        assert settings.get("tol", 1e-6) == 1e-6


class TestModelResultConstruction:
    def test_default_fields(self):
        result = ModelResult(expr=["X_0"], error=0.1)
        assert result.parameters is None
        # Each instance must get its own dict, not a shared default
        result2 = ModelResult(expr=["X_1"], error=0.2)
        result.augmentations["key"] = {}
        assert "key" not in result2.augmentations


class TestModelResultAddAugmentation:
    def test_stores_data_and_injects_type(self):
        result = ModelResult(expr=["X_0"], error=0.1)
        result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
        assert "latex" in result.augmentations
        assert result.augmentations["latex"]["value"] == "$X_0$"
        assert result.augmentations["latex"]["_type"] == "LaTeXAugmenter"

    def test_duplicate_key_gets_suffix(self):
        result = ModelResult(expr=["X_0"], error=0.1)
        result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
        result.add_augmentation("latex", {"value": "$X_1$"}, "LaTeXAugmenter")
        assert "latex" in result.augmentations
        assert "latex_1" in result.augmentations
        assert result.augmentations["latex_1"]["value"] == "$X_1$"

    def test_multiple_duplicates_increment_counter(self):
        result = ModelResult(expr=["X_0"], error=0.1)
        result.add_augmentation("latex", {"value": "a"}, "LaTeXAugmenter")
        result.add_augmentation("latex", {"value": "b"}, "LaTeXAugmenter")
        result.add_augmentation("latex", {"value": "c"}, "LaTeXAugmenter")
        assert "latex" in result.augmentations
        assert "latex_1" in result.augmentations
        assert "latex_2" in result.augmentations
        assert result.augmentations["latex_2"]["value"] == "c"

    def test_different_names_do_not_collide(self):
        result = ModelResult(expr=["X_0"], error=0.1)
        result.add_augmentation("foo", {"x": 1}, "A")
        result.add_augmentation("bar", {"x": 2}, "B")
        assert len(result.augmentations) == 2


class TestModelResultToDict:
    def test_basic_serialization(self):
        result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
        d = result.to_dict()
        assert d["expr"] == ["X_0", "+", "C"]
        assert d["error"] == 0.25
        assert isinstance(d["error"], float)
        assert d["parameters"] is None
        assert d["augmentations"] == {}

    def test_numpy_parameters_serialized(self):
        params = np.array([1.5, -2.0])
        result = ModelResult(expr=["C", "*", "X_0", "+", "C"], error=np.float64(0.1))
        result.parameters = params
        d = result.to_dict()
        assert isinstance(d["error"], float)
        assert d["parameters"] == {"__ndarray__": True, "data": [1.5, -2.0]}


class TestModelResultFromDict:
    def test_round_trip_with_parameters(self):
        params = np.array([1.5, -2.0])
        original = ModelResult(expr=["C", "*", "X_0"], error=0.42, parameters=params)
        original.add_augmentation("latex", {"value": "$C X_0$"}, "LaTeXAugmenter")

        restored = ModelResult.from_dict(original.to_dict())

        assert restored.expr == ["C", "*", "X_0"]
        assert restored.error == 0.42
        assert isinstance(restored.parameters, np.ndarray)
        assert np.array_equal(restored.parameters, params)
        assert restored.augmentations["latex"]["value"] == "$C X_0$"
        assert restored.augmentations["latex"]["_type"] == "LaTeXAugmenter"

    def test_round_trip_status(self):
        original = ModelResult(expr=["X_0", "X_1"], error=np.inf, status="invalid", status_detail="bad")
        restored = ModelResult.from_dict(json.loads(json.dumps(original.to_dict())))
        assert (restored.status, restored.status_detail, restored.error) == ("invalid", "bad", np.inf)

    def test_missing_status_raises(self):
        data = ModelResult(expr=["X_0"], error=0.1).to_dict()
        del data["status"]
        with pytest.raises(ValueError, match="without an evaluation status"):
            ModelResult.from_dict(data)


def _make_eval_result(**kwargs):
    model = ModelResult(expr=["X_0"], error=0.05)
    defaults = dict(
        all_models=[model],
        trace=[0, 0],
        approach_name="TestApproach",
        success_threshold=0.1,
    )
    defaults.update(kwargs)
    return EvalResult(**defaults)


def _models(*errors):
    return [ModelResult(expr=[f"X_{i}"], error=e) for i, e in enumerate(errors)]


class TestEvalResultConstruction:
    def test_default_fields(self):
        result = _make_eval_result()
        assert result.dataset_name is None
        assert result.metadata is None
        assert result.top_k == 20
        # Each instance must get its own augmentations dict
        result2 = _make_eval_result()
        result.augmentations["key"] = {}
        assert "key" not in result2.augmentations


class TestEvalResultDerivedFields:
    def test_counts(self):
        result = _make_eval_result(all_models=_models(0.3, 0.2), trace=[0, 1, 0, 0])
        assert result.num_evaluated == 2
        assert result.evaluation_calls == 4

    def test_best_and_success(self):
        result = _make_eval_result(all_models=_models(0.3, 0.05, 0.2), trace=[0, 1, 2])
        assert result.min_error == 0.05
        assert result.best_expr == "X_1"
        assert result.success is True

    def test_success_false_above_threshold_or_without_threshold(self):
        assert _make_eval_result(all_models=_models(0.5), trace=[0]).success is False
        assert _make_eval_result(success_threshold=None).success is False

    def test_top_models_sorted_stable_nan_last(self):
        result = _make_eval_result(all_models=_models(np.nan, 0.2, 0.1, 0.2), trace=[0, 1, 2, 3])
        assert [m.expr for m in result.top_models()] == [["X_2"], ["X_1"], ["X_3"], ["X_0"]]

    def test_ties_broken_by_length_then_first_seen(self):
        models = [
            ModelResult(expr=["X_0", "+", "X_1"], error=0.1),
            ModelResult(expr=["X_1", "+", "X_0"], error=0.1),
            ModelResult(expr=["X_0"], error=0.1),
            ModelResult(expr=["C"], error=0.05),
        ]
        result = _make_eval_result(all_models=models, trace=[0, 1, 2, 3])
        assert [m.expr for m in result.top_models()] == [
            ["C"],
            ["X_0"],
            ["X_0", "+", "X_1"],
            ["X_1", "+", "X_0"],
        ]

    def test_length_does_not_override_error(self):
        models = [ModelResult(expr=["X_0"], error=0.2), ModelResult(expr=["X_0", "+", "C"], error=0.1)]
        assert _make_eval_result(all_models=models, trace=[0, 1]).best_expr == "X_0+C"

    def test_top_models_respects_top_k(self):
        models = _models(0.3, 0.1, 0.2)
        assert [m.expr for m in _make_eval_result(all_models=models, trace=[0], top_k=2).top_models()] == [
            ["X_1"],
            ["X_2"],
        ]
        assert len(_make_eval_result(all_models=models, trace=[0], top_k=-1).top_models()) == 3
        assert len(_make_eval_result(all_models=models, trace=[0], top_k=10).top_models()) == 3

    def test_top_models_k_overrides_top_k(self):
        result = _make_eval_result(all_models=_models(0.3, 0.1, 0.2), trace=[0], top_k=1)
        assert [m.expr for m in result.top_models(2)] == [["X_1"], ["X_2"]]
        assert len(result.top_models(k=-1)) == 3
        assert result.top_models(0) == []
        assert len(result.top_models()) == 1

    def test_top_models_are_shared_objects(self):
        result = _make_eval_result(all_models=_models(0.3, 0.1), trace=[0, 1])
        assert result.top_models()[0] is result.all_models[1]

    def test_top_models_follow_appended_models(self):
        result = _make_eval_result(all_models=_models(0.3), trace=[0])
        _ = result.top_models()
        result.all_models.append(ModelResult(expr=["C"], error=0.01))
        assert result.best_expr == "C"

    def test_all_nan(self):
        result = _make_eval_result(all_models=_models(np.nan, np.nan), trace=[0, 1])
        assert np.isnan(result.min_error)
        assert result.best_expr == "X_0"
        assert result.success is False

    def test_empty(self):
        result = _make_eval_result(all_models=[], trace=[])
        assert np.isnan(result.min_error)
        assert result.best_expr == ""
        assert result.success is False
        assert result.top_models() == []
        assert result.evaluation_calls == 0


class TestEvalResultTrace:
    def test_best_so_far_strict_improvements_skip_nan(self):
        result = _make_eval_result(all_models=_models(np.nan, 0.5, 0.5, 0.1), trace=[0, 1, 2, 1, 3, 0])
        assert result.best_so_far() == [(2, 1), (5, 3)]

    def test_best_so_far_skips_inf(self):
        result = _make_eval_result(all_models=_models(np.inf, 1.0), trace=[0, 1])
        assert result.best_so_far() == [(2, 1)]

    def test_best_so_far_empty(self):
        assert _make_eval_result(all_models=[], trace=[]).best_so_far() == []

    def test_calls_to_threshold_uses_success_threshold(self):
        result = _make_eval_result(all_models=_models(0.5, 0.05), trace=[0, 0, 0, 1, 1])
        assert result.calls_to_threshold() == 4

    def test_calls_to_threshold_explicit_threshold(self):
        result = _make_eval_result(all_models=_models(0.5, 0.05), trace=[0, 0, 0, 1, 1])
        assert result.calls_to_threshold(threshold=1.0) == 1
        assert result.calls_to_threshold(threshold=0.01) is None

    def test_calls_to_threshold_without_threshold_raises(self):
        result = _make_eval_result(success_threshold=None)
        with pytest.raises(ValueError):
            result.calls_to_threshold()

    def test_best_error_within_budgets(self):
        result = _make_eval_result(all_models=_models(np.nan, 0.5, 0.1), trace=[0, 1, 0, 2])
        curve = result.best_error_within([0, 1, 2, 3, 4, 100])
        np.testing.assert_array_equal(curve, [np.nan, np.nan, 0.5, 0.5, 0.1, 0.1])

    def test_best_error_within_empty_trace(self):
        curve = _make_eval_result(all_models=[], trace=[]).best_error_within([1, 5])
        assert np.all(np.isnan(curve))


def _status_models(*statuses):
    return [
        ModelResult(expr=[f"X_{i}"], error=0.1 * (i + 1) if st == "ok" else np.inf, status=st)
        for i, st in enumerate(statuses)
    ]


class TestEvalResultStatuses:
    def test_status_counts_unique_and_call(self):
        result = _make_eval_result(all_models=_status_models("ok", "invalid", "error"), trace=[0, 1, 1, 2])
        assert result.status_counts() == {"ok": 1, "invalid": 1, "rejected": 0, "undefined": 0, "error": 1}
        assert result.status_counts(per="call") == {"ok": 1, "invalid": 2, "rejected": 0, "undefined": 0, "error": 1}
        assert list(result.status_counts()) == list(STATUSES)

    def test_status_counts_bad_per_raises(self):
        with pytest.raises(ValueError, match="per must be"):
            _make_eval_result().status_counts(per="model")  # type: ignore[arg-type]

    # Tests for EvalResult.validity_rate, which is commented out in SRToolkit/utils/types.py.
    #
    # def test_validity_rate_counts_rejected_and_undefined_as_valid(self):
    #     result = _make_eval_result(
    #         all_models=_status_models("ok", "rejected", "undefined", "invalid"), trace=[0, 1, 2, 3]
    #     )
    #     assert result.validity_rate() == 0.75
    #
    # def test_validity_rate_excludes_error(self):
    #     result = _make_eval_result(all_models=_status_models("ok", "invalid", "error", "error"), trace=[0, 1, 2, 3])
    #     assert result.validity_rate() == 0.5
    #
    # def test_validity_rate_per_call(self):
    #     result = _make_eval_result(all_models=_status_models("ok", "invalid"), trace=[0, 1, 1, 1])
    #     assert result.validity_rate(per="call") == 0.25
    #
    # def test_validity_rate_nan_without_countable_models(self):
    #     assert np.isnan(_make_eval_result(all_models=[], trace=[]).validity_rate())
    #     assert np.isnan(_make_eval_result(all_models=_status_models("error"), trace=[0]).validity_rate())

    def test_failed_models_rank_last(self):
        models = _status_models("invalid", "ok", "undefined", "ok")
        result = _make_eval_result(all_models=models, trace=[0, 1, 2, 3])
        assert [m.status for m in result.top_models(k=-1)] == ["ok", "ok", "invalid", "undefined"]
        assert result.best_expr == "X_1"

    def test_inf_and_nan_rank_after_finite(self):
        result = _make_eval_result(all_models=_models(np.nan, np.inf, 0.3), trace=[0, 1, 2])
        assert result.top_models(k=-1)[0].error == 0.3

    def test_min_error_inf_when_all_failed(self):
        result = _make_eval_result(all_models=_status_models("invalid", "rejected"), trace=[0, 1])
        assert result.min_error == np.inf
        assert result.success is False

    def test_best_error_within_ignores_inf(self):
        result = _make_eval_result(all_models=_models(np.inf, 0.5), trace=[0, 1])
        assert np.isnan(result.best_error_within([1])[0])
        assert result.best_error_within([2])[0] == 0.5

    def test_to_dict_contains_status_counts(self):
        result = _make_eval_result(all_models=_status_models("ok", "invalid"), trace=[0, 1])
        d = result.to_dict()
        assert d["status_counts"]["invalid"] == 1
        assert EvalResult.from_dict(json.loads(json.dumps(d))).all_models[1].status == "invalid"


class TestEvalResultAddAugmentation:
    def test_stores_data_and_injects_type(self):
        result = _make_eval_result()
        result.add_augmentation("summary", {"value": 42}, "SummaryAugmenter")
        assert "summary" in result.augmentations
        assert result.augmentations["summary"]["value"] == 42
        assert result.augmentations["summary"]["_type"] == "SummaryAugmenter"

    def test_duplicate_key_gets_suffix(self):
        result = _make_eval_result()
        result.add_augmentation("summary", {"value": 42}, "SummaryAugmenter")
        result.add_augmentation("summary", {"value": 99}, "SummaryAugmenter")
        assert "summary" in result.augmentations
        assert "summary_1" in result.augmentations
        assert result.augmentations["summary_1"]["value"] == 99


class TestEvalResultToDict:
    def test_basic_serialization(self):
        result = _make_eval_result()
        d = result.to_dict()
        assert d["min_error"] == 0.05
        assert d["best_expr"] == "X_0"
        assert d["approach_name"] == "TestApproach"
        assert d["success"] is True
        assert d["num_evaluated"] == 1
        assert d["evaluation_calls"] == 2
        assert d["trace"] == [0, 0]
        assert d["success_threshold"] == 0.1
        assert d["top_k"] == 20
        assert len(d["all_models"]) == 1
        assert "top_models" not in d
        assert d["dataset_name"] is None
        assert d["metadata"] is None

    def test_numpy_scalar_casts(self):
        result = _make_eval_result(
            all_models=[ModelResult(expr=["X_0"], error=np.float64(0.05))],
            trace=[np.int64(0)],
            success_threshold=np.float64(0.1),
            top_k=np.int64(5),
        )
        d = result.to_dict()
        assert isinstance(d["min_error"], float)
        assert isinstance(d["num_evaluated"], int)
        assert isinstance(d["evaluation_calls"], int)
        assert isinstance(d["success"], bool)
        assert isinstance(d["success_threshold"], float)
        assert isinstance(d["top_k"], int)
        assert all(type(i) is int for i in d["trace"])
        json.dumps(d)


class TestEvalResultFromDict:
    def test_round_trip_with_optional_fields(self):
        result = _make_eval_result(
            dataset_name="Feynman_I_1",
            metadata={"source": "feynman"},
            top_k=3,
        )
        result.add_augmentation("summary", {"value": 42}, "SummaryAugmenter")

        restored = EvalResult.from_dict(result.to_dict())

        assert restored.min_error == 0.05
        assert restored.best_expr == "X_0"
        assert restored.trace == [0, 0]
        assert restored.success_threshold == 0.1
        assert restored.top_k == 3
        assert restored.dataset_name == "Feynman_I_1"
        assert restored.metadata == {"source": "feynman"}
        assert restored.augmentations["summary"]["value"] == 42
        assert len(restored.top_models()) == 1
        assert isinstance(restored.top_models()[0], ModelResult)

    def test_round_trip_missing_optional_fields(self):
        d = _make_eval_result().to_dict()
        del d["dataset_name"]
        del d["metadata"]

        restored = EvalResult.from_dict(d)

        assert restored.dataset_name is None
        assert restored.metadata is None

    def test_writes_format_version(self):
        assert _make_eval_result().to_dict()["format_version"] == 2

    def test_old_format_without_version_raises_clear_error(self):
        d = _make_eval_result().to_dict()
        del d["format_version"]
        with pytest.raises(ValueError, match="older SRToolkit version"):
            EvalResult.from_dict(d)

    def test_unknown_format_version_raises(self):
        d = _make_eval_result().to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError, match="Unsupported format_version: 99"):
            EvalResult.from_dict(d)

    def test_round_trip_preserves_nested_model_results(self):
        params = np.array([2.0])
        model = ModelResult(expr=["C", "*", "X_0"], error=0.01, parameters=params)
        result = _make_eval_result(all_models=[model], trace=[0])

        restored = EvalResult.from_dict(result.to_dict())

        assert restored.top_models()[0].expr == ["C", "*", "X_0"]
        assert np.array_equal(restored.top_models()[0].parameters, params)

    def test_round_trip_keeps_model_augmentations_on_top_models(self):
        result = _make_eval_result(all_models=_models(0.3, 0.1), trace=[0, 1])
        result.top_models()[0].add_augmentation("latex", {"expr_latex": "X_1"}, "LaTeXAugmenter")

        restored = EvalResult.from_dict(result.to_dict())

        assert restored.top_models()[0].augmentations["latex"]["expr_latex"] == "X_1"
        assert restored.top_models()[0] is restored.all_models[1]
