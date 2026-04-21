import numpy as np

from SRToolkit.utils.types import (
    CONST,
    FN,
    LIT,
    OP,
    VALID_SYMBOL_TYPES,
    VAR,
    EstimationSettings,
    EvalResult,
    ModelResult,
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


class TestEstimationSettings:
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
        "bed_X",
        "num_consts_sampled",
        "domain_bounds",
    }

    def test_all_fields_in_annotations(self):
        assert self.EXPECTED_FIELDS == set(EstimationSettings.__annotations__.keys())

    def test_all_fields_optional(self):
        assert EstimationSettings.__required_keys__ == frozenset()
        assert self.EXPECTED_FIELDS == set(EstimationSettings.__optional_keys__)

    def test_dict_interface(self):
        settings: EstimationSettings = {"method": "L-BFGS-B", "max_iter": 200}
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


def _make_eval_result(**kwargs):
    model = ModelResult(expr=["X_0"], error=0.05)
    defaults = dict(
        min_error=0.05,
        best_expr="X_0",
        num_evaluated=100,
        evaluation_calls=120,
        top_models=[model],
        all_models=[model],
        approach_name="TestApproach",
        success=True,
    )
    defaults.update(kwargs)
    return EvalResult(**defaults)


class TestEvalResultConstruction:
    def test_default_fields(self):
        result = _make_eval_result()
        assert result.dataset_name is None
        assert result.metadata is None
        # Each instance must get its own augmentations dict
        result2 = _make_eval_result()
        result.augmentations["key"] = {}
        assert "key" not in result2.augmentations


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
        model = ModelResult(expr=["X_0"], error=0.05)
        result = _make_eval_result(top_models=[model], all_models=[model])
        d = result.to_dict()
        assert d["min_error"] == 0.05
        assert d["best_expr"] == "X_0"
        assert d["approach_name"] == "TestApproach"
        assert d["success"] is True
        assert len(d["top_models"]) == 1
        assert len(d["all_models"]) == 1
        assert d["dataset_name"] is None
        assert d["metadata"] is None

    def test_numpy_scalar_casts(self):
        result = _make_eval_result(
            min_error=np.float64(0.05),
            num_evaluated=np.int64(100),
            evaluation_calls=np.int64(120),
            success=np.bool_(True),
        )
        d = result.to_dict()
        assert isinstance(d["min_error"], float)
        assert isinstance(d["num_evaluated"], int)
        assert isinstance(d["evaluation_calls"], int)
        assert isinstance(d["success"], bool)


class TestEvalResultFromDict:
    def test_round_trip_with_optional_fields(self):
        result = _make_eval_result(
            dataset_name="Feynman_I_1",
            metadata={"source": "feynman"},
        )
        result.add_augmentation("summary", {"value": 42}, "SummaryAugmenter")

        restored = EvalResult.from_dict(result.to_dict())

        assert restored.min_error == 0.05
        assert restored.best_expr == "X_0"
        assert restored.dataset_name == "Feynman_I_1"
        assert restored.metadata == {"source": "feynman"}
        assert restored.augmentations["summary"]["value"] == 42
        assert len(restored.top_models) == 1
        assert isinstance(restored.top_models[0], ModelResult)

    def test_round_trip_missing_optional_fields(self):
        d = _make_eval_result().to_dict()
        del d["dataset_name"]
        del d["metadata"]

        restored = EvalResult.from_dict(d)

        assert restored.dataset_name is None
        assert restored.metadata is None

    def test_round_trip_preserves_nested_model_results(self):
        params = np.array([2.0])
        model = ModelResult(expr=["C", "*", "X_0"], error=0.01, parameters=params)
        result = _make_eval_result(top_models=[model], all_models=[model])

        restored = EvalResult.from_dict(result.to_dict())

        assert restored.top_models[0].expr == ["C", "*", "X_0"]
        assert np.array_equal(restored.top_models[0].parameters, params)
