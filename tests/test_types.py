"""Tests for SRToolkit.utils.types — EstimationSettings, ModelResult, EvalResult."""

import numpy as np
import pytest

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(expr=None, error=0.1, parameters=None, augmentations=None):
    kwargs = dict(expr=expr or ["X_0"], error=error)
    if parameters is not None:
        kwargs["parameters"] = parameters
    if augmentations is not None:
        kwargs["augmentations"] = augmentations
    return ModelResult(**kwargs)


def _make_eval(model=None, **overrides):
    m = model or _make_model()
    defaults = dict(
        min_error=0.05,
        best_expr="X_0",
        num_evaluated=10,
        evaluation_calls=12,
        top_models=[m],
        all_models=[m],
        approach_name="TestApproach",
        success=True,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestSymbolTypeConstants:
    def test_constant_values(self):
        assert VAR == "var"
        assert CONST == "const"
        assert FN == "fn"
        assert OP == "op"
        assert LIT == "lit"

    def test_valid_symbol_types_set(self):
        assert VALID_SYMBOL_TYPES == {"var", "const", "fn", "op", "lit"}

    def test_valid_symbol_types_is_set(self):
        assert isinstance(VALID_SYMBOL_TYPES, set)


# ---------------------------------------------------------------------------
# EstimationSettings
# ---------------------------------------------------------------------------


class TestEstimationSettings:
    """TypedDict — verify accepted keys and dict protocol."""

    def test_empty_dict_is_valid(self):
        s: EstimationSettings = {}
        assert s == {}

    def test_single_key(self):
        s: EstimationSettings = {"method": "L-BFGS-B"}
        assert s["method"] == "L-BFGS-B"

    def test_all_keys_accepted(self):
        s: EstimationSettings = {
            "method": "Nelder-Mead",
            "tol": 1e-8,
            "gtol": 1e-4,
            "max_iter": 500,
            "constant_bounds": (-10.0, 10.0),
            "initialization": "mean",
            "max_constants": 4,
            "max_expr_length": 20,
            "num_points_sampled": 128,
            "bed_X": np.zeros((5, 2)),
            "num_consts_sampled": 64,
            "domain_bounds": [(-1.0, 1.0), (0.0, 2.0)],
        }
        assert s["method"] == "Nelder-Mead"
        assert s["max_iter"] == 500

    def test_get_with_default(self):
        s: EstimationSettings = {}
        assert s.get("tol", 1e-6) == 1e-6

    def test_bed_X_none(self):
        s: EstimationSettings = {"bed_X": None}
        assert s["bed_X"] is None

    def test_domain_bounds_none(self):
        s: EstimationSettings = {"domain_bounds": None}
        assert s["domain_bounds"] is None


# ---------------------------------------------------------------------------
# ModelResult — construction
# ---------------------------------------------------------------------------


class TestModelResultConstruction:
    def test_minimal_construction(self):
        r = ModelResult(expr=["X_0"], error=0.5)
        assert r.expr == ["X_0"]
        assert r.error == 0.5
        assert r.parameters is None
        assert r.augmentations == {}

    def test_with_parameters(self):
        params = np.array([1.0, 2.0])
        r = ModelResult(expr=["C", "+", "X_0"], error=0.0, parameters=params)
        np.testing.assert_array_equal(r.parameters, params)

    def test_augmentations_default_independent(self):
        r1 = ModelResult(expr=["X_0"], error=0.1)
        r2 = ModelResult(expr=["X_1"], error=0.2)
        r1.augmentations["key"] = "val"
        assert "key" not in r2.augmentations


# ---------------------------------------------------------------------------
# ModelResult.add_augmentation
# ---------------------------------------------------------------------------


class TestModelResultAddAugmentation:
    def test_first_augmentation_stored_under_original_name(self):
        r = _make_model()
        r.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
        assert "latex" in r.augmentations
        assert r.augmentations["latex"]["value"] == "$X_0$"

    def test_type_key_injected(self):
        r = _make_model()
        r.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
        assert r.augmentations["latex"]["_type"] == "LaTeXAugmenter"

    def test_duplicate_name_gets_suffix_1(self):
        r = _make_model()
        r.add_augmentation("a", {"v": 1}, "T")
        r.add_augmentation("a", {"v": 2}, "T")
        assert "a" in r.augmentations
        assert "a_1" in r.augmentations
        assert r.augmentations["a_1"]["v"] == 2

    def test_triple_duplicate_increments_counter(self):
        r = _make_model()
        r.add_augmentation("a", {"v": 1}, "T")
        r.add_augmentation("a", {"v": 2}, "T")
        r.add_augmentation("a", {"v": 3}, "T")
        assert "a_2" in r.augmentations
        assert r.augmentations["a_2"]["v"] == 3

    def test_different_names_do_not_collide(self):
        r = _make_model()
        r.add_augmentation("foo", {"x": 1}, "A")
        r.add_augmentation("bar", {"x": 2}, "B")
        assert len(r.augmentations) == 2


# ---------------------------------------------------------------------------
# ModelResult.to_dict / from_dict
# ---------------------------------------------------------------------------


class TestModelResultSerialization:
    def test_to_dict_basic_fields(self):
        r = _make_model(expr=["X_0", "+", "C"], error=0.25)
        d = r.to_dict()
        assert d["expr"] == ["X_0", "+", "C"]
        assert d["error"] == 0.25
        assert d["parameters"] is None
        assert d["augmentations"] == {}

    def test_to_dict_error_is_python_float(self):
        r = _make_model(error=np.float64(0.5))
        d = r.to_dict()
        assert isinstance(d["error"], float)

    def test_to_dict_parameters_serialized(self):
        params = np.array([1.5, 2.5])
        r = _make_model(parameters=params)
        d = r.to_dict()
        assert d["parameters"] == {"__ndarray__": True, "data": [1.5, 2.5]}

    def test_roundtrip_no_parameters(self):
        r = _make_model(expr=["X_0", "-", "X_1"], error=0.3)
        r2 = ModelResult.from_dict(r.to_dict())
        assert r2.expr == r.expr
        assert r2.error == r.error
        assert r2.parameters is None

    def test_roundtrip_with_parameters(self):
        params = np.array([3.14])
        r = _make_model(expr=["C"], error=0.0, parameters=params)
        r2 = ModelResult.from_dict(r.to_dict())
        np.testing.assert_array_almost_equal(r2.parameters, params)

    def test_roundtrip_with_augmentations(self):
        r = _make_model()
        r.add_augmentation("info", {"score": 42}, "ScoreAugmenter")
        r2 = ModelResult.from_dict(r.to_dict())
        assert r2.augmentations["info"]["score"] == 42
        assert r2.augmentations["info"]["_type"] == "ScoreAugmenter"

    def test_from_dict_reconstructs_correctly(self):
        data = {
            "expr": ["X_1"],
            "error": 0.99,
            "parameters": None,
            "augmentations": {},
        }
        r = ModelResult.from_dict(data)
        assert r.expr == ["X_1"]
        assert r.error == 0.99


# ---------------------------------------------------------------------------
# EvalResult — construction
# ---------------------------------------------------------------------------


class TestEvalResultConstruction:
    def test_minimal_required_fields(self):
        r = _make_eval()
        assert r.min_error == 0.05
        assert r.best_expr == "X_0"
        assert r.approach_name == "TestApproach"
        assert r.success is True
        assert r.dataset_name is None
        assert r.metadata is None
        assert r.augmentations == {}

    def test_with_dataset_name(self):
        r = _make_eval(dataset_name="Nguyen-1")
        assert r.dataset_name == "Nguyen-1"

    def test_with_metadata(self):
        r = _make_eval(metadata={"seed": 7})
        assert r.metadata["seed"] == 7

    def test_augmentations_default_independent(self):
        r1 = _make_eval()
        r2 = _make_eval()
        r1.augmentations["k"] = "v"
        assert "k" not in r2.augmentations


# ---------------------------------------------------------------------------
# EvalResult.add_augmentation
# ---------------------------------------------------------------------------


class TestEvalResultAddAugmentation:
    def test_first_augmentation_stored_under_original_name(self):
        r = _make_eval()
        r.add_augmentation("complexity", {"value": 3}, "ComplexityAugmenter")
        assert r.augmentations["complexity"]["value"] == 3

    def test_type_key_injected(self):
        r = _make_eval()
        r.add_augmentation("complexity", {"value": 3}, "ComplexityAugmenter")
        assert r.augmentations["complexity"]["_type"] == "ComplexityAugmenter"

    def test_duplicate_name_gets_suffix_1(self):
        r = _make_eval()
        r.add_augmentation("x", {"v": 1}, "T")
        r.add_augmentation("x", {"v": 2}, "T")
        assert "x_1" in r.augmentations
        assert r.augmentations["x_1"]["v"] == 2

    def test_triple_duplicate_increments_counter(self):
        r = _make_eval()
        r.add_augmentation("x", {"v": 1}, "T")
        r.add_augmentation("x", {"v": 2}, "T")
        r.add_augmentation("x", {"v": 3}, "T")
        assert "x_2" in r.augmentations

    def test_different_names_do_not_collide(self):
        r = _make_eval()
        r.add_augmentation("a", {"x": 1}, "A")
        r.add_augmentation("b", {"x": 2}, "B")
        assert len(r.augmentations) == 2


# ---------------------------------------------------------------------------
# EvalResult.to_dict / from_dict
# ---------------------------------------------------------------------------


class TestEvalResultSerialization:
    def test_to_dict_scalar_types(self):
        r = _make_eval()
        d = r.to_dict()
        assert isinstance(d["min_error"], float)
        assert isinstance(d["num_evaluated"], int)
        assert isinstance(d["evaluation_calls"], int)
        assert isinstance(d["success"], bool)

    def test_to_dict_nested_models(self):
        m = _make_model(expr=["X_0"], error=0.1)
        r = _make_eval(model=m)
        d = r.to_dict()
        assert len(d["top_models"]) == 1
        assert d["top_models"][0]["expr"] == ["X_0"]

    def test_to_dict_dataset_name_none(self):
        r = _make_eval()
        d = r.to_dict()
        assert d["dataset_name"] is None

    def test_to_dict_metadata_preserved(self):
        r = _make_eval(metadata={"info": "test"})
        d = r.to_dict()
        assert d["metadata"] == {"info": "test"}

    def test_roundtrip_basic(self):
        r = _make_eval()
        r2 = EvalResult.from_dict(r.to_dict())
        assert r2.min_error == r.min_error
        assert r2.best_expr == r.best_expr
        assert r2.approach_name == r.approach_name
        assert r2.success == r.success
        assert r2.dataset_name is None
        assert r2.metadata is None

    def test_roundtrip_with_dataset_name_and_metadata(self):
        r = _make_eval(dataset_name="Feynman-I.12.1", metadata={"seed": 3})
        r2 = EvalResult.from_dict(r.to_dict())
        assert r2.dataset_name == "Feynman-I.12.1"
        assert r2.metadata == {"seed": 3}

    def test_roundtrip_nested_models_preserved(self):
        m = _make_model(expr=["C", "*", "X_0"], error=0.01, parameters=np.array([2.0]))
        r = _make_eval(model=m)
        r2 = EvalResult.from_dict(r.to_dict())
        assert len(r2.top_models) == 1
        assert r2.top_models[0].expr == ["C", "*", "X_0"]
        np.testing.assert_array_almost_equal(r2.top_models[0].parameters, np.array([2.0]))

    def test_roundtrip_with_augmentations(self):
        r = _make_eval()
        r.add_augmentation("meta", {"tag": "unit"}, "MetaAugmenter")
        r2 = EvalResult.from_dict(r.to_dict())
        assert r2.augmentations["meta"]["tag"] == "unit"

    def test_from_dict_missing_optional_keys_use_defaults(self):
        """from_dict uses dict.get for optional keys, so omitting them yields None."""
        m = _make_model()
        data = {
            "min_error": 0.1,
            "best_expr": "X_0",
            "num_evaluated": 5,
            "evaluation_calls": 5,
            "top_models": [m.to_dict()],
            "all_models": [m.to_dict()],
            "approach_name": "Test",
            "success": False,
            "augmentations": {},
            # dataset_name and metadata intentionally omitted
        }
        r = EvalResult.from_dict(data)
        assert r.dataset_name is None
        assert r.metadata is None

    def test_to_dict_numpy_min_error_converted(self):
        r = _make_eval(min_error=np.float64(0.123))
        d = r.to_dict()
        assert isinstance(d["min_error"], float)
        assert d["min_error"] == pytest.approx(0.123)

    def test_to_dict_numpy_counts_converted(self):
        r = _make_eval(num_evaluated=np.int64(42), evaluation_calls=np.int64(50))
        d = r.to_dict()
        assert isinstance(d["num_evaluated"], int)
        assert isinstance(d["evaluation_calls"], int)
