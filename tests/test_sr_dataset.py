"""Tests for SRToolkit.dataset.sr_dataset."""

import tempfile

import numpy as np
import pytest

from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.utils.symbol_library import SymbolLibrary


@pytest.fixture
def simple_dataset():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    return SR_dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=np.array([3.0, 7.0, 11.0]),
        max_evaluations=10000,
        original_equation="z = x + y",
        success_threshold=1e-6,
    )


class TestSRDatasetInit:
    def test_basic_init(self, simple_dataset):
        assert simple_dataset.X.shape == (3, 2)
        assert simple_dataset.ranking_function == "rmse"
        assert simple_dataset.success_threshold == 1e-6

    def test_contains_constants(self, simple_dataset):
        assert simple_dataset.contains_constants is True

    def test_str(self, simple_dataset):
        s = str(simple_dataset)
        assert "z = x + y" in s
        assert "RMSE" in s


class TestSRDatasetEvaluator:
    def test_create_evaluator(self, simple_dataset):
        evaluator = simple_dataset.create_evaluator()
        error = evaluator.evaluate_expr(["X_0", "+", "X_1"])
        assert error < 1e-6

    def test_evaluator_inherits_settings(self, simple_dataset):
        evaluator = simple_dataset.create_evaluator()
        assert evaluator.max_evaluations == 10000
        assert evaluator.success_threshold == 1e-6

    def test_evaluator_metadata(self, simple_dataset):
        evaluator = simple_dataset.create_evaluator(metadata={"key": "val"})
        assert evaluator.metadata["key"] == "val"


class TestSRDatasetSerialization:
    def test_to_dict_round_trip(self, simple_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = simple_dataset.to_dict(tmpdir)
            restored = SR_dataset.from_dict(d)

            np.testing.assert_array_equal(restored.X, simple_dataset.X)
            np.testing.assert_array_equal(restored.y, simple_dataset.y)
            assert restored.ranking_function == simple_dataset.ranking_function
            assert restored.success_threshold == simple_dataset.success_threshold
            assert restored.original_equation == simple_dataset.original_equation
            assert restored.max_evaluations == simple_dataset.max_evaluations

    def test_to_dict_format_version(self, simple_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = simple_dataset.to_dict(tmpdir)
            assert d["format_version"] == 1

    def test_from_dict_bad_format_version(self, simple_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = simple_dataset.to_dict(tmpdir)
            d["format_version"] = 99
            with pytest.raises(ValueError, match="Unsupported format_version"):
                SR_dataset.from_dict(d)

    def test_from_dict_missing_data(self):
        with pytest.raises(Exception):
            SR_dataset.from_dict({"format_version": 1, "dataset_path": "/tmp/nonexistent.npz"})
