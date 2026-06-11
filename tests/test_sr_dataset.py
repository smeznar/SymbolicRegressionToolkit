"""Tests for SRToolkit.dataset.sr_dataset."""

import numpy as np
import pytest

from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.sampling import UniformSampling
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


@pytest.fixture
def serialisable_dataset(tmp_path, monkeypatch):
    """Dataset with benchmark/version/data_source set, using a patched cache root."""
    monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)
    samplers = [UniformSampling(0, 5), UniformSampling(0, 5)]
    dataset = SR_dataset(
        X,
        SymbolLibrary.default_symbols(2),
        ground_truth=["X_0", "+", "X_1"],
        y=np.array([3.0, 7.0, 11.0], dtype=float),
        max_evaluations=10000,
        original_equation="z = x + y",
        success_threshold=1e-6,
        dataset_name="test_ds",
        samplers=samplers,
        benchmark="test_bench",
        version="1.0.0",
    )
    dataset.data_source = SampleSource(n_samples=3, seed=0)
    return dataset


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

    def test_new_fields_default_none(self, simple_dataset):
        assert simple_dataset.benchmark is None
        assert simple_dataset.version is None
        assert simple_dataset.data_source is None


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
    def test_to_dict_requires_benchmark(self, simple_dataset):
        with pytest.raises(ValueError, match="benchmark"):
            simple_dataset.to_dict()

    def test_to_dict_requires_version(self):
        X = np.array([[1.0, 2.0]])
        ds = SR_dataset(X, SymbolLibrary.default_symbols(2), benchmark="b", version=None)
        with pytest.raises(ValueError, match="version"):
            ds.to_dict()

    def test_to_dict_format_version(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        assert d["format_version"] == 2

    def test_to_dict_includes_benchmark_version(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        assert d["benchmark"] == "test_bench"
        assert d["version"] == "1.0.0"

    def test_to_dict_pure_no_filesystem_side_effects(self, serialisable_dataset, tmp_path):
        before = list(tmp_path.iterdir())
        serialisable_dataset.to_dict()
        after = list(tmp_path.iterdir())
        assert before == after  # no files written

    def test_from_dict_round_trip(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        restored = SR_dataset.from_dict(d)
        assert restored.X.shape == serialisable_dataset.X.shape
        assert restored.ranking_function == serialisable_dataset.ranking_function
        assert restored.success_threshold == serialisable_dataset.success_threshold
        assert restored.benchmark == "test_bench"
        assert restored.version == "1.0.0"

    def test_from_dict_bad_format_version_raises(self, serialisable_dataset):
        d = serialisable_dataset.to_dict()
        d["format_version"] = 99
        with pytest.raises(ValueError, match="Unsupported format_version"):
            SR_dataset.from_dict(d)

    def test_from_dict_v1_legacy_missing_data_raises(self):
        """v1 format with missing dataset_path raises an exception."""
        with pytest.raises(Exception):
            SR_dataset.from_dict(
                {
                    "format_version": 1,
                    "dataset_path": "/tmp/nonexistent_srtk_test.npz",
                }
            )


class TestSRDatasetRefresh:
    def test_refresh_null_source_raises(self, simple_dataset):
        simple_dataset.benchmark = "b"
        simple_dataset.version = "1.0.0"
        simple_dataset.data_source = None
        with pytest.raises(ValueError, match="data_source is null"):
            simple_dataset.refresh()

    def test_refresh_reruns_sampling(self, serialisable_dataset):
        serialisable_dataset.refresh()
        # After refresh the cache should be regenerated; X should be reloaded
        assert serialisable_dataset.X.shape[1] == 2


class TestSRDatasetResample:
    def test_resample_returns_correct_shape(self, serialisable_dataset):
        X_new, y_new = serialisable_dataset.resample(50)
        assert X_new.shape == (50, 2)
        assert y_new.shape == (50,)

    def test_resample_no_samplers_raises(self, simple_dataset):
        with pytest.raises(ValueError, match="samplers"):
            simple_dataset.resample(10)
