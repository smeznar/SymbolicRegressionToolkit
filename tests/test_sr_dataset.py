"""Tests for SRToolkit.dataset.sr_dataset."""

import json

import numpy as np
import pytest

from SRToolkit.dataset import data_cache
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.data_split import DataSplit
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.measures import create_behavior_matrix
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

    def test_str_bed_describes_bed(self):
        dataset = SR_dataset(
            np.zeros((3, 2)),
            SymbolLibrary.default_symbols(2),
            ranking_function="bed",
            ground_truth=["X_0", "+", "C"],
            success_threshold=0.5,
        )
        s = str(dataset)
        assert "BED ranking function" in s
        assert "behavioral expression distance is less than 0.5" in s
        assert "root mean squared error" not in s

    def test_new_fields_default_none(self, simple_dataset):
        assert simple_dataset.benchmark is None
        assert simple_dataset.version is None
        assert simple_dataset.data_source is None


def _bed_dataset(**kwargs):
    """BED dataset whose samplers draw from |x| in [10, 11], far from X in [0, 1)."""
    return SR_dataset(
        np.random.default_rng(0).random((20, 2)),
        SymbolLibrary.default_symbols(2),
        ranking_function="bed",
        ground_truth=["X_0", "+", "C"],
        success_threshold=1.0,
        samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
        num_consts_sampled=4,
        **kwargs,
    )


class TestSRDatasetEvaluator:
    def test_bed_draws_points_from_dataset_samplers(self):
        bed_X = _bed_dataset(num_points_sampled=8).create_evaluator(seed=0).ranking.bed_X
        assert bed_X.shape == (8, 2)
        assert np.all(np.abs(bed_X) >= 10)

    def test_bed_precomputed_bed_gt_scores_on_X_not_samplers(self):
        X = np.random.default_rng(0).random((20, 2))
        gt = create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=4, seed=0)
        dataset = _bed_dataset(bed_gt=gt)
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, dataset.X)

    def test_bed_fixed_per_dataset_across_run_seeds(self):
        dataset = _bed_dataset(num_points_sampled=8, seed=4)
        a, b = dataset.create_evaluator(seed=1), dataset.create_evaluator(seed=2)
        np.testing.assert_array_equal(a.ranking.bed_X, b.ranking.bed_X)
        assert a.evaluate_expr(["X_0", "*", "C"]) == b.evaluate_expr(["X_0", "*", "C"])

    def test_bed_fixed_without_dataset_seed_falls_back_to_zero(self):
        assert _bed_dataset().create_evaluator(seed=1).ranking.seed == 0

    def test_bed_resample_uses_run_seed(self):
        dataset = _bed_dataset(num_points_sampled=8, bed_resample=True)
        a, b = dataset.create_evaluator(seed=1), dataset.create_evaluator(seed=2)
        assert not np.array_equal(a.ranking.bed_X, b.ranking.bed_X)

    def test_bed_without_ground_truth_uses_y_not_samplers(self):
        X = np.random.default_rng(0).random((20, 2))
        dataset = SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            ranking_function="bed",
            y=X[:, 0],
            success_threshold=1.0,
            samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
            num_points_sampled=-1,
        )
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, X)

    def test_bed_num_points_minus_one_uses_all_of_X(self):
        dataset = _bed_dataset(num_points_sampled=-1)
        np.testing.assert_array_equal(dataset.create_evaluator(seed=0).ranking.bed_X, dataset.X)

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
        assert d["format_version"] == 3

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

    def test_resample_free_constant_raises(self):
        # A free constant ("C") has no value to evaluate against during generation.
        # This must raise clearly rather than pass an empty constant array (which the
        # compiled Cython backend would read out of bounds, silently producing garbage).
        with pytest.raises(ValueError, match="free constant"):
            SR_dataset.from_samplers(
                ground_truth=["X_0", "*", "C"],
                samplers=[UniformSampling(0.5, 5.0)],
                n_samples=16,
                seed=1,
            )


# ── Data splits (train / test / aux) ──────────────────────────────────────────


class TestDataSplits:
    def _dataset_with_extras(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            ground_truth=["X_0", "+", "X_1"],
            y=np.array([3.0, 7.0, 11.0]),
            test_X=np.array([[7.0, 8.0], [9.0, 10.0]]),
            test_y=np.array([15.0, 19.0]),
            aux_data={"grid": {"X": np.array([[0.0, 0.0]])}},
            dataset_name="ds",
            benchmark="bench",
            version="1.0.0",
        )

    def test_x_setter_updates_the_data(self, simple_dataset):
        simple_dataset.X = np.zeros((3, 2))
        assert simple_dataset.X.sum() == 0.0

    def test_y_setter_none_drops_target(self, simple_dataset):
        simple_dataset.y = None
        assert simple_dataset.y is None

    def test_test_and_aux_accessors(self):
        ds = self._dataset_with_extras()
        assert isinstance(ds.test, DataSplit)
        assert ds.test_X is ds.test.X
        assert ds.test_y.shape == (2,)
        assert set(ds.aux_data) == {"grid"}
        assert isinstance(ds.aux_data["grid"], DataSplit)
        assert ds.aux_data["grid"].X.shape == (1, 2)

    def test_no_test_split_returns_none(self, simple_dataset):
        assert simple_dataset.test is None
        assert simple_dataset.test_X is None
        assert simple_dataset.test_y is None
        assert simple_dataset.aux_data == {}

    def test_ndarray_ground_truth_rejected(self):
        with pytest.raises(ValueError, match="ndarray ground_truth"):
            SR_dataset(np.zeros((2, 1)), SymbolLibrary.default_symbols(1), ground_truth=np.zeros((4, 4)))

    @pytest.mark.parametrize("bad_key", ["", ".", "..", "a/b", "a\\b", "a:b", "a?b"])
    def test_invalid_aux_key_rejected(self, bad_key):
        with pytest.raises(ValueError, match="aux keys become file names"):
            SR_dataset(
                np.zeros((2, 1)),
                SymbolLibrary.default_symbols(1),
                y=np.zeros(2),
                aux_data={bad_key: {"X": np.zeros((1, 1))}},
            )

    @pytest.mark.parametrize("key", ["train", "test", "a__b", "grid 2"])
    def test_aux_keys_only_need_to_be_valid_file_names(self, key):
        ds = SR_dataset(
            np.zeros((2, 1)), SymbolLibrary.default_symbols(1), y=np.zeros(2), aux_data={key: {"X": np.ones((1, 1))}}
        )
        assert ds.aux_data[key].X.sum() == 1.0

    def test_splits_cached_in_dataset_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        self._dataset_with_extras().to_dict()
        for key in ("ds", "ds/test", "ds/aux/grid"):
            assert data_cache.dataset_path("bench", "1.0.0", key).exists()

    def test_dataset_names_with_separators_do_not_collide(self, tmp_path, monkeypatch):
        # A dataset named "ds__test" used to share its file with the test data of "ds".
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        other = SR_dataset(
            np.full((2, 2), 5.0),
            SymbolLibrary.default_symbols(2),
            y=np.zeros(2),
            dataset_name="ds__test",
            benchmark="bench",
            version="1.0.0",
        )
        ds.to_dict()
        other.to_dict()
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test_X, ds.test_X)

    def test_aux_value_can_be_a_data_split(self):
        split = DataSplit({"X": np.ones((2, 2))})
        ds = SR_dataset(np.zeros((2, 2)), SymbolLibrary.default_symbols(2), y=np.zeros(2), aux_data={"grid": split})
        assert ds.aux_data["grid"] is split

    def test_aux_split_with_source_regenerated_not_stored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        far = DataSplit({"X": np.zeros((5, 2))}, data_source=SampleSource(n_samples=5, seed=3))
        ds = SR_dataset(
            np.zeros((3, 2)),
            SymbolLibrary.default_symbols(2),
            y=np.zeros(3),
            ground_truth=["X_0", "+", "X_1"],
            samplers=[UniformSampling(10, 11), UniformSampling(10, 11)],
            aux_data={"far": far},
            dataset_name="ds",
            benchmark="bench",
            version="1.0.0",
        )
        d = ds.to_dict()
        assert not data_cache.dataset_path("bench", "1.0.0", "ds/aux/far").exists()
        restored = SR_dataset.from_dict(d)
        assert restored.aux_data["far"].X.shape == (5, 2)
        assert np.all(np.abs(restored.aux_data["far"].X) >= 10)
        assert restored.aux_data["far"].data_source is not None

    def test_aux_entry_added_later_is_saved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.aux_data["weights"] = DataSplit({"w": np.arange(3.0)})
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.aux_data["weights"].arrays["w"], np.arange(3.0))

    def test_invalid_aux_key_added_later_rejected_on_save(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.aux_data["a/b"] = DataSplit({"X": np.zeros((1, 2))})
        with pytest.raises(ValueError, match="aux keys become file names"):
            ds.to_dict()

    def test_extra_test_arrays_survive_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        ds.test.arrays["weights"] = np.array([1.0, 2.0])
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test.arrays["weights"], np.array([1.0, 2.0]))

    def test_v2_config_loads_without_splits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        d = self._dataset_with_extras().to_dict()
        d["format_version"] = 2
        del d["test"], d["aux"]
        restored = SR_dataset.from_dict(d)
        assert restored.X.shape == (3, 2)
        assert restored.test is None and restored.aux_data == {}

    def test_refresh_regenerates_test_split(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 5), UniformSampling(0, 5)],
            n_samples=10,
            n_test=4,
            seed=0,
            benchmark="bench",
            version="1.0.0",
            dataset_name="gen",
        )
        expected = ds.test_X.copy()
        ds.test.arrays["X"] = np.zeros((4, 2))
        ds.refresh()
        np.testing.assert_array_equal(ds.test_X, expected)

    def test_cache_remove_deletes_split_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        self._dataset_with_extras().to_dict()
        assert [e["key"] for e in data_cache.list()] == ["ds"]
        data_cache.remove("bench", "1.0.0", "ds")
        assert not data_cache.dataset_path("bench", "1.0.0", "ds/test").exists()
        assert not (data_cache.dataset_path("bench", "1.0.0", "ds").parent / "ds").exists()

    def test_from_samplers_n_test_creates_test_split(self):
        ds = SR_dataset.from_samplers(
            ["X_0", "+", "X_1"],
            [UniformSampling(0, 5), UniformSampling(0, 5)],
            n_samples=40,
            n_test=12,
            seed=0,
        )
        assert ds.X.shape == (40, 2)
        assert ds.test_X.shape == (12, 2)
        assert ds.test_y.shape == (12,)
        # The test split carries its own source, so it is regenerated (not persisted).
        assert ds.test.data_source is not None

    def test_to_dict_from_dict_round_trips_test_and_aux(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        restored = SR_dataset.from_dict(ds.to_dict())
        np.testing.assert_array_equal(restored.test_X, ds.test_X)
        np.testing.assert_array_equal(restored.test_y, ds.test_y)
        np.testing.assert_array_equal(restored.aux_data["grid"].X, ds.aux_data["grid"].X)

    def test_constant_bounds_survive_json_round_trip(self, tmp_path, monkeypatch):
        # JSON turns the tuple into a list; expressions with constants must still be fitted.
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        X = np.random.default_rng(0).random((10, 2))
        ds = SR_dataset(
            X,
            SymbolLibrary.default_symbols(2),
            y=2 * X[:, 0] + X[:, 1],
            benchmark="b",
            version="1",
            dataset_name="cb",
            constant_bounds=(-3, 3),
        )
        restored = SR_dataset.from_dict(json.loads(json.dumps(ds.to_dict())))
        assert restored.create_evaluator().evaluate_expr(["C", "*", "X_0", "+", "X_1"]) < 1e-6

    def test_node_ground_truth_serialised_with_dataset_library(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        lib = SymbolLibrary.from_symbol_list(["+", "*", "sin", "C"], num_variables=2)
        gt = tokens_to_tree(["sin", "(", "X_0", ")", "*", "X_1"], lib)
        ds = SR_dataset(np.zeros((3, 2)), lib, y=np.zeros(3), ground_truth=gt, benchmark="b", version="1")
        assert ds.to_dict()["ground_truth"] == gt.to_list(symbol_library=lib)

    def test_to_archive_from_archive_round_trips_test(self, tmp_path, monkeypatch):
        monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path)
        ds = self._dataset_with_extras()
        archive_path = tmp_path / "ds.zip"
        ds.to_archive(archive_path)
        restored = SR_dataset.from_archive(archive_path)
        np.testing.assert_array_equal(restored.test_X, ds.test_X)
        np.testing.assert_array_equal(restored.aux_data["grid"].X, ds.aux_data["grid"].X)
