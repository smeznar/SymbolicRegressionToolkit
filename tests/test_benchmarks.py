"""
Benchmark tests that require downloaded data in data/feynman/ and data/nguyen/.

Run with:
    pytest tests/test_benchmarks.py

These tests are intentionally excluded from the default test run (testpaths = ["SRToolkit"])
because they depend on external data files.
"""

import pytest

from SRToolkit.dataset import SR_benchmark, SR_dataset

FEYNMAN_DIR = "data/feynman"
NGUYEN_DIR = "data/nguyen"


@pytest.mark.benchmark
def test_feynman_loads():
    benchmark = SR_benchmark.feynman(FEYNMAN_DIR)
    assert len(benchmark.list_datasets(verbose=False)) == 100


@pytest.mark.benchmark
def test_feynman_add_dataset_instance():
    benchmark = SR_benchmark.feynman(FEYNMAN_DIR)
    dataset = benchmark.create_dataset("I.16.6")
    assert isinstance(dataset, SR_dataset)

    bm = SR_benchmark("BM", "data/bm")
    bm.add_dataset_instance("I.16.6", dataset)
    assert len(bm.list_datasets(verbose=False)) == 1


@pytest.mark.benchmark
def test_feynman_save_load():
    b1 = SR_benchmark.feynman(FEYNMAN_DIR)
    b2 = SR_benchmark.load_benchmark(FEYNMAN_DIR)
    assert len(b1.list_datasets(verbose=False)) == len(b2.list_datasets(verbose=False))

    dataset_name = b2.list_datasets(verbose=False)[0]
    dataset = b2.create_dataset(dataset_name)
    rmse = dataset.create_evaluator().evaluate_expr(dataset.ground_truth)
    assert rmse < dataset.success_threshold


@pytest.mark.benchmark
def test_feynman_ground_truth():
    benchmark = SR_benchmark.feynman(FEYNMAN_DIR)
    failures = []
    for dataset_name in benchmark.list_datasets(verbose=False):
        ds = benchmark.create_dataset(dataset_name)
        rmse = ds.create_evaluator().evaluate_expr(ds.ground_truth)
        if rmse > ds.success_threshold:
            failures.append(f"{dataset_name}: RMSE={rmse}")
    assert not failures, "Ground truth check failed for:\n" + "\n".join(failures)


@pytest.mark.benchmark
def test_nguyen_ground_truth():
    benchmark = SR_benchmark.nguyen(NGUYEN_DIR)
    failures = []
    for dataset_name in benchmark.list_datasets(verbose=False):
        ds = benchmark.create_dataset(dataset_name)
        rmse = ds.create_evaluator().evaluate_expr(ds.ground_truth)
        if rmse > ds.success_threshold:
            failures.append(f"{dataset_name}: RMSE={rmse}")
    assert not failures, "Ground truth check failed for:\n" + "\n".join(failures)


@pytest.mark.benchmark
def test_proged_on_feynman():
    from SRToolkit.approaches import ProGED

    benchmark = SR_benchmark.feynman(FEYNMAN_DIR)
    dataset = benchmark.create_dataset("I.16.6")
    # Change max_evaluations to 100 to speed up the test as results here don't matter.
    dataset.max_evaluations = 100

    model = ProGED(dataset.symbol_library, verbose=False)
    results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=18, verbose=False)

    assert len(results) == 1
    assert results[0].num_evaluated > 0
    assert results[0].evaluation_calls <= 100
