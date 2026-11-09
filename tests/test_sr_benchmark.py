"""Tests for SRToolkit.dataset.sr_benchmark (stage C: background knowledge, format 4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from SRToolkit.dataset import Feynman, data_cache
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.data_source import SampleSource
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.dataset.sr_benchmark import SR_benchmark
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTokens, MaxTreeDepth, RequiredVariables
from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.grammar.constraints import MaxDepth
from SRToolkit.utils.symbol_library import SymbolLibrary

_GT = ["X_0", "+", "X_1"]
_X = np.random.default_rng(0).random((5, 2))
_XY = (_X, _X.sum(axis=1))
_NINE_CONSTANTS = ["C", "*", "X_0", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C", "+", "C"]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr("SRToolkit.dataset.data_cache.data_root", lambda: tmp_path / "cache")


def _lib2c():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _samplers():
    return {"X_1": UniformSampling(10, 11, uses_negative=False), "X_0": UniformSampling(0, 1, uses_negative=False)}


def _dicts(items):
    return [c.to_dict() for c in items]


def test_add_dataset_v4_entry_builds_dataset():
    flat = dict(
        samplers=_samplers(),
        variables={"X_0": VariableInfo("x", "position")},
        target=VariableInfo("z"),
        description="Toy",
        constant_bounds=(-3, 3),
    )
    bm = SR_benchmark("b")
    bm.add_dataset(
        _lib2c(),
        None,
        dataset_name="a",
        ground_truth=_GT,
        original_equation="z = x + y",
        data_source=SampleSource(n_samples=20, seed=0),
        max_expr_length=12,
        required_constraints=[MaxTreeDepth(5)],
        **flat,
    )
    entry = bm.datasets["a"]
    assert entry["format_version"] == 4
    assert "samplers" not in entry and "symbol_library" not in entry
    assert "constant_bounds" not in entry["kwargs"]
    ds = bm.create_dataset("a")
    expected = SR_dataset(np.zeros((1, 2)), _lib2c(), **flat).background_knowledge
    assert ds.background_knowledge.to_dict() == expected.to_dict()
    assert _dicts(ds.required_constraints) == _dicts([MaxConstants(8), MaxTokens(12), MaxTreeDepth(5)])
    assert ds.X.shape == (20, 2)
    assert ds.original_equation == "z = x + y"


def _lib_sq():
    # "sq" is not a default symbol, so y can only be computed with this library.
    lib = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
    lib.add_symbol("sq", "fn", 5, "np.square({})")
    return lib


@pytest.mark.parametrize(
    "dataset, ranking_function, library, ground_truth, expected_y",
    [
        ((_X, _X.sum(axis=1)), "rmse", _lib2c, _GT, _X.sum(axis=1)),
        (_X, "rmse", _lib2c, _GT, _X.sum(axis=1)),
        (_X, "rmse", _lib_sq, ["sq", "(", "X_0", ")", "+", "X_1"], _X[:, 0] ** 2 + _X[:, 1]),
        (_X, "bed", _lib2c, _GT, None),
    ],
    ids=["tuple", "X-only-rmse-y-from-gt", "X-only-rmse-custom-library", "X-only-bed"],
)
def test_add_dataset_writes_arrays_to_cache(dataset, ranking_function, library, ground_truth, expected_y):
    bm = SR_benchmark("b")
    bm.add_dataset(library(), dataset, "a", ranking_function=ranking_function, ground_truth=ground_truth)
    ds = bm.create_dataset("a")
    np.testing.assert_array_equal(ds.X, _X)
    np.testing.assert_equal(ds.y, expected_y)  # also matches None to None


def test_add_dataset_node_ground_truth_saved_as_tokens():
    bm = SR_benchmark("b")
    bm.add_dataset(_lib2c(), _XY, "a", ground_truth=tokens_to_tree(_GT, _lib2c()))
    assert bm.create_dataset("a").ground_truth == _GT


def test_add_from_samplers_forwards_arguments():
    bm = SR_benchmark("b")
    bm.add_from_samplers(
        _GT,
        [UniformSampling(0, 1)] * 2,
        symbol_library=_lib2c(),
        n_samples=8,
        seed=3,
        ranking_function="bed",
        dataset_name="a",
        original_equation="z = x + y",
        success_threshold=0.1,
        max_evaluations=50,
        dataset_metadata={"citation": "x"},
        variables={"X_0": VariableInfo("x")},
        target=VariableInfo("z"),
        description="Toy",
        max_constants=2,
        max_expr_length=9,
        required_constraints=[MaxTreeDepth(5)],
        method="L-BFGS-B",
    )
    ds = bm.create_dataset("a")
    reference = SR_dataset.from_samplers(_GT, [UniformSampling(0, 1)] * 2, n_samples=8, seed=3)
    np.testing.assert_array_equal(ds.X, reference.X)
    assert (ds.ranking_function, ds.seed, ds.original_equation) == ("bed", 3, "z = x + y")
    assert (ds.success_threshold, ds.max_evaluations, ds.dataset_metadata) == (0.1, 50, {"citation": "x"})
    assert ds.background_knowledge.variables == {"X_0": VariableInfo("x")}
    assert (ds.background_knowledge.target, ds.background_knowledge.description) == (VariableInfo("z"), "Toy")
    assert _dicts(ds.required_constraints) == _dicts([MaxConstants(2), MaxTokens(9), MaxTreeDepth(5)])
    assert ds.kwargs == {"method": "L-BFGS-B"}


def test_create_dataset_resamples_v4():
    bm = SR_benchmark("b")
    bm.add_from_samplers(_GT, _samplers(), symbol_library=_lib2c(), dataset_name="a", n_samples=10, seed=0)
    ds = bm.create_dataset("a", n_samples=50, seed=1)
    assert ds.X.shape == (50, 2)
    assert np.all((ds.X[:, 0] >= 0) & (ds.X[:, 0] <= 1))
    assert np.all((ds.X[:, 1] >= 10) & (ds.X[:, 1] <= 11))
    np.testing.assert_allclose(ds.y, ds.X[:, 0] + ds.X[:, 1])
    np.testing.assert_array_equal(bm.create_dataset("a", n_samples=50, seed=1).X, ds.X)


def _add_existing(bm):
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="x", n_samples=5, seed=0)


@pytest.mark.parametrize(
    "add, exc",
    [
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, samplers=[UniformSampling(0, 1)] * 3),
            ValueError,
        ),
        (lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, max_constants=True), ValueError),
        (lambda bm: (_add_existing(bm), bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT)), ValueError),
        (lambda bm: bm.add_dataset(_lib2c(), (_X,), "x", ground_truth=_GT), ValueError),
        (lambda bm: bm.add_dataset(_lib2c(), (_X, [1.0] * 5), "x", ground_truth=_GT), ValueError),
        (lambda bm: bm.add_dataset(_lib2c(), _X, "x"), ValueError),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, required_constraints=[MaxDepth(3)]),
            TypeError,
        ),
        (
            lambda bm: bm.add_dataset(
                _lib2c(), _XY, "x", ground_truth=_GT, required_constraints=[RequiredVariables("X_9")]
            ),
            ValueError,
        ),
        (lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=np.zeros((5, 4))), ValueError),
        (lambda bm: bm.add_dataset(_lib2c(), _X, "x", ranking_function="bed"), ValueError),
        (lambda bm: bm.add_dataset(_lib2c(), (np.zeros((5, 3)), np.zeros(5)), "x", ground_truth=_GT), ValueError),
    ],
    ids=[
        "sampler-count",
        "max-constants-bool",
        "duplicate-name",
        "tuple-not-two-arrays",
        "tuple-y-not-array",
        "rmse-array-without-gt",
        "grammar-constraint",
        "misfit-required-constraint",
        "ndarray-ground-truth",
        "bed-without-gt",
        "columns-differ-from-variables",
    ],
)
def test_add_dataset_rejects_before_writing(add, exc):
    bm = SR_benchmark("b")
    with pytest.raises(exc):
        add(bm)
    assert not data_cache.dataset_path("b", bm.version, "x").exists()


@pytest.mark.parametrize(
    "name, n_samples, match",
    [("missing", None, "missing"), ("x", 5, "samplers")],
    ids=["unknown-name", "resample-without-samplers"],
)
def test_create_dataset_errors(name, n_samples, match):
    bm = SR_benchmark("b")
    bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT)
    with pytest.raises(ValueError, match=match):
        bm.create_dataset(name, n_samples=n_samples)


def test_add_dataset_gt_eval_error_chained():
    bm = SR_benchmark("b")
    with pytest.raises(ValueError) as info:
        bm.add_dataset(SymbolLibrary.default_symbols(1), np.ones((5, 1)), "x", ground_truth=["X_0", "*", "C"])
    assert info.value.__cause__ is not None


def test_add_from_samplers_forwards_background_knowledge():
    bk = BackgroundKnowledge(
        _lib2c(), samplers=[UniformSampling(0, 1)] * 2, description="Toy", constraints=[MaxTreeDepth(3)]
    )
    bm = SR_benchmark("b")
    bm.add_from_samplers(
        _GT,
        [UniformSampling(0, 1)] * 2,
        symbol_library=_lib2c(),
        description="Toy",
        advisory_constraints=[MaxTreeDepth(3)],
        dataset_name="a",
        n_samples=30,
        seed=0,
    )
    assert not data_cache.dataset_path("b", bm.version, "a").exists()
    ds = bm.create_dataset("a")
    assert ds.background_knowledge.to_dict() == bk.to_dict()
    assert ds.X.shape == (30, 2)
    assert ds.original_equation == "y = X_0+X_1"


@pytest.mark.parametrize("samplers", [None, []], ids=["none", "empty"])
def test_add_from_samplers_errors(samplers):
    with pytest.raises(ValueError):
        SR_benchmark("b").add_from_samplers(_GT, samplers, symbol_library=_lib2c(), dataset_name="a")


def test_v3_benchmark_entry_migrates(tmp_path):
    # A dataset entry as SR_benchmark wrote it before format 4 (benchmark file format 2).
    v3 = json.loads((Path(__file__).parent / "data" / "v3_cache" / "config_v3.json").read_text())
    del v3["test"], v3["aux"]
    v3.update(benchmark="b", dataset_name="a")  # no "num_variables": inferred from the samplers
    bm = SR_benchmark("b", metadata={"citation": "x"})
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="a", n_samples=5, seed=0)
    d = bm.to_dict()
    d["datasets"]["a"] = v3
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(d))
    loaded = SR_benchmark.from_dict(path)
    assert loaded.metadata == {"citation": "x"}
    assert loaded.list_datasets(verbose=False, num_variables=2) == ["a"]
    ds = loaded.create_dataset("a")
    assert ds.constant_bounds == (-3.0, 3.0)
    assert _dicts(ds.required_constraints) == [MaxConstants(4).to_dict()]
    assert ds.X.shape == (5, 2)


# Nguyen's libraries have no free-constant symbol, so the MaxConstants limit cannot show there.
@pytest.mark.parametrize("benchmark, name", [(Feynman, "I.12.1"), (Feynman, "I.16.6")], ids=["I.12.1", "I.16.6"])
def test_builtin_scores_unchanged(benchmark, name):
    bm = benchmark(n_samples=100, force_generate=True)
    assert bm.datasets[name]["format_version"] == 4
    ds = bm.create_dataset(name)
    ev = ds.create_evaluator()
    reference = SR_evaluator(ds.X, ds.y, symbol_library=ds.symbol_library, ground_truth=ds.ground_truth, seed=0)

    ev.evaluate_expr(_NINE_CONSTANTS)
    assert (ev.models[-1].status, ev.models[-1].status_detail) == ("rejected", "MaxConstants: 9 constants, limit 8")
    ev.evaluate_expr(["X_0", "+"])
    assert ev.models[-1].status == "invalid"
    assert ev.evaluate_expr(["C", "*", "X_0"]) == pytest.approx(reference.evaluate_expr(["C", "*", "X_0"]))
    # Last: reaching the success threshold stops the evaluator.
    assert ev.evaluate_expr(ds.ground_truth) < 1e-9
    assert ev.models[-1].status == "ok"


# ── Regression tests for older SR_benchmark behaviour (fixed with stage C) ────


def test_create_dataset_keeps_documented_exception_types():
    # regression: every from_dict error used to become a bare ValueError without its cause.
    bm = SR_benchmark("b")
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="a", n_samples=5, seed=0)
    bm.datasets["a"]["background_knowledge"]["samplers"][0]["sampler_class"] = "nope.Missing"
    with pytest.raises(ImportError):
        bm.create_dataset("a")
    bm.datasets["a"]["format_version"] = 99
    with pytest.raises(ValueError) as info:
        bm.create_dataset("a")
    assert info.value.__cause__ is not None


def test_add_dataset_without_data_or_source_rejected():
    # regression: an entry that can never be materialised was stored silently.
    bm = SR_benchmark("b")
    with pytest.raises(ValueError, match="data_source"):
        bm.add_dataset(_lib2c(), None, "a", ground_truth=_GT)
    assert "a" not in bm.datasets


def test_from_dict_does_not_modify_input():
    # regression: from_dict wrote num_variables into the caller's entry dicts.
    bm = SR_benchmark("b")
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="a", n_samples=5, seed=0)
    d = bm.to_dict()
    del d["datasets"]["a"]["num_variables"]
    loaded = SR_benchmark.from_dict(d)
    assert "num_variables" not in d["datasets"]["a"]
    assert loaded.datasets["a"]["num_variables"] == 2
