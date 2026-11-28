"""Tests for SRToolkit.dataset.sr_benchmark (stage C: background knowledge, format 4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from SRToolkit.dataset import Feynman, data_cache
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
from SRToolkit.dataset.data_source import FallbackSource, SampleSource, UrlSource
from SRToolkit.dataset.sampling import LogUniformSampling, UniformSampling
from SRToolkit.dataset.sr_benchmark import SR_benchmark
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils import units as u
from SRToolkit.utils.expression import Expression
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTokens, MaxTreeDepth, RequiredVariables
from SRToolkit.utils.expression_tree import ExpressionParseError
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.grammar.constraints import MaxDepth
from SRToolkit.utils.grammar.grammar import Grammar, Rule
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.units import Unit

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
    return {"X_1": UniformSampling(10, 11), "X_0": UniformSampling(0, 1)}


_GT_C0 = ["C_0", "*", "X_0", "+", "C_0"]
_FREE = Constant("C_0", prior=UniformSampling(0, 1))
_LU_DEFAULT = Constant(prior=LogUniformSampling(0.1, 10), bounds=(0, float("inf")))


def _grammar():
    return Grammar(Rule.from_line("E -> E '+' F [0.4] | F [0.6]") + Rule.from_line("F -> 'X_0' [1.0]"))


def _dicts(items):
    return [c.to_dict() for c in items]


def test_add_dataset_v5_entry_builds_dataset():
    flat = dict(
        samplers=_samplers(),
        variables={"X_0": VariableInfo("x", "position")},
        target=VariableInfo("z"),
        description="Toy",
        constant_bounds=(-3, 3),
        default_constant=_LU_DEFAULT,
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
        settings=Settings(minimize={"max_iter": 5}),
        ranking_resample=True,
        **flat,
    )
    entry = bm.datasets["a"]
    assert entry["format_version"] == 5
    assert "samplers" not in entry and "symbol_library" not in entry
    assert (entry["settings"], entry["ranking_resample"]) == (Settings(minimize={"max_iter": 5}).to_dict(), True)
    ds = SR_benchmark.from_dict(json.loads(json.dumps(bm.to_dict()))).create_dataset("a")
    assert (ds.settings, ds.ranking_resample) == (Settings(minimize={"max_iter": 5}), True)
    expected = SR_dataset(np.zeros((1, 2)), _lib2c(), **flat).background_knowledge
    assert ds.background_knowledge.to_dict() == expected.to_dict()
    assert ds.default_constant.to_dict() == _LU_DEFAULT.to_dict()
    assert _dicts(ds.required_constraints) == _dicts([MaxConstants(8), MaxTokens(12), MaxTreeDepth(5)])
    assert ds.X.shape == (20, 2)
    assert ds.original_equation == "z = x + y"


def _lib_sq():
    # "sq" is not a default symbol, so y can only be computed with this library.
    lib = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
    lib.add_symbol("sq", "fn", 5, "np.square({})")
    return lib


def _lib_c1():
    # "C_1" is a named constant (value 3) here, not a labelled free constant, so it needs no spec.
    lib = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)
    lib.add_symbol("C_1", "lit", 5, "3.0")
    return lib


@pytest.mark.parametrize(
    "dataset, ranking_function, library, ground_truth, gt_constants, expected_y",
    [
        ((_X, _X.sum(axis=1)), "rmse", _lib2c, _GT, None, _X.sum(axis=1)),
        (_X, "rmse", _lib2c, _GT, None, _X.sum(axis=1)),
        (_X, "rmse", _lib_sq, ["sq", "(", "X_0", ")", "+", "X_1"], None, _X[:, 0] ** 2 + _X[:, 1]),
        (_X, "bed", _lib2c, _GT, None, None),
        (_X, "rmse", _lib_c1, ["X_0", "+", "C_1"], None, _X[:, 0] + 3.0),
        # A free (non-fixed) constant spec is fine with (X, y): y is given, not computed.
        ((_X, _X.sum(axis=1)), "rmse", _lib2c, _GT_C0, [_FREE], _X.sum(axis=1)),
    ],
    ids=[
        "tuple",
        "X-only-rmse-y-from-gt",
        "X-only-rmse-custom-library",
        "X-only-bed",
        "X-only-named-constant-C_1",
        "tuple-free-gt-constant",
    ],
)
def test_add_dataset_writes_arrays_to_cache(dataset, ranking_function, library, ground_truth, gt_constants, expected_y):
    bm = SR_benchmark("b")
    bm.add_dataset(
        library(),
        dataset,
        "a",
        ranking_function=ranking_function,
        ground_truth=ground_truth if gt_constants is None else Expression(ground_truth, free_constants=gt_constants),
    )
    entry = bm.datasets["a"]
    assert entry["data_source"] is None
    assert (entry["ranking_resample"], entry["settings"]) == (False, Settings().to_dict())
    ds = bm.create_dataset("a")
    # The entry is the dataset's to_dict plus num_variables; add_dataset makes no test/aux splits.
    assert set(entry) == (set(ds.to_dict()) - {"test", "aux"}) | {"num_variables"}
    np.testing.assert_array_equal(ds.X, _X)
    np.testing.assert_equal(ds.y, expected_y)  # also matches None to None


# Reason: regression (add_from_samplers with symbol_library None ignored an Expression's library) +
# clause (None uses the Expression's own library).
def test_add_from_samplers_uses_the_expression_library():
    lib = SymbolLibrary.default_symbols(1, const_symbol="K")
    bm = SR_benchmark("b")
    bm.add_from_samplers(
        Expression(["K", "*", "X_0"], lib),
        [UniformSampling(0, 1)],
        dataset_name="a",
        n_samples=4,
        ranking_function="bed",
    )
    assert bm.create_dataset("a").symbol_library.constant_symbol == "K"


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
        settings=Settings(minimize={"method": "Nelder-Mead"}),
        ranking_resample=True,
    )
    ds = bm.create_dataset("a")
    reference = SR_dataset.from_samplers(_GT, [UniformSampling(0, 1)] * 2, n_samples=8, seed=3)
    np.testing.assert_array_equal(ds.X, reference.X)
    assert (ds.ranking_function, ds.seed, ds.original_equation) == ("bed", 3, "z = x + y")
    assert (ds.success_threshold, ds.max_evaluations, ds.dataset_metadata) == (0.1, 50, {"citation": "x"})
    assert ds.background_knowledge.variables == {"X_0": VariableInfo("x")}
    assert (ds.background_knowledge.target, ds.background_knowledge.description) == (VariableInfo("z"), "Toy")
    assert _dicts(ds.required_constraints) == _dicts([MaxConstants(2), MaxTokens(9), MaxTreeDepth(5)])
    assert (ds.settings, ds.ranking_resample) == (Settings(minimize={"method": "Nelder-Mead"}), True)


def test_create_dataset_resamples_v4():
    bm = SR_benchmark("b")
    bm.add_from_samplers(_GT, _samplers(), symbol_library=_lib2c(), dataset_name="a", n_samples=10, seed=0)
    ds = bm.create_dataset("a", n_samples=50, seed=1)
    assert ds.X.shape == (50, 2)
    assert np.all((ds.X[:, 0] >= 0) & (ds.X[:, 0] <= 1))
    assert np.all((ds.X[:, 1] >= 10) & (ds.X[:, 1] <= 11))
    np.testing.assert_allclose(ds.y, ds.X[:, 0] + ds.X[:, 1])
    np.testing.assert_array_equal(bm.create_dataset("a", n_samples=50, seed=1).X, ds.X)
    assert (ds.settings, ds.ranking_resample) == (Settings(), False)


_FIX2 = Constant("C_0", value=2.0, fixed=True)


# Reason: clause (add_dataset / add_from_samplers store the ground truth's specs in the entry; y of a bare
# ndarray and of a SampleSource is computed with the fixed values). y = 2x + 2.
@pytest.mark.parametrize(
    "add, n_samples",
    [
        (lambda bm: bm.add_dataset(_lib2c(), _X, "a", ground_truth=Expression(_GT_C0, free_constants=[_FIX2])), None),
        (
            lambda bm: bm.add_from_samplers(
                Expression(_GT_C0, _lib2c(), [_FIX2]), [UniformSampling(0, 1)] * 2, dataset_name="a",
                n_samples=6, seed=0,
            ),
            None,
        ),
        (
            lambda bm: bm.add_from_samplers(
                Expression(_GT_C0, _lib2c(), [_FIX2]), [UniformSampling(0, 1)] * 2, dataset_name="a",
                n_samples=6, seed=0,
            ),
            7,
        ),
    ],
    ids=["add_dataset-ndarray", "add_from_samplers", "add_from_samplers-resampled"],
)  # fmt: skip
def test_ground_truth_constants_reach_dataset_and_y(add, n_samples):
    bm = SR_benchmark("b")
    add(bm)
    assert bm.datasets["a"]["ground_truth_constants"] == [_FIX2.to_dict()]
    ds = bm.create_dataset("a", n_samples=n_samples, seed=1)
    assert ds.ground_truth_constants == (_FIX2,)
    np.testing.assert_allclose(ds.y, 2 * ds.X[:, 0] + 2)


def _add_existing(bm):
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="x", n_samples=5, seed=0)


@pytest.mark.parametrize(
    "add, exc, match",
    [
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, samplers=[UniformSampling(0, 1)] * 3),
            ValueError,
            "samplers",
        ),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, max_constants=True),
            ValueError,
            "max_constants",
        ),
        (lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, settings={"bed": {}}), TypeError, "Settings"),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, ranking_resample=1),
            TypeError,
            "ranking_resample",
        ),
        (
            lambda bm: bm.add_from_samplers(
                _GT,
                [UniformSampling(0, 1)] * 2,
                dataset_name="x",
                n_samples=5,
                settings=Settings(evaluator={"max_expr_length": 5}),
            ),
            ValueError,
            "evaluator.max_expr_length",
        ),  # fmt: skip
        (
            lambda bm: (_add_existing(bm), bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT)),
            ValueError,
            "already exists",
        ),
        (lambda bm: bm.add_dataset(_lib2c(), (_X,), "x", ground_truth=_GT), ValueError, "tuple"),
        (lambda bm: bm.add_dataset(_lib2c(), (_X, [1.0] * 5), "x", ground_truth=_GT), ValueError, "tuple"),
        (lambda bm: bm.add_dataset(_lib2c(), _X, "x"), ValueError, "ground truth must be provided"),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, required_constraints=[MaxDepth(3)]),
            TypeError,
            "ExpressionConstraint",
        ),
        (
            lambda bm: bm.add_dataset(
                _lib2c(), _XY, "x", ground_truth=_GT, required_constraints=[RequiredVariables("X_9")]
            ),
            ValueError,
            "X_9",
        ),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=np.zeros((5, 4))),
            ValueError,
            "ndarray ground_truth",
        ),
        (
            lambda bm: bm.add_dataset(_lib2c(), _X, "x", ground_truth=np.zeros((5, 4))),
            ValueError,
            "ndarray ground_truth",
        ),
        (lambda bm: bm.add_dataset(_lib2c(), _X, "x", ranking_function="bed"), ValueError, "'bed' ranking"),
        (
            lambda bm: bm.add_dataset(_lib2c(), _XY, "x", ground_truth=_GT, max_evaluations=0),
            ValueError,
            "max_evaluations",
        ),
        (
            lambda bm: bm.add_dataset(_lib2c(), (np.zeros((5, 3)), np.zeros(5)), "x", ground_truth=_GT),
            ValueError,
            "columns",
        ),
        (lambda bm: bm.add_dataset(_lib2c(), np.zeros((5, 3)), "x", ground_truth=_GT), ValueError, "columns"),
        (
            lambda bm: bm.add_dataset(_lib2c(), _X, "x", ground_truth=Expression(_GT_C0, free_constants=[_FREE])),
            ValueError,
            "free constant",
        ),
        (
            lambda bm: bm.add_dataset(
                _lib2c(),
                None,
                "x",
                ground_truth=Expression(_GT_C0, free_constants=[_FREE]),
                samplers=[UniformSampling(0, 1)] * 2,
                data_source=SampleSource(n_samples=5, seed=0),
            ),
            ValueError,
            "free constant",
        ),  # fmt: skip
        (
            lambda bm: bm.add_dataset(
                _lib2c(),
                None,
                "x",
                ground_truth=Expression(_GT_C0, free_constants=[_FREE]),
                samplers=[UniformSampling(0, 1)] * 2,
                data_source=FallbackSource([UrlSource("http://example.invalid/x.zip"), SampleSource(n_samples=5)]),
            ),
            ValueError,
            "free constant",
        ),  # fmt: skip
        (
            lambda bm: bm.add_from_samplers(
                Expression(_GT_C0, _lib2c(), [_FREE]),
                [UniformSampling(0, 1)] * 2,
                dataset_name="x",
                n_samples=5,
                seed=0,
            ),
            ValueError,
            "free constant",
        ),  # fmt: skip
    ],
    ids=[
        "sampler-count",
        "max-constants-bool",
        "settings-not-settings",
        "ranking-resample-int",
        "from-samplers-settings-max-expr-length",
        "duplicate-name",
        "tuple-not-two-arrays",
        "tuple-y-not-array",
        "rmse-array-without-gt",
        "grammar-constraint",
        "misfit-required-constraint",
        "ndarray-ground-truth",
        "ndarray-ground-truth-ndarray-data",
        "bed-without-gt",
        "zero-budget",
        "columns-differ-from-variables",
        "ndarray-columns-differ-from-variables",
        "gt-constant-not-fixed-ndarray",
        "gt-constant-not-fixed-sample-source",
        "gt-constant-not-fixed-fallback-source",
        "gt-constant-not-fixed-from-samplers",
    ],
)
def test_add_dataset_rejects_before_writing(add, exc, match):
    bm = SR_benchmark("b")
    with pytest.raises(exc, match=match):
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
    # The ground truth parses but cannot be evaluated: its function's body names nothing that exists.
    lib = SymbolLibrary.default_symbols(1)
    lib.add_symbol("bad", "fn", 5, "no_such_function({})")
    bm = SR_benchmark("b")
    with pytest.raises(ValueError, match="Could not evaluate the ground truth") as info:
        bm.add_dataset(lib, np.ones((5, 1)), "x", ground_truth=["bad", "(", "X_0", ")"])
    assert info.value.__cause__ is not None


# Reason: clause (Raises: a ground truth that does not parse is rejected before anything is written).
def test_add_dataset_rejects_unparsable_ground_truth():
    bm = SR_benchmark("b")
    with pytest.raises(ExpressionParseError, match=r"^\[SR_benchmark.add_dataset\] 'ground_truth' does not parse"):
        bm.add_dataset(SymbolLibrary.default_symbols(1), np.ones((5, 1)), "x", ground_truth=["X_0", "*"])
    assert not data_cache.dataset_path("b", bm.version, "x").exists()


def test_add_from_samplers_forwards_background_knowledge():
    bk = BackgroundKnowledge(
        _lib2c(),
        samplers=[UniformSampling(0, 1)] * 2,
        description="Toy",
        constraints=[MaxTreeDepth(3)],
        constant_bounds=(-2, 3),
        default_constant=_LU_DEFAULT,
        reference_grammar=_grammar(),
    )
    bm = SR_benchmark("b")
    bm.add_from_samplers(
        _GT,
        [UniformSampling(0, 1)] * 2,
        symbol_library=_lib2c(),
        description="Toy",
        advisory_constraints=[MaxTreeDepth(3)],
        constant_bounds=(-2, 3),
        default_constant=_LU_DEFAULT,
        reference_grammar=_grammar(),
        dataset_name="a",
        n_samples=30,
        seed=0,
    )
    assert not data_cache.dataset_path("b", bm.version, "a").exists()
    ds = bm.create_dataset("a")
    assert ds.background_knowledge.to_dict() == bk.to_dict()
    assert ds.background_knowledge.reference_grammar.to_dict() == _grammar().to_dict()
    assert ds.X.shape == (30, 2)
    assert ds.original_equation == "y = X_0+X_1"


@pytest.mark.parametrize("samplers", [None, []], ids=["none", "empty"])
def test_add_from_samplers_errors(samplers):
    with pytest.raises(ValueError, match="non-empty"):
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
    assert bm.datasets[name]["format_version"] == 5
    ds = bm.create_dataset(name)
    ev = ds.create_evaluator()
    reference = SR_evaluator(ds.X, ds.y, symbol_library=ds.symbol_library, ground_truth=ds.ground_truth, seed=0)

    ev.evaluate_expr(_NINE_CONSTANTS)
    assert (ev.models[-1].status, ev.models[-1].status_detail) == ("rejected", "MaxConstants: 9 constants, limit 8")
    ev.evaluate_expr(["X_0", "+"])
    assert ev.models[-1].status == "invalid"
    assert ev.evaluate_expr(["C", "*", "X_0"]).error == pytest.approx(reference.evaluate_expr(["C", "*", "X_0"]).error)
    # Last: reaching the success threshold stops the evaluator.
    assert ev.evaluate_expr(ds.ground_truth).error < 1e-9
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


def test_from_dict_rejects_zip_path():
    with pytest.raises(ValueError, match="share.load"):
        SR_benchmark.from_dict("benchmark.zip")


def test_from_dict_does_not_modify_input():
    # regression: from_dict wrote num_variables into the caller's entry dicts.
    bm = SR_benchmark("b")
    bm.add_from_samplers(_GT, [UniformSampling(0, 1)] * 2, dataset_name="a", n_samples=5, seed=0)
    d = bm.to_dict()
    del d["datasets"]["a"]["num_variables"]
    loaded = SR_benchmark.from_dict(d)
    assert "num_variables" not in d["datasets"]["a"]
    assert loaded.datasets["a"]["num_variables"] == 2


# ── original_equation: stored only when given, derived when built or listed ────

_NAMES = dict(variables={"X_0": VariableInfo("x")}, target=VariableInfo("z"))


@pytest.mark.parametrize(
    "add",
    [
        lambda bm: bm.add_dataset(_lib2c(), _XY, "a", ground_truth=_GT, **_NAMES),
        lambda bm: bm.add_from_samplers(
            _GT, [UniformSampling(0, 1)] * 2, symbol_library=_lib2c(), dataset_name="a", n_samples=5, seed=0, **_NAMES
        ),
    ],
    ids=["add_dataset", "add_from_samplers"],
)
def test_entry_stores_none_and_dataset_derives(add):
    bm = SR_benchmark("b")
    add(bm)
    assert bm.datasets["a"]["original_equation"] is None
    assert bm.create_dataset("a").original_equation == "z = x+X_1"


def test_list_datasets_prints_equations(capsys):
    bm = SR_benchmark("b")
    bm.add_dataset(_lib2c(), _XY, "given_entry", ground_truth=_GT, original_equation="z = x + y")
    bm.add_dataset(_lib2c(), _XY, "derived_entry", ground_truth=_GT, **_NAMES)
    bm.add_dataset(_lib2c(), _XY, "untargeted_entry", ground_truth=_GT, variables={"X_1": VariableInfo("v")})
    instance = SR_dataset(
        _X, _lib2c(), y=_XY[1], ground_truth=["X_0", "*", "X_1"],
        variables={"X_0": VariableInfo("m")}, target=VariableInfo("F"),
    )  # fmt: skip
    bm.add_dataset_instance("instance_entry", instance)
    bm.add_dataset(_lib2c(), _XY, "no_eq_entry")
    bm.list_datasets(verbose=True)
    out = capsys.readouterr().out
    assert "z = x + y" in out
    assert "z = x+X_1" in out
    assert "F = m*X_1" in out
    assert "y = X_0+v" in out
    assert "None" not in out
    [no_eq_line] = [line for line in out.splitlines() if "no_eq_entry" in line]
    assert "Expression" not in no_eq_line


def test_srsd_bonus_6_squares_epsilon():
    # Regression: the ground truth had epsilon where the formula has epsilon**2.
    from SRToolkit.dataset import SRSD_Feynman
    from SRToolkit.utils.expression_compiler import compile_expr

    entry = SRSD_Feynman().datasets["SRSD Bonus 6"]
    library = SymbolLibrary.from_dict(entry["background_knowledge"]["symbol_library"])
    eps, e_n, ang, m, z1, z2, q = 2.0, 3.0, 0.5, 1.5, 2.0, 3.0, 0.7
    X = np.array([[eps, e_n, ang, m, z1, z2, q]])
    expected = np.sqrt(1 + 2 * eps**2 * e_n * ang**2 / (m * (z1 * z2 * q**2) ** 2))
    assert compile_expr(entry["ground_truth"], library)(X, None)[0] == pytest.approx(expected)


# ── Named constants and units survive the benchmark entry ─────────────────────

_UNIT_INFOS = dict(
    variables={"X_0": VariableInfo("x", unit=u.m)}, named_constants={"pi": VariableInfo("π", unit=Unit())}
)


def _lib2pi():
    return SymbolLibrary.from_symbol_list(["+", "*", "C", "pi"], num_variables=2)


@pytest.mark.parametrize(
    "add",
    [
        lambda bm: bm.add_dataset(_lib2pi(), _XY, "a", ground_truth=_GT, **_UNIT_INFOS),
        lambda bm: bm.add_from_samplers(
            _GT,
            [UniformSampling(0, 1)] * 2,
            symbol_library=_lib2pi(),
            dataset_name="a",
            n_samples=5,
            seed=0,
            **_UNIT_INFOS,
        ),
    ],
    ids=["add_dataset", "add_from_samplers"],
)
def test_entry_keeps_constants_and_units(add, tmp_path):
    bm = SR_benchmark("b")
    add(bm)
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(bm.to_dict()))
    bk = SR_benchmark.from_dict(path).create_dataset("a").background_knowledge
    assert bk.named_constants == {"pi": VariableInfo("π", unit=Unit())}
    assert bk.variables["X_0"].unit == u.m
