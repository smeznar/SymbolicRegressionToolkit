"""Spec tests for SRToolkit.utils.types (constants, RankingSettings, ModelResult, EvalResult)."""

import dataclasses
import json
import math
import re
import typing

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
    CheckResult,
    EvalResult,
    ModelResult,
    RankingSettings,
    Status,
)

INF = float("inf")
NAN = float("nan")


def M(expr, error, status="ok", **kwargs):
    """A ModelResult; ``expr`` is a token list."""
    return ModelResult(expr=list(expr), error=error, status=status, **kwargs)


def F(expr=("X_0", "X_1")):
    """A failed model: invalid expression with error +inf."""
    return ModelResult(expr=list(expr), error=INF, status="invalid")


def R(models, trace, **kwargs):
    return EvalResult(all_models=models, trace=trace, approach_name=kwargs.pop("approach_name", ""), **kwargs)


def json_round_trip(obj):
    return type(obj).from_dict(json.loads(json.dumps(obj.to_dict())))


# ---------------------------------------------------------------------------------------------
# Constants and RankingSettings
# ---------------------------------------------------------------------------------------------


class TestConstants:
    def test_valid_symbol_types_is_exactly_the_five_types(self):
        assert VALID_SYMBOL_TYPES == {VAR, CONST, FN, OP, LIT}
        assert len(VALID_SYMBOL_TYPES) == 5

    def test_precedence_levels_are_ordered(self):
        assert OP_ADDITIVE < OP_MULTIPLICATIVE < OP_POWER < FN_PREFIX
        assert OP_POWER < LEAF
        assert FN_POSTFIX < 0
        assert FN_POSTFIX != -1

    def test_statuses_match_status_literal_in_order(self):
        assert STATUSES == typing.get_args(Status)


class TestRankingSettings:
    def test_ranking_settings_keys_are_the_documented_optional_keys(self):
        documented = {
            "method",
            "tol",
            "gtol",
            "max_iter",
            "constant_bounds",
            "initialization",
            "max_constants",
            "max_expr_length",
            "backend",
            "num_points_sampled",
            "samplers",
            "bed_gt",
            "ranking_seed",
            "bed_resample",
            "num_consts_sampled",
        }
        assert RankingSettings.__optional_keys__ == documented
        assert RankingSettings.__required_keys__ == frozenset()


# ---------------------------------------------------------------------------------------------
# ModelResult
# ---------------------------------------------------------------------------------------------


class TestCheckResult:
    def test_check_result_defaults(self):
        # A constraint fills passed/message/amount; the evaluator fills constraint and stage.
        result = CheckResult()
        assert (result.passed, result.message, result.amount) == (True, "", 0.0)
        assert (result.implemented, result.constraint, result.stage) == (True, "", "")
        assert result.error is None
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False


class TestModelResult:
    def test_model_result_defaults_and_independent_containers(self):
        a = ModelResult(expr=["X_0"], error=0.1)
        b = ModelResult(expr=["X_1"], error=0.2)
        assert a.parameters is None
        assert a.ranking_data == {}
        assert a.augmentations == {}
        assert a.status == "ok"
        assert a.status_detail is None
        assert a.advisory_violations == []
        a.ranking_data["k"] = 1
        a.augmentations["k"] = {}
        a.advisory_violations.append(CheckResult(passed=False))
        assert b.ranking_data == {}
        assert b.augmentations == {}
        assert b.advisory_violations == []

    def test_model_result_json_round_trip_restores_every_field(self):
        model = ModelResult(
            expr=["C", "*", "X_0", "+", "C"],
            error=INF,
            parameters=np.array([1.5, -2.0], dtype=np.float32),
            ranking_data={"t": (1, 2), 3: np.arange(4).reshape(2, 2)},
            augmentations={"aug": {"value": "s", "_type": "Aug"}},
            status="undefined",
            status_detail="nan score",
            advisory_violations=[
                CheckResult(False, "depth 3, limit 1", 2.0, True, "MaxTreeDepth", "pre"),
                CheckResult(False, "no fitted parameters", INF, True, "OutputBounds", "post"),
                CheckResult(False, "raised ValueError: x", INF, True, "Custom", "pre", "ValueError: x"),
                CheckResult(False, "numpy amount", np.int64(3), True, "Custom", "post"),
            ],
        )
        back = json_round_trip(model)
        assert back.expr == ["C", "*", "X_0", "+", "C"]
        assert back.error == INF
        assert back.parameters.dtype == np.float32
        assert back.parameters.shape == (2,)
        np.testing.assert_array_equal(back.parameters, [1.5, -2.0])
        assert type(back.ranking_data["t"]) is tuple
        assert back.ranking_data["t"] == (1, 2)
        assert list(back.ranking_data) == ["t", 3]
        assert back.ranking_data[3].dtype == np.arange(4).dtype
        np.testing.assert_array_equal(back.ranking_data[3], [[0, 1], [2, 3]])
        assert back.augmentations == {"aug": {"value": "s", "_type": "Aug"}}
        assert back.status == "undefined"
        assert back.status_detail == "nan score"
        assert back.advisory_violations == [
            CheckResult(False, "depth 3, limit 1", 2.0, True, "MaxTreeDepth", "pre"),
            CheckResult(False, "no fitted parameters", INF, True, "OutputBounds", "post"),
            CheckResult(False, "raised ValueError: x", INF, True, "Custom", "pre", "ValueError: x"),
            CheckResult(False, "numpy amount", 3, True, "Custom", "post"),
        ]
        assert type(back.advisory_violations[-1].amount) is int

    def test_model_result_numpy_scalars_come_back_as_python_scalars(self):
        model = ModelResult(expr=["X_0"], error=np.float64(0.25), ranking_data={"n": np.int64(3)})
        back = json_round_trip(model)
        assert type(back.error) is float
        assert back.error == 0.25
        assert type(back.ranking_data["n"]) is int
        assert back.ranking_data["n"] == 3

    def test_model_result_from_dict_missing_advisory_violations_loads_empty(self):
        # Results saved before advisory violations were recorded have no such key.
        data = ModelResult(expr=["X_0"], error=0.1).to_dict()
        data.pop("advisory_violations", None)
        assert ModelResult.from_dict(data).advisory_violations == []

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: d.pop("status"),
            lambda d: d.update(status="bogus"),
        ],
        ids=["missing", "bogus"],
    )
    def test_model_result_from_dict_rejects_bad_status(self, mutate):
        data = ModelResult(expr=["X_0"], error=0.1).to_dict()
        mutate(data)
        with pytest.raises(ValueError, match="status"):
            ModelResult.from_dict(data)


# ---------------------------------------------------------------------------------------------
# Shared behaviour of ModelResult and EvalResult
# ---------------------------------------------------------------------------------------------

FACTORIES = [
    pytest.param(lambda: ModelResult(expr=["X_0"], error=0.1), id="ModelResult"),
    pytest.param(lambda: R([M(["X_0"], 0.1)], [0]), id="EvalResult"),
]


class TestAddAugmentation:
    @pytest.mark.parametrize("make", FACTORIES)
    def test_add_augmentation_stores_typed_entry_under_first_free_key(self, make):
        result = make()
        result.add_augmentation("a", {"v": 1, "_type": "old"}, "T")
        result.add_augmentation("a_3", {"v": 2}, "T")
        result.add_augmentation("a", {"v": 3}, "U")
        result.add_augmentation("a", {"v": 4}, "U")
        result.add_augmentation("a", {"v": 5}, "U")
        # first free key among a_1, a_2, ...: a_1, a_2, then a_4 because a_3 is taken
        assert list(result.augmentations) == ["a", "a_3", "a_1", "a_2", "a_4"]
        assert result.augmentations["a"] == {"v": 1, "_type": "T"}
        assert result.augmentations["a_3"] == {"v": 2, "_type": "T"}
        assert result.augmentations["a_1"] == {"v": 3, "_type": "U"}
        assert result.augmentations["a_4"] == {"v": 5, "_type": "U"}

    @pytest.mark.parametrize("make", FACTORIES)
    def test_add_augmentation_stores_shallow_copy_and_does_not_modify_caller_dict(self, make):
        result = make()
        nested = [1]
        data = {"v": 1, "nested": nested}
        result.add_augmentation("a", data, "T")
        assert data == {"v": 1, "nested": [1]}
        data["v"] = 2
        assert result.augmentations["a"]["v"] == 1
        assert result.augmentations["a"]["nested"] is nested


def _mr(**kwargs):
    return ModelResult(expr=["X_0"], error=0.1, **kwargs)


def _er(models=None, **kwargs):
    models = models if models is not None else [M(["X_0"], 0.1)]
    return R(models, list(range(len(models))), **kwargs)


def _er_bad_model():
    return _er([M(["X_0"], 0.1), M(["X_1"], 0.2), M(["X_2"], 0.3), M(["X_3"], 0.4, ranking_data={"z": {1}})])


class TestToDictErrors:
    @pytest.mark.parametrize(
        "make, type_name",
        [
            (lambda: _mr(parameters=np.array([1j])), "complex"),
            (lambda: _mr(ranking_data={"z": {1}}), "set"),
            (lambda: _mr(augmentations={"a": {"v": b"x", "_type": "T"}}), "bytes"),
            (lambda: _er(augmentations={"a": {"v": {1}, "_type": "T"}}), "set"),
            (_er_bad_model, "set"),
        ],
        ids=[
            "MR-parameters-complex",
            "MR-ranking_data-set",
            "MR-augmentations-bytes",
            "ER-augmentations-set",
            "ER-model-ranking_data-set",
        ],
    )
    def test_to_dict_raises_type_error_for_unsaveable_value(self, make, type_name):
        with pytest.raises(TypeError, match=type_name):
            make().to_dict()

    @pytest.mark.parametrize(
        "make, path",
        [
            (lambda: _mr(parameters=np.array([1j])), "parameters"),
            (lambda: _mr(ranking_data={"z": {1}}), "ranking_data['z']"),
            (lambda: _mr(augmentations={"a": {"v": b"x", "_type": "T"}}), "augmentations['a']['v']"),
            (lambda: _er(augmentations={"a": {"v": {1}, "_type": "T"}}), "augmentations['a']['v']"),
            (lambda: _er(metadata={"m": {1}}), "metadata['m']"),
            (_er_bad_model, "all_models[3].ranking_data['z']"),
        ],
        ids=[
            "MR-parameters",
            "MR-ranking_data",
            "MR-augmentations",
            "ER-augmentations",
            "ER-metadata",
            "ER-model-ranking_data",
        ],
    )
    def test_to_dict_type_error_names_full_path(self, make, path):
        # Anchored: the path must be the whole location, not just a suffix of it.
        with pytest.raises(TypeError, match=rf" at {re.escape(path)}\.$"):
            make().to_dict()


# ---------------------------------------------------------------------------------------------
# EvalResult: construction and derived summaries
# ---------------------------------------------------------------------------------------------


class TestEvalResultSummaries:
    def test_eval_result_defaults(self):
        result = R([], [])
        assert result.success_threshold is None
        assert result.top_k == 20
        assert result.dataset_name is None
        assert result.metadata is None
        assert result.wall_time is None
        assert result.augmentations == {}

    def test_num_evaluated_and_evaluation_calls(self):
        # model 3 is never referenced by the trace but still counts as evaluated
        models = [M(["X_0"], 0.1), M(["X_1"], 0.2), M(["X_2"], 0.3), M(["X_3"], 0.4)]
        result = R(models, [0, 1, 0, 2, 2])
        assert result.num_evaluated == 4
        assert result.evaluation_calls == 5

    def test_top_models_sorted_by_error_failed_last(self):
        models = [M(["X_0"], 0.5), F(), M(["X_1"], 0.1), M(["X_2"], 0.3)]
        result = R(models, [0, 1, 2, 3])
        assert result.top_models(-1) == [models[2], models[3], models[0], models[1]]

    def test_top_models_ties_by_length_then_first_evaluated(self):
        # error decides, not status: the "rejected" model with a finite error ranks by error
        models = [
            M(["X_0", "+", "X_1"], 0.1),
            M(["X_0"], 0.1),
            M(["X_1"], 0.1),
            M(["X_0", "X_1", "X_2"], INF, status="invalid"),
            M(["X_0", "X_1"], NAN, status="undefined"),
            M(["X_1", "X_0"], INF, status="error"),
            M(["X_2"], 0.05, status="rejected"),
        ]
        result = R(models, list(range(7)))
        expected = [models[6], models[1], models[2], models[0], models[4], models[5], models[3]]
        assert [id(m) for m in result.top_models(-1)] == [id(m) for m in expected]

    @pytest.mark.parametrize(
        "top_k, k, expected",
        [
            (2, None, [1, 2]),
            (0, None, []),
            (2, 0, []),
            (2, -1, [1, 2, 0]),
            (2, 10, [1, 2, 0]),
            (2, 1, [1]),
        ],
        ids=["default-top_k", "top_k-zero", "k-zero", "k-negative", "k-too-large", "k-one"],
    )
    def test_top_models_k(self, top_k, k, expected):
        models = [M(["X_0"], 0.3), M(["X_1"], 0.1), M(["X_2"], 0.2)]
        result = R(models, [0, 1, 2], top_k=top_k)
        assert result.top_models(k) == [models[i] for i in expected]

    def test_summaries_follow_appended_models(self):
        models = [M(["X_0"], 0.5)]
        result = R(models, [0])
        assert result.min_error == 0.5
        new = M(["X_1"], 0.1)
        result.all_models.append(new)
        result.trace.append(1)
        assert result.min_error == 0.1
        assert result.best_expr == "X_1"
        assert result.top_models(1)[0] is new

    @pytest.mark.parametrize(
        "models, expected",
        [
            ([], NAN),
            ([F(), F()], INF),
            ([M(["X_0", "X_1"], INF, status="invalid"), M(["X_0"], NAN, status="undefined")], NAN),
            ([F(), M(["X_0"], 0.5), M(["X_1"], 0.1)], 0.1),
        ],
        ids=["empty", "all-inf", "all-failed-shorter-nan-ranks-first", "mixed"],
    )
    def test_min_error(self, models, expected):
        result = R(models, list(range(len(models))))
        np.testing.assert_equal(result.min_error, expected)

    @pytest.mark.parametrize(
        "models, expected",
        [
            ([], ""),
            ([M(["X_0"], 0.5), M(["C", "*", "X_0"], 0.05)], "C*X_0"),
        ],
        ids=["empty", "best"],
    )
    def test_best_expr(self, models, expected):
        assert R(models, list(range(len(models)))).best_expr == expected

    @pytest.mark.parametrize(
        "errors, threshold, expected",
        [
            ([0.05], 0.1, True),
            ([0.1], 0.1, False),
            ([0.2], 0.1, False),
            ([0.05], None, False),
            ([0.05], NAN, False),
            ([], 0.1, False),
        ],
        ids=["below", "equal", "above", "no-threshold", "nan-threshold", "empty"],
    )
    def test_success(self, errors, threshold, expected):
        models = [M([f"X_{i}"], e) for i, e in enumerate(errors)]
        assert R(models, list(range(len(models))), success_threshold=threshold).success is expected


# ---------------------------------------------------------------------------------------------
# EvalResult: trace-based metrics
# ---------------------------------------------------------------------------------------------


class TestEvalResultTrace:
    def test_best_so_far_records_strict_improvements(self):
        models = [M(["X_0", "+", "C"], 0.5), F(), M(["C", "*", "X_0"], 0.2), M(["C"], 0.2)]
        result = R(models, [1, 0, 0, 2, 3])
        assert result.best_so_far() == [(2, 0), (4, 2)]
        assert result.top_models(1)[0] is models[3]

    @pytest.mark.parametrize(
        "models, trace",
        [([], []), ([F(), M(["X_0"], NAN, status="undefined")], [0, 1, 0])],
        ids=["no-models", "all-failed"],
    )
    def test_best_so_far_empty(self, models, trace):
        assert R(models, trace).best_so_far() == []

    @pytest.mark.parametrize(
        "threshold, expected",
        [(None, 4), (0.6, 1), (0.5, 4), (0.001, None), (NAN, None)],
        ids=["success_threshold", "explicit-overrides", "strictly-below", "never", "nan"],
    )
    def test_calls_to_threshold(self, threshold, expected):
        models = [M(["X_0"], 0.5), M(["X_1"], INF, status="rejected"), M(["C"], 0.01)]
        result = R(models, [0, 1, 0, 2, 2], success_threshold=0.1)
        assert result.calls_to_threshold(threshold) == expected

    def test_calls_to_threshold_skips_failed_models(self):
        models = [M(["X_0"], -INF, status="undefined"), M(["C"], 0.01)]
        assert R(models, [0, 1]).calls_to_threshold(0.1) == 2

    def test_calls_to_threshold_without_threshold_raises(self):
        result = R([M(["X_0"], 0.5)], [0])
        with pytest.raises(ValueError, match="threshold"):
            result.calls_to_threshold()

    @pytest.mark.parametrize(
        "models, trace, budgets, expected",
        [
            (
                [F(), M(["X_1"], 0.5), M(["C"], 0.1)],
                [0, 1, 1, 2],
                [-1, 0, 1, 2, 3, 4, 100],
                [NAN, NAN, NAN, 0.5, 0.5, 0.1, 0.1],
            ),
            (
                [F(), M(["X_1"], 0.5), M(["C"], 0.1)],
                [0, 1, 1, 2],
                np.array([4, 1, 2], dtype=np.int64),
                [0.1, NAN, 0.5],
            ),
            ([M(["X_1"], 0.5)], [0], [1, 2], [0.5, 0.5]),
            ([F(), M(["X_1"], 0.5)], [0, 1], [], np.empty(0)),
            ([], [], [1, 5], [NAN, NAN]),
        ],
        ids=["budgets", "unsorted-numpy-ints", "single-call", "no-budgets", "no-models"],
    )
    def test_best_error_within(self, models, trace, budgets, expected):
        out = R(models, trace).best_error_within(budgets)
        assert out.shape == (len(expected),)
        np.testing.assert_array_equal(out, expected)

    def test_status_counts_unique_and_call(self):
        models = [
            M(["X_0"], 0.5),
            M(["X_0", "X_1"], INF, status="invalid"),
            M(["X_1", "X_0"], INF, status="invalid"),
            M(["X_1"], INF, status="error"),
        ]
        result = R(models, [0, 1, 1, 2, 3, 3, 3])
        unique = result.status_counts()
        assert unique == {"ok": 1, "invalid": 2, "rejected": 0, "undefined": 0, "error": 1}
        assert list(unique) == list(STATUSES)
        assert result.status_counts(per="call") == {"ok": 1, "invalid": 3, "rejected": 0, "undefined": 0, "error": 3}

    @pytest.mark.parametrize("per", ["calls", None], ids=["calls", "None"])
    def test_status_counts_rejects_bad_per(self, per):
        with pytest.raises(ValueError, match="per"):
            R([M(["X_0"], 0.5)], [0]).status_counts(per=per)


# ---------------------------------------------------------------------------------------------
# EvalResult: serialization
# ---------------------------------------------------------------------------------------------


def _summary_result():
    models = [M(["X_0"], 0.5), F(), M(["C", "*", "X_0"], 0.05)]
    return R(models, [0, 1, 1, 2, 2], success_threshold=0.1)


class TestEvalResultSerialization:
    def test_eval_result_json_round_trip_restores_every_field(self):
        model = M(
            ["C", "*", "X_0"],
            0.05,
            parameters=np.array([2.0, 3.0], dtype=np.float32),
            augmentations={"latex": {"value": "C X_0", "_type": "LaTeXAugmenter"}},
        )
        result = EvalResult(
            all_models=[M(["X_0"], 0.5), model, F()],
            trace=[0, 1, 2, 1],
            approach_name="MyApproach",
            success_threshold=0.1,
            top_k=5,
            dataset_name="ds",
            metadata={"seed": 3, "model": "m", "dataset_metadata": {"n": [1, 2]}},
            wall_time=1.5,
            augmentations={"aug": {"t": (1, "a"), 7: [1.0, INF], "_type": "Aug"}},
        )
        back = json_round_trip(result)
        assert [m.expr for m in back.all_models] == [["X_0"], ["C", "*", "X_0"], ["X_0", "X_1"]]
        assert [m.error for m in back.all_models] == [0.5, 0.05, INF]
        assert [m.status for m in back.all_models] == ["ok", "ok", "invalid"]
        assert back.all_models[1].parameters.dtype == np.float32
        np.testing.assert_array_equal(back.all_models[1].parameters, [2.0, 3.0])
        assert back.all_models[1].augmentations == {"latex": {"value": "C X_0", "_type": "LaTeXAugmenter"}}
        assert back.trace == [0, 1, 2, 1]
        assert back.approach_name == "MyApproach"
        assert back.success_threshold == 0.1
        assert back.top_k == 5
        assert back.dataset_name == "ds"
        assert back.metadata == {"seed": 3, "model": "m", "dataset_metadata": {"n": [1, 2]}}
        assert back.wall_time == 1.5
        assert back.augmentations == {"aug": {"t": (1, "a"), 7: [1.0, INF], "_type": "Aug"}}
        assert type(back.augmentations["aug"]["t"]) is tuple

    def test_eval_result_metadata_is_encoded(self):
        with pytest.raises(TypeError, match="set"):
            R([M(["X_0"], 0.1)], [0], metadata={"m": {1}}).to_dict()
        result = R([M(["X_0"], 0.1)], [0], metadata={"seed": 3, "bounds": (0, 1), "X": np.arange(3)})
        back = json_round_trip(result)
        assert back.metadata["seed"] == 3
        assert type(back.metadata["bounds"]) is tuple
        assert back.metadata["bounds"] == (0, 1)
        assert back.metadata["X"].dtype == np.arange(3).dtype
        np.testing.assert_array_equal(back.metadata["X"], [0, 1, 2])

    @pytest.mark.parametrize(
        "kwargs, get, expected_type, expected",
        [
            ({"trace": [np.int64(0)]}, lambda r: r.trace[0], int, 0),
            ({"top_k": np.int64(5)}, lambda r: r.top_k, int, 5),
            ({"success_threshold": np.float64(0.1)}, lambda r: r.success_threshold, float, 0.1),
            ({"wall_time": np.float32(1.5)}, lambda r: r.wall_time, float, 1.5),
        ],
        ids=["trace", "top_k", "success_threshold", "wall_time"],
    )
    def test_eval_result_numpy_scalars_come_back_as_python_scalars(self, kwargs, get, expected_type, expected):
        result = EvalResult(**{"all_models": [M(["X_0"], 0.1)], "trace": [0], "approach_name": "", **kwargs})
        value = get(json_round_trip(result))
        assert type(value) is expected_type
        assert value == expected

    def test_eval_result_to_dict_contains_version_and_summaries(self):
        d = _summary_result().to_dict()
        json.dumps(d)
        assert d["format_version"] == 2
        assert d["min_error"] == 0.05
        assert d["best_expr"] == "C*X_0"
        assert d["success"] is True
        assert d["num_evaluated"] == 3
        assert d["evaluation_calls"] == 5
        assert d["status_counts"] == {"ok": 2, "invalid": 1, "rejected": 0, "undefined": 0, "error": 0}

    def test_eval_result_from_dict_ignores_saved_summaries_and_unknown_keys(self):
        d = json.loads(json.dumps(_summary_result().to_dict()))
        d["min_error"] = 99.0
        d["best_expr"] = "X"
        d["success"] = False
        d["num_evaluated"] = 0
        d["unknown_key"] = 1
        d["all_models"][0]["unknown_key"] = 1
        back = EvalResult.from_dict(d)
        assert back.min_error == 0.05
        assert back.best_expr == "C*X_0"
        assert back.success is True
        assert back.num_evaluated == 3
        assert back.all_models[0].expr == ["X_0"]

    def test_eval_result_without_models_round_trips(self):
        result = R([], [], approach_name="A")
        text = json.dumps(result.to_dict())
        assert "NaN" in text
        back = EvalResult.from_dict(json.loads(text))
        assert back.all_models == []
        assert back.trace == []
        assert math.isnan(back.min_error)

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda d: d.pop("format_version"), "evaluation trace"),
            (lambda d: d.update(format_version=1), "format_version"),
            (lambda d: d.update(format_version=3), "format_version"),
            (lambda d: d["all_models"][0].pop("status"), "status"),
        ],
        ids=["version-missing", "version-1", "version-3", "model-without-status"],
    )
    def test_eval_result_from_dict_rejects(self, mutate, match):
        d = json.loads(json.dumps(_summary_result().to_dict()))
        mutate(d)
        with pytest.raises(ValueError, match=match):
            EvalResult.from_dict(d)
