import contextlib
import json
import os
import resource
import signal
import tempfile
import warnings

import numpy as np
import pytest
from strategies import result_dict

from SRToolkit.evaluation.results import ResultAugmenter, SR_results
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EvalResult

# ── Shared data ────────────────────────────────────────────────────────────────

_X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
_y = np.array([3.0, 0.0, 3.0, 11.0])  # y = 2*X_1 - X_0; optimal: ["C","*","X_1","-","X_0"]


# ── Helpers ───────────────────────────────────────────────────────────────────


class _TestAugmenter(ResultAugmenter):
    def write_results(self, results: EvalResult) -> None:
        results.add_augmentation(self.name, {"value": 42}, self._type)

    def to_dict(self, base_path: str, name: str) -> dict:
        return {"name": self.name}


class _FailingAugmenter(ResultAugmenter):
    def write_results(self, results: EvalResult) -> None:
        raise RuntimeError("intentional failure")

    def to_dict(self, base_path: str, name: str) -> dict:
        return {}


def _make_se(**kwargs):
    defaults = dict(seed=42, success_threshold=-1)
    defaults.update(kwargs)
    return SR_evaluator(_X, _y, **defaults)


def _single_result(approach_name="", top_k=5):
    se = _make_se()
    se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
    return se.get_results(approach_name=approach_name, top_k=top_k)


def _two_result(approach_name="", top_k=5):
    se = _make_se()
    se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
    se.evaluate_expr(["X_0", "+", "X_1"])
    return se.get_results(approach_name=approach_name, top_k=top_k)


# ── ResultAugmenter ────────────────────────────────────────────────────────────


class TestResultAugmenterSubclass:
    def test_type_attribute_set_to_qualified_name(self):
        assert hasattr(_TestAugmenter, "_type")
        assert _TestAugmenter._type.endswith("_TestAugmenter")
        assert "." in _TestAugmenter._type


class TestResultAugmenterInit:
    def test_name_stored(self):
        aug = _TestAugmenter("my_aug")
        assert aug.name == "my_aug"


class TestResultAugmenterFormatEvalResult:
    def test_formats_keys_excluding_type(self):
        result = ResultAugmenter.format_eval_result({"_type": "T", "key": "val"})
        assert "key: val" in result
        assert "_type" not in result

    def test_empty_after_excluding_type(self):
        result = ResultAugmenter.format_eval_result({"_type": "T"})
        assert result == ""


class TestResultAugmenterFormatModelResult:
    def test_formats_as_comma_joined_kv(self):
        result = ResultAugmenter.format_model_result({"_type": "T", "a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result

    def test_excludes_type_key(self):
        result = ResultAugmenter.format_model_result({"_type": "T", "x": 9})
        assert "_type" not in result


class TestResultAugmenterFromDict:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ResultAugmenter.from_dict({})


# ── SR_results.__init__ ────────────────────────────────────────────────────────


class TestSRResultsInit:
    def test_empty_results_list(self):
        assert SR_results().results == []


# ── SR_results.add_results ────────────────────────────────────────────────────


class TestSRResultsAddResults:
    def test_best_is_lowest_error(self):
        se = _make_se()
        se.evaluate_expr(["X_0", "+", "X_1"])  # worse
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])  # better
        r = se.get_results()[0]
        assert r.min_error < 1e-6

    def test_success_true_when_below_threshold(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=1.0)
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        assert se.get_results()[0].success is True

    def test_success_false_when_above_threshold(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=1e-20)
        se.evaluate_expr(["X_0", "+", "X_1"])
        assert se.get_results()[0].success is False

    def test_success_false_when_threshold_none(self):
        se = SR_evaluator(_X, _y, seed=42, success_threshold=-1)
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        # success_threshold=-1 (< 0) means threshold check never passes
        r = se.get_results()[0]
        assert r.success is False

    def test_dataset_name_extracted_from_metadata(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "ds", "extra": 1}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name == "ds"
        assert r.metadata == {"extra": 1}

    def test_metadata_without_dataset_name_fully_preserved(self):
        se = SR_evaluator(_X, _y, metadata={"source": "feynman"}, seed=0)
        se.evaluate_expr(["X_0"])
        r = se.get_results()[0]
        assert r.dataset_name is None
        assert r.metadata == {"source": "feynman"}

    def test_only_dataset_name_in_metadata_gives_none_remaining(self):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "only"}, seed=0)
        se.evaluate_expr(["X_0"])
        assert se.get_results()[0].metadata is None


# ── SR_results.print_results ──────────────────────────────────────────────────


class TestSRResultsPrintResults:
    def test_prints_header_for_all_experiments(self, capsys):
        results = _single_result()
        results.print_results()
        assert "=== Experiment 1/1 ===" in capsys.readouterr().out

    def test_prints_specific_experiment_no_header(self, capsys):
        results = _single_result()
        results.print_results(experiment_number=0)
        out = capsys.readouterr().out
        assert "===" not in out
        assert "Best expression" in out

    def test_out_of_bounds_experiment_number_raises(self):
        results = _single_result()
        with pytest.raises(IndexError):
            results.print_results(experiment_number=99)

    def test_negative_experiment_number_prints_from_end(self, capsys):
        results = _single_result()
        results.print_results(experiment_number=-1)
        assert "Best expression" in capsys.readouterr().out

    def test_prints_dataset_and_approach_when_set(self, capsys):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "MyDS"}, seed=0)
        se.evaluate_expr(["X_0"])
        se.get_results(approach_name="MyApproach").print_results()
        out = capsys.readouterr().out
        assert "MyDS" in out
        assert "MyApproach" in out

    def test_prints_stop_reason_only_when_set(self, capsys):
        results = _single_result()
        results.print_results(experiment_number=0)
        assert "Stopped: " in capsys.readouterr().out
        results[0].stop_reason = None
        results.print_results(experiment_number=0)
        assert "Stopped" not in capsys.readouterr().out

    def test_prints_metadata_block(self, capsys):
        se = SR_evaluator(_X, _y, metadata={"source": "feynman"}, seed=0)
        se.evaluate_expr(["X_0"])
        se.get_results().print_results()
        assert "source: feynman" in capsys.readouterr().out

    def test_detailed_shows_models_section(self, capsys):
        results = _single_result()
        results.print_results(detailed=True)
        assert "Models:" in capsys.readouterr().out

    def test_model_scope_best_shows_one_model(self, capsys):
        results = _two_result(top_k=2)
        results.print_results(detailed=True, model_scope="best")
        out = capsys.readouterr().out
        # Only the best model expression should appear in the Models section
        lines = [line for line in out.splitlines() if line.startswith("  ") and "error=" in line]
        assert len(lines) == 1

    def test_model_scope_all_shows_all_models(self, capsys):
        results = _two_result(top_k=1)
        results.print_results(detailed=True, model_scope="all")
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.startswith("  ") and "error=" in line]
        assert len(lines) == 2

    def test_eval_augmentation_printed_fallback(self, capsys):
        # augmentation with unresolvable _type → k/v fallback printing
        results = _single_result()
        results[0].add_augmentation("mykey", {"info": "hello"}, "NonExistent.Class")
        results.print_results()
        out = capsys.readouterr().out
        assert "mykey" in out
        assert "hello" in out

    def test_augmentation_with_non_string_type_printed_fallback(self, capsys):
        results = _single_result()
        results[0].augmentations["mykey"] = {"_type": None, "info": "hello"}
        results.print_results()
        assert "hello" in capsys.readouterr().out

    def test_model_augmentation_printed_fallback(self, capsys):
        results = _single_result()
        results[0].top_models()[0].add_augmentation("aug", {"x": 5}, "NonExistent.Class")
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "x=5" in out

    def test_model_augmentation_only_type_not_printed(self, capsys):
        # When augmentation data has only _type, parts=[] so nothing printed for that aug
        results = _single_result()
        results[0].top_models()[0].add_augmentation("empty_aug", {}, "NonExistent.Class")
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "empty_aug" not in out

    def test_augmentations_filter(self, capsys):
        results = _single_result()
        results[0].add_augmentation("keep", {"val": 1}, "NonExistent.Class")
        results[0].add_augmentation("skip", {"val": 2}, "NonExistent.Class")
        results.print_results(augmentations=["keep"])
        out = capsys.readouterr().out
        assert "keep" in out
        assert "skip" not in out

    def test_model_augmentation_resolved_prints_line(self, capsys):
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.print_results(detailed=True)
        capsys.readouterr()  # discard

    def test_model_augmentation_filter_applied(self, capsys):
        # Covers line 984: augmentations_filter is applied to model-level augmentation keys
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        results = se.get_results(top_k=1)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.print_results(detailed=True, augmentations=["HoldoutMetric"])
        out = capsys.readouterr().out
        assert "HoldoutMetric" in out

    def test_augmentation_resolved_to_result_augmenter_class(self, capsys):
        # Use a real augmenter whose _type is importable
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.print_results()
        out = capsys.readouterr().out
        assert "ExpressionToLatex" in out

    def test_model_augmentation_resolved_to_result_augmenter_class(self, capsys):
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.print_results(detailed=True)
        out = capsys.readouterr().out
        assert "ExpressionToLatex" in out


# ── SR_results.augment ────────────────────────────────────────────────────────


class TestSRResultsAugment:
    def test_augment_all_experiments(self):
        se1 = _make_se()
        se1.evaluate_expr(["X_0"])
        se2 = _make_se()
        se2.evaluate_expr(["X_1"])
        combined = se1.get_results() + se2.get_results()
        combined.augment([_TestAugmenter("aug")])
        assert "aug" in combined[0].augmentations
        assert "aug" in combined[1].augmentations

    @pytest.mark.parametrize("number, augmented", [(0, [True, False]), (-1, [False, True])])
    def test_augment_specific_experiment(self, number, augmented):
        combined = _single_result() + _single_result()
        combined.augment([_TestAugmenter("aug")], experiment_number=number)
        assert ["aug" in r.augmentations for r in combined] == augmented

    def test_augment_experiment_out_of_range(self):
        with pytest.raises(IndexError):
            _single_result().augment([_TestAugmenter("aug")], experiment_number=1)

    def test_augment_single_augmenter_not_in_list(self):
        results = _single_result()
        results.augment(_TestAugmenter("aug"))
        assert "aug" in results[0].augmentations

    def test_augment_specific_experiment_exception_warns(self):
        results = _single_result()
        with pytest.warns(UserWarning, match="intentional failure"):
            results.augment([_FailingAugmenter("bad")], experiment_number=0)

    def test_augment_exception_warns_and_continues(self):
        combined = _single_result() + _single_result()
        with pytest.warns(UserWarning, match="intentional failure"):
            combined.augment([_FailingAugmenter("bad"), _TestAugmenter("good")])
        assert "good" in combined[0].augmentations
        assert "good" in combined[1].augmentations


# ── SR_results container ops ──────────────────────────────────────────────────


class TestSRResultsContainerOps:
    def test_for_loop_visits_every_result(self):
        combined = _single_result() + _single_result()
        seen = [r for r in combined]
        assert len(seen) == 2

    def test_iteration_matches_indexing(self):
        combined = _single_result() + _single_result()
        for i, r in enumerate(combined):
            assert r is combined[i]

    def test_iteration_yields_eval_results(self):
        combined = _single_result() + _single_result()
        assert all(isinstance(r, EvalResult) for r in combined)

    def test_iteration_over_empty_results(self):
        assert list(SR_results()) == []

    def test_can_iterate_more_than_once(self):
        combined = _single_result() + _single_result()
        assert len(list(combined)) == len(list(combined)) == 2

    def test_len(self):
        assert len(_single_result()) == 1

    def test_getitem_returns_eval_result(self):
        assert isinstance(_single_result()[0], EvalResult)

    def test_getitem_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            _single_result()[99]

    def test_getitem_negative_index(self):
        results = _single_result() + _single_result()
        assert results[-1] is results.results[1]
        assert results[-2] is results.results[0]

    def test_getitem_negative_out_of_bounds_raises(self):
        with pytest.raises(IndexError):
            _single_result()[-2]

    def test_getitem_non_int_raises(self):
        with pytest.raises(TypeError):
            _single_result()["0"]

    def test_add_creates_new_object_without_mutating_operands(self):
        r1 = _single_result()
        r2 = _single_result()
        r3 = r1 + r2
        assert len(r1) == 1
        assert len(r3) == 2
        assert r3 is not r1

    def test_iadd_modifies_in_place(self):
        r1 = _single_result()
        r2 = _single_result()
        r1 += r2
        assert len(r1) == 2


# ── SR_results.save / load ────────────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore")
class TestSRResultsSaveLoad:
    def test_save_to_explicit_json_file(self, tmp_path):
        results = _single_result()
        path = str(tmp_path / "out.json")
        results.save(path)
        assert os.path.isfile(path)

    def test_saved_file_has_a_normal_files_mode(self, tmp_path):
        umask = os.umask(0o022)
        try:
            _single_result().save(str(tmp_path / "out.json"))
            with open(tmp_path / "plain", "w"):
                pass
        finally:
            os.umask(umask)
        assert os.stat(tmp_path / "out.json").st_mode == os.stat(tmp_path / "plain").st_mode

    def test_save_to_existing_directory(self, tmp_path):
        results = _single_result()
        results.save(str(tmp_path))
        assert os.path.isfile(tmp_path / "results.json")

    def test_save_extensionless_path_treated_as_directory(self, tmp_path):
        results = _single_result()
        target = str(tmp_path / "mydir")
        results.save(target)
        assert os.path.isfile(os.path.join(target, "results.json"))

    def test_save_invalid_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file extension"):
            _single_result().save(str(tmp_path / "out.txt"))

    def test_save_creates_nested_directories(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "c")
        _single_result().save(path)
        assert os.path.isfile(os.path.join(path, "results.json"))

    def test_load_from_file(self, tmp_path):
        results = _single_result()
        path = str(tmp_path / "r.json")
        results.save(path)
        loaded = SR_results.load(path)
        assert len(loaded) == 1

    def test_load_from_directory(self, tmp_path):
        _single_result().save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert len(loaded) == 1

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SR_results.load(str(tmp_path / "missing.json"))

    def test_load_invalid_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid file extension"):
            SR_results.load(str(tmp_path / "out.txt"))

    def test_load_invalid_format_version_raises(self, tmp_path):
        _single_result().save(str(tmp_path))
        path = str(tmp_path / "results.json")
        with open(path) as f:
            data = json.load(f)
        data["format_version"] = 99
        with open(path, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="Unsupported format_version"):
            SR_results.load(str(tmp_path))

    def test_save_load_round_trip(self, tmp_path):
        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        results = se.get_results(top_k=2)
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert loaded[0].best_expr == results[0].best_expr
        assert abs(loaded[0].min_error - results[0].min_error) < 1e-12
        assert loaded[0].num_evaluated == results[0].num_evaluated
        assert len(loaded[0].top_models()) == len(results[0].top_models())
        np.testing.assert_array_almost_equal(
            loaded[0].top_models()[0].parameters, results[0].top_models()[0].parameters
        )

    def test_save_load_multiple_experiments(self, tmp_path):
        r1 = _single_result()
        r2 = _single_result()
        combined = r1 + r2
        combined.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert len(loaded) == 2
        assert loaded[0].best_expr == combined[0].best_expr
        assert loaded[1].best_expr == combined[1].best_expr

    def test_save_load_with_metadata(self, tmp_path):
        se = SR_evaluator(_X, _y, metadata={"dataset_name": "ds", "fold": 3}, seed=42)
        se.evaluate_expr(["X_0"])
        results = se.get_results(approach_name="Ap")
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert loaded[0].dataset_name == "ds"
        assert loaded[0].metadata == {"fold": 3}
        assert loaded[0].approach_name == "Ap"

    def test_save_load_with_augmentations(self, tmp_path):
        from SRToolkit.evaluation.result_augmentation import ExpressionToLatex

        sl = SymbolLibrary.default_symbols(num_variables=2)
        results = _single_result()
        results.augment([ExpressionToLatex(sl)])
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        assert "ExpressionToLatex" in loaded[0].augmentations
        orig = results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]
        assert loaded[0].augmentations["ExpressionToLatex"]["best_expr_latex"] == orig

    def test_save_load_holdout_augmentation_round_trip(self, tmp_path):
        from SRToolkit.evaluation.result_augmentation import HoldoutMetric

        se = _make_se()
        se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
        se.evaluate_expr(["X_0", "+", "X_1"])
        results = se.get_results(top_k=2)
        results.augment([HoldoutMetric((se.X, se.y), metric="rmse", symbol_library=se.symbol_library)])
        results.save(str(tmp_path))
        loaded = SR_results.load(str(tmp_path))
        orig_aug = results[0].augmentations["HoldoutMetric"]
        rest_aug = loaded[0].augmentations["HoldoutMetric"]
        assert rest_aug["_type"] == "SRToolkit.evaluation.result_augmentation.HoldoutMetric"
        assert abs(rest_aug["best_value"] - orig_aug["best_value"]) < 1e-12
        # model-level values survive the JSON round-trip
        orig_value = results[0].top_models()[0].augmentations["HoldoutMetric"]["value"]
        rest_value = loaded[0].top_models()[0].augmentations["HoldoutMetric"]["value"]
        assert abs(rest_value - orig_value) < 1e-12


# ── from_checkpoint ───────────────────────────────────────────────────────────

_CALLS = [["X_0"], ["C", "*", "X_1"], ["X_0"], ["C", "*", "X_1", "-", "X_0"]]


def _run(checkpoint, budget, calls):
    se = SR_evaluator(
        _X,
        _y,
        seed=0,
        success_threshold=-1,
        max_evaluations=budget,
        checkpoint=checkpoint,
        metadata={"dataset_name": "d", "k": 1},
        initialization="mean",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the call after the budget is used up
        for expr in calls:
            se.evaluate_expr(expr)
    return se


class TestFromCheckpoint:
    @pytest.mark.parametrize(
        "budget, finish, stop_reason",
        [
            (3, lambda se: (se.get_results(), se.close()), "max_evaluations"),
            (3, lambda se: None, "max_evaluations"),  # writer still open; the stop committed
            (-1, lambda se: se.get_results(), "interrupted"),  # results taken during the run do not finish it
            (-1, lambda se: se.close(), "interrupted"),  # no reason saved
        ],
        ids=["stopped-closed", "stopped-open", "results-mid-run", "unstopped-closed"],
    )
    def test_from_checkpoint_equals_get_results(self, tmp_path, budget, finish, stop_reason):
        path = tmp_path / "run.sqlite"
        finish(_run(path, budget, _CALLS))
        twin = result_dict(_run(None, budget, _CALLS).get_results("A", 2)[0])
        results = SR_results.from_checkpoint(path, "A", 2)
        assert len(results) == 1
        assert result_dict(results[0]) == {**twin, "stop_reason": stop_reason}

    def test_from_checkpoint_sees_committed_calls_without_blocking_writer(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = _run(path, -1, _CALLS[:3])
        se.get_results()  # commits
        se.evaluate_expr(["X_1"])
        se.evaluate_expr(["X_0"])  # both pending
        assert SR_results.from_checkpoint(path)[0].trace == [0, 1, 0]
        se.get_results()  # the writer still commits
        assert SR_results.from_checkpoint(path)[0].trace == [0, 1, 0, 2, 0]

    @pytest.mark.parametrize(
        "make_file, error, match",
        [(None, FileNotFoundError, None), (lambda path: path.write_text("keep"), ValueError, "checkpoint")],
        ids=["missing", "not-a-checkpoint"],
    )
    def test_from_checkpoint_errors(self, tmp_path, make_file, error, match):
        path = tmp_path / "run.sqlite"
        if make_file is not None:
            make_file(path)
        with pytest.raises(error, match=match):
            SR_results.from_checkpoint(path)


# ── Atomic save ───────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _file_size_limit(limit):
    # A disk that fills up: writes past ``limit`` bytes raise OSError (EFBIG).
    old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    old_handler = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, old_limit[1]))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, old_limit)
        signal.signal(signal.SIGXFSZ, old_handler)


def _unsaveable(result):
    result.add_augmentation("broken", {"value": object()}, "test")


def _too_big(result):
    result.add_augmentation("big", {"value": "x" * 100_000}, "test")  # past the 20 kB limit below


@pytest.mark.parametrize(
    "previous, spoil, error",
    [(True, _unsaveable, TypeError), (True, _too_big, OSError), (False, _too_big, OSError)],
    ids=["unsaveable-data", "disk-full", "disk-full-no-previous-file"],
)
def test_failed_save_keeps_previous_file(tmp_path, previous, spoil, error):
    path = tmp_path / "results.json"
    se = SR_evaluator(_X, _y, seed=0)
    se.evaluate_expr(["X_0"])
    if previous:
        se.get_results().save(str(path))
    before = path.read_bytes() if previous else None
    bad = se.get_results()
    spoil(bad[0])
    with _file_size_limit(20_000), pytest.raises(error):
        bad.save(str(path))
    assert (path.read_bytes() if path.exists() else None) == before  # written whole or not at all
    assert sorted(p.name for p in tmp_path.iterdir()) == (["results.json"] if previous else [])  # no temp file


@pytest.mark.parametrize("relative", [False, True], ids=["other-directory", "bare-file-name"])
def test_save_writes_its_temporary_file_next_to_the_target(tmp_path, monkeypatch, relative):
    # "the JSON goes to a temporary file next to it": saving works although the working directory
    # and the system temp directory are read-only.
    locked, out = tmp_path / "locked", tmp_path / "out"
    locked.mkdir()
    out.mkdir()
    locked.chmod(0o500)
    monkeypatch.setattr(tempfile, "tempdir", str(locked))
    monkeypatch.chdir(out if relative else locked)
    se = SR_evaluator(_X, _y, seed=0)
    se.evaluate_expr(["X_0"])
    try:
        se.get_results().save("results.json" if relative else str(out / "results.json"))
    finally:
        locked.chmod(0o700)
    assert sorted(os.listdir(out)) == ["results.json"]
    assert SR_results.load(str(out / "results.json"))[0].trace == [0]
