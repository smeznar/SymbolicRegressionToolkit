import os
from unittest.mock import MagicMock, patch

import pytest

from SRToolkit.evaluation.callbacks import (
    BestExpressionFound,
    CallbackDispatcher,
    EarlyStoppingCallback,
    ExperimentEvent,
    ExprEvaluated,
    LoggingCallback,
    ProgressBarCallback,
    SRCallbacks,
)
from SRToolkit.utils.types import EvalResult, ModelResult

# ── Helpers ────────────────────────────────────────────────────────────────────


class _ConcreteCallback(SRCallbacks):
    """Minimal concrete subclass used to instantiate the abstract base."""

    pass


def _make_expr_evaluated(**kwargs):
    defaults = dict(expression="X_0", error=0.5, evaluation_number=1, experiment_id="exp", is_new_best=False)
    defaults.update(kwargs)
    return ExprEvaluated(**defaults)


def _make_best_found(**kwargs):
    defaults = dict(experiment_id="exp", expression="X_0", error=0.5, evaluation_number=1)
    defaults.update(kwargs)
    return BestExpressionFound(**defaults)


def _make_experiment_event(**kwargs):
    defaults = dict(dataset_name="ds", approach_name="ap", max_evaluations=100, success_threshold=1e-4, seed=42)
    defaults.update(kwargs)
    return ExperimentEvent(**defaults)


def _make_eval_result():
    model = ModelResult(expr=["X_0"], error=0.05)
    return EvalResult(
        min_error=0.05,
        best_expr="X_0",
        num_evaluated=100,
        evaluation_calls=120,
        top_models=[model],
        all_models=[model],
        approach_name="ap",
        success=True,
    )


# ── ExprEvaluated ──────────────────────────────────────────────────────────────


class TestExprEvaluatedDataclass:
    def test_fields_stored_correctly(self):
        e = ExprEvaluated(expression="X_0+C", error=0.25, evaluation_number=7, experiment_id="exp1", is_new_best=True)
        assert e.expression == "X_0+C"
        assert e.error == 0.25
        assert e.evaluation_number == 7
        assert e.experiment_id == "exp1"
        assert e.is_new_best is True


# ── BestExpressionFound ────────────────────────────────────────────────────────


class TestBestExpressionFoundDataclass:
    def test_fields_stored_correctly(self):
        b = BestExpressionFound(experiment_id="e1", expression="sin(X_0)", error=0.01, evaluation_number=42)
        assert b.experiment_id == "e1"
        assert b.expression == "sin(X_0)"
        assert b.error == 0.01
        assert b.evaluation_number == 42


# ── ExperimentEvent ────────────────────────────────────────────────────────────


class TestExperimentEventDataclass:
    def test_fields_stored_correctly(self):
        ev = ExperimentEvent(
            dataset_name="Feynman", approach_name="ProGED", max_evaluations=500, success_threshold=1e-6, seed=0
        )
        assert ev.dataset_name == "Feynman"
        assert ev.approach_name == "ProGED"
        assert ev.max_evaluations == 500
        assert ev.success_threshold == 1e-6
        assert ev.seed == 0

    def test_optional_fields_accept_none(self):
        ev = ExperimentEvent(
            dataset_name="ds", approach_name="ap", max_evaluations=None, success_threshold=None, seed=None
        )
        assert ev.max_evaluations is None
        assert ev.success_threshold is None
        assert ev.seed is None


# ── SRCallbacks defaults ───────────────────────────────────────────────────────


class TestSRCallbacksDefaults:
    def test_on_expr_evaluated_returns_none(self):
        cb = _ConcreteCallback()
        assert cb.on_expr_evaluated(_make_expr_evaluated()) is None

    def test_on_best_expression_returns_none(self):
        cb = _ConcreteCallback()
        assert cb.on_best_expression(_make_best_found()) is None

    def test_on_experiment_start_returns_none(self):
        cb = _ConcreteCallback()
        assert cb.on_experiment_start(_make_experiment_event()) is None

    def test_on_experiment_end_returns_none(self):
        cb = _ConcreteCallback()
        assert cb.on_experiment_end(_make_experiment_event(), _make_eval_result()) is None

    def test_to_dict_contains_callback_class_key(self):
        cb = _ConcreteCallback()
        d = cb.to_dict()
        assert "callback_class" in d
        assert d["callback_class"].endswith("_ConcreteCallback")

    def test_from_dict_returns_instance(self):
        cb = _ConcreteCallback.from_dict({})
        assert isinstance(cb, _ConcreteCallback)


# ── CallbackDispatcher – init and basic accessors ──────────────────────────────


class TestCallbackDispatcherInit:
    def test_default_gives_empty_list(self):
        d1 = CallbackDispatcher()
        d2 = CallbackDispatcher()
        d1._callbacks.append(_ConcreteCallback())
        assert d2._callbacks == []

    def test_provided_list_is_stored(self):
        cbs = [_ConcreteCallback()]
        dispatcher = CallbackDispatcher(callbacks=cbs)
        assert dispatcher._callbacks is cbs


class TestCallbackDispatcherGetCallbacks:
    def test_returns_same_list(self):
        cbs = [_ConcreteCallback()]
        dispatcher = CallbackDispatcher(callbacks=cbs)
        assert dispatcher.get_callbacks() is cbs


class TestCallbackDispatcherAdd:
    def test_add_single_callback(self):
        dispatcher = CallbackDispatcher()
        cb = _ConcreteCallback()
        dispatcher.add(cb)
        assert dispatcher._callbacks == [cb]

    def test_add_multiple_callbacks_preserves_order(self):
        dispatcher = CallbackDispatcher()
        cb1, cb2 = _ConcreteCallback(), _ConcreteCallback()
        dispatcher.add(cb1)
        dispatcher.add(cb2)
        assert dispatcher._callbacks == [cb1, cb2]


class TestCallbackDispatcherRemove:
    def test_remove_registered_callback(self):
        cb = _ConcreteCallback()
        dispatcher = CallbackDispatcher(callbacks=[cb])
        dispatcher.remove(cb)
        assert dispatcher._callbacks == []

    def test_remove_nonexistent_raises_value_error(self):
        dispatcher = CallbackDispatcher()
        with pytest.raises(ValueError):
            dispatcher.remove(_ConcreteCallback())


# ── CallbackDispatcher – on_expr_evaluated ─────────────────────────────────────


class TestCallbackDispatcherOnExprEvaluated:
    def test_empty_callbacks_returns_true(self):
        assert CallbackDispatcher().on_expr_evaluated(_make_expr_evaluated()) is True

    def test_callback_returning_none_continues(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = None
        assert CallbackDispatcher(callbacks=[cb]).on_expr_evaluated(_make_expr_evaluated()) is True

    def test_callback_returning_true_continues(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = True
        assert CallbackDispatcher(callbacks=[cb]).on_expr_evaluated(_make_expr_evaluated()) is True

    def test_callback_returning_false_stops(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_expr_evaluated.return_value = False
        assert CallbackDispatcher(callbacks=[cb]).on_expr_evaluated(_make_expr_evaluated()) is False

    def test_all_callbacks_called_even_after_one_stops(self):
        cb1 = MagicMock(spec=SRCallbacks)
        cb1.on_expr_evaluated.return_value = False
        cb2 = MagicMock(spec=SRCallbacks)
        cb2.on_expr_evaluated.return_value = None
        event = _make_expr_evaluated()
        CallbackDispatcher(callbacks=[cb1, cb2]).on_expr_evaluated(event)
        cb1.on_expr_evaluated.assert_called_once_with(event)
        cb2.on_expr_evaluated.assert_called_once_with(event)

    def test_one_stop_among_multiple_returns_false(self):
        cb_none = MagicMock(spec=SRCallbacks)
        cb_none.on_expr_evaluated.return_value = None
        cb_false = MagicMock(spec=SRCallbacks)
        cb_false.on_expr_evaluated.return_value = False
        cb_true = MagicMock(spec=SRCallbacks)
        cb_true.on_expr_evaluated.return_value = True
        result = CallbackDispatcher(callbacks=[cb_none, cb_false, cb_true]).on_expr_evaluated(_make_expr_evaluated())
        assert result is False


# ── CallbackDispatcher – on_best_expression ────────────────────────────────────


class TestCallbackDispatcherOnBestExpression:
    def test_empty_callbacks_returns_true(self):
        assert CallbackDispatcher().on_best_expression(_make_best_found()) is True

    def test_callback_returning_none_continues(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_best_expression.return_value = None
        assert CallbackDispatcher(callbacks=[cb]).on_best_expression(_make_best_found()) is True

    def test_callback_returning_true_continues(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_best_expression.return_value = True
        assert CallbackDispatcher(callbacks=[cb]).on_best_expression(_make_best_found()) is True

    def test_callback_returning_false_stops(self):
        cb = MagicMock(spec=SRCallbacks)
        cb.on_best_expression.return_value = False
        assert CallbackDispatcher(callbacks=[cb]).on_best_expression(_make_best_found()) is False

    def test_all_callbacks_called_even_after_one_stops(self):
        cb1 = MagicMock(spec=SRCallbacks)
        cb1.on_best_expression.return_value = False
        cb2 = MagicMock(spec=SRCallbacks)
        cb2.on_best_expression.return_value = None
        event = _make_best_found()
        CallbackDispatcher(callbacks=[cb1, cb2]).on_best_expression(event)
        cb1.on_best_expression.assert_called_once_with(event)
        cb2.on_best_expression.assert_called_once_with(event)

    def test_one_stop_among_multiple_returns_false(self):
        cb_none = MagicMock(spec=SRCallbacks)
        cb_none.on_best_expression.return_value = None
        cb_false = MagicMock(spec=SRCallbacks)
        cb_false.on_best_expression.return_value = False
        cb_true = MagicMock(spec=SRCallbacks)
        cb_true.on_best_expression.return_value = True
        result = CallbackDispatcher(callbacks=[cb_none, cb_false, cb_true]).on_best_expression(_make_best_found())
        assert result is False


# ── CallbackDispatcher – on_experiment_start / end ────────────────────────────


class TestCallbackDispatcherOnExperimentStart:
    def test_dispatches_to_all_callbacks(self):
        cb1, cb2 = MagicMock(spec=SRCallbacks), MagicMock(spec=SRCallbacks)
        event = _make_experiment_event()
        CallbackDispatcher(callbacks=[cb1, cb2]).on_experiment_start(event)
        cb1.on_experiment_start.assert_called_once_with(event)
        cb2.on_experiment_start.assert_called_once_with(event)

    def test_empty_dispatcher_is_no_op(self):
        CallbackDispatcher().on_experiment_start(_make_experiment_event())


class TestCallbackDispatcherOnExperimentEnd:
    def test_dispatches_to_all_callbacks(self):
        cb1, cb2 = MagicMock(spec=SRCallbacks), MagicMock(spec=SRCallbacks)
        event = _make_experiment_event()
        result = _make_eval_result()
        CallbackDispatcher(callbacks=[cb1, cb2]).on_experiment_end(event, result)
        cb1.on_experiment_end.assert_called_once_with(event, result)
        cb2.on_experiment_end.assert_called_once_with(event, result)


# ── ProgressBarCallback ────────────────────────────────────────────────────────


class TestProgressBarCallbackInit:
    def test_default_no_desc(self):
        cb = ProgressBarCallback()
        assert cb.desc is None

    def test_custom_desc_stored(self):
        cb = ProgressBarCallback(desc="my search")
        assert cb.desc == "my search"

    def test_pbar_initially_none(self):
        assert ProgressBarCallback().pbar is None


class TestProgressBarCallbackOnExperimentStart:
    def test_creates_bounded_pbar_when_max_evaluations_set(self):
        cb = ProgressBarCallback()
        event = _make_experiment_event(max_evaluations=200)
        with patch("tqdm.tqdm") as mock_tqdm:
            cb.on_experiment_start(event)
            mock_tqdm.assert_called_once()
            _, kwargs = mock_tqdm.call_args
            assert kwargs["total"] == 200

    def test_creates_unbounded_pbar_when_max_evaluations_none(self):
        cb = ProgressBarCallback()
        event = _make_experiment_event(max_evaluations=None)
        with patch("tqdm.tqdm") as mock_tqdm:
            cb.on_experiment_start(event)
            mock_tqdm.assert_called_once()
            _, kwargs = mock_tqdm.call_args
            assert "total" not in kwargs

    def test_uses_custom_desc(self):
        cb = ProgressBarCallback(desc="custom")
        with patch("tqdm.tqdm") as mock_tqdm:
            cb.on_experiment_start(_make_experiment_event())
            _, kwargs = mock_tqdm.call_args
            assert kwargs["desc"] == "custom"

    def test_auto_generates_desc_from_event(self):
        cb = ProgressBarCallback()
        event = _make_experiment_event(approach_name="ProGED", dataset_name="Feynman")
        with patch("tqdm.tqdm") as mock_tqdm:
            cb.on_experiment_start(event)
            _, kwargs = mock_tqdm.call_args
            assert kwargs["desc"] == "ProGED on Feynman"


class TestProgressBarCallbackOnExprEvaluated:
    def test_updates_pbar_when_present(self):
        cb = ProgressBarCallback()
        cb.pbar = MagicMock()
        cb.on_expr_evaluated(_make_expr_evaluated())
        cb.pbar.update.assert_called_once_with(1)

    def test_no_error_when_pbar_none(self):
        cb = ProgressBarCallback()
        cb.on_expr_evaluated(_make_expr_evaluated())

    def test_returns_none(self):
        cb = ProgressBarCallback()
        assert cb.on_expr_evaluated(_make_expr_evaluated()) is None


class TestProgressBarCallbackOnExperimentEnd:
    def test_closes_pbar_and_sets_to_none(self):
        cb = ProgressBarCallback()
        mock_pbar = MagicMock()
        cb.pbar = mock_pbar
        cb.on_experiment_end(_make_experiment_event(), _make_eval_result())
        mock_pbar.close.assert_called_once()
        assert cb.pbar is None

    def test_no_error_when_pbar_none(self):
        cb = ProgressBarCallback()
        cb.on_experiment_end(_make_experiment_event(), _make_eval_result())


class TestProgressBarCallbackSerialization:
    def test_to_dict_includes_desc(self):
        d = ProgressBarCallback(desc="test").to_dict()
        assert d["desc"] == "test"

    def test_to_dict_none_desc(self):
        d = ProgressBarCallback().to_dict()
        assert "desc" in d
        assert d["desc"] is None

    def test_from_dict_with_desc(self):
        cb = ProgressBarCallback.from_dict({"desc": "hello"})
        assert cb.desc == "hello"

    def test_from_dict_without_desc_key_gives_none(self):
        cb = ProgressBarCallback.from_dict({})
        assert cb.desc is None


# ── EarlyStoppingCallback ──────────────────────────────────────────────────────


class TestEarlyStoppingCallbackInit:
    def test_stores_threshold_and_max_evaluations(self):
        cb = EarlyStoppingCallback(threshold=1e-4, max_evaluations=500)
        assert cb.threshold == 1e-4
        assert cb.max_evaluations == 500

    def test_max_evaluations_defaults_to_none(self):
        cb = EarlyStoppingCallback(threshold=1e-4)
        assert cb.max_evaluations is None


class TestEarlyStoppingCallbackOnExperimentStart:
    def test_threshold_set_from_event_when_self_is_none(self):
        cb = EarlyStoppingCallback(threshold=None)
        cb.on_experiment_start(_make_experiment_event(success_threshold=1e-6))
        assert cb.threshold == 1e-6

    def test_threshold_not_overridden_when_already_set(self):
        cb = EarlyStoppingCallback(threshold=1e-3)
        cb.on_experiment_start(_make_experiment_event(success_threshold=1e-6))
        assert cb.threshold == 1e-3

    def test_threshold_unchanged_when_event_threshold_also_none(self):
        cb = EarlyStoppingCallback(threshold=None)
        cb.on_experiment_start(_make_experiment_event(success_threshold=None))
        assert cb.threshold is None

    def test_max_evaluations_set_from_event(self):
        cb = EarlyStoppingCallback(threshold=None)
        cb.on_experiment_start(_make_experiment_event(max_evaluations=300))
        assert cb.max_evaluations == 300

    def test_max_evaluations_not_overridden_when_already_set(self):
        cb = EarlyStoppingCallback(threshold=None, max_evaluations=200)
        cb.on_experiment_start(_make_experiment_event(max_evaluations=300))
        assert cb.max_evaluations == 200

    def test_max_evaluations_not_set_when_event_value_is_zero(self):
        cb = EarlyStoppingCallback(threshold=None)
        cb.on_experiment_start(_make_experiment_event(max_evaluations=0))
        assert cb.max_evaluations is None


class TestEarlyStoppingCallbackOnExprEvaluated:
    def test_stops_when_evaluation_number_equals_max(self):
        cb = EarlyStoppingCallback(threshold=None, max_evaluations=10)
        assert cb.on_expr_evaluated(_make_expr_evaluated(evaluation_number=10)) is False

    def test_stops_when_evaluation_number_exceeds_max(self):
        cb = EarlyStoppingCallback(threshold=None, max_evaluations=10)
        assert cb.on_expr_evaluated(_make_expr_evaluated(evaluation_number=11)) is False

    def test_continues_when_below_max(self):
        cb = EarlyStoppingCallback(threshold=None, max_evaluations=10)
        assert cb.on_expr_evaluated(_make_expr_evaluated(evaluation_number=9)) is True

    def test_continues_when_max_evaluations_none(self):
        cb = EarlyStoppingCallback(threshold=None)
        assert cb.on_expr_evaluated(_make_expr_evaluated(evaluation_number=10000)) is True

    def test_negative_max_evaluations_never_stops(self):
        # Chained comparison: eval_number >= max_evaluations >= 0 is False when max < 0
        cb = EarlyStoppingCallback(threshold=None, max_evaluations=-1)
        assert cb.on_expr_evaluated(_make_expr_evaluated(evaluation_number=99999)) is True


class TestEarlyStoppingCallbackOnBestExpression:
    def test_stops_when_error_below_threshold(self):
        cb = EarlyStoppingCallback(threshold=1e-4)
        assert cb.on_best_expression(_make_best_found(error=1e-5)) is False

    def test_continues_when_error_equals_threshold(self):
        cb = EarlyStoppingCallback(threshold=1e-4)
        assert cb.on_best_expression(_make_best_found(error=1e-4)) is True

    def test_continues_when_error_above_threshold(self):
        cb = EarlyStoppingCallback(threshold=1e-4)
        assert cb.on_best_expression(_make_best_found(error=0.5)) is True

    def test_continues_when_threshold_none(self):
        cb = EarlyStoppingCallback(threshold=None)
        assert cb.on_best_expression(_make_best_found(error=0.0)) is True


class TestEarlyStoppingCallbackSerialization:
    def test_to_dict_contains_threshold_and_max_evaluations(self):
        d = EarlyStoppingCallback(threshold=1e-4, max_evaluations=500).to_dict()
        assert d["threshold"] == 1e-4
        assert d["max_evaluations"] == 500

    def test_from_dict_round_trip(self):
        cb = EarlyStoppingCallback(threshold=1e-4, max_evaluations=500)
        restored = EarlyStoppingCallback.from_dict(cb.to_dict())
        assert restored.threshold == 1e-4
        assert restored.max_evaluations == 500

    def test_from_dict_missing_keys_gives_none(self):
        cb = EarlyStoppingCallback.from_dict({})
        assert cb.threshold is None
        assert cb.max_evaluations is None


# ── LoggingCallback ────────────────────────────────────────────────────────────


class TestLoggingCallbackInit:
    def test_default_no_log_file(self):
        cb = LoggingCallback()
        assert cb.log_file is None
        assert cb._resolved_log_file is None

    def test_with_log_file_stores_both(self):
        cb = LoggingCallback(log_file="logs/out.log")
        assert cb.log_file == "logs/out.log"
        assert cb._resolved_log_file == "logs/out.log"


class TestLoggingCallbackOnExperimentStart:
    def test_resolves_all_placeholders(self):
        cb = LoggingCallback(log_file="logs/{dataset_name}_{approach_name}_{seed}.log")
        cb.on_experiment_start(_make_experiment_event(dataset_name="Feynman", approach_name="ProGED", seed=7))
        assert cb._resolved_log_file == "logs/Feynman_ProGED_7.log"

    def test_partial_placeholders(self):
        cb = LoggingCallback(log_file="logs/{dataset_name}.log")
        cb.on_experiment_start(_make_experiment_event(dataset_name="Nguyen", approach_name="ap", seed=1))
        assert cb._resolved_log_file == "logs/Nguyen.log"

    def test_sets_resolved_to_none_when_log_file_none(self):
        cb = LoggingCallback(log_file=None)
        cb._resolved_log_file = "stale/path.log"
        cb.on_experiment_start(_make_experiment_event())
        assert cb._resolved_log_file is None


class TestLoggingCallbackOnBestExpression:
    def test_prints_to_stdout_when_no_file(self, capsys):
        cb = LoggingCallback()
        cb.on_best_expression(
            BestExpressionFound(experiment_id="exp1", expression="X_0", error=0.001, evaluation_number=10)
        )
        captured = capsys.readouterr()
        assert "[Experiment exp1]" in captured.out
        assert "X_0" in captured.out

    def test_message_format(self, capsys):
        cb = LoggingCallback()
        cb.on_best_expression(BestExpressionFound(experiment_id="E", expression="C", error=1.5e-3, evaluation_number=5))
        captured = capsys.readouterr()
        assert captured.out == "[Experiment E] New best: C (error=1.500000e-03)\n"

    def test_writes_to_file(self, tmp_path):
        log_file = str(tmp_path / "out.log")
        cb = LoggingCallback(log_file=log_file)
        cb.on_best_expression(BestExpressionFound(experiment_id="E", expression="X_0", error=0.1, evaluation_number=1))
        content = open(log_file).read()
        assert "[Experiment E]" in content
        assert "X_0" in content

    def test_appends_to_existing_file(self, tmp_path):
        log_file = str(tmp_path / "out.log")
        cb = LoggingCallback(log_file=log_file)
        cb.on_best_expression(BestExpressionFound(experiment_id="E", expression="X_0", error=0.2, evaluation_number=1))
        cb.on_best_expression(BestExpressionFound(experiment_id="E", expression="X_1", error=0.1, evaluation_number=2))
        lines = open(log_file).readlines()
        assert len(lines) == 2
        assert "X_0" in lines[0]
        assert "X_1" in lines[1]

    def test_writes_to_file_when_flock_raises_os_error(self, tmp_path):
        log_file = str(tmp_path / "out.log")
        cb = LoggingCallback(log_file=log_file)
        with patch("fcntl.flock", side_effect=OSError):
            cb.on_best_expression(
                BestExpressionFound(experiment_id="E", expression="X_0", error=0.1, evaluation_number=1)
            )
        content = open(log_file).read()
        assert "[Experiment E]" in content

    def test_creates_parent_directories(self, tmp_path):
        log_file = str(tmp_path / "nested" / "dir" / "out.log")
        cb = LoggingCallback(log_file=log_file)
        cb.on_best_expression(BestExpressionFound(experiment_id="E", expression="X_0", error=0.1, evaluation_number=1))
        assert os.path.isfile(log_file)


class TestLoggingCallbackSerialization:
    def test_to_dict_with_log_file(self):
        d = LoggingCallback(log_file="logs/out.log").to_dict()
        assert d["log_file"] == "logs/out.log"

    def test_to_dict_without_log_file(self):
        d = LoggingCallback().to_dict()
        assert "log_file" in d
        assert d["log_file"] is None

    def test_from_dict_round_trip(self):
        cb = LoggingCallback(log_file="logs/{seed}.log")
        restored = LoggingCallback.from_dict(cb.to_dict())
        assert restored.log_file == "logs/{seed}.log"

    def test_from_dict_missing_key_gives_none(self):
        cb = LoggingCallback.from_dict({})
        assert cb.log_file is None
