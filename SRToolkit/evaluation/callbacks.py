"""Event-driven callback system for monitoring and controlling SR evaluation.

Provides event dataclasses fired during evaluation, the
[Callback][SRToolkit.evaluation.callbacks.Callback] base class for implementing custom callbacks,
a [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] for managing multiple callbacks,
and built-in implementations for progress display, early stopping, and logging.
"""

import os
import time
import warnings
from abc import ABC
from dataclasses import dataclass
from datetime import timedelta

from tqdm import tqdm

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.utils import EvalResult


@dataclass
class ExprEvaluated:
    """Fired after each expression is evaluated by
    [evaluate_expr][SRToolkit.evaluation.evaluator.Evaluator.evaluate_expr].

    Attributes:
        expression: String representation of the evaluated expression.
        error: Error value returned by the ranking function (RMSE or BED).
        evaluation_number: Total number of
            [evaluate_expr][SRToolkit.evaluation.evaluator.Evaluator.evaluate_expr]
            calls made so far, including cache hits.
        experiment_id: Identifier of the current experiment.
        is_new_best: ``True`` if this expression achieved a lower error than all previous ones.
        status: How the evaluation ended; see [Status][SRToolkit.utils.types.Status].
    """

    expression: str
    error: float
    evaluation_number: int
    experiment_id: str
    is_new_best: bool
    status: str = "ok"


@dataclass
class ExprSkipped:
    """Event fired after each skip counted by ``count_skip``.

    See [count_skip][SRToolkit.evaluation.evaluator.Evaluator.count_skip].

    Attributes:
        reason: The reason passed to ``count_skip``, e.g. ``"rejected"``.
        skips_in_a_row: Skips in a row, this one included (see
            [count_skip][SRToolkit.evaluation.evaluator.Evaluator.count_skip]).
        evaluation_number: Total number of
            [evaluate_expr][SRToolkit.evaluation.evaluator.Evaluator.evaluate_expr]
            calls made so far, including cache hits.
        experiment_id: Identifier of the current experiment.
    """

    reason: str
    skips_in_a_row: int
    evaluation_number: int
    experiment_id: str


@dataclass
class BestExpressionFound:
    """Fired when a new best expression is found during evaluation.

    Attributes:
        experiment_id: Identifier of the current experiment.
        expression: String representation of the new best expression.
        error: Error value of the new best expression.
        evaluation_number: Total number of
            [evaluate_expr][SRToolkit.evaluation.evaluator.Evaluator.evaluate_expr]
            calls made at the time this event is fired.
    """

    experiment_id: str
    expression: str
    error: float
    evaluation_number: int


@dataclass
class ExperimentEvent:
    """Fired at experiment start and end.

    Attributes:
        dataset_name: Name of the dataset being evaluated.
        approach_name: Name of the SR approach being run.
        max_evaluations: Maximum number of evaluations allowed for this experiment.
        success_threshold: Error threshold for success, or ``None`` if not set.
        seed: Random seed used for this experiment, or ``None`` if not set.
    """

    dataset_name: str
    approach_name: str
    max_evaluations: int | None
    success_threshold: float | None
    seed: int | None


class Callback(ABC):
    """Abstract base class for SR evaluation callbacks.

    Implement only the methods you need. To stop the search, return a reason (a ``str``) from
    [on_expr_evaluated][SRToolkit.evaluation.callbacks.Callback.on_expr_evaluated],
    [on_best_expression][SRToolkit.evaluation.callbacks.Callback.on_best_expression] or
    [on_skip][SRToolkit.evaluation.callbacks.Callback.on_skip]; it is
    recorded as the run's ``stop_reason`` unless evaluation has already stopped (the first reason
    is kept). ``False`` also stops, with the reason
    ``"callback:<ClassName>"``. Return ``True`` or ``None`` to continue.

    Examples:
        >>> class PrintBestCallback(Callback):
        ...     def on_best_expression(self, event):
        ...         print(f"New best: {event.expression} (error={event.error:.4g})")
        >>> cb = PrintBestCallback()
        >>> cb.on_best_expression(BestExpressionFound("", "X_0+C", 0.01, 5))
        New best: X_0+C (error=0.01)
    """

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool | str | None:
        """Called after each expression is evaluated.

        Args:
            event: Data about the evaluated expression.

        Returns:
            A reason, or ``False``, to stop the search; ``True`` or ``None`` to continue.
        """
        return None

    def on_best_expression(self, event: BestExpressionFound) -> bool | str | None:
        """Called when a new best expression is found.

        Args:
            event: Data about the new best expression.

        Returns:
            A reason, or ``False``, to stop the search; ``True`` or ``None`` to continue.
        """
        return None

    def on_skip(self, event: ExprSkipped) -> bool | str | None:
        """Called after each skip counted by ``count_skip``, also after evaluation has stopped.

        See [count_skip][SRToolkit.evaluation.evaluator.Evaluator.count_skip].

        Args:
            event: Data about the skip.

        Returns:
            A reason, or ``False``, to stop the search; ``True`` or ``None`` to continue.
        """
        return None

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        """Called before an experiment starts.

        Args:
            event: Data about the experiment that is about to begin.
        """
        pass

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        """Called after an experiment completes.

        Args:
            event: Data about the experiment that just ended.
            results: Final [EvalResult][SRToolkit.utils.types.EvalResult] for this experiment.
        """
        pass

    def get_state(self) -> dict:
        """Return the runtime state to save in a checkpoint, so a resumed run continues from it.

        The evaluator saves it with each snapshot of its checkpoint (see
        [Checkpoints][SRToolkit.evaluation.evaluator]) and, in a resumed run, passes it to
        [set_state][SRToolkit.evaluation.callbacks.Callback.set_state] of the matching callback
        (see [CallbackDispatcher.set_state][SRToolkit.evaluation.callbacks.CallbackDispatcher.set_state]).
        Configuration belongs in [to_dict][SRToolkit.evaluation.callbacks.Callback.to_dict],
        not here.

        Returns:
            A dict that ``json.dumps`` accepts; otherwise saving the snapshot raises ``TypeError``. The
            default is ``{}``.
        """
        return {}

    def set_state(self, state: dict) -> None:
        """Restore the runtime state saved in an earlier session.

        In a resumed run the evaluator calls it once, with what
        [get_state][SRToolkit.evaluation.callbacks.Callback.get_state] returned, just before the
        first ``evaluate_expr`` or ``count_skip`` call after the resume, so after
        ``on_experiment_start`` (see
        [Checkpoints][SRToolkit.evaluation.evaluator]). A callback without saved state is not
        called. The default does nothing.

        Args:
            state: The saved state.
        """
        pass

    def to_dict(self) -> dict:
        """Serialise this callback to a JSON-safe dictionary.

        The default implementation stores only the fully-qualified class path.
        Override in subclasses to include constructor parameters so that
        [from_dict][SRToolkit.evaluation.callbacks.Callback.from_dict] can
        reconstruct a functionally identical instance.

        Returns:
            A JSON-safe dict with at least a ``"callback_class"`` key.
        """
        return {"callback_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}

    @classmethod
    def from_dict(cls, d: dict) -> "Callback":
        """Reconstruct a callback from a serialised dictionary.

        The default implementation calls ``cls()`` with no arguments. Override in
        subclasses that require constructor parameters.

        Args:
            d: Dictionary produced by
                [to_dict][SRToolkit.evaluation.callbacks.Callback.to_dict].

        Returns:
            A new instance of this callback class.
        """
        return cls()

    @classmethod
    def from_config_dict(cls, config: dict) -> "Callback":
        """Reconstruct a callback from a config dict that includes a ``callback_class`` path.

        This is the self-dispatching counterpart to
        [from_dict][SRToolkit.evaluation.callbacks.Callback.from_dict]: it binds the
        config to any installed ``.srtk`` bundle, resolves the concrete callback class named
        by ``callback_class``, and delegates to that class's
        [from_dict][SRToolkit.evaluation.callbacks.Callback.from_dict]. Mirrors
        [Dataset.from_dict][SRToolkit.dataset.dataset.Dataset.from_dict], which
        resolves its own class the same way.

        Args:
            config: A serialised callback config containing a ``callback_class`` key, as
                produced by
                [to_dict][SRToolkit.evaluation.callbacks.Callback.to_dict].

        Returns:
            A new instance of the concrete callback class.

        Raises:
            ImportError: If ``callback_class`` cannot be imported (e.g. its ``.srtk`` bundle
                is not installed, or the config has no ``_bundle`` key and was never bound).
            TypeError: If ``callback_class`` is not a Callback subclass.
        """
        config = _auto_bind(config)
        target_cls = _import_class(config["callback_class"], "callback", Callback)
        return target_cls.from_dict(config)


def _stop_reason(callback: Callback, signal: bool | str | None) -> str | None:
    # A callback's return value as a stop reason: its text, its class for False or "", else None.
    if isinstance(signal, str) or (isinstance(signal, bool) and not signal):
        return signal or f"callback:{type(callback).__name__}"
    return None


class CallbackDispatcher:
    """Manages multiple [Callback][SRToolkit.evaluation.callbacks.Callback] instances and
    dispatches events to all of them.

    Examples:
        >>> dispatcher = CallbackDispatcher([ProgressBarCallback(), EarlyStoppingCallback(threshold=1e-6)])
        >>> dispatcher.on_best_expression(BestExpressionFound("", "X_0", 1e-7, 42))
        'early_stopping:success_threshold'
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        """Args:
        callbacks: Initial list of callbacks. Defaults to an empty list.
        """
        if callbacks is None:
            self._callbacks: list[Callback] = []
        else:
            self._callbacks = callbacks

    def get_callbacks(self) -> list[Callback]:
        """Returns the list of callbacks.

        Returns:
            A list of [Callback][SRToolkit.evaluation.callbacks.Callback] instances in this dispatcher.
        """
        return self._callbacks

    def add(self, callback: Callback) -> None:
        """Add a callback to the dispatcher.

        Args:
            callback: The [Callback][SRToolkit.evaluation.callbacks.Callback] instance to add.
        """
        self._callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a callback from the dispatcher.

        Args:
            callback: The [Callback][SRToolkit.evaluation.callbacks.Callback] instance to remove.

        Raises:
            ValueError: If ``callback`` is not currently registered.
        """
        self._callbacks.remove(callback)

    def on_expr_evaluated(self, event: ExprEvaluated) -> str | None:
        """Dispatch to all callbacks and return the first stop reason.

        Args:
            event: Data about the evaluated expression.

        Returns:
            The reason of the first callback (in order) that asked to stop, ``"callback:<ClassName>"``
            if it returned ``False`` or an empty string; ``None`` if none did.
        """
        reasons = [_stop_reason(cb, cb.on_expr_evaluated(event)) for cb in self._callbacks]
        return next((r for r in reasons if r is not None), None)

    def on_best_expression(self, event: BestExpressionFound) -> str | None:
        """Dispatch to all callbacks and return the first stop reason.

        Args:
            event: Data about the new best expression.

        Returns:
            As for [on_expr_evaluated][SRToolkit.evaluation.callbacks.CallbackDispatcher.on_expr_evaluated].
        """
        reasons = [_stop_reason(cb, cb.on_best_expression(event)) for cb in self._callbacks]
        return next((r for r in reasons if r is not None), None)

    def on_skip(self, event: ExprSkipped) -> str | None:
        """Dispatch to all callbacks and return the first stop reason.

        Args:
            event: Data about the skip.

        Returns:
            As for [on_expr_evaluated][SRToolkit.evaluation.callbacks.CallbackDispatcher.on_expr_evaluated].
        """
        reasons = [_stop_reason(cb, cb.on_skip(event)) for cb in self._callbacks]
        return next((r for r in reasons if r is not None), None)

    def get_state(self) -> list[dict]:
        """Return the state of every callback, in order.

        Returns:
            One ``{"callback_class": <import path>, "state": <get_state()>}`` per callback.
        """
        return [{"callback_class": cb.to_dict()["callback_class"], "state": cb.get_state()} for cb in self._callbacks]

    def set_state(self, states: list[dict]) -> None:
        """Restore the states saved by an earlier dispatcher's ``get_state``.

        The state at each position goes to the callback at the same position if its import path
        (``callback_class``) is the same, and callbacks without a saved state are not called. A
        non-empty state (not ``{}``, which
        [get_state][SRToolkit.evaluation.callbacks.Callback.get_state] returns by default) whose
        class is only at other positions is dropped with a ``UserWarning``: the callbacks were
        reordered, so e.g. a time limit would start counting from zero. A state of a callback that
        is no longer there is dropped silently.

        Args:
            states: The saved states.
        """
        classes = [cb.to_dict()["callback_class"] for cb in self._callbacks]
        for i, saved in enumerate(states):
            if i < len(classes) and saved["callback_class"] == classes[i]:
                self._callbacks[i].set_state(saved["state"])
            elif saved["state"] and saved["callback_class"] in classes:
                warnings.warn(
                    f"[CallbackDispatcher] The saved state of {saved['callback_class']} (position {i}) is "
                    "dropped: that class is now at another position.",
                    UserWarning,
                )

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        """Dispatch to all callbacks.

        Args:
            event: Data about the experiment that is about to begin.
        """
        for cb in self._callbacks:
            cb.on_experiment_start(event)

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        """Dispatch to all callbacks.

        Args:
            event: Data about the experiment that just ended.
            results: Final [EvalResult][SRToolkit.utils.types.EvalResult] for this experiment.
        """
        for cb in self._callbacks:
            cb.on_experiment_end(event, results)


class ProgressBarCallback(Callback):
    """Displays a tqdm progress bar that updates after each expression evaluation.

    The bar shows the event's ``evaluation_number``, so in a resumed run it continues from the
    calls made before.

    Examples:
        >>> cb = ProgressBarCallback(desc="My search")
        >>> cb.desc
        'My search'
    """

    def __init__(self, desc: str | None = None):
        """Args:
        desc: Description label shown on the progress bar. If ``None``, the label
            is auto-generated as ``"<approach> on <dataset>"`` when the experiment starts.
        """
        self.pbar = None
        self.desc = desc

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        desc = self.desc or f"{event.approach_name} on {event.dataset_name}"
        if event.max_evaluations is not None:
            self.pbar = tqdm(total=event.max_evaluations, desc=desc, unit=" expr")
        else:
            self.pbar = tqdm(desc=desc, unit=" expr")

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool | None:
        if self.pbar is not None:
            self.pbar.update(event.evaluation_number - self.pbar.n)
        return None

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        if self.pbar:
            self.pbar.close()
            self.pbar = None

    def to_dict(self) -> dict:
        return {**super().to_dict(), "desc": self.desc}

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressBarCallback":
        return cls(desc=d.get("desc"))


class EarlyStoppingCallback(Callback):
    """Stops the search when the best error falls below a threshold or after a number of evaluations.

    The evaluator already enforces its own ``success_threshold`` and ``max_evaluations``; use this
    callback for a tighter threshold or budget, or as an example of a stopping callback. Its stop
    reasons, ``"early_stopping:success_threshold"`` and ``"early_stopping:max_evaluations"``, differ
    from the evaluator's, so a run it stops counts as one that ended with budget left.

    Examples:
        >>> cb = EarlyStoppingCallback(threshold=1e-6)
        >>> cb.on_best_expression(BestExpressionFound("", "X_0", 1e-7, 42))
        'early_stopping:success_threshold'
        >>> cb.on_best_expression(BestExpressionFound("", "X_0", 1e-5, 43))
        True
    """

    def __init__(self, threshold: float | None, max_evaluations: int | None = None):
        """Args:
        threshold: Error value below which the search is stopped. If ``None``, each
            experiment's ``success_threshold`` is used.
        max_evaluations: Number of evaluations after which the search is stopped. If
            ``None``, each experiment's ``max_evaluations`` is used (if it has one).
        """
        self.threshold = threshold
        self.max_evaluations = max_evaluations
        # The limits of the current experiment: the given ones, or the experiment's where None.
        self._threshold = threshold
        self._max_evaluations = max_evaluations

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        self._threshold = event.success_threshold if self.threshold is None else self.threshold
        self._max_evaluations = self.max_evaluations
        if self._max_evaluations is None and event.max_evaluations is not None and event.max_evaluations > 0:
            self._max_evaluations = event.max_evaluations

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool | str | None:
        if self._max_evaluations is not None and event.evaluation_number >= self._max_evaluations >= 0:
            return "early_stopping:max_evaluations"
        return True

    def on_best_expression(self, event: BestExpressionFound) -> bool | str | None:
        if self._threshold is not None and event.error < self._threshold:
            return "early_stopping:success_threshold"
        return True

    def to_dict(self) -> dict:
        return {**super().to_dict(), "threshold": self.threshold, "max_evaluations": self.max_evaluations}

    @classmethod
    def from_dict(cls, d: dict) -> "EarlyStoppingCallback":
        return cls(threshold=d.get("threshold"), max_evaluations=d.get("max_evaluations"))


class TimeLimitCallback(Callback):
    """Stops the search once a wall-clock time limit has elapsed.

    The timer starts when the experiment begins
    ([on_experiment_start][SRToolkit.evaluation.callbacks.Callback.on_experiment_start])
    and uses `time.monotonic` so it is unaffected by system clock adjustments. The stop
    request (reason ``"time_limit"``) is issued from the next expression evaluation or skip once
    ``time_limit`` seconds have elapsed, so the search may run slightly past ``time_limit``
    depending on the time between consecutive evaluations and skips.

    In a resumed run the limit spans all sessions: the time used up to the last snapshot is
    restored from the checkpoint. Time after that snapshot in a session that crashed is not counted.

    Examples:
        >>> cb = TimeLimitCallback(time_limit=60.0)
        >>> cb.time_limit
        60.0
        >>> from datetime import timedelta
        >>> TimeLimitCallback(time_limit=timedelta(minutes=2)).time_limit
        120.0
    """

    def __init__(self, time_limit: float | timedelta):
        """Args:
            time_limit: Maximum wall-clock duration before the search is stopped at the next
                expression evaluation or skip. Either a number of seconds or a
                `datetime.timedelta`, which is converted to seconds and stored as a float.
                Must be strictly positive.

        Raises:
            ValueError: If ``time_limit`` is not strictly positive.
        """
        if isinstance(time_limit, timedelta):
            time_limit = time_limit.total_seconds()
        if time_limit <= 0:
            raise ValueError(f"time_limit must be strictly positive, got {time_limit}.")
        self.time_limit: float = time_limit
        self._start_time: float | None = None

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        self._start_time = time.monotonic()

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool | str | None:
        if self._start_time is not None and time.monotonic() - self._start_time >= self.time_limit:
            return "time_limit"
        return True

    def on_skip(self, event: ExprSkipped) -> bool | str | None:
        if self._start_time is not None and time.monotonic() - self._start_time >= self.time_limit:
            return "time_limit"
        return True

    def get_state(self) -> dict:
        """Return ``{"elapsed": seconds}``, the time counted so far, or ``{}`` before the timer starts."""
        if self._start_time is None:
            return {}
        return {"elapsed": time.monotonic() - self._start_time}

    def set_state(self, state: dict) -> None:
        """Continue from the saved ``"elapsed"`` time; does nothing without it.

        Afterwards the elapsed time is the saved one plus the time since this call, whether or not
        ``on_experiment_start`` was called.
        """
        if "elapsed" in state:
            self._start_time = time.monotonic() - state["elapsed"]

    def to_dict(self) -> dict:
        return {**super().to_dict(), "time_limit": self.time_limit}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeLimitCallback":
        return cls(time_limit=d["time_limit"])


class LoggingCallback(Callback):
    """Logs each new best expression to stdout or a file.

    ``log_file`` may contain placeholders that are resolved at experiment start
    using fields from [ExperimentEvent][SRToolkit.evaluation.callbacks.ExperimentEvent].
    Available placeholders: ``{dataset_name}``,``{approach_name}``, ``{seed}``. Using
    per-experiment placeholders (e.g. ``{seed}``) gives each job its own file, which is
    the recommended approach for parallel execution.

    When multiple jobs share the same resolved file path, writes are protected
    by ``fcntl.flock`` (POSIX advisory locking) so concurrent processes on
    Linux / macOS do not corrupt each other's output.  On Windows or network
    filesystems where ``flock`` is unavailable the lock is silently skipped.

    Examples:
        >>> cb = LoggingCallback()
        >>> cb.on_best_expression(BestExpressionFound("Nguyen-1_ProGED_42", "X_0+C", 0.001, 10))
        [Experiment Nguyen-1_ProGED_42] New best: X_0+C (error=1.000000e-03)
        >>> cb = LoggingCallback(log_file="logs/{dataset_name}_{seed}.log")
        >>> cb.on_experiment_start(ExperimentEvent(dataset_name="test", max_evaluations=10, seed=1,
        ...                                        success_threshold=0, approach_name="ta"))
        >>> cb._resolved_log_file
        'logs/test_1.log'
    """

    def __init__(self, log_file: str | None = None):
        """Args:
        log_file: Destination for log messages.  If ``None``, messages are
            printed to stdout.  May be a plain path or a template string with
            placeholders ``{dataset_name}``, ``{approach_name}``, ``{seed}``
            that are resolved when the experiment starts.
        """
        self.log_file = log_file
        self._resolved_log_file: str | None = log_file

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        if self.log_file is not None:
            self._resolved_log_file = self.log_file.format(
                dataset_name=event.dataset_name,
                approach_name=event.approach_name,
                seed=event.seed,
            )
        else:
            self._resolved_log_file = None

    def on_best_expression(self, event: BestExpressionFound) -> None:
        log_msg = f"[Experiment {event.experiment_id}] New best: {event.expression} (error={event.error:.6e})\n"
        if self._resolved_log_file is not None:
            os.makedirs(os.path.dirname(os.path.abspath(self._resolved_log_file)), exist_ok=True)
            with open(self._resolved_log_file, "a") as f:
                try:
                    import fcntl

                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(log_msg)
                    fcntl.flock(f, fcntl.LOCK_UN)
                except (ImportError, OSError):
                    f.write(log_msg)
        else:
            print(log_msg, end="")

    def to_dict(self) -> dict:
        return {**super().to_dict(), "log_file": self.log_file}

    @classmethod
    def from_dict(cls, d: dict) -> "LoggingCallback":
        return cls(log_file=d.get("log_file"))
