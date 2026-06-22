"""
Event-driven callback system for monitoring and controlling SR evaluation.

Provides event dataclasses fired during evaluation, the
[SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] base class for implementing custom callbacks,
a [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] for managing multiple callbacks,
and built-in implementations for progress display, early stopping, and logging.
"""

import importlib
import os
import time
from abc import ABC
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Union

from tqdm import tqdm

from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.utils import EvalResult


@dataclass
class ExprEvaluated:
    """
    Fired after each expression is evaluated by
    [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr].

    Attributes:
        expression: String representation of the evaluated expression.
        error: Error value returned by the ranking function (RMSE or BED).
        evaluation_number: Total number of
            [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]
            calls made so far, including cache hits.
        experiment_id: Identifier of the current experiment.
        is_new_best: ``True`` if this expression achieved a lower error than all previous ones.
    """

    expression: str
    error: float
    evaluation_number: int
    experiment_id: str
    is_new_best: bool


@dataclass
class BestExpressionFound:
    """
    Fired when a new best expression is found during evaluation.

    Attributes:
        experiment_id: Identifier of the current experiment.
        expression: String representation of the new best expression.
        error: Error value of the new best expression.
        evaluation_number: Total number of
            [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]
            calls made at the time this event is fired.
    """

    experiment_id: str
    expression: str
    error: float
    evaluation_number: int


@dataclass
class ExperimentEvent:
    """
    Fired at experiment start and end.

    Attributes:
        dataset_name: Name of the dataset being evaluated.
        approach_name: Name of the SR approach being run.
        max_evaluations: Maximum number of evaluations allowed for this experiment.
        success_threshold: Error threshold for success, or ``None`` if not set.
        seed: Random seed used for this experiment, or ``None`` if not set.
    """

    dataset_name: str
    approach_name: str
    max_evaluations: Optional[int]
    success_threshold: Optional[float]
    seed: Optional[int]


class SRCallbacks(ABC):
    """
    Abstract base class for SR evaluation callbacks.

    Implement only the methods you need. Return ``False`` from
    [on_expr_evaluated][SRToolkit.evaluation.callbacks.SRCallbacks.on_expr_evaluated] or
    [on_best_expression][SRToolkit.evaluation.callbacks.SRCallbacks.on_best_expression]
    to request early stopping; return ``True`` or ``None`` to continue.

    Examples:
        >>> class PrintBestCallback(SRCallbacks):
        ...     def on_best_expression(self, event):
        ...         print(f"New best: {event.expression} (error={event.error:.4g})")
        >>> cb = PrintBestCallback()
        >>> cb.on_best_expression(BestExpressionFound("", "X_0+C", 0.01, 5))
        New best: X_0+C (error=0.01)
    """

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        """
        Called after each expression is evaluated.

        Args:
            event: Data about the evaluated expression.

        Returns:
            ``False`` to stop the search early, ``True`` or ``None`` to continue.
        """
        return None

    def on_best_expression(self, event: BestExpressionFound) -> Optional[bool]:
        """
        Called when a new best expression is found.

        Args:
            event: Data about the new best expression.

        Returns:
            ``False`` to stop the search early, ``True`` or ``None`` to continue.
        """
        return None

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        """
        Called before an experiment starts.

        Args:
            event: Data about the experiment that is about to begin.
        """
        pass

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        """
        Called after an experiment completes.

        Args:
            event: Data about the experiment that just ended.
            results: Final [EvalResult][SRToolkit.utils.types.EvalResult] for this experiment.
        """
        pass

    def to_dict(self) -> dict:
        """
        Serialise this callback to a JSON-safe dictionary.

        The default implementation stores only the fully-qualified class path.
        Override in subclasses to include constructor parameters so that
        [from_dict][SRToolkit.evaluation.callbacks.SRCallbacks.from_dict] can
        reconstruct a functionally identical instance.

        Returns:
            A JSON-safe dict with at least a ``"callback_class"`` key.
        """
        return {"callback_class": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}

    @classmethod
    def from_dict(cls, d: dict) -> "SRCallbacks":
        """
        Reconstruct a callback from a serialised dictionary.

        The default implementation calls ``cls()`` with no arguments. Override in
        subclasses that require constructor parameters.

        Args:
            d: Dictionary produced by
                [to_dict][SRToolkit.evaluation.callbacks.SRCallbacks.to_dict].

        Returns:
            A new instance of this callback class.
        """
        return cls()

    @classmethod
    def from_config_dict(cls, config: dict) -> "SRCallbacks":
        """
        Reconstruct a callback from a config dict that includes a ``callback_class`` path.

        This is the self-dispatching counterpart to
        [from_dict][SRToolkit.evaluation.callbacks.SRCallbacks.from_dict]: it binds the
        config to any installed ``.srtk`` bundle, resolves the concrete callback class named
        by ``callback_class``, and delegates to that class's
        [from_dict][SRToolkit.evaluation.callbacks.SRCallbacks.from_dict]. Mirrors
        [SR_dataset.from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict], which
        resolves its own class the same way.

        Args:
            config: A serialised callback config containing a ``callback_class`` key, as
                produced by
                [to_dict][SRToolkit.evaluation.callbacks.SRCallbacks.to_dict].

        Returns:
            A new instance of the concrete callback class.

        Raises:
            ImportError: If ``callback_class`` cannot be imported (e.g. its ``.srtk`` bundle
                is not installed, or the config has no ``_bundle`` key and was never bound).
        """
        config = _auto_bind(config)
        class_path = config["callback_class"]
        module_path, cls_name = class_path.rsplit(".", 1)
        try:
            target_cls = getattr(importlib.import_module(module_path), cls_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Cannot import callback class {class_path!r}. "
                "If this is a bundle class, install the bundle first. "
                "If the config has no '_bundle' key, call bind_config(config) manually."
            ) from None
        return target_cls.from_dict(config)


class CallbackDispatcher:
    """
    Manages multiple [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instances and
    dispatches events to all of them.

    Examples:
        >>> dispatcher = CallbackDispatcher()
        >>> dispatcher.add(EarlyStoppingCallback(threshold=1e-6))
        >>> len(dispatcher._callbacks)
        1
    """

    def __init__(self, callbacks: Optional[List[SRCallbacks]] = None):
        """
        Args:
            callbacks: Initial list of callbacks. Defaults to an empty list.
        """
        if callbacks is None:
            self._callbacks: List[SRCallbacks] = []
        else:
            self._callbacks = callbacks

    def get_callbacks(self) -> List[SRCallbacks]:
        """
        Returns the list of callbacks.

        Returns:
            A list of [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instances in this dispatcher.
        """
        return self._callbacks

    def add(self, callback: SRCallbacks) -> None:
        """
        Add a callback to the dispatcher.

        Args:
            callback: The [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instance to add.
        """
        self._callbacks.append(callback)

    def remove(self, callback: SRCallbacks) -> None:
        """
        Remove a callback from the dispatcher.

        Args:
            callback: The [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instance to remove.

        Raises:
            ValueError: If ``callback`` is not currently registered.
        """
        self._callbacks.remove(callback)

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool:
        """
        Dispatch to all callbacks and aggregate the stop signal.

        Args:
            event: Data about the evaluated expression.

        Returns:
            ``False`` if any callback returned ``False`` (requesting early stop), ``True`` otherwise.
        """
        should_continue = True
        for cb in self._callbacks:
            cont = cb.on_expr_evaluated(event)
            if isinstance(cont, bool) and not cont:
                should_continue = False
        return should_continue

    def on_best_expression(self, event: BestExpressionFound) -> bool:
        """
        Dispatch to all callbacks and aggregate the stop signal.

        Args:
            event: Data about the new best expression.

        Returns:
            ``False`` if any callback returned ``False`` (requesting early stop), ``True`` otherwise.
        """
        should_continue = True
        for cb in self._callbacks:
            cont = cb.on_best_expression(event)
            if isinstance(cont, bool) and not cont:
                should_continue = False
        return should_continue

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the experiment that is about to begin.
        """
        for cb in self._callbacks:
            cb.on_experiment_start(event)

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the experiment that just ended.
            results: Final [EvalResult][SRToolkit.utils.types.EvalResult] for this experiment.
        """
        for cb in self._callbacks:
            cb.on_experiment_end(event, results)


class ProgressBarCallback(SRCallbacks):
    """
    Displays a tqdm progress bar that updates after each expression evaluation.

    Examples:
        >>> cb = ProgressBarCallback(desc="My search")
        >>> cb.desc
        'My search'
    """

    def __init__(self, desc: Optional[str] = None):
        """
        Args:
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

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        if self.pbar is not None:
            self.pbar.update(1)
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


class EarlyStoppingCallback(SRCallbacks):
    """
    Stops the search when the best expression error falls below a threshold.

    Examples:
        >>> cb = EarlyStoppingCallback(threshold=1e-6)
        >>> cb.on_best_expression(BestExpressionFound("", "X_0", 1e-7, 42))
        False
        >>> cb.on_best_expression(BestExpressionFound("", "X_0", 1e-5, 43))
        True
    """

    def __init__(self, threshold: Optional[float], max_evaluations: Optional[int] = None):
        """
        Args:
            threshold: Error value below which the search is stopped.
        """
        self.threshold = threshold
        self.max_evaluations = max_evaluations

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        if self.threshold is None and event.success_threshold is not None:
            self.threshold = event.success_threshold
        if self.max_evaluations is None and event.max_evaluations is not None and event.max_evaluations > 0:
            self.max_evaluations = event.max_evaluations

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        if self.max_evaluations is not None and event.evaluation_number >= self.max_evaluations >= 0:
            return False
        return True

    def on_best_expression(self, event: BestExpressionFound) -> Optional[bool]:
        if self.threshold is not None and event.error < self.threshold:
            return False
        return True

    def to_dict(self) -> dict:
        return {**super().to_dict(), "threshold": self.threshold, "max_evaluations": self.max_evaluations}

    @classmethod
    def from_dict(cls, d: dict) -> "EarlyStoppingCallback":
        return cls(threshold=d.get("threshold"), max_evaluations=d.get("max_evaluations"))


class TimeLimitCallback(SRCallbacks):
    """
    Stops the search once a wall-clock time limit has elapsed.

    The timer starts when the experiment begins
    ([on_experiment_start][SRToolkit.evaluation.callbacks.SRCallbacks.on_experiment_start])
    and uses `time.monotonic` so it is unaffected by system clock adjustments. The stop
    request is issued from the next expression evaluation after the limit is exceeded, so the
    search may run slightly past ``time_limit`` depending on how long a single evaluation takes.

    Examples:
        >>> cb = TimeLimitCallback(time_limit=60.0)
        >>> cb.time_limit
        60.0
        >>> from datetime import timedelta
        >>> TimeLimitCallback(time_limit=timedelta(minutes=2)).time_limit
        120.0
    """

    def __init__(self, time_limit: Union[float, timedelta]):
        """
        Args:
            time_limit: Maximum wall-clock duration before the search is stopped at the next
                expression evaluation. Either a number of seconds or a
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
        self._start_time: Optional[float] = None

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        self._start_time = time.monotonic()

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        if self._start_time is not None:
            if time.monotonic() - self._start_time >= self.time_limit:
                return False
        return True

    def to_dict(self) -> dict:
        return {**super().to_dict(), "time_limit": self.time_limit}

    @classmethod
    def from_dict(cls, d: dict) -> "TimeLimitCallback":
        return cls(time_limit=d["time_limit"])


class LoggingCallback(SRCallbacks):
    """
    Logs each new best expression to stdout or a file.

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

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Destination for log messages.  If ``None``, messages are
                printed to stdout.  May be a plain path or a template string with
                placeholders ``{dataset_name}``, ``{approach_name}``, ``{seed}``
                that are resolved when the experiment starts.
        """
        self.log_file = log_file
        self._resolved_log_file: Optional[str] = log_file

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
