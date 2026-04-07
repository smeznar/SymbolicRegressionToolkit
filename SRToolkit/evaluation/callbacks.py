"""
Event-driven callback system for monitoring and controlling SR evaluation.

Provides event dataclasses fired during evaluation, the
[SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] base class for implementing custom callbacks,
a [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] for managing multiple callbacks,
and built-in implementations for progress display, early stopping, and logging.
"""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

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
    experiment_id: int
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

    experiment_id: int
    expression: str
    error: float
    evaluation_number: int


@dataclass
class ExperimentEvent:
    """
    Fired at experiment start and end.

    Attributes:
        experiment_id: Identifier of the experiment.
        dataset_name: Name of the dataset being evaluated.
        approach_name: Name of the SR approach being run.
        max_evaluations: Maximum number of evaluations allowed for this experiment.
        success_threshold: Error threshold for success, or ``None`` if not set.
        seed: Random seed used for this experiment, or ``None`` if not set.
    """

    experiment_id: int
    dataset_name: str
    approach_name: str
    max_evaluations: Optional[int]
    success_threshold: Optional[float]
    seed: Optional[int]


@dataclass
class DatasetEvent:
    """
    Fired at dataset start and end.

    Attributes:
        dataset_name: Name of the dataset.
    """

    dataset_name: str


@dataclass
class ApproachEvent:
    """
    Fired at approach start and end.

    Attributes:
        approach_name: Name of the SR approach.
    """

    approach_name: str


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
        >>> cb.on_best_expression(BestExpressionFound(0, "X_0+C", 0.01, 5))
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

    def on_dataset_start(self, event: DatasetEvent) -> None:
        """
        Called before processing a dataset.

        Args:
            event: Data about the dataset that is about to be processed.
        """
        pass

    def on_dataset_end(self, event: DatasetEvent, results: dict) -> None:
        """
        Called after all experiments on a dataset complete.

        Args:
            event: Data about the dataset.
            results: Aggregated results for the dataset.
        """
        pass

    def on_approach_start(self, event: ApproachEvent) -> None:
        """
        Called before an approach starts processing.

        Args:
            event: Data about the approach that is about to run.
        """
        pass

    def on_approach_end(self, event: ApproachEvent, results: dict) -> None:
        """
        Called after an approach finishes processing all datasets.

        Args:
            event: Data about the approach.
            results: Aggregated results for the approach.
        """
        pass


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
            ``False`` if any callback returned ``False`` (requesting early stop),
            ``True`` otherwise.
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
            ``False`` if any callback returned ``False`` (requesting early stop),
            ``True`` otherwise.
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

    def on_dataset_start(self, event: DatasetEvent) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the dataset that is about to be processed.
        """
        for cb in self._callbacks:
            cb.on_dataset_start(event)

    def on_dataset_end(self, event: DatasetEvent, results: dict) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the dataset.
            results: Aggregated results for the dataset.
        """
        for cb in self._callbacks:
            cb.on_dataset_end(event, results)

    def on_approach_start(self, event: ApproachEvent) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the approach that is about to run.
        """
        for cb in self._callbacks:
            cb.on_approach_start(event)

    def on_approach_end(self, event: ApproachEvent, results: dict) -> None:
        """
        Dispatch to all callbacks.

        Args:
            event: Data about the approach.
            results: Aggregated results for the approach.
        """
        for cb in self._callbacks:
            cb.on_approach_end(event, results)


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
        from tqdm import tqdm

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


class EarlyStoppingCallback(SRCallbacks):
    """
    Stops the search when the best expression error falls below a threshold.

    Examples:
        >>> cb = EarlyStoppingCallback(threshold=1e-6)
        >>> cb.on_best_expression(BestExpressionFound(0, "X_0", 1e-7, 42))
        False
        >>> cb.on_best_expression(BestExpressionFound(0, "X_0", 1e-5, 43))
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


class LoggingCallback(SRCallbacks):
    """
    Logs each new best expression to stdout or a file.

    Examples:
        >>> cb = LoggingCallback()
        >>> cb.on_best_expression(BestExpressionFound(0, "X_0+C", 0.001, 10))
        [Experiment 0] New best: X_0+C (error=1.000000e-03)
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Path to a file where log messages are appended. If ``None``,
                messages are printed to stdout.
        """
        self.log_file = log_file

    def on_best_expression(self, event: BestExpressionFound) -> None:
        log_msg = f"[Experiment {event.experiment_id}] New best: {event.expression} (error={event.error:.6e})"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        else:
            print(log_msg)
