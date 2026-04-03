"""
Callback system for SR execution events.

This module provides the callback infrastructure for monitoring and controlling
SR experiments. Users can implement custom callbacks or use built-in ones.
"""

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

from SRToolkit.utils import EvalResult


@dataclass
class ExprEvaluated:
    """Fired after each expression evaluation."""

    expression: str
    error: float
    evaluation_number: int
    experiment_id: int
    is_new_best: bool


@dataclass
class BestExpressionFound:
    """Fired when a new best expression is found."""

    experiment_id: int
    expression: str
    error: float
    evaluation_number: int


@dataclass
class ExperimentEvent:
    """Fired at experiment start/end."""

    experiment_id: int
    dataset_name: str
    approach_name: str
    seed: Optional[int]


@dataclass
class DatasetEvent:
    """Fired at dataset start/end."""

    dataset_name: str


@dataclass
class ApproachEvent:
    """Fired at approach start/end."""

    approach_name: str


class SRCallbacks(ABC):
    """
    Base class for SR callbacks.

    Implement only the methods you need. Methods return None by default,
    which is treated as "continue". Return False to signal early stopping.
    """

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        """
        Called after each expression is evaluated.

        Return False to stop search early, True or None to continue.
        """
        return None

    def on_best_expression(self, event: BestExpressionFound) -> Optional[bool]:
        """
        Called when a new best expression is found.

        Return False to stop search early.
        """
        return None

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        """Called before an experiment starts."""
        pass

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        """Called after an experiment completes."""
        pass

    def on_dataset_start(self, event: DatasetEvent) -> None:
        """Called before processing a dataset."""
        pass

    def on_dataset_end(self, event: DatasetEvent, results: dict) -> None:
        """Called after processing all experiments on a dataset."""
        pass

    def on_approach_start(self, event: ApproachEvent) -> None:
        """Called before processing an approach."""
        pass

    def on_approach_end(self, event: ApproachEvent, results: dict) -> None:
        """Called after processing an approach on all datasets."""
        pass


class CallbackDispatcher:
    """
    Manages multiple callbacks and dispatches events to all of them.

    Usage:
        dispatcher = CallbackDispatcher()
        dispatcher.add(ProgressBarCallback())
        dispatcher.add(EarlyStoppingCallback(threshold=1e-6))

        results = dataset.evaluate_approach(approach, callbacks=dispatcher)
    """

    def __init__(self, callbacks: Optional[List[SRCallbacks]] = None):
        if callbacks is None:
            self._callbacks: List[SRCallbacks] = []
        else:
            self._callbacks = callbacks

    def add(self, callback: SRCallbacks) -> None:
        """Add a callback. Returns self for chaining."""
        self._callbacks.append(callback)

    def remove(self, callback: SRCallbacks) -> None:
        """Remove a callback."""
        self._callbacks.remove(callback)

    def on_expr_evaluated(self, event: ExprEvaluated) -> bool:
        """
        Dispatch to all callbacks. Return False if any callback requests stop.
        """
        should_continue = True
        for cb in self._callbacks:
            cont = cb.on_expr_evaluated(event)
            if isinstance(cont, bool) and not cont:
                should_continue = False
        return should_continue

    def on_best_expression(self, event: BestExpressionFound) -> bool:
        should_continue = True
        for cb in self._callbacks:
            cont = cb.on_best_expression(event)
            if isinstance(cont, bool) and not cont:
                should_continue = False
        return should_continue

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        for cb in self._callbacks:
            cb.on_experiment_start(event)

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        for cb in self._callbacks:
            cb.on_experiment_end(event, results)

    def on_dataset_start(self, event: DatasetEvent) -> None:
        for cb in self._callbacks:
            cb.on_dataset_start(event)

    def on_dataset_end(self, event: DatasetEvent, results: dict) -> None:
        for cb in self._callbacks:
            cb.on_dataset_end(event, results)

    def on_approach_start(self, event: ApproachEvent) -> None:
        for cb in self._callbacks:
            cb.on_approach_start(event)

    def on_approach_end(self, event: ApproachEvent, results: dict) -> None:
        for cb in self._callbacks:
            cb.on_approach_end(event, results)


class ProgressBarCallback(SRCallbacks):
    """Progress bar using tqdm."""

    def __init__(self, desc: Optional[str] = None):
        self.pbar = None
        self.desc = desc

    def on_experiment_start(self, event: ExperimentEvent) -> None:
        desc = self.desc or f"{event.approach_name} on {event.dataset_name}"
        from tqdm import tqdm

        self.pbar = tqdm(desc=desc, unit="expr")

    def on_expr_evaluated(self, event: ExprEvaluated) -> Optional[bool]:
        if self.pbar:
            self.pbar.update(1)
        return None

    def on_experiment_end(self, event: ExperimentEvent, results: EvalResult) -> None:
        if self.pbar:
            self.pbar.close()
            self.pbar = None


class EarlyStoppingCallback(SRCallbacks):
    """Stop search when success threshold is reached."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def on_best_expression(self, event: BestExpressionFound) -> Optional[bool]:
        if event.error < self.threshold:
            return False
        return True


class LoggingCallback(SRCallbacks):
    """Log best expressions to file or stdout."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file

    def on_best_expression(self, event: BestExpressionFound) -> None:
        log_msg = f"[Experiment {event.experiment_id}] New best: {event.expression} (error={event.error:.6e})"
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
        else:
            print(log_msg)
