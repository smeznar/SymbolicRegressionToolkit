"""Classes and functions for evaluating symbolic regression approaches.

Modules:
    parameter_estimator: [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator] — fits
        free constants in expressions and ranks them by RMSE.
    evaluator: [Evaluator][SRToolkit.evaluation.evaluator.Evaluator] — expression
        evaluation.
    model_store: [ModelStore][SRToolkit.evaluation.model_store.ModelStore] — the SQLite storage
        behind an evaluator's ``models`` and ``trace``, optionally a checkpoint file.
    results: [Results][SRToolkit.evaluation.results.Results] and
        [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] — experiment results and
        the base class for their post-processing.
    result_augmentation: [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] implementations
        that post-process results with LaTeX, simplified forms, and held-out metrics.
    callbacks: [Callback][SRToolkit.evaluation.callbacks.Callback] and
        [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] — event-driven hooks
        for monitoring and early stopping during evaluation.
"""

from SRToolkit.utils.types import EvalResult, ModelResult

from .callbacks import (
    BestExpressionFound,
    Callback,
    CallbackDispatcher,
    EarlyStoppingCallback,
    ExperimentEvent,
    ExprEvaluated,
    ExprSkipped,
    LoggingCallback,
    ProgressBarCallback,
    TimeLimitCallback,
)
from .evaluator import Evaluator, evaluate_metric
from .parameter_estimator import ParameterEstimator
from .result_augmentation import (
    ExpressionSimplifier,
    ExpressionToLatex,
    HoldoutMetric,
)
from .results import ResultAugmenter, Results

__all__ = [
    "ParameterEstimator",
    "Evaluator",
    "Results",
    "EvalResult",
    "ModelResult",
    "ResultAugmenter",
    "evaluate_metric",
    "ExpressionSimplifier",
    "HoldoutMetric",
    "ExpressionToLatex",
    "Callback",
    "CallbackDispatcher",
    "EarlyStoppingCallback",
    "ProgressBarCallback",
    "LoggingCallback",
    "TimeLimitCallback",
    "ExperimentEvent",
    "BestExpressionFound",
    "ExprEvaluated",
    "ExprSkipped",
]
