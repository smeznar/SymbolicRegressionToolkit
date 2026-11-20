"""Classes and functions for evaluating symbolic regression approaches.

Modules:
    parameter_estimator: [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator] — fits
        free constants in expressions and ranks them by RMSE.
    sr_evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] — expression
        evaluation.
    results: [SR_results][SRToolkit.evaluation.results.SR_results] and
        [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] — experiment results and
        the base class for their post-processing.
    result_augmentation: [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] implementations
        that post-process results with LaTeX, simplified forms, and held-out metrics.
    callbacks: [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] and
        [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] — event-driven hooks
        for monitoring and early stopping during evaluation.
"""

from SRToolkit.utils.types import EvalResult, ModelResult

from .callbacks import (
    BestExpressionFound,
    CallbackDispatcher,
    EarlyStoppingCallback,
    ExperimentEvent,
    ExprEvaluated,
    LoggingCallback,
    ProgressBarCallback,
    SRCallbacks,
    TimeLimitCallback,
)
from .parameter_estimator import ParameterEstimator
from .result_augmentation import (
    ExpressionSimplifier,
    ExpressionToLatex,
    HoldoutMetric,
)
from .results import ResultAugmenter, SR_results
from .sr_evaluator import SR_evaluator, evaluate_metric

__all__ = [
    "ParameterEstimator",
    "SR_evaluator",
    "SR_results",
    "EvalResult",
    "ModelResult",
    "ResultAugmenter",
    "evaluate_metric",
    "ExpressionSimplifier",
    "HoldoutMetric",
    "ExpressionToLatex",
    "SRCallbacks",
    "CallbackDispatcher",
    "EarlyStoppingCallback",
    "ProgressBarCallback",
    "LoggingCallback",
    "TimeLimitCallback",
    "ExperimentEvent",
    "BestExpressionFound",
    "ExprEvaluated",
]
