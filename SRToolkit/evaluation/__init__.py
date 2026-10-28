"""
Classes and functions for evaluating symbolic regression approaches.

Modules:
    parameter_estimator: [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator] — fits
        free constants in expressions and ranks them by RMSE.
    sr_evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] and
        [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] — expression evaluation and result management.
    result_augmentation: [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] implementations
        that post-process results with LaTeX, simplified forms, and held-out metrics.
    callbacks: [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] and
        [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] — event-driven hooks
        for monitoring and early stopping during evaluation.
"""

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
from .sr_evaluator import EvalResult, ModelResult, ResultAugmenter, SR_evaluator, SR_results, evaluate_metric

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
