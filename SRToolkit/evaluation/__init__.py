"""
This module contains classes and functions for evaluating symbolic regression approaches. Mainly it contains classes that
can be used for parameter estimation and evaluation of mathematical expressions on some dataset.

Modules:
    parameter_estimator: The module containing classes and functions for parameter estimation.
    sr_evaluator: The module containing classes and functions for expressions on some dataset.
    result_augmentation: The module containing classes and functions for result augmentation.
"""

from .parameter_estimator import ParameterEstimator
from .result_augmentation import (
    BED,
    R2,
    RESULT_AUGMENTERS,
    RMSE,
    EvalResult,
    ExpressionSimplifier,
    ExpressionToLatex,
    ModelResult,
    register_augmenter,
)
from .sr_evaluator import ResultAugmenter, SR_evaluator, SR_results

__all__ = [
    "ParameterEstimator",
    "SR_evaluator",
    "SR_results",
    "EvalResult",
    "ModelResult",
    "ResultAugmenter",
    "ExpressionSimplifier",
    "RMSE",
    "R2",
    "BED",
    "ExpressionToLatex",
    "RESULT_AUGMENTERS",
    "register_augmenter",
]
