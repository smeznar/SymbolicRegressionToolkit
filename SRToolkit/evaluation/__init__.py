"""
This module contains classes and functions for evaluating symbolic regression approaches. Mainly it contains classes that
can be used for parameter estimation and evaluation of mathematical expressions on some dataset.

Modules:
    parameter_estimator: The module containing classes and functions for parameter estimation.
    sr_evaluator: The module containing classes and functions for expressions on some dataset.
    result_augmentation: The module containing classes and functions for result augmentation.
"""

from .parameter_estimator import ParameterEstimator
from .sr_evaluator import SR_evaluator
from .result_augmentation import ResultAugmenter, ExpressionSimplifier, RMSE

__all__ = ["ParameterEstimator", "SR_evaluator", "ResultAugmenter", "ExpressionSimplifier", "RMSE"]