"""
Symbolic Regression Toolkit

This package provides a Python-based toolkit for equation discovery/symbolic regression.

Modules:
    dataset: The module containing classes for working with Datasets and Benchmarks.
    utils: The module contains utility classes and functions.
    evaluation: The module contains classes and functions for estimating parameters and evaluating Symbolic Regression models.
    approaches: The module containing SR approach implementations and configurations.
    experiments: The module containing the job-based experiment runner for multi-dataset, multi-approach experiments.
"""

__version__ = "1.4.0"
__license__ = "GPL-3.0 License"
__author__ = "Sebastian Mežnar, Jure Brence"
__credits__ = "Jožef Stefan Institute"

from . import approaches, dataset, evaluation, experiments, utils

__all__ = utils.__all__ + evaluation.__all__ + dataset.__all__ + approaches.__all__ + experiments.__all__
