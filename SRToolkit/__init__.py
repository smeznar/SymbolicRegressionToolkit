"""
Symbolic Regression Toolkit

This package provides a Python-based toolkit for equation discovery/symbolic regression.

Modules:
    dataset: The module containing classes for working with Datasets and Benchmarks.
    utils: The module containing utility classes and functions.
    evaluation: The module containing classes and functions for estimating parameters and evaluating Symbolic Regression models.
"""

__version__ = "1.3.0"
__author__ = "Sebastian Mežnar, Jure Brence"
__credits__ = "Jožef Stefan Institute"

from . import utils
from . import evaluation
from . import dataset