"""
Symbolic Regression Toolkit

This package provides a Python-based toolkit for equation discovery/symbolic regression.

Modules:
    dataset: The module containing classes for working with Datasets and Benchmarks.
    utils: The module contains utility classes and functions.
    evaluation: The module contains classes and functions for estimating parameters and evaluating Symbolic Regression models.
    approaches: The module containing SR approach implementations and configurations.
    experiments: The module containing the job-based experiment runner for multi-dataset, multi-approach experiments.
    bundle: Tools for packing, installing, loading, and sharing user-defined bundles.

The dataset cache management utilities (list, gc, dataset_path, refresh) live in
[data_cache][SRToolkit.dataset.data_cache].
"""

import sys as _sys

from . import approaches, bundle, dataset, evaluation, experiments, utils
from .bundle._store import bundles_root as _bundles_root

_bundles_parent = str(_bundles_root().parent)
if _bundles_parent not in _sys.path:
    _sys.path.insert(0, _bundles_parent)

__version__ = "1.5.0"
__license__ = "GPL-3.0 License"
__author__ = "Sebastian Mežnar, Jure Brence"
__credits__ = "Jožef Stefan Institute"


__all__ = (
    utils.__all__ + evaluation.__all__ + dataset.__all__ + approaches.__all__ + experiments.__all__ + bundle.__all__
)
