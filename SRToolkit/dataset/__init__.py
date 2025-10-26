"""
This module contains data sets and benchmarks for symbolic regression. A dataset represents a single equation with specific data, constraints for evaluation, etc.
A benchmark is a collection of datasets. Our library provides the user with two modified version of popular equation discovery benchmarks. Specifically Feynman and Nguyen.

Modules:
    sr_dataset: The module containing the SRDataset class, which can be used to create a dataset and easily evaluate equation discovery approaches.
    sr_benchmark: The module containing the SRBenchmark class, which can be used to create a benchmark i.e. a collection of datasets.
"""

from .sr_dataset import SR_dataset
from .sr_benchmark import SR_benchmark

__all__ = ["SR_dataset", "SR_benchmark"]
