"""
Datasets and benchmarks for symbolic regression.

A dataset ([SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset]) represents a single equation
with associated data, a symbol library, and evaluation constraints. A benchmark
([SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark]) is a collection of datasets.

Modules:
    sr_dataset: [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] — wraps input data and
        evaluation settings for a single equation discovery problem.
    sr_benchmark: [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] — manages a
        collection of datasets.
    feynman: [Feynman][SRToolkit.dataset.feynman.Feynman] — 100-equation physics benchmark.
    nguyen: [Nguyen][SRToolkit.dataset.nguyen.Nguyen] — 10-equation polynomial/trig benchmark.
"""

from .feynman import Feynman
from .nguyen import Nguyen
from .sr_benchmark import SR_benchmark
from .sr_dataset import SR_dataset

__all__ = ["SR_dataset", "SR_benchmark", "Feynman", "Nguyen"]
