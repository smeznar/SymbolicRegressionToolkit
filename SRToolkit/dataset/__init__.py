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
    srsd_feynman: [SRSD_Feynman][SRToolkit.dataset.srsd_feynman.SRSD_Feynman] — 120-equation
        SRSD physics benchmark with per-variable sampling strategies.
    sampling: [DefaultSampling][SRToolkit.dataset.sampling.DefaultSampling],
        [SimpleSampling][SRToolkit.dataset.sampling.SimpleSampling],
        [IntegerSampling][SRToolkit.dataset.sampling.IntegerSampling] — variable samplers.
"""

from .feynman import Feynman
from .nguyen import Nguyen
from .sampling import DefaultSampling, IntegerSampling, SimpleSampling
from .sr_benchmark import SR_benchmark
from .sr_dataset import SR_dataset
from .srsd_feynman import SRSD_Feynman

__all__ = [
    "SR_dataset",
    "SR_benchmark",
    "Feynman",
    "Nguyen",
    "SRSD_Feynman",
    "DefaultSampling",
    "SimpleSampling",
    "IntegerSampling",
]
