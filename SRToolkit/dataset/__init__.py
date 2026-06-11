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
    sampling: [LogUniformSampling][SRToolkit.dataset.sampling.LogUniformSampling],
        [UniformSampling][SRToolkit.dataset.sampling.UniformSampling],
        [IntegerUniformSampling][SRToolkit.dataset.sampling.IntegerUniformSampling] — variable
        samplers with serialisation support.
    data_source: [DataSource][SRToolkit.dataset.data_source.DataSource] and its concrete
        types [UrlSource][SRToolkit.dataset.data_source.UrlSource] and
        [SampleSource][SRToolkit.dataset.data_source.SampleSource] — describe where a
        dataset's cached data originates.
    data_cache: dataset cache management — list, gc, remove, refresh, dataset_path —
        plus the materialisation engine used internally by the dataset machinery.
"""

from . import data_cache
from .data_source import DataSource, SampleSource, UrlSource, source_from_dict
from .feynman import Feynman
from .nguyen import Nguyen
from .sampling import IntegerUniformSampling, LogUniformSampling, Sampler, UniformSampling, sampler_from_dict
from .sr_benchmark import SR_benchmark
from .sr_dataset import SR_dataset
from .srsd_feynman import SRSD_Feynman

__all__ = [
    "SR_dataset",
    "SR_benchmark",
    "Feynman",
    "Nguyen",
    "SRSD_Feynman",
    "Sampler",
    "LogUniformSampling",
    "UniformSampling",
    "IntegerUniformSampling",
    "sampler_from_dict",
    "DataSource",
    "UrlSource",
    "SampleSource",
    "source_from_dict",
    "data_cache",
]
