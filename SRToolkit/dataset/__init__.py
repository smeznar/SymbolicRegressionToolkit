"""Datasets and benchmarks for symbolic regression.

A dataset ([Dataset][SRToolkit.dataset.dataset.Dataset]) represents a single equation
with associated data, a symbol library, and evaluation constraints. A benchmark
([Benchmark][SRToolkit.dataset.benchmark.Benchmark]) is a collection of datasets.

Modules:
    dataset: [Dataset][SRToolkit.dataset.dataset.Dataset] — wraps input data and
        evaluation settings for a single equation discovery problem.
    background_knowledge: [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]
        and [VariableInfo][SRToolkit.dataset.background_knowledge.VariableInfo] — what a dataset
        tells the approach besides the data (symbols, input distribution, hints, descriptions).
    benchmark: [Benchmark][SRToolkit.dataset.benchmark.Benchmark] — manages a
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
    data_split: [DataSplit][SRToolkit.dataset.data_split.DataSplit] — named arrays (e.g. a
        dataset's held-out test data) with an optional data source.
    data_cache: dataset cache management — list, gc, remove, refresh, dataset_path —
        plus the materialisation engine used internally by the dataset machinery.
"""

from . import data_cache
from .background_knowledge import BackgroundKnowledge, VariableInfo
from .benchmark import Benchmark
from .data_source import DataSource, FallbackSource, SampleSource, UrlSource
from .data_split import DataSplit
from .dataset import Dataset
from .feynman import Feynman
from .nguyen import Nguyen
from .sampling import IntegerUniformSampling, LogUniformSampling, Sampler, UniformSampling
from .srsd_feynman import SRSD_Feynman

__all__ = [
    "Dataset",
    "Benchmark",
    "BackgroundKnowledge",
    "VariableInfo",
    "Feynman",
    "Nguyen",
    "SRSD_Feynman",
    "Sampler",
    "LogUniformSampling",
    "UniformSampling",
    "IntegerUniformSampling",
    "DataSource",
    "UrlSource",
    "SampleSource",
    "FallbackSource",
    "DataSplit",
    "data_cache",
]
