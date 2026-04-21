"""
Sampling strategies for variable generation in symbolic regression benchmarks.
"""

import importlib
from abc import ABC, abstractmethod

import numpy as np


def log_uniform_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    """Sample from log-uniform distribution over both positive and negative ranges."""
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    pos_samples = 10.0 ** np.random.uniform(log10_min, log10_max, size=num_positives)
    neg_samples = -(10.0 ** np.random.uniform(log10_min, log10_max, size=num_negatives))
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def log_uniform_positive_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    """Sample from log-uniform distribution over the positive range."""
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    return 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)


def log_uniform_negative_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    """Sample from log-uniform distribution over the negative range."""
    log10_min = np.log10(abs(min_value))
    log10_max = np.log10(abs(max_value))
    return -(10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size))


def uniform_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over both positive and negative ranges."""
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    pos_samples = np.random.uniform(min_value, max_value, size=num_positives)
    neg_samples = -np.random.uniform(min_value, max_value, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def uniform_positive_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over the positive range."""
    return np.random.uniform(min_value, max_value, size=sample_size)


def uniform_negative_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over the negative range."""
    return -np.random.uniform(min_value, max_value, size=sample_size)


def integer_uniform_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from both positive and negative ranges."""
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    pos_samples = np.random.randint(min_value, max_value, size=num_positives)
    neg_samples = -np.random.randint(min_value, max_value, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def integer_uniform_positive_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from the positive range."""
    return np.random.randint(min_value, max_value, size=sample_size)


def integer_uniform_negative_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from the negative range."""
    return -np.random.randint(min_value, max_value, size=sample_size)


class Sampler(ABC):
    """
    Abstract base class for variable samplers.

    Concrete subclasses must implement ``__call__``,
    [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict], and
    [from_dict][SRToolkit.dataset.sampling.Sampler.from_dict]. The dictionary produced by
    [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict] must include a ``"sampler_class"``
    key holding the fully-qualified class path (e.g.
    ``"SRToolkit.dataset.sampling.UniformSampling"``), so that
    [sampling_from_dict][SRToolkit.dataset.sampling.sampling_from_dict] can reconstruct any
    subclass — including user-defined ones — via ``importlib`` without a central registry.
    """

    @abstractmethod
    def __call__(self, sample_size: int) -> np.ndarray:
        """Draw ``sample_size`` samples and return them as a 1-D numpy array."""

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Serialize this sampler to a JSON-compatible dictionary.

        The returned dict **must** include ``"sampler_class"`` set to the
        fully-qualified class path of this sampler.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Sampler":
        """Reconstruct a sampler from a dictionary produced by [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict]."""


class LogUniformSampling(Sampler):
    """
    Log-uniform sampler with configurable sign constraints.

    Samples from `U(\\log_{10}(\\text{min}), \\log_{10}(\\text{max}))` in log space,
    optionally drawing from positive and/or negative ranges.

    Args:
        min_value: Lower bound of the log-uniform range (must be > 0).
        max_value: Upper bound of the log-uniform range (must be > 0).
        uses_positive: If ``True``, positive samples are included.
        uses_negative: If ``True``, negative samples are included.
    """

    def __init__(self, min_value: float, max_value: float, uses_positive: bool = True, uses_negative: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative

    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return log_uniform_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return log_uniform_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return log_uniform_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.LogUniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "uses_positive": self.uses_positive,
            "uses_negative": self.uses_negative,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogUniformSampling":
        """Deserialize a [LogUniformSampling][SRToolkit.dataset.sampling.LogUniformSampling] from a dictionary produced by [to_dict][SRToolkit.dataset.sampling.LogUniformSampling.to_dict]."""
        return cls(d["min_value"], d["max_value"], d["uses_positive"], d["uses_negative"])


class UniformSampling(Sampler):
    """
    Linear uniform sampler with configurable sign constraints.

    Samples from`U(\\text{min}, \\text{max})`, optionally drawing from positive
    and/or negative ranges.

    Args:
        min_value: Lower bound of the uniform range.
        max_value: Upper bound of the uniform range.
        uses_positive: If ``True``, positive samples are included.
        uses_negative: If ``True``, negative samples are included.
    """

    def __init__(self, min_value: float, max_value: float, uses_positive: bool = True, uses_negative: bool = True):
        self.min_value = min_value
        self.max_value = max_value
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative

    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return uniform_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return uniform_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return uniform_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.UniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "uses_positive": self.uses_positive,
            "uses_negative": self.uses_negative,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UniformSampling":
        """Deserialize a [UniformSampling][SRToolkit.dataset.sampling.UniformSampling] from a dictionary produced by [to_dict][SRToolkit.dataset.sampling.UniformSampling.to_dict]."""
        return cls(d["min_value"], d["max_value"], d["uses_positive"], d["uses_negative"])


class IntegerUniformSampling(Sampler):
    """
    Integer uniform sampler with configurable sign constraints.

    Samples integers from :math:`\\{\\text{min}, ..., \\text{max}-1\\}`, optionally drawing
    from positive and/or negative ranges.

    Args:
        min_value: Lower bound of the integer range.
        max_value: Upper bound (exclusive) of the integer range.
        uses_positive: If ``True``, positive samples are included.
        uses_negative: If ``True``, negative samples are included.
    """

    def __init__(self, min_value: int, max_value: int, uses_positive: bool = True, uses_negative: bool = True):
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        assert uses_positive or uses_negative
        self.uses_positive = uses_positive
        self.uses_negative = uses_negative

    def __call__(self, sample_size):
        if self.uses_positive and self.uses_negative:
            return integer_uniform_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return integer_uniform_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return integer_uniform_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.IntegerUniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "uses_positive": self.uses_positive,
            "uses_negative": self.uses_negative,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntegerUniformSampling":
        """Deserialize a [IntegerUniformSampling][SRToolkit.dataset.sampling.IntegerUniformSampling] from a dictionary produced by [to_dict][SRToolkit.dataset.sampling.IntegerUniformSampling.to_dict]."""
        return cls(d["min_value"], d["max_value"], d["uses_positive"], d["uses_negative"])


def sampling_from_dict(d: dict) -> Sampler:
    """
    Deserialize a sampler from a dictionary produced by its [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict] method.

    Uses ``importlib`` to load the class from the ``"sampler_class"`` key, so any
    user-defined [Sampler][SRToolkit.dataset.sampling.Sampler] subclass round-trips without a central registry.

    Args:
        d: Dictionary with a ``"sampler_class"`` key (fully-qualified class path, e.g.
            ``"SRToolkit.dataset.sampling.UniformSampling"``) and the sampler's parameters.

    Returns:
        A reconstructed [Sampler][SRToolkit.dataset.sampling.Sampler] instance.

    Raises:
        KeyError: If ``"sampler_class"`` is missing from ``d``.
        ImportError: If the module cannot be imported.
        AttributeError: If the class cannot be found in the module.
    """
    module_path, cls_name = d["sampler_class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls.from_dict(d)
