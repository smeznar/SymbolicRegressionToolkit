"""
Sampling strategies for variable generation in symbolic regression benchmarks.
"""

import numpy as np


def default_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
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


def default_positive_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    """Sample from log-uniform distribution over the positive range."""
    log10_min = np.log10(min_value)
    log10_max = np.log10(max_value)
    return 10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size)


def default_negative_sampling(sample_size, min_value=1.0e-1, max_value=1.0e1):
    """Sample from log-uniform distribution over the negative range."""
    log10_min = np.log10(abs(min_value))
    log10_max = np.log10(abs(max_value))
    return -(10.0 ** np.random.uniform(log10_min, log10_max, size=sample_size))


def simple_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over both positive and negative ranges."""
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    pos_samples = np.random.uniform(min_value, max_value, size=num_positives)
    neg_samples = -np.random.uniform(min_value, max_value, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def simple_positive_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over the positive range."""
    return np.random.uniform(min_value, max_value, size=sample_size)


def simple_negative_sampling(sample_size, min_value=0.0, max_value=1.0):
    """Sample from linear uniform distribution over the negative range."""
    return -np.random.uniform(min_value, max_value, size=sample_size)


def integer_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from both positive and negative ranges."""
    num_positives = sum(np.random.uniform(0.0, 1.0, size=sample_size) > 0.5)
    num_negatives = sample_size - num_positives
    pos_samples = np.random.randint(min_value, max_value, size=num_positives)
    neg_samples = -np.random.randint(min_value, max_value, size=num_negatives)
    all_samples = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(all_samples)
    return all_samples


def integer_positive_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from the positive range."""
    return np.random.randint(min_value, max_value, size=sample_size)


def integer_negative_sampling(sample_size, min_value=1, max_value=100):
    """Sample integers from the negative range."""
    return -np.random.randint(min_value, max_value, size=sample_size)


class DefaultSampling:
    """
    Log-uniform sampler with configurable sign constraints.

    Samples from :math:`U(\\log_{10}(\\text{min}), \\log_{10}(\\text{max}))` in log space,
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
            return default_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return default_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return default_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )


class SimpleSampling:
    """
    Linear uniform sampler with configurable sign constraints.

    Samples from :math:`U(\\text{min}, \\text{max})`, optionally drawing from positive
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
            return simple_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return simple_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return simple_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )


class IntegerSampling:
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
            return integer_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_positive:
            return integer_positive_sampling(sample_size, self.min_value, self.max_value)
        elif self.uses_negative:
            return integer_negative_sampling(sample_size, self.min_value, self.max_value)
        raise AttributeError(
            f"Either uses_positive ({self.uses_positive}) or uses_negative ({self.uses_negative}) must be True"
        )
