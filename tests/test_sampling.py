"""Tests for SRToolkit.dataset.sampling (spec-first; see the module docstrings)."""

import pytest

from SRToolkit.dataset.sampling import IntegerUniformSampling, LogUniformSampling, Sampler, UniformSampling


@pytest.mark.parametrize(
    "sampler, text",
    [
        (UniformSampling(1, 5), "uniform on [-5, -1] ∪ [1, 5]"),
        (UniformSampling(1, 5, uses_negative=False), "uniform on [1, 5]"),
        (UniformSampling(1, 5, uses_positive=False), "uniform on [-5, -1]"),
        (UniformSampling(0, 5), "uniform on [-5, 5]"),
        (UniformSampling(0.5, 2.5, uses_negative=False), "uniform on [0.5, 2.5]"),
        (UniformSampling(1, 1234567, uses_negative=False), "uniform on [1, 1234567]"),
        (UniformSampling(0.1234567, 2, uses_negative=False), "uniform on [0.1234567, 2]"),
        (LogUniformSampling(0.1, 10), "log-uniform on [-10, -0.1] ∪ [0.1, 10]"),
        (LogUniformSampling(0.1, 10, uses_negative=False), "log-uniform on [0.1, 10]"),
        (IntegerUniformSampling(1, 100), "integers from -99 to -1 and from 1 to 99, uniform"),
        (IntegerUniformSampling(1, 100, uses_negative=False), "integers from 1 to 99, uniform"),
        (IntegerUniformSampling(1, 100, uses_positive=False), "integers from -99 to -1, uniform"),
    ],
    ids=[
        "uniform-both",
        "uniform-positive",
        "uniform-negative",
        "uniform-from-zero",
        "uniform-floats",
        "uniform-large-not-rounded",
        "uniform-precise-not-rounded",
        "log-both",
        "log-positive",
        "integer-both",
        "integer-positive",
        "integer-negative",
    ],
)
def test_builtin_describe(sampler, text):
    # The magnitude is in [min_value, max_value] (integers: max_value excluded); the sign is chosen
    # separately, so a two-signed sampler covers the mirrored range too.
    assert sampler.describe() == text


class _Constant(Sampler):
    """A custom sampler that does not override describe."""

    def __init__(self, value):
        self.value = value

    def __call__(self, sample_size):
        return [self.value] * sample_size

    def to_dict(self):
        return {"sampler_class": f"{__name__}._Constant", "value": self.value}


def test_default_describe_from_to_dict():
    assert _Constant(3).describe() == "_Constant(value=3)"
