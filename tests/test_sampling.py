"""Tests for SRToolkit.dataset.sampling (spec-first; see the module docstrings)."""

import numpy as np
import pytest
from scipy import stats

from SRToolkit.dataset.sampling import IntegerUniformSampling, LogUniformSampling, Sampler, UniformSampling


@pytest.mark.parametrize(
    "sampler, text",
    [
        (UniformSampling(1, 5, mirror=True), "uniform on [-5, -1] ∪ [1, 5]"),
        (UniformSampling(1, 5), "uniform on [1, 5]"),
        (UniformSampling(-5, -1), "uniform on [-5, -1]"),
        (UniformSampling(-2, 5), "uniform on [-2, 5]"),
        (UniformSampling(0, 5, mirror=True), "uniform on [-5, 5]"),
        (UniformSampling(0.5, 2.5), "uniform on [0.5, 2.5]"),
        (UniformSampling(1, 1234567), "uniform on [1, 1234567]"),
        (UniformSampling(0.1234567, 2), "uniform on [0.1234567, 2]"),
        (LogUniformSampling(0.1, 10, mirror=True), "log-uniform on [-10, -0.1] ∪ [0.1, 10]"),
        (LogUniformSampling(0.1, 10), "log-uniform on [0.1, 10]"),
        (LogUniformSampling(-10, -0.1), "log-uniform on [-10, -0.1]"),
        (IntegerUniformSampling(1, 99, mirror=True), "integers from -99 to -1 and from 1 to 99, uniform"),
        (IntegerUniformSampling(1, 99), "integers from 1 to 99, uniform"),
        (IntegerUniformSampling(-99, -1), "integers from -99 to -1, uniform"),
    ],
    ids=[
        "uniform-mirror",
        "uniform-positive",
        "uniform-negative",
        "uniform-across-zero",
        "uniform-mirror-from-zero",
        "uniform-floats",
        "uniform-large-not-rounded",
        "uniform-precise-not-rounded",
        "log-mirror",
        "log-positive",
        "log-negative",
        "integer-mirror",
        "integer-positive",
        "integer-negative",
    ],
)
def test_builtin_describe(sampler, text):
    assert sampler.describe() == text


# Draws from the global state, pinned from the pre-mirror code (570f5f6) with np.random.seed(1):
# datasets sampled without an rng must regenerate identically.
@pytest.mark.parametrize(
    "sampler, expected",
    [
        (
            UniformSampling(1, 5, mirror=True),
            [-2.382242908172191, 1.5870235632684522, -1.3693543790751912, -1.7450408455106836],
        ),
        (UniformSampling(1, 5), [2.668088018810296, 3.8812979737686324, 1.0004574992693795, 2.209330290527359]),
        (
            LogUniformSampling(0.1, 10, mirror=True),
            [-0.49104518184659673, 0.19656753051200318, -0.1529949829431263, -0.23578730726467376],
        ),
        (
            LogUniformSampling(-1e-27, -1e-29),
            [-6.82407842709882e-29, -2.758347554916672e-28, -1.0005268542378315e-29, -4.024066464714621e-29],
        ),
        (IntegerUniformSampling(1, 99, mirror=True), [-17, -2, -65, 80]),
    ],
    ids=["uniform-mirror", "uniform", "log-mirror", "log-negative", "integer-mirror"],
)
def test_global_state_draws_unchanged(sampler, expected):
    np.random.seed(1)
    assert sampler(4).tolist() == expected


_SAMPLERS = [
    UniformSampling(-2, 5),
    UniformSampling(1, 5, mirror=True),
    LogUniformSampling(0.1, 10),
    LogUniformSampling(-10, -0.1),
    LogUniformSampling(0.1, 10, mirror=True),
    IntegerUniformSampling(-3, 6),
    IntegerUniformSampling(1, 9, mirror=True),
]
_IDS = ["uniform", "uniform-mirror", "log", "log-negative", "log-mirror", "integer", "integer-mirror"]


@pytest.mark.parametrize("sampler", _SAMPLERS, ids=_IDS)
def test_rng_reproducible_and_leaves_global_state(sampler):
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    a = sampler(50, rng=np.random.default_rng(7))
    b = sampler(50, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], before)


@pytest.mark.parametrize("sampler", _SAMPLERS, ids=_IDS)
def test_ppf_matches_call_distribution(sampler):
    # ppf pushed through uniforms is distributed like __call__ (two-sample KS, fixed seeds).
    rng = np.random.default_rng(0)
    from_ppf = sampler.ppf(rng.random(4000))
    from_call = sampler(4000, rng=np.random.default_rng(1))
    assert stats.ks_2samp(from_ppf, from_call).pvalue > 0.01


@pytest.mark.parametrize(
    "sampler, at_zero, near_one",
    [
        (UniformSampling(-2, 5), -2, 5),
        (UniformSampling(1, 5, mirror=True), -5, 5),
        (LogUniformSampling(0.1, 10), 0.1, 10),
        (LogUniformSampling(-10, -0.1), -10, -0.1),
        (IntegerUniformSampling(-3, 6), -3, 6),
        (IntegerUniformSampling(1, 9, mirror=True), -9, 9),
    ],
    ids=["uniform", "uniform-mirror", "log", "log-negative", "integer", "integer-mirror"],
)
def test_ppf_support_edges(sampler, at_zero, near_one):
    low, high = sampler.ppf(np.array([0.0, 1 - 1e-12]))
    assert low == pytest.approx(at_zero)
    assert high == pytest.approx(near_one, rel=1e-6)


def test_ppf_mirror_splits_at_half():
    s = UniformSampling(1, 5, mirror=True)
    assert np.all(s.ppf(np.linspace(0, 0.499, 50)) <= -1)
    assert np.all(s.ppf(np.linspace(0.5, 0.999, 50)) >= 1)


@pytest.mark.parametrize(
    "sampler, support",
    [
        (UniformSampling(1, 5, mirror=True), lambda x: (np.abs(x) >= 1) & (np.abs(x) <= 5)),
        (LogUniformSampling(0.1, 10, mirror=True), lambda x: (np.abs(x) >= 0.1) & (np.abs(x) <= 10)),
        (IntegerUniformSampling(1, 9, mirror=True), lambda x: (np.abs(x) >= 1) & (np.abs(x) <= 9)),
    ],
    ids=["uniform", "log", "integer"],
)
def test_mirror_support_and_half_split(sampler, support):
    x = sampler(4000, rng=np.random.default_rng(0))
    assert np.all(support(x))
    assert abs(np.mean(x < 0) - 0.5) < 0.03


def test_plain_samples_interval_as_written():
    x = UniformSampling(-2, 5)(4000, rng=np.random.default_rng(0))
    assert x.min() >= -2 and x.max() <= 5 and np.mean(x < 0) == pytest.approx(2 / 7, abs=0.03)


@pytest.mark.parametrize(
    "make",
    [
        lambda: UniformSampling(-1, 5, mirror=True),
        lambda: UniformSampling(5, 1),
        lambda: LogUniformSampling(0, 10),
        lambda: LogUniformSampling(-1, 10),
        lambda: LogUniformSampling(-10, -1, mirror=True),
        lambda: IntegerUniformSampling(-1, 5, mirror=True),
        lambda: IntegerUniformSampling(0, 5, mirror=True),
        lambda: IntegerUniformSampling(5, 4),
    ],
    ids=[
        "mirror-negative-low",
        "low-above-high",
        "log-zero",
        "log-mixed-sign",
        "log-mirror-negative",
        "integer-mirror-negative",
        "integer-mirror-zero",
        "integer-empty",
    ],
)
def test_invalid_bounds_raise(make):
    with pytest.raises(ValueError):
        make()


def test_integer_bounds_inclusive():
    # Both bounds are drawn; ppf gives the same values as int64.
    s = IntegerUniformSampling(-1, 1)
    assert set(s(200, rng=np.random.default_rng(0)).tolist()) == {-1, 0, 1}
    q = s.ppf(np.array([0.0, 0.5, 1 - 1e-12]))
    assert q.dtype == np.int64 and q.tolist() == [-1, 0, 1]


def test_integer_error_names_given_max():
    with pytest.raises(ValueError, match=r"max_value \(4\)"):
        IntegerUniformSampling(5, 4)


@pytest.mark.parametrize("sampler", _SAMPLERS, ids=_IDS)
def test_dict_round_trip(sampler):
    assert Sampler.from_dict(sampler.to_dict()).to_dict() == sampler.to_dict()


@pytest.mark.parametrize("sampler", _SAMPLERS, ids=_IDS)
def test_equal_by_to_dict(sampler):
    loaded = Sampler.from_dict(sampler.to_dict())
    assert loaded == sampler and hash(loaded) == hash(sampler)
    assert len({loaded, sampler}) == 1


def test_unequal_parameters_class_or_type():
    assert UniformSampling(0, 2) == UniformSampling(0.0, 2.0)
    assert UniformSampling(0, 2) != UniformSampling(0, 3)
    assert UniformSampling(1, 2) != LogUniformSampling(1, 2)
    assert UniformSampling(0, 2) != {"min_value": 0, "max_value": 2}


class _Constant(Sampler):
    """A custom sampler that overrides neither describe nor ppf."""

    def __init__(self, value):
        self.value = value

    def __call__(self, sample_size, rng=None):
        return [self.value] * sample_size

    def to_dict(self):
        return {"sampler_class": f"{__name__}._Constant", "value": self.value}


class _UniformChild(UniformSampling):
    """A subclass that keeps UniformSampling.to_dict."""


def test_custom_sampler_equal_and_hashable():
    # Reason: implied (equal samplers hash equally, also for user subclasses whose to_dict holds
    # an unhashable list or that inherit to_dict).
    assert _Constant([1, 2]) == _Constant([1, 2]) and _Constant([1, 2]) != _Constant([1, 3])
    assert len({_Constant([1, 2]), _Constant([1, 2])}) == 1
    child = _UniformChild(0, 2)
    assert child == UniformSampling(0, 2) and hash(child) == hash(UniformSampling(0, 2))


def test_default_describe_from_to_dict():
    assert _Constant(3).describe() == "_Constant(value=3)"


def test_default_ppf_raises():
    with pytest.raises(NotImplementedError):
        _Constant(3).ppf(np.array([0.5]))
