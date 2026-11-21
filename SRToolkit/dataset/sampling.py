"""Sampling strategies for variable generation in symbolic regression benchmarks."""

from abc import ABC, abstractmethod

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind, _import_class


def _check_mirror(min_value, max_value, mirror: bool, name: str) -> None:
    """Raise if ``min_value > max_value``, or if ``mirror`` is set and ``min_value < 0``."""
    if min_value > max_value:
        raise ValueError(f"[{name}] min_value ({min_value}) is greater than max_value ({max_value}).")
    if mirror and min_value < 0:
        raise ValueError(
            f"[{name}] mirror=True requires min_value >= 0 (got {min_value}); the mirrored halves would overlap."
        )


def _draw(sample_size: int, draw, mirror: bool, rng) -> np.ndarray:
    """Draw ``sample_size`` values with ``draw(n)``, negating each with probability ½ when ``mirror`` is set."""
    if not mirror:
        return draw(sample_size)
    src = np.random if rng is None else rng
    num_positives = int(np.count_nonzero(src.uniform(0.0, 1.0, size=sample_size) > 0.5))
    all_samples = np.concatenate([draw(num_positives), -draw(sample_size - num_positives)])
    src.shuffle(all_samples)
    return all_samples


def _mirror_ppf(u, ppf, mirror: bool) -> np.ndarray:
    """Apply ``ppf``; with ``mirror``, ``u < ½`` maps onto the negated interval and ``u >= ½`` onto the interval."""
    u = np.asarray(u, dtype=float)
    if not mirror:
        return ppf(u)
    return np.where(u < 0.5, -ppf(1.0 - 2.0 * u), ppf(2.0 * u - 1.0))


def _intervals(low: float, high: float, mirror: bool) -> str:
    """``[low, high]``, or with ``mirror`` also ``[-high, -low]`` (one interval if ``low == 0``)."""
    if not mirror:
        return f"[{low:.15g}, {high:.15g}]"
    if low == 0:
        return f"[{-high:.15g}, {high:.15g}]"
    return f"[{-high:.15g}, {-low:.15g}] ∪ [{low:.15g}, {high:.15g}]"


def _sample_inputs(samplers, n: int) -> np.ndarray:
    """Draw ``n`` input rows, one column per sampler: the single place inputs are sampled."""
    return np.column_stack([s(n) for s in samplers])


class Sampler(ABC):
    """Abstract base class for variable samplers.

    Concrete subclasses must implement ``__call__``,
    [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict], and
    [from_dict][SRToolkit.dataset.sampling.Sampler.from_dict], and may implement
    [ppf][SRToolkit.dataset.sampling.Sampler.ppf]. The dictionary produced by
    [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict] must include a ``"sampler_class"``
    key holding the fully-qualified class path (e.g.
    ``"SRToolkit.dataset.sampling.UniformSampling"``), so that
    [Sampler.from_dict][SRToolkit.dataset.sampling.Sampler.from_dict] can reconstruct any
    subclass — including user-defined ones — via ``importlib`` without a central registry.
    """

    @abstractmethod
    def __call__(self, sample_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``sample_size`` samples and return them as a 1-D numpy array.

        Args:
            sample_size: Number of samples.
            rng: Generator to draw from. If ``None``, draws from NumPy's global random state.
        """

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map values ``u`` in ``[0, 1)`` to samples through the inverse CDF.

        Pushing uniform (or stratified, e.g. Latin Hypercube) values through ``ppf`` gives samples
        from this sampler's distribution. Optional: the base implementation raises, and callers fall
        back to ``__call__``.

        Args:
            u: Values in ``[0, 1)``.

        Returns:
            Samples of the same shape as ``u``.

        Raises:
            NotImplementedError: If the sampler does not implement it.
        """
        raise NotImplementedError(f"{type(self).__name__}.ppf is not implemented.")

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary.

        The returned dict **must** include ``"sampler_class"`` set to the
        fully-qualified class path of this sampler.
        """

    def describe(self) -> str:
        """Return a short phrase saying what the sampler draws, for people and LLM prompts.

        [BackgroundKnowledge.describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe]
        lists it under "Input distribution". The base implementation renders
        ``ClassName(key=repr(value), ...)`` from
        [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict] without ``"sampler_class"``; the
        built-ins return a phrase such as ``"uniform on [1, 5]"``.

        Returns:
            The phrase.
        """
        args = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "sampler_class")
        return f"{type(self).__name__}({args})"

    @classmethod
    def from_dict(cls, d: dict) -> "Sampler":
        """Reconstruct a sampler from a dictionary produced by
        [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict].

        When called on the base [Sampler][SRToolkit.dataset.sampling.Sampler] class,
        dispatches to the concrete subclass named by the ``"sampler_class"`` key using
        ``importlib`` — both built-in and user-defined subclasses round-trip without a
        central registry. When called on a concrete subclass, that subclass must override
        this method.

        Args:
            d: Dictionary with a ``"sampler_class"`` key (fully-qualified class path, e.g.
                ``"SRToolkit.dataset.sampling.UniformSampling"``) and the sampler's parameters.

        Returns:
            A reconstructed [Sampler][SRToolkit.dataset.sampling.Sampler] instance.

        Raises:
            KeyError: If ``"sampler_class"`` is missing from ``d`` (dispatch path).
            ImportError: If the class cannot be imported (dispatch path).
            TypeError: If the class is not a Sampler subclass (dispatch path).
            NotImplementedError: If called on a subclass that has not overridden this method.
        """
        if cls is Sampler:
            d = _auto_bind(d)
            resolved = _import_class(d["sampler_class"], "sampler", Sampler)
            return resolved.from_dict(d)
        raise NotImplementedError(f"{cls.__name__}.from_dict is not implemented.")


class LogUniformSampling(Sampler):
    """Log-uniform sampler: ``log10(|x|)`` is uniform, the sign is that of the bounds.

    Samples ``[min_value, max_value]`` with a log-uniform magnitude. Both bounds must be nonzero and
    of the same sign; negative bounds (e.g. ``(-10, -0.1)``) give negative samples. With
    ``mirror=True`` (positive bounds only), each sample is negated with probability ½, so the
    support is ``[-max_value, -min_value] ∪ [min_value, max_value]``.

    Args:
        min_value: Lower bound.
        max_value: Upper bound.
        mirror: If ``True``, also sample the negated interval.

    Raises:
        ValueError: If a bound is zero, the bounds differ in sign, ``min_value > max_value``, or
            ``mirror`` is ``True`` with negative bounds.
    """

    def __init__(self, min_value: float, max_value: float, mirror: bool = False):
        _check_mirror(min_value, max_value, mirror, "LogUniformSampling")
        if min_value == 0 or max_value == 0 or (min_value < 0) != (max_value < 0):
            raise ValueError(
                f"[LogUniformSampling] Bounds must be nonzero and of the same sign (got {min_value}, {max_value})."
            )
        self.min_value = min_value
        self.max_value = max_value
        self.mirror = mirror
        self._negative = min_value < 0
        # log10 of the smallest and largest magnitude
        self._log_low, self._log_high = np.log10(sorted((abs(min_value), abs(max_value))))

    def __call__(self, sample_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        src = np.random if rng is None else rng

        def draw(n):
            magnitudes = 10.0 ** src.uniform(self._log_low, self._log_high, size=n)
            return -magnitudes if self._negative else magnitudes

        return _draw(sample_size, draw, self.mirror, rng)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map ``u`` in ``[0, 1)`` to samples: ``u = 0`` gives ``min_value`` (or ``-max_value`` with ``mirror``).

        Args:
            u: Values in ``[0, 1)``.

        Returns:
            Samples of the same shape as ``u``.
        """

        def ppf(v):
            if self._negative:  # u = 0 is the largest magnitude, i.e. min_value
                return -(10.0 ** (self._log_high - v * (self._log_high - self._log_low)))
            return 10.0 ** (self._log_low + v * (self._log_high - self._log_low))

        return _mirror_ppf(u, ppf, self.mirror)

    def describe(self) -> str:
        """Return ``"log-uniform on <intervals>"``, e.g. ``"log-uniform on [-10, -0.1] ∪ [0.1, 10]"``.

        Returns:
            The phrase.
        """
        return "log-uniform on " + _intervals(self.min_value, self.max_value, self.mirror)

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.LogUniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mirror": self.mirror,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogUniformSampling":
        """Deserialize a [LogUniformSampling][SRToolkit.dataset.sampling.LogUniformSampling] from a dictionary produced by
        [to_dict][SRToolkit.dataset.sampling.LogUniformSampling.to_dict].

        Args:
            d: The serialized sampler (``min_value``, ``max_value``, ``mirror``).

        Returns:
            The reconstructed sampler.
        """
        return cls(d["min_value"], d["max_value"], d["mirror"])


class UniformSampling(Sampler):
    """Uniform sampler on ``[min_value, max_value]``.

    The interval may contain zero (e.g. ``(-2, 5)``). With ``mirror=True`` (``min_value >= 0``
    only), each sample is negated with probability ½, so the support is
    ``[-max_value, -min_value] ∪ [min_value, max_value]``.

    Args:
        min_value: Lower bound.
        max_value: Upper bound.
        mirror: If ``True``, also sample the negated interval.

    Raises:
        ValueError: If ``min_value > max_value``, or ``mirror`` is ``True`` and ``min_value < 0``.
    """

    def __init__(self, min_value: float, max_value: float, mirror: bool = False):
        _check_mirror(min_value, max_value, mirror, "UniformSampling")
        self.min_value = min_value
        self.max_value = max_value
        self.mirror = mirror

    def __call__(self, sample_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        src = np.random if rng is None else rng
        return _draw(sample_size, lambda n: src.uniform(self.min_value, self.max_value, size=n), self.mirror, rng)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map ``u`` in ``[0, 1)`` to ``min_value + u * (max_value - min_value)`` (halved per side with ``mirror``).

        Args:
            u: Values in ``[0, 1)``.

        Returns:
            Samples of the same shape as ``u``.
        """
        low, high = self.min_value, self.max_value
        return _mirror_ppf(u, lambda v: low + v * (high - low), self.mirror)

    def describe(self) -> str:
        """Return ``"uniform on <intervals>"``, e.g. ``"uniform on [-5, -1] ∪ [1, 5]"``.

        A mirrored interval starting at ``0`` is written as one interval (``"uniform on [-5, 5]"``).

        Returns:
            The phrase.
        """
        return "uniform on " + _intervals(self.min_value, self.max_value, self.mirror)

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.UniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mirror": self.mirror,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UniformSampling":
        """Deserialize a [UniformSampling][SRToolkit.dataset.sampling.UniformSampling] from a dictionary produced by
        [to_dict][SRToolkit.dataset.sampling.UniformSampling.to_dict].

        Args:
            d: The serialized sampler (``min_value``, ``max_value``, ``mirror``).

        Returns:
            The reconstructed sampler.
        """
        return cls(d["min_value"], d["max_value"], d["mirror"])


class IntegerUniformSampling(Sampler):
    """Uniform sampler on the integers ``min_value, ..., max_value`` (both included).

    The range may contain zero. With ``mirror=True`` (``min_value >= 1`` only, so that ``0`` is not
    drawn from both halves), each sample is negated with probability ½.

    Args:
        min_value: Lower bound.
        max_value: Upper bound.
        mirror: If ``True``, also sample the negated range.

    Raises:
        ValueError: If ``min_value > max_value``, or ``mirror`` is ``True`` and ``min_value < 1``.
    """

    def __init__(self, min_value: int, max_value: int, mirror: bool = False):
        min_value, max_value = int(min_value), int(max_value)
        _check_mirror(min_value, max_value, mirror, "IntegerUniformSampling")
        if mirror and min_value == 0:
            raise ValueError(
                "[IntegerUniformSampling] mirror=True requires min_value >= 1 (got 0); 0 would be drawn from both halves."
            )
        self.min_value = min_value
        self.max_value = max_value
        self.mirror = mirror

    def __call__(self, sample_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        integers = np.random.randint if rng is None else rng.integers  # both exclude the upper bound
        return _draw(sample_size, lambda n: integers(self.min_value, self.max_value + 1, size=n), self.mirror, rng)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Map ``u`` in ``[0, 1)`` to ``floor(min_value + u * (max_value - min_value + 1))`` (halved per side with ``mirror``).

        Args:
            u: Values in ``[0, 1)``.

        Returns:
            Integer samples (``int64``) of the same shape as ``u``.
        """
        low, high = self.min_value, self.max_value
        # The clip keeps u -> 1 (reached by the mirrored half at u = 0) at max_value.
        return _mirror_ppf(u, lambda v: np.minimum(np.floor(low + v * (high - low + 1)), high), self.mirror).astype(
            np.int64
        )

    def describe(self) -> str:
        """Return ``"integers from <a> to <b>, uniform"``.

        With ``mirror``: ``"integers from -<b> to -<a> and from <a> to <b>, uniform"``.

        Returns:
            The phrase.
        """
        low, high = self.min_value, self.max_value
        span = f"from {low} to {high}"
        return "integers " + (f"from {-high} to {-low} and " if self.mirror else "") + span + ", uniform"

    def to_dict(self) -> dict:
        """Serialize this sampler to a JSON-compatible dictionary."""
        return {
            "sampler_class": "SRToolkit.dataset.sampling.IntegerUniformSampling",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mirror": self.mirror,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntegerUniformSampling":
        """Deserialize a [IntegerUniformSampling][SRToolkit.dataset.sampling.IntegerUniformSampling] from a dictionary produced by
        [to_dict][SRToolkit.dataset.sampling.IntegerUniformSampling.to_dict].

        Args:
            d: The serialized sampler (``min_value``, ``max_value``, ``mirror``).

        Returns:
            The reconstructed sampler.
        """
        return cls(d["min_value"], d["max_value"], d["mirror"])
