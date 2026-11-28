"""Specifications of free constants: priors, bounds and fixed values.

A **free constant** is a token of type ``"const"`` (``C`` in the default
[SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary]) whose value is fitted to the data. It
is unrelated to a **named constant** (a ``"lit"`` symbol such as ``pi``), whose value is fixed by
the library.

A bare ``C`` is a new parameter at every occurrence; a **labelled** constant ``C_<int>`` (e.g.
``C_5``) is one parameter however often it occurs. Labels are local to an expression: ``C_0`` in two
expressions are unrelated parameters. The parameters of an expression form a vector ordered by first
appearance, as listed by
[Expression.parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels].

A [Constant][SRToolkit.utils.free_constants.Constant] describes how one labelled constant is
treated; an [Expression][SRToolkit.utils.expression.Expression] carries the specs of its labelled
constants. A spec gives the prior its starting points (and BED samples) are drawn from, the hard
bounds of the fit, an optional value, and whether that value is fixed instead of fitted. Specs
address constants by label, so to specify a constant write it labelled (``C_0`` instead of ``C``).
A bare ``C``, and a labelled constant without a spec, use the **default spec**: a ``Constant``
template (no label) that applies to every parameter without a spec of its own. A spec's unset ``prior`` or ``bounds``
are also taken from it, so a fit is unbounded only when infinite bounds are given explicitly. A
dataset's default spec is
[BackgroundKnowledge.default_constant][SRToolkit.dataset.background_knowledge.BackgroundKnowledge];
[uniform_default][SRToolkit.utils.free_constants.uniform_default] builds the usual one, a uniform
prior on a constant range with that range as bounds.
[Expression.resolve][SRToolkit.utils.expression.Expression.resolve] combines an expression's specs
with the default spec into one complete spec per parameter.

Without specs, how constants are written does not matter: ``C_0*X_0 + C_0`` and ``C_1*X_0 + C_1``
are the same model. [canonicalise][SRToolkit.utils.expression.canonicalise] writes them one way
(a constant occurring once bare, a shared one labelled), e.g. to deduplicate candidates of an
approach that writes labelled constants (the evaluator setting ``canonicalise_constants``).
"""

import json
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this package, and expression
    # imports this module.
    from SRToolkit.dataset.sampling import Sampler

    from .expression import Expression


@dataclass(frozen=True)
class Constant:
    """Specification of one free constant.

    A spec with a ``label`` applies to every occurrence of that labelled constant in an expression.
    A spec without one is a **template**. A template used as the default spec (applied to every
    parameter that has no spec of its own, see
    [uniform_default][SRToolkit.utils.free_constants.uniform_default]) must set ``prior`` and
    ``bounds`` and leave ``value`` and ``fixed`` unset.

    Args:
        label: The labelled-constant token the spec applies to, e.g. ``"C_5"``. ``None`` makes the
            spec a template. Whether the label occurs in an expression is checked when an
            [Expression][SRToolkit.utils.expression.Expression] is built; a token the
            symbol library registers as a symbol (e.g. a named constant ``C_1``) is never a
            labelled constant.
        prior: Distribution that starting points and BED samples are drawn from. ``None``: the
            default spec's prior, filled in by
            [Expression.resolve][SRToolkit.utils.expression.Expression.resolve].
        bounds: Hard ``(lower, upper)`` bounds of the fit: a pair of real numbers (length-2 sequence
            or array) with ``lower < upper``. Either end may be infinite; ``(-inf, inf)`` fits
            without bounds. ``None``: the default spec's bounds, filled in by
            [Expression.resolve][SRToolkit.utils.expression.Expression.resolve].
        value: The constant's value: used as is when ``fixed``, otherwise the first starting point
            of the fit, which must lie in the closed interval ``bounds`` (checked against inherited
            bounds by [Expression.resolve][SRToolkit.utils.expression.Expression.resolve]).
            ``None`` draws the starting point from the prior.
        fixed: If ``True``, the constant is not fitted and ``value`` is used as is, whatever the
            bounds (it must still be finite).

    Attributes:
        label: As given.
        prior: As given.
        bounds: A tuple of two ``float``s, or ``None``.
        value: A ``float``, or ``None``.
        fixed: As given.

    Raises:
        TypeError: If ``label`` is not ``None`` or a string, ``prior`` is not ``None`` or a
            [Sampler][SRToolkit.dataset.sampling.Sampler], ``bounds`` is not ``None`` or a
            pair of real numbers, ``value`` is not ``None`` or a real number, or
            ``fixed`` is not a ``bool`` (``bool`` is not accepted as a number).
        ValueError: If ``bounds`` contains ``nan`` or does not satisfy ``lower < upper``,
            ``value`` is not finite, ``fixed`` is ``True`` without a ``value``, or ``fixed`` is
            ``False`` and ``value`` lies outside ``bounds``.

    Examples:
        >>> from SRToolkit.dataset import UniformSampling
        >>> g = Constant(label="C_0", value=9.81, fixed=True)
        >>> g.fixed
        True
        >>> Constant(label="C_1", prior=UniformSampling(0.0, 2.0), bounds=[0, float("inf")]).bounds
        (0.0, inf)
    """

    label: str | None = None
    prior: "Sampler | None" = None
    bounds: tuple[float, float] | None = None
    value: float | None = None
    fixed: bool = False

    def __post_init__(self) -> None:
        # Imported here: SRToolkit.dataset imports this package.
        from SRToolkit.dataset.sampling import Sampler

        if self.label is not None and not isinstance(self.label, str):
            raise TypeError(f"Constant 'label' must be None or a string, got {self.label!r}.")
        if self.prior is not None and not isinstance(self.prior, Sampler):
            raise TypeError(f"Constant 'prior' must be None or a Sampler, got {self.prior!r}.")
        if self.bounds is not None:
            try:
                pair = tuple(self.bounds)
            except TypeError:
                pair = ()
            if len(pair) != 2 or not all(isinstance(b, numbers.Real) and not isinstance(b, bool) for b in pair):
                raise TypeError(f"Constant 'bounds' must be a pair of real numbers, got {self.bounds!r}.")
            lower, upper = float(pair[0]), float(pair[1])
            if not lower < upper:  # also rejects nan
                raise ValueError(f"Constant 'bounds' must satisfy lower < upper, got {self.bounds!r}.")
            object.__setattr__(self, "bounds", (lower, upper))
        if self.value is not None:
            if not isinstance(self.value, numbers.Real) or isinstance(self.value, bool):
                raise TypeError(f"Constant 'value' must be None or a real number, got {self.value!r}.")
            object.__setattr__(self, "value", float(self.value))
            if not math.isfinite(self.value):
                raise ValueError(f"Constant 'value' must be finite, got {self.value!r}.")
        if not isinstance(self.fixed, bool):
            raise TypeError(f"Constant 'fixed' must be a bool, got {self.fixed!r}.")
        if self.fixed and self.value is None:
            raise ValueError("Constant with fixed=True needs a 'value'.")
        if not self.fixed and self.value is not None and self.bounds is not None:
            lower, upper = self.bounds
            if not lower <= self.value <= upper:
                raise ValueError(
                    f"Constant {self.label!r}: 'value' {self.value!r} lies outside 'bounds' {self.bounds!r}."
                )

    def to_dict(self) -> dict:
        """Return a JSON-safe dict of the spec.

        The prior is stored with its [to_dict][SRToolkit.dataset.sampling.Sampler.to_dict] and the
        bounds as a ``[lower, upper]`` list; infinite bounds stay ``float`` infinities (see
        [serialization][SRToolkit.utils.serialization]).

        Returns:
            A dict with the keys ``"label"``, ``"prior"``, ``"bounds"``, ``"value"`` and
            ``"fixed"``; unset fields are ``None``.

        Examples:
            >>> Constant(label="C_0", value=1.5).to_dict()
            {'label': 'C_0', 'prior': None, 'bounds': None, 'value': 1.5, 'fixed': False}
        """
        return {
            "label": self.label,
            "prior": None if self.prior is None else self.prior.to_dict(),
            "bounds": None if self.bounds is None else list(self.bounds),
            "value": self.value,
            "fixed": self.fixed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Constant":
        """Rebuild a spec from [to_dict][SRToolkit.utils.free_constants.Constant.to_dict] output.

        Args:
            d: A dict as returned by ``to_dict``, also after a JSON round trip. Unknown keys are
                ignored.

        Returns:
            The spec, with the same ``to_dict`` as the one that was saved.

        Raises:
            KeyError: If a key of ``to_dict`` is missing from ``d``.
            ImportError: If the prior's sampler class cannot be imported.
            TypeError: If a value has the wrong type (see
                [Constant][SRToolkit.utils.free_constants.Constant]).
            ValueError: If the values are invalid (see
                [Constant][SRToolkit.utils.free_constants.Constant]).

        Examples:
            >>> from SRToolkit.dataset import UniformSampling
            >>> c = Constant(label="C_0", prior=UniformSampling(0.0, 2.0), bounds=(0.0, float("inf")))
            >>> Constant.from_dict(c.to_dict()).to_dict() == c.to_dict()
            True
        """
        # Imported here: SRToolkit.dataset imports this package.
        from SRToolkit.dataset.sampling import Sampler

        label, prior, bounds, value, fixed = d["label"], d["prior"], d["bounds"], d["value"], d["fixed"]
        return cls(
            label=label,
            prior=None if prior is None else Sampler.from_dict(prior),
            bounds=bounds,
            value=value,
            fixed=fixed,
        )


def uniform_default(constant_bounds: tuple[float, float]) -> Constant:
    """Return the default spec with a uniform prior on, and bounds of, a constant range.

    Args:
        constant_bounds: The ``(lower, upper)`` range, as in
            [BackgroundKnowledge.constant_bounds][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]:
            a pair of finite real numbers (length-2 sequence or array) with ``lower < upper``.

    Returns:
        A new template (``label=None``) with prior ``UniformSampling(lower, upper)`` (not mirrored)
        and bounds ``(lower, upper)`` as ``float``s.

    Raises:
        TypeError: If ``constant_bounds`` is not a pair of real numbers (``bool`` is
            not accepted as a number).
        ValueError: If ``constant_bounds`` is not finite or does not satisfy ``lower < upper``.

    Examples:
        >>> uniform_default((-5.0, 5.0)).bounds
        (-5.0, 5.0)
    """
    # Imported here: SRToolkit.dataset imports this package.
    from SRToolkit.dataset.sampling import UniformSampling

    template = Constant(bounds=constant_bounds)
    if template.bounds is None:
        raise TypeError("'constant_bounds' must be a pair of real numbers, got None.")
    lower, upper = template.bounds
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise ValueError(f"'constant_bounds' must be finite, got {constant_bounds!r}.")
    return replace(template, prior=UniformSampling(lower, upper))


def _check_default_constant(default_constant: Constant) -> None:
    """Raise ``ValueError`` unless ``default_constant`` is a template with a prior and bounds and no label, value or fixed."""
    d = default_constant
    if d.label is not None or d.value is not None or d.fixed or d.prior is None or d.bounds is None:
        raise ValueError(
            f"'default_constant' must be a template with a prior and bounds and no label, value or fixed, got {d!r}."
        )


def _specs_by_label(free_constants: Sequence[Constant] | None, labels: Sequence[str | None]) -> dict[str, Constant]:
    """Return the specs by label.

    Raises:
        ValueError: If a spec has no label, two specs share a label, or a spec's label is not in
            ``labels`` (the parameter labels of the expression).
    """
    specs: dict[str, Constant] = {}
    for spec in free_constants or ():
        if spec.label is None:
            raise ValueError(f"Every spec in 'free_constants' needs a label, got {spec!r}.")
        if spec.label in specs:
            raise ValueError(f"Two specs in 'free_constants' share the label {spec.label!r}.")
        specs[spec.label] = spec
    unknown = sorted(set(specs) - set(labels))
    if unknown:
        raise ValueError(f"'free_constants' labels {unknown} are not labelled constants of the expression.")
    return specs


def _ground_truth_parameters(ground_truth: "Expression", context: str) -> np.ndarray:
    """Return the parameter vector of a ground truth whose parameters are all fixed by its specs.

    One value per parameter, in the order of ``parameter_labels``. Used to evaluate a ground truth,
    e.g. to compute ``y``.

    Raises:
        ValueError: Starting with ``context``, naming the free constants without a fixed value.
    """
    fixed = {c.label: c.value for c in ground_truth.free_constants if c.fixed}
    labels = ground_truth.parameter_labels
    constant = str(ground_truth.symbol_library.constant_symbol)
    unfixed = sorted({label or constant for label in labels if label not in fixed})
    if unfixed:
        raise ValueError(
            f"{context} ground truth contains a free constant ({', '.join(unfixed)}) without a fixed value, so it "
            "cannot be evaluated. Fix its value with a spec (label it, e.g. C_0, and give the ground truth as an "
            "Expression with Constant(label=..., value=..., fixed=True))."
        )
    return np.array([fixed[label] for label in labels], dtype=np.float64)


def _spec_key(free_constants: Sequence[Constant] | None) -> str:
    """Return a canonical key of labelled specs; ``""`` for ``None`` or empty.

    The key covers only score-affecting fields (label, prior, bounds, value, fixed), priors by their
    ``to_dict`` content with numbers compared as floats (``UniformSampling(0, 2)`` equals
    ``UniformSampling(0.0, 2.0)``): it is equal for specs with equal fields in any order, differs
    when any of these fields differ, and is the same in every process (it feeds seeds and cache keys).
    """

    def as_floats(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: as_floats(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [as_floats(v) for v in x]
        if isinstance(x, numbers.Real) and not isinstance(x, bool):
            return float(x)
        return x

    if not free_constants:
        return ""
    # to_dict holds exactly the score-affecting fields; a descriptive field added later (D28)
    # must be dropped here.
    specs = sorted(free_constants, key=lambda c: c.label or "")
    return json.dumps([as_floats(c.to_dict()) for c in specs], sort_keys=True, separators=(",", ":"))
