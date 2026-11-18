"""Specifications of free constants: priors, bounds and fixed values.

A **free constant** is a token of type ``"const"`` (``C`` in the default
[SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary]) whose value is fitted to the data. It
is unrelated to a **named constant** (a ``"lit"`` symbol such as ``pi``), whose value is fixed by
the library.

A bare ``C`` is a new parameter at every occurrence; a **labelled** constant ``C_<int>`` (e.g.
``C_5``) is one parameter however often it occurs. Labels are local to an expression: ``C_0`` in two
expressions are unrelated parameters. The parameters of an expression form a vector ordered by first
appearance, as listed by [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels].

A [Constant][SRToolkit.utils.free_constants.Constant] describes how one labelled constant is
treated: the prior its starting points (and BED samples) are drawn from, the hard bounds of the fit,
an optional value, and whether that value is fixed instead of fitted. Specs address constants by
label, so to specify a constant write it labelled (``C_0`` instead of ``C``). A bare ``C``, and a
labelled constant without a spec, use the **default template**: a ``Constant`` without a label that
applies to every parameter without a spec of its own. A spec's unset ``prior`` or ``bounds`` are
also taken from it, so a fit is unbounded only when infinite bounds are given explicitly. A
dataset's default comes from its
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge];
[uniform_default][SRToolkit.utils.free_constants.uniform_default] builds the usual one, a uniform
prior on a constant range with that range as bounds.
[resolve_constants][SRToolkit.utils.free_constants.resolve_constants] combines the specs given for
an expression with the default into one complete spec per parameter.
"""

import json
import math
import numbers
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from .expression_tree import Node, parameter_labels
from .symbol_library import SymbolLibrary

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this package.
    from SRToolkit.dataset.sampling import Sampler


@dataclass(frozen=True)
class Constant:
    """Specification of one free constant.

    A spec with a ``label`` applies to every occurrence of that labelled constant in an expression.
    A spec without one is a **template**: the default applied to every parameter that has no spec
    of its own (see [uniform_default][SRToolkit.utils.free_constants.uniform_default]); a template
    used as the default must set ``prior`` and ``bounds``.

    Args:
        label: The labelled-constant token the spec applies to, e.g. ``"C_5"``. ``None`` makes the
            spec a template. Whether the label occurs in an expression is checked by
            [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]; a token the
            symbol library registers as a symbol (e.g. a named constant ``C_1``) is never a
            labelled constant.
        prior: Distribution that starting points and BED samples are drawn from. ``None``: the
            default's prior, filled in by [resolve_constants][SRToolkit.utils.free_constants.resolve_constants].
        bounds: Hard ``(lower, upper)`` bounds of the fit: a pair of real numbers (length-2 sequence
            or array) with ``lower < upper``. Either end may be infinite; ``(-inf, inf)`` fits without bounds.
            ``None``: the default's bounds, filled in by [resolve_constants][SRToolkit.utils.free_constants.resolve_constants].
        value: The constant's value: used as is when ``fixed``, otherwise the first starting point
            of the fit, which must lie in the closed interval ``bounds`` (checked against inherited
            bounds by [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]).
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

    label: Optional[str] = None
    prior: Optional["Sampler"] = None
    bounds: Optional[Tuple[float, float]] = None
    value: Optional[float] = None
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


def uniform_default(constant_bounds: Tuple[float, float]) -> Constant:
    """Return the default template with a uniform prior on, and bounds of, a constant range.

    Args:
        constant_bounds: The ``(lower, upper)`` range free constants are fitted in, as in
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


def resolve_constants(
    expr: Union[List[str], Node],
    free_constants: Optional[Sequence[Constant]],
    default: Constant,
    symbol_library: Optional[SymbolLibrary] = None,
) -> Tuple[Constant, ...]:
    """Return one complete spec per parameter of an expression.

    Each labelled constant with a spec gets that spec, with a ``prior`` or ``bounds`` of ``None``
    taken from ``default``. Every other parameter (each bare ``C``, and labelled constants without
    a spec) gets ``default`` itself. Use it to see what an expression is fitted under, e.g. to
    resample the constants of a saved result.

    Args:
        expr: Expression as an infix token list or a [Node][SRToolkit.utils.expression_tree.Node].
        free_constants: Specs for labelled constants of ``expr``, in any order. ``None`` or empty
            gives every parameter the default.
        default: The template for parameters without a spec, with ``prior`` and ``bounds`` set,
            e.g. from [uniform_default][SRToolkit.utils.free_constants.uniform_default].
        symbol_library: Symbol library used to recognise constants. ``None`` uses the active or
            default library.

    Returns:
        One spec per parameter, in the order of
        [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels]; ``label`` is ``None``
        for every parameter that uses the default, labelled or not. Empty if ``expr`` has no free
        constants.

    Raises:
        TypeError: If ``expr`` is neither a list of token strings nor a
            [Node][SRToolkit.utils.expression_tree.Node].
        ValueError: If ``default`` has a label or lacks a ``prior`` or ``bounds``, a spec has no
            label, two specs share a label, a spec's label is not a labelled constant of ``expr``,
            or a spec that is not fixed has a ``value`` outside the bounds it takes from
            ``default``.

    Examples:
        >>> default = uniform_default((-5.0, 5.0))
        >>> specs = resolve_constants(
        ...     ["C", "*", "X_0", "+", "C_1"], [Constant(label="C_1", value=2.0, fixed=True)], default
        ... )
        >>> [(s.label, s.value, s.bounds) for s in specs]
        [(None, None, (-5.0, 5.0)), ('C_1', 2.0, (-5.0, 5.0))]
    """
    if default.label is not None or default.prior is None or default.bounds is None:
        raise ValueError(f"'default' must be a template with a prior and bounds and no label, got {default!r}.")
    labels = parameter_labels(expr, symbol_library)
    specs: Dict[str, Constant] = {}
    for spec in free_constants or ():
        if spec.label is None:
            raise ValueError(f"Every spec in 'free_constants' needs a label, got {spec!r}.")
        if spec.label in specs:
            raise ValueError(f"Two specs share the label {spec.label!r}.")
        specs[spec.label] = spec
    unknown = sorted(set(specs) - set(labels))
    if unknown:
        raise ValueError(f"Spec labels {unknown} are not labelled constants of the expression.")
    resolved = []
    for label in labels:
        given = specs.get(label) if label is not None else None
        if given is None:
            resolved.append(default)
        else:
            resolved.append(
                replace(
                    given,
                    prior=default.prior if given.prior is None else given.prior,
                    bounds=default.bounds if given.bounds is None else given.bounds,
                )
            )
    return tuple(resolved)


def _spec_key(free_constants: Optional[Sequence[Constant]]) -> str:
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
