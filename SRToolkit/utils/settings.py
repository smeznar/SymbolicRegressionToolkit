"""Settings of an evaluation run: how expressions are evaluated, fitted and ranked.

A [Settings][SRToolkit.utils.settings.Settings] object holds named **sections**, one per
component that reads them. Each section is a frozen dataclass, a subclass of
[SettingsSection][SRToolkit.utils.settings.SettingsSection], with its defaults and validation;
its class attribute ``name`` is its key in ``Settings``. The built-in sections are:

| Section | Class | Read by |
|---|---|---|
| ``evaluator`` | [EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings] | the evaluator |
| ``estimation`` | [EstimationSettings][SRToolkit.utils.settings.EstimationSettings] | the parameter estimator |
| ``minimize`` | [MinimizeSettings][SRToolkit.utils.settings.MinimizeSettings] | the parameter estimator's optimizer |
| ``rmse`` | [RMSESettings][SRToolkit.utils.settings.RMSESettings] | RMSE ranking |
| ``bed`` | [BEDSettings][SRToolkit.utils.settings.BEDSettings] | BED ranking |

A section that is not given holds its defaults, so ``Settings()`` is the default run. Each
component reads its own section and ignores the others, so one ``Settings`` can configure several
rankings at once. A custom ranking adds its own section under its own name.

Facts about the task (constant bounds, the default spec of free constants, samplers, the
ground truth's constants) are not settings; they belong to the dataset and its
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge].

Only the fields that differ from their defaults are saved (``to_dict``), and two ``Settings`` are
equal when their saved forms are.

Examples:
    >>> settings = Settings(minimize={"max_iter": 200}, evaluator=EvaluatorSettings(simplify_expr=True))
    >>> settings.minimize.max_iter, settings.minimize.method
    (200, 'L-BFGS-B')
    >>> settings.replace(minimize={"tol": 1e-8}).minimize.max_iter
    200
    >>> settings.to_dict()["sections"]["minimize"]
    {'max_iter': 200}

    A custom section:

    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True, eq=False)
    ... class MyRankingSettings(SettingsSection):
    ...     name = "my_ranking"
    ...     alpha: float = 0.5
    >>> Settings(my_ranking=MyRankingSettings(alpha=0.3)).get(MyRankingSettings).alpha
    0.3
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.utils.expression_simplifier import DefaultRules, SimplificationRules, _check_fold_mode
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe

_S = TypeVar("_S", bound="SettingsSection")


def _check_int(value: Any, name: str, minimum: int | None = None) -> int:
    # value as an int if it is an int or numpy integer (bool rejected), else TypeError; ValueError
    # if it is below minimum.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"[Settings] {name} must be an int, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"[Settings] {name} must be at least {minimum}, got {value!r}.")
    return int(value)


def _check_positive_real(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"[Settings] {name} must be a number, got {value!r}.")
    if not 0 < float(value) < math.inf:
        raise ValueError(f"[Settings] {name} must be positive and finite, got {value!r}.")


def _check_choice(value: Any, choices: tuple[str, ...], name: str) -> None:
    if value not in choices:
        raise ValueError(f"[Settings] {name} must be one of {list(choices)}, got {value!r}.")


class SettingsSection:
    """Base class of a settings section.

    A subclass is a ``@dataclass(frozen=True, eq=False)`` with a default for every field and sets
    ``name``.
    It validates its fields in ``__post_init__``, raising ``TypeError`` for a value of the wrong
    type and ``ValueError`` for one out of range or not one of the allowed values, with a message
    naming ``<name>.<field>``. Field values must be saveable (see
    [serialization][SRToolkit.utils.serialization]); a field holding an object with ``to_dict`` /
    ``from_dict`` declares that class as ``field(metadata={"codec": cls})``.

    Two sections are equal when they have the same class and the same ``to_dict()``, as
    ``Settings`` are. (Without ``eq=False``, the dataclass compares field by field instead.)

    Attributes:
        name: The section's key in [Settings][SRToolkit.utils.settings.Settings] (class
            attribute).
    """

    name: ClassVar[str] = ""

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.to_dict() == self.to_dict()  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]

    def _encode(self, f: dataclasses.Field, value: Any) -> Any:
        codec = f.metadata.get("codec")
        if codec is not None and value is not None:
            return value.to_dict()
        return _to_json_safe(value, f"{self.name}.{f.name}")

    def changed(self) -> dict[str, Any]:
        """Return the fields whose values differ from their defaults, as ``{field: value}``.

        Values are compared by their saved form, so a value equal to the default is left out.
        """
        result = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if f.default is not dataclasses.MISSING:
                default = f.default
            else:
                assert f.default_factory is not dataclasses.MISSING  # every settings field has a default
                default = f.default_factory()
            encoded = json.dumps(self._encode(f, value), sort_keys=True)
            if encoded != json.dumps(self._encode(f, default), sort_keys=True):
                result[f.name] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the changed fields (see ``changed``) in their saved, JSON-compatible form.

        A field with a codec is saved through its value's ``to_dict``.
        """
        fields = {f.name: f for f in dataclasses.fields(self)}  # type: ignore[arg-type]
        return {k: self._encode(fields[k], v) for k, v in self.changed().items()}

    @classmethod
    def from_dict(cls: type[_S], data: Mapping[str, Any]) -> _S:
        """Build the section from the output of ``to_dict``.

        Args:
            data: Saved field values by field name; missing fields take their defaults.

        Returns:
            The section.

        Raises:
            TypeError: If ``data`` names a field the section does not have, or as for the
                section's validation.
            ValueError: As for the section's validation.
        """
        fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise TypeError(f"[Settings] Section {cls.name!r} has no field(s) {unknown}.")
        values = {}
        for k, v in data.items():
            codec = fields[k].metadata.get("codec")
            values[k] = codec.from_dict(v) if codec is not None and v is not None else _from_json_safe(v)
        return cls(**values)


@dataclass(frozen=True, eq=False)
class EvaluatorSettings(SettingsSection):
    """Settings of the evaluator: the stored form of expressions, limits and seeding.

    The ``simplify_*`` fields are the arguments of
    [simplify][SRToolkit.utils.expression_simplifier.simplify] without the prefix.

    Attributes:
        simplify_expr: Simplify every expression the evaluator scores or looks up, and score and
            store the simplified form, so inputs that simplify alike share one cache entry.
            Slower; the stored expression is never longer than the input (brackets aside).
            Default ``False``.
        canonicalise_constants: [Canonicalise][SRToolkit.utils.expression.canonicalise] every
            expression after simplifying and folding, so ``C_1*X_0 + C_1`` after
            ``C_0*X_0 + C_0`` is a cache hit. Not applied to an expression that still has
            free-constant specs after simplifying and folding. Default ``False``.
        fold_constants: ``mode`` of [fold][SRToolkit.utils.expression_simplifier.fold]
            (``"none"``, ``"default"`` or ``"keep"``) for every expression scored or looked up;
            the folded form is a different hypothesis. With ``simplify_expr``, ``simplify`` folds
            as part of simplifying. Default ``"none"``.
        simplify_rules: Rewrite rules. Default
            [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules]``()``.
        simplify_max_iterations: Maximum number of rewrite steps, a non-negative int. Default
            ``30``.
        simplify_max_nodes: E-graph growth limit (see ``simplify``'s ``max_nodes``), a
            non-negative int. Default ``10_000``.
        simplify_max_time: Time limit in seconds, a positive finite number, or ``None`` for none. With a
            time limit, what is scored can depend on the machine. Default ``None``.
        simplify_fold_aware: Used when ``fold_constants`` is not ``"none"``: ``False`` is faster
            and folds less. Default ``True``.
        max_consecutive_skips: Evaluation stops with the reason ``"max_consecutive_skips"`` after
            this many skips in a row (see
            [count_skip][SRToolkit.evaluation.evaluator.Evaluator.count_skip]); negative for
            no limit. Default ``10000``.
        max_constants: Shorthand for the required constraint
            [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants], an int; negative
            for no limit. Default ``8``.
        max_expr_length: Shorthand for the required constraint
            [MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] (infix tokens, brackets
            included), an int; negative for no limit. Default ``-1``.
        backend: How expressions are compiled, by every ranking and the parameter estimator:
            ``"stack"`` (Cython stack machine), ``"codegen"`` (generated NumPy source) or
            ``"stack_py"`` (pure-Python stack machine, faster with many custom symbols).
            Default ``"stack"``.
        ranking_seed: Seed of the ranking's randomness (BED's points and behaviours, the starting
            points of fits), a non-negative int, in place of the evaluator's seed; ``None`` uses
            the evaluator's seed.
            [Dataset.create_evaluator][SRToolkit.dataset.dataset.Dataset.create_evaluator]
            sets it to the dataset's seed unless the dataset resamples its objective each run.
            Default ``None``.

    Raises:
        TypeError: If ``simplify_expr``, ``canonicalise_constants`` or ``simplify_fold_aware`` is
            not a ``bool``; ``simplify_rules`` is not a
            [SimplificationRules][SRToolkit.utils.expression_simplifier.SimplificationRules];
            ``simplify_max_time`` is not a number or ``None``; or ``simplify_max_iterations``,
            ``simplify_max_nodes``, ``max_consecutive_skips``, ``max_constants``,
            ``max_expr_length`` or ``ranking_seed`` is not an int (``None`` allowed for
            ``ranking_seed``).
        ValueError: If ``fold_constants`` or ``backend`` is not one of its values,
            ``simplify_max_iterations`` / ``simplify_max_nodes`` / ``ranking_seed`` is negative,
            ``simplify_max_time`` is not positive and finite, or ``max_consecutive_skips`` is ``0``.
    """

    name: ClassVar[str] = "evaluator"

    simplify_expr: bool = False
    canonicalise_constants: bool = False
    fold_constants: Literal["none", "default", "keep"] = "none"
    simplify_rules: SimplificationRules = field(default_factory=DefaultRules, metadata={"codec": SimplificationRules})
    simplify_max_iterations: int = 30
    simplify_max_nodes: int = 10_000
    simplify_max_time: float | None = None
    simplify_fold_aware: bool = True
    max_consecutive_skips: int = 10000
    max_constants: int = 8
    max_expr_length: int = -1
    backend: Literal["stack", "codegen", "stack_py"] = "stack"
    ranking_seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("simplify_expr", "canonicalise_constants", "simplify_fold_aware"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"[Settings] evaluator.{name} must be a bool, got {getattr(self, name)!r}.")
        if not isinstance(self.simplify_rules, SimplificationRules):
            raise TypeError(
                f"[Settings] evaluator.simplify_rules must be a SimplificationRules, got {self.simplify_rules!r}."
            )
        if self.simplify_max_time is not None:
            _check_positive_real(self.simplify_max_time, "evaluator.simplify_max_time")
        _check_fold_mode(self.fold_constants, "[Settings] evaluator.fold_constants")
        _check_choice(self.backend, ("stack", "codegen", "stack_py"), "evaluator.backend")
        skips = _check_int(self.max_consecutive_skips, "evaluator.max_consecutive_skips")
        if skips == 0:
            raise ValueError("[Settings] evaluator.max_consecutive_skips must be positive, or negative for no limit.")
        values = {
            "simplify_max_iterations": _check_int(self.simplify_max_iterations, "evaluator.simplify_max_iterations", 0),
            "simplify_max_nodes": _check_int(self.simplify_max_nodes, "evaluator.simplify_max_nodes", 0),
            "max_consecutive_skips": skips,
            "max_constants": _check_int(self.max_constants, "evaluator.max_constants"),
            "max_expr_length": _check_int(self.max_expr_length, "evaluator.max_expr_length"),
        }
        if self.ranking_seed is not None:
            values["ranking_seed"] = _check_int(self.ranking_seed, "evaluator.ranking_seed", 0)
        for k, v in values.items():
            object.__setattr__(self, k, v)


# The methods of scipy.optimize.minimize, by their lowercase name (scipy ignores letter case).
_MINIMIZE_METHODS = {
    name.lower(): name
    for name in (
        "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP",
        "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov",
    )
}  # fmt: skip


@dataclass(frozen=True, eq=False)
class EstimationSettings(SettingsSection):
    """Settings of the [parameter estimator][SRToolkit.evaluation.parameter_estimator].

    The optimizer's own settings are in the ``minimize`` section
    ([MinimizeSettings][SRToolkit.utils.settings.MinimizeSettings]).

    Attributes:
        initialization: Starting point of a free constant without a ``value``: ``"random"`` draws
            it from its prior, ``"mean"`` takes the prior's ``ppf(0.5)``. Default ``"random"``.

    Raises:
        ValueError: If ``initialization`` is not one of its values.
    """

    name: ClassVar[str] = "estimation"

    initialization: Literal["random", "mean"] = "random"

    def __post_init__(self) -> None:
        _check_choice(self.initialization, ("random", "mean"), "estimation.initialization")


@dataclass(frozen=True, eq=False)
class MinimizeSettings(SettingsSection):
    """Settings of ``scipy.optimize.minimize``, which the [parameter estimator][SRToolkit.evaluation.parameter_estimator] runs.

    Attributes:
        method: A ``scipy.optimize.minimize`` method, in any letter case; stored in SciPy's
            spelling (``"l-bfgs-b"`` becomes ``"L-BFGS-B"``). Default ``"L-BFGS-B"``.
        tol: ``minimize``'s termination tolerance ``tol``, a positive finite number. Default
            ``1e-6``.
        gtol: Gradient-norm tolerance (``options["gtol"]``, used by gradient-based methods), a
            positive finite number. Default ``1e-3``.
        max_iter: Maximum optimizer iterations (``options["maxiter"]``), a non-negative int.
            Default ``100``.

    Raises:
        TypeError: If ``method`` is not a string, ``tol`` / ``gtol`` is not a number or
            ``max_iter`` is not an int.
        ValueError: If ``method`` is not a ``minimize`` method, ``tol`` / ``gtol`` is not
            positive and finite (``nan`` included) or ``max_iter`` is negative.
    """

    name: ClassVar[str] = "minimize"

    method: str = "L-BFGS-B"
    tol: float = 1e-6
    gtol: float = 1e-3
    max_iter: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.method, str):
            raise TypeError(f"[Settings] minimize.method must be a string, got {self.method!r}.")
        if self.method.lower() not in _MINIMIZE_METHODS:
            raise ValueError(
                f"[Settings] minimize.method must be a scipy.optimize.minimize method, got {self.method!r}."
            )
        object.__setattr__(self, "method", _MINIMIZE_METHODS[self.method.lower()])
        _check_positive_real(self.tol, "minimize.tol")
        _check_positive_real(self.gtol, "minimize.gtol")
        object.__setattr__(self, "max_iter", _check_int(self.max_iter, "minimize.max_iter", 0))


@dataclass(frozen=True, eq=False)
class RMSESettings(SettingsSection):
    """Settings of [RMSERanking][SRToolkit.evaluation.ranking.RMSERanking]; none yet.

    Constants are fitted with the ``estimation`` and ``minimize`` sections.
    """

    name: ClassVar[str] = "rmse"


@dataclass(frozen=True, eq=False)
class BEDSettings(SettingsSection):
    """Settings of [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking].

    Attributes:
        num_points_sampled: Number of points at which behaviours are compared (drawn from the
            samplers or subsampled from the rows of ``X``, see
            [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking]), a positive int; ``-1`` uses
            every row of ``X``. Default ``64``.
        num_consts_sampled: Number of constant vectors sampled per expression, a positive int.
            Default ``32``.
        bed_gt: Precomputed target behaviour, an array of shape
            ``(n_samples, num_consts_sampled)`` with one row per row of ``X`` (stored as a
            read-only copy), or ``None`` to compute it from the ground truth or ``y``. Rows are checked
            against ``X`` when the ranking is set up. Default ``None``.

    Raises:
        TypeError: If ``num_points_sampled`` / ``num_consts_sampled`` is not an int.
        ValueError: If ``num_points_sampled`` is neither positive nor ``-1``,
            ``num_consts_sampled`` is not positive, or ``bed_gt`` is not 2-D with
            ``num_consts_sampled`` columns.
    """

    name: ClassVar[str] = "bed"

    num_points_sampled: int = 64
    num_consts_sampled: int = 32
    bed_gt: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = _check_int(self.num_points_sampled, "bed.num_points_sampled")
        if points < 1 and points != -1:
            raise ValueError(f"[Settings] bed.num_points_sampled must be positive or -1, got {points}.")
        consts = _check_int(self.num_consts_sampled, "bed.num_consts_sampled", 1)
        object.__setattr__(self, "num_points_sampled", points)
        object.__setattr__(self, "num_consts_sampled", consts)
        if self.bed_gt is not None:
            bed_gt = np.array(self.bed_gt)  # a copy, read-only below
            if bed_gt.ndim != 2 or bed_gt.shape[1] != consts:
                raise ValueError(
                    f"[Settings] bed.bed_gt must be a 2-D array with num_consts_sampled ({consts}) columns, "
                    f"got shape {bed_gt.shape}."
                )
            bed_gt.flags.writeable = False
            object.__setattr__(self, "bed_gt", bed_gt)


_BUILT_IN: dict[str, type[SettingsSection]] = {
    "evaluator": EvaluatorSettings,
    "estimation": EstimationSettings,
    "minimize": MinimizeSettings,
    "rmse": RMSESettings,
    "bed": BEDSettings,
}


def _as_section(name: str, value: Any, current: SettingsSection | None) -> SettingsSection:
    # The section stored under name: an instance as given (checked), or a dict of fields applied
    # to the current section (or the built-in defaults). Built-in names take exactly their class.
    if isinstance(value, SettingsSection):
        if type(value).name != name or (name in _BUILT_IN and type(value) is not _BUILT_IN[name]):
            raise ValueError(
                f"[Settings] Section {name!r} got a {type(value).__name__} (section {type(value).name!r})."
            )
        section = value
    elif isinstance(value, Mapping):
        if current is None:
            if name not in _BUILT_IN:
                raise ValueError(
                    f"[Settings] Unknown section {name!r}; give a custom section as an instance of its class."
                )
            current = _BUILT_IN[name]()
        section = dataclasses.replace(current, **value)  # type: ignore[type-var]
    else:
        raise TypeError(f"[Settings] Section {name!r} must be a SettingsSection or a dict, got {type(value).__name__}.")
    section.to_dict()  # TypeError now, rather than when saved, for a value that cannot be saved
    return section


class Settings:
    """The settings of a run, as named sections (see the [module][SRToolkit.utils.settings]).

    Immutable; change it with ``replace``. Two ``Settings`` are equal when their ``to_dict``
    forms are, so a value given at its default equals one left out. Copying and pickling go
    through ``to_dict`` / ``from_dict``, so custom section classes must be importable.

    A built-in section is given as an instance of exactly its class or as a dict of fields applied
    over its defaults; a section given as ``None`` or not given holds its defaults.

    Args:
        evaluator: [EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings] or a dict.
        estimation: [EstimationSettings][SRToolkit.utils.settings.EstimationSettings] or a dict.
        minimize: [MinimizeSettings][SRToolkit.utils.settings.MinimizeSettings] or a dict.
        rmse: [RMSESettings][SRToolkit.utils.settings.RMSESettings] or a dict.
        bed: [BEDSettings][SRToolkit.utils.settings.BEDSettings] or a dict.
        **custom: Custom sections, each an instance of its class under its ``name``.

    Raises:
        ValueError: If a section's key differs from its ``name``, a built-in key holds an instance
            of another class (a subclass included), or a dict is given for a custom section; as
            for the section's validation.
        TypeError: If a section is neither a ``SettingsSection`` nor a dict, a dict names an
            unknown field, or a field value cannot be saved; as for the section's validation.
    """

    __slots__ = ("_sections",)
    _sections: dict[str, SettingsSection]

    def __init__(
        self,
        evaluator: EvaluatorSettings | Mapping[str, Any] | None = None,
        estimation: EstimationSettings | Mapping[str, Any] | None = None,
        minimize: MinimizeSettings | Mapping[str, Any] | None = None,
        rmse: RMSESettings | Mapping[str, Any] | None = None,
        bed: BEDSettings | Mapping[str, Any] | None = None,
        **custom: SettingsSection,
    ) -> None:
        given = {
            "evaluator": evaluator,
            "estimation": estimation,
            "minimize": minimize,
            "rmse": rmse,
            "bed": bed,
            **custom,
        }
        sections = {name: _as_section(name, value, None) for name, value in given.items() if value is not None}
        for name, cls in _BUILT_IN.items():
            sections.setdefault(name, cls())
        object.__setattr__(self, "_sections", sections)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Settings is immutable; use replace().")

    @property
    def evaluator(self) -> EvaluatorSettings:
        """The ``evaluator`` section."""
        return self._sections["evaluator"]  # type: ignore[return-value]

    @property
    def estimation(self) -> EstimationSettings:
        """The ``estimation`` section."""
        return self._sections["estimation"]  # type: ignore[return-value]

    @property
    def minimize(self) -> MinimizeSettings:
        """The ``minimize`` section."""
        return self._sections["minimize"]  # type: ignore[return-value]

    @property
    def rmse(self) -> RMSESettings:
        """The ``rmse`` section."""
        return self._sections["rmse"]  # type: ignore[return-value]

    @property
    def bed(self) -> BEDSettings:
        """The ``bed`` section."""
        return self._sections["bed"]  # type: ignore[return-value]

    def get(self, cls: type[_S]) -> _S:
        """Return the section stored under ``cls.name``, or ``cls()`` (its defaults) if there is none.

        Args:
            cls: The section class.

        Returns:
            The stored section, or ``cls()``.

        Raises:
            TypeError: If the section stored under ``cls.name`` is not a ``cls``.
        """
        section = self._sections.get(cls.name)
        if section is None:
            return cls()
        if not isinstance(section, cls):
            raise TypeError(f"[Settings] Section {cls.name!r} holds a {type(section).__name__}, not a {cls.__name__}.")
        return section

    def replace(self, **sections: SettingsSection | Mapping[str, Any]) -> Settings:
        """Return a copy with sections replaced or updated; ``self`` is unchanged.

        Args:
            **sections: Sections by name. An instance replaces the section; a dict of fields
                updates the current one (a custom section only if it is present). ``None`` is not
                accepted; reset a section with an instance of its defaults. Sections not named are
                kept.

        Returns:
            The new settings.

        Raises:
            ValueError: As for the constructor (a dict for a custom section that is not present).
            TypeError: As for the constructor, or if a value is ``None``.
        """
        new = dict(self._sections)
        for name, value in sections.items():
            new[name] = _as_section(name, value, new.get(name))
        result = Settings.__new__(Settings)
        object.__setattr__(result, "_sections", new)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a JSON-compatible dict (arrays as serialization markers).

        Returns:
            ``{"format_version": 1, "sections": {name: fields}}`` with the changed fields of each
            section (see [changed][SRToolkit.utils.settings.SettingsSection.changed]); sections
            without changes are left out. The fields of a custom section also hold
            ``"section_class"``, its import path (``module.QualName``).
        """
        out: dict[str, Any] = {}
        for name, section in self._sections.items():
            fields = section.to_dict()
            if not fields:
                continue
            if name not in _BUILT_IN:
                fields = {"section_class": f"{type(section).__module__}.{type(section).__qualname__}", **fields}
            out[name] = fields
        return {"format_version": 1, "sections": out}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Settings:
        """Rebuild settings from the output of ``to_dict``.

        A dict with a top-level ``_bundle`` key is first bound with
        [bind_config][SRToolkit.bundle.bind_config], so custom section classes resolve to an
        installed ``.srtk`` bundle.

        Args:
            data: The output of [to_dict][SRToolkit.utils.settings.Settings.to_dict].

        Returns:
            The settings.

        Raises:
            ValueError: If ``format_version`` is not ``1`` or a section without
                ``"section_class"`` is not built in; as for the constructor.
            ImportError: If a ``"section_class"`` cannot be imported.
            TypeError: If ``"sections"`` is not a dict of dicts, or a ``"section_class"`` is not a
                subclass of ``SettingsSection`` or is ``SettingsSection`` itself; as for the
                constructor.
        """
        data = _auto_bind(dict(data))
        if data.get("format_version") != 1:
            raise ValueError(f"[Settings] Unsupported format_version: {data.get('format_version')!r}. Expected 1.")
        sections: dict[str, Any] = {}
        sections_data = data.get("sections", {})
        if not isinstance(sections_data, Mapping) or not all(isinstance(e, Mapping) for e in sections_data.values()):
            raise TypeError("[Settings] 'sections' must be a dict of dicts, one per section.")
        for name, entry in sections_data.items():
            entry = dict(entry)
            path = entry.pop("section_class", None)
            if path is not None:
                section_cls = _import_class(path, "settings section", SettingsSection)
            elif name in _BUILT_IN:
                section_cls = _BUILT_IN[name]
            else:
                raise ValueError(f"[Settings] Unknown section {name!r} without a section_class.")
            sections[name] = section_cls.from_dict(entry)
        return cls(**sections)

    def __reduce__(self) -> tuple[Any, ...]:
        # Copied and pickled through the saved form.
        return (Settings.from_dict, (self.to_dict(),))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Settings) and other.to_dict() == self.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        parts = [f"{name}={section.changed()!r}" for name, section in self._sections.items() if section.changed()]
        return f"Settings({', '.join(parts)})"
