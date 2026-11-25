"""Shared constants, settings types and result dataclasses for SRToolkit.

- **Symbol types** (``VAR``, ``CONST``, ``FN``, ``OP``, ``LIT``) tag the entries of a
  [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary].
- **Precedence levels** (``OP_ADDITIVE``, ``OP_MULTIPLICATIVE``, ``OP_POWER``, ``FN_PREFIX``,
  ``FN_POSTFIX``, ``LEAF``) order symbols for parsing and printing. Custom operators can use values
  between the levels.
- [RankingSettings][SRToolkit.utils.types.RankingSettings] lists the keyword settings understood by
  the built-in ranking functions and the parameter estimator.
- [ModelResult][SRToolkit.utils.types.ModelResult] (one evaluated expression) and
  [EvalResult][SRToolkit.utils.types.EvalResult] (one SR run) hold experiment outcomes. Each model
  has a [Status][SRToolkit.utils.types.Status]. Both save to JSON through
  [serialization][SRToolkit.utils.serialization], so everything stored in them must be a value
  listed there. [CheckResult][SRToolkit.utils.types.CheckResult] is the outcome of one check of an
  [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]; a
  ``ModelResult`` saves the failed advisory ones. The trace-based metrics of ``EvalResult`` are explained in
  [sr_evaluator][SRToolkit.evaluation.sr_evaluator].
"""

import bisect
import dataclasses
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict, get_args

import numpy as np

from .serialization import _from_json_safe, _to_json_safe

if TYPE_CHECKING:
    # Imported for annotations only: free_constants imports symbol_library, which imports this module.
    from SRToolkit.evaluation.model_store import ModelStore

    from .expression_simplifier import SimplificationRules
    from .free_constants import Constant

VAR = "var"
"""Symbol type of a variable (e.g. ``X_0``)."""
CONST = "const"
"""Symbol type of a free constant whose value is fitted (e.g. ``C``)."""
FN = "fn"
"""Symbol type of a function (e.g. ``sin``, or postfix ``^2``)."""
OP = "op"
"""Symbol type of a binary operator (e.g. ``+``)."""
LIT = "lit"
"""Symbol type of a named constant, a symbol with a fixed value (e.g. ``pi``, ``e``)."""
VALID_SYMBOL_TYPES: set[str] = {VAR, CONST, FN, OP, LIT}
"""The set of valid symbol types: ``VAR``, ``CONST``, ``FN``, ``OP`` and ``LIT``."""

OP_ADDITIVE = 100
"""Precedence of additive operators (``+``, ``-``)."""
OP_MULTIPLICATIVE = 200
"""Precedence of multiplicative operators (``*``, ``/``)."""
OP_POWER = 300
"""Precedence of the power operator (``^``)."""

FN_PREFIX = 1000
"""Precedence of prefix functions (``sin(E)``). It is higher than every operator precedence."""
FN_POSTFIX = -1000
"""Precedence of postfix functions (``E^2``). Negative: any function precedence ``<= 0`` makes a
function postfix."""

LEAF = 1000
"""Precedence stored for atoms (``VAR``, ``CONST``, ``LIT``). Atoms never take part in precedence
comparisons. The large value prevents spurious parentheses if a high-precedence operator is added."""

Status = Literal["ok", "invalid", "rejected", "undefined", "error"]
"""
Outcome of evaluating one expression, stored as [ModelResult.status][SRToolkit.utils.types.ModelResult].

- ``"ok"``: scored, with a finite error.
- ``"invalid"``: not a well-formed expression; the parser raised an
  [ExpressionParseError][SRToolkit.utils.expression_tree.ExpressionParseError].
- ``"rejected"``: well-formed, but it fails a check of a required constraint of the evaluator (by
  default the ``max_constants`` / ``max_expr_length`` limits), or the ranking function rejected it.
  A failed pre-check means it was not scored; a failed post-check means it was scored and its fit
  kept, but its error is still ``+inf``.
- ``"undefined"``: scored, but the ranking function gave a non-finite error, e.g. because the
  expression is undefined at some inputs (``sqrt`` of a negative number, ``log(0)``) or
  overflows. For RMSE this describes the fitted model: other constants might have been fine.
- ``"error"``: the ranking function, or a check of a required constraint, raised an exception for
  a well-formed expression. This is a failure of the evaluation, not of the expression.

Every status except ``"ok"`` gets the error ``+inf``.
"""

STATUSES: tuple[str, ...] = get_args(Status)
"""The values of [Status][SRToolkit.utils.types.Status], in the same order. This is also the key order
of [status_counts][SRToolkit.utils.types.EvalResult.status_counts]."""


class RankingSettings(TypedDict, total=False):
    """Settings for the ranking functions, the parameter estimator and the evaluator's limits.

    Passed as ``**kwargs`` to ``SR_dataset``, ``SR_evaluator``, and ``ParameterEstimator``, then
    handed to the active ranking's [setup][SRToolkit.evaluation.ranking.RankingFunction.setup] as a
    plain ``dict``. All fields are optional.

    These are the keys the built-in rankings understand. A custom
    [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] may read *additional* keys from
    the settings dict at runtime. Passing them through works, but a type checker will not know
    about them, so extend this ``TypedDict`` (or silence the call site) if you want them checked.

    Attributes:
        method: Optimization algorithm for parameter fitting. Default: ``"L-BFGS-B"``.
        tol: Termination tolerance for the optimizer. Default: ``1e-6``.
        gtol: Gradient-norm termination tolerance. Default: ``1e-3``.
        max_iter: Maximum optimizer iterations. Default: ``100``.
        constant_bounds: ``(lower, upper)`` range of the default spec
            [uniform_default(constant_bounds)][SRToolkit.utils.free_constants.uniform_default],
            used unless ``default_constant`` is given (the evaluator also completes a partial
            ``default_constant`` from it). Validated only when used. Default: ``(-5, 5)``.
        default_constant: The default spec of free constants (see
            [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]): a
            [Constant][SRToolkit.utils.free_constants.Constant] template with ``prior`` and
            ``bounds`` set, e.g.
            [BackgroundKnowledge.default_constant][SRToolkit.dataset.background_knowledge.BackgroundKnowledge].
            The evaluator completes a partial template from ``constant_bounds`` before passing it
            on; rankings, ``ParameterEstimator`` and ``HoldoutMetric`` do not, and an incomplete
            one raises ``ValueError`` when used. ``None``: ``uniform_default(constant_bounds)``.
            Default: ``None``.
        initialization: Starting point of a free constant without a ``value``: ``"random"`` draws
            it from its prior, ``"mean"`` takes the prior's ``ppf(0.5)`` (the midpoint for the
            default spec). See
            [parameter_estimator][SRToolkit.evaluation.parameter_estimator]. Default: ``"random"``.
        max_constants: Maximum number of free constants in an expression (counted as by
            [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants]), an ``int`` or numpy
            integer (``bool`` and other types raise ``ValueError``). Read only by the evaluator
            and datasets, not by rankings or ``ParameterEstimator``: a shorthand for the required
            constraint
            [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants]: an expression
            with more gets status ``"rejected"`` and error ``+inf``, for every ranking. A negative
            value disables the limit. Adds nothing if the evaluator's ``required_constraints``
            already contain a ``MaxConstants``; a non-negative setting given together with one
            must have the same limit. Default: ``8``.
        max_expr_length: Maximum expression length in infix tokens, brackets included; same type
            rules as ``max_constants``. Read only by the evaluator and datasets: a shorthand for the
            required constraint [MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens], enforced like
            ``max_constants``. A negative value disables the limit. Same rule for a
            ``MaxTokens`` in ``required_constraints``. Default: ``-1``.
        backend: Backend used to compile expressions, by both rankings (``ParameterEstimator`` and BED).
            ``"stack"`` uses the Cython-backed postfix stack machine;
            ``"codegen"`` generates Python/NumPy source via ``exec()``;
            ``"stack_py"`` uses the pure-Python stack machine (faster than
            ``"stack"`` when the symbol library contains many custom symbols).
            Default: ``"stack"``.
        num_points_sampled: Number of domain points used when evaluating expression
            behavior for BED. ``-1`` uses all points in ``X``. Default: ``64``.
        samplers: The input distribution: one [Sampler][SRToolkit.dataset.sampling.Sampler] per
            variable. BED draws its ``num_points_sampled`` points from them when the behavior comes
            from a ground-truth expression and ``num_points_sampled`` is not ``-1``; otherwise it
            uses rows of ``X``. The evaluator also puts them in its background knowledge.
            Default: ``None``.
        bed_gt: Precomputed ground-truth behavior matrix for BED, one row per row of ``X`` and
            ``num_consts_sampled`` columns (so ``X`` must be the points it was computed on;
            ``samplers`` are ignored with it). If ``None``, the behavior is computed from the ground
            truth, or taken from ``y``. Default: ``None``.
        ranking_seed: Seed for the ranking's own randomness (BED points, behavior matrices and
            threshold; the starting points of RMSE's fits), overriding the evaluator's seed;
            ``None`` uses the evaluator's seed. Set by
            [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator]
            as described under ``ranking_resample``. Default: ``None``.
        ranking_resample: Read by [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset]. ``False``
            fixes the objective per dataset (``ranking_seed`` = dataset seed, ``0`` if unset);
            ``True`` resamples it every run from the evaluator's seed (``create_evaluator(seed=...)``,
            random if ``None``). An explicit ``ranking_seed`` wins in both cases. Default: ``False``.
        num_consts_sampled: Number of constant vectors sampled per expression for
            BED. Default: ``32``.
        ground_truth_constants: Specs for the labelled constants of the ground truth (see
            [Constant][SRToolkit.utils.free_constants.Constant]). Read only when the ground-truth
            expression is evaluated: RMSE uses their fixed values to generate ``y`` when it is
            missing; BED samples the ground truth's constants from them (the default spec for the
            rest) for its target behavior or default threshold. Passed straight to
            ``SR_evaluator``, they are validated by the ranking's ``setup``. Set by
            [SR_dataset.ranking_settings][SRToolkit.dataset.sr_dataset.SR_dataset.ranking_settings]
            from the dataset's ``ground_truth_constants``. Default: ``None``.
        simplify_rules: Rewrite rules of
            [simplify][SRToolkit.utils.expression_simplifier.simplify] for
            ``evaluate_expr(simplify_expr=True)``. Read only by the evaluator. ``None``:
            [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules]. Default: ``None``.
        simplify_max_iterations: ``max_iterations`` of
            [simplify][SRToolkit.utils.expression_simplifier.simplify] for
            ``evaluate_expr(simplify_expr=True)``, a non-negative ``int`` or numpy integer
            (not ``bool``). Read only by the evaluator. Default: ``simplify``'s (``30``).
        simplify_max_nodes: ``max_nodes`` of
            [simplify][SRToolkit.utils.expression_simplifier.simplify] for
            ``evaluate_expr(simplify_expr=True)``, a non-negative ``int`` or numpy integer
            (not ``bool``). Read only by the evaluator. Default: ``simplify``'s (``10_000``).
        simplify_max_time: ``max_time`` of
            [simplify][SRToolkit.utils.expression_simplifier.simplify] for
            ``evaluate_expr(simplify_expr=True)``: seconds, a positive number (not ``bool``), or
            ``None`` for no time limit. With a time limit the simplified expression, and so what is scored, can
            depend on the machine. Read only by the evaluator. Default: ``None``.

    Examples:
        >>> settings: RankingSettings = {"method": "L-BFGS-B", "max_iter": 200}
    """

    method: str
    tol: float
    gtol: float
    max_iter: int
    constant_bounds: tuple[float, float]
    initialization: str
    max_constants: int
    max_expr_length: int
    backend: str
    num_points_sampled: int
    samplers: list[Any] | None
    bed_gt: np.ndarray | None
    ranking_seed: int | None
    ranking_resample: bool
    num_consts_sampled: int
    ground_truth_constants: Sequence["Constant"] | None
    default_constant: "Constant | None"
    simplify_rules: "SimplificationRules | None"
    simplify_max_iterations: int
    simplify_max_nodes: int
    simplify_max_time: float | None


def _add_augmentation(augmentations: dict[str, dict[str, Any]], name: str, data: dict[str, Any], aug_type: str) -> None:
    # Shared body of ModelResult/EvalResult.add_augmentation: store a shallow copy under the first
    # free key among name, name_1, name_2, ...
    resolved = name
    counter = 1
    while resolved in augmentations:
        resolved = f"{name}_{counter}"
        counter += 1
    augmentations[resolved] = {**data, "_type": aug_type}


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check of one [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint].

    Immutable (a frozen dataclass), so results can be shared. A constraint's ``pre_check`` / ``post_check`` returns one; it fills in ``passed``, ``message``
    and ``amount`` and leaves ``constraint`` and ``stage`` empty. When the
    [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] records a result, it stores a
    copy with ``constraint`` and ``stage`` filled in.

    Attributes:
        passed: Whether the expression satisfies the constraint. ``True`` also when the check is
            not implemented.
        message: Why the check failed (e.g. ``"9 constants, limit 8"``); may be empty. For a check
            that is not implemented, says so. The built-in constraints leave it empty when the
            check passes.
        amount: How far outside the constraint the expression is, in the constraint's own unit
            (e.g. ``1.0`` for one constant too many). The built-in constraints return a ``float``
            that is greater than 0 exactly when the check fails, and ``0.0`` when it passes. Not validated, so a custom
            constraint may return any value.
        implemented: ``False`` if the constraint has no such check. The evaluator then ignores the
            result, whatever ``passed`` says: nothing is recorded and nothing is rejected.
        constraint: The class name (``type(constraint).__name__``) of the constraint that produced
            it, filled in by the evaluator; ``""`` otherwise.
        stage: ``"pre"`` or ``"post"`` for the check that produced it, filled in by the evaluator;
            ``""`` otherwise.
        error: ``None``, unless the check raised an exception or returned something other than a
            ``CheckResult`` (treated as a ``TypeError``). The evaluator then records the check as
            failed, with ``error`` set to ``"<ExceptionType>: <message>"``, ``message``
            to ``"raised <error>"`` and ``amount`` to ``inf``. Such a result says nothing about
            the expression; leave it out when counting violations.

    Examples:
        >>> CheckResult().passed
        True
        >>> CheckResult(passed=False, message="9 constants, limit 8", amount=1.0).amount
        1.0
    """

    passed: bool = True
    message: str = ""
    amount: float = 0.0
    implemented: bool = True
    constraint: str = ""
    stage: str = ""
    error: str | None = None


@dataclass
class ModelResult:
    """One evaluated expression: an entry of [EvalResult.all_models][SRToolkit.utils.types.EvalResult].

    Two models are equal when their [to_dict][SRToolkit.utils.types.ModelResult.to_dict] are equal.

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED). By convention ``+inf``
            unless ``status`` is ``"ok"``; this is not validated.
        parameters: Fitted values, one per parameter in the order of
            [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels], shape
            ``(n_parameters,)``, fixed values included; ``free_constants`` holds the specs they were
            fitted under. Set by the RMSE ranking only. ``None`` for other rankings and when nothing
            was fitted.
        ranking_data: Freeform payload carried over from the
            [RankingResult][SRToolkit.evaluation.ranking.RankingResult] ``extra`` dict of the
            ranking function that scored this expression. Empty for rankings that emit nothing
            beyond ``score``/``parameters``.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.ModelResult.add_augmentation].
        status: How the evaluation ended; see [Status][SRToolkit.utils.types.Status].
        status_detail: Why the status is not ``"ok"``: the parse error, the failed check of a
            required constraint (e.g. ``"MaxConstants: 9 constants, limit 8"``), the exception, or the
            non-finite score. ``None`` for ``"ok"``.
        advisory_violations: The failed checks of the evaluator's advisory constraints, as
            [CheckResult][SRToolkit.utils.types.CheckResult]s with ``constraint`` and ``stage``
            filled in; a check that raised is included, with ``error`` set. Pre-checks are recorded for every expression the evaluator could parse,
            whatever its status; post-checks only for models with status ``"ok"``. Pre-check
            results come first; within a stage, results follow the order of the constraints. Empty
            if no check failed, if the evaluator has no advisory constraints, or if the status is
            ``"invalid"``. It never affects ``error``. See
            [expression_constraints][SRToolkit.utils.expression_constraints].
        free_constants: The specs ``parameters`` were fitted under (priors, bounds, fixed values),
            as [Constant][SRToolkit.utils.free_constants.Constant]s sorted by label; ``None`` when
            no specs were given. Parameters without a spec (bare ``C`` always), and a spec's unset
            ``prior`` or ``bounds``, used [EvalResult.default_constant][SRToolkit.utils.types.EvalResult].

    Examples:
        >>> result = ModelResult(expr=["C", "*", "X_0"], error=0.42)
        >>> result.error, result.status
        (0.42, 'ok')
    """

    expr: list[str]
    error: float
    parameters: "np.ndarray | None" = None
    ranking_data: dict[str, Any] = field(default_factory=dict)
    augmentations: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: Status = "ok"
    status_detail: str | None = None
    advisory_violations: list[CheckResult] = field(default_factory=list)
    free_constants: tuple["Constant", ...] | None = None

    def __eq__(self, other: object) -> bool:
        # Field-wise dataclass equality raises for arrays with several elements; to_dict does not.
        if not isinstance(other, ModelResult):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def add_augmentation(self, name: str, data: dict[str, Any], aug_type: str) -> None:
        """Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter].

        A shallow copy of ``data`` is stored in [augmentations][SRToolkit.utils.types.ModelResult] with
        ``"_type"`` set to ``aug_type``. The caller's dict is not modified, but nested values are
        shared with it. If ``name`` is already taken, the first free key among ``name_1``,
        ``name_2``, … is used instead, so existing data is never overwritten.

        Args:
            name: Key under which the augmentation is stored.
            data: Augmentation data. Its values must be ones that can be saved (see
                [serialization][SRToolkit.utils.serialization]); otherwise
                [to_dict][SRToolkit.utils.types.ModelResult.to_dict] raises ``TypeError``. A
                ``"_type"`` key in it is replaced in the stored copy.
            aug_type: Augmenter class name, stored under ``"_type"``.

        Examples:
            >>> result = ModelResult(expr=["X_0"], error=0.1)
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> list(result.augmentations)
            ['latex', 'latex_1']
        """
        _add_augmentation(self.augmentations, name, data, aug_type)

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary of this model result.

        The dictionary can be passed to ``json.dump``. After a JSON round trip,
        [from_dict][SRToolkit.utils.types.ModelResult.from_dict] restores every field exactly, as
        guaranteed by [serialization][SRToolkit.utils.serialization] (arrays keep dtype and shape,
        tuples and non-str dict keys are kept). The one exception is that numpy scalars come back as
        Python scalars.

        Returns:
            A dictionary with one entry per field; ``advisory_violations`` is saved as a list of
            dicts with one entry per [CheckResult][SRToolkit.utils.types.CheckResult] field, and
            ``free_constants`` as a list of [Constant.to_dict][SRToolkit.utils.free_constants.Constant.to_dict]
            (or ``None``).

        Raises:
            TypeError: If ``parameters`` has an unsupported dtype (e.g. complex), or
                ``ranking_data`` or ``augmentations`` contain a value that cannot be saved. The
                message gives its full path, e.g. ``ranking_data['z']``.

        Examples:
            >>> ModelResult(expr=["X_0", "+", "C"], error=0.25).to_dict()["error"]
            0.25
        """
        return self._to_dict("")

    def _to_dict(self, prefix: str) -> dict:
        # to_dict with ``prefix`` (e.g. "all_models[3].") prepended to the paths in error messages.
        return {
            "expr": self.expr,
            "error": float(self.error),
            "parameters": _to_json_safe(self.parameters, f"{prefix}parameters"),
            "ranking_data": _to_json_safe(self.ranking_data, f"{prefix}ranking_data"),
            "augmentations": _to_json_safe(self.augmentations, f"{prefix}augmentations"),
            "status": self.status,
            "status_detail": self.status_detail,
            "advisory_violations": [
                _to_json_safe(dataclasses.asdict(r), f"{prefix}advisory_violations[{i}]")
                for i, r in enumerate(self.advisory_violations)
            ],
            "free_constants": (
                None
                if self.free_constants is None
                else _to_json_safe([c.to_dict() for c in self.free_constants], f"{prefix}free_constants")
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResult":
        """Reconstruct a [ModelResult][SRToolkit.utils.types.ModelResult] from the output of [to_dict][SRToolkit.utils.types.ModelResult.to_dict].

        Args:
            data: Dictionary produced by [to_dict][SRToolkit.utils.types.ModelResult.to_dict],
                possibly after a JSON round trip. Unknown keys are ignored; a missing
                ``advisory_violations`` (results saved before it was recorded) loads as an empty
                list, a missing ``free_constants`` as ``None``; other malformed input has undefined
                behaviour. Each saved
                [CheckResult][SRToolkit.utils.types.CheckResult] comes back equal, field by field
                (an ``inf`` amount included).

        Returns:
            The reconstructed [ModelResult][SRToolkit.utils.types.ModelResult].

        Raises:
            ValueError: If ``data`` has no ``status`` (it was saved before statuses were recorded),
                or its ``status`` is not one of [STATUSES][SRToolkit.utils.types.STATUSES].
            ImportError: If the prior of a saved spec cannot be imported.

        Examples:
            >>> result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
            >>> ModelResult.from_dict(result.to_dict()).expr
            ['X_0', '+', 'C']
        """
        if "status" not in data:
            raise ValueError(
                "[ModelResult.from_dict] This model was saved by an older SRToolkit version, without an "
                "evaluation status, and cannot be loaded. Rerun the experiment."
            )
        if data["status"] not in STATUSES:
            raise ValueError(f"[ModelResult.from_dict] Unknown status {data['status']!r}; expected one of {STATUSES}.")
        # Imported here: free_constants imports symbol_library, which imports this module.
        from .free_constants import Constant

        specs = data.get("free_constants")
        return cls(
            expr=data["expr"],
            error=data["error"],
            parameters=_from_json_safe(data["parameters"]),
            ranking_data=_from_json_safe(data["ranking_data"]),
            augmentations=_from_json_safe(data["augmentations"]),
            status=data["status"],
            status_detail=data["status_detail"],
            advisory_violations=[CheckResult(**r) for r in data.get("advisory_violations", [])],
            free_constants=None if specs is None else tuple(Constant.from_dict(c) for c in _from_json_safe(specs)),
        )


@dataclass
class RunSummary:
    """What an [EvalResult][SRToolkit.utils.types.EvalResult] keeps of a run instead of all its models.

    Made by ``get_results(keep_all_models=False)`` (see
    [SR_evaluator.get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results]) and
    stored in [EvalResult.summary][SRToolkit.utils.types.EvalResult]. Each field equals what the full
    result would give, so the methods of ``EvalResult`` answer the same with either.

    Attributes:
        num_evaluated: The number of distinct models evaluated.
        evaluation_calls: The number of counted calls (the length of the trace).
        incumbents: One ``(call_number, model_index, model)`` per call where the best finite error
            strictly improved, in call order: as
            [best_so_far][SRToolkit.utils.types.EvalResult.best_so_far], with each model.
            ``model_index`` is the model's position in the full run (and in its checkpoint, also
            after it is compacted). Loaded from a file, each entry is again a tuple.
        status_counts: The number of distinct models with each status, as ``status_counts()``.
        call_status_counts: The number of counted calls with each status, as
            ``status_counts(per="call")``.
        pareto_front: The models of [pareto_front][SRToolkit.utils.types.EvalResult.pareto_front].
            ``None`` if missing from a saved summary.
    """

    num_evaluated: int
    evaluation_calls: int
    incumbents: list[tuple[int, int, ModelResult]]
    status_counts: dict[str, int]
    call_status_counts: dict[str, int]
    pareto_front: list[ModelResult] | None = None


def _ranking_key(model: ModelResult) -> tuple[bool, float, int]:
    # Non-finite errors (failed models, or NaN in hand-built results) sort after every finite one;
    # NaN compares false with everything, so the flag leads and the error slot is a placeholder.
    # Among equal errors, shorter expressions come first.
    failed = not math.isfinite(model.error)
    return failed, 0.0 if failed else model.error, len(model.expr)


@dataclass
class EvalResult:
    """Result of a single SR experiment, as returned by ``SR_results[i]``.

    The stored data is the list of evaluated models and the evaluation trace; summary values
    such as ``min_error``, ``best_expr``, ``success`` and ``top_models()`` are derived from them.
    The ranking behind these summaries is cached. After construction models must not be modified
    or replaced, or the summaries can be stale.

    A result of a long run can instead keep only its top models and a
    [RunSummary][SRToolkit.utils.types.RunSummary] (``keep_all_models=False``, see
    [SR_evaluator.get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results]); its
    methods then give the same answers, except where noted. All models of such a run stay in its
    checkpoint, if it was kept (see
    [open_checkpoint][SRToolkit.utils.types.EvalResult.open_checkpoint]).

    Attributes:
        all_models: All evaluated models, in the order they were first evaluated. ``trace``
            indexes into this list. With a ``summary``, only the ``top_k`` best models, best
            first.
        trace: The evaluation trace. ``trace[k]`` is the index into ``all_models`` of the
            expression requested by the ``(k+1)``-th counted call to ``evaluate_expr`` (a call
            that counts toward ``max_evaluations``; see
            [sr_evaluator][SRToolkit.evaluation.sr_evaluator]). Cache hits appear as repeated
            indices; calls made after evaluation was stopped are not recorded. Every entry must
            be a valid, non-negative index into ``all_models``; otherwise behaviour is undefined.
            Empty with a ``summary``.
        approach_name: Name of the SR approach, or empty string if not provided.
        success_threshold: The experiment succeeds if an error strictly below this is found. A
            ``nan`` threshold is never reached. ``None`` if not set.
        top_k: Default number of models returned by ``top_models()``. ``0`` means none; negative
            means all models.
        dataset_name: Name of the dataset. ``None`` if not provided.
        metadata: The evaluator's metadata without ``dataset_name`` (e.g. model name, seed, and
            the dataset's metadata under ``"dataset_metadata"``). ``None`` if there is none.
        wall_time: Seconds the run took, all sessions of a resumed run together (see
            [SR_evaluator.wall_time][SRToolkit.evaluation.sr_evaluator.SR_evaluator.wall_time]).
            ``None`` if not known, e.g. from a checkpoint of a run that crashed before its first
            commit.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].
        default_constant: The default spec of the run (a
            [Constant][SRToolkit.utils.free_constants.Constant] template): the spec of every
            parameter without one of its own (see
            [ModelResult.free_constants][SRToolkit.utils.types.ModelResult]). With it,
            [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]
            ``(model.expr, model.free_constants, result.default_constant, symbol_library)`` gives
            the complete specs of a saved model. ``None`` for results saved in format 2.
        stop_reason: Why the run ended: ``"success_threshold"``, ``"max_evaluations"``, the reason
            of a callback (e.g. ``"time_limit"``) or of the approach (see
            [stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop]), or ``"returned"`` if the
            search returned without stopping the evaluator. Any other reason than the first two
            marks a run that ended with budget left. ``None`` for results saved before it was
            recorded.
        summary: What is kept of the run instead of all its models, or ``None`` (default) if
            ``all_models`` and ``trace`` are complete.
        checkpoint: A reference to the run's checkpoint, ``{"path": ..., "fingerprint": ...}``
            (the absolute path and the evaluator's fingerprint, see
            [Checkpoints][SRToolkit.evaluation.sr_evaluator]), or ``None``.
            [SR_results.from_checkpoint][SRToolkit.evaluation.results.SR_results.from_checkpoint],
            [evaluate_approach][SRToolkit.dataset.sr_dataset.SR_dataset.evaluate_approach] and
            [ExperimentJob.run][SRToolkit.experiments.ExperimentJob.run] set it when the file
            still exists as they return; ``get_results`` does not. Only
            [open_checkpoint][SRToolkit.utils.types.EvalResult.open_checkpoint] uses it.
        limits: The run's limit segments: one for its start and one per
            [extension][SRToolkit.dataset.sr_dataset.SR_dataset.extend_run] that changed something
            (a plain resume adds none; see [Extending runs][SRToolkit.evaluation.sr_evaluator]).
            Each is a dict with ``calls`` (the calls kept when the segment starts),
            ``max_evaluations`` (negative: no limit), ``success_threshold`` (``None``: none),
            ``callbacks`` (the ``to_dict()`` of each callback; ``[]`` if calls were made without
            callbacks, ``None`` if the segment made no call) and ``previous_stop_reason`` (why the segment before ended; ``None`` for
            the first, or after a crash). Check [extended][SRToolkit.utils.types.EvalResult.extended]
            beside ``stop_reason`` before comparing runs. ``[]`` if not recorded (results saved
            before it was, or built by hand).

    Examples:
        >>> models = [ModelResult(expr=["X_0"], error=0.5), ModelResult(expr=["C", "*", "X_0"], error=0.05)]
        >>> result = EvalResult(
        ...     all_models=models,
        ...     trace=[0, 1, 0],
        ...     approach_name="MyApproach",
        ...     success_threshold=0.1,
        ... )
        >>> result.best_expr, result.min_error, result.success
        ('C*X_0', 0.05, True)
    """

    all_models: list[ModelResult]
    trace: list[int]
    approach_name: str
    success_threshold: float | None = None
    top_k: int = 20
    dataset_name: str | None = None
    metadata: dict | None = None
    wall_time: float | None = None
    augmentations: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_constant: "Constant | None" = None
    stop_reason: str | None = None
    summary: RunSummary | None = None
    checkpoint: dict[str, Any] | None = None
    limits: list[dict[str, Any]] = field(default_factory=list)
    _ranking_cache: list[int] | None = field(default=None, init=False, repr=False, compare=False)

    def _ranking(self) -> list[int]:
        # Indices into all_models from best to worst. The sort is stable, so models that tie on
        # error and length keep their first-seen order.
        if self._ranking_cache is None or len(self._ranking_cache) != len(self.all_models):
            self._ranking_cache = sorted(range(len(self.all_models)), key=lambda i: _ranking_key(self.all_models[i]))
        return self._ranking_cache

    def _incumbents(self) -> list[tuple[int, int, float]]:
        # (call_number, model_index, error) where the best finite error strictly improved.
        if self.summary is not None:
            return [(call, index, model.error) for call, index, model in self.summary.incumbents]
        history: list[tuple[int, int, float]] = []
        for call, index in enumerate(self.trace, start=1):
            error = self.all_models[index].error
            if math.isfinite(error) and (not history or error < history[-1][2]):
                history.append((call, index, error))
        return history

    @property
    def extended(self) -> bool:
        """Whether the run was extended with new limits: ``limits`` has more than one segment.

        An extension that changed nothing adds no segment, so it does not count. An extended run
        may differ from one that had its final limits from the start: e.g. an approach that reads
        its budget ran its first segment under the old one, and a time limit is not reproducible.
        """
        return len(self.limits) > 1

    @property
    def num_evaluated(self) -> int:
        """Number of distinct models evaluated (an expression with its specs).

        ``len(all_models)``, or ``summary.num_evaluated``. Includes models the trace never
        references.
        """
        return len(self.all_models) if self.summary is None else self.summary.num_evaluated

    @property
    def evaluation_calls(self) -> int:
        """Number of counted calls to ``evaluate_expr``, including cache hits (``len(trace)``, or ``summary.evaluation_calls``)."""
        return len(self.trace) if self.summary is None else self.summary.evaluation_calls

    def top_models(self, k: int | None = None) -> list[ModelResult]:
        """Return the ``k`` best models, best first.

        Models with a finite error come first, sorted by error. Models whose error is not finite
        (``inf`` or ``nan``, not distinguished) come last. Only the error decides this, not the
        status; for results produced by the evaluator the two agree. Ties in error, and models
        that are all non-finite, are ordered by length (fewer tokens first), then by the order in
        which they were first evaluated. The returned objects are the ones in ``all_models``, not
        copies.

        Args:
            k: Number of models to return. Defaults to ``top_k``. ``0`` returns ``[]``; a negative
                value or one larger than the number of models returns all of them (with a
                ``summary``, the ``top_k`` kept ones).

        Returns:
            The best ``k`` models, best first.

        Examples:
            >>> models = [ModelResult(expr=["X_0"], error=0.5), ModelResult(expr=["X_1"], error=0.1)]
            >>> result = EvalResult(all_models=models, trace=[0, 1], approach_name="", top_k=1)
            >>> [m.expr for m in result.top_models()]
            [['X_1']]
            >>> [m.expr for m in result.top_models(k=-1)]
            [['X_1'], ['X_0']]
        """
        if k is None:
            k = self.top_k
        ranking = self._ranking()
        if 0 <= k < len(ranking):
            ranking = ranking[:k]
        return [self.all_models[i] for i in ranking]

    @property
    def min_error(self) -> float:
        """Lowest finite error across all models.

        If no error is finite, the error of the top-ranked model (see
        [top_models][SRToolkit.utils.types.EvalResult.top_models]); this is ``+inf`` for results
        produced by the evaluator. ``NaN`` if no model was evaluated.
        """
        ranking = self._ranking()
        return self.all_models[ranking[0]].error if ranking else float("nan")

    @property
    def best_expr(self) -> str:
        """Tokens of the best model joined without a separator (e.g. ``"C*X_0"``). Empty if no expression was evaluated."""
        ranking = self._ranking()
        return "".join(self.all_models[ranking[0]].expr) if ranking else ""

    @property
    def success(self) -> bool:
        """Whether ``min_error < success_threshold``. ``False`` if no threshold is set or it is ``nan``."""
        return self.success_threshold is not None and bool(self.min_error < self.success_threshold)

    def best_so_far(self) -> list[tuple[int, int]]:
        """Return the incumbent history: each call in the trace where the best error strictly improved.

        Failed models (non-finite error) never become the incumbent. Only a strictly lower error
        changes the incumbent, so after a tie broken by length the last entry need not be
        ``top_models(1)[0]``. With a ``summary``, read from ``summary.incumbents``, which also
        holds each model.

        Returns:
            A list of ``(call_number, model_index)`` pairs, where ``call_number`` is 1-based and
            ``model_index`` indexes into ``all_models`` (with a ``summary``, into the full run's
            models, as in its checkpoint). Empty if no model was scored successfully.

        Examples:
            >>> failed = ModelResult(["X_1", "X_0"], float("inf"), status="invalid")
            >>> models = [ModelResult(["X_0"], 0.5), failed, ModelResult(["C"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[1, 0, 0, 2], approach_name="")
            >>> result.best_so_far()
            [(2, 0), (4, 2)]
        """
        return [(call, index) for call, index, _ in self._incumbents()]

    def calls_to_threshold(self, threshold: float | None = None) -> int | None:
        """Return the number of calls needed to first reach an error below ``threshold``.

        With a ``summary``, read from ``summary.incumbents`` (the same answer).

        Args:
            threshold: Error threshold. Defaults to ``success_threshold``.

        Returns:
            The 1-based call number at which an expression with a finite error strictly below
            ``threshold`` was first requested, or ``None`` if that never happened (always for a
            ``nan`` threshold). Failed models (non-finite error) never count. When aggregating
            over runs, treat
            ``None`` as censored at ``evaluation_calls`` rather than dropping the run (see
            [sr_evaluator][SRToolkit.evaluation.sr_evaluator]).

        Raises:
            ValueError: If ``threshold`` is not given and ``success_threshold`` is ``None``.

        Examples:
            >>> models = [ModelResult(["X_0"], 0.5), ModelResult(["C"], 0.01)]
            >>> result = EvalResult(all_models=models, trace=[0, 0, 1], approach_name="", success_threshold=0.1)
            >>> result.calls_to_threshold()
            3
        """
        if threshold is None:
            threshold = self.success_threshold
        if threshold is None:
            raise ValueError("[EvalResult.calls_to_threshold] No threshold given and success_threshold is None.")
        # Errors strictly decrease along the incumbents, so the first one below the threshold is the first call.
        return next((call for call, _, error in self._incumbents() if error < threshold), None)

    def best_error_within(self, budgets: Sequence[int]) -> np.ndarray:
        """Return the best error found within each of the given numbers of counted calls.

        With a ``summary``, read from ``summary.incumbents`` (the same answer).

        A budget larger than ``evaluation_calls`` gets the run's final best error. This is exact
        for a run that reached ``success_threshold``, but only an upper bound for a run that
        stopped early for another reason (see [sr_evaluator][SRToolkit.evaluation.sr_evaluator]).
        Keep budgets at or below the experiment's ``max_evaluations``.

        Args:
            budgets: Numbers of calls, as ``int`` or numpy integers. Non-integers have undefined
                behaviour.

        Returns:
            Array of shape ``(len(budgets),)``, in the order of ``budgets``, with the best finite
            error within each budget.
            ``NaN`` where no model had been scored successfully yet, including every budget
            ``<= 0``.

        Examples:
            >>> failed = ModelResult(["X_0", "X_1"], float("inf"), status="invalid")
            >>> models = [failed, ModelResult(["X_1"], 0.5), ModelResult(["C"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[0, 1, 2], approach_name="")
            >>> result.best_error_within([1, 2, 3, 10])
            array([nan, 0.5, 0.1, 0.1])
        """
        incumbents = self._incumbents()
        calls = [call for call, _, _ in incumbents]
        curve = np.full(len(budgets), np.nan)
        for j, budget in enumerate(budgets):
            position = bisect.bisect_right(calls, budget)  # incumbents found within the budget
            if position > 0:
                curve[j] = incumbents[position - 1][2]
        return curve

    def status_counts(self, per: Literal["unique", "call"] = "unique") -> dict[str, int]:
        """Return the number of models with each [Status][SRToolkit.utils.types.Status].

        With a ``summary``, its stored counts.

        Args:
            per: ``"unique"`` counts each model (an expression with its specs) once; ``"call"`` counts every
                counted call in the trace, cache hits included.

        Returns:
            A dictionary with one entry per status in the order of
            [STATUSES][SRToolkit.utils.types.STATUSES], zeros included.

        Raises:
            ValueError: If ``per`` is not ``"unique"`` or ``"call"``.

        Examples:
            >>> models = [ModelResult(["X_0"], 0.5), ModelResult(["X_0", "X_1"], float("inf"), status="invalid")]
            >>> result = EvalResult(all_models=models, trace=[0, 1, 1], approach_name="")
            >>> result.status_counts()
            {'ok': 1, 'invalid': 1, 'rejected': 0, 'undefined': 0, 'error': 0}
            >>> result.status_counts(per="call")["invalid"]
            2
        """
        if per not in ("unique", "call"):
            raise ValueError(f"[EvalResult.status_counts] per must be 'unique' or 'call', got {per!r}.")
        if self.summary is not None:
            return dict(self.summary.status_counts if per == "unique" else self.summary.call_status_counts)
        models = self.all_models if per == "unique" else [self.all_models[i] for i in self.trace]
        counts = {status: 0 for status in STATUSES}
        for model in models:
            counts[model.status] += 1
        return counts

    def pareto_front(self) -> list[ModelResult]:
        """Return the models on the error-complexity Pareto front, shortest first.

        Complexity is the number of tokens (``len(model.expr)``). Failed models (non-finite error)
        are excluded. Of the models of each length, the one with the lowest error is taken (the
        first evaluated on ties); it is on the front if its error is strictly lower than that of
        every shorter model on it. So errors strictly decrease along the front, and its last model
        is ``top_models(1)[0]`` whenever that has a finite error. With a ``summary``, read from
        ``summary.pareto_front``. The returned objects are the stored ones, not copies.

        Returns:
            The models on the front, by increasing length. Empty if no model has a finite error.

        Raises:
            ValueError: If the result has a ``summary`` without a front (written before fronts were
                recorded).

        Examples:
            >>> models = [ModelResult(["X_0"], 0.5), ModelResult(["X_1"], 0.7), ModelResult(["C", "*", "X_0"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[0, 1, 2], approach_name="")
            >>> ["".join(m.expr) for m in result.pareto_front()]
            ['X_0', 'C*X_0']
        """
        if self.summary is not None:
            if self.summary.pareto_front is None:
                raise ValueError("[EvalResult.pareto_front] The summary of this result has no Pareto front.")
            return list(self.summary.pareto_front)
        best: dict[int, ModelResult] = {}  # per length, the first model with the lowest finite error
        for model in self.all_models:
            length = len(model.expr)
            if math.isfinite(model.error) and (length not in best or model.error < best[length].error):
                best[length] = model
        front: list[ModelResult] = []
        for length in sorted(best):
            if not front or best[length].error < front[-1].error:
                front.append(best[length])
        return front

    def open_checkpoint(self, path: "str | os.PathLike | None" = None) -> "ModelStore":
        """Open the run's checkpoint read-only, to reach all its models.

        Its ``models`` and ``trace`` are read-only sequences that read from the file on demand, so
        a run of any length can be gone through without loading it (see
        [ModelStore][SRToolkit.evaluation.model_store.ModelStore]). Close it when done.

        Args:
            path: Where the checkpoint is now, if it was moved. Defaults to the path in
                ``checkpoint``. The fingerprint is checked either way.

        Returns:
            The checkpoint, opened with ``mode="read"``.

        Raises:
            ValueError: If ``checkpoint`` is ``None``; if the file is a checkpoint of another run
                (its fingerprint differs from the one in ``checkpoint``; the message names the
                differing parts), which includes the run after an
                [extension][SRToolkit.dataset.sr_dataset.SR_dataset.extend_run] that changed its
                limits; if it is not a checkpoint, or one of another format version.
            FileNotFoundError: If no file exists at the path; the message gives the path.
        """
        # Imported here: model_store imports this module.
        from SRToolkit.evaluation.model_store import ModelStore

        if self.checkpoint is None:
            raise ValueError("[EvalResult.open_checkpoint] This result has no checkpoint reference.")
        target = self.checkpoint["path"] if path is None else path
        store = ModelStore(target, "read")
        try:
            saved = store.get_header("fingerprint", {})
        except BaseException:
            store.close()
            raise
        expected = self.checkpoint["fingerprint"]
        changed = sorted(k for k in saved.keys() | expected.keys() if saved.get(k) != expected.get(k))
        if changed:
            store.close()
            hint = (
                " If the run was extended, use the extended run's results."
                if set(changed) <= {"max_evaluations", "success_threshold"}
                else ""
            )
            raise ValueError(
                f"[EvalResult.open_checkpoint] The checkpoint at {os.fspath(target)!r} is of another run: "
                f"these parameters differ: {', '.join(changed)}.{hint}"
            )
        return store

    def add_augmentation(self, name: str, data: dict[str, Any], aug_type: str) -> None:
        """Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter].

        A shallow copy of ``data`` is stored in [augmentations][SRToolkit.utils.types.EvalResult] with
        ``"_type"`` set to ``aug_type``. The caller's dict is not modified, but nested values are
        shared with it. If ``name`` is already taken, the first free key among ``name_1``,
        ``name_2``, … is used instead, so existing data is never overwritten.

        Args:
            name: Key under which the augmentation is stored.
            data: Augmentation data. Its values must be ones that can be saved (see
                [serialization][SRToolkit.utils.serialization]); otherwise
                [to_dict][SRToolkit.utils.types.EvalResult.to_dict] raises ``TypeError``. A
                ``"_type"`` key in it is replaced in the stored copy.
            aug_type: Augmenter class name, stored under ``"_type"``.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0], approach_name="MyApproach")
            >>> result.add_augmentation("complexity", {"value": 3}, "ComplexityAugmenter")
            >>> result.augmentations["complexity"]
            {'value': 3, '_type': 'ComplexityAugmenter'}
        """
        _add_augmentation(self.augmentations, name, data, aug_type)

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary of this evaluation result.

        The dictionary can be passed to ``json.dump``. After a JSON round trip,
        [from_dict][SRToolkit.utils.types.EvalResult.from_dict] restores every field exactly
        (including ``metadata``, ``augmentations`` and each model, see
        [ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict]), as guaranteed by
        [serialization][SRToolkit.utils.serialization]. The one exception is that numpy scalars come
        back as Python scalars.

        Returns:
            A dictionary with ``"format_version": 4`` and one entry per field
            (``default_constant`` as its ``to_dict``, ``summary`` as a dictionary of its fields
            with each model as its ``to_dict``, each ``None`` if not set). It also holds the
            derived summaries ``min_error``, ``best_expr``, ``success``, ``num_evaluated``,
            ``evaluation_calls`` and ``status_counts`` (per model), for reading the file
            directly (with a ``summary``, the counts are its own). ``from_dict`` ignores them.
            ``min_error`` is ``nan`` for a result without
            models, which ``json`` writes as the non-standard token ``NaN``.

        Raises:
            TypeError: If ``metadata``, ``augmentations`` or ``limits`` contain a value that cannot be saved,
                or a model's ``to_dict`` raises (see
                [ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict]). The
                message gives its full path, e.g. ``all_models[3].ranking_data['z']``.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0, 0], approach_name="MyApproach")
            >>> d = result.to_dict()
            >>> d["trace"], d["min_error"]
            ([0, 0], 0.05)
        """
        summary = None
        if self.summary is not None:
            summary = {
                "num_evaluated": self.summary.num_evaluated,
                "evaluation_calls": self.summary.evaluation_calls,
                "incumbents": [
                    [call, index, model._to_dict(f"summary.incumbents[{i}].")]
                    for i, (call, index, model) in enumerate(self.summary.incumbents)
                ],
                "status_counts": dict(self.summary.status_counts),
                "call_status_counts": dict(self.summary.call_status_counts),
                "pareto_front": (
                    None
                    if self.summary.pareto_front is None
                    else [m._to_dict(f"summary.pareto_front[{i}].") for i, m in enumerate(self.summary.pareto_front)]
                ),
            }
        return {
            "format_version": 4,
            "min_error": float(self.min_error),
            "best_expr": self.best_expr,
            "success": self.success,
            "num_evaluated": self.num_evaluated,
            "evaluation_calls": self.evaluation_calls,
            "status_counts": self.status_counts(),
            "approach_name": self.approach_name,
            "dataset_name": self.dataset_name,
            "success_threshold": None if self.success_threshold is None else float(self.success_threshold),
            "top_k": int(self.top_k),
            "metadata": _to_json_safe(self.metadata, "metadata"),
            "wall_time": None if self.wall_time is None else float(self.wall_time),
            "stop_reason": self.stop_reason,
            "augmentations": _to_json_safe(self.augmentations, "augmentations"),
            "trace": [int(i) for i in self.trace],
            "all_models": [m._to_dict(f"all_models[{i}].") for i, m in enumerate(self.all_models)],
            "default_constant": (
                None
                if self.default_constant is None
                else _to_json_safe(self.default_constant.to_dict(), "default_constant")
            ),
            "summary": summary,
            "checkpoint": _to_json_safe(self.checkpoint, "checkpoint"),
            "limits": _to_json_safe(self.limits, "limits"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        """Reconstruct an [EvalResult][SRToolkit.utils.types.EvalResult] from the output of [to_dict][SRToolkit.utils.types.EvalResult.to_dict].

        Args:
            data: Dictionary produced by [to_dict][SRToolkit.utils.types.EvalResult.to_dict],
                possibly after a JSON round trip, of format 4, 3 (written before summaries and
                checkpoint references; ``summary`` and ``checkpoint`` load as ``None``) or 2
                (written before specs were recorded; ``default_constant`` and each model's
                ``free_constants`` load as ``None`` too). A missing ``stop_reason``, or a missing
                field of a saved ``summary`` that has a default, loads as ``None``; missing
                ``limits`` load as ``[]``. Unknown keys are
                ignored; other malformed input has undefined behaviour.

        Returns:
            The reconstructed [EvalResult][SRToolkit.utils.types.EvalResult].

        Raises:
            ValueError: If ``format_version`` is missing (results saved before the evaluation trace
                was recorded) or not ``2``, ``3`` or ``4``, or if a model cannot be loaded (see
                [ModelResult.from_dict][SRToolkit.utils.types.ModelResult.from_dict]).
            ImportError: If the prior of a saved spec cannot be imported.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0], approach_name="MyApproach")
            >>> EvalResult.from_dict(result.to_dict()).best_expr
            'X_0'
        """
        fmt = data.get("format_version")
        if fmt not in (2, 3, 4):
            if fmt is None:
                raise ValueError(
                    "[EvalResult.from_dict] These results were saved by an older SRToolkit version, without "
                    "an evaluation trace, and cannot be loaded. Rerun the experiment; with ExperimentGrid, "
                    "delete the old result files first, since existing files count as completed jobs."
                )
            raise ValueError(f"[EvalResult.from_dict] Unsupported format_version: {fmt!r}. Expected 2, 3 or 4.")
        # Imported here: free_constants imports symbol_library, which imports this module.
        from .free_constants import Constant

        default = data.get("default_constant")
        summary = data.get("summary")
        if summary is not None:
            # A field with a default may be missing (saved before it existed): it loads as None.
            values = {
                f.name: summary.get(f.name) if f.default is not dataclasses.MISSING else summary[f.name]
                for f in dataclasses.fields(RunSummary)
            }
            values["incumbents"] = [(call, index, ModelResult.from_dict(m)) for call, index, m in values["incumbents"]]
            if values["pareto_front"] is not None:
                values["pareto_front"] = [ModelResult.from_dict(m) for m in values["pareto_front"]]
            summary = RunSummary(**values)
        return cls(
            all_models=[ModelResult.from_dict(m) for m in data["all_models"]],
            trace=list(data["trace"]),
            approach_name=data["approach_name"],
            success_threshold=data["success_threshold"],
            top_k=data["top_k"],
            dataset_name=data["dataset_name"],
            metadata=_from_json_safe(data["metadata"]),
            wall_time=data["wall_time"],
            augmentations=_from_json_safe(data["augmentations"]),
            default_constant=None if default is None else Constant.from_dict(_from_json_safe(default)),
            stop_reason=data.get("stop_reason"),
            summary=summary,
            checkpoint=_from_json_safe(data.get("checkpoint")),
            limits=_from_json_safe(data.get("limits", [])),
        )
