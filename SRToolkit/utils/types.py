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
  listed there. The trace-based metrics of ``EvalResult`` are explained in
  [sr_evaluator][SRToolkit.evaluation.sr_evaluator].
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypedDict, get_args

import numpy as np

from .serialization import _from_json_safe, _to_json_safe

VAR = "var"
"""Symbol type of a variable (e.g. ``X_0``)."""
CONST = "const"
"""Symbol type of a free constant whose value is fitted (e.g. ``C``)."""
FN = "fn"
"""Symbol type of a function (e.g. ``sin``, or postfix ``^2``)."""
OP = "op"
"""Symbol type of a binary operator (e.g. ``+``)."""
LIT = "lit"
"""Symbol type of a literal with a fixed value (e.g. ``pi``, ``1``)."""
VALID_SYMBOL_TYPES: Set[str] = {VAR, CONST, FN, OP, LIT}
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
"""Precedence of postfix functions (``E^2``). Negative, and distinct from the ``-1`` that
[get_precedence][SRToolkit.utils.symbol_library.SymbolLibrary.get_precedence] returns for unknown
symbols."""

LEAF = 1000
"""Precedence stored for atoms (``VAR``, ``CONST``, ``LIT``). Atoms never take part in precedence
comparisons. The large value prevents spurious parentheses if a high-precedence operator is added."""

Status = Literal["ok", "invalid", "rejected", "undefined", "error"]
"""
Outcome of evaluating one expression, stored as [ModelResult.status][SRToolkit.utils.types.ModelResult].

- ``"ok"``: scored, with a finite error.
- ``"invalid"``: not a well-formed expression; the parser raised an
  [ExpressionParseError][SRToolkit.utils.expression_tree.ExpressionParseError].
- ``"rejected"``: well-formed, but it violates a limit of the task (``max_constants``,
  ``max_expr_length``), so it was not scored.
- ``"undefined"``: scored, but the ranking function gave a non-finite error, e.g. because the
  expression is undefined at some inputs (``sqrt`` of a negative number, ``log(0)``) or
  overflows. For RMSE this describes the fitted model: other constants might have been fine.
- ``"error"``: the ranking function raised an exception for a well-formed expression. This is a
  failure of the evaluation, not of the expression.

Every status except ``"ok"`` gets the error ``+inf``.
"""

STATUSES: Tuple[str, ...] = get_args(Status)
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
        constant_bounds: ``(lower, upper)`` bounds for constant values. The RMSE ranking samples
            initial values in them and passes them to the optimizer as bounds (used by methods
            that support bounds, such as ``"L-BFGS-B"``); BED samples its constant vectors in them.
            Default: ``(-5, 5)``.
        initialization: Constant initialization strategy. ``"random"`` samples
            uniformly within ``constant_bounds``; ``"mean"`` sets all constants
            to the midpoint. Default: ``"random"``.
        max_constants: Maximum number of free constants in an expression, enforced by the
            evaluator for every ranking. An expression with more gets status ``"rejected"``
            and error ``+inf``. A negative value disables the limit. Default: ``8``.
        max_expr_length: Maximum expression length in tokens, enforced like ``max_constants``.
            A negative value disables the limit. Default: ``-1``.
        backend: Backend used to compile expressions, by both rankings (``ParameterEstimator`` and BED).
            ``"stack"`` uses the Cython-backed postfix stack machine;
            ``"codegen"`` generates Python/NumPy source via ``exec()``;
            ``"stack_py"`` uses the pure-Python stack machine (faster than
            ``"stack"`` when the symbol library contains many custom symbols).
            Default: ``"stack"``.
        num_points_sampled: Number of domain points used when evaluating expression
            behavior for BED. ``-1`` uses all points in ``X``. Default: ``64``.
        samplers: Per-feature list of [Sampler][SRToolkit.dataset.sampling.Sampler] that draw
            the BED evaluation points (``num_points_sampled`` of them). Takes precedence over
            subsampling ``X``. Default: ``None``.
        bed_gt: Precomputed ground-truth behavior matrix for BED, one row per row of ``X`` and
            ``num_consts_sampled`` columns (so ``X`` must be the points it was computed on;
            ``samplers`` are ignored with it). If ``None``, the behavior is computed from the ground
            truth, or taken from ``y``. Default: ``None``.
        ranking_seed: Seed for the ranking's own randomness (BED points, behavior matrices,
            threshold), overriding the evaluator's seed. Set by
            [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator]
            to the dataset's seed. Read only by BED. Default: ``None``.
        bed_resample: Read by [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset]. ``False``
            fixes the BED objective per dataset (``ranking_seed`` = dataset seed, ``0`` if unset);
            ``True`` resamples it every run from the run's seed. Default: ``False``.
        num_consts_sampled: Number of constant vectors sampled per expression for
            BED. Default: ``32``.

    Examples:
        >>> settings: RankingSettings = {"method": "L-BFGS-B", "max_iter": 200}
    """

    method: str
    tol: float
    gtol: float
    max_iter: int
    constant_bounds: Tuple[float, float]
    initialization: str
    max_constants: int
    max_expr_length: int
    backend: str
    num_points_sampled: int
    samplers: Optional[List[Any]]
    bed_gt: Optional[np.ndarray]
    ranking_seed: Optional[int]
    bed_resample: bool
    num_consts_sampled: int


def _add_augmentation(augmentations: Dict[str, Dict[str, Any]], name: str, data: Dict[str, Any], aug_type: str) -> None:
    # Shared body of ModelResult/EvalResult.add_augmentation: store a shallow copy under the first
    # free key among name, name_1, name_2, ...
    resolved = name
    counter = 1
    while resolved in augmentations:
        resolved = f"{name}_{counter}"
        counter += 1
    augmentations[resolved] = {**data, "_type": aug_type}


@dataclass
class ModelResult:
    """One evaluated expression: an entry of [EvalResult.all_models][SRToolkit.utils.types.EvalResult].

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED). By convention ``+inf``
            unless ``status`` is ``"ok"``; this is not validated.
        parameters: Fitted constant values, shape ``(n_constants,)``. Set by the RMSE ranking
            only. ``None`` for other rankings and when nothing was fitted.
        ranking_data: Freeform payload carried over from the
            [RankingResult][SRToolkit.evaluation.ranking.RankingResult] ``extra`` dict of the
            ranking function that scored this expression. Empty for rankings that emit nothing
            beyond ``score``/``parameters``.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.ModelResult.add_augmentation].
        status: How the evaluation ended; see [Status][SRToolkit.utils.types.Status].
        status_detail: Why the status is not ``"ok"``: the parse error, the violated limit, the
            exception, or the non-finite score. ``None`` for ``"ok"``.

    Examples:
        >>> result = ModelResult(expr=["C", "*", "X_0"], error=0.42)
        >>> result.error, result.status
        (0.42, 'ok')
    """

    expr: List[str]
    error: float
    parameters: Optional["np.ndarray"] = None
    ranking_data: Dict[str, Any] = field(default_factory=dict)
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: Status = "ok"
    status_detail: Optional[str] = None

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        """Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter].

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
            A dictionary with one entry per field.

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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelResult":
        """Reconstruct a [ModelResult][SRToolkit.utils.types.ModelResult] from the output of [to_dict][SRToolkit.utils.types.ModelResult.to_dict].

        Args:
            data: Dictionary produced by [to_dict][SRToolkit.utils.types.ModelResult.to_dict],
                possibly after a JSON round trip. Unknown keys are ignored; other malformed
                input has undefined behaviour.

        Returns:
            The reconstructed [ModelResult][SRToolkit.utils.types.ModelResult].

        Raises:
            ValueError: If ``data`` has no ``status`` (it was saved before statuses were recorded),
                or its ``status`` is not one of [STATUSES][SRToolkit.utils.types.STATUSES].

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
        return cls(
            expr=data["expr"],
            error=data["error"],
            parameters=_from_json_safe(data["parameters"]),
            ranking_data=_from_json_safe(data["ranking_data"]),
            augmentations=_from_json_safe(data["augmentations"]),
            status=data["status"],
            status_detail=data["status_detail"],
        )


def _ranking_key(model: ModelResult) -> Tuple[bool, float, int]:
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

    Attributes:
        all_models: All evaluated models, in the order they were first evaluated. ``trace``
            indexes into this list.
        trace: The evaluation trace. ``trace[k]`` is the index into ``all_models`` of the
            expression requested by the ``(k+1)``-th counted call to ``evaluate_expr`` (a call
            that counts toward ``max_evaluations``; see
            [sr_evaluator][SRToolkit.evaluation.sr_evaluator]). Cache hits appear as repeated
            indices; calls made after evaluation was stopped are not recorded. Every entry must
            be a valid, non-negative index into ``all_models``; otherwise behaviour is undefined.
        approach_name: Name of the SR approach, or empty string if not provided.
        success_threshold: The experiment succeeds if an error strictly below this is found. A
            ``nan`` threshold is never reached. ``None`` if not set.
        top_k: Default number of models returned by ``top_models()``. ``0`` means none; negative
            means all models.
        dataset_name: Name of the dataset. ``None`` if not provided.
        metadata: The evaluator's metadata without ``dataset_name`` (e.g. model name, seed, and
            the dataset's metadata under ``"dataset_metadata"``). ``None`` if there is none.
        wall_time: Wall-clock duration in seconds of the approach's search for this
            experiment, measured around ``search()`` by the experiment runner. ``None`` if
            the result was not produced by a runner that records timing.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].

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

    all_models: List[ModelResult]
    trace: List[int]
    approach_name: str
    success_threshold: Optional[float] = None
    top_k: int = 20
    dataset_name: Optional[str] = None
    metadata: Optional[dict] = None
    wall_time: Optional[float] = None
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _ranking_cache: Optional[List[int]] = field(default=None, init=False, repr=False, compare=False)

    def _ranking(self) -> List[int]:
        # Indices into all_models from best to worst. The sort is stable, so models that tie on
        # error and length keep their first-seen order.
        if self._ranking_cache is None or len(self._ranking_cache) != len(self.all_models):
            self._ranking_cache = sorted(range(len(self.all_models)), key=lambda i: _ranking_key(self.all_models[i]))
        return self._ranking_cache

    @property
    def num_evaluated(self) -> int:
        """Number of unique expressions evaluated: ``len(all_models)``, including models the trace never references."""
        return len(self.all_models)

    @property
    def evaluation_calls(self) -> int:
        """Number of counted calls to ``evaluate_expr``, including cache hits."""
        return len(self.trace)

    def top_models(self, k: Optional[int] = None) -> List[ModelResult]:
        """Return the ``k`` best models, best first.

        Models with a finite error come first, sorted by error. Models whose error is not finite
        (``inf`` or ``nan``, not distinguished) come last. Only the error decides this, not the
        status; for results produced by the evaluator the two agree. Ties in error, and models
        that are all non-finite, are ordered by length (fewer tokens first), then by the order in
        which they were first evaluated. The returned objects are the ones in ``all_models``, not
        copies.

        Args:
            k: Number of models to return. Defaults to ``top_k``. ``0`` returns ``[]``; a negative
                value or one larger than the number of models returns all of them.

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

    def best_so_far(self) -> List[Tuple[int, int]]:
        """Return the incumbent history: each call in the trace where the best error strictly improved.

        Failed models (non-finite error) never become the incumbent. Only a strictly lower error
        changes the incumbent, so after a tie broken by length the last entry need not be
        ``top_models(1)[0]``.

        Returns:
            A list of ``(call_number, model_index)`` pairs, where ``call_number`` is 1-based and
            ``model_index`` indexes into ``all_models``. Empty if no model was scored successfully.

        Examples:
            >>> failed = ModelResult(["X_1", "X_0"], float("inf"), status="invalid")
            >>> models = [ModelResult(["X_0"], 0.5), failed, ModelResult(["C"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[1, 0, 0, 2], approach_name="")
            >>> result.best_so_far()
            [(2, 0), (4, 2)]
        """
        history: List[Tuple[int, int]] = []
        best = None
        for call, index in enumerate(self.trace, start=1):
            error = self.all_models[index].error
            if math.isfinite(error) and (best is None or error < best):
                best = error
                history.append((call, index))
        return history

    def calls_to_threshold(self, threshold: Optional[float] = None) -> Optional[int]:
        """Return the number of calls needed to first reach an error below ``threshold``.

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
        for call, index in enumerate(self.trace, start=1):
            error = self.all_models[index].error
            if math.isfinite(error) and error < threshold:
                return call
        return None

    def best_error_within(self, budgets: Sequence[int]) -> np.ndarray:
        """Return the best error found within each of the given numbers of counted calls.

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
        curve = np.full(len(budgets), np.nan)
        if not self.trace:
            return curve
        errors = np.array([self.all_models[i].error for i in self.trace], dtype=float)
        errors[~np.isfinite(errors)] = np.nan
        running = np.fmin.accumulate(errors)
        for j, budget in enumerate(budgets):
            if budget > 0:
                curve[j] = running[min(budget, len(running)) - 1]
        return curve

    def status_counts(self, per: Literal["unique", "call"] = "unique") -> Dict[str, int]:
        """Return the number of models with each [Status][SRToolkit.utils.types.Status].

        Args:
            per: ``"unique"`` counts each evaluated expression once; ``"call"`` counts every
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
        if per == "unique":
            models = self.all_models
        elif per == "call":
            models = [self.all_models[i] for i in self.trace]
        else:
            raise ValueError(f"[EvalResult.status_counts] per must be 'unique' or 'call', got {per!r}.")
        counts = {status: 0 for status in STATUSES}
        for model in models:
            counts[model.status] += 1
        return counts

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        """Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter].

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
            A dictionary with ``"format_version": 2`` and one entry per field. It also holds the
            derived summaries ``min_error``, ``best_expr``, ``success``, ``num_evaluated``,
            ``evaluation_calls`` and ``status_counts`` (per unique expression), for reading the file
            directly. ``from_dict`` ignores them. ``min_error`` is ``nan`` for a result without
            models, which ``json`` writes as the non-standard token ``NaN``.

        Raises:
            TypeError: If ``metadata`` or ``augmentations`` contain a value that cannot be saved,
                or a model's ``to_dict`` raises (see
                [ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict]). The
                message gives its full path, e.g. ``all_models[3].ranking_data['z']``.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0, 0], approach_name="MyApproach")
            >>> d = result.to_dict()
            >>> d["trace"], d["min_error"]
            ([0, 0], 0.05)
        """
        return {
            "format_version": 2,
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
            "augmentations": _to_json_safe(self.augmentations, "augmentations"),
            "trace": [int(i) for i in self.trace],
            "all_models": [m._to_dict(f"all_models[{i}].") for i, m in enumerate(self.all_models)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalResult":
        """Reconstruct an [EvalResult][SRToolkit.utils.types.EvalResult] from the output of [to_dict][SRToolkit.utils.types.EvalResult.to_dict].

        Args:
            data: Dictionary produced by [to_dict][SRToolkit.utils.types.EvalResult.to_dict],
                possibly after a JSON round trip. Unknown keys are ignored; other malformed
                input has undefined behaviour.

        Returns:
            The reconstructed [EvalResult][SRToolkit.utils.types.EvalResult].

        Raises:
            ValueError: If ``format_version`` is not ``2`` (results saved before the evaluation
                trace was recorded have none), or if a model cannot be loaded (see
                [ModelResult.from_dict][SRToolkit.utils.types.ModelResult.from_dict]).

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0], approach_name="MyApproach")
            >>> EvalResult.from_dict(result.to_dict()).best_expr
            'X_0'
        """
        fmt = data.get("format_version")
        if fmt != 2:
            if fmt is None:
                raise ValueError(
                    "[EvalResult.from_dict] These results were saved by an older SRToolkit version, without "
                    "an evaluation trace, and cannot be loaded. Rerun the experiment; with ExperimentGrid, "
                    "delete the old result files first, since existing files count as completed jobs."
                )
            raise ValueError(f"[EvalResult.from_dict] Unsupported format_version: {fmt!r}. Expected 2.")
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
        )
