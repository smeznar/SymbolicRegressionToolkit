"""
Shared type definitions, constants, and result dataclasses for the SRToolkit package.

Defines symbol-type constants (``VAR``, ``CONST``, ``FN``, ``OP``, ``LIT``),
``RankingSettings`` for ranking-function and parameter-estimation configuration, and
``ModelResult`` / ``EvalResult`` for representing SR experiment outcomes.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, TypedDict

import numpy as np

from .serialization import _from_json_safe, _to_json_safe

VAR = "var"
CONST = "const"
FN = "fn"
OP = "op"
LIT = "lit"
VALID_SYMBOL_TYPES: Set[str] = {VAR, CONST, FN, OP, LIT}

# Operator precedence levels — use these when registering operators so that
# user-defined operators at custom precedences can be inserted between levels.
OP_ADDITIVE = 100  # +, -
OP_MULTIPLICATIVE = 200  # *, /
OP_POWER = 300  # ^

# Precedence sentinels for functions (stored in the precedence field of "fn" symbols).
# Values are chosen to be well outside any reasonable operator precedence range.
FN_PREFIX = 1000  # prefix functions: sin(E), cos(E) — outranks all ops in the shunting-yard
FN_POSTFIX = -1000  # postfix functions: E '^2' — clearly negative, distinct from get_precedence's -1 fallback

# Atoms (VAR, CONST, LIT) never participate in operator precedence comparisons,
# but storing a large value prevents accidental parenthesisation if a high-precedence
# operator is added.
LEAF = 1000

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

STATUSES: Tuple[str, ...] = ("ok", "invalid", "rejected", "undefined", "error")


class RankingSettings(TypedDict, total=False):
    """
    Settings consumed by the ranking functions (and the RMSE ranking's parameter estimator).

    Passed as ``**kwargs`` to ``SR_dataset``, ``SR_evaluator``, and ``ParameterEstimator``, then
    handed to the active ranking's [setup][SRToolkit.evaluation.ranking.RankingFunction.setup] as a
    plain ``dict``. All fields are optional.

    These are the keys the built-in rankings understand. A custom
    [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] may read *additional* keys from
    the settings dict at runtime — passing them through works, but a type checker will not know
    about them, so extend this ``TypedDict`` (or silence the call site) if you want them checked.

    Examples:
        >>> settings: RankingSettings = {"method": "L-BFGS-B", "max_iter": 200}
        >>> settings.get("method")
        'L-BFGS-B'
        >>> settings.get("tol", 1e-6)
        1e-06

    Attributes:
        method: Optimization algorithm for parameter fitting. Default: ``"L-BFGS-B"``.
        tol: Termination tolerance for the optimizer. Default: ``1e-6``.
        gtol: Gradient-norm termination tolerance. Default: ``1e-3``.
        max_iter: Maximum optimizer iterations. Default: ``100``.
        constant_bounds: ``(lower, upper)`` bounds for sampled constant values.
            Default: ``(-5, 5)``.
        initialization: Constant initialization strategy — ``"random"`` samples
            uniformly within ``constant_bounds``; ``"mean"`` sets all constants
            to the midpoint. Default: ``"random"``.
        max_constants: Maximum number of free constants permitted in a single
            expression. Expressions exceeding this limit score ``NaN``.
            Default: ``8``.
        max_expr_length: Maximum expression length in tokens. ``-1`` disables the
            limit. Default: ``-1``.
        backend: Evaluation backend used by ``ParameterEstimator``.
            ``"stack"`` uses the Cython-backed postfix stack machine (default);
            ``"codegen"`` generates Python/NumPy source via ``exec()``;
            ``"stack_py"`` uses the pure-Python stack machine (faster than
            ``"stack"`` when the symbol library contains many custom symbols).
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


@dataclass
class ModelResult:
    """
    A single model entry in ``EvalResult.all_models`` (and so in ``EvalResult.top_models()``).

    Examples:
        >>> result = ModelResult(expr=["C", "*", "X_0"], error=0.42)
        >>> result.expr
        ['C', '*', 'X_0']
        >>> result.error
        0.42
        >>> result.parameters is None
        True
        >>> result.status
        'ok'
        >>> ModelResult(expr=["X_0", "X_1"], error=float("inf"), status="invalid").status_detail is None
        True

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED). ``+inf`` unless ``status``
            is ``"ok"``.
        parameters: Fitted constant values. Present for RMSE ranking only, ``None`` otherwise.
        status: How the evaluation ended; see [Status][SRToolkit.utils.types.Status].
        status_detail: Why the status is not ``"ok"``: the parse error, the violated limit, the
            exception, or the non-finite score. ``None`` for ``"ok"``.
        ranking_data: Freeform payload carried over from the
            [RankingResult][SRToolkit.evaluation.ranking.RankingResult] ``extra`` dict of the
            ranking function that scored this expression. Empty for rankings that emit nothing
            beyond ``score``/``parameters``.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.ModelResult.add_augmentation].
    """

    expr: List[str]
    error: float
    parameters: Optional["np.ndarray"] = None
    ranking_data: Dict[str, Any] = field(default_factory=dict)
    augmentations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    status: Status = "ok"
    status_detail: Optional[str] = None

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        """
        Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] to this result.

        If ``name`` is already present in :attr:`augmentations`, a numeric suffix is
        appended (``name_1``, ``name_2``, …) to avoid overwriting existing data.

        Examples:
            >>> result = ModelResult(expr=["X_0"], error=0.1)
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> result.augmentations["latex"]["value"]
            '$X_0$'
            >>> result.add_augmentation("latex", {"value": "$X_0$"}, "LaTeXAugmenter")
            >>> "latex_1" in result.augmentations
            True

        Args:
            name: Key under which the augmentation is stored in :attr:`augmentations`.
                A suffix is added automatically if the key already exists.
            data: Arbitrary dict of augmentation data. Any existing ``"_type"`` key
                will be overwritten with ``aug_type``.
            aug_type: Augmenter class name, stored as ``data["_type"]``.
        """
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """
        Serialize this model result to a JSON-safe dictionary.

        NumPy arrays and scalars are converted to native Python types so the
        result can be passed directly to ``json.dump``.

        Examples:
            >>> result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
            >>> d = result.to_dict()
            >>> d["expr"]
            ['X_0', '+', 'C']
            >>> d["error"]
            0.25
            >>> d["parameters"] is None
            True

        Returns:
            A JSON-safe dictionary suitable for passing to [from_dict][SRToolkit.utils.types.ModelResult.from_dict].
        """
        return {
            "expr": self.expr,
            "error": float(self.error),
            "parameters": _to_json_safe(self.parameters),
            "ranking_data": _to_json_safe(self.ranking_data),
            "augmentations": _to_json_safe(self.augmentations),
            "status": self.status,
            "status_detail": self.status_detail,
        }

    @staticmethod
    def from_dict(data: dict) -> "ModelResult":
        """
        Reconstruct a [ModelResult][SRToolkit.utils.types.ModelResult] from a dictionary produced by [to_dict][SRToolkit.utils.types.ModelResult.to_dict].

        Examples:
            >>> result = ModelResult(expr=["X_0", "+", "C"], error=0.25)
            >>> result2 = ModelResult.from_dict(result.to_dict())
            >>> result2.expr
            ['X_0', '+', 'C']
            >>> result2.error
            0.25

        Args:
            data: Dictionary representation of a [ModelResult][SRToolkit.utils.types.ModelResult], as produced
                by [to_dict][SRToolkit.utils.types.ModelResult.to_dict].

        Returns:
            The reconstructed :class:`ModelResult`.

        Raises:
            ValueError: If ``data`` has no ``status``, i.e. it was saved before statuses were
                recorded.
        """
        if "status" not in data:
            raise ValueError(
                "[ModelResult.from_dict] This model was saved by an older SRToolkit version, without an "
                "evaluation status, and cannot be loaded. Rerun the experiment."
            )
        return ModelResult(
            expr=data["expr"],
            error=data["error"],
            parameters=_from_json_safe(data["parameters"]),
            ranking_data=_from_json_safe(data.get("ranking_data", {})),
            augmentations=_from_json_safe(data["augmentations"]),
            status=data["status"],
            status_detail=data.get("status_detail"),
        )


def _ranking_key(model: ModelResult) -> Tuple[bool, float, int]:
    # Non-finite errors (failed models, or NaN in hand-built results) sort after every finite one;
    # NaN compares false with everything, so the flag leads and the error slot is a placeholder.
    # Among equal errors, shorter expressions come first.
    failed = not math.isfinite(model.error)
    return failed, 0.0 if failed else model.error, len(model.expr)


@dataclass
class EvalResult:
    """
    Result for a single SR experiment, as returned by ``SR_results[i]``.

    The stored data is the list of evaluated models and the evaluation trace; summary values
    such as ``min_error``, ``best_expr``, ``success`` and ``top_models()`` are derived from them.

    Examples:
        >>> models = [ModelResult(expr=["X_0"], error=0.5), ModelResult(expr=["C", "*", "X_0"], error=0.05)]
        >>> result = EvalResult(
        ...     all_models=models,
        ...     trace=[0, 1, 0],
        ...     approach_name="MyApproach",
        ...     success_threshold=0.1,
        ... )
        >>> result.best_expr
        'C*X_0'
        >>> result.min_error
        0.05
        >>> result.success
        True
        >>> result.num_evaluated, result.evaluation_calls
        (2, 3)
        >>> result.dataset_name is None
        True

    Attributes:
        all_models: All evaluated models, in the order they were first evaluated. ``trace``
            indexes into this list.
        trace: The evaluation trace. ``trace[k]`` is the index into ``all_models`` of the
            expression requested by the ``(k+1)``-th counted call to ``evaluate_expr``. Cache
            hits appear as repeated indices; calls made after evaluation was stopped are not
            recorded.
        approach_name: Name of the SR approach, or empty string if not provided.
        success_threshold: Error threshold below which the experiment counts as a success.
            ``None`` if not set.
        top_k: Default number of models returned by ``top_models()``. Negative means all models.
        dataset_name: Name of the dataset. ``None`` if not provided.
        metadata: Arbitrary metadata dict associated with the dataset. ``None`` if not provided.
        wall_time: Wall-clock duration in seconds of the approach's search for this
            experiment, measured around ``search()`` by the experiment runner. ``None`` if
            the result was not produced by a runner that records timing.
        augmentations: Per-augmenter data keyed by augmenter name. Populated by
            [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclasses via
            [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].
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
        """Number of unique expressions evaluated."""
        return len(self.all_models)

    @property
    def evaluation_calls(self) -> int:
        """Number of counted calls to ``evaluate_expr``, including cache hits."""
        return len(self.trace)

    def top_models(self, k: Optional[int] = None) -> List[ModelResult]:
        """
        The ``k`` best models, sorted by error, with models whose error is not finite (every
        status except ``"ok"``) last.

        Models with equal error are ordered by length (number of tokens), shorter first, and
        then by the order in which they were first evaluated. These are the same objects as in
        ``all_models``, not copies.

        Examples:
            >>> models = [ModelResult(expr=["X_0"], error=0.5), ModelResult(expr=["X_1"], error=0.1)]
            >>> result = EvalResult(all_models=models, trace=[0, 1], approach_name="", top_k=1)
            >>> [m.expr for m in result.top_models()]
            [['X_1']]
            >>> [m.expr for m in result.top_models(k=-1)]
            [['X_1'], ['X_0']]
            >>> result.top_models()[0] is result.all_models[1]
            True
            >>> long, short = ModelResult(["X_0", "+", "0"], 0.1), ModelResult(["X_0"], 0.1)
            >>> EvalResult(all_models=[long, short], trace=[0, 1], approach_name="").best_expr
            'X_0'

        Args:
            k: Number of models to return. Defaults to ``top_k``; a negative value or one larger
                than the number of models returns all of them.

        Returns:
            The best ``k`` models, best first.
        """
        if k is None:
            k = self.top_k
        ranking = self._ranking()
        if 0 <= k < len(ranking):
            ranking = ranking[:k]
        return [self.all_models[i] for i in ranking]

    @property
    def min_error(self) -> float:
        """Lowest error across all evaluated models. ``NaN`` if there are none, and ``+inf`` if
        every model failed."""
        ranking = self._ranking()
        return self.all_models[ranking[0]].error if ranking else float("nan")

    @property
    def best_expr(self) -> str:
        """The best expression as a string. Empty if no expression was evaluated."""
        ranking = self._ranking()
        return "".join(self.all_models[ranking[0]].expr) if ranking else ""

    @property
    def success(self) -> bool:
        """Whether ``min_error`` is below ``success_threshold``. ``False`` if no threshold is set."""
        return self.success_threshold is not None and bool(self.min_error < self.success_threshold)

    def best_so_far(self) -> List[Tuple[int, int]]:
        """
        The incumbent history: each point in the trace where the best error so far strictly improved.

        Failed models (non-finite error) never become the incumbent.

        Examples:
            >>> failed = ModelResult(["X_1", "X_0"], float("inf"), status="invalid")
            >>> models = [ModelResult(["X_0"], 0.5), failed, ModelResult(["C"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[1, 0, 0, 2], approach_name="")
            >>> result.best_so_far()
            [(2, 0), (4, 2)]

        Returns:
            A list of ``(call_number, model_index)`` pairs, where ``call_number`` is 1-based and
            ``model_index`` indexes into ``all_models``.
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
        """
        Number of calls needed to first reach an error below ``threshold``.

        Examples:
            >>> models = [ModelResult(["X_0"], 0.5), ModelResult(["C"], 0.01)]
            >>> result = EvalResult(all_models=models, trace=[0, 0, 1], approach_name="", success_threshold=0.1)
            >>> result.calls_to_threshold()
            3
            >>> result.calls_to_threshold(threshold=0.001) is None
            True

        Args:
            threshold: Error threshold. Defaults to ``success_threshold``.

        Returns:
            The 1-based call number at which an expression with error below ``threshold`` was first
            requested, or ``None`` if that never happened. When aggregating over runs, treat
            ``None`` as censored at ``evaluation_calls`` (more calls were needed) rather than
            dropping the run. A run that was stopped before using its whole budget (e.g. by a
            time limit) is censored at fewer calls than the budget; the result cannot tell such
            a run apart from one that used its budget.

        Raises:
            ValueError: If ``threshold`` is not given and ``success_threshold`` is ``None``.
        """
        if threshold is None:
            threshold = self.success_threshold
        if threshold is None:
            raise ValueError("[EvalResult.calls_to_threshold] No threshold given and success_threshold is None.")
        for call, index in enumerate(self.trace, start=1):
            if self.all_models[index].error < threshold:
                return call
        return None

    def best_error_within(self, budgets: Sequence[int]) -> np.ndarray:
        """
        Best error found within each of the given numbers of counted calls.

        A result does not record why its search stopped. For a budget larger than
        ``evaluation_calls``, the run's final best error is returned, as if the run had
        continued without finding anything better. How far to trust that depends on why the
        run stopped:

        - It reached ``success_threshold``: the value is exact, since a run stops as soon
          as it succeeds.
        - It stopped early for another reason, such as a time limit or an approach that ran
          out of candidates: the value is only an upper bound, because the run might have
          improved with more calls.

        Keep budgets at or below the experiment's ``max_evaluations``. No run got more calls
        than that, so larger budgets don't measure anything.

        Examples:
            >>> failed = ModelResult(["X_0", "X_1"], float("inf"), status="invalid")
            >>> models = [failed, ModelResult(["X_1"], 0.5), ModelResult(["C"], 0.1)]
            >>> result = EvalResult(all_models=models, trace=[0, 1, 2], approach_name="")
            >>> result.best_error_within([0, 1, 2, 3, 10])
            array([nan, nan, 0.5, 0.1, 0.1])

        Args:
            budgets: Numbers of calls. Budgets beyond ``evaluation_calls`` give the final best
                error (see above).

        Returns:
            An array with the best finite error within each budget, or ``NaN`` where no model had
            been scored successfully yet.
        """
        errors = np.array([self.all_models[i].error for i in self.trace], dtype=float)
        errors[~np.isfinite(errors)] = np.nan
        running = np.fmin.accumulate(errors) if len(errors) > 0 else errors
        curve = np.full(len(budgets), np.nan)
        for j, budget in enumerate(budgets):
            if budget > 0 and len(running) > 0:
                curve[j] = running[min(budget, len(running)) - 1]
        return curve

    def status_counts(self, per: Literal["unique", "call"] = "unique") -> Dict[str, int]:
        """
        Number of models with each [Status][SRToolkit.utils.types.Status].

        Examples:
            >>> models = [ModelResult(["X_0"], 0.5), ModelResult(["X_0", "X_1"], float("inf"), status="invalid")]
            >>> result = EvalResult(all_models=models, trace=[0, 1, 1], approach_name="")
            >>> result.status_counts()
            {'ok': 1, 'invalid': 1, 'rejected': 0, 'undefined': 0, 'error': 0}
            >>> result.status_counts(per="call")["invalid"]
            2

        Args:
            per: ``"unique"`` counts each evaluated expression once; ``"call"`` counts every
                counted call in the trace, cache hits included.

        Returns:
            A dictionary with one entry per status, zeros included.

        Raises:
            ValueError: If ``per`` is not ``"unique"`` or ``"call"``.
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

    # validity_rate is left out to keep the API lean: it is rarely needed, and it is a one-line
    # formula on status_counts (the definition is in the "Evaluation statuses" section of the
    # sr_evaluator module docstring). Uncomment it, and its tests in tests/utils/test_types.py,
    # if a canonical method turns out to be useful.
    #
    # def validity_rate(self, per: Literal["unique", "call"] = "unique") -> float:
    #     """
    #     Fraction of expressions that are well-formed.
    #
    #     Every expression with a tree counts as valid (``"ok"``, ``"rejected"``, ``"undefined"``);
    #     ``"invalid"`` ones do not. Models with status ``"error"`` are left out of both counts,
    #     since their failure lies in the evaluation, not in the expression.
    #
    #     Args:
    #         per: ``"unique"`` counts each evaluated expression once; ``"call"`` counts every
    #             counted call in the trace, cache hits included.
    #
    #     Returns:
    #         The fraction of valid expressions, or ``NaN`` if there are none to count.
    #     """
    #     counts = self.status_counts(per)
    #     total = sum(counts.values()) - counts["error"]
    #     if total == 0:
    #         return float("nan")
    #     return (total - counts["invalid"]) / total

    def add_augmentation(self, name: str, data: Dict[str, Any], aug_type: str) -> None:
        """
        Attach augmentation data produced by a [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] to this result.

        If ``name`` is already present in :attr:`augmentations`, a numeric suffix is
        appended (``name_1``, ``name_2``, …) to avoid overwriting existing data.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0], approach_name="MyApproach")
            >>> result.add_augmentation("complexity", {"value": 3}, "ComplexityAugmenter")
            >>> result.augmentations["complexity"]["value"]
            3
            >>> result.add_augmentation("complexity", {"value": 5}, "ComplexityAugmenter")
            >>> "complexity_1" in result.augmentations
            True

        Args:
            name: Key under which the augmentation is stored in :attr:`augmentations`.
                A suffix is added automatically if the key already exists.
            data: Arbitrary dict of augmentation data. Any existing ``"_type"`` key
                will be overwritten with ``aug_type``.
            aug_type: Augmenter class name, stored as ``data["_type"]``.
        """
        resolved = name
        counter = 1
        while resolved in self.augmentations:
            resolved = f"{name}_{counter}"
            counter += 1
        data["_type"] = aug_type
        self.augmentations[resolved] = data

    def to_dict(self) -> dict:
        """
        Serialize this evaluation result to a JSON-safe dictionary.

        NumPy arrays and scalars within nested [ModelResult][SRToolkit.utils.types.ModelResult] entries are
        converted to native Python types so the result can be passed directly
        to ``json.dump``. Besides the stored fields, the dictionary contains the derived summary
        values (``min_error``, ``best_expr``, ``success``, ``num_evaluated``, ``evaluation_calls``,
        ``status_counts`` per unique expression) for reading the file directly; [from_dict][SRToolkit.utils.types.EvalResult.from_dict]
        ignores them.

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0, 0], approach_name="MyApproach")
            >>> d = result.to_dict()
            >>> d["min_error"]
            0.05
            >>> d["trace"]
            [0, 0]
            >>> d["approach_name"]
            'MyApproach'

        Returns:
            A JSON-safe dictionary suitable for passing to [from_dict][SRToolkit.utils.types.EvalResult.from_dict].
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
            "metadata": self.metadata,
            "wall_time": self.wall_time,
            "augmentations": _to_json_safe(self.augmentations),
            "trace": [int(i) for i in self.trace],
            "all_models": [m.to_dict() for m in self.all_models],
        }

    @staticmethod
    def from_dict(data: dict) -> "EvalResult":
        """
        Reconstruct an [EvalResult][SRToolkit.utils.types.EvalResult] from a dictionary produced by [to_dict][SRToolkit.utils.types.EvalResult.to_dict].

        Examples:
            >>> result = EvalResult(all_models=[ModelResult(["X_0"], 0.05)], trace=[0], approach_name="MyApproach")
            >>> result2 = EvalResult.from_dict(result.to_dict())
            >>> result2.min_error
            0.05
            >>> result2.best_expr
            'X_0'
            >>> result2.trace
            [0]

        Args:
            data: Dictionary representation of an [EvalResult][SRToolkit.utils.types.EvalResult], as produced
                by [to_dict][SRToolkit.utils.types.EvalResult.to_dict].

        Returns:
            The reconstructed [EvalResult][SRToolkit.utils.types.EvalResult].

        Raises:
            ValueError: If ``format_version`` is not ``2``. Results saved before the evaluation
                trace was recorded have no ``format_version`` and cannot be loaded.
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
        return EvalResult(
            all_models=[ModelResult.from_dict(m) for m in data["all_models"]],
            trace=list(data["trace"]),
            approach_name=data["approach_name"],
            success_threshold=data.get("success_threshold"),
            top_k=data.get("top_k", 20),
            dataset_name=data.get("dataset_name"),
            metadata=data.get("metadata"),
            wall_time=data.get("wall_time"),
            augmentations=_from_json_safe(data["augmentations"]),
        )
