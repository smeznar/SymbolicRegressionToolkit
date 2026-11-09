"""Expression evaluation and results management for symbolic regression.

Contains [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator], which ranks expressions
with a [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] (RMSE by default);
[SR_results][SRToolkit.evaluation.sr_evaluator.SR_results], a container for experiment results; and
[ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter], the base class for
post-processing results.

## Evaluation statuses

Each evaluated expression gets a [Status][SRToolkit.utils.types.Status] and, unless it is ``"ok"``,
a ``status_detail`` saying why. The evaluator checks in this order and stops at the first match:

1. ``"invalid"``: the parser raises an
   [ExpressionParseError][SRToolkit.utils.expression_tree.ExpressionParseError].
2. ``"rejected"``: a pre-check of a required constraint fails (see below), so the expression is
   not scored; or the ranking function rejects it while scoring.
3. ``"error"``: the ranking function, or a check of a required constraint, raises. The evaluator
   also emits a warning.
4. ``"undefined"``: the ranking function returns a non-finite error.
5. ``"rejected"``: a post-check of a required constraint fails. The fit is kept in ``parameters``.
6. ``"ok"``: otherwise.

Every status except ``"ok"`` gets the error ``+inf``, so an evaluated expression never gives
``NaN`` and failed models rank after every scored one. Every call counts toward
``max_evaluations``, whatever its status. Once evaluation has stopped (threshold or budget
reached), further calls are not evaluated: they warn, return ``nan`` and are not counted.

The statuses are saved with the results, so validity can be measured after the run from
[status_counts][SRToolkit.utils.types.EvalResult.status_counts]. Count every well-formed
expression as valid, including rejected and undefined ones, and leave ``"error"`` out entirely,
because those failures belong to the evaluation, not to the approach:

```python
counts = result.status_counts()  # per unique expression; per="call" counts every call
validity_rate = 1 - counts["invalid"] / (sum(counts.values()) - counts["error"])
```

## Constraints

The evaluator enforces the ``required_constraints`` and records the advisory ones, which are the
``constraints`` of its background knowledge (see below). Both are
[ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]s; that module
explains what the checks look at and the two roles. The evaluator calls ``setup`` with its symbol
library on its own copies, so the instances passed in are not changed.

- ``required_constraints``: pre-checks run in order after parsing, post-checks in order after an
  ``"ok"`` score. The first failure gives ``"rejected"`` with the ``status_detail``
  ``"<ClassName>: <message>"``, e.g. ``"MaxConstants: 9 constants, limit 8"``, and the later checks
  of that stage are not run. The settings
  ``max_constants`` (default 8) and ``max_expr_length`` add a ``MaxConstants`` and a ``MaxTokens``
  in front of the list, unless a negative setting disables them or the list already has one of
  that class (see [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator]), so ``[]``
  still allows at most 8 constants.
- ``advisory_constraints``: pre-checks run on every parsed expression, post-checks on every ``"ok"``
  model. Each failed check is recorded in
  [ModelResult.advisory_violations][SRToolkit.utils.types.ModelResult] as a
  [CheckResult][SRToolkit.utils.types.CheckResult] with ``constraint`` and ``stage`` filled in; the
  error is not affected.

For one expression the order is: parse, advisory pre-checks, required pre-checks, score, required
post-checks, advisory post-checks. So rejected models and models with status ``"error"`` still
carry their advisory pre-check results. Pre-checks get ``ModelResult.expr`` (simplified if
``simplify_expr=True``), post-checks the ``ModelResult``. Checks a constraint does not implement
are skipped.

A check that raises an exception, or returns something other than a ``CheckResult`` (treated as a
``TypeError``), emits a ``UserWarning`` and counts as failed, with
[CheckResult.error][SRToolkit.utils.types.CheckResult] set to ``"<ExceptionType>: <message>"``. For
a required constraint the status is then ``"error"`` with the ``status_detail``
``"<ClassName>.<pre_check|post_check> raised <ExceptionType>: <message>"``. For an advisory
constraint the result is recorded in ``advisory_violations`` like any failed check; leave results
with ``error`` set out when counting violations.

To avoid spending evaluations on expressions that would fail a pre-check, or to filter candidates
by the advisory constraints, an approach can call
[check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints] first: it runs
the pre-checks only and is not billed.

## Background knowledge

The search sees only the evaluator, so the evaluator carries the task's
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge] as
[background_knowledge][SRToolkit.evaluation.sr_evaluator.SR_evaluator.background_knowledge].
[describe][SRToolkit.evaluation.sr_evaluator.SR_evaluator.describe] and
[priors][SRToolkit.evaluation.sr_evaluator.SR_evaluator.priors] give the task as text and as
records, the same as the dataset's, e.g. to build an LLM prompt. Neither mentions the answer, but
the evaluator does not hide it: ``ground_truth`` is a public attribute, which an approach must not
read.

## Evaluation trace

The evaluator records every counted call to
[evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] in order, and each
[EvalResult][SRToolkit.utils.types.EvalResult] carries this record as ``trace``: ``trace[k]`` is the
index into ``all_models`` of the expression requested by call ``k + 1``. Cache hits appear as
repeated indices, since they count toward ``max_evaluations``; calls made after evaluation stopped
are not recorded. Three methods derive the usual fixed-target and fixed-budget quantities from it:

- [best_so_far][SRToolkit.utils.types.EvalResult.best_so_far]: the calls at which the best error
  strictly improved;
- [calls_to_threshold][SRToolkit.utils.types.EvalResult.calls_to_threshold]: the number of calls
  needed to first reach an error below a threshold (``success_threshold`` by default), or ``None``;
- [best_error_within][SRToolkit.utils.types.EvalResult.best_error_within]: the best error within each
  of the given numbers of calls.

A run that never reaches the threshold is censored, not missing: it shows that more than
``evaluation_calls`` calls were needed. Keep such runs when aggregating, for example as the
fraction of runs solved within each budget:

```python
import numpy as np
from SRToolkit.experiments import ExperimentGrid

grid = ExperimentGrid.load("results/grid.json")
results = grid.load_results("Nguyen-1", "ProGED")

budgets = np.array([100, 1_000, 10_000])  # at most the experiment's max_evaluations
needed = [r.calls_to_threshold() for r in results]
solved = [np.mean([n is not None and n <= b for n in needed]) for b in budgets]

curves = np.stack([r.best_error_within(budgets) for r in results])
median_curve = np.nanmedian(curves, axis=0)
```

A result does not record ``max_evaluations`` or why its search stopped. For a run that stopped
early for another reason than reaching the threshold (e.g. a time limit), ``best_error_within``
past its last call is only an upper bound, and a ``None`` from ``calls_to_threshold`` is censored
at fewer calls than the budget. If your runs can stop early, check ``evaluation_calls`` against the
budget before comparing approaches.

Note:
    [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] is defined here rather than
    in ``result_augmentation`` to avoid circular imports.
"""

import dataclasses
import importlib
import json
import logging
import math
import os
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from typing_extensions import Literal, Unpack

from SRToolkit.evaluation.callbacks import (
    BestExpressionFound,
    CallbackDispatcher,
    EarlyStoppingCallback,
    ExprEvaluated,
    SRCallbacks,
)
from SRToolkit.evaluation.parameter_estimator import DEFAULT_ESTIMATION_SETTINGS
from SRToolkit.evaluation.ranking import RankingFunction, build_ranking
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_constraints import (
    ExpressionConstraint,
    MaxConstants,
    MaxTokens,
    _set_up_copy,
    _validate_limit,
)
from SRToolkit.utils.expression_simplifier import simplify
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, _as_tokens, tokens_to_tree
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.measures import REGRESSION_METRICS
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, EvalResult, ModelResult, RankingSettings, Status

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this module.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo

logger = logging.getLogger(__name__)


def evaluate_metric(
    expr: Union[List[str], Node],
    parameters: Optional[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    metric: Union[str, Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float]] = "r2",
    symbol_library: Optional[SymbolLibrary] = None,
    backend: str = "stack",
) -> float:
    """Score an expression on ``(X, y)`` by reusing already-fitted ``parameters`` — no refitting.

    Builds a constants-bound prediction function ``predict(X) -> y_pred`` from ``expr`` and
    ``parameters`` and applies ``metric``. This is the shared scoring core behind
    [SR_evaluator.get_metric][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_metric] (train
    data) and [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric] (held-out
    data) — the same call against different arrays.

    Args:
        expr: Expression as an infix token list or a [Node][SRToolkit.utils.expression_tree.Node].
        parameters: Fitted constants to bind in. ``None`` is treated as an empty array.
        X: Inputs of shape ``(n_samples, n_features)`` to score on.
        y: Targets of shape ``(n_samples,)`` to score against.
        metric: A built-in name in
            [REGRESSION_METRICS][SRToolkit.utils.measures.REGRESSION_METRICS]
            (``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"``, ``"nrmse"``) or a callable
            ``metric(predict, X, y) -> float``.
        symbol_library: Symbol library used to compile ``expr``. Defaults to the package default.
        backend: Compilation backend forwarded to
            [compile_expr][SRToolkit.utils.expression_compiler.compile_expr]. Default ``"stack"``.

    Returns:
        The metric value as a float, or ``NaN`` if the metric raises during computation.

    Raises:
        ValueError: If ``metric`` is a string that is not a known built-in metric.
        TypeError: If ``metric`` is neither a string nor a callable.
    """
    if isinstance(metric, str):
        if metric not in REGRESSION_METRICS:
            raise ValueError(
                f"[evaluate_metric] Unknown metric '{metric}'. Available built-in metrics: "
                f"{sorted(REGRESSION_METRICS)}. Alternatively pass a callable with signature "
                "metric(predict, X, y) -> float."
            )
        metric_fn = REGRESSION_METRICS[metric]
    elif callable(metric):
        metric_fn = metric
    else:
        raise TypeError(f"[evaluate_metric] metric must be a string or a callable, got {type(metric).__name__}.")

    expr_list = _as_tokens(expr, symbol_library)

    raw = compile_expr(expr_list, symbol_library, backend=backend)
    params = parameters if parameters is not None else np.array([])

    def predict(X_in: np.ndarray) -> np.ndarray:
        return raw(X_in, params)

    try:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            return float(metric_fn(predict, X, y))
    except Exception as e:
        warnings.warn(f"[evaluate_metric] Error computing metric for '{''.join(expr_list)}': {e}")
        return float("nan")


class ResultAugmenter(ABC):
    _type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._type = f"{cls.__module__}.{cls.__qualname__}"

    def __init__(self, name: str):
        """Base class for result augmenters. Subclasses implement
        [write_results][SRToolkit.evaluation.sr_evaluator.ResultAugmenter.write_results] to compute
        and store additional data in an [EvalResult][SRToolkit.utils.types.EvalResult] via
        [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation].

        For concrete implementations, see
        [result_augmentation][SRToolkit.evaluation.result_augmentation].

        Args:
            name: Identifier used as the key in
                ``augmentations`` dict of [EvalResult][SRToolkit.utils.types.EvalResult] and
                [ModelResult][SRToolkit.utils.types.ModelResult].
                If two augmenters share the same name,
                [add_augmentation][SRToolkit.utils.types.EvalResult.add_augmentation] appends a
                numeric suffix automatically.
        """
        self.name = name

    @abstractmethod
    def write_results(
        self,
        results: "EvalResult",
    ) -> None:
        """Compute and write augmentation data into *results* and its models.

        Call ``results.add_augmentation(self.name, data, self._type)`` for experiment-level
        data and ``model.add_augmentation(self.name, data, self._type)`` for per-model data.
        ``data`` may hold only values that can be saved (see
        [serialization][SRToolkit.utils.serialization]), or saving the results raises ``TypeError``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.
        """

    @abstractmethod
    def to_dict(self, base_path: str, name: str) -> dict:
        """Transforms the augmenter into a dictionary. This is used for saving the augmenter to disk.

        Args:
            base_path: The base path used for saving the data inside the augmenter, if needed.
            name: The name/identifier used by the augmenter for saving.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        """Returns a formatted string for experiment-level augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        return "\n".join(f"  {k}: {v}" for k, v in data.items() if k != "_type")

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        """Returns a formatted string for a single model's augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
        return ", ".join(parts)

    @staticmethod
    def from_dict(data: dict) -> "ResultAugmenter":
        """Creates an instance of the ResultAugmenter class from the dictionary with the relevant data.

        Subclasses should override this method if they support serialization. The default
        implementation raises ``NotImplementedError``, allowing custom augmenters to skip
        serialization if not needed.

        Args:
            data: the dictionary containing the data needed to recreate the augmenter.

        Returns:
            An instance of the ResultAugmenter class with the same configuration as in the data dictionary.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "from_dict is not implemented for this augmenter. "
            "Override this method if your augmenter supports serialization."
        )


def _as_constraint_tuple(constraints: Any, name: str) -> Tuple[ExpressionConstraint, ...]:
    # The items of a constraint list as a tuple; TypeError naming the offending argument or item.
    try:
        items = tuple(constraints)
    except TypeError:
        raise TypeError(
            f"[SR_evaluator] {name} must be an iterable of ExpressionConstraint, got {constraints!r}."
        ) from None
    for item in items:
        if not isinstance(item, ExpressionConstraint):
            raise TypeError(f"[SR_evaluator] {name} must contain ExpressionConstraint instances, got {item!r}.")
    return items


def _add_limit_constraints_from_kwargs(
    constraints: Tuple[ExpressionConstraint, ...], kwargs: Mapping[str, Any]
) -> Tuple[ExpressionConstraint, ...]:
    # MaxConstants then MaxTokens from the max_constants / max_expr_length settings (negative = none),
    # followed by the given constraints. A setting adds nothing when the list already has a
    # constraint of its class; a given non-negative setting with a different limit is a ValueError.
    built: List[ExpressionConstraint] = []
    for key, cls in (("max_constants", MaxConstants), ("max_expr_length", MaxTokens)):
        value = _validate_limit(kwargs.get(key, DEFAULT_ESTIMATION_SETTINGS[key]), key, allow_negative=True)
        # cls is MaxConstants or MaxTokens, so every listed constraint has a limit.
        listed = cast(List[Union[MaxConstants, MaxTokens]], [c for c in constraints if isinstance(c, cls)])
        if not listed:
            if value >= 0:
                built.append(cls(value))
        else:
            clash = next((c for c in listed if c.limit != value), None)
            if key in kwargs and value >= 0 and clash is not None:
                raise ValueError(
                    f"[SR_evaluator] {key}={value} disagrees with {cls.__name__}({clash.limit}) in "
                    f"required_constraints; remove the setting or the constraint."
                )
    return tuple(built) + constraints


def _failed_checks(
    constraints: Sequence[ExpressionConstraint], stage: str, subject: Any, first_only: bool = False
) -> List[CheckResult]:
    # The failed checks of one stage, in order, with constraint and stage filled in. A check that
    # raises, or returns something other than a CheckResult, counts as failed: it gives
    # passed=False, amount inf and error "<Type>: <message>", and a UserWarning. Passed and
    # unimplemented checks are left out. With first_only, stops after the first failure.
    failed: List[CheckResult] = []
    for constraint in constraints:
        name = type(constraint).__name__
        try:
            result = constraint.pre_check(subject) if stage == "pre" else constraint.post_check(subject)
            if not isinstance(result, CheckResult):
                raise TypeError(f"{stage}_check returned {type(result).__name__}, expected CheckResult")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            warnings.warn(f"[SR_evaluator] {name}.{stage}_check raised {error}", UserWarning)
            result = CheckResult(passed=False, message=f"raised {error}", amount=math.inf, error=error)
        if not result.implemented or result.passed:
            continue
        failed.append(dataclasses.replace(result, constraint=name, stage=stage))
        if first_only:
            break
    return failed


class SR_evaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        symbol_library: Optional[SymbolLibrary] = None,
        max_evaluations: int = -1,
        success_threshold: Optional[float] = None,
        ranking_function: Union[str, RankingFunction] = "rmse",
        ground_truth: Optional[Union[List[str], Node]] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
        required_constraints: Optional[Sequence[ExpressionConstraint]] = None,
        advisory_constraints: Optional[Sequence[ExpressionConstraint]] = None,
        variables: Optional[Mapping[str, "VariableInfo"]] = None,
        target: Optional["VariableInfo"] = None,
        description: str = "",
        reference_grammar: Optional[Grammar] = None,
        **kwargs: Unpack[RankingSettings],
    ):
        """Evaluate expressions on data with a ranking function and collect the results.

        Each distinct expression is scored once and cached; repeating it returns the cached error
        but still counts toward ``max_evaluations``. Statuses, constraints and the evaluation trace
        are described in the [module docs][SRToolkit.evaluation.sr_evaluator]; results are
        collected with [get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results].

        Note:
            Whether two expressions are equivalent is undecidable, so ``success_threshold`` is only
            a proxy for success. Inspect the best expressions manually.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``. RMSE needs ``y`` or a constant-free
                ``ground_truth`` to generate it from; BED can use it in place of a ground truth.
            symbol_library: Symbol library defining the tokens. ``None`` (default): the active
                library ([get_or_default][SRToolkit.utils.symbol_library.SymbolLibrary.get_or_default]).
            max_evaluations: Budget of counted calls to
                [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr],
                cache hits included. ``-1`` (default) means no limit.
            success_threshold: Error below which an expression counts as successful and the search
                stops. ``None`` (default) uses the ranking's default (``1e-7`` for RMSE; for BED
                estimated from the ground truth, or none without one). A negative value is never
                reached.
            ranking_function: ``"rmse"`` (default), ``"bed"``, a
                [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] instance, or a
                ``"module.ClassName"`` import path (see
                [build_ranking][SRToolkit.evaluation.ranking.build_ranking]).
            ground_truth: The target expression, as a token list or a
                [Node][SRToolkit.utils.expression_tree.Node]. BED compares against it; RMSE uses it
                to generate ``y`` when ``y`` is not given.
            seed: Random seed. If given, the constructor also calls ``np.random.seed(seed)``, which
                sets NumPy's global random state.
            metadata: Optional information about this evaluation (e.g. dataset name). A
                ``"dataset_name"`` key is copied to
                [EvalResult][SRToolkit.utils.types.EvalResult] ``dataset_name``.
            required_constraints: Constraints every evaluated expression must satisfy (see the
                [module docs][SRToolkit.evaluation.sr_evaluator]); ``None`` means ``[]``.
                ``max_constants`` and ``max_expr_length`` add a
                [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants] and a
                [MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] in front, unless a
                negative setting disables them or the list already has one of that class.
            advisory_constraints: Constraints whose failed checks are recorded in each model's
                ``advisory_violations`` without changing its error; stored like
                ``required_constraints``. ``None`` (default) means none.
            variables: Information per variable, keyed by token (e.g. ``"X_0"``).
            target: Information about the target.
            description: Free text about the task.
            reference_grammar: A [Grammar][SRToolkit.utils.grammar.grammar.Grammar] an approach may
                start from.
            **kwargs: Settings from [RankingSettings][SRToolkit.utils.types.RankingSettings],
                passed to the ranking function. The ranking's parameter estimator gets
                ``max_constants`` and ``max_expr_length`` as ``-1``, since the evaluator enforces
                them.

        Raises:
            TypeError: If an item of ``required_constraints`` or ``advisory_constraints`` is not
                an [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]
                (e.g. the grammar constraint [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth],
                or a single constraint passed instead of a sequence). The message names the item.
                Also raised as for
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge],
                and by [build_ranking][SRToolkit.evaluation.ranking.build_ranking] for a
                ``ranking_function`` of the wrong type.
            ValueError: If ``max_constants`` or ``max_expr_length`` is not an ``int`` or numpy
                integer (``bool`` is rejected), or is given, non-negative and different from the
                limit of a constraint of its class in ``required_constraints`` (a negative setting
                adds nothing and never clashes); the message names the setting. Also raised as
                for ``BackgroundKnowledge`` (e.g. a ``constant_bounds`` setting that is not finite
                with ``lower < upper``, or ``samplers`` without one sampler per variable), and by
                the ranking's ``setup`` for missing or malformed data (e.g. RMSE with neither ``y``
                nor ``ground_truth``).
            ImportError: If an import-path ``ranking_function`` cannot be imported.

        Attributes:
            X: The input data.
            y: The target values as given (``None`` if RMSE generated them).
            symbol_library: The symbol library used.
            background_knowledge: The background knowledge (read-only).
            ranking: The active [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction]. It
                owns the ranking's state: the
                [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator]
                for RMSE, or the behaviour matrix and points ``bed_X`` for BED.
            kwargs: The settings passed as ``**kwargs``.
            ground_truth: The target expression as passed, or ``None``.
            max_evaluations: The budget of counted calls; ``-1`` for no limit.
            success_threshold: The ``success_threshold`` passed, or the ranking's default if it
                was ``None`` (still ``None`` if the ranking has no default).
            seed: The random seed.
            metadata: The metadata dict as passed.
            models: Cached [ModelResult][SRToolkit.utils.types.ModelResult] for every evaluated
                expression, in the order the expressions were first evaluated.
            trace: One entry per counted call to
                [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]
                (cache hits included, calls after stopping excluded): the index into ``models`` of
                the requested expression.
            should_stop: Whether evaluation has stopped (threshold or budget reached). Later calls
                to ``evaluate_expr`` warn and return ``nan`` without being counted.
        """
        self.kwargs = kwargs
        self.models: List[ModelResult] = list()
        self._model_index: Dict[str, int] = dict()
        self.trace: List[int] = list()
        self.success_threshold = success_threshold
        self.metadata = metadata
        self.ground_truth = ground_truth
        self._experiment_id: str = ""
        self.should_stop = False
        self._current_best_error = float("inf")

        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        self.symbol_library = symbol_library
        self.max_evaluations = max_evaluations
        self.seed = seed

        # Constraints: types first (both lists), then the limit settings, then set-up copies.
        required = (
            () if required_constraints is None else _as_constraint_tuple(required_constraints, "required_constraints")
        )
        advisory = (
            () if advisory_constraints is None else _as_constraint_tuple(advisory_constraints, "advisory_constraints")
        )
        required = _add_limit_constraints_from_kwargs(required, kwargs)
        self._required_constraints = tuple(_set_up_copy(c, symbol_library) for c in required)

        # Background knowledge from the same values the evaluator fits with; it holds the set-up
        # copies of the advisory constraints. Imported here: SRToolkit.dataset imports this module.
        from SRToolkit.dataset.background_knowledge import BackgroundKnowledge

        self._background_knowledge = BackgroundKnowledge(
            symbol_library,
            samplers=kwargs.get("samplers"),
            constant_bounds=kwargs.get("constant_bounds", DEFAULT_ESTIMATION_SETTINGS["constant_bounds"]),
            constraints=advisory,
            reference_grammar=reference_grammar,
            description=description,
            variables={} if variables is None else variables,
            target=target,
        )
        # BackgroundKnowledge stores its constraints as a tuple; the field is typed Iterable.
        self._advisory_constraints = cast(Tuple[ExpressionConstraint, ...], self._background_knowledge.constraints)

        if seed is not None:
            np.random.seed(seed)

        # The ranking function owns all of its data-derived state (parameter estimator for RMSE,
        # behaviour matrix + evaluation points bed_X for BED); the evaluator just calls score().
        # The evaluator enforces the limits itself, so the ranking's parameter estimator does not.
        ranking_settings: Dict[str, Any] = dict(kwargs)
        ranking_settings["max_constants"] = -1
        ranking_settings["max_expr_length"] = -1
        self.ranking: RankingFunction = build_ranking(ranking_function)
        self.ranking.setup(X, y, ground_truth, symbol_library, seed, ranking_settings)

        if self.success_threshold is None:
            self.success_threshold = self.ranking.default_success_threshold()

        self._callbacks: Optional[Union[CallbackDispatcher, SRCallbacks]] = CallbackDispatcher(
            callbacks=[EarlyStoppingCallback(threshold=self.success_threshold, max_evaluations=max_evaluations)]
        )
        self.X = X
        self.y = y

    @property
    def total_evaluations(self) -> int:
        """Number of counted calls to
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr], including
        cache hits but not calls made after evaluation was stopped. Equal to ``len(trace)``.
        """
        return len(self.trace)

    @property
    def invalid(self) -> List[str]:
        """Strings of the evaluated expressions with status ``"invalid"`` (not well-formed), in
        the order they were first evaluated.
        """
        return ["".join(m.expr) for m in self.models if m.status == "invalid"]

    @property
    def ranking_function(self) -> str:
        """The name of the active [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction]
        (e.g. ``"rmse"``, ``"bed"``).
        """
        return self.ranking.name

    @property
    def parameter_estimator(self) -> Any:
        """The [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator]
        owned by the ranking function, or ``None`` if the active ranking fits no constants.
        """
        return getattr(self.ranking, "parameter_estimator", None)

    @property
    def required_constraints(self) -> Tuple[ExpressionConstraint, ...]:
        """The set-up copies of the required constraints, in the order they are checked.

        The ``MaxConstants`` and ``MaxTokens`` added by the settings (if any), then copies of the
        ``required_constraints`` argument. Passing this tuple back as ``required_constraints``
        with the same settings gives the same constraints.
        """
        return self._required_constraints

    @property
    def advisory_constraints(self) -> Tuple[ExpressionConstraint, ...]:
        """The set-up copies of the advisory constraints, in the order given."""
        return self._advisory_constraints

    @property
    def background_knowledge(self) -> "BackgroundKnowledge":
        """What the task tells the approach besides the data (read-only).

        Built in the constructor from the symbol library, ``advisory_constraints``, ``variables``,
        ``target``, ``description``, ``reference_grammar`` and the ``constant_bounds`` /
        ``samplers`` settings, so it describes the task the evaluator scores. Its ``constraints``
        are the advisory constraints.
        """
        return self._background_knowledge

    def describe(self) -> str:
        r"""Return the task as text for people and LLM prompts, without the answer.

        [BackgroundKnowledge.describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe]
        of [background_knowledge][SRToolkit.evaluation.sr_evaluator.SR_evaluator.background_knowledge]
        with [required_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.required_constraints]
        listed as enforced (including a ``MaxConstants`` / ``MaxTokens`` added by the settings).
        For an evaluator from
        [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator] it
        equals [SR_dataset.describe][SRToolkit.dataset.sr_dataset.SR_dataset.describe]. It does not
        mention the ground truth, success threshold, ranking or budget.

        Returns:
            The description, lines separated by ``"\n"``, without a trailing newline.

        Examples:
            >>> from SRToolkit.utils.expression_constraints import MaxTreeDepth
            >>> X = np.array([[1.0], [2.0]])
            >>> se = SR_evaluator(X, X[:, 0], symbol_library=SymbolLibrary.from_symbol_list(["+", "*", "C"], 1),
            ...     advisory_constraints=[MaxTreeDepth(4)])
            >>> print(se.describe())
            Symbols: +, *, C, X_0
            Free constants are fitted in [-5.0, 5.0].
            Enforced constraints (expressions that break them are rejected):
              - at most 8 free constants
            Hints (advisory, may be ignored):
              - expression tree at most 4 levels deep
        """
        return self._background_knowledge.describe(self._required_constraints)

    def priors(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the constraints as records, the structured form of ``describe``.

        For an evaluator from ``create_evaluator`` it equals
        [SR_dataset.priors][SRToolkit.dataset.sr_dataset.SR_dataset.priors].

        Returns:
            ``{"required": [...], "advisory": [...]}``: the
            [summary][SRToolkit.utils.expression_constraints.ExpressionConstraint.summary] of each of
            ``required_constraints`` and ``advisory_constraints``, in order.
        """
        return {
            "required": [c.summary() for c in self._required_constraints],
            "advisory": [c.summary() for c in self._advisory_constraints],
        }

    def get_model(self, expr: Union[List[str], Node]) -> Optional[ModelResult]:
        """Return the result of an already evaluated expression, or ``None``.

        Does not count toward ``max_evaluations`` and fires no callbacks. The expression is matched
        token for token, without simplifying it.

        Args:
            expr: The expression, as a token list or a [Node][SRToolkit.utils.expression_tree.Node].

        Returns:
            The cached [ModelResult][SRToolkit.utils.types.ModelResult], or ``None`` if ``expr`` has
            not been evaluated.

        Examples:
            >>> X = np.array([[1.0], [2.0]])
            >>> se = SR_evaluator(X, X[:, 0], symbol_library=SymbolLibrary.from_symbol_list(["+"], 1))
            >>> se.get_model(["X_0"]) is None
            True
            >>> _ = se.evaluate_expr(["X_0"])
            >>> se.get_model(["X_0"]).status
            'ok'
        """
        index = self._model_index.get("".join(_as_tokens(expr, self.symbol_library)))
        return None if index is None else self.models[index]

    def check_constraints(
        self,
        expr: Union[List[str], Node],
    ) -> Tuple[List[CheckResult], List[CheckResult]]:
        """Run the constraints' pre-checks on an expression without evaluating it.

        The call is free: it reads no data, does not count toward ``max_evaluations``, fires no
        callbacks and records nothing. An approach can use it to skip candidates that would be
        rejected, or to filter them by the advisory constraints. Only pre-checks run, since
        post-checks need a scored model.

        The pre-checks are the ones
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] runs, on the
        same input (the expression is not simplified), so an empty ``required`` list means
        ``evaluate_expr(expr)`` will neither reject ``expr`` in a pre-check nor give it the status
        ``"error"`` there. Unlike evaluation, every failed pre-check is reported, not only the
        first. A check that raises, or returns something other than a ``CheckResult``, is reported
        like a failed one, with ``error`` set (see
        [CheckResult][SRToolkit.utils.types.CheckResult]), and emits a ``UserWarning``. Checks a
        constraint does not implement are skipped.

        Args:
            expr: The expression, as an infix token list or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A tuple ``(required, advisory)`` of lists with one
            [CheckResult][SRToolkit.utils.types.CheckResult] per failed pre-check of
            ``required_constraints`` and of ``advisory_constraints``, in the order of each list,
            with ``constraint`` and ``stage="pre"`` filled in as in
            [ModelResult.advisory_violations][SRToolkit.utils.types.ModelResult]. A list is empty if
            all its pre-checks pass.

        Raises:
            TypeError: If ``expr`` is neither a token list nor a ``Node``.
            ExpressionParseError: If ``expr`` is not a well-formed expression. It is always
                parsed, even when there are no constraints.

        Examples:
            >>> from SRToolkit.utils.expression_constraints import MaxConstants, MaxTreeDepth
            >>> X = np.array([[1.0], [2.0], [3.0]])
            >>> se = SR_evaluator(X, X[:, 0], symbol_library=SymbolLibrary.default_symbols(1),
            ...     required_constraints=[MaxConstants(1)], advisory_constraints=[MaxTreeDepth(2)])
            >>> required, advisory = se.check_constraints(["C", "*", "sin", "(", "X_0", "+", "C", ")"])
            >>> [(r.constraint, r.message) for r in required]
            [('MaxConstants', '2 constants, limit 1')]
            >>> [(r.constraint, r.message) for r in advisory]
            [('MaxTreeDepth', 'depth 4, limit 2')]
            >>> se.check_constraints(["C", "*", "X_0"])
            ([], [])
            >>> se.total_evaluations
            0
        """
        expr_list = _as_tokens(expr, self.symbol_library)
        tokens_to_tree(expr_list, self.symbol_library)
        return (
            _failed_checks(self._required_constraints, "pre", expr_list),
            _failed_checks(self._advisory_constraints, "pre", expr_list),
        )

    def register_callbacks(self, callbacks: Optional[Union[SRCallbacks, CallbackDispatcher]] = None) -> None:
        """Register callbacks for monitoring and early stopping.

        A single [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instance is
        automatically wrapped in a
        [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher].

        Examples:
            >>> from SRToolkit.evaluation.callbacks import EarlyStoppingCallback
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> se.register_callbacks(EarlyStoppingCallback(threshold=1e-6))
            >>> se._callbacks is not None
            True

        Args:
            callbacks: A [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher]
                or a single [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instance.
        """
        if isinstance(callbacks, CallbackDispatcher):
            if self._callbacks is not None:
                if isinstance(self._callbacks, SRCallbacks):
                    old_callbacks = [self._callbacks]
                if isinstance(self._callbacks, CallbackDispatcher):
                    old_callbacks = self._callbacks.get_callbacks()
                else:
                    old_callbacks = []

            self._callbacks = callbacks
            for cb in old_callbacks:
                self._callbacks.add(cb)
        elif isinstance(callbacks, SRCallbacks):
            if isinstance(self._callbacks, CallbackDispatcher):
                self._callbacks.add(callbacks)
            elif isinstance(self._callbacks, SRCallbacks):
                self._callbacks = CallbackDispatcher(callbacks=[self._callbacks, callbacks])
            else:
                self._callbacks = CallbackDispatcher(callbacks=[callbacks])

    def evaluate_expr(
        self,
        expr: Union[List[str], Node],
        simplify_expr: bool = False,
        verbose: int = 0,
    ) -> float:
        """Evaluate an expression, or return its cached error.

        A new expression is parsed, checked against the constraints, scored by the ranking
        function and stored in ``models`` (statuses and the order of checks are in the
        [module docs][SRToolkit.evaluation.sr_evaluator]). An expression evaluated before is not
        scored again; its stored error is returned. Either way the call is appended to ``trace``,
        counts toward ``max_evaluations`` and is reported to the callbacks: ``on_expr_evaluated``
        for every call, plus ``on_best_expression`` when the error improves on the best so far. A
        callback returning ``False`` stops evaluation; by default this happens when an error is
        below ``success_threshold`` or ``max_evaluations`` is reached.

        Note:
            ``simplify_expr=True`` makes a call much slower: simplifying with SymPy usually takes
            longer than scoring the expression. Use it to inspect the best results after a search,
            not on every candidate during it.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> print(se.evaluate_expr(["C", "*", "X_1", "-", "X_0"]) < 1e-6)
            True
            >>> # With simplify_expr, the simplified expression is scored and stored
            >>> X = np.array([[0, 1], [0, 2], [0, 3]])
            >>> se = SR_evaluator(X, np.array([2, 3, 4]), seed=42, success_threshold=-1)
            >>> _ = se.evaluate_expr(["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"], simplify_expr=True)
            >>> "".join(se.models[0].expr), bool(0.99 < se.models[0].parameters[0] < 1.01)
            ('C+X_1', True)
            >>> # A malformed expression gets the error inf and the status "invalid"
            >>> print(se.evaluate_expr(["C", "*", "X_1", "X_0"]))
            inf
            >>> se.models[-1].status
            'invalid'
            >>> # BED is stochastic and scaled by the ground truth's output, so an exact match may
            >>> # still have a small positive error
            >>> X = np.random.rand(10, 2) - 0.5
            >>> se = SR_evaluator(X, ground_truth=["X_0", "+", "C"], ranking_function="bed", seed=42)
            >>> print(se.evaluate_expr(["C", "+", "X_0"]) < 0.2)
            True

        Args:
            expr: The expression, as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            simplify_expr: If ``True``, simplify the expression first; the simplified
                expression is then checked, scored, cached and stored as ``ModelResult.expr``, so
                inputs that simplify alike share one cache entry. If simplification fails, a
                warning is emitted and the expression is evaluated as given. Default ``False``.
            verbose: ``0`` (default): no logging. ``1``: log at DEBUG level (logger
                ``SRToolkit.evaluation.sr_evaluator``) each new expression with its final status,
                error and ``status_detail``, and each cache hit. ``2``: also let NumPy
                floating-point warnings through while scoring.

        Returns:
            The error under the active ranking function, or ``+inf`` if the status is not
            ``"ok"``. ``nan`` if evaluation has already stopped: the call then emits a warning and
            is not counted.

        Raises:
            TypeError: If ``expr`` is neither a token list nor a ``Node``.
        """
        if self.should_stop:
            warnings.warn(
                f"Evaluation stopped because max_evaluations ({self.max_evaluations}) reached or an expression with error lower than success_threshold ({self.success_threshold}) was found. "
            )
            return np.nan
        else:
            if simplify_expr:
                try:
                    expr = simplify(expr, self.symbol_library)
                except Exception as e:
                    expr_list = _as_tokens(expr, self.symbol_library)
                    warnings.warn(f"Unable to simplify: {''.join(expr_list)}, problems with subexpression {e}")

            expr_list = _as_tokens(expr, self.symbol_library)

            expr_str = "".join(expr_list)
            index = self._model_index.get(expr_str)
            if index is not None:
                if verbose > 0:
                    logger.debug("Already evaluated %s", expr_str)
                self.trace.append(index)
                if self._callbacks is not None:
                    event = ExprEvaluated(
                        expression=expr_str,
                        error=self.models[index].error,
                        evaluation_number=self.total_evaluations,
                        experiment_id=self._experiment_id,
                        is_new_best=False,
                        status=self.models[index].status,
                    )
                    if not self._callbacks.on_expr_evaluated(event):
                        self.should_stop = True
                return self.models[index].error

            else:
                model = self._evaluate_new(expr, expr_list, expr_str, verbose)
                if verbose > 0:
                    logger.debug(
                        "Evaluated expression %s: status %s, error %s%s",
                        expr_str,
                        model.status,
                        model.error,
                        f" ({model.status_detail})" if model.status_detail else "",
                    )
                error = model.error
                self._model_index[expr_str] = len(self.models)
                self.trace.append(len(self.models))
                self.models.append(model)

                if self._callbacks is not None:
                    is_new_best = error < self._current_best_error
                    if is_new_best:
                        self._current_best_error = error
                    event = ExprEvaluated(
                        expression=expr_str,
                        error=error,
                        evaluation_number=self.total_evaluations,
                        experiment_id=self._experiment_id,
                        is_new_best=is_new_best,
                        status=model.status,
                    )
                    if not self._callbacks.on_expr_evaluated(event):
                        self.should_stop = True
                    if is_new_best:
                        best_event = BestExpressionFound(
                            experiment_id=self._experiment_id,
                            expression=expr_str,
                            error=error,
                            evaluation_number=self.total_evaluations,
                        )
                        if not self._callbacks.on_best_expression(best_event):
                            self.should_stop = True

                return error

    def _evaluate_new(
        self, expr: Union[List[str], Node], expr_list: List[str], expr_str: str, verbose: int
    ) -> ModelResult:
        # Parse first, so that each failure gets its own status.
        try:
            tokens_to_tree(expr_list, self.symbol_library)
        except ExpressionParseError as e:
            return ModelResult(expr=expr_list, error=np.inf, status="invalid", status_detail=str(e))

        advisory = _failed_checks(self._advisory_constraints, "pre", expr_list)

        failed = _failed_checks(self._required_constraints, "pre", expr_list, first_only=True)
        if failed:
            if failed[0].error is not None:
                return self._constraint_error(failed[0], expr_list, advisory)
            return ModelResult(
                expr=expr_list,
                error=np.inf,
                status="rejected",
                status_detail=f"{failed[0].constraint}: {failed[0].message}",
                advisory_violations=advisory,
            )

        model = self._score(expr, expr_list, expr_str, verbose)
        model.advisory_violations = advisory
        if model.status != "ok":
            return model

        failed = _failed_checks(self._required_constraints, "post", model, first_only=True)
        if failed:
            if failed[0].error is not None:
                return self._constraint_error(failed[0], expr_list, advisory, model.parameters, model.ranking_data)
            model.error = np.inf
            model.status = "rejected"
            model.status_detail = f"{failed[0].constraint}: {failed[0].message}"
            return model

        model.advisory_violations.extend(_failed_checks(self._advisory_constraints, "post", model))
        return model

    def _score(self, expr: Union[List[str], Node], expr_list: List[str], expr_str: str, verbose: int) -> ModelResult:
        try:
            with (
                np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
                if verbose < 2
                else nullcontext()
            ):
                result = self.ranking.score(expr)
        except Exception as e:
            # The expression is well-formed, so this is a failure of the evaluation; warn, since it
            # usually points at a bug rather than at the expression.
            message = f"{type(e).__name__}: {e}"
            warnings.warn(f"[SR_evaluator] Error scoring expression {expr_str}: {message}")
            return ModelResult(expr=expr_list, error=np.inf, status="error", status_detail=message)

        score = float(result.score)
        detail = result.status_detail
        if result.status not in (None, "undefined", "rejected"):
            # Only "rejected" and "undefined" may come from a ranking; see RankingResult.status.
            message = f"the {self.ranking.name} ranking returned a status it may not set: {result.status!r}"
            warnings.warn(f"[SR_evaluator] Error scoring expression {expr_str}: {message}")
            return ModelResult(expr=expr_list, error=np.inf, status="error", status_detail=message)
        status: Status
        if result.status == "rejected":
            status = "rejected"
        elif result.status == "undefined" or not np.isfinite(score):
            status = "undefined"
        else:
            status = "ok"

        if status == "ok":
            detail = None
        elif detail is None:
            detail = f"{self.ranking.name} score is {score}" if status == "undefined" else "rejected by the ranking"

        return ModelResult(
            expr=expr_list,
            error=score if status == "ok" else np.inf,
            parameters=result.parameters,
            ranking_data=result.extra,
            status=status,
            status_detail=detail,
        )

    @staticmethod
    def _constraint_error(
        result: CheckResult,
        expr_list: List[str],
        advisory: List[CheckResult],
        parameters: Optional[np.ndarray] = None,
        ranking_data: Optional[Dict[str, Any]] = None,
    ) -> ModelResult:
        # A required check raised: a failure of the evaluation, not of the expression.
        return ModelResult(
            expr=expr_list,
            error=np.inf,
            parameters=parameters,
            ranking_data=ranking_data if ranking_data is not None else {},
            status="error",
            status_detail=f"{result.constraint}.{result.stage}_check raised {result.error}",
            advisory_violations=advisory,
        )

    def get_metric(
        self,
        expr: Union[List[str], Node],
        metric: Union[str, Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float]] = "r2",
    ) -> float:
        """Compute an alternative, read-only metric for an already-evaluated expression.

        Unlike [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr],
        this does **not** count toward the evaluation budget, does not affect ranking,
        success, caching, or callbacks, and never changes the canonical ranking function.
        It exists so an approach can structure its search around a measure other than the
        ranking error (e.g. R² or NRMSE) without re-running the (budgeted) evaluation and
        without being given direct access to ``X``/``y`` as attributes.

        The expression **must already have been evaluated** via ``evaluate_expr`` so its
        constants are fitted and cached — otherwise a ``ValueError`` is raised. This is the
        guard that keeps the budget meaningful: a custom ``metric`` callable cannot be used to
        score un-evaluated expressions for free.

        The ``metric`` is computed from a constants-bound prediction function
        ``predict(X) -> y_pred`` built from the cached fit. Built-in string metrics and
        custom callables share the signature ``metric(predict, X, y) -> float``; passing
        ``predict`` (rather than a precomputed ``y_pred``) lets a metric also evaluate the
        model away from ``X`` (e.g. for smoothness or extrapolation measures).

        Only available in ``ranking_function="rmse"`` mode. In BED mode (no ``y``,
        no fitted constants), and for models that could not be scored (status other than
        ``"ok"``), this returns ``NaN``.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(se.get_metric(["C", "*", "X_1", "-", "X_0"], "r2") > 0.999)
            True
            >>> # A custom measure: worst-case absolute residual.
            >>> def max_abs_residual(predict, X, y):
            ...     return float(np.max(np.abs(predict(X) - y)))
            >>> print(se.get_metric(["C", "*", "X_1", "-", "X_0"], max_abs_residual) < 1e-6)
            True

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree. Must already have been
                passed to ``evaluate_expr``.
            metric: Either the name of a built-in metric in
                [REGRESSION_METRICS][SRToolkit.utils.measures.REGRESSION_METRICS]
                (``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"``, ``"nrmse"``), or a callable with the
                signature ``metric(predict, X, y) -> float`` where ``predict(X) -> y_pred``
                has the fitted constants bound in. Default ``"r2"``.

        Returns:
            The metric value as a float. Returns ``NaN`` in BED mode, for models that could
            not be scored, or if the metric raises during computation.

        Raises:
            ValueError: If ``expr`` has not been evaluated yet, or if ``metric`` is a
                string that is not a known built-in metric.
            TypeError: If ``metric`` is neither a string nor a callable.
        """
        expr_list = _as_tokens(expr, self.symbol_library)
        expr_str = "".join(expr_list)

        index = self._model_index.get(expr_str)
        if index is None:
            raise ValueError(
                f"[SR_evaluator.get_metric] Expression '{expr_str}' has not been evaluated yet. "
                "Call evaluate_expr() on it first so its constants are fitted and cached."
            )

        # Validate the metric up front so unknown names raise regardless of mode.
        if isinstance(metric, str):
            if metric not in REGRESSION_METRICS:
                raise ValueError(
                    f"[SR_evaluator.get_metric] Unknown metric '{metric}'. Available built-in metrics: "
                    f"{sorted(REGRESSION_METRICS)}. Alternatively pass a callable with signature "
                    "metric(predict, X, y) -> float."
                )
        elif not callable(metric):
            raise TypeError(
                f"[SR_evaluator.get_metric] metric must be a string or a callable, got {type(metric).__name__}."
            )

        model = self.models[index]

        # Prediction-based metrics need targets and a fitted model: undefined in BED mode
        # and for models that could not be scored.
        if self.ranking_function != "rmse" or self.y is None:
            return float("nan")
        if model.status != "ok":
            return float("nan")

        backend = str(self.parameter_estimator.estimation_settings["backend"])
        return evaluate_metric(
            expr_list, model.parameters, self.X, self.y, metric, self.symbol_library, backend=backend
        )

    def get_results(
        self, approach_name: str = "", top_k: int = 20, results: Optional["SR_results"] = None
    ) -> "SR_results":
        """Returns the results of the equation discovery/symbolic regression process/evaluation.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> print(results[0].num_evaluated)
            1
            >>> print(results[0].evaluation_calls)
            1
            >>> print(results[0].best_expr)
            C*X_1-X_0
            >>> print(results[0].min_error < 1e-6)
            True
            >>> print(1.99 < results[0].top_models()[0].parameters[0] < 2.01)
            True

        Args:
            approach_name: The name of the approach used to discover the equations.
            top_k: The number of top results to include in the output. If `top_k`
                is greater than the number of evaluated expressions, all
                evaluated expressions are included. If `top_k` is less than 0,
                all evaluated expressions are included.
            results: An SR_results object containing the results of the previous evaluation. If provided,
                the results of the current evaluation are appended to the existing results. Otherwise, a new SR_results
                object is created.

        Returns:
            An instance of the SR_results object with the results of the evaluation.
        """
        if results is None:
            results = SR_results()

        results.add_results(
            self.models,
            self.trace,
            top_k,
            self.success_threshold,
            approach_name,
            self.metadata,
        )

        return results


class SR_results:
    def __init__(self):
        """Container for SR experiment results, typically obtained via
        [SR_evaluator.get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results].

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> print(results[0].best_expr)
            C*X_1-X_0
            >>> print(results[0].min_error < 1e-6)
            True
            >>> len(results)
            1

        Attributes:
            results: List of [EvalResult][SRToolkit.utils.types.EvalResult] instances,
                one per experiment.
        """
        self.results = list()

    def add_results(
        self,
        models: List[ModelResult],
        trace: List[int],
        top_k: int,
        success_threshold: Optional[float],
        approach_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Adds the results of an evaluation to the results object.

        Args:
            models: Evaluated models in the order they were first evaluated.
            trace: The evaluation trace: the index into ``models`` requested by each counted call.
            top_k: Default number of models returned by ``top_models()``. Negative means all.
            success_threshold: The success threshold used to determine whether the evaluation was successful.
            approach_name: The name of the approach used to discover the equations.
            metadata: A dictionary containing additional metadata about the evaluation.
        """
        dataset_name = None
        remaining_metadata = None
        if metadata is not None and "dataset_name" in metadata:
            dataset_name = metadata["dataset_name"]
            remaining_metadata = {key: value for key, value in metadata.items() if key != "dataset_name"}
            if len(remaining_metadata) == 0:
                remaining_metadata = None
        elif metadata is not None:
            remaining_metadata = metadata

        results_obj = EvalResult(
            all_models=list(models),
            trace=list(trace),
            approach_name=approach_name,
            success_threshold=success_threshold,
            top_k=top_k,
            dataset_name=dataset_name,
            metadata=remaining_metadata,
        )

        self.results.append(results_obj)

    def print_results(
        self,
        experiment_number: Optional[int] = None,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations: Optional[List[str]] = None,
    ):
        r"""Prints the results of the SR_evaluator.

        Displays the minimum error, best expression, evaluation counts, success status,
        metadata, and approach name. When *detailed* is ``True``, also prints per-model
        information. Augmentation data is formatted by the corresponding
        [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] subclass,
        looked up from the global registry via the ``_type`` field stored in each
        augmentation entry.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> results.print_results()  # doctest: +ELLIPSIS
            === Experiment 1/1 ===
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            <BLANKLINE>
            >>> results.print_results(detailed=True, experiment_number=0)  # doctest: +ELLIPSIS
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            <BLANKLINE>
            Models:
              C*X_1-X_0  (error=..., params=...)
            <BLANKLINE>

        Args:
            experiment_number: Number of the experiment to print. If None, prints all.
            detailed: If True, prints per-model information.
            model_scope: Which models to show when *detailed* is True.
                ``"best"`` shows only the top model, ``"top"`` shows the top-k,
                ``"all"`` shows all evaluated models.
            augmentations: Filter which augmenters to display by name.
                If None, all augmentations present in the data are shown.

        Raises:
            TypeError: If ``experiment_number`` is not an integer.
            IndexError: If ``experiment_number`` is out of range. Negative numbers count from the end.
        """
        if experiment_number is None:
            for i, result in enumerate(self.results):
                print(f"=== Experiment {i + 1}/{len(self.results)} ===")
                SR_results._print_result_(result, detailed, model_scope, augmentations)
                print()
        else:
            SR_results._print_result_(self[experiment_number], detailed, model_scope, augmentations)

    @staticmethod
    def _print_result_(
        result: EvalResult,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations_filter: Optional[List[str]] = None,
    ):
        def _get_augmenter_class(type_str: str) -> Optional[type]:
            try:
                module_path, cls_name = type_str.rsplit(".", 1)
                return getattr(importlib.import_module(module_path), cls_name)
            except Exception:
                return None

        # --- Core info ---
        if result.dataset_name is not None:
            print(f"Dataset: {result.dataset_name}")
        if result.approach_name != "":
            print(f"Approach: {result.approach_name}")
        print(f"Best expression: {result.best_expr}")
        print(f"Error: {result.min_error}")
        print(
            f"Evaluated: {result.num_evaluated} expressions | Calls: {result.evaluation_calls} | Success: {result.success}"
        )
        if result.wall_time is not None:
            print(f"Wall time: {result.wall_time:.3f}s")
        if result.metadata is not None and len(result.metadata) > 0:
            print("Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        print()

        # --- Resolve which augmenter keys to show ---
        eval_aug_keys = list(result.augmentations.keys())
        if augmentations_filter is not None:
            eval_aug_keys = [k for k in eval_aug_keys if k in augmentations_filter]

        # --- Experiment-level augmentation sections ---
        for key in eval_aug_keys:
            data = result.augmentations[key]
            type_str = data.get("_type", "")
            cls = _get_augmenter_class(type_str) if type_str else None
            print(f"--- {key} ---")
            if cls is not None and issubclass(cls, ResultAugmenter):
                line = cls.format_eval_result(data)
                if line:
                    print(line)
            else:
                for k, v in data.items():
                    if k != "_type":
                        print(f"  {k}: {v}")

        # --- Per-model output ---
        if detailed:
            # Pick model list based on scope
            if model_scope == "best":
                models = result.top_models(1)
            elif model_scope == "top":
                models = result.top_models()
            else:
                models = result.all_models

            if models:
                print()
                print("Models:")
                for model in models:
                    parts = [f"error={model.error:.6g}"]
                    if model.parameters is not None:
                        parts.append(f"params={np.round(model.parameters, 4).tolist()}")
                    expr_str = "".join(model.expr)
                    print(f"  {expr_str}  ({', '.join(parts)})")

                    # Model-level augmentations
                    model_aug_keys = list(model.augmentations.keys())
                    if augmentations_filter is not None:
                        model_aug_keys = [k for k in model_aug_keys if k in augmentations_filter]
                    for key in model_aug_keys:
                        data = model.augmentations[key]
                        type_str = data.get("_type", "")
                        cls = _get_augmenter_class(type_str) if type_str else None
                        if cls is not None and issubclass(cls, ResultAugmenter):
                            line = cls.format_model_result(data)
                            if line:
                                print(f"    {key}: {line}")
                        else:
                            parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
                            if parts:
                                print(f"    {key}: {', '.join(parts)}")

    def augment(
        self, augmenters: Union[List[ResultAugmenter], ResultAugmenter], experiment_number: Optional[int] = None
    ) -> None:
        r"""Applies the given [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter]
        instances to the stored results. Augmenters add post-hoc information such as LaTeX
        representations, simplified expressions, or R² scores.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> from SRToolkit.evaluation.result_augmentation import ExpressionToLatex
            >>> results.augment([ExpressionToLatex(SymbolLibrary.default_symbols(2))])
            >>> results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]  # doctest: +ELLIPSIS
            '$C_{0} \\cdot X_{1} - X_{0}$'

        Args:
            augmenters: A [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] or a list of [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] objects to apply to the results.
            experiment_number: If provided, apply augmenters only to this experiment's result.
                If None, apply to all results.
        """
        if isinstance(augmenters, ResultAugmenter):
            augmenters = [augmenters]

        if experiment_number is not None:
            assert experiment_number < len(self.results), "[SR_results.augment] experiment number out of bounds"
            for augmenter in augmenters:
                try:
                    augmenter.write_results(self.results[experiment_number])
                except Exception as e:
                    warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")
        else:
            for result in self.results:
                for augmenter in augmenters:
                    try:
                        augmenter.write_results(result)
                    except Exception as e:
                        warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")

    def __add__(self, other: "SR_results") -> "SR_results":
        """Returns a new SR_results object that is the concatenation of the current SR_results object with the other SR_results object.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            A new SR_results object containing the concatenated results.
        """
        new = SR_results()
        new.results = self.results + other.results
        return new

    def __iadd__(self, other: "SR_results") -> "SR_results":
        """In-place concatenation of SR_results objects.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            self
        """
        self.results += other.results
        return self

    def __getitem__(self, item: int) -> EvalResult:
        """Returns the results of the experiment with the given index.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> result_of_first_experiment = results[0]

        Args:
            item: the index of the experiment. Negative indices count from the end, as for a list.

        Returns:
            The results of the experiment with the given index.

        Raises:
            TypeError: If ``item`` is not an integer.
            IndexError: If ``item`` is out of range.
        """
        if not isinstance(item, int):
            raise TypeError(f"[SR_results.__getitem__] Index must be an integer, got {type(item).__name__}.")
        if not -len(self.results) <= item < len(self.results):
            raise IndexError(f"[SR_results.__getitem__] Index {item} out of range for {len(self.results)} results.")
        return self.results[item]

    def __iter__(self) -> Iterator[EvalResult]:
        """Iterates over the results stored in the results object, in the order they were added. Usually, each result corresponds to a single experiment.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> for r in results:
            ...     print(r.best_expr)
            C*X_1-X_0

        Returns:
            An iterator over the [EvalResult][SRToolkit.utils.types.EvalResult] objects stored in the results object.
        """
        return iter(self.results)

    def __len__(self) -> int:
        """Returns the number of results stored in the results object. Usually, each result corresponds to a single experiment.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> len(results)
            1

        Returns:
            The number of results stored in the results object.
        """
        return len(self.results)

    def save(self, path: str) -> None:
        """Saves the results to a specific file or directory as JSON.

        If *path* is an existing directory, it writes ``results.json`` inside it.
        If *path* is a file path, it must end with the ``.json`` extension.

        Examples:
            >>> import tempfile
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     results.save(tmpdir + "/my_results/results.json")
            ...     loaded = SR_results.load(tmpdir + "/my_results/results.json")
            ...     print(loaded[0].best_expr)
            C*X_1-X_0

        Args:
            path: Directory path or specific .json file path.

        Raises:
            ValueError: If the path is a file with an extension other than .json.
            OSError: If the directory cannot be created.
            TypeError: If a result holds augmentation or ranking data that cannot be saved (see
                [serialization][SRToolkit.utils.serialization]).
        """
        if os.path.isdir(path):
            target_file = os.path.join(path, "results.json")
        else:
            _, extension = os.path.splitext(path)

            if extension == "":
                target_file = os.path.join(path, "results.json")
            elif extension.lower() == ".json":
                target_file = path
            else:
                raise ValueError(f"Invalid file extension '{extension}'. Results must be saved to a '.json' file.")

        target_dir = os.path.dirname(target_file)
        if target_dir and not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        # 3. Prepare data
        output = {
            "format_version": 1,
            "type": "SR_results",
            "results": [r.to_dict() for r in self.results],
        }

        with open(target_file, "w") as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def load(path: str) -> "SR_results":
        """Load results previously saved with [save][SRToolkit.evaluation.sr_evaluator.SR_results.save].

        If *path* is a directory, it looks for ``results.json`` inside it.
        If *path* is a file, it must end with the ``.json`` extension.

        Args:
            path: Directory path containing ``results.json`` or path to a specific .json file.

        Returns:
            A new [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] instance with the loaded data.

        Raises:
            FileNotFoundError: If the specified file or directory does not exist.
            ValueError: If the file extension is not .json or if ``format_version`` is not ``1``.
        """
        if os.path.isdir(path):
            results_path = os.path.join(path, "results.json")
        else:
            _, extension = os.path.splitext(path)
            if extension.lower() != ".json":
                raise ValueError(
                    f"Invalid file extension '{extension}'. SR_results can only be loaded from '.json' files."
                )
            results_path = path

        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Could not find results file at: {results_path}")

        with open(results_path, "r") as f:
            data = json.load(f)

        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[SR_results.load] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )

        sr_results = SR_results()
        sr_results.results = [EvalResult.from_dict(r) for r in data["results"]]
        return sr_results
