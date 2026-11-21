"""Expression evaluation for symbolic regression.

Contains [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator], which ranks expressions
with a [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] (RMSE by default) and
collects them into [SR_results][SRToolkit.evaluation.results.SR_results].

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

Whatever its status above, a model that cannot be saved (see
[ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict], e.g. a ranking's ``extra`` with
a value that cannot be saved) becomes ``"error"``: it is stored with empty ``ranking_data`` and
``advisory_violations``, ``parameters`` ``None`` and a ``status_detail`` starting
``"the model cannot be saved: "``, and the evaluator emits a warning.

Every status except ``"ok"`` gets the error ``+inf``, so an evaluated expression never gives
``NaN`` and failed models rank after every scored one. Every call counts toward
``max_evaluations``, whatever its status.

## Stopping

Evaluation stops at the first of these, which is recorded as ``stop_reason`` and saved in the
results as [EvalResult.stop_reason][SRToolkit.utils.types.EvalResult]:

- ``"success_threshold"``: a new expression has an error below ``success_threshold``;
- ``"max_evaluations"``: the call used the last evaluation of the budget (a call that does both
  records ``"success_threshold"``);
- a callback asked to stop (see [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks]), e.g.
  ``"time_limit"``;
- the approach called [stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop] with its own
  reason, e.g. because its search space is exhausted.

An approach loops while ``should_stop`` is ``False``. If its search returns without any of these,
the results record ``"returned"``. Once evaluation has stopped, further calls are not evaluated:
they warn, return ``nan`` and are not counted. A ``stop_reason`` other than
``"success_threshold"`` or ``"max_evaluations"`` marks a run that ended with budget left.

The statuses are saved with the results, so validity can be measured after the run from
[status_counts][SRToolkit.utils.types.EvalResult.status_counts]. Count every well-formed
expression as valid, including rejected and undefined ones, and leave ``"error"`` out entirely,
because those failures belong to the evaluation, not to the approach:

```python
counts = result.status_counts()  # per unique expression; per="call" counts every call
validity_rate = 1 - counts["invalid"] / (sum(counts.values()) - counts["error"])
```

## Free constants

An approach can pass specs for the labelled constants of an expression (priors, bounds, fixed
values; see [free_constants][SRToolkit.utils.free_constants]) as the ``free_constants`` argument of
[evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]. A **model** is an
expression together with its specs: the same tokens with different specs are two models, each
scored, cached, counted in ``num_evaluated`` and ``status_counts`` and stored in ``models`` on its
own. Specs are matched by value (the fields that affect the score) and in any order. They are stored
in [ModelResult.free_constants][SRToolkit.utils.types.ModelResult] sorted by label (string order),
or ``None`` when none (or an empty list) were given. Parameters without a spec, and a spec's unset
``prior`` or ``bounds``, use the evaluator's ``default_constant``, which is recorded in the results
as [EvalResult.default_constant][SRToolkit.utils.types.EvalResult]. Expressions that differ only in
how their constants are written (``C_0*X_0 + C_0`` and ``C_1*X_0 + C_1``) are separate models
unless the approach passes ``canonicalise_constants=True``, which
[canonicalises][SRToolkit.utils.free_constants.canonicalise] them first.

Specs are checked right after parsing: a malformed expression is ``"invalid"`` (and stored with its
specs) whatever they are. Specs that do not fit a well-formed expression, and specs combined with
``simplify_expr=True``, raise; nothing is then recorded or counted. The ranking's
[score][SRToolkit.evaluation.ranking.RankingFunction.score] receives the specs as
``free_constants``.
Callback events, [invalid][SRToolkit.evaluation.sr_evaluator.SR_evaluator.invalid] and
[best_expr][SRToolkit.utils.types.EvalResult.best_expr] carry only the token string, so two models
with equal tokens look alike there.

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

grid = ExperimentGrid.from_dict("results/grid.json")
results = grid.load_results("Nguyen-1", "ProGED")

budgets = np.array([100, 1_000, 10_000])  # at most the experiment's max_evaluations
needed = [r.calls_to_threshold() for r in results]
solved = [np.mean([n is not None and n <= b for n in needed]) for b in budgets]

curves = np.stack([r.best_error_within(budgets) for r in results])
median_curve = np.nanmedian(curves, axis=0)
```

A run whose ``stop_reason`` is neither ``"success_threshold"`` nor ``"max_evaluations"`` (e.g.
``"time_limit"``) ended with budget left: ``best_error_within`` past its last call is only an upper
bound, and a ``None`` from ``calls_to_threshold`` is censored at fewer calls than the budget. Check
``stop_reason`` before comparing approaches.

## Storage

``models`` and ``trace`` are read-only views of a SQLite database, in memory by default or in the
``checkpoint`` file, which keeps memory constant during long runs (see
[model_store][SRToolkit.evaluation.model_store]). ``se.models[i]`` returns a new
[ModelResult][SRToolkit.utils.types.ModelResult] each time, so changing it saves nothing. An
evaluator holds an open database connection, so it cannot be pickled or deep-copied. It is not
thread-safe: call it from one thread at a time.
"""

import dataclasses
import logging
import math
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np
from typing_extensions import Unpack

from SRToolkit.evaluation.callbacks import (
    BestExpressionFound,
    CallbackDispatcher,
    ExprEvaluated,
    SRCallbacks,
)
from SRToolkit.evaluation.model_store import ModelStore
from SRToolkit.evaluation.parameter_estimator import DEFAULT_ESTIMATION_SETTINGS
from SRToolkit.evaluation.ranking import RankingFunction, build_ranking
from SRToolkit.evaluation.results import SR_results
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
from SRToolkit.utils.free_constants import Constant, _spec_key, canonicalise, resolve_constants
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.measures import REGRESSION_METRICS
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, ModelResult, RankingSettings, Status

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this module.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo

logger = logging.getLogger(__name__)


def evaluate_metric(
    expr: list[str] | Node,
    parameters: np.ndarray | None,
    X: np.ndarray,
    y: np.ndarray,
    metric: str | Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float] = "r2",
    symbol_library: SymbolLibrary | None = None,
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


def _as_constraint_tuple(constraints: Any, name: str) -> tuple[ExpressionConstraint, ...]:
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


# Defaults of the limit settings, turned into MaxConstants / MaxTokens (negative = none).
_DEFAULT_LIMITS: dict[str, int] = {"max_constants": 8, "max_expr_length": -1}


def _add_limit_constraints_from_kwargs(
    constraints: tuple[ExpressionConstraint, ...], kwargs: Mapping[str, Any]
) -> tuple[ExpressionConstraint, ...]:
    # MaxConstants then MaxTokens from the max_constants / max_expr_length settings (negative = none),
    # followed by the given constraints. A setting adds nothing when the list already has a
    # constraint of its class; a given non-negative setting with a different limit is a ValueError.
    built: list[ExpressionConstraint] = []
    for key, cls in (("max_constants", MaxConstants), ("max_expr_length", MaxTokens)):
        value = _validate_limit(kwargs.get(key, _DEFAULT_LIMITS[key]), key, allow_negative=True)
        # cls is MaxConstants or MaxTokens, so every listed constraint has a limit.
        listed = cast(list[MaxConstants | MaxTokens], [c for c in constraints if isinstance(c, cls)])
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
) -> list[CheckResult]:
    # The failed checks of one stage, in order, with constraint and stage filled in. A check that
    # raises, or returns something other than a CheckResult, counts as failed: it gives
    # passed=False, amount inf and error "<Type>: <message>", and a UserWarning. Passed and
    # unimplemented checks are left out. With first_only, stops after the first failure.
    failed: list[CheckResult] = []
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
        y: np.ndarray | None = None,
        symbol_library: SymbolLibrary | None = None,
        max_evaluations: int = -1,
        success_threshold: float | None = None,
        ranking_function: str | RankingFunction = "rmse",
        ground_truth: list[str] | Node | None = None,
        seed: int | None = None,
        metadata: dict | None = None,
        required_constraints: Sequence[ExpressionConstraint] | None = None,
        advisory_constraints: Sequence[ExpressionConstraint] | None = None,
        variables: Mapping[str, "VariableInfo"] | None = None,
        target: "VariableInfo | None" = None,
        named_constants: Mapping[str, "VariableInfo"] | None = None,
        description: str = "",
        reference_grammar: Grammar | None = None,
        checkpoint: str | os.PathLike | None = None,
        **kwargs: Unpack[RankingSettings],
    ):
        """Evaluate expressions on data with a ranking function and collect the results.

        Each distinct model (an expression with its free-constant specs) is scored once and
        cached; repeating it returns the cached error but still counts toward
        ``max_evaluations``. Statuses, free constants, constraints and the evaluation trace
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
                cache hits included. A negative value (default ``-1``) means no limit.
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
                sets NumPy's global random state. The ranking gets it too, unless the
                ``ranking_seed`` setting is given.
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
            named_constants: Information per named constant, keyed by ``"lit"`` token (e.g. ``"c"``).
            description: Free text about the task.
            reference_grammar: A [Grammar][SRToolkit.utils.grammar.grammar.Grammar] an approach may
                start from.
            checkpoint: A file to create and store the models and trace in, committed as described
                in [model_store][SRToolkit.evaluation.model_store]. ``None`` (default): they are
                kept in memory.
            **kwargs: Settings from [RankingSettings][SRToolkit.utils.types.RankingSettings],
                passed to the ranking function. A partial ``default_constant`` template is completed from
                ``constant_bounds`` (as by ``BackgroundKnowledge``) before the ranking gets it.

        Raises:
            TypeError: If ``max_evaluations`` is not an ``int`` or numpy integer (``bool`` is
                rejected). If an item of ``required_constraints`` or ``advisory_constraints`` is not
                an [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]
                (e.g. the grammar constraint [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth],
                or a single constraint passed instead of a sequence). The message names the item.
                Also raised as for
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge],
                and by [build_ranking][SRToolkit.evaluation.ranking.build_ranking] for a
                ``ranking_function`` of the wrong type.
            ValueError: If ``max_evaluations`` is ``0``, or ``checkpoint`` is an empty path. If ``max_constants`` or ``max_expr_length``
                is not an ``int`` or numpy integer (``bool`` is rejected), or is given, non-negative and different from the
                limit of a constraint of its class in ``required_constraints`` (a negative setting
                adds nothing and never clashes); the message names the setting. Also raised as
                for ``BackgroundKnowledge`` (e.g. a ``constant_bounds`` setting that is not finite
                with ``lower < upper``, a ``default_constant`` setting with a ``label``, a
                ``value`` or ``fixed=True``, or ``samplers`` without one sampler per variable), and by
                the ranking's ``setup`` for missing or malformed data (e.g. RMSE with neither ``y``
                nor ``ground_truth``).
            ImportError: If an import-path ``ranking_function`` cannot be imported.
            FileExistsError: If a file exists at ``checkpoint``.
            sqlite3.OperationalError: If the ``checkpoint`` file cannot be created (e.g. its
                directory does not exist or is not writable).

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
            max_evaluations: The budget of counted calls; negative for no limit.
            success_threshold: The ``success_threshold`` passed, or the ranking's default if it
                was ``None`` (still ``None`` if the ranking has no default).
            seed: The random seed.
            metadata: The metadata dict as passed.
            models: Read-only [view][SRToolkit.evaluation.model_store.ModelsView] of the
                [ModelResult][SRToolkit.utils.types.ModelResult] of every evaluated model, in the
                order they were first evaluated.
            trace: Read-only [view][SRToolkit.evaluation.model_store.TraceView] with one entry per
                counted call to
                [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]
                (cache hits included, calls after stopping excluded): the index into ``models`` of
                the requested expression.
            stop_reason: Why evaluation stopped (see the
                [module docs][SRToolkit.evaluation.sr_evaluator]), or ``None`` while it runs.
            default_constant: The run's default spec, complete:
                [background_knowledge.default_constant][SRToolkit.dataset.background_knowledge.BackgroundKnowledge],
                built from the ``default_constant`` and ``constant_bounds`` settings. The ranking
                gets it as its ``default_constant`` setting and the results record it; ``kwargs``
                keeps the setting as passed.
        """
        self.kwargs = kwargs
        self.success_threshold = success_threshold
        self.metadata = metadata
        self.ground_truth = ground_truth
        self._experiment_id: str = ""
        self.stop_reason: str | None = None
        self._current_best_error = float("inf")

        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        self.symbol_library = symbol_library
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer)):
            raise TypeError(f"[SR_evaluator] max_evaluations must be an int, got {type(max_evaluations).__name__}.")
        if max_evaluations == 0:
            raise ValueError("[SR_evaluator] max_evaluations must be positive, or negative for no limit; got 0.")
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
            named_constants={} if named_constants is None else named_constants,
            default_constant=kwargs.get("default_constant"),
        )
        # BackgroundKnowledge stores its constraints as a tuple; the field is typed Iterable.
        self._advisory_constraints = cast(tuple[ExpressionConstraint, ...], self._background_knowledge.constraints)
        self.default_constant = cast(Constant, self._background_knowledge.default_constant)

        if seed is not None:
            np.random.seed(seed)

        # The ranking function owns all of its data-derived state (parameter estimator for RMSE,
        # behaviour matrix + evaluation points bed_X for BED); the evaluator just calls score().
        # The limits are enforced here, as required constraints; the ranking does not get them.
        ranking_settings: dict[str, Any] = {k: v for k, v in kwargs.items() if k not in _DEFAULT_LIMITS}
        ranking_settings["default_constant"] = self.default_constant  # completed by the background knowledge
        self.ranking: RankingFunction = build_ranking(ranking_function)
        # A ranking_seed (fixed per dataset) replaces the evaluator's seed here, so every ranking,
        # custom ones included, follows it without reading the setting itself.
        ranking_seed = kwargs.get("ranking_seed")
        self.ranking.setup(
            X, y, ground_truth, symbol_library, seed if ranking_seed is None else ranking_seed, ranking_settings
        )

        if self.success_threshold is None:
            self.success_threshold = self.ranking.default_success_threshold()

        self._callbacks = CallbackDispatcher()
        self.X = X
        self.y = y

        # Last, so that a failing constructor leaves no checkpoint file behind.
        self._store = ModelStore(checkpoint)
        self.models = self._store.models
        self.trace = self._store.trace

    @property
    def should_stop(self) -> bool:
        """Whether evaluation has stopped: ``stop_reason is not None``. Later calls to
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] warn and
        return ``nan`` without being counted.
        """
        return self.stop_reason is not None

    def stop(self, reason: str) -> None:
        """Stop evaluation and record why, unless it has already stopped (the first reason is kept).

        An approach calls this when it ends its search with budget left, e.g. because its search
        space is exhausted, so the results explain the early end.

        Examples:
            >>> se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
            >>> se.stop("grammar exhausted")
            >>> se.stop("other")
            >>> se.should_stop, se.stop_reason
            (True, 'grammar exhausted')

        Args:
            reason: Short free text, saved as
                [EvalResult.stop_reason][SRToolkit.utils.types.EvalResult].

        Raises:
            ValueError: If ``reason`` is not a non-empty ``str``.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"[SR_evaluator.stop] reason must be a non-empty str, got {reason!r}.")
        if self.stop_reason is None:
            self.stop_reason = reason
            self._store.commit()

    @property
    def total_evaluations(self) -> int:
        """Number of counted calls to
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr], including
        cache hits but not calls made after evaluation was stopped. Equal to ``len(trace)``.
        """
        return len(self.trace)

    @property
    def invalid(self) -> list[str]:
        """Strings of the evaluated expressions with status ``"invalid"`` (not well-formed), in
        the order they were first evaluated.
        """
        return self._store.invalid_exprs()

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
    def required_constraints(self) -> tuple[ExpressionConstraint, ...]:
        """The set-up copies of the required constraints, in the order they are checked.

        The ``MaxConstants`` and ``MaxTokens`` added by the settings (if any), then copies of the
        ``required_constraints`` argument. Passing this tuple back as ``required_constraints``
        with the same settings gives the same constraints.
        """
        return self._required_constraints

    @property
    def advisory_constraints(self) -> tuple[ExpressionConstraint, ...]:
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

    def priors(self) -> dict[str, list[dict[str, Any]]]:
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

    def get_model(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
        canonicalise_constants: bool = False,
    ) -> ModelResult | None:
        """Return the result of an already evaluated expression, or ``None``.

        Does not count toward ``max_evaluations`` and fires no callbacks. The expression is matched
        token for token, without simplifying it, together with its specs.

        Args:
            expr: The expression, as a token list or a [Node][SRToolkit.utils.expression_tree.Node].
            free_constants: The specs it was evaluated with, in any order; ``None`` or empty for none.
            canonicalise_constants: As in
                [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]; pass
                the value ``expr`` was evaluated with. Default ``False``.

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
        tokens = _as_tokens(expr, self.symbol_library)
        if canonicalise_constants and not free_constants:
            tokens = canonicalise(tokens, self.symbol_library)
        found = self._store.lookup(tokens, _spec_key(free_constants))
        return None if found is None else self.models[found[0]]

    def check_constraints(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
    ) -> tuple[list[CheckResult], list[CheckResult]]:
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
            free_constants: Specs for its labelled constants, validated against ``expr`` by
                [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]. The
                pre-checks themselves see only the tokens.

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
            ValueError: If ``free_constants`` does not fit ``expr`` (see
                [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]).

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
        resolve_constants(expr_list, free_constants, self.default_constant, self.symbol_library)
        return (
            _failed_checks(self._required_constraints, "pre", expr_list),
            _failed_checks(self._advisory_constraints, "pre", expr_list),
        )

    def register_callbacks(self, callbacks: SRCallbacks | CallbackDispatcher | None = None) -> None:
        """Add callbacks for monitoring and early stopping to those already registered.

        The evaluator enforces ``success_threshold`` and ``max_evaluations`` itself; callbacks add
        further stop conditions (e.g. a time limit).

        Examples:
            >>> from SRToolkit.evaluation.callbacks import TimeLimitCallback
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> se.register_callbacks(TimeLimitCallback(time_limit=60))

        Args:
            callbacks: A single [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks], or a
                [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] whose
                callbacks are added (the dispatcher itself is not changed). ``None`` adds nothing.

        Raises:
            TypeError: If ``callbacks`` is of another type (e.g. a list).
        """
        if isinstance(callbacks, SRCallbacks):
            self._callbacks.add(callbacks)
        elif isinstance(callbacks, CallbackDispatcher):
            for cb in callbacks.get_callbacks():
                self._callbacks.add(cb)
        elif callbacks is not None:
            raise TypeError(
                "[SR_evaluator.register_callbacks] Expected an SRCallbacks, a CallbackDispatcher or None, "
                f"got {type(callbacks).__name__}."
            )

    def evaluate_expr(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
        simplify_expr: bool = False,
        canonicalise_constants: bool = False,
        verbose: int = 0,
    ) -> float:
        """Evaluate an expression, or return its cached error.

        A new expression is parsed, checked against the constraints, scored by the ranking
        function and stored in ``models`` (statuses and the order of checks are in the
        [module docs][SRToolkit.evaluation.sr_evaluator]). An expression evaluated before with the
        same specs is not scored again; its stored error is returned. Either way the call is
        appended to ``trace``,
        counts toward ``max_evaluations`` and is reported to the callbacks: ``on_expr_evaluated``
        for every call, plus ``on_best_expression`` when the error improves on the best so far.
        Evaluation then stops if the error is below ``success_threshold``, the budget is used up,
        or a callback asks to stop (see [Stopping][SRToolkit.evaluation.sr_evaluator]).

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
            >>> # A labelled constant with a spec of its own: started from U(0, 10), fitted within (0, 10)
            >>> from SRToolkit.dataset import UniformSampling
            >>> from SRToolkit.utils.free_constants import Constant
            >>> spec = Constant(label="C_0", prior=UniformSampling(0, 10), bounds=(0.0, 10.0))
            >>> se = SR_evaluator(X, y, seed=42)
            >>> print(se.evaluate_expr(["C_0", "*", "X_1", "-", "X_0"], [spec]) < 1e-6)
            True
            >>> se.models[-1].free_constants == (spec,)
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
            free_constants: Specs for labelled constants of ``expr`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]), in any order. Constants
                without one, and a spec's unset ``prior`` or ``bounds``, use ``default_constant``.
                Passed to the ranking and stored as ``ModelResult.free_constants``. ``None`` or
                empty: no specs.
            simplify_expr: If ``True``, simplify the expression first; the simplified
                expression is then checked, scored, cached and stored as ``ModelResult.expr``, so
                inputs that simplify alike share one cache entry. If simplification fails, a
                warning is emitted and the expression is evaluated as given. Default ``False``.
            canonicalise_constants: If ``True``, [canonicalise][SRToolkit.utils.free_constants.canonicalise]
                the expression (after simplifying); the canonical form replaces it for the rest of
                the call, so ``C_1*X_0 + C_1`` after ``C_0*X_0 + C_0`` is a cache hit. For approaches
                that write labelled constants; pass the same value to ``get_model`` and ``get_metric``.
                Ignored with ``free_constants``: labels with specs are kept as written. Default
                ``False``.
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
            ValueError: If ``free_constants`` is given with ``simplify_expr=True`` (simplifying
                renumbers constants and loses their labels), or does not fit a well-formed ``expr``
                (see [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]).
                Nothing is then recorded or counted.
            TypeError: If a spec in ``free_constants`` cannot be saved (see
                [Constant.to_dict][SRToolkit.utils.free_constants.Constant.to_dict]). Nothing is
                then recorded or counted.
        """
        specs = tuple(sorted(free_constants, key=lambda c: c.label or "")) if free_constants else None
        # Temporary: simplify() renumbers constants and drops their labels, so specs would no longer
        # fit. Remove once simplify() takes free_constants and returns the specs with the simplified
        # expression.
        if specs is not None and simplify_expr:
            raise ValueError("free_constants cannot be combined with simplify_expr=True.")
        if self.should_stop:
            warnings.warn(f"Evaluation has already stopped ({self.stop_reason}); the call is not evaluated or counted.")
            return np.nan
        else:
            if simplify_expr:
                try:
                    expr = simplify(expr, self.symbol_library)
                except Exception as e:
                    expr_list = _as_tokens(expr, self.symbol_library)
                    warnings.warn(f"Unable to simplify: {''.join(expr_list)}, problems with subexpression {e}")

            expr_list = _as_tokens(expr, self.symbol_library)
            if canonicalise_constants and specs is None:
                canonical = canonicalise(expr_list, self.symbol_library)
                if canonical != expr_list:
                    # The ranking gets the canonical tokens, so the fit's seed is that of the canonical form.
                    expr = expr_list = canonical

            expr_str = "".join(expr_list)
            spec_key = _spec_key(specs)
            found = self._store.lookup(expr_list, spec_key)
            if found is not None:
                index, error, status = found
                if verbose > 0:
                    logger.debug("Already evaluated %s", expr_str)
                self._store.add_call(index)
                if 0 <= self.max_evaluations <= self.total_evaluations:
                    self.stop("max_evaluations")
                event = ExprEvaluated(
                    expression=expr_str,
                    error=error,
                    evaluation_number=self.total_evaluations,
                    experiment_id=self._experiment_id,
                    is_new_best=False,
                    status=status,
                )
                reason = self._callbacks.on_expr_evaluated(event)
                if reason is not None:
                    self.stop(reason)
                return error

            else:
                model = self._evaluate_new(expr, expr_list, expr_str, verbose, specs)
                model.free_constants = specs
                try:
                    self._store.add(expr_list, spec_key, model)
                except TypeError as e:
                    # The ranking's output cannot be saved: a failure of the evaluation.
                    message = f"the model cannot be saved: {e}"
                    warnings.warn(f"[SR_evaluator] Error scoring expression {expr_str}: {message}")
                    model.ranking_data = {}
                    model.parameters = None
                    model.advisory_violations = []
                    model.error = np.inf
                    model.status = "error"
                    model.status_detail = message
                    self._store.add(expr_list, spec_key, model)
                if verbose > 0:
                    logger.debug(
                        "Evaluated expression %s: status %s, error %s%s",
                        expr_str,
                        model.status,
                        model.error,
                        f" ({model.status_detail})" if model.status_detail else "",
                    )
                error = model.error

                # Threshold before budget, so a call that does both records "success_threshold".
                if self.success_threshold is not None and error < self.success_threshold:
                    self.stop("success_threshold")
                elif 0 <= self.max_evaluations <= self.total_evaluations:
                    self.stop("max_evaluations")

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
                reason = self._callbacks.on_expr_evaluated(event)
                if reason is not None:
                    self.stop(reason)
                if is_new_best:
                    best_event = BestExpressionFound(
                        experiment_id=self._experiment_id,
                        expression=expr_str,
                        error=error,
                        evaluation_number=self.total_evaluations,
                    )
                    reason = self._callbacks.on_best_expression(best_event)
                    if reason is not None:
                        self.stop(reason)

                return error

    def _evaluate_new(
        self,
        expr: list[str] | Node,
        expr_list: list[str],
        expr_str: str,
        verbose: int,
        specs: tuple[Constant, ...] | None,
    ) -> ModelResult:
        # Parse first, so that each failure gets its own status.
        try:
            tokens_to_tree(expr_list, self.symbol_library)
        except ExpressionParseError as e:
            return ModelResult(expr=expr_list, error=np.inf, status="invalid", status_detail=str(e))
        # Specs that do not fit are the caller's mistake, not the expression's: raise.
        if specs is not None:
            resolve_constants(expr_list, specs, self.default_constant, self.symbol_library)

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

        model = self._score(expr, expr_list, expr_str, verbose, specs)
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

    def _score(
        self,
        expr: list[str] | Node,
        expr_list: list[str],
        expr_str: str,
        verbose: int,
        specs: tuple[Constant, ...] | None,
    ) -> ModelResult:
        try:
            with (
                np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore")
                if verbose < 2
                else nullcontext()
            ):
                result = self.ranking.score(expr, free_constants=specs)
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
        expr_list: list[str],
        advisory: list[CheckResult],
        parameters: np.ndarray | None = None,
        ranking_data: dict[str, Any] | None = None,
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
        expr: list[str] | Node,
        metric: str | Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float] = "r2",
        free_constants: Sequence[Constant] | None = None,
        canonicalise_constants: bool = False,
    ) -> float:
        """Compute an alternative, read-only metric for an already-evaluated expression.

        Unlike [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr],
        this does **not** count toward the evaluation budget, does not affect ranking,
        success, caching, or callbacks, and never changes the canonical ranking function.
        It exists so an approach can structure its search around a measure other than the
        ranking error (e.g. R² or NRMSE) without re-running the (budgeted) evaluation and
        without being given direct access to ``X``/``y`` as attributes.

        The expression **must already have been evaluated** with the same specs via
        ``evaluate_expr`` so its constants are fitted and cached — otherwise a ``ValueError`` is
        raised. It is matched without simplifying it, as in
        [get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model]. This is the
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
            free_constants: The specs ``expr`` was evaluated with, as in ``evaluate_expr``;
                ``None`` or empty for none.
            canonicalise_constants: As in ``evaluate_expr``; pass the value ``expr`` was evaluated
                with. Default ``False``.

        Returns:
            The metric value as a float. Returns ``NaN`` in BED mode, for models that could
            not be scored, or if the metric raises during computation.

        Raises:
            ValueError: If ``expr`` has not been evaluated yet with these specs, or if ``metric``
                is a string that is not a known built-in metric.
            TypeError: If ``metric`` is neither a string nor a callable.
        """
        expr_list = _as_tokens(expr, self.symbol_library)
        if canonicalise_constants and not free_constants:
            expr_list = canonicalise(expr_list, self.symbol_library)
        expr_str = "".join(expr_list)

        found = self._store.lookup(expr_list, _spec_key(free_constants))
        if found is None:
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

        model = self.models[found[0]]

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
        self, approach_name: str = "", top_k: int = 20, results: "SR_results | None" = None
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
            An instance of the SR_results object with the results of the evaluation. Its
            ``stop_reason`` is the evaluator's, or ``"returned"`` if evaluation has not stopped.
        """
        self._store.commit()
        if results is None:
            results = SR_results()

        results.add_results(
            self.models,
            self.trace,
            top_k,
            self.success_threshold,
            approach_name,
            self.metadata,
            self.default_constant,
            self.stop_reason or "returned",
        )

        return results
