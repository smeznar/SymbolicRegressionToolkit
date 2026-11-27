"""Expression evaluation for symbolic regression.

Contains [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator], which ranks expressions
with a [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] (RMSE by default) and
collects them into [SR_results][SRToolkit.evaluation.results.SR_results].

## Settings

A run is configured by one [Settings][SRToolkit.utils.settings.Settings] object, passed as
``settings``. The evaluator reads its ``evaluator`` section
([EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings]: stored form, limits, skip limit,
backend, ``ranking_seed``); the ranking reads its own section and shared ones (see
[ranking][SRToolkit.evaluation.ranking]). Facts about the task are not settings: ``constant_bounds``,
``default_constant``, ``samplers`` and ``ground_truth_constants`` are arguments of the evaluator,
which passes the first three to the ranking in its
[background_knowledge][SRToolkit.evaluation.sr_evaluator.SR_evaluator.background_knowledge].

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

The statuses are saved with the results, so validity can be measured after the run from
[status_counts][SRToolkit.utils.types.EvalResult.status_counts]. Count every well-formed
expression as valid, including rejected and undefined ones, and leave ``"error"`` out entirely,
because those failures belong to the evaluation, not to the approach:

```python
counts = result.status_counts()  # per unique expression; per="call" counts every call
validity_rate = 1 - counts["invalid"] / (sum(counts.values()) - counts["error"])
```

## Stopping

Evaluation stops at the first of these, which is recorded as ``stop_reason`` and saved in the
results as [EvalResult.stop_reason][SRToolkit.utils.types.EvalResult]:

- ``"success_threshold"``: a new expression has an error below ``success_threshold``;
- ``"max_evaluations"``: the call used the last evaluation of the budget (a call that does both
  records ``"success_threshold"``);
- ``"max_consecutive_skips"``: the approach skipped ``max_consecutive_skips`` expressions in a row
  (the setting ``evaluator.max_consecutive_skips``, default ``10000``; see
  [count_skip][SRToolkit.evaluation.sr_evaluator.SR_evaluator.count_skip]);
- a callback asked to stop (see [SR_callback][SRToolkit.evaluation.callbacks.SR_callback]), e.g.
  ``"time_limit"``;
- the approach called [stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop] with its own
  reason, e.g. because its search space is exhausted.

An approach loops while ``should_stop`` is ``False``. If its search returns without any of these,
the results record ``"returned"``. Once evaluation has stopped, further ``evaluate_expr``
calls are not evaluated: they warn, return ``None`` and are not counted.

## Free constants

An approach can pass specs for the labelled constants of an expression (priors, bounds, fixed
values; see [free_constants][SRToolkit.utils.free_constants]) as the ``free_constants`` argument of
[evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]. A **model** is an
expression together with its specs: the same tokens with different specs are two models, each
scored, cached, counted in ``num_evaluated`` and ``status_counts`` and stored in ``models`` on its
own. Specs are matched by value (the fields that affect the score) and in any order. They are stored
in [ModelResult.free_constants][SRToolkit.utils.types.ModelResult] sorted by label,
or ``None`` when none (or an empty list) were given. Parameters without a spec, and a spec's unset
``prior`` or ``bounds``, use the evaluator's ``default_constant``, which is recorded in the results
as [EvalResult.default_constant][SRToolkit.utils.types.EvalResult]. Expressions that differ only in
how their constants are written (``C_0*X_0 + C_0`` and ``C_1*X_0 + C_1``) are separate models
unless the setting ``evaluator.canonicalise_constants`` makes them one (see Stored form below).

Specs are checked right after parsing, against the expression as given: a malformed expression is
``"invalid"`` (and stored with its specs) whatever they are. Specs that do not fit a well-formed
expression raise; nothing is then recorded or counted. The ranking's
[score][SRToolkit.evaluation.ranking.RankingFunction.score] receives the specs as
``free_constants``.

## Stored form

An expression is checked, scored and stored in its **stored form**, which three settings of the
``evaluator`` section (see [EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings])
produce, in this order:

1. ``simplify_expr``: [simplified][SRToolkit.utils.expression_simplifier.simplify] with the
   ``simplify_*`` settings. If simplification fails, the expression is kept as given;
   [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] then warns,
   unless the expression is malformed (it is scored ``"invalid"``, which says why). A malformed
   expression is still canonicalised token by token in step 3.
2. ``fold_constants``: [folded][SRToolkit.utils.expression_simplifier.fold] (during simplification
   with ``simplify_expr``, otherwise on the expression as given).
3. ``canonicalise_constants``: [canonicalised][SRToolkit.utils.free_constants.canonicalise], only
   if no specs are left.

Simplifying and folding drop the specs of the labels they remove; if none remain, the model is one
without specs (``ModelResult.free_constants`` is ``None``). Inputs with the same stored form and
the same remaining specs are one model: scored once, cached once. ``ModelResult.expr`` holds the
stored form, the pre-checks see it, and
[get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model],
[get_metric][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_metric] and
[check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints] bring their
input to it first. [stored_tokens][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stored_tokens]
returns it.

## Constraints

The evaluator enforces the ``required_constraints`` and records the advisory ones, which are the
``constraints`` of its background knowledge (see below). Both are
[ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]s; that module
explains what the checks look at and the two roles. The evaluator calls ``setup`` with its symbol
library on its own copies, so the instances passed in are not changed.

- ``required_constraints``: pre-checks run in order after parsing, post-checks in order after an
  ``"ok"`` score. The first failure gives ``"rejected"`` with the ``status_detail``
  ``"<ClassName>: <message>"``, e.g. ``"MaxConstants: 9 constants, limit 8"``, and the later checks
  of that stage are not run. The settings ``evaluator.max_constants`` (default 8) and
  ``evaluator.max_expr_length`` add a ``MaxConstants`` and a ``MaxTokens`` in front of the list,
  unless a negative setting disables them or the list already has one of that class (see
  [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator]), so ``[]`` still allows at most
  8 constants.
- ``advisory_constraints``: pre-checks run on every parsed expression, post-checks on every ``"ok"``
  model. Each failed check is recorded in
  [ModelResult.advisory_violations][SRToolkit.utils.types.ModelResult] as a
  [CheckResult][SRToolkit.utils.types.CheckResult] with ``constraint`` and ``stage`` filled in; the
  error is not affected.

For one expression the order is: parse, advisory pre-checks, required pre-checks, score, required
post-checks, advisory post-checks. So rejected models and models with status ``"error"`` still
carry their advisory pre-check results. Pre-checks get ``ModelResult.expr`` (the stored form),
post-checks the ``ModelResult``. Checks a constraint does not implement are skipped.

A check that raises warns and counts as failed, which gives the status ``"error"`` for a required
constraint (see [Checks that raise][SRToolkit.utils.expression_constraints]).

To avoid spending evaluations on expressions that would fail a pre-check, or to filter candidates
by the advisory constraints, an approach can call
[check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints] first: it runs
the pre-checks only, on the stored form, and is not billed.

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
``stop_reason``, and [extended][SRToolkit.utils.types.EvalResult.extended] (see Extending runs
below), before comparing approaches.

## Storage

``models`` and ``trace`` are read-only views of a SQLite database, in memory by default or in the
``checkpoint`` file, which keeps memory constant during long runs (see
[model_store][SRToolkit.evaluation.model_store]). ``se.models[i]`` returns a new
[ModelResult][SRToolkit.utils.types.ModelResult] each time, so changing it saves nothing. An
evaluator holds an open database connection, so it cannot be pickled or deep-copied; release it
with [close][SRToolkit.evaluation.sr_evaluator.SR_evaluator.close] when done (a dropped evaluator
loses only what was not committed). It is not thread-safe: call it from one thread at a time.

## Checkpoints

With ``checkpoint=path`` the evaluator writes the models and trace to that file as it goes, so a
run that crashes, is killed or is stopped keeps what it evaluated: convert it to results with
[SR_results.from_checkpoint][SRToolkit.evaluation.results.SR_results.from_checkpoint], or continue
it with ``resume=True``. A resume must match the saved run in:

- the data, ranking, symbols, ground truth and seed;
- the constraints (``evaluator.max_constants`` and ``evaluator.max_expr_length`` are compared as
  the ``MaxConstants`` / ``MaxTokens`` they add to the required constraints);
- ``default_constant``, ``constant_bounds``, ``samplers`` and ``ground_truth_constants`` (in any
  order; ``[]`` is the same as ``None``);
- ``max_evaluations`` and ``success_threshold``, which only
  [extending the run][SRToolkit.dataset.sr_dataset.SR_dataset.extend_run] changes;
- every setting that differs from its default, in every section, whether the ranking reads it or
  not. Settings are compared by their saved form, so a setting written at its default is the same
  as one left out.

Otherwise it raises ``ValueError`` naming what differs (a setting as ``<section>.<field>``).
``snapshot_interval`` and the approach are not checked, and other callbacks only warn.

Resuming continues from the run's last [snapshot][SRToolkit.evaluation.model_store.Snapshot], the
approach's state saved between two steps. Approaches that support snapshots (see
[SR_approach][SRToolkit.approaches.sr_approach.SR_approach]) save one every ``snapshot_interval``
seconds (5 minutes by default) and when they stop. A crash loses only the work since the last
snapshot. A run without any snapshot starts over. A finished run (stopped for any reason except an
interrupt) is not run again. Snapshots may be pickled, so resume only checkpoints you trust.

## Extending runs

To give a run a larger budget, a lower threshold or a longer time limit, extend it with
[SR_dataset.extend_run][SRToolkit.dataset.sr_dataset.SR_dataset.extend_run]:

```python
results = dataset.evaluate_approach(approach, checkpoint_dir="runs", keep_checkpoint=True)
path = results[0].checkpoint["path"]
results = dataset.extend_run(approach, path, max_evaluations=50_000)
```

The run goes back to its last snapshot and continues from there with the new limits. So the result
is the same as a run that had these limits from the start. The work since that snapshot is done
again. If the run already meets the new limits, it stops right away. Each extension is recorded in
[EvalResult.limits][SRToolkit.utils.types.EvalResult].

To change callbacks, extend the run too: a plain resume with other callbacks only warns.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import signal
import threading
import time
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from SRToolkit.evaluation.callbacks import (
    BestExpressionFound,
    CallbackDispatcher,
    ExprEvaluated,
    ExprSkipped,
    SR_callback,
)
from SRToolkit.evaluation.model_store import ModelStore, Snapshot, _capture_rngs
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
from SRToolkit.utils.expression_simplifier import fold, simplify
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, _as_tokens, _as_tree, tokens_to_tree
from SRToolkit.utils.free_constants import Constant, _spec_key, canonicalise, resolve_constants
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.measures import REGRESSION_METRICS
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe
from SRToolkit.utils.settings import EvaluatorSettings, Settings
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import CheckResult, ModelResult, Status

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports this module.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge, VariableInfo
    from SRToolkit.dataset.sampling import Sampler

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


def _fingerprint_hash(value: Any, name: str) -> str:
    # Hash of a fingerprint part: arrays by their bytes, objects with to_dict through it.
    def encode(obj: Any) -> Any:
        return obj.to_dict() if hasattr(obj, "to_dict") else _to_json_safe(obj, name)

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        data = f"{array.dtype}|{array.shape}|".encode() + array.tobytes()
    else:
        data = json.dumps(value, sort_keys=True, default=encode).encode()
    return hashlib.blake2b(data, digest_size=16).hexdigest()


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


def _add_limit_constraints(
    constraints: tuple[ExpressionConstraint, ...], limits: Mapping[str, Any]
) -> tuple[ExpressionConstraint, ...]:
    # MaxConstants then MaxTokens from the max_constants / max_expr_length settings (negative = none),
    # followed by the given constraints. A setting adds nothing when the list already has a
    # constraint of its class; a given non-negative setting with a different limit is a ValueError.
    built: list[ExpressionConstraint] = []
    for key, cls in (("max_constants", MaxConstants), ("max_expr_length", MaxTokens)):
        value = _validate_limit(limits.get(key, getattr(EvaluatorSettings(), key)), key, allow_negative=True)
        # cls is MaxConstants or MaxTokens, so every listed constraint has a limit.
        listed = cast(list[MaxConstants | MaxTokens], [c for c in constraints if isinstance(c, cls)])
        if not listed:
            if value >= 0:
                built.append(cls(value))
        else:
            clash = next((c for c in listed if c.limit != value), None)
            if key in limits and value >= 0 and clash is not None:
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
        constant_bounds: tuple[float, float] | None = None,
        default_constant: Constant | None = None,
        samplers: "Mapping[str, Sampler] | Sequence[Sampler] | None" = None,
        ground_truth_constants: Sequence[Constant] | None = None,
        settings: Settings | None = None,
        checkpoint: str | os.PathLike | None = None,
        resume: bool = False,
        snapshot_interval: float | None = 300.0,
    ):
        """Evaluate expressions on data with a ranking function and collect the results.

        Each distinct model (an expression with its free-constant specs) is scored once and
        cached; repeating it returns the cached model but still counts toward
        ``max_evaluations``. Statuses, free constants, constraints and the evaluation trace
        are described in the [module docs][SRToolkit.evaluation.sr_evaluator]; results are
        collected with [get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results].

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values of shape ``(n_samples,)``. RMSE needs ``y`` or a ``ground_truth``
                to generate it from, each of whose free constants has a fixed value in
                ``ground_truth_constants``; BED can use it in place of a ground truth.
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
                ``evaluator.ranking_seed`` setting is set.
            metadata: Optional information about this evaluation (e.g. dataset name). A
                ``"dataset_name"`` key is copied to
                [EvalResult][SRToolkit.utils.types.EvalResult] ``dataset_name``.
            required_constraints: Constraints every evaluated expression must satisfy (see the
                [module docs][SRToolkit.evaluation.sr_evaluator]); ``None`` means ``[]``. The
                settings ``evaluator.max_constants`` and ``evaluator.max_expr_length`` add a
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
            constant_bounds: ``(lower, upper)`` range used only to build or complete
                ``default_constant`` (see
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]).
                ``None``: ``(-5, 5)``.
            default_constant: The default spec of free constants, a
                [Constant][SRToolkit.utils.free_constants.Constant] template; a partial one is
                completed from ``constant_bounds``. ``None``:
                [uniform_default(constant_bounds)][SRToolkit.utils.free_constants.uniform_default].
            samplers: The input distribution, one [Sampler][SRToolkit.dataset.sampling.Sampler] per
                variable (a sequence, or a mapping by token). BED draws its points from them when
                ``bed.num_points_sampled > 0``. ``None`` (default): no input distribution.
            ground_truth_constants: Specs for the labelled constants of ``ground_truth`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]), in any order. RMSE uses their
                fixed values to generate a missing ``y``; BED keeps fixed values, samples the other
                constants from their priors, and uses ``default_constant`` for constants without a
                spec. ``None`` (default) or empty: no specs.
            settings: The run's [Settings][SRToolkit.utils.settings.Settings]: the ``evaluator``
                section (see [EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings]) and
                the ranking's sections. ``None``: the defaults.
            checkpoint: A file to store the models and trace in, committed as described in
                [model_store][SRToolkit.evaluation.model_store]; created, or continued with
                ``resume=True`` (see [Checkpoints][SRToolkit.evaluation.sr_evaluator]). ``None``
                (default): they are kept in memory.
            resume: Continue the run saved in ``checkpoint`` (see
                [Checkpoints][SRToolkit.evaluation.sr_evaluator]). If the run already meets a limit,
                the evaluator starts stopped. Default ``False``.
            snapshot_interval: Seconds between snapshots during a run with a ``checkpoint`` (see
                [Checkpoints][SRToolkit.evaluation.sr_evaluator]). ``None``: a snapshot only when
                the step loop ends. Default ``300.0``. Not checked on resume, so a resume may change
                it.

        Raises:
            TypeError: If ``max_evaluations`` is not an ``int`` or numpy integer (``bool`` is
                rejected), or ``settings`` is not a ``Settings``. If an item of
                ``required_constraints`` or ``advisory_constraints`` is not an
                [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint]
                (e.g. the grammar constraint [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth],
                or a single constraint passed instead of a sequence). The message names the item.
                Also raised as for
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge],
                and by [build_ranking][SRToolkit.evaluation.ranking.build_ranking] for a
                ``ranking_function`` of the wrong type. With a ``checkpoint`` (only then), also if
                ``metadata``, ``samplers``, the ranking or a constraint cannot be saved (see [serialization][SRToolkit.utils.serialization]). No
                file is then left behind.
            ValueError: If ``max_evaluations`` is ``0``, ``snapshot_interval`` is not positive, or
                ``checkpoint`` is an empty path. If ``resume=True`` and ``checkpoint`` is ``None``,
                is not a checkpoint (or one of another format version), or was saved with another
                run (see [Checkpoints][SRToolkit.evaluation.sr_evaluator]); the message names what
                differs. If ``evaluator.max_constants`` or ``evaluator.max_expr_length`` differs
                from its default (a value written at its default counts as not given), is
                non-negative and differs from the limit of a constraint of its class in
                ``required_constraints``. Also raised as for ``BackgroundKnowledge`` (e.g.
                ``constant_bounds`` not finite with ``lower < upper``, a ``default_constant`` with a
                ``label``, a ``value`` or ``fixed=True``, or ``samplers`` without one sampler per
                variable), and by the ranking's ``setup`` for missing or malformed data (e.g. RMSE
                with neither ``y`` nor ``ground_truth``).
            ImportError: If an import-path ``ranking_function`` cannot be imported.
            FileExistsError: If a file exists at ``checkpoint`` and ``resume`` is ``False``.
            FileNotFoundError: If ``resume=True`` and no file exists at ``checkpoint``.
            sqlite3.OperationalError: If the ``checkpoint`` file cannot be created or opened (e.g.
                its directory does not exist or is not writable).

        Attributes:
            X: The input data.
            y: The target values as given (``None`` if RMSE generated them).
            symbol_library: The symbol library used.
            background_knowledge: The background knowledge (read-only).
            ranking: The active [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction]. It
                owns the ranking's state: the
                [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator]
                for RMSE, or the behaviour matrix and points ``bed_X`` for BED.
            settings: The settings used.
            ground_truth_constants: The ``ground_truth_constants`` as passed.
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
            checkpoint: The ``checkpoint`` passed, or ``None``.
            snapshot_interval: The ``snapshot_interval`` passed.
            resumed_from: The [Snapshot][SRToolkit.evaluation.model_store.Snapshot] a resumed
                evaluator continues from, or ``None`` (a new run, or a checkpoint without one).
            limits: The run's limit segments (see
                [EvalResult.limits][SRToolkit.utils.types.EvalResult]); a resume reads them from
                the checkpoint.
            skips: How many expressions the approach skipped without evaluating them, by reason; a
                resume restores them (see
                [count_skip][SRToolkit.evaluation.sr_evaluator.SR_evaluator.count_skip]).
            default_constant: The run's default spec, complete:
                [background_knowledge.default_constant][SRToolkit.dataset.background_knowledge.BackgroundKnowledge],
                built from ``default_constant`` and ``constant_bounds``. The ranking and the results
                use it.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"]).error
            >>> print(rmse < 1e-6)
            True

        Note:
            Whether two expressions are equivalent is undecidable, so ``success_threshold`` is only
            a proxy for success. Inspect the best expressions manually.
        """
        if settings is None:
            settings = Settings()
        if not isinstance(settings, Settings):
            raise TypeError(f"[SR_evaluator] settings must be a Settings, got {type(settings).__name__}.")
        self.settings = settings
        self.ground_truth_constants = ground_truth_constants
        self.success_threshold = success_threshold
        self.metadata = metadata
        self.ground_truth = ground_truth
        self._experiment_id: str = ""
        self.stop_reason: str | None = None
        self._interrupted = False  # set by a signal caught by stop_on_signals
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
        required = _add_limit_constraints(required, settings.evaluator.changed())
        self._required_constraints = tuple(_set_up_copy(c, symbol_library) for c in required)

        # Background knowledge from the same values the evaluator fits with; it holds the set-up
        # copies of the advisory constraints. Imported here: SRToolkit.dataset imports this module.
        from SRToolkit.dataset.background_knowledge import BackgroundKnowledge

        self._background_knowledge = BackgroundKnowledge(
            symbol_library,
            samplers=samplers,
            constraints=advisory,
            reference_grammar=reference_grammar,
            description=description,
            variables={} if variables is None else variables,
            target=target,
            named_constants={} if named_constants is None else named_constants,
            default_constant=default_constant,
            **({} if constant_bounds is None else {"constant_bounds": constant_bounds}),
        )
        # BackgroundKnowledge stores its constraints as a tuple; the field is typed Iterable.
        self._advisory_constraints = cast(tuple[ExpressionConstraint, ...], self._background_knowledge.constraints)
        self.default_constant = cast(Constant, self._background_knowledge.default_constant)

        if seed is not None:
            np.random.seed(seed)

        # The ranking function owns all of its data-derived state (parameter estimator for RMSE,
        # behaviour matrix + evaluation points bed_X for BED); the evaluator just calls score().
        # A ranking_seed (fixed per dataset) replaces the evaluator's seed here, so every ranking,
        # custom ones included, follows it without reading the setting itself.
        self.ranking: RankingFunction = build_ranking(ranking_function)
        ranking_seed = settings.evaluator.ranking_seed
        self.ranking.setup(
            X,
            y,
            ground_truth,
            ground_truth_constants,
            self._background_knowledge,
            seed if ranking_seed is None else ranking_seed,
            settings,
        )

        if self.success_threshold is None:
            self.success_threshold = self.ranking.default_success_threshold()

        self._callbacks = CallbackDispatcher()
        self.X = X
        self.y = y

        # The callbacks' saved state, restored before the first evaluation event after a resume.
        self._callback_states: list[dict] | None = None
        self.resumed_from: Snapshot | None = None
        self._in_step = False  # inside se.step(): a signal takes effect when the step ends
        self.checkpoint = checkpoint
        if snapshot_interval is not None and not snapshot_interval > 0:
            raise ValueError(f"[SR_evaluator] snapshot_interval must be positive or None, got {snapshot_interval!r}.")
        self.snapshot_interval = snapshot_interval
        if resume and checkpoint is None:
            raise ValueError("[SR_evaluator] resume=True needs a checkpoint to resume.")
        # The run's limit segments (see EvalResult.limits); a resume reads them from the checkpoint.
        self.limits: list[dict[str, Any]] = [
            {
                "calls": 0,
                "max_evaluations": int(max_evaluations),
                "success_threshold": None if self.success_threshold is None else float(self.success_threshold),
                "callbacks": None,  # not known yet: registered after construction, recorded at the first evaluation
                "previous_stop_reason": None,
            }
        ]
        self._callbacks_ready = False  # whether the callbacks' states are restored and recorded in limits
        self.skips: dict[str, int] = {}
        self._skips_in_a_row = 0  # skips since the last counted evaluate_expr call
        if checkpoint is not None:
            # Computed before the store, so that unsaveable values leave no checkpoint file behind.
            fingerprint = self._fingerprint()
            self._results_header()  # raises TypeError for unsaveable metadata

        # Last, so that a failing constructor leaves no checkpoint file behind.
        self._store = ModelStore(checkpoint, "resume" if resume else "create")
        self.models = self._store.models
        self.trace = self._store.trace
        self._started = time.monotonic()
        self._stopped_at: float | None = None  # when evaluation stopped, which ends wall_time
        self._earlier_time = 0.0  # wall_time of the run's earlier sessions
        if checkpoint is None:
            return
        if resume:
            saved = self._store.get_header("fingerprint", {})
            changed = sorted(k for k in fingerprint.keys() | saved.keys() if fingerprint.get(k) != saved.get(k))
            if changed:
                self._store.close()
                hint = (
                    " Change limits with SR_dataset.extend_run."
                    if set(changed) <= {"max_evaluations", "success_threshold"}
                    else ""
                )
                raise ValueError(
                    f"[SR_evaluator] Cannot resume {os.fspath(checkpoint)!r}: these parameters differ from the "
                    f"saved run: {', '.join(changed)}.{hint}"
                )
            self.limits = _from_json_safe(self._store.get_header("limits"))
            self.resumed_from = self._store.snapshot
            if self.resumed_from is not None:
                self._callback_states = self.resumed_from.callbacks
            self._earlier_time = self._store.get_header("wall_time") or 0.0
            saved_reason = self._store.get_header("stop_reason")
            if saved_reason not in (None, "interrupted"):  # finished: kept whole, and stays stopped
                self.stop_reason = saved_reason
                self._stopped_at = self._started
                self.skips = self._store.get_header("skips", {})
            else:
                self._store.rewind()
                self._store.set_header("stop_reason", None)
                if self.resumed_from is not None:
                    saved_skips = self._store.get_header("snapshot_skips", {})
                    self.skips = saved_skips.get("counts", {})
                    self._skips_in_a_row = saved_skips.get("in_a_row", 0)
                self._store.set_header("skips", self.skips)
            self._current_best_error = self._store.min_error
        else:
            self._store.set_header("fingerprint", fingerprint)
            self._store.set_header("seed", None if seed is None else int(seed))
            self._store.set_header("limits", _to_json_safe(self.limits, "limits"))
        self._store.set_header("results", self._results_header())
        if self.stop_reason is None:
            # Limits the run already meets (e.g. after an extension that did not raise them).
            if self.success_threshold is not None and self._current_best_error < self.success_threshold:
                self.stop("success_threshold")
            elif 0 <= self.max_evaluations <= self.total_evaluations:
                self.stop("max_evaluations")
            elif 0 <= self.settings.evaluator.max_consecutive_skips <= self._skips_in_a_row:
                self.stop("max_consecutive_skips")
        self._store.commit()

    def _fingerprint(self) -> dict[str, str]:
        # One hash per setting a resumed run must share with the saved one (see Checkpoints).
        parts: dict[str, Any] = {
            "X": self.X,
            "y": self.y,
            "ground_truth": None if self.ground_truth is None else _as_tokens(self.ground_truth, self.symbol_library),
            "ranking_function": self.ranking,
            "required_constraints": list(self._required_constraints),
            "advisory_constraints": list(self._advisory_constraints),
            "symbol_library": self.symbol_library,
            "default_constant": self.default_constant,
            "seed": self.seed,
            # Normalised as in limits, so a threshold given as -1 and one read back as -1.0 match.
            "max_evaluations": int(self.max_evaluations),
            "success_threshold": None if self.success_threshold is None else float(self.success_threshold),
        }
        bk = self._background_knowledge
        parts["samplers"] = bk.samplers
        parts["constant_bounds"] = bk.constant_bounds
        if self.ground_truth_constants:  # in any order; [] is the same as None
            parts["ground_truth_constants"] = sorted(self.ground_truth_constants, key=lambda c: c.label or "")
        # One entry per setting that differs from its default; the limits are in required_constraints.
        for section, fields in self.settings.to_dict()["sections"].items():
            for field, value in fields.items():
                if field == "section_class":  # a custom section's class path, not a setting
                    continue
                if section == "evaluator" and field in ("max_constants", "max_expr_length"):
                    continue
                parts[f"{section}.{field}"] = value
        return {name: _fingerprint_hash(value, name) for name, value in parts.items()}

    def _results_header(self) -> dict[str, Any]:
        # What SR_results.from_checkpoint needs besides the models and trace.
        return {
            "success_threshold": self.success_threshold,
            "metadata": _to_json_safe(self.metadata, "metadata"),
            "default_constant": self.default_constant.to_dict(),
        }

    @property
    def should_stop(self) -> bool:
        """Whether evaluation has stopped: ``stop_reason is not None``.

        Later calls to [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr]
        warn and return ``None`` without being counted. After a signal caught by
        [stop_on_signals][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop_on_signals], reading
        it outside a [step][SRToolkit.evaluation.sr_evaluator.SR_evaluator.step] stops evaluation
        with the reason ``"interrupted"``.
        """
        if self._interrupted and not self._in_step and self.stop_reason is None:
            self.stop("interrupted")
        return self.stop_reason is not None

    @contextmanager
    def step(self) -> Generator[None, None, None]:
        """Defer an interrupt to the end of one step of an approach.

        Inside the ``with`` block a signal caught by
        [stop_on_signals][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop_on_signals] does not stop
        evaluation; the stop applies when the block ends (also with an exception). So the step's
        candidates are all evaluated and told: without it, Ctrl-C after the 2nd of 5 candidates would
        stop at the 3rd, the approach would be told 2, and its final snapshot would hold a state an
        uninterrupted run never reaches, so a resumed run would differ from it. Other stops (budget,
        threshold, callbacks, [stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop]) apply at
        once, since they happen at the same call in every run. The default step loop of
        [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] wraps each step in it.
        """
        self._in_step = True
        try:
            yield
        finally:
            self._in_step = False
            if self._interrupted and self.stop_reason is None:
                self.stop("interrupted")

    @property
    def snapshot_due(self) -> bool:
        """Whether a snapshot is due: with a ``checkpoint``, at the first new model or call
        ``snapshot_interval`` seconds or more after the last snapshot (or the start). Always
        ``False`` without a ``checkpoint`` or with ``snapshot_interval=None``.
        """
        return (
            self.checkpoint is not None
            and self.snapshot_interval is not None
            and self._store.snapshot_due(self.snapshot_interval)
        )

    def save_snapshot(self, state: bytes) -> None:
        """Save a snapshot of the run, replacing the previous one, and commit.

        The snapshot holds ``state``, the global random states (Python's ``random``, NumPy's and,
        if PyTorch is imported, its CPU and CUDA states), the callbacks' states and the
        skip counts (``skips`` and the count of skips in a row), at the current
        number of calls and models. Does nothing without a ``checkpoint``. Only the step loop of
        [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] should call it, between two
        steps, where ``state`` is the approach's whole search state: a resumed run always
        continues with that loop.

        Args:
            state: The approach's search state, as returned by its
                [dump_search_state][SRToolkit.approaches.sr_approach.SR_approach.dump_search_state].

        Raises:
            TypeError: If a callback's state cannot be saved.
            sqlite3.Error: If the commit fails (see
                [ModelStore.commit][SRToolkit.evaluation.model_store.ModelStore.commit]).
        """
        if self.checkpoint is None:
            return
        # Callbacks not yet restored from the snapshot this run resumed from keep their saved state.
        callbacks = self._callback_states if self._callback_states is not None else self._callbacks.get_state()
        headers = {
            "wall_time": self.wall_time,
            "skips": self.skips,
            "snapshot_skips": {"counts": self.skips, "in_a_row": self._skips_in_a_row},  # what a resume restores
        }
        self._store.save_snapshot(state, _capture_rngs(), callbacks, headers)

    @contextmanager
    def stop_on_signals(self) -> Generator[None, None, None]:
        """Stop evaluation on SIGINT (Ctrl-C) or SIGTERM while the ``with`` block runs.

        The first signal stops evaluation with the reason ``"interrupted"`` at the next read of
        ``should_stop`` or call to
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] outside a
        [step][SRToolkit.evaluation.sr_evaluator.SR_evaluator.step], or at the end of the block, and
        commits the checkpoint. A second signal raises ``KeyboardInterrupt`` at once.

        Cluster schedulers send SIGTERM some time before they kill a job (30 s by default on SLURM):
        if the current step ends in that time and the approach implements the
        [step interface][SRToolkit.approaches.sr_approach.SR_approach.supports_steps], the job
        resumes without losing work.

        The previous handlers are restored at the end of the block. Outside the main thread, where
        Python cannot install signal handlers, the block runs without them.

        Examples:
            >>> import os
            >>> se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 3.0]))
            >>> with se.stop_on_signals():
            ...     os.kill(os.getpid(), signal.SIGINT)  # as Ctrl-C would
            ...     se.should_stop
            True
            >>> se.stop_reason
            'interrupted'

        Raises:
            KeyboardInterrupt: At a second signal.
        """
        # signal.signal raises ValueError outside the main thread, so run the block unprotected there.
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        # Python runs a handler in the main thread between two bytecodes, so it can interrupt the
        # evaluator anywhere, e.g. halfway through a store insert. It therefore only sets a flag;
        # should_stop, evaluate_expr and the end of the block or of a step turn the flag into a
        # stop at a point where the evaluator is consistent.
        def handler(signum: int, frame: Any) -> None:
            if self._interrupted:  # a second signal: the user insists, abort at once
                raise KeyboardInterrupt
            self._interrupted = True

        # signal.signal returns the handler it replaces, kept to be restored at the end.
        previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield
        finally:
            # Runs on a normal end and on an exception (including the KeyboardInterrupt above).
            for sig, old in previous.items():
                # None: the old handler was not installed from Python; fall back to the default.
                signal.signal(sig, signal.SIG_DFL if old is None else old)
            if self._interrupted:
                # A flag not yet turned into a stop (e.g. the search returned right after the
                # signal) still records "interrupted" and commits; an earlier stop reason is kept.
                self.stop("interrupted")
                # Reset, so a later block or should_stop does not see this signal again.
                self._interrupted = False

    def stop(self, reason: str) -> None:
        """Stop evaluation and record why, unless it has already stopped (the first reason is kept).

        An approach calls this when it ends its search with budget left, e.g. because its search
        space is exhausted, so the results explain the early end. The first stop saves ``reason``
        and commits the checkpoint.

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
            sqlite3.Error: If the commit fails (see
                [ModelStore.commit][SRToolkit.evaluation.model_store.ModelStore.commit]).
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"[SR_evaluator.stop] reason must be a non-empty str, got {reason!r}.")
        if self.stop_reason is None:
            self.stop_reason = reason
            self._stopped_at = time.monotonic()
            if self.checkpoint is not None:
                self._store.set_header("stop_reason", reason)
            self._commit()

    def _setup_callbacks(self) -> None:
        if self._callbacks_ready:
            return
        # Callbacks are registered after construction, so they are set up at the first counted
        # call of evaluate_expr or count_skip: restore their saved states, then record them in the
        # current limit segment, or warn if a plain resume has other ones. A config that cannot be
        # saved raises here, at the start, also without a checkpoint (results are saved with their
        # limits).
        configs = [cb.to_dict() for cb in self._callbacks.get_callbacks()]
        saved = _to_json_safe(configs, "callbacks")
        if self._callback_states is not None:
            self._callbacks.set_state(self._callback_states)
            self._callback_states = None
        segment = self.limits[-1]
        if segment["callbacks"] is None:
            if self.checkpoint is not None:
                recorded = [*self.limits[:-1], {**segment, "callbacks": configs}]
                self._store.set_header("limits", _to_json_safe(recorded, "limits"))
            segment["callbacks"] = configs
        elif saved != _to_json_safe(segment["callbacks"], "callbacks"):
            warnings.warn(
                "[SR_evaluator] The callbacks differ from those this part of the run started with; "
                "change limits with SR_dataset.extend_run.",
                UserWarning,
            )
        self._callbacks_ready = True

    def count_skip(self, reason: str) -> None:
        """Count one expression the approach skipped without evaluating it, by reason.

        Call it for each expression the approach discards without passing it to
        ``evaluate_expr``, so the results show how many it discarded and why (see
        [EvalResult.skips][SRToolkit.utils.types.EvalResult]). It uses no budget and may be called
        from ``search``, ``ask`` or ``tell``. The built-in approaches use the reasons
        ``"evaluated"`` (evaluated before), ``"rejected"`` (breaks a required constraint) and
        ``"not_expression"`` (e.g. a latent vector that does not decode); others are allowed.

        Skips in a row are counted too. Every counted call of ``evaluate_expr`` (cache hits
        included) resets them; lookups (``get_model``, ``check_constraints``) do not. When they
        reach the setting ``evaluator.max_consecutive_skips`` (see
        [EvaluatorSettings][SRToolkit.utils.settings.EvaluatorSettings]), evaluation
        [stops][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop] with the reason
        ``"max_consecutive_skips"``.

        Each skip, like each evaluation, is then reported to the callbacks'
        [on_skip][SRToolkit.evaluation.callbacks.SR_callback.on_skip], which may also stop
        evaluation; on the same skip ``"max_consecutive_skips"`` is checked first, so its reason is
        kept. Skips are still counted and reported after evaluation has stopped. Step approaches
        report skipped proposals through [Candidate][SRToolkit.approaches.sr_approach.Candidate]
        instead; the step loop counts them here.

        The counts are saved with each commit and snapshot. A resumed run continues from the counts
        of its snapshot, or from zero without one, so they match its models; a finished run resumed
        as is keeps its counts.

        Args:
            reason: Why the expression was skipped.

        Raises:
            TypeError: At the first counted call of ``count_skip`` or ``evaluate_expr``, if a
                callback's ``to_dict`` cannot be saved; nothing is then counted.
            ValueError: If ``reason`` is not a non-empty ``str``.
            sqlite3.Error: If the commit of a stop fails, whether the stop comes from
                ``max_consecutive_skips`` or a callback (see
                [ModelStore.commit][SRToolkit.evaluation.model_store.ModelStore.commit]).

        Examples:
            >>> se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
            >>> se.count_skip("rejected")
            >>> se.count_skip("rejected")
            >>> se.skips
            {'rejected': 2}
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"[SR_evaluator.count_skip] reason must be a non-empty str, got {reason!r}.")
        self._setup_callbacks()
        self.skips[reason] = self.skips.get(reason, 0) + 1
        self._skips_in_a_row += 1
        if 0 <= self.settings.evaluator.max_consecutive_skips <= self._skips_in_a_row:
            self.stop("max_consecutive_skips")
        event = ExprSkipped(
            reason=reason,
            skips_in_a_row=self._skips_in_a_row,
            evaluation_number=self.total_evaluations,
            experiment_id=self._experiment_id,
        )
        callback_reason = self._callbacks.on_skip(event)
        if callback_reason is not None:
            self.stop(callback_reason)

    def close(self) -> None:
        """Commit, with the wall time and skips so far, and close the database of ``models`` and ``trace``.

        Call it when done with the evaluator: a ``checkpoint`` file is then complete on its own,
        without its ``-wal`` and ``-shm`` files. Collect the results first: afterwards every method that reads or writes ``models`` or ``trace`` raises
        ``sqlite3.ProgrammingError``. Closing again does nothing.

        Raises:
            sqlite3.Error: If the final commit fails; the database is closed anyway, and the
                checkpoint keeps what was committed before.
        """
        if self.checkpoint is not None and not self._store._closed:
            self._store.set_header("wall_time", self.wall_time)
            self._store.set_header("skips", self.skips)
        self._store.close()

    def _commit(self) -> None:
        # Commit with the wall time and skips so far, so a checkpoint keeps them also after a crash.
        if self.checkpoint is not None:
            self._store.set_header("wall_time", self.wall_time)
            self._store.set_header("skips", self.skips)
        self._store.commit()

    @property
    def wall_time(self) -> float:
        """Seconds the run has taken: from the evaluator's creation until evaluation stopped, or
        until now if it has not. A resumed run adds the time of its earlier sessions, saved in the
        checkpoint.
        """
        end = time.monotonic() if self._stopped_at is None else self._stopped_at
        return self._earlier_time + end - self._started

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
        ``target``, ``named_constants``, ``description``, ``reference_grammar``,
        ``constant_bounds``, ``default_constant`` and ``samplers``, so it describes the task the
        evaluator scores. Its ``constraints`` are the advisory constraints.
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

    def _stored_form(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None,
        warn: bool = False,
    ) -> tuple[list[str] | Node, list[str], tuple[Constant, ...] | None]:
        # The stored form (see the module docs): (expr, its tokens, its specs sorted by label or None).
        # Simplified with simplify_expr (as given if that fails, with a warning if warn and the
        # expression parses), folded with fold_constants, then canonicalised with
        # canonicalise_constants if no specs are left. Shared by every lookup so they match.
        s = self.settings.evaluator
        mode = s.fold_constants
        folded = False
        if s.simplify_expr:
            try:
                expr, free_constants = simplify(
                    expr,
                    self.symbol_library,
                    free_constants,
                    rules=s.simplify_rules,
                    max_iterations=s.simplify_max_iterations,
                    max_nodes=s.simplify_max_nodes,
                    max_time=s.simplify_max_time,
                    fold=mode,
                    fold_aware=s.simplify_fold_aware,
                )
                folded = True
            except Exception as e:
                # A malformed expression is scored "invalid" as given; its status says why.
                if warn and not isinstance(e, ExpressionParseError):
                    expr_str = "".join(_as_tokens(expr, self.symbol_library))
                    warnings.warn(f"Unable to simplify: {expr_str}, problems with subexpression {e}")
        if mode != "none" and not folded:
            try:
                expr, free_constants = fold(expr, self.symbol_library, free_constants, mode)
            except ValueError:
                # Malformed (ExpressionParseError): scored as given, as "invalid". Specs that do not
                # fit: evaluate_expr has raised already, so this is a lookup, which then misses.
                pass
        specs = tuple(sorted(free_constants, key=lambda c: c.label or "")) if free_constants else None
        tokens = _as_tokens(expr, self.symbol_library)
        if s.canonicalise_constants and specs is None:
            canonical = canonicalise(tokens, self.symbol_library)
            if canonical != tokens:
                # The ranking gets the canonical tokens, so the fit's seed is that of the canonical form.
                expr = tokens = canonical
        return expr, tokens, specs

    def get_model(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
    ) -> ModelResult | None:
        """Return the result of an already evaluated expression, or ``None``.

        Does not count toward ``max_evaluations`` and fires no callbacks. The expression is matched
        in its [stored form][SRToolkit.evaluation.sr_evaluator], together with the specs that
        remain, as by [stored_tokens][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stored_tokens].

        Args:
            expr: The expression, as a token list or a [Node][SRToolkit.utils.expression_tree.Node].
            free_constants: The specs it was evaluated with, in any order; ``None`` or empty for none.

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
        _, tokens, specs = self._stored_form(expr, free_constants)
        found = self._store.lookup(tokens, _spec_key(specs))
        return None if found is None else self.models[found[0]]

    def stored_tokens(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
    ) -> list[str]:
        """Return the tokens of the stored form of ``expr``.

        These are the tokens
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] stores
        ``expr`` under (see [Stored form][SRToolkit.evaluation.sr_evaluator]), so two
        inputs with the same stored tokens and the same remaining specs are one model. Approaches
        use it to recognise repeats among expressions not yet evaluated. Does not count toward
        ``max_evaluations``, records nothing and emits no warnings. The specs are not checked:
        specs that do not fit ``expr`` match no model. With the ``evaluator.simplify_max_time``
        setting, the result can differ between calls (the time limit depends on the machine's
        load), so a lookup can miss a model that was evaluated.

        Args:
            expr: The expression, as a token list or a [Node][SRToolkit.utils.expression_tree.Node].
            free_constants: Its specs, in any order; ``None`` or empty for none.

        Returns:
            The stored form as a list of infix tokens.

        Raises:
            TypeError: If ``expr`` is neither a token list nor a ``Node``.

        Examples:
            >>> X = np.array([[1.0], [2.0]])
            >>> se = SR_evaluator(X, X[:, 0], settings=Settings(evaluator={"canonicalise_constants": True}))
            >>> "".join(se.stored_tokens(["C_3", "*", "X_0", "+", "C_3"]))
            'C_0*X_0+C_0'
        """
        return self._stored_form(expr, free_constants)[1]

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
        same input: the [stored form][SRToolkit.evaluation.sr_evaluator] of ``expr``. So an empty
        ``required`` list means ``evaluate_expr(expr, free_constants)`` will neither reject
        ``expr`` in a pre-check nor give it the status ``"error"`` there (unless a
        ``simplify_max_time`` limit simplifies it differently). Unlike evaluation, every failed
        pre-check is reported, not only the first. A check that raises, or returns something other
        than a ``CheckResult``, is reported like a failed one, with ``error`` set (see
        [CheckResult][SRToolkit.utils.types.CheckResult]), and emits a ``UserWarning``. Checks a
        constraint does not implement are skipped.

        Args:
            expr: The expression, as an infix token list or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            free_constants: Specs for its labelled constants, validated against ``expr`` as given
                by [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]. They
                shape the stored form (see above); the pre-checks themselves see only its tokens.

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
        expr_list = self._stored_form(expr_list, free_constants)[1]
        return (
            _failed_checks(self._required_constraints, "pre", expr_list),
            _failed_checks(self._advisory_constraints, "pre", expr_list),
        )

    def register_callbacks(self, callbacks: SR_callback | CallbackDispatcher | None = None) -> None:
        """Add callbacks for monitoring and early stopping to those already registered.

        The evaluator enforces ``success_threshold`` and ``max_evaluations`` itself; callbacks add
        further stop conditions (e.g. a time limit).

        Register callbacks before the first counted call of ``evaluate_expr`` or ``count_skip``:
        that call restores their saved states (in a resumed run) and records their configs in
        [limits][SRToolkit.utils.types.EvalResult]. A callback registered later receives events,
        but it is not recorded and its saved state is not restored.

        Examples:
            >>> from SRToolkit.evaluation.callbacks import TimeLimitCallback
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> se.register_callbacks(TimeLimitCallback(time_limit=60))

        Args:
            callbacks: A single [SR_callback][SRToolkit.evaluation.callbacks.SR_callback], or a
                [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] whose
                callbacks are added (the dispatcher itself is not changed). ``None`` adds nothing.

        Raises:
            TypeError: If ``callbacks`` is of another type (e.g. a list).
        """
        if isinstance(callbacks, SR_callback):
            self._callbacks.add(callbacks)
        elif isinstance(callbacks, CallbackDispatcher):
            for cb in callbacks.get_callbacks():
                self._callbacks.add(cb)
        elif callbacks is not None:
            raise TypeError(
                "[SR_evaluator.register_callbacks] Expected an SR_callback, a CallbackDispatcher or None, "
                f"got {type(callbacks).__name__}."
            )

    def evaluate_expr(
        self,
        expr: list[str] | Node,
        free_constants: Sequence[Constant] | None = None,
        verbose: int = 0,
    ) -> ModelResult | None:
        """Evaluate an expression, or return its cached model.

        A new expression is parsed, checked against the constraints, scored by the ranking
        function and stored in ``models`` (statuses and the order of checks are in the
        [module docs][SRToolkit.evaluation.sr_evaluator]). An expression whose stored form and
        remaining specs match a stored model is not scored again; that model is returned. Either
        way the call is appended to ``trace``, counts toward ``max_evaluations``, resets the count
        of skips in a row (see [count_skip][SRToolkit.evaluation.sr_evaluator.SR_evaluator.count_skip])
        and is reported to the callbacks: ``on_expr_evaluated`` for every call, plus
        ``on_best_expression`` when the error improves on the best so far. Evaluation then stops
        if the error is below ``success_threshold``, the budget is used up, or a callback asks to
        stop (see [Stopping][SRToolkit.evaluation.sr_evaluator]).

        The expression is checked, scored and stored in its
        [stored form][SRToolkit.evaluation.sr_evaluator], set by the settings
        ``evaluator.simplify_expr``, ``evaluator.fold_constants`` and
        ``evaluator.canonicalise_constants``, with the specs that remain. If a well-formed
        expression cannot be simplified, a ``UserWarning`` is emitted and it is used as given
        (still folded and canonicalised); a malformed one is ``"invalid"`` without a warning.

        Args:
            expr: The expression, as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            free_constants: Specs for labelled constants of ``expr`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]), in any order. Constants
                without one, and a spec's unset ``prior`` or ``bounds``, use ``default_constant``.
                Passed to the ranking and stored as ``ModelResult.free_constants``. ``None`` or
                empty: no specs.
            verbose: ``0`` (default): no logging. ``1``: log at DEBUG level (logger
                ``SRToolkit.evaluation.sr_evaluator``) each new expression with its final status,
                error and ``status_detail``, and each cache hit. ``2``: also let NumPy
                floating-point warnings through while scoring.

        Returns:
            The expression's [ModelResult][SRToolkit.utils.types.ModelResult], as ``models``
            stores it: its ``expr`` in the stored form (simplified, folded, canonicalised), its
            ``error`` under the active ranking function (``+inf`` if the ``status`` is not
            ``"ok"``), the fitted ``parameters`` (if any) and the ``status_detail`` of a failure;
            for a model that could not be saved, the stored ``"error"`` model (see the module
            docs). A new copy each call, like ``models[i]``: changing it saves nothing. ``None``
            if evaluation has already stopped: the call then emits a warning and is not counted.

        Raises:
            TypeError: If ``expr`` is neither a token list nor a ``Node``. If a spec in
                ``free_constants`` cannot be saved (see
                [Constant.to_dict][SRToolkit.utils.free_constants.Constant.to_dict]); nothing is
                then recorded or counted. At the first counted call of ``evaluate_expr`` or
                ``count_skip``, if a callback's ``to_dict`` cannot be saved; nothing is then
                counted.
            ValueError: If ``free_constants`` does not fit a well-formed ``expr`` (see
                [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]).
                Nothing is then recorded or counted.
            sqlite3.Error: If a commit of the checkpoint fails (see
                [ModelStore.commit][SRToolkit.evaluation.model_store.ModelStore.commit]).

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9]])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> model = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(model.error < 1e-6, model.status, round(float(model.parameters[0]), 3))
            True ok 2.0
            >>> # A labelled constant with a spec of its own: started from U(0, 10), fitted within (0, 10)
            >>> from SRToolkit.dataset import UniformSampling
            >>> from SRToolkit.utils.free_constants import Constant
            >>> spec = Constant(label="C_0", prior=UniformSampling(0, 10), bounds=(0.0, 10.0))
            >>> se = SR_evaluator(X, y, seed=42)
            >>> model = se.evaluate_expr(["C_0", "*", "X_1", "-", "X_0"], [spec])
            >>> print(model.error < 1e-6, model.free_constants == (spec,))
            True True
            >>> # With the setting evaluator.simplify_expr, the simplified expression is scored and stored
            >>> X = np.array([[0, 1], [0, 2], [0, 3]])
            >>> settings = Settings(evaluator={"simplify_expr": True})
            >>> se = SR_evaluator(X, np.array([2, 3, 4]), seed=42, success_threshold=-1, settings=settings)
            >>> model = se.evaluate_expr(["C", "+", "X_0", "*", "X_1", "/", "X_0"])
            >>> "".join(model.expr), bool(0.99 < model.parameters[0] < 1.01)
            ('C+X_1', True)
        """
        if self.should_stop:
            warnings.warn(
                f"[SR_evaluator] Evaluation has already stopped ({self.stop_reason}); "
                "the call is not evaluated or counted.",
                UserWarning,
            )
            return None
        else:
            if (
                self.settings.evaluator.simplify_expr or self.settings.evaluator.fold_constants != "none"
            ) and free_constants:
                # Specs that do not fit raise here, not as a failed simplification or fold (see _evaluate_new).
                try:
                    tree = _as_tree(expr, self.symbol_library)
                except ExpressionParseError:
                    tree = None  # malformed: simplify fails too, and the model is "invalid"
                if tree is not None:
                    resolve_constants(tree, free_constants, self.default_constant, self.symbol_library)
            expr, expr_list, specs = self._stored_form(expr, free_constants, warn=True)

            expr_str = "".join(expr_list)
            spec_key = _spec_key(specs)
            self._setup_callbacks()

            found = self._store.lookup(expr_list, spec_key)
            self._skips_in_a_row = 0  # either way a counted call follows
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
                if self._store.commit_due:
                    self._commit()
                return self.models[index]

            else:
                model = self._evaluate_new(expr, expr_list, expr_str, verbose, specs)
                model.free_constants = specs
                try:
                    index = self._store.add(expr_list, spec_key, model)
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
                    index = self._store.add(expr_list, spec_key, model)
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
                if self._store.commit_due:
                    self._commit()

                return self.models[index]

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
    ) -> float:
        """Return another metric of an already-evaluated model, without using the budget.

        Lets an approach steer its search by a measure other than the ranking error (e.g. R² or a
        custom callable) without access to ``X`` and ``y``. It does not count toward
        ``max_evaluations`` and changes nothing: not the ranking, the cache, ``trace`` or the
        callbacks. The model must already have been evaluated with the same specs by
        [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr], so that
        un-evaluated expressions cannot be scored for free; it is matched as in
        [get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model]. The metric is
        computed with the model's fitted parameters on the evaluator's ``X`` and ``y``.

        Args:
            expr: Expression as a token list in infix notation or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            metric: The name of a built-in metric in
                [REGRESSION_METRICS][SRToolkit.utils.measures.REGRESSION_METRICS] (``"mse"``,
                ``"rmse"``, ``"mae"``, ``"r2"``, ``"nrmse"``), or a callable
                ``metric(predict, X, y) -> float``, where ``predict(X) -> y_pred`` has the fitted
                parameters bound in (so a metric can also evaluate the model away from ``X``).
                Default ``"r2"``.
            free_constants: The specs ``expr`` was evaluated with, as in ``evaluate_expr``;
                ``None`` or empty for none.

        Returns:
            The metric value. ``NaN`` if the ranking is not ``"rmse"``, if ``y`` was not given to
            the evaluator (e.g. RMSE generated it from the ground truth), if the model's status is
            not ``"ok"``, or if the metric raises.

        Raises:
            ValueError: If ``expr`` has not been evaluated yet with these specs, or ``metric`` is a
                string that is not a built-in metric.
            TypeError: If ``metric`` is neither a string nor a callable.

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
        """
        _, expr_list, specs = self._stored_form(expr, free_constants)
        expr_str = "".join(expr_list)

        found = self._store.lookup(expr_list, _spec_key(specs))
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

        return evaluate_metric(
            expr_list,
            model.parameters,
            self.X,
            self.y,
            metric,
            self.symbol_library,
            backend=self.settings.evaluator.backend,
        )

    def get_results(
        self,
        approach_name: str = "",
        top_k: int = 20,
        results: "SR_results | None" = None,
        keep_all_models: bool = True,
    ) -> "SR_results":
        """Return the results of the evaluation.

        Saves the [wall_time][SRToolkit.evaluation.sr_evaluator.SR_evaluator.wall_time] and
        ``skips`` it reports and commits the checkpoint first. It does not stop evaluation, so it can be called during a
        run; a checkpoint records the run as finished only once evaluation stops.
        [SR_approach.run][SRToolkit.approaches.sr_approach.SR_approach.run] stops it when the search
        returns; after calling ``search`` directly, call ``stop`` yourself.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
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
            keep_all_models: If ``True`` (default), the result holds every model and the trace. If
                ``False``, only the ``top_k`` best models and a
                [RunSummary][SRToolkit.utils.types.RunSummary] (counts, the best-so-far history with
                its models, the Pareto front), so memory and file size do not grow with the run;
                all models stay in the checkpoint, if one is kept. With ``True``, warns as
                [add_results][SRToolkit.evaluation.results.SR_results.add_results] for a very large
                run. The result's [checkpoint][SRToolkit.utils.types.EvalResult] reference is not
                set.

        Returns:
            An instance of the SR_results object with the results of the evaluation. Its
            ``stop_reason`` is the evaluator's, or ``"returned"`` if evaluation has not stopped.

        Raises:
            ValueError: If ``keep_all_models`` is ``False`` and ``top_k <= 0``.
            sqlite3.Error: If the commit fails (see
                [ModelStore.commit][SRToolkit.evaluation.model_store.ModelStore.commit]).
        """
        if not keep_all_models and top_k <= 0:
            raise ValueError(f"[SR_evaluator.get_results] keep_all_models=False needs top_k >= 1, got {top_k}.")
        wall_time = self.wall_time  # read once: it still grows if evaluation has not stopped
        if self.checkpoint is not None:
            self._store.set_header("wall_time", wall_time)
            self._store.set_header("skips", self.skips)
        self._store.commit()
        if results is None:
            results = SR_results()

        summary = None
        models: Sequence[ModelResult] = self.models
        if not keep_all_models:
            models, summary = self._store.summary(top_k)
        results.add_results(
            models,
            self.trace,
            top_k,
            self.success_threshold,
            approach_name,
            self.metadata,
            self.default_constant,
            self.stop_reason or "returned",
            summary,
            self.limits,  # add_results copies limits and skips
            self.skips,
        )
        results.results[-1].wall_time = wall_time
        return results
