"""Concrete [ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] implementations
that post-process SR results with additional information.

Available augmenters: [ExpressionToLatex][SRToolkit.evaluation.result_augmentation.ExpressionToLatex],
[ExpressionSimplifier][SRToolkit.evaluation.result_augmentation.ExpressionSimplifier],
[HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric].
Custom augmenters work automatically without registration — just subclass
[ResultAugmenter][SRToolkit.evaluation.results.ResultAugmenter] and implement the
required methods.
"""

import copy
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.evaluation.ranking import RankingFunction
from SRToolkit.evaluation.results import ResultAugmenter
from SRToolkit.evaluation.sr_evaluator import evaluate_metric
from SRToolkit.utils import SymbolLibrary, simplify
from SRToolkit.utils.expression import Expression, _to_expression
from SRToolkit.utils.expression_simplifier import (
    DefaultRules,
    SimplificationRules,
    _check_fold_mode,
    _check_settings,
)
from SRToolkit.utils.free_constants import Constant, _spec_key
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.types import EvalResult, ModelResult

if TYPE_CHECKING:
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
    from SRToolkit.dataset.data_split import DataSplit


class ExpressionToLatex(ResultAugmenter):
    def __init__(
        self,
        symbol_library: SymbolLibrary,
        scope: str = "top",
        verbose: bool = False,
        name: str = "ExpressionToLatex",
    ) -> None:
        """Converts expressions inside the results to LaTeX strings.

        Args:
            symbol_library: Symbol library used to produce LaTeX templates for each token.
            scope: Which expressions to convert.

                - ``"best"``: only the best expression.
                - ``"top"``: the best expression and all top-k models.
                - ``"all"``: everything in ``"top"`` plus all evaluated expressions.
            verbose: If ``True``, emits a warning when LaTeX conversion fails for an expression.
                Default ``False``.
            name: Key used in
                ``augmentations`` dict of [EvalResult][SRToolkit.utils.types.EvalResult] and
                [ModelResult][SRToolkit.utils.types.ModelResult].
                Default ``"ExpressionToLatex"``.
        """
        super().__init__(name)
        self.symbol_library = symbol_library

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        self.verbose = verbose

    def write_results(self, results: EvalResult) -> None:
        """Write LaTeX representations into *results* and its models.

        Stores ``{"best_expr_latex": ...}`` in
        [EvalResult][SRToolkit.utils.types.EvalResult] ``augmentations``.
        Also stores ``{"expr_latex": ...}`` in each model's augmentations when
        ``scope`` is ``"top"`` or ``"all"``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.

        Raises:
            ValueError: If ``scope`` is ``"all"`` and *results* does not keep all its models (it
                has a [summary][SRToolkit.utils.types.EvalResult]); nothing is added then.
        """
        if self.scope == "all" and results.summary is not None:
            raise ValueError(
                f"[{type(self).__name__}] scope='all' needs all models, but this result keeps only its top models "
                "(keep_all_models=False)."
            )
        eval_data: dict[str, Any] = {}
        try:
            eval_data["best_expr_latex"] = Expression(results.top_models(1)[0].expr, self.symbol_library).to_latex()
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Unable to convert best expression to LaTeX: {e}")
        results.add_augmentation(self.name, eval_data, self._type)

        scoped = results.all_models if self.scope == "all" else results.top_models() if self.scope == "top" else []
        for model in scoped:
            try:
                model.add_augmentation(
                    self.name,
                    {"expr_latex": Expression(model.expr, self.symbol_library).to_latex()},
                    self._type,
                )
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Unable to convert expression {''.join(model.expr)} to LaTeX: {e}")

    @classmethod
    def format_eval_result(cls, data: dict[str, Any]) -> str:
        """Format experiment-level LaTeX augmentation data for display.

        Args:
            data: Augmentation dict containing ``"best_expr_latex"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        latex = data.get("best_expr_latex", "")
        return f"LaTeX of the best expression: {latex}" if latex else ""

    @classmethod
    def format_model_result(cls, data: dict[str, Any]) -> str:
        """Format per-model LaTeX augmentation data for display.

        Args:
            data: Augmentation dict containing ``"expr_latex"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        latex = data.get("expr_latex", "")
        return f"LaTeX: {latex}" if latex else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """Creates a dictionary representation of the ExpressionToLatex augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "type": "ExpressionToLatex",
            "name": self.name,
            "symbol_library": self.symbol_library.to_dict(),
            "scope": self.scope,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExpressionToLatex":
        """Creates an instance of the ExpressionToLatex augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.

        Returns:
            An instance of the ExpressionToLatex augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[ExpressionToLatex.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        return ExpressionToLatex(
            symbol_library=data["symbol_library"],
            scope=data["scope"],
            verbose=data["verbose"],
            name=data["name"],
        )


class ExpressionSimplifier(ResultAugmenter):
    def __init__(
        self,
        symbol_library: SymbolLibrary,
        scope: str = "top",
        verbose: bool = False,
        name: str = "ExpressionSimplifier",
        rules: SimplificationRules | None = None,
        max_iterations: int | None = None,
        max_nodes: int | None = None,
        max_time: float | None = None,
        fold: Literal["none", "default", "keep"] | None = None,
        fold_aware: bool | None = None,
    ) -> None:
        """Simplified expressions of the best, top or all models of a result.

        Each expression is simplified by [simplify][SRToolkit.utils.expression_simplifier.simplify]
        with its model's ``free_constants``.

        Args:
            symbol_library: Symbol library of the expressions. It also decides which of the
                [DefaultRules][SRToolkit.utils.expression_simplifier.DefaultRules] apply.
            scope: Which expressions to simplify.

                - ``"best"``: only the best expression.
                - ``"top"``: the best expression and all top-k models.
                - ``"all"``: everything in ``"top"`` plus all evaluated expressions.
            verbose: If ``True``, emits a warning when simplification fails for an expression.
                Default ``False``.
            name: Key used in
                ``augmentations`` dict of [EvalResult][SRToolkit.utils.types.EvalResult] and
                [ModelResult][SRToolkit.utils.types.ModelResult].
                Default ``"ExpressionSimplifier"``.
            rules: As in [simplify][SRToolkit.utils.expression_simplifier.simplify].
            max_iterations: As in [simplify][SRToolkit.utils.expression_simplifier.simplify]; a
                non-negative ``int``, or ``None`` for its default.
            max_nodes: As in [simplify][SRToolkit.utils.expression_simplifier.simplify]; a
                non-negative ``int``, or ``None`` for its default.
            max_time: As in [simplify][SRToolkit.utils.expression_simplifier.simplify]; a positive
                number or ``None``.
            fold: As in [simplify][SRToolkit.utils.expression_simplifier.simplify], or ``None`` for
                its default (no folding).
            fold_aware: As in [simplify][SRToolkit.utils.expression_simplifier.simplify], or
                ``None`` for its default.

        Raises:
            TypeError: If ``rules`` is neither a
                [SimplificationRules][SRToolkit.utils.expression_simplifier.SimplificationRules] nor
                ``None``, or ``fold_aware`` is neither a ``bool`` nor ``None``.
            ValueError: If ``scope`` is not ``"best"``, ``"top"`` or ``"all"``; if
                ``max_iterations`` or ``max_nodes`` is not a non-negative ``int`` or numpy integer
                (``bool`` is rejected); if ``max_time`` is not ``None`` or a positive number
                (``bool`` and ``nan`` are rejected); if ``fold`` is neither ``None`` nor
                ``"none"``, ``"default"`` or ``"keep"``.
        """
        super().__init__(name)
        if scope not in ["best", "top", "all"]:
            raise ValueError(f"[ExpressionSimplifier] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        _check_settings(rules, max_iterations, max_nodes, max_time, fold_aware)
        if fold is not None:
            _check_fold_mode(fold, "fold")
        self.symbol_library = symbol_library
        self.rules = DefaultRules() if rules is None else rules
        self.max_iterations = max_iterations
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.fold = fold
        self.fold_aware = fold_aware
        self.scope = scope

        self.verbose = verbose

    def write_results(self, results: EvalResult) -> None:
        """Write simplified expressions into *results* and its models.

        Stores ``{"simplified_best_expr": ...}`` in the
        [EvalResult][SRToolkit.utils.types.EvalResult] ``augmentations``, and
        ``{"simplified_expr": ...}`` in each model's when ``scope`` is ``"top"`` or ``"all"``. A
        value is the simplified tokens joined into one string (e.g. ``"(C+C)*X_0"``). With folding
        (``fold`` other than ``None`` or ``"none"``), each entry also has ``"folded": True``: the
        simplified expression is then a different hypothesis. If
        simplifying an expression fails, its entry has no such key (with a warning if
        ``verbose``). A result without models gets an entry without ``"simplified_best_expr"``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.

        Raises:
            ValueError: If ``scope`` is ``"all"`` and *results* does not keep all its models (it
                has a [summary][SRToolkit.utils.types.EvalResult]); nothing is added then.
        """
        if self.scope == "all" and results.summary is not None:
            raise ValueError(
                f"[{type(self).__name__}] scope='all' needs all models, but this result keeps only its top models "
                "(keep_all_models=False)."
            )
        if not results.top_models(1):  # nothing was evaluated
            results.add_augmentation(self.name, {}, self._type)
            return
        best = results.top_models(1)[0]
        scoped = results.all_models if self.scope == "all" else results.top_models() if self.scope == "top" else []
        for model, key in [(best, "simplified_best_expr"), *((m, "simplified_expr") for m in scoped)]:
            data: dict[str, Any] = {}
            try:
                options: dict[str, Any] = {
                    "max_iterations": self.max_iterations,
                    "max_nodes": self.max_nodes,
                    "fold": self.fold,
                    "fold_aware": self.fold_aware,
                }
                simplified = simplify(
                    Expression(model.expr, self.symbol_library, model.free_constants),
                    rules=self.rules,
                    max_time=self.max_time,
                    **{k: v for k, v in options.items() if v is not None},
                )
                data[key] = str(simplified)
                if self.fold not in (None, "none"):
                    data["folded"] = True
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Unable to simplify {''.join(model.expr)}: {e}")
            (results if key == "simplified_best_expr" else model).add_augmentation(self.name, data, self._type)

    @classmethod
    def format_eval_result(cls, data: dict[str, Any]) -> str:
        """Format experiment-level simplification data for display.

        Args:
            data: Augmentation dict containing ``"simplified_best_expr"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        simplified = data.get("simplified_best_expr", "")
        label = "Simplified (folded)" if data.get("folded") else "Simplified"
        return f"{label}: {simplified}" if simplified else ""

    @classmethod
    def format_model_result(cls, data: dict[str, Any]) -> str:
        """Format per-model simplification data for display.

        Args:
            data: Augmentation dict containing ``"simplified_expr"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        simplified = data.get("simplified_expr", "")
        label = "Simplified (folded)" if data.get("folded") else "Simplified"
        return f"{label}: {simplified}" if simplified else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """Serialize the augmenter to a JSON-compatible dictionary.

        Args:
            base_path: Ignored.
            name: Ignored.

        Returns:
            Format 2: ``format_version``, ``type``, ``name``, ``symbol_library``, ``scope``,
            ``verbose``, ``rules``, ``max_iterations``, ``max_nodes``, ``max_time``, ``fold`` and
            ``fold_aware``.
        """
        return {
            "format_version": 2,
            "type": "ExpressionSimplifier",
            "name": self.name,
            "symbol_library": self.symbol_library.to_dict(),
            "scope": self.scope,
            "verbose": self.verbose,
            "rules": self.rules.to_dict(),
            "max_iterations": self.max_iterations,
            "max_nodes": self.max_nodes,
            "max_time": self.max_time,
            "fold": self.fold,
            "fold_aware": self.fold_aware,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExpressionSimplifier":
        """Reconstruct the augmenter from the output of ``to_dict``.

        Args:
            data: A dictionary of format 1 or 2 (``format_version``; a dictionary without it is
                read as format 1). Format 1 (before the rules and limits were saved) gets the
                default rules and limits; a dictionary without ``fold`` or ``fold_aware`` gets
                ``None`` for them.

        Returns:
            The augmenter.

        Raises:
            ValueError: If ``format_version`` is neither 1 nor 2.
            ImportError: If the class of the saved rules cannot be imported (see
                [SimplificationRules.from_dict][SRToolkit.utils.expression_simplifier.SimplificationRules.from_dict]).
        """
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError(
                f"[ExpressionSimplifier.from_dict] Unsupported format_version: {data.get('format_version')!r}. "
                "Expected 1 or 2."
            )
        return ExpressionSimplifier(
            symbol_library=SymbolLibrary.from_dict(data["symbol_library"]),
            scope=data["scope"],
            verbose=data["verbose"],
            name=data["name"],
            rules=SimplificationRules.from_dict(data["rules"]) if "rules" in data else None,
            max_iterations=data.get("max_iterations"),
            max_nodes=data.get("max_nodes"),
            max_time=data.get("max_time"),
            fold=data.get("fold"),
            fold_aware=data.get("fold_aware"),
        )


class HoldoutMetric(ResultAugmenter):
    def __init__(
        self,
        data: "DataSplit | tuple[np.ndarray, ...]",
        metric: str
        | Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float]
        | RankingFunction = "rmse",
        scope: str = "top",
        ground_truth: list[str] | Expression | None = None,
        background_knowledge: "BackgroundKnowledge | None" = None,
        settings: Settings | None = None,
        seed: int | None = 0,
        refit_constants: bool = False,
        name: str = "HoldoutMetric",
    ) -> None:
        """Scores a result's models on held-out data.

        Use it to measure how the expressions found by a search perform on data the search never
        saw. Apply it with [SR_results.augment][SRToolkit.evaluation.results.SR_results.augment];
        it adds the score of the best model to the
        [EvalResult][SRToolkit.utils.types.EvalResult] and, depending on ``scope``, to each model.

        The kind of score depends on ``metric``:

        - **A built-in metric name or a callable** (``"rmse"``, ``"r2"``, ``"nrmse"``, ``"mae"``,
          ``"mse"``, or ``metric(predict, X, y) -> float``): each model is evaluated on the
          held-out ``X`` with the constants fitted during the search, and compared with the
          held-out ``y``. This measures how well the fitted model generalizes. Models without
          fitted constants (e.g. from a BED search) have them fitted on the held-out data first.
          With ``refit_constants=True``, every model's constants are fitted on the held-out data,
          which measures how well the expression's form fits new data instead; these values are
          marked as refit. Requires ``y``. Fits use each model's specs
          (``ModelResult.free_constants``).
        - **A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction]**: the ranking is
          set up on the held-out data and scores each model, with its specs, e.g. BED on the held-out
          ``X``.

        Scoring follows ``background_knowledge`` and ``settings``, not the run's: in fits and
        ranking scores, constants without a spec, and a spec's unset ``prior`` or ``bounds``, use
        the background knowledge's ``default_constant``, not
        [EvalResult.default_constant][SRToolkit.utils.types.EvalResult]. To score under the
        search's task and settings, pass the ``background_knowledge`` and ``settings`` of the
        search's evaluator (e.g. from
        [create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator], which adds
        the dataset's ``ranking_seed``).

        With every kind of metric, models the search could not score (any
        [status][SRToolkit.utils.types.Status] other than ``"ok"``, e.g. rejected by a required
        constraint) get ``NaN``; they are not scored on the held-out data.

        Args:
            data: The held-out data: a [DataSplit][SRToolkit.dataset.data_split.DataSplit]
                (e.g. ``dataset.test``) or an ``(X[, y])`` tuple.
            metric: A built-in metric name, a callable, or a ``RankingFunction`` (see above).
                Built-in names and callables are computed against the held-out ``y``, so ``data``
                must include it. Measures that need only the expression (e.g. complexity) belong in
                a ``RankingFunction`` or a separate augmenter. Default ``"rmse"``.
            scope: Which models to score: ``"best"``, ``"top"`` (the result's top models), or
                ``"all"``. Default ``"top"``.
            ground_truth: Ground-truth expression, as a token list or an
                [Expression][SRToolkit.utils.expression.Expression] with its specs, read in the
                symbol library of ``background_knowledge`` and passed to a ``RankingFunction``
                metric's setup (e.g. for BED). Unused otherwise.
            background_knowledge: The task. Expressions are compiled with its ``symbol_library``
                and fitted with its ``default_constant``; a ``RankingFunction`` metric's setup gets
                all of it (e.g. BED's ``samplers``). ``None``: a
                [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge]
                with only the active symbol library.
            settings: [Settings][SRToolkit.utils.settings.Settings] for fitting constants on the
                held-out data (``estimation``, ``minimize``, ``evaluator.backend``) and for a
                ``RankingFunction`` metric's setup. ``None``: the defaults.
            seed: Seed for the ranking setup (e.g. BED point sampling) and the constant fitting.
                The ``evaluator.ranking_seed`` setting overrides it for a ranking metric, but not
                for the refit of a string or callable metric. Default ``0``.
            refit_constants: Fit every model's constants on the held-out data instead of reusing
                those from the search. Only for built-in or callable metrics. Default ``False``.
            name: Key under which the values are stored in ``augmentations``. Default
                ``"HoldoutMetric"``.

        Raises:
            ValueError: If ``scope`` is invalid, if a built-in or callable ``metric`` is given
                held-out data without ``y``, or if ``refit_constants=True`` is combined with a
                ``RankingFunction`` metric.
            TypeError: If ``data`` is neither a ``DataSplit`` nor an ``(X[, y])`` tuple,
                ``metric`` is not a string, callable, or ``RankingFunction``, ``settings`` is not a
                ``Settings`` or ``background_knowledge`` is not a ``BackgroundKnowledge``, or a
                ``RankingFunction`` metric gets a ``ground_truth`` that is neither a list of token
                strings nor an ``Expression``.

        Examples:
            >>> from SRToolkit.evaluation import SR_evaluator
            >>> X = np.array([[1.0, 2.0], [8.0, 4.0], [5.0, 4.0], [7.0, 9.0]])
            >>> se = SR_evaluator(X, 2 * X[:, 1] - X[:, 0], seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> X_test = np.array([[2.0, 3.0], [4.0, 1.0], [0.0, 5.0]])
            >>> results.augment(HoldoutMetric((X_test, 2 * X_test[:, 1] - X_test[:, 0]), metric="rmse"))
            >>> bool(results[0].augmentations["HoldoutMetric"]["best_value"] < 1e-6)
            True
        """
        super().__init__(name)

        if scope not in ["best", "top", "all"]:
            raise ValueError(f"[HoldoutMetric] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope
        # Imported here: SRToolkit.dataset imports this module.
        from SRToolkit.dataset.background_knowledge import BackgroundKnowledge

        if background_knowledge is None:
            background_knowledge = BackgroundKnowledge(SymbolLibrary.get_or_default())
        if not isinstance(background_knowledge, BackgroundKnowledge):
            raise TypeError(
                "[HoldoutMetric] background_knowledge must be a BackgroundKnowledge, got "
                f"{type(background_knowledge).__name__}."
            )
        if settings is None:
            settings = Settings()
        if not isinstance(settings, Settings):
            raise TypeError(f"[HoldoutMetric] settings must be a Settings, got {type(settings).__name__}.")
        self.background_knowledge = background_knowledge
        self.symbol_library = background_knowledge.symbol_library
        self.settings = settings
        self.metric = metric
        self.seed = seed
        self.refit_constants = refit_constants
        self._estimator: ParameterEstimator | None = None

        if isinstance(data, tuple):
            self._X = data[0]
            self._y = data[1] if len(data) > 1 else None
        elif hasattr(data, "arrays"):  # DataSplit (duck-typed to avoid an import cycle)
            self._X, self._y = data.X, data.y
        else:
            raise TypeError(
                f"[HoldoutMetric] data must be a DataSplit or an (X[, y]) tuple, got {type(data).__name__}."
            )

        if isinstance(metric, RankingFunction):
            if refit_constants:
                raise ValueError(
                    "[HoldoutMetric] refit_constants applies to string/callable metrics only; a "
                    "RankingFunction metric is set up on the data provided and refits constants there "
                    "if it uses any (e.g. RMSE); it never uses the constants from the search."
                )
            self._ranking_mode = True
            self._metric_label = metric.name
            # Deep-copy so setup() mutates our own instance, not the caller's (which may be shared
            # with an SR_evaluator or another HoldoutMetric).
            self.metric = copy.deepcopy(metric)
            # As in SR_evaluator, a ranking_seed replaces the seed for the ranking.
            ranking_seed = self.settings.evaluator.ranking_seed
            self.metric.setup(
                self._X,
                self._y,
                None if ground_truth is None else _to_expression(ground_truth, self.symbol_library, "ground_truth"),
                background_knowledge,
                seed if ranking_seed is None else ranking_seed,
                self.settings,
            )
        elif isinstance(metric, str):
            self._ranking_mode = False
            self._metric_label = metric
        elif callable(metric):
            self._ranking_mode = False
            self._metric_label = getattr(metric, "__name__", "custom")
        else:
            raise TypeError(
                f"[HoldoutMetric] metric must be a string, a callable, or a RankingFunction, got {type(metric).__name__}."
            )

        # A string/callable metric is computed against held-out targets, and missing model
        # parameters are refit against them too — both need y. Fail fast rather than silently
        # producing NaN for every model.
        if not self._ranking_mode and self._y is None:
            raise ValueError(
                f"[HoldoutMetric] metric {self._metric_label!r} requires held-out targets y, but the "
                "data provided none (a single-element (X,) tuple or a DataSplit without a 'y' array)."
            )

    def _value(self, model: ModelResult) -> float:
        # A model the search could not score (invalid, rejected e.g. for max_constants, ...) gets
        # NaN with every metric; scoring it here would make a rejected expression look like an answer.
        if model.status != "ok":
            return float("nan")
        # NaN on failure, as evaluate_metric does: an expression that cannot be scored, parsed or
        # fitted must not abort augmentation of the remaining models.
        try:
            expr = Expression(model.expr, self.symbol_library, model.free_constants)
            if isinstance(self.metric, RankingFunction):  # same as self._ranking_mode; narrows the type
                return float(self.metric.score(expr).score)
            # Reuse the model's fitted parameters when it has them (true generalization; an empty
            # array is a valid fit of a constant-free expression). When it has none (results from
            # a ranking that fits no constants, such as BED) or refit_constants is set, fit the
            # free constants on the held-out data.
            # __init__ rejects string/callable metrics without y, so it is set in this mode.
            assert self._y is not None
            parameters = None if self.refit_constants else model.parameters
            if parameters is None and expr.num_parameters == 0:
                parameters = np.array([])  # nothing to fit
            elif parameters is None:
                fit_error, parameters = self._fit_parameters(expr)
                # No evaluation of the fit was finite: the expression is undefined on the held-out data.
                if not np.isfinite(fit_error):
                    return float("nan")
            return evaluate_metric(
                expr, parameters, self._X, self._y, self.metric, self.symbol_library, self.settings.evaluator.backend
            )
        except Exception as e:
            warnings.warn(f"[HoldoutMetric] Error scoring expression '{''.join(model.expr)}': {e}")
            return float("nan")

    def _fit_parameters(self, expr: Expression) -> tuple[float, np.ndarray]:
        """Return ``(error, parameters)`` of a fit of ``expr`` under its specs on the held-out data."""
        # __init__ rejects string/callable metrics without y, so it is set in this mode.
        assert self._y is not None
        if self._estimator is None:
            self._estimator = ParameterEstimator(
                self._X,
                self._y,
                self.symbol_library,
                self.seed,
                self.settings,
                self.background_knowledge.default_constant,
            )
        return self._estimator.estimate_parameters(expr)

    def write_results(self, results: EvalResult) -> None:
        """Write the held-out metric into *results* and its models.

        Stores ``{"best_value", "metric", "refit_constants"}`` on the
        [EvalResult][SRToolkit.utils.types.EvalResult] and ``{"value", "metric", "refit_constants"}``
        on each scoped model.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.

        Raises:
            ValueError: If ``scope`` is ``"all"`` and *results* does not keep all its models (it
                has a [summary][SRToolkit.utils.types.EvalResult]); nothing is added then.
        """
        if self.scope == "all" and results.summary is not None:
            raise ValueError(
                f"[{type(self).__name__}] scope='all' needs all models, but this result keeps only its top models "
                "(keep_all_models=False)."
            )
        # The best model is also in the scoped models, so score each expression once: rescoring would
        # cost a fit each time and, with a refit, could give a different value.
        values: dict[tuple[tuple[str, ...], str], float] = {}

        def value(model: ModelResult) -> float:
            key = (tuple(model.expr), _spec_key(model.free_constants))
            if key not in values:
                values[key] = self._value(model)
            return values[key]

        best = results.top_models(1)[0]
        results.add_augmentation(
            self.name,
            {"best_value": value(best), "metric": self._metric_label, "refit_constants": self.refit_constants},
            self._type,
        )

        scoped = results.all_models if self.scope == "all" else results.top_models() if self.scope == "top" else []
        for model in scoped:
            model.add_augmentation(
                self.name,
                {"value": value(model), "metric": self._metric_label, "refit_constants": self.refit_constants},
                self._type,
            )

    @classmethod
    def format_eval_result(cls, data: dict[str, Any]) -> str:
        """Format experiment-level holdout data for display.

        Args:
            data: The augmentation stored on the
                [EvalResult][SRToolkit.utils.types.EvalResult] (``best_value``, ``metric``,
                ``refit_constants``).

        Returns:
            A display line, or an empty string if there is no value.
        """
        val = data.get("best_value", "")
        if val == "":
            return ""
        refit = " (refit)" if data.get("refit_constants") else ""
        return f"Holdout {data.get('metric', 'metric')}{refit} (best): {val}"

    @classmethod
    def format_model_result(cls, data: dict[str, Any]) -> str:
        """Format per-model holdout data for display.

        Args:
            data: The augmentation stored on a [ModelResult][SRToolkit.utils.types.ModelResult]
                (``value``, ``metric``, ``refit_constants``).

        Returns:
            A display fragment, or an empty string if there is no value.
        """
        val = data.get("value", "")
        if val == "":
            return ""
        refit = " (refit)" if data.get("refit_constants") else ""
        return f"holdout {data.get('metric', '')}{refit}={val}"

    def to_dict(self, base_path: str, name: str) -> dict:
        """Serialize this augmenter and write its held-out arrays to ``base_path``.

        Supported only with a built-in (string) metric.

        Args:
            base_path: Directory to write the held-out ``.npz`` into.
            name: Identifier used for the data filename.

        Returns:
            A dictionary sufficient to recreate the augmenter via
            [from_dict][SRToolkit.evaluation.result_augmentation.HoldoutMetric.from_dict]. The
            settings are saved with [Settings.to_dict][SRToolkit.utils.settings.Settings.to_dict];
            of the background knowledge only ``symbol_library`` and ``default_constant`` are kept,
            the parts a built-in metric uses.

        Raises:
            NotImplementedError: If ``metric`` is a callable or a ``RankingFunction`` (not
                serialisable — reconstruct such augmenters manually).
            TypeError: If a field of a custom settings section cannot be saved.
        """
        if not isinstance(self.metric, str):
            raise NotImplementedError(
                "[HoldoutMetric.to_dict] Only a built-in (string) metric is serialisable; "
                "reconstruct callable/RankingFunction holdout metrics manually."
            )
        import os

        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        # Absolute, so loading does not depend on the working directory at save time.
        data_path = os.path.abspath(os.path.join(base_path, f"{name}_holdout.npz"))
        np.savez(data_path, X=self._X, y=cast(np.ndarray, self._y))  # a string metric always has held-out y

        return {
            "format_version": 1,
            "type": "HoldoutMetric",
            "name": self.name,
            "scope": self.scope,
            "metric": self.metric,
            "data": data_path,
            "symbol_library": self.symbol_library.to_dict(),
            "seed": self.seed,
            "refit_constants": self.refit_constants,
            "settings": self.settings.to_dict(),
            "default_constant": cast(Constant, self.background_knowledge.default_constant).to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "HoldoutMetric":
        """Recreate a ``HoldoutMetric`` from its saved form.

        Args:
            data: A dictionary produced by
                [to_dict][SRToolkit.evaluation.result_augmentation.HoldoutMetric.to_dict].

        Returns:
            The reconstructed augmenter.

        Raises:
            ValueError: If ``data["format_version"]`` is not ``1``.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[HoldoutMetric.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        arrays = np.load(data["data"])
        # Imported here: SRToolkit.dataset imports this module.
        from SRToolkit.dataset.background_knowledge import BackgroundKnowledge

        background_knowledge = BackgroundKnowledge(
            SymbolLibrary.from_dict(data["symbol_library"]),
            default_constant=Constant.from_dict(data["default_constant"]),
        )
        return HoldoutMetric(
            (arrays["X"], arrays["y"]),
            metric=data["metric"],
            scope=data["scope"],
            background_knowledge=background_knowledge,
            settings=Settings.from_dict(data["settings"]),
            seed=data.get("seed", 0),
            refit_constants=data.get("refit_constants", False),
            name=data["name"],
        )
