"""
Concrete [ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] implementations
that post-process SR results with additional information.

Available augmenters: [ExpressionToLatex][SRToolkit.evaluation.result_augmentation.ExpressionToLatex],
[ExpressionSimplifier][SRToolkit.evaluation.result_augmentation.ExpressionSimplifier],
[HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric].
Custom augmenters work automatically without registration — just subclass
[ResultAugmenter][SRToolkit.evaluation.sr_evaluator.ResultAugmenter] and implement the
required methods.
"""

import copy
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from SRToolkit.evaluation.parameter_estimator import DEFAULT_ESTIMATION_SETTINGS, ParameterEstimator
from SRToolkit.evaluation.ranking import RankingFunction
from SRToolkit.evaluation.sr_evaluator import EvalResult, ModelResult, ResultAugmenter, evaluate_metric
from SRToolkit.utils import Node, SymbolLibrary, simplify, tokens_to_tree
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe

if TYPE_CHECKING:
    from SRToolkit.dataset.data_split import DataSplit


class ExpressionToLatex(ResultAugmenter):
    def __init__(
        self,
        symbol_library: SymbolLibrary,
        scope: str = "top",
        verbose: bool = False,
        name: str = "ExpressionToLatex",
    ) -> None:
        """
        Converts expressions inside the results to LaTeX strings.

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
        """
        Write LaTeX representations into *results* and its models.

        Stores ``{"best_expr_latex": ...}`` in
        [EvalResult][SRToolkit.utils.types.EvalResult] ``augmentations``.
        Also stores ``{"expr_latex": ...}`` in each model's augmentations when
        ``scope`` is ``"top"`` or ``"all"``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.
        """
        eval_data: Dict[str, Any] = {}
        try:
            eval_data["best_expr_latex"] = tokens_to_tree(results.top_models(1)[0].expr, self.symbol_library).to_latex(
                self.symbol_library
            )
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Unable to convert best expression to LaTeX: {e}")
        results.add_augmentation(self.name, eval_data, self._type)

        scoped = results.all_models if self.scope == "all" else results.top_models() if self.scope == "top" else []
        for model in scoped:
            try:
                model.add_augmentation(
                    self.name,
                    {"expr_latex": tokens_to_tree(model.expr, self.symbol_library).to_latex(self.symbol_library)},
                    self._type,
                )
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Unable to convert expression {''.join(model.expr)} to LaTeX: {e}")

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        """
        Format experiment-level LaTeX augmentation data for display.

        Args:
            data: Augmentation dict containing ``"best_expr_latex"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        latex = data.get("best_expr_latex", "")
        return f"LaTeX of the best expression: {latex}" if latex else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        """
        Format per-model LaTeX augmentation data for display.

        Args:
            data: Augmentation dict containing ``"expr_latex"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        latex = data.get("expr_latex", "")
        return f"LaTeX: {latex}" if latex else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the ExpressionToLatex augmenter.

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
        """
        Creates an instance of the ExpressionToLatex augmenter from a dictionary.

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
    ) -> None:
        """
        Algebraically simplifies expressions inside the results using SymPy.

        Args:
            symbol_library: Symbol library used by the simplifier to resolve token types.
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
        """
        super().__init__(name)
        self.symbol_library = symbol_library

        if scope not in ["best", "top", "all"]:
            raise Exception(f"[RMSE augmenter] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope

        self.verbose = verbose

    def write_results(self, results: EvalResult) -> None:
        """
        Write simplified expressions into *results* and its models.

        Stores ``{"simplified_best_expr": ...}`` in
        [EvalResult][SRToolkit.utils.types.EvalResult] ``augmentations`` if
        simplification succeeds. Also stores ``{"simplified_expr": ...}`` in each model's
        augmentations when ``scope`` is ``"top"`` or ``"all"``.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.
        """
        eval_data: Dict[str, Any] = {}
        try:
            simplified_expr = simplify(results.top_models(1)[0].expr, self.symbol_library)
            if isinstance(simplified_expr, list):
                eval_data["simplified_best_expr"] = "".join(simplified_expr)
            elif isinstance(simplified_expr, Node):
                eval_data["simplified_best_expr"] = "".join(simplified_expr.to_list(self.symbol_library))
            else:
                raise Exception(f"Simplified expression is not a list or Node: {simplified_expr}")
        except Exception as e:
            if self.verbose:
                warnings.warn(f"Unable to simplify {results.best_expr}: {e}")
        results.add_augmentation(self.name, eval_data, self._type)

        scoped = results.all_models if self.scope == "all" else results.top_models() if self.scope == "top" else []
        for model in scoped:
            model_data: Dict[str, Any] = {}
            try:
                simplified_expr = simplify(model.expr, self.symbol_library)
                if isinstance(simplified_expr, list):
                    model_data["simplified_expr"] = "".join(simplified_expr)
                elif isinstance(simplified_expr, Node):
                    model_data["simplified_expr"] = "".join(simplified_expr.to_list(self.symbol_library))
                else:
                    raise Exception(f"Simplified expression is not a list or Node: {simplified_expr}")
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Unable to simplify {''.join(model.expr)}: {e}")
            model.add_augmentation(self.name, model_data, self._type)

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        """
        Format experiment-level simplification data for display.

        Args:
            data: Augmentation dict containing ``"simplified_best_expr"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        simplified = data.get("simplified_best_expr", "")
        return f"Simplified: {simplified}" if simplified else ""

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        """
        Format per-model simplification data for display.

        Args:
            data: Augmentation dict containing ``"simplified_expr"``.

        Returns:
            A human-readable string, or empty string if no data is present.
        """
        simplified = data.get("simplified_expr", "")
        return f"Simplified: {simplified}" if simplified else ""

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the ExpressionSimplifier augmenter.

        Args:
            base_path: Unused and ignored
            name: Unused and ignored

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        return {
            "format_version": 1,
            "type": "ExpressionSimplifier",
            "name": self.name,
            "symbol_library": self.symbol_library.to_dict(),
            "scope": self.scope,
            "verbose": self.verbose,
        }

    @staticmethod
    def from_dict(data: dict) -> "ExpressionSimplifier":
        """
        Creates an instance of the ExpressionSimplifier augmenter from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the augmenter.
        Returns:
            An instance of the ExpressionSimplifier augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[ExpressionSimplifier.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        return ExpressionSimplifier(
            symbol_library=data["symbol_library"],
            scope=data["scope"],
            verbose=data["verbose"],
            name=data["name"],
        )


class HoldoutMetric(ResultAugmenter):
    def __init__(
        self,
        data: "Union[DataSplit, Tuple[np.ndarray, ...]]",
        metric: Union[
            str, Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray, np.ndarray], float], RankingFunction
        ] = "rmse",
        scope: str = "top",
        symbol_library: Optional[SymbolLibrary] = None,
        ground_truth: Optional[Union[list, Node]] = None,
        ranking_settings: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 0,
        refit_constants: bool = False,
        name: str = "HoldoutMetric",
    ) -> None:
        """
        Scores a result's models on held-out data.

        Use it to measure how the expressions found by a search perform on data the search never
        saw. Apply it with [SR_results.augment][SRToolkit.evaluation.sr_evaluator.SR_results.augment];
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
          marked as refit. Requires ``y``.
        - **A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction]**: the ranking is
          set up on the held-out data and scores each model, e.g. BED on the held-out ``X``.

        With a metric name or a callable, models the search could not score (any
        [status][SRToolkit.utils.types.Status] other than ``"ok"``) get ``NaN``.

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

        Args:
            data: The held-out data: a [DataSplit][SRToolkit.dataset.data_split.DataSplit]
                (e.g. ``dataset.test``) or an ``(X[, y])`` tuple.
            metric: A built-in metric name, a callable, or a ``RankingFunction`` (see above).
                Built-in names and callables are computed against the held-out ``y``, so ``data``
                must include it. Measures that need only the expression (e.g. complexity) belong in
                a ``RankingFunction`` or a separate augmenter. Default ``"rmse"``.
            scope: Which models to score: ``"best"``, ``"top"`` (the result's top models), or
                ``"all"``. Default ``"top"``.
            symbol_library: Symbol library used to compile expressions. Defaults to the active
                library.
            ground_truth: Ground-truth expression passed to a ``RankingFunction`` metric's setup
                (e.g. for BED). Unused otherwise.
            ranking_settings: Settings for a ``RankingFunction`` metric's setup and for fitting
                constants on the held-out data (e.g. ``constant_bounds``, ``max_constants``).
                Pass the dataset's settings (``dataset.kwargs``) to fit under the same limits as
                the search.
            seed: Seed for the ranking setup (e.g. BED point sampling) and the constant fitting.
                Default ``0``.
            refit_constants: Fit every model's constants on the held-out data instead of reusing
                those from the search. Only for built-in or callable metrics. Default ``False``.
            name: Key under which the values are stored in ``augmentations``. Default
                ``"HoldoutMetric"``.

        Raises:
            ValueError: If ``scope`` is invalid, if a built-in or callable ``metric`` is given
                held-out data without ``y``, or if ``refit_constants=True`` is combined with a
                ``RankingFunction`` metric.
            TypeError: If ``data`` is neither a ``DataSplit`` nor an ``(X[, y])`` tuple, or if
                ``metric`` is not a string, callable, or ``RankingFunction``.
        """
        super().__init__(name)

        if scope not in ["best", "top", "all"]:
            raise ValueError(f"[HoldoutMetric] Invalid scope: {scope}. Must be one of 'best', 'top', 'all'.")
        self.scope = scope
        # Resolve the library once, here, so setup() and the reuse-params path always get a concrete
        # one (the active context-manager library, or the package default).
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        self.symbol_library = symbol_library
        self.metric = metric
        self.seed = seed
        self.refit_constants = refit_constants
        self.ranking_settings: Dict[str, Any] = dict(ranking_settings or {})
        self._estimator: Optional[ParameterEstimator] = None

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
            self.metric.setup(self._X, self._y, ground_truth, symbol_library, seed, self.ranking_settings)
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
        expr = model.expr
        if isinstance(self.metric, RankingFunction):  # same as self._ranking_mode; narrows the type
            # Match the NaN-on-failure contract of the reuse-params branch (evaluate_metric): a
            # single expression that raises must not abort the whole augmentation.
            try:
                return float(self.metric.score(expr).score)
            except Exception as e:
                warnings.warn(f"[HoldoutMetric] Error scoring expression '{''.join(expr)}': {e}")
                return float("nan")
        # A model the search could not score (invalid, rejected e.g. for max_constants, ...) has no
        # usable parameters; scoring it here would make a rejected expression look like an answer.
        if model.status != "ok":
            return float("nan")
        # Reuse the model's fitted parameters when it has them (true generalization; an empty array
        # is a valid fit of a constant-free expression). When it has none — results from a ranking
        # that fits no constants, such as BED — or refit_constants is set, fit the free constants on
        # the held-out data. Same NaN-on-failure contract: an expression that cannot be parsed or
        # fitted must not abort augmentation of the remaining models.
        # __init__ rejects string/callable metrics without y, so it is set in this mode.
        assert self._y is not None
        parameters = None if self.refit_constants else model.parameters
        try:
            if parameters is None and not self._has_constants(expr):
                # Nothing to fit in a constant-free expression.
                parameters = np.array([])
            elif parameters is None:
                fit_error, parameters = self._fit_parameters(expr)
                # The fit rejected the expression (e.g. over max_constants) and returned no
                # constants; scoring with an empty array would read past it.
                if np.isnan(fit_error):
                    return float("nan")
            return evaluate_metric(expr, parameters, self._X, self._y, self.metric, self.symbol_library)
        except Exception as e:
            warnings.warn(f"[HoldoutMetric] Error scoring expression '{''.join(expr)}': {e}")
            return float("nan")

    def _has_constants(self, expr: list) -> bool:
        const_symbols = set(self.symbol_library.get_symbols_of_type("const"))
        return any(token in const_symbols for token in expr)

    def _fit_parameters(self, expr: list) -> Tuple[float, np.ndarray]:
        """Fit the free constants in *expr* against the held-out data (used when a model carries no
        fitted parameters). Returns the fit's error and the constants (an empty array for
        constant-free expressions, and with a ``NaN`` error when the fit rejects the expression)."""
        # __init__ rejects string/callable metrics without y, so it is set in this mode.
        assert self._y is not None
        if self._estimator is None:
            self._estimator = ParameterEstimator(
                self._X, self._y, self.symbol_library, seed=self.seed, **self.ranking_settings
            )
        return self._estimator.estimate_parameters(expr)

    def write_results(self, results: EvalResult) -> None:
        """
        Write the held-out metric into *results* and its models.

        Stores ``{"best_value", "metric", "refit_constants"}`` on the
        [EvalResult][SRToolkit.utils.types.EvalResult] and ``{"value", "metric", "refit_constants"}``
        on each scoped model.

        Args:
            results: The [EvalResult][SRToolkit.utils.types.EvalResult] to augment.
        """
        # The best model is also in the scoped models, so score each expression once: rescoring would
        # cost a fit each time and, with a refit, could give a different value.
        values: Dict[str, float] = {}

        def value(model: ModelResult) -> float:
            key = "".join(model.expr)
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
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        """
        Format experiment-level holdout data for display.

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
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        """
        Format per-model holdout data for display.

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
        """
        Serialize this augmenter. Supported only in reuse-params mode with a built-in (string)
        metric; the held-out arrays are written to ``base_path``.

        Args:
            base_path: Directory to write the held-out ``.npz`` into.
            name: Identifier used for the data filename.

        Returns:
            A dictionary sufficient to recreate the augmenter via
            [from_dict][SRToolkit.evaluation.result_augmentation.HoldoutMetric.from_dict].

        Raises:
            NotImplementedError: If ``metric`` is a callable or a ``RankingFunction`` (not
                serialisable — reconstruct such augmenters manually).
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
        if self._y is not None:
            np.savez(data_path, X=self._X, y=self._y)
        else:
            np.savez(data_path, X=self._X)

        return {
            "format_version": 1,
            "type": "HoldoutMetric",
            "name": self.name,
            "scope": self.scope,
            "metric": self.metric,
            "data": data_path,
            "symbol_library": self.symbol_library.to_dict() if self.symbol_library is not None else None,
            "seed": self.seed,
            "refit_constants": self.refit_constants,
            # Only the constant-fitting settings are used in this (string-metric) mode, and they are
            # all JSON-safe; ranking-only settings such as samplers are not saved.
            "ranking_settings": _to_json_safe(
                {k: v for k, v in self.ranking_settings.items() if k in DEFAULT_ESTIMATION_SETTINGS}
            ),
        }

    @staticmethod
    def from_dict(data: dict) -> "HoldoutMetric":
        """
        Recreate a [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric] from a
        dictionary produced by
        [to_dict][SRToolkit.evaluation.result_augmentation.HoldoutMetric.to_dict].

        Args:
            data: The serialized dictionary.

        Returns:
            The reconstructed augmenter.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[HoldoutMetric.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        arrays = np.load(data["data"])
        X = arrays["X"]
        y = arrays["y"] if "y" in arrays else None
        symbol_library = SymbolLibrary.from_dict(data["symbol_library"]) if data.get("symbol_library") else None
        ranking_settings = _from_json_safe(data.get("ranking_settings") or {})
        return HoldoutMetric(
            (X,) if y is None else (X, y),
            metric=data["metric"],
            scope=data["scope"],
            symbol_library=symbol_library,
            ranking_settings=ranking_settings,
            seed=data.get("seed", 0),
            refit_constants=data.get("refit_constants", False),
            name=data["name"],
        )
