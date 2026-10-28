"""
Pluggable ranking functions for [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].

A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] turns an expression into a
single lower-is-better scalar. Built-ins are [RMSERanking][SRToolkit.evaluation.ranking.RMSERanking]
(alias ``"rmse"``) and [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking] (alias ``"bed"``);
custom rankings are referenced by their fully-qualified import path.
"""

import importlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.measures import bed, create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary


@dataclass
class RankingResult:
    """
    The outcome of scoring a single expression with a
    [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction].

    Examples:
        >>> r = RankingResult(score=0.5, parameters=np.array([1.0, 2.0]))
        >>> r.score
        0.5
        >>> r.parameters.tolist()
        [1.0, 2.0]
        >>> RankingResult(score=float("nan")).parameters is None
        True

    Attributes:
        score: The lower-is-better scalar. **The only field that drives ordering,
            ``success_threshold`` comparison, and best-tracking.**
        parameters: Fitted constants (RMSE). ``None`` when the ranking fits nothing (BED).
            This is threaded into [ModelResult.parameters][SRToolkit.utils.types.ModelResult]
            and reused for held-out scoring.
        extra: Freeform, ranking-owned payload for anything beyond ``parameters``
            (diagnostics, auxiliary fitted arrays, …). Threaded into
            [ModelResult.ranking_data][SRToolkit.utils.types.ModelResult]. A ranking that
            wants *typed* richness may instead subclass ``RankingResult``; the evaluator only
            ever reads ``score`` and ``parameters``.
    """

    score: float
    parameters: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class RankingFunction(ABC):
    """
    Abstract single-objective ranking strategy for
    [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].

    Subclasses own all of their data-derived state (built in
    [setup][SRToolkit.evaluation.ranking.RankingFunction.setup]) and implement
    [score][SRToolkit.evaluation.ranking.RankingFunction.score]. See
    [RMSERanking][SRToolkit.evaluation.ranking.RMSERanking] and
    [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking].

    Attributes:
        name: Registry key and serialized tag (``"rmse"``, ``"bed"``).
        needs_targets: Whether the ranking requires ``y`` (drives data-shape checks).
        needs_ground_truth: Whether the ranking requires a ``ground_truth``.
    """

    name: str = "ranking"
    needs_targets: bool = False
    needs_ground_truth: bool = False

    @abstractmethod
    def setup(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        ground_truth: Optional[Union[List[str], Node]],
        symbol_library: SymbolLibrary,
        seed: Optional[int],
        settings: Dict[str, Any],
    ) -> None:
        """
        Validate required inputs and precompute all ranking-specific state.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values; required when ``needs_targets``.
            ground_truth: Target expression; required when ``needs_ground_truth``.
            symbol_library: Token vocabulary.
            seed: Random seed for reproducible setup.
            settings: Extra settings (the evaluator's ``kwargs``); each subclass reads only the
                keys it needs.
        """

    @abstractmethod
    def score(self, expr: Union[List[str], Node]) -> RankingResult:
        """
        Score a single expression. Lower is better.

        Args:
            expr: Expression as a token list (infix) or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.

        Returns:
            A [RankingResult][SRToolkit.evaluation.ranking.RankingResult] whose ``score`` drives
            ordering. May raise — the evaluator marks the expression invalid and records ``NaN``.
        """

    def default_success_threshold(self) -> Optional[float]:
        """
        The success threshold to use when the evaluator was not given an explicit one.

        Returns:
            A scalar threshold, or ``None`` for no default. Called only when the user passed no
            ``success_threshold``.
        """
        return None

    def describe(self, success_threshold: Optional[float]) -> str:
        """
        Human-readable description for [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset]
        ``__str__``. Each subclass describes itself correctly.

        Args:
            success_threshold: The active success threshold, for inclusion in the text.

        Returns:
            A description string.
        """
        return f"Expressions are ranked based on the {self.name.upper()} ranking function."

    def to_dict(self) -> dict:
        """
        JSON-safe config: the ``ranking_class`` import path (plus any scalar params a subclass
        adds). No arrays.

        Returns:
            A dictionary sufficient to reconstruct the ranking via
            [from_dict][SRToolkit.evaluation.ranking.RankingFunction.from_dict].
        """
        return {"ranking_class": f"{type(self).__module__}.{type(self).__qualname__}"}

    @classmethod
    def from_dict(cls, data: dict) -> "RankingFunction":
        """
        Reconstruct a ranking from its [to_dict][SRToolkit.evaluation.ranking.RankingFunction.to_dict]
        form.

        Mirrors [Sampler.from_dict][SRToolkit.dataset.sampling.Sampler.from_dict] and
        [SR_approach.from_config_dict][SRToolkit.approaches.sr_approach.SR_approach.from_config_dict]:
        binds the config to any installed ``.srtk`` bundle, then resolves ``ranking_class`` (a
        fully-qualified import path or a built-in alias) via
        [build_ranking][SRToolkit.evaluation.ranking.build_ranking].

        Args:
            data: A dict with a ``"ranking_class"`` import path (or built-in alias).

        Returns:
            The reconstructed ranking instance (un-``setup``).
        """
        data = _auto_bind(data)
        return build_ranking(data["ranking_class"])


class RMSERanking(RankingFunction):
    """
    Ranks expressions by the root-mean-squared error of the best constant fit.

    Owns a [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator]
    (built in [setup][SRToolkit.evaluation.ranking.RankingFunction.setup]); ``score`` returns the
    RMSE together with the fitted ``parameters``.

    When ``y`` is not provided, targets are generated by evaluating a fully specified
    ``ground_truth`` (one with no free constants) over ``X``.
    """

    name = "rmse"
    needs_targets = True
    needs_ground_truth = False

    def setup(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        ground_truth: Optional[Union[List[str], Node]],
        symbol_library: SymbolLibrary,
        seed: Optional[int],
        settings: Dict[str, Any],
    ) -> None:
        """
        Build the parameter estimator that fits each expression's constants to ``y``.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values. If ``None``, they are generated by evaluating ``ground_truth`` on
                ``X``, which then must contain no free constants.
            ground_truth: Target expression; used only to generate ``y`` when it is missing.
            symbol_library: Token vocabulary.
            seed: Seed for the random initial values of the constants.
            settings: Estimation settings (``method``, ``tol``, ``constant_bounds``,
                ``max_constants``, …); other keys are ignored.

        Raises:
            ValueError: If neither ``y`` nor ``ground_truth`` is given, or ``y`` must be generated
                from a ground truth that contains a free constant.
        """
        if y is None:
            if ground_truth is None:
                raise ValueError(
                    "RMSE ranking requires either target values y or a ground truth expression to generate them from."
                )
            y = self._targets_from_ground_truth(X, ground_truth, symbol_library)
        self.parameter_estimator = ParameterEstimator(X, y, symbol_library=symbol_library, seed=seed, **settings)

    @staticmethod
    def _targets_from_ground_truth(
        X: np.ndarray,
        ground_truth: Union[List[str], Node],
        symbol_library: SymbolLibrary,
    ) -> np.ndarray:
        tokens = ground_truth if isinstance(ground_truth, list) else ground_truth.to_list(symbol_library)
        # A free constant ("C") has no value to evaluate against — reject rather than feed the
        # compiled backend an empty constant array (which would read out of bounds).
        const_symbols = set(symbol_library.get_symbols_of_type("const"))
        if any(tok in const_symbols for tok in tokens):
            raise ValueError(
                "Cannot generate y from a ground truth containing a free constant "
                f"({', '.join(sorted(const_symbols))}) — it has no value to evaluate. Provide y "
                "explicitly or use a fully specified ground truth."
            )
        return compile_expr(ground_truth, symbol_library)(X, np.array([]))

    def score(self, expr: Union[List[str], Node]) -> RankingResult:
        error, parameters = self.parameter_estimator.estimate_parameters(expr)
        return RankingResult(score=error, parameters=parameters)

    def default_success_threshold(self) -> Optional[float]:
        return 1e-7

    def describe(self, success_threshold: Optional[float]) -> str:
        text = "Expressions are ranked based on the RMSE ranking function.\n"
        if success_threshold is not None:
            text += (
                f"Expressions are deemed successful if the root mean squared error is less than "
                f"{success_threshold}. However, we advise that you check the best performing "
                "expressions manually to ensure they are correct."
            )
        return text


class BEDRanking(RankingFunction):
    """
    Ranks expressions by their Behavior-aware Expression Dissimilarity (BED) to a target behavior.

    BED compares what two expressions do rather than how they are written. Each expression is
    evaluated at a fixed set of points ``bed_X`` under ``num_consts_sampled`` random draws of its
    free parameters (within ``constant_bounds``), and the output distributions at each point are
    compared with the target behavior there. No parameters are fitted, so ``score`` returns only
    the distance and ``parameters`` is ``None``. See [bed][SRToolkit.utils.measures.bed].

    [setup][SRToolkit.evaluation.ranking.RankingFunction.setup] fixes the points ``bed_X`` and the
    target behavior once. The target behavior comes from the first available of:

    1. **A precomputed ``bed_gt`` matrix**, one row per row of ``X`` and ``num_consts_sampled``
       columns. If a ``ground_truth`` is also given, it is used only for the default threshold.
    2. **The ``ground_truth`` expression**, possibly with free parameters, evaluated over
       ``bed_X``. The points are drawn by ``samplers`` (one
       [Sampler][SRToolkit.dataset.sampling.Sampler] per variable, ``num_points_sampled`` fresh
       points; an [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] passes its own samplers
       here) or, without samplers, subsampled from the rows of ``X``.
    3. **The observed targets ``y``**, treated as a deterministic behavior. This lets BED rank
       expressions on measured data with no known expression.

    For ``bed_gt`` and ``y``, the behavior is known only at the rows of ``X``, so the points are
    rows of ``X`` (the same rows are kept in the behavior) and ``samplers`` are ignored, with a
    warning. Rows of
    ``X`` are subsampled without replacement to ``num_points_sampled``; ``-1`` uses all of them,
    which is how to score on exactly a given set of points.

    All randomness (points, behavior matrices, threshold) follows one seed: the ``ranking_seed``
    setting when given, otherwise the evaluator's seed.
    [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator] sets
    ``ranking_seed`` to the dataset's seed, so every run on a dataset is scored by the same
    objective, unless ``bed_resample=True``.

    The default success threshold is the ground truth's own noise floor: its largest BED to
    itself over 100 resamplings of its free parameters, times 1.1. Without a ground-truth
    expression there is nothing to resample, so there is no default threshold; set one.
    """

    name = "bed"
    needs_targets = False
    needs_ground_truth = False

    def setup(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        ground_truth: Optional[Union[List[str], Node]],
        symbol_library: SymbolLibrary,
        seed: Optional[int],
        settings: Dict[str, Any],
    ) -> None:
        """
        Fix the evaluation points and the target behavior that expressions are compared against.

        See the class description for the order in which the target behavior (``bed_gt``,
        ``ground_truth``, ``y``) and the points (``samplers`` or rows of ``X``) are chosen.

        Args:
            X: Input data of shape ``(n_samples, n_features)``; the points are taken from its rows
                unless samplers are used.
            y: Observed targets, used as the target behavior when neither ``bed_gt`` nor
                ``ground_truth`` is given.
            ground_truth: Target expression, possibly with free parameters.
            symbol_library: Token vocabulary.
            seed: Seed for all of BED's randomness, unless ``settings`` has a ``ranking_seed``.
            settings: BED settings (``num_points_sampled``, ``num_consts_sampled``,
                ``constant_bounds``, ``samplers``, ``bed_gt``, ``ranking_seed``); other keys are
                ignored.

        Raises:
            ValueError: If no target behavior can be built, or if ``ground_truth``, ``bed_gt``,
                ``y`` or ``samplers`` do not fit ``X`` (wrong type, shape or count).
        """
        # Runtime guard: configs loaded from JSON bypass the annotation, so a string or array can get here.
        if ground_truth is not None and not isinstance(ground_truth, (list, Node)):
            raise ValueError("Ground truth must be provided as a list of tokens or a Node object.")  # pyright: ignore[reportUnreachable]

        self.ground_truth = ground_truth
        self.symbol_library = symbol_library
        # All of BED's randomness (points, behaviour matrices, threshold) follows one seed: an
        # explicit ranking_seed (fixed per dataset) or else the evaluator's seed.
        ranking_seed = settings.get("ranking_seed")
        self.seed = ranking_seed if ranking_seed is not None else seed
        self.num_consts_sampled = settings.get("num_consts_sampled", 32)
        self.num_points_sampled = settings.get("num_points_sampled", 64)
        self.constant_bounds = settings.get("constant_bounds", (-5, 5))
        samplers = settings.get("samplers", None)

        # The behaviour to compare against, in order: a precomputed bed_gt matrix, the ground-truth
        # expression, or the observed targets y (a deterministic behaviour).
        bed_gt = settings.get("bed_gt", None)
        row_behavior: Optional[np.ndarray] = None
        if bed_gt is not None:
            row_behavior = np.asarray(bed_gt)
            self._validate_bed_gt(row_behavior, X)
        elif ground_truth is not None:
            pass  # behaviour is computed from the expression below
        elif y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"Targets y must be 1-D with one value per row of X ({X.shape[0]}), got shape {y.shape}."
                )
            row_behavior = np.repeat(y[:, None], self.num_consts_sampled, axis=1)
        else:
            raise ValueError(
                "The bed ranking function requires a ground truth (a list of tokens or a Node object), "
                "a precomputed bed_gt behaviour matrix, or target values y."
            )

        if row_behavior is not None:
            # A matrix or y describes behaviour at the rows of X, so the points are rows of X, and
            # any subsampling applies the same rows to both. Samplers would need behaviour at fresh
            # points, which neither has, so they are ignored.
            if samplers is not None:
                warnings.warn(
                    "A precomputed bed_gt matrix or targets y give behaviour at the rows of X only, so "
                    "samplers are ignored; the points are rows of X."
                )
            indices = self._subsample_rows(X.shape[0])
            self.bed_X = X[indices, :]
            self.gt_behavior = row_behavior[indices, :]
            return

        # Ground-truth expression: points from samplers (a fresh domain) or subsampled from X, and
        # the behaviour computed over them.
        if samplers is not None:
            if self.num_points_sampled == -1:
                raise ValueError(
                    "num_points_sampled=-1 (use all of X) is not valid together with samplers, "
                    "which generate a fresh domain — set it to a positive integer."
                )
            if len(samplers) != X.shape[1]:
                raise ValueError(f"Got {len(samplers)} samplers for {X.shape[1]} variables; give one per variable.")
            # One sampler per feature; each draws a column (cf. SR_dataset.resample). Samplers
            # draw from the global np.random state, so seed it here for reproducibility.
            if self.seed is not None:
                np.random.seed(self.seed)
            self.bed_X = np.column_stack([s(self.num_points_sampled) for s in samplers])
        else:
            self.bed_X = X[self._subsample_rows(X.shape[0]), :]

        # row_behavior is None only when the ground-truth branch above was taken.
        assert ground_truth is not None
        self.gt_behavior = create_behavior_matrix(
            ground_truth,
            self.bed_X,
            num_consts_sampled=self.num_consts_sampled,
            consts_bounds=self.constant_bounds,
            symbol_library=self.symbol_library,
            seed=self.seed,
        )

    def _validate_bed_gt(self, bed_gt: np.ndarray, X: np.ndarray) -> None:
        if bed_gt.ndim != 2:
            raise ValueError(f"bed_gt must be a 2-D behaviour matrix, got {bed_gt.ndim} dimension(s).")
        if bed_gt.shape[0] != X.shape[0]:
            raise ValueError(
                f"bed_gt has {bed_gt.shape[0]} rows but X has {X.shape[0]} points; a precomputed "
                "behaviour matrix must be passed with the X it was computed on."
            )
        # Candidates are sampled with num_consts_sampled columns, and the distance assumes both
        # matrices have the same number, so a mismatch would silently give wrong values.
        if bed_gt.shape[1] != self.num_consts_sampled:
            raise ValueError(
                f"bed_gt has {bed_gt.shape[1]} columns but num_consts_sampled is {self.num_consts_sampled}; "
                "they must match."
            )

    def _subsample_rows(self, n_rows: int) -> np.ndarray:
        # Subsample rows without replacement: each point contributes its own output distribution,
        # so duplicate rows would add no information. -1 (or a count that meets/exceeds the data)
        # uses all rows.
        if self.num_points_sampled == -1 or self.num_points_sampled >= n_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.seed)
        return rng.choice(n_rows, size=self.num_points_sampled, replace=False)

    def score(self, expr: Union[List[str], Node]) -> RankingResult:
        error = bed(
            expr,
            self.gt_behavior,
            self.bed_X,
            num_consts_sampled=self.num_consts_sampled,
            consts_bounds=self.constant_bounds,
            symbol_library=self.symbol_library,
            seed=self.seed,
        )
        return RankingResult(score=error)

    def default_success_threshold(self) -> Optional[float]:
        # Estimating the noise floor relies on resampling the ground truth's constants; with only a
        # precomputed bed_gt matrix or targets y there is nothing to resample, so no default exists.
        # (With both bed_gt and a ground truth, the matrix is scored and the expression sets this.)
        if self.ground_truth is None:
            return None
        # Each resampling needs its own constant draws; derive them from the ranking seed so the
        # threshold is reproducible whenever the seed is set.
        seeds = (
            np.random.default_rng(self.seed).integers(0, 2**31 - 1, size=100) if self.seed is not None else [None] * 100
        )
        distances = [
            bed(
                self.ground_truth,
                self.gt_behavior,
                self.bed_X,
                num_consts_sampled=self.num_consts_sampled,
                consts_bounds=self.constant_bounds,
                symbol_library=self.symbol_library,
                seed=None if s is None else int(s),
            )
            for s in seeds
        ]
        return float(np.max(distances) * 1.1)

    def describe(self, success_threshold: Optional[float]) -> str:
        text = "Expressions are ranked based on the BED ranking function.\n"
        if success_threshold is not None:
            text += (
                f"Expressions are deemed successful if the behavioral expression distance is less "
                f"than {success_threshold}. However, we advise that you check the best performing "
                "expressions manually to ensure they are correct."
            )
        return text


# Friendly short-name aliases for built-in rankings. Anything else resolves by import path.
_RANKING_ALIASES: Dict[str, type] = {
    "rmse": RMSERanking,
    "bed": BEDRanking,
}


def build_ranking(spec: Union[str, RankingFunction], fallback: str = "rmse") -> RankingFunction:
    """
    Resolve a ranking specification to an (un-``setup``) instance.

    Args:
        spec: A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] instance (returned
            as-is), a built-in alias (``"rmse"``, ``"bed"``), or a fully-qualified
            ``"module.ClassName"`` import path resolved via ``importlib``.
        fallback: Alias to fall back to (with a warning) when ``spec`` is an unknown bare name.

    Returns:
        A ranking instance, not yet ``setup``.

    Raises:
        TypeError: If ``spec`` is neither a string nor a ``RankingFunction``.
        ImportError: If an import-path ``spec`` cannot be imported (e.g. its ``.srtk`` bundle
            is not installed).
        ValueError: If an import-path ``spec`` resolves to something that is not a
            ``RankingFunction`` subclass.
    """
    if isinstance(spec, RankingFunction):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"[build_ranking] ranking_function must be a string or a RankingFunction, got {type(spec).__name__}."
        )

    key = spec.lower()
    if key in _RANKING_ALIASES:
        return _RANKING_ALIASES[key]()

    if "." in spec:
        module_path, class_name = spec.rsplit(".", 1)
        try:
            klass = getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Cannot import ranking class {spec!r}. "
                "If this is a bundle class, install the bundle first. "
                "If the config has no '_bundle' key, call bind_config(config) manually."
            ) from None
        if not (isinstance(klass, type) and issubclass(klass, RankingFunction)):
            raise ValueError(f"[build_ranking] '{spec}' is not a RankingFunction subclass.")
        return klass()

    warnings.warn(f"ranking_function '{spec}' not supported. Using {fallback} instead.")
    return _RANKING_ALIASES[fallback]()
