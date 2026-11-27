"""Pluggable ranking functions for [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].

A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] turns an expression into a
single lower-is-better scalar. Built-ins are [RMSERanking][SRToolkit.evaluation.ranking.RMSERanking]
(alias ``"rmse"``) and [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking] (alias ``"bed"``);
custom rankings are referenced by their fully-qualified import path.

A ranking is configured through the run's [Settings][SRToolkit.utils.settings.Settings]: it reads
its own section (``rmse``, ``bed``, or a custom
[SettingsSection][SRToolkit.utils.settings.SettingsSection]) and shared ones such as
``evaluator.backend``. Facts about the task (the default spec of constants, the input samplers)
reach it through the evaluator's
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge].
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from SRToolkit.bundle._relocate import _auto_bind, _import_class
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.free_constants import Constant, _ground_truth_parameters
from SRToolkit.utils.measures import bed, create_behavior_matrix
from SRToolkit.utils.settings import Settings

if TYPE_CHECKING:
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge


@dataclass
class RankingResult:
    """The outcome of scoring a single expression with a
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
        score: The lower-is-better scalar. It becomes the model's error when the status is
            ``"ok"``, and so drives ordering, the ``success_threshold`` comparison and
            best-tracking; with any other status the error is ``+inf`` and the score is ignored.
        parameters: Fitted values (RMSE), one per parameter in the order of
            [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels], fixed values
            included. ``None`` when the ranking fits nothing (BED). This is threaded into
            [ModelResult.parameters][SRToolkit.utils.types.ModelResult] and reused for held-out
            scoring.
        extra: Freeform, ranking-owned payload for anything beyond ``parameters``
            (diagnostics, auxiliary fitted arrays, …). Threaded into
            [ModelResult.ranking_data][SRToolkit.utils.types.ModelResult]. Its values must be
            ones that can be saved (see [serialization][SRToolkit.utils.serialization]);
            otherwise the evaluator stores the model with status ``"error"``, error ``inf``,
            empty ``ranking_data`` and ``parameters`` ``None``. A ranking that wants *typed*
            richness may instead subclass ``RankingResult``; the evaluator reads only
            ``score``, ``parameters``, ``status`` and ``status_detail``.
        status: Set only when the ranking knows that it could not score the expression:
            ``"rejected"`` (the expression breaks a rule the ranking enforces) or ``"undefined"``
            (it has no finite score). Leave it ``None`` and the evaluator decides: a finite
            ``score`` is ``"ok"``, anything else ``"undefined"``. ``"invalid"`` and ``"error"``
            cannot be set here: the evaluator marks unparseable expressions ``"invalid"`` before
            ``score`` is called, and a ranking reports an ``"error"`` by raising an exception.
            See [Status][SRToolkit.utils.types.Status].
        status_detail: The reason, stored as
            [ModelResult.status_detail][SRToolkit.utils.types.ModelResult]. Optional; without it
            the evaluator writes a generic one (e.g. ``"rmse score is nan"``).
    """

    score: float
    parameters: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    status: Literal["rejected", "undefined"] | None = None
    status_detail: str | None = None


class RankingFunction(ABC):
    """Abstract single-objective ranking strategy for
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
        y: np.ndarray | None,
        ground_truth: list[str] | Node | None,
        ground_truth_constants: Sequence[Constant] | None,
        background_knowledge: "BackgroundKnowledge",
        seed: int | None,
        settings: Settings,
    ) -> None:
        """Validate required inputs and precompute all ranking-specific state.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values; required when ``needs_targets``.
            ground_truth: Target expression; required when ``needs_ground_truth``.
            ground_truth_constants: Specs for the labelled constants of ``ground_truth`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]), or ``None``.
            background_knowledge: The task, e.g. its ``symbol_library``, ``default_constant``
                (the default spec of free constants, already complete) and ``samplers`` (the input distribution, or ``None``).
            seed: Seed for the ranking's randomness. The evaluator passes the
                ``evaluator.ranking_seed`` setting when set, otherwise its own seed.
            settings: The run's [Settings][SRToolkit.utils.settings.Settings]. A ranking reads its
                own section (a custom one with ``settings.get(MySection)``, which gives the
                section's defaults when the run did not set it) and shared ones such as
                ``settings.evaluator.backend``.
        """

    @abstractmethod
    def score(self, expr: list[str] | Node, free_constants: Sequence[Constant] | None = None) -> RankingResult:
        """Score a single expression. Lower is better.

        Args:
            expr: Expression as a token list (infix) or a
                [Node][SRToolkit.utils.expression_tree.Node] tree.
            free_constants: Specs for labelled constants of ``expr`` (see
                [Constant][SRToolkit.utils.free_constants.Constant]). The evaluator passes a
                non-empty tuple sorted by label, checked against ``expr`` beforehand, or ``None``.
                Parameters without a spec, and a spec's unset ``prior`` or ``bounds``, use the
                background knowledge's ``default_constant``.

        Returns:
            A [RankingResult][SRToolkit.evaluation.ranking.RankingResult] whose ``score`` drives
            ordering. The evaluator has already parsed ``expr`` and run the pre-checks of its
            required constraints. May raise; the evaluator then records the status ``"error"``
            with the error ``+inf``.
        """

    def default_success_threshold(self) -> float | None:
        """The success threshold to use when the evaluator was not given an explicit one.

        Returns:
            A scalar threshold, or ``None`` for no default. Called only when the user passed no
            ``success_threshold``.
        """
        return None

    def describe(self, success_threshold: float | None) -> str:
        """Human-readable description for [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset]
        ``__str__``. Each subclass describes itself correctly.

        Args:
            success_threshold: The active success threshold, for inclusion in the text.

        Returns:
            A description string.
        """
        return f"Expressions are ranked based on the {self.name.upper()} ranking function."

    def to_dict(self) -> dict:
        """JSON-safe config: the ``ranking_class`` import path (plus any scalar params a subclass
        adds). No arrays.

        Returns:
            A dictionary sufficient to reconstruct the ranking via
            [from_dict][SRToolkit.evaluation.ranking.RankingFunction.from_dict].
        """
        return {"ranking_class": f"{type(self).__module__}.{type(self).__qualname__}"}

    @classmethod
    def from_dict(cls, data: dict) -> "RankingFunction":
        """Reconstruct a ranking from its [to_dict][SRToolkit.evaluation.ranking.RankingFunction.to_dict]
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
    """Ranks expressions by the root-mean-squared error of the best constant fit.

    Owns a [ParameterEstimator][SRToolkit.evaluation.parameter_estimator.ParameterEstimator]
    (built in [setup][SRToolkit.evaluation.ranking.RankingFunction.setup]); ``score`` returns the
    RMSE together with the fitted ``parameters``, and in ``extra`` (stored as
    [ModelResult.ranking_data][SRToolkit.utils.types.ModelResult]) how the fit went: whether the
    optimizer converged, how many RMSE evaluations and iterations it took, and its message when it
    did not converge. ``extra`` is empty when no constants were fitted. See
    [estimate_parameters_with_info][SRToolkit.evaluation.parameter_estimator.ParameterEstimator.estimate_parameters_with_info].

    Constants are fitted with the ``estimation`` settings. When ``y`` is not provided, targets are
    generated by evaluating ``ground_truth`` over ``X``, with the fixed values of
    ``ground_truth_constants`` for its free constants.

    The random starting points of the fits are seeded by the ``evaluator.ranking_seed`` setting
    when set, otherwise by the evaluator's seed, combined with each expression's tokens and specs;
    with neither seed they are random.
    [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator] sets
    ``ranking_seed`` (when the settings do not) to the dataset's seed, so every run on a dataset
    is scored by the same objective, unless the dataset sets
    [ranking_resample][SRToolkit.dataset.sr_dataset.SR_dataset].
    """

    name = "rmse"
    needs_targets = True
    needs_ground_truth = False

    def setup(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        ground_truth: list[str] | Node | None,
        ground_truth_constants: Sequence[Constant] | None,
        background_knowledge: "BackgroundKnowledge",
        seed: int | None,
        settings: Settings,
    ) -> None:
        """Build the parameter estimator that fits each expression's constants to ``y``.

        Args:
            X: Input data of shape ``(n_samples, n_features)``.
            y: Target values. If ``None``, they are generated by evaluating ``ground_truth`` on
                ``X``; each of its free constants then needs a fixed value in
                ``ground_truth_constants``.
            ground_truth: Target expression; used only to generate ``y`` when it is missing.
            ground_truth_constants: Specs for the labelled constants of ``ground_truth``; used
                only to generate ``y``.
            background_knowledge: Its ``symbol_library`` and ``default_constant`` are used.
            seed: Seed for the random initial values of the constants.
            settings: The ``estimation`` section and ``evaluator.backend`` are used.

        Raises:
            ValueError: If neither ``y`` nor ``ground_truth`` is given, or ``y`` must be generated
                from ``ground_truth`` and a ``ground_truth_constants`` label is not a labelled
                constant of it or one of its free constants has no fixed value.
        """
        symbol_library = background_knowledge.symbol_library
        if y is None:
            if ground_truth is None:
                raise ValueError(
                    "RMSE ranking requires either target values y or a ground truth expression to generate them from."
                )
            parameters = _ground_truth_parameters(ground_truth, ground_truth_constants, symbol_library, "[RMSERanking]")
            y = compile_expr(ground_truth, symbol_library)(X, parameters)
        self.parameter_estimator = ParameterEstimator(
            X, y, symbol_library, seed, settings, background_knowledge.default_constant
        )

    def score(self, expr: list[str] | Node, free_constants: Sequence[Constant] | None = None) -> RankingResult:
        """Return the RMSE of the best fit of ``expr`` under its specs, with the fitted parameters."""
        error, parameters, info = self.parameter_estimator.estimate_parameters_with_info(expr, free_constants)
        return RankingResult(score=error, parameters=parameters, extra=info)

    def default_success_threshold(self) -> float | None:
        return 1e-7

    def describe(self, success_threshold: float | None) -> str:
        text = "Expressions are ranked based on the RMSE ranking function.\n"
        if success_threshold is not None:
            text += (
                f"Expressions are deemed successful if the root mean squared error is less than "
                f"{success_threshold}. However, we advise that you check the best performing "
                "expressions manually to ensure they are correct."
            )
        return text


class BEDRanking(RankingFunction):
    """Ranks expressions by their Behavior-aware Expression Dissimilarity (BED) to a target behavior.

    BED compares what two expressions do rather than how they are written. Each expression is
    evaluated at a fixed set of points ``bed_X`` under ``num_consts_sampled`` random draws of its
    free parameters, each drawn from its spec's prior, or the default spec's (see
    [score][SRToolkit.evaluation.ranking.RankingFunction.score]); fixed values are kept. The output
    distributions at each point are compared with the target behavior there. No parameters are
    fitted, so ``score`` returns only the distance and ``parameters`` is ``None``. See
    [bed][SRToolkit.utils.measures.bed]. The settings read are the ``bed`` section
    (``num_points_sampled``, ``num_consts_sampled``, ``bed_gt``) and ``evaluator.backend``.

    [setup][SRToolkit.evaluation.ranking.RankingFunction.setup] fixes the points ``bed_X`` and the
    target behavior once. The target behavior comes from the first available of:

    1. **A precomputed ``bed_gt`` matrix**, one row per row of ``X`` and ``num_consts_sampled``
       columns. If a ``ground_truth`` is also given, it is used only for the default threshold.
    2. **The ``ground_truth`` expression**, possibly with free parameters, evaluated over
       ``bed_X``; its parameters follow the specs in ``ground_truth_constants`` (fixed values,
       priors), and the default spec otherwise. With the background knowledge's ``samplers`` (the
       input distribution, one [Sampler][SRToolkit.dataset.sampling.Sampler] per variable) and
       ``num_points_sampled > 0``, the points are ``num_points_sampled`` fresh draws from them;
       otherwise they are rows of ``X``.
    3. **The observed targets ``y``**, treated as a deterministic behavior. This lets BED rank
       expressions on measured data with no known expression.

    For ``bed_gt`` and ``y``, the behavior is known only at the rows of ``X``, so the points are
    rows of ``X`` (the same rows are kept in the behavior) and ``samplers`` are not used. Rows of
    ``X`` are subsampled without replacement to ``num_points_sampled``; ``-1`` uses all of them,
    which is how to score on exactly a given set of points.

    All randomness (points, behavior matrices, threshold) follows one seed: the
    ``evaluator.ranking_seed`` setting when set, otherwise the evaluator's seed.
    [SR_dataset.create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator] sets
    ``ranking_seed`` (when the settings do not) to the dataset's seed, so every run on a dataset
    is scored by the same objective, unless the dataset sets
    [ranking_resample][SRToolkit.dataset.sr_dataset.SR_dataset].

    The default success threshold is the ground truth's own noise floor: its largest BED to
    itself over 100 resamplings of its unfixed parameters, times 1.1, and at least ``1e-7`` (so a
    ground truth without unfixed parameters gets ``1e-7``). Without a ground-truth expression
    there is nothing to resample, so there is no default threshold; set one.
    """

    name = "bed"
    needs_targets = False
    needs_ground_truth = False

    def setup(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        ground_truth: list[str] | Node | None,
        ground_truth_constants: Sequence[Constant] | None,
        background_knowledge: "BackgroundKnowledge",
        seed: int | None,
        settings: Settings,
    ) -> None:
        """Fix the evaluation points and the target behavior that expressions are compared against.

        See the class description for the order in which the target behavior (``bed_gt``,
        ``ground_truth``, ``y``) and the points (``samplers`` or rows of ``X``) are chosen.

        Args:
            X: Input data of shape ``(n_samples, n_features)``; the points are taken from its rows
                unless samplers are used.
            y: Observed targets, used as the target behavior when neither ``bed_gt`` nor
                ``ground_truth`` is given.
            ground_truth: Target expression, possibly with free parameters.
            ground_truth_constants: Specs for the labelled constants of ``ground_truth``.
            background_knowledge: Its ``symbol_library``, ``samplers`` and ``default_constant``
                are used.
            seed: Seed for all of BED's randomness.
            settings: The ``bed`` section and ``evaluator.backend`` are used.

        Raises:
            ValueError: If no target behavior can be built; ``ground_truth`` is not a token list
                or ``Node``; ``bed_gt`` does not have one row per row of ``X``; ``y`` is not 1-D
                with one value per row of ``X``; the number of ``samplers`` is not the number of
                columns of ``X`` (when they are used); or the target behavior
                comes from ``ground_truth`` and the ``ground_truth_constants`` specs do not fit it
                (see [resolve_constants][SRToolkit.utils.free_constants.resolve_constants]).
        """
        # Runtime guard: configs loaded from JSON bypass the annotation, so a string or array can get here.
        if ground_truth is not None and not isinstance(ground_truth, (list, Node)):
            raise ValueError("Ground truth must be provided as a list of tokens or a Node object.")  # pyright: ignore[reportUnreachable]

        self.ground_truth = ground_truth
        self.symbol_library = background_knowledge.symbol_library
        # All of BED's randomness (points, behaviour matrices, threshold) follows this one seed.
        self.seed = seed
        self.num_consts_sampled = settings.bed.num_consts_sampled
        self.num_points_sampled = settings.bed.num_points_sampled
        self.ground_truth_constants = ground_truth_constants
        self.default_constant = background_knowledge.default_constant
        self.backend = settings.evaluator.backend
        samplers = background_knowledge.samplers

        # The behaviour to compare against, in order: a precomputed bed_gt matrix, the ground-truth
        # expression, or the observed targets y (a deterministic behaviour).
        bed_gt = settings.bed.bed_gt
        row_behavior: np.ndarray | None = None
        if bed_gt is not None:
            if bed_gt.shape[0] != X.shape[0]:
                raise ValueError(
                    f"bed_gt has {bed_gt.shape[0]} rows but X has {X.shape[0]} points; a precomputed "
                    "behaviour matrix must be passed with the X it was computed on."
                )
            row_behavior = bed_gt
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
            # points, which neither has, so they are not used.
            indices = self._subsample_rows(X.shape[0])
            self.bed_X = X[indices, :]
            self.gt_behavior = row_behavior[indices, :]
            return

        # Ground-truth expression: points drawn from the samplers, or rows of X when there are none
        # or num_points_sampled=-1 asks for all of X; the behaviour is computed over them.
        if samplers is not None and self.num_points_sampled != -1:
            if len(samplers) != X.shape[1]:
                raise ValueError(f"Got {len(samplers)} samplers for {X.shape[1]} variables; give one per variable.")
            # One sampler per feature; each draws a column (cf. SR_dataset.resample). Samplers
            # draw from the global np.random state, so seed it here for reproducibility.
            if self.seed is not None:
                np.random.seed(self.seed)
            # Imported here: SRToolkit.dataset imports this module.
            from SRToolkit.dataset.sampling import _sample_inputs

            self.bed_X = _sample_inputs(samplers, self.num_points_sampled)
        else:
            self.bed_X = X[self._subsample_rows(X.shape[0]), :]

        # row_behavior is None only when the ground-truth branch above was taken.
        assert ground_truth is not None
        self.gt_behavior = create_behavior_matrix(
            ground_truth,
            self.bed_X,
            num_consts_sampled=self.num_consts_sampled,
            symbol_library=self.symbol_library,
            seed=self.seed,
            backend=self.backend,
            free_constants=self.ground_truth_constants,
            default_constant=self.default_constant,
        )

    def _subsample_rows(self, n_rows: int) -> np.ndarray:
        # Subsample rows without replacement: each point contributes its own output distribution,
        # so duplicate rows would add no information. -1 (or a count that meets/exceeds the data)
        # uses all rows.
        if self.num_points_sampled == -1 or self.num_points_sampled >= n_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.seed)
        return rng.choice(n_rows, size=self.num_points_sampled, replace=False)

    def score(self, expr: list[str] | Node, free_constants: Sequence[Constant] | None = None) -> RankingResult:
        """Return the BED of ``expr``, its constants sampled under its specs, to the target behavior."""
        error = bed(
            expr,
            self.gt_behavior,
            self.bed_X,
            num_consts_sampled=self.num_consts_sampled,
            symbol_library=self.symbol_library,
            seed=self.seed,
            backend=self.backend,
            free_constants1=free_constants,
            default_constant=self.default_constant,
        )
        return RankingResult(score=error)

    def default_success_threshold(self) -> float | None:
        """Return the ground truth's noise floor (see the class description), or ``None`` without one."""
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
                symbol_library=self.symbol_library,
                seed=None if s is None else int(s),
                backend=self.backend,
                free_constants1=self.ground_truth_constants,
                default_constant=self.default_constant,
            )
            for s in seeds
        ]
        # Floor: a ground truth without unfixed parameters has a noise floor of 0, which no error is below.
        return float(max(np.max(distances) * 1.1, 1e-7))

    def describe(self, success_threshold: float | None) -> str:
        text = "Expressions are ranked based on the BED ranking function.\n"
        if success_threshold is not None:
            text += (
                f"Expressions are deemed successful if the behavioral expression distance is less "
                f"than {success_threshold}. However, we advise that you check the best performing "
                "expressions manually to ensure they are correct."
            )
        return text


# Friendly short-name aliases for built-in rankings. Anything else resolves by import path.
_RANKING_ALIASES: dict[str, type] = {
    "rmse": RMSERanking,
    "bed": BEDRanking,
}


def build_ranking(spec: str | RankingFunction, fallback: str = "rmse") -> RankingFunction:
    """Resolve a ranking specification to an (un-``setup``) instance.

    Args:
        spec: A [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction] instance (returned
            as-is), a built-in alias (``"rmse"``, ``"bed"``), or a fully-qualified
            ``"module.ClassName"`` import path resolved via ``importlib``.
        fallback: Alias to fall back to (with a warning) when ``spec`` is an unknown bare name.

    Returns:
        A ranking instance, not yet ``setup``.

    Raises:
        TypeError: If ``spec`` is neither a string nor a ``RankingFunction``, or an import-path
            ``spec`` names something that is not a ``RankingFunction`` subclass.
        ImportError: If an import-path ``spec`` cannot be imported (e.g. its ``.srtk`` bundle
            is not installed).
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
        return _import_class(spec, "ranking", RankingFunction)()

    warnings.warn(f"ranking_function '{spec}' not supported. Using {fallback} instead.")
    return _RANKING_ALIASES[fallback]()
