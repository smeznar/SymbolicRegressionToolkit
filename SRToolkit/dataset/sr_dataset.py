"""A single symbolic regression task: data, objective and background knowledge.

[SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] bundles three things.

## Data

The data lives in [DataSplit][SRToolkit.dataset.data_split.DataSplit]s: the train split (``X`` /
``y``, the data the search runs on), an optional held-out ``test`` split, and optional ``aux_data``
splits for the user's own measures. A split either holds arrays given directly or regenerates them
from a [DataSource][SRToolkit.dataset.data_source.DataSource]; the
[data cache][SRToolkit.dataset.data_cache] stores them between runs.

## Background knowledge and objective

What the task tells the approach besides the data (symbol library, samplers, constant bounds,
advisory constraints, reference grammar, descriptions) is given as separate arguments, from which
the dataset builds one
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge], available as
``dataset.background_knowledge``.

The objective is how expressions are scored: the ranking function, the success threshold, the
evaluation budget and the ``required_constraints``. An expression that breaks a required constraint
is rejected with error ``+inf``. The ``max_constants`` and ``max_expr_length`` arguments are turned
into [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants] (8 by default) and
[MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] at the front of
``required_constraints``. A ``MaxConstants`` / ``MaxTokens`` already in the list takes the place
of its limit (a different non-negative limit is a ``ValueError``), and a negative limit adds
nothing.
The advisory constraints of the background knowledge only record violations. The evaluator from
[create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator] gets the background
knowledge and the required constraints, and so does an approach's
[adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt]. Both constraint lists are stored as
copies set up on the symbol library, so their checks can run.

## Text

[describe][SRToolkit.dataset.sr_dataset.SR_dataset.describe] renders only the background knowledge
and the required constraints, so it can go into a prompt; an evaluator from ``create_evaluator``
gives the same text with
[SR_evaluator.describe][SRToolkit.evaluation.sr_evaluator.SR_evaluator.describe], so an approach
can read it during the search. ``str(dataset)`` adds the ranking and the answer,
[original_equation][SRToolkit.dataset.sr_dataset.SR_dataset.original_equation]: given, or derived
from ``ground_truth`` with the variable and target names.

## File format

[to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict] writes format 4: the background
knowledge under ``"background_knowledge"`` and the required constraints under
``"required_constraints"``. [from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict] also
reads formats 2 and 3 (symbol library and samplers at the top level, ``constant_bounds`` /
``max_constants`` / ``max_expr_length`` in ``kwargs``) and gives the same objective as before;
saving the result writes format 4.
"""

import io
import json
import os
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from urllib.request import urlopen

import numpy as np
from typing_extensions import Unpack

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.evaluation.callbacks import CallbackDispatcher, ExperimentEvent, SRCallbacks
from SRToolkit.evaluation.ranking import build_ranking
from SRToolkit.evaluation.sr_evaluator import (
    SR_evaluator,
    SR_results,
    _add_limit_constraints_from_kwargs,
    _as_constraint_tuple,
)
from SRToolkit.utils import Node, SymbolLibrary
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_constraints import ExpressionConstraint, _set_up_copy
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.types import RankingSettings

from . import data_cache
from .background_knowledge import BackgroundKnowledge, VariableInfo
from .data_source import DataSource, SampleSource, _check_no_free_constants
from .data_split import DataSplit
from .sampling import Sampler, _sample_inputs


def _resolve_task(
    flat: Dict[str, Any],
    max_constants: Optional[int],
    max_expr_length: Optional[int],
    required_constraints: Optional[Iterable[ExpressionConstraint]],
    kwargs: Mapping[str, Any],
    X: Optional[np.ndarray] = None,
) -> Tuple[BackgroundKnowledge, Tuple[ExpressionConstraint, ...], Dict[str, Any]]:
    """Return the background knowledge, required constraints and remaining ranking settings of a task."""
    # Background knowledge from its parts; None means not given (the field's default).
    bk = BackgroundKnowledge(**{k: v for k, v in flat.items() if v is not None})
    n_variables = len(bk.symbol_library.get_symbols_of_type("var"))
    if X is not None and X.ndim == 2 and X.shape[1] != n_variables:
        raise ValueError(f"X has {X.shape[1]} columns but the symbol library has {n_variables} variables.")
    limits = {
        key: value
        for key, value in (("max_constants", max_constants), ("max_expr_length", max_expr_length))
        if value is not None
    }
    required = _add_limit_constraints_from_kwargs(
        _as_constraint_tuple(() if required_constraints is None else required_constraints, "required_constraints"),
        limits,
    )
    return bk, tuple(_set_up_copy(c, bk.symbol_library) for c in required), dict(kwargs)


def _derive_equation(tokens: List[str], variables: Mapping[str, VariableInfo], target: Optional[VariableInfo]) -> str:
    """Return ``"<target name> = "`` plus ``tokens`` joined, variable tokens replaced by their names.

    See [SR_dataset.original_equation][SRToolkit.dataset.sr_dataset.SR_dataset.original_equation].
    """
    names = {"u-": "-", **{token: info.name for token, info in variables.items() if info.name}}
    if target is not None and target.name:
        lhs = target.name
    else:
        lhs, i = "y", 0
        while lhs in names.values():
            i += 1
            lhs = f"y_{i}"
    return f"{lhs} = " + "".join(names.get(token, token) for token in tokens)


class SR_dataset:
    """One symbolic regression task: data splits, objective and background knowledge.

    How background knowledge is given and how ``max_constants`` / ``max_expr_length`` become
    required constraints is explained in the [module docstring][SRToolkit.dataset.sr_dataset].

    Args:
        X: Inputs of shape ``(n_samples, n_variables)`` that the search runs on.
        symbol_library: Tokens expressions may use.
        ranking_function: ``"rmse"``, ``"bed"`` (see [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking])
            or the import path of a custom [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction].
        y: Targets of shape ``(n_samples,)``, or ``None``.
        max_evaluations: Evaluation budget; negative means no limit.
        ground_truth: The answer as infix tokens or a [Node][SRToolkit.utils.expression_tree.Node], or
            ``None``.
        original_equation: Human-readable answer, e.g. ``"F = m*a"``; ``None`` derives it (see
            [original_equation][SRToolkit.dataset.sr_dataset.SR_dataset.original_equation]).
        success_threshold: Error below which an expression counts as a success; ``None`` for none.
        seed: Default seed of evaluators, experiments and the ranking; ``None`` for none.
        dataset_metadata: Metadata about the dataset artefact (e.g. citation).
        dataset_name: Name of the dataset, also its key in the data cache.
        samplers: Input distribution: one [Sampler][SRToolkit.dataset.sampling.Sampler] per variable,
            as ``{token: sampler}`` or a list in the library's variable order. Used by
            [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample] and passed to the ranking.
        benchmark: Benchmark name, e.g. ``"feynman"``. Required by
            [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict].
        version: Version string, e.g. ``"1.0.0"``. Required by ``to_dict``.
        test_X: Held-out inputs the search never sees, e.g. for
            [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric].
        test_y: Held-out targets for ``test_X``.
        aux_data: Extra data for your own measures, as ``{key: {array_name: array}}`` or
            ``{key: DataSplit}``. Saved with the dataset; nothing in the toolkit reads it.
        variables: Information per variable, keyed by token (e.g. ``"X_0"``).
        target: Information about the target.
        description: Free text about the task.
        constant_bounds: ``(lower, upper)`` range free constants are fitted in; ``None``:
            ``(-5, 5)``.
        reference_grammar: A [Grammar][SRToolkit.utils.grammar.grammar.Grammar] an approach may
            start from.
        max_constants: Limit on free constants, added as ``MaxConstants``. ``None``: 8, unless
            ``required_constraints`` lists one. Negative: no limit.
        max_expr_length: Limit on tokens, added as ``MaxTokens``. ``None`` or negative: no limit.
        required_constraints: Further constraints every expression must satisfy; ``None`` for none.
        advisory_constraints: Hints about the search space: violations are recorded, the score is
            not changed; ``None`` for none.
        **kwargs: Ranking settings (see [RankingSettings][SRToolkit.utils.types.RankingSettings]),
            passed on to the evaluator.

    Raises:
        ValueError: If ``X`` has a different number of columns than the symbol library has
            variables; ``max_constants`` / ``max_expr_length`` is not an integer
            or disagrees with a listed ``MaxConstants`` / ``MaxTokens``; ``ground_truth`` is an ndarray
            (pass a BED behaviour matrix as ``bed_gt``); an ``aux_data`` key is not a valid file name;
            a required constraint does not fit the symbol library (raised by its ``setup``); or as
            for [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge].
        TypeError: If ``required_constraints`` holds something that is not an
            ``ExpressionConstraint``, or as for ``BackgroundKnowledge``.

    Attributes:
        X: Inputs the search runs on.
        y: Targets the search runs on, or ``None``.
        test: The held-out [DataSplit][SRToolkit.dataset.data_split.DataSplit], or ``None``.
        test_X: ``test.X``, or ``None``.
        test_y: ``test.y``, or ``None``.
        aux_data: Extra data as ``{key: DataSplit}``; you may add entries.
        data_source: Origin of ``X`` / ``y`` (e.g. a
            [SampleSource][SRToolkit.dataset.data_source.SampleSource]), used to regenerate them;
            ``None`` if they were given directly.
        contains_constants: Whether the symbol library has a free-constant symbol.
        kwargs: The ranking settings.
        background_knowledge: The background knowledge (read-only).
        symbol_library: ``background_knowledge.symbol_library`` (read-only).
        samplers: ``background_knowledge.samplers``: a tuple in variable order, or ``None`` (read-only).
        constant_bounds: ``background_knowledge.constant_bounds`` (read-only).
        required_constraints: The required constraints as a tuple (read-only).
        ranking_function: As given.
        max_evaluations: As given.
        success_threshold: As given.
        ground_truth: As given.
        original_equation: The given answer, or one derived from ``ground_truth``; ``None`` if
            neither (read-only).
        seed: As given.
        dataset_metadata: As given.
        dataset_name: As given.
        benchmark: As given.
        version: As given.

    Examples:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]),
        ...     ground_truth=["X_0", "+", "X_1"], success_threshold=1e-6)
        >>> evaluator = dataset.create_evaluator()
        >>> bool(evaluator.evaluate_expr(["C", "*", "X_1", "+", "X_0"]) < dataset.success_threshold)
        True
    """

    def __init__(  # type: ignore[misc]
        self,
        X: np.ndarray,
        symbol_library: SymbolLibrary,
        ranking_function: str = "rmse",
        y: Optional[np.ndarray] = None,
        max_evaluations: int = -1,
        ground_truth: Optional[Union[List[str], Node]] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        dataset_name: str = "unnamed",
        samplers: Optional[Union[List[Sampler], Mapping[str, Sampler]]] = None,
        benchmark: Optional[str] = None,
        version: Optional[str] = None,
        test_X: Optional[np.ndarray] = None,
        test_y: Optional[np.ndarray] = None,
        aux_data: Optional[Dict[str, Union[Dict[str, np.ndarray], DataSplit]]] = None,
        variables: Optional[Mapping[str, VariableInfo]] = None,
        target: Optional[VariableInfo] = None,
        description: Optional[str] = None,
        constant_bounds: Optional[Tuple[float, float]] = None,
        reference_grammar: Optional[Grammar] = None,
        max_constants: Optional[int] = None,
        max_expr_length: Optional[int] = None,
        required_constraints: Optional[Iterable[ExpressionConstraint]] = None,
        advisory_constraints: Optional[Iterable[ExpressionConstraint]] = None,
        **kwargs: Unpack[RankingSettings],
    ) -> None:
        self._background_knowledge, self._required_constraints, kwargs = _resolve_task(  # type: ignore[assignment]
            {
                "symbol_library": symbol_library,
                "samplers": samplers,
                "variables": variables,
                "target": target,
                "description": description,
                "constant_bounds": constant_bounds,
                "reference_grammar": reference_grammar,
                "constraints": advisory_constraints,
            },
            max_constants,
            max_expr_length,
            required_constraints,
            kwargs,
            X,
        )

        self.max_evaluations = max_evaluations
        self.success_threshold = success_threshold
        self.ranking_function = ranking_function
        if isinstance(ground_truth, np.ndarray):
            raise ValueError(
                "[SR_dataset] An ndarray ground_truth (a precomputed BED behaviour matrix) is no longer "
                "supported on SR_dataset; provide a symbolic ground truth (a token list or a Node). A "
                "precomputed behaviour matrix is ranking-owned state — pass it to the BED ranking instead."
            )
        self.ground_truth = ground_truth
        self._original_equation = original_equation
        self.kwargs = kwargs
        self.dataset_name = dataset_name

        # The data lives in DataSplits: the train split (X / y, the data the search runs on), an
        # optional held-out test split, and optional extra splits for the user's own measures.
        train_arrays: Dict[str, np.ndarray] = {"X": X}
        if y is not None:
            train_arrays["y"] = y
        self._train = DataSplit(train_arrays)
        self.test: Optional[DataSplit] = None
        if test_X is not None:
            test_arrays: Dict[str, np.ndarray] = {"X": test_X}
            if test_y is not None:
                test_arrays["y"] = test_y
            self.test = DataSplit(test_arrays)
        self.aux_data: Dict[str, DataSplit] = {}
        for key, value in (aux_data or {}).items():
            self._validate_aux_key(key)
            self.aux_data[key] = value if isinstance(value, DataSplit) else DataSplit(dict(value))

        # See if symbols contain a symbol for constants
        symbols_metadata = self.symbol_library.symbols.values()
        self.contains_constants = any([symbol["type"] == "const" for symbol in symbols_metadata])

        self.seed = seed
        self.dataset_metadata = dataset_metadata

        # Cache / serialisation metadata
        self.benchmark = benchmark
        self.version = version
        # Origin of the cached data (UrlSource / SampleSource / None). Not a constructor
        # argument: it is set by the factory methods (from_dict, from_samplers) and by
        # SR_benchmark.add_dataset. End users select an origin through those entry points.
        # data_source is a view over the train split's source (see the property below).
        self.data_source = None

    # ── Background knowledge and objective (read-only) ───────────────────────

    @property
    def background_knowledge(self) -> BackgroundKnowledge:
        """The background knowledge (read-only)."""
        return self._background_knowledge

    @property
    def symbol_library(self) -> SymbolLibrary:
        """Shortcut to ``background_knowledge.symbol_library`` (read-only)."""
        return self._background_knowledge.symbol_library

    @property
    def samplers(self) -> Optional[Tuple[Sampler, ...]]:
        """Shortcut to ``background_knowledge.samplers`` (read-only)."""
        return self._background_knowledge.samplers  # type: ignore[return-value]

    @property
    def constant_bounds(self) -> Tuple[float, float]:
        """Shortcut to ``background_knowledge.constant_bounds`` (read-only)."""
        return self._background_knowledge.constant_bounds

    @property
    def required_constraints(self) -> Tuple[ExpressionConstraint, ...]:
        """The required constraints (read-only)."""
        return self._required_constraints

    @property
    def original_equation(self) -> Optional[str]:
        """Human-readable answer, e.g. ``"F = m*a"`` (read-only).

        The given ``original_equation`` if it is not ``None``. Otherwise ``"<target name> = "``
        followed by the infix tokens of ``ground_truth`` (a ``Node`` is converted with the dataset's
        symbol library) joined without spaces, each variable token replaced by its name in
        ``background_knowledge.variables`` and the unary minus ``u-`` written ``-``. A variable with
        no entry or an empty name keeps its token. With no target or an empty target name the left
        side is the first of ``y``, ``y_1``, ``y_2``, … that is not a variable's name. ``None`` if no
        ``original_equation`` was given and ``ground_truth`` is ``None``.

        Examples:
            >>> ds = SR_dataset(np.zeros((1, 2)), SymbolLibrary.default_symbols(2),
            ...     ground_truth=["X_0", "*", "X_1"], variables={"X_0": VariableInfo("m")},
            ...     target=VariableInfo("F"))
            >>> ds.original_equation
            'F = m*X_1'
        """
        if self._original_equation is not None or self.ground_truth is None:
            return self._original_equation
        tokens = (
            self.ground_truth
            if isinstance(self.ground_truth, list)
            else self.ground_truth.to_list(symbol_library=self.symbol_library)
        )
        bk = self._background_knowledge
        return _derive_equation(tokens, bk.variables, bk.target)

    # ── Data accessors ───────────────────────────────────────────────────────
    # X / y / test_X / test_y are shortcuts to the splits' arrays, and data_source is the train
    # split's origin, so all `self.X` / `self.data_source` uses work on the splits directly.

    # Characters that are not allowed in file names on at least one supported OS.
    _FORBIDDEN_KEY_CHARS = '/\\:*?"<>|'

    @classmethod
    def _validate_aux_key(cls, key: str) -> None:
        """Raise ``ValueError`` if ``key`` cannot be a file name in the data cache."""
        if (
            not isinstance(key, str)
            or key in ("", ".", "..")
            or any(c in cls._FORBIDDEN_KEY_CHARS for c in key)
            or any(ord(c) < 32 for c in key)
        ):
            raise ValueError(
                f"[SR_dataset] Invalid aux data key {key!r}: aux keys become file names in the data "
                f"cache, so a key must be a non-empty string other than '.' or '..', without any of "
                f"{' '.join(cls._FORBIDDEN_KEY_CHARS)}."
            )

    def _splits(self) -> List[Tuple[str, DataSplit]]:
        """Return every split with its cache key: ``<name>``, ``<name>/test``, ``<name>/aux/<key>``."""
        splits = [(self.dataset_name, self._train)]
        if self.test is not None:
            splits.append((f"{self.dataset_name}/test", self.test))
        for key, split in self.aux_data.items():
            # aux_data is a plain dict the user may add to, so check entries here too.
            self._validate_aux_key(key)
            splits.append((f"{self.dataset_name}/aux/{key}", split))
        return splits

    @property
    def X(self) -> np.ndarray:
        """Input data the search runs on."""
        return self._train.arrays["X"]

    @X.setter
    def X(self, value: np.ndarray) -> None:
        self._train.arrays["X"] = value

    @property
    def y(self) -> Optional[np.ndarray]:
        """Target values the search runs on, or ``None`` if the dataset has no targets."""
        return self._train.arrays.get("y")

    @y.setter
    def y(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self._train.arrays.pop("y", None)
        else:
            self._train.arrays["y"] = value

    @property
    def test_X(self) -> Optional[np.ndarray]:
        """Inputs of the held-out test split, or ``None`` if there is none."""
        return self.test.X if self.test is not None else None

    @property
    def test_y(self) -> Optional[np.ndarray]:
        """Targets of the held-out test split, or ``None`` if there are none."""
        return self.test.y if self.test is not None else None

    @property
    def data_source(self) -> Optional[DataSource]:
        """Origin of the data the search runs on (``SampleSource`` / ``UrlSource`` / ``None``)."""
        return self._train.data_source

    @data_source.setter
    def data_source(self, value: Optional[DataSource]) -> None:
        self._train.data_source = value

    def evaluate_approach(
        self,
        sr_approach: SR_approach,
        num_experiments: int = 1,
        top_k: int = 20,
        initial_seed: Optional[int] = None,
        results: Optional[SR_results] = None,
        callbacks: Optional[Union[SRCallbacks, CallbackDispatcher, List[SRCallbacks]]] = None,
        verbose: bool = True,
        adaptation_path: Optional[str] = None,
    ) -> SR_results:
        """Run an SR approach on this dataset and collect the results.

        Each experiment searches with a new evaluator from
        [create_evaluator][SRToolkit.dataset.sr_dataset.SR_dataset.create_evaluator]. Experiment ``i``
        (from 0) is seeded with ``seed + i``, where ``seed`` is ``initial_seed`` or the dataset seed; it
        is unseeded if both are ``None``. Approaches with ``adaptation_scope="once"`` adapt before the
        first experiment, those with ``"experiment"`` before each one;
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt] gets ``X``, ``background_knowledge``
        and ``required_constraints``.

        Args:
            sr_approach: The approach to run.
            num_experiments: Number of independent runs.
            top_k: Number of best expressions kept per run.
            initial_seed: Base seed; ``None`` uses the dataset seed.
            results: [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] to append to in place;
                ``None`` creates a new one.
            callbacks: An [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks], a list of them, or a
                [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher].
            verbose: Print a line as each experiment starts.
            adaptation_path: For ``adaptation_scope="once"``: load the adapted state from this file if it
                exists, otherwise adapt and save it there, creating its parent directories. ``None``
                adapts without saving.

        Returns:
            ``results`` (or a new ``SR_results``) with one entry per experiment.

        Examples:
            >>> results = dataset.evaluate_approach(my_approach, num_experiments=5)  # doctest: +SKIP
        """
        base_seed = self.seed if initial_seed is None else initial_seed

        if results is None:
            results = SR_results()

        if isinstance(callbacks, SRCallbacks):
            dispatcher = CallbackDispatcher(callbacks=[callbacks])
            callbacks = dispatcher
        elif isinstance(callbacks, list):
            if len(callbacks) == 0:
                callbacks = None
            else:
                callbacks = CallbackDispatcher(callbacks=callbacks)

        dataset_name = self.dataset_name

        for experiment in range(num_experiments):
            if verbose:
                print(f"Running experiment {experiment + 1}/{num_experiments}")
            # Experiment i gets base_seed + i, as job i of an ExperimentGrid does.
            seed = None if base_seed is None else base_seed + experiment

            event = ExperimentEvent(
                dataset_name=dataset_name,
                approach_name=sr_approach.name,
                success_threshold=self.success_threshold,
                max_evaluations=self.max_evaluations,
                seed=seed,
            )
            if callbacks is not None:
                callbacks.on_experiment_start(event)

            sr_approach.prepare()

            if sr_approach.adaptation_scope == "once" and experiment == 0:
                if adaptation_path is not None and os.path.exists(adaptation_path):
                    sr_approach.load_adapted_state(adaptation_path)
                else:
                    sr_approach.adapt(self.X, self.background_knowledge, self.required_constraints)
                    if adaptation_path is not None:
                        dir_name = os.path.dirname(adaptation_path)
                        if dir_name:
                            os.makedirs(dir_name, exist_ok=True)
                        sr_approach.save_adapted_state(adaptation_path)
            elif sr_approach.adaptation_scope == "experiment":
                sr_approach.adapt(self.X, self.background_knowledge, self.required_constraints)

            evaluator = self.create_evaluator(seed=seed)
            evaluator._experiment_id = f"{dataset_name}_{sr_approach.name}_{seed}"
            evaluator.register_callbacks(callbacks)
            start_time = time.monotonic()
            sr_approach.search(evaluator, seed)
            elapsed = time.monotonic() - start_time
            experiment_results = evaluator.get_results(sr_approach.name, top_k)
            experiment_results.results[-1].wall_time = elapsed
            results += experiment_results

            if callbacks is not None:
                callbacks.on_experiment_end(event, results.results[-1])
        return results

    def ranking_settings(self) -> Dict[str, Any]:
        """Return the settings the ranking function gets, as in an evaluator from ``create_evaluator``.

        They are ``kwargs`` plus ``constant_bounds``, plus ``samplers`` if the dataset has them
        (BED draws its points from them when it can; other rankings ignore them). Unless ``bed_resample``
        is set, ``ranking_seed`` defaults to the dataset seed (``0`` if ``None``), so every run is
        scored by the same objective. Pass them to
        [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric] to score held-out
        data under the search's settings.

        Returns:
            A new dict of settings; the limits ``max_constants`` / ``max_expr_length`` are not in it
            (they are in ``required_constraints``).
        """
        settings: Dict[str, Any] = dict(self.kwargs)
        settings["constant_bounds"] = self.constant_bounds
        if self.samplers is not None:
            settings["samplers"] = list(self.samplers)
        # Fix the ranking's randomness per dataset, unless the user asked for it to be resampled each run.
        if not settings.get("bed_resample", False):
            settings.setdefault("ranking_seed", self.seed if self.seed is not None else 0)
        return settings

    def create_evaluator(self, metadata: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> SR_evaluator:
        """Create an [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] for this dataset.

        The evaluator gets the parts of the background knowledge (its ``constraints`` as advisory
        constraints) and ``required_constraints`` as required, adds no limits of its own, and passes
        [ranking_settings][SRToolkit.dataset.sr_dataset.SR_dataset.ranking_settings] to the ranking.
        Its ``background_knowledge`` is rebuilt from these parts, so its ``to_dict()``,
        ``describe()`` and ``priors()`` equal this dataset's.

        Args:
            metadata: Metadata to attach to the evaluator; ``dataset_name`` and ``dataset_metadata`` are
                added to it in place.
            seed: Seed of the evaluator; ``None`` uses the dataset seed.

        Returns:
            The configured evaluator.

        Raises:
            ValueError: If the ranking rejects the dataset (e.g. RMSE with neither ``y`` nor
                ``ground_truth``).
            ImportError: If an import-path ``ranking_function`` cannot be imported.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]))
            >>> float(dataset.create_evaluator().evaluate_expr(["X_1", "+", "X_0"]))  # doctest: +ELLIPSIS
            0.0...
        """
        if metadata is None:
            metadata = dict()
        if self.dataset_metadata is not None:
            metadata["dataset_metadata"] = self.dataset_metadata
        metadata["dataset_name"] = self.dataset_name

        return SR_evaluator(
            X=self.X,
            y=self.y,
            max_evaluations=self.max_evaluations,
            success_threshold=self.success_threshold,
            ranking_function=self.ranking_function,
            ground_truth=self.ground_truth,
            symbol_library=self.symbol_library,
            seed=self.seed if seed is None else seed,
            metadata=metadata,
            required_constraints=self.required_constraints,
            advisory_constraints=tuple(self.background_knowledge.constraints),
            variables=self.background_knowledge.variables,
            target=self.background_knowledge.target,
            description=self.background_knowledge.description,
            reference_grammar=self.background_knowledge.reference_grammar,
            # The limits are already in required_constraints; -1 adds none of the evaluator's own.
            max_constants=-1,
            max_expr_length=-1,
            **self.ranking_settings(),
        )

    def describe(self) -> str:
        """Return the task as text for people and LLM prompts, without the answer.

        An evaluator from ``create_evaluator`` gives the same text with
        [SR_evaluator.describe][SRToolkit.evaluation.sr_evaluator.SR_evaluator.describe].

        Returns:
            [BackgroundKnowledge.describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe]
            with ``required_constraints`` listed as enforced. Unlike ``str(dataset)``, it has no ground
            truth, equation or success threshold.
        """
        return self.background_knowledge.describe(self.required_constraints)

    def priors(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return the constraints as records, the structured form of ``describe``.

        An evaluator from ``create_evaluator`` gives the same with
        [SR_evaluator.priors][SRToolkit.evaluation.sr_evaluator.SR_evaluator.priors].

        Returns:
            ``{"required": [...], "advisory": [...]}``: the
            [summary][SRToolkit.utils.expression_constraints.ExpressionConstraint.summary] of each
            required constraint and
            [BackgroundKnowledge.priors][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.priors],
            each in order.
        """
        return {
            "required": [c.summary() for c in self.required_constraints],
            "advisory": self.background_knowledge.priors(),
        }

    def __str__(self) -> str:
        """Return a description of the dataset, including the answer.

        It gives [original_equation][SRToolkit.dataset.sr_dataset.SR_dataset.original_equation] (omitted
        when ``None``), the ranking with its success rule and non-default settings, then
        [describe][SRToolkit.dataset.sr_dataset.SR_dataset.describe] and a pointer to
        ``dataset_metadata``. Use ``describe`` for prompts.

        Returns:
            The multi-line description.
        """
        equation = self.original_equation
        text = "" if equation is None else f"Dataset for target expression {equation}.\n"
        # Each ranking describes itself (and its success rule); no setup() is needed for that.
        text += build_ranking(self.ranking_function).describe(self.success_threshold)
        if not text.endswith("\n"):
            text += "\n"
        if self.kwargs:
            settings = ", ".join(f"{key}={value}" for key, value in self.kwargs.items())
            text += f"Non-default ranking settings: {settings}.\n"
        text += self.describe() + "\n"
        text += "For other metadata, please refer to the attribute self.dataset_metadata."
        return text

    def resample(self, n: int, seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Draw new data from the samplers without changing the dataset.

        ``y`` is computed from ``ground_truth`` unless ``ranking_function`` is ``"bed"``.

        Args:
            n: Number of rows.
            seed: Seeds NumPy's global random state before the draw; ``None`` leaves it as is.

        Returns:
            ``X`` of shape ``(n, n_variables)`` if ``ranking_function`` is ``"bed"``, otherwise
            ``(X, y)`` with ``y`` of shape ``(n,)``.

        Raises:
            ValueError: If the dataset has no samplers, or ``y`` is needed and ``ground_truth`` is
                ``None`` or has a free constant.
        """
        if self.samplers is None:
            raise ValueError(
                f"[SR_dataset.resample] Dataset '{self.dataset_name}' has no samplers defined. "
                "Provide samplers when constructing the dataset."
            )
        if seed is not None:
            np.random.seed(seed)
        X = _sample_inputs(self.samplers, n)
        if self.ranking_function == "bed":
            return X
        if self.ground_truth is None or isinstance(self.ground_truth, np.ndarray):
            raise ValueError(
                f"[SR_dataset.resample] Dataset '{self.dataset_name}' has no token-list ground truth — "
                "cannot evaluate y. ground_truth must be a list of tokens or a Node."
            )

        _check_no_free_constants(
            self.ground_truth, self.symbol_library, f"[SR_dataset.resample] Dataset '{self.dataset_name}'"
        )

        f = compile_expr(self.ground_truth, self.symbol_library)
        y = f(X, np.array([]))
        return X, y

    def resample_inplace(self, n: int, seed: Optional[int] = None) -> "SR_dataset":
        """Replace ``X`` and ``y`` with data from [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample].

        ``y`` becomes ``None`` if ``ranking_function`` is ``"bed"``.

        Args:
            n: Number of rows.
            seed: As for ``resample``.

        Returns:
            ``self``.

        Raises:
            ValueError: As for ``resample``.
        """
        result = self.resample(n, seed=seed)
        if isinstance(result, tuple):
            self.X, self.y = result
        else:
            self.X, self.y = result, None
        return self

    def _persist_to_cache(self) -> None:
        """Write every split without a ``data_source`` to its file in the data cache."""
        assert self.benchmark is not None and self.version is not None, (
            "_persist_to_cache requires benchmark/version to be set; the caller must validate this."
        )

        for key, split in self._splits():
            if split.data_source is not None:
                continue
            cache_path = data_cache.dataset_path(self.benchmark, self.version, key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(cache_path), **split.arrays)  # type: ignore[arg-type]  #

    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary of this dataset's configuration (format 4).

        Arrays are not embedded but live in the [data cache][SRToolkit.dataset.data_cache]: this call
        writes every split without a ``data_source`` there, so the result reloads with
        [from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict].

        Returns:
            A dict with ``format_version`` (``4``), ``dataset_name``, ``benchmark``, ``version``,
            ``ranking_function``, ``max_evaluations``, ``success_threshold``, ``original_equation``
            (the given string, or ``None`` when derived), ``seed``, ``dataset_metadata``, ``kwargs``,
            ``ground_truth`` (token list or ``None``),
            ``data_source``, ``test``, ``aux``, ``background_knowledge``
            ([BackgroundKnowledge.to_dict][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.to_dict])
            and ``required_constraints`` (each constraint's ``to_dict``, the limit ones included).

        Raises:
            ValueError: If ``benchmark`` or ``version`` is ``None``.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]),
            ...     benchmark="test", version="1.0.0")
            >>> dataset.to_dict()["format_version"]
            4
        """
        if self.benchmark is None:
            raise ValueError("[SR_dataset.to_dict] 'benchmark' is None. Set self.benchmark before serialising.")
        if self.version is None:
            raise ValueError("[SR_dataset.to_dict] 'version' is None. Set self.version before serialising.")

        # Persist any data split whose arrays are the only copy (no data_source) into the
        # cache so the returned config is reloadable by from_dict. Source-backed splits are
        # regenerated on load, so _persist_to_cache skips them and this stays side-effect-free.
        self._persist_to_cache()

        # Serialise kwargs, converting ndarray values
        kwargs_out = {}
        for k, v in self.kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs_out[k] = v.tolist()
            else:
                kwargs_out[k] = v

        # Serialise ground truth (symbolic only — ndarray behaviour matrices are rejected at
        # construction and belong to the ranking function instead).
        if self.ground_truth is None:
            ground_truth_out = None
        elif isinstance(self.ground_truth, list):
            ground_truth_out = self.ground_truth
        elif isinstance(self.ground_truth, Node):
            ground_truth_out = self.ground_truth.to_list(symbol_library=self.symbol_library)
        else:
            raise ValueError("[SR_dataset.to_dict] Ground truth must be either a list or a Node")

        # The test and aux splits: which arrays each holds and where it comes from (the train
        # split's source is carried top-level as `data_source`).
        def split_entry(split: DataSplit) -> dict:
            return {
                "arrays": list(split.arrays.keys()),
                "data_source": split.data_source.to_dict() if split.data_source is not None else None,
            }

        return {
            "format_version": 4,
            "dataset_name": self.dataset_name,
            "benchmark": self.benchmark,
            "version": self.version,
            "ranking_function": self.ranking_function,
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "original_equation": self._original_equation,
            "seed": self.seed,
            "dataset_metadata": self.dataset_metadata,
            "kwargs": kwargs_out,
            "ground_truth": ground_truth_out,
            "data_source": self.data_source.to_dict() if self.data_source is not None else None,
            "test": split_entry(self.test) if self.test is not None else None,
            "aux": {key: split_entry(split) for key, split in self.aux_data.items()},
            "background_knowledge": self.background_knowledge.to_dict(),
            "required_constraints": [c.to_dict() for c in self.required_constraints],
        }

    @classmethod
    def from_dict(cls, d: Union[dict, str, Path]) -> "SR_dataset":
        """Load a dataset from its dictionary representation or a JSON file.

        Arrays come from the data cache, or are generated from their ``data_source``. Formats 2 and 3
        are migrated (see the [module docstring][SRToolkit.dataset.sr_dataset]). For
        ``d = ds.to_dict()``, ``from_dict(d).to_dict() == d``. Load ``.zip`` archives with
        [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive].

        Args:
            d: Output of [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict], or the path of a
                JSON file holding it.

        Returns:
            The dataset.

        Raises:
            ValueError: If ``d`` is a ``.zip`` path, ``format_version`` is not 2, 3 or 4, or the
                constructor rejects the configuration.
            TypeError: As for the constructor.
            ImportError: If a saved sampler, constraint or ranking class cannot be imported.
            FileNotFoundError: If a split's data is not in the cache and it has no ``data_source``.

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> ds = SR_dataset.from_samplers(["X_0", "+", "X_1"], [UniformSampling(0, 5), UniformSampling(0, 5)],
            ...     n_samples=3, seed=0, benchmark="test_bench", version="1.0.0")
            >>> SR_dataset.from_dict(ds.to_dict()).X.shape
            (3, 2)
        """
        if isinstance(d, (str, Path)):
            if str(d).endswith(".zip"):
                raise ValueError(
                    "[SR_dataset.from_dict] Received a '.zip' path. Load self-contained "
                    "archives with SR_dataset.from_archive(path) instead."
                )
            with open(d) as f:
                dd = json.load(f)
        else:
            dd = dict(d)

        # Apply bundle relocation if needed
        dd = _auto_bind(dd)

        # Accept the current v4 layout and the v2 / v3 configs of earlier releases. A v2 dict has no
        # "test" / "aux" entries, so only the train split loads; its cache key is the dataset name in
        # every version, and symbolic ground truth still loads from dd["ground_truth"]. (v2 datasets
        # that stored an ndarray ground truth in a separate _gt.npy are not migrated — ndarray ground
        # truth is no longer supported; provide a symbolic ground truth or rebuild such datasets.)
        fmt = dd.get("format_version")
        if fmt not in (2, 3, 4):
            raise ValueError(f"[SR_dataset.from_dict] Unsupported format_version: {fmt!r}. Expected 2, 3 or 4.")
        # Read the task part in either layout, validating it before any data is loaded or generated.
        task_args: Dict[str, Any] = dict(dd.get("kwargs") or {})
        if "background_knowledge" in dd:  # format 4
            required = tuple(ExpressionConstraint.from_dict(c) for c in dd.get("required_constraints") or ())
            bk = BackgroundKnowledge.from_dict(dd["background_knowledge"])
            task_args.update(
                symbol_library=bk.symbol_library,
                samplers=bk.samplers,
                constant_bounds=bk.constant_bounds,
                advisory_constraints=bk.constraints,
                reference_grammar=bk.reference_grammar,
                description=bk.description,
                variables=bk.variables,
                target=bk.target,
                required_constraints=required,
                # The saved list holds the effective limits; -1 adds no default to it.
                max_constants=-1,
                max_expr_length=-1,
            )
        else:  # formats 2 and 3: symbol library and samplers at the top level, the rest in kwargs
            samplers = dd.get("samplers")
            task_args.update(
                symbol_library=SymbolLibrary.from_dict(dd["symbol_library"]),
                samplers=[Sampler.from_dict(s) for s in samplers] if samplers is not None else None,
            )

        benchmark = dd["benchmark"]
        version = dd["version"]
        dataset_name = dd["dataset_name"]

        def load_split(key: str, source_dict: Optional[dict]) -> DataSplit:
            # Resolve from the cache, materialising from the split's own source if needed. The
            # config slice keeps the dataset's samplers / ground truth, which a SampleSource needs.
            path = data_cache.resolve(benchmark, version, key, {**dd, "data_source": source_dict})
            data = np.load(str(path))
            return DataSplit({k: data[k] for k in data.files}, data_source=DataSource.from_dict(source_dict))

        train = load_split(dataset_name, dd.get("data_source"))
        test_entry = dd.get("test")
        test = load_split(f"{dataset_name}/test", test_entry.get("data_source")) if test_entry else None
        aux = {}
        for key, entry in (dd.get("aux") or {}).items():
            cls._validate_aux_key(key)
            aux[key] = load_split(f"{dataset_name}/aux/{key}", entry.get("data_source"))

        dataset = cls(
            train.X,
            ranking_function=dd["ranking_function"],
            y=train.y,
            max_evaluations=dd["max_evaluations"],
            ground_truth=dd.get("ground_truth"),
            original_equation=dd["original_equation"],
            success_threshold=dd["success_threshold"],
            seed=dd["seed"],
            dataset_metadata=dd.get("dataset_metadata"),
            dataset_name=dataset_name,
            benchmark=benchmark,
            version=version,
            **task_args,
        )
        # Attach the loaded splits whole, so every array and each split's source is kept.
        dataset._train = train
        dataset.test = test
        dataset.aux_data = aux
        return dataset

    def to_archive(self, path: "Union[str, Path]") -> None:
        """Write the dataset, configuration and all data, to a self-contained ``.zip`` archive.

        The archive holds ``dataset.json`` and one ``data/<cache key>.npz`` per split, source-backed
        ones included, so loading it never regenerates or downloads anything. Like ``to_dict``, it
        writes splits without a ``data_source`` to the data cache. Load it with
        [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive] or
        [from_url][SRToolkit.dataset.sr_dataset.SR_dataset.from_url].

        Args:
            path: Destination file. A suffix other than ``.zip`` gives a warning.

        Raises:
            ValueError: If ``benchmark`` or ``version`` is ``None``.
        """
        path = Path(path)
        if path.suffix.lower() != ".zip":
            warnings.warn(
                f"[SR_dataset.to_archive] path '{path}' does not end in '.zip'. "
                "The file will be a ZIP archive regardless of the extension.",
                stacklevel=2,
            )

        # to_dict validates that benchmark/version are set.
        dataset_json = json.dumps(self.to_dict(), indent=2)

        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dataset.json", dataset_json)

            for key, split in self._splits():
                buf = io.BytesIO()
                np.savez(buf, **split.arrays)  # type: ignore[arg-type]
                zf.writestr(f"data/{key}.npz", buf.getvalue())

    @classmethod
    def from_archive(cls, path: "Union[str, Path]") -> "SR_dataset":
        """Load a dataset from a ``.zip`` archive written by [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive].

        The archive's data is extracted into the data cache, so nothing is regenerated.

        Args:
            path: Path of the archive.

        Returns:
            The dataset.
        """
        with zipfile.ZipFile(str(path), "r") as zf:
            d = json.loads(zf.read("dataset.json"))

        benchmark = d["benchmark"]
        version = d["version"]

        # Extract every data file into the cache version directory.
        data_cache.import_archive(Path(path), benchmark, version)

        # The cache is now fully populated, so no split needs to materialise from a source.
        d["data_source"] = None
        if d.get("test"):
            d["test"]["data_source"] = None
        for entry in (d.get("aux") or {}).values():
            entry["data_source"] = None
        return cls.from_dict(d)

    @classmethod
    def from_url(cls, url: str) -> "SR_dataset":
        """Download an archive written by [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive] and load it.

        Loads as [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive]. For a bare data
        file, use [UrlSource][SRToolkit.dataset.data_source.UrlSource] instead.

        Args:
            url: URL of the archive.

        Returns:
            The dataset.
        """
        with urlopen(url) as response:
            data = response.read()

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            return cls.from_archive(tmp.name)
        finally:
            os.unlink(tmp.name)

    @classmethod
    def from_samplers(  # type: ignore[misc]
        cls,
        ground_truth: Union[List[str], Node],
        samplers: Optional[Union[List[Sampler], Mapping[str, Sampler]]] = None,
        symbol_library: Optional[SymbolLibrary] = None,
        n_samples: int = 10000,
        seed: Optional[int] = None,
        n_test: Optional[int] = None,
        ranking_function: str = "rmse",
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        max_evaluations: int = -1,
        dataset_name: str = "unnamed",
        dataset_metadata: Optional[dict] = None,
        benchmark: Optional[str] = None,
        version: Optional[str] = None,
        variables: Optional[Mapping[str, VariableInfo]] = None,
        target: Optional[VariableInfo] = None,
        description: Optional[str] = None,
        constant_bounds: Optional[Tuple[float, float]] = None,
        reference_grammar: Optional[Grammar] = None,
        max_constants: Optional[int] = None,
        max_expr_length: Optional[int] = None,
        required_constraints: Optional[Iterable[ExpressionConstraint]] = None,
        advisory_constraints: Optional[Iterable[ExpressionConstraint]] = None,
        **kwargs: Unpack[RankingSettings],
    ) -> "SR_dataset":
        """Build a dataset by sampling inputs and evaluating a ground truth on them.

        ``X`` is drawn from the samplers and ``y`` is computed as in
        [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample]. The data gets a
        [SampleSource][SRToolkit.dataset.data_source.SampleSource] as ``data_source``, so ``to_dict``
        stores no arrays and [refresh][SRToolkit.dataset.sr_dataset.SR_dataset.refresh] regenerates it.

        Args:
            ground_truth: The answer as infix tokens or a [Node][SRToolkit.utils.expression_tree.Node];
                it must have no free constants unless ``ranking_function`` is ``"bed"``.
            samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per variable, as a list in
                variable order or ``{token: sampler}``.
            symbol_library: ``None`` uses
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols] with one
                variable per sampler.
            n_samples: Number of rows of ``X``.
            seed: Seed of the draw, stored on the ``SampleSource``; ``None`` for none.
            n_test: Rows of a held-out test split drawn the same way with seed ``seed + 1``, with its own
                ``SampleSource``; ``None`` for no test split.
            ranking_function: As for ``SR_dataset``; with ``"bed"`` no ``y`` is generated.
            original_equation: As for ``SR_dataset``.
            success_threshold: As for ``SR_dataset``.
            max_evaluations: As for ``SR_dataset``.
            dataset_name: As for ``SR_dataset``.
            dataset_metadata: As for ``SR_dataset``.
            benchmark: As for ``SR_dataset``.
            version: As for ``SR_dataset``.
            variables: As for ``SR_dataset``.
            target: As for ``SR_dataset``.
            description: As for ``SR_dataset``.
            constant_bounds: As for ``SR_dataset``.
            reference_grammar: As for ``SR_dataset``.
            max_constants: As for ``SR_dataset``.
            max_expr_length: As for ``SR_dataset``.
            required_constraints: As for ``SR_dataset``.
            advisory_constraints: As for ``SR_dataset``.
            **kwargs: As for ``SR_dataset``.

        Returns:
            The dataset with freshly generated data.

        Raises:
            ValueError: If ``samplers`` is ``None`` or empty, ``y`` cannot be computed (see ``resample``),
                or the constructor rejects the arguments.
            TypeError: As for the constructor.

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> ds = SR_dataset.from_samplers(["X_0", "+", "X_1"],
            ...     [UniformSampling(0, 5), UniformSampling(0, 5)], n_samples=100, seed=0)
            >>> ds.X.shape
            (100, 2)
            >>> ds.original_equation
            'y = X_0+X_1'
        """
        if not samplers:
            raise ValueError(
                "[SR_dataset.from_samplers] 'samplers' must be non-empty (one sampler per input variable)."
            )
        n_vars = len(samplers)
        if symbol_library is None:
            symbol_library = SymbolLibrary.default_symbols(n_vars)

        dataset = cls(
            np.empty((0, n_vars)),
            symbol_library,
            ranking_function=ranking_function,
            ground_truth=ground_truth,
            original_equation=original_equation,
            success_threshold=success_threshold,
            max_evaluations=max_evaluations,
            dataset_name=dataset_name,
            dataset_metadata=dataset_metadata,
            samplers=samplers,
            benchmark=benchmark,
            version=version,
            variables=variables,
            target=target,
            description=description,
            constant_bounds=constant_bounds,
            reference_grammar=reference_grammar,
            max_constants=max_constants,
            max_expr_length=max_expr_length,
            required_constraints=required_constraints,
            advisory_constraints=advisory_constraints,
            **kwargs,
        )
        dataset.data_source = SampleSource(n_samples=n_samples, seed=seed)

        # Reuse the canonical generation path (handles rmse/bed and its validation).
        dataset.resample_inplace(n_samples, seed=seed)

        # Optional held-out test split: same samplers + ground truth, its own source/seed.
        if n_test is not None:
            test_seed = None if seed is None else seed + 1
            test_result = dataset.resample(n_test, seed=test_seed)
            if isinstance(test_result, tuple):
                test_arrays: Dict[str, np.ndarray] = {"X": test_result[0], "y": test_result[1]}
            else:
                test_arrays = {"X": test_result}
            dataset.test = DataSplit(test_arrays, data_source=SampleSource(n_samples=n_test, seed=test_seed))

        return dataset

    def refresh(self) -> None:
        """Regenerate every split that has a ``data_source`` and reload it.

        The new arrays overwrite the split's files in the data cache.

        Raises:
            ValueError: If ``data_source``, ``benchmark`` or ``version`` is ``None``.
        """
        if self.data_source is None:
            raise ValueError(
                "[SR_dataset.refresh] Cannot refresh: data_source is null. "
                "Set self.data_source to a valid source config before calling refresh()."
            )
        if self.benchmark is None or self.version is None:
            raise ValueError("[SR_dataset.refresh] Cannot refresh: 'benchmark' or 'version' is None.")

        config = self.to_dict()
        for key, split in self._splits():
            if split.data_source is None:
                continue
            cache_path = data_cache.resolve(
                self.benchmark,
                self.version,
                key,
                {**config, "data_source": split.data_source.to_dict()},
                force=True,
            )
            data = np.load(str(cache_path))
            split.arrays = {k: data[k] for k in data.files}
