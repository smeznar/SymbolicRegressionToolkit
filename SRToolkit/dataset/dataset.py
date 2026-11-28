"""A single symbolic regression task: data, objective and background knowledge.

[Dataset][SRToolkit.dataset.dataset.Dataset] bundles three things.

## Data

The data lives in [DataSplit][SRToolkit.dataset.data_split.DataSplit]s: the train split (``X`` /
``y``, the data the search runs on), an optional held-out ``test`` split, and optional ``aux_data``
splits for the user's own measures. A split either holds arrays given directly or regenerates them
from a [DataSource][SRToolkit.dataset.data_source.DataSource]; the
[data cache][SRToolkit.dataset.data_cache] stores them between runs.

## Background knowledge and objective

What the task tells the approach besides the data (symbol library, samplers, constant bounds and
default spec of free constants, advisory constraints, reference grammar, descriptions) is given as
separate arguments, from which the dataset builds one
[BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge], available as
``dataset.background_knowledge``.

The objective is how expressions are scored: the ranking function, the success threshold, the
evaluation budget and the ``required_constraints``. An expression that breaks a required constraint
is rejected with error ``+inf``. The ``max_constants`` and ``max_expr_length`` arguments are turned
into [MaxConstants][SRToolkit.utils.expression_constraints.MaxConstants] (8 by default) and
[MaxTokens][SRToolkit.utils.expression_constraints.MaxTokens] at the front of
``required_constraints``. A ``MaxConstants`` / ``MaxTokens`` already in the list takes the place
of its limit (a different non-negative limit is a ``ValueError``), and a negative limit adds
nothing. The other settings of the evaluator, the ranking and parameter estimation are one
[Settings][SRToolkit.utils.settings.Settings]; the dataset's limits stay arguments.

The advisory constraints of the background knowledge only record violations. The evaluator from
[create_evaluator][SRToolkit.dataset.dataset.Dataset.create_evaluator] gets the background
knowledge and the required constraints, and so does an approach's
[adapt][SRToolkit.approaches.approach.Approach.adapt]. Both constraint lists are stored as
copies set up on the symbol library, so their checks can run.

## Text

[describe][SRToolkit.dataset.dataset.Dataset.describe] renders only the background knowledge
and the required constraints, so it can go into a prompt; an evaluator from ``create_evaluator``
gives the same text with
[Evaluator.describe][SRToolkit.evaluation.evaluator.Evaluator.describe], so an approach
can read it during the search. ``str(dataset)`` adds the ranking and the answer,
[original_equation][SRToolkit.dataset.dataset.Dataset.original_equation]: given, or derived
from ``ground_truth`` with the variable and target names.

## File format

[to_dict][SRToolkit.dataset.dataset.Dataset.to_dict] writes format 5: the background
knowledge under ``"background_knowledge"``, the required constraints under
``"required_constraints"``, the settings under ``"settings"``
([Settings.to_dict][SRToolkit.utils.settings.Settings.to_dict]), and the answer
(``ground_truth`` as tokens, its specs as ``ground_truth_constants``) and ``ranking_resample`` at
the top level.
[from_dict][SRToolkit.dataset.dataset.Dataset.from_dict] also reads formats 2–4 of
earlier releases, with the same objective; saving the result writes format 5. Their flat
``"kwargs"`` are sorted: ``constant_bounds``, ``max_constants``, ``max_expr_length`` and
``ranking_resample`` become the arguments of the same name, each built-in setting goes to its
section (e.g. ``method`` becomes ``minimize.method``), and any other key is a ``ValueError``.
A ``None`` value is the default, except ``ranking_seed: None``, which becomes
``ranking_resample=True`` (the run's seed, as before).
"""

import copy
import json
import numbers
import os
import warnings
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np

from SRToolkit.approaches.approach import Approach
from SRToolkit.bundle._relocate import _auto_bind
from SRToolkit.evaluation.callbacks import Callback, CallbackDispatcher, ExperimentEvent
from SRToolkit.evaluation.evaluator import (
    Evaluator,
    _add_limit_constraints,
    _as_constraint_tuple,
    _as_ground_truth,
)
from SRToolkit.evaluation.model_store import (
    ModelStore,
    _checkpoint_reference,
    _compact_checkpoint,
    _remove_checkpoint,
    _run_finished,
)
from SRToolkit.evaluation.ranking import build_ranking
from SRToolkit.evaluation.results import Results
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.expression import Expression
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_constraints import ExpressionConstraint, _set_up_copy
from SRToolkit.utils.free_constants import Constant, _ground_truth_parameters
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.serialization import _from_json_safe, _to_json_safe
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.types import EvalResult

from . import data_cache
from .background_knowledge import BackgroundKnowledge, VariableInfo
from .data_source import DataSource, SampleSource
from .data_split import DataSplit
from .sampling import Sampler, _sample_inputs

# Format 2-4 "kwargs": the keys that are constructor arguments, and the section of each built-in setting.
_OLD_ARGUMENTS = {"constant_bounds", "max_constants", "max_expr_length", "ranking_resample"}
_OLD_SETTING_SECTIONS = {
    **dict.fromkeys(
        (
            "simplify_expr",
            "canonicalise_constants",
            "fold_constants",
            "simplify_rules",
            "simplify_max_iterations",
            "simplify_max_nodes",
            "simplify_max_time",
            "simplify_fold_aware",
            "max_consecutive_skips",
            "backend",
            "ranking_seed",
        ),
        "evaluator",
    ),
    **dict.fromkeys(("method", "tol", "gtol", "max_iter"), "minimize"),
    "initialization": "estimation",
    **dict.fromkeys(("num_points_sampled", "num_consts_sampled", "bed_gt"), "bed"),
}


def _resolve_task(
    flat: dict[str, Any],
    max_constants: int | None,
    max_expr_length: int | None,
    required_constraints: Iterable[ExpressionConstraint] | None,
    settings: Settings | None,
    ranking_resample: bool,
    X: np.ndarray | None = None,
) -> tuple[BackgroundKnowledge, tuple[ExpressionConstraint, ...], Settings]:
    """Return the background knowledge, required constraints and settings of a task."""
    if not isinstance(ranking_resample, bool):
        raise TypeError(f"ranking_resample must be a bool, got {type(ranking_resample).__name__}.")
    if settings is None:
        settings = Settings()
    elif not isinstance(settings, Settings):
        raise TypeError(f"settings must be a Settings, got {type(settings).__name__}.")
    # A dataset's limits are its max_constants / max_expr_length arguments (saved as required constraints).
    in_settings = sorted({"max_constants", "max_expr_length"} & set(settings.evaluator.changed()))
    if in_settings:
        raise ValueError(
            f"Set the limits with the {', '.join(k + '=' for k in in_settings)} argument(s), not with "
            f"{', '.join('evaluator.' + k for k in in_settings)} in settings."
        )
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
    required = _add_limit_constraints(
        _as_constraint_tuple(() if required_constraints is None else required_constraints, "required_constraints"),
        limits,
    )
    return bk, tuple(_set_up_copy(c, bk.symbol_library) for c in required), settings


def _derive_equation(tokens: list[str], variables: Mapping[str, VariableInfo], target: VariableInfo | None) -> str:
    """Return ``"<target name> = "`` plus ``tokens`` joined, variable tokens replaced by their names.

    See [Dataset.original_equation][SRToolkit.dataset.dataset.Dataset.original_equation].
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


class Dataset:
    """One symbolic regression task: data splits, objective and background knowledge.

    How background knowledge is given and how ``max_constants`` / ``max_expr_length`` become
    required constraints is explained in the [module docstring][SRToolkit.dataset.dataset].

    Args:
        X: Inputs of shape ``(n_samples, n_variables)`` that the search runs on.
        symbol_library: Tokens expressions may use.
        ranking_function: ``"rmse"``, ``"bed"`` (see [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking])
            or the import path of a custom [RankingFunction][SRToolkit.evaluation.ranking.RankingFunction].
        y: Targets of shape ``(n_samples,)``, or ``None``.
        max_evaluations: Evaluation budget; negative means no limit. ``0`` is rejected.
        ground_truth: The answer, as infix tokens or an
            [Expression][SRToolkit.utils.expression.Expression] carrying the specs of its labelled
            constants (see [Constant][SRToolkit.utils.free_constants.Constant]); constants without
            a spec use ``default_constant``. Read in ``symbol_library`` (an ``Expression`` with
            another library is rebuilt from its tokens); it must parse. ``y`` can be computed from
            it only if all its constants are fixed; BED samples the others from their priors.
            ``None`` for none.
        original_equation: Human-readable answer, e.g. ``"F = m*a"``; ``None`` derives it (see
            [original_equation][SRToolkit.dataset.dataset.Dataset.original_equation]).
        success_threshold: Error below which an expression counts as a success; ``None`` for none.
        seed: Default seed of evaluators, experiments and the ranking; ``None`` for none.
        dataset_metadata: Metadata about the dataset artefact (e.g. citation).
        dataset_name: Name of the dataset, also its key in the data cache.
        samplers: Input distribution: one [Sampler][SRToolkit.dataset.sampling.Sampler] per variable,
            as ``{token: sampler}`` or a list in the library's variable order. Used by
            [resample][SRToolkit.dataset.dataset.Dataset.resample] and passed to the ranking.
        benchmark: Benchmark name, e.g. ``"feynman"``. Required by
            [to_dict][SRToolkit.dataset.dataset.Dataset.to_dict].
        version: Version string, e.g. ``"1.0.0"``. Required by ``to_dict``.
        test_X: Held-out inputs the search never sees, e.g. for
            [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric].
        test_y: Held-out targets for ``test_X``.
        aux_data: Extra data for your own measures, as ``{key: {array_name: array}}`` or
            ``{key: DataSplit}``. Saved with the dataset; nothing in the toolkit reads it.
        variables: Information per variable, keyed by token (e.g. ``"X_0"``).
        target: Information about the target.
        named_constants: Information per named constant, keyed by ``"lit"`` token (e.g. ``"c"``).
        description: Free text about the task.
        constant_bounds: ``(lower, upper)`` range of the default spec, as for
            [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge];
            ``None``: ``(-5, 5)``.
        default_constant: The default spec of free constants, a
            [Constant][SRToolkit.utils.free_constants.Constant] template, as for
            ``BackgroundKnowledge``; ``None``:
            [uniform_default(constant_bounds)][SRToolkit.utils.free_constants.uniform_default].
        reference_grammar: A [Grammar][SRToolkit.utils.grammar.grammar.Grammar] an approach may
            start from.
        max_constants: Limit on free constants, added as ``MaxConstants``. ``None``: 8, unless
            ``required_constraints`` lists one. Negative: no limit.
        max_expr_length: Limit on tokens, added as ``MaxTokens``. ``None`` or negative: no limit.
        required_constraints: Further constraints every expression must satisfy; ``None`` for none.
        advisory_constraints: Hints about the search space: violations are recorded, the score is
            not changed; ``None`` for none.
        settings: The [Settings][SRToolkit.utils.settings.Settings] of the evaluator, the ranking
            and parameter estimation; ``None``: ``Settings()``. It may not change
            ``evaluator.max_constants`` / ``evaluator.max_expr_length``; use the ``max_constants`` /
            ``max_expr_length`` arguments.
        ranking_resample: If ``True``, the ranking's randomness (BED points and behaviours, the
            starting point of each fit) follows the evaluator's seed (the run seed) instead of the
            dataset seed; a ``ranking_seed`` in ``settings`` still wins (see
            [create_evaluator][SRToolkit.dataset.dataset.Dataset.create_evaluator]).

    Raises:
        ValueError: If ``X`` has a different number of columns than the symbol library has
            variables; ``max_evaluations`` is ``0``; ``max_constants`` / ``max_expr_length`` is not an integer
            or disagrees with a listed ``MaxConstants`` / ``MaxTokens``; ``settings`` sets
            ``evaluator.max_constants`` or ``evaluator.max_expr_length`` to a non-default value;
            ``ground_truth`` is an ndarray (pass a BED behaviour matrix to the BED ranking) or a
            free spec of it has a ``value`` outside the bounds it takes from ``default_constant`` (see
            [Expression.resolve][SRToolkit.utils.expression.Expression.resolve]); an
            ``aux_data`` key is not a valid file name; a required constraint does not fit the symbol
            library (raised by its ``setup``); or as for
            [BackgroundKnowledge][SRToolkit.dataset.background_knowledge.BackgroundKnowledge].
        TypeError: If ``required_constraints`` holds something that is not an
            ``ExpressionConstraint``, ``settings`` is not a ``Settings``, ``ranking_resample`` is
            not a ``bool``, ``ground_truth`` is neither a list of token strings nor an
            ``Expression``, or as for ``BackgroundKnowledge``.
        ExpressionParseError: If ``ground_truth`` does not parse.

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
        settings: As given (``Settings()`` for ``None``).
        ranking_resample: As given.
        background_knowledge: The background knowledge (read-only).
        symbol_library: ``background_knowledge.symbol_library`` (read-only).
        samplers: ``background_knowledge.samplers``: a tuple in variable order, or ``None`` (read-only).
        constant_bounds: ``background_knowledge.constant_bounds`` (read-only).
        default_constant: ``background_knowledge.default_constant``, complete (read-only).
        required_constraints: The required constraints as a tuple (read-only).
        ranking_function: As given.
        max_evaluations: As given.
        success_threshold: As given.
        ground_truth: The answer as an ``Expression`` in ``symbol_library``, or ``None``.
        ground_truth_constants: The specs of ``ground_truth`` sorted by label, or ``None`` if it
            has none (read-only).
        original_equation: The given answer, or one derived from ``ground_truth``; ``None`` if
            neither (read-only).
        seed: As given.
        dataset_metadata: As given.
        dataset_name: As given.
        benchmark: As given.
        version: As given.

    Examples:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> dataset = Dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]),
        ...     ground_truth=["X_0", "+", "X_1"], success_threshold=1e-6)
        >>> evaluator = dataset.create_evaluator()
        >>> bool(evaluator.evaluate_expr(["C", "*", "X_1", "+", "X_0"]).error < dataset.success_threshold)
        True
    """

    def __init__(
        self,
        X: np.ndarray,
        symbol_library: SymbolLibrary,
        ranking_function: str = "rmse",
        y: np.ndarray | None = None,
        max_evaluations: int = -1,
        ground_truth: list[str] | Expression | None = None,
        original_equation: str | None = None,
        success_threshold: float | None = None,
        seed: int | None = None,
        dataset_metadata: dict | None = None,
        dataset_name: str = "unnamed",
        samplers: list[Sampler] | Mapping[str, Sampler] | None = None,
        benchmark: str | None = None,
        version: str | None = None,
        test_X: np.ndarray | None = None,
        test_y: np.ndarray | None = None,
        aux_data: dict[str, dict[str, np.ndarray] | DataSplit] | None = None,
        variables: Mapping[str, VariableInfo] | None = None,
        target: VariableInfo | None = None,
        named_constants: Mapping[str, VariableInfo] | None = None,
        description: str | None = None,
        constant_bounds: tuple[float, float] | None = None,
        default_constant: Constant | None = None,
        reference_grammar: Grammar | None = None,
        max_constants: int | None = None,
        max_expr_length: int | None = None,
        required_constraints: Iterable[ExpressionConstraint] | None = None,
        advisory_constraints: Iterable[ExpressionConstraint] | None = None,
        settings: Settings | None = None,
        ranking_resample: bool = False,
    ) -> None:
        self._background_knowledge, self._required_constraints, self.settings = _resolve_task(
            {
                "symbol_library": symbol_library,
                "samplers": samplers,
                "variables": variables,
                "target": target,
                "named_constants": named_constants,
                "description": description,
                "constant_bounds": constant_bounds,
                "default_constant": default_constant,
                "reference_grammar": reference_grammar,
                "constraints": advisory_constraints,
            },
            max_constants,
            max_expr_length,
            required_constraints,
            settings,
            ranking_resample,
            X,
        )
        self.ranking_resample = ranking_resample

        if max_evaluations == 0:
            raise ValueError("[Dataset] max_evaluations must be positive, or negative for no limit; got 0.")
        self.max_evaluations = max_evaluations
        self.success_threshold = success_threshold
        self.ranking_function = ranking_function
        try:
            self.ground_truth = _as_ground_truth(
                ground_truth, self.symbol_library, cast(Constant, self._background_knowledge.default_constant)
            )
        except (TypeError, ValueError) as e:
            raise type(e)(f"[Dataset] {e}") from None
        self._original_equation = original_equation

        self.dataset_name = dataset_name

        # The data lives in DataSplits: the train split (X / y, the data the search runs on), an
        # optional held-out test split, and optional extra splits for the user's own measures.
        train_arrays: dict[str, np.ndarray] = {"X": X}
        if y is not None:
            train_arrays["y"] = y
        self._train = DataSplit(train_arrays)
        self.test: DataSplit | None = None
        if test_X is not None:
            test_arrays: dict[str, np.ndarray] = {"X": test_X}
            if test_y is not None:
                test_arrays["y"] = test_y
            self.test = DataSplit(test_arrays)
        self.aux_data: dict[str, DataSplit] = {}
        for key, value in (aux_data or {}).items():
            self._validate_aux_key(key)
            self.aux_data[key] = value if isinstance(value, DataSplit) else DataSplit(dict(value))

        self.contains_constants = self.symbol_library.constant_symbol is not None

        self.seed = seed
        self.dataset_metadata = dataset_metadata

        # Cache / serialisation metadata
        self.benchmark = benchmark
        self.version = version
        # Origin of the cached data (UrlSource / SampleSource / None). Not a constructor
        # argument: it is set by the factory methods (from_dict, from_samplers) and by
        # Benchmark.add_dataset. End users select an origin through those entry points.
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
    def samplers(self) -> tuple[Sampler, ...] | None:
        """Shortcut to ``background_knowledge.samplers`` (read-only)."""
        return self._background_knowledge.samplers  # type: ignore[return-value]

    @property
    def constant_bounds(self) -> tuple[float, float]:
        """Shortcut to ``background_knowledge.constant_bounds`` (read-only)."""
        return self._background_knowledge.constant_bounds

    @property
    def default_constant(self) -> Constant:
        """Shortcut to ``background_knowledge.default_constant``, complete (read-only)."""
        return cast(Constant, self._background_knowledge.default_constant)

    @property
    def required_constraints(self) -> tuple[ExpressionConstraint, ...]:
        """The required constraints (read-only)."""
        return self._required_constraints

    @property
    def ground_truth_constants(self) -> tuple[Constant, ...] | None:
        """The specs of ``ground_truth``, sorted by label; ``None`` if it has none (read-only)."""
        if self.ground_truth is None or not self.ground_truth.free_constants:
            return None
        return tuple(sorted(self.ground_truth.free_constants, key=lambda c: c.label or ""))

    @property
    def original_equation(self) -> str | None:
        """Human-readable answer, e.g. ``"F = m*a"`` (read-only).

        The given ``original_equation`` if it is not ``None``. Otherwise ``"<target name> = "``
        followed by the infix tokens of ``ground_truth`` joined without spaces, each variable token
        replaced by its name in ``background_knowledge.variables`` and the unary minus ``u-``
        written ``-``. A variable with no entry or an empty name keeps its token. With no target or an empty target name the left
        side is the first of ``y``, ``y_1``, ``y_2``, … that is not a variable's name. ``None`` if no
        ``original_equation`` was given and ``ground_truth`` is ``None``.

        Examples:
            >>> ds = Dataset(np.zeros((1, 2)), SymbolLibrary.default_symbols(2),
            ...     ground_truth=["X_0", "*", "X_1"], variables={"X_0": VariableInfo("m")},
            ...     target=VariableInfo("F"))
            >>> ds.original_equation
            'F = m*X_1'
        """
        if self._original_equation is not None or self.ground_truth is None:
            return self._original_equation
        bk = self._background_knowledge
        return _derive_equation(list(self.ground_truth.tokens), bk.variables, bk.target)

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
                f"[Dataset] Invalid aux data key {key!r}: aux keys become file names in the data "
                f"cache, so a key must be a non-empty string other than '.' or '..', without any of "
                f"{' '.join(cls._FORBIDDEN_KEY_CHARS)}."
            )

    def _splits(self) -> list[tuple[str, DataSplit]]:
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
    def y(self) -> np.ndarray | None:
        """Target values the search runs on, or ``None`` if the dataset has no targets."""
        return self._train.arrays.get("y")

    @y.setter
    def y(self, value: np.ndarray | None) -> None:
        if value is None:
            self._train.arrays.pop("y", None)
        else:
            self._train.arrays["y"] = value

    @property
    def test_X(self) -> np.ndarray | None:
        """Inputs of the held-out test split, or ``None`` if there is none."""
        return self.test.X if self.test is not None else None

    @property
    def test_y(self) -> np.ndarray | None:
        """Targets of the held-out test split, or ``None`` if there are none."""
        return self.test.y if self.test is not None else None

    @property
    def data_source(self) -> DataSource | None:
        """Origin of the data the search runs on (``SampleSource`` / ``UrlSource`` / ``None``)."""
        return self._train.data_source

    @data_source.setter
    def data_source(self, value: DataSource | None) -> None:
        self._train.data_source = value

    def evaluate_approach(
        self,
        approach: Approach,
        num_experiments: int = 1,
        top_k: int = 20,
        initial_seed: int | None = None,
        results: Results | None = None,
        callbacks: Callback | CallbackDispatcher | list[Callback] | None = None,
        verbose: bool = True,
        adaptation_path: str | None = None,
        checkpoint_dir: str | os.PathLike | None = None,
        resume: bool = False,
        keep_checkpoint: bool = False,
        snapshot_interval: float | None = 300.0,
        keep_all_models: bool = True,
    ) -> Results:
        """Run an SR approach on this dataset and collect the results.

        Each experiment searches with a new evaluator from
        [create_evaluator][SRToolkit.dataset.dataset.Dataset.create_evaluator]. Experiment ``i``
        (from 0) is seeded with ``seed + i``, where ``seed`` is ``initial_seed`` or the dataset seed; it
        is unseeded if both are ``None``. Approaches with ``adaptation_scope="once"`` adapt before the
        first experiment, those with ``"experiment"`` before each one.

        With ``checkpoint_dir``, each experiment keeps its models and trace in the checkpoint
        ``checkpoint_dir/{dataset_name}/{approach_name}/exp_{seed}.sqlite`` (the layout of an
        [ExperimentGrid][SRToolkit.experiments.ExperimentGrid]; see
        [Checkpoints][SRToolkit.evaluation.evaluator]), and its search, not its adaptation, runs
        inside [stop_on_signals][SRToolkit.evaluation.evaluator.Evaluator.stop_on_signals]:

        - when every experiment has finished, their checkpoints are deleted (or, with
          ``keep_checkpoint=True``, kept and compacted, each into one self-contained file);
        - if an experiment is interrupted (Ctrl-C or SIGTERM), no further experiments run: its
          results are added with stop reason ``"interrupted"``, all checkpoints are kept, and a
          ``UserWarning`` says where;
        - if a search raises, all checkpoints are kept (committed and closed), and the exception
          propagates.

        With ``resume=True``, the call continues where an earlier one stopped: finished experiments
        are read from their checkpoints as they are (without adapting, running or calling
        callbacks), the interrupted one continues from its last snapshot (or starts anew, with a
        ``UserWarning``, if the approach lacks the
        [step interface][SRToolkit.approaches.approach.Approach.supports_steps]), and
        the rest run. A finished run is not extended, and its settings are not checked against this
        call's. Without
        ``checkpoint_dir``, nothing is written to disk and no signal handlers are installed.

        A result refers to its experiment's checkpoint (its
        [checkpoint][SRToolkit.utils.types.EvalResult]) exactly when the file remains as the call
        returns or raises: kept, interrupted or failed.

        Every experiment's results are kept in memory until the call returns; for long runs, use
        ``keep_all_models=False``, or an [ExperimentGrid][SRToolkit.experiments.ExperimentGrid],
        which saves each experiment's results to its own file.

        Args:
            approach: The approach to run.
            num_experiments: Number of independent runs.
            top_k: Number of best expressions kept per run.
            initial_seed: Base seed; ``None`` uses the dataset seed.
            results: [Results][SRToolkit.evaluation.results.Results] to append to in place;
                ``None`` creates a new one.
            callbacks: An [Callback][SRToolkit.evaluation.callbacks.Callback], a list of them, or a
                [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher].
            verbose: Print a line as each experiment starts.
            adaptation_path: For ``adaptation_scope="once"``: the state file passed to
                [adapt_or_load][SRToolkit.dataset.dataset.Dataset.adapt_or_load]; ``None``
                adapts without saving.
            checkpoint_dir: Directory for the experiments' checkpoints, created if missing;
                ``None`` (default) keeps them in memory.
            resume: Continue the experiments whose checkpoints exist in ``checkpoint_dir``.
                Default ``False``: an existing checkpoint raises ``FileExistsError``.
            keep_checkpoint: Keep the checkpoints of finished experiments, e.g. to inspect all their
                models later. Default ``False``.
            snapshot_interval: Seconds between snapshots in an experiment with a checkpoint;
                ``None``: only when the approach's step loop ends. Default ``300.0``.
            keep_all_models: As for
                [get_results][SRToolkit.evaluation.evaluator.Evaluator.get_results], for every
                experiment, including those read from checkpoints. Default ``True``.

        Returns:
            ``results`` (or a new ``Results``) with one entry per experiment run.

        Raises:
            ValueError: If ``checkpoint_dir`` is given and the experiments are unseeded (``initial_seed``
                and the dataset seed are ``None``): a checkpoint is named by its seed; if
                ``keep_all_models`` is ``False`` and ``top_k <= 0`` (before anything runs). Also as for
                [Evaluator][SRToolkit.evaluation.evaluator.Evaluator] with ``resume=True``
                when an interrupted experiment continues (e.g. its checkpoint is of a run with other
                settings).
            FileExistsError: If ``resume`` is ``False`` and an experiment's checkpoint exists.
        """
        base_seed = self.seed if initial_seed is None else initial_seed
        if checkpoint_dir is not None and base_seed is None:
            raise ValueError(
                "[Dataset.evaluate_approach] checkpoint_dir needs seeded experiments: pass initial_seed or "
                "give the dataset a seed."
            )

        if not keep_all_models and top_k <= 0:
            raise ValueError(f"[Dataset.evaluate_approach] keep_all_models=False needs top_k >= 1, got {top_k}.")
        if results is None:
            results = Results()

        if isinstance(callbacks, Callback):
            dispatcher = CallbackDispatcher(callbacks=[callbacks])
            callbacks = dispatcher
        elif isinstance(callbacks, list):
            if len(callbacks) == 0:
                callbacks = None
            else:
                callbacks = CallbackDispatcher(callbacks=callbacks)

        dataset_name = self.dataset_name
        checkpoints: list[str] = []
        referring: list[EvalResult] = []  # results of this call that refer to their checkpoint
        adapted = False
        for experiment in range(num_experiments):
            reference = None
            # Experiment i gets base_seed + i, as job i of an ExperimentGrid does.
            seed = None if base_seed is None else base_seed + experiment
            checkpoint = None
            continued = False
            if checkpoint_dir is not None:
                checkpoint = os.path.join(checkpoint_dir, dataset_name, approach.name, f"exp_{seed}.sqlite")
                os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
                checkpoints.append(checkpoint)
                if resume and os.path.exists(checkpoint):
                    if _run_finished(checkpoint):  # from an earlier call: read it, run nothing
                        if verbose:
                            print(f"Experiment {experiment + 1}/{num_experiments}: read from its checkpoint")
                        results += Results.from_checkpoint(checkpoint, approach.name, top_k, keep_all_models)
                        referring.append(results.results[-1])
                        continue
                    if approach.supports_steps:
                        continued = True
                    else:
                        warnings.warn(
                            f"[Dataset.evaluate_approach] {approach.name} cannot continue from a snapshot, "
                            f"so experiment {experiment + 1}/{num_experiments} starts afresh and replaces its "
                            f"checkpoint {checkpoint}.",
                            UserWarning,
                        )
                        _remove_checkpoint(checkpoint)
            if verbose:
                print(f"Running experiment {experiment + 1}/{num_experiments}")

            event = ExperimentEvent(
                dataset_name=dataset_name,
                approach_name=approach.name,
                success_threshold=self.success_threshold,
                max_evaluations=self.max_evaluations,
                seed=seed,
            )
            if callbacks is not None:
                callbacks.on_experiment_start(event)

            if approach.adaptation_scope == "once" and not adapted:
                self.adapt_or_load(approach, adaptation_path)
                adapted = True
            elif approach.adaptation_scope == "experiment":
                approach.adapt(self.X, self.background_knowledge, self.required_constraints)

            evaluator = self.create_evaluator(
                seed=seed, checkpoint=checkpoint, resume=continued, snapshot_interval=snapshot_interval
            )
            try:
                evaluator._experiment_id = f"{dataset_name}_{approach.name}_{seed}"
                evaluator.register_callbacks(callbacks)
                with evaluator.stop_on_signals() if checkpoint is not None else nullcontext():
                    approach.run(evaluator, seed)
                results += evaluator.get_results(approach.name, top_k, keep_all_models=keep_all_models)
                if checkpoint is not None:
                    reference = _checkpoint_reference(evaluator._store, checkpoint)
            finally:
                evaluator.close()

            result = results.results[-1]
            # A callback sees the reference only if the file is sure to stay; it is added afterwards otherwise,
            # since the file remains until the call ends (and is deleted then, see below).
            if reference is not None and (keep_checkpoint or evaluator.stop_reason == "interrupted"):
                result.checkpoint = reference
            if callbacks is not None:
                callbacks.on_experiment_end(event, result)
            if reference is not None:
                result.checkpoint = reference
                referring.append(result)
            if evaluator.stop_reason == "interrupted":
                warnings.warn(
                    f"[Dataset.evaluate_approach] Experiment {experiment + 1}/{num_experiments} was interrupted; "
                    f"its checkpoint is kept at {checkpoint}. Call again with resume=True to continue.",
                    UserWarning,
                )
                return results

        for checkpoint in checkpoints:
            if keep_checkpoint:
                _compact_checkpoint(checkpoint)
            else:
                _remove_checkpoint(checkpoint)
        if not keep_checkpoint:
            for result in referring:
                result.checkpoint = None
        return results

    def extend_run(
        self,
        approach: Approach,
        checkpoint: str | os.PathLike,
        *,
        max_evaluations: int | None = None,
        success_threshold: float | None = None,
        callbacks: Callback | CallbackDispatcher | list[Callback] | None = None,
        top_k: int = 20,
        keep_all_models: bool = True,
        snapshot_interval: float | None = 300.0,
        adaptation_path: str | None = None,
    ) -> Results:
        """Continue the run saved in ``checkpoint`` with new limits.

        Use it to give a finished or interrupted run a larger budget, a lower threshold or other
        callbacks (e.g. a longer time limit). The run goes back to its last snapshot and continues
        from there, with its seed (see [Extending runs][SRToolkit.evaluation.evaluator]). If the
        approach's search depends only on its saved state, not on its limits, the result equals a
        run that had the new limits from the start.

        Limits left as ``None`` stay as they were; this dataset's own ``max_evaluations`` and
        ``success_threshold`` are not used. If nothing changes (the same limits and callbacks), the
        call is a plain resume: a finished run stays finished and an interrupted one continues. As
        in [evaluate_approach][SRToolkit.dataset.dataset.Dataset.evaluate_approach], the
        approach adapts first, the callbacks get ``on_experiment_start`` and ``on_experiment_end``,
        and Ctrl-C stops the run cleanly.

        The checkpoint must come from this dataset: the same settings apart from the limits. The
        setting ``evaluator.max_consecutive_skips`` is not a limit and cannot change, so a run stopped by it
        stops again at once. The approach is not checked, so pass the run's approach. If an extension is interrupted, call
        ``extend_run`` again with ``None`` limits to continue it. When the run ends, the checkpoint
        is compacted into one file, which the result refers to.

        Note:
            The checkpoint is extended in place, so afterwards it holds the extended run. The
            results of the original run do not change: keep them (e.g. with
            [save][SRToolkit.evaluation.results.Results.save]), or copy the checkpoint first if
            you need the original file. Their
            [open_checkpoint][SRToolkit.utils.types.EvalResult.open_checkpoint] raises once the
            limits change. For an approach that is deterministic from its state, the extended run
            starts with the original run's calls, so the statistics of its first ``n`` calls (e.g.
            [best_error_within][SRToolkit.utils.types.EvalResult.best_error_within] ``([n])``) are
            those of the original run, with ``n`` its ``evaluation_calls``.

        Args:
            approach: The approach of the run, which must implement the step interface (see
                [supports_steps][SRToolkit.approaches.approach.Approach.supports_steps]).
            checkpoint: The run's checkpoint.
            max_evaluations: The new total budget, at least the calls the run made; negative for
                no limit; ``None`` (default) keeps the last segment's.
            success_threshold: The new threshold (a negative one is never reached); ``None``
                (default) keeps the last segment's.
            callbacks: An [Callback][SRToolkit.evaluation.callbacks.Callback], a list of them,
                or a [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher];
                ``None`` for none; the run's earlier callbacks are not reused. Give them in the run's
                order, so they get their saved states (e.g. a longer ``TimeLimitCallback`` counts on from the
                time saved with the snapshot).
            top_k: As for [get_results][SRToolkit.evaluation.evaluator.Evaluator.get_results].
            keep_all_models: As for ``get_results``.
            snapshot_interval: As for [Evaluator][SRToolkit.evaluation.evaluator.Evaluator].
                Default ``300.0``.
            adaptation_path: For ``adaptation_scope="once"``: the state file passed to
                [adapt_or_load][SRToolkit.dataset.dataset.Dataset.adapt_or_load].

        Returns:
            New results with the run's [EvalResult][SRToolkit.utils.types.EvalResult]; its
            ``stop_reason`` is ``"interrupted"`` after Ctrl-C or SIGTERM (with a ``UserWarning``).

        Raises:
            TypeError: Before anything is changed: if ``max_evaluations``
                is not an ``int``, numpy integer or ``None``, or ``success_threshold`` is not a real
                number or ``None`` (``bool`` is rejected for both); if a callback's ``to_dict``
                holds a value that cannot be saved (see
                [serialization][SRToolkit.utils.serialization]).
            ValueError: Before anything is changed: if ``max_evaluations`` is ``0``, or
                non-negative and below the calls the run made (an extension does not shorten a
                run); if ``keep_all_models`` is ``False`` and ``top_k <= 0``; if the approach does
                not support snapshots; if the file is not a checkpoint, or is of a run with other
                settings than this dataset's (the message names them).
            FileNotFoundError: If no file exists at ``checkpoint``.

        Examples:
            >>> import tempfile
            >>> from SRToolkit.approaches import ProGED
            >>> dataset = Dataset(np.array([[1.0], [2.0], [3.0]]), SymbolLibrary.default_symbols(1),
            ...                      y=np.array([2.0, 4.0, 6.0]), max_evaluations=5, success_threshold=-1, seed=0)
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     first = dataset.evaluate_approach(ProGED(), checkpoint_dir=tmpdir, keep_checkpoint=True,
            ...                                       verbose=False)
            ...     extended = dataset.extend_run(ProGED(), first[0].checkpoint["path"], max_evaluations=10)
            >>> extended[0].evaluation_calls, extended[0].extended
            (10, True)
        """
        if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer, type(None))):
            raise TypeError(
                f"[Dataset.extend_run] max_evaluations must be an int or None, got {type(max_evaluations).__name__}."
            )
        if max_evaluations == 0:
            raise ValueError("[Dataset.extend_run] max_evaluations must be positive, or negative for no limit.")
        if isinstance(success_threshold, bool) or not isinstance(success_threshold, (numbers.Real, type(None))):
            raise TypeError(
                "[Dataset.extend_run] success_threshold must be a real number or None, got "
                f"{type(success_threshold).__name__}."
            )
        if not keep_all_models and top_k <= 0:
            raise ValueError(f"[Dataset.extend_run] keep_all_models=False needs top_k >= 1, got {top_k}.")
        if not approach.supports_steps:
            raise ValueError(
                f"[Dataset.extend_run] {approach.name} cannot continue from a snapshot, so its runs cannot be extended."
            )
        if isinstance(callbacks, Callback):
            callbacks = CallbackDispatcher(callbacks=[callbacks])
        elif isinstance(callbacks, list):
            callbacks = CallbackDispatcher(callbacks=callbacks) if callbacks else None
        # Raises TypeError for a config that cannot be saved, before anything is changed.
        configs = _to_json_safe([cb.to_dict() for cb in (callbacks.get_callbacks() if callbacks else [])], "callbacks")
        store = ModelStore(checkpoint, "read")
        try:
            seed = store.get_header("seed")
            last = _from_json_safe(store.get_header("limits"))[-1]
            saved = store.get_header("fingerprint", {})
            calls = len(store.trace)
        finally:
            store.close()
        if max_evaluations is not None and 0 <= max_evaluations < calls:
            raise ValueError(
                f"[Dataset.extend_run] max_evaluations={max_evaluations} is below the {calls} calls the run "
                "made; an extension does not shorten a run."
            )
        # A copy of this dataset with the new limits builds the run's evaluators.
        dataset = copy.copy(self)
        dataset.max_evaluations = last["max_evaluations"] if max_evaluations is None else int(max_evaluations)
        dataset.success_threshold = last["success_threshold"] if success_threshold is None else float(success_threshold)
        probe = dataset.create_evaluator(seed=seed)
        fingerprint = probe._fingerprint()
        probe.close()
        changed = sorted(
            k
            for k in fingerprint.keys() | saved.keys()
            if k not in ("max_evaluations", "success_threshold") and fingerprint.get(k) != saved.get(k)
        )
        if changed:
            raise ValueError(
                f"[Dataset.extend_run] {os.fspath(checkpoint)!r} is of a run with other settings than this "
                f"dataset's: {', '.join(changed)}."
            )

        if approach.adaptation_scope == "once":
            self.adapt_or_load(approach, adaptation_path)
        elif approach.adaptation_scope == "experiment":
            approach.adapt(self.X, self.background_knowledge, self.required_constraints)

        # An extension that changes nothing is a plain resume: no new segment, and a finished run
        # stays finished. A segment without recorded callbacks (it made no call) matches any.
        unchanged = (last["max_evaluations"], last["success_threshold"]) == (
            dataset.max_evaluations,
            probe.success_threshold,
        ) and (last["callbacks"] is None or _to_json_safe(last["callbacks"], "callbacks") == configs)
        if not unchanged:
            # Reopen the run with the new limits: the new fingerprint, a limit segment starting at the
            # snapshot, and no stop reason, so the resume below rewinds it to the snapshot.
            store = ModelStore(checkpoint, "resume")
            try:
                limits = _from_json_safe(store.get_header("limits"))
                limits.append(
                    {
                        "calls": 0 if store.snapshot is None else store.snapshot.num_calls,
                        "max_evaluations": dataset.max_evaluations,
                        "success_threshold": probe.success_threshold,
                        "callbacks": None,  # recorded at the first evaluation, as for a new run
                        "previous_stop_reason": store.get_header("stop_reason"),
                    }
                )
                store.set_header("fingerprint", fingerprint)
                store.set_header("limits", _to_json_safe(limits, "limits"))
                store.set_header("stop_reason", None)
                store.commit()
            finally:
                store.close()
        evaluator = dataset.create_evaluator(
            seed=seed, checkpoint=checkpoint, resume=True, snapshot_interval=snapshot_interval
        )
        event = ExperimentEvent(
            dataset_name=self.dataset_name,
            approach_name=approach.name,
            success_threshold=evaluator.success_threshold,
            max_evaluations=evaluator.max_evaluations,
            seed=seed,
        )
        try:
            evaluator._experiment_id = f"{self.dataset_name}_{approach.name}_{seed}"
            evaluator.register_callbacks(callbacks)
            if callbacks is not None:
                callbacks.on_experiment_start(event)
            with evaluator.stop_on_signals():
                approach.run(evaluator, seed)
            results = evaluator.get_results(approach.name, top_k, keep_all_models=keep_all_models)
            reference = _checkpoint_reference(evaluator._store, checkpoint)
        finally:
            evaluator.close()
        interrupted = evaluator.stop_reason == "interrupted"
        if not interrupted:
            _compact_checkpoint(checkpoint)
        result = results.results[-1]
        result.checkpoint = reference
        if callbacks is not None:
            callbacks.on_experiment_end(event, result)
        if interrupted:
            warnings.warn(
                f"[Dataset.extend_run] The run was interrupted; its checkpoint is kept at {checkpoint}. Call "
                "extend_run again to continue it.",
                UserWarning,
            )
        return results

    def adapt_or_load(self, approach: Approach, path: str | None, force: bool = False) -> None:
        """Adapt ``approach`` to this dataset, or load its saved adapted state.

        If ``path`` names an existing file and ``force`` is off, the state is loaded from it with
        [load_adapted_state][SRToolkit.approaches.approach.Approach.load_adapted_state].
        Otherwise the approach runs [adapt][SRToolkit.approaches.approach.Approach.adapt] on
        ``X``, ``background_knowledge`` and ``required_constraints`` and, if ``path`` is set, saves
        the state there with
        [save_adapted_state][SRToolkit.approaches.approach.Approach.save_adapted_state],
        creating its parent directories. The state is written to a temporary file next to ``path``
        and moved into place, so jobs adapting in parallel never load a half-written file; if saving
        fails, the exception propagates and no file is left behind. An existing file is loaded as
        is; it is not checked against this dataset. Does not check ``adaptation_scope``.

        Args:
            approach: The approach to adapt.
            path: File the adapted state is loaded from or saved to; ``None`` adapts without saving.
            force: Adapt and overwrite the file even if it exists.

        Raises:
            NotImplementedError: If ``path`` is set and the approach does not implement
                ``load_adapted_state`` or ``save_adapted_state`` (whichever is needed).
        """
        if path is not None and not force and os.path.exists(path):
            approach.load_adapted_state(path)
            return
        approach.adapt(self.X, self.background_knowledge, self.required_constraints)
        if path is not None:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            # The temporary name keeps the extension, since e.g. np.save appends one otherwise.
            root, ext = os.path.splitext(path)
            tmp_path = f"{root}.tmp{os.getpid()}{ext}"
            try:
                approach.save_adapted_state(tmp_path)
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def create_evaluator(
        self,
        metadata: dict[str, Any] | None = None,
        seed: int | None = None,
        checkpoint: str | os.PathLike | None = None,
        resume: bool = False,
        snapshot_interval: float | None = 300.0,
    ) -> Evaluator:
        """Create an [Evaluator][SRToolkit.evaluation.evaluator.Evaluator] for this dataset.

        The evaluator gets the parts of the background knowledge (its ``constraints`` as advisory
        constraints), ``required_constraints`` as required (and adds no limits of its own), and
        ``ground_truth`` with its specs. Its ``background_knowledge`` is rebuilt from these parts, so its
        ``to_dict()``, ``describe()`` and ``priors()`` equal this dataset's.

        It gets ``settings`` with ``evaluator.max_constants`` and ``evaluator.max_expr_length`` set
        to ``-1`` (the limits are in ``required_constraints``) and ``evaluator.ranking_seed``:

        - the one in ``settings``, if set;
        - otherwise unset if ``ranking_resample``, so the ranking uses the evaluator's seed;
        - otherwise the dataset seed (``0`` if ``None``), so every run is scored by the same
          objective: the same BED points and behaviours, and the same starting point for each
          expression's fit.

        To score held-out data as the search did, pass the returned evaluator's ``settings`` and
        ``background_knowledge`` to
        [HoldoutMetric][SRToolkit.evaluation.result_augmentation.HoldoutMetric].

        Args:
            metadata: Metadata to attach to the evaluator; ``dataset_name`` and ``dataset_metadata`` are
                added to it in place.
            seed: Seed of the evaluator; ``None`` uses the dataset seed.
            checkpoint: The evaluator's checkpoint file; ``None`` (default) keeps its models in
                memory.
            resume: Continue the run saved in ``checkpoint``. Default ``False``.
            snapshot_interval: Seconds between the evaluator's snapshots; ``None``: only when the
                approach's step loop ends. Default ``300.0``.

        Returns:
            The configured evaluator.

        Raises:
            ValueError: If the ranking rejects the dataset (e.g. RMSE with neither ``y`` nor
                ``ground_truth``), ``snapshot_interval`` is not positive, or as for
                [Evaluator][SRToolkit.evaluation.evaluator.Evaluator] with ``resume=True``.
            ImportError: If an import-path ``ranking_function`` cannot be imported.
            FileExistsError: As for ``Evaluator`` with a ``checkpoint``.
            FileNotFoundError: As for ``Evaluator`` with ``resume=True``.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = Dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]))
            >>> float(dataset.create_evaluator().evaluate_expr(["X_1", "+", "X_0"]).error)  # doctest: +ELLIPSIS
            0.0...
        """
        if metadata is None:
            metadata = dict()
        if self.dataset_metadata is not None:
            metadata["dataset_metadata"] = self.dataset_metadata
        metadata["dataset_name"] = self.dataset_name

        # The limits are already in required_constraints; -1 adds none of the evaluator's own.
        evaluator = {"max_constants": -1, "max_expr_length": -1}
        # Fix the ranking's randomness per dataset, unless it is resampled each run.
        if not self.ranking_resample and self.settings.evaluator.ranking_seed is None:
            evaluator["ranking_seed"] = self.seed if self.seed is not None else 0
        bk = self.background_knowledge
        return Evaluator(
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
            named_constants=self.background_knowledge.named_constants,
            description=self.background_knowledge.description,
            reference_grammar=self.background_knowledge.reference_grammar,
            constant_bounds=bk.constant_bounds,
            default_constant=bk.default_constant,
            samplers=None if self.samplers is None else list(self.samplers),
            settings=self.settings.replace(evaluator=evaluator),
            checkpoint=checkpoint,
            resume=resume,
            snapshot_interval=snapshot_interval,
        )

    def describe(self) -> str:
        """Return the task as text for people and LLM prompts, without the answer.

        An evaluator from ``create_evaluator`` gives the same text with
        [Evaluator.describe][SRToolkit.evaluation.evaluator.Evaluator.describe].

        Returns:
            [BackgroundKnowledge.describe][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.describe]
            with ``required_constraints`` listed as enforced. Unlike ``str(dataset)``, it has no ground
            truth, equation or success threshold.
        """
        return self.background_knowledge.describe(self.required_constraints)

    def priors(self) -> dict[str, list[dict[str, Any]]]:
        """Return the constraints as records, the structured form of ``describe``.

        An evaluator from ``create_evaluator`` gives the same with
        [Evaluator.priors][SRToolkit.evaluation.evaluator.Evaluator.priors].

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

        It gives [original_equation][SRToolkit.dataset.dataset.Dataset.original_equation] (omitted
        when ``None``), the ranking with its success rule, a line
        ``Non-default settings: section.field=value, ...`` (values in their saved form, as in
        [Settings.to_dict][SRToolkit.utils.settings.Settings.to_dict]; sections in stored order,
        fields in declaration order, without a custom section's ``section_class``; omitted when all
        settings are default), then
        [describe][SRToolkit.dataset.dataset.Dataset.describe] and a pointer to
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
        changed = [
            f"{section}.{key}={value}"
            for section, fields in self.settings.to_dict()["sections"].items()
            for key, value in fields.items()
            if key != "section_class"
        ]
        if changed:
            text += f"Non-default settings: {', '.join(changed)}.\n"
        text += self.describe() + "\n"
        text += "For other metadata, please refer to the attribute self.dataset_metadata."
        return text

    def resample(self, n: int, seed: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
                ``None`` or has a free constant without a fixed value in its specs.
        """
        if self.samplers is None:
            raise ValueError(
                f"[Dataset.resample] Dataset '{self.dataset_name}' has no samplers defined. "
                "Provide samplers when constructing the dataset."
            )
        if seed is not None:
            np.random.seed(seed)
        X = _sample_inputs(self.samplers, n)
        if self.ranking_function == "bed":
            return X
        if self.ground_truth is None:
            raise ValueError(f"[Dataset.resample] Dataset '{self.dataset_name}' has no ground truth to compute y from.")

        parameters = _ground_truth_parameters(self.ground_truth, f"[Dataset.resample] Dataset '{self.dataset_name}'")
        y = compile_expr(self.ground_truth)(X, parameters)
        return X, y

    def resample_inplace(self, n: int, seed: int | None = None) -> "Dataset":
        """Replace ``X`` and ``y`` with data from [resample][SRToolkit.dataset.dataset.Dataset.resample].

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
        """Return a JSON-safe dictionary of this dataset's configuration (format 5).

        Arrays are not embedded but live in the [data cache][SRToolkit.dataset.data_cache]: this call
        writes every split without a ``data_source`` there, so the result reloads with
        [from_dict][SRToolkit.dataset.dataset.Dataset.from_dict].

        Returns:
            A dict with ``format_version`` (``5``), ``dataset_name``, ``benchmark``, ``version``,
            ``ranking_function``, ``max_evaluations``, ``success_threshold``, ``original_equation``
            (the given string, or ``None`` when derived), ``seed``, ``dataset_metadata``,
            ``settings`` ([Settings.to_dict][SRToolkit.utils.settings.Settings.to_dict]),
            ``ranking_resample``,
            ``ground_truth`` (token list or ``None``), ``ground_truth_constants`` (a list of
            [Constant.to_dict][SRToolkit.utils.free_constants.Constant.to_dict], or ``None``),
            ``data_source``, ``test``, ``aux``, ``background_knowledge``
            ([BackgroundKnowledge.to_dict][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.to_dict])
            and ``required_constraints`` (each constraint's ``to_dict``, the limit ones included).

        Raises:
            ValueError: If ``benchmark`` or ``version`` is ``None``.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = Dataset(X, SymbolLibrary.default_symbols(2), y=np.array([3, 7, 11]),
            ...     benchmark="test", version="1.0.0")
            >>> dataset.to_dict()["format_version"]
            5
        """
        if self.benchmark is None:
            raise ValueError("[Dataset.to_dict] 'benchmark' is None. Set self.benchmark before serialising.")
        if self.version is None:
            raise ValueError("[Dataset.to_dict] 'version' is None. Set self.version before serialising.")

        # Persist any data split whose arrays are the only copy (no data_source) into the
        # cache so the returned config is reloadable by from_dict. Source-backed splits are
        # regenerated on load, so _persist_to_cache skips them and this stays side-effect-free.
        self._persist_to_cache()

        # The test and aux splits: which arrays each holds and where it comes from (the train
        # split's source is carried top-level as `data_source`).
        def split_entry(split: DataSplit) -> dict:
            return {
                "arrays": list(split.arrays.keys()),
                "data_source": split.data_source.to_dict() if split.data_source is not None else None,
            }

        return {
            "format_version": 5,
            "dataset_name": self.dataset_name,
            "benchmark": self.benchmark,
            "version": self.version,
            "ranking_function": self.ranking_function,
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "original_equation": self._original_equation,
            "seed": self.seed,
            "dataset_metadata": self.dataset_metadata,
            "settings": self.settings.to_dict(),
            "ranking_resample": self.ranking_resample,
            "ground_truth": None if self.ground_truth is None else list(self.ground_truth.tokens),
            "ground_truth_constants": (
                None if self.ground_truth_constants is None else [c.to_dict() for c in self.ground_truth_constants]
            ),
            "data_source": self.data_source.to_dict() if self.data_source is not None else None,
            "test": split_entry(self.test) if self.test is not None else None,
            "aux": {key: split_entry(split) for key, split in self.aux_data.items()},
            "background_knowledge": self.background_knowledge.to_dict(),
            "required_constraints": [c.to_dict() for c in self.required_constraints],
        }

    @classmethod
    def from_dict(cls, d: dict | str | Path) -> "Dataset":
        """Load a dataset from its dictionary representation or a JSON file.

        Arrays come from the data cache, or are generated from their ``data_source``. Formats 2–4
        are migrated (see the [module docstring][SRToolkit.dataset.dataset]): their flat
        ``"kwargs"`` are sorted into settings sections, except ``constant_bounds``,
        ``max_constants``, ``max_expr_length`` and ``ranking_resample``, which become the arguments
        of the same name. For
        ``d = ds.to_dict()``, ``from_dict(d).to_dict() == d``. Load ``.zip`` archives with
        [share.load][SRToolkit.share.load].

        Args:
            d: Output of [to_dict][SRToolkit.dataset.dataset.Dataset.to_dict], or the path of a
                JSON file holding it.

        Returns:
            The dataset.

        Raises:
            ValueError: If ``d`` is a ``.zip`` path, ``format_version`` is not 2–5, a format 2–4
                ``"kwargs"`` key is neither one of those arguments nor a built-in setting,
                ``"ground_truth_constants"`` are given without a ``"ground_truth"`` or do not fit
                it, or the constructor rejects the configuration.
            TypeError: As for the constructor.
            ImportError: If a saved sampler, constraint, ranking or settings section class cannot be
                imported.
            FileNotFoundError: If a split's data is not in the cache and it has no ``data_source``.

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> ds = Dataset.from_samplers(["X_0", "+", "X_1"], [UniformSampling(0, 5), UniformSampling(0, 5)],
            ...     n_samples=3, seed=0, benchmark="test_bench", version="1.0.0")
            >>> Dataset.from_dict(ds.to_dict()).X.shape
            (3, 2)
        """
        if isinstance(d, (str, Path)):
            if str(d).endswith(".zip"):
                raise ValueError(
                    "[Dataset.from_dict] Received a '.zip' path. Load archives with SRToolkit.share.load(path)."
                )
            with open(d) as f:
                dd = json.load(f)
        else:
            dd = dict(d)

        # Apply bundle relocation if needed
        dd = _auto_bind(dd)

        # Accept the current v5 layout and the v2-v4 configs of earlier releases. A v2 dict has no
        # "test" / "aux" entries, so only the train split loads; its cache key is the dataset name in
        # every version, and symbolic ground truth still loads from dd["ground_truth"]. (v2 datasets
        # that stored an ndarray ground truth in a separate _gt.npy are not migrated — ndarray ground
        # truth is no longer supported; provide a symbolic ground truth or rebuild such datasets.)
        fmt = dd.get("format_version")
        if fmt not in (2, 3, 4, 5):
            raise ValueError(f"[Dataset.from_dict] Unsupported format_version: {fmt!r}. Expected 2, 3, 4 or 5.")
        # Read the task part in either layout, validating it before any data is loaded or generated.
        task_args: dict[str, Any] = {}
        if fmt == 5:
            task_args["settings"] = Settings.from_dict(dd["settings"])
            task_args["ranking_resample"] = dd["ranking_resample"]
        else:  # flat kwargs: sort them into settings sections and task arguments
            sections: dict[str, dict[str, Any]] = {}
            old = dd.get("kwargs") or {}
            for key, value in old.items():
                if key not in _OLD_ARGUMENTS and key not in _OLD_SETTING_SECTIONS:
                    raise ValueError(
                        f"[Dataset.from_dict] {key!r} in 'kwargs' is not a built-in setting; move it to "
                        "a settings section named after its ranking (see Settings)."
                    )
                if value is None:  # None meant the default
                    continue
                if key in _OLD_ARGUMENTS:
                    task_args[key] = value
                else:
                    sections.setdefault(_OLD_SETTING_SECTIONS[key], {})[key] = value
            # An explicit ranking_seed None kept the ranking on the run's seed.
            if "ranking_seed" in old and old["ranking_seed"] is None:
                task_args["ranking_resample"] = True
            task_args["settings"] = Settings.from_dict({"format_version": 1, "sections": sections})
        if "background_knowledge" in dd:  # formats 4 and 5
            required = tuple(ExpressionConstraint.from_dict(c) for c in dd.get("required_constraints") or ())
            bk = BackgroundKnowledge.from_dict(dd["background_knowledge"])
            task_args.update(
                symbol_library=bk.symbol_library,
                samplers=bk.samplers,
                constant_bounds=bk.constant_bounds,
                default_constant=bk.default_constant,
                advisory_constraints=bk.constraints,
                reference_grammar=bk.reference_grammar,
                description=bk.description,
                variables=bk.variables,
                target=bk.target,
                named_constants=bk.named_constants,
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

        def load_split(key: str, source_dict: dict | None) -> DataSplit:
            # Resolve from the cache, materialising from the split's own source if needed. The
            # config slice keeps the dataset's samplers / ground truth, which a SampleSource needs.
            path = data_cache.resolve(benchmark, version, key, {**dd, "data_source": source_dict})
            data = np.load(str(path))
            return DataSplit({k: data[k] for k in data.files}, data_source=DataSource.from_dict(source_dict))

        # The ground truth with its specs, which are saved next to its tokens.
        specs = [Constant.from_dict(c) for c in dd.get("ground_truth_constants") or ()]
        ground_truth = None
        if dd.get("ground_truth") is not None:
            try:
                ground_truth = Expression(dd["ground_truth"], task_args["symbol_library"], specs)
            except (TypeError, ValueError) as e:
                raise type(e)(f"[Dataset.from_dict] 'ground_truth_constants' do not fit 'ground_truth': {e}") from None
        elif specs:
            raise ValueError("[Dataset.from_dict] 'ground_truth_constants' are given without a 'ground_truth'.")

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
            ground_truth=ground_truth,
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

    @classmethod
    def from_samplers(
        cls,
        ground_truth: list[str] | Expression,
        samplers: list[Sampler] | Mapping[str, Sampler] | None = None,
        symbol_library: SymbolLibrary | None = None,
        n_samples: int = 10000,
        seed: int | None = None,
        n_test: int | None = None,
        ranking_function: str = "rmse",
        original_equation: str | None = None,
        success_threshold: float | None = None,
        max_evaluations: int = -1,
        dataset_name: str = "unnamed",
        dataset_metadata: dict | None = None,
        benchmark: str | None = None,
        version: str | None = None,
        variables: Mapping[str, VariableInfo] | None = None,
        target: VariableInfo | None = None,
        named_constants: Mapping[str, VariableInfo] | None = None,
        description: str | None = None,
        constant_bounds: tuple[float, float] | None = None,
        default_constant: Constant | None = None,
        reference_grammar: Grammar | None = None,
        max_constants: int | None = None,
        max_expr_length: int | None = None,
        required_constraints: Iterable[ExpressionConstraint] | None = None,
        advisory_constraints: Iterable[ExpressionConstraint] | None = None,
        settings: Settings | None = None,
        ranking_resample: bool = False,
    ) -> "Dataset":
        """Build a dataset by sampling inputs and evaluating a ground truth on them.

        ``X`` is drawn from the samplers and ``y`` is computed as in
        [resample][SRToolkit.dataset.dataset.Dataset.resample]. The data gets a
        [SampleSource][SRToolkit.dataset.data_source.SampleSource] as ``data_source``, so ``to_dict``
        stores no arrays and [refresh][SRToolkit.dataset.dataset.Dataset.refresh] regenerates it.

        Args:
            ground_truth: The answer as infix tokens or an
                [Expression][SRToolkit.utils.expression.Expression] with its specs; unless
                ``ranking_function`` is ``"bed"``, every free constant in it needs a fixed value in
                its specs.
            samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per variable, as a list in
                variable order or ``{token: sampler}``.
            symbol_library: ``None`` uses the library of an ``Expression`` ground truth, and for a
                token list
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols] with
                one variable per sampler.
            n_samples: Number of rows of ``X``.
            seed: Seed of the draw, stored on the ``SampleSource``; ``None`` for none.
            n_test: Rows of a held-out test split drawn the same way with seed ``seed + 1``, with its own
                ``SampleSource``; ``None`` for no test split.
            ranking_function: As for ``Dataset``; with ``"bed"`` no ``y`` is generated.
            original_equation: As for ``Dataset``.
            success_threshold: As for ``Dataset``.
            max_evaluations: As for ``Dataset``.
            dataset_name: As for ``Dataset``.
            dataset_metadata: As for ``Dataset``.
            benchmark: As for ``Dataset``.
            version: As for ``Dataset``.
            variables: As for ``Dataset``.
            target: As for ``Dataset``.
            named_constants: As for ``Dataset``.
            description: As for ``Dataset``.
            constant_bounds: As for ``Dataset``.
            default_constant: As for ``Dataset``.
            reference_grammar: As for ``Dataset``.
            max_constants: As for ``Dataset``.
            max_expr_length: As for ``Dataset``.
            required_constraints: As for ``Dataset``.
            advisory_constraints: As for ``Dataset``.
            settings: As for ``Dataset``.
            ranking_resample: As for ``Dataset``.

        Returns:
            The dataset with freshly generated data.

        Raises:
            ValueError: If ``samplers`` is ``None`` or empty, ``y`` cannot be computed (see
                ``resample``), or the constructor rejects the arguments.
            TypeError: As for the constructor.
            ExpressionParseError: As for the constructor.

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> ds = Dataset.from_samplers(["X_0", "+", "X_1"],
            ...     [UniformSampling(0, 5), UniformSampling(0, 5)], n_samples=100, seed=0)
            >>> ds.X.shape
            (100, 2)
            >>> ds.original_equation
            'y = X_0+X_1'
        """
        if not samplers:
            raise ValueError("[Dataset.from_samplers] 'samplers' must be non-empty (one sampler per input variable).")
        n_vars = len(samplers)
        if symbol_library is None:
            symbol_library = (
                ground_truth.symbol_library
                if isinstance(ground_truth, Expression)
                else SymbolLibrary.default_symbols(n_vars)
            )

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
            named_constants=named_constants,
            description=description,
            constant_bounds=constant_bounds,
            default_constant=default_constant,
            reference_grammar=reference_grammar,
            max_constants=max_constants,
            max_expr_length=max_expr_length,
            required_constraints=required_constraints,
            advisory_constraints=advisory_constraints,
            settings=settings,
            ranking_resample=ranking_resample,
        )
        dataset.data_source = SampleSource(n_samples=n_samples, seed=seed)

        # Reuse the canonical generation path (handles rmse/bed and its validation).
        dataset.resample_inplace(n_samples, seed=seed)

        # Optional held-out test split: same samplers + ground truth, its own source/seed.
        if n_test is not None:
            test_seed = None if seed is None else seed + 1
            test_result = dataset.resample(n_test, seed=test_seed)
            if isinstance(test_result, tuple):
                test_arrays: dict[str, np.ndarray] = {"X": test_result[0], "y": test_result[1]}
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
                "[Dataset.refresh] Cannot refresh: data_source is null. "
                "Set self.data_source to a valid source config before calling refresh()."
            )
        if self.benchmark is None or self.version is None:
            raise ValueError("[Dataset.refresh] Cannot refresh: 'benchmark' or 'version' is None.")

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
