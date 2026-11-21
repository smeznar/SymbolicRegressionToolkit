"""Job-based experiment runner for symbolic regression experiments.

Provides three public classes:

- [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] — lightweight metadata (seed, paths) for a
  single run.
- [ExperimentJob][SRToolkit.experiments.ExperimentJob] — one atomic experiment: a single dataset ×
  approach × seed triple. Can be run in-process or dispatched to a CLI worker.
- [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] — a full cross-product grid of datasets and
  approaches.  Manages serialization, parallelism via HPC command files, progress tracking, and
  result loading.
"""

import dataclasses
import json
import os
import shlex
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.bundle._relocate import _auto_bind, _class_refs, _import_class
from SRToolkit.dataset.sr_benchmark import SR_benchmark
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.callbacks import CallbackDispatcher, ExperimentEvent, SRCallbacks
from SRToolkit.evaluation.results import SR_results


def _warn_missing_class_deps(configs: list[dict]) -> None:
    """Soft-check that every ``*_class`` path in ``configs`` is importable.

    Emits a single warning listing any unimportable classes. Does not raise:
    [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] stays lazy and the
    hard failure is deferred to job-run time (in
    [SR_approach.from_config_dict][SRToolkit.approaches.sr_approach.SR_approach.from_config_dict]
    / [SRCallbacks.from_config_dict][SRToolkit.evaluation.callbacks.SRCallbacks.from_config_dict]).
    Run *after* ``_auto_bind`` so bundle paths are
    already rewritten to their installed import prefix.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for cfg in configs:
        for class_path, _ in _class_refs(cfg):
            if class_path in seen:
                continue
            seen.add(class_path)
            if "." not in class_path:  # a built-in alias such as ranking_class "rmse"
                continue
            try:
                _import_class(class_path, "referenced")
            except ImportError:
                missing.append(class_path)
    if missing:
        warnings.warn(
            "[ExperimentGrid] These classes referenced by the grid could not be imported "
            "on this machine:\n  - " + "\n  - ".join(missing) + "\n"
            "Install the providing .srtk bundle(s) before running jobs, or the grid will "
            "fail at run time. If a config has no '_bundle' key, call bind_config(config) first.",
            stacklevel=3,
        )


def _warn_if_outdated(writer_version: str | None) -> None:
    """Warn if the SRToolkit that wrote a grid is newer than the one loading it."""
    if not writer_version:
        return
    try:
        from packaging.version import Version

        from SRToolkit import __version__

        if Version(__version__) < Version(writer_version):
            warnings.warn(
                f"[ExperimentGrid] This grid was written with SRToolkit {writer_version}, "
                f"but {__version__} is installed. Loading may fail or behave unexpectedly.",
                stacklevel=3,
            )
    except Exception:
        # Version comparison is best-effort; never block loading on it.
        pass


def _safe_auto_bind(config: dict) -> dict:
    """``_auto_bind`` that degrades to a warning if the bundle can't be resolved.

    Keeps grid loading soft: a not-yet-installed bundle warns at load (the class stays
    unresolved and the hard failure surfaces at job-run time) instead of crashing
    ``from_dict``.
    """
    try:
        return _auto_bind(config)
    except Exception as exc:  # noqa: BLE001 - missing/!resolvable bundle shouldn't break load
        warnings.warn(
            f"[ExperimentGrid] Could not bind a config to its bundle ({exc}). "
            "Install the providing .srtk bundle before running jobs.",
            stacklevel=3,
        )
        return config


@dataclass
class ExperimentInfo:
    """Metadata for a single experiment run.

    Holds all job-specific information not contained in the dataset or approach
    config: the random seed, the path where the result should be written, how many top
    expressions to keep, and — for approaches with ``adaptation_scope="once"`` — where
    the pre-adapted state is stored.

    [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] builds these in memory inside
    [build_job][SRToolkit.experiments.ExperimentGrid.build_job] — both
    [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] and the ``run_job`` CLI
    derive the seed and result path from the grid plus the (dataset, approach, seed)
    identifiers, so no per-job ``info`` files are written to disk.

    Examples:
        >>> info = ExperimentInfo(seed=42, result_path="/results/exp_42.json")
        >>> info.seed
        42
        >>> info.top_k
        20
        >>> d = info.to_dict()
        >>> ExperimentInfo.from_dict(d) == info
        True

    Attributes:
        seed: Random seed passed to the evaluator and the approach's ``search()`` method.
        result_path: File path where the result JSON will be written.  If a directory is
            passed to [ExperimentJob][SRToolkit.experiments.ExperimentJob], the filename
            ``exp_{seed}.json`` is appended automatically.
        top_k: Number of top-ranked expressions to retain in the result. Default ``20``.
        adapted_state_path: Path to the pre-adapted state for ``"once"``-scope approaches; its
            parent directories are created when the state is saved. ``None`` means the approach
            will adapt from scratch on every run and the state will not be saved.
    """

    seed: int
    result_path: str
    top_k: int = 20
    adapted_state_path: str | None = None

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary.

        Returns:
            A flat dictionary with keys ``seed``, ``result_path``, ``top_k``, and ``adapted_state_path``, suitable for passing to [from_dict][SRToolkit.experiments.ExperimentInfo.from_dict].
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentInfo":
        """Restore an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] from a dictionary
        produced by [to_dict][SRToolkit.experiments.ExperimentInfo.to_dict].

        Args:
            d: Dictionary with keys ``seed``, ``result_path``, ``top_k``, and
                ``adapted_state_path``.

        Returns:
            The reconstructed [ExperimentInfo][SRToolkit.experiments.ExperimentInfo].
        """
        return cls(**d)


class ExperimentJob:
    """A single atomic experiment: one dataset × one approach × one seed.

    An ``ExperimentJob`` is built from three components:

    - **dataset**: the dataset to evaluate on — an ``SR_dataset`` instance, a path to a
      ``SR_dataset.to_dict()`` JSON file, or the dict itself.
    - **approach**: the SR approach — an ``SR_approach`` instance, a path to an
      ``ApproachConfig.to_dict()`` JSON file, or the dict itself.
    - **info**: job metadata — an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] instance,
      a path to an ``ExperimentInfo.to_dict()`` JSON file, or the dict itself.

    For Python use, you can pass instances directly::

        job = ExperimentJob(my_dataset, my_approach,
                            ExperimentInfo(seed=0, result_path="/out/"))
        job.run()

    Within a grid you rarely build these by hand —
    [ExperimentGrid.build_job][SRToolkit.experiments.ExperimentGrid.build_job] assembles
    one from the grid's stored configs, which is also how the ``run_job`` CLI executes a
    single (dataset, approach, seed) triple from just the ``grid.json`` file.

    Attributes:
        dataset_name: Name of the dataset, resolved at construction time.
        approach_name: Name of the approach, resolved at construction time.
        seed: Random seed (from ``info``).
        result_path: File path where the experiment result is saved (from ``info``).
        info: The [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] for this job.
        is_complete: ``True`` if the result file already exists on disk.
    """

    def __init__(
        self,
        dataset: SR_dataset | str | dict,
        approach: SR_approach | str | dict,
        info: ExperimentInfo | str | dict,
        callbacks: SRCallbacks | list[SRCallbacks] | dict | list[dict] | None = None,
    ) -> None:
        """Args:
            dataset: The dataset.  One of:

                - ``SR_dataset`` instance — used directly in memory.
                - ``str`` — path to a JSON produced by ``SR_dataset.to_dict()``.
                - ``dict`` — the ``SR_dataset.to_dict()`` output directly.

            approach: The SR approach.  One of:

                - ``SR_approach`` instance — used directly in memory.
                - ``str`` — path to a JSON produced by ``ApproachConfig.to_dict()``.
                - ``dict`` — the ``ApproachConfig.to_dict()`` output directly.

            info: Job metadata.  One of:

                - [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] instance.
                - ``str`` — path to a JSON produced by ``ExperimentInfo.to_dict()``.
                - ``dict`` — the ``ExperimentInfo.to_dict()`` output directly.

            callbacks: Optional callbacks to attach during
                [run][SRToolkit.experiments.ExperimentJob.run].  Accepts a single
                [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks] instance, a list
                of instances, a single serialised callback dict, or a list of dicts.
                Instances are serialised to dicts immediately so that
                [run][SRToolkit.experiments.ExperimentJob.run] always reconstructs fresh
                instances (no shared state between jobs).  Defaults to ``None``.

        Raises:
            ValueError: If ``info.result_path`` is not a directory and does not end
                with ``.json``.
        """
        if isinstance(info, ExperimentInfo):
            self.info = info
        elif isinstance(info, str):
            with open(info) as f:
                self.info = ExperimentInfo.from_dict(json.load(f))
        else:
            self.info = ExperimentInfo.from_dict(dict(info))

        if isinstance(dataset, SR_dataset):
            self._dataset_instance: SR_dataset | None = dataset
            self._dataset_dict: dict | None = None
            self.dataset_name: str = dataset.dataset_name
        elif isinstance(dataset, str):
            with open(dataset) as f:
                self._dataset_dict = json.load(f)
            self._dataset_instance = None
            self.dataset_name = self._dataset_dict.get("dataset_name", "unnamed")
        else:
            self._dataset_dict = dict(dataset)
            self._dataset_instance = None
            self.dataset_name = self._dataset_dict.get("dataset_name", "unnamed")

        if isinstance(approach, SR_approach):
            self._approach_instance: SR_approach | None = approach
            self._approach_dict: dict | None = None
            self.approach_name: str = approach.name
        elif isinstance(approach, str):
            with open(approach) as f:
                self._approach_dict = json.load(f)
            self._approach_instance = None
            self.approach_name = self._approach_dict.get("name", "unknown")
        else:
            self._approach_dict = dict(approach)
            self._approach_instance = None
            self.approach_name = self._approach_dict.get("name", "unknown")

        self.seed = self.info.seed

        if callbacks is None:
            self._callback_configs: list[dict] | None = None
        elif isinstance(callbacks, list):
            self._callback_configs = [cb if isinstance(cb, dict) else cb.to_dict() for cb in callbacks]
        else:
            self._callback_configs = [callbacks if isinstance(callbacks, dict) else callbacks.to_dict()]

        if os.path.isdir(self.info.result_path):
            self.result_path = os.path.join(self.info.result_path, f"exp_{self.seed}.json")
        else:
            _, extension = os.path.splitext(self.info.result_path)
            if extension.lower() != ".json":
                raise ValueError(
                    f"Invalid file extension '{extension}'. SR_results can only be saved as '.json' files."
                )
            self.result_path = self.info.result_path

    @property
    def is_complete(self) -> bool:
        """``True`` if the result file at ``result_path`` already exists on disk."""
        return os.path.exists(self.result_path)

    def run(self) -> None:
        """Execute this experiment and save the result to ``result_path``.

        Handles adaptation according to
        [SR_approach.adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope]:

        - ``"never"``: no adaptation.
        - ``"once"``: loads pre-adapted state from
          [ExperimentInfo][SRToolkit.experiments.ExperimentInfo]'s ``adapted_state_path``
          if a path is set and the file exists, otherwise adapts (and saves if a path is set).
        - ``"experiment"``: adapts fresh every run.

        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt] gets the dataset's ``X``,
        ``background_knowledge`` and ``required_constraints``.
        The result is saved via
        [SR_results.save][SRToolkit.evaluation.results.SR_results.save] to ``result_path``.
        """
        # The constructor sets exactly one of the instance and the dict of each.
        if self._dataset_instance is not None:
            dataset = self._dataset_instance
        else:
            assert self._dataset_dict is not None
            dataset = SR_dataset.from_dict(self._dataset_dict)

        if self._approach_instance is not None:
            approach = self._approach_instance
        else:
            assert self._approach_dict is not None
            approach = SR_approach.from_config_dict(self._approach_dict)

        # Honors the contract for callers passing a live instance; a no-op for the
        # grid's own from-config reconstruction (nothing has accumulated yet).
        approach.prepare()

        if approach.adaptation_scope == "once":
            state_path = self.info.adapted_state_path
            if state_path is None:
                approach.adapt(dataset.X, dataset.background_knowledge, dataset.required_constraints)
            else:
                if os.path.exists(state_path):
                    approach.load_adapted_state(state_path)
                else:
                    approach.adapt(dataset.X, dataset.background_knowledge, dataset.required_constraints)
                    dir_name = os.path.dirname(state_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    approach.save_adapted_state(state_path)
        elif approach.adaptation_scope == "experiment":
            approach.adapt(dataset.X, dataset.background_knowledge, dataset.required_constraints)

        evaluator = dataset.create_evaluator(seed=self.info.seed)
        evaluator._experiment_id = f"{self.dataset_name}_{self.approach_name}_{self.info.seed}"
        dispatcher: CallbackDispatcher | None = None
        if self._callback_configs:
            cbs = [SRCallbacks.from_config_dict(d) for d in self._callback_configs]
            dispatcher = CallbackDispatcher(callbacks=cbs)
            evaluator.register_callbacks(dispatcher)

        event = ExperimentEvent(
            dataset_name=self.dataset_name,
            approach_name=self.approach_name,
            max_evaluations=evaluator.max_evaluations,
            success_threshold=evaluator.success_threshold,
            seed=self.info.seed,
        )
        if dispatcher is not None:
            dispatcher.on_experiment_start(event)

        start_time = time.monotonic()
        approach.search(evaluator, self.info.seed)
        elapsed = time.monotonic() - start_time

        results = evaluator.get_results(self.approach_name, self.info.top_k)
        results.results[-1].wall_time = elapsed
        if dispatcher is not None:
            dispatcher.on_experiment_end(event, results.results[-1])
        results.save(self.result_path)

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "pending"
        return (
            f"ExperimentJob(dataset={self.dataset_name!r}, approach={self.approach_name!r}, "
            f"seed={self.seed}, status={status!r})"
        )


class ExperimentGrid:
    """A grid of symbolic regression experiments over datasets, approaches and seeds.

    Each experiment is an independent
    [ExperimentJob][SRToolkit.experiments.ExperimentJob] that runs one approach on one
    dataset with one seed. Jobs can be executed locally (iterate and call
    ``.run()``) or on HPC clusters (generate a commands file with
    [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands]).

    The grid is written to ``results_dir/grid.json`` by
    [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint] and reloaded with
    [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict]; share it with
    [share.save][SRToolkit.share.save]. Results are saved per-experiment to
    ``results_dir/{dataset}/{approach}/exp_{seed}.json``, so parallel workers never write to
    the same file.

    **Seed scheme**: job *i* (0-indexed) receives ``seed = initial_seed + i``.

    Examples:
        >>> from SRToolkit.dataset import Nguyen
        >>> from SRToolkit.approaches import ProGED
        >>> from SRToolkit.experiments import ExperimentGrid
        >>> bench = Nguyen()
        >>> approach = ProGED()
        >>> grid = ExperimentGrid(bench, approach, num_experiments=3,
        ...                       results_dir="/tmp/sr_run")  # doctest: +SKIP

    Args:
        datasets: One or more datasets to run experiments on. Accepts a single
            [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset], a
            [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] (all
            datasets in the benchmark are included), or a list containing
            ``SR_dataset``-s and ``SR_benchmark``-s.
        approaches: One or more SR approaches.  Accepts a single
            [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] or a list of them.
        num_experiments: Number of independent experiments per (dataset, approach) pair.
        results_dir: Root directory where all results and grid metadata are stored.
        initial_seed: Seed for the first experiment.  Subsequent experiments use
            ``initial_seed + 1``, ``initial_seed + 2``, etc.
        top_k: Number of top expressions to highlight per experiment.
        adapted_states: Optional mapping ``{approach_name: {dataset_name: path}}``
            providing paths for pre-adapted state files; relative paths are made absolute against
            the current directory. Jobs for listed (approach, dataset) pairs will load state from
            the given path if it exists, or adapt and save to it otherwise (creating its parent
            directories). Pairs not listed will adapt on every run without saving. Several datasets may deliberately share one
            ``path`` for approaches whose adaptation depends only on what those datasets share
            (e.g. EDHiE, for datasets with the same symbol library, reference grammar and
            required constraints): the path is then adapted once on the first dataset listed for
            it and the others reuse that state. An informational warning lists any such
            shared paths so an accidental collision can be spotted; give each dataset a
            distinct ``path`` if adaptation must differ per dataset.
        callbacks: Optional callback or list of callbacks forwarded to every job
            created by [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs].
            Stored as configs, so each job builds fresh instances (no shared state between
            jobs) and they travel inside ``grid.json``, from which
            [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] rebuilds them.
    """

    def __init__(
        self,
        datasets: SR_dataset | list[SR_dataset | SR_benchmark] | SR_benchmark,
        approaches: SR_approach | list[SR_approach],
        num_experiments: int,
        results_dir: str,
        initial_seed: int = 0,
        top_k: int = 20,
        adapted_states: dict[str, dict[str, str]] | None = None,
        callbacks: SRCallbacks | list[SRCallbacks] | None = None,
    ) -> None:
        self.num_experiments = num_experiments
        self.results_dir = results_dir
        self.initial_seed = initial_seed
        self.top_k = top_k
        self._adapted_states: dict[str, dict[str, str]] = {
            approach: {dataset: os.path.abspath(path) for dataset, path in paths.items()}
            for approach, paths in (adapted_states or {}).items()
        }

        if callbacks is None:
            self.callback_configs: list[dict] | None = None
        elif isinstance(callbacks, list):
            self.callback_configs = [cb.to_dict() for cb in callbacks]
        else:
            self.callback_configs = [callbacks.to_dict()]

        # Build approach configs (plain serialisable dicts, no instance caching)
        self.approach_configs: list[dict] = []
        if isinstance(approaches, SR_approach):
            approaches = [approaches]
        for approach in approaches:
            self.add_approach(approach)

        # Serialise all datasets eagerly into config dicts (no arrays embedded)
        self.datasets: dict[str, dict] = dict()
        if isinstance(datasets, list):
            for ds in datasets:
                self.add_dataset(ds)
        else:
            self.add_dataset(datasets)

    def add_approach(self, approach: SR_approach) -> None:
        """Add an approach to the grid.

        The approach's config is serialised immediately (its ``adaptation_scope`` is
        recorded alongside) so the grid never caches a live instance. Safe to call after
        [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] to extend a loaded grid.

        Args:
            approach: The [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] to add.

        Raises:
            ValueError: If an approach with the same ``name`` is already in the grid
                (names map to result subdirectories, so duplicates would collide).
        """
        cfg = approach.config.to_dict()
        cfg["adaptation_scope"] = approach.adaptation_scope
        name = cfg["name"]
        if any(c["name"] == name for c in self.approach_configs):
            raise ValueError(
                f"[ExperimentGrid] Duplicate approach name '{name}'. "
                f"Rename one of the approaches before adding it (e.g. set its config name)."
            )
        self.approach_configs.append(cfg)

    def add_dataset(self, dataset: SR_dataset | SR_benchmark) -> None:
        """Add a dataset, or every dataset in a benchmark, to the grid.

        Each dataset is serialised immediately to its config dict (no data arrays
        embedded). Safe to call after
        [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] to extend a loaded grid.

        Args:
            dataset: An [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] or an
                [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] (all of its
                datasets are added).

        Raises:
            ValueError: If a dataset name already exists in the grid, or if ``dataset``
                is neither an ``SR_dataset`` nor an ``SR_benchmark``.
        """
        if isinstance(dataset, SR_benchmark):
            for name in dataset.list_datasets(verbose=False):
                self._add_one_dataset(dataset.create_dataset(name), name)
        elif isinstance(dataset, SR_dataset):
            self._add_one_dataset(dataset, dataset.dataset_name)
        else:
            raise ValueError(f"[ExperimentGrid] datasets must be an SR_dataset or SR_benchmark, got {type(dataset)}")

    def _add_one_dataset(self, ds: SR_dataset, name: str) -> None:
        if name in self.datasets:
            raise ValueError(
                f"[ExperimentGrid] Duplicate dataset name '{name}'. "
                f"Rename one of the datasets before passing it to ExperimentGrid "
                f"(e.g. set ds.dataset_name = 'new_name')."
            )
        self.datasets[name] = ds.to_dict()

    def _get_adapted_state_ref_path(self, approach_name: str, dataset_name: str) -> str | None:
        """Return the pickle path for an approach × dataset adapted state."""
        return self._adapted_states.get(approach_name, {}).get(dataset_name)

    def materialize_data(self) -> None:
        """Materialise every dataset's data into the local cache, once, in this process.

        Reconstructing each dataset via
        [SR_dataset.from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict] triggers
        its ``data_source`` to download or sample the arrays into the shared data cache (a
        no-op for datasets already cached). Because this runs single-process, it is
        race-free — the intended counterpart to running it before parallel workers, which
        then only *read* the cache rather than each materialising the same dataset.

        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] calls this by
        default, so the data is obtained once on the machine that writes the commands
        (which, on a cluster, is typically the only node with network access for
        ``UrlSource`` datasets).

        Raises:
            FileNotFoundError: If a ``null``-source dataset has no data in the local cache
                (there is nothing to download or sample, so it cannot be materialised here).
        """
        for dataset_name in self.datasets:
            SR_dataset.from_dict(self.datasets[dataset_name])

    def _pairs_needing_adaptation(self) -> "list[tuple[str, str]]":
        """Return ``(approach_name, dataset_name)`` pairs whose ``"once"`` state is absent.

        A pair qualifies when the approach is ``adaptation_scope="once"``, has a state path
        registered in the ``adapted_states`` mapping, and that state file does not yet exist
        on disk.

        When several pairs map to the *same* ``state_path``, only the first is returned and an
        informational warning is emitted: the path is adapted once on the first dataset listed
        for it and the rest reuse the result. This is intended for approaches whose adaptation
        depends only on what those datasets share (e.g. EDHiE, for datasets with the same symbol
        library, reference grammar and required constraints); give each dataset a distinct
        ``state_path`` if adaptation must differ per dataset.
        """
        pairs: "list[tuple[str, str]]" = []
        seen_paths: dict[str, "tuple[str, str]"] = {}
        conflicts: "list[str]" = []
        for approach_config in self.approach_configs:
            if approach_config.get("adaptation_scope", "never") != "once":
                continue
            approach_name = approach_config["name"]
            for dataset_name, state_path in self._adapted_states.get(approach_name, {}).items():
                if os.path.exists(state_path):
                    continue
                if state_path in seen_paths:
                    kept_approach, kept_dataset = seen_paths[state_path]
                    conflicts.append(
                        f"  - {state_path!r}: ({approach_name!r}, {dataset_name!r}) reuses the "
                        f"state adapted for ({kept_approach!r}, {kept_dataset!r})"
                    )
                    continue
                seen_paths[state_path] = (approach_name, dataset_name)
                pairs.append((approach_name, dataset_name))
        if conflicts:
            warnings.warn(
                "[ExperimentGrid] Some datasets share an adapted-state path, so each such path is "
                "adapted only once — on the first dataset listed for it — and the rest reuse that "
                "state. This is the intended setup for approaches whose adaptation depends only on what "
                "those datasets share (e.g. EDHiE with the same symbol library, reference grammar and "
                "required constraints):\n" + "\n".join(conflicts) + "\n"
                "If adaptation should instead differ per dataset, give each dataset a distinct "
                "'state_path' in the 'adapted_states' mapping.",
                stacklevel=2,
            )
        return pairs

    def adapt_one(self, approach_name: str, dataset_name: str, force: bool = False) -> None:
        """Adapt a single (approach, dataset) pair and persist its state.

        This is the unit the per-pair adaptation commands in ``prepare.txt`` invoke (via the
        ``adapt`` CLI with ``--dataset``/``--approach``), so independent pairs can adapt in
        parallel across a cluster. It loads the dataset, calls
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt] with the dataset's ``X``,
        background knowledge and required constraints, and saves the state via
        [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state].

        Silently returns (no-op) when there is nothing to do: the approach is not
        ``"once"``-scope, or has no registered state path for this dataset. By default it also
        no-ops when the state file already exists; pass ``force=True`` to re-adapt and overwrite
        it (e.g. after the adaptation logic or the dataset changed, or to replace a corrupt
        state). Because the dataset is named explicitly here, ``force`` is unambiguous even when
        several datasets share one state path — it adapts on *this* dataset and overwrites.

        Args:
            approach_name: Name of an approach in this grid.
            dataset_name: Name of a dataset in this grid.
            force: Re-adapt and overwrite an existing state file instead of skipping it.
                Defaults to ``False``.

        Raises:
            ValueError: If ``approach_name`` is not in the grid.
            KeyError: If ``dataset_name`` is not in the grid.
        """
        try:
            approach_config = next(c for c in self.approach_configs if c["name"] == approach_name)
        except StopIteration:
            raise ValueError(f"[ExperimentGrid] No approach named {approach_name!r} in this grid.") from None
        if approach_config.get("adaptation_scope", "never") != "once":
            return
        state_path = self._get_adapted_state_ref_path(approach_name, dataset_name)
        if state_path is None or (os.path.exists(state_path) and not force):
            return
        dataset = SR_dataset.from_dict(self.datasets[dataset_name])
        approach = SR_approach.from_config_dict(approach_config)
        approach.adapt(dataset.X, dataset.background_knowledge, dataset.required_constraints)
        dir_name = os.path.dirname(state_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        approach.save_adapted_state(state_path)

    def adapt_if_missing(self):
        """Pre-adapt all ``adaptation_scope="once"`` approaches where the state file is absent.

        For each (approach, dataset) pair whose state file does not yet exist on disk,
        this method loads the dataset, calls
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt] once with the dataset's ``X``,
        background knowledge and required constraints, then persists the
        state via
        [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state].
        Pairs whose state file already exists are skipped.

        Approaches whose ``adaptation_scope`` is not ``"once"``, or that have no entry in
        the ``adapted_states`` mapping passed at construction, are skipped entirely.

        This adapts every pair sequentially in this process. To distribute the work, use
        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands], which emits one
        [adapt_one][SRToolkit.experiments.ExperimentGrid.adapt_one] command per pair into
        ``prepare.txt``.
        """
        for approach_name, dataset_name in self._pairs_needing_adaptation():
            self.adapt_one(approach_name, dataset_name)

    def build_job(self, dataset_name: str, approach_name: str, seed: int) -> ExperimentJob:
        """Construct the single [ExperimentJob][SRToolkit.experiments.ExperimentJob] for one
        (dataset, approach, seed) triple, using this grid's stored configs.

        This is the unit of work both
        [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] and the
        ``run_job`` CLI build from, so a worker only needs the grid file plus the three
        identifiers — no per-job config files on disk.

        Args:
            dataset_name: Name of a dataset in this grid.
            approach_name: Name of an approach in this grid.
            seed: Random seed for the run.

        Returns:
            The corresponding [ExperimentJob][SRToolkit.experiments.ExperimentJob].

        Raises:
            KeyError: If ``dataset_name`` is not in the grid.
            ValueError: If ``approach_name`` is not in the grid.
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"[ExperimentGrid] No dataset named {dataset_name!r} in this grid.")
        dataset_dict = self.datasets[dataset_name]
        try:
            approach_config = next(c for c in self.approach_configs if c["name"] == approach_name)
        except StopIteration:
            raise ValueError(f"[ExperimentGrid] No approach named {approach_name!r} in this grid.") from None
        info = ExperimentInfo(
            seed=seed,
            result_path=os.path.join(self.results_dir, dataset_name, approach_name, f"exp_{seed}.json"),
            top_k=self.top_k,
            adapted_state_path=self._get_adapted_state_ref_path(approach_name, dataset_name),
        )
        return ExperimentJob(dataset=dataset_dict, approach=approach_config, info=info, callbacks=self.callback_configs)

    def create_jobs(self, skip_completed: bool = True) -> list[ExperimentJob]:
        """Return the list of [ExperimentJob][SRToolkit.experiments.ExperimentJob] instances for
        this grid.

        Does **not** trigger adaptation — call
        [adapt_if_missing][SRToolkit.experiments.ExperimentGrid.adapt_if_missing] first if any
        approach has ``adaptation_scope="once"``. It also does **not** materialise data:
        running the returned jobs sequentially warms the cache lazily on first use, but if you
        dispatch them in parallel yourself, call
        [materialize_data][SRToolkit.experiments.ExperimentGrid.materialize_data] first so
        workers don't race to materialise the same dataset.
        ([save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] does both for you.)

        Args:
            skip_completed: If ``True`` (default), omit jobs whose result file
                (``exp_{seed}.json``) already exists on disk.

        Returns:
            List of jobs, one per (dataset, approach, seed) triple that has not yet completed.
        """
        jobs: list[ExperimentJob] = []
        for approach_config in self.approach_configs:
            for dataset_name in self.datasets:
                for i in range(self.num_experiments):
                    job = self.build_job(dataset_name, approach_config["name"], self.initial_seed + i)
                    if skip_completed and job.is_complete:
                        continue
                    jobs.append(job)
        return jobs

    def save_commands(
        self,
        path: str,
        python_executable: str = "python",
        skip_completed: bool = True,
        materialize_data: bool = True,
    ) -> str | None:
        """Write the experiment commands file (and, if needed, a ``prepare`` commands file).

        Calls [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint] first to persist the grid
        to ``results_dir/grid.json``. Every command line references that single file plus
        the identifiers needed to rebuild the work — no per-dataset, per-approach, or
        per-job files are written.

        The output is split into (up to) two files so a run is correct on any platform
        without scheduler-specific dependency tricks — **run the prepare file to completion,
        then the experiments file**:

        - ``path`` — one ``run_job`` line per pending (dataset, approach, seed) triple::

              python -m SRToolkit.experiments run_job \\
                  --grid /path/grid.json --dataset NG-1 --approach ProGED --seed 0

        - ``prepare_<name>`` (sibling of ``path``) — written **only** when one or more
          ``adaptation_scope="once"`` approaches still need state, one parallel-safe line
          per (approach, dataset) pair::

              python -m SRToolkit.experiments adapt \\
                  --grid /path/grid.json --dataset NG-1 --approach ProGED

        Data is materialised, once, in this process (``materialize_data=True``):
        every dataset's ``data_source`` is resolved into the shared cache before any command
        runs. This is single-process and therefore race-free, fetches each ``UrlSource``
        exactly once, and happens on the machine writing the commands — typically the only
        node with network access on a cluster. Workers (adaptation and experiments) then
        only *read* the warm cache.

        Args:
            path: File path to write the experiment commands to.
            python_executable: Python executable to use in the commands.
            skip_completed: If ``True`` (default), omit already-completed jobs.
            materialize_data: If ``True`` (default), resolve every dataset's data into the
                local cache before writing commands. Set ``False`` to only (re)generate the
                command text without touching data (e.g. the cache is already warm, or this
                machine cannot reach the data).

        Returns:
            The path to the prepare commands file if one was written, otherwise ``None``.
        """
        self.checkpoint()
        if materialize_data:
            self.materialize_data()
        grid_path = shlex.quote(os.path.join(self.results_dir, "grid.json"))

        prepare_path: str | None = None
        prepare_pairs = self._pairs_needing_adaptation()
        if prepare_pairs:
            abs_path = os.path.abspath(path)
            prepare_path = os.path.join(os.path.dirname(abs_path), "prepare_" + os.path.basename(abs_path))
            prepare_lines = [
                "# Adaptation prerequisites — run ALL of these to completion before the experiment commands.",
                "# Lines are independent and may run in parallel.",
            ]
            prepare_lines += [
                f"{python_executable} -m SRToolkit.experiments adapt "
                f"--grid {grid_path} "
                f"--dataset {shlex.quote(dataset_name)} "
                f"--approach {shlex.quote(approach_name)}"
                for approach_name, dataset_name in prepare_pairs
            ]

        header = [f"# results_dir: {self.results_dir}"]
        if prepare_path is not None:
            header.append(f"# Run '{os.path.basename(prepare_path)}' to completion FIRST, then these commands.")
        lines = list(header)
        for approach_config in self.approach_configs:
            for dataset_name in self.datasets:
                approach_name = approach_config["name"]
                for i in range(self.num_experiments):
                    seed = self.initial_seed + i
                    if skip_completed and self.build_job(dataset_name, approach_name, seed).is_complete:
                        continue
                    lines.append(
                        f"{python_executable} -m SRToolkit.experiments run_job "
                        f"--grid {grid_path} "
                        f"--dataset {shlex.quote(dataset_name)} "
                        f"--approach {shlex.quote(approach_name)} "
                        f"--seed {seed}"
                    )

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        if prepare_path is not None:
            with open(prepare_path, "w") as f:
                f.write("\n".join(prepare_lines) + "\n")
        return prepare_path

    def progress(self) -> None:
        """Print a dataset × approach progress table to stdout.

        Each cell shows ``done/total`` experiments completed for that pair, based on
        the presence of the per-experiment ``exp_{seed}.json`` files on disk.

        Example output::

            Dataset        ProGED    EDHiE
            -----------   -------   ------
            NG-1             5/5      3/5
            NG-2             2/5      0/5
        """
        dataset_names = list(self.datasets.keys())
        approach_names = [cfg["name"] for cfg in self.approach_configs]

        total_str = str(self.num_experiments)
        ds_w = max(len(n) for n in dataset_names + ["Dataset"]) + 2
        col_w = max(len(n) for n in approach_names + [f"{total_str}/{total_str}"]) + 2

        header = f"{'Dataset':<{ds_w}}" + "".join(f"{ap:>{col_w}}" for ap in approach_names)
        separator = "-" * ds_w + "".join("-" * col_w for _ in approach_names)
        print(header)
        print(separator)

        for dataset_name in dataset_names:
            row = f"{dataset_name:<{ds_w}}"
            for approach_name in approach_names:
                done = sum(
                    1
                    for i in range(self.num_experiments)
                    if os.path.exists(
                        os.path.join(
                            self.results_dir,
                            dataset_name,
                            approach_name,
                            f"exp_{self.initial_seed + i}.json",
                        )
                    )
                )
                row += f"{done}/{self.num_experiments}".rjust(col_w)
            print(row)

    def load_results(self, dataset_name: str, approach_name: str) -> SR_results:
        """Load and merge all completed per-experiment results for a (dataset, approach) pair.

        Examples:
            >>> results = grid.load_results("Nguyen-1", "ProGED")  # doctest: +SKIP
            >>> len(results)  # number of completed experiments  # doctest: +SKIP
            5

        Args:
            dataset_name: Name of the dataset.
            approach_name: Name of the approach.

        Returns:
            An [SR_results][SRToolkit.evaluation.results.SR_results] object containing one [EvalResult][SRToolkit.utils.types.EvalResult] per completed experiment.  Returns an empty ``SR_results`` if no experiments have completed yet.
        """
        merged = SR_results()
        for i in range(self.num_experiments):
            seed = self.initial_seed + i
            result_path = os.path.join(self.results_dir, dataset_name, approach_name, f"exp_{seed}.json")
            if os.path.exists(result_path):
                merged += SR_results.load(result_path)
        return merged

    def to_dict(self) -> dict:
        """Serialize the grid to a single self-contained, JSON-safe dict — the *recipe*.

        The returned dict inlines every dataset config, approach config, and callback
        config alongside the run metadata (``num_experiments``, ``initial_seed``,
        ``top_k``). It is the portable, one-file form of the grid: pass it (or a JSON
        file holding it) straight to [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict].

        Unlike [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint], it omits the
        machine-local, absolute ``adapted_states`` paths; ``results_dir`` is supplied at load
        time. The per-approach ``adaptation_scope`` travels inside each approach config, so a
        recipient re-adapts to their own paths.

        Dataset configs are the full per-dataset recipe (``data_source`` + samplers +
        ground truth), not bare references, and contain no data arrays — those are
        regenerated or downloaded from each dataset's ``data_source`` on first use. A
        ``null``-source dataset's data must already be present in the recipient's cache;
        ship it with [share.save][SRToolkit.share.save] otherwise.

        Returns:
            A JSON-safe dict with ``format_version`` ``2``, suitable for ``json.dump``.
        """
        from SRToolkit import __version__

        return {
            "format_version": 2,
            "type": "ExperimentGrid",
            "srtk_version": __version__,
            "num_experiments": self.num_experiments,
            "initial_seed": self.initial_seed,
            "top_k": self.top_k,
            "datasets": {name: dict(cfg) for name, cfg in self.datasets.items()},
            "approaches": [dict(cfg) for cfg in self.approach_configs],
            "callbacks": ([dict(cfg) for cfg in self.callback_configs] if self.callback_configs is not None else None),
        }

    @classmethod
    def from_dict(
        cls,
        config: dict | str | Path,
        results_dir: str | None = None,
        adapted_states: dict[str, dict[str, str]] | None = None,
    ) -> "ExperimentGrid":
        """Rebuild a grid from a recipe dict or a JSON file.

        ``*_class`` paths pointing at an installed ``.srtk`` bundle resolve automatically.
        Datasets and approaches are built lazily when jobs run; a ``*_class`` path that cannot
        be imported triggers a warning here and fails when the job runs. Load ``.zip`` archives
        with [share.load][SRToolkit.share.load].

        Args:
            config: The recipe. One of:

                - ``dict`` — the [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict]
                  output directly. ``results_dir`` is then **required**.
                - ``str`` / ``Path`` — path to a JSON file holding that dict, e.g. the
                  ``grid.json`` written by
                  [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint]. The
                  directory containing the file becomes the grid's ``results_dir``
                  unless ``results_dir`` is given explicitly.

            results_dir: Where results are read from and written to. Overrides the
                file-anchored default; required when ``config`` is a dict.
            adapted_states: As in the [class docstring][SRToolkit.experiments.ExperimentGrid].
                ``None`` uses the paths stored in ``config`` by
                [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint], if any.

        Returns:
            A fully configured ``ExperimentGrid``.

        Raises:
            ValueError: If ``config`` is a ``.zip`` path or not an ``ExperimentGrid`` recipe, its
                ``format_version`` is unsupported, or ``results_dir`` is omitted when
                loading from a dict.
        """
        default_results_dir: str | None = None
        if isinstance(config, (str, Path)):
            if str(config).endswith(".zip"):
                raise ValueError(
                    "[ExperimentGrid.from_dict] Received a '.zip' path. Load archives with SRToolkit.share.load(path)."
                )
            file_path = os.path.abspath(str(config))
            with open(file_path) as f:
                d = json.load(f)
            default_results_dir = os.path.dirname(file_path)
        else:
            d = dict(config)

        if d.get("type") != "ExperimentGrid":
            raise ValueError(
                f"[ExperimentGrid.from_dict] Config is not an ExperimentGrid recipe (type={d.get('type')!r})."
            )
        fmt = d.get("format_version")
        if fmt != 2:
            raise ValueError(f"[ExperimentGrid.from_dict] Unsupported format_version: {fmt!r}. Expected 2.")
        _warn_if_outdated(d.get("srtk_version"))

        resolved_results_dir = results_dir if results_dir is not None else default_results_dir
        if resolved_results_dir is None:
            raise ValueError(
                "[ExperimentGrid.from_dict] results_dir is required when loading from a dict "
                "(there is no file location to anchor it to)."
            )

        grid = cls.__new__(cls)
        grid.num_experiments = d["num_experiments"]
        grid.results_dir = os.path.abspath(resolved_results_dir)
        grid.initial_seed = d["initial_seed"]
        grid.top_k = d["top_k"]
        states = adapted_states if adapted_states is not None else d.get("adapted_states", {})
        grid._adapted_states = {
            approach: {dataset: os.path.abspath(path) for dataset, path in paths.items()}
            for approach, paths in states.items()
        }
        grid.datasets = {name: _safe_auto_bind(cfg) for name, cfg in d["datasets"].items()}
        grid.approach_configs = [_safe_auto_bind(cfg) for cfg in d["approaches"]]
        callbacks = d.get("callbacks")
        grid.callback_configs = [_safe_auto_bind(cfg) for cfg in callbacks] if callbacks is not None else None

        _warn_missing_class_deps(list(grid.datasets.values()) + grid.approach_configs + (grid.callback_configs or []))
        return grid

    def checkpoint(self) -> None:
        """Write the grid to ``results_dir/grid.json``.

        Resume it with [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict]. Creates
        ``results_dir`` if needed and overwrites an existing ``grid.json``. The file is the [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] recipe plus the
        machine-local ``adapted_states`` paths. To share a grid with someone else, use
        [share.save][SRToolkit.share.save].
        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] calls this.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        payload = self.to_dict()
        if self._adapted_states:
            payload["adapted_states"] = self._adapted_states
        with open(os.path.join(self.results_dir, "grid.json"), "w") as f:
            json.dump(payload, f, indent=2)

    def __repr__(self) -> str:
        ds_count = len(self.datasets)
        ap_count = len(self.approach_configs)
        total = ds_count * ap_count * self.num_experiments
        return (
            f"ExperimentGrid({ds_count} datasets × {ap_count} approaches × "
            f"{self.num_experiments} experiments = {total} jobs, "
            f"results_dir={self.results_dir!r})"
        )
