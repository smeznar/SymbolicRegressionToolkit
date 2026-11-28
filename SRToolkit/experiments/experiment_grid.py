"""Job-based experiment runner for symbolic regression experiments.

Provides three public classes:

- [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] — lightweight metadata (seed, paths) for a
  single run.
- [ExperimentJob][SRToolkit.experiments.ExperimentJob] — one atomic experiment: a single dataset ×
  approach × seed triple. Can be run in-process or dispatched to a CLI worker.
- [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] — a full cross-product grid of datasets and
  approaches. Manages serialization, parallelism via HPC command files, progress tracking, and
  result loading.

Each job keeps its files in one directory (in a grid, ``results_dir/{dataset}/{approach}/``): its
result ``exp_{seed}.json``, which marks it done; its checkpoint ``exp_{seed}.sqlite`` while it is
unfinished (see [Checkpoints][SRToolkit.evaluation.evaluator]); and the lock file
``exp_{seed}.lock``, locked while it runs and deleted once its result is saved.
[ExperimentJob.status][SRToolkit.experiments.ExperimentJob.status] reads the job's state from
them, and running an unfinished job again resumes it.
[ExperimentJob.extend][SRToolkit.experiments.ExperimentJob.extend] continues a job's run with new
limits (a larger budget, a longer time limit); a done job can be extended only if its checkpoint
was kept (``keep_checkpoint=True``), and stays done with its old result until the extension
finishes. The grid and ``grid.json`` keep the limits the grid was created with. Reading a running
job's checkpoint from another node over a network filesystem is not guaranteed to work.
"""

import contextlib
import dataclasses
import json
import os
import shlex
import sqlite3
import warnings
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import fcntl
except ImportError:  # e.g. Windows: jobs are not locked
    fcntl = None  # type: ignore[assignment]

from SRToolkit.approaches.approach import Approach
from SRToolkit.bundle._relocate import _auto_bind, _class_refs, _import_class
from SRToolkit.dataset.benchmark import Benchmark
from SRToolkit.dataset.dataset import Dataset
from SRToolkit.evaluation.callbacks import Callback, CallbackDispatcher, ExperimentEvent
from SRToolkit.evaluation.model_store import (
    ModelStore,
    _checkpoint_reference,
    _compact_checkpoint,
    _remove_checkpoint,
    _run_finished,
)
from SRToolkit.evaluation.results import Results

# Lock files this process holds. status() must not open them: on NFS, flock is emulated by fcntl
# locks, and closing any descriptor of a file drops all of this process's locks on it.
_held_locks: set[str] = set()


def _warn_missing_class_deps(configs: list[dict]) -> None:
    # One warning listing the *_class paths in configs that cannot be imported; loading stays lazy,
    # so the job fails when it runs. Call after binding, so bundle paths are already rewritten.
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


def _bind_or_warn(config: dict) -> dict:
    # _auto_bind, but a bundle that cannot be resolved only warns: the config stays unbound and the
    # job fails when it runs, instead of from_dict.
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

    [ExperimentGrid.build_job][SRToolkit.experiments.ExperimentGrid.build_job] builds these; you need one
    only to run an [ExperimentJob][SRToolkit.experiments.ExperimentJob] yourself.

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
        result_path: File path where the result JSON will be written. If a directory is
            passed to [ExperimentJob][SRToolkit.experiments.ExperimentJob], the filename
            ``exp_{seed}.json`` is appended automatically.
        top_k: Number of top-ranked expressions to retain in the result. Default ``20``.
        adapted_state_path: Path to the adapted state of a ``"once"``-scope approach; its parent
            directories are created when the state is saved. ``None``: adapt without saving.
        keep_checkpoint: Keep the job's checkpoint after its result is saved, compacted into one
            self-contained file, e.g. to inspect all its models later. Default ``False``: it is
            deleted.
        snapshot_interval: Seconds between the job's snapshots (see
            [Checkpoints][SRToolkit.evaluation.evaluator]); ``None``: only when the approach's
            step loop ends. Default ``300.0``.
        keep_all_models: As for
            [get_results][SRToolkit.evaluation.evaluator.Evaluator.get_results]: ``False``
            saves only the ``top_k`` models and a summary of the run. Default ``True``.
    """

    seed: int
    result_path: str
    top_k: int = 20
    adapted_state_path: str | None = None
    keep_checkpoint: bool = False
    snapshot_interval: float | None = 300.0
    keep_all_models: bool = True

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dictionary.

        Returns:
            A flat dictionary with keys ``seed``, ``result_path``, ``top_k``, ``adapted_state_path``,
            ``keep_checkpoint``, ``snapshot_interval`` and ``keep_all_models``, suitable for passing to
            [from_dict][SRToolkit.experiments.ExperimentInfo.from_dict].
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentInfo":
        """Restore an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] from a dictionary
        produced by [to_dict][SRToolkit.experiments.ExperimentInfo.to_dict].

        Args:
            d: Dictionary with keys ``seed``, ``result_path``, ``top_k``, ``adapted_state_path``
                and, optionally, ``keep_checkpoint`` (default ``False``), ``snapshot_interval``
                (default ``300.0``) and ``keep_all_models`` (default ``True``).

        Returns:
            The reconstructed [ExperimentInfo][SRToolkit.experiments.ExperimentInfo].
        """
        return cls(**d)


class ExperimentJob:
    """A single atomic experiment: one dataset × one approach × one seed.

    Built from a dataset, an approach and an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo],
    each given as an instance, a dict or the path of a JSON file::

        job = ExperimentJob(my_dataset, my_approach,
                            ExperimentInfo(seed=0, result_path="/out/"))
        job.run()

    In a grid, [build_job][SRToolkit.experiments.ExperimentGrid.build_job] builds them.

    Attributes:
        dataset_name: Name of the dataset, resolved at construction time.
        approach_name: Name of the approach, resolved at construction time.
        seed: Random seed (from ``info``).
        result_path: File path where the experiment result is saved (from ``info``).
        checkpoint_path: The job's checkpoint: ``result_path`` with the extension ``.sqlite``.
        info: The [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] for this job.
        is_complete: ``True`` if the result file already exists on disk.
        status: The job's state (see [status][SRToolkit.experiments.ExperimentJob.status]).
    """

    def __init__(
        self,
        dataset: Dataset | str | dict,
        approach: Approach | str | dict,
        info: ExperimentInfo | str | dict,
        callbacks: Callback | list[Callback] | dict | list[dict] | None = None,
    ) -> None:
        """Args:
            dataset: An ``Dataset``, or its ``to_dict()`` as a dict or the path of a JSON file.
            approach: An ``Approach``, or its config's ``to_dict()`` as a dict or the path of a
                JSON file.
            info: An [ExperimentInfo][SRToolkit.experiments.ExperimentInfo], or its ``to_dict()``
                as a dict or the path of a JSON file.
            callbacks: Callbacks for [run][SRToolkit.experiments.ExperimentJob.run]: an
                [Callback][SRToolkit.evaluation.callbacks.Callback], a callback's
                ``to_dict()``, or a list of either. Stored as configs, so each run builds fresh
                instances. ``None`` for none.

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

        if isinstance(dataset, Dataset):
            self._dataset_instance: Dataset | None = dataset
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

        if isinstance(approach, Approach):
            self._approach_instance: Approach | None = approach
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
                raise ValueError(f"Invalid file extension '{extension}'. Results can only be saved as '.json' files.")
            self.result_path = self.info.result_path
        self.checkpoint_path = os.path.splitext(self.result_path)[0] + ".sqlite"
        self._lock_path = os.path.splitext(self.result_path)[0] + ".lock"

    @property
    def is_complete(self) -> bool:
        """``True`` if the result file at ``result_path`` already exists on disk."""
        return os.path.exists(self.result_path)

    @property
    def status(self) -> Literal["done", "running", "not started", "interrupted", "crashed"]:
        """The job's state, read from its files.

        The first that applies (for the files, see
        [experiment_grid][SRToolkit.experiments.experiment_grid]):

        - ``"done"``: its result exists (even while a process holds its lock);
        - ``"running"``: a process holds its lock (see [run][SRToolkit.experiments.ExperimentJob.run]);
        - ``"not started"``: it has no checkpoint;
        - ``"interrupted"``: its checkpoint records a stop by Ctrl-C or SIGTERM;
        - ``"crashed"``: it has another checkpoint: the process was killed or the search raised, its
          result was not saved, or the checkpoint cannot be read.

        Running an interrupted or crashed job resumes it. A done job whose
        [extension][SRToolkit.experiments.ExperimentJob.extend] was interrupted or crashed stays
        ``"done"``. Where files cannot be locked, a running job shows as ``"not started"`` or
        ``"crashed"``.
        """
        if self.is_complete:
            return "done"
        if os.path.abspath(self._lock_path) in _held_locks:
            return "running"
        if fcntl is not None:
            try:
                with open(self._lock_path, "rb") as lock:
                    # Shared: on NFS (flock emulated by fcntl locks) an exclusive lock needs write access.
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)  # released when closed
            except BlockingIOError:
                return "running"
            except OSError:  # no lock file, or no locking
                pass
        if not os.path.exists(self.checkpoint_path):
            return "not started"
        try:
            store = ModelStore(self.checkpoint_path, "read")
            try:
                reason = store.get_header("stop_reason")
            finally:
                store.close()
        except (ValueError, sqlite3.Error):  # not a readable checkpoint
            return "crashed"
        return "interrupted" if reason == "interrupted" else "crashed"

    def run(self, rerun: bool = False) -> None:
        """Execute this experiment and save the result to ``result_path``.

        A job whose result exists does nothing, unless ``rerun``. For the whole run, adaptation
        included, the job holds an exclusive ``fcntl.flock`` on its lock file (``result_path`` with
        the extension ``.lock``), so it never runs in two processes at once. The lock file is
        deleted once the result is saved; after an interrupt or an exception it stays, unlocked.
        Where files cannot be locked (Windows, or a filesystem without ``flock``), it warns with a
        ``UserWarning`` and runs unlocked.

        Adapts as
        [adaptation_scope][SRToolkit.approaches.approach.Approach.adaptation_scope] says
        (``"once"``: [adapt_or_load][SRToolkit.dataset.dataset.Dataset.adapt_or_load] with
        ``info.adapted_state_path``).

        The evaluator keeps its models and trace in the checkpoint at ``checkpoint_path`` and
        continues it from its last snapshot if it exists (see
        [Checkpoints][SRToolkit.evaluation.evaluator]), so running an unfinished job again
        resumes it. An approach whose runs cannot be resumed (see
        [supports_steps][SRToolkit.approaches.approach.Approach.supports_steps]) starts
        its job afresh, with a ``UserWarning``, and replaces the checkpoint. A checkpoint of a
        finished run (e.g. killed while saving its result) gives the result directly, without
        adapting or running anything. The search runs inside
        [stop_on_signals][SRToolkit.evaluation.evaluator.Evaluator.stop_on_signals]:

        - a job that finishes saves its result, and only then deletes its checkpoint (or, with
          ``info.keep_checkpoint``, compacts it into one self-contained file, which the saved
          result then refers to as its [checkpoint][SRToolkit.utils.types.EvalResult]; a result
          whose checkpoint is deleted has no reference);
        - a job stopped by Ctrl-C or SIGTERM keeps its checkpoint, saves no result (a result marks
          the job done), calls ``on_experiment_end`` with its results so far and raises
          ``KeyboardInterrupt``, so a loop or scheduler sees it did not finish (those results
          refer to the checkpoint);
        - a job whose search raises keeps its checkpoint (committed and closed), and the exception
          propagates.

        Args:
            rerun: Run the job afresh: once the dataset and approach are set up (adapted), delete its
                result and checkpoint (the adapted state of a ``"once"`` approach is kept). A setup
                that fails keeps the old result; a new run that does not finish loses it. Default
                ``False``.

        Raises:
            KeyboardInterrupt: If the search was stopped by a signal, after the checkpoint is
                closed.
            RuntimeError: If another process holds the job's lock, or (with ``rerun``) finished the
                job while this one was taking it, before any job file is changed.
            ValueError: If ``info.keep_all_models`` is ``False`` and ``info.top_k <= 0``, before
                any job file is changed; as for
                [Evaluator][SRToolkit.evaluation.evaluator.Evaluator] with
                ``resume=True``, if the checkpoint is from a run with other settings or
                ``info.snapshot_interval`` is not positive.
        """
        info = self.info
        if not info.keep_all_models and info.top_k <= 0:
            raise ValueError(f"[ExperimentJob] keep_all_models=False needs top_k >= 1, got {info.top_k}.")
        if self.is_complete and not rerun:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.result_path)), exist_ok=True)
        with self._locked() as valid:
            if not valid:
                if self.is_complete and not rerun:
                    return
                raise RuntimeError("[ExperimentJob] The job just finished in another process; run it again.")
            if not rerun and self.is_complete:  # finished by another process in the meantime
                if os.path.exists(self._lock_path):
                    os.remove(self._lock_path)
                return

            if not rerun and os.path.exists(self.checkpoint_path) and _run_finished(self.checkpoint_path):
                results = Results.from_checkpoint(
                    self.checkpoint_path, self.approach_name, info.top_k, info.keep_all_models
                )
                self._save_result(results)
                return

            dataset, approach = self._dataset_and_approach()
            if approach.adaptation_scope == "once":
                dataset.adapt_or_load(approach, self.info.adapted_state_path)
            elif approach.adaptation_scope == "experiment":
                approach.adapt(dataset.X, dataset.background_knowledge, dataset.required_constraints)

            if rerun:  # only now that setup succeeded
                if os.path.exists(self.result_path):
                    os.remove(self.result_path)
                _remove_checkpoint(self.checkpoint_path)
            resume = os.path.exists(self.checkpoint_path)
            if resume and not approach.supports_steps:
                warnings.warn(
                    f"[ExperimentJob] {self.approach_name} cannot continue from a snapshot, so its checkpoint "
                    f"{self.checkpoint_path} cannot be resumed; the job starts afresh and replaces it.",
                    UserWarning,
                )
                _remove_checkpoint(self.checkpoint_path)
                resume = False
            evaluator = dataset.create_evaluator(
                seed=self.info.seed,
                checkpoint=self.checkpoint_path,
                resume=resume,
                snapshot_interval=self.info.snapshot_interval,
            )
            try:
                evaluator._experiment_id = f"{self.dataset_name}_{self.approach_name}_{self.info.seed}"
                dispatcher: CallbackDispatcher | None = None
                if self._callback_configs:
                    cbs = [Callback.from_config_dict(d) for d in self._callback_configs]
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

                with evaluator.stop_on_signals():
                    approach.run(evaluator, self.info.seed)
                results = evaluator.get_results(self.approach_name, info.top_k, keep_all_models=info.keep_all_models)
                if info.keep_checkpoint or evaluator.stop_reason == "interrupted":  # the file remains
                    results.results[-1].checkpoint = _checkpoint_reference(evaluator._store, self.checkpoint_path)
            finally:
                evaluator.close()
            if dispatcher is not None:
                dispatcher.on_experiment_end(event, results.results[-1])
            # An interrupted job saves no result: a result marks the job done.
            if evaluator.stop_reason == "interrupted":
                raise KeyboardInterrupt(f"The job was interrupted; its checkpoint is kept at {self.checkpoint_path}.")
            self._save_result(results)

    def extend(
        self,
        max_evaluations: int | None = None,
        success_threshold: float | None = None,
        callbacks: Callback | list[Callback] | None = None,
    ) -> None:
        """Continue the job's run with new limits and save the result.

        The run, done or not, continues from the job's checkpoint as in
        [extend_run][SRToolkit.dataset.dataset.Dataset.extend_run], under the job's lock and
        after adapting as in [run][SRToolkit.experiments.ExperimentJob.run] (``"once"``: with
        ``info.adapted_state_path``). The result replaces the old one; the
        checkpoint is then kept compacted if ``info.keep_checkpoint``, else deleted. An interrupt
        keeps the checkpoint and raises ``KeyboardInterrupt``. A done job keeps its old result, and
        its status ``"done"``, until an ``extend`` finishes, so only ``extend`` continues it; a
        plain ``run`` does nothing.

        Args:
            max_evaluations: As for ``extend_run``.
            success_threshold: As for ``extend_run``.
            callbacks: As for ``extend_run``. The job's own callbacks are not used, also for a job
                that is not done: give new ones, e.g. a ``TimeLimitCallback`` to keep a time limit.

        Raises:
            KeyboardInterrupt: If the run was stopped by a signal.
            RuntimeError: If another process holds the job's lock, or deleted the lock file (by
                finishing the job) while this one was taking it; then extend again.
            ValueError: If the job has no checkpoint: it was never started, or it is done and its
                checkpoint was not kept (rerun it instead). As for ``extend_run``.
        """
        info = self.info
        if not os.path.exists(self.checkpoint_path):
            if self.is_complete:
                raise ValueError(
                    f"[ExperimentJob] {self.result_path} has no checkpoint to extend (keep_checkpoint was off); "
                    "rerun the job instead."
                )
            raise ValueError("[ExperimentJob] The job was never started, so there is nothing to extend; run it.")
        with self._locked() as valid:
            if not valid:
                raise RuntimeError("[ExperimentJob] The job just finished in another process; extend it again.")
            dataset, approach = self._dataset_and_approach()
            results = dataset.extend_run(
                approach,
                self.checkpoint_path,
                max_evaluations=max_evaluations,
                success_threshold=success_threshold,
                callbacks=callbacks,
                top_k=info.top_k,
                keep_all_models=info.keep_all_models,
                snapshot_interval=info.snapshot_interval,
                adaptation_path=info.adapted_state_path,
            )
            if results.results[-1].stop_reason == "interrupted":
                raise KeyboardInterrupt(f"The job was interrupted; its checkpoint is kept at {self.checkpoint_path}.")
            self._save_result(results, already_compacted=True)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[bool]:
        # Hold the job's lock while the body runs. Yields False if a process that finished the job
        # deleted the lock file after this one opened it; warns and runs unlocked where files cannot
        # be locked.
        lock_path = os.path.abspath(self._lock_path)
        with open(lock_path, "a") if fcntl is not None else contextlib.nullcontext() as lock:
            try:
                if fcntl is None or lock is None:
                    raise OSError("no fcntl")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"[ExperimentJob] The job is running in another process (it holds {lock_path})."
                ) from None
            except OSError:
                warnings.warn(
                    "[ExperimentJob] Files cannot be locked here, so the same job could run in two processes at once.",
                    UserWarning,
                )
            else:
                if not os.path.exists(lock_path) or os.stat(lock_path).st_ino != os.fstat(lock.fileno()).st_ino:
                    yield False
                    return
                _held_locks.add(lock_path)
            try:
                yield True
            finally:
                _held_locks.discard(lock_path)

    def _dataset_and_approach(self) -> tuple[Dataset, Approach]:
        # The job's dataset and approach; the constructor sets exactly one of the instance and the
        # dict of each.
        if self._dataset_instance is not None:
            dataset = self._dataset_instance
        else:
            assert self._dataset_dict is not None
            dataset = Dataset.from_dict(self._dataset_dict)
        if self._approach_instance is not None:
            approach = self._approach_instance
        else:
            assert self._approach_dict is not None
            approach = Approach.from_config_dict(self._approach_dict)
        return dataset, approach

    def _save_result(self, results: Results, already_compacted: bool = False) -> None:
        # Save the result, then delete (or compact, unless already compacted) the checkpoint and
        # delete the lock file: a result marks the job done. The lock file goes only once the result
        # exists, so a process that opened it just before sees the result after taking the lock and
        # does nothing.
        if not self.info.keep_checkpoint:
            results.results[-1].checkpoint = None
        results.save(self.result_path)
        if self.info.keep_checkpoint:
            if not already_compacted:
                _compact_checkpoint(self.checkpoint_path)
        else:
            _remove_checkpoint(self.checkpoint_path)
        if os.path.exists(self._lock_path):
            os.remove(self._lock_path)

    def __repr__(self) -> str:
        return (
            f"ExperimentJob(dataset={self.dataset_name!r}, approach={self.approach_name!r}, "
            f"seed={self.seed}, status={self.status!r})"
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
            [Dataset][SRToolkit.dataset.dataset.Dataset], a
            [Benchmark][SRToolkit.dataset.benchmark.Benchmark] (all
            datasets in the benchmark are included), or a list containing
            ``Dataset``-s and ``Benchmark``-s.
        approaches: One or more SR approaches. Accepts a single
            [Approach][SRToolkit.approaches.approach.Approach] or a list of them.
        num_experiments: Number of independent experiments per (dataset, approach) pair.
        results_dir: Root directory where all results and grid metadata are stored.
        initial_seed: Seed for the first experiment. Subsequent experiments use
            ``initial_seed + 1``, ``initial_seed + 2``, etc.
        top_k: Number of top expressions to highlight per experiment.
        adapted_states: ``{approach_name: {dataset_name: path}}`` of state files for ``"once"``
            approaches; relative paths are made absolute. A listed pair loads its state, or adapts
            and saves it; an unlisted pair adapts on every run without saving. Datasets may share a
            path when the approach's adaptation depends only on what they share (e.g. EDHiE with one
            symbol library): it is adapted on the first dataset listed and reused, with a warning.
        callbacks: A callback or a list of them for every job. Stored as configs in ``grid.json``,
            so each job builds fresh instances.
        keep_all_models: Passed to every job (see
            [ExperimentInfo][SRToolkit.experiments.ExperimentInfo]); ``False`` keeps result files
            small for long runs. Default ``True``.
        keep_checkpoint: Passed to every job: keep each job's checkpoint, e.g. to reach all models
            of a run whose result keeps only the top ones. Default ``False``.
        snapshot_interval: Passed to every job (see
            [ExperimentInfo][SRToolkit.experiments.ExperimentInfo]): seconds between its snapshots,
            positive; ``None``: a snapshot only when the approach's step loop ends. Default
            ``300.0``.

    Raises:
        ValueError: If ``keep_all_models`` is ``False`` and ``top_k <= 0``, or if
            ``snapshot_interval`` is neither positive nor ``None``.
    """

    def __init__(
        self,
        datasets: Dataset | list[Dataset | Benchmark] | Benchmark,
        approaches: Approach | list[Approach],
        num_experiments: int,
        results_dir: str,
        initial_seed: int = 0,
        top_k: int = 20,
        adapted_states: dict[str, dict[str, str]] | None = None,
        callbacks: Callback | list[Callback] | None = None,
        keep_all_models: bool = True,
        keep_checkpoint: bool = False,
        snapshot_interval: float | None = 300.0,
    ) -> None:
        if not keep_all_models and top_k <= 0:
            raise ValueError(f"[ExperimentGrid] keep_all_models=False needs top_k >= 1, got {top_k}.")
        if snapshot_interval is not None and not snapshot_interval > 0:
            raise ValueError(f"[ExperimentGrid] snapshot_interval must be positive or None, got {snapshot_interval!r}.")
        self.keep_all_models = keep_all_models
        self.keep_checkpoint = keep_checkpoint
        self.snapshot_interval = snapshot_interval
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
        if isinstance(approaches, Approach):
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

    def add_approach(self, approach: Approach) -> None:
        """Add an approach to the grid.

        The approach's config is serialised immediately (its ``adaptation_scope`` is
        recorded alongside) so the grid never caches a live instance. Safe to call after
        [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] to extend a loaded grid.

        Args:
            approach: The [Approach][SRToolkit.approaches.approach.Approach] to add.

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

    def add_dataset(self, dataset: Dataset | Benchmark) -> None:
        """Add a dataset, or every dataset in a benchmark, to the grid.

        Each dataset is serialised immediately to its config dict (no data arrays
        embedded). Safe to call after
        [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] to extend a loaded grid.

        Args:
            dataset: An [Dataset][SRToolkit.dataset.dataset.Dataset] or an
                [Benchmark][SRToolkit.dataset.benchmark.Benchmark] (all of its
                datasets are added).

        Raises:
            ValueError: If a dataset name already exists in the grid, or if ``dataset``
                is neither an ``Dataset`` nor an ``Benchmark``.
        """
        if isinstance(dataset, Benchmark):
            named = [(name, dataset.create_dataset(name)) for name in dataset.list_datasets(verbose=False)]
        elif isinstance(dataset, Dataset):
            named = [(dataset.dataset_name, dataset)]
        else:
            raise ValueError(f"[ExperimentGrid] datasets must be a Dataset or Benchmark, got {type(dataset)}")
        for name, ds in named:
            if name in self.datasets:
                raise ValueError(
                    f"[ExperimentGrid] Duplicate dataset name '{name}'. "
                    f"Rename one of the datasets before passing it to ExperimentGrid "
                    f"(e.g. set ds.dataset_name = 'new_name')."
                )
            self.datasets[name] = ds.to_dict()

    def materialize_data(self) -> None:
        """Load every dataset once in this process, so its ``data_source`` fills the data cache
        before parallel workers only read it.

        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] calls this by default.

        Raises:
            FileNotFoundError: If a ``null``-source dataset has no data in the local cache
                (there is nothing to download or sample, so it cannot be materialised here).
        """
        for dataset_name in self.datasets:
            Dataset.from_dict(self.datasets[dataset_name])

    def _pairs_needing_adaptation(self) -> "list[tuple[str, str]]":
        # The (approach_name, dataset_name) pairs of "once" approaches whose state path is listed in
        # adapted_states but missing. Of pairs sharing a path only the first, with a warning.
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
        """Adapt one (approach, dataset) pair and save its state.

        The ``adapt`` commands of [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] call it
        per pair, so pairs can adapt in parallel. Does nothing if the approach is not
        ``"once"``-scoped, has no state path for this dataset, or the state file exists (unless
        ``force``).

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
        state_path = self._adapted_states.get(approach_name, {}).get(dataset_name)
        if state_path is None or (os.path.exists(state_path) and not force):
            return
        dataset = Dataset.from_dict(self.datasets[dataset_name])
        approach = Approach.from_config_dict(approach_config)
        dataset.adapt_or_load(approach, state_path, force=True)

    def adapt_if_missing(self):
        """Adapt, in this process, every ``"once"`` pair whose state file is missing (see
        [adapt_one][SRToolkit.experiments.ExperimentGrid.adapt_one]). To adapt in parallel, use
        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands].
        """
        for approach_name, dataset_name in self._pairs_needing_adaptation():
            self.adapt_one(approach_name, dataset_name)

    def build_job(self, dataset_name: str, approach_name: str, seed: int) -> ExperimentJob:
        """Construct the single [ExperimentJob][SRToolkit.experiments.ExperimentJob] for one
        (dataset, approach, seed) triple, using this grid's stored configs.

        [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] and the ``run_job`` CLI use it, so a
        worker needs only ``grid.json`` and the three identifiers.

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
            adapted_state_path=self._adapted_states.get(approach_name, {}).get(dataset_name),
            keep_all_models=self.keep_all_models,
            keep_checkpoint=self.keep_checkpoint,
            snapshot_interval=self.snapshot_interval,
        )
        return ExperimentJob(dataset=dataset_dict, approach=approach_config, info=info, callbacks=self.callback_configs)

    def create_jobs(self, skip_completed: bool = True) -> list[ExperimentJob]:
        """Return the grid's jobs, one per (dataset, approach, seed) triple.

        Neither adapts nor loads data: call
        [adapt_if_missing][SRToolkit.experiments.ExperimentGrid.adapt_if_missing] first for ``"once"``
        approaches, and [materialize_data][SRToolkit.experiments.ExperimentGrid.materialize_data] before
        running the jobs in parallel. [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands]
        does both.

        Args:
            skip_completed: If ``True`` (default), omit done and running jobs (see
                [status][SRToolkit.experiments.ExperimentJob.status]); interrupted and crashed ones
                stay, and running them resumes them. ``False``: every job; the
                [run][SRToolkit.experiments.ExperimentJob.run] of a done one does nothing unless
                ``rerun=True``, and that of a running one raises ``RuntimeError``.

        Returns:
            The [ExperimentJob][SRToolkit.experiments.ExperimentJob]-s in approach, dataset, seed
            order, without the skipped ones.
        """
        jobs: list[ExperimentJob] = []
        for approach_config in self.approach_configs:
            for dataset_name in self.datasets:
                for i in range(self.num_experiments):
                    job = self.build_job(dataset_name, approach_config["name"], self.initial_seed + i)
                    if skip_completed and job.status in ("done", "running"):
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

        - ``path`` — one ``run_job`` line per job
          [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] returns with the same
          ``skip_completed``; the line of a done job ends with ``--rerun``, so it reruns the job
          each time the file is run::

              python -m SRToolkit.experiments run_job \\
                  --grid /path/grid.json --dataset NG-1 --approach ProGED --seed 0

        - ``prepare_<name>`` (sibling of ``path``) — written **only** when one or more
          ``adaptation_scope="once"`` approaches still need state, one parallel-safe line
          per (approach, dataset) pair::

              python -m SRToolkit.experiments adapt \\
                  --grid /path/grid.json --dataset NG-1 --approach ProGED

        With ``materialize_data``, every dataset's data is fetched into the cache first, once, on
        this machine (on a cluster often the only one with network access), so workers only read
        it.

        Args:
            path: File path to write the experiment commands to.
            python_executable: Python executable to use in the commands.
            skip_completed: If ``True`` (default), omit jobs that are done or running, as
                [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] does. ``False``:
                every job, done ones with ``--rerun``.
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
        for job in self.create_jobs(skip_completed):
            lines.append(
                f"{python_executable} -m SRToolkit.experiments run_job "
                f"--grid {grid_path} "
                f"--dataset {shlex.quote(job.dataset_name)} "
                f"--approach {shlex.quote(job.approach_name)} "
                f"--seed {job.seed}" + (" --rerun" if not skip_completed and job.status == "done" else "")
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

        Each cell shows ``done/total`` for that pair (see
        [status][SRToolkit.experiments.ExperimentJob.status]). If any job is running, interrupted or
        crashed, a blank line follows, then their counts (zero counts left out) and one line per
        interrupted or crashed job, in dataset, approach, seed order; running these again resumes
        them.

        Example output::

            Dataset    ProGED   EDHiE
            -------------------------
            NG-1          5/5     3/5
            NG-2          2/5     0/5

            running 2, crashed 1
            crashed      NG-2 / ProGED / seed 3
        """
        dataset_names = list(self.datasets.keys())
        approach_names = [cfg["name"] for cfg in self.approach_configs]
        statuses = {
            (dataset_name, approach_name, seed): self.build_job(dataset_name, approach_name, seed).status
            for dataset_name in dataset_names
            for approach_name in approach_names
            for seed in range(self.initial_seed, self.initial_seed + self.num_experiments)
        }

        total_str = str(self.num_experiments)
        ds_w = max(len(n) for n in dataset_names + ["Dataset"]) + 2
        col_w = max(len(n) for n in approach_names + [f"{total_str}/{total_str}"]) + 2

        header = f"{'Dataset':<{ds_w}}" + "".join(f"{ap:>{col_w}}" for ap in approach_names)
        separator = "-" * ds_w + "".join("-" * col_w for _ in approach_names)
        print(header)
        print(separator)

        done = Counter((ds, ap) for (ds, ap, _), status in statuses.items() if status == "done")
        for dataset_name in dataset_names:
            row = f"{dataset_name:<{ds_w}}"
            for approach_name in approach_names:
                row += f"{done[dataset_name, approach_name]}/{self.num_experiments}".rjust(col_w)
            print(row)

        counts: Counter[str] = Counter(statuses.values())
        unfinished = [f"{name} {counts[name]}" for name in ("running", "interrupted", "crashed") if counts[name]]
        if unfinished:
            print()
            print(", ".join(unfinished))
            for (dataset_name, approach_name, seed), status in statuses.items():
                if status in ("interrupted", "crashed"):
                    print(f"{status:<13}{dataset_name} / {approach_name} / seed {seed}")

    def load_results(self, dataset_name: str, approach_name: str) -> Results:
        """Load and merge all completed per-experiment results for a (dataset, approach) pair.

        Args:
            dataset_name: Name of the dataset.
            approach_name: Name of the approach.

        Returns:
            [Results][SRToolkit.evaluation.results.Results] with one
            [EvalResult][SRToolkit.utils.types.EvalResult] per completed experiment; empty if none.
        """
        merged = Results()
        for i in range(self.num_experiments):
            seed = self.initial_seed + i
            result_path = os.path.join(self.results_dir, dataset_name, approach_name, f"exp_{seed}.json")
            if os.path.exists(result_path):
                merged += Results.load(result_path)
        return merged

    def to_dict(self) -> dict:
        """Return the grid as one self-contained, JSON-safe dict.

        It holds every dataset, approach and callback config plus ``num_experiments``,
        ``initial_seed``, ``top_k``, ``keep_all_models``, ``keep_checkpoint`` and
        ``snapshot_interval``; load it with [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict].

        Unlike [checkpoint][SRToolkit.experiments.ExperimentGrid.checkpoint], it leaves out the machine-local
        ``adapted_states`` paths, and ``results_dir`` is given when loading. Datasets hold no
        arrays; a dataset without a ``data_source`` must already be in the recipient's cache, or be
        shared with [share.save][SRToolkit.share.save].

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
            "keep_all_models": self.keep_all_models,
            "keep_checkpoint": self.keep_checkpoint,
            "snapshot_interval": self.snapshot_interval,
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
        with [share.load][SRToolkit.share.load]. A recipe without ``keep_all_models``,
        ``keep_checkpoint`` or ``snapshot_interval`` (written before they existed) loads them as
        ``True``, ``False`` and ``300.0``.

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
                loading from a dict; if it has ``keep_all_models`` false and ``top_k <= 0``, or a
                ``snapshot_interval`` that is neither positive nor ``None``.
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
        writer_version = d.get("srtk_version")
        if writer_version:
            try:  # best effort: never block loading on the version check
                from packaging.version import Version

                from SRToolkit import __version__

                if Version(__version__) < Version(writer_version):
                    warnings.warn(
                        f"[ExperimentGrid] This grid was written with SRToolkit {writer_version}, "
                        f"but {__version__} is installed. Loading may fail or behave unexpectedly.",
                        stacklevel=2,
                    )
            except Exception:
                pass

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
        grid.keep_all_models = d.get("keep_all_models", True)
        grid.keep_checkpoint = d.get("keep_checkpoint", False)
        grid.snapshot_interval = d.get("snapshot_interval", 300.0)
        if not grid.keep_all_models and grid.top_k <= 0:
            raise ValueError(f"[ExperimentGrid.from_dict] keep_all_models=False needs top_k >= 1, got {grid.top_k}.")
        if grid.snapshot_interval is not None and not grid.snapshot_interval > 0:
            raise ValueError(
                f"[ExperimentGrid.from_dict] snapshot_interval must be positive or None, got {grid.snapshot_interval!r}."
            )
        states = adapted_states if adapted_states is not None else d.get("adapted_states", {})
        grid._adapted_states = {
            approach: {dataset: os.path.abspath(path) for dataset, path in paths.items()}
            for approach, paths in states.items()
        }
        grid.datasets = {name: _bind_or_warn(cfg) for name, cfg in d["datasets"].items()}
        grid.approach_configs = [_bind_or_warn(cfg) for cfg in d["approaches"]]
        callbacks = d.get("callbacks")
        grid.callback_configs = [_bind_or_warn(cfg) for cfg in callbacks] if callbacks is not None else None

        _warn_missing_class_deps(list(grid.datasets.values()) + grid.approach_configs + (grid.callback_configs or []))
        return grid

    def checkpoint(self) -> None:
        """Write the grid to ``results_dir/grid.json``.

        Resume it with [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict]. Creates ``results_dir`` if
        needed and overwrites an existing ``grid.json``. The file is the
        [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] recipe plus the machine-local
        ``adapted_states`` paths. To share a grid with someone else, use
        [share.save][SRToolkit.share.save]. [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands]
        calls this.
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
