"""
Job-based experiment runner for symbolic regression experiments.

Provides three public classes:

- [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] — lightweight metadata (seed, paths) for a
  single run.
- [ExperimentJob][SRToolkit.experiments.ExperimentJob] — one atomic experiment: a single dataset ×
  approach × seed triple. Can be run in-process or dispatched to a CLI worker.
- [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] — a full cross-product grid of datasets and
  approaches.  Manages serialization, parallelism via HPC command files, progress tracking, and
  result loading.

Typical workflow::

    from SRToolkit.dataset import Nguyen
    from SRToolkit.approaches import ProGED
    from SRToolkit.experiments import ExperimentGrid

    grid = ExperimentGrid(
        datasets=Nguyen(),
        approaches=ProGED(),
        num_experiments=5,
        results_dir="/results/my_run",
    )

    # Run all jobs locally (sequential):
    for job in grid.create_jobs():
        job.run()

    # — or — generate a commands file for a SLURM / GNU Parallel cluster:
    grid.save_commands("/results/my_run/commands.txt")

    # Check progress and load results afterwards:
    grid.progress()
    sr_results = grid.load_results("Nguyen-1", "ProGED")
"""

import dataclasses
import importlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.dataset.sr_benchmark import SR_benchmark
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.callbacks import CallbackDispatcher, SRCallbacks
from SRToolkit.evaluation.sr_evaluator import SR_results


def _approach_from_config(config_dict: dict) -> SR_approach:
    """Reconstruct an SR_approach from a config dict that includes ``approach_class``."""
    class_path = config_dict["approach_class"]
    module_path, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls.from_config(config_dict)


def _callback_from_config(config_dict: dict) -> SRCallbacks:
    """Reconstruct an SRCallbacks instance from a config dict that includes ``callback_class``."""
    class_path = config_dict["callback_class"]
    module_path, cls_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls.from_dict(config_dict)


@dataclass
class ExperimentInfo:
    """
    Metadata for a single experiment run.

    Holds all job-specific information not contained in the dataset or approach
    config: the random seed, the path where the result should be written, how many top
    expressions to keep, and — for approaches with ``adaptation_scope="once"`` — where
    the pre-adapted state is stored.

    [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] constructs these automatically when
    you call [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs]. When running jobs
    via the CLI, ``info.json`` files are written by
    [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] and passed with
    ``--info``.

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
        adapted_state_path: Base path to the pre-adapted state for ``"once"``-scope
            approaches. ``None`` means the approach will adapt from scratch on every run and the
            state will not be saved.
    """

    seed: int
    result_path: str
    top_k: int = 20
    adapted_state_path: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-safe dictionary.

        Returns:
            A flat dictionary with keys ``seed``, ``result_path``, ``top_k``, and ``adapted_state_path``, suitable for passing to [from_dict][SRToolkit.experiments.ExperimentInfo.from_dict].
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentInfo":
        """
        Restore an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] from a dictionary
        produced by [to_dict][SRToolkit.experiments.ExperimentInfo.to_dict].

        Args:
            d: Dictionary with keys ``seed``, ``result_path``, ``top_k``, and
                ``adapted_state_path``.

        Returns:
            The reconstructed [ExperimentInfo][SRToolkit.experiments.ExperimentInfo].
        """
        return cls(**d)


class ExperimentJob:
    """
    A single atomic experiment: one dataset × one approach × one seed.

    An ``ExperimentJob`` is built from three components:

    - **dataset**: the dataset to evaluate on — an ``SR_dataset`` instance, a path to a
      ``SR_dataset.to_dict()`` JSON file, or the dict itself.
    - **approach**: the SR approach — an ``SR_approach`` instance, a path to an
      ``ApproachConfig.to_dict()`` JSON file, or the dict itself.
    - **info**: job metadata — an [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] instance,
      a path to an ``ExperimentInfo.to_dict()`` JSON file, or the dict itself.

    The three-path form makes standalone CLI execution trivial::

        python -m SRToolkit.experiments run_job \\
            --dataset /data/DS1.json \\
            --approach /configs/proged_config.json \\
            --info /out/DS1/ProGED/exp_0/info.json

    For Python use, pass instances directly::

        job = ExperimentJob(my_dataset, my_approach,
                            ExperimentInfo(seed=0, result_path="/out/"))
        job.run()

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
        dataset: Union[SR_dataset, str, dict],
        approach: Union[SR_approach, str, dict],
        info: Union[ExperimentInfo, str, dict],
        callbacks: Optional[Union[SRCallbacks, List[SRCallbacks], dict, List[dict]]] = None,
    ) -> None:
        """
        Args:
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
            self._dataset_instance: Optional[SR_dataset] = dataset
            self._dataset_dict: Optional[dict] = None
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
            self._approach_instance: Optional[SR_approach] = approach
            self._approach_dict: Optional[dict] = None
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
            self._callback_configs: Optional[List[dict]] = None
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
                    f"Invalid file extension '{extension}'. SR_results can only be loaded from '.json' files."
                )
            self.result_path = self.info.result_path

    @property
    def is_complete(self) -> bool:
        """``True`` if the result file at ``result_path`` already exists on disk."""
        return os.path.exists(self.result_path)

    def run(self) -> None:
        """
        Execute this experiment and save the result to ``result_path``.

        Handles adaptation according to
        [SR_approach.adaptation_scope][SRToolkit.approaches.sr_approach.SR_approach.adaptation_scope]:

        - ``"never"``: no adaptation.
        - ``"once"``: loads pre-adapted state from
          [ExperimentInfo][SRToolkit.experiments.ExperimentInfo]'s ``adapted_state_path``
          if a path is set and the file exists, otherwise adapts (and saves if a path is set).
        - ``"experiment"``: adapts fresh every run.

        The result is saved via
        [SR_results.save][SRToolkit.evaluation.sr_evaluator.SR_results.save] to ``result_path``.
        """
        if self._dataset_instance is not None:
            dataset = self._dataset_instance
        else:
            if self._dataset_dict is None:
                raise ValueError("No dataset provided: pass a dataset instance or a dataset dict.")
            dataset = SR_dataset.from_dict(self._dataset_dict)

        if self._approach_instance is not None:
            approach = self._approach_instance
        else:
            if self._approach_dict is None:
                raise ValueError("No approach provided: pass an approach instance or an approach dict.")
            approach = _approach_from_config(self._approach_dict)

        approach.prepare()

        if approach.adaptation_scope == "once":
            state_path = self.info.adapted_state_path
            if state_path is None:
                approach.adapt(dataset.X, dataset.symbol_library)
            else:
                if os.path.exists(state_path):
                    approach.load_adapted_state(state_path)
                else:
                    approach.adapt(dataset.X, dataset.symbol_library)
                    dir_name = os.path.dirname(state_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    approach.save_adapted_state(state_path)
        elif approach.adaptation_scope == "experiment":
            approach.adapt(dataset.X, dataset.symbol_library)

        evaluator = dataset.create_evaluator(seed=self.info.seed)
        evaluator._experiment_id = f"{self.dataset_name}_{self.approach_name}_{self.info.seed}"
        if self._callback_configs:
            cbs = [_callback_from_config(d) for d in self._callback_configs]
            evaluator.set_callbacks(CallbackDispatcher(callbacks=cbs))
        approach.search(evaluator, self.info.seed)
        results = evaluator.get_results(self.approach_name, self.info.top_k)
        results.save(self.result_path)

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "pending"
        return (
            f"ExperimentJob(dataset={self.dataset_name!r}, approach={self.approach_name!r}, "
            f"seed={self.seed}, status={status!r})"
        )


class ExperimentGrid:
    """
    Defines and manages a grid of symbolic regression experiments across multiple
    datasets and approaches.

    Each experiment is an independent
    [ExperimentJob][SRToolkit.experiments.ExperimentJob] that runs one approach on one
    dataset with one seed. Jobs can be executed locally (iterate and call
    ``.run()``) or on HPC clusters (generate a commands file with
    [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands]).

    The grid spec is persisted via [save][SRToolkit.experiments.ExperimentGrid.save] and
    reloaded via [load][SRToolkit.experiments.ExperimentGrid.load].  Results are saved
    per-experiment to ``results_dir/{dataset}/{approach}/exp_{seed}.json``,
    so parallel workers never write to the same file.

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
        datasets: One or more datasets to run experiments on.  Accepts a single
            [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset], a list of
            ``SR_dataset`` instances, or an
            [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] (all
            datasets in the benchmark are included).
        approaches: One or more SR approaches.  Accepts a single
            [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] or a list.
        num_experiments: Number of independent experiments per (dataset, approach) pair.
        results_dir: Root directory where all results and grid metadata are stored.
        initial_seed: Seed for the first experiment.  Subsequent experiments use
            ``initial_seed + 1``, ``initial_seed + 2``, etc.
        top_k: Number of top expressions to retain per experiment.
        adapted_states: Optional mapping ``{approach_name: {dataset_name: path}}``
            providing paths for pre-adapted state files.  Jobs for listed
            (approach, dataset) pairs will load state from the given path if it
            exists, or adapt and save to it otherwise.  Pairs not listed will adapt
            on every run without saving.
        callbacks: Optional callback or list of callbacks forwarded to every job
            created by [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs].
            Callbacks are serialised to dicts immediately so that each job reconstructs
            fresh instances in [run][SRToolkit.experiments.ExperimentJob.run] (no shared
            state between jobs).  When the grid is saved via
            [save][SRToolkit.experiments.ExperimentGrid.save] or
            [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands], a
            ``_callbacks.json`` file is written alongside the grid and the
            ``--callbacks`` flag is added to every CLI command.  Defaults to ``None``.
    """

    def __init__(
        self,
        datasets: Union[SR_dataset, List[Union[SR_dataset, SR_benchmark]], SR_benchmark],
        approaches: Union[SR_approach, List[SR_approach]],
        num_experiments: int,
        results_dir: str,
        initial_seed: int = 0,
        top_k: int = 20,
        adapted_states: Optional[Dict[str, Dict[str, str]]] = None,
        callbacks: Optional[Union[SRCallbacks, List[SRCallbacks]]] = None,
    ) -> None:
        self.num_experiments = num_experiments
        self.results_dir = results_dir
        self.initial_seed = initial_seed
        self.top_k = top_k
        self._adapted_states: Dict[str, Dict[str, str]] = adapted_states or {}

        if callbacks is None:
            self.callback_configs: Optional[List[dict]] = None
        elif isinstance(callbacks, list):
            self.callback_configs = [cb.to_dict() for cb in callbacks]
        else:
            self.callback_configs = [callbacks.to_dict()]

        # Build approach configs (plain serialisable dicts, no instance caching)
        if isinstance(approaches, SR_approach):
            approaches = [approaches]
        self.approach_configs: List[dict] = []
        for approach in approaches:
            cfg = approach.config.to_dict()
            cfg["adaptation_scope"] = approach.adaptation_scope
            self.approach_configs.append(cfg)

        # Serialise all datasets eagerly to results_dir/_datasets/
        self.datasets: Dict[str, dict] = dict()

        seen_names: Set[str] = set()

        def _add_dataset(ds: SR_dataset, name: str) -> None:
            if name in seen_names:
                raise ValueError(
                    f"[ExperimentGrid] Duplicate dataset name '{name}'. "
                    f"Rename one of the datasets before passing it to ExperimentGrid "
                    f"(e.g. set ds.dataset_name = 'new_name')."
                )
            seen_names.add(name)
            save_dir = os.path.join(results_dir, "_datasets", name)
            self.datasets[name] = ds.to_dict(save_dir)

        if isinstance(datasets, SR_benchmark):
            for name in datasets.list_datasets(verbose=False):
                _add_dataset(datasets.create_dataset(name), name)
        elif isinstance(datasets, SR_dataset):
            _add_dataset(datasets, datasets.dataset_name)
        elif isinstance(datasets, list):
            for ds in datasets:
                if isinstance(ds, SR_dataset):
                    _add_dataset(ds, ds.dataset_name)
                elif isinstance(ds, SR_benchmark):
                    for name in ds.list_datasets(verbose=False):
                        _add_dataset(ds.create_dataset(name), name)
                else:
                    raise ValueError(
                        f"[ExperimentGrid] Each element of datasets must be an SR_dataset "
                        f"or SR_benchmark, got {type(ds)}"
                    )
        else:
            raise ValueError(
                f"[ExperimentGrid] datasets must be SR_dataset, SR_benchmark, or a list, got {type(datasets)}"
            )

    def _get_adapted_state_ref_path(self, approach_name: str, dataset_name: str) -> Optional[str]:
        """Return the pickle path for an approach × dataset adapted state."""
        return self._adapted_states.get(approach_name, {}).get(dataset_name)

    def adapt_if_missing(self):
        """
        Pre-adapt all ``adaptation_scope="once"`` approaches where the state file is absent.

        For each (approach, dataset) pair whose state file does not yet exist on disk,
        this method loads the dataset, calls
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt] once, then persists the
        state via
        [save_adapted_state][SRToolkit.approaches.sr_approach.SR_approach.save_adapted_state].
        Pairs whose state file already exists are skipped.

        Approaches whose ``adaptation_scope`` is not ``"once"``, or that have no entry in
        the ``adapted_states`` mapping passed at construction, are skipped entirely.

        Call this before [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs] to ensure
        all states are ready before parallel workers start.

        """
        for approach_config in self.approach_configs:
            if approach_config.get("adaptation_scope", "never") != "once":
                continue
            approach_name = approach_config["name"]
            if approach_name not in self._adapted_states:
                continue
            for dataset_name, adapted_state_path in self._adapted_states[approach_name].items():
                if os.path.exists(adapted_state_path):
                    continue
                dataset = SR_dataset.from_dict(self.datasets[dataset_name])
                approach = _approach_from_config(approach_config)
                approach.prepare()
                approach.adapt(dataset.X, dataset.symbol_library)
                approach.save_adapted_state(adapted_state_path)

    def create_jobs(self, skip_completed: bool = True) -> List[ExperimentJob]:
        """
        Return the list of [ExperimentJob][SRToolkit.experiments.ExperimentJob] instances for
        this grid.

        Does **not** trigger adaptation — call
        [adapt_if_missing][SRToolkit.experiments.ExperimentGrid.adapt_if_missing] first if any
        approach has ``adaptation_scope="once"``.

        Args:
            skip_completed: If ``True`` (default), omit jobs whose result file
                (``exp_{seed}.json``) already exists on disk.

        Returns:
            List of jobs, one per (dataset, approach, seed) triple that has not yet completed.
        """
        jobs: List[ExperimentJob] = []
        for approach_config in self.approach_configs:
            for dataset_dict in self.datasets.values():
                approach_name = approach_config["name"]
                dataset_name = dataset_dict["dataset_name"]
                adapted_state_ref_path = self._get_adapted_state_ref_path(approach_name, dataset_name)
                for i in range(self.num_experiments):
                    seed = self.initial_seed + i
                    result_path = os.path.join(self.results_dir, dataset_name, approach_name, f"exp_{seed}.json")
                    info = ExperimentInfo(
                        seed=seed,
                        result_path=result_path,
                        top_k=self.top_k,
                        adapted_state_path=adapted_state_ref_path,
                    )
                    job = ExperimentJob(dataset=dataset_dict, approach=approach_config, info=info, callbacks=self.callback_configs)
                    if skip_completed and job.is_complete:
                        continue
                    jobs.append(job)
        return jobs

    def save_commands(
        self,
        path: str,
        python_executable: str = "python",
        skip_completed: bool = True,
    ) -> None:
        """
        Write a commands file with one CLI line per pending job.

        Calls [save][SRToolkit.experiments.ExperimentGrid.save] first to persist the grid.
        Also writes per-dataset JSON files, per-approach config JSON files, and per-job
        ``info.json`` files.

        Each line has the form::

            python -m SRToolkit.experiments run_job \\
                --dataset /path/dataset.json \\
                --approach /path/config.json \\
                --info /path/exp_N/info.json \\
                --callbacks /path/_callbacks.json

        The ``--callbacks`` flag is included only when callbacks are configured.

        Args:
            path: File path to write commands to.
            python_executable: Python executable to use in the commands.
            skip_completed: If ``True`` (default), omit already-completed jobs.
        """
        self.save()

        # Derive paths from the same convention used by save()
        ds_json_paths = {
            name: os.path.join(self.results_dir, "_datasets", name, f"{name}.json") for name in self.datasets
        }
        config_json_paths = {
            cfg["name"]: os.path.join(self.results_dir, "_approaches", f"{cfg['name']}_config.json")
            for cfg in self.approach_configs
        }

        callbacks_path = os.path.join(self.results_dir, "_callbacks.json")
        callbacks_arg = f" --callbacks {callbacks_path}" if os.path.exists(callbacks_path) else ""

        # Write per-job info.json files and collect command lines
        jobs = self.create_jobs(skip_completed=skip_completed)
        lines = [f"# results_dir: {self.results_dir}"]
        for job in jobs:
            os.makedirs(os.path.dirname(job.result_path), exist_ok=True)
            info_path = os.path.join(os.path.dirname(job.result_path), f"info_{job.seed}.json")
            with open(info_path, "w") as f:
                json.dump(job.info.to_dict(), f, indent=2)
            lines.append(
                f"{python_executable} -m SRToolkit.experiments run_job "
                f"--dataset {ds_json_paths[job.dataset_name]} "
                f"--approach {config_json_paths[job.approach_name]} "
                f"--info {info_path}"
                f"{callbacks_arg}"
            )

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def progress(self) -> None:
        """
        Print a dataset × approach progress table to stdout.

        Each cell shows ``done/total`` experiments completed for that pair, based on
        the presence of ``results.json`` files on disk.

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
        """
        Load and merge all completed per-experiment results for a (dataset, approach) pair.

        Examples:
            >>> results = grid.load_results("Nguyen-1", "ProGED")  # doctest: +SKIP
            >>> len(results)  # number of completed experiments  # doctest: +SKIP
            5

        Args:
            dataset_name: Name of the dataset.
            approach_name: Name of the approach.

        Returns:
            An [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] object containing one [EvalResult][SRToolkit.evaluation.result_augmentation.EvalResult] per completed experiment.  Returns an empty ``SR_results`` if no experiments have completed yet.
        """
        merged = SR_results()
        for i in range(self.num_experiments):
            seed = self.initial_seed + i
            result_path = os.path.join(self.results_dir, dataset_name, approach_name, f"exp_{seed}.json")
            if os.path.exists(result_path):
                merged += SR_results.load(result_path)
        return merged

    def save(self) -> None:
        """
        Persist the grid specification and supporting files to ``results_dir``.

        Writes the following files (all idempotent — existing files are not overwritten):

        - ``results_dir/grid.json`` — the grid specification.
        - ``results_dir/_datasets/{name}/{name}.json`` — one JSON file per dataset.
        - ``results_dir/_approaches/{name}_config.json`` — one JSON file per approach config.
        - ``results_dir/_callbacks.json`` — serialised callbacks, written only when callbacks
          are set.

        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] calls this
        automatically, so a separate ``save()`` call is only needed when checkpointing
        the grid without generating a commands file.
        """
        os.makedirs(self.results_dir, exist_ok=True)

        # Write per-dataset JSON files
        for name, dataset in self.datasets.items():
            ds_path = os.path.join(self.results_dir, "_datasets", name, f"{name}.json")
            if not os.path.exists(ds_path):
                os.makedirs(os.path.dirname(ds_path), exist_ok=True)
                with open(ds_path, "w") as f:
                    json.dump(dataset, f, indent=2)

        # Write per-approach config JSON files
        for approach_config in self.approach_configs:
            cfg_path = os.path.join(self.results_dir, "_approaches", f"{approach_config['name']}_config.json")
            if not os.path.exists(cfg_path):
                os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
                with open(cfg_path, "w") as f:
                    json.dump(approach_config, f, indent=2)

        # Write callbacks file when callbacks are set
        if self.callback_configs is not None:
            callbacks_path = os.path.join(self.results_dir, "_callbacks.json")
            with open(callbacks_path, "w") as f:
                json.dump(self.callback_configs, f, indent=2)

        grid_dict = {
            "format_version": 1,
            "type": "ExperimentGrid",
            "results_dir": self.results_dir,
            "num_experiments": self.num_experiments,
            "initial_seed": self.initial_seed,
            "top_k": self.top_k,
            "adapted_states": self._adapted_states,
            "dataset_names": list(self.datasets.keys()),
            "approach_names": [cfg["name"] for cfg in self.approach_configs],
        }
        with open(os.path.join(self.results_dir, "grid.json"), "w") as f:
            json.dump(grid_dict, f, indent=2)

    @staticmethod
    def load(path: str) -> "ExperimentGrid":
        """
        Load an [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] from a previously saved
        ``grid.json``.

        Dataset and approach instances are **not** created at load time — they are
        reconstructed lazily when jobs are executed.

        Args:
            path: Path to the ``grid.json`` file written by
                [save][SRToolkit.experiments.ExperimentGrid.save].

        Returns:
            A fully configured ``ExperimentGrid``

        Raises:
            ValueError: If ``format_version`` is not supported.
        """
        with open(path) as f:
            d = json.load(f)
        if d.get("format_version", 1) != 1:
            raise ValueError(
                f"[ExperimentGrid.load] Unsupported format_version: {d.get('format_version')!r}. Expected 1."
            )
        grid = ExperimentGrid.__new__(ExperimentGrid)
        grid.num_experiments = d["num_experiments"]
        grid.results_dir = d["results_dir"]
        grid.initial_seed = d["initial_seed"]
        grid.top_k = d["top_k"]
        grid._adapted_states = d.get("adapted_states", {})

        grid.datasets = {}
        for name in d["dataset_names"]:
            ds_path = os.path.join(grid.results_dir, "_datasets", name, f"{name}.json")
            with open(ds_path) as f:
                grid.datasets[name] = json.load(f)

        grid.approach_configs = []
        for name in d["approach_names"]:
            cfg_path = os.path.join(grid.results_dir, "_approaches", f"{name}_config.json")
            with open(cfg_path) as f:
                grid.approach_configs.append(json.load(f))

        callbacks_path = os.path.join(grid.results_dir, "_callbacks.json")
        if os.path.exists(callbacks_path):
            with open(callbacks_path) as f:
                grid.callback_configs = json.load(f)
        else:
            grid.callback_configs = None

        return grid

    def __repr__(self) -> str:
        ds_count = len(self.datasets)
        ap_count = len(self.approach_configs)
        total = ds_count * ap_count * self.num_experiments
        return (
            f"ExperimentGrid({ds_count} datasets × {ap_count} approaches × "
            f"{self.num_experiments} experiments = {total} jobs, "
            f"results_dir={self.results_dir!r})"
        )
