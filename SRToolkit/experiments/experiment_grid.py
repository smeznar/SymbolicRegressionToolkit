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
"""

import dataclasses
import importlib
import inspect
import json
import os
import shlex
import shutil
import tempfile
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.bundle import list_installed, read_manifest
from SRToolkit.bundle._install import _repack
from SRToolkit.bundle._relocate import _auto_bind, bundle_class_index
from SRToolkit.dataset.sr_benchmark import SR_benchmark
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation.callbacks import CallbackDispatcher, ExperimentEvent, SRCallbacks
from SRToolkit.evaluation.sr_evaluator import SR_results


def _iter_class_paths(obj: Any) -> Iterator[str]:
    """Yield every ``*_class`` dotted-path value nested anywhere in ``obj``.

    Walks dicts and lists recursively. A grid's code dependencies are not only
    its ``approach_class`` entries — they also include ``callback_class`` and the
    ``sampler_class`` / ``source_class`` / ``constraint_class`` paths nested
    inside each dataset config — so the whole structure must be scanned.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.endswith("_class"):
                yield v
            else:
                yield from _iter_class_paths(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_class_paths(v)


def _warn_missing_class_deps(configs: List[dict]) -> None:
    """Soft-check that every ``*_class`` path in ``configs`` is importable.

    Emits a single warning listing any unimportable classes. Does not raise:
    [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] stays lazy and the
    hard failure is deferred to job-run time (in
    [SR_approach.from_config_dict][SRToolkit.approaches.sr_approach.SR_approach.from_config_dict]
    / [SRCallbacks.from_config_dict][SRToolkit.evaluation.callbacks.SRCallbacks.from_config_dict]).
    Run *after* ``_auto_bind`` so bundle paths are
    already rewritten to their installed import prefix.
    """
    missing: List[str] = []
    seen: Set[str] = set()
    for cfg in configs:
        for class_path in _iter_class_paths(cfg):
            if class_path in seen:
                continue
            seen.add(class_path)
            module_path, _, cls_name = class_path.rpartition(".")
            if not module_path:
                continue
            try:
                getattr(importlib.import_module(module_path), cls_name)
            except (ImportError, AttributeError):
                missing.append(class_path)
    if missing:
        warnings.warn(
            "[ExperimentGrid] These classes referenced by the grid could not be imported "
            "on this machine:\n  - " + "\n  - ".join(missing) + "\n"
            "Install the providing .srtk bundle(s) before running jobs, or the grid will "
            "fail at run time. If a config has no '_bundle' key, call bind_config(config) first.",
            stacklevel=3,
        )


def _warn_if_outdated(writer_version: Optional[str]) -> None:
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


def _classify_dataset_source(cfg: dict) -> "tuple[bool, str]":
    """Return ``(needs_archiving, human_reason)`` for a dataset config.

    Data that the recipient can reach on their own is *referenced* (left to its
    ``data_source``); data that would otherwise be lost is *archived* into the export.
    Per the settled design: ``null`` sources and seedless ``sample`` sources are
    archived; ``url`` and seeded ``sample`` sources are referenced.
    """
    src = cfg.get("data_source")
    if src is None:
        return True, "null source — data exists only in the cache; shipped inline"
    source_class = src.get("source_class", "")
    if source_class.endswith("SampleSource"):
        if src.get("seed") is None:
            return True, "sample source without a fixed seed — exact arrays shipped for reproducibility"
        return False, f"regenerated from samplers (seed={src.get('seed')})"
    if source_class.endswith("UrlSource"):
        return False, f"downloaded on first use from {src.get('url')}"
    if source_class.endswith("FallbackSource"):
        return False, "fallback chain — canonical data downloaded, regenerated only if unavailable"
    return False, f"referenced via {source_class or 'custom data source'}"


def _custom_class_paths(configs: List[dict]) -> List[str]:
    """Sorted unique ``*_class`` paths that are not built into SRToolkit."""
    found: Set[str] = set()
    for cfg in configs:
        for class_path in _iter_class_paths(cfg):
            if not class_path.startswith("SRToolkit."):
                found.add(class_path)
    return sorted(found)


def _annotate_bundles(obj: Any, class_to_bundle: "Dict[str, tuple[str, str]]") -> Any:
    """Deep-copy ``obj``, stamping ``_bundle``/``_version`` onto each dict that directly
    holds a custom ``*_class`` key.

    The annotation is placed at the granularity each ``from_dict`` dispatcher binds: a
    sampler dict, a source dict, an approach cfg, a callback cfg. Since such a dict holds
    a single ``*_class``, the recipient's ``_auto_bind`` relocates it against exactly the
    right bundle — sibling configs (sampler vs. source) bind independently. A single dict
    that somehow references two different bundles can't be expressed with one ``_bundle``
    and is left unannotated with a warning.
    """
    if isinstance(obj, dict):
        bundles = {
            class_to_bundle[v]
            for k, v in obj.items()
            if isinstance(v, str) and k.endswith("_class") and v in class_to_bundle
        }
        annotated = {k: _annotate_bundles(v, class_to_bundle) for k, v in obj.items()}
        if len(bundles) == 1:
            name, version = next(iter(bundles))
            annotated["_bundle"] = name
            annotated["_version"] = version
        elif len(bundles) > 1:
            warnings.warn(
                f"[ExperimentGrid.export] A single config references classes from multiple "
                f"bundles {sorted(n for n, _ in bundles)}; it cannot be annotated with one "
                "'_bundle'. Build a combined bundle and bind it manually.",
                stacklevel=3,
            )
        return annotated
    if isinstance(obj, list):
        return [_annotate_bundles(v, class_to_bundle) for v in obj]
    return obj


def _describe_class_source(class_path: str) -> str:
    """Best-effort ``'class_path (file.py)'`` label for a warning, never raising.

    Resolves the defining module's source file so the warning can name the *file* the
    user must pack — falling back to the bare class path when the source can't be located
    (``__main__``, C extension, no source on disk).
    """
    module_path = class_path.rpartition(".")[0]
    if module_path in ("", "__main__"):
        return f"{class_path} (defined in __main__)"
    try:
        source = inspect.getsourcefile(importlib.import_module(module_path))
    except Exception:  # noqa: BLE001 - locating the source is best-effort cosmetics
        source = None
    return f"{class_path} ({source})" if source else class_path


def _pack_custom_code(
    configs: List[dict],
    bundles_dir: Path,
    additional_bundles: Optional[List[Union[str, Path]]] = None,
    strict: bool = False,
) -> "tuple[List[tuple[str, str, str]], List[tuple[str, str]], Dict[str, tuple[str, str]]]":
    """Resolve every non-SRToolkit class referenced by ``configs`` to a shippable bundle.

    A custom class is *covered* when some bundle provides a class of that name. Two bundle
    sources are matched, ``additional_bundles`` first so an explicitly supplied ``.srtk``
    wins over a coincidentally-installed one:

    - **User-supplied** ``.srtk`` files (``additional_bundles``) are matched by the class
      names they define (read straight from the archive, no install) and copied into
      ``bundles_dir`` verbatim — the user built them with [pack][SRToolkit.bundle.pack],
      so multi-file implementations, dependencies, and the real version all travel
      correctly.
    - **Already-installed** bundles whose ``import_prefix`` a class path already sits under
      are **re-packed once** from the installed source files via ``_repack``, which carries
      their ``python_deps`` / ``srtk_min_version``.

    The class path in the recipe is left untouched in both cases: only the owning
    ``(name, version)`` is recorded, and the recipient's bind machinery relocates the class
    by name at load time. Custom classes matching neither source are *not* guessed at: they
    are returned in ``manual`` for the manifest to flag, and a warning naming each class and
    its source file is emitted. When ``strict`` is set, an uncovered class raises
    ``ValueError`` instead.

    Returns ``(packed, manual, class_to_bundle)``: ``packed`` is ``[(label,
    bundle_filename, note)]``, ``manual`` is ``[(class_path, reason)]``, and
    ``class_to_bundle`` maps each covered class path to the ``(name, version)`` of the
    bundle the recipient will install — used to annotate the recipe so the existing bind
    machinery relocates each class on load.

    Raises:
        ValueError: If ``strict`` and any custom class is covered by neither an installed
            bundle nor ``additional_bundles``; or if an ``additional_bundles`` path is not
            a readable ``.srtk``.
    """
    custom = _custom_class_paths(configs)
    if not custom:
        return [], [], {}

    packed: List["tuple[str, str, str]"] = []
    manual: List["tuple[str, str]"] = []
    class_to_bundle: "Dict[str, tuple[str, str]]" = {}

    # Read each supplied bundle's manifest + the class names it defines, without installing.
    supplied: "List[dict]" = []  # {path, manifest, classes, covered}
    for raw in additional_bundles or []:
        srtk_path = Path(raw)
        manifest = read_manifest(srtk_path)  # raises on a bad/missing archive
        supplied.append(
            {"path": srtk_path, "manifest": manifest, "classes": bundle_class_index(srtk_path), "covered": []}
        )

    installed = list_installed()

    # Partition each custom class to its covering bundle. Supplied bundles take precedence
    # and are matched by class name (mirroring how bind_config resolves on the recipient);
    # installed bundles are matched by the import_prefix the class path already sits under.
    installed_classes: Dict[str, List[str]] = {}
    installed_entry: Dict[str, dict] = {}
    uncovered: List[str] = []
    for class_path in custom:
        cls_name = class_path.rpartition(".")[2]
        match = next((b for b in supplied if cls_name in b["classes"]), None)
        if match is not None:
            match["covered"].append(class_path)
            continue
        entry = next((e for e in installed if class_path.startswith(e["import_prefix"] + ".")), None)
        if entry is not None:
            installed_classes.setdefault(entry["import_prefix"], []).append(class_path)
            installed_entry[entry["import_prefix"]] = entry
        else:
            uncovered.append(class_path)

    # Copy each matched user-supplied bundle once; class paths are left unchanged.
    for b in supplied:
        if not b["covered"]:
            continue
        bundles_dir.mkdir(parents=True, exist_ok=True)
        manifest = b["manifest"]
        dest = bundles_dir / b["path"].name
        shutil.copy2(b["path"], dest)
        packed.append((f"bundle '{manifest.name}' v{manifest.version}", dest.name, "supplied via additional_bundles"))
        for class_path in b["covered"]:
            class_to_bundle[class_path] = (manifest.name, manifest.version)

    # Re-pack each matched installed bundle exactly once. _repack owns the install-layout
    # details and carries the bundle's python_deps / srtk_min_version.
    for prefix, entry in installed_entry.items():
        try:
            out = bundles_dir / f"{entry['name']}_{entry['version']}.srtk".replace(" ", "_")
            _repack(entry["name"], entry["version"], out)
            packed.append(
                (f"bundle '{entry['name']}' v{entry['version']}", out.name, "re-packed from the installed bundle")
            )
            for class_path in installed_classes[prefix]:
                class_to_bundle[class_path] = (entry["name"], entry["version"])
        except Exception as exc:  # noqa: BLE001 - best-effort; report and move on
            manual.append((prefix, f"could not re-pack installed bundle: {exc}"))

    # Anything left is not guessed at — the user must supply a .srtk for it.
    if uncovered:
        for class_path in uncovered:
            manual.append(
                (class_path, "no installed bundle and no additional_bundles entry covers it — pack it and pass it")
            )
        described = "\n  - ".join(_describe_class_source(c) for c in uncovered)
        message = (
            "[ExperimentGrid.export] These custom classes are not covered by any installed bundle "
            "or additional_bundles entry and were NOT packed into the export:\n  - " + described + "\n"
            "Build a .srtk for each (SRToolkit.bundle.pack) and pass it via additional_bundles, "
            "or the recipient will be unable to load the grid."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)

    # Warn about supplied bundles that matched nothing — likely a mistake.
    unused = [str(b["path"]) for b in supplied if not b["covered"]]
    if unused:
        warnings.warn(
            "[ExperimentGrid.export] These additional_bundles did not match any class referenced "
            "by the grid and were not included:\n  - " + "\n  - ".join(unused),
            stacklevel=3,
        )

    return packed, manual, class_to_bundle


@dataclass
class ExperimentInfo:
    """
    Metadata for a single experiment run.

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
                    f"Invalid file extension '{extension}'. SR_results can only be saved as '.json' files."
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
            approach = SR_approach.from_config_dict(self._approach_dict)

        # Honors the contract for callers passing a live instance; a no-op for the
        # grid's own from-config reconstruction (nothing has accumulated yet).
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
        dispatcher: Optional[CallbackDispatcher] = None
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
        datasets: One or more datasets to run experiments on. Accepts a single
            [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset], a
            [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] (all
            datasets in the benchmark are included), or a list containing
            ``SR_dataset``-s and ``SR_dataset``-s.
        approaches: One or more SR approaches.  Accepts a single
            [SR_approach][SRToolkit.approaches.sr_approach.SR_approach] or a list of them.
        num_experiments: Number of independent experiments per (dataset, approach) pair.
        results_dir: Root directory where all results and grid metadata are stored.
        initial_seed: Seed for the first experiment.  Subsequent experiments use
            ``initial_seed + 1``, ``initial_seed + 2``, etc.
        top_k: Number of top expressions to highlight per experiment.
        adapted_states: Optional mapping ``{approach_name: {dataset_name: path}}``
            providing paths for pre-adapted state files. Jobs for listed
            (approach, dataset) pairs will load state from the given path if it
            exists, or adapt and save to it otherwise. Pairs not listed will adapt
            on every run without saving.  Several datasets may deliberately share one
            ``path`` for approaches that adapt per symbol space rather than per dataset
            (e.g. EDHiE): the path is then adapted once on the first dataset listed for
            it and the others reuse that state. An informational warning lists any such
            shared paths so an accidental collision can be spotted; give each dataset a
            distinct ``path`` if adaptation must differ per dataset.
        callbacks: Optional callback or list of callbacks forwarded to every job
            created by [create_jobs][SRToolkit.experiments.ExperimentGrid.create_jobs].
            Callbacks are serialised to dicts immediately so that each job reconstructs
            fresh instances in [run][SRToolkit.experiments.ExperimentJob.run] (no shared
            state between jobs). They are inlined into the grid recipe by
            [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict], so they travel inside
            ``grid.json`` — both [save][SRToolkit.experiments.ExperimentGrid.save] and the
            ``run_job`` CLI reconstruct them from that single file, with no extra files or
            flags.  Defaults to ``None``.
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
        self.approach_configs: List[dict] = []
        if isinstance(approaches, SR_approach):
            approaches = [approaches]
        for approach in approaches:
            self.add_approach(approach)

        # Serialise all datasets eagerly into config dicts (no arrays embedded)
        self.datasets: Dict[str, dict] = dict()
        if isinstance(datasets, list):
            for ds in datasets:
                self.add_dataset(ds)
        else:
            self.add_dataset(datasets)

    def add_approach(self, approach: SR_approach) -> None:
        """
        Add an approach to the grid.

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

    def add_dataset(self, dataset: Union[SR_dataset, SR_benchmark]) -> None:
        """
        Add a dataset, or every dataset in a benchmark, to the grid.

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

    def _get_adapted_state_ref_path(self, approach_name: str, dataset_name: str) -> Optional[str]:
        """Return the pickle path for an approach × dataset adapted state."""
        return self._adapted_states.get(approach_name, {}).get(dataset_name)

    def materialize_data(self) -> None:
        """
        Materialise every dataset's data into the local cache, once, in this process.

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

    def _pairs_needing_adaptation(self) -> "List[tuple[str, str]]":
        """Return ``(approach_name, dataset_name)`` pairs whose ``"once"`` state is absent.

        A pair qualifies when the approach is ``adaptation_scope="once"``, has a state path
        registered in the ``adapted_states`` mapping, and that state file does not yet exist
        on disk.

        When several pairs map to the *same* ``state_path``, only the first is returned and an
        informational warning is emitted: the path is adapted once on the first dataset listed
        for it and the rest reuse the result. This is intended for approaches that adapt per
        symbol space rather than per dataset (e.g. EDHiE); give each dataset a distinct
        ``state_path`` if adaptation must differ per dataset.
        """
        pairs: "List[tuple[str, str]]" = []
        seen_paths: Dict[str, "tuple[str, str]"] = {}
        conflicts: "List[str]" = []
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
                "state. This is the intended setup for approaches that adapt per symbol space "
                "(e.g. EDHiE) rather than per dataset:\n" + "\n".join(conflicts) + "\n"
                "If adaptation should instead differ per dataset, give each dataset a distinct "
                "'state_path' in the 'adapted_states' mapping.",
                stacklevel=2,
            )
        return pairs

    def adapt_one(self, approach_name: str, dataset_name: str, force: bool = False) -> None:
        """
        Adapt a single (approach, dataset) pair and persist its state.

        This is the unit the per-pair adaptation commands in ``prepare.txt`` invoke (via the
        ``adapt`` CLI with ``--dataset``/``--approach``), so independent pairs can adapt in
        parallel across a cluster. It loads the dataset, calls
        [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt], and saves the state via
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
        approach.adapt(dataset.X, dataset.symbol_library)
        approach.save_adapted_state(state_path)

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

        This adapts every pair sequentially in this process. To distribute the work, use
        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands], which emits one
        [adapt_one][SRToolkit.experiments.ExperimentGrid.adapt_one] command per pair into
        ``prepare.txt``.
        """
        for approach_name, dataset_name in self._pairs_needing_adaptation():
            self.adapt_one(approach_name, dataset_name)

    def build_job(self, dataset_name: str, approach_name: str, seed: int) -> ExperimentJob:
        """
        Construct the single [ExperimentJob][SRToolkit.experiments.ExperimentJob] for one
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

    def create_jobs(self, skip_completed: bool = True) -> List[ExperimentJob]:
        """
        Return the list of [ExperimentJob][SRToolkit.experiments.ExperimentJob] instances for
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
        jobs: List[ExperimentJob] = []
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
    ) -> Optional[str]:
        """
        Write the experiment commands file (and, if needed, a ``prepare`` commands file).

        Calls [save][SRToolkit.experiments.ExperimentGrid.save] first to persist the grid
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
        self.save()
        if materialize_data:
            self.materialize_data()
        grid_path = shlex.quote(os.path.join(self.results_dir, "grid.json"))

        prepare_path: Optional[str] = None
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
        """
        Print a dataset × approach progress table to stdout.

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

    def to_dict(self) -> dict:
        """
        Serialize the grid to a single self-contained, JSON-safe dict — the *recipe*.

        The returned dict inlines every dataset config, approach config, and callback
        config alongside the run metadata (``num_experiments``, ``initial_seed``,
        ``top_k``). It is the portable, one-file form of the grid: pass it (or a JSON
        file holding it) straight to [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict].

        Unlike [save][SRToolkit.experiments.ExperimentGrid.save], this carries **no
        results** and **no machine-local execution state**: ``results_dir`` is supplied
        at load time (anchored to the file's location) and the absolute
        ``adapted_states`` paths are dropped — the per-approach ``adaptation_scope``
        already travels inside each approach config, so a recipient re-adapts to their
        own paths.

        Dataset configs are the full per-dataset recipe (``data_source`` + samplers +
        ground truth), not bare references, and contain no data arrays — those are
        regenerated or downloaded from each dataset's ``data_source`` on first use. A
        ``null``-source dataset's data must already be present in the recipient's cache;
        ship it via [export][SRToolkit.experiments.ExperimentGrid.export] otherwise.

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
        config: Union[dict, str, Path],
        results_dir: Optional[str] = None,
        adapted_states: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> "ExperimentGrid":
        """
        Reconstruct an [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] from a
        recipe produced by [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict].

        Every embedded config (approaches, callbacks, and the samplers / sources /
        constraints nested inside each dataset config) is run through ``_auto_bind`` so
        that ``*_class`` paths pointing at an installed ``.srtk`` bundle resolve
        automatically. Dataset and approach **instances are not created here** — they
        are reconstructed lazily when jobs run. As a courtesy, any ``*_class`` path that
        cannot be imported on this machine triggers a warning naming the missing class;
        the hard failure is deferred to job-run time.

        Args:
            config: The recipe. One of:

                - ``dict`` — the [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict]
                  output directly. ``results_dir`` is then **required**.
                - ``str`` / ``Path`` — path to a JSON file holding that dict. The
                  directory containing the file becomes the grid's ``results_dir``
                  unless ``results_dir`` is given explicitly.

            results_dir: Where results are read from and written to. Overrides the
                file-anchored default; required when ``config`` is a dict.
            adapted_states: Optional mapping ``{approach_name: {dataset_name: path}}``
                providing paths for pre-adapted state files. Jobs for listed
                (approach, dataset) pairs will load state from the given path if it
                exists, or adapt and save to it otherwise. Pairs not listed will adapt
                on every run without saving.  Several datasets may deliberately share one
                ``path`` for approaches that adapt per symbol space rather than per dataset
                (e.g. EDHiE): the path is then adapted once on the first dataset listed for
                it and the others reuse that state. An informational warning lists any such
                shared paths so an accidental collision can be spotted; give each dataset a
                distinct ``path`` if adaptation must differ per dataset.

        Returns:
            A fully configured ``ExperimentGrid``.

        Raises:
            ValueError: If the config is not an ``ExperimentGrid`` recipe, its
                ``format_version`` is unsupported, or ``results_dir`` is omitted when
                loading from a dict.
        """
        default_results_dir: Optional[str] = None
        if isinstance(config, (str, Path)):
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
        grid._adapted_states = adapted_states or {}
        grid.datasets = {name: _safe_auto_bind(cfg) for name, cfg in d["datasets"].items()}
        grid.approach_configs = [_safe_auto_bind(cfg) for cfg in d["approaches"]]
        callbacks = d.get("callbacks")
        grid.callback_configs = [_safe_auto_bind(cfg) for cfg in callbacks] if callbacks is not None else None

        _warn_missing_class_deps(list(grid.datasets.values()) + grid.approach_configs + (grid.callback_configs or []))
        return grid

    def export(
        self,
        path: Union[str, Path],
        include_results: bool = False,
        additional_bundles: Optional[List[Union[str, Path]]] = None,
        strict: bool = False,
    ) -> None:
        """
        Write a self-contained, shareable ``.zip`` gathering everything a recipient needs.

        Unlike [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] (a recipe that
        *assumes* the recipient can reach the data and code), ``export`` bundles the
        dependencies that wouldn't otherwise travel into one archive whose entries are:

        ```
        grid.json            # the recipe (to_dict output), annotated with _bundle refs
        data/<name>.zip      # only for datasets whose data isn't otherwise reachable
        bundles/<name>.srtk  # only when custom (non-SRToolkit) classes are referenced
        results/...          # only when include_results=True
        MANIFEST.md          # inventory + recipient instructions
        ```

        A dataset's data is archived when its source is ``null`` (data lives only in the
        cache) or a seedless ``sample`` source (regeneration wouldn't reproduce the exact
        numbers). ``url`` and seeded ``sample`` datasets are left referenced — the
        recipient downloads or regenerates them.

        **Custom code.** Any non-SRToolkit class referenced anywhere in the grid must be
        covered by a ``.srtk`` bundle so the recipient can import it. Two sources are
        used, in order:

        - **already-installed bundles** — a referenced class whose import path falls under
          an installed bundle is re-packed automatically from the installed source files
          (carrying its declared ``python_deps`` / ``srtk_min_version``);
        - **``additional_bundles``** — ``.srtk`` files you built yourself with
          [pack][SRToolkit.bundle.pack] for code that isn't installed as a bundle. Because
          you build them, multi-file implementations, dependencies, and the real version
          all travel correctly. ``export`` never tries to guess a bundle from a loose
          source file.

        The recipe is annotated with ``_bundle`` / ``_version`` so the recipient's bind
        machinery relocates each class on load. Any custom class covered by neither source
        is **not** packed: it is listed in ``MANIFEST.md`` and a warning names each class
        and its source file. Pass ``strict=True`` to raise instead of warning.

        The bundles are **not** installed automatically.
        [from_export][SRToolkit.experiments.ExperimentGrid.from_export] extracts them next to
        the archive so the recipient can ``install`` each before running jobs (installing
        executes third-party code and mutates the global bundle store, so it stays an
        explicit step); class binding is deferred to job-run time, so install order doesn't
        matter as long as it happens before the jobs run.

        Load the result with
        [from_export][SRToolkit.experiments.ExperimentGrid.from_export].

        Args:
            path: Destination ``.zip`` path (parent directories created if absent). A
                non-``.zip`` suffix is allowed but warns.
            include_results: If ``True``, copy completed ``exp_*.json`` result files into
                ``results/`` so the export carries the run's outputs too.
            additional_bundles: Optional list of ``.srtk`` paths (built with
                [pack][SRToolkit.bundle.pack]) providing custom classes that aren't
                installed as bundles. Each is copied verbatim into ``bundles/`` and matched
                to the classes it provides; an explicitly supplied bundle takes precedence
                over a coincidentally-installed one. An entry that matches no referenced
                class warns.
            strict: If ``True``, raise ``ValueError`` when any referenced custom class is
                covered by neither an installed bundle nor ``additional_bundles``. Defaults
                to ``False`` (warn and flag it in ``MANIFEST.md`` instead).

        Raises:
            ValueError: If ``strict`` and a referenced custom class is uncovered, or if an
                ``additional_bundles`` path is not a readable ``.srtk``.
        """
        path = Path(path)
        if path.suffix.lower() != ".zip":
            warnings.warn(
                f"[ExperimentGrid.export] Export path {str(path)!r} does not end in '.zip'; "
                "a zip archive is written regardless.",
                stacklevel=2,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Build the export contents in a temp dir, then zip them at the archive root.
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            self._write_export_dir(staging, include_results, additional_bundles, strict)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in sorted(staging.rglob("*")):
                    if file.is_file():
                        zf.write(file, file.relative_to(staging))

    def _write_export_dir(
        self,
        directory: Path,
        include_results: bool,
        additional_bundles: Optional[List[Union[str, Path]]] = None,
        strict: bool = False,
    ) -> None:
        """Write the export contents (``grid.json``, ``data/``, ``bundles/``, ``MANIFEST.md``)
        into ``directory``. Shared by [export][SRToolkit.experiments.ExperimentGrid.export],
        which zips the result."""
        directory.mkdir(parents=True, exist_ok=True)

        archived: List["tuple[str, str]"] = []
        referenced: List["tuple[str, str]"] = []
        for name, cfg in self.datasets.items():
            needs_archiving, reason = _classify_dataset_source(cfg)
            if needs_archiving:
                (directory / "data").mkdir(exist_ok=True)
                # Materialise from the cache (null) or samplers (seedless sample), then
                # freeze the exact arrays into the archive.
                SR_dataset.from_dict(cfg).to_archive(directory / "data" / f"{name}.zip")
                archived.append((name, reason))
            else:
                referenced.append((name, reason))

        # Pack custom code, then annotate the recipe so the recipient's bind machinery
        # relocates each custom class to its installed bundle. Annotation is stamped on
        # the dict that directly holds each *_class, so sibling configs (e.g. a custom
        # sampler and a custom source in one dataset) bind to their own bundles.
        all_configs = list(self.datasets.values()) + self.approach_configs + (self.callback_configs or [])
        packed, manual, class_to_bundle = _pack_custom_code(
            all_configs, directory / "bundles", additional_bundles, strict
        )

        recipe = self.to_dict()
        if class_to_bundle:
            recipe = _annotate_bundles(recipe, class_to_bundle)
        (directory / "grid.json").write_text(json.dumps(recipe, indent=2))

        results_included = self._copy_results(directory / "results") if include_results else False

        self._write_manifest(directory / "MANIFEST.md", archived, referenced, packed, manual, results_included)

    @classmethod
    def from_export(
        cls,
        path: Union[str, Path],
        results_dir: Optional[str] = None,
    ) -> "ExperimentGrid":
        """
        Load a grid from a ``.zip`` produced by
        [export][SRToolkit.experiments.ExperimentGrid.export].

        The archive is unpacked and:

        - any ``data/*.zip`` archives are imported into the local data cache (so the bundled
          datasets resolve without a network or samplers);
        - any ``bundles/`` and ``MANIFEST.md`` are written into ``results_dir`` so the
          recipient has the loose ``.srtk`` files to ``install`` (``from_export`` never
          auto-installs — that runs third-party code and mutates the global bundle store);
        - the grid is rebuilt from ``grid.json`` and any shipped ``results/`` are copied into
          ``results_dir`` so [progress][SRToolkit.experiments.ExperimentGrid.progress] and
          [load_results][SRToolkit.experiments.ExperimentGrid.load_results] see them.

        Custom classes still need their ``.srtk`` bundle(s) installed before jobs run — the
        returned grid is lazy and a warning names any that aren't importable yet. Class
        binding happens at job-run time, so installing after this call (before running) is
        fine. Install from ``<results_dir>/bundles/`` per the extracted ``MANIFEST.md``.

        Args:
            path: A ``.zip`` written by ``export``.
            results_dir: Where this machine's runs read/write results, and where ``bundles/``
                / ``MANIFEST.md`` are extracted. Defaults to the archive path with its suffix
                removed (e.g. ``run.zip`` → ``run/``), created on load.

        Returns:
            A fully configured ``ExperimentGrid``.

        Raises:
            FileNotFoundError: If ``path`` does not exist or the archive has no ``grid.json``.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"[ExperimentGrid.from_export] No export archive at {str(path)!r}.")

        if results_dir is not None:
            resolved_results_dir = os.path.abspath(results_dir)
        else:
            resolved_results_dir = os.path.abspath(str(path.with_suffix("")))
        os.makedirs(resolved_results_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(staging)

            grid_json = staging / "grid.json"
            if not grid_json.is_file():
                raise FileNotFoundError(
                    f"[ExperimentGrid.from_export] {str(path)!r} is not an export archive (no grid.json)."
                )

            # Import bundled data into the local cache.
            data_dir = staging / "data"
            if data_dir.is_dir():
                for zip_path in sorted(data_dir.glob("*.zip")):
                    # Side effect: extracts the archive's arrays into the data cache.
                    SR_dataset.from_archive(zip_path)

            # Persist bundles + manifest so the recipient can install the loose .srtk files.
            bundles_src = staging / "bundles"
            if bundles_src.is_dir():
                shutil.copytree(bundles_src, Path(resolved_results_dir) / "bundles", dirs_exist_ok=True)
            manifest_src = staging / "MANIFEST.md"
            if manifest_src.is_file():
                shutil.copy2(manifest_src, Path(resolved_results_dir) / "MANIFEST.md")

            grid = cls.from_dict(str(grid_json), results_dir=resolved_results_dir)

            # Copy any shipped results into results_dir (staging is a temp dir, so always copy).
            shipped_results = staging / "results"
            if shipped_results.is_dir():
                for src in shipped_results.rglob("exp_*.json"):
                    target = Path(resolved_results_dir) / src.relative_to(shipped_results)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target)

        return grid

    def _copy_results(self, dest: Path) -> bool:
        """Copy completed ``exp_*.json`` result files into ``dest``. Returns True if any."""
        copied = False
        for dataset_name in self.datasets:
            for approach_config in self.approach_configs:
                src_dir = Path(self.results_dir) / dataset_name / approach_config["name"]
                if not src_dir.is_dir():
                    continue
                for result_file in src_dir.glob("exp_*.json"):
                    target = dest / dataset_name / approach_config["name"] / result_file.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(result_file, target)
                    copied = True
        return copied

    def _write_manifest(
        self,
        path: Path,
        archived: "List[tuple[str, str]]",
        referenced: "List[tuple[str, str]]",
        packed: "List[tuple[str, str, str]]",
        manual: "List[tuple[str, str]]",
        results_included: bool,
    ) -> None:
        from SRToolkit import __version__

        lines = [
            "# ExperimentGrid export",
            "",
            f"Written with SRToolkit {__version__}.",
            "",
            "## Load on the recipient machine",
            "",
            "```python",
            "from SRToolkit.experiments import ExperimentGrid",
            "# Imports bundled data into the cache and extracts bundles/ next to the archive.",
            'grid = ExperimentGrid.from_export("<this-export>.zip")',
        ]
        if packed:
            lines += [
                "",
                "# Install the bundled code before running jobs (paths are under the extracted",
                "# results folder, e.g. '<this-export>/bundles/...').",
                "from SRToolkit.bundle import install",
            ]
            lines += [f'install("<this-export>/bundles/{fname}")' for _, fname, _ in packed]
        lines += [
            "```",
            "",
            "## Datasets",
            "",
        ]
        for name, reason in archived:
            lines.append(f"- `{name}` — **archived** ({reason}); data in `data/{name}.zip`.")
        for name, reason in referenced:
            lines.append(f"- `{name}` — referenced ({reason}).")
        if not (archived or referenced):
            lines.append("- _(none)_")

        lines += ["", "## Custom code", ""]
        if packed:
            lines.append("Bundled into `bundles/` — install each before running jobs (see snippet above):")
            lines.append("")
            lines += [f"- `bundles/{fname}` — {label}; {note}." for label, fname, note in packed]
        if manual:
            lines.append("")
            lines.append("**Could not pack automatically — handle these manually:**")
            lines.append("")
            lines += [f"- `{cls}` — {reason}." for cls, reason in manual]
        if not (packed or manual):
            lines.append("None — every referenced class is built into SRToolkit.")

        lines += [
            "",
            "## Results",
            "",
            "Included in `results/`." if results_included else "Not included (recipe only).",
            "",
        ]
        path.write_text("\n".join(lines))

    def save(self) -> None:
        """
        Persist the grid to a single ``results_dir/grid.json`` file.

        The file is the [to_dict][SRToolkit.experiments.ExperimentGrid.to_dict] recipe
        plus the machine-local ``adapted_states`` paths (which ``to_dict`` omits, since
        they don't travel between machines). This is the local-persistence counterpart to
        the shareable [export][SRToolkit.experiments.ExperimentGrid.export]: it keeps your
        run resumable in place, but is not meant for handing to someone else.

        [save_commands][SRToolkit.experiments.ExperimentGrid.save_commands] calls this
        automatically, so a separate ``save()`` call is only needed when checkpointing the
        grid without generating a commands file.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        payload = self.to_dict()
        if self._adapted_states:
            payload["adapted_states"] = self._adapted_states
        with open(os.path.join(self.results_dir, "grid.json"), "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(path: str) -> "ExperimentGrid":
        """
        Load a grid from a ``grid.json`` written by
        [save][SRToolkit.experiments.ExperimentGrid.save].

        Thin wrapper over [from_dict][SRToolkit.experiments.ExperimentGrid.from_dict] that
        also restores the local ``adapted_states`` paths. Dataset and approach instances
        are **not** created at load time — they are reconstructed lazily when jobs run.

        ``results_dir`` is taken from the directory containing ``path``, not from any value
        stored in the file, so a grid directory can be moved, mounted under a different
        path, or copied elsewhere and still load (and write its results) correctly.

        Args:
            path: Path to the ``grid.json`` file. The directory holding it becomes the
                loaded grid's ``results_dir``.

        Returns:
            A fully configured ``ExperimentGrid``.

        Raises:
            ValueError: If the file is not a supported ``ExperimentGrid`` recipe.
        """
        path = os.path.abspath(path)
        with open(path) as f:
            d = json.load(f)
        grid = ExperimentGrid.from_dict(d, results_dir=os.path.dirname(path))
        grid._adapted_states = d.get("adapted_states", {})
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
