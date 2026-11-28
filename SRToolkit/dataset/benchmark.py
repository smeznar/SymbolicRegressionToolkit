"""Benchmark collection for symbolic regression datasets."""

import copy
import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from SRToolkit.evaluation.evaluator import _as_ground_truth
from SRToolkit.utils.expression import Expression
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_constraints import ExpressionConstraint
from SRToolkit.utils.free_constants import Constant, _ground_truth_parameters
from SRToolkit.utils.grammar.grammar import Grammar
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.symbol_library import SymbolLibrary

from . import data_cache
from .background_knowledge import VariableInfo
from .data_source import DataSource, SampleSource, _config_samplers
from .dataset import Dataset, _derive_equation, _resolve_task
from .sampling import Sampler


def _save_arrays_to_cache(benchmark: str, version: str, name: str, X: np.ndarray, y: np.ndarray | None = None) -> None:
    """Write ``X`` (and optional ``y``) to the dataset's ``.npz`` in the cache version dir."""
    cache_path = data_cache.dataset_path(benchmark, version, name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if y is not None:
        np.savez(str(cache_path), X=X, y=y, allow_pickle=False)
    else:
        np.savez(str(cache_path), X=X, allow_pickle=False)


def _count_variables(symbol_library: SymbolLibrary) -> int:
    """Number of variable-type symbols in ``symbol_library``, or ``-1`` if there are none."""
    num_vars = sum(1 for sym in symbol_library.symbols.values() if sym.type == "var")
    return num_vars if num_vars > 0 else -1


class Benchmark:
    def __init__(
        self,
        benchmark_name: str,
        datasets: list[Dataset | tuple[str, Dataset]] | None = None,
        metadata: dict[str, Any] | None = None,
        version: str = "1.0.0",
        base_dir: str | None = None,
    ):
        """A named, persistent collection of symbolic regression datasets.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of Benchmark with additional functionality
            >>> len(benchmark.list_datasets(verbose=False))
            100

        Args:
            benchmark_name: Name of this benchmark.
            datasets: Initial datasets to add. Each element can be an
                [Dataset][SRToolkit.dataset.dataset.Dataset] instance (auto-named as
                ``"<benchmark_name>_<index>"``) or a ``(name, Dataset)`` tuple.
            metadata: Optional dictionary of benchmark-level metadata (e.g. citation, description).
            version: Version string for this benchmark. Defaults to ``"1.0.0"``.
            base_dir: Directory where dataset files are stored or will be written.
                Optional — if omitted, the data cache is used exclusively.

        Raises:
            ValueError: If any element of ``datasets`` is not an
                [Dataset][SRToolkit.dataset.dataset.Dataset] or a valid ``(name, Dataset)`` tuple.
        """
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.version = version
        self.datasets: dict[str, dict[str, Any]] = {}
        self.metadata = {} if metadata is None else metadata
        if datasets is not None:
            for i, dataset in enumerate(datasets):
                if isinstance(dataset, Dataset):
                    self.add_dataset_instance(benchmark_name + "_" + str(i + 1), dataset)
                elif isinstance(dataset, tuple) and isinstance(dataset[0], str) and isinstance(dataset[1], Dataset):
                    self.add_dataset_instance(dataset[0], dataset[1])
                else:
                    raise ValueError(
                        "[Benchmark] Dataset inside the datasets argument must be either a tuple "
                        "(name, Dataset) or a Dataset instance."
                    )

    def add_dataset_instance(self, dataset_name: str, dataset: Dataset):
        """Adds an instance of the Dataset class to the benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of Benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> isinstance(dataset, Dataset)
            True
            >>> bm = Benchmark("BM")
            >>> bm.add_dataset_instance("I.16.6", dataset)

        Args:
             dataset_name: The name of the dataset.
             dataset: An instance of the Dataset class.

        Raises:
            Exception: If the dataset name already exists in the benchmark.
        """
        if dataset_name in self.datasets:
            raise ValueError(f"Dataset {dataset_name} already exists in the benchmark.")
        else:
            self.datasets[dataset_name] = {}
        self.datasets[dataset_name]["sr_dataset"] = dataset
        self.datasets[dataset_name]["num_variables"] = dataset.X.shape[1]

    def add_dataset(
        self,
        symbol_library: SymbolLibrary,
        dataset: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        dataset_name: str | None = None,
        ranking_function: str = "rmse",
        max_evaluations: int = -1,
        ground_truth: list[str] | Expression | None = None,
        original_equation: str | None = None,
        success_threshold: float | None = None,
        seed: int | None = None,
        dataset_metadata: dict | None = None,
        samplers: list[Sampler] | Mapping[str, Sampler] | None = None,
        data_source: DataSource | None = None,
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
    ):
        """Add a dataset to the benchmark without building it.

        The entry is stored in format 5 (see
        [Dataset.to_dict][SRToolkit.dataset.dataset.Dataset.to_dict]) and built by
        [create_dataset][SRToolkit.dataset.benchmark.Benchmark.create_dataset]. The
        arguments are validated as by ``Dataset`` before any cache file is written. Array data
        given in ``dataset`` is written to the data cache. A ``UserWarning`` is issued when
        ``ground_truth`` is ``None``.

        Args:
            symbol_library: The symbol library to use.
            dataset: Direct data for the dataset: a bare 2-D ndarray ``X`` (features only; for
                ``"rmse"``, ``y`` is computed from ``ground_truth``), or an ``(X, y)`` tuple. Use ``None`` together with ``data_source`` to have the
                cache layer materialise the data instead. When ``data_source`` is provided
                this argument is ignored. To use data from a local file, load it
                yourself (e.g. with ``numpy.load``) and pass the arrays here.
            dataset_name: The name of the dataset. Auto-generated if ``None``.
            ranking_function: ``"rmse"`` or ``"bed"``.
            max_evaluations: Maximum expressions to evaluate. Negative (default ``-1``) means no limit;
                ``0`` is rejected.
            ground_truth: Ground truth expression, as a token list or an
                [Expression][SRToolkit.utils.expression.Expression] carrying the specs of its
                labelled constants, as for ``Dataset``; read in ``symbol_library``. Saved as
                tokens, its specs as ``ground_truth_constants``.
            original_equation: Human-readable answer, as for
                [Dataset][SRToolkit.dataset.dataset.Dataset]; ``None`` is stored as ``None``
                and derived when the dataset is built (see
                [Dataset.original_equation][SRToolkit.dataset.dataset.Dataset.original_equation]).
            success_threshold: Error threshold for success. ``None`` means no threshold.
            seed: Random seed.
            dataset_metadata: Optional dataset-level metadata dict, merged over the benchmark's
                metadata.
            samplers: One sampler per variable, as a list in the library's variable order or a
                mapping ``{token: sampler}``, as for ``Dataset``. They define the problem's
                input distribution and power
                [resample][SRToolkit.dataset.dataset.Dataset.resample]; a
                [SampleSource][SRToolkit.dataset.data_source.SampleSource] draws from them, and a
                ``"bed"`` ranking draws its evaluation points from them.
            data_source: Optional [DataSource][SRToolkit.dataset.data_source.DataSource]
                describing where the data comes from
                ([UrlSource][SRToolkit.dataset.data_source.UrlSource] or
                [SampleSource][SRToolkit.dataset.data_source.SampleSource]). When provided,
                the ``dataset`` argument is ignored and the cache layer manages
                materialisation.
            variables: Information per variable, keyed by token, as for
                [Dataset][SRToolkit.dataset.dataset.Dataset].
            target: Information about the target, as for ``Dataset``.
            named_constants: Information per named constant, as for ``Dataset``.
            description: Free text about the task, as for ``Dataset``.
            constant_bounds: Range of the default spec, as for ``Dataset``.
            default_constant: The default spec of free constants, as for ``Dataset``.
            reference_grammar: A grammar an approach may start from, as for ``Dataset``.
            max_constants: Free-constant limit, as for ``Dataset`` (``None``: 8).
            max_expr_length: Token limit, as for ``Dataset`` (``None``: no limit).
            required_constraints: Further required constraints, as for ``Dataset``.
            advisory_constraints: Hints about the search space, as for ``Dataset``.
            settings: Settings of the evaluator, ranking and parameter estimation, as for
                ``Dataset``.
            ranking_resample: Resample the ranking's randomness each run, as for ``Dataset``.

        Raises:
            ValueError: If the dataset name already exists; ``max_evaluations`` is ``0``; the
                background-knowledge, limit or constraint arguments are rejected as by
                [Dataset][SRToolkit.dataset.dataset.Dataset]; ``ground_truth`` is an
                ndarray, or missing where the ranking needs it (``"bed"``, or ``"rmse"`` with a
                bare ndarray ``dataset``, not an ``(X, y)`` tuple); ``dataset`` is a tuple that is
                not two numpy arrays; the data has a different number of columns than the symbol library
                has variables; both ``dataset`` and ``data_source`` are ``None`` (the
                dataset could never be loaded); a free spec of ``ground_truth`` has a ``value``
                outside the bounds it takes from ``default_constant``; ``"rmse"`` has to generate ``y`` (a bare ndarray ``dataset``, or a
                ``SampleSource``, also inside a ``FallbackSource``) and a free constant of
                ``ground_truth`` has no fixed value in its specs; or ``ground_truth`` cannot be
                evaluated on a bare ndarray ``dataset`` (raised from the original error).
            TypeError: As for ``Dataset``.
            ExpressionParseError: If ``ground_truth`` does not parse.

        Examples:
            >>> import numpy as np
            >>> from SRToolkit.utils import SymbolLibrary
            >>> bm = Benchmark("BM")
            >>> X = np.random.rand(10, 2)
            >>> bm.add_dataset(SymbolLibrary.default_symbols(2), (X, X[:, 0] + X[:, 1]), dataset_name="test_ds",
            ...     ground_truth=["X_0", "+", "X_1"], original_equation="y = x0 + x1")
            >>> len(bm.list_datasets(verbose=False))
            1
        """
        if dataset_name is None:
            dataset_name = f"{self.benchmark_name}_{len(self.datasets) + 1}"

        # Fail fast before any cache files are written, mirroring add_dataset_instance.
        if dataset_name in self.datasets:
            raise ValueError(f"[Benchmark.add_dataset] Dataset '{dataset_name}' already exists in the benchmark.")
        if max_evaluations == 0:
            raise ValueError(
                "[Benchmark.add_dataset] max_evaluations must be positive, or negative for no limit; got 0."
            )

        # Validate the task (background knowledge, required constraints) before anything is written.
        # Array data is checked against the library too (dataset is ignored when data_source is given).
        X = None
        if data_source is None:
            if isinstance(dataset, np.ndarray):
                X = dataset
            elif isinstance(dataset, tuple) and len(dataset) == 2 and isinstance(dataset[0], np.ndarray):
                X = dataset[0]
        bk, required, settings = _resolve_task(
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
        try:
            gt = _as_ground_truth(ground_truth, bk.symbol_library, cast(Constant, bk.default_constant))
        except (TypeError, ValueError) as e:
            raise type(e)(f"[Benchmark.add_dataset] {e}") from None
        # RMSE generates y from the ground truth for a bare X or a SampleSource: its free constants
        # need fixed values, checked here before anything is written.
        parameters = None
        # A SampleSource generates y, also as one of a FallbackSource's sources.
        sources = [data_source, *getattr(data_source, "sources", [])]
        generates_y = any(isinstance(s, SampleSource) for s in sources) or (
            data_source is None and isinstance(dataset, np.ndarray)
        )
        if ranking_function == "rmse" and gt is not None and generates_y:
            parameters = _ground_truth_parameters(gt, f"[Benchmark.add_dataset] Dataset '{dataset_name}'")
        symbol_library = bk.symbol_library
        if dataset is None and data_source is None:
            raise ValueError(
                "[Benchmark.add_dataset] Give the data ('dataset') or a 'data_source' to generate it from; "
                "without either the dataset could never be loaded."
            )

        merged_metadata = copy.deepcopy(self.metadata)
        if dataset_metadata:
            merged_metadata.update(dataset_metadata)

        entry: dict[str, Any] = {
            "format_version": 5,
            "dataset_name": dataset_name,
            "benchmark": self.benchmark_name,
            "version": self.version,
            "ranking_function": ranking_function,
            "max_evaluations": max_evaluations,
            "success_threshold": success_threshold,
            "seed": seed,
            "dataset_metadata": merged_metadata,
            "original_equation": original_equation,
            "settings": settings.to_dict(),
            "ranking_resample": ranking_resample,
            "ground_truth": None if gt is None else list(gt.tokens),
            "ground_truth_constants": (
                [c.to_dict() for c in sorted(gt.free_constants, key=lambda c: c.label or "")]
                if gt is not None and gt.free_constants
                else None
            ),
            "background_knowledge": bk.to_dict(),
            "required_constraints": [c.to_dict() for c in required],
            # Samplers and array data are validated to match the library's variables.
            "num_variables": _count_variables(symbol_library),
        }

        if ground_truth is None:
            if ranking_function == "bed":
                raise ValueError("[Benchmark.add_dataset] For 'bed' ranking, the ground truth must be provided.")
            warnings.warn(
                "[Benchmark.add_dataset] 'ground_truth' argument not provided. We recommend providing it "
                "for more transparent evaluation."
            )

        # Resolve the data source / write the arrays.
        if data_source is not None:
            # Lazy: the cache layer materialises X (and y) on first use; nothing written here.
            entry["data_source"] = data_source.to_dict()
        else:
            entry["data_source"] = None
            if isinstance(dataset, np.ndarray):
                if ranking_function == "rmse":
                    if gt is None:
                        raise ValueError(
                            "[Benchmark.add_dataset] For 'rmse' ranking, if the dataset argument is a numpy "
                            "array, the ground truth must be provided."
                        )
                    try:
                        y = compile_expr(gt)(dataset, parameters)
                    except Exception as e:
                        raise ValueError(
                            f"[Benchmark.add_dataset] Could not evaluate the ground truth. Original error: {e}"
                        ) from e
                    _save_arrays_to_cache(self.benchmark_name, self.version, dataset_name, dataset, y)
                elif ranking_function == "bed":
                    _save_arrays_to_cache(self.benchmark_name, self.version, dataset_name, dataset)

            elif isinstance(dataset, tuple):
                if (
                    len(dataset) != 2
                    or not isinstance(dataset[0], np.ndarray)
                    or not isinstance(dataset[1], np.ndarray)
                ):
                    raise ValueError(
                        "[Benchmark.add_dataset] When the dataset argument is a tuple, it must be (X, y) "
                        "with both values numpy arrays."
                    )
                if ranking_function == "bed":
                    warnings.warn(
                        "[Benchmark.add_dataset] 'bed' ranking only utilizes the array with features. "
                        "Array with targets will be ignored."
                    )
                _save_arrays_to_cache(self.benchmark_name, self.version, dataset_name, dataset[0], dataset[1])

        self.datasets[dataset_name] = entry

    def add_from_samplers(
        self,
        ground_truth: list[str] | Expression,
        samplers: list[Sampler] | Mapping[str, Sampler] | None = None,
        symbol_library: SymbolLibrary | None = None,
        n_samples: int = 10000,
        seed: int | None = None,
        ranking_function: str = "rmse",
        dataset_name: str | None = None,
        original_equation: str | None = None,
        success_threshold: float | None = None,
        max_evaluations: int = -1,
        dataset_metadata: dict | None = None,
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
        """Add a dataset described by a ground-truth expression and per-variable samplers.

        The benchmark-level counterpart to
        [Dataset.from_samplers][SRToolkit.dataset.dataset.Dataset.from_samplers]: it
        attaches a [SampleSource][SRToolkit.dataset.data_source.SampleSource], so the data is
        generated lazily from the samplers (and, for RMSE, ``ground_truth``) the first time the
        dataset is materialised via
        [create_dataset][SRToolkit.dataset.benchmark.Benchmark.create_dataset]. Nothing is
        written to the data cache here.

        Args:
            ground_truth: Ground-truth expression as a token list or an
                [Expression][SRToolkit.utils.expression.Expression] with its specs; with
                ``ranking_function="rmse"``, every free constant in it needs a fixed value in its
                specs.
            samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per variable, as a list
                in the library's variable order or a mapping ``{token: sampler}``.
            symbol_library: Token vocabulary. ``None`` uses the library of an ``Expression`` ground
                truth, and for a token list
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]
                with one variable per sampler.
            n_samples: Number of input rows to generate on materialisation. Defaults to ``10000``.
            seed: Random seed stored on the
                [SampleSource][SRToolkit.dataset.data_source.SampleSource].
            ranking_function: ``"rmse"`` or ``"bed"``.
            dataset_name: Name of the dataset. Auto-generated if ``None``.
            original_equation: As for
                [add_dataset][SRToolkit.dataset.benchmark.Benchmark.add_dataset].
            success_threshold: Error threshold for success. ``None`` means no threshold.
            max_evaluations: Maximum expressions to evaluate. Negative (default ``-1``) means no limit;
                ``0`` is rejected.
            dataset_metadata: Optional dataset-level metadata dict.
            variables: Information per variable, as for
                [Dataset][SRToolkit.dataset.dataset.Dataset].
            target: Information about the target, as for ``Dataset``.
            named_constants: Information per named constant, as for ``Dataset``.
            description: Free text about the task, as for ``Dataset``.
            constant_bounds: Range of the default spec, as for ``Dataset``.
            default_constant: The default spec of free constants, as for ``Dataset``.
            reference_grammar: A grammar an approach may start from, as for ``Dataset``.
            max_constants: Free-constant limit, as for ``Dataset`` (``None``: 8).
            max_expr_length: Token limit, as for ``Dataset`` (``None``: no limit).
            required_constraints: Further required constraints, as for ``Dataset``.
            advisory_constraints: Hints about the search space, as for ``Dataset``.
            settings: Settings of the evaluator, ranking and parameter estimation, as for
                ``Dataset``.
            ranking_resample: Resample the ranking's randomness each run, as for ``Dataset``.

        Raises:
            ValueError: If ``samplers`` is ``None`` or empty, or as for
                [add_dataset][SRToolkit.dataset.benchmark.Benchmark.add_dataset].
            TypeError: As for ``add_dataset``.
            ExpressionParseError: As for ``add_dataset``.

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> bm = Benchmark("BM")
            >>> bm.add_from_samplers(["X_0", "+", "X_1"],
            ...     [UniformSampling(0, 5), UniformSampling(0, 5)], dataset_name="add",
            ...     n_samples=100, seed=0)
            >>> ds = bm.create_dataset("add")
            >>> ds.X.shape
            (100, 2)
        """
        if not samplers:
            raise ValueError(
                "[Benchmark.add_from_samplers] 'samplers' must be non-empty (one sampler per input variable)."
            )
        if symbol_library is None:
            symbol_library = (
                ground_truth.symbol_library
                if isinstance(ground_truth, Expression)
                else SymbolLibrary.default_symbols(len(samplers))
            )

        self.add_dataset(
            symbol_library=symbol_library,
            dataset=None,
            dataset_name=dataset_name,
            ranking_function=ranking_function,
            max_evaluations=max_evaluations,
            ground_truth=ground_truth,
            original_equation=original_equation,
            success_threshold=success_threshold,
            seed=seed,
            dataset_metadata=dataset_metadata,
            samplers=samplers,
            data_source=SampleSource(n_samples=n_samples, seed=seed),
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

    def create_dataset(
        self,
        dataset_name: str,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> Dataset:
        """Create the dataset stored under ``dataset_name``.

        When ``n_samples`` is provided the returned dataset contains freshly sampled data
        instead of the pre-generated data on disk. The dataset must have samplers defined
        (see ``samplers`` argument of
        [add_dataset][SRToolkit.dataset.benchmark.Benchmark.add_dataset]). Entries of formats
        2–4 from older saved benchmarks are migrated as by
        [Dataset.from_dict][SRToolkit.dataset.dataset.Dataset.from_dict].

        Args:
            dataset_name: The name of the dataset to create.
            n_samples: If provided, generate a fresh dataset with this many samples using
                the stored samplers instead of loading pre-generated data from disk.
            seed: Random seed used when ``n_samples`` is provided. If ``None``, no seed is set.

        Returns:
            An [Dataset][SRToolkit.dataset.dataset.Dataset] instance containing the
            data, ground truth expression, and metadata for the given dataset.

        Raises:
            ValueError: If the dataset name is not found; ``n_samples`` is provided but the
                dataset has no samplers defined; or ``Dataset.from_dict`` raises a ``ValueError``
                for the entry (raised from it, with the dataset name).
            TypeError: As for ``Dataset.from_dict``, unchanged.
            ImportError: As for ``Dataset.from_dict``, unchanged.
            FileNotFoundError: As for ``Dataset.from_dict``, unchanged.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of Benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> dataset.X.shape
            (10000, 3)
            >>> dataset_small = benchmark.create_dataset('I.16.6', n_samples=500, seed=0)
            >>> dataset_small.X.shape
            (500, 3)
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        if n_samples is not None:
            config = self.datasets[dataset_name]
            if _config_samplers(config) is None:
                raise ValueError(f"[Benchmark.create_dataset] Cannot resample '{dataset_name}': no samplers defined.")
            # _create_from_entry may return the benchmark's stored Dataset instance
            # (for entries added via add_dataset_instance). Copy before mutating X/y so
            # resampling never corrupts the stored dataset or aliases across callers.
            dataset = copy.deepcopy(self._create_from_entry(config))
            return dataset.resample_inplace(n_samples, seed=seed)

        # Loading canonical data: materialisation is driven entirely by each dataset's
        # ``data_source`` — the built-ins use a FallbackSource that downloads the canonical
        # archive once and regenerates from samplers only if the download is unavailable.
        return self._create_from_entry(self.datasets[dataset_name])

    def _create_from_entry(self, entry: dict) -> Dataset:
        """Internal helper: create a Dataset from an entry dict."""
        if "sr_dataset" in entry:
            return entry["sr_dataset"]
        try:
            return Dataset.from_dict(entry)
        except ValueError as e:
            # Name the benchmark entry; other exception types pass through as Dataset.from_dict raises them.
            raise ValueError(
                f"[Benchmark.create_dataset] Could not create dataset '{entry.get('dataset_name')}' from its entry: {e}"
            ) from e

    def list_datasets(self, verbose: bool = True, num_variables: int = -1) -> list[str]:
        """Return the names of the datasets, optionally printing a line per dataset.

        Nothing is built. With ``verbose``, each line gives the name, the number of variables and
        the equation: the entry's ``original_equation``, or derived as for
        [Dataset.original_equation][SRToolkit.dataset.dataset.Dataset.original_equation]
        when not given (for an entry added with ``add_dataset_instance``, that dataset's
        ``original_equation``). An entry without an equation shows only its name and number of
        variables.

        Args:
            verbose: If ``True``, print one line per dataset.
            num_variables: If not ``-1``, only return datasets with this many input variables.

        Returns:
            The dataset names, sorted by number of variables, then by name.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of Benchmark with additional functionality
            >>> len(benchmark.list_datasets(num_variables=2, verbose=False))
            15
            >>> datasets_with_8_vars = benchmark.list_datasets(num_variables=8, verbose=False)
            >>> datasets_with_8_vars[0]
            'II.36.38'
        """
        datasets = [
            dataset_name
            for dataset_name in self.datasets
            if num_variables < 0 or self.datasets[dataset_name].get("num_variables") == num_variables
        ]
        datasets = sorted(
            datasets,
            key=lambda dataset_name: (
                self.datasets[dataset_name].get("num_variables", -1),
                dataset_name,
            ),
        )

        if verbose:
            part1 = []
            part2 = []
            part3 = []
            max_length_1 = 0
            max_length_2 = 0
            for d in datasets:
                nv = self.datasets[d].get("num_variables", -1)
                if nv == 1:
                    variable_str = "1 variable"
                elif nv is None or nv < 1:
                    variable_str = "Amount of variables unknown"
                else:
                    variable_str = f"{nv} variables"
                part1.append(d + ":")
                part2.append(variable_str)
                entry = self.datasets[d]
                if "sr_dataset" in entry:
                    equation = entry["sr_dataset"].original_equation
                elif entry.get("original_equation") is not None or entry.get("ground_truth") is None:
                    equation = entry.get("original_equation")
                else:
                    bk = entry.get("background_knowledge") or {}
                    target = bk.get("target")
                    equation = _derive_equation(
                        entry["ground_truth"],
                        {k: VariableInfo.from_dict(v) for k, v in (bk.get("variables") or {}).items()},
                        VariableInfo.from_dict(target) if target is not None else None,
                    )
                part3.append(equation)
                if len(d) + 1 > max_length_1:
                    max_length_1 = len(d) + 1
                if len(variable_str) > max_length_2:
                    max_length_2 = len(variable_str)

            for p1, p2, p3 in zip(part1, part2, part3):
                line = f"{p1:<{max_length_1}} {p2:<{max_length_2}}"
                print(line if p3 is None else f"{line}, Expression: {p3}")
        return datasets

    def to_dict(self) -> dict:
        """Serialise the benchmark to a pure JSON-safe dictionary.

        Dataset entries that have an ``sr_dataset`` key (added via
        [add_dataset_instance][SRToolkit.dataset.benchmark.Benchmark.add_dataset_instance])
        are serialised via ``Dataset.to_dict()``.

        Returns:
            A JSON-safe dict representing the full benchmark configuration.
        """
        datasets_out = {}
        for name, entry in self.datasets.items():
            if "sr_dataset" in entry:
                datasets_out[name] = entry["sr_dataset"].to_dict()
            else:
                datasets_out[name] = {k: v for k, v in entry.items() if k != "sr_dataset"}

        return {
            "format_version": 2,
            "type": "Benchmark",
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "metadata": self.metadata,
            "datasets": datasets_out,
        }

    @classmethod
    def from_dict(cls, d: dict | str | Path) -> "Benchmark":
        """Reconstruct a Benchmark from a config dict or a saved JSON file.

        Load ``.zip`` archives with [share.load][SRToolkit.share.load].
        Dataset entries of formats 2–4 in older files are accepted and migrated as by
        [Dataset.from_dict][SRToolkit.dataset.dataset.Dataset.from_dict]. The entries are
        copied, so ``d`` is not changed.

        Args:
            d: A dict produced by [to_dict][SRToolkit.dataset.benchmark.Benchmark.to_dict],
                or a path to a JSON file.

        Returns:
            An [Benchmark][SRToolkit.dataset.benchmark.Benchmark] instance.

        Raises:
            ValueError: If ``d`` is a ``.zip`` path.
        """
        if isinstance(d, (str, Path)):
            if str(d).endswith(".zip"):
                raise ValueError(
                    "[Benchmark.from_dict] Received a '.zip' path. Load archives with SRToolkit.share.load(path)."
                )
            with open(d) as f:
                dd = json.load(f)
        else:
            dd = d

        benchmark_name = dd["benchmark_name"]
        version = dd.get("version", "1.0.0")
        metadata = dd.get("metadata", {})

        b = cls(benchmark_name, version=version, metadata=metadata)

        for name, entry in dd.get("datasets", {}).items():
            # Store a copy of the entry (the caller's dict is not changed); materialisation is lazy
            # via create_dataset.
            entry = copy.deepcopy(entry)
            b.datasets[name] = entry
            if "num_variables" not in entry:
                # Try to infer
                samplers = _config_samplers(entry)
                if samplers is not None:
                    entry["num_variables"] = len(samplers)
                else:
                    entry["num_variables"] = -1

        return b
