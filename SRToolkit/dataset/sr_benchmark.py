"""
Benchmark collection for symbolic regression datasets.
"""

import copy
import json
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Unpack

from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EstimationSettings

from .data_source import DataSource, SampleSource
from .sampling import Sampler
from .sr_dataset import SR_dataset


def _save_arrays_to_cache(
    benchmark: str, version: str, name: str, X: np.ndarray, y: Optional[np.ndarray] = None
) -> None:
    """Write ``X`` (and optional ``y``) to the dataset's ``.npz`` in the cache version dir."""
    from SRToolkit.dataset import data_cache

    cache_path = data_cache.dataset_path(benchmark, version, name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if y is not None:
        np.savez(str(cache_path), X=X, y=y, allow_pickle=False)
    else:
        np.savez(str(cache_path), X=X, allow_pickle=False)


def _save_gt_array_to_cache(benchmark: str, version: str, name: str, gt: np.ndarray) -> None:
    """
    Write an ndarray (a ``bed`` behaviour matrix) ground truth to ``<name>_gt.npy`` beside the
    dataset ``.npz``. Such a ground truth is not JSON-safe, so the entry stores ``None`` and
    [SR_dataset.from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict] reloads it from here.
    """
    from SRToolkit.dataset import data_cache

    cache_path = data_cache.dataset_path(benchmark, version, name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path.parent / f"{name}_gt.npy"), gt)


def _count_variables(symbol_library: SymbolLibrary) -> int:
    """Number of variable-type symbols in ``symbol_library``, or ``-1`` if there are none."""
    num_vars = sum(1 for sym in symbol_library.symbols.values() if sym.get("type") == "var")
    return num_vars if num_vars > 0 else -1


class SR_benchmark:
    def __init__(
        self,
        benchmark_name: str,
        datasets: Optional[List[Union[SR_dataset, Tuple[str, SR_dataset]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        base_dir: Optional[str] = None,
    ):
        """
        A named, persistent collection of symbolic regression datasets.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> len(benchmark.list_datasets(verbose=False))
            100

        Args:
            benchmark_name: Name of this benchmark.
            datasets: Initial datasets to add. Each element can be an
                [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance (auto-named as
                ``"<benchmark_name>_<index>"``) or a ``(name, SR_dataset)`` tuple.
            metadata: Optional dictionary of benchmark-level metadata (e.g. citation, description).
            version: Version string for this benchmark. Defaults to ``"1.0.0"``.
            base_dir: Directory where dataset files are stored or will be written.
                Optional — if omitted, the data cache is used exclusively.

        Raises:
            ValueError: If any element of ``datasets`` is not an
                [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] or a valid ``(name, SR_dataset)`` tuple.
        """
        self.benchmark_name = benchmark_name
        self.base_dir = base_dir
        self.version = version
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.metadata = {} if metadata is None else metadata
        if datasets is not None:
            for i, dataset in enumerate(datasets):
                if isinstance(dataset, SR_dataset):
                    self.add_dataset_instance(benchmark_name + "_" + str(i + 1), dataset)
                elif isinstance(dataset, tuple) and isinstance(dataset[0], str) and isinstance(dataset[1], SR_dataset):
                    self.add_dataset_instance(dataset[0], dataset[1])
                else:
                    raise ValueError(
                        "[SR_benchmark] Dataset inside the datasets argument must be either a tuple "
                        "(name, SR_dataset) or a SR_dataset instance."
                    )

    def add_dataset_instance(self, dataset_name: str, dataset: SR_dataset):
        """
        Adds an instance of the SR_dataset class to the benchmark.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> isinstance(dataset, SR_dataset)
            True
            >>> bm = SR_benchmark("BM")
            >>> bm.add_dataset_instance("I.16.6", dataset)

        Args:
             dataset_name: The name of the dataset.
             dataset: An instance of the SR_dataset class.

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
        dataset: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None,
        dataset_name: Optional[str] = None,
        ranking_function: str = "rmse",
        max_evaluations: int = -1,
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        samplers: Optional[List[Sampler]] = None,
        data_source: Optional[DataSource] = None,
        **kwargs: Unpack[EstimationSettings],
    ):
        """
        Adds a dataset to the benchmark.

        Examples:
            >>> import tempfile, numpy as np
            >>> from SRToolkit.utils import SymbolLibrary
            >>> bm = SR_benchmark("BM")
            >>> X = np.random.rand(10, 2)
            >>> y = X[:, 0] + X[:, 1]
            >>> bm.add_dataset(SymbolLibrary.default_symbols(2),(X, y),dataset_name="test_ds",ground_truth=["X_0", "+", "X_1"],original_equation="y = x0 + x1")
            >>> len(bm.list_datasets(verbose=False))
            1

        Args:
            dataset: Direct data for the dataset: a 2-D numpy array (features) or a
                ``(X, y)`` tuple. Use ``None`` together with ``data_source`` to have the
                cache layer materialise the data instead. When ``data_source`` is provided
                this argument is ignored. To use data from a local file, load it
                yourself (e.g. with ``numpy.load``) and pass the arrays here.
            symbol_library: The symbol library to use.
            dataset_name: The name of the dataset. Auto-generated if ``None``.
            ranking_function: ``"rmse"`` or ``"bed"``.
            max_evaluations: Maximum expressions to evaluate. ``-1`` means no limit.
            ground_truth: Ground truth expression.
            original_equation: Human-readable equation string.
            success_threshold: Error threshold for success. ``None`` means no threshold.
            seed: Random seed.
            dataset_metadata: Optional dataset-level metadata dict.
            samplers: Optional list of samplers (one per input variable). They define the
                problem's input distribution and power
                [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample]; a
                [SampleSource][SRToolkit.dataset.data_source.SampleSource] draws from them.
            data_source: Optional [DataSource][SRToolkit.dataset.data_source.DataSource]
                describing where the data comes from
                ([UrlSource][SRToolkit.dataset.data_source.UrlSource] or
                [SampleSource][SRToolkit.dataset.data_source.SampleSource]). When provided,
                the ``dataset`` argument is ignored and the cache layer manages
                materialisation.
            **kwargs: Estimation settings forwarded to
                [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].

        Raises:
            ValueError: Various validation errors (see below).
        """

        if dataset_name is None:
            dataset_name = f"{self.benchmark_name}_{len(self.datasets) + 1}"

        # Fail fast before any cache files are written, mirroring add_dataset_instance.
        if dataset_name in self.datasets:
            raise ValueError(f"[SR_benchmark.add_dataset] Dataset '{dataset_name}' already exists in the benchmark.")

        if "bed_X" in kwargs and kwargs["bed_X"] is not None:
            kwargs["bed_X"] = kwargs["bed_X"].tolist()

        # An ndarray ('bed' behaviour matrix) ground truth is not JSON-safe: the entry
        # stores None and the array is written to <name>_gt.npy at the end.
        if isinstance(ground_truth, Node):
            ground_truth_out: Optional[Union[List[str], str]] = ground_truth.to_list()
        elif isinstance(ground_truth, np.ndarray):
            ground_truth_out = None
        else:
            ground_truth_out = ground_truth

        merged_metadata = copy.deepcopy(self.metadata)
        if dataset_metadata:
            merged_metadata.update(dataset_metadata)

        entry: Dict[str, Any] = {
            "format_version": 2,
            "dataset_name": dataset_name,
            "benchmark": self.benchmark_name,
            "version": self.version,
            "symbol_library": symbol_library.to_dict(),
            "ranking_function": ranking_function,
            "max_evaluations": max_evaluations,
            "success_threshold": success_threshold,
            "seed": seed,
            "dataset_metadata": merged_metadata,
            "original_equation": original_equation,
            "kwargs": kwargs,
            "ground_truth": ground_truth_out,
            "samplers": [s.to_dict() for s in samplers] if samplers is not None else None,
            # Default count: one variable per sampler, else the symbol library. The array
            # branches below override this with the real column count when data is supplied.
            "num_variables": len(samplers) if samplers is not None else _count_variables(symbol_library),
        }

        # Derive a human-readable equation from a token-list ground truth if absent.
        if ground_truth is None:
            if ranking_function == "bed":
                raise ValueError("[SR_benchmark.add_dataset] For 'bed' ranking, the ground truth must be provided.")
            warnings.warn(
                "[SR_benchmark.add_dataset] 'ground_truth' argument not provided. We recommend providing it "
                "for more transparent evaluation."
            )
        elif original_equation is None and isinstance(ground_truth, list):
            entry["original_equation"] = "y = " + "".join(ground_truth)

        # Resolve the data source / write the arrays. num_variables defaults to the sampler /
        # symbol-library count set above; the array branches override it with the real shape.
        if data_source is not None:
            # Lazy: the cache layer materialises X (and y) on first use; nothing written here.
            entry["data_source"] = data_source.to_dict()
        else:
            entry["data_source"] = None
            if isinstance(dataset, np.ndarray):
                if ranking_function == "rmse":
                    if ground_truth is None:
                        raise ValueError(
                            "[SR_benchmark.add_dataset] For 'rmse' ranking, if the dataset argument is a numpy "
                            "array, the ground truth must be provided."
                        )
                    if isinstance(ground_truth, np.ndarray):
                        raise ValueError(
                            "[SR_benchmark.add_dataset] For 'rmse' ranking, the ground truth must be "
                            "a list of tokens from the symbol library or a SRToolkit.utils.Node object."
                        )
                    try:
                        y = compile_expr(ground_truth, symbol_library)(dataset, None)
                    except Exception as e:
                        raise Exception(
                            f"[SR_benchmark.add_dataset] Could not evaluate the ground truth. Original error: {e}"
                        )
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
                        "[SR_benchmark.add_dataset] When the dataset argument is a tuple, it must be (X, y) "
                        "with both values numpy arrays."
                    )
                if ranking_function == "bed":
                    warnings.warn(
                        "[SR_benchmark.add_dataset] 'bed' ranking only utilizes the array with features. "
                        "Array with targets will be ignored."
                    )
                _save_arrays_to_cache(self.benchmark_name, self.version, dataset_name, dataset[0], dataset[1])

        if isinstance(ground_truth, np.ndarray):
            _save_gt_array_to_cache(self.benchmark_name, self.version, dataset_name, ground_truth)

        self.datasets[dataset_name] = entry

    def add_from_samplers(
        self,
        ground_truth: Union[List[str], Node],
        samplers: List[Sampler],
        symbol_library: Optional[SymbolLibrary] = None,
        n_samples: int = 10000,
        seed: Optional[int] = None,
        ranking_function: str = "rmse",
        dataset_name: Optional[str] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        max_evaluations: int = -1,
        dataset_metadata: Optional[dict] = None,
        **kwargs: Unpack[EstimationSettings],
    ) -> None:
        """
        Add a dataset described only by a ground-truth expression and per-variable samplers.

        This is the benchmark-level counterpart to
        [SR_dataset.from_samplers][SRToolkit.dataset.sr_dataset.SR_dataset.from_samplers]:
        it attaches a [SampleSource][SRToolkit.dataset.data_source.SampleSource] so the data
        is generated lazily from ``samplers`` (and, for RMSE, ``ground_truth``) the first
        time the dataset is materialised via
        [create_dataset][SRToolkit.dataset.sr_benchmark.SR_benchmark.create_dataset].

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> bm = SR_benchmark("BM")
            >>> bm.add_from_samplers(["X_0", "+", "X_1"],
            ...     [UniformSampling(0, 5), UniformSampling(0, 5)], dataset_name="add",
            ...     n_samples=100, seed=0)
            >>> ds = bm.create_dataset("add")
            >>> ds.X.shape
            (100, 2)

        Args:
            ground_truth: Ground-truth expression as a token list or
                [Node][SRToolkit.utils.expression_tree.Node].
            samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per input variable.
            symbol_library: Token vocabulary. Defaults to
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]
                with one variable per sampler.
            n_samples: Number of input rows to generate on materialisation. Defaults to ``10000``.
            seed: Random seed stored on the
                [SampleSource][SRToolkit.dataset.data_source.SampleSource].
            ranking_function: ``"rmse"`` or ``"bed"``.
            dataset_name: Name of the dataset. Auto-generated if ``None``.
            original_equation: Human-readable equation string. Auto-filled from a token-list
                ``ground_truth`` when ``None``.
            success_threshold: Error threshold for success. ``None`` means no threshold.
            max_evaluations: Maximum expressions to evaluate. ``-1`` means no limit.
            dataset_metadata: Optional dataset-level metadata dict.
            **kwargs: Estimation settings forwarded to
                [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].
        """
        if symbol_library is None:
            if not samplers:
                raise ValueError(
                    "[SR_benchmark.add_from_samplers] 'samplers' must be a non-empty list "
                    "(one sampler per input variable)."
                )
            symbol_library = SymbolLibrary.default_symbols(len(samplers))

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
            **kwargs,
        )

    def create_dataset(
        self,
        dataset_name: str,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> SR_dataset:
        """
        Creates an instance of a dataset from the given dataset name.

        When ``n_samples`` is provided the returned dataset contains freshly sampled data
        instead of the pre-generated data on disk. The dataset must have samplers defined
        (see ``samplers`` argument of
        [add_dataset][SRToolkit.dataset.sr_benchmark.SR_benchmark.add_dataset]).

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> dataset.X.shape
            (10000, 3)
            >>> dataset_small = benchmark.create_dataset('I.16.6', n_samples=500, seed=0)
            >>> dataset_small.X.shape
            (500, 3)

        Args:
            dataset_name: The name of the dataset to create.
            n_samples: If provided, generate a fresh dataset with this many samples using
                the stored samplers instead of loading pre-generated data from disk.
            seed: Random seed used when ``n_samples`` is provided. If ``None``, no seed is set.

        Returns:
            An [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance containing the
            data, ground truth expression, and metadata for the given dataset.

        Raises:
            ValueError: If the dataset name is not found, or if ``n_samples`` is provided but
                the dataset has no samplers defined.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")

        if n_samples is not None:
            config = self.datasets[dataset_name]
            if config.get("samplers") is None:
                raise ValueError(
                    f"[SR_benchmark.create_dataset] Cannot resample '{dataset_name}': no samplers defined."
                )
            # _create_from_entry may return the benchmark's stored SR_dataset instance
            # (for entries added via add_dataset_instance). Copy before mutating X/y so
            # resampling never corrupts the stored dataset or aliases across callers.
            # A fresh draw is requested, so the canonical cached data would only be loaded
            # and immediately overwritten — skip _ensure_data and let the data_source (a
            # SampleSource for the built-ins) materialise whatever is needed to construct.
            dataset = copy.deepcopy(self._create_from_entry(config))
            return dataset.resample_inplace(n_samples, seed=seed)

        # Loading canonical data: give subclasses a chance to prefetch it into the cache
        # (e.g. download a benchmark archive once) before materialisation runs.
        self._ensure_data(dataset_name)
        return self._create_from_entry(self.datasets[dataset_name])

    def _ensure_data(self, dataset_name: str) -> None:
        """
        Hook called before loading a dataset's canonical (non-resampled) data.

        The base implementation is a no-op. Subclasses (e.g. the built-in benchmarks) may
        override it to prefetch authoritative data into the cache — typically by downloading
        a benchmark archive once — so that the subsequent
        [data_cache.resolve][SRToolkit.dataset.data_cache.resolve] is a cache hit and each
        dataset's own ``data_source`` (a fallback) is not consulted. Implementations should
        be idempotent and degrade gracefully (warn, don't raise) when the prefetch fails, so
        the per-dataset ``data_source`` can still serve as a fallback.

        Args:
            dataset_name: The dataset about to be loaded.
        """
        pass

    def _create_from_entry(self, entry: dict) -> SR_dataset:
        """Internal helper: create an SR_dataset from an entry dict."""
        if "sr_dataset" in entry:
            return entry["sr_dataset"]
        try:
            return SR_dataset.from_dict(entry)
        except Exception as e:
            raise ValueError(
                f"[SR_benchmark.create_dataset] Could not create SR_dataset from the given "
                f"dictionary. Original error: {e}"
            )

    def list_datasets(self, verbose: bool = True, num_variables: int = -1) -> List[str]:
        """
        Lists the available datasets.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman() # Feynman is a specific instance of SR_benchmark with additional functionality
            >>> len(benchmark.list_datasets(num_variables=2, verbose=False))
            15
            >>> datasets_with_8_vars = benchmark.list_datasets(num_variables=8, verbose=False)
            >>> datasets_with_8_vars[0]
            'II.36.38'

        Args:
            verbose: If ``True``, also prints a description of each dataset.
            num_variables: If not ``-1``, only return datasets with this many input variables.

        Returns:
            A list of dataset names.
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
                part3.append(self.datasets[d].get("original_equation"))
                if len(d) + 1 > max_length_1:
                    max_length_1 = len(d) + 1
                if len(variable_str) > max_length_2:
                    max_length_2 = len(variable_str)

            for p1, p2, p3 in zip(part1, part2, part3):
                print(f"{p1:<{max_length_1}} {p2:<{max_length_2}}, Expression: {p3}")
        return datasets

    def to_dict(self) -> dict:
        """
        Serialise the benchmark to a pure JSON-safe dictionary.

        Dataset entries that have an ``sr_dataset`` key (added via
        [add_dataset_instance][SRToolkit.dataset.sr_benchmark.SR_benchmark.add_dataset_instance])
        are serialised via ``SR_dataset.to_dict()``.

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
            "type": "SR_benchmark",
            "benchmark_name": self.benchmark_name,
            "version": self.version,
            "metadata": self.metadata,
            "datasets": datasets_out,
        }

    @classmethod
    def from_dict(cls, d: Union[dict, str, Path]) -> "SR_benchmark":
        """
        Reconstruct an SR_benchmark from a config dict or a saved JSON file.

        To load a self-contained ``.zip`` archive (written by
        [to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive]) use
        [from_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.from_archive] instead.

        Args:
            d: A dict produced by [to_dict][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_dict],
                or a path to a JSON file.

        Returns:
            An [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] instance.
        """
        if isinstance(d, (str, Path)):
            if str(d).endswith(".zip"):
                raise ValueError(
                    "[SR_benchmark.from_dict] Received a '.zip' path. Load self-contained "
                    "archives with SR_benchmark.from_archive(path) instead."
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
            # Store the entry dict directly; materialisation is lazy via create_dataset
            b.datasets[name] = entry
            if "num_variables" not in entry:
                # Try to infer
                samplers = entry.get("samplers")
                if samplers is not None:
                    entry["num_variables"] = len(samplers)
                else:
                    entry["num_variables"] = -1

        return b

    def to_archive(self, path: Union[str, Path]) -> None:
        """
        Write the benchmark (config + data) to a ``.zip`` archive.

        The archive contains:

        - ``benchmark.json``: the benchmark configuration dict.
        - ``data/<dataset_name>.npz``: the cached data for each dataset.
        - ``data/<dataset_name>_gt.npy``: ground-truth behaviour array (if present).

        Args:
            path: Destination path for the archive.  Non-``.zip`` suffixes trigger a
                warning but are still accepted.
        """
        path = Path(path)
        if path.suffix.lower() != ".zip":
            warnings.warn(
                f"[SR_benchmark.to_archive] path '{path}' does not end in '.zip'. "
                "The file will be a ZIP archive regardless of the extension.",
                stacklevel=2,
            )

        from SRToolkit.dataset import data_cache

        benchmark_json = json.dumps(self.to_dict(), indent=2)

        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("benchmark.json", benchmark_json)

            for name, entry in self.datasets.items():
                if "sr_dataset" in entry:
                    # In-memory dataset — write its data
                    ds = entry["sr_dataset"]
                    cache_p = data_cache.dataset_path(self.benchmark_name, self.version, name)
                    if not cache_p.exists():
                        # Write directly from arrays
                        import io

                        buf = io.BytesIO()
                        if ds.y is not None:
                            np.savez(buf, X=ds.X, y=ds.y)
                        else:
                            np.savez(buf, X=ds.X)
                        zf.writestr(f"data/{name}.npz", buf.getvalue())
                        if isinstance(ds.ground_truth, np.ndarray):
                            gt_buf = io.BytesIO()
                            np.save(gt_buf, ds.ground_truth)
                            zf.writestr(f"data/{name}_gt.npy", gt_buf.getvalue())
                        continue
                else:
                    ds_name = entry.get("dataset_name", name)
                    try:
                        cache_p = data_cache.resolve(self.benchmark_name, self.version, ds_name, entry)
                    except Exception as e:
                        warnings.warn(f"[SR_benchmark.to_archive] Could not materialise '{name}': {e}. Skipping.")
                        continue

                zf.write(str(cache_p), f"data/{name}.npz")

                gt_path = cache_p.parent / f"{name}_gt.npy"
                if gt_path.exists():
                    zf.write(str(gt_path), f"data/{name}_gt.npy")

    @classmethod
    def from_archive(cls, path: Union[str, Path]) -> "SR_benchmark":
        """
        Load a benchmark from a self-contained ``.zip`` archive.

        This is the counterpart to
        [to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive]: it reads
        ``benchmark.json`` from the archive, extracts the bundled ``data/*.npz`` (and any
        ``_gt.npy``) into the data cache, and returns a benchmark whose datasets read
        from that populated cache.

        Args:
            path: Path to a ``.zip`` archive written by
                [to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive].

        Returns:
            An [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] instance.
        """
        from SRToolkit.dataset import data_cache

        with zipfile.ZipFile(str(path), "r") as zf:
            benchmark_dict = json.loads(zf.read("benchmark.json"))

        benchmark_name = benchmark_dict["benchmark_name"]
        version = benchmark_dict.get("version", "1.0.0")

        # Extract data into the cache
        data_cache.import_archive(Path(path), benchmark_name, version)

        b = cls(benchmark_name, version=version, metadata=benchmark_dict.get("metadata", {}))
        for name, entry in benchmark_dict.get("datasets", {}).items():
            b.datasets[name] = entry
            if "num_variables" not in entry:
                entry["num_variables"] = -1

        return b

    @classmethod
    def from_url(cls, url: str) -> "SR_benchmark":
        """
        Download a self-contained ``.zip`` archive from a URL and load it.

        This is the remote counterpart to
        [from_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.from_archive]: the
        archive is downloaded to a temporary file and then loaded exactly as
        ``from_archive`` would. The ``url`` must point at an archive written by
        [to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive] (a
        ``benchmark.json`` plus a ``data/`` directory) — not a bare ``.npz``/data zip
        (that is what [UrlSource][SRToolkit.dataset.data_source.UrlSource] is for).

        Args:
            url: URL of a ``.zip`` archive written by
                [to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive].

        Returns:
            An [SR_benchmark][SRToolkit.dataset.sr_benchmark.SR_benchmark] instance.
        """
        import tempfile
        from urllib.request import urlopen

        with urlopen(url) as response:
            data = response.read()

        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            return cls.from_archive(tmp.name)
        finally:
            os.unlink(tmp.name)
