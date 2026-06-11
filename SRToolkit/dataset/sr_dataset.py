"""
Dataset wrapper for a single symbolic regression problem.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Unpack

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.evaluation.callbacks import CallbackDispatcher, ExperimentEvent, SRCallbacks
from SRToolkit.evaluation.sr_evaluator import SR_evaluator, SR_results
from SRToolkit.utils import Node, SymbolLibrary
from SRToolkit.utils.expression_compiler import compile_expr
from SRToolkit.utils.types import EstimationSettings

from .data_source import DataSource, SampleSource, source_from_dict
from .sampling import Sampler, sampler_from_dict


class SR_dataset:
    def __init__(
        self,
        X: np.ndarray,
        symbol_library: SymbolLibrary,
        ranking_function: str = "rmse",
        y: Optional[np.ndarray] = None,
        max_evaluations: int = -1,
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        dataset_name: str = "unnamed",
        samplers: Optional[List[Sampler]] = None,
        benchmark: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Unpack[EstimationSettings],
    ) -> None:
        """
        Wraps input data and evaluation settings for a single symbolic regression problem.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> evaluator = dataset.create_evaluator()
            >>> bool(evaluator.evaluate_expr(["sin", "(", "X_0", ")"]) < dataset.success_threshold)
            False
            >>> bool(evaluator.evaluate_expr(["u-", "C", "*", "X_1", "+", "X_0"]) < dataset.success_threshold)
            True

        Args:
            X: Input data of shape ``(num_samples, num_variables)`` used to evaluate expressions.
            symbol_library: The symbol library defining the tokens used for the discovery task.
            ranking_function: Ranking function to use. ``"rmse"`` calculates the RMSE between ground truth
                values and expression outputs with fitted free parameters. ``"bed"`` is a stochastic measure of
                behavioral distance between expressions; it is less sensitive to overfitting and focuses more on
                structure identification (see [bed][SRToolkit.utils.measures.bed] for more details).
            y: Target values used for parameter estimation when ``ranking_function="rmse"``.
            max_evaluations: Maximum number of expressions to evaluate. Values less than 0 mean no limit.
            ground_truth: The ground truth expression, as a list of tokens in infix notation, a
                [Node][SRToolkit.utils.expression_tree.Node] tree, or a numpy array of behavior matrix
                (see [create_behavior_matrix][SRToolkit.utils.measures.create_behavior_matrix]).
                Numpy array is only applicable when ``ranking_function="bed"``.
            original_equation: Human-readable string of the original equation (e.g. ``"z = x + y"``).
            success_threshold: Error threshold below which an expression is considered successful. If ``None``,
                no threshold is applied.
            seed: Random seed for reproducibility. ``None`` means no seed is set.
            dataset_metadata: Optional dictionary of metadata about the dataset (e.g. citation, variable names).
            dataset_name: Name for this dataset. Defaults to ``"unnamed"``.
            samplers: Optional list of [Sampler][SRToolkit.dataset.sampling.Sampler]
                instances (one per input variable). The built-in
                [LogUniformSampling][SRToolkit.dataset.sampling.LogUniformSampling],
                [UniformSampling][SRToolkit.dataset.sampling.UniformSampling], and
                [IntegerUniformSampling][SRToolkit.dataset.sampling.IntegerUniformSampling]
                implement this interface.
            benchmark: Optional benchmark name (e.g. ``"feynman"``). Required for serialisation
                via [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict].
            version: Optional version string (e.g. ``"1.0.0"``). Required for serialisation.
            **kwargs: Optional estimation settings passed to
                [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].
                Supported keys: ``method``, ``tol``, ``gtol``, ``max_iter``, ``constant_bounds``,
                ``initialization``, ``max_constants``, ``max_expr_length``, ``num_points_sampled``,
                ``bed_X``, ``num_consts_sampled``, ``domain_bounds``.
        """
        self.X = X
        self.symbol_library = symbol_library
        self.y = y
        self.max_evaluations = max_evaluations
        self.success_threshold = success_threshold
        self.ranking_function = ranking_function
        self.ground_truth = ground_truth
        self.original_equation = original_equation
        self.kwargs = kwargs
        self.dataset_name = dataset_name

        # See if symbols contain a symbol for constants
        symbols_metadata = self.symbol_library.symbols.values()
        self.contains_constants = any([symbol["type"] == "const" for symbol in symbols_metadata])

        self.seed = seed
        self.dataset_metadata = dataset_metadata
        self.samplers = samplers

        # Cache / serialisation metadata
        self.benchmark = benchmark
        self.version = version
        # Origin of the cached data (UrlSource / SampleSource / None). Not a constructor
        # argument: it is set by the factory methods (from_dict, from_samplers) and by
        # SR_benchmark.add_dataset. End users select an origin through those entry points.
        self.data_source: Optional[DataSource] = None

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
        """
        Evaluates an SR approach on this dataset.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y")
            >>> results = dataset.evaluate_approach(my_approach, num_experiments=5)  # doctest: +SKIP

        Args:
            sr_approach: The SR approach to evaluate.
            num_experiments: Number of independent experiments (runs) to perform.
            top_k: Number of top expressions to retain per experiment.
            initial_seed: Seed for random number generation. If ``None``, the dataset seed is used.
            results: Existing [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] object to append
                results to. If ``None``, a new one is created.
            callbacks: Optional list of [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks], [SRCallbacks][SRToolkit.evaluation.callbacks.SRCallbacks], or
                [CallbackDispatcher][SRToolkit.evaluation.callbacks.CallbackDispatcher] for monitoring
                and controlling the search.
            verbose: If ``True``, prints progress for each experiment.
            adaptation_path: Path to save/load the adapted state for approaches with
                ``adaptation_scope="once"``. If the file already exists it is loaded directly,
                skipping adaptation. If it does not exist, the approach is adapted and the state
                is saved to this path. If ``None``, adaptation runs without saving.

        Returns:
            An [SR_results][SRToolkit.evaluation.sr_evaluator.SR_results] object containing results from all experiments.
        """
        if initial_seed is None:
            seed = self.seed
        else:
            seed = initial_seed

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

        if sr_approach.adaptation_scope == "once":
            if adaptation_path is not None and os.path.exists(adaptation_path):
                sr_approach.load_adapted_state(adaptation_path)
            else:
                sr_approach.adapt(self.X, self.symbol_library)
                if adaptation_path is not None:
                    dir_name = os.path.dirname(adaptation_path)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    sr_approach.save_adapted_state(adaptation_path)

        for experiment in range(num_experiments):
            if verbose:
                print(f"Running experiment {experiment + 1}/{num_experiments}")
            if seed is not None:
                seed += 1

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

            if sr_approach.adaptation_scope == "experiment":
                sr_approach.adapt(self.X, self.symbol_library)

            evaluator = self.create_evaluator(seed=seed)
            evaluator._experiment_id = f"{dataset_name}_{sr_approach.name}_{seed}"
            evaluator.register_callbacks(callbacks)
            sr_approach.search(evaluator, seed)
            results += evaluator.get_results(sr_approach.name, top_k)

            if callbacks is not None:
                callbacks.on_experiment_end(event, results.results[-1])
        return results

    def create_evaluator(self, metadata: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> SR_evaluator:
        """
        Creates an instance of the SR_evaluator class from this dataset.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> evaluator = dataset.create_evaluator()
            >>> float(evaluator.evaluate_expr(["sin", "(", "X_0", ")"]))  # doctest: +ELLIPSIS
            8.05645397...
            >>> float(evaluator.evaluate_expr(["X_1", "+", "X_0"]))  # doctest: +ELLIPSIS
            0.0...

        Args:
            metadata: Optional dictionary of metadata to attach to the evaluator (e.g. model name, seed).
                Dataset metadata is merged in automatically.
            seed: Seed for the random number generator. If ``None``, the dataset seed is used.

        Returns:
            A configured [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] ready to evaluate expressions against this dataset.

        Raises:
            Exception: If [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] cannot be
                instantiated with the current dataset settings.
        """
        if metadata is None:
            metadata = dict()
        if self.dataset_metadata is not None:
            metadata["dataset_metadata"] = self.dataset_metadata
        metadata["dataset_name"] = self.dataset_name

        if seed is None:
            seed = self.seed

        return SR_evaluator(
            X=self.X,
            y=self.y,
            max_evaluations=self.max_evaluations,
            success_threshold=self.success_threshold,
            ranking_function=self.ranking_function,
            ground_truth=self.ground_truth,
            symbol_library=self.symbol_library,
            seed=seed,
            metadata=metadata,
            **self.kwargs,
        )

    def __str__(self) -> str:
        r"""
        Returns a string describing this dataset.

        The string describes the target expression, symbols that should be used,
        and the success threshold. It also includes any constraints that should
        be followed when evaluating a model on this dataset. These constraints include the maximum
        number of expressions to evaluate, the maximum length of the expression,
        and the maximum number of constants allowed in the expression. If the
        symbol library contains a symbol for constants, the string also includes
        the range of constants.

        For other metadata, please refer to the attribute self.dataset_metadata.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> str(dataset)
            'Dataset for target expression z = x + y. When evaluating your model on this dataset, you should limit your generative model to only produce expressions using the following symbols: +, -, *, /, ^, u-, sqrt, sin, cos, exp, tan, arcsin, arccos, arctan, sinh, cosh, tanh, floor, ceil, ln, log, ^-1, ^2, ^3, ^4, ^5, pi, e, C, X_0, X_1.\nExpressions will be ranked based on the RMSE ranking function.\nExpressions are deemed successful if the root mean squared error is less than 1e-06. However, we advise that you check the best performing expressions manually to ensure they are correct.\nDataset uses the default limitations (extra arguments) from the SR_evaluator.The expressions in the dataset can contain constants/free parameters.\nFor other metadata, please refer to the attribute self.dataset_metadata.'

        Returns:
            A string describing this dataset.
        """
        description = f"Dataset for target expression {self.original_equation}."
        description += (
            f" When evaluating your model on this dataset, you should limit your generative model to only "
            f"produce expressions using the following symbols: {str(self.symbol_library)}.\nExpressions will be "
            f"ranked based on the {self.ranking_function.upper()} ranking function.\n"
        )

        if self.success_threshold is not None:
            description += (
                "Expressions are deemed successful if the root mean squared error is less than "
                f"{self.success_threshold}. However, we advise that you check the best performing "
                f"expressions manually to ensure they are correct.\n"
            )

        if len(self.kwargs) == 0:
            description += "Dataset uses the default limitations (extra arguments) from the SR_evaluator."
        else:
            limitations = "Non default limitations (extra arguments) from the SR_evaluators are:"
            for key, value in self.kwargs.items():
                limitations += f" {key}={value}, "
            limitations = limitations[:-2] + ".\n"
            description += limitations

        if self.contains_constants:
            description += "The expressions in the dataset can contain constants/free parameters.\n"

        description += "For other metadata, please refer to the attribute self.dataset_metadata."

        return description

    def resample(self, n: int, seed: Optional[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate fresh data by applying the stored samplers to produce new inputs.

        For ``ranking_function="rmse"``, the ground truth expression is also evaluated and
        ``(X, y)`` is returned. For ``ranking_function="bed"``, only ``X`` is returned.

        Args:
            n: Number of samples to generate.
            seed: Random seed for reproducibility. If ``None``, no seed is set.

        Returns:
            For RMSE: a tuple ``(X, y)`` with shapes ``(n, n_features)`` and ``(n,)``.
            For BED: a single array ``X`` with shape ``(n, n_features)``.

        Raises:
            ValueError: If ``samplers`` is ``None``, or if ``ranking_function="rmse"`` and
                ``ground_truth`` is ``None`` or a behaviour array.
        """
        if self.samplers is None:
            raise ValueError(
                f"[SR_dataset.resample] Dataset '{self.dataset_name}' has no samplers defined. "
                "Provide samplers when constructing the dataset."
            )
        if seed is not None:
            np.random.seed(seed)
        X = np.column_stack([s(n) for s in self.samplers])
        if self.ranking_function == "bed":
            return X
        if self.ground_truth is None or isinstance(self.ground_truth, np.ndarray):
            raise ValueError(
                f"[SR_dataset.resample] Dataset '{self.dataset_name}' has no token-list ground truth — "
                "cannot evaluate y. ground_truth must be a list of tokens or a Node."
            )
        f = compile_expr(self.ground_truth, self.symbol_library)
        y = f(X, np.array([]))
        return X, y

    def resample_inplace(self, n: int, seed: Optional[int] = None) -> "SR_dataset":
        """
        Resample via the stored samplers and write the result back into ``self.X``/``self.y``.

        A convenience wrapper around [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample]
        that handles the RMSE ``(X, y)`` vs BED ``X``-only return shapes. For BED datasets,
        ``self.y`` is set to ``None``.

        Args:
            n: Number of samples to generate.
            seed: Random seed for reproducibility. If ``None``, no seed is set.

        Returns:
            ``self``, with ``X`` (and ``y``) replaced by the freshly sampled data.

        Raises:
            ValueError: Propagated from [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample]
                if ``samplers`` is ``None`` or the ground truth cannot produce ``y``.
        """
        result = self.resample(n, seed=seed)
        if isinstance(result, tuple):
            self.X, self.y = result
        else:
            self.X, self.y = result, None
        return self

    def _persist_to_cache(self) -> None:
        """
        Write this dataset's in-memory arrays into the data cache version directory.

        Used by [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict] when
        ``data_source`` is ``None`` — in that case the in-memory arrays are the only
        copy of the data, so they must be written to
        ``<benchmark>/<version>/<dataset_name>.npz`` for
        [from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict] to be able to
        reload them. An array-shaped ground truth (a ``bed`` behaviour matrix) is
        written to the sibling ``<dataset_name>_gt.npy``.

        Requires ``benchmark`` and ``version`` to be set (the caller validates this).
        """
        from SRToolkit.dataset import data_cache

        assert self.benchmark is not None and self.version is not None, (
            "_persist_to_cache requires benchmark/version to be set; the caller must validate this."
        )

        cache_path = data_cache.dataset_path(self.benchmark, self.version, self.dataset_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.y is not None:
            np.savez(str(cache_path), X=self.X, y=self.y)
        else:
            np.savez(str(cache_path), X=self.X)
        if isinstance(self.ground_truth, np.ndarray):
            np.save(str(cache_path.parent / f"{self.dataset_name}_gt.npy"), self.ground_truth)

    def to_dict(self) -> dict:
        r"""
        Creates a JSON-safe dictionary representation of this dataset.

        The data arrays are not embedded in the dictionary — they live in the
        data cache (see [data_cache][SRToolkit.dataset.data_cache]). Use
        [from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict] to
        reconstruct the full dataset including data.

        When ``data_source`` is ``None`` the in-memory arrays are the only copy of the
        data, so calling this method writes them into the cache version directory (via
        the private ``_persist_to_cache`` helper) so the returned config stays
        reloadable. For
        [SampleSource][SRToolkit.dataset.data_source.SampleSource] /
        [UrlSource][SRToolkit.dataset.data_source.UrlSource] datasets there are no arrays
        to write and the call has no filesystem side effects.

        Examples:
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y",
            ...     success_threshold=1e-6, benchmark="test", version="1.0.0")
            >>> d = dataset.to_dict()
            >>> d["format_version"]
            2
            >>> d["benchmark"]
            'test'

        Returns:
            A JSON-safe dictionary representing this dataset's configuration.

        Raises:
            ValueError: If ``benchmark`` or ``version`` is ``None`` (both are required for
                serialisation so the cache layer can locate the data).
            ValueError: If ``ground_truth`` is not the correct type.
        """
        if self.benchmark is None:
            raise ValueError("[SR_dataset.to_dict] 'benchmark' is None. Set self.benchmark before serialising.")
        if self.version is None:
            raise ValueError("[SR_dataset.to_dict] 'version' is None. Set self.version before serialising.")

        # When there is no DataSource, the in-memory arrays are the only copy of the
        # data. Persist them into the cache so the returned config is actually
        # reloadable by from_dict (which materialises from the cache). For sample/url
        # sources there are no arrays to write, so to_dict stays side-effect-free.
        if self.data_source is None and self.X is not None:
            self._persist_to_cache()

        # Serialise kwargs, converting ndarray values
        kwargs_out = {}
        for k, v in self.kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs_out[k] = v.tolist()
            else:
                kwargs_out[k] = v

        # Serialise ground truth
        if self.ground_truth is None:
            ground_truth_out = None
        elif isinstance(self.ground_truth, list):
            ground_truth_out = self.ground_truth
        elif isinstance(self.ground_truth, Node):
            ground_truth_out = self.ground_truth.to_list()
        elif isinstance(self.ground_truth, np.ndarray):
            # ndarray ground truth lives in the cache as a separate file
            ground_truth_out = None
        else:
            raise ValueError("[SR_dataset.to_dict] Ground truth must be either a list, a Node, or a numpy array")

        return {
            "format_version": 2,
            "dataset_name": self.dataset_name,
            "benchmark": self.benchmark,
            "version": self.version,
            "symbol_library": self.symbol_library.to_dict(),
            "ranking_function": self.ranking_function,
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "original_equation": self.original_equation,
            "seed": self.seed,
            "dataset_metadata": self.dataset_metadata,
            "kwargs": kwargs_out,
            "samplers": [s.to_dict() for s in self.samplers] if self.samplers is not None else None,
            "ground_truth": ground_truth_out,
            "data_source": self.data_source.to_dict() if self.data_source is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Union[dict, str, Path]) -> "SR_dataset":
        """
        Creates an instance of the SR_dataset class from its dictionary representation.

        If ``d`` is a string or Path, it is treated as a JSON file path and read
        from disk. To load a self-contained ``.zip`` archive (written by
        [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive]) use
        [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive] instead.
        The data arrays are loaded from the data cache (or materialised on demand).

        Examples:
            >>> import tempfile, os, json
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> from SRToolkit.dataset.data_source import SampleSource
            >>> X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
            >>> ds = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11], dtype=float), max_evaluations=10000, original_equation="z = x + y",
            ...     success_threshold=1e-6, benchmark="test_bench", version="1.0.0",
            ...     samplers=[UniformSampling(0, 5), UniformSampling(0, 5)])
            >>> ds.data_source = SampleSource(n_samples=3, seed=0)
            >>> d = ds.to_dict()
            >>> ds2 = SR_dataset.from_dict(d)
            >>> ds2.benchmark
            'test_bench'

        Args:
            d: Dictionary produced by
                [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict], or a path to a
                JSON file containing such a dictionary.

        Returns:
            A new [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance.

        Raises:
            ValueError: If the ``format_version`` is not 2.
            FileNotFoundError: If the cached data file does not exist and ``data_source`` is
                ``None``.
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
        from SRToolkit.bundle._relocate import _auto_bind

        dd = _auto_bind(dd)

        fmt = dd.get("format_version", 1)
        if fmt == 1:
            # Legacy format — delegate to old-style loading
            return cls._from_dict_v1(dd)
        if fmt != 2:
            raise ValueError(f"[SR_dataset.from_dict] Unsupported format_version: {fmt!r}. Expected 2.")

        from SRToolkit.dataset import data_cache

        benchmark = dd["benchmark"]
        version = dd["version"]
        dataset_name = dd["dataset_name"]

        cache_path = data_cache.resolve(benchmark, version, dataset_name, dd)

        data = np.load(str(cache_path))
        X = data["X"]
        y = data["y"] if "y" in data else None

        # Check for separate ground-truth array file
        gt_path = cache_path.parent / f"{dataset_name}_gt.npy"
        if gt_path.exists():
            ground_truth = np.load(str(gt_path))
        else:
            ground_truth = dd.get("ground_truth")

        kwargs = dict(dd.get("kwargs") or {})
        if "bed_X" in kwargs and kwargs["bed_X"] is not None:
            kwargs["bed_X"] = np.array(kwargs["bed_X"])

        samplers = None
        if dd.get("samplers") is not None:
            samplers = [sampler_from_dict(s) for s in dd["samplers"]]

        dataset = cls(
            X,
            SymbolLibrary.from_dict(dd["symbol_library"]),
            ranking_function=dd["ranking_function"],
            y=y,
            max_evaluations=dd["max_evaluations"],
            ground_truth=ground_truth,
            original_equation=dd["original_equation"],
            success_threshold=dd["success_threshold"],
            seed=dd["seed"],
            dataset_metadata=dd.get("dataset_metadata"),
            dataset_name=dataset_name,
            samplers=samplers,
            benchmark=benchmark,
            version=version,
            **kwargs,
        )
        dataset.data_source = source_from_dict(dd.get("data_source"))
        return dataset

    def to_archive(self, path: "Union[str, Path]") -> None:
        """
        Write this dataset (config + data) to a self-contained ``.zip`` archive.

        The archive mirrors the per-dataset layout of
        [SR_benchmark.to_archive][SRToolkit.dataset.sr_benchmark.SR_benchmark.to_archive]
        and contains:

        - ``dataset.json``: this dataset's configuration dict (see
          [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict]).
        - ``data/<dataset_name>.npz``: the ``X`` (and ``y`` for RMSE) arrays.
        - ``data/<dataset_name>_gt.npy``: ground-truth behaviour array, only when
          ``ground_truth`` is a numpy array (a ``bed`` behaviour matrix).

        Load it back with
        [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive], or
        from a URL with
        [from_url][SRToolkit.dataset.sr_dataset.SR_dataset.from_url].

        Args:
            path: Destination path for the archive. A non-``.zip`` suffix triggers a
                warning but is still written as a ZIP archive.

        Raises:
            ValueError: If ``benchmark`` or ``version`` is ``None`` (both are required
                so the cache layer can locate the data on load — raised by
                [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict]).
        """
        import io
        import warnings
        import zipfile

        path = Path(path)
        if path.suffix.lower() != ".zip":
            warnings.warn(
                f"[SR_dataset.to_archive] path '{path}' does not end in '.zip'. "
                "The file will be a ZIP archive regardless of the extension.",
                stacklevel=2,
            )

        # to_dict validates that benchmark/version are set.
        dataset_json = json.dumps(self.to_dict(), indent=2)
        name = self.dataset_name

        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dataset.json", dataset_json)

            buf = io.BytesIO()
            if self.y is not None:
                np.savez(buf, X=self.X, y=self.y)
            else:
                np.savez(buf, X=self.X)
            zf.writestr(f"data/{name}.npz", buf.getvalue())

            if isinstance(self.ground_truth, np.ndarray):
                gt_buf = io.BytesIO()
                np.save(gt_buf, self.ground_truth)
                zf.writestr(f"data/{name}_gt.npy", gt_buf.getvalue())

    @classmethod
    def from_archive(cls, path: "Union[str, Path]") -> "SR_dataset":
        """
        Load a dataset from a self-contained ``.zip`` archive.

        This is the counterpart to
        [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive]: it reads
        ``dataset.json`` from the archive, extracts the bundled ``data/*.npz`` (and any
        ``_gt.npy``) into the data cache, and reconstructs the dataset from them. Unlike
        [from_dict][SRToolkit.dataset.sr_dataset.SR_dataset.from_dict], no
        ``data_source`` materialisation is needed — the data travels inside the archive.

        Args:
            path: Path to a ``.zip`` archive written by
                [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive].

        Returns:
            A new [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance.
        """
        import zipfile

        from SRToolkit.dataset import data_cache

        with zipfile.ZipFile(str(path), "r") as zf:
            d = json.loads(zf.read("dataset.json"))

        benchmark = d["benchmark"]
        version = d["version"]

        # Extract data/*.npz (and any _gt.npy) into the cache version directory.
        data_cache.import_archive(Path(path), benchmark, version)

        # The cache is now populated, so no source needs to materialise it.
        d["data_source"] = None
        return cls.from_dict(d)

    @classmethod
    def from_url(cls, url: str) -> "SR_dataset":
        """
        Download a self-contained ``.zip`` archive from a URL and load it.

        This is the remote counterpart to
        [from_archive][SRToolkit.dataset.sr_dataset.SR_dataset.from_archive]: the archive
        is downloaded to a temporary file and then loaded exactly as ``from_archive``
        would. The ``url`` must point at an archive written by
        [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive] (a
        ``dataset.json`` plus a ``data/`` directory) — not a bare ``.npz``/data zip
        (that is what [UrlSource][SRToolkit.dataset.data_source.UrlSource] is for).

        Args:
            url: URL of a ``.zip`` archive written by
                [to_archive][SRToolkit.dataset.sr_dataset.SR_dataset.to_archive].

        Returns:
            A new [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance.
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

    @classmethod
    def from_samplers(
        cls,
        ground_truth: Union[List[str], Node],
        samplers: List[Sampler],
        symbol_library: Optional[SymbolLibrary] = None,
        n_samples: int = 10000,
        seed: Optional[int] = None,
        ranking_function: str = "rmse",
        original_equation: Optional[str] = None,
        success_threshold: Optional[float] = None,
        max_evaluations: int = -1,
        dataset_name: str = "unnamed",
        dataset_metadata: Optional[dict] = None,
        benchmark: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Unpack[EstimationSettings],
    ) -> "SR_dataset":
        r"""
        Build a dataset from just a ground-truth expression and per-variable samplers.

        This is the convenience constructor for the common case where you have the
        expression you want to recover and a sampling spec for its inputs, but no data
        arrays yet. The inputs ``X`` are drawn from ``samplers`` (one per variable) and,
        for ``ranking_function="rmse"``, the targets ``y`` are produced by evaluating
        ``ground_truth`` on them. The result carries a
        [SampleSource][SRToolkit.dataset.data_source.SampleSource], so it round-trips via
        [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict], regenerates with
        [refresh][SRToolkit.dataset.sr_dataset.SR_dataset.refresh], and resamples with
        [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample].

        Examples:
            >>> from SRToolkit.dataset.sampling import UniformSampling
            >>> ds = SR_dataset.from_samplers(["X_0", "+", "X_1"],
            ...     [UniformSampling(0, 5), UniformSampling(0, 5)], n_samples=100, seed=0)
            >>> ds.X.shape
            (100, 2)
            >>> ds.y.shape
            (100,)
            >>> ds.original_equation
            'y = X_0+X_1'

        Args:
            ground_truth: The ground-truth expression as a list of infix tokens or a
                [Node][SRToolkit.utils.expression_tree.Node]. For ``"rmse"`` it is
                evaluated to produce ``y``; for ``"bed"`` it is stored as the target.
            samplers: One [Sampler][SRToolkit.dataset.sampling.Sampler] per input variable.
            symbol_library: Token vocabulary. Defaults to
                [default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols]
                with one variable per sampler.
            n_samples: Number of input rows to generate. Defaults to ``10000``.
            seed: Random seed for the generation (stored on the
                [SampleSource][SRToolkit.dataset.data_source.SampleSource]). ``None`` means
                no seed is set.
            ranking_function: ``"rmse"`` or ``"bed"``.
            original_equation: Human-readable equation string. If ``None`` and
                ``ground_truth`` is a token list, it is auto-filled as ``"y = <tokens>"``.
            success_threshold: Error threshold for success. ``None`` means no threshold.
            max_evaluations: Maximum expressions to evaluate. ``-1`` means no limit.
            dataset_name: Name for this dataset.
            dataset_metadata: Optional dataset-level metadata dict.
            benchmark: Optional benchmark name (needed only for serialisation).
            version: Optional version string (needed only for serialisation).
            **kwargs: Estimation settings forwarded to
                [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator].

        Returns:
            A new [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] with freshly
            generated data.

        Raises:
            ValueError: If ``samplers`` is empty, or (via
                [resample][SRToolkit.dataset.sr_dataset.SR_dataset.resample]) if
                ``ranking_function="rmse"`` and ``ground_truth`` cannot be evaluated.
        """
        if not samplers:
            raise ValueError(
                "[SR_dataset.from_samplers] 'samplers' must be a non-empty list (one sampler per input variable)."
            )
        if symbol_library is None:
            symbol_library = SymbolLibrary.default_symbols(len(samplers))
        if original_equation is None and isinstance(ground_truth, list):
            original_equation = "y = " + "".join(ground_truth)

        dataset = cls(
            np.empty((0, len(samplers))),
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
            **kwargs,
        )
        dataset.data_source = SampleSource(n_samples=n_samples, seed=seed)

        # Reuse the canonical generation path (handles rmse/bed and its validation).
        return dataset.resample_inplace(n_samples, seed=seed)

    @classmethod
    def _from_dict_v1(cls, d: dict) -> "SR_dataset":
        """Load a format_version=1 dict (legacy path)."""
        try:
            data = np.load(d["dataset_path"])
            X = data["X"]
            y = data["y"] if "y" in data else None
        except Exception:
            raise Exception(f"[SR_dataset.from_dict] Could not load dataset from {d['dataset_path']}")

        if "ground_truth" in d and (isinstance(d["ground_truth"], list) or d["ground_truth"] is None):
            ground_truth = d["ground_truth"]
        else:
            try:
                ground_truth = np.load(d["ground_truth"])
            except Exception:
                raise Exception(f"[SR_dataset.from_dict] Could not load ground truth from {d['ground_truth']}")

        kwargs = dict(d.get("kwargs") or {})
        if "bed_X" in kwargs and kwargs["bed_X"] is not None:
            kwargs["bed_X"] = np.array(kwargs["bed_X"])

        samplers = None
        if d.get("samplers") is not None:
            samplers = [sampler_from_dict(s) for s in d["samplers"]]

        return cls(
            X,
            SymbolLibrary.from_dict(d["symbol_library"]),
            ranking_function=d["ranking_function"],
            y=y,
            max_evaluations=d["max_evaluations"],
            ground_truth=ground_truth,
            original_equation=d["original_equation"],
            success_threshold=d["success_threshold"],
            seed=d["seed"],
            dataset_metadata=d.get("dataset_metadata"),
            dataset_name=d["dataset_name"],
            samplers=samplers,
            **kwargs,
        )

    def refresh(self) -> None:
        """
        Force-refresh the cached data for this dataset by re-materialising it
        from ``data_source``.

        After refreshing, ``self.X`` and ``self.y`` are reloaded from the new cache.

        Raises:
            ValueError: If ``data_source``, ``benchmark``, or ``version`` is ``None``.
        """
        if self.data_source is None:
            raise ValueError(
                "[SR_dataset.refresh] Cannot refresh: data_source is null. "
                "Set self.data_source to a valid source config before calling refresh()."
            )
        if self.benchmark is None or self.version is None:
            raise ValueError("[SR_dataset.refresh] Cannot refresh: 'benchmark' or 'version' is None.")

        from SRToolkit.dataset import data_cache

        data_cache.resolve(
            self.benchmark,
            self.version,
            self.dataset_name,
            self.to_dict(),
            force=True,
        )

        cache_path = data_cache.dataset_path(self.benchmark, self.version, self.dataset_name)
        data = np.load(str(cache_path))
        self.X = data["X"]
        if "y" in data:
            self.y = data["y"]
