"""
Dataset wrapper for a single symbolic regression problem.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from typing_extensions import Unpack

from SRToolkit.approaches.sr_approach import SR_approach
from SRToolkit.evaluation.callbacks import CallbackDispatcher, ExperimentEvent, SRCallbacks
from SRToolkit.evaluation.sr_evaluator import SR_evaluator, SR_results
from SRToolkit.utils import Node, SymbolLibrary
from SRToolkit.utils.types import EstimationSettings


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
            X: Input data of shape ``(n_samples, n_features)`` used to evaluate expressions.
            symbol_library: The symbol library defining the token vocabulary.
            ranking_function: Ranking function to use. ``"rmse"`` calculates the error between ground truth
                values and expression outputs with fitted free parameters. ``"bed"`` is a stochastic measure of
                behavioral distance between expressions; it is less sensitive to overfitting and focuses more on
                structure identification.
            y: Target values used for parameter estimation when ``ranking_function="rmse"``.
            max_evaluations: Maximum number of expressions to evaluate. Values less than 0 mean no limit.
            ground_truth: The ground truth expression, as a list of tokens in infix notation, a
                [Node][SRToolkit.utils.expression_tree.Node] tree, or a numpy array of behavior vectors
                (see [create_behavior_matrix][SRToolkit.utils.measures.create_behavior_matrix]).
            original_equation: Human-readable string of the original equation (e.g. ``"z = x + y"``).
            success_threshold: Error threshold below which an expression is considered successful. If ``None``,
                no threshold is applied.
            seed: Random seed for reproducibility. ``None`` means no seed is set.
            dataset_metadata: Optional dictionary of metadata about the dataset (e.g. citation, variable names).
            dataset_name: Name for this dataset. Defaults to ``"unnamed"``.
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
            evaluator.set_callbacks(callbacks)
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

        try:
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
        except Exception as e:
            raise e

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

    # Once SR_approach base class is implemented, we can add a function to run experiments
    # def run_experiments(self, approach: SR_approach, num_runs: int=10):

    def to_dict(self, base_path: str) -> dict:
        r"""
        Creates a dictionary representation of this dataset. This is mainly used for saving the dataset to disk.

        Examples:
            >>> import tempfile
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> dataset.to_dict("data/example_ds")  # doctest: +SKIP

        Args:
            base_path: The path to the directory where the data in the dataset should be saved.

        Returns:
            A dictionary representation of this dataset.
        """
        output = {
            "format_version": 1,
            "symbol_library": self.symbol_library.to_dict(),
            "ranking_function": self.ranking_function,
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "original_equation": self.original_equation,
            "seed": self.seed,
            "dataset_metadata": self.dataset_metadata,
            "dataset_name": self.dataset_name,
        }

        if self.kwargs is not None and "bed_X" in self.kwargs and isinstance(self.kwargs["bed_X"], np.ndarray):
            self.kwargs["bed_X"] = self.kwargs["bed_X"].tolist()

        output["kwargs"] = self.kwargs

        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        if self.ground_truth is None:
            output["ground_truth"] = None
        else:
            if isinstance(self.ground_truth, list):
                output["ground_truth"] = self.ground_truth
            elif isinstance(self.ground_truth, Node):
                output["ground_truth"] = self.ground_truth.to_list()
            elif isinstance(self.ground_truth, np.ndarray) and not os.path.exists(
                f"{base_path}/{self.dataset_name}_gt.npy"
            ):
                np.save(f"{base_path}/{self.dataset_name}_gt.npy", self.ground_truth)
                output["ground_truth"] = f"{base_path}/{self.dataset_name}_gt.npy"

        if not os.path.exists(f"{base_path}/{self.dataset_name}.npz"):
            if self.y is None:
                np.savez(f"{base_path}/{self.dataset_name}.npz", X=self.X)
            else:
                np.savez(f"{base_path}/{self.dataset_name}.npz", X=self.X, y=self.y)
        output["dataset_path"] = f"{base_path}/{self.dataset_name}.npz"

        return output

    @staticmethod
    def from_dict(d: dict) -> "SR_dataset":
        """
        Creates an instance of the SR_dataset class from its dictionary representation. This is mainly used for
        loading the dataset from disk.

        Examples:
            >>> import tempfile, os
            >>> tmpdir = tempfile.mkdtemp()
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> ds = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> dataset_dict = ds.to_dict(tmpdir)
            >>> dataset = SR_dataset.from_dict(dataset_dict)
            >>> dataset.X.shape
            (3, 2)

        Args:
            d: Dictionary representation of the dataset, as produced by
                [to_dict][SRToolkit.dataset.sr_dataset.SR_dataset.to_dict].

        Returns:
            A new [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset] instance.

        Raises:
            ValueError: If the ``format_version`` in ``d`` is not supported.
            Exception: If the dataset file or ground truth file cannot be loaded.
        """
        if d.get("format_version", 1) != 1:
            raise ValueError(
                f"[SR_dataset.from_dict] Unsupported format_version: {d.get('format_version')!r}. Expected 1."
            )
        try:
            data = np.load(d["dataset_path"])
            X = data["X"]
            if "y" in data:
                y = data["y"]
            else:
                y = None
        except Exception:
            raise Exception(f"[SR_dataset.from_dict] Could not load dataset from {d['dataset_path']}")

        if "ground_truth" in d and (isinstance(d["ground_truth"], list) or d["ground_truth"] is None):
            ground_truth = d["ground_truth"]
        else:
            try:
                ground_truth = np.load(d["ground_truth"])
            except Exception:
                raise Exception(f"[SR_dataset.from_dict] Could not load ground truth from {d['ground_truth']}")

        if "bed_X" in d["kwargs"] and d["kwargs"]["bed_X"] is not None:
            d["kwargs"]["bed_X"] = np.array(d["kwargs"]["bed_X"])

        try:
            return SR_dataset(
                X,
                SymbolLibrary.from_dict(d["symbol_library"]),
                ranking_function=d["ranking_function"],
                y=y,
                max_evaluations=d["max_evaluations"],
                ground_truth=ground_truth,
                original_equation=d["original_equation"],
                success_threshold=d["success_threshold"],
                seed=d["seed"],
                dataset_metadata=d["dataset_metadata"],
                dataset_name=d["dataset_name"],
                **d["kwargs"],
            )
        except Exception as e:
            raise Exception(f"[SR_dataset.from_dict] Error creating dataset: {e}")
