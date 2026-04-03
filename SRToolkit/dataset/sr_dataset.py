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
        dataset_name: Optional[str] = None,
        **kwargs: Unpack[EstimationSettings],
    ) -> None:
        """
        Initializes an instance of the SR_dataset class.

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
            X: The input data to be used in calculation of the error/ranking function. We assume that X is a 2D array
                with the shape (n_samples, n_features).
            symbol_library: The symbol library to use.
            ranking_function: The ranking function to use. Currently, "rmse" and "bed" are supported. RMSE is the
                standard ranking function in symbolic regression, calculating the error between the ground truth values
                and outputs of expressions with fitted free parameters. BED is a stochastic measure that calculates
                the behavioral distance between two expressions that can contain free parameters. Its advantage is that
                expressions with lots of parameters are less likely to overfit, and thus the measure focuses more on
                structure identification.
            y: The target values to be used in parameter estimation if the ranking function is "rmse".
            max_evaluations: The maximum number of expressions to evaluate. Less than 0 means no limit.
            ground_truth: The ground truth expression, represented as a list of tokens (strings) in the infix notation,
                a SRToolkit.utils.Node object, or a numpy array representing behavior
                (see SRToolkit.utils.create_behavior_matrix for more details).
            original_equation: The original equation from which the ground truth expression was generated).
            success_threshold: The threshold for determining whether an expression is successful or not. If None,
            seed: The seed to use for random number generation/reproducibility. Default is None, which means no seed is used.
            dataset_metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the name of the dataset, a citation for the dataset, number of variables, etc.
            dataset_name: An optional string containing the name of the dataset.

        Keyword Arguments:
            method (str): The method to be used for minimization. Currently, only "L-BFGS-B" is supported/tested.
                Default is "L-BFGS-B".
            tol (float): The tolerance for termination. Default is 1e-6.
            gtol (float): The tolerance for the gradient norm. Default is 1e-3.
            max_iter (int): The maximum number of iterations. Default is 100.
            constant_bounds (Tuple[float, float]): A tuple of two elements, specifying the lower and upper bounds for
                the constant values. Default is (-5, 5).
            initialization (str): The method to use for initializing the constant values. Currently, only "random" and
                "mean" are supported. "random" creates a vector with random values sampled within the bounds. "mean"
                creates a vector where all values are calculated as (lower_bound + upper_bound)/2. Default is "random".
            max_constants (int): The maximum number of constants allowed in the expression. Default is 8.
            max_expr_length (int): The maximum length of the expression. Default is -1 (no limit).
            num_points_sampled (int): The number of points to sample when estimating the behavior of an expression.
                Default is 64. If num_points_sampled==-1, then the number of points sampled is equal to the number of
                points in the dataset.
            bed_X (Optional[np.ndarray]): Points used for BED evaluation. If None and domain_bounds are given, points
                are sampled from the domain. If None and domain_bounds are not givem, points are randomly selected
                from X. Default is None.
            num_consts_sampled (int): Number of constants sampled for BED evaluation. Default is 32.
            domain_bounds (Optional[List[Tuple[float, float]]]): Bounds for the domain to be used if bed_X is None to
                sample random points. Default is None.
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
        callbacks: Optional[Union[SRCallbacks, CallbackDispatcher]] = None,
        verbose=True,
    ) -> SR_results:
        """
        Evaluates an SR_approach on this dataset.

        Args:
            sr_approach: An instance of SR_approach that will be evaluated on this dataset.
            num_experiments: The number of times the approach should be evaluated on this dataset.
            top_k: Number of the best expressions presented in the results
            initial_seed: The seed used for random number generation. If None, the seed from the dataset is used.
            results: An optional SR_results object to which the results of the evaluation will be added. If None,
                a new SR_results object will be created.
            callbacks: Optional callbacks for monitoring and controlling the search. Can be a single
                SRCallbacks instance, a CallbackDispatcher, or None.
            verbose: If true, prints the progress of the evaluation.

        Returns:
            The results of the evaluation.
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

        dataset_name = self.dataset_name or "unknown"

        adaptation_cache = None

        for experiment in range(num_experiments):
            if verbose:
                print(f"Running experiment {experiment + 1}/{num_experiments}")
            if seed is not None:
                seed += 1

            event = ExperimentEvent(
                experiment_id=experiment,
                dataset_name=dataset_name,
                approach_name=sr_approach.name,
                seed=seed,
            )
            if callbacks is not None:
                callbacks.on_experiment_start(event)

            sr_approach.prepare()

            if sr_approach.adaptation_scope == "never":
                pass
            elif sr_approach.adaptation_scope == "once":
                key = sr_approach.name
                if adaptation_cache is not None:
                    sr_approach.load_adapted_state(adaptation_cache[key])
                else:
                    sr_approach.adapt(self.X, self.symbol_library)
                    adaptation_cache = sr_approach.save_adapted_state()
            else:  # "experiment"
                sr_approach.adapt(self.X, self.symbol_library)
                if sr_approach.save_adapted_model:
                    sr_approach.save_adapted_state()

            evaluator = self.create_evaluator(seed=seed)
            if callbacks is not None:
                evaluator.set_callbacks(callbacks)
            sr_approach.search(evaluator, seed)
            results += evaluator.get_results(sr_approach.name, top_k)

            if callbacks:
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
            metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the dataset used, the model used, seed, etc.
            seed: An optional seed to be used for the random number generator. If None, the seed from the dataset is used.

        Returns:
            An instance of the SR_evaluator class.

        Raises:
            Exception: if an error occurs when creating the evaluator.
        """
        if metadata is None:
            metadata = dict()
        if self.dataset_metadata is not None:
            metadata["dataset_metadata"] = self.dataset_metadata
        if self.dataset_name is not None:
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

    def to_dict(self, base_path: str, name: str) -> dict:
        r"""
        Creates a dictionary representation of this dataset. This is mainly used for saving the dataset to disk.

        Examples:
            >>> import tempfile
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> dataset = SR_dataset(X, SymbolLibrary.default_symbols(2), ground_truth=["X_0", "+", "X_1"],
            ...     y=np.array([3, 7, 11]), max_evaluations=10000, original_equation="z = x + y", success_threshold=1e-6)
            >>> dataset.to_dict("data/example_ds", "test_dataset")  # doctest: +SKIP

        Args:
            base_path: The path to the directory where the data in the dataset should be saved.
            name: The name of the dataset. This will be used to name the files containing the dataset data.

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
            "dataset_name": self.dataset_name if self.dataset_name is not None else name,
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
            elif isinstance(self.ground_truth, np.ndarray) and not os.path.exists(f"{base_path}/{name}_gt.npy"):
                np.save(f"{base_path}/{name}_gt.npy", self.ground_truth)
                output["ground_truth"] = f"{base_path}/{name}_gt.npy"

        if not os.path.exists(f"{base_path}/{name}.npz"):
            if self.y is None:
                np.savez(f"{base_path}/{name}.npz", X=self.X)
            else:
                np.savez(f"{base_path}/{name}.npz", X=self.X, y=self.y)
        output["dataset_path"] = f"{base_path}/{name}.npz"

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
            >>> dataset_dict = ds.to_dict(tmpdir, "test_dataset")
            >>> dataset = SR_dataset.from_dict(dataset_dict)
            >>> dataset.X.shape
            (3, 2)

        Args:
            d: The dictionary representation of the dataset.
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
