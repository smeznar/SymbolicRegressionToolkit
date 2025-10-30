import os
from typing import List, Optional, Union

import numpy as np

from SRToolkit.evaluation import SR_evaluator, ResultAugmenter
from SRToolkit.utils import SymbolLibrary, Node, create_behavior_matrix


# TODO: Rewrite this class a bit adapt to changes from SR_evaluator


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
        result_augmenters: Optional[List[ResultAugmenter]] = None,
        seed: Optional[int] = None,
        dataset_metadata: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the SR_dataset class.

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
            result_augmenters: Optional list of objects that augment the results returned by the "get_results" function.
            seed: The seed to use for random number generation/reproducibility. Default is None, which means no seed is used.
            dataset_metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the name of the dataset, a citation for the dataset, number of variables, etc.

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
        self.result_augmenters = result_augmenters
        self.kwargs = kwargs

        # See if symbols contain a symbol for constants
        symbols_metadata = self.symbol_library.symbols.values()
        self.contains_constants = any(
            [symbol["type"] == "const" for symbol in symbols_metadata]
        )

        self.seed = seed
        self.dataset_metadata = dataset_metadata

    def create_evaluator(self, metadata: dict = None) -> SR_evaluator:
        """
        Creates an instance of the SR_evaluator class from this dataset.

        Args:
            metadata: An optional dictionary containing metadata about this evaluation. This could include
                information such as the dataset used, the model used, seed, etc.

        Returns:
            An instance of the SR_evaluator class.

        Raises:
            Exception: if an error occurs when creating the evaluator.
        """
        if metadata is None:
            metadata = dict()
        metadata["dataset_metadata"] = self.dataset_metadata

        try:
            return SR_evaluator(
                X=self.X,
                y=self.y,
                max_evaluations=self.max_evaluations,
                success_threshold=self.success_threshold,
                ranking_function=self.ranking_function,
                ground_truth=self.ground_truth,
                result_augmenters=self.result_augmenters,
                symbol_library=self.symbol_library,
                seed=self.seed,
                metadata=metadata,
                **self.kwargs,
            )
        except Exception as e:
            print(f"Error creating evaluator: {e}")
            raise e

    def __str__(self) -> str:
        """
        Returns a string describing this dataset.

        The string describes the target expression, symbols that should be used,
        and the success threshold. It also includes any constraints that should
        be followed when evaluating a model on this dataset. These constraints include the maximum
        number of expressions to evaluate, the maximum length of the expression,
        and the maximum number of constants allowed in the expression. If the
        symbol library contains a symbol for constants, the string also includes
        the range of constants.

        For other metadata, please refer to the attribute self.dataset_metadata.

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
            description += ("Expressions are deemed successful if the root mean squared error is less than "
                            f"{self.success_threshold}. However, we advise that you check the best performing "
                            f"expressions manually to ensure they are correct.\n")

        has_limitations = False
        if len(self.kwargs) == 0:
            description += "Dataset uses the default limitations (extra arguments) from the SR_evaluator."
        else:
            limitations = "Non default limitations (extra arguments) from the SR_evaluators are:"
            for key, value in self.kwargs.items():
                limitations += f" {key}={value}, "
            limitations = limitations[:-2] + ".\n"
            description += limitations

        if self.contains_constants:
            description += f"The expressions in the dataset can contain constants/free parameters.\n"

        description += "For other metadata, please refer to the attribute self.dataset_metadata."

        return description

    # Once SR_approach base class is implemented, we can add a function to run experiments
    # def run_experiments(self, approach: SR_approach, num_runs: int=10):

    def to_dict(self, base_path: str, name: str):
        output = {
            "symbol_library": self.symbol_library.to_dict(),
            "ranking_function": self.ranking_function,
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "original_equation": self.original_equation,
            "seed": self.seed,
            "dataset_metadata": self.dataset_metadata,
            "kwargs": self.kwargs
        }

        if self.result_augmenters is None:
            output["result_augmenters"] = None
        else:
            output["result_augmenters"] = [ag.to_dict(base_path, name) for ag in self.result_augmenters]

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
    def from_dict(d: dict, augmentation_map: dict):
        try:
            data = np.load(d["dataset_path"])
            X = data["X"]
            if "y" in data:
                y = data["y"]
            else:
                y = None
        except:
            raise Exception(f"[SR_dataset.from_dict] Could not load dataset from {d['dataset_path']}")

        if "ground_truth" in d and isinstance(d["ground_truth"], list) or d["ground_truth"] is None:
            ground_truth = d["ground_truth"]
        else:
            try:
                ground_truth = np.load(d["ground_truth"])
            except:
                raise Exception(f"[SR_dataset.from_dict] Could not load ground truth from {d['ground_truth']}")

        if not "result_augmenters" in d:
            raise Exception("[SR_dataset.from_dict] Could not find result_augmenters keyword in the provided dictionary.")

        if d["result_augmenters"] is None:
            result_augmenters = None
        else:
            result_augmenters = [augmentation_map[ag_data["type"]].from_dict(ag_data, augmentation_map)
                                 for ag_data in d["result_augmenters"]]

        try:
            return SR_dataset(X,
                              SymbolLibrary.from_dict(d["symbol_library"]),
                              ranking_function=d["ranking_function"],
                              y=y,
                              max_evaluations=d["max_evaluations"],
                              ground_truth=ground_truth,
                              original_equation=d["original_equation"],
                              success_threshold=d["success_threshold"],
                              result_augmenters=result_augmenters,
                              seed=d["seed"],
                              dataset_metadata=d["dataset_metadata"],
                              **d["kwargs"])
        except Exception as e:
            raise Exception(f"[SR_dataset.from_dict] Error creating dataset: {e}")
