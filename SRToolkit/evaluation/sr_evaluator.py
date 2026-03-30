"""
This module contains the SR_evaluator class, which is used for evaluating symbolic regression approaches. Additionally,
the generic ResultAugmenter class is defined here to avoid circular imports.
"""

import logging
import os
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import numpy as np
from scipy.stats.qmc import LatinHypercube

from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_simplifier import simplify
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.measures import bed, create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary

logger = logging.getLogger(__name__)


class _ModelResultBase(TypedDict):
    """Required fields present in every model entry."""

    expr: List[str]
    error: float


class ModelResult(_ModelResultBase, total=False):
    """
    A single model entry in ``EvalResult["top_models"]`` and ``EvalResult["all_models"]``.

    Attributes:
        expr: Token list representing the expression, e.g. ``["C", "*", "X_0"]``.
        error: Numeric error under the ranking function (RMSE or BED).
        parameters: (Optional) Fitted constant values. Present for RMSE ranking only.
        expr_latex: (Optional) LaTeX string. Added by :class:`ExpressionToLatex`.
        simplified_expr: (Optional) Simplified token string. Added by :class:`ExpressionSimplifier`.
        bed: (Optional) BED score. Added by :class:`BED` augmenter.
    """

    parameters: "np.ndarray"
    expr_latex: str
    simplified_expr: str
    bed: float


class _EvalResultBase(TypedDict):
    """Required fields present in every experiment result."""

    min_error: float
    best_expr: str
    num_evaluated: int
    evaluation_calls: int
    top_models: List[ModelResult]
    all_models: List[ModelResult]
    approach_name: str
    success: bool


class EvalResult(_EvalResultBase, total=False):
    """
    Result dictionary for a single SR experiment, as returned by ``SR_results[i]``.

    Attributes:
        min_error: Lowest error achieved across all evaluated expressions.
        best_expr: String representation of the best expression found.
        num_evaluated: Number of unique expressions evaluated.
        evaluation_calls: Number of times ``evaluate_expr`` was called (includes cache hits).
        top_models: Top-*k* models sorted by error, each a :class:`ModelResult`.
        all_models: All evaluated models sorted by error, each a :class:`ModelResult`.
        approach_name: Name of the SR approach, or empty string if not provided.
        success: Whether ``min_error`` is below the configured ``success_threshold``.
        dataset_name: (Optional) Name of the dataset, extracted from metadata.
        metadata: (Optional) Remaining metadata dict after ``dataset_name`` is popped.
        best_expr_latex: (Optional) LaTeX string of the best expression. Added by :class:`ExpressionToLatex`.
        simplified_best_expr: (Optional) Simplified best expression string. Added by :class:`ExpressionSimplifier`.
        best_expr_bed: (Optional) BED score of the best expression. Added by :class:`BED` augmenter.

    Custom augmenters may add arbitrary keys at runtime. Static type checkers
    will flag access to those keys unless you subclass ``EvalResult`` to declare
    them or suppress the error with ``cast``/``# type: ignore``.
    """

    dataset_name: str
    metadata: dict
    best_expr_latex: str
    simplified_best_expr: str
    best_expr_bed: float


class ResultAugmenter:
    def __init__(self):
        """
        Generic class that defines the interface for result augmentation. For examples, see the implementations of
        this class at SRToolkit.evaluation.result_augmentation.py.
        """
        raise NotImplementedError("ResultAugmenter is an abstract class and cannot be instantiated.")

    def augment_results(
        self,
        results: "EvalResult",
        models: List[ModelResult],
    ) -> "EvalResult":
        """
        Augments the results dictionary with additional information. The model variable contains all models, for only
        top models, results["top_models"] should be used.

        Args:
            results: The dictionary containing the results to augment.
            models: A list of :class:`ModelResult` dicts sorted by error (best first).

        Returns:
            The augmented results dictionary.
        """
        raise NotImplementedError("augment_results is not implemented as ResultAugmenter is an abstract class.")

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Transforms the augmenter into a dictionary. This is used for saving the augmenter to disk.

        Args:
            base_path: The base path used for saving the data inside the augmenter, if needed.
            name: The name/identifier used by the augmenter for saving.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """
        raise NotImplementedError("to_dict is not implemented as ResultAugmenter is an abstract class.")

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "ResultAugmenter":
        """
        Creates an instance of the ResultAugmenter class from the dictionary with the relevant data.

        Args:
            data: the dictionary containing the data needed to recreate the augmenter.
            augmenter_map: A dictionary mapping augmenter names to their classes.

        Returns:
            An instance of the ResultAugmenter class with the same configuration as in the data dictionary.
        """
        raise NotImplementedError("from_dict is not implemented as ResultAugmenter is an abstract class.")


class SR_evaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
        max_evaluations: int = -1,
        success_threshold: Optional[float] = None,
        ranking_function: str = "rmse",
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        result_augmenters: Optional[List[ResultAugmenter]] = None,
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the SR_evaluator class. This class is used for evaluating symbolic regression approaches.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True

        Args:
            X: The input data to be used in parameter estimation for variables. We assume that X is a 2D array with
                shape (n_samples, n_features).
            y: The target values to be used in parameter estimation.
            max_evaluations: The maximum number of expressions to evaluate. Default is -1, which means no limit.
            success_threshold: The threshold used for determining whether an expression is considered successful. If
                None, the threshold is set to 1e-7 for RMSE and calculated automatically for BED. For BED we calculate
                this value by evaluating the distance of ground truth to itself 100 times and setting the threshold to
                np.max(distances)*1.1. For this calculation to be helpful, ground_truth must be provided as a list of
                tokens or SRToolkit.utils.Node object.
            metadata: An optional dictionary containing metadata about this evaluation. This could include information
                such as the dataset used, the model used, seed, etc.
            symbol_library: The symbol library to use.
            ranking_function: The function used for ranking the expressions and fitting parameters if needed.
                Currently, "rmse" and "bed" are supported. Default is "rmse".
            ground_truth: The ground truth for the BED evaluation. This should be a list of tokens, a Node object, or a
                numpy array representing behavior (see SRToolkit.utils.create_behavior_matrix for more details).
            result_augmenters: Optional list of objects that augment the results returned by the "get_results" function.
                For example, SRToolkit.evaluation.result_augmentation.ExpressionSimplifier simplifies the evaluated
                expressions. Possible augmenters can be found in SRToolkit.evaluation.result_augmentation.py or customly
                defined by inheriting from SRToolkit.evaluation.result_augmentation.ResultAugmenter class.
            seed: The seed to use for random number generation.

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
                are sampled from the domain. If None and domain_bounds are not given, points are randomly selected
                from X. Default is None.
            num_consts_sampled (int): Number of constants sampled for BED evaluation. Default is 32.
            domain_bounds (Optional[List[Tuple[float, float]]]): Bounds for the domain to be used if bed_X is None to
                sample random points. Default is None.

        Attributes:
            models: A dictionary containing the results of previously evaluated expressions.
            invalid: A list containing the expressions that could not be evaluated.
            ground_truth: The ground truth we are trying to find.
            gt_behavior: The behavior matrix for the ground truth that is used when BED is chosen as the ranking function.
            max_evaluations: The maximum number of expressions to evaluate.
            bed_evaluation_parameters: A dictionary containing parameters used for BED evaluation.
            metadata: An optional dictionary containing metadata about this evaluation. This could include information
                such as the dataset used, the model used, seed, etc.
            symbol_library: The symbol library to use.
            total_evaluations: The number of times the "evaluate_expr" function was called.
            seed: The seed to use for random number generation.
            parameter_estimator: An instance of the ParameterEstimator class used for parameter estimation.
            ranking_function: The function used for ranking the expressions and fitting parameters if needed.
            success_threshold: The threshold used for determining whether an expression is considered successful.
            result_augmenters: A list of SRToolkit.evaluation.result_augmentation.ResultAugmenter objects that augment
                the results returned by the get_results.

        Methods:
            evaluate_expr(expr): Evaluates an expression in infix notation and stores the result in memory to prevent re-evaluation.
            get_results(top_k): Returns the results of the evaluation.

        Notes:
            Determining if two expressions are equivalent is undecidable. Furthermore, random sampling, parameter
            fitting, and numerical errors all make it hard to determine whether we found the correct expression.
            Because of this, the success threshold is only a proxy for determining the success of an expression.
            We recommend checking the best performing expression manually for a better indication of success.
        """
        self.kwargs = kwargs
        self.models: Dict[str, ModelResult] = dict()
        self.invalid: List[str] = list()
        self.success_threshold = success_threshold
        self.metadata = metadata
        self.ground_truth = ground_truth
        self.gt_behavior = None
        self.bed_evaluation_parameters: Dict[str, Any] = {
            "bed_X": None,
            "num_consts_sampled": 32,
            "num_points_sampled": 64,
            "domain_bounds": None,
            "constant_bounds": (-5, 5),
        }
        if kwargs:
            for k in self.bed_evaluation_parameters.keys():
                if k in kwargs:
                    self.bed_evaluation_parameters[k] = kwargs[k]
        if self.bed_evaluation_parameters["num_points_sampled"] == -1:
            self.bed_evaluation_parameters["num_points_sampled"] = X.shape[0]

        self.symbol_library = symbol_library
        self.max_evaluations = max_evaluations
        self.total_evaluations = 0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        if ranking_function not in ["rmse", "bed"]:
            warnings.warn(f"ranking_function '{ranking_function}' not supported. Using rmse instead.")
            ranking_function = "rmse"
        self.ranking_function = ranking_function

        self.result_augmenters = []
        if result_augmenters is not None:
            for ra in result_augmenters:
                if not isinstance(ra, ResultAugmenter):
                    warnings.warn(f"result_augmenter {ra} is not an instance of ResultAugmenter. Skipping.")
                else:
                    self.result_augmenters.append(ra)

        if ranking_function == "rmse":
            if y is None:
                raise ValueError("Target values must be provided for RMSE ranking function.")
            self.parameter_estimator = ParameterEstimator(X, y, symbol_library=symbol_library, seed=seed, **kwargs)

            if self.success_threshold is None:
                self.success_threshold = 1e-7

        elif ranking_function == "bed":
            if ground_truth is None:
                raise ValueError(
                    "Ground truth must be provided for bed ranking function. The ground truth must be "
                    "provided as a list of tokens, a Node object, or a numpy array representing behavior. "
                    "The behavior matrix is a matrix representing the distribution of outputs of an "
                    "expression with free parameters at different points in the domain. This matrix "
                    "should be of size (num_points_sampled, num_consts_sampled). See "
                    "SRToolkit.utils.create_behavior_matrix for more details."
                )
            else:
                if self.bed_evaluation_parameters["bed_X"] is None:
                    if self.bed_evaluation_parameters["domain_bounds"] is not None:
                        db = self.bed_evaluation_parameters["domain_bounds"]
                        assert isinstance(db, List), "Domain bounds should be a list of tuples."
                        interval_length = np.array([ub - lb for (lb, ub) in db])
                        lower_bound = np.array([lb for (lb, ub) in db])
                        lho = LatinHypercube(len(db), optimization="random-cd", seed=seed)
                        self.bed_evaluation_parameters["bed_X"] = (
                            lho.random(self.bed_evaluation_parameters["num_points_sampled"]) * interval_length
                            + lower_bound
                        )
                    else:
                        indices = np.random.choice(
                            X.shape[0],
                            size=self.bed_evaluation_parameters["num_points_sampled"],
                        )
                        self.bed_evaluation_parameters["bed_X"] = X[indices, :]

            if isinstance(ground_truth, (list, Node)):
                self.gt_behavior = create_behavior_matrix(
                    ground_truth,
                    self.bed_evaluation_parameters["bed_X"],
                    num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                    consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                    symbol_library=self.symbol_library,
                    seed=self.seed,
                )
            elif isinstance(ground_truth, np.ndarray):
                self.gt_behavior = ground_truth
            else:
                raise ValueError(
                    "Ground truth must be provided as a list of tokens, a Node object, or a numpy array representing behavior."
                )

            if self.success_threshold is None:
                assert self.ground_truth is not None, "Ground truth must be provided for BED ranking function."
                distances = [
                    bed(
                        self.ground_truth,
                        self.gt_behavior,
                        self.bed_evaluation_parameters["bed_X"],
                        num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                        num_points_sampled=self.bed_evaluation_parameters["num_points_sampled"],
                        domain_bounds=self.bed_evaluation_parameters["domain_bounds"],
                        consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                        symbol_library=self.symbol_library,
                    )
                    for i in range(100)
                ]
                self.success_threshold = np.max(distances) * 1.1

        self.X = X
        self.y = y

    def evaluate_expr(
        self,
        expr: Union[List[str], Node],
        simplify_expr: bool = False,
        verbose: int = 0,
    ) -> float:
        """
        Evaluates an expression in infix notation and stores the result in
        memory to prevent re-evaluation.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> X = np.array([[0, 1], [0, 2], [0, 3]])
            >>> y = np.array([2, 3, 4])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"], simplify_expr=True)
            >>> print(rmse < 1e-6)
            True
            >>> list(se.models.keys())[0]
            'C+X_1'
            >>> print(0.99 < se.models["C+X_1"]["parameters"][0] < 1.01)
            True
            >>> # Evaluating invalid expression returns nan and adds it to invalid list
            >>> print(se.evaluate_expr(["C", "*", "X_1", "X_0"]))
            nan
            >>> se.invalid
            ['C*X_1X_0']
            >>> X = np.random.rand(10, 2) - 0.5
            >>> gt = ["X_0", "+", "C"]
            >>> se = SR_evaluator(X, ground_truth=gt, ranking_function="bed", seed=42)
            >>> print(se.evaluate_expr(["C", "+", "X_1"]) < 1)
            True
            >>> # When evaluating using BED as the ranking function, the error depends on the scale of output of the
            >>> # ground truth. Because of stochasticity of BED, error might be high even when expressions match exactly.
            >>> print(se.evaluate_expr(["C", "+", "X_0"]) < 0.2)
            True
            >>> # X can also be sampled from a domain by providing domain_bounds
            >>> se = SR_evaluator(X, ground_truth=gt, ranking_function="bed", domain_bounds=[(-1, 1), (-1, 1)], seed=42)
            >>> print(se.evaluate_expr(["C", "+", "X_0"]) < 0.2)
            True

        Args:
            expr: An expression. This should be an instance of the SRToolkit.utils.expression_tree.Node class or a list
                  of tokens in the infix notation.
            simplify_expr: If True, simplifies the expression using SymPy before evaluating it. This typically slows down
                           evaluation. We recommend simplifying only the best expressions when getting results using
                           the get_results method.
            verbose: When 0, no additional output is printed, when 1, prints the expression being evaluated, RMSE, and
                     estimated parameters, and when 2, also prints numpy errors produced during evaluation.

        Returns:
            The root-mean-square error of the expression.

        Warnings:
            Maximum number of evaluations reached: If the maximum number of evaluations has been reached, a warning is printed and np.nan is returned.

        Notes:
            If the expression has already been evaluated, its stored value is returned instead of re-evaluating the expression.
            When the maximum number of evaluations has been reached, a warning is printed and np.nan is returned.
        """
        self.total_evaluations += 1

        if 0 <= self.max_evaluations < self.total_evaluations:
            warnings.warn(f"Maximum number of evaluations ({self.max_evaluations}) reached. Stopping evaluation.")
            return np.nan
        else:
            if simplify_expr:
                try:
                    expr = simplify(expr, self.symbol_library)
                except Exception as e:
                    if isinstance(expr, Node):
                        expr_list = expr.to_list(symbol_library=self.symbol_library)
                    else:
                        expr_list = expr
                    warnings.warn(f"Unable to simplify: {''.join(expr_list)}, problems with subexpression {e}")

            if isinstance(expr, Node):
                expr_list = expr.to_list(symbol_library=self.symbol_library)
            else:
                expr_list = expr

            expr_str = "".join(expr_list)
            if expr_str in self.models:
                if verbose > 0:
                    logger.debug("Already evaluated %s", expr_str)
                return self.models[expr_str]["error"]

            else:
                if self.ranking_function == "rmse":
                    try:
                        with (
                            np.errstate(
                                divide="ignore",
                                invalid="ignore",
                                over="ignore",
                                under="ignore",
                            )
                            if verbose < 2
                            else nullcontext()
                        ):
                            error, parameters = self.parameter_estimator.estimate_parameters(expr)

                        if verbose > 0:
                            if parameters.size > 0:
                                parameter_string = (
                                    f" Best parameters found are [{', '.join([str(round(p, 3)) for p in parameters])}]"
                                )
                            else:
                                parameter_string = ""
                            logger.debug("Evaluated expression %s with RMSE: %s.%s", expr_str, error, parameter_string)

                    except Exception as e:
                        if verbose > 0:
                            logger.debug("Error evaluating expression %s: %s", expr_str, e)

                        self.invalid.append(expr_str)
                        error, parameters = np.nan, np.array([])

                    self.models[expr_str] = {
                        "error": error,
                        "parameters": parameters,
                        "expr": expr_list,
                    }

                elif self.ranking_function == "bed":
                    try:
                        with (
                            np.errstate(
                                divide="ignore",
                                invalid="ignore",
                                over="ignore",
                                under="ignore",
                            )
                            if verbose < 2
                            else nullcontext()
                        ):
                            assert self.gt_behavior is not None, (
                                "Ground truth must be provided for BED ranking function."
                            )
                            error = bed(
                                expr,
                                self.gt_behavior,
                                self.bed_evaluation_parameters["bed_X"],
                                num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                                num_points_sampled=self.bed_evaluation_parameters["num_points_sampled"],
                                domain_bounds=self.bed_evaluation_parameters["domain_bounds"],
                                consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                                symbol_library=self.symbol_library,
                                seed=self.seed,
                            )

                            if verbose > 0:
                                logger.debug("Evaluated expression %s with BED: %s.", expr_str, error)

                    except Exception as e:
                        if verbose > 0:
                            logger.debug("Error evaluating expression %s: %s", expr_str, e)

                        self.invalid.append(expr_str)
                        error = np.nan

                    self.models[expr_str] = {
                        "error": error,
                        "expr": expr_list,
                    }

                else:
                    raise ValueError(f"Ranking function {self.ranking_function} not supported.")

                return error

    # TODO: Add a way to add custom information to a given expression (e.g. probability of expression in ProGED)

    def get_results(
        self, approach_name: str = "", top_k: int = 20, results: Optional["SR_results"] = None
    ) -> "SR_results":
        """
        Returns the results of the equation discovery/symbolic regression process/evaluation.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> print(results[0]["num_evaluated"])
            1
            >>> print(results[0]["evaluation_calls"])
            1
            >>> print(results[0]["best_expr"])
            C*X_1-X_0
            >>> print(results[0]["min_error"] < 1e-6)
            True
            >>> print(1.99 < results[0]["top_models"][0]["parameters"][0] < 2.01)
            True

        Args:
            approach_name: The name of the approach used to discover the equations.
            top_k: The number of top results to include in the output. If `top_k`
                is greater than the number of evaluated expressions, all
                evaluated expressions are included. If `top_k` is less than 0,
                all evaluated expressions are included.
            results: An SR_results object containing the results of the previous evaluation. If provided,
                the results of the current evaluation are appended to the existing results. Otherwise, a new SR_results
                object is created.

        Returns:
            An instance of the SR_results object with the results of the evaluation.
        """
        if top_k > len(self.models) or top_k < 0:
            top_k = len(self.models)

        if results is None:
            results = SR_results()
        results.add_results(
            self.models,
            top_k,
            self.result_augmenters,
            self.total_evaluations,
            self.success_threshold,
            approach_name,
            self.metadata,
        )

        return results

    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Creates a dictionary representation of the SR_evaluator.

        Args:
            base_path: Used to save the data of the evaluator to disk.
            name: Used to save the data of the evaluator to disk.

        Returns:
            A dictionary containing the necessary information to recreate the evaluator from disk.
        """
        output = {
            "format_version": 1,
            "type": "SR_evaluator",
            "metadata": self.metadata,
            "symbol_library": self.symbol_library.to_dict(),
            "max_evaluations": self.max_evaluations,
            "success_threshold": self.success_threshold,
            "ranking_function": self.ranking_function,
            "result_augmenters": [ra.to_dict(base_path, name) for ra in self.result_augmenters],
            "seed": self.seed,
            "kwargs": self.kwargs,
        }

        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        X_path = f"{base_path}/{name}_X.npy"
        np.save(X_path, self.X)
        output["X"] = X_path

        if self.y is not None:
            y_path = f"{base_path}/{name}_y.npy"
            np.save(y_path, self.y)
            output["y"] = y_path
        else:
            output["y"] = None

        if self.ground_truth is None:
            output["ground_truth"] = None
        else:
            if isinstance(self.ground_truth, list):
                output["ground_truth"] = self.ground_truth
            elif isinstance(self.ground_truth, Node):
                output["ground_truth"] = self.ground_truth.to_list(self.symbol_library)
            else:
                gt_path = f"{base_path}/{name}_gt.npy"
                np.save(gt_path, self.ground_truth)
                output["ground_truth"] = gt_path

        return output

    @staticmethod
    def from_dict(data: dict, augmenter_map: Optional[dict] = None) -> "SR_evaluator":
        """
        Creates an instance of the SR_evaluator from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the evaluator.
            augmenter_map: A dictionary mapping the names of the augmenters to the augmenter classes.

        Returns:
            An instance of the SR_evaluator.

        Raises:
            Exception: if unable to load data for X/y/ground truth data, if result augmenters provided but not the
                augmenter map or if the result augmentor does not occur in the augmenter map.
        """
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[SR_evaluator.from_dict] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )

        try:
            X = np.load(data["X"])

            if data["y"] is not None:
                y = np.load(data["y"])
            else:
                y = None

            if data["ground_truth"] is None:
                gt = None
            else:
                if isinstance(data["ground_truth"], list):
                    gt = data["ground_truth"]
                else:
                    gt = np.load(data["ground_truth"])
        except Exception as e:
            raise ValueError(f"[SR_evaluator.from_dict] Unable to load data for X/y/ground truth due to {e}")

        result_augmenters = []
        for ra_data in data["result_augmenters"]:
            if augmenter_map is None:
                raise ValueError(
                    "[SR_evaluator.from_dict] Argument augmenter_map must be provided when loading "
                    "the dictionary contains result augmenters."
                )
            if ra_data["type"] not in augmenter_map:
                raise ValueError(
                    f"[SR_evaluator.from_dict] Result augmenter {ra_data['type']} not found in the augmenter map."
                )
            result_augmenters.append(augmenter_map[ra_data["type"]].from_dict(ra_data, augmenter_map))

        symbol_library = SymbolLibrary.from_dict(data["symbol_library"])
        return SR_evaluator(
            X,
            y=y,
            ground_truth=gt,
            symbol_library=symbol_library,
            max_evaluations=data["max_evaluations"],
            success_threshold=data["success_threshold"],
            ranking_function=data["ranking_function"],
            result_augmenters=result_augmenters,
            seed=data["seed"],
            metadata=data["metadata"],
            **data["kwargs"],
        )


class SR_results:
    def __init__(self):
        """
        Initializes an SR_results object. This object stores the results of an equation discovery/symbolic regression experiments.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1) # Obtain an instance of SR_results
            >>> print(results[0]["num_evaluated"])
            1
            >>> print(results[0]["evaluation_calls"])
            1
            >>> print(results[0]["best_expr"])
            C*X_1-X_0
            >>> print(results[0]["min_error"] < 1e-6)
            True
            >>> print(1.99 < results[0]["top_models"][0]["parameters"][0] < 2.01)
            True

        Attributes:
            results: A list of dictionaries containing the results of each evaluation.

        Methods:
            add_results: Adds the results of an evaluation to the results object. If needed, the results are
                additionally augmented using the provided result augmenters.
            print_results: Prints the results of the evaluation.
            __len__: Returns the number of results stored in the results object.
        """
        self.results = list()

    def add_results(
        self,
        models: Dict[str, ModelResult],
        top_k: int,
        result_augmenters: Optional[List[ResultAugmenter]],
        total_evaluations: int,
        success_threshold: Optional[float],
        approach_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Adds the results of an evaluation to the results object. If needed, the results are additionally augmented
        using the provided result augmenters. For an example of how to use this function, look at the SR_evaluator.get_results method.

        Args:
            models: A dictionary mapping expressions to their evaluation results.
            top_k: The number of top results to include in the output.
            result_augmenters: A list of result augmenters to use for augmenting the results.
            total_evaluations: The total number of evaluations performed during the evaluation.
            success_threshold: The success threshold used to determine whether the evaluation was successful.
            approach_name: The name of the approach used to discover the equations.
            metadata: A dictionary containing additional metadata about the evaluation.
        """
        models_list = list(models.values())
        best_indices = np.argsort([v["error"] for v in models_list])
        best_models = [models_list[i] for i in best_indices]

        results_dict = cast(
            EvalResult,
            cast(
                object,
                {
                    "min_error": best_models[0]["error"],
                    "best_expr": "".join(best_models[0]["expr"]),
                    "num_evaluated": len(models_list),
                    "evaluation_calls": total_evaluations,
                    "top_models": list(),
                    "all_models": models_list,
                    "approach_name": approach_name,
                },
            ),
        )

        if metadata is not None and "dataset_name" in metadata:
            results_dict["dataset_name"] = metadata["dataset_name"]
            metadata.pop("dataset_name")
            if len(metadata) > 0:
                results_dict["metadata"] = metadata
        elif metadata is not None:
            results_dict["metadata"] = metadata

        # Determine success based on the predefined success threshold
        if success_threshold is not None and results_dict["min_error"] < success_threshold:
            results_dict["success"] = True
        else:
            results_dict["success"] = False

        for model in best_models[:top_k]:
            m: ModelResult = {"expr": model["expr"], "error": model["error"]}
            if "parameters" in model:
                m["parameters"] = model["parameters"]

            results_dict["top_models"].append(m)

        if result_augmenters is not None:
            for augmenter in result_augmenters:
                try:
                    results_dict = augmenter.augment_results(results_dict, models_list)
                except Exception as e:
                    warnings.warn(
                        f"Error augmenting results, skipping current augmentor because of the following error: {e}"
                    )

        self.results.append(results_dict)

    def print_results(self, experiment_number: Optional[int] = None, detailed: bool = False):
        r"""
        Prints the results of the SR_evaluator. Specifically, prints the minimum error, the best expression,
        the number of evaluated expressions, the number of times the "evaluate_expr" function was called, whether
        the evaluation was successful, and the metadata and the approach name, if present. If detailed is True, prints
        all the information about the top models.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> results.print_results()  # doctest: +ELLIPSIS
            Experiment 1/1:
            Best expression found: C*X_1-X_0
            Error: ...
            Number of evaluated expressions: 1
            Number of times evaluate_expr was called: 1
            Success: ...
            <BLANKLINE>
            -----------------------------------------
            >>> results.print_results(detailed=True, experiment_number=0)  # doctest: +ELLIPSIS
            Best expression found: C*X_1-X_0
            Error: ...
            Number of evaluated expressions: 1
            Number of times evaluate_expr was called: 1
            Success: ...
            <BLANKLINE>
            Top models:
            Model 1 - expr: ['C', '*', 'X_1', '-', 'X_0'], error: ..., parameters: [...]
            <BLANKLINE>

        Args:
            experiment_number: Number of the experiment you want to print the results for. If None, prints the results for all experiments.
            detailed: If True, prints all the information about the top models.

        """
        if experiment_number is None:
            for i, result in enumerate(self.results):
                print(f"Experiment {i + 1}/{len(self.results)}:")
                SR_results._print_result_(result, detailed)
                print("-----------------------------------------")

        else:
            assert experiment_number < len(self.results), "[SR_Results.print_results] experiment number out of bounds"
            SR_results._print_result_(self.results[experiment_number], detailed)

    @staticmethod
    def _print_result_(result, detailed: bool = False):
        if "dataset_name" in result:
            print(f"Dataset: {result['dataset_name']}")
        if result["approach_name"] != "":
            print(f"Approach: {result['approach_name']}")
        print(f"Best expression found: {result['best_expr']}")
        print(f"Error: {result['min_error']}")
        print(f"Number of evaluated expressions: {result['num_evaluated']}")
        print(f"Number of times evaluate_expr was called: {result['evaluation_calls']}")
        print(f"Success: {result['success']}")
        print()
        if "metadata" in result and result["metadata"] is not None and len(result["metadata"]) > 0:
            print("Metadata:")
            for key, value in result["metadata"].items():
                print(f"{key}: {value}")
            print()
        if detailed:
            print("Top models:")
            for j, model in enumerate(result["top_models"]):
                print(f"Model {j + 1} - " + ", ".join(["{}: {}".format(key, value) for key, value in model.items()]))
            print()

    # TODO: Function that creates a pareto front
    # TODO: Function that creates a table with results
    # TODO: Function that returns the best expression

    def __add__(self, other):
        """
        Returns a new SR_results object that is the concatenation of the current SR_results object with the other SR_results object.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            A new SR_results object containing the concatenated results.
        """
        new = SR_results()
        new.results = self.results + other.results
        return new

    def __iadd__(self, other):
        """
        In-place concatenation of SR_results objects.

        Args:
            other: SR_results object to concatenate with the current SR_results object.

        Returns:
            self
        """
        self.results += other.results
        return self

    def __getitem__(self, item) -> EvalResult:
        """
        Returns the results of the experiment with the given index.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> result_of_first_experiment = results[0]

        Args:
            item: the index of the experiment.

        Returns:
            The results of the experiment with the given index.

        """
        assert isinstance(item, int), "[SR_Results.__getitem__] Item must be an integer."
        assert 0 <= item < len(self.results), "[SR_Results.__getitem__] Item out of bounds."
        return self.results[item]

    def __len__(self) -> int:
        """
        Returns the number of results stored in the results object. Usually, each result corresponds to a single experiment.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> len(results)
            1

        Returns:
            The number of results stored in the results object.
        """
        return len(self.results)
