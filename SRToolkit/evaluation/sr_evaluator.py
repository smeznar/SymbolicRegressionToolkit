"""
This module contains the SR_evaluator class, which is used for evaluating symbolic regression approaches. Additionally,
the generic ResultAugmenter class is defined here to avoid circular imports.
"""

import json
import logging
import os
import warnings
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from scipy.stats.qmc import LatinHypercube
from typing_extensions import Literal, Unpack

from SRToolkit.evaluation.callbacks import BestExpressionFound, CallbackDispatcher, ExprEvaluated, SRCallbacks
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.utils.expression_simplifier import simplify
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.measures import bed, create_behavior_matrix
from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.types import EstimationSettings, EvalResult, ModelResult

logger = logging.getLogger(__name__)


class ResultAugmenter(ABC):
    _type: ClassVar[str] = ""

    def __init__(self, name: str):
        """
        Base class for result augmenters. Subclasses implement :meth:`write_results` to compute
        and store additional data in an :class:`EvalResult` via :meth:`EvalResult.add_augmentation`.

        For examples, see the implementations in ``SRToolkit.evaluation.result_augmentation``.

        Args:
            name: Identifier used as the key in :attr:`EvalResult.augmentations` and
                :attr:`ModelResult.augmentations`. If two augmenters share the same name,
                :meth:`EvalResult.add_augmentation` appends a numeric suffix automatically.
        """
        self.name = name

    @abstractmethod
    def write_results(
        self,
        results: "EvalResult",
    ) -> None:
        """
        Computes and writes augmentation data into *results* and its top models.

        Call ``results.add_augmentation(self.name, data, self._type)`` for experiment-level
        data and ``model.add_augmentation(self.name, data, self._type)`` for per-model data.

        Args:
            results: The :class:`EvalResult` to augment.
        """

    @abstractmethod
    def to_dict(self, base_path: str, name: str) -> dict:
        """
        Transforms the augmenter into a dictionary. This is used for saving the augmenter to disk.

        Args:
            base_path: The base path used for saving the data inside the augmenter, if needed.
            name: The name/identifier used by the augmenter for saving.

        Returns:
            A dictionary containing the necessary information to recreate the augmenter.
        """

    @classmethod
    def format_eval_result(cls, data: Dict[str, Any]) -> str:
        """
        Returns a formatted string for experiment-level augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        return "\n".join(f"  {k}: {v}" for k, v in data.items() if k != "_type")

    @classmethod
    def format_model_result(cls, data: Dict[str, Any]) -> str:
        """
        Returns a formatted string for a single model's augmentation data.

        Subclasses override this for custom formatting. The *data* dict is the inner
        augmentation dictionary (includes ``_type``).

        Args:
            data: The augmentation data dictionary.

        Returns:
            A formatted string, or empty string if no relevant data exists.
        """
        parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
        return ", ".join(parts)

    @staticmethod
    def from_dict(data: dict) -> "ResultAugmenter":
        """
        Creates an instance of the ResultAugmenter class from the dictionary with the relevant data.

        Subclasses should override this method if they support serialization. The default
        implementation raises ``NotImplementedError``, allowing custom augmenters to skip
        serialization if not needed.

        Args:
            data: the dictionary containing the data needed to recreate the augmenter.

        Returns:
            An instance of the ResultAugmenter class with the same configuration as in the data dictionary.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "from_dict is not implemented for this augmenter. "
            "Override this method if your augmenter supports serialization."
        )


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
        seed: Optional[int] = None,
        metadata: Optional[dict] = None,
        **kwargs: Unpack[EstimationSettings],
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
        self._callbacks: Optional[Union[CallbackDispatcher, SRCallbacks]] = None
        self._should_stop = False
        self._current_best_error = float("inf")
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
                    self.bed_evaluation_parameters[k] = kwargs[k]  # type: ignore[literal-required]
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

    def set_callbacks(self, callbacks: Union[SRCallbacks, CallbackDispatcher]) -> None:
        """
        Set callbacks for monitoring and controlling the search.

        Args:
            callbacks: A CallbackDispatcher instance or a single SRCallbacks instance.
        """
        if isinstance(callbacks, CallbackDispatcher):
            self._callbacks = callbacks
        elif isinstance(callbacks, SRCallbacks):
            dispatcher = CallbackDispatcher(callbacks=[callbacks])
            self._callbacks = dispatcher

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
            >>> print(0.99 < se.models["C+X_1"].parameters[0] < 1.01)
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
                return self.models[expr_str].error

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

                    self.models[expr_str] = ModelResult(
                        expr=expr_list,
                        error=error,
                        parameters=parameters,
                    )

                    if self._callbacks is not None:
                        is_new_best = error < self._current_best_error
                        if is_new_best:
                            self._current_best_error = error
                        event = ExprEvaluated(
                            expression=expr_str,
                            error=error,
                            evaluation_number=self.total_evaluations,
                            experiment_id=0,  # TODO: Change through meta-data when defined
                            is_new_best=is_new_best,
                        )
                        should_continue = self._callbacks.on_expr_evaluated(event)
                        if should_continue is False:
                            self._should_stop = True
                        if is_new_best:
                            best_event = BestExpressionFound(
                                experiment_id=0,  # TODO: Change through meta-data when defined
                                expression=expr_str,
                                error=error,
                                evaluation_number=self.total_evaluations,
                            )
                            self._callbacks.on_best_expression(best_event)

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

                    self.models[expr_str] = ModelResult(
                        expr=expr_list,
                        error=error,
                    )

                    if self._callbacks is not None:
                        is_new_best = error < self._current_best_error
                        if is_new_best:
                            self._current_best_error = error
                        event = ExprEvaluated(
                            expression=expr_str,
                            error=error,
                            evaluation_number=self.total_evaluations,
                            experiment_id=0,  # TODO: Change through meta-data when defined
                            is_new_best=is_new_best,
                        )
                        should_continue = self._callbacks.on_expr_evaluated(event)
                        if should_continue is False:
                            self._should_stop = True
                        if is_new_best:
                            best_event = BestExpressionFound(
                                experiment_id=0,  # TODO: Change through meta-data when defined
                                expression=expr_str,
                                error=error,
                                evaluation_number=self.total_evaluations,
                            )
                            self._callbacks.on_best_expression(best_event)

                else:
                    raise ValueError(f"Ranking function {self.ranking_function} not supported.")

                return error

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
            >>> print(results[0].num_evaluated)
            1
            >>> print(results[0].evaluation_calls)
            1
            >>> print(results[0].best_expr)
            C*X_1-X_0
            >>> print(results[0].min_error < 1e-6)
            True
            >>> print(1.99 < results[0].top_models[0].parameters[0] < 2.01)
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
    def from_dict(data: dict) -> "SR_evaluator":
        """
        Creates an instance of the SR_evaluator from a dictionary.

        Args:
            data: A dictionary containing the necessary information to recreate the evaluator.

        Returns:
            An instance of the SR_evaluator.
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

        symbol_library = SymbolLibrary.from_dict(data["symbol_library"])
        return SR_evaluator(
            X,
            y=y,
            ground_truth=gt,
            symbol_library=symbol_library,
            max_evaluations=data["max_evaluations"],
            success_threshold=data["success_threshold"],
            ranking_function=data["ranking_function"],
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
            >>> print(results[0].num_evaluated)
            1
            >>> print(results[0].evaluation_calls)
            1
            >>> print(results[0].best_expr)
            C*X_1-X_0
            >>> print(results[0].min_error < 1e-6)
            True
            >>> print(1.99 < results[0].top_models[0].parameters[0] < 2.01)
            True

        Attributes:
            results: A list of :class:`EvalResult` instances containing the results of each evaluation.

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
        total_evaluations: int,
        success_threshold: Optional[float],
        approach_name: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Adds the results of an evaluation to the results object.

        Args:
            models: A dictionary mapping expressions to their evaluation results.
            top_k: The number of top results to include in the output.
            total_evaluations: The total number of evaluations performed during the evaluation.
            success_threshold: The success threshold used to determine whether the evaluation was successful.
            approach_name: The name of the approach used to discover the equations.
            metadata: A dictionary containing additional metadata about the evaluation.
        """
        models_list = list(models.values())
        sorted_indices = np.argsort([v.error for v in models_list])
        sorted_models = [models_list[i] for i in sorted_indices]

        dataset_name = None
        remaining_metadata = None
        if metadata is not None and "dataset_name" in metadata:
            dataset_name = metadata["dataset_name"]
            remaining_metadata = {key: value for key, value in metadata.items() if key != "dataset_name"}
            if len(remaining_metadata) == 0:
                remaining_metadata = None
        elif metadata is not None:
            remaining_metadata = metadata

        success = success_threshold is not None and sorted_models[0].error < success_threshold

        results_obj = EvalResult(
            min_error=sorted_models[0].error,
            best_expr="".join(sorted_models[0].expr),
            num_evaluated=len(models_list),
            evaluation_calls=total_evaluations,
            top_models=sorted_models[:top_k],
            all_models=models_list,
            approach_name=approach_name,
            success=success,
            dataset_name=dataset_name,
            metadata=remaining_metadata,
        )

        self.results.append(results_obj)

    def print_results(
        self,
        experiment_number: Optional[int] = None,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations: Optional[List[str]] = None,
    ):
        r"""
        Prints the results of the SR_evaluator.

        Displays the minimum error, best expression, evaluation counts, success status,
        metadata, and approach name. When *detailed* is True, also prints per-model
        information. Augmentation data is formatted by the corresponding
        :class:`ResultAugmenter` class, looked up from the global registry via the
        ``_type`` field stored in each augmentation entry.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> results.print_results()  # doctest: +ELLIPSIS
            === Experiment 1/1 ===
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            <BLANKLINE>
            >>> results.print_results(detailed=True, experiment_number=0)  # doctest: +ELLIPSIS
            Best expression: C*X_1-X_0
            Error: ...
            Evaluated: 1 expressions | Calls: 1 | Success: ...
            <BLANKLINE>
            Models:
              C*X_1-X_0  (error=..., params=...)
            <BLANKLINE>

        Args:
            experiment_number: Number of the experiment to print. If None, prints all.
            detailed: If True, prints per-model information.
            model_scope: Which models to show when *detailed* is True.
                ``"best"`` shows only the top model, ``"top"`` shows the top-k,
                ``"all"`` shows all evaluated models.
            augmentations: Filter which augmenters to display by name.
                If None, all augmentations present in the data are shown.
        """
        if experiment_number is None:
            for i, result in enumerate(self.results):
                print(f"=== Experiment {i + 1}/{len(self.results)} ===")
                SR_results._print_result_(result, detailed, model_scope, augmentations)
                print()
        else:
            assert experiment_number < len(self.results), "[SR_results.print_results] experiment number out of bounds"
            SR_results._print_result_(self.results[experiment_number], detailed, model_scope, augmentations)

    @staticmethod
    def _print_result_(
        result: EvalResult,
        detailed: bool = False,
        model_scope: Literal["best", "top", "all"] = "top",
        augmentations_filter: Optional[List[str]] = None,
    ):
        # TODO: Check if this works for custom augmenters even after registering
        from SRToolkit.evaluation.result_augmentation import RESULT_AUGMENTERS

        # --- Core info ---
        if result.dataset_name is not None:
            print(f"Dataset: {result.dataset_name}")
        if result.approach_name != "":
            print(f"Approach: {result.approach_name}")
        print(f"Best expression: {result.best_expr}")
        print(f"Error: {result.min_error}")
        print(
            f"Evaluated: {result.num_evaluated} expressions | Calls: {result.evaluation_calls} | Success: {result.success}"
        )
        if result.metadata is not None and len(result.metadata) > 0:
            print("Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        print()

        # --- Resolve which augmenter keys to show ---
        eval_aug_keys = list(result.augmentations.keys())
        if augmentations_filter is not None:
            eval_aug_keys = [k for k in eval_aug_keys if k in augmentations_filter]

        # --- Experiment-level augmentation sections ---
        for key in eval_aug_keys:
            data = result.augmentations[key]
            type_str = data.get("_type", "")
            cls = RESULT_AUGMENTERS.get(type_str) if type_str else None
            print(f"--- {key} ---")
            if cls is not None:
                line = cls.format_eval_result(data)
                if line:
                    print(line)
            else:
                if type_str:
                    warnings.warn(
                        f"No registered augmenter for type '{type_str}'. "
                        f"Register with register_augmenter() or fall back to default dump."
                    )
                for k, v in data.items():
                    if k != "_type":
                        print(f"  {k}: {v}")

        # --- Per-model output ---
        if detailed:
            # Pick model list based on scope
            if model_scope == "best":
                models = result.top_models[:1]
            elif model_scope == "top":
                models = result.top_models
            else:
                models = result.all_models

            if models:
                print()
                print("Models:")
                for model in models:
                    parts = [f"error={model.error:.6g}"]
                    if model.parameters is not None:
                        parts.append(f"params={np.round(model.parameters, 4).tolist()}")
                    expr_str = "".join(model.expr)
                    print(f"  {expr_str}  ({', '.join(parts)})")

                    # Model-level augmentations
                    model_aug_keys = list(model.augmentations.keys())
                    if augmentations_filter is not None:
                        model_aug_keys = [k for k in model_aug_keys if k in augmentations_filter]
                    for key in model_aug_keys:
                        data = model.augmentations[key]
                        type_str = data.get("_type", "")
                        cls = RESULT_AUGMENTERS.get(type_str) if type_str else None
                        if cls is not None:
                            line = cls.format_model_result(data)
                            if line:
                                print(f"    {key}: {line}")
                        else:
                            parts = [f"{k}={v}" for k, v in data.items() if k != "_type"]
                            if parts:
                                print(f"    {key}: {', '.join(parts)}")

    def augment(self, augmenters: List[ResultAugmenter], experiment_number: Optional[int] = None) -> None:
        r"""
        Applies the given augmenters to the results. Augmenters add additional information to the results,
        such as LaTeX representations, simplified expressions, or R2 scores.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> from SRToolkit.evaluation.result_augmentation import ExpressionToLatex
            >>> results.augment([ExpressionToLatex(SymbolLibrary.default_symbols(2))])
            >>> results[0].augmentations["ExpressionToLatex"]["best_expr_latex"]  # doctest: +ELLIPSIS
            '$C_{0} \\cdot X_{1} - X_{0}$'

        Args:
            augmenters: A list of ResultAugmenter objects to apply to the results.
            experiment_number: If provided, apply augmenters only to this experiment's result.
                If None, apply to all results.
        """
        if experiment_number is not None:
            assert experiment_number < len(self.results), "[SR_results.augment] experiment number out of bounds"
            for augmenter in augmenters:
                try:
                    augmenter.write_results(self.results[experiment_number])
                except Exception as e:
                    warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")
        else:
            for result in self.results:
                for augmenter in augmenters:
                    try:
                        augmenter.write_results(result)
                    except Exception as e:
                        warnings.warn(f"Error augmenting results with {augmenter.name}, skipping: {e}")

    def __add__(self, other) -> "SR_results":
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

    def __iadd__(self, other) -> "SR_results":
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

    def save(self, path: str) -> None:
        """
        Saves the results to a directory as JSON.

        Creates *path* if it does not exist, then writes ``results.json`` inside it.

        Examples:
            >>> import tempfile
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y, seed=42)
            >>> _ = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     results.save(tmpdir + "/my_results")
            ...     loaded = SR_results.load(tmpdir + "/my_results")
            ...     print(loaded[0].best_expr)
            C*X_1-X_0

        Args:
            path: Directory path where the results will be saved.
        """
        if not os.path.isdir(path):
            os.makedirs(path)
        output = {
            "format_version": 1,
            "type": "SR_results",
            "results": [r.to_dict() for r in self.results],
        }
        with open(os.path.join(path, "results.json"), "w") as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def load(path: str) -> "SR_results":
        """
        Loads results previously saved with :meth:`save`.

        Args:
            path: Directory path containing a ``results.json`` file.

        Returns:
            A new :class:`SR_results` instance with the loaded data.
        """
        results_path = os.path.join(path, "results.json")
        with open(results_path, "r") as f:
            data = json.load(f)
        if data.get("format_version", 1) != 1:
            raise ValueError(
                f"[SR_results.load] Unsupported format_version: {data.get('format_version')!r}. Expected 1."
            )
        sr_results = SR_results()
        sr_results.results = [EvalResult.from_dict(r) for r in data["results"]]
        return sr_results
