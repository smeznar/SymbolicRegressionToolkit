"""
This module contains the SR_evaluator class, which is used for evaluating symbolic regression approaches.
"""
# TODO: Fix documentation examples
from typing import Optional, List, Union, Tuple, TypedDict
import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube

from SRToolkit.utils import Node, SymbolLibrary, simplify, create_behavior_matrix, bed
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator

# Maybe add an __all__ variable with public methods
_MEASURE_DICT = {
    "rmse": lambda expr1, expr2, y1, y2, params1, params2, X: np.sqrt(np.mean((y1 - expr1(X, params1)) ** 2)),
    "bed": lambda expr1, expr2, y1, y2, params1, params2, X: 0 # TODO: popravi
}

class EvaluatorKwargs(TypedDict, total=False):
    method: str
    tol: float
    gtol: float
    max_iter: int
    constant_bounds: Tuple[float, float]
    initialization: str
    max_constants: int
    max_expr_length: int
    num_points_sampled: int
    bed_X: Optional[np.ndarray]
    num_consts_sampled: int
    domain_bounds: Optional[List[Tuple[float, float]]]

class SR_evaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_evaluations: int = -1,
        metadata: Optional[dict] = None,
        symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
        ranking_function: str = "rmse",
        ground_truth: Optional[Union[List[str], Node, np.ndarray]] = None,
        evaluation_measures: Optional[List[Union[str, Tuple[str, callable]]]]=None,
        seed: Optional[int] = None,
        **kwargs: EvaluatorKwargs
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


        Attributes:
            models: A dictionary containing the results of previously evaluated expressions.
            max_evaluations: The maximum number of expressions to evaluate.
            metadata: An optional dictionary containing metadata about this evaluation. This could include information such as the dataset used, the model used, seed, etc.
            symbol_library: The symbol library to use.
            total_expressions: The total number of expressions considered.
            parameter_estimator: An instance of the ParameterEstimator class used for parameter estimation.

        Args:
            X: The input data to be used in parameter estimation for variables. We assume that X is a 2D array with shape (n_samples, n_features).
            y: The target values to be used in parameter estimation.
            max_evaluations: The maximum number of expressions to evaluate. Default is -1, which means no limit.
            metadata: An optional dictionary containing metadata about this evaluation. This could include information such as the dataset used, the model used, seed, etc.
            symbol_library: The symbol library to use.
            ranking_function: The function used for ranking the expressions and fitting parameters if needed.
                Currently, "rmse" and "bed" are supported. Default is "rmse".
            evaluation_measures: Optional list of additional metrics to compute for the top expressions reported by
                get_results (as controlled by the "top_k" parameter). Each item may be either:
                  - a string name of a built-in metric (e.g., "rmse"), or
                  - a tuple (name, fn) where fn is a callable with signature
                    fn(expr1: List[str], expr2: List[str], y1: np.ndarray, y2: np.ndarray,
                       params1: np.ndarray, params2: np.ndarray, X: np.ndarray) -> float.
                These metrics do not affect ranking (which is controlled by "ranking_function"); they are computed for
                reporting. If None, it defaults to ["rmse"] (i.e., only RMSE is computed).
            seed: The seed to use for random number generation.

        Keyword Arguments:
            method str: The method to be used for minimization. Currently, only "L-BFGS-B" is supported/tested. Default is "L-BFGS-B".
            tol float: The tolerance for termination. Default is 1e-6.
            gtol float: The tolerance for the gradient norm. Default is 1e-3.
            max_iter int: The maximum number of iterations. Default is 100.
            constant_bounds Tuple[float, float]: A tuple of two elements, specifying the lower and upper bounds for the constant values. Default is (-5, 5).
            initialization str: The method to use for initializing the constant values. Currently, only "random" and "mean" are supported. "random" creates a vector with random values
                                sampled within the bounds. "mean" creates a vector where all values are calculated as (lower_bound + upper_bound)/2. Default is "random".
            max_constants int: The maximum number of constants allowed in the expression. Default is 8.
            max_expr_length int: The maximum length of the expression. Default is -1 (no limit).
            num_points_sampled int: The number of points to sample when estimating the behavior of an expression. Default is 64.
            bed_X: Optional[np.ndarray]=None,
            num_consts_sampled: int=32,
            num_points_sampled: int=64,
            domain_bounds: Optional[List[Tuple[float, float]]]=None,

        Methods:
            evaluate_expr(expr): Evaluates an expression in infix notation and stores the result in memory to prevent re-evaluation.
            get_results(top_k): Returns the results of the evaluation.
        """

        self.models = dict()
        self.invalid = list()
        self.metadata = metadata
        self.ground_truth = ground_truth
        self.gt_behavior = None
        self.bed_evaluation_parameters = {
            "bed_X": None,
            "num_consts_sampled": 32,
            "num_points_sampled": 64,
            "domain_bounds": None,
            "constant_bounds": (-5, 5)
        }
        if kwargs:
            for k in self.bed_evaluation_parameters.keys():
                if k in kwargs:
                    self.bed_evaluation_parameters[k] = kwargs[k]

        self.symbol_library = symbol_library
        self.max_evaluations = max_evaluations
        self.total_expressions = 0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.parameter_estimator = ParameterEstimator(
            X, y, symbol_library=symbol_library, seed=seed, **kwargs)

        if ranking_function not in ["rmse", "bed"]:
            print(f"Warning: ranking_function {ranking_function} not supported. Using rmse instead.")
            ranking_function = "rmse"
        self.ranking_function = ranking_function

        if evaluation_measures is None:
            evaluation_measures = ["rmse"]
        self.evaluation_metrics = []
        for measure in evaluation_measures:
            if isinstance(measure, str):
                if measure not in ["rmse"]:
                    print(f"Warning: evaluation measure {measure} not supported. Ignoring.")
                else:
                    self.evaluation_metrics.append((measure, _MEASURE_DICT[measure]))
            elif isinstance(measure, tuple):
                self.evaluation_metrics.append(measure)

        if ranking_function == "bed":
            if ground_truth is None:
                raise ValueError("Ground truth must be provided for bed ranking function. The ground truth must be "
                                 "provided as a list of tokens, a Node object, or a numpy array representing behavior. "
                                 "The behavior matrix is a matrix representing the distribution of outputs of an "
                                 "expression with free parameters at different points in the domain. This matrix "
                                 "should be of size (num_points_sampled, num_consts_sampled). The behavior of an "
                                 "expressions without free parameters can be expressed as a matrix of size "
                                 "(num_points_sampled, 1) with values equal to the output of the expression at these points.")
            else:
                if self.bed_evaluation_parameters["bed_X"] is None:
                    if self.bed_evaluation_parameters["domain_bounds"] is not None:
                        db = self.bed_evaluation_parameters["domain_bounds"]
                        interval_length = np.array([ub - lb for (lb, ub) in db])
                        lower_bound = np.array([lb for (lb, ub) in db])
                        lho = LatinHypercube(len(db), optimization="random-cd", seed=seed)
                        self.bed_evaluation_parameters["bed_X"] = lho.random(self.bed_evaluation_parameters["num_points_sampled"]) * interval_length + lower_bound
                    else:
                        self.bed_evaluation_parameters["bed_X"] = np.random.choice(X, size=self.bed_evaluation_parameters["num_points_sampled"])
            if isinstance(ground_truth, (list, Node)):
                self.gt_behavior = create_behavior_matrix(ground_truth, self.bed_evaluation_parameters["bed_X"],
                                                          num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                                                          consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                                                          symbol_library=self.symbol_library, seed=self.seed)
            elif isinstance(ground_truth, np.ndarray):
                self.gt_behavior = ground_truth
            else:
                raise ValueError("Ground truth must be provided as a list of tokens, a Node object, or a numpy array representing behavior.")

    def evaluate_expr(self, expr: Union[List[str], Node], simplify_expr: bool = False, verbose: int=0) -> float:
        """
        Evaluates an expression in infix notation and stores the result in
        memory to prevent re-evaluation.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> X = np.array([[0, 1], [0, 2], [0, 3]])
            >>> y = np.array([2, 3, 4])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "+", "C" "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"], simplify_expr=True)
            >>> print(rmse < 1e-6)
            True
            >>> list(se.models.keys())[0]
            'C+X_1'
            >>> print(0.99 < se.models["C+X_1"]["parameters"][0] < 1.01)
            True

        Args:
            expr: An expression. This should be an istance of the SRToolkit.utils.expression_tree.Node class or a list
                  of tokens in the infix notation.
            simplify_expr: If True, simplifies the expression using SymPy before evaluating it.
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
        self.total_expressions += 1

        if 0 <= self.max_evaluations < self.total_expressions:
            warnings.warn(
                f"Maximum number of evaluations ({self.max_evaluations}) reached. Stopping evaluation.")
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
                    print(f"Unable to simplify: {''.join(expr_list)}, problems with subexpression {e}")

            if isinstance(expr, Node):
                expr_list = expr.to_list(symbol_library=self.symbol_library)
            else:
                expr_list = expr

            expr_str = "".join(expr_list)
            if expr_str in self.models:
                if verbose > 0:
                    print(f"Already evaluated {expr_str}")
                return self.models[expr_str]["error"]
            else:
                if self.ranking_function == "rmse":
                    if verbose < 2:
                        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                            error, parameters = self.parameter_estimator.estimate_parameters(expr)
                    else:
                        error, parameters = self.parameter_estimator.estimate_parameters(expr)

                    if verbose > 0:
                        if parameters.size > 0:
                            parameter_string = f" Best parameters found are [{', '.join([str(round(p, 3)) for p in parameters])}]"
                        else:
                            parameter_string = ""
                        print(f"Evaluated expression {expr_str} with RMSE: {error}." + parameter_string)
                    self.models[expr_str] = {
                        "error": error,
                        "parameters": parameters,
                        "expr": expr_list,
                    }
                elif self.ranking_function == "bed":
                    if verbose < 2:
                        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                            error = bed(expr, self.gt_behavior, self.bed_evaluation_parameters["bed_X"],
                                        num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                                        num_points_sampled=self.bed_evaluation_parameters["num_points_sampled"],
                                        domain_bounds=self.bed_evaluation_parameters["domain_bounds"],
                                        consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                                        symbol_library=self.symbol_library, seed=self.seed)
                    else:
                        error = bed(expr, self.gt_behavior, self.bed_evaluation_parameters["bed_X"],
                                    num_consts_sampled=self.bed_evaluation_parameters["num_consts_sampled"],
                                    num_points_sampled=self.bed_evaluation_parameters["num_points_sampled"],
                                    domain_bounds=self.bed_evaluation_parameters["domain_bounds"],
                                    consts_bounds=self.bed_evaluation_parameters["constant_bounds"],
                                    symbol_library=self.symbol_library, seed=self.seed)
                    if verbose > 0:
                        print(f"Evaluated expression {expr_str} with BED: {error}.")
                    self.models[expr_str] = {
                        "error": error,
                        "expr": expr_list,
                    }
                else:
                    raise ValueError(f"Ranking function {self.ranking_function} not supported.")
                return error

    # def evaluate_exprs(
    #     self, exprs: List[List[str]], num_processes: int = 1
    # ) -> List[float]:
    #     if num_processes > 1:
    #         pool = Pool(num_processes)
    #         results = pool.map(self.evaluate_expr, exprs)
    #         pool.close()
    #         for r in results:
    #             self.models
    #         return results
    #     else:
    #         return [self.evaluate_expr(expr) for expr in exprs]

    def get_results(self, top_k: int = 20, success_threshold: float = 1e-7) -> dict:
        """
        Returns the results of the equation discovery/symbolic regression process/evaluation.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> se = SR_evaluator(X, y)
            >>> rmse = se.evaluate_expr(["C", "*", "X_1", "-", "X_0"])
            >>> results = se.get_results(top_k=1)
            >>> print(results["num_evaluated"])
            1
            >>> print(results["total_expressions"])
            1
            >>> print(results["best_expr"])
            C*X_1-X_0
            >>> print(results["min_rmse"] < 1e-6)
            True
            >>> print(1.99 < results["results"][0]["parameters"][0] < 2.01)
            True

        Args:
            top_k: The number of top results to include in the output. If `top_k`
                is greater than the number of evaluated expressions, all
                evaluated expressions are included. If `top_k` is less than 0,
                all evaluated expressions are included.
            success_threshold: The threshold below which the evaluation is
                considered successful. Default is 1e-7.

        Returns:
            A dictionary containing the results of the equation discovery/symbolic regression process. The keys are:

                - "metadata" : The metadata provided in the constructor.
                - "min_rmse" : The minimum root mean squared error.
                - "best_expr" : The expression with the minimum root mean
                  squared error.
                - "num_evaluated" : The number of evaluated expressions.
                - "total_expressions" : The total number of expressions
                  considered.
                - "success" : Whether the evaluation was successful.
                - "results" : A list of dictionaries, where each dictionary
                  contains the root mean squared error, the expression, and the
                  estimated parameters of the expression. The list is sorted in
                  ascending order of the root mean squared error.
        """
        if top_k > len(self.models) or top_k < 0:
            top_k = len(self.models)

        models = list(self.models.values())
        best_indices = np.argsort([v["error"] for v in models])

        results = {
            "metadata": self.metadata,
            "min_rmse": models[best_indices[0]]["error"],
            "best_expr": "".join(models[best_indices[0]]["expr"]),
            "num_evaluated": len(models),
            "total_expressions": self.total_expressions,
            "results": list(),
        }

        # Determine success based on the predefined success threshold
        if success_threshold is not None and results["min_rmse"] < success_threshold:
            results["success"] = True
        else:
            results["success"] = False

        for i in best_indices[:top_k]:
            results["results"].append(models[i])

        return results
