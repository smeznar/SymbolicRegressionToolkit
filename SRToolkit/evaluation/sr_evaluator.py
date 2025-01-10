"""
This module contains the SR_evaluator class, which is used for evaluating symbolic regression approaches.
"""
from typing import Optional, List
import warnings

import numpy as np

from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator


class SR_evaluator:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_evaluations: int = -1,
        metadata: Optional[dict] = None,
        symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
        **kwargs
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

        Keyword Arguments:
            method str: The method to be used for minimization. Currently, only "L-BFGS-B" is supported/tested. Default is "L-BFGS-B".
            tol float: The tolerance for termination. Default is 1e-6.
            gtol float: The tolerance for the gradient norm. Default is 1e-3.
            max_iter int: The maximum number of iterations. Default is 100.
            bounds List[float]: A list of two elements, specifying the lower and upper bounds for the constant values. Default is [-5, 5].
            initialization str: The method to use for initializing the constant values. Currently, only "random" and "mean" are supported. "random" creates a vector with random values
                                sampled within the bounds. "mean" creates a vector where all values are calculated as (lower_bound + upper_bound)/2. Default is "random".
            max_constants int: The maximum number of constants allowed in the expression. Default is 8.
            max_expr_length int: The maximum length of the expression. Default is -1 (no limit).

        Methods:
            evaluate_expr(expr): Evaluates an expression in infix notation and stores the result in memory to prevent re-evaluation.
            get_results(top_k): Returns the results of the evaluation.
        """
        self.models = dict()
        self.metadata = metadata
        self.symbol_library = symbol_library
        self.max_evaluations = max_evaluations
        self.total_expressions = 0
        self.parameter_estimator = ParameterEstimator(
            X, y, symbol_library=symbol_library, **kwargs)

    def evaluate_expr(self, expr: List[str]) -> float:
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

        Args:
            expr: A list of strings representing the expression in infix notation.

        Returns:
            The root mean square error of the expression.

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
            expr_str = "".join(expr)
            if expr_str in self.models:
                # print(f"Already evaluated {expr_str}")
                # print(self.models[expr_str])
                return self.models[expr_str]["rmse"]
            else:
                rmse, parameters = self.parameter_estimator.estimate_parameters(expr)
                self.models[expr_str] = {
                    "rmse": rmse,
                    "parameters": parameters,
                    "expr": expr,
                }
                return rmse

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
        best_indices = np.argsort([v["rmse"] for v in models])

        results = {
            "metadata": self.metadata,
            "min_rmse": models[best_indices[0]]["rmse"],
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
