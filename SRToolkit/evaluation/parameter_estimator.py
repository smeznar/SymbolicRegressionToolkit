"""
This module contains the ParameterEstimator class, which is used to estimate the parameters of an expression.
"""
from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import minimize

from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.expression_compiler import expr_to_error_function


class ParameterEstimator:
    def __init__(self, X: np.ndarray, y: np.ndarray, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols(), **kwargs):
        """
        Initializes an instance of the ParameterEstimator class.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True

        Args:
            X: The input data to be used in parameter estimation for variables. We assume that X is a 2D array
                with shape (n_samples, n_features).
            y: The target values to be used in parameter estimation.
            symbol_library: The symbol library to use. Defaults to SymbolLibrary.default_symbols().

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
            estimate_parameters(expr: List[str]): Estimates the parameters of an expression by minimizing the error between the predicted and actual values.
        """
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        # self.stats = {"success": 0, "failure": 0, "steps": dict(), "num_constants": dict(), "failed_constants": dict()}

        self.estimation_settings = {
                "method": "L-BFGS-B",
                "tol": 1e-6,
                "gtol": 1e-3,
                "max_iter": 100,
                "bounds": [-5, 5],
                "initialization": "random", # random, mean
                "max_constants": 8,
                "max_expr_length": -1
        }

        if kwargs:
            self.estimation_settings.update(kwargs)

    def estimate_parameters(self, expr: List[str]) -> Tuple[float, np.ndarray]:
        """
        Estimates the parameters of an expression by minimizing the error between the predicted and actual values.

        Examples:
            >>> X = np.array([[1, 2], [8, 4], [5, 4], [7, 9], ])
            >>> y = np.array([3, 0, 3, 11])
            >>> pe = ParameterEstimator(X, y)
            >>> rmse, constants = pe.estimate_parameters(["C", "*", "X_1", "-", "X_0"])
            >>> print(rmse < 1e-6)
            True
            >>> print(1.99 < constants[0] < 2.01)
            True

        Args:
            expr: A list of strings representing the expression to be evaluated. The expression should include the
                  symbol 'C' for constants whose values need to be estimated.

        Returns:
            the root mean square error (RMSE) of the optimized expression.
            An array containing the optimized constant values.

        Notes:
            if the length of the expression exceeds the maximum allowed, NaN and an empty array are returned.
            If the number of constants in the expression exceeds the maximum allowed, NaN and an empty array are returned.
            If there are no constants in the expression, the RMSE is calculated directly without optimization.
        """
        num_constants = sum([1 for t in expr if t == "C"])
        if 0 <= self.estimation_settings["max_constants"] < num_constants:
            return np.nan, np.array([])

        if 0 <= self.estimation_settings["max_expr_length"] < len(expr):
            return np.nan, np.array([])

        executable_error_fn = expr_to_error_function(expr, self.symbol_library)

        if num_constants == 0:
            rmse = executable_error_fn(self.X, np.array([]), self.y)
            return rmse, np.array([])
        else:
            return self._optimize_parameters(executable_error_fn, num_constants)

    def _optimize_parameters(self, executable_error_fn: callable, num_constants: int) -> Tuple[float, np.ndarray]:
        """
        Optimizes the parameters of a given expression by minimizing the root mean squared error between the predicted and actual values.

        Parameters
        ----------
        executable_error_fn : callable
            A function that takes in the input values, the constant values, and the target values and returns the root mean squared error.
        num_constants : int
            The number of constants in the expression.

        Returns
        -------
        float
            The root mean square error of the optimized expression.
        np.ndarray
            An array containing the optimized constant values.
        """
        if self.estimation_settings["initialization"] == "random":
            x0 = np.random.rand(num_constants) * (self.estimation_settings["bounds"][1] - self.estimation_settings["bounds"][0]) + self.estimation_settings["bounds"][0]
        else:
            x0 = np.array([np.mean(self.estimation_settings["bounds"]) for _ in range(num_constants)])

        res = minimize(lambda c: executable_error_fn(self.X, c, self.y), x0, method=self.estimation_settings["method"],
                       tol=self.estimation_settings["tol"],
                       options={
                           "maxiter": self.estimation_settings["max_iter"],
                           "gtol": self.estimation_settings["gtol"]
                                },
                       bounds=[(self.estimation_settings["bounds"][0], self.estimation_settings["bounds"][1]) for _ in range(num_constants)])

        # if res.success:
        #     self.stats["success"] += 1
        # else:
        #     self.stats["failure"] += 1
        #     if num_constants in self.stats["failed_constants"]:
        #         self.stats["failed_constants"][num_constants] += 1
        #     else:
        #         self.stats["failed_constants"][num_constants] = 1
        #
        # if res.nit in self.stats["steps"]:
        #     self.stats["steps"][res.nit] += 1
        # else:
        #     self.stats["steps"][res.nit] = 1
        #
        # if num_constants in self.stats["num_constants"]:
        #     self.stats["num_constants"][num_constants] += 1
        # else:
        #     self.stats["num_constants"][num_constants] = 1

        return res.fun, res.x


