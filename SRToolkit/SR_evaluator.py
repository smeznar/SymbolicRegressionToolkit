from typing import Optional, List

import numpy as np

from .symbol_library import SymbolLibrary
from .parameter_estimator import ParameterEstimator


class SR_evaluator:
    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: Optional[dict]=None,
                 estimation_settings: Optional[dict]= None, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
        """
        Initializes an instance of the SR_evaluator class. This class is used for evaluating symbolic regression approaches.

        Parameters
        ----------
        X : np.ndarray
            The input data to be used in parameter estimation for variables. We assume that X is a 2D array
            with shape (n_samples, n_features).
        y : np.ndarray
            The target values to be used in parameter estimation.
        metadata : Optional[dict]
            An optional dictionary containing metadata about the expressions to be evaluated.
        estimation_settings : Optional[dict]
            An optional dictionary of settings for the parameter estimation process. The
            following settings are available:
                - method : str
                    The method to be used for minimization. Currently, only
                    "L-BFGS-B" is supported/tested. Default is "L-BFGS-B".
                - tol : float
                    The tolerance for termination. Default is 1e-6.
                - gtol : float
                    The tolerance for the gradient norm. Default is 1e-3.
                - maxiter : int
                    The maximum number of iterations. Default is 100.
                - bounds : list
                    A list of two elements, specifying the lower and upper bounds
                    for the constant values. Default is [-5, 5].
                - initialization : str
                    The method to use for initializing the constant values.
                    Currently, only "random" and "mean" are supported. "random" creates a vector with random values
                    sampled within the bounds. "mean" creates a vector where all values are calculated as
                    (lower_bound + upper_bound)/2. Default is "random".
                - max_constants : int
                    The maximum number of constants allowed in an expression.
                    Default is 8.
        symbol_library : SymbolLibrary, optional
            An instance of SymbolLibrary, specifying the symbols and their
            properties to be used for parameter estimation. Default is
            SymbolLibrary.default_symbols().
        """
        self.models = {}
        self.metadata = metadata
        self.estimation_settings = estimation_settings
        self.symbol_library = symbol_library
        self.total_expressions = 0
        self.parameter_estimator = ParameterEstimator(X, y, estimation_settings=estimation_settings, symbol_library=symbol_library)

    def evaluate_expr(self, expr: List[str]) -> float:
        """
        Evaluates an expression in infix notation and stores the result in
        memory to prevent re-evaluation.

        Parameters
        ----------
        expr : list[str]
            The expression in infix notation.

        Returns
        -------
        float
            The root mean square error of the expression.

        Notes
        -----
        If the expression has already been evaluated, its stored value is
        returned instead of re-evaluating the expression.
        """
        self.total_expressions += 1

        expr_str = ''.join(expr)
        if expr_str in self.models:
            # print(f"Already evaluated {expr_str}")
            # print(self.models[expr_str])
            return self.models[expr_str]["rmse"]
        else:
            rmse, parameters = self.parameter_estimator.estimate_parameters(expr)
            self.models[expr_str] = {"rmse": rmse, "parameters": parameters, "expr": expr}
            return rmse

    def get_results(self, top_k: int=20)-> dict:
        # print(self.parameter_estimator.stats)

        """
        Returns the results of the equation discovery/symbolic regression process.

        Parameters
        ----------
        top_k : int
            The number of top results to include in the output. If `top_k` is
            greater than the number of evaluated expressions, all evaluated
            expressions are included. If `top_k` is less than 0, all evaluated
            expressions are included. Default is 20.

        Returns
        -------
        dict
            A dictionary containing the results of the symbolic regression
            process. The keys are:

                - "metadata" : The metadata provided in the constructor.
                - "min_rmse" : The minimum root mean squared error.
                - "best_expr" : The expression with the minimum root mean
                  squared error.
                - "num_evaluated" : The number of evaluated expressions.
                - "total_expressions" : The total number of expressions
                  considered.
                - "results" : A list of dictionaries, where each dictionary
                  contains the root mean squared error, the expression, and the
                  estimated parameters of the expression. The list is sorted in
                  ascending order of the root mean squared error.
        """
        if top_k > len(self.models) or top_k < 0:
            top_k = len(self.models)

        models = list(self.models.values())
        best_indices = np.argsort([v["rmse"] for v in models])

        results = {"metadata": self.metadata,
                   "min_rmse": models[best_indices[0]]["rmse"],
                   "best_expr": "".join(models[best_indices[0]]["expr"]),
                   "num_evaluated": len(models),
                   "total_expressions": self.total_expressions,
                   "results": list()}

        for i in best_indices[:top_k]:
            results["results"].append(models[i])

        return results