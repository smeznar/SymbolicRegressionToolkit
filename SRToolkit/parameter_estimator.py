from typing import Optional, List, Tuple

import numpy as np
from scipy.optimize import minimize


from .utils import Node, tokens_to_tree, is_float
from .symbol_library import SymbolLibrary


class ParameterEstimator:
    def __init__(self, X: np.ndarray, y: np.ndarray, estimation_settings: Optional[dict]=None, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
        """
        Initializes an instance of the ParameterEstimator class.

        Parameters
        ----------
        X : array
            The input data to be used in parameter estimation for variables. We assume that X is a 2D array
            with shape (n_samples, n_features).
        y : array
            The target values to be used in parameter estimation.
        estimation_settings : dict, optional
            A dictionary of settings for the parameter estimation process. The
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
        self.symbol_library = symbol_library
        self.X = X
        self.y = y
        # self.stats = {"success": 0, "failure": 0, "steps": dict(), "num_constants": dict(), "failed_constants": dict()}

        self.estimation_settings = {
                "method": "L-BFGS-B",
                "tol": 1e-6,
                "gtol": 1e-3,
                "maxiter": 100,
                "bounds": [-5, 5],
                "initialization": "random", # random, mean
                "max_constants": 8
        }

        if estimation_settings is not None:
            self.estimation_settings.update(estimation_settings)

    def estimate_parameters(self, expr: List[str]) -> Tuple[float, np.ndarray]:
        """
        Estimates the parameters of an expression by minimizing the error between the predicted and actual values.

        Parameters
        ----------
        expr : list[str]
            A list of strings representing the expression to be evaluated. The expression should include the
            symbol 'C' for constants whose values need to be estimated.

        Returns
        -------
        float
            The root mean square error (RMSE) of the optimized expression.
        np.ndarray
            An array containing the optimized constant values.

        Notes
        -----
        If the number of constants in the expression exceeds the maximum allowed, NaN and an empty array are returned.
        If there are no constants in the expression, the RMSE is calculated directly without optimization.
        """
        num_constants = sum([1 for t in expr if t == "C"])
        if 0 <= self.estimation_settings["max_constants"] < num_constants:
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
                           "maxiter": self.estimation_settings["maxiter"],
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


def expr_to_executable_function(expr: List[str], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> callable:
    """
    Converts an expression in infix notation to an executable function.

    Parameters
    ----------
    expr : list[str]
        The expression in infix notation.
    symbol_library : SymbolLibrary, optional
        The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns
    -------
    callable
        An executable function that takes in a 2D array of input values and a 1D array of constant values and returns
        the output of the expression.
    """
    tree = tokens_to_tree(expr, symbol_library)
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "def _executable_expression_(X, C):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += "\treturn " + symbol

    exec(fun_string)
    return locals()["_executable_expression_"]


def expr_to_error_function(expr: List[str], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> callable:
    """
    Converts an expression in infix notation to an executable function that returns the root mean squared error between
    the output of the expression and the target values.

    Parameters
    ----------
    expr : list[str]
        The expression in infix notation.
    symbol_library : SymbolLibrary, optional
        The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns
    -------
    callable
        An executable function that takes in a 2D array of input values `X`,
        a 1D array of constant values `C`, and a 1D array of target values `y`.
        It returns the root mean squared error between the output of the expression
        and the target values.
    """
    tree = tokens_to_tree(expr, symbol_library)
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "def _executable_expression_(X, C, y):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))" # TODO: Maybe add different error functions

    exec(fun_string)
    return locals()["_executable_expression_"]


def tree_to_function_rec(tree: Node, symbol_library: SymbolLibrary, var_counter: int=0, const_counter: int=0) -> Tuple[List[str], str, int, int]:
    """
    Recursively converts a parse tree into a string of Python code that can be executed to evaluate the expression
    represented by the tree.

    Parameters
    ----------
    tree : Node
        The root of the parse tree to convert.
    symbol_library : SymbolLibrary
        The symbol library to use when converting the tree. This library defines the properties of the symbols in the tree.
    var_counter : int
        The number of variables encountered so far. This is used to create a unique variable name for each variable.
    const_counter : int
        The number of constants encountered so far. This is used to select the correct constant value from the constant array.

    Returns
    -------
    list[str]
        A list of strings, where each string contains a line of Python code to execute to evaluate the expression represented by the tree.
    str
        The name of the variable that represents the output of the expression.
    int
        The updated value of `var_counter`.
    int
        The updated value of `const_counter`.
    """
    if tree.left is None and tree.right is None:
        if symbol_library.get_type(tree.symbol) in ["var", "lit"]:
            return [], symbol_library.get_np_fn(tree.symbol), var_counter, const_counter
        elif symbol_library.get_type(tree.symbol) == "const":
            return [], symbol_library.get_np_fn(tree.symbol).format(const_counter), var_counter, const_counter + 1
        else:
            if is_float(tree.symbol):
                return [], tree.symbol, var_counter, const_counter
            else:
                raise Exception(f"Error while parsing expression {tree.symbol}.")

    elif tree.left is not None and tree.right is None:
        code, symbol, var_counter, const_counter = tree_to_function_rec(tree.left, symbol_library, var_counter, const_counter)
        output_symbol = "y_{}".format(var_counter)
        code.append(symbol_library.get_np_fn(tree.symbol).format(output_symbol, symbol))
        return code, output_symbol, var_counter + 1, const_counter

    else:
        left_code, left_symbol, var_counter, const_counter = tree_to_function_rec(tree.left, symbol_library, var_counter, const_counter)
        right_code, right_symbol, var_counter, const_counter = tree_to_function_rec(tree.right, symbol_library, var_counter, const_counter)
        output_symbol = "y_{}".format(var_counter)
        code = left_code + right_code
        code.append(symbol_library.get_np_fn(tree.symbol).format(output_symbol, left_symbol, right_symbol))
        return code, output_symbol, var_counter + 1, const_counter