import numpy as np
from scipy.optimize import minimize

from .utils import Node, tokens_to_tree, is_float
from .symbol_library import SymbolLibrary


class ParameterEstimator:
    def __init__(self, X, y, estimation_settings=None, symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
        self.symbol_library = symbol_library
        self.X = X
        self.y = y

        if estimation_settings is None:
            self.estimation_settings = {
                "method": "L-BFGS-B",
                "tol": 1e-6,
                "maxiter": 100,
                "bounds": [-5, 5],
                "initialization": "random" # random, mean
            }
        else:
            self.estimation_settings = estimation_settings

    def estimate_parameters(self, expr: list[str]):
        num_constants = sum([1 for t in expr if t == "C"])
        executable_error_fn = expr_to_error_function(expr, self.symbol_library)

        if num_constants == 0:
            rmse = executable_error_fn(self.X, self.y)
            return rmse, np.array([])
        else:
            return self._optimize_parameters(executable_error_fn, num_constants)

    def _optimize_parameters(self, executable_error_fn, num_constants: int):
        x0 = np.random.rand(num_constants) * (self.estimation_settings["bounds"][1] - self.estimation_settings["bounds"][0]) + self.estimation_settings["bounds"][0]
        res = minimize(lambda c: executable_error_fn(self.X, c, self.y), x0, method=self.estimation_settings["method"],
                       tol=self.estimation_settings["tol"], options={"maxiter": self.estimation_settings["maxiter"]},
                       bounds=[(self.estimation_settings["bounds"][0], self.estimation_settings["bounds"][1]) for _ in range(num_constants)])
        return res.fun, res.x


def expr_to_executable_function(expr: list[str], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
    tree = tokens_to_tree(expr, symbol_library)
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "def _executable_expression_(X, C):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += "\treturn " + symbol

    exec(fun_string)
    return locals()["_executable_expression_"]


def expr_to_error_function(expr: list[str], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()):
    tree = tokens_to_tree(expr, symbol_library)
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "def _executable_expression_(X, C, y):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))" # TODO: Maybe add different error functions

    exec(fun_string)
    return locals()["_executable_expression_"]


def tree_to_function_rec(tree: Node, symbol_library: SymbolLibrary, var_counter: int=0, const_counter: int=0) -> tuple[list[str], str, int, int]:
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