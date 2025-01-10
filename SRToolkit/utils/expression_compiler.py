"""
This module contains functions that convert an expression in infix notation to an executable python function.
"""
from typing import List, Tuple

from SRToolkit.utils import Node, tokens_to_tree, is_float
from SRToolkit.utils.symbol_library import SymbolLibrary

# Generated functions are defined (through exec) here, so numpy needs to be imported
import numpy as np

def expr_to_executable_function(expr: List[str], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> callable:
    """
    Converts an expression in infix notation to an executable function.

    Examples:
        >>> executable_fun = expr_to_executable_function(["X_0", "+", "1"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])
        >>> executable_fun = expr_to_executable_function(["pi"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([1]))
        array([3.14159265, 3.14159265, 3.14159265, 3.14159265])
        >>> executable_fun = expr_to_executable_function(["C"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([1]))
        array([1, 1, 1, 1])

    Args:
        expr : The expression in infix notation.
        symbol_library : The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns:
        An executable function that takes in a 2D array of input values and a 1D array of constant values and returns the output of the expression.
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

    Examples:
        >>> executable_fun = expr_to_error_function(["X_0", "+", "1"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))
        0.0

    Args:
        expr : The expression in infix notation.
        symbol_library : The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns:
        An executable function that takes in a 2D array of input values `X`, a 1D array of constant values `C`, and a 1D array of target values `y`. It returns the root mean squared error between the output of the expression and the target values.
    """
    tree = tokens_to_tree(expr, symbol_library)
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "def _executable_expression_(X, C, y):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))"

    exec(fun_string)
    return locals()["_executable_expression_"]


def tree_to_function_rec(tree: Node, symbol_library: SymbolLibrary, var_counter: int=0, const_counter: int=0) -> Tuple[List[str], str, int, int]:
    """
    Recursively converts a parse tree into a string of Python code that can be executed to evaluate the expression
    represented by the tree.

    Args:
        tree: The root of the parse tree to convert.
        symbol_library: The symbol library to use when converting the tree. This library defines the properties of the symbols in the tree.
        var_counter: The number of variables encountered so far. This is used to create a unique variable name for each variable.
        const_counter: The number of constants encountered so far. This is used to select the correct constant value from the constant array.

    Returns:
        A list of strings, where each string contains a line of Python code to execute to evaluate the expression represented by the tree.
        The name of the variable that represents the output of the expression.
        The updated value of `var_counter`.
        The updated value of `const_counter`.

    Raises:
        Exception: If the parse tree contains an invalid symbol.

    Notes:
        This function is a helper function for `expr_to_executable_function` and similar and should not be called directly
        unless you want to customize the way the expression is defined. For examples, see the code of `expr_to_executable_function` and `expr_to_error_function` in this module.


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
                raise Exception(f"Encountered invalid symbol {tree.symbol} while converting to function.")

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