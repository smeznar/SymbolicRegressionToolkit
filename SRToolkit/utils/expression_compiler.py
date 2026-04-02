"""
Functions for compiling symbolic expressions into executable Python callables.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from SRToolkit.utils.expression_tree import Node, is_float, tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary


def expr_to_executable_function(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Compile an expression into an executable Python function.

    The returned callable evaluates the expression over a batch of inputs and a vector
    of constant values. To use a backend other than NumPy, set
    ``symbol_library.preamble`` to the required import statements.

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
        >>> tree = tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        >>> executable_fun = expr_to_executable_function(tree)
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])
        >>> # In case you need libraries other than numpy for the evaluation of your expressions,
        >>> # you can add them to the preamble in the SymbolLibrary. Here is how a preamble would look like:
        >>> symbol_library = SymbolLibrary.default_symbols(1)
        >>> symbol_library.preamble = ["import numpy as np"]
        >>> # Usually this is done when initializing the SymbolLibrary as SymbolLibrary(preamble=preamble)
        >>> executable_fun = expr_to_executable_function(tree, symbol_library)
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, preamble imports).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D array of shape
        ``(n_samples, n_features)`` and ``C`` is a 1-D array of constant values.
        Returns a 1-D output array of shape ``(n_samples,)``.

    Raises:
        Exception: If ``expr`` is neither a list nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if not (isinstance(expr, list) or isinstance(expr, Node)):
        raise Exception(
            "Expression must be given as either a list of tokens or a tree (an instance of the "
            "SRToolkit.utils.expression_tree.Node class)"
        )

    if isinstance(expr, list):
        tree = tokens_to_tree(expr, symbol_library)
    else:
        tree = expr
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "\n".join(symbol_library.preamble) + "\ndef _executable_expression_(X, C):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += "\treturn " + symbol

    fun_assignment_dict: Dict[str, Callable] = {}
    exec(fun_string, {"np": np}, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def expr_to_error_function(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Compile an expression into a callable that computes the RMSE against target values.

    To use a backend other than NumPy, set ``symbol_library.preamble`` to the required
    import statements.

    Examples:
        >>> executable_fun = expr_to_error_function(["X_0", "+", "1"])
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0
        >>> tree = tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        >>> executable_fun = expr_to_error_function(tree)
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0
        >>> # In case you need libraries other than numpy for the evaluation of your expressions,
        >>> # you can add them to the preamble in the SymbolLibrary. Here is how a preamble would look like:
        >>> symbol_library = SymbolLibrary.default_symbols(1)
        >>> symbol_library.preamble = ["import numpy as np"]
        >>> # Usually this is done when initializing the SymbolLibrary as SymbolLibrary(preamble=preamble)
        >>> executable_fun = expr_to_error_function(tree, symbol_library)
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, preamble imports).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C, y)`` where ``X`` is a 2-D array of shape
        ``(n_samples, n_features)``, ``C`` is a 1-D array of constant values, and ``y``
        is a 1-D target array. Returns the scalar RMSE as a float.

    Raises:
        Exception: If ``expr`` is neither a list nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if not (isinstance(expr, list) or isinstance(expr, Node)):
        raise Exception(
            "Expression must be given as either a list of tokens or a tree (an instance of the "
            "SRToolkit.utils.expression_tree.Node class)"
        )

    if isinstance(expr, list):
        tree = tokens_to_tree(expr, symbol_library)
    else:
        tree = expr
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree, symbol_library)

    fun_string = "\n".join(symbol_library.preamble) + "\ndef _executable_expression_(X, C, y):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))"

    fun_assignment_dict: Dict[str, Callable] = {}
    exec(fun_string, {"np": np}, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def tree_to_function_rec(
    tree: Node,
    symbol_library: SymbolLibrary,
    var_counter: int = 0,
    const_counter: int = 0,
) -> Tuple[List[str], str, int, int]:
    """
    Recursively convert a parse tree into lines of Python code for expression evaluation.

    This is a low-level helper for [expr_to_executable_function][SRToolkit.utils.expression_compiler.expr_to_executable_function] and
    [expr_to_error_function][SRToolkit.utils.expression_compiler.expr_to_error_function]. Call those functions directly unless you need
    fine-grained control over code generation.

    Args:
        tree: Root of the subtree to convert.
        symbol_library: Provides NumPy function strings for each token.
        var_counter: Running count of intermediate variables, used to generate unique
            names. Default ``0``.
        const_counter: Running count of constants consumed; used to index into the ``C``
            array. Default ``0``.

    Returns:
        A 4-tuple ``(code, symbol, var_counter, const_counter)`` where ``code`` is a list
        of Python assignment strings forming the expression body, ``symbol`` is the name
        of the variable holding this subtree's result, and ``var_counter`` /
        ``const_counter`` are the updated counters.

    Raises:
        Exception: If the tree contains a token that is neither a recognized symbol nor
            a numeric literal.
    """
    if tree.left is None and tree.right is None:
        if symbol_library.get_type(tree.symbol) in ["var", "lit"]:
            return [], symbol_library.get_np_fn(tree.symbol), var_counter, const_counter
        elif symbol_library.get_type(tree.symbol) == "const":
            return (
                [],
                symbol_library.get_np_fn(tree.symbol).format(const_counter),
                var_counter,
                const_counter + 1,
            )
        else:
            if is_float(tree.symbol):
                return [], tree.symbol, var_counter, const_counter
            else:
                raise Exception(f"Encountered invalid symbol {tree.symbol} while converting to function.")

    elif tree.left is not None and tree.right is None:
        code, symbol, var_counter, const_counter = tree_to_function_rec(
            tree.left, symbol_library, var_counter, const_counter
        )
        output_symbol = "y_{}".format(var_counter)
        code.append(symbol_library.get_np_fn(tree.symbol).format(output_symbol, symbol))
        return code, output_symbol, var_counter + 1, const_counter

    else:
        assert tree.right is not None, "Right child should be present in this branch."
        assert tree.left is not None, "Left child should be present if right child is present."
        left_code, left_symbol, var_counter, const_counter = tree_to_function_rec(
            tree.left, symbol_library, var_counter, const_counter
        )
        right_code, right_symbol, var_counter, const_counter = tree_to_function_rec(
            tree.right, symbol_library, var_counter, const_counter
        )
        output_symbol = "y_{}".format(var_counter)
        code = left_code + right_code
        code.append(symbol_library.get_np_fn(tree.symbol).format(output_symbol, left_symbol, right_symbol))
        return code, output_symbol, var_counter + 1, const_counter
