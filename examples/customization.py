import numpy as np

from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.expression_compiler import expr_to_executable_function

if __name__ == "__main__":
    # ------------------------------------------ SymbolLibrary -------------------------------------------------
    # One can create a custom symbol library by adding symbols manually. Such instance of SymbolLibrary can be used
    # in the same way as the default one by providing it as function argument.
    custom_symbol_library = SymbolLibrary()
    custom_symbol_library.add_symbol(
        "sin", symbol_type="fn", precedence=5, np_fn="{} = np.sin({})"
    )
    custom_symbol_library.add_symbol(
        "cos", symbol_type="fn", precedence=5, np_fn="{} = np.cos({})"
    )
    custom_symbol_library.add_symbol(
        "+", symbol_type="op", precedence=0, np_fn="{} = {} + {}"
    )
    custom_symbol_library.add_symbol(
        "C", symbol_type="const", precedence=5, np_fn="C[{}]"
    )
    custom_symbol_library.add_symbol("X", "var", 5, "X")

    # Additionally, one can create a custom symbol library by using the default_symbols function. For example,
    # we can create the default library with a different set of variable names.
    custom_symbol_library = SymbolLibrary.default_symbols(num_variables=0)
    custom_symbol_library.add_symbol("X1", "var", 5, "X[:, 0]")
    custom_symbol_library.add_symbol("X2", "var", 5, "X[:, 1]")

    # ---------------------------- List of symbols (tokens) to callable python function --------------------------
    # First, lets select an expression that we want to evaluate.
    expr = "( X1 + X2 ) ^2".split(" ")
    print("Tokenized expression: ", expr)
    # This creates a list of tokens ['(', 'X1', '+', 'X2', ')', '^2']

    # Next, we need to map each token to a callable python function.
    executable_function = expr_to_executable_function(expr, custom_symbol_library)

    # Finally, we can evaluate the expression. An executable function created using expr_to_executable_function
    # accepts a 2D array of input values and a 1D array of constant values.
    X = np.array([[1, 2], [3, 4]])
    print(
        "".join(expr) + " evaluated at points x1=[1, 3], x2=[2, 4]: ",
        executable_function(X, np.array([])),
    )
    # This should print [9, 49]

    # If we have an expression that contains constants, we need to pass them in as a 1D array with the values.
    # Even if the expression does not contain all the variables, we still need to pass in the whole 2D array.
    expr = "( X1 + C ) ^2 - C".split(" ")
    executable_function = expr_to_executable_function(expr, custom_symbol_library)
    print(
        "".join(expr) + " evaluated at points x1=[1, 3], x2=[2, 4], C=[3, 1]: ",
        executable_function(X, np.array([3, 1])),
    )
    # This should print [15, 35]
