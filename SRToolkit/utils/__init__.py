"""The module containing the `utils`.

The `utils` module provides a set of utilities used in the package and for expression compilation.

Modules:
    symbol_library: The module containing the symbol library data structure for managing symbols that can occur in expressions and their properties.
    expression_tree: The module containing the expression tree data structure and functions for transforming expressions into trees and back.
    expression_compiler: The module containing functions that transform expressions in the infix notation (represented as lists of tokens) to executable python functions.

"""

from .expression_tree import Node, tokens_to_tree, is_float
from .symbol_library import SymbolLibrary
from .expression_compiler import tree_to_function_rec, expr_to_executable_function, expr_to_error_function


__all__ = ["SymbolLibrary", "Node", "tokens_to_tree", "is_float", "tree_to_function_rec", "expr_to_executable_function",
           "expr_to_error_function"]