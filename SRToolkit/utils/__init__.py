"""The module containing the `utils`.

The `utils` module provides a set of utilities used in the package and for expression compilation.

Modules:
    symbol_library: The module containing the symbol library data structure for managing symbols that can occur in expressions and their properties.
    expression_tree: The module containing the expression tree data structure and functions for transforming expressions into trees and back.
    expression_compiler: The module containing functions that transform expressions in the infix notation (represented as lists of tokens) to executable python functions.
    expression_simplifier: The module containing functions that simplify an expression using SymPy
    expression_generator: The module containing helper functions for generating expressions
    measures: The module containing functions for computing various performance measures on expressions
"""

from .expression_compiler import (
    expr_to_error_function,
    expr_to_executable_function,
    tree_to_function_rec,
)
from .expression_generator import (
    create_generic_pcfg,
    generate_from_pcfg,
    generate_n_expressions,
)
from .expression_simplifier import simplify
from .expression_tree import Node, expr_to_latex, is_float, tokens_to_tree
from .measures import bed, create_behavior_matrix, edit_distance, tree_edit_distance
from .symbol_library import SymbolLibrary

__all__ = [
    "SymbolLibrary",
    "Node",
    "tokens_to_tree",
    "is_float",
    "tree_to_function_rec",
    "expr_to_executable_function",
    "expr_to_error_function",
    "simplify",
    "generate_from_pcfg",
    "create_generic_pcfg",
    "generate_n_expressions",
    "bed",
    "create_behavior_matrix",
    "edit_distance",
    "tree_edit_distance",
    "expr_to_latex",
]
