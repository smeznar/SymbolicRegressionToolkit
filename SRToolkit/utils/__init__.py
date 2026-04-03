"""
Utilities for expression representation, compilation, generation, and evaluation.

Modules:
    symbol_library: The [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] class — manages the token vocabulary and
        token properties.
    expression_tree: The [Node][SRToolkit.utils.expression_tree.Node] binary-tree representation and conversion utilities
        for expressions.
    expression_compiler: Compiles token-list or tree expressions into executable Python
        callables.
    expression_simplifier: SymPy-backed algebraic simplification, including constant folding.
    expression_generator: PCFG construction from a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] and Monte-Carlo
        expression sampling.
    measures: Distance and similarity measures: edit distance, tree edit distance,
        and Behavior-aware Expression Distance (BED).
    serialization: Internal JSON serialization utilities for numpy types.
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
from .serialization import _from_json_safe, _to_json_safe
from .symbol_library import SymbolLibrary
from .types import (
    CONST,
    FN,
    LIT,
    OP,
    VALID_SYMBOL_TYPES,
    VAR,
    EstimationSettings,
    EvalResult,
    ModelResult,
)

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
    "EstimationSettings",
    "VAR",
    "CONST",
    "FN",
    "OP",
    "LIT",
    "VALID_SYMBOL_TYPES",
    "EvalResult",
    "ModelResult",
    "_to_json_safe",
    "_from_json_safe",
]
