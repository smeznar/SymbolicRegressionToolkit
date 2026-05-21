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
    grammar: CFG/PCFG representation, constraint protocol, and stateful derivation —
        [Rule][SRToolkit.utils.grammar.grammar.Rule],
        [Grammar][SRToolkit.utils.grammar.grammar.Grammar],
        [Constraint][SRToolkit.utils.grammar.constraints.Constraint],
        [Derivation][SRToolkit.utils.grammar.derivation.Derivation], and more.
    measures: Distance and similarity measures: edit distance, tree edit distance,
        and Behavior-aware Expression Distance (BED).
    serialization: Internal JSON serialization utilities for numpy types.
"""

from .expression_compiler import compile_expr, compile_expr_rmse
from .expression_generator import generate_n_expressions
from .expression_simplifier import simplify
from .expression_tree import Node, expr_to_latex, is_float, tokens_to_tree
from .grammar import (
    AncestorInfo,
    Constraint,
    Derivation,
    DimensionalConsistency,
    ExpansionContext,
    Grammar,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    NoNested,
    ParseTree,
    ParseTreeNode,
    Rule,
)
from .measures import bed, create_behavior_matrix, edit_distance, tree_edit_distance
from .symbol_library import SymbolLibrary
from .types import (
    CONST,
    FN,
    FN_POSTFIX,
    FN_PREFIX,
    LEAF,
    LIT,
    OP,
    OP_ADDITIVE,
    OP_MULTIPLICATIVE,
    OP_POWER,
    VALID_SYMBOL_TYPES,
    VAR,
    EstimationSettings,
    EvalResult,
    ModelResult,
)

__all__ = [
    "Grammar",
    "Rule",
    "ParseTree",
    "ParseTreeNode",
    "Derivation",
    "Constraint",
    "ExpansionContext",
    "AncestorInfo",
    "MaxDepth",
    "MaxNodes",
    "MaxOccurrences",
    "NoNested",
    "DimensionalConsistency",
    "SymbolLibrary",
    "Node",
    "tokens_to_tree",
    "is_float",
    "compile_expr",
    "compile_expr_rmse",
    "simplify",
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
    "FN_PREFIX",
    "FN_POSTFIX",
    "OP",
    "OP_ADDITIVE",
    "OP_MULTIPLICATIVE",
    "OP_POWER",
    "LIT",
    "LEAF",
    "VALID_SYMBOL_TYPES",
    "EvalResult",
    "ModelResult",
]
