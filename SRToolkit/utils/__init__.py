"""Utilities for expression representation, compilation, generation, and evaluation.

Modules:
    symbol_library: The [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] class — manages the token vocabulary and
        token properties.
    expression_tree: The [Node][SRToolkit.utils.expression_tree.Node] binary-tree representation and conversion utilities
        for expressions.
    expression_compiler: Compiles token-list or tree expressions into executable Python
        callables.
    expression_simplifier: SymPy-backed algebraic simplification, including constant folding.
    expression_constraints: Constraints on candidate expressions, checked before scoring (on the
        expression) or after (on the fitted model), and enforced by the evaluator (required) or
        recorded (advisory) —
        [ExpressionConstraint][SRToolkit.utils.expression_constraints.ExpressionConstraint] and
        the built-in constraints.
    expression_generator: PCFG construction from a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] and Monte-Carlo
        expression sampling.
    grammar: CFG/PCFG representation, constraint protocol, and stateful derivation —
        [Rule][SRToolkit.utils.grammar.grammar.Rule],
        [Grammar][SRToolkit.utils.grammar.grammar.Grammar],
        [GrammarConstraint][SRToolkit.utils.grammar.constraints.GrammarConstraint],
        [Derivation][SRToolkit.utils.grammar.derivation.Derivation], and more.
    measures: Distance and similarity measures: edit distance, tree edit distance,
        and Behavior-aware Expression Distance (BED).
    serialization: Which values can be saved in results and configs, and how they are stored as
        JSON.
"""

from .expression_compiler import compile_expr, compile_expr_rmse
from .expression_constraints import (
    ExpressionConstraint,
    MaxConstants,
    MaxSymbolCount,
    MaxTokens,
    MaxTreeDepth,
    MaxTreeSize,
    NoNesting,
    OutputBounds,
    RequiredVariables,
)
from .expression_generator import generate_n_expressions
from .expression_simplifier import simplify
from .expression_tree import ExpressionParseError, Node, expr_to_latex, is_float, tokens_to_tree
from .grammar import (
    AncestorInfo,
    Derivation,
    DimensionalConsistency,
    ExpansionContext,
    Grammar,
    GrammarConstraint,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    NoNested,
    ParseTree,
    ParseTreeNode,
    Rule,
)
from .measures import (
    REGRESSION_METRICS,
    PredictFn,
    bed,
    create_behavior_matrix,
    edit_distance,
    mae,
    mse,
    nrmse,
    r2,
    rmse,
    tree_edit_distance,
)
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
    STATUSES,
    VALID_SYMBOL_TYPES,
    VAR,
    CheckResult,
    EvalResult,
    ModelResult,
    RankingSettings,
    Status,
)

__all__ = [
    "Grammar",
    "Rule",
    "ParseTree",
    "ParseTreeNode",
    "Derivation",
    "GrammarConstraint",
    "ExpansionContext",
    "AncestorInfo",
    "MaxDepth",
    "MaxNodes",
    "MaxOccurrences",
    "NoNested",
    "DimensionalConsistency",
    "ExpressionConstraint",
    "MaxConstants",
    "MaxTokens",
    "MaxTreeDepth",
    "MaxTreeSize",
    "MaxSymbolCount",
    "NoNesting",
    "OutputBounds",
    "RequiredVariables",
    "CheckResult",
    "SymbolLibrary",
    "Node",
    "tokens_to_tree",
    "ExpressionParseError",
    "is_float",
    "compile_expr",
    "compile_expr_rmse",
    "simplify",
    "generate_n_expressions",
    "bed",
    "create_behavior_matrix",
    "edit_distance",
    "tree_edit_distance",
    "mse",
    "rmse",
    "mae",
    "r2",
    "nrmse",
    "REGRESSION_METRICS",
    "PredictFn",
    "expr_to_latex",
    "RankingSettings",
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
    "Status",
    "STATUSES",
]
