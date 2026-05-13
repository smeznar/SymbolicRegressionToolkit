"""
CFG/PCFG grammar representation, constraint protocol, and stateful derivation engine.

Provides:
    [Rule][SRToolkit.utils.grammar.grammar.Rule],
    [ParseTreeNode][SRToolkit.utils.grammar.grammar.ParseTreeNode],
    [ParseTree][SRToolkit.utils.grammar.grammar.ParseTree],
    [Grammar][SRToolkit.utils.grammar.grammar.Grammar],
    [Constraint][SRToolkit.utils.grammar.constraints.Constraint],
    [AncestorInfo][SRToolkit.utils.grammar.constraints.AncestorInfo],
    [ExpansionContext][SRToolkit.utils.grammar.constraints.ExpansionContext],
    [MaxDepth][SRToolkit.utils.grammar.constraints.MaxDepth],
    [MaxNodes][SRToolkit.utils.grammar.constraints.MaxNodes],
    [MaxOccurrences][SRToolkit.utils.grammar.constraints.MaxOccurrences],
    [NoNested][SRToolkit.utils.grammar.constraints.NoNested],
    [DimensionalConsistency][SRToolkit.utils.grammar.constraints.DimensionalConsistency],
    [Derivation][SRToolkit.utils.grammar.derivation.Derivation].
"""

from .constraints import (
    AncestorInfo,
    Constraint,
    DimensionalConsistency,
    ExpansionContext,
    MaxDepth,
    MaxNodes,
    MaxOccurrences,
    NoNested,
)
from .derivation import Derivation
from .grammar import Grammar, ParseTree, ParseTreeNode, Rule

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
]
