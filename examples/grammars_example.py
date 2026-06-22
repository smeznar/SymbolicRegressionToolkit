import json

from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.grammar import (
    Grammar,
    MaxDepth,
    MaxOccurrences,
    NoNested,
    Rule,
)

if __name__ == "__main__":
    # ------------------------------------------ Building a grammar --------------------------------------------
    # A grammar is a set of production rules. Each Rule expands a non-terminal (lhs) into a sequence of
    # terminals and non-terminals (rhs). A symbol is a non-terminal iff it appears on some rule's lhs. When all
    # rules for a non-terminal share weight 1.0 the grammar is a plain CFG; differing weights make it a PCFG.
    g = Grammar(
        [
            Rule("E", ["E", "+", "F"], weight=0.4, name="E_add"),
            Rule("E", ["F"], weight=0.6, name="E_to_F"),
            Rule("F", ["x"], name="F_x"),
            Rule("F", ["y"], name="F_y"),
        ],
        start="E",
    )
    print("Non-terminals:", g.nonterminals)
    print("Is PCFG:", g.is_pcfg())

    # generate_one samples the grammar to completion. It returns None if every attempt exceeds the step budget.
    for _ in range(5):
        print("  generated:", g.generate_one())

    # ----------------------------------- Driving a derivation manually ----------------------------------------
    # For full control, start a stateful Derivation: inspect the constraint-filtered options with options(),
    # then either apply() a rule you picked or sample() one by weight. Here we sample to completion -- picking
    # options()[0] every time would always take the recursive E_add rule and never terminate.
    d = g.start_derivation("E")
    while not d.complete:
        d.sample()  # apply one rule chosen by weight among the allowed options
    print("\nManual derivation tokens:", d.to_token_list())
    print("Productions used:", [r.name for r in d.to_parse_tree().productions_used()])

    # --------------------------------------------- Constraints ------------------------------------------------
    # Constraints are hard filters applied at every derivation step; a rule is offered only if every constraint
    # accepts it. They compose, and validate() never raises (returns False on any violation).
    gc = Grammar(
        [
            Rule("E", ["E", "+", "E"], name="E_add"),
            Rule("E", ["sin", "(", "E", ")"], name="E_sin"),
            Rule("E", ["x"], name="E_x"),
        ],
        start="E",
    )
    gc.add_constraint(MaxDepth(3))  # at most 3 levels of nesting
    gc.add_constraint(MaxOccurrences("sin", 1))  # at most one sin in the finished expression
    gc.add_constraint(NoNested("sin"))  # never sin(... sin ...)
    print("\nConstrained samples:")
    for _ in range(5):
        print("  ", gc.generate_one())

    # ------------------------------------- Grammar from a SymbolLibrary ---------------------------------------
    # from_symbol_library builds a PCFG automatically from a symbol library using a standard operator-precedence
    # hierarchy, with rule names (E_add_+, R_fn_sin, V_X_0, ...) the constraint system relies on.
    sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "sin", "^2"], num_variables=2)
    g_auto = Grammar.from_symbol_library(sl)
    print("\nFrom symbol library:")
    for _ in range(5):
        print("  ", g_auto.generate_one())

    # ------------------------------------------- Serialisation ------------------------------------------------
    # Grammars and all built-in constraints round-trip through to_dict/from_dict. Constraints are identified by
    # their fully-qualified class path and reconstructed via importlib, so custom constraints travel too.
    blob = gc.to_dict()
    with open("grammar.json", "w") as fh:
        json.dump(blob, fh, indent=2)
    with open("grammar.json") as fh:
        g_loaded = Grammar.from_dict(json.load(fh))
    print("\nReloaded grammar sample:", g_loaded.generate_one())
