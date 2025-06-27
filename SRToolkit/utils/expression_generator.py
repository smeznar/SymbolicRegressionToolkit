"""
This module contains helper functions for creating a PCFG with generic probabilities from the SymbolLibrary and to use it for generating random expressions.
"""
import nltk
import numpy as np

from SRToolkit.utils import SymbolLibrary


def create_generic_pcfg(symbol_library: SymbolLibrary) -> str:
    """
    Creates a generic PCFG from the SymbolLibrary.

    Examples:
        >>> sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "sin", "^2", "pi"], 2)
        >>> print(create_generic_pcfg(sl))
        E -> E '+' F [0.2]
        E -> E '-' F [0.2]
        E -> F [0.6]
        F -> F '*' B [0.4]
        F -> B [0.6]
        B -> T [1.0]
        T -> R [0.2]
        T -> C [0.2]
        T -> V [0.6]
        C -> 'pi' [1.0]
        R -> 'sin' '(' E ')' [0.4]
        R -> P [0.15]
        R -> '(' E ')' [0.45]
        P -> '(' E ')' '^2' [1.0]
        V -> 'X_0' [0.5]
        V -> 'X_1' [0.5]
        <BLANKLINE>

    Args:
        symbol_library: The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns:
        A PCFG with generic probabilities, written as a string.
    """
    symbols = symbol_library.symbols.values()
    E = [s["symbol"] for s in symbols if s["type"]=="op" and s["precedence"]==0]
    F = [s["symbol"] for s in symbols if s["type"]=="op" and s["precedence"]==1]
    BP = [s["symbol"] for s in symbols if s["type"]=="op" and s["precedence"]==2]
    R = [s["symbol"] for s in symbols if s["type"]=="fn" and s["precedence"]==5]
    P = [s["symbol"] for s in symbols if s["type"]=="fn" and s["precedence"]==-1]
    V = [s["symbol"] for s in symbols if s["type"]=="var"]
    Cc = [s["symbol"] for s in symbols if s["type"]=="const"]
    Cl = [s["symbol"] for s in symbols if s["type"]=="lit"]

    grammar = ""
    if len(E) > 0:
        for s in E:
            grammar += f"E -> E '{s}' F [{0.4/len(E)}]\n"
        grammar += "E -> F [0.6]\n"
    else:
        grammar += "E -> F [1.0]\n"

    if len(F) > 0:
        for s in F:
            grammar += f"F -> F '{s}' B [{0.4/len(F)}]\n"
        grammar += "F -> B [0.6]\n"
    else:
        grammar += "F -> B [1.0]\n"

    if len(BP) > 0:
        for s in BP:
            grammar += f"B -> B '{s}' T [{0.05/len(BP)}]\n"
        grammar += "B -> T [0.95]\n"
    else:
        grammar += "B -> T [1.0]\n"

    if len(Cc) + len(Cl) > 0:
        grammar += "T -> R [0.2]\n"
        grammar += "T -> C [0.2]\n"
        grammar += "T -> V [0.6]\n"
        if len(Cl) > 0 and len(Cc) > 0:
            for s in Cl:
                grammar += f"C -> '{s}' [{0.2/len(Cl)}]\n"
            for s in Cc:
                grammar += f"C -> '{s}' [{0.8/len(Cc)}]\n"
        elif len(Cl) > 0:
            for s in Cl:
                grammar += f"C -> '{s}' [{1/len(Cl)}]\n"
        elif len(Cc) > 0:
            for s in Cc:
                grammar += f"C -> '{s}' [{1/len(Cc)}]\n"
    else:
        grammar += "T -> R [0.3]\n"
        grammar += "T -> V [0.7]\n"

    if len(R) > 0:
        for s in R:
            grammar += f"R -> '{s}' '(' E ')' [{0.4/len(R)}]\n"
        if len(P) > 0:
            grammar += "R -> P [0.15]\n"
            grammar += "R -> '(' E ')' [0.45]\n"
        else:
            grammar += "R -> '(' E ')' [0.6]\n"
    else:
        if len(P) > 0:
            grammar += "R -> P [0.15]\n"
            grammar += "R -> '(' E ')' [0.85]\n"
        else:
            grammar += "R -> '(' E ')' [1.0]\n"

    if len(P) > 0:
        total = sum([1/abs(float(s[1:])) for s in P])
        for s in P:
            grammar += f"P -> '(' E ')' '{s}' [{(1/abs(float(s[1:])))/total}]\n"

    if len(V) > 0:
        for s in V:
            grammar += f"V -> '{s}' [{1/len(V)}]\n"

    return grammar


def _expand(grammar, symbol, current_depth, max_depth=40):
    if current_depth > max_depth:
        return None

    if isinstance(symbol, nltk.grammar.Nonterminal):
        productions = grammar.productions(lhs=symbol)
        if not productions:
            return [str(symbol)]

        weights = [p.prob() for p in productions]
        chosen_production = np.random.choice(productions, p=weights)

        generated_sequence = []
        for rhs_symbol in chosen_production.rhs():
            expanded_part = _expand(grammar, rhs_symbol, current_depth + 1)
            if expanded_part is None:
                return None
            generated_sequence.extend(expanded_part)
        return generated_sequence
    else:
        return [str(symbol)]


def generate_from_pcfg(grammar_str: str, start_symbol="E", max_depth=40, limit=100):
    """
    Generates a random expression from a PCFG with monte-carlo sampling.

    Examples:
        >>> generate_from_pcfg("E -> '1' [1.0]")
        ['1']
        >>> grammar = create_generic_pcfg(SymbolLibrary.default_symbols())
        >>> len(generate_from_pcfg(grammar)) > 0
        True

    Args:
        symbol_library: The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Returns:
        A PCFG with generic probabilities, written as a string.
    """
    start_symbol = nltk.grammar.Nonterminal(start_symbol)
    grammar = nltk.PCFG.fromstring(grammar_str)
    expr = _expand(grammar, start_symbol, 0, max_depth)
    tries = 1
    while expr is None and tries < limit:
        expr = _expand(grammar, start_symbol, 0, max_depth)

    if expr is None:
        raise Exception(f"[Expression generation] Couldn't find an expression with max_depth {max_depth} from this grammar in {limit} tries.")

    return expr


if __name__ == '__main__':
    sl = SymbolLibrary.default_symbols(5)
    grammar = create_generic_pcfg(sl)
    print(generate_from_pcfg(grammar))
