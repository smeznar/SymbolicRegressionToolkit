"""
PCFG construction from a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] and Monte-Carlo sampling of symbolic
expressions.
"""

from typing import List, Union

import nltk
import numpy as np
from tqdm import tqdm

from .symbol_library import SymbolLibrary


def create_generic_pcfg(symbol_library: SymbolLibrary) -> str:
    """
    Construct a generic Probabilistic Context-Free Grammar (PCFG) from a symbol library.

    The grammar encodes standard mathematical operator precedence through a fixed
    non-terminal hierarchy:

    - ``E`` — additive level (precedence 0 operators)
    - ``F`` — multiplicative level (precedence 1 operators)
    - ``B`` — power level (precedence 2 operators)
    - ``T`` — terminal: function application (``R``), constant (``C``), or variable (``V``)
    - ``R`` — unary functions (precedence 5) and parenthesised sub-expressions
    - ``P`` — postfix functions (precedence -1, e.g. ``^2``)

    The returned string is in NLTK PCFG format and can be passed directly to
    [generate_from_pcfg][SRToolkit.utils.expression_generator.generate_from_pcfg] or [generate_n_expressions][SRToolkit.utils.expression_generator.generate_n_expressions].

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
        symbol_library: Symbol library defining the available tokens, their types,
            and precedences.

    Returns:
        NLTK-formatted PCFG string with generic probabilities.
    """
    symbols = symbol_library.symbols.values()
    E = [s["symbol"] for s in symbols if s["type"] == "op" and s["precedence"] == 0]
    F = [s["symbol"] for s in symbols if s["type"] == "op" and s["precedence"] == 1]
    BP = [s["symbol"] for s in symbols if s["type"] == "op" and s["precedence"] == 2]
    R = [s["symbol"] for s in symbols if s["type"] == "fn" and s["precedence"] == 5]
    P = [s["symbol"] for s in symbols if s["type"] == "fn" and s["precedence"] == -1]
    V = [s["symbol"] for s in symbols if s["type"] == "var"]
    Cc = [s["symbol"] for s in symbols if s["type"] == "const"]
    Cl = [s["symbol"] for s in symbols if s["type"] == "lit"]

    grammar = ""
    if len(E) > 0:
        for s in E:
            grammar += f"E -> E '{s}' F [{0.4 / len(E)}]\n"
        grammar += "E -> F [0.6]\n"
    else:
        grammar += "E -> F [1.0]\n"

    if len(F) > 0:
        for s in F:
            grammar += f"F -> F '{s}' B [{0.4 / len(F)}]\n"
        grammar += "F -> B [0.6]\n"
    else:
        grammar += "F -> B [1.0]\n"

    if len(BP) > 0:
        for s in BP:
            grammar += f"B -> B '{s}' T [{0.05 / len(BP)}]\n"
        grammar += "B -> T [0.95]\n"
    else:
        grammar += "B -> T [1.0]\n"

    if len(Cc) + len(Cl) > 0:
        grammar += "T -> R [0.2]\n"
        grammar += "T -> C [0.2]\n"
        grammar += "T -> V [0.6]\n"
        if len(Cl) > 0 and len(Cc) > 0:
            for s in Cl:
                grammar += f"C -> '{s}' [{0.2 / len(Cl)}]\n"
            for s in Cc:
                grammar += f"C -> '{s}' [{0.8 / len(Cc)}]\n"
        elif len(Cl) > 0:
            for s in Cl:
                grammar += f"C -> '{s}' [{1 / len(Cl)}]\n"
        elif len(Cc) > 0:
            for s in Cc:
                grammar += f"C -> '{s}' [{1 / len(Cc)}]\n"
    else:
        grammar += "T -> R [0.3]\n"
        grammar += "T -> V [0.7]\n"

    if len(R) > 0:
        for s in R:
            grammar += f"R -> '{s}' '(' E ')' [{0.4 / len(R)}]\n"
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
        total = sum([1 / abs(float(s[1:])) for s in P])
        for s in P:
            grammar += f"P -> '(' E ')' '{s}' [{(1 / abs(float(s[1:]))) / total}]\n"

    if len(V) > 0:
        for s in V:
            grammar += f"V -> '{s}' [{1 / len(V)}]\n"

    return grammar


def _expand(grammar, symbol, current_depth, max_depth=40):
    """
    Recursively expand a grammar symbol by weighted random production selection.

    Args:
        grammar: Parsed NLTK PCFG object.
        symbol: Current symbol to expand (terminal or non-terminal).
        current_depth: Current recursion depth.
        max_depth: Maximum allowed recursion depth. Values ≤ 0 allow unbounded depth.

    Returns:
        List of terminal token strings, or ``None`` if ``max_depth`` is exceeded.
    """
    if current_depth > max_depth > 0:
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


def generate_from_pcfg(grammar_str: str, start_symbol: str = "E", max_depth: int = 40, limit: int = 100) -> List[str]:
    """
    Sample a single expression from a PCFG by Monte-Carlo tree expansion.

    Examples:
        >>> generate_from_pcfg("E -> '1' [1.0]")
        ['1']
        >>> grammar = create_generic_pcfg(SymbolLibrary.default_symbols())
        >>> len(generate_from_pcfg(grammar)) > 0
        True

    Args:
        grammar_str: Grammar in NLTK PCFG notation.
        start_symbol: Non-terminal from which expansion begins. Default ``"E"``.
        max_depth: Maximum parse-tree depth. Values below ``0`` allow unbounded depth.
            Default ``40``.
        limit: Maximum number of sampling attempts before raising an exception.
            Default ``100``.

    Returns:
        A single expression as a list of string tokens in infix notation.

    Raises:
        Exception: If a valid expression cannot be produced within ``limit`` attempts.
    """
    start_symbol = nltk.grammar.Nonterminal(start_symbol)
    grammar = nltk.PCFG.fromstring(grammar_str)
    expr = _expand(grammar, start_symbol, 0, max_depth)
    tries = 1
    while expr is None and tries < limit:
        expr = _expand(grammar, start_symbol, 0, max_depth)

    if expr is None:
        raise Exception(
            f"[Expression generation] Couldn't find an expression with max_depth {max_depth} from this grammar in {limit} tries."
        )

    return expr


def generate_n_expressions(
    expression_description: Union[str, SymbolLibrary],
    num_expressions: int,
    unique: bool = True,
    max_expression_length: int = 50,
    verbose: bool = True,
) -> List[List[str]]:
    """
    Sample ``num_expressions`` expressions from a grammar or symbol library.

    Examples:
        >>> len(generate_n_expressions(SymbolLibrary.default_symbols(5), 100, verbose=False))
        100
        >>> generate_n_expressions(SymbolLibrary.from_symbol_list([], 1), 3, unique=False, verbose=False, max_expression_length=1)
        [['X_0'], ['X_0'], ['X_0']]

    Args:
        expression_description: Grammar source as a NLTK PCFG string or a
            [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] (a generic PCFG is built automatically via
            [create_generic_pcfg][SRToolkit.utils.expression_generator.create_generic_pcfg]).
        num_expressions: Number of expressions to generate.
        unique: If ``True``, every expression in the output is lexicographically distinct
            (though not necessarily semantically inequivalent). Default ``True``.
        max_expression_length: Maximum token count per expression. Values below ``0``
            allow unbounded length. Default ``50``.
        verbose: Display a progress bar. Default ``True``.

    Returns:
        List of expressions, each represented as a list of string tokens in infix notation.
    """
    if isinstance(expression_description, SymbolLibrary):
        grammar = create_generic_pcfg(expression_description)
    elif isinstance(expression_description, str):
        grammar = expression_description
    else:
        raise Exception(
            "Description of expressions must be either a grammar written as a string or an instance of SymbolLibrary."
        )

    expressions: List[List[str]] = []
    expression_strings = set()
    if verbose:
        pbar = tqdm(total=num_expressions)
    while len(expressions) < num_expressions:
        try:
            expr = generate_from_pcfg(grammar, max_depth=max_expression_length * 10)
        except Exception:
            print("Couldn't generate a valid expression in 100 tries")
            continue
        if len(expr) > max_expression_length > 0:
            continue

        expr_string = "".join(expr)
        if expr_string not in expression_strings or not unique:
            expressions.append(expr)
            expression_strings.add(expr_string)
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
    return expressions


if __name__ == "__main__":
    sl = SymbolLibrary.default_symbols(5)
    a = generate_n_expressions(sl, 1000, unique=False, max_expression_length=1)
    b = 0
    # grammar = create_generic_pcfg(sl)
    # print(generate_from_pcfg(grammar))
