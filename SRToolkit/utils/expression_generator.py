"""
Monte-Carlo sampling of symbolic expressions from a grammar or symbol library.
"""

import warnings
from typing import List, Union

from tqdm import tqdm

from .grammar import Grammar
from .symbol_library import SymbolLibrary

_INVALID_RATIO_THRESHOLD = 0.8
_MIN_ATTEMPTS_BEFORE_WARNING = 500


def generate_n_expressions(
    expression_description: Union[str, SymbolLibrary, Grammar],
    num_expressions: int,
    unique: bool = True,
    max_expression_length: int = 50,
    verbose: bool = True,
    max_consecutive_generation_failures: int = 100,
    max_consecutive_uniqueness_failures: int = 200,
    max_derivation_steps: int = 1000,
    start: str = "E",
) -> List[List[str]]:
    """
    Sample ``num_expressions`` expressions from a grammar or symbol library.

    Examples:
        >>> len(generate_n_expressions(SymbolLibrary.default_symbols(5), 100, verbose=False))
        100
        >>> generate_n_expressions(SymbolLibrary.from_symbol_list([], 1), 3, unique=False, verbose=False, max_expression_length=1)
        [['X_0'], ['X_0'], ['X_0']]

    Args:
        expression_description: Grammar source — one of:

            - A [Grammar][SRToolkit.utils.grammar.Grammar] object (used directly, including
              any registered constraints).
            - A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] (a generic PCFG
              is built automatically via
              [Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library]).
            - A grammar string in the NLTK notation (with an optional ``# start: nonterminal``
              line where nonterminal indicates the start symbol) used by
              [Grammar.from_grammar_string][SRToolkit.utils.grammar.Grammar.from_grammar_string].

        num_expressions: Number of expressions to generate.
        unique: If ``True``, every expression in the output is lexicographically distinct
            (though semantically equivalent expressions may still appear). Default ``True``.
        max_expression_length: Maximum token count per expression. Values ≤ ``0``
            allow unbounded length. Default ``50``.
        verbose: Display a progress bar showing total attempts, the ratio of invalid
            expressions (derivation failed or exceeded ``max_expression_length``), and —
            when ``unique=True`` — the ratio of duplicate expressions among valid ones
            and the total number of generation attempts.
        max_consecutive_generation_failures: Maximum number of consecutive attempts that
            produce an invalid expression (derivation failed *or* result exceeded
            ``max_expression_length``) before raising an exception. Resets on any valid expression.
            Default ``100``.
        max_consecutive_uniqueness_failures: Maximum number of consecutive valid expressions
            that are already in the output set before stopping early and returning what has
            been collected so far. Only relevant when ``unique=True``. Resets whenever a new
            unique expression is found. Default ``200``.
        max_derivation_steps: Maximum number of rule applications per single derivation
            attempt before it is abandoned. Increase for grammars with deep recursion that
            legitimately require many steps. Default ``1000``.
        start: Start non-terminal used when ``expression_description`` is a grammar
            string. Ignored for [Grammar][SRToolkit.utils.grammar.Grammar] and
            [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] inputs.
            Default ``"E"``.

    Returns:
        List of expressions, each represented as a list of string tokens in infix notation.
        May contain fewer than ``num_expressions`` entries if ``unique=True`` and the
        search space is exhausted before the target count is reached.

    Raises:
        Exception: If ``expression_description`` is not a
            [Grammar][SRToolkit.utils.grammar.Grammar],
            [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary], or ``str``.
        Exception: If generation fails ``max_consecutive_generation_failures`` times in a
            row, indicating the grammar or length constraint may be too restrictive.

    Warns:
        UserWarning: If more than 80 % of attempts produce an invalid expression (after at
            least ``500`` total attempts), suggesting the grammar
            or constraints are overly restrictive. Emitted at most once per call.
        UserWarning: If ``unique=True`` and no new unique expression is found after
            ``max_consecutive_uniqueness_failures`` consecutive valid attempts, indicating
            the search space may be exhausted. The expressions collected so far are returned.
    """
    if isinstance(expression_description, Grammar):
        grammar = expression_description
    elif isinstance(expression_description, SymbolLibrary):
        grammar = Grammar.from_symbol_library(expression_description)
    elif isinstance(expression_description, str):
        grammar = Grammar.from_grammar_string(expression_description, start=start)
    else:
        raise Exception("expression_description must be a Grammar, SymbolLibrary, or grammar string.")

    expressions: List[List[str]] = []
    expression_strings: set = set()

    total_attempts = 0
    total_invalid = 0
    total_duplicates = 0
    consecutive_generation_failures = 0
    consecutive_uniqueness_failures = 0
    ratio_warning_sent = False

    pbar = tqdm(total=num_expressions) if verbose else None

    def update_postfix() -> None:
        if pbar is None:
            return
        fail_pct = total_invalid / total_attempts if total_attempts > 0 else 0.0
        postfix: dict = {"attempts": total_attempts, "fail%": f"{fail_pct:.0%}"}
        if unique:
            total_valid = total_attempts - total_invalid
            dup_pct = total_duplicates / total_valid if total_valid > 0 else 0.0
            postfix["dup%"] = f"{dup_pct:.0%}"
        pbar.set_postfix(postfix)

    while len(expressions) < num_expressions:
        expr = grammar.generate_one(max_retries=1, max_steps=max_derivation_steps)
        total_attempts += 1

        if expr is None or (max_expression_length > 0 and len(expr) > max_expression_length):
            total_invalid += 1
            consecutive_generation_failures += 1

            if (
                not ratio_warning_sent
                and total_attempts >= _MIN_ATTEMPTS_BEFORE_WARNING
                and total_invalid / total_attempts > _INVALID_RATIO_THRESHOLD
            ):
                warnings.warn(
                    f"[Expression generation] {total_invalid / total_attempts:.0%} of "
                    f"{total_attempts} attempts produced an invalid expression (derivation "
                    "failed or exceeded max_expression_length). Consider relaxing grammar "
                    "constraints or increasing max_expression_length.",
                    stacklevel=2,
                )
                ratio_warning_sent = True

            if consecutive_generation_failures >= max_consecutive_generation_failures:
                if pbar is not None:
                    pbar.close()
                raise Exception(
                    f"[Expression generation] Failed to generate a valid expression "
                    f"{consecutive_generation_failures} times in a row "
                    f"({total_invalid} invalid out of {total_attempts} total attempts). "
                    "The grammar or length constraint may be too restrictive."
                )
            update_postfix()
            continue

        consecutive_generation_failures = 0

        expr_string = "".join(expr)
        if unique and expr_string in expression_strings:
            total_duplicates += 1
            consecutive_uniqueness_failures += 1
            if consecutive_uniqueness_failures >= max_consecutive_uniqueness_failures:
                warnings.warn(
                    f"[Expression generation] Failed to find a new unique expression "
                    f"{consecutive_uniqueness_failures} times in a row — stopping early "
                    f"with {len(expressions)} of {num_expressions} expressions collected. "
                    "The expression search space may be exhausted.",
                    stacklevel=2,
                )
                break
            update_postfix()
            continue

        consecutive_uniqueness_failures = 0
        expressions.append(expr)
        if unique:
            expression_strings.add(expr_string)
        if pbar is not None:
            pbar.update(1)
        update_postfix()

    if pbar is not None:
        pbar.close()
    return expressions
