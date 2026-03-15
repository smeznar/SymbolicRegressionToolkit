"""
This module contains the ProGED approach - Probabilistic grammar-based equation discovery by Brence et. al.
"""
from typing import Any, Dict, Optional, Union

import numpy as np

from SRToolkit.approaches import SR_approach
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import generate_n_expressions, SymbolLibrary


class ProGED(SR_approach):
    r"""
    A slimmed-down version of the ProGED approach. You can find the full version of the approach at
    <https://github.com/brencej/ProGED> and the paper presenting the approach at
    <https://www.sciencedirect.com/science/article/pii/S0950705121003403>.

    The approach randomly samples expressions from a probabilistic grammar and evaluates them on the dataset.

    Examples:
        >>> from SRToolkit.dataset import SR_benchmark  # doctest: +SKIP
        >>> benchmark = SR_benchmark.feynman('../../data/feynman/')  # doctest: +SKIP
        >>> dataset = benchmark.create_dataset('I.16.6')  # doctest: +SKIP
        >>> dataset.max_evaluations = 100  # doctest: +SKIP
        >>> model = ProGED(dataset.symbol_library, verbose=False)  # doctest: +SKIP
        >>> results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=18, verbose=False)  # doctest: +SKIP
        >>> results.print_results(0)  # doctest: +SKIP
        Dataset: I.16.6
        Approach: ProGED
        Best expression found: X_0/C/C
        Error: 0.39676872335771524
        Number of evaluated expressions: 77
        Number of times evaluate_expr was called: 100
        Success: False
        <BLANKLINE>

    Args:
        grammar: The grammar to use for sampling expressions. Can be either a string or a SymbolLibrary object. Using
            a string let's you define a custom grammar.
        verbose: If True, prints the expression and its error if the expression is better than the current best.
    """
    def __init__(self, grammar: Union[str, SymbolLibrary], verbose: bool = False):
        super().__init__("ProGED")
        self.grammar = grammar
        self.verbose = verbose

    def prepare(self) -> None:
        """
        ProGED is stateless, so this method does nothing.
        """
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        """
        Samples expressions from the grammar using the Monte Carlo approach and evaluates them on the dataset.

        Args:
            sr_evaluator: The evaluator used for scoring expressions.
            seed: The seed used for random number generation.
        """
        np.random.seed(seed)
        min_error = float("inf")
        success = sr_evaluator.success_threshold is not None and min_error <= sr_evaluator.success_threshold
        budget_exhausted = 0 < sr_evaluator.max_evaluations <= sr_evaluator.total_evaluations
        while not success and not budget_exhausted:
            expr = generate_n_expressions(self.grammar, 1, verbose=False)[0]
            error = sr_evaluator.evaluate_expr(expr)
            if error < min_error:
                min_error = error
                if self.verbose:
                    print(f"New best expression {''.join(expr)} with error {min_error} "
                          f"after {sr_evaluator.total_evaluations} evaluations.")
            success = sr_evaluator.success_threshold is not None and min_error <= sr_evaluator.success_threshold
            budget_exhausted = 0 < sr_evaluator.max_evaluations <= sr_evaluator.total_evaluations

