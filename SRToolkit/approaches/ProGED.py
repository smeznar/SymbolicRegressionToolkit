"""
ProGED approach — probabilistic grammar-based equation discovery by Brence et al.
"""

from typing import Optional, Union

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary, generate_n_expressions

from .sr_approach import SR_approach


class ProGED(SR_approach):
    def __init__(self, grammar: Union[str, SymbolLibrary], verbose: bool = False) -> None:
        r"""
        A slimmed-down version of ProGED — probabilistic grammar-based equation discovery.

        Randomly samples expressions from a probabilistic context-free grammar (PCFG) and evaluates
        them using the provided evaluator. The full version of the approach is available at
        https://github.com/brencej/ProGED; see also Brence et al. (2021),
        https://doi.org/10.1016/j.knosys.2021.107077.

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman()
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> dataset.max_evaluations = 100
            >>> model = ProGED(dataset.symbol_library, verbose=False)
            >>> results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=18, verbose=False)
            >>> r = results[0]
            >>> r.dataset_name
            'I.16.6'
            >>> r.approach_name
            'ProGED'
            >>> r.best_expr
            'X_0/C/C'
            >>> r.num_evaluated
            77
            >>> bool(r.success)
            False

        Args:
            grammar: Grammar used for sampling. Either a
                [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] (grammar is derived
                automatically) or a custom grammar string.
            verbose: If ``True``, prints each new best expression and its error during search.
        """
        super().__init__("ProGED")
        self.grammar = grammar
        self.verbose = verbose

    def prepare(self) -> None:
        """
        ProGED is stateless — this method does nothing.

        Returns:
            None
        """
        pass

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """
        Randomly sample expressions from the grammar and evaluate them until the budget is exhausted
        or the success threshold is reached.

        Args:
            sr_evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] used to
                score candidate expressions.
            seed: Optional random seed for reproducible sampling.

        Returns:
            None
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
                    print(
                        f"New best expression {''.join(expr)} with error {min_error} "
                        f"after {sr_evaluator.total_evaluations} evaluations."
                    )
            success = sr_evaluator.success_threshold is not None and min_error <= sr_evaluator.success_threshold
            budget_exhausted = 0 < sr_evaluator.max_evaluations <= sr_evaluator.total_evaluations
