"""
This module contains the ProGED approach - Probabilistic grammar-based equation discovery.
"""
from typing import Optional, Union

from SRToolkit.approaches import SR_approach
from SRToolkit.dataset import SR_benchmark
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import generate_n_expressions, SymbolLibrary


class ProGED(SR_approach):
    """
    A slimmed down version of the ProGED approach. You can find the full version of the approach at
    <https://github.com/brencej/ProGED> and the paper presenting the approach at
    <https://www.sciencedirect.com/science/article/pii/S0950705121003403>.

    The approach randomly samples expressions from a probabilistic grammar and evaluates them on the dataset.

    Args:
        grammar: The grammar to use for sampling expressions. Can be either a string or a SymbolLibrary object. Using
            a string let's you define a custom grammar.
        verbose: If True, prints the expression and its error if the expression is better than the current best.
    """
    def __init__(self, grammar: Union[str, SymbolLibrary], verbose: bool = False):
        super().__init__("ProGED")
        self.grammar = grammar
        self.verbose = verbose

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        """
        Samples expressions from the grammar using the Monte Carlo approach and evaluates them on the dataset.

        Args:
            sr_evaluator: The evaluator used for scoring expressions.
            seed: The seed used for random number generation.
        """
        # TODO: Take care of seeding for the generate_n_expressions function
        min_error = float("inf")
        while sr_evaluator.total_evaluations < sr_evaluator.max_evaluations and min_error > sr_evaluator.success_threshold:
            expr = generate_n_expressions(self.grammar, 1, verbose=False)[0]
            error = sr_evaluator.evaluate_expr(expr)
            if error < min_error:
                min_error = error
                if self.verbose:
                    print(f"New best expression {''.join(expr)} with error {min_error} after {sr_evaluator.total_evaluations} evaluations.")
            min_error = min(min_error, error)

    def clone(self):
        """
        Clones the ProGED approach.

        Returns:
            The approach is stateless, so this method only returns the object itself.
        """
        return self

if __name__ == "__main__":
    benchmark = SR_benchmark.feynman('../../data/feynman/')
    dataset = benchmark.create_dataset('I.16.6')
    dataset.max_evaluations = 10000
    model = ProGED(dataset.symbol_library, verbose=True)
    results = dataset.evaluate_approach(model, num_experiments=3)
    results.print_results()
