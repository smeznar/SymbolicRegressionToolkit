"""
ProGED approach — probabilistic grammar-based equation discovery by Brence et al.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import SymbolLibrary, generate_n_expressions

from .sr_approach import ApproachConfig, SR_approach


@dataclass
class ProGEDConfig(ApproachConfig):
    """
    Configuration dataclass for the [ProGED][SRToolkit.approaches.ProGED.ProGED] approach.

    Examples:
        >>> cfg = ProGEDConfig()
        >>> cfg.name
        'ProGED'
        >>> d = cfg.to_dict()
        >>> ProGEDConfig.from_dict(d).grammar
    """

    name: str = "ProGED"
    grammar: Optional[str] = None


class ProGED(SR_approach):
    def __init__(self, grammar: Optional[str] = None) -> None:
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
            >>> model = ProGED()
            >>> model.adapt(dataset.X, dataset.symbol_library) # Since we don't put a custom grammar into ProGED we will need an automatically created PCFG.
            >>> results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=18, verbose=False)
            >>> r = results[0]
            >>> r.dataset_name
            'I.16.6'
            >>> r.approach_name
            'ProGED'
            >>> r.best_expr
            'C*X_0'
            >>> r.num_evaluated
            74
            >>> bool(r.success)
            False

        Args:
            grammar: Grammar used for sampling. Either a
                [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] (grammar is derived
                automatically) or a custom grammar string.
        """
        grammar_str = grammar if isinstance(grammar, str) else None
        super().__init__(ProGEDConfig(grammar=grammar_str))
        self.grammar: Optional[Union[str, SymbolLibrary]] = grammar

    def prepare(self) -> None:
        """
        ProGED is stateless — this method does nothing.

        Returns:
            None
        """
        pass

    @property
    def adaptation_scope(self) -> str:
        return "experiment" if self.grammar is None else "never"

    def adapt(self, X: np.ndarray, symbol_library: SymbolLibrary) -> None:
        self.grammar = symbol_library

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
        while not sr_evaluator.should_stop:
            if self.grammar is not None:
                expr = generate_n_expressions(self.grammar, 1, verbose=False)[0]
                _ = sr_evaluator.evaluate_expr(expr)
            else:
                raise RuntimeError("ProGED.search() must be called after adapt() if grammar is not provided.")

    @classmethod
    def from_config(cls, config: dict) -> "ProGED":
        return cls(
            grammar=config.get("grammar", None),
        )
