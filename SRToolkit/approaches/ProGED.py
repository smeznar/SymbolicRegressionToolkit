"""
ProGED approach — probabilistic grammar-based equation discovery by Brence et al.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Grammar, SymbolLibrary, generate_n_expressions

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
    """

    name: str = "ProGED"
    grammar: Optional[dict] = None


class ProGED(SR_approach):
    def __init__(self, grammar: Optional[Union[str, Grammar, SymbolLibrary]] = None) -> None:
        r"""
        A slimmed-down version of ProGED — probabilistic grammar-based equation discovery.

        Randomly samples expressions from a probabilistic context-free grammar (PCFG).
        The grammar defines both which expressions are structurally valid (hard constraints via
        registered [Constraint][SRToolkit.utils.grammar.Constraint] objects) and a probability
        distribution over them (soft constraints via production rule weights), biasing sampling
        toward shorter or domain-appropriate expressions. Providing a domain-specific grammar is
        the main strength of this approach — it encodes prior knowledge about the expected form
        of the target expression and makes the search significantly more efficient. The full
        version of the approach is available at https://github.com/brencej/ProGED.

        References:
            [Brence et al. (2021)][cite-proged]

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> benchmark = Feynman()
            >>> dataset = benchmark.create_dataset('I.16.6')
            >>> dataset.max_evaluations = 100
            >>> model = ProGED()
            >>> # Since we don't put a custom grammar into ProGED we will need an automatically created PCFG.
            >>> model.adapt(dataset.X, dataset.symbol_library)
            >>> results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=18, verbose=False)
            >>> r = results[0]
            >>> r.dataset_name
            'I.16.6'
            >>> r.approach_name
            'ProGED'
            >>> r.best_expr
            'C-X_0/C-X_0'
            >>> r.num_evaluated
            63
            >>> bool(r.success)
            False

        Args:
            grammar: Grammar used for sampling. Its production rule weights define a probability
                distribution over expressions — higher-weight rules are sampled more often,
                acting as a soft prior. Registered constraints impose hard structural limits.
                Providing a carefully designed, domain-specific grammar is strongly recommended,
                as it is the primary way to incorporate prior knowledge and improve search
                efficiency. One of:

                - A [Grammar][SRToolkit.utils.grammar.Grammar] object (used directly,
                  including its weights and any registered constraints).
                - A grammar string in the notation used by
                  [Grammar.from_grammar_string][SRToolkit.utils.grammar.Grammar.from_grammar_string].
                - A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] — converted
                  automatically via
                  [Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library].
                - ``None`` — a generic grammar is derived automatically from the symbol library
                  via [adapt][SRToolkit.approaches.sr_approach.SR_approach.adapt], but this forgoes
                  the approach's main advantage.
        """
        if isinstance(grammar, Grammar):
            grammar_dict = grammar.to_dict()
        elif isinstance(grammar, SymbolLibrary):
            grammar = Grammar.from_symbol_library(grammar)
            grammar_dict = grammar.to_dict()
        elif isinstance(grammar, str):
            grammar = Grammar.from_grammar_string(grammar)
            grammar_dict = grammar.to_dict()
        else:
            grammar = None
            grammar_dict = None
        super().__init__(ProGEDConfig(grammar=grammar_dict))
        self.grammar: Optional[Grammar] = grammar

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
        if self.grammar is None:
            self.grammar = Grammar.from_symbol_library(symbol_library)

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
        grammar = config.get("grammar", None)
        if grammar is not None:
            return cls(grammar=Grammar.from_dict(grammar))
        return cls()
