"""ProGED approach — probabilistic grammar-based equation discovery by Brence et al."""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Grammar, SymbolLibrary, generate_n_expressions
from SRToolkit.utils.expression_constraints import ExpressionConstraint

from .sr_approach import ApproachConfig, SR_approach

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports the approaches.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge


@dataclass
class ProGEDConfig(ApproachConfig):
    """Configuration dataclass for the [ProGED][SRToolkit.approaches.ProGED.ProGED] approach.

    Attributes:
        grammar: The constructor's grammar, saved with
            [Grammar.to_dict][SRToolkit.utils.grammar.grammar.Grammar.to_dict], or ``None``.
        max_consecutive_skips: See [ProGED][SRToolkit.approaches.ProGED.ProGED].
        skip_evaluated: See [ProGED][SRToolkit.approaches.ProGED.ProGED].
        skip_rejected: See [ProGED][SRToolkit.approaches.ProGED.ProGED].

    Examples:
        >>> cfg = ProGEDConfig()
        >>> cfg.name
        'ProGED'
        >>> d = cfg.to_dict()
    """

    name: str = "ProGED"
    grammar: Optional[dict] = None
    max_consecutive_skips: int = 10000
    skip_evaluated: bool = True
    skip_rejected: bool = True


class ProGED(SR_approach):
    def __init__(
        self,
        grammar: Optional[Union[str, Grammar, SymbolLibrary]] = None,
        max_consecutive_skips: int = 10000,
        skip_evaluated: bool = True,
        skip_rejected: bool = True,
    ) -> None:
        r"""A slimmed-down version of ProGED — probabilistic grammar-based equation discovery.

        Randomly samples expressions from a probabilistic context-free grammar (PCFG).
        The grammar defines both which expressions are structurally valid (hard constraints via
        registered [GrammarConstraint][SRToolkit.utils.grammar.GrammarConstraint] objects) and a probability
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
            >>> results = dataset.evaluate_approach(model, num_experiments=1, initial_seed=19, verbose=False)
            >>> r = results[0]
            >>> r.dataset_name
            'I.16.6'
            >>> r.approach_name
            'ProGED'
            >>> r.best_expr
            'C-X_0/C-X_0'
            >>> r.num_evaluated
            100
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
                - ``None`` — the grammar is chosen per task by
                  [adapt][SRToolkit.approaches.ProGED.ProGED.adapt] from the task's background
                  knowledge: its ``reference_grammar`` if set, else a generic grammar from the
                  symbol library. A generic grammar forgoes the approach's main advantage.
            max_consecutive_skips: Number of samples in a row that may be skipped (see
                [search][SRToolkit.approaches.ProGED.ProGED.search]) before the search stops early.
            skip_evaluated: Skip samples that were already evaluated. ``False`` evaluates repeats
                too, as cache hits that use the budget.
            skip_rejected: Skip samples that break a required constraint. ``False`` evaluates them,
                so they use the budget and appear in the results as ``"rejected"``.
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
        super().__init__(
            ProGEDConfig(
                grammar=grammar_dict,
                max_consecutive_skips=max_consecutive_skips,
                skip_evaluated=skip_evaluated,
                skip_rejected=skip_rejected,
            )
        )
        self.grammar: Optional[Grammar] = grammar
        self.max_consecutive_skips = max_consecutive_skips
        self.skip_evaluated = skip_evaluated
        self.skip_rejected = skip_rejected
        # The grammar search samples from: the constructor's, or the one adapt chose for this task.
        self._sampling_grammar: Optional[Grammar] = grammar

    def prepare(self) -> None:
        """ProGED is stateless — this method does nothing.

        Returns:
            None
        """
        pass

    @property
    def adaptation_scope(self) -> str:
        """``"never"`` if a grammar was given to the constructor, else ``"experiment"``.

        Without a grammar, [adapt][SRToolkit.approaches.ProGED.ProGED.adapt] chooses one for each
        task.

        Returns:
            ``"never"`` or ``"experiment"``.
        """
        return "experiment" if self.grammar is None else "never"

    def adapt(
        self,
        X: np.ndarray,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> None:
        """Choose the grammar to sample from, if none was given to the constructor.

        Uses [BackgroundKnowledge.sampling_grammar][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.sampling_grammar]:
        the reference grammar if it is set, otherwise a generic grammar for the symbol library.

        Args:
            X: Inputs; not used.
            background_knowledge: Source of the reference grammar or the symbol library.
            required_constraints: Not used.
        """
        if self.grammar is not None:
            return
        self._sampling_grammar = background_knowledge.sampling_grammar()

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """Sample expressions from the grammar and evaluate the new ones until the evaluator stops.

        A sample is skipped, without using the budget, if it was already evaluated
        ([get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model]) and
        ``skip_evaluated`` is set, or if it breaks a required constraint
        ([check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints]) and
        ``skip_rejected`` is set.
        After ``max_consecutive_skips`` skipped samples in a row, the search stops early with a
        warning: the grammar yields few new valid expressions.

        Args:
            sr_evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] used to
                score candidate expressions.
            seed: Optional random seed for reproducible sampling.

        Raises:
            RuntimeError: If no grammar was given to the constructor and ``adapt`` was not called.
        """
        if self._sampling_grammar is None:
            raise RuntimeError("ProGED.search() must be called after adapt() if grammar is not provided.")
        np.random.seed(seed)
        skipped = 0
        while not sr_evaluator.should_stop:
            expr = generate_n_expressions(self._sampling_grammar, 1, verbose=False)[0]
            if (self.skip_evaluated and sr_evaluator.get_model(expr) is not None) or (
                self.skip_rejected and sr_evaluator.check_constraints(expr)[0]
            ):
                skipped += 1
                if skipped >= self.max_consecutive_skips:
                    warnings.warn(
                        f"[ProGED] Stopping early: {skipped} samples in a row were already evaluated or broke a "
                        "required constraint, so the grammar yields few new valid expressions."
                    )
                    return
                continue
            skipped = 0
            sr_evaluator.evaluate_expr(expr)

    @classmethod
    def from_config(cls, config: dict) -> "ProGED":
        grammar = config.get("grammar", None)
        return cls(
            grammar=None if grammar is None else Grammar.from_dict(grammar),
            max_consecutive_skips=config.get("max_consecutive_skips", 10000),
            skip_evaluated=config.get("skip_evaluated", True),
            skip_rejected=config.get("skip_rejected", True),
        )
