"""ProGED approach — probabilistic grammar-based equation discovery by Brence et al."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Grammar, Node, SymbolLibrary
from SRToolkit.utils.expression_constraints import ExpressionConstraint

from .sr_approach import ApproachConfig, Candidate, SR_approach

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports the approaches.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge


@dataclass
class ProGEDConfig(ApproachConfig):
    """Configuration of [ProGED][SRToolkit.approaches.ProGED.ProGED].

    Attributes:
        grammar: The constructor's grammar as a
            [Grammar.to_dict][SRToolkit.utils.grammar.grammar.Grammar.to_dict] dict, or ``None``.
        skip_evaluated: See [ProGED][SRToolkit.approaches.ProGED.ProGED].
        skip_rejected: See [ProGED][SRToolkit.approaches.ProGED.ProGED].
        canonicalise_constants: See [ProGED][SRToolkit.approaches.ProGED.ProGED].
    """

    name: str = "ProGED"
    grammar: dict | None = None
    skip_evaluated: bool = True
    skip_rejected: bool = True
    canonicalise_constants: bool = False


class ProGED(SR_approach):
    def __init__(
        self,
        grammar: str | Grammar | SymbolLibrary | None = None,
        skip_evaluated: bool = True,
        skip_rejected: bool = True,
        canonicalise_constants: bool = False,
    ) -> None:
        r"""Random sampling from a probabilistic context-free grammar (a slimmed-down ProGED).

        The grammar's rule weights bias sampling toward likely expressions and its registered
        [GrammarConstraint][SRToolkit.utils.grammar.GrammarConstraint]s exclude invalid ones; a
        domain-specific grammar is how this approach uses prior knowledge. Full version:
        https://github.com/brencej/ProGED.

        Each step of the [step interface][SRToolkit.approaches.sr_approach] proposes one sample, and
        a stopped run can be resumed. Resuming needs the same grammar (the same constructor
        argument, or the same task for ``adapt``); a different one is not detected.

        References:
            [Brence et al. (2021)][cite-proged]

        Examples:
            >>> from SRToolkit.dataset import Feynman
            >>> dataset = Feynman().create_dataset('I.16.6')
            >>> dataset.max_evaluations = 100
            >>> results = dataset.evaluate_approach(ProGED(), num_experiments=1, initial_seed=19, verbose=False)
            >>> results[0].best_expr
            'C-X_0/C-X_0'

        Args:
            grammar: Grammar to sample from: a [Grammar][SRToolkit.utils.grammar.Grammar], a string
                for [Grammar.from_grammar_string][SRToolkit.utils.grammar.Grammar.from_grammar_string],
                a [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] (converted with
                [Grammar.from_symbol_library][SRToolkit.utils.grammar.Grammar.from_symbol_library]),
                or ``None`` to let [adapt][SRToolkit.approaches.ProGED.ProGED.adapt] choose one per
                task. Sample length is bounded only by the derivation limit of
                [ask][SRToolkit.approaches.ProGED.ProGED.ask]; a ``MaxTokens`` required constraint
                rejects long samples.
            skip_evaluated: Skip samples that were already evaluated. ``False`` evaluates repeats
                too, as cache hits that use the budget.
            skip_rejected: Skip samples that break a required constraint. ``False`` evaluates them,
                so they use the budget and appear in the results as ``"rejected"``.
            canonicalise_constants: Treat samples that differ only in constant labels
                (``C_0*X_0 + C_0`` and ``C_1*X_0 + C_1``) as one expression. Useful only for
                grammars with labelled constants as terminals.
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
                skip_evaluated=skip_evaluated,
                skip_rejected=skip_rejected,
                canonicalise_constants=canonicalise_constants,
            )
        )
        self.grammar: Grammar | None = grammar
        self.skip_evaluated = skip_evaluated
        self.skip_rejected = skip_rejected
        self.canonicalise_constants = canonicalise_constants
        # The grammar search samples from: the constructor's, or the one adapt chose for this task.
        self._sampling_grammar: Grammar | None = grammar

    @property
    def adaptation_scope(self) -> str:
        """``"never"`` if a grammar was given to the constructor, else ``"experiment"``."""
        return "experiment" if self.grammar is None else "never"

    def adapt(
        self,
        X: np.ndarray,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> None:
        """Choose the grammar for this task, unless the constructor was given one.

        Uses [BackgroundKnowledge.sampling_grammar][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.sampling_grammar]:
        the reference grammar if set, else a generic grammar for the symbol library. ``X`` and
        ``required_constraints`` are not used.
        """
        if self.grammar is not None:
            return
        self._sampling_grammar = background_knowledge.sampling_grammar()

    def start(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> None:
        """Seed NumPy's global random state, unless ``seed`` is ``None``.

        The same seed, grammar and evaluator settings give the same samples.

        Raises:
            RuntimeError: If no grammar was given to the constructor and ``adapt`` was not called.
        """
        if self._sampling_grammar is None:
            raise RuntimeError("ProGED.start() must be called after adapt() if grammar is not provided.")
        if seed is not None:
            np.random.seed(seed)

    def ask(self, sr_evaluator: SR_evaluator, state: None) -> list[Candidate | list[str] | Node | None]:
        """Propose one sample from the grammar.

        A derivation that needs more than 1000 rule applications, or in which no rule fits, fails
        and is proposed as ``None`` (the skip ``"not_expression"``), not redrawn. A sample is
        proposed with the skip reason ``"evaluated"`` if ``skip_evaluated`` is set and it was
        evaluated before, else ``"rejected"`` if ``skip_rejected`` is set and it breaks a required
        constraint.

        Returns:
            A one-element list: a [Candidate][SRToolkit.approaches.sr_approach.Candidate], or
            ``None`` if the derivation failed.
        """
        assert self._sampling_grammar is not None  # set by the constructor or adapt; checked in start
        expr = self._sampling_grammar.generate_one(max_retries=1)
        if expr is None:
            return [None]
        if (
            self.skip_evaluated
            and sr_evaluator.get_model(expr, canonicalise_constants=self.canonicalise_constants) is not None
        ):
            reason = "evaluated"
        elif self.skip_rejected and sr_evaluator.check_constraints(expr)[0]:
            reason = "rejected"
        else:
            return [Candidate(expr, canonicalise_constants=self.canonicalise_constants)]
        return [Candidate(expr, canonicalise_constants=self.canonicalise_constants, skip=reason)]

    @classmethod
    def from_config(cls, config: dict) -> "ProGED":
        """Rebuild the approach from a config dict; missing keys take the constructor defaults.

        Unknown keys (e.g. the removed ``max_consecutive_skips``) are ignored.

        Returns:
            The approach.
        """
        grammar = config.get("grammar", None)
        return cls(
            grammar=None if grammar is None else Grammar.from_dict(grammar),
            skip_evaluated=config.get("skip_evaluated", True),
            skip_rejected=config.get("skip_rejected", True),
            canonicalise_constants=config.get("canonicalise_constants", False),
        )
