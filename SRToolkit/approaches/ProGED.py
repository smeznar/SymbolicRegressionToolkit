"""ProGED approach — probabilistic grammar-based equation discovery by Brence et al."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Grammar, Node, SymbolLibrary, generate_n_expressions
from SRToolkit.utils.expression_constraints import ExpressionConstraint

from .sr_approach import ApproachConfig, Candidate, SR_approach

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
        canonicalise_constants: See [ProGED][SRToolkit.approaches.ProGED.ProGED].

    Examples:
        >>> cfg = ProGEDConfig()
        >>> cfg.name
        'ProGED'
        >>> d = cfg.to_dict()
    """

    name: str = "ProGED"
    grammar: dict | None = None
    max_consecutive_skips: int = 10000
    skip_evaluated: bool = True
    skip_rejected: bool = True
    canonicalise_constants: bool = False


class ProGED(SR_approach):
    def __init__(
        self,
        grammar: str | Grammar | SymbolLibrary | None = None,
        max_consecutive_skips: int = 10000,
        skip_evaluated: bool = True,
        skip_rejected: bool = True,
        canonicalise_constants: bool = False,
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
                [ask][SRToolkit.approaches.ProGED.ProGED.ask]) before the search stops early.
            skip_evaluated: Skip samples that were already evaluated. ``False`` evaluates repeats
                too, as cache hits that use the budget.
            skip_rejected: Skip samples that break a required constraint. ``False`` evaluates them,
                so they use the budget and appear in the results as ``"rejected"``.
            canonicalise_constants: Treat samples that differ only in how their constants are
                labelled (``C_0*X_0 + C_0`` and ``C_1*X_0 + C_1``) as one expression; passed to
                [evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] and
                ``get_model``. Useful only for grammars with labelled constants as terminals.

        Raises:
            ValueError: If ``max_consecutive_skips`` is less than 1.
        """
        if max_consecutive_skips < 1:
            raise ValueError(f"[ProGED] max_consecutive_skips must be at least 1, got {max_consecutive_skips}.")
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
                canonicalise_constants=canonicalise_constants,
            )
        )
        self.grammar: Grammar | None = grammar
        self.max_consecutive_skips = max_consecutive_skips
        self.skip_evaluated = skip_evaluated
        self.skip_rejected = skip_rejected
        self.canonicalise_constants = canonicalise_constants
        # The grammar search samples from: the constructor's, or the one adapt chose for this task.
        self._sampling_grammar: Grammar | None = grammar
        self._skipped = 0  # samples skipped in a row

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

    def start(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> None:
        """Seed NumPy's global random state, which the grammar samples from.

        ProGED implements the step interface (see [sr_approach][SRToolkit.approaches.sr_approach]):
        each step proposes one expression and evaluates it, so it can be resumed from a snapshot.

        Args:
            sr_evaluator: The evaluator of the run.
            seed: Optional random seed for reproducible sampling.

        Raises:
            RuntimeError: If no grammar was given to the constructor and ``adapt`` was not called.
        """
        if self._sampling_grammar is None:
            raise RuntimeError("ProGED.start() must be called after adapt() if grammar is not provided.")
        np.random.seed(seed)
        self._skipped = 0

    def ask(self, sr_evaluator: SR_evaluator) -> list[Candidate | list[str] | Node]:
        """Sample expressions from the grammar until one is not skipped, and propose it.

        A sample is skipped, without using the budget, if it was already evaluated
        ([get_model][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_model]) and
        ``skip_evaluated`` is set, or if it breaks a required constraint
        ([check_constraints][SRToolkit.evaluation.sr_evaluator.SR_evaluator.check_constraints]) and
        ``skip_rejected`` is set. After ``max_consecutive_skips`` skipped samples in a row, the grammar
        yields few new valid expressions: it [stops][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop]
        the evaluator with a reason starting with ``"ProGED"`` and returns no candidate.

        Args:
            sr_evaluator: The evaluator of the run.

        Returns:
            One candidate, or none after too many skipped samples or if the run is interrupted
            while sampling.
        """
        assert self._sampling_grammar is not None  # set by the constructor or adapt; checked in start
        while self._skipped < self.max_consecutive_skips:
            if sr_evaluator.should_stop:  # interrupted while sampling; the snapshot keeps the count
                return []
            expr = generate_n_expressions(self._sampling_grammar, 1, verbose=False)[0]
            evaluated = self.skip_evaluated and (
                sr_evaluator.get_model(expr, canonicalise_constants=self.canonicalise_constants) is not None
            )
            if not (evaluated or (self.skip_rejected and sr_evaluator.check_constraints(expr)[0])):
                self._skipped = 0
                return [Candidate(expr, canonicalise_constants=self.canonicalise_constants)]
            self._skipped += 1
        sr_evaluator.stop(
            f"ProGED: {self.max_consecutive_skips} samples in a row were already evaluated or broke a required "
            "constraint"
        )
        return []

    def get_search_state(self) -> bytes:
        """Return the sampling grammar and the number of samples skipped in a row (the search state
        besides NumPy's global random state, which the framework saves), as JSON bytes.
        """
        assert self._sampling_grammar is not None
        return json.dumps({"grammar": self._sampling_grammar.to_dict(), "skipped": self._skipped}).encode()

    def set_search_state(self, state: bytes) -> None:
        """Restore the state returned by ``get_search_state``.

        Args:
            state: The saved state.
        """
        saved = json.loads(state)
        self._sampling_grammar = Grammar.from_dict(saved["grammar"])
        self._skipped = saved["skipped"]

    @classmethod
    def from_config(cls, config: dict) -> "ProGED":
        grammar = config.get("grammar", None)
        return cls(
            grammar=None if grammar is None else Grammar.from_dict(grammar),
            max_consecutive_skips=config.get("max_consecutive_skips", 10000),
            skip_evaluated=config.get("skip_evaluated", True),
            skip_rejected=config.get("skip_rejected", True),
            canonicalise_constants=config.get("canonicalise_constants", False),
        )
