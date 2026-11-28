"""EDHiE approach — equation discovery with hierarchical variational autoencoders by Mežnar et al.

Requires the optional ``approaches`` extra (PyTorch and pymoo). Install with::

    pip install 'symbolic-regression-toolkit[approaches]'
"""

import hashlib
import pickle
import random
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Expression, Node, SymbolLibrary, generate_n_expressions, tokens_to_tree
from SRToolkit.utils.expression_constraints import ExpressionConstraint

from .sr_approach import ApproachConfig, Candidate, SR_approach, check_dependencies

if TYPE_CHECKING:
    # Imported for annotations only: SRToolkit.dataset imports the approaches.
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
    from SRToolkit.utils.types import ModelResult

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.population import Population
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.core.termination import NoTermination
except ImportError:
    raise ImportError(
        "EDHiE requires pymoo. Install dependencies either manually or with "
        "the command: pip install 'symbolic-regression-toolkit[approaches]'"
    )

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import Dataset, Sampler
except ImportError:
    raise ImportError(
        "EDHiE requires PyTorch. Install dependencies either manually or with "
        "the command: pip install 'symbolic-regression-toolkit[approaches]'"
    )


@dataclass
class EDHiEConfig(ApproachConfig):
    """Configuration of [EDHiE][SRToolkit.approaches.EDHiE.EDHiE]; the fields are its constructor arguments."""

    name: str = "EDHiE"
    latent_size: int = 32
    num_expressions: int = 20000
    max_expression_length: int = 30
    only_unique_expressions: bool = True
    epochs: int = 20
    batch_size: int = 32
    max_beta: float = 0.035
    population_size: int = 40
    weights_path: str | None = None
    verbose: bool = True
    max_training_skips: int = 10000
    skip_evaluated: bool = True
    skip_rejected: bool = True
    training_seed: int | None = None


@dataclass
class _GASearch:
    """EDHiE's search state: the GA and, between ``ask`` and ``tell``, the asked generation."""

    ga: GA
    pending: Population | None = None


class EDHiE(SR_approach):
    def __init__(
        self,
        latent_size: int = 32,
        num_expressions: int = 20000,
        max_expression_length: int = 30,
        only_unique_expressions: bool = True,
        epochs: int = 20,
        batch_size: int = 32,
        max_beta: float = 0.035,
        population_size: int = 40,
        weights_path: str | None = None,
        verbose: bool = True,
        max_training_skips: int = 10000,
        skip_evaluated: bool = True,
        skip_rejected: bool = True,
        training_seed: int | None = None,
    ) -> None:
        r"""EDHiE: equation discovery with hierarchical variational autoencoders by Mežnar et al.

        [adapt][SRToolkit.approaches.EDHiE.EDHiE.adapt] trains a hierarchical VAE (HVAE) on random
        expressions for the task; the search runs a genetic algorithm in its latent space. In the
        [step interface][SRToolkit.approaches.sr_approach], one step is one generation and is never
        empty, so the search runs until the evaluator stops it.

        A stopped run resumes only with the same HVAE, bit for bit: load it from a file
        (``weights_path`` or a saved adapted state), or train it again with the same
        ``training_seed`` on the same device and PyTorch version. Other settings are not checked.

        Args:
            latent_size: Dimension of the HVAE latent space.
            num_expressions: Number of training expressions.
            max_expression_length: Longest training expression, in tokens.
            only_unique_expressions: Train on distinct expressions while the grammar yields new ones
                (see [training_expressions][SRToolkit.approaches.EDHiE.EDHiE.training_expressions]).
                ``False`` keeps repeats, so likely expressions weigh more.
            epochs: Number of HVAE training epochs.
            batch_size: HVAE training batch size.
            max_beta: Largest KL weight of the annealing schedule (regularisation strength).
            population_size: GA population size. A resumed run keeps the snapshot's.
            weights_path: File written by [save_adapted_state][SRToolkit.approaches.EDHiE.EDHiE.save_adapted_state];
                the HVAE is loaded from it here and used for every task.
            verbose: Print the training loss and reconstructions while training (the search prints nothing).
            max_training_skips: Limit for sampling the training expressions (see
                ``training_expressions``). It does not limit the search; the setting
                ``evaluator.max_consecutive_skips`` does.
            skip_evaluated: Skip expressions evaluated before (see [ask][SRToolkit.approaches.EDHiE.EDHiE.ask]).
                ``False`` evaluates them again, as cache hits that use the budget.
            skip_rejected: Skip expressions that break a required constraint. ``False`` evaluates
                them, so they use the budget and appear in the results as ``"rejected"``.
            training_seed: Seed for ``adapt``, or ``None`` for unseeded.

        Raises:
            ValueError: If ``max_training_skips`` is below 1.

        Examples:
            >>> model = EDHiE(latent_size=24, num_expressions=100, epochs=1, verbose=False)
            >>> model.name
            'EDHiE'

        References:
            [Mežnar et al. (2023)][cite-edhie]
        """
        super().__init__(
            EDHiEConfig(
                latent_size=latent_size,
                num_expressions=num_expressions,
                max_expression_length=max_expression_length,
                only_unique_expressions=only_unique_expressions,
                epochs=epochs,
                batch_size=batch_size,
                max_beta=max_beta,
                population_size=population_size,
                weights_path=weights_path,
                verbose=verbose,
                max_training_skips=max_training_skips,
                skip_evaluated=skip_evaluated,
                skip_rejected=skip_rejected,
                training_seed=training_seed,
            )
        )
        if max_training_skips < 1:
            raise ValueError(f"[EDHiE] max_training_skips must be at least 1, got {max_training_skips}.")
        check_dependencies(["pytorch", "pymoo"])
        self.latent_size = latent_size
        self.num_expressions = num_expressions
        self.max_expression_length = max_expression_length
        self.only_unique_expressions = only_unique_expressions
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_beta = max_beta
        self.population_size = population_size
        self.verbose = verbose
        self.max_training_skips = max_training_skips
        self.skip_evaluated = skip_evaluated
        self.skip_rejected = skip_rejected
        self.training_seed = training_seed

        self.model: HVAE | None = None
        self._pretrained = weights_path is not None
        if weights_path is not None:
            self.load_adapted_state(weights_path)

    @property
    def adaptation_scope(self) -> str:
        """When the framework trains the HVAE: ``"never"`` if ``weights_path`` was given, else ``"once"``.

        Returns:
            ``"once"`` or ``"never"``.
        """
        return "never" if self._pretrained else "once"

    def adapt(
        self,
        X: np.ndarray,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> None:
        """Train a new HVAE on the task's training expressions, replacing any earlier one.

        The expressions come from [training_expressions][SRToolkit.approaches.EDHiE.EDHiE.training_expressions].
        With ``training_seed`` set, sampling and training are reproducible and the global random
        states are restored afterwards.

        Args:
            X: Not used.
            background_knowledge: Supplies the training expressions.
            required_constraints: Every training expression passes them.

        Raises:
            ValueError: If no valid training expressions can be sampled.
        """
        # Seed this training only: the global states are restored afterwards, so later unseeded
        # searches stay random.
        states = None
        if self.training_seed is not None:
            states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            random.seed(self.training_seed)
            np.random.seed(self.training_seed)
            torch.manual_seed(self.training_seed)
        try:
            symbol_library = background_knowledge.symbol_library
            expressions = self.training_expressions(background_knowledge, required_constraints)
            expr_trees: list[Node | None] = [tokens_to_tree(expr, symbol_library) for expr in expressions]
            self.model = HVAE(len(symbol_library), self.latent_size, symbol_library)
            train_hvae(
                self.model,
                TreeDataset(expr_trees),
                symbol_library,
                epochs=self.epochs,
                batch_size=self.batch_size,
                max_beta=self.max_beta,
                verbose=self.verbose,
            )
        finally:
            if states is not None:
                random.setstate(states[0])
                np.random.set_state(states[1])
                torch.set_rng_state(states[2])

    def training_expressions(
        self,
        background_knowledge: "BackgroundKnowledge",
        required_constraints: Sequence[ExpressionConstraint],
    ) -> list[list[str]]:
        """Sample the expressions the HVAE is trained on.

        Samples from [sampling_grammar][SRToolkit.dataset.background_knowledge.BackgroundKnowledge.sampling_grammar]
        and keeps those of at most ``max_expression_length`` tokens that pass the pre-checks of
        ``required_constraints``. With
        ``only_unique_expressions``, they are distinct until ``max_training_skips`` valid samples in a
        row are repeats; then repeats fill the rest, with a warning.

        Args:
            background_knowledge: Supplies the grammar.
            required_constraints: Set-up constraints every expression must pass.

        Returns:
            ``num_expressions`` token lists.

        Raises:
            ValueError: If ``max_training_skips`` samples in a row break a required constraint.
        """
        grammar = background_knowledge.sampling_grammar()
        symbol_library = background_knowledge.symbol_library
        constant = str(symbol_library.constant_symbol)

        expressions: list[list[str]] = []
        seen = set()
        allow_repeats = not self.only_unique_expressions
        rejected = repeats = 0  # consecutive samples of each kind
        while len(expressions) < self.num_expressions:
            # unique=False: uniqueness is checked below, across batches and after the constraint filter.
            batch = generate_n_expressions(
                grammar,
                max(self.num_expressions - len(expressions), 64),
                unique=False,
                max_expression_length=self.max_expression_length,
                verbose=False,
            )
            for expr in batch:
                # The HVAE's vocabulary has only the library's symbols, so labelled constants become bare.
                expr = [constant if (s := symbol_library.get(t)) is not None and s.type == "const" else t for t in expr]
                if any(r.implemented and not r.passed for r in (c.pre_check(expr) for c in required_constraints)):
                    rejected += 1
                    if rejected >= self.max_training_skips:
                        raise ValueError(
                            f"[EDHiE] {rejected} sampled expressions in a row broke a required constraint; "
                            "the grammar cannot produce valid training expressions."
                        )
                    continue
                rejected = 0
                key = "".join(expr)
                if not allow_repeats and key in seen:
                    repeats += 1
                    if repeats >= self.max_training_skips:
                        allow_repeats = True
                        warnings.warn(
                            f"[EDHiE] Only {len(expressions)} distinct valid training expressions found; "
                            f"filling the remaining {self.num_expressions - len(expressions)} with repeats."
                        )
                    continue
                repeats = 0
                seen.add(key)
                expressions.append(expr)
                if len(expressions) == self.num_expressions:
                    break
        return expressions

    def save_adapted_state(self, path: str) -> None:
        """Save the HVAE (architecture, symbol library and weights) to ``path``.

        Args:
            path: File to write, e.g. ``path/model.pt``.

        Raises:
            RuntimeError: If there is no HVAE (no ``adapt`` and no ``weights_path``).
        """
        if self.model is None:
            raise RuntimeError("EDHiE has no HVAE to save: call adapt() or pass weights_path first.")
        torch.save(
            {
                "symbol_library": self.model.decoder.symbol_library.to_dict(),
                "latent_size": self.latent_size,
                "hidden_size": self.model.decoder.hidden_size,
                "max_height": self.model.decoder.max_height,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    def load_adapted_state(self, path: str) -> None:
        """Load the HVAE from ``path``, replacing any current one.

        Args:
            path: File written by [save_adapted_state][SRToolkit.approaches.EDHiE.EDHiE.save_adapted_state].
        """
        checkpoint = torch.load(path, weights_only=False)
        symbol_library = SymbolLibrary.from_dict(checkpoint["symbol_library"])
        self.model = HVAE(
            input_size=len(symbol_library),
            output_size=checkpoint["latent_size"],
            symbol_library=symbol_library,
            hidden_size=checkpoint["hidden_size"],
            max_height=checkpoint["max_height"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])

    def start(self, sr_evaluator: SR_evaluator, seed: int | None = None) -> _GASearch:
        """Seed the random states and start a new search.

        See [start][SRToolkit.approaches.sr_approach.SR_approach.start]. With ``seed`` set, it seeds the global
        random states (Python, NumPy, PyTorch; not restored) and the GA's own generator, so the same
        seed, HVAE, approach and evaluator settings give the same run; with ``None``, the GA's
        generator draws fresh entropy.

        Args:
            sr_evaluator: The evaluator of the run.
            seed: The run's seed, or ``None``.

        Returns:
            The search state.

        Raises:
            RuntimeError: If there is no HVAE (no ``adapt`` and no ``weights_path``).
        """
        if self.model is None:
            raise RuntimeError("EDHiE needs an HVAE: call adapt() or pass weights_path before start().")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        return _GASearch(self._new_ga(seed))

    def ask(self, sr_evaluator: SR_evaluator, state: _GASearch) -> list[Candidate | list[str] | Expression | None]:
        """Decode the next generation and propose one entry per individual, in order.

        See [ask][SRToolkit.approaches.sr_approach.SR_approach.ask]. Each individual is proposed as the first of
        these that applies:

        1. ``None`` if it does not decode;
        2. a skip ``"evaluated"`` if ``skip_evaluated`` is set and its stored form (see
           [stored_tokens][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stored_tokens]) is that of
           an expression evaluated before or proposed for evaluation earlier in this generation;
        3. a skip ``"rejected"`` if ``skip_rejected`` is set and it breaks a required constraint;
        4. its expression, to be evaluated.

        Args:
            sr_evaluator: The evaluator of the run.
            state: The search state.

        Returns:
            One entry per individual; never empty.
        """
        assert self.model is not None  # checked by start and load_search_state
        state.pending = state.ga.ask()
        X = state.pending.get("X")
        device = next(self.model.parameters()).device
        trees = self.model.decode(torch.tensor(X, device=device, dtype=torch.float32))
        candidates: list[Candidate | list[str] | Expression | None] = []
        evaluated: set[tuple[str, ...]] = set()  # stored forms of the expressions proposed for evaluation
        for tree in trees:
            if tree is None:  # Shouldn't happen but just to be safe
                candidates.append(None)
                continue
            # One Expression per individual, so the lookups below and the evaluation parse it once.
            expr = Expression.from_tree(tree, sr_evaluator.symbol_library)
            key = tuple(sr_evaluator.stored_tokens(expr))
            if self.skip_evaluated and (key in evaluated or sr_evaluator.get_model(expr) is not None):
                candidates.append(Candidate(expr, skip="evaluated"))
            elif self.skip_rejected and sr_evaluator.check_constraints(expr)[0]:
                candidates.append(Candidate(expr, skip="rejected"))
            else:
                candidates.append(expr)
                evaluated.add(key)
        return candidates

    def tell(
        self,
        sr_evaluator: SR_evaluator,
        state: _GASearch,
        candidates: list[Candidate | list[str] | Expression | None],
        results: "list[ModelResult | None]",
    ) -> None:
        """Score the generation from ``ask`` and select the next population.

        See [tell][SRToolkit.approaches.sr_approach.SR_approach.tell]. Individual *i* gets the error of
        ``results[i]`` (lower is better), or ``inf`` if that is ``None``, its error is ``nan``, or
        the step stopped before it.

        Args:
            sr_evaluator: The evaluator of the run.
            state: The search state.
            candidates: Not used.
            results: The models of the entries from ``ask`` that the step loop reached.
        """
        assert state.pending is not None  # set by ask
        errors = np.full(len(state.pending), np.inf)
        for i, result in enumerate(results):
            if result is not None and not np.isnan(result.error):
                errors[i] = result.error
        state.pending.set("F", errors[:, None])
        state.ga.tell(infills=state.pending)
        state.pending = None

    def dump_search_state(self, state: _GASearch) -> bytes:
        """Serialise the search state between two steps.

        See [dump_search_state][SRToolkit.approaches.sr_approach.SR_approach.dump_search_state].

        Args:
            state: The search state.

        Returns:
            The search state without the HVAE; only a fingerprint of the HVAE is saved, and
            ``load_search_state`` checks it.
        """
        assert self.model is not None  # checked by start and load_search_state
        # The GA itself does not pickle (tournament selection holds a local function); mating is
        # configuration only and the problem holds the HVAE, so both are rebuilt on load.
        ga = {k: v for k, v in vars(state.ga).items() if k not in ("mating", "problem")}
        return pickle.dumps({"ga": ga, "model": _digest(self.model)})

    def load_search_state(self, data: bytes) -> _GASearch:
        """Rebuild a search state from ``dump_search_state``, on this approach's HVAE.

        See [load_search_state][SRToolkit.approaches.sr_approach.SR_approach.load_search_state].

        Args:
            data: The serialised state.

        Returns:
            The search state, with the dumped GA settings.

        Raises:
            RuntimeError: If there is no HVAE (no ``adapt`` and no ``weights_path``).
            ValueError: If the HVAE differs from the one the state was saved with (state dicts
                compared bit for bit).
        """
        if self.model is None:
            raise RuntimeError("EDHiE needs an HVAE to restore a search state: call adapt() or pass weights_path.")
        saved = pickle.loads(data)
        if saved["model"] != _digest(self.model):
            raise ValueError(
                "EDHiE: the HVAE differs from the one the search state was saved with (weights must match "
                "bit for bit); load the same model (weights_path or a saved adapted state) or train it with "
                "the same training_seed on the same device and PyTorch version."
            )
        ga = self._new_ga(None)
        vars(ga).update(saved["ga"])
        return _GASearch(ga)

    def _new_ga(self, seed: int | None) -> GA:
        """Return a GA with this approach's settings, set up on the latent space."""
        assert self.model is not None
        ga = GA(
            pop_size=self.population_size,
            sampling=_TorchNormalSampling(),
            crossover=_LICrossover(),
            mutation=_RandomMutation(),
            eliminate_duplicates=False,
            seed=seed,
        )
        ga.setup(_LatentSpace(self.model, self.latent_size), termination=NoTermination())
        return ga

    @classmethod
    def from_config(cls, config: dict) -> "EDHiE":
        """Rebuild the approach from a config dict.

        Missing keys take the constructor defaults and unknown keys are ignored.
        ``max_training_skips`` falls back to the old key ``max_consecutive_skips``.

        Returns:
            The approach.
        """
        return cls(
            latent_size=config.get("latent_size", 32),
            num_expressions=config.get("num_expressions", 20000),
            max_expression_length=config.get("max_expression_length", 30),
            only_unique_expressions=config.get("only_unique_expressions", True),
            epochs=config.get("epochs", 20),
            batch_size=config.get("batch_size", 32),
            max_beta=config.get("max_beta", 0.035),
            population_size=config.get("population_size", 40),
            weights_path=config.get("weights_path", None),
            verbose=config.get("verbose", True),
            max_training_skips=config.get("max_training_skips", config.get("max_consecutive_skips", 10000)),
            skip_evaluated=config.get("skip_evaluated", True),
            skip_rejected=config.get("skip_rejected", True),
            training_seed=config.get("training_seed", None),
        )


class BatchedNode:
    """Batched binary tree node for vectorised HVAE forward/backward passes.

    Mirrors the recursive structure of [Node][SRToolkit.utils.expression_tree.Node] but stores a
    batch of trees simultaneously. Each position in ``symbols`` corresponds to one tree in the
    batch; empty strings ``""`` mark padding (absent subtrees). ``left`` and ``right`` are
    themselves ``BatchedNode`` instances (or ``None``) representing the batched subtrees.

    After all trees have been added via [add_tree][SRToolkit.approaches.EDHiE.BatchedNode.add_tree],
    call [create_target][SRToolkit.approaches.EDHiE.BatchedNode.create_target] to build the
    one-hot target tensors and masks required by the HVAE loss.

    We recommend using [SRToolkit.approaches.EDHiE.create_batch] to create batched trees.
    """

    def __init__(self, symbol2index: dict[str, int], size: int = 0, trees: list[Node | None] | None = None):
        """Args:
        symbol2index: Mapping from symbol strings to vocabulary indices.
        size: Number of padding slots to pre-allocate (filled with ``""``).
        trees: Optional list of [Node][SRToolkit.utils.expression_tree.Node] trees to add
            immediately via [add_tree][SRToolkit.approaches.EDHiE.BatchedNode.add_tree].
        """
        self.symbols: list[str] = ["" for _ in range(size)]  # For when a new leaf node is added
        self.left: BatchedNode | None = None
        self.right: BatchedNode | None = None
        self.symbol2index: dict[str, int] = symbol2index
        self.mask: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.target_indices: torch.Tensor | None = None
        self.prediction: torch.Tensor | None = None

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)

    def add_tree(self, tree: Node | None = None) -> None:
        """Append one tree (or a padding slot) to the batch.

        If ``tree`` is ``None``, an empty padding entry (``""``) is appended at every level.
        Otherwise the tree's symbol is appended and its subtrees are recursively merged into
        ``self.left`` / ``self.right``, creating new ``BatchedNode`` children as needed.

        Args:
            tree: A [Node][SRToolkit.utils.expression_tree.Node] to add, or ``None`` to append
                a padding slot.

        Returns:
            None
        """
        # Add an empty subtree to the batch. This is used to pad the batch so that it
        # has the same number of symbols in each node.
        if tree is None:
            self.symbols.append("")

            # Recursively fill subtrees if they exist
            if self.left is not None:
                self.left.add_tree()
            if self.right is not None:
                self.right.add_tree()
        # Add the given subtree to the batch
        else:
            self.symbols.append(tree.symbol)

            # Add the left subtree
            if isinstance(self.left, BatchedNode) and isinstance(tree.left, Node):
                self.left.add_tree(tree.left)
            # Add the empty subtree to the right batched node
            elif isinstance(self.left, BatchedNode):
                self.left.add_tree()
            # Add new batched nodes to the left
            elif isinstance(tree.left, Node):
                self.left = BatchedNode(self.symbol2index, size=len(self.symbols) - 1)
                self.left.add_tree(tree.left)

            # Add the right subtree
            if isinstance(self.right, BatchedNode) and isinstance(tree.right, Node):
                self.right.add_tree(tree.right)
            # Add the empty subtree to the right batched node
            elif isinstance(self.right, BatchedNode):
                self.right.add_tree()
            # Add the new batched node to the right
            elif isinstance(tree.right, Node):
                self.right = BatchedNode(self.symbol2index, size=len(self.symbols) - 1)
                self.right.add_tree(tree.right)

    def create_target(self) -> None:
        """Build one-hot target tensors, integer target indices, and binary masks for the entire subtree.

        Populates ``self.target`` (one-hot, shape ``(batch, vocab)``), ``self.target_indices``
        (integer class indices, shape ``(batch,)``, ``-1`` for padding), and ``self.mask``
        (``1.0`` for real symbols, ``0.0`` for padding). Recurses into ``left`` and ``right``.

        Returns:
            None
        """
        target = torch.zeros((len(self.symbols), len(self.symbol2index)))
        mask = torch.ones(len(self.symbols))
        target_indices = torch.full((len(self.symbols),), -1, dtype=torch.long)

        for i, s in enumerate(self.symbols):
            if s == "":
                mask[i] = 0
            else:
                target[i, self.symbol2index[s]] = 1
                target_indices[i] = self.symbol2index[s]

        self.mask = mask
        self.target = target
        self.target_indices = target_indices

        if self.left is not None:
            self.left.create_target()
        if self.right is not None:
            self.right.create_target()

    def to_expr_list(self) -> list[Node | None]:
        """Extract one [Node][SRToolkit.utils.expression_tree.Node] tree per batch element.

        Returns:
            A list of [Node][SRToolkit.utils.expression_tree.Node] trees (or ``None`` for padding
            slots), one per position in the batch.
        """
        return [self.get_expr_at_idx(i) for i in range(len(self.symbols))]

    def get_expr_at_idx(self, idx: int) -> Node | None:
        """Reconstruct the expression tree for a single batch element.

        Args:
            idx: Batch index to retrieve.

        Returns:
            A [Node][SRToolkit.utils.expression_tree.Node] tree, or ``None`` if the slot is padding.
        """
        symbol = self.symbols[idx]
        if symbol == "":
            return None

        left = self.left.get_expr_at_idx(idx) if isinstance(self.left, BatchedNode) else None
        right = self.right.get_expr_at_idx(idx) if isinstance(self.right, BatchedNode) else None

        return Node(symbol, left=left, right=right)

    def get_prediction(self) -> torch.Tensor:
        """Collect decoder logit predictions from all nodes in infix order.

        Returns:
            Stacked logit tensor of shape ``(batch, vocab, n_nodes)``.
        """
        predictions = self.get_prediction_rec()
        return torch.stack(predictions, dim=2)

    def get_prediction_rec(self) -> list[torch.Tensor]:
        if self.prediction is None:
            return []

        reps = []
        if isinstance(self.left, BatchedNode):
            reps += self.left.get_prediction_rec()

        reps.append(self.prediction)

        if isinstance(self.right, BatchedNode):
            reps += self.right.get_prediction_rec()

        return reps

    def get_target(self) -> torch.Tensor:
        """Collect one-hot target indices from all nodes in infix order.

        Returns:
            Stacked target tensor of shape ``(batch, n_nodes)`` with ``-1`` for padding.
        """
        targets = self.get_target_rec()
        return torch.stack(targets, dim=1)

    def get_target_rec(self) -> list[torch.Tensor]:
        if self.target_indices is None:
            raise RuntimeError(
                "BatchedNode.create_target() must be called before get_target_rec(). We recommend using SRToolkit.approaches.EDHiE.create_batch tp create batched trees."
            )

        reps = []
        if isinstance(self.left, BatchedNode):
            reps += self.left.get_target_rec()

        reps.append(self.target_indices)

        if isinstance(self.right, BatchedNode):
            reps += self.right.get_target_rec()

        return reps


class HVAE(nn.Module):
    """Hierarchical Variational Autoencoder (HVAE) for expression trees.

    Combines a tree-recursive [Encoder][SRToolkit.approaches.EDHiE.Encoder] with a
    tree-recursive [Decoder][SRToolkit.approaches.EDHiE.Decoder] to learn a continuous
    latent representation of symbolic expressions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        symbol_library: SymbolLibrary,
        hidden_size: int | None = None,
        max_height: int = 20,
    ):
        """Args:
        input_size: Vocabulary size (number of symbols in the symbol library).
        output_size: Dimensionality of the latent space.
        symbol_library: [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] used by
            the decoder to constrain leaf/non-leaf symbol selection.
        hidden_size: Hidden state size for the GRU cells. Defaults to ``output_size``.
        max_height: Maximum tree depth the decoder will generate before forcing leaf symbols.
        """
        super().__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size, symbol_library, max_height)

    def forward(self, tree: BatchedNode) -> tuple[torch.Tensor, torch.Tensor, BatchedNode]:
        """Run the full VAE forward pass: encode, sample latent vector, decode.

        Args:
            tree: Batched expression trees (training mode — teacher-forced decoding).

        Returns:
            A 3-tuple ``(mu, logvar, reconstructed_tree)`` where ``mu`` and ``logvar`` are the
            approximate posterior parameters and ``reconstructed_tree`` is the decoder output.
        """
        mu, logvar = self.encoder(tree)
        z = self.sample(mu, logvar)
        out = self.decoder(z, tree)
        return mu, logvar, out

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw a latent sample using the reparameterisation trick.

        Args:
            mu: Mean of the approximate posterior, shape ``(batch, latent_size)``.
            logvar: Log-variance of the approximate posterior, same shape.

        Returns:
            Sampled latent vector ``z = mu + eps * exp(logvar / 2)``.
        """
        eps = torch.randn_like(mu)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree: BatchedNode) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of trees to posterior parameters without sampling.

        Args:
            tree: Batched expression trees.

        Returns:
            A 2-tuple ``(mu, logvar)`` — the approximate posterior mean and log-variance.
        """
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> list[Node | None]:
        """Decode a batch of latent vectors to expression trees (inference mode).

        Args:
            z: Latent vectors of shape ``(batch, latent_size)``.

        Returns:
            A list of [Node][SRToolkit.utils.expression_tree.Node] trees (or ``None`` for failed
            decodes), one per row of ``z``.
        """
        return self.decoder.decode(z)


class Encoder(nn.Module):
    """Tree-recursive GRU encoder that maps a [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode]
    tree to approximate posterior parameters ``(mu, logvar)``.

    Uses a [GRU221][SRToolkit.approaches.EDHiE.GRU221] cell to combine left/right child hidden
    states bottom-up, then projects to ``mu`` and ``logvar`` via linear layers.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Args:
        input_size: Vocabulary size (one-hot input dimension).
        hidden_size: Hidden state dimensionality of the GRU.
        output_size: Dimensionality of the latent space (``mu`` and ``logvar`` output size).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = GRU221(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree: BatchedNode) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batched tree to posterior parameters.

        Args:
            tree: Batched expression tree with targets and masks populated.

        Returns:
            A 2-tuple ``(mu, logvar)`` of shape ``(batch, latent_size)`` each.
        """
        if tree.target is None or tree.mask is None:
            raise RuntimeError(
                "[Encoder.forward] BatchedNode.create_target() must be called before encoding. We recommend using SRToolkit.approaches.EDHiE.create_batch() to create batched trees."
            )
        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        return mu, logvar

    def recursive_forward(self, tree: BatchedNode) -> torch.Tensor:
        """Recursively encode a subtree bottom-up using the GRU cell.

        Args:
            tree: Current subtree node (batched).

        Returns:
            Hidden state tensor of shape ``(batch, hidden_size)`` for this subtree root.
        """
        if isinstance(tree.left, BatchedNode):
            h_left = self.recursive_forward(tree.left)
        else:
            # Type checking suppressed because if tree.target exists is checked in forward() and target is added recursively.
            h_left = torch.zeros(len(tree.symbols), self.hidden_size, device=tree.target.device)  # type: ignore[union-attr]

        if isinstance(tree.right, BatchedNode):
            h_right = self.recursive_forward(tree.right)
        else:
            # Type checking suppressed because if tree.target exists is checked in forward() and target is added recursively.
            h_right = torch.zeros(len(tree.symbols), self.hidden_size, device=tree.target.device)  # type: ignore[union-attr]

        hidden = self.gru(tree.target, h_left, h_right)
        # Type checking suppressed because if tree.mask exists is checked in forward() and mask is added recursively.
        hidden = hidden.mul(tree.mask[:, None])  # type: ignore[index]
        return hidden


class Decoder(nn.Module):
    """Tree-recursive GRU decoder that maps a latent vector to an expression tree.

    Uses a [GRU122][SRToolkit.approaches.EDHiE.GRU122] cell to split each parent hidden state
    top-down into left and right child hidden states, then predicts the symbol at each node.
    """

    leaf_symbols_mask: torch.Tensor

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, symbol_library: SymbolLibrary, max_height: int = 20
    ):
        """Args:
        input_size: Latent space dimensionality (decoder input).
        hidden_size: Hidden state dimensionality of the GRU.
        output_size: Vocabulary size (logit output dimension).
        symbol_library: Used to identify leaf symbols and enforce the ``max_height`` constraint.
        max_height: Maximum tree depth; beyond this depth only leaf symbols are sampled.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru = GRU122(input_size=output_size, hidden_size=hidden_size)

        self.max_height = max_height
        self.symbol_library = symbol_library
        self.symbol2index = symbol_library.symbols2index()
        self.index2symbol = {i: s for s, i in self.symbol2index.items()}
        leaf_symbols_mask = torch.zeros(len(self.symbol2index))
        for s, i in self.symbol2index.items():
            if self.symbol_library[s].type in ["var", "const", "lit"]:
                leaf_symbols_mask[i] = 1
        self.register_buffer("leaf_symbols_mask", leaf_symbols_mask)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, z: torch.Tensor, tree: BatchedNode) -> BatchedNode:
        """Teacher-forced decoding pass used during training.

        Stores logit predictions in each node of ``tree`` (in-place) using the ground-truth
        tree structure to guide recursion.

        Args:
            z: Latent vectors of shape ``(batch, latent_size)``.
            tree: Ground-truth batched tree; predictions are written into each node's
                ``prediction`` attribute.

        Returns:
            The same ``tree`` object with ``prediction`` populated at every node.
        """
        hidden = self.z2h(z)
        self.recursive_forward(hidden, tree)
        return tree

    def recursive_forward(self, hidden: torch.Tensor, tree: BatchedNode) -> None:  # type: ignore[override]
        """Recursively write predictions into a subtree (teacher-forced, training only).

        Args:
            hidden: Current node hidden state, shape ``(batch, hidden_size)``.
            tree: Current subtree node; ``prediction`` is set in-place.

        Returns:
            None
        """
        prediction = self.h2o(hidden)
        tree.prediction = prediction
        symbol_probs = F.softmax(prediction, dim=1)
        continue_left = isinstance(tree.left, BatchedNode)
        continue_right = isinstance(tree.right, BatchedNode)
        if continue_left or continue_right:
            left, right = self.gru(symbol_probs, hidden)
            if continue_left:
                self.recursive_forward(left, tree.left)  # type: ignore[arg-type]
            if continue_right:
                self.recursive_forward(right, tree.right)  # type: ignore[arg-type]

    def decode(self, z: torch.Tensor) -> list[Node | None]:
        """Autoregressively decode latent vectors to expression trees (inference mode).

        Args:
            z: Latent vectors of shape ``(batch, latent_size)``.

        Returns:
            A list of [Node][SRToolkit.utils.expression_tree.Node] trees, one per row of ``z``.
        """
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch = self.recursive_decode(hidden, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden: torch.Tensor, mask: torch.Tensor, height: int = 0) -> BatchedNode:
        """Recursively decode hidden states into a batched subtree (inference mode).

        Args:
            hidden: Current node hidden state, shape ``(batch, hidden_size)``.
            mask: Boolean mask; ``True`` for active (non-padding) batch elements.
            height: Current tree depth (used to enforce ``max_height``).

        Returns:
            A [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode] representing this subtree.
        """
        prediction = F.softmax(self.h2o(hidden), dim=1)
        # Sample symbol in a given node
        symbols, left_mask, right_mask = self.sample_symbol(prediction, mask, height)
        has_left = torch.any(left_mask)
        has_right = torch.any(right_mask)
        if has_left or has_right:
            left, right = self.gru(prediction, hidden)
            l_tree = self.recursive_decode(left, left_mask, height + 1) if has_left else None
            r_tree = self.recursive_decode(right, right_mask, height + 1) if has_right else None
        else:
            l_tree = None
            r_tree = None

        node = BatchedNode(self.symbol2index)
        node.symbols = symbols
        node.left = l_tree
        node.right = r_tree
        return node

    def sample_symbol(
        self, prediction: torch.Tensor, mask: torch.Tensor, height: int
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Greedily select one symbol per batch element and compute child masks.

        Args:
            prediction: Softmax probabilities over the vocabulary, shape ``(batch, vocab)``.
            mask: Boolean mask for active batch elements.
            height: Current tree depth; at ``max_height`` only leaf symbols are eligible.

        Returns:
            A 3-tuple ``(symbols, left_mask, right_mask)`` where ``symbols`` is a list of
            selected symbol strings (``""`` for masked positions), and the masks indicate which
            batch elements should recurse into the left/right child.
        """
        symbols = []
        left_mask = mask.clone()
        right_mask = mask.clone()

        if height >= self.max_height:
            prediction = prediction * self.leaf_symbols_mask

        for i in range(prediction.size(0)):
            if mask[i]:
                symbol = self.index2symbol[int(torch.argmax(prediction[i, :]))]
                symbols.append(symbol)
                kind = self.symbol_library[symbol].type
                if kind == "fn":
                    right_mask[i] = False
                if kind in ["var", "const", "lit"]:
                    left_mask[i] = False
                    right_mask[i] = False
            else:
                symbols.append("")
        return symbols, left_mask, right_mask


class GRU221(nn.Module):
    """Two-input, one-output GRU cell used by the [Encoder][SRToolkit.approaches.EDHiE.Encoder].

    Combines an input vector with two hidden states (left child and right child) into a
    single parent hidden state. The two child states are concatenated before being passed
    through the standard GRU gating equations.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """Args:
        input_size: Dimensionality of the input vector (vocabulary one-hot size).
        hidden_size: Dimensionality of each child hidden state and the output hidden state.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whr = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whz = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.whn = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Compute parent hidden state from input and two child hidden states.

        Args:
            x: Input (symbol embedding / one-hot), shape ``(batch, input_size)``.
            h1: Left child hidden state, shape ``(batch, hidden_size)``.
            h2: Right child hidden state, shape ``(batch, hidden_size)``.

        Returns:
            Parent hidden state of shape ``(batch, hidden_size)``.
        """
        h = torch.cat([h1, h2], dim=1)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2


class GRU122(nn.Module):
    """One-input, two-output GRU cell used by the [Decoder][SRToolkit.approaches.EDHiE.Decoder].

    Splits a parent hidden state into two child hidden states (left and right) driven by
    an input vector. The output is split evenly along the hidden dimension.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """Args:
        input_size: Dimensionality of the input vector (symbol probability distribution).
        hidden_size: Dimensionality of each output child hidden state (output is
            ``2 * hidden_size`` before splitting).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wir = nn.Linear(in_features=input_size, out_features=2 * hidden_size)
        self.whr = nn.Linear(in_features=hidden_size, out_features=2 * hidden_size)
        self.wiz = nn.Linear(in_features=input_size, out_features=2 * hidden_size)
        self.whz = nn.Linear(in_features=hidden_size, out_features=2 * hidden_size)
        self.win = nn.Linear(in_features=input_size, out_features=2 * hidden_size)
        self.whn = nn.Linear(in_features=hidden_size, out_features=2 * hidden_size)
        torch.nn.init.xavier_uniform_(self.wir.weight)
        torch.nn.init.xavier_uniform_(self.whr.weight)
        torch.nn.init.xavier_uniform_(self.wiz.weight)
        torch.nn.init.xavier_uniform_(self.whz.weight)
        torch.nn.init.xavier_uniform_(self.win.weight)
        torch.nn.init.xavier_uniform_(self.whn.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute two child hidden states from input and parent hidden state.

        Args:
            x: Input vector (symbol probabilities), shape ``(batch, input_size)``.
            h: Parent hidden state, shape ``(batch, hidden_size)``.

        Returns:
            A 2-tuple ``(h_left, h_right)`` of child hidden states, each of shape
            ``(batch, hidden_size)``.
        """
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 2)
        out = (1 - z) * n + z * dh
        h_left, h_right = torch.split(out, self.hidden_size, dim=1)
        return h_left, h_right


def create_batch(trees: list[Node | None], symbol2index: dict[str, int]) -> BatchedNode:
    """Pack a list of expression trees into a single [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode]
    with targets and masks populated.

    Args:
        trees: List of [Node][SRToolkit.utils.expression_tree.Node] trees to batch.
        symbol2index: Mapping from symbol strings to vocabulary indices.

    Returns:
        A [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode] ready for the HVAE forward pass.
    """
    t = BatchedNode(symbol2index, trees=trees)
    t.create_target()
    return t


class TreeBatchSampler(Sampler[list[int]]):
    """PyTorch sampler that yields randomly permuted mini-batches of tree indices.

    Each call to ``__iter__`` produces a fresh permutation. Batches are contiguous slices
    of the permutation; the last incomplete batch is dropped.
    """

    def __init__(self, batch_size: int, num_eq: int):
        """Args:
        batch_size: Number of trees per mini-batch.
        num_eq: Total number of trees in the dataset.
        """
        self.batch_size = batch_size
        self.num_eq = num_eq

    def __iter__(self) -> Iterator[list[int]]:
        permutation = np.random.permutation(self.num_eq)
        for i in range(len(self)):
            yield permutation[i * self.batch_size : (i + 1) * self.batch_size].tolist()

    def __len__(self) -> int:
        return self.num_eq // self.batch_size


class TreeDataset(Dataset):
    """Minimal PyTorch ``Dataset`` wrapping a list of expression trees."""

    def __init__(self, trees: list[Node | None]):
        """Args:
        trees: List of [Node][SRToolkit.utils.expression_tree.Node] expression trees.
        """
        self.trees: list[Node] = [t for t in trees if isinstance(t, Node)]

    def __getitem__(self, idx: int) -> Node:
        return self.trees[idx]

    def __len__(self) -> int:
        return len(self.trees)


def annealing_schedule(it: int, total_iters: int, supremum: float = 0.04) -> float:
    """Linear KL annealing schedule mapping iteration count to a ``[0, supremum]`` coefficient.

    Args:
        it: Current iteration index.
        total_iters: Total number of training iterations.
        supremum: Maximum value returned at ``it == total_iters``.

    Returns:
        Annealing coefficient ``supremum * (it / total_iters)``.
    """
    x = it / total_iters
    return x * supremum


def hvae_loss(
    outputs: "BatchedNode",
    mu: torch.Tensor,
    logvar: torch.Tensor,
    lmbda: float,
    criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the HVAE ELBO loss: reconstruction (cross-entropy) + KL divergence.

    Args:
        outputs: Decoded [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode] produced by the HVAE forward pass.
        mu: Mean of the approximate posterior, shape ``(batch, latent_size)``.
        logvar: Log-variance of the approximate posterior, same shape as ``mu``.
        lmbda: KL annealing coefficient that scales the KL term.
        criterion: Reconstruction loss callable (typically ``CrossEntropyLoss``).

    Returns:
        A 3-tuple ``(total_loss, bce, kld)`` where ``total_loss = bce + lmbda * kld``.
    """
    BCE = criterion(outputs.get_prediction(), outputs.get_target())
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)
    return BCE + lmbda * KLD, BCE, KLD


def train_hvae(
    model: HVAE,
    trainset: TreeDataset,
    symbol_library: SymbolLibrary,
    epochs: int = 20,
    batch_size: int = 32,
    max_beta: float = 0.04,
    verbose: bool = True,
) -> None:
    """Train an [HVAE][SRToolkit.approaches.EDHiE.HVAE] on a dataset of expression trees.

    Uses the Adam optimiser and cross-entropy reconstruction loss with KL annealing controlled
    by [annealing_schedule][SRToolkit.approaches.EDHiE.annealing_schedule]. If ``verbose=True``,
    prints one decoded reconstruction sample per epoch at the midpoint batch.

    Args:
        model: The [HVAE][SRToolkit.approaches.EDHiE.HVAE] to train (modified in-place).
        trainset: [TreeDataset][SRToolkit.approaches.EDHiE.TreeDataset] of expression trees.
        symbol_library: Symbol library defining the vocabulary.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        max_beta: Maximum KL annealing coefficient.
        verbose: If ``True``, prints loss statistics and a sample reconstruction per epoch.

    Returns:
        None
    """
    symbol2index = symbol_library.symbols2index()
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    total_iters = epochs * (len(trainset) // batch_size)
    midpoint = len(trainset) // (2 * batch_size)
    sampler = TreeBatchSampler(batch_size, len(trainset))

    for epoch in range(epochs):
        bce, kl, loss_sum, num_iters = 0.0, 0.0, 0.0, 0.0

        with tqdm(total=len(trainset), desc=f"Training HVAE - Epoch: {epoch + 1}/{epochs}", unit="chunks") as prog_bar:
            for i, tree_ids in enumerate(sampler):
                lmbda = annealing_schedule(iter_counter, total_iters, max_beta)
                iter_counter += 1

                batch = create_batch([trainset[j] for j in tree_ids], symbol2index)

                optimizer.zero_grad()
                mu, logvar, outputs = model(batch)
                loss, bcel, kll = hvae_loss(outputs, mu, logvar, lmbda, criterion)
                loss.backward()
                optimizer.step()

                num_iters += 1
                bce += bcel.detach().item()
                kl += kll.detach().item()
                loss_sum += loss.detach().item()
                prog_bar.set_postfix(
                    **{"run:": "HVAE", "loss": loss_sum / num_iters, "BCE": bce / num_iters, "KLD": kl / num_iters}
                )
                prog_bar.update(batch_size)

                if verbose and i == midpoint:
                    original_trees = batch.to_expr_list()
                    decoded_trees = model.decode(mu.detach())
                    print()
                    if original_trees[0] is not None and decoded_trees[0] is not None:
                        print(f"O: {''.join(original_trees[0].to_list(symbol_library=symbol_library))}")
                        print(f"P: {''.join(decoded_trees[0].to_list(symbol_library=symbol_library))}")


def _digest(model: HVAE) -> str:
    """Return the sha256 hex digest of the model's state dict (names and weights, read on the CPU)."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class _LatentSpace(Problem):
    """pymoo ``Problem`` for the HVAE latent space.

    EDHiE scores the individuals itself (``ask`` / ``tell``), so the problem only carries the model
    and the symbol indices for [_RandomMutation][SRToolkit.approaches.EDHiE._RandomMutation].
    """

    def __init__(self, model: HVAE, dim: int):
        """Set up the problem on a trained HVAE.

        Args:
            model: Trained [HVAE][SRToolkit.approaches.EDHiE.HVAE].
            dim: Dimensionality of the latent space (number of decision variables for pymoo).
        """
        self.model = model
        self.symbol2index = model.decoder.symbol_library.symbols2index()
        super().__init__(n_var=dim, n_obj=1)


class _TorchNormalSampling(Sampling):
    """pymoo ``Sampling`` that initialises the GA population by drawing samples from a
    standard normal distribution in latent space.
    """

    def _do(self, problem: _LatentSpace, n_samples: int, **kwargs) -> np.ndarray:
        """Args:
            problem: The [_LatentSpace][SRToolkit.approaches.EDHiE._LatentSpace].
            n_samples: Number of individuals to generate.

        Returns:
            Array of shape ``(n_samples, n_var)`` sampled from ``N(0, 1)`` in latent space.
        """
        return np.random.standard_normal((n_samples, problem.n_var))


class _LICrossover(Crossover):
    """Linear interpolation (LI) crossover operator for latent vectors.

    Combines two parent vectors as a random convex combination:
    ``child = w * parent1 + (1 - w) * parent2`` with per-dimension weights ``w ~ U(0, 1)``.
    """

    def __init__(self):
        """Initialises a 2-parent → 1-child crossover operator."""
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        weights = np.random.random(X.shape[2])  # per-variable weights, shape (n_vars,)
        return (X[0, :] * weights[None, :] + X[1, :] * (1 - weights[None, :]))[None, :, :]


class _RandomMutation(Mutation):
    """Latent-space mutation that re-encodes individuals and perturbs them proportionally to
    the posterior variance learned by the HVAE encoder.

    Each individual is decoded to an expression tree, re-encoded to obtain the posterior
    log-variance, and then perturbed with Gaussian noise scaled by the standard deviation.
    """

    def __init__(self):
        """Initialises the mutation operator."""
        super().__init__()

    def _do(self, problem: _LatentSpace, X, **kwargs):
        device = next(problem.model.parameters()).device
        trees = problem.model.decode(torch.tensor(X, device=device, dtype=torch.float32))
        batch = create_batch(trees, problem.symbol2index)
        var = problem.model.encode(batch)[1].detach().cpu().numpy()
        mutation_scale = np.random.random((X.shape[0], 1))
        std = np.multiply(mutation_scale, (np.exp(var / 2.0) - 1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)
