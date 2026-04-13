"""
EDHiE approach — equation discovery with hierarchical variational autoencoders by Mežnar et al.
"""

import random
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Node, SymbolLibrary, generate_n_expressions, tokens_to_tree

from .sr_approach import ApproachConfig, SR_approach, check_dependencies

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.core.termination import Termination
    from pymoo.optimize import minimize
except ImportError:
    raise ImportError("pymoo is not installed.")

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import Dataset, Sampler
except ImportError:
    raise ImportError("PyTorch is not installed.")


@dataclass
class EDHiEConfig(ApproachConfig):
    """
    Configuration dataclass for the [EDHiE][SRToolkit.approaches.EDHiE.EDHiE] approach.

    Examples:
        >>> cfg = EDHiEConfig(latent_size=32, epochs=10)
        >>> cfg.name
        'EDHiE'
        >>> d = cfg.to_dict()
        >>> EDHiEConfig.from_dict(d).latent_size
        32
    """

    name: str = "EDHiE"
    latent_size: int = 32
    num_expressions: int = 20000
    max_expression_length: int = 30
    only_unique_expressions: bool = True
    epochs: int = 20
    batch_size: int = 32
    max_beta: float = 0.035
    population_size: int = 40
    weights_path: Optional[str] = None
    verbose: bool = True


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
        weights_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        r"""
        EDHiE — Equation Discovery with Hierarchical variational autoEncoders by Mežnar et al.

        Trains a Hierarchical VAE (HVAE) on randomly generated expressions from the symbol library
        (``adapt``), then explores the learned latent space with a genetic algorithm to find
        expressions that best fit the target dataset (``search``).

        Examples:
            >>> model = EDHiE(latent_size=24, num_expressions=100, epochs=1, verbose=False)
            >>> model.name
            'EDHiE'

        Args:
            latent_size: Dimensionality of the HVAE latent space.
            num_expressions: Number of random expressions generated to train the HVAE.
            max_expression_length: Maximum token length of generated training expressions.
            only_unique_expressions: If ``True``, only unique expressions are generated.
            epochs: Number of training epochs for the HVAE.
            batch_size: Batch size used during HVAE training.
            max_beta: Maximum value of the KL annealing coefficient (controls regularization strength).
            population_size: GA population size used during latent space search.
            weights_path: Optional path to load/save HVAE weights from/to disk.
            verbose: If ``True``, prints training loss and new best expressions during search.
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
            )
        )
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

        self.model: Optional[HVAE] = None
        if weights_path is not None:
            self.load_adapted_state(weights_path)

    @property
    def adaptation_scope(self) -> str:
        return "once" if self.model is None else "never"

    def prepare(self) -> None:
        """
        Reset per-experiment state.

        The trained HVAE weights are preserved across experiments; only the evolutionary search
        is re-initialised on each call to
        [search][SRToolkit.approaches.EDHiE.EDHiE.search].

        Returns:
            None
        """
        pass

    def adapt(self, X: np.ndarray, symbol_library: SymbolLibrary) -> None:
        """
        Train the HVAE on randomly generated expressions from the symbol library.

        Args:
            X: Input data from the domain (shape ``(n_samples, n_variables)``). Only used to
                determine the number of variables; target values are not accessed.
            symbol_library: Symbol library defining available tokens.

        Returns:
            None
        """
        expressions = generate_n_expressions(
            symbol_library,
            self.num_expressions,
            max_expression_length=self.max_expression_length,
            unique=self.only_unique_expressions,
        )
        expr_trees: List[Optional[Node]] = [tokens_to_tree(expr, symbol_library) for expr in expressions]
        trainset = TreeDataset(expr_trees)

        self.model = HVAE(len(symbol_library), self.latent_size, symbol_library)
        train_hvae(
            self.model,
            trainset,
            symbol_library,
            epochs=self.epochs,
            batch_size=self.batch_size,
            max_beta=self.max_beta,
            verbose=self.verbose,
        )

    def save_adapted_state(self, path: str) -> None:
        """
        Save the trained HVAE model weights and architecture metadata to disk.

        Saves a single checkpoint file containing the symbol library, latent size,
        hidden size, max height, and model weights — enough to fully reconstruct
        the model in [load_adapted_state][SRToolkit.approaches.EDHiE.EDHiE.load_adapted_state]
        without needing to call [adapt][SRToolkit.approaches.EDHiE.EDHiE.adapt] first.

        Args:
            path: File path to save the checkpoint to, including the file extension, e.g. ``path/model.pt``.
        """
        if self.model is None:
            warnings.warn("[EDHiE.save_adapted_state] No model to save.")
            return
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
        """
        Restore a previously trained HVAE model from disk.

        Reconstructs the [HVAE][SRToolkit.approaches.EDHiE.HVAE] from the architecture
        metadata saved alongside the weights, so no prior call to
        [adapt][SRToolkit.approaches.EDHiE.EDHiE.adapt] is required.

        Args:
            path: File path previously passed to
                [save_adapted_state][SRToolkit.approaches.EDHiE.EDHiE.save_adapted_state].
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

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """
        Explore the HVAE latent space with a genetic algorithm to find the best expression.

        Args:
            sr_evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] used to
                score candidate expressions.
            seed: Optional random seed for reproducibility.

        Returns:
            None

        Raises:
            RuntimeError: If [adapt][SRToolkit.approaches.EDHiE.EDHiE.adapt] has not been called
                before ``search``.
        """
        if self.model is None:
            raise RuntimeError("EDHiE.adapt() must be called before search().")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

        problem = _HVAEProblem(self.model, sr_evaluator, self.latent_size)
        ga = GA(
            pop_size=self.population_size,
            sampling=_TorchNormalSampling(),
            crossover=_LICrossover(),
            mutation=_RandomMutation(),
            eliminate_duplicates=False,
        )
        minimize(
            problem,
            ga,
            _BestTermination(),
            verbose=False,
        )

    @classmethod
    def from_config(cls, config: dict) -> "EDHiE":
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
        )


class BatchedNode:
    """
    Batched binary tree node for vectorised HVAE forward/backward passes.

    Mirrors the recursive structure of [Node][SRToolkit.utils.expression_tree.Node] but stores a
    batch of trees simultaneously. Each position in ``symbols`` corresponds to one tree in the
    batch; empty strings ``""`` mark padding (absent subtrees). ``left`` and ``right`` are
    themselves ``BatchedNode`` instances (or ``None``) representing the batched subtrees.

    After all trees have been added via [add_tree][SRToolkit.approaches.EDHiE.BatchedNode.add_tree],
    call [create_target][SRToolkit.approaches.EDHiE.BatchedNode.create_target] to build the
    one-hot target tensors and masks required by the HVAE loss.

    We recommend using [SRToolkit.approaches.EDHiE.create_batch] to create batched trees.
    """

    def __init__(self, symbol2index: Dict[str, int], size: int = 0, trees: Optional[List[Optional[Node]]] = None):
        """
        Args:
            symbol2index: Mapping from symbol strings to vocabulary indices.
            size: Number of padding slots to pre-allocate (filled with ``""``).
            trees: Optional list of [Node][SRToolkit.utils.expression_tree.Node] trees to add
                immediately via [add_tree][SRToolkit.approaches.EDHiE.BatchedNode.add_tree].
        """
        self.symbols: List[str] = ["" for _ in range(size)]  # For when a new leaf node is added
        self.left: Optional[BatchedNode] = None
        self.right: Optional[BatchedNode] = None
        self.symbol2index: Dict[str, int] = symbol2index
        self.mask: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.target_indices: Optional[torch.Tensor] = None
        self.prediction: Optional[torch.Tensor] = None

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)

    def add_tree(self, tree: Optional[Node] = None) -> None:
        """
        Append one tree (or a padding slot) to the batch.

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
        """
        Build one-hot target tensors, integer target indices, and binary masks for the entire subtree.

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

    def to_expr_list(self) -> List[Optional[Node]]:
        """
        Extract one [Node][SRToolkit.utils.expression_tree.Node] tree per batch element.

        Returns:
            A list of [Node][SRToolkit.utils.expression_tree.Node] trees (or ``None`` for padding
            slots), one per position in the batch.
        """
        return [self.get_expr_at_idx(i) for i in range(len(self.symbols))]

    def get_expr_at_idx(self, idx: int) -> Optional[Node]:
        """
        Reconstruct the expression tree for a single batch element.

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
        """
        Collect decoder logit predictions from all nodes in infix order.

        Returns:
            Stacked logit tensor of shape ``(batch, vocab, n_nodes)``.
        """
        predictions = self.get_prediction_rec()
        return torch.stack(predictions, dim=2)

    def get_prediction_rec(self) -> List[torch.Tensor]:
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
        """
        Collect one-hot target indices from all nodes in infix order.

        Returns:
            Stacked target tensor of shape ``(batch, n_nodes)`` with ``-1`` for padding.
        """
        targets = self.get_target_rec()
        return torch.stack(targets, dim=1)

    def get_target_rec(self) -> List[torch.Tensor]:
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
    """
    Hierarchical Variational Autoencoder (HVAE) for expression trees.

    Combines a tree-recursive [Encoder][SRToolkit.approaches.EDHiE.Encoder] with a
    tree-recursive [Decoder][SRToolkit.approaches.EDHiE.Decoder] to learn a continuous
    latent representation of symbolic expressions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        symbol_library: SymbolLibrary,
        hidden_size: Optional[int] = None,
        max_height: int = 20,
    ):
        """
        Args:
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

    def forward(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor, BatchedNode]:
        """
        Run the full VAE forward pass: encode, sample latent vector, decode.

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
        """
        Draw a latent sample using the reparameterisation trick.

        Args:
            mu: Mean of the approximate posterior, shape ``(batch, latent_size)``.
            logvar: Log-variance of the approximate posterior, same shape.

        Returns:
            Sampled latent vector ``z = mu + eps * exp(logvar / 2)``.
        """
        eps = torch.randn_like(mu)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of trees to posterior parameters without sampling.

        Args:
            tree: Batched expression trees.

        Returns:
            A 2-tuple ``(mu, logvar)`` — the approximate posterior mean and log-variance.
        """
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> List[Optional[Node]]:
        """
        Decode a batch of latent vectors to expression trees (inference mode).

        Args:
            z: Latent vectors of shape ``(batch, latent_size)``.

        Returns:
            A list of [Node][SRToolkit.utils.expression_tree.Node] trees (or ``None`` for failed
            decodes), one per row of ``z``.
        """
        return self.decoder.decode(z)


class Encoder(nn.Module):
    """
    Tree-recursive GRU encoder that maps a [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode]
    tree to approximate posterior parameters ``(mu, logvar)``.

    Uses a [GRU221][SRToolkit.approaches.EDHiE.GRU221] cell to combine left/right child hidden
    states bottom-up, then projects to ``mu`` and ``logvar`` via linear layers.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Args:
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

    def forward(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batched tree to posterior parameters.

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
        """
        Recursively encode a subtree bottom-up using the GRU cell.

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
    """
    Tree-recursive GRU decoder that maps a latent vector to an expression tree.

    Uses a [GRU122][SRToolkit.approaches.EDHiE.GRU122] cell to split each parent hidden state
    top-down into left and right child hidden states, then predicts the symbol at each node.
    """

    leaf_symbols_mask: torch.Tensor

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, symbol_library: SymbolLibrary, max_height: int = 20
    ):
        """
        Args:
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
            if self.symbol_library.get_type(s) in ["var", "const", "lit"]:
                leaf_symbols_mask[i] = 1
        self.register_buffer("leaf_symbols_mask", leaf_symbols_mask)

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, z: torch.Tensor, tree: BatchedNode) -> BatchedNode:
        """
        Teacher-forced decoding pass used during training.

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
        """
        Recursively write predictions into a subtree (teacher-forced, training only).

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

    def decode(self, z: torch.Tensor) -> List[Optional[Node]]:
        """
        Autoregressively decode latent vectors to expression trees (inference mode).

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
        """
        Recursively decode hidden states into a batched subtree (inference mode).

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
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Greedily select one symbol per batch element and compute child masks.

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
                if self.symbol_library.get_type(symbol) == "fn":
                    right_mask[i] = False
                if self.symbol_library.get_type(symbol) in ["var", "const", "lit"]:
                    left_mask[i] = False
                    right_mask[i] = False
            else:
                symbols.append("")
        return symbols, left_mask, right_mask


class GRU221(nn.Module):
    """
    Two-input, one-output GRU cell used by the [Encoder][SRToolkit.approaches.EDHiE.Encoder].

    Combines an input vector with two hidden states (left child and right child) into a
    single parent hidden state. The two child states are concatenated before being passed
    through the standard GRU gating equations.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
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
        """
        Compute parent hidden state from input and two child hidden states.

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
    """
    One-input, two-output GRU cell used by the [Decoder][SRToolkit.approaches.EDHiE.Decoder].

    Splits a parent hidden state into two child hidden states (left and right) driven by
    an input vector. The output is split evenly along the hidden dimension.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Args:
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

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute two child hidden states from input and parent hidden state.

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


def create_batch(trees: List[Optional[Node]], symbol2index: Dict[str, int]) -> BatchedNode:
    """
    Pack a list of expression trees into a single [BatchedNode][SRToolkit.approaches.EDHiE.BatchedNode]
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


class TreeBatchSampler(Sampler[List[int]]):
    """
    PyTorch sampler that yields randomly permuted mini-batches of tree indices.

    Each call to ``__iter__`` produces a fresh permutation. Batches are contiguous slices
    of the permutation; the last incomplete batch is dropped.
    """

    def __init__(self, batch_size: int, num_eq: int):
        """
        Args:
            batch_size: Number of trees per mini-batch.
            num_eq: Total number of trees in the dataset.
        """
        self.batch_size = batch_size
        self.num_eq = num_eq

    def __iter__(self) -> Iterator[List[int]]:
        permutation = np.random.permutation(self.num_eq)
        for i in range(len(self)):
            yield permutation[i * self.batch_size : (i + 1) * self.batch_size].tolist()

    def __len__(self) -> int:
        return self.num_eq // self.batch_size


class TreeDataset(Dataset):
    """
    Minimal PyTorch ``Dataset`` wrapping a list of expression trees.
    """

    def __init__(self, trees: List[Optional[Node]]):
        """
        Args:
            trees: List of [Node][SRToolkit.utils.expression_tree.Node] expression trees.
        """
        self.trees: List[Node] = [t for t in trees if isinstance(t, Node)]

    def __getitem__(self, idx: int) -> Node:
        return self.trees[idx]

    def __len__(self) -> int:
        return len(self.trees)


def annealing_schedule(it: int, total_iters: int, supremum: float = 0.04) -> float:
    """
    Linear KL annealing schedule mapping iteration count to a ``[0, supremum]`` coefficient.

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the HVAE ELBO loss: reconstruction (cross-entropy) + KL divergence.

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
    """
    Train an [HVAE][SRToolkit.approaches.EDHiE.HVAE] on a dataset of expression trees.

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


class _HVAEProblem(Problem):
    """
    pymoo ``Problem`` that wraps HVAE decoding and SR expression evaluation.

    Each call to ``_evaluate`` decodes a batch of latent vectors to expression trees,
    evaluates them via the [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator],
    and returns the error values as the objective.
    """

    _FALLBACK_SCORE = float("inf")

    def __init__(self, model: HVAE, evaluator: SR_evaluator, dim: int):
        """
        Args:
            model: Trained [HVAE][SRToolkit.approaches.EDHiE.HVAE] used to decode latent vectors.
            evaluator: [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] used to score
                decoded expressions.
            dim: Dimensionality of the latent space (number of decision variables for pymoo).
        """
        self.model = model
        self.evaluator = evaluator
        self.symbol2index = evaluator.symbol_library.symbols2index()
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        device = next(self.model.parameters()).device
        trees = self.model.decode(torch.tensor(x[:, :], device=device, dtype=torch.float32))
        errors = []
        for tree in trees:
            if self.evaluator.should_stop or tree is None:
                errors.append(self._FALLBACK_SCORE)
                continue
            error = self.evaluator.evaluate_expr(tree)
            errors.append(error if not np.isnan(error) else self._FALLBACK_SCORE)
        out["F"] = np.array(errors)


class _TorchNormalSampling(Sampling):
    """
    pymoo ``Sampling`` that initialises the GA population by drawing samples from a
    standard normal distribution in latent space.
    """

    def _do(self, problem: _HVAEProblem, n_samples: int, **kwargs) -> np.ndarray:
        """
        Args:
            problem: The [_HVAEProblem][SRToolkit.approaches.EDHiE._HVAEProblem].
            n_samples: Number of individuals to generate.

        Returns:
            Array of shape ``(n_samples, n_var)`` sampled from ``N(0, 1)`` in latent space.
        """
        return np.random.standard_normal((n_samples, problem.n_var))


class _BestTermination(Termination):
    """
    pymoo ``Termination`` that stops the GA when the evaluator signals early stopping
    (success threshold reached or budget exhausted).
    """

    def _update(self, algorithm):
        if algorithm.problem.evaluator.should_stop:
            self.terminate()
        return 0.0


class _LICrossover(Crossover):
    """
    Linear interpolation (LI) crossover operator for latent vectors.

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
    """
    Latent-space mutation that re-encodes individuals and perturbs them proportionally to
    the posterior variance learned by the HVAE encoder.

    Each individual is decoded to an expression tree, re-encoded to obtain the posterior
    log-variance, and then perturbed with Gaussian noise scaled by the standard deviation.
    """

    def __init__(self):
        """Initialises the mutation operator."""
        super().__init__()

    def _do(self, problem: _HVAEProblem, X, **kwargs):
        device = next(problem.model.parameters()).device
        trees = problem.model.decode(torch.tensor(X, device=device, dtype=torch.float32))
        batch = create_batch(trees, problem.symbol2index)
        var = problem.model.encode(batch)[1].detach().cpu().numpy()
        mutation_scale = np.random.random((X.shape[0], 1))
        std = np.multiply(mutation_scale, (np.exp(var / 2.0) - 1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)
