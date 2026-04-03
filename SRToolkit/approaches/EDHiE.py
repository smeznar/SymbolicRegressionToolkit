"""
This module contains the EDHiE (Equation Discovery with Hierarchical variational autoEncoders) approach by Mežnar et. al.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

from SRToolkit.approaches.sr_approach import ApproachConfig, SR_approach, check_dependencies
from SRToolkit.dataset import SR_benchmark, SR_dataset
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils import Node, SymbolLibrary, generate_n_expressions, tokens_to_tree

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import Dataset, Sampler
    from tqdm import tqdm
except ImportError:
    raise ImportError("PyTorch is not installed.")


class EDHiE(SR_approach):
    r"""
    EDHiE (Equation Discovery with Hierarchical variational autoEncoders) approach by Mežnar et. al.

    The approach trains a Hierarchical VAE (HVAE) on randomly generated expressions from the symbol
    library, then explores the learned latent space using a genetic algorithm to find expressions
    that best fit the dataset.

    Args:
        latent_size: Dimensionality of the HVAE latent space.
        num_expressions: Number of random expressions to generate for HVAE training.
        max_expression_length: Maximum token length of generated training expressions.
        epochs: Number of training epochs for the HVAE.
        batch_size: Batch size used during HVAE training.
        max_beta: Maximum value of the KL annealing coefficient during HVAE training.
        population_size: GA population size used during latent space search.
        max_generations: Maximum number of GA generations during search.
        verbose: If True, prints training and search progress.
    """

    def __init__(
        self,
        latent_size: int = 24,
        num_expressions: int = 20000,
        max_expression_length: int = 30,
        epochs: int = 20,
        batch_size: int = 32,
        max_beta: float = 0.035,
        population_size: int = 40,
        max_generations: int = 250,
        verbose: bool = True,
    ) -> None:
        super().__init__("EDHiE")
        check_dependencies(["pytorch", "pymoo"])
        self.latent_size = latent_size
        self.num_expressions = num_expressions
        self.max_expression_length = max_expression_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_beta = max_beta
        self.population_size = population_size
        self.max_generations = max_generations
        self.verbose = verbose

        self.model: Optional[HVAE] = None

    @property
    def adaptation_scope(self) -> str:
        return "once"

    def prepare(self) -> None:
        """
        Resets per-experiment state. The trained HVAE weights are preserved;
        only the evolutionary search is re-initialised on each call to search().
        """
        pass

    def adapt(self, X: np.ndarray, symbol_library: SymbolLibrary) -> None:
        """
        Trains the HVAE on randomly generated expressions from the symbol library.

        Args:
            X: Input data from the domain (used only to determine variable count via the symbol library).
            symbol_library: The symbol library defining available tokens.
        """
        expressions = generate_n_expressions(
            symbol_library,
            self.num_expressions,
            max_expression_length=self.max_expression_length,
        )
        expr_trees = [tokens_to_tree(expr, symbol_library) for expr in expressions]
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

    def save_adapted_state(self) -> Any:
        """
        Returns the trained HVAE model so the framework can cache it in memory.
        """
        return self.model

    def load_adapted_state(self, state: Any) -> None:
        """
        Restores a previously trained HVAE model from the cached state.

        Args:
            state: The HVAE model returned by save_adapted_state().
        """
        self.model = state

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None) -> None:
        """
        Explores the HVAE latent space with a genetic algorithm to find the best expression.

        Args:
            sr_evaluator: The evaluator used to score candidate expressions.
            seed: Optional random seed for reproducibility.
        """
        if self.model is None:
            raise RuntimeError("EDHiE.adapt() must be called before search().")

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

        success_threshold = sr_evaluator.success_threshold if sr_evaluator.success_threshold is not None else 1e-10

        problem = SRProblem(self.model, sr_evaluator, self.latent_size, verbose=self.verbose)
        ga = GA(
            pop_size=self.population_size,
            sampling=TorchNormalSampling(),
            crossover=LICrossover(),
            mutation=RandomMutation(),
            eliminate_duplicates=False,
        )
        minimize(
            problem,
            ga,
            BestTermination(min_f=success_threshold, n_max_gen=self.max_generations),
            verbose=False,
        )


@dataclass
class EDHiEConfig(ApproachConfig):
    """Configuration for EDHiE approach."""

    name: str = "EDHiE"
    latent_size: int = 24
    num_expressions: int = 20000
    max_expression_length: int = 30
    epochs: int = 20
    batch_size: int = 32
    max_beta: float = 0.035
    population_size: int = 40
    max_generations: int = 250
    verbose: bool = True


class BatchedNode:
    def __init__(self, symbol2index: Dict[str, int], size: int = 0, trees: Union[None, List[Node]] = None):
        self.symbols: List[str] = ["" for _ in range(size)]
        self.left: Union[BatchedNode, None] = None
        self.right: Union[BatchedNode, None] = None
        self.symbol2index: Dict[str, int] = symbol2index
        self.mask: Union[None, torch.Tensor] = None
        self.target: Union[None, torch.Tensor] = None
        self.prediction: Union[None, torch.Tensor] = None

        if trees is not None:
            for tree in trees:
                self.add_tree(tree)

    def add_tree(self, tree=None):
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

            # Add left subtree
            if isinstance(self.left, BatchedNode) and isinstance(tree.left, Node):
                self.left.add_tree(tree.left)
            # Add empty subtree to the right batched node
            elif isinstance(self.left, BatchedNode):
                self.left.add_tree()
            # Add new batched nodes to the left
            elif isinstance(tree.left, Node):
                self.left = BatchedNode(self.symbol2index, size=len(self.symbols) - 1)
                self.left.add_tree(tree.left)

            # Add right subtree
            if isinstance(self.right, BatchedNode) and isinstance(tree.right, Node):
                self.right.add_tree(tree.right)
            # Add empty subtree to the right batched node
            elif isinstance(self.right, BatchedNode):
                self.right.add_tree()
            # Add new batched node to the right
            elif isinstance(tree.right, Node):
                self.right = BatchedNode(self.symbol2index, size=len(self.symbols) - 1)
                self.right.add_tree(tree.right)

    def loss(
        self, mu: torch.Tensor, logvar: torch.Tensor, lmbda: float, criterion
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = self.get_prediction()
        target = self.get_target()
        BCE = criterion(pred, target)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size(0)
        return BCE + lmbda * KLD, BCE, KLD

    def create_target(self):
        target = torch.zeros((len(self.symbols), len(self.symbol2index)))
        mask = torch.ones(len(self.symbols))

        for i, s in enumerate(self.symbols):
            if s == "":
                mask[i] = 0
            else:
                target[i, self.symbol2index[s]] = 1

        self.mask = mask
        self.target = Variable(target)

        if self.left is not None:
            self.left.create_target()
        if self.right is not None:
            self.right.create_target()

    def to_expr_list(self) -> List[Optional[Node]]:
        expressions = []
        for i in range(len(self.symbols)):
            expressions.append(self.get_expr_at_idx(i))
        return expressions

    def get_expr_at_idx(self, idx: int) -> Optional[Node]:
        symbol = self.symbols[idx]
        if symbol == "":
            return None

        left = self.left.get_expr_at_idx(idx) if isinstance(self.left, BatchedNode) else None
        right = self.right.get_expr_at_idx(idx) if isinstance(self.right, BatchedNode) else None

        return Node(symbol, left=left, right=right)

    def get_prediction(self) -> torch.Tensor:
        predictions = self.get_prediction_rec()
        return torch.stack(predictions, dim=2)

    def get_prediction_rec(self) -> List[torch.Tensor]:
        if self.prediction is None:
            return []

        reps = []
        if isinstance(self.left, BatchedNode):
            reps += self.left.get_prediction_rec()

        reps.append(self.prediction)

        if self.right is not None:
            reps += self.right.get_prediction_rec()

        return reps

    def get_target(self) -> torch.Tensor:
        targets = self.get_target_rec()
        return torch.stack(targets, dim=1)

    def get_target_rec(self) -> List[torch.Tensor]:
        reps = []
        if isinstance(self.left, BatchedNode):
            reps += self.left.get_target_rec()

        target = torch.zeros(len(self.symbols)).long()
        for i, s in enumerate(self.symbols):
            if s == "":
                target[i] = -1
            else:
                target[i] = self.symbol2index[s]
        reps.append(target)

        if isinstance(self.right, BatchedNode):
            reps += self.right.get_target_rec()

        return reps


class HVAE(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        symbol_library: SymbolLibrary,
        hidden_size: Union[None, int] = None,
        max_height: int = 20,
    ):
        super(HVAE, self).__init__()

        if hidden_size is None:
            hidden_size = output_size

        self.encoder = Encoder(input_size, hidden_size, output_size)
        self.decoder = Decoder(output_size, hidden_size, input_size, symbol_library, max_height)

    def forward(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor, BatchedNode]:
        mu, logvar = self.encoder(tree)
        z = self.sample(mu, logvar)
        out = self.decoder(z, tree)
        return mu, logvar, out

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = Variable(torch.randn(mu.size()))
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def encode(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(tree)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> List[Optional[Node]]:
        return self.decoder.decode(z)


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = GRU221(input_size=input_size, hidden_size=hidden_size)
        self.mu = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.logvar = nn.Linear(in_features=hidden_size, out_features=output_size)

        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.logvar.weight)

    def forward(self, tree: BatchedNode) -> Tuple[torch.Tensor, torch.Tensor]:
        tree_encoding = self.recursive_forward(tree)
        mu = self.mu(tree_encoding)
        logvar = self.logvar(tree_encoding)
        return mu, logvar

    def recursive_forward(self, tree: BatchedNode) -> torch.Tensor:
        if isinstance(tree.left, BatchedNode):
            h_left = self.recursive_forward(tree.left)
        else:
            h_left = torch.zeros(len(tree.symbols), self.hidden_size)

        if isinstance(tree.right, BatchedNode):
            h_right = self.recursive_forward(tree.right)
        else:
            h_right = torch.zeros(len(tree.symbols), self.hidden_size)

        hidden = self.gru(tree.target, h_left, h_right)
        assert tree.mask is not None, "Mask is not defined"
        hidden = hidden.mul(tree.mask[:, None])
        return hidden


class Decoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, symbol_library: SymbolLibrary, max_height: int = 20
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.z2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.gru = GRU122(input_size=output_size, hidden_size=hidden_size)

        self.max_height = max_height
        self.symbol_library = symbol_library
        self.symbol2index = symbol_library.symbols2index()
        self.index2symbol = {i: s for s, i in self.symbol2index.items()}
        self.leaf_symbols_mask = torch.zeros(len(self.symbol2index))
        for s, i in self.symbol2index.items():
            if self.symbol_library.get_type(s) in ["var", "const", "lit"]:
                self.leaf_symbols_mask[i] = 1

        torch.nn.init.xavier_uniform_(self.z2h.weight)
        torch.nn.init.xavier_uniform_(self.h2o.weight)

    # Used during training to guide the learning process
    def forward(self, z: torch.Tensor, tree: BatchedNode) -> BatchedNode:
        hidden = self.z2h(z)
        self.recursive_forward(hidden, tree)
        return tree

    def recursive_forward(self, hidden: torch.Tensor, tree: BatchedNode):
        prediction = self.h2o(hidden)
        tree.prediction = prediction
        symbol_probs = F.softmax(prediction, dim=1)
        if isinstance(tree.left, BatchedNode) or isinstance(tree.right, BatchedNode):
            left, right = self.gru(symbol_probs, hidden)
            if isinstance(tree.left, BatchedNode):
                self.recursive_forward(left, tree.left)
            if isinstance(tree.right, BatchedNode):
                self.recursive_forward(right, tree.right)

    # Used for inference to generate expression trees from latent vectorS
    def decode(self, z: torch.Tensor) -> List[Optional[Node]]:
        with torch.no_grad():
            mask = torch.ones(z.size(0)).bool()
            hidden = self.z2h(z)
            batch = self.recursive_decode(hidden, mask)
            return batch.to_expr_list()

    def recursive_decode(self, hidden: torch.Tensor, mask: torch.Tensor, height: int = 0) -> BatchedNode:
        # print(height)
        prediction = F.softmax(self.h2o(hidden), dim=1)
        # Sample symbol in a given node
        symbols, left_mask, right_mask = self.sample_symbol(prediction, mask, height)
        left, right = self.gru(prediction, hidden)
        if torch.any(left_mask):
            l_tree = self.recursive_decode(left, left_mask, height + 1)
        else:
            l_tree = None

        if torch.any(right_mask):
            r_tree = self.recursive_decode(right, right_mask, height + 1)
        else:
            r_tree = None

        node = BatchedNode(self.symbol2index)
        node.symbols = symbols
        node.left = l_tree
        node.right = r_tree
        return node

    def sample_symbol(
        self, prediction: torch.Tensor, mask: torch.Tensor, height: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        # Select the symbol with the highest value ("probability")
        symbols = []
        left_mask = torch.clone(mask)
        right_mask = torch.clone(mask)

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
    def __init__(self, input_size: int, hidden_size: int):
        super(GRU221, self).__init__()
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
        h = torch.cat([h1, h2], dim=1)
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        return (1 - z) * n + (z / 2) * h1 + (z / 2) * h2


class GRU122(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(GRU122, self).__init__()
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

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        r = torch.sigmoid(self.wir(x) + self.whr(h))
        z = torch.sigmoid(self.wiz(x) + self.whz(h))
        n = torch.tanh(self.win(x) + r * self.whn(h))
        dh = h.repeat(1, 2)
        out = (1 - z) * n + z * dh
        return torch.split(out, self.hidden_size, dim=1)


def create_batch(trees, symbol2index):
    t = BatchedNode(symbol2index, trees=trees)
    t.create_target()
    return t


class TreeBatchSampler(Sampler):
    def __init__(self, batch_size, num_eq):
        self.batch_size = batch_size
        self.num_eq = num_eq
        self.permute = np.random.permutation(self.num_eq)

    def __iter__(self):
        for i in range(len(self)):
            batch = self.permute[(i * self.batch_size) : ((i + 1) * self.batch_size)]
            yield batch

    def __len__(self):
        return self.num_eq // self.batch_size


class TreeDataset(Dataset):
    def __init__(self, train):
        self.train = train

    def __getitem__(self, idx):
        return self.train[idx]

    def __len__(self):
        return len(self.train)


def logistic_function(it, total_iters, supremum=0.04):
    x = it / total_iters
    return x * supremum


def train_hvae(model, trainset, symbol_library, epochs=20, batch_size=32, max_beta=0.04, verbose=True):
    symbol2index = symbol_library.symbols2index()
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(ignore_index=-1, reduction="mean")

    iter_counter = 0
    total_iters = epochs * (len(trainset) // batch_size)
    lmbda = logistic_function(iter_counter, total_iters, max_beta)

    midpoint = len(trainset) // (2 * batch_size)

    for epoch in range(epochs):
        sampler = TreeBatchSampler(batch_size, len(trainset))
        bce, kl, los, num_iters = 0, 0, 0, 0

        with tqdm(total=len(trainset), desc=f"Training HVAE - Epoch: {epoch + 1}/{epochs}", unit="chunks") as prog_bar:
            for i, tree_ids in enumerate(sampler):
                batch = create_batch([trainset[j] for j in tree_ids], symbol2index)

                mu, logvar, outputs = model(batch)
                loss, bcel, kll = outputs.loss(mu, logvar, lmbda, criterion)
                bce += bcel.detach().item()
                kl += kll.detach().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_iters += 1
                los += bcel.detach().item() + lmbda * kll.detach().item()
                prog_bar.set_postfix(
                    **{"run:": "HVAE", "loss": los / num_iters, "BCE": bce / num_iters, "KLD": kl / num_iters}
                )
                prog_bar.update(batch_size)

                lmbda = logistic_function(iter_counter, total_iters, max_beta)
                iter_counter += 1

                if verbose and i == midpoint:
                    original_trees = batch.to_expr_list()
                    z = model.encode(batch)[0]
                    decoded_trees = model.decode(z)
                    for i in range(1):
                        print()
                        print(f"O: {''.join(original_trees[i].to_list(symbol_library=symbol_library))}")
                        print(f"P: {''.join(decoded_trees[i].to_list(symbol_library=symbol_library))}")


class SRProblem(Problem):
    def __init__(self, model, evaluator: SR_evaluator, dim, default_value=1e10, verbose=True):
        self.model = model
        self.default_value = np.array(default_value)
        self.evaluator = evaluator
        self.symbol2index = self.evaluator.symbol_library.symbols2index()
        self.input_mean = torch.zeros(next(model.decoder.parameters()).size(0))
        self.best_f = 9e50
        self.verbose = verbose
        super().__init__(n_var=dim, n_obj=1)

    def _evaluate(self, x, out, *args, **kwargs):
        trees = self.model.decode(torch.tensor(x[:, :]))

        errors = []
        for tree in trees:
            expr = tree.to_list(self.evaluator.symbol_library, "infix")
            error = self.evaluator.evaluate_expr(expr)
            if error < self.best_f:
                self.best_f = error
                if self.verbose:
                    print(f"New best expression with score {error}: {''.join(expr)}")
            errors.append(error)

        out["F"] = np.array(errors)


class TorchNormalSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return [torch.normal(problem.input_mean).numpy() for _ in range(n_samples)]


class BestTermination(Termination):
    def __init__(self, min_f=1e-10, n_max_gen=500) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)

    def _update(self, algorithm):
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        return self.max_gen.update(algorithm)


class LICrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        weights = np.random.random(X.shape[1])
        return (X[0, :] * weights[:, None] + X[1, :] * (1 - weights[:, None]))[None, :, :]


class RandomMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        trees = problem.model.decode(torch.tensor(X))
        batch = create_batch(trees, problem.symbol2index)
        var = problem.model.encode(batch)[1].detach().numpy()
        mutation_scale = np.random.random((X.shape[0], 1))
        std = np.multiply(mutation_scale, (np.exp(var / 2.0) - 1)) + 1
        return np.random.normal(mutation_scale * X, std).astype(np.float32)


def symbolic_regression_run(
    model, approach, dataset: SR_dataset, seed, population_size=40, latent_size=24, max_generations=250, verbose=True
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    evaluator = dataset.create_evaluator()
    if approach == "EDHiE":
        ga = GA(
            pop_size=population_size,
            sampling=TorchNormalSampling(),
            crossover=LICrossover(),
            mutation=RandomMutation(),
            eliminate_duplicates=False,
        )
        problem = SRProblem(model, evaluator, latent_size, verbose)
        minimize(
            problem, ga, BestTermination(min_f=dataset.success_threshold, n_max_gen=max_generations), verbose=verbose
        )

        return evaluator.get_results(top_k=-1)


if __name__ == "__main__":
    dataset = SR_benchmark.feynman("../data/fey_data").create_dataset("I.12.4")
    latent_size = 24
    num_expressions = 20000
    max_beta = 0.035
    max_expression_length = 30
    model_name = "24random"

    # Possibly create a training set or load expressions
    expressions = generate_n_expressions(
        dataset.symbol_library, num_expressions, max_expression_length=max_expression_length
    )
    expr_tree = [tokens_to_tree(expr, dataset.symbol_library) for expr in expressions]
    # Create a training set
    trainset = TreeDataset(expr_tree)

    # Train the model
    model = HVAE(len(dataset.symbol_library), latent_size, dataset.symbol_library)
    train_hvae(model, trainset, dataset.symbol_library, epochs=20, max_beta=max_beta)
    # torch.save(model.state_dict(), f"../params/{model_name}.pt")
