"""
Distance and similarity measures between symbolic expressions: edit distance,
tree edit distance, and Behavioral Embedding Distance (BED).
"""

from typing import List, Optional, Tuple, Union

import editdistance
import numpy as np
import zss
from scipy.stats.qmc import LatinHypercube

from .expression_compiler import compile_expr
from .expression_tree import Node, tokens_to_tree
from .symbol_library import SymbolLibrary


def edit_distance(
    expr1: Union[List[str], Node],
    expr2: Union[List[str], Node],
    notation: str = "postfix",
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> int:
    """
    Compute the edit distance between two expressions.

    Both expressions are first converted to the requested notation, so the result
    is independent of whether a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree is passed.
    Levenshtein distance is then computed on the serialised token sequences.

    Examples:
        >>> edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"])
        0
        >>> edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> edit_distance(tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1)), tokens_to_tree(["X_0", "-", "1"], SymbolLibrary.default_symbols(1)))
        1

    Args:
        expr1: First expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        expr2: Second expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        notation: Notation used for comparison: ``"infix"``, ``"prefix"``, or
            ``"postfix"``. Defaults to ``"postfix"`` to avoid parenthesis artefacts.
        symbol_library: Symbol library used when converting expressions to the target
            notation. Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        Integer edit distance between the two serialised expressions.
    """
    if isinstance(expr1, Node):
        expr1 = expr1.to_list(symbol_library=symbol_library, notation=notation)
    elif isinstance(expr1, list):
        expr1 = tokens_to_tree(expr1, symbol_library).to_list(symbol_library=symbol_library, notation=notation)

    if isinstance(expr2, Node):
        expr2 = expr2.to_list(symbol_library=symbol_library, notation=notation)
    elif isinstance(expr2, list):
        expr2 = tokens_to_tree(expr2, symbol_library).to_list(symbol_library=symbol_library, notation=notation)

    return editdistance.eval(expr1, expr2)


def _expr_to_zss(expr: Node) -> zss.Node:
    """Convert a [Node][SRToolkit.utils.expression_tree.Node] tree to a ``zss.Node`` tree for Zhang-Shasha distance computation."""
    zexpr = zss.Node(expr.symbol)
    if expr.left is not None:
        zexpr.addkid(_expr_to_zss(expr.left))
    if expr.right is not None:
        zexpr.addkid(_expr_to_zss(expr.right))

    return zexpr


def tree_edit_distance(
    expr1: Union[Node, List[str]],
    expr2: Union[Node, List[str]],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> int:
    """
    Compute the Zhang-Shasha tree edit distance between two expressions.

    Unlike [edit_distance][SRToolkit.utils.measures.edit_distance], which operates on flattened
    token sequences, tree edit distance considers the expression's hierarchical structure.
    The cost is the minimum number of node insertions, deletions, and relabellings needed
    to transform one tree into the other.

    Examples:
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"])
        0
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> tree_edit_distance(tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1)), tokens_to_tree(["X_0", "-", "1"], SymbolLibrary.default_symbols(1)))
        1

    Args:
        expr1: First expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        expr2: Second expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used when converting token lists to trees.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        Integer tree edit distance.
    """
    if isinstance(expr1, Node):
        zss1 = _expr_to_zss(expr1)
    elif isinstance(expr1, list):
        zss1 = _expr_to_zss(tokens_to_tree(expr1, symbol_library))

    if isinstance(expr2, Node):
        zss2 = _expr_to_zss(expr2)
    elif isinstance(expr2, list):
        zss2 = _expr_to_zss(tokens_to_tree(expr2, symbol_library))

    return int(zss.simple_distance(zss1, zss2))


def create_behavior_matrix(
    expr: Union[Node, List[str]],
    X: np.ndarray,
    num_consts_sampled: int = 32,
    consts_bounds: Tuple[float, float] = (-5, 5),
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate an expression over multiple constant samples to produce a behavior matrix.

    For expressions with free constants, constants are drawn via Latin Hypercube Sampling
    within ``consts_bounds``. For constant-free expressions, all columns are identical.

    Examples:
        >>> X = np.random.rand(10, 2) - 0.5
        >>> create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32).shape
        (10, 32)
        >>> mean_0_1 = np.mean(create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32, consts_bounds=(0, 1)))
        >>> mean_1_5 = np.mean(create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32, consts_bounds=(1, 5)))
        >>> print(bool(mean_0_1 < mean_1_5))
        True
        >>> # Deterministic expressions always produce the same behavior matrix
        >>> bm1 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        >>> bm2 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        >>> print(bool(np.array_equal(bm1, bm2)))
        True

    Args:
        expr: Expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        X: Input data of shape ``(n_samples, n_features)`` at which the expression is
            evaluated.
        num_consts_sampled: Number of constant vectors to sample; sets the number of
            output columns. Default ``32``.
        consts_bounds: ``(lower, upper)`` bounds for constant sampling. Default ``(-5, 5)``.
        symbol_library: Symbol library used to compile the expression. Defaults to
            [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        seed: Random seed for reproducible constant sampling. Default ``None``.

    Returns:
        Behavior matrix of shape ``(n_samples, num_consts_sampled)``.

    Raises:
        Exception: If ``expr`` is neither a token list nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.default_symbols()
    const_symbols = symbol_library.get_symbols_of_type("const")

    if isinstance(expr, list):
        tokens = expr
    elif isinstance(expr, Node):
        tokens = expr.to_list(notation="postfix")
    else:
        raise TypeError("Expression must be given as a list of strings or a Node tree.")

    num_constants = sum(tokens.count(c) for c in const_symbols)

    callable_expr = compile_expr(expr, symbol_library, backend="stack")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if num_constants > 0:
            lho = LatinHypercube(num_constants, rng=seed)
            constants = lho.random(num_consts_sampled) * (consts_bounds[1] - consts_bounds[0]) + consts_bounds[0]
            ys = []
            for c in constants:
                ys.append(callable_expr(X, c))
            return np.array(ys).T
        else:
            return np.repeat(callable_expr(X, None)[:, None], num_consts_sampled, axis=1)


def _custom_wasserstein(u: np.ndarray, v: np.ndarray):
    """
    Compute the 1-D Wasserstein distance between two sample arrays.

    Uses a direct CDF comparison over the sorted union of values, without scipy.

    Args:
        u: First sample array. Must be non-empty.
        v: Second sample array. Must be non-empty.

    Returns:
        Wasserstein distance as a float.
    """
    u = np.sort(u)
    v = np.sort(v)
    all_values = np.sort(np.concatenate((u, v)))
    deltas = np.diff(all_values)
    u_cdf_indices = u.searchsorted(all_values[:-1], "right")
    v_cdf_indices = v.searchsorted(all_values[:-1], "right")
    u_cdf = u_cdf_indices / u.size
    v_cdf = v_cdf_indices / v.size
    return np.sum(np.abs(u_cdf - v_cdf) * deltas)


def _vectorized_wasserstein_batch(expr1: np.ndarray, expr2: np.ndarray) -> np.ndarray:
    """
    Compute per-row Wasserstein distances between two fully-finite behavior matrices.

    Both matrices must have identical shape ``(n_rows, k)`` with no NaN or inf values.
    Processes data in memory-bounded chunks so that peak usage stays ≤ ~50 MB.

    Args:
        expr1: Behavior matrix, shape ``(n_rows, k)``.
        expr2: Behavior matrix, shape ``(n_rows, k)``.

    Returns:
        1-D array of shape ``(n_rows,)`` containing the Wasserstein distance for each row.
    """
    n, k = expr1.shape
    u_s = np.sort(expr1, axis=1)  # (n, k)
    v_s = np.sort(expr2, axis=1)  # (n, k)
    all_vals = np.sort(np.concatenate([u_s, v_s], axis=1), axis=1)  # (n, 2k)
    deltas = np.diff(all_vals, axis=1)  # (n, 2k-1)

    m = 2 * k - 1  # number of CDF evaluation points
    # Limit intermediate (n_chunk × k × m) boolean array to ≈50 MB
    max_chunk = max(1, 50_000_000 // (k * m))

    wds = np.empty(n, dtype=np.float64)
    for start in range(0, n, max_chunk):
        end = min(start + max_chunk, n)
        cu = u_s[start:end]  # (chunk, k)
        cv = v_s[start:end]  # (chunk, k)
        ca = all_vals[start:end, :-1]  # (chunk, m)
        # Broadcasting: (chunk, k, 1) <= (chunk, 1, m) → (chunk, k, m); sum over k axis
        u_cdf = np.sum(cu[:, :, None] <= ca[:, None, :], axis=1) / k  # (chunk, m)
        v_cdf = np.sum(cv[:, :, None] <= ca[:, None, :], axis=1) / k  # (chunk, m)
        wds[start:end] = np.sum(np.abs(u_cdf - v_cdf) * deltas[start:end], axis=1)

    return wds


def bed(
    expr1: Union[Node, List[str], np.ndarray],
    expr2: Union[Node, List[str], np.ndarray],
    X: Optional[np.ndarray] = None,
    num_consts_sampled: int = 32,
    num_points_sampled: int = 64,
    domain_bounds: Optional[List[Tuple[float, float]]] = None,
    consts_bounds: Tuple[float, float] = (-5, 5),
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    seed: Optional[int] = None,
) -> float:
    """
    Compute the Behavior-aware Expression Distance (BED) between two expressions.

    BED measures how similarly two expressions behave over a domain by comparing
    their output distributions point-by-point using the Wasserstein distance. Free
    constants are marginalised by sampling multiple constant vectors via Latin
    Hypercube Sampling.

    Either ``X`` or ``domain_bounds`` must be provided when expressions are given as
    token lists or [Node][SRToolkit.utils.expression_tree.Node] trees. Pre-computed behavior matrices can be passed
    directly to avoid redundant evaluation.

    Examples:
        >>> X = np.random.rand(10, 2) - 0.5
        >>> expr1 = ["X_0", "+", "C"] # instances of SRToolkit.utils.expression_tree.Node work as well
        >>> expr2 = ["X_1", "+", "C"]
        >>> bed(expr1, expr2, X) < 1
        True
        >>> # Changing the number of sampled constants
        >>> bed(expr1, expr2, X, num_consts_sampled=128, consts_bounds=(-2, 2)) < 1
        True
        >>> # Sampling X instead of giving it directly by defining a domain
        >>> bed(expr1, expr2, domain_bounds=[(0, 1), (0, 1)]) < 1
        True
        >>> bed(expr1, expr2, domain_bounds=[(0, 1), (0, 1)], num_points_sampled=128) < 1
        True
        >>> # You can use behavior matrices instead of expressions (this has potential computational advantages if same expression is used multiple times)
        >>> bm1 = create_behavior_matrix(expr1, X)
        >>> bed(bm1, expr2, X) < 1
        True
        >>> bm2 = create_behavior_matrix(expr2, X)
        >>> bed(bm1, bm2) < 1
        True

    Args:
        expr1: First expression as a token list, a [Node][SRToolkit.utils.expression_tree.Node] tree, or a pre-computed
            behavior matrix of shape ``(n_samples, num_consts_sampled)``.
        expr2: Second expression in the same format as ``expr1``.
        X: Evaluation points of shape ``(n_samples, n_features)``. Required unless both
            expressions are behavior matrices or ``domain_bounds`` is provided.
        num_consts_sampled: Number of constant vectors sampled per expression. Default ``32``.
        num_points_sampled: Number of points sampled from ``domain_bounds`` when ``X`` is
            ``None``. Default ``64``.
        domain_bounds: Per-variable ``(lower, upper)`` bounds used to sample ``X`` via
            Latin Hypercube Sampling when ``X`` is ``None``.
        consts_bounds: ``(lower, upper)`` bounds for constant sampling. Default ``(-5, 5)``.
        symbol_library: Symbol library used to compile expressions. Defaults to
            [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        seed: Random seed for reproducible sampling. Default ``None``.

    Returns:
        BED between the expressions as a non-negative float. A value of ``0.0``
        indicates identical behavior over the sampled domain; larger values indicate
        greater behavioral dissimilarity. Returns ``inf`` if any evaluation point
        produces finite outputs for one expression but not the other.

    Raises:
        Exception: If ``X`` is ``None`` and neither ``domain_bounds`` is provided nor
            both expressions are pre-computed behavior matrices.
        Exception: If ``X`` is ``None`` and exactly one expression is a pre-computed
            behavior matrix (the two matrices would be over different domains).
        ValueError: If any entry in ``domain_bounds`` has a lower bound greater than
            or equal to its upper bound.
        ValueError: If the two behavior matrices have different numbers of rows.
        ValueError: If the behavior matrices have zero rows.
    """

    if X is None and not isinstance(expr1, np.ndarray) and not isinstance(expr2, np.ndarray):
        if domain_bounds is None:
            raise Exception(
                "If X is not given and both expressions are not given as a behavior matrix, "
                "then domain_bounds parameter must be given"
            )
        for i, (lb, ub) in enumerate(domain_bounds):
            if lb >= ub:
                raise ValueError(f"domain_bounds[{i}] has lower bound ({lb}) >= upper bound ({ub}).")
        interval_length = np.array([ub - lb for (lb, ub) in domain_bounds])
        lower_bound = np.array([lb for (lb, ub) in domain_bounds])
        lho = LatinHypercube(len(domain_bounds), optimization="random-cd", rng=seed)
        X = lho.random(num_points_sampled) * interval_length + lower_bound
    elif X is None and (isinstance(expr1, np.ndarray) != isinstance(expr2, np.ndarray)):
        raise Exception(
            "If X is not given, both expressions must be given as a behavior matrix or as a list of "
            "tokens/SRToolkit.utils.Node objects. Otherwise, behavior matrices are uncomparable."
        )

    if isinstance(expr1, list) or isinstance(expr1, Node):
        assert X is not None
        expr1 = create_behavior_matrix(expr1, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    if isinstance(expr2, list) or isinstance(expr2, Node):
        assert X is not None
        expr2 = create_behavior_matrix(expr2, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    if expr1.shape[0] != expr2.shape[0]:
        raise ValueError("Behavior matrices must have the same number of rows (points on which behavior is evaluated).")
    if expr1.shape[0] == 0:
        raise ValueError(
            "Behavior matrices must have at least one row. If your expressions are given as behavior "
            "matrices, make sure they are not empty. Otherwise, if X is given, make sure it contains "
            "at least one point. If X is not given, make sure num_points_sampled is greater than 0."
        )
    n = expr1.shape[0]

    finite_any1 = np.any(np.isfinite(expr1), axis=1)
    finite_any2 = np.any(np.isfinite(expr2), axis=1)

    if np.any(finite_any1 ^ finite_any2):
        return np.inf

    active = finite_any1 & finite_any2
    e1, e2 = expr1[active], expr2[active]

    if e1.shape[0] == 0:
        return 0.0

    if np.any(np.isinf(e1)) or np.any(np.isinf(e2)):
        return np.inf

    has_nan = ~(np.all(np.isfinite(e1), axis=1) & np.all(np.isfinite(e2), axis=1))
    wds = np.empty(e1.shape[0])

    if np.any(~has_nan):
        wds[~has_nan] = _vectorized_wasserstein_batch(e1[~has_nan], e2[~has_nan])

    for i in np.where(has_nan)[0]:
        wds[i] = _custom_wasserstein(e1[i][~np.isnan(e1[i])], e2[i][~np.isnan(e2[i])])

    return float(np.sum(wds) / n)
