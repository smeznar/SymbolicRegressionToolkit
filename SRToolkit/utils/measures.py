"""
Distance and similarity measures between symbolic expressions: edit distance,
tree edit distance, and Behavioral Embedding Distance (BED).
"""

from typing import List, Optional, Tuple, Union

import editdistance
import numpy as np
import zss
from scipy.stats.qmc import LatinHypercube

from .expression_compiler import expr_to_executable_function
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

    Both expressions are normalised to the requested notation before computing
    Levenshtein distance, making the result independent of input serialisation.

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


def _expr_to_zss(expr):
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
        expr1 = _expr_to_zss(expr1)
    elif isinstance(expr1, list):
        expr1 = _expr_to_zss(tokens_to_tree(expr1, symbol_library))

    if isinstance(expr2, Node):
        expr2 = _expr_to_zss(expr2)
    elif isinstance(expr2, list):
        expr2 = _expr_to_zss(tokens_to_tree(expr2, symbol_library))

    return int(zss.simple_distance(expr1, expr2))


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
        raise Exception("Expression should be a list of strings (tokens) or a Node")

    num_constants = sum(tokens.count(c) for c in const_symbols)

    callable_expr = expr_to_executable_function(expr, symbol_library)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if num_constants > 0:
            lho = LatinHypercube(num_constants, seed=seed)
            constants = lho.random(num_consts_sampled) * (consts_bounds[1] - consts_bounds[0]) + consts_bounds[0]
            ys = []
            for c in constants:
                ys.append(callable_expr(X, c))
            return np.array(ys).T
        else:
            return np.repeat(callable_expr(X, None)[:, None], num_consts_sampled, axis=1)


def _custom_wasserstein(u, v):
    """
    Compute the 1-D Wasserstein distance between two sample arrays.

    Uses a direct CDF comparison over the sorted union of values, without scipy.

    Args:
        u: First sample array.
        v: Second sample array.

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
        BED between the expressions, given as a float.

    Raises:
        Exception: If ``X`` is ``None`` and neither ``domain_bounds`` is provided nor
            both expressions are pre-computed behavior matrices.
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
        lho = LatinHypercube(len(domain_bounds), optimization="random-cd", seed=seed)
        X = lho.random(num_points_sampled) * interval_length + lower_bound
    elif X is None and (isinstance(expr1, np.ndarray) != isinstance(expr2, np.ndarray)):
        raise Exception(
            "If X is not given, both expressions must be given as a behavior matrix or as a list of "
            "tokens/SRToolkit.utils.Node objects. Otherwise, behavior matrices are uncomparable."
        )

    if isinstance(expr1, list) or isinstance(expr1, Node):
        if X is None:
            raise ValueError(
                "Either X must be given, domain_bounds must be given, or both expressions must be given as behavior matrices."
            )
        expr1 = create_behavior_matrix(expr1, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    if isinstance(expr2, list) or isinstance(expr2, Node):
        if X is None:
            raise ValueError(
                "Either X must be given, domain_bounds must be given, or both expressions must be given as behavior matrices."
            )
        expr2 = create_behavior_matrix(expr2, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    if expr1.shape[0] != expr2.shape[0]:
        raise ValueError("Behavior matrices must have the same number of rows (points on which behavior is evaluated).")
    if expr1.shape[0] == 0:
        raise ValueError(
            "Behavior matrices must have at least one row. If your expressions are given as behavior "
            "matrices, make sure they are not empty. Otherwise, if X is given, make sure it contains "
            "at least one point. If X is not given, make sure num_points_sampled is greater than 0."
        )
    wds = []
    for i in range(expr1.shape[0]):
        u = expr1[i][np.isfinite(expr1[i])]
        v = expr2[i][np.isfinite(expr2[i])]
        if u.shape[0] > 0 and v.shape[0] > 0:
            wds.append(_custom_wasserstein(u, v))
        elif u.shape[0] == 0 and v.shape[0] == 0:
            wds.append(0)
        else:
            wds.append(np.inf)

    return float(np.mean(wds))


if __name__ == "__main__":
    expr = ["X_0", "+", "C"]
    expr2 = ["sqrt", "(", "C", ")", "+", "X_0"]
    X = np.random.rand(10, 2) - 0.5
    b = bed(expr, expr2, X)
    print(b)
