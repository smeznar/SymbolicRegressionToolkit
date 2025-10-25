"""
This module contains measures for evaluating the similarity between two expressions.
"""
from typing import List, Union, Tuple, Optional

import numpy as np
import editdistance
import zss
from scipy.stats.qmc import LatinHypercube


from SRToolkit.utils import SymbolLibrary, Node, tokens_to_tree, expr_to_executable_function

def edit_distance(expr1: Union[List[str], Node], expr2: Union[List[str], Node], notation: str="postfix",
                  symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> int:
    """
    Calculates the edit distance between two expressions.

    Examples:
        >>> edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"])
        0
        >>> edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> edit_distance(tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1)), tokens_to_tree(["X_0", "-", "1"], SymbolLibrary.default_symbols(1)))
        1

    Args:
        expr1: Expression given as a list of tokens in the infix notation or as an instance of SRToolkit.utils.expression_tree.Node
        expr2: Expression given as a list of tokens in the infix notation or as an instance of SRToolkit.utils.expression_tree.Node
        notation: The notation in which the distance between the two expressions is computed. Can be one of "infix", "postfix", or "prefix".
            By default, "postfix" is used to avoid inconsistencies that occur because of parenthesis.
        symbol_library: The symbol library to use when converting the expressions to lists of tokens and vice versa. Defaults to SymbolLibrary.default_symbols().

    Returns:
        The edit distance between the two expressions written in a given notation.
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
    zexpr = zss.Node(expr.symbol)
    if expr.left is not None:
        zexpr.addkid(_expr_to_zss(expr.left))
    if expr.right is not None:
        zexpr.addkid(_expr_to_zss(expr.right))

    return zexpr

def tree_edit_distance(expr1: Union[Node, List[str]], expr2: Union[Node, List[str]],
                       symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> int:
    """
    Calculates the tree edit distance between two expressions.

    Examples:
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0", "+", "1"])
        0
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> tree_edit_distance(tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1)), tokens_to_tree(["X_0", "-", "1"], SymbolLibrary.default_symbols(1)))
        1

    Args:
        expr1: Expression given as a list of tokens in the infix notation or as an instance of SRToolkit.utils.expression_tree.Node
        expr2: Expression given as a list of tokens in the infix notation or as an instance of SRToolkit.utils.expression_tree.Node
        symbol_library: Symbol library to use when converting the lists of tokens into an instance of SRToolkit.utils.expression_tree.Node.

    Returns:
        The tree edit distance between the two expressions.
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


def create_behavior_matrix(expr: Union[Node, List[str]], X: np.ndarray, num_consts_sampled: int=32,
                           consts_bounds: Tuple[float, float]=(-5, 5),
                           symbol_library: SymbolLibrary=SymbolLibrary.default_symbols(), seed: int=None) -> np.ndarray:
    """
    Creates a behavior matrix from an expression with free parameters. The shape of the matrix is (X.shape[0], num_consts_sampled).

    Examples:
        >>> X = np.random.rand(10, 2) - 0.5
        >>> create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32).shape
        (10, 32)
        >>> mean_0_1 = np.mean(create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32, consts_bounds=(0, 1)))
        >>> mean_1_5 = np.mean(create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=32, consts_bounds=(1, 5)))
        >>> mean_0_1 < mean_1_5
        True
        >>> # Deterministic expressions always produce the same behavior matrix
        >>> bm1 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        >>> bm2 = create_behavior_matrix(["X_0", "+", "X_1"], X)
        >>> np.array_equal(bm1, bm2)
        True


    Args:
        expr: An expression given as a list of tokens in the infix notation.
        X: Points on which the expression is evaluated to determine the behavior
        num_consts_sampled: Number of sets of constants sampled
        consts_bounds: Bounds between which constant values are sampled
        symbol_library: Symbol library used to transform the expression into an executable function.
        seed: Random seed. If None, generation will be random.

    Raises:
        Exception: If expr is not a list of tokens or an instance of SRToolkit.utils.expression_tree.Node.

    Returns:
        A matrix of size (X.shape[0], num_consts_sampled) that represents the behavior of an expression.
    """
    if isinstance(expr, list):
        num_constants = expr.count("C")
    elif isinstance(expr, Node):
        num_constants = expr.to_list(notation="postfix").count("C")
    else:
        raise Exception("Expression should be a list of strings (tokens) or a Node")

    expr = expr_to_executable_function(expr, symbol_library)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
        if num_constants > 0:
            lho = LatinHypercube(num_constants, seed=seed)
            constants = lho.random(num_consts_sampled) * (consts_bounds[1]-consts_bounds[0]) + consts_bounds[0]
            ys = []
            for c in constants:
                ys.append(expr(X, c))
            return np.array(ys).T
        else:
            return np.repeat(expr(X, None)[:, None], num_consts_sampled, axis=1)


def _custom_wasserstein(u, v):
    u = np.sort(u)
    v = np.sort(v)
    all_values = np.sort(np.concatenate((u, v)))
    deltas = np.diff(all_values)
    u_cdf_indices = u.searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v.searchsorted(all_values[:-1], 'right')
    u_cdf = u_cdf_indices / u.size
    v_cdf = v_cdf_indices / v.size
    return np.sum(np.abs(u_cdf - v_cdf) * deltas)


def bed(expr1: Union[Node, List[str], np.ndarray], expr2: Union[Node, List[str], np.ndarray], X: Optional[np.ndarray]=None,
        num_consts_sampled: int=32, num_points_sampled: int=64, domain_bounds: Optional[List[Tuple[float, float]]]=None,
        consts_bounds: Tuple[float, float]=(-5, 5), symbol_library: SymbolLibrary=SymbolLibrary.default_symbols(),
        seed: int=None) -> float:
    """
    Computes the Behavioral Embedding Distance (BED) between two expressions or behavior matrices over a
    given dataset or domain, using Wasserstein distance as a metric.

    The BED is computed either by using precomputed behavior matrices or by sampling points from a
    domain and evaluating the expressions over them.

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
        expr1: The first expression or behavior matrix. If it is
            an expression, it must be provided as a Node or a list of string representations. If it is
            already a behavior matrix, it should be a numpy array of size (num_points_sampled, num_consts_sampled).
        expr2: The second expression or behavior matrix. Similar
            to expr1, it should be either a Node, list of strings representing the expression, or a
            numpy array representing the behavior matrix.
        X: Array of points over which behavior is evaluated. If not provided, the domain bounds parameter will be
            used to sample points.
        num_consts_sampled: Number of constants sampled for behavior evaluation if expressions
            are given as Nodes or lists rather than matrices. Default is 32.
        num_points_sampled: Number of points sampled from the domain if X is not provided. Default is 64.
        domain_bounds: The bounds of the domain for sampling points when X is not given. Each tuple represents the
            lower and upper bounds for a domain feature (e.g., [(0, 1), (0, 2)]).
        consts_bounds: The lower and upper bounds for sampling constants when evaluating expressions. Default is (-5, 5).
        symbol_library: The library of symbols used to parse and evaluate expressions. Default is the default symbol
            library from SymbolLibrary.
        seed: Seed for random number generation during sampling for deterministic results. Default is None.

    Returns:
        float: The mean Wasserstein distance computed between the behaviors of the two expressions or
        matrices over the sampled points.

    Raises:
        Exception: If X is not provided and domain_bounds is missing, this exception is raised to
            ensure proper sampling of points for behavior evaluation.
        AssertionError: Raised when the shapes of the behavior matrices or sampling points do not match
            the expected dimensions.
    """

    if X is None and not isinstance(expr1, np.ndarray) and not isinstance(expr2, np.ndarray):
        if domain_bounds is None:
            raise Exception("If X is not given and both expressions are not given as a behavior matrix, "
                            "then domain_bounds parameter must be given")
        interval_length = np.array([ub - lb for (lb, ub) in domain_bounds])
        lower_bound = np.array([lb for (lb, ub) in domain_bounds])
        lho = LatinHypercube(len(domain_bounds), optimization="random-cd", seed=seed)
        X = lho.random(num_points_sampled) * interval_length + lower_bound
    elif X is None and (isinstance(expr1, np.ndarray) != isinstance(expr2, np.ndarray)):
        raise Exception("If X is not given, both expressions must be given as a behavior matrix or as a list of "
                        "tokens/SRToolkit.utils.Node objects. Otherwise, behavior matrices are uncomparable.")

    if isinstance(expr1, list) or isinstance(expr1, Node):
        expr1 = create_behavior_matrix(expr1, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    if isinstance(expr2, list) or isinstance(expr2, Node):
        expr2 = create_behavior_matrix(expr2, X, num_consts_sampled, consts_bounds, symbol_library, seed)

    assert expr1.shape[0] == expr2.shape[0], ("Behavior matrices must have the same number rows (points "
                                                            "on which behavior is evaluated.)")
    assert expr1.shape[0] > 0, ("Behavior matrices must have at least one row. if your expressions are given as behavior"
                                "matrices, make sure they are not empty. Otherwise, if X is given, make sure it contains"
                                "at least one point. If X is not given, make sure num_points_sampled is greater than 0.")
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

if __name__ == '__main__':
    expr = ["X_0", "+", "C"]
    expr2 = ["sqrt", "(", "C", ")", "+", "X_0"]
    X = np.random.rand(10, 2) - 0.5
    b = bed(expr, expr2, X)
    print(b)
