"""Measures that compare expressions with each other, and metrics that score predictions against data.

**Expression-to-expression measures** take two expressions, each given as a list of token strings in
infix notation (as parsed by [tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree]) or as a
[Node][SRToolkit.utils.expression_tree.Node] tree:

- [edit_distance][SRToolkit.utils.measures.edit_distance] and
  [tree_edit_distance][SRToolkit.utils.measures.tree_edit_distance] are syntactic: they count
  token or node edits.
- [bed][SRToolkit.utils.measures.bed] computes the Behavior-aware Expression Dissimilarity (BED),
  which compares what expressions compute rather than how they are written.

An expression with free constants defines a family of functions, not a single one. BED evaluates
each expression at points ``X`` for many constant vectors drawn by Latin Hypercube Sampling, giving
a *behavior matrix* of shape ``(n_points, n_constant_samples)``
([create_behavior_matrix][SRToolkit.utils.measures.create_behavior_matrix]). Row ``i`` is read as
the distribution of the expression's output at ``X[i]``, and BED is the mean over points of the
1-D Wasserstein distance between corresponding rows. No constants are fitted, so BED can rank
candidate structures during search; [BEDRanking][SRToolkit.evaluation.ranking.BEDRanking] uses it
as a ranking function. A behavior matrix can be computed once and passed to
[bed][SRToolkit.utils.measures.bed] in place of an expression. See Mežnar et al.,
[arXiv:2408.11515](https://arxiv.org/abs/2408.11515).

**Regression metrics** ([mse][SRToolkit.utils.measures.mse], [rmse][SRToolkit.utils.measures.rmse],
[mae][SRToolkit.utils.measures.mae], [r2][SRToolkit.utils.measures.r2],
[nrmse][SRToolkit.utils.measures.nrmse]) compare a model's predictions with targets. All share the
signature ``metric(predict, X, y) -> float``, where ``predict`` is a
[PredictFn][SRToolkit.utils.measures.PredictFn] with the model's fitted constants already bound
in. Taking ``predict`` rather than precomputed predictions lets a metric also evaluate the model
away from ``X``, e.g. at perturbed or held-out points. Built-in metrics are listed by name in
[REGRESSION_METRICS][SRToolkit.utils.measures.REGRESSION_METRICS], which
[SR_evaluator.get_metric][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_metric] uses; a user
callable with the same signature can be used in place of a name.

All metrics return a Python float and never emit NumPy warnings: an undefined or infinite result is
returned as ``nan`` or ``inf`` (each metric lists when). ``predict(X)`` and ``y`` must both have
shape ``(n_samples,)``; anything else raises ``ValueError``.

Functions that take ``symbol_library=None`` use
[SymbolLibrary.get_or_default][SRToolkit.utils.symbol_library.SymbolLibrary.get_or_default]: the
library of the active context, else the one set with
[set_default][SRToolkit.utils.symbol_library.SymbolLibrary.set_default], else
[default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

Parameters are validated only when they are used. Invalid arguments raise ``TypeError`` or
``ValueError`` with a message that names the offending parameter.
"""

from types import MappingProxyType
from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import zss
from scipy.stats.qmc import LatinHypercube

from .expression_compiler import _variable_column, compile_expr
from .expression_tree import Node, _as_tree
from .symbol_library import SymbolLibrary

_NOTATIONS = ("infix", "prefix", "postfix")


def _check_interval(bounds: Tuple[float, float], name: str) -> None:
    """Raise ``ValueError`` unless ``bounds`` is an interval with ``lower < upper``."""
    if not bounds[0] < bounds[1]:
        raise ValueError(f"{name} must satisfy lower < upper, got {tuple(bounds)}.")


def _levenshtein(a: List[str], b: List[str]) -> int:
    """Return the Levenshtein distance between two token sequences.

    Returns:
        Minimum number of token insertions, deletions and substitutions turning ``a`` into ``b``.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


def edit_distance(
    expr1: Union[List[str], Node],
    expr2: Union[List[str], Node],
    notation: str = "postfix",
    symbol_library: Optional[SymbolLibrary] = None,
) -> int:
    """Return the Levenshtein distance between the token sequences of two expressions.

    Both expressions are parsed and re-serialised in ``notation`` before comparison, so the result
    does not depend on whether a token list or a tree is passed, and redundant parentheses in a
    token list have no effect. Each token insertion, deletion or substitution costs 1, so the
    distance is symmetric.

    Args:
        expr1: First expression as an infix token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        expr2: Second expression, in either form.
        notation: Notation in which the two expressions are compared: ``"infix"``, ``"prefix"`` or
            ``"postfix"``. It does not affect how token lists are read (always infix). In ``"infix"``,
            the parentheses needed to serialise an expression count as tokens; prefix and postfix
            contain none, hence the default ``"postfix"``.
        symbol_library: Symbol library used to parse and serialise the expressions. ``None`` uses
            the active or default library (see the module docstring).

    Returns:
        Number of token edits, ``>= 0``; ``0`` if and only if the serialised sequences are equal.

    Raises:
        TypeError: If an expression is neither a list of token strings nor a
            [Node][SRToolkit.utils.expression_tree.Node] (e.g. a string, a tuple, or a list with a
            non-string element).
        ValueError: If ``notation`` is not ``"infix"``, ``"prefix"`` or ``"postfix"``.
        ExpressionParseError: If a token list cannot be parsed: it is empty, malformed, or contains a
            symbol missing from ``symbol_library``.

    Examples:
        >>> edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> # The two parentheses count in infix notation
        >>> edit_distance(["(", "X_0", "+", "X_1", ")", "*", "X_0"], ["X_0", "+", "X_1", "*", "X_0"], notation="infix")
        2
    """
    if notation not in _NOTATIONS:
        raise ValueError(f"notation must be one of {_NOTATIONS}, got {notation!r}.")
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    tokens1 = _as_tree(expr1, symbol_library, "expr1").to_list(symbol_library=symbol_library, notation=notation)
    tokens2 = _as_tree(expr2, symbol_library, "expr2").to_list(symbol_library=symbol_library, notation=notation)
    return _levenshtein(tokens1, tokens2)


def _expr_to_zss(expr: Node) -> zss.Node:
    """Convert a [Node][SRToolkit.utils.expression_tree.Node] tree to a ``zss.Node`` tree.

    Returns:
        A ``zss.Node`` tree with the same symbols and children ordered left, then right.
    """
    zexpr = zss.Node(expr.symbol)
    if expr.left is not None:
        zexpr.addkid(_expr_to_zss(expr.left))
    if expr.right is not None:
        zexpr.addkid(_expr_to_zss(expr.right))

    return zexpr


def tree_edit_distance(
    expr1: Union[Node, List[str]],
    expr2: Union[Node, List[str]],
    symbol_library: Optional[SymbolLibrary] = None,
) -> int:
    """Return the Zhang-Shasha tree edit distance between two expressions.

    Unlike [edit_distance][SRToolkit.utils.measures.edit_distance], this compares expression trees,
    so notation and parentheses play no role. Each node insertion, deletion or relabelling costs 1;
    nodes match when their symbols are equal, and children are ordered (left, then right). The
    distance is symmetric.

    Args:
        expr1: First expression as an infix token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        expr2: Second expression, in either form.
        symbol_library: Symbol library used to parse token lists. ``None`` uses the active or
            default library (see the module docstring).

    Returns:
        Minimum number of node edits, ``>= 0``; ``0`` if and only if the trees have the same symbols in
        the same ordered structure.

    Raises:
        TypeError: If an expression is neither a list of token strings nor a
            [Node][SRToolkit.utils.expression_tree.Node] (e.g. a string, a tuple, or a list with a
            non-string element).
        ExpressionParseError: If a token list cannot be parsed: it is empty, malformed, or contains a
            symbol missing from ``symbol_library``.

    Examples:
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
        1
        >>> tree_edit_distance(["X_0", "+", "1"], ["X_0"])
        2
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    zss1 = _expr_to_zss(_as_tree(expr1, symbol_library, "expr1"))
    zss2 = _expr_to_zss(_as_tree(expr2, symbol_library, "expr2"))

    # Unit costs: zss otherwise uses the `editdistance` package (when installed) to price relabelling
    # by the character edit distance between symbols. zss is untyped; without return_operations=True
    # it returns just the cost.
    distance = cast(float, zss.simple_distance(zss1, zss2, label_dist=_unit_label_cost))
    return int(distance)


def _unit_label_cost(a: str, b: str) -> int:
    """Return the cost of relabelling ``a`` as ``b`` (insertion/deletion compare with ``""``): 0 or 1."""
    return int(a != b)


def create_behavior_matrix(
    expr: Union[Node, List[str]],
    X: np.ndarray,
    num_consts_sampled: int = 32,
    consts_bounds: Tuple[float, float] = (-5, 5),
    symbol_library: Optional[SymbolLibrary] = None,
    seed: Optional[int] = None,
    backend: str = "stack",
) -> np.ndarray:
    """Evaluate an expression at the points ``X`` for many sampled constant vectors.

    Column ``j`` holds the expression's values at ``X`` with the ``j``-th constant vector. Each
    occurrence of a constant symbol is a separate free parameter; with ``n_consts`` occurrences, the
    constant vectors are one Latin Hypercube sample of ``num_consts_sampled`` points in the
    ``n_consts``-dimensional box with every constant in ``[lower, upper)`` (``consts_bounds``). A
    constant-free expression gives ``num_consts_sampled`` identical columns. Non-finite outputs
    (``nan``, ``±inf``) are kept. NumPy floating-point warnings are suppressed during the call, and
    NumPy's error settings are restored afterwards. Sampling is random unless ``seed`` is set.

    Args:
        expr: Expression as an infix token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        X: Evaluation points of shape ``(n_samples, n_features)``, with a column for every variable
            ``expr`` uses. ``n_samples`` may be 0, giving a result of shape ``(0, num_consts_sampled)``.
        num_consts_sampled: Number of constant vectors, i.e. output columns; at least 1.
        consts_bounds: ``(lower, upper)`` interval for every constant, with ``lower < upper``.
        symbol_library: Symbol library used to compile ``expr``. ``None`` uses the active or default
            library (see the module docstring).
        seed: Seed for constant sampling. ``None`` gives a different sample on each call.
        backend: Backend used to compile expressions: ``"stack"`` (default), ``"stack_py"`` or
            ``"codegen"`` (see [compile_expr][SRToolkit.utils.expression_compiler.compile_expr]). All give
            the same values; ``"stack_py"`` can be faster for libraries with many custom symbols.

    Returns:
        Behavior matrix of shape ``(n_samples, num_consts_sampled)``.

    Raises:
        TypeError: If ``expr`` is neither a list of token strings nor a
            [Node][SRToolkit.utils.expression_tree.Node].
        ValueError: If ``X`` is not 2-D or has no column for a variable that ``expr`` uses, if
            ``num_consts_sampled < 1``, if ``consts_bounds[0] >= consts_bounds[1]``, or if ``backend``
            is not one of the backends above.
        ExpressionParseError: If ``expr`` cannot be parsed or compiled with ``symbol_library``.

    Examples:
        >>> X = np.array([[0.0], [1.0], [2.0]])
        >>> create_behavior_matrix(["X_0", "+", "C"], X, num_consts_sampled=8, seed=0).shape
        (3, 8)
        >>> create_behavior_matrix(["X_0", "*", "X_0"], X, num_consts_sampled=2)
        array([[0., 0.],
               [1., 1.],
               [4., 4.]])
    """
    if num_consts_sampled < 1:
        raise ValueError(f"num_consts_sampled must be at least 1, got {num_consts_sampled}.")
    _check_interval(consts_bounds, "consts_bounds")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D with shape (n_samples, n_features), got shape {X.shape}.")
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()

    tree = _as_tree(expr, symbol_library, "expr")
    tokens = tree.to_list(symbol_library=symbol_library, notation="postfix")
    num_constants = sum(symbol_library.get_type(token) == "const" for token in tokens)
    callable_expr = compile_expr(tree, symbol_library, backend=backend)

    with np.errstate(all="ignore"):
        if num_constants == 0:
            return np.repeat(callable_expr(X, None)[:, None], num_consts_sampled, axis=1)
        lower, upper = consts_bounds
        constants = LatinHypercube(num_constants, rng=seed).random(num_consts_sampled) * (upper - lower) + lower
        columns = [callable_expr(X, c) for c in constants]
        return np.stack(columns, axis=1)


def _custom_wasserstein(u: np.ndarray, v: np.ndarray):
    """Return the 1-D Wasserstein distance between two samples.

    Equal to ``scipy.stats.wasserstein_distance(u, v)``; the samples may differ in length.

    Args:
        u: First sample, 1-D and non-empty.
        v: Second sample, 1-D and non-empty.

    Returns:
        The distance as a numpy float, ``>= 0``.
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
    """Return the row-wise 1-D Wasserstein distances between two finite behavior matrices.

    Equivalent to ``_custom_wasserstein`` on each pair
    of rows, computed in chunks that keep temporary memory under about 50 MB.

    Args:
        expr1: Finite matrix of shape ``(n_rows, k1)``.
        expr2: Finite matrix of shape ``(n_rows, k2)``; ``k2`` may differ from ``k1``.

    Returns:
        Array of shape ``(n_rows,)`` with the distance between row ``i`` of each matrix.
    """
    n, k1 = expr1.shape
    k2 = expr2.shape[1]
    u_s = np.sort(expr1, axis=1)  # (n, k1)
    v_s = np.sort(expr2, axis=1)  # (n, k2)
    all_vals = np.sort(np.concatenate([u_s, v_s], axis=1), axis=1)  # (n, k1 + k2)
    deltas = np.diff(all_vals, axis=1)  # (n, m)

    m = k1 + k2 - 1  # number of CDF evaluation points
    # Limit the intermediate (n_chunk × max(k1, k2) × m) boolean array to ≈50 MB
    max_chunk = max(1, 50_000_000 // (max(k1, k2) * m))

    wds = np.empty(n, dtype=np.float64)
    for start in range(0, n, max_chunk):
        end = min(start + max_chunk, n)
        cu = u_s[start:end]  # (chunk, k1)
        cv = v_s[start:end]  # (chunk, k2)
        ca = all_vals[start:end, :-1]  # (chunk, m)
        # Broadcasting: (chunk, k, 1) <= (chunk, 1, m) → (chunk, k, m); sum over k axis
        u_cdf = np.sum(cu[:, :, None] <= ca[:, None, :], axis=1) / k1  # (chunk, m)
        v_cdf = np.sum(cv[:, :, None] <= ca[:, None, :], axis=1) / k2  # (chunk, m)
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
    symbol_library: Optional[SymbolLibrary] = None,
    seed: Optional[int] = None,
    backend: str = "stack",
) -> float:
    """Return the Behavior-aware Expression Dissimilarity (BED) between two expressions.

    Each expression is turned into a behavior matrix over the same points (see
    [create_behavior_matrix][SRToolkit.utils.measures.create_behavior_matrix]), and BED is the mean
    over points of the 1-D Wasserstein distance between the two rows. A precomputed behavior matrix
    can be passed in place of an expression to avoid re-evaluating it. The points depend on the
    arguments:

    - **Two matrices** are compared directly, and must have been computed at the same points. If
      ``X`` is given, it is only used to check that each matrix has ``len(X)`` rows.
    - **At least one expression, with ``X``**: expressions are evaluated at ``X``, and a matrix
      argument must have been computed at ``X`` (``len(X)`` rows). ``domain_bounds`` is ignored.
    - **Two expressions, without ``X``**: ``num_points_sampled`` points are drawn from
      ``domain_bounds`` by Latin Hypercube Sampling.
    - **An expression and a matrix, without ``X``** is an error: the matrix's points are unknown.

    Args:
        expr1: First expression as an infix token list, a [Node][SRToolkit.utils.expression_tree.Node] tree,
            or a behavior matrix of shape ``(n_samples, k1)`` with ``k1 >= 1``.
        expr2: Second expression, in any of the same forms. A matrix may have a different number of
            columns than ``expr1``.
        X: Points of shape ``(n_samples, n_features)`` at which expression arguments are evaluated,
            with a column for every variable used. Required when exactly one argument is a matrix;
            with two matrices, only their row count is checked against it.
        num_consts_sampled: Constant vectors sampled per expression argument (its matrix columns);
            at least 1. Not used for matrix arguments.
        num_points_sampled: Number of points drawn from ``domain_bounds``; at least 1. Used only in
            that mode.
        domain_bounds: One ``(lower, upper)`` interval per variable, with ``lower < upper``, covering
            every variable used. Used only when ``X`` is ``None`` and neither argument is a matrix.
        consts_bounds: ``(lower, upper)`` interval for every constant, with ``lower < upper``. Not used
            for matrix arguments, but validated whenever an argument is an expression.
        symbol_library: Symbol library used to compile expressions. ``None`` uses the active or
            default library (see the module docstring).
        seed: Seed for all sampling: the points and the constants of both expressions. ``None``
            makes the result random whenever points are sampled or an expression has constants.
        backend: Backend used to compile expression arguments: ``"stack"`` (default), ``"stack_py"`` or
            ``"codegen"`` (see [compile_expr][SRToolkit.utils.expression_compiler.compile_expr]). All give
            the same values; ``"stack_py"`` can be faster for libraries with many custom symbols.

    Returns:
        BED as a float ``>= 0``; ``0.0`` for identical behavior at the points. Each point's (row's)
        distance is found by these steps, in order:

        1. ``nan`` entries are dropped from both rows.
        2. If both rows are now empty, the distance is 0; if only one is, it is ``inf``.
        3. If either row contains ``±inf``: 0 if both rows hold only the same infinity, else ``inf``.
        4. Otherwise, the distance is the Wasserstein distance between the remaining samples.

        If any point's distance is ``inf``, BED is ``inf``.

    Raises:
        TypeError: If an argument is neither a list of token strings, a
            [Node][SRToolkit.utils.expression_tree.Node] nor a ``numpy.ndarray``.
        ValueError: If ``X`` is ``None`` and either exactly one argument is a matrix, or neither
            ``domain_bounds`` nor two matrices are given; if a matrix argument is not 2-D or has no
            columns; if the two behaviors have different numbers of rows or none (including an
            ``X`` with 0 rows at which an expression is evaluated); if ``X`` is given and is not
            2-D or a matrix argument's row count differs from ``len(X)``;
            if points are sampled and ``num_points_sampled < 1`` or an interval in
            ``domain_bounds`` has ``lower >= upper``; if an argument is an expression (even one
            without constants) and ``num_consts_sampled < 1`` or ``consts_bounds`` has
            ``lower >= upper`` or ``backend`` is unknown; or if ``X`` or ``domain_bounds`` lacks a
            column for a variable used.
        ExpressionParseError: If an expression cannot be parsed or compiled with ``symbol_library``.

    Examples:
        >>> X = np.linspace(-1, 1, 20).reshape(-1, 1)
        >>> bed(["X_0"], ["X_0", "+", "1"], X)
        1.0
        >>> bed(["C", "*", "X_0"], ["C", "+", "X_0"], domain_bounds=[(-1, 1)], seed=0) > 0
        True
        >>> # Reuse a precomputed behavior matrix, e.g. of a ground truth
        >>> gt = create_behavior_matrix(["X_0", "+", "C"], X, seed=0)
        >>> bed(["X_0", "+", "C"], gt, X, seed=0)
        0.0

    Notes:
        With rows ``u_i`` and ``v_i`` of the two behavior matrices read as empirical distributions,
        ``BED = (1 / n_samples) * sum_i d_i``, where ``d_i`` is the 1-D Wasserstein distance
        ``W1(u_i, v_i)`` after the non-finite steps in Returns. Every point counts in ``n_samples``,
        including those given distance 0 by those steps.

        As a consequence of seeding, with ``seed`` set ``bed`` is deterministic, symmetric in
        ``expr1`` and ``expr2``, and ``bed(e, e) == 0``. With ``seed=None``, ``bed(e, e)`` can be
        greater than 0 for an expression with constants.

    References:
        S. Mežnar, S. Džeroski, L. Todorovski. Quantifying Behavioral Dissimilarity Between
        Mathematical Expressions. [arXiv:2408.11515](https://arxiv.org/abs/2408.11515).
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    # Expressions become Node trees (parsed once); matrices are checked for shape.
    trees: Dict[str, Node] = {}
    matrices: Dict[str, np.ndarray] = {}
    for name, arg in (("expr1", expr1), ("expr2", expr2)):
        if isinstance(arg, np.ndarray):
            if arg.ndim != 2 or arg.shape[1] == 0:
                raise ValueError(
                    f"{name} must be a behavior matrix of shape (n_samples, k) with k >= 1, got {arg.shape}."
                )
            matrices[name] = arg
        elif isinstance(arg, (list, Node)):
            trees[name] = _as_tree(arg, symbol_library, name)
        else:
            raise TypeError(
                f"{name} must be a list of token strings, a Node or a numpy.ndarray, got {type(arg).__name__}."
            )

    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D with shape (n_samples, n_features), got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one point (row).")
        for name, matrix in matrices.items():
            if matrix.shape[0] != X.shape[0]:
                raise ValueError(f"{name} has {matrix.shape[0]} rows, but X has {X.shape[0]} points.")
    elif len(matrices) == 1:
        raise ValueError("X must be given when exactly one of expr1 and expr2 is a behavior matrix.")
    elif not matrices:
        X = _sample_points(trees["expr1"], trees["expr2"], num_points_sampled, domain_bounds, symbol_library, seed)
    else:
        rows1, rows2 = matrices["expr1"].shape[0], matrices["expr2"].shape[0]
        if rows1 != rows2:
            raise ValueError(f"expr1 and expr2 must have the same number of rows, got {rows1} and {rows2}.")
        if rows1 == 0:
            raise ValueError("expr1 and expr2 must have at least one row.")

    def behavior(name: str) -> np.ndarray:
        if name in matrices:
            return matrices[name]
        # An expression argument always has points: X was given, or it was sampled above.
        assert X is not None
        return create_behavior_matrix(trees[name], X, num_consts_sampled, consts_bounds, symbol_library, seed, backend)

    return _mean_row_distance(behavior("expr1"), behavior("expr2"))


def _sample_points(
    expr1: Node,
    expr2: Node,
    num_points_sampled: int,
    domain_bounds: Optional[List[Tuple[float, float]]],
    symbol_library: SymbolLibrary,
    seed: Optional[int],
) -> np.ndarray:
    """Return ``num_points_sampled`` Latin Hypercube points in ``domain_bounds``, for ``bed``.

    Returns:
        Points of shape ``(num_points_sampled, len(domain_bounds))``.

    Raises:
        ValueError: If ``domain_bounds`` is missing, has an empty interval or no column for a variable
            used by an expression, or if ``num_points_sampled < 1``.
    """
    if domain_bounds is None:
        raise ValueError("domain_bounds must be given when X is not given and neither argument is a behavior matrix.")
    if num_points_sampled < 1:
        raise ValueError(f"num_points_sampled must be at least 1, got {num_points_sampled}.")
    for i, bounds in enumerate(domain_bounds):
        _check_interval(bounds, f"domain_bounds[{i}]")
    for tree in (expr1, expr2):
        columns = [
            _variable_column(token, symbol_library)
            for token in tree.to_list(symbol_library=symbol_library, notation="postfix")
            if symbol_library.get_type(token) == "var"
        ]
        if columns and max(columns) >= len(domain_bounds):
            raise ValueError(
                f"domain_bounds has {len(domain_bounds)} interval(s), but an expression uses variable column "
                f"{max(columns)}."
            )
    lower = np.array([lb for lb, _ in domain_bounds], dtype=np.float64)
    width = np.array([ub - lb for lb, ub in domain_bounds], dtype=np.float64)
    return (
        LatinHypercube(len(domain_bounds), optimization="random-cd", rng=seed).random(num_points_sampled) * width
        + lower
    )


def _mean_row_distance(b1: np.ndarray, b2: np.ndarray) -> float:
    """Return BED for two behavior matrices with equal row counts, applying the non-finite steps per row.

    Returns:
        The mean over rows of the row distances defined in [bed][SRToolkit.utils.measures.bed]'s
        Returns section, or ``inf`` if any row distance is ``inf``.
    """
    nan1, nan2 = np.isnan(b1), np.isnan(b2)
    empty1, empty2 = nan1.all(axis=1), nan2.all(axis=1)
    if np.any(empty1 != empty2):  # step 2: only one row has values left after dropping nan
        return np.inf

    # Step 3: a row with an infinity has distance 0 only if every remaining value of both rows is the
    # same infinity. (Rows that are empty in both matrices satisfy these conditions vacuously.)
    has_inf = (np.isinf(b1).any(axis=1) | np.isinf(b2).any(axis=1)) & ~empty1
    all_pos = ((b1 == np.inf) | nan1).all(axis=1) & ((b2 == np.inf) | nan2).all(axis=1)
    all_neg = ((b1 == -np.inf) | nan1).all(axis=1) & ((b2 == -np.inf) | nan2).all(axis=1)
    if np.any(has_inf & ~(all_pos | all_neg)):
        return np.inf

    # Step 4: Wasserstein distance on finite rows; rows with nan go through the slower per-row path.
    distances = np.zeros(b1.shape[0])
    finite = ~empty1 & ~has_inf
    clean = finite & ~nan1.any(axis=1) & ~nan2.any(axis=1)
    if np.any(clean):
        distances[clean] = _vectorized_wasserstein_batch(b1[clean], b2[clean])
    for i in np.flatnonzero(finite & ~clean):
        distances[i] = _custom_wasserstein(b1[i][~nan1[i]], b2[i][~nan2[i]])
    return float(distances.mean())


# ---------------------------------------------------------------------------
# Regression metrics: metric(predict, X, y) -> float (see the module docstring)
# ---------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], np.ndarray]
"""Type of a prediction function ``predict(X) -> y_pred``, the first argument of every regression metric.

It maps inputs of shape ``(n_samples, n_features)`` to predictions of shape ``(n_samples,)``, with the
model's fitted constants already bound in.
"""


def _predictions_and_targets(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Call ``predict(X)`` and return the predictions and targets as float arrays of shape ``(n_samples,)``.

    Returns:
        ``(y_pred, y)``, both 1-D float64 arrays of length ``len(X)``.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.
    """
    n_samples = len(X)
    y_pred = np.asarray(predict(X), dtype=np.float64)
    if y_pred.shape != (n_samples,):
        raise ValueError(
            f"predict(X) must return shape ({n_samples},) for X with {n_samples} rows, got {y_pred.shape}."
        )
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (n_samples,):
        raise ValueError(f"y must have shape ({n_samples},) for X with {n_samples} rows, got {y.shape}.")
    return y_pred, y


def _mean_squared_error(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Return ``mean((y_pred - y) ** 2)``, or ``nan`` for empty input."""
    if y.size == 0:
        return float("nan")
    return float(np.mean((y_pred - y) ** 2))


def _normalised_metric_special_value(
    y_pred: np.ndarray, y: np.ndarray, perfect: float, worst: float
) -> Optional[float]:
    """Return the special value of [r2][SRToolkit.utils.measures.r2] / [nrmse][SRToolkit.utils.measures.nrmse], if any.

    Returns:
        ``nan`` for empty input, a non-finite target or a ``nan`` prediction; for constant ``y``,
        ``perfect`` if every prediction is exact, else ``worst``; ``worst`` if any prediction is
        infinite; otherwise ``None`` (the regular formula applies).
    """
    if y.size == 0 or not np.all(np.isfinite(y)) or np.any(np.isnan(y_pred)):
        return float("nan")
    if np.all(y == y[0]):
        return perfect if np.all(y_pred == y) else worst
    if np.any(np.isinf(y_pred)):
        return worst
    return None


def mse(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the mean squared error of a model's predictions.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``. Exceptions it
            raises propagate.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``, as any array-like (converted with ``np.asarray``).

    Returns:
        ``mean((predict(X) - y) ** 2)``, ``>= 0``; lower is better. Special cases:

        - ``nan`` if there are no samples, or an error ``predict(X)[i] - y[i]`` is ``nan`` (e.g. ``inf - inf``);
        - otherwise ``inf`` if an error is infinite or too large to square.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.

    Examples:
        >>> mse(lambda X: X[:, 0], np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 4.0]))
        0.3333333333333333
    """
    y_pred, y = _predictions_and_targets(predict, X, y)
    with np.errstate(all="ignore"):
        return _mean_squared_error(y_pred, y)


def rmse(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the root-mean-square error of a model's predictions.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``. Exceptions it
            raises propagate.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``, as any array-like (converted with ``np.asarray``).

    Returns:
        ``sqrt(mean((predict(X) - y) ** 2))``, ``>= 0``; lower is better. Special cases:

        - ``nan`` if there are no samples, or an error ``predict(X)[i] - y[i]`` is ``nan`` (e.g. ``inf - inf``);
        - otherwise ``inf`` if an error is infinite or too large to square.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.

    Examples:
        >>> rmse(lambda X: X[:, 0], np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 4.0]))
        0.5773502691896257
    """
    y_pred, y = _predictions_and_targets(predict, X, y)
    with np.errstate(all="ignore"):
        return float(np.sqrt(_mean_squared_error(y_pred, y)))


def mae(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the mean absolute error of a model's predictions.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``. Exceptions it
            raises propagate.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``, as any array-like (converted with ``np.asarray``).

    Returns:
        ``mean(abs(predict(X) - y))``, ``>= 0``; lower is better. Special cases:

        - ``nan`` if there are no samples, or an error ``predict(X)[i] - y[i]`` is ``nan`` (e.g. ``inf - inf``);
        - otherwise ``inf`` if an error is infinite.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.

    Examples:
        >>> mae(lambda X: X[:, 0], np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 4.0]))
        0.3333333333333333
    """
    y_pred, y = _predictions_and_targets(predict, X, y)
    if y.size == 0:
        return float("nan")
    with np.errstate(all="ignore"):
        return float(np.mean(np.abs(y_pred - y)))


def r2(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the coefficient of determination R² of a model's predictions.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``. Exceptions it
            raises propagate.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``, as any array-like (converted with ``np.asarray``).

    Returns:
        ``1 - SS_res / SS_tot``, with ``SS_res = sum((y - predict(X)) ** 2)`` and
        ``SS_tot = sum((y - mean(y)) ** 2)``; at most 1, higher is better. Special cases, checked
        in this order (a prediction is exact if it equals its target exactly):

        - ``nan`` if there are no samples, a target is ``nan`` or infinite, or a prediction is ``nan``;
        - if all targets are equal (also for one sample): ``1.0`` if all predictions are exact, else ``-inf``;
        - ``-inf`` if a prediction is infinite.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.

    Examples:
        >>> r2(lambda X: X[:, 0], np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 4.0]))
        0.7857142857142857
    """
    y_pred, y = _predictions_and_targets(predict, X, y)
    special = _normalised_metric_special_value(y_pred, y, perfect=1.0, worst=-np.inf)
    if special is not None:
        return special
    with np.errstate(all="ignore"):
        return float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def nrmse(predict: PredictFn, X: np.ndarray, y: np.ndarray) -> float:
    """Return the root-mean-square error of a model's predictions, normalised by the spread of the targets.

    Args:
        predict: Callable mapping ``X`` to predictions of shape ``(n_samples,)``. Exceptions it
            raises propagate.
        X: Input data of shape ``(n_samples, n_features)``.
        y: Targets of shape ``(n_samples,)``, as any array-like (converted with ``np.asarray``).

    Returns:
        ``rmse / std(y)``, with the population standard deviation (``ddof=0``); ``>= 0``, lower is
        better. Special cases, checked in this order (a prediction is exact if it equals its target
        exactly):

        - ``nan`` if there are no samples, a target is ``nan`` or infinite, or a prediction is ``nan``;
        - if all targets are equal (also for one sample): ``0.0`` if all predictions are exact, else ``inf``;
        - ``inf`` if a prediction is infinite.

    Raises:
        ValueError: If ``predict(X)`` or ``y`` is not 1-D of length ``n_samples``.

    Examples:
        >>> nrmse(lambda X: X[:, 0], np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 4.0]))
        0.46291004988627577
    """
    y_pred, y = _predictions_and_targets(predict, X, y)
    special = _normalised_metric_special_value(y_pred, y, perfect=0.0, worst=np.inf)
    if special is not None:
        return special
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.mean((y - y_pred) ** 2)) / np.std(y))


REGRESSION_METRICS: Mapping[str, Callable[[PredictFn, np.ndarray, np.ndarray], float]] = MappingProxyType(
    {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse": nrmse,
    }
)
"""Read-only mapping from name to built-in regression metric.

Keys are ``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"`` and ``"nrmse"``; every value has the signature
``metric(predict, X, y) -> float``.
[SR_evaluator.get_metric][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_metric] uses it to
resolve a metric given by name. The mapping cannot be modified; to use another metric, pass a
callable with the same signature instead of a name.
"""
