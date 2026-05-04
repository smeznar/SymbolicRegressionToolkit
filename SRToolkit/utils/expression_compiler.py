"""
Functions for compiling symbolic expressions into executable Python callables.

The primary interface are two functions:

- [compile_expr][SRToolkit.utils.expression_compiler.compile_expr]:
  returns a callable ``f(X, C) → np.ndarray``.
- [compile_expr_rmse][SRToolkit.utils.expression_compiler.compile_expr_rmse]:
  returns an RMSE callable ``f(X, C, y) → float``.

Both accept a ``backend`` parameter that selects the evaluation engine:

- ``"stack"`` (default): postfix stack-machine evaluator backed by Cython; falls
  back to pure-Python automatically when the compiled extension is unavailable.
- ``"codegen"``: generates Python/NumPy source code via ``exec()``. Compatible
  with any custom symbol library and requires no compiled extensions.
- ``"stack_py"``: pure-Python stack-machine evaluator; avoids the Cython→Python
  boundary overhead when the library contains many custom symbols.

The ``backend`` parameter selects the engine; lower-level per-backend functions are
internal and prefixed with ``_``.

Use ``"stack"`` (the default) for best performance in most cases. Use ``"stack_py"``
when the library contains many custom symbols to avoid Cython→Python call overhead per
instruction. Use ``"codegen"`` when evaluating on large datasets, as it generates a
single vectorised NumPy function with less per-instruction dispatch overhead.
"""

# Load the pure-Python evaluator explicitly, bypassing the compiled .so.
# Used when backend="stack_py" is requested — avoids Cython→Python boundary
# overhead for expressions with many custom (non-default) symbols.
import importlib.util as _ilu
import pathlib as _pl
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Import the evaluator backend. Python prefers the compiled `_eval_cython.so`
# (or `.pyd` on Windows) over `_eval_cython.py` when both are present, so this
# single import transparently selects the fast Cython path when available and
# falls back to the pure-Python implementation otherwise.
from SRToolkit.utils import _eval_cython as _eval  # type: ignore[attr-defined]
from SRToolkit.utils.expression_tree import Node, is_float, tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary

_eval_py_spec = _ilu.spec_from_file_location(
    "_eval_cython_py",
    _pl.Path(__file__).parent / "_eval_cython.py",
)
_eval_py = _ilu.module_from_spec(_eval_py_spec)  # type: ignore[arg-type]
_eval_py_spec.loader.exec_module(_eval_py)  # type: ignore[union-attr]


# ── Codegen evaluator backend ──────────────────────────────────────────────


def _expr_to_executable_function(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Compile an expression into an executable Python function.

    The returned callable evaluates the expression over a batch of inputs and a vector
    of constant values. To use a backend other than NumPy, set
    ``symbol_library.preamble`` to the required import statements.

    Examples:
        >>> executable_fun = _expr_to_executable_function(["X_0", "+", "1"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])
        >>> executable_fun = _expr_to_executable_function(["pi"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([1]))
        array([3.14159265, 3.14159265, 3.14159265, 3.14159265])
        >>> executable_fun = _expr_to_executable_function(["C"])
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([1]))
        array([1, 1, 1, 1])
        >>> tree = tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        >>> executable_fun = _expr_to_executable_function(tree)
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])
        >>> # In case you need libraries other than numpy for the evaluation of your expressions,
        >>> # you can add them to the preamble in the SymbolLibrary. Here is how a preamble would look like:
        >>> symbol_library = SymbolLibrary.default_symbols(1)
        >>> symbol_library.preamble = ["import numpy as np"]
        >>> # Usually this is done when initializing the SymbolLibrary as SymbolLibrary(preamble=preamble)
        >>> executable_fun = _expr_to_executable_function(tree, symbol_library)
        >>> executable_fun(np.array([[1], [2], [3], [4]]), np.array([]))
        array([2, 3, 4, 5])

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, preamble imports).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D array of shape ``(n_samples, n_features)`` and ``C`` is a 1-D array of constant values. Returns a 1-D output array of shape ``(n_samples,)``.

    Raises:
        Exception: If ``expr`` is neither a list nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if not (isinstance(expr, list) or isinstance(expr, Node)):
        raise Exception(
            "Expression must be given as either a list of tokens or a tree (an instance of the "
            "SRToolkit.utils.expression_tree.Node class)"
        )

    if isinstance(expr, list):
        tree = tokens_to_tree(expr, symbol_library)
    else:
        tree = expr
    code, symbol, _, _ = _tree_to_function_rec(tree, symbol_library)

    fun_string = "\n".join(symbol_library.preamble) + "\ndef _executable_expression_(X, C):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += "\treturn " + symbol

    fun_assignment_dict: Dict[str, Callable] = {}
    exec(fun_string, {"np": np}, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def _expr_to_error_function(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Compile an expression into a callable that computes the RMSE against target values.

    To use a backend other than NumPy, set ``symbol_library.preamble`` to the required
    import statements.

    Examples:
        >>> executable_fun = _expr_to_error_function(["X_0", "+", "1"])
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0
        >>> tree = tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        >>> executable_fun = _expr_to_error_function(tree)
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0
        >>> # In case you need libraries other than numpy for the evaluation of your expressions,
        >>> # you can add them to the preamble in the SymbolLibrary. Here is how a preamble would look like:
        >>> symbol_library = SymbolLibrary.default_symbols(1)
        >>> symbol_library.preamble = ["import numpy as np"]
        >>> # Usually this is done when initializing the SymbolLibrary as SymbolLibrary(preamble=preamble)
        >>> executable_fun = _expr_to_error_function(tree, symbol_library)
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, preamble imports).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C, y)`` where ``X`` is a 2-D array of shape ``(n_samples, n_features)``, ``C`` is a 1-D array of constant values, and ``y`` is a 1-D target array. Returns the scalar RMSE as a float.

    Raises:
        Exception: If ``expr`` is neither a list nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if not (isinstance(expr, list) or isinstance(expr, Node)):
        raise Exception(
            "Expression must be given as either a list of tokens or a tree (an instance of the "
            "SRToolkit.utils.expression_tree.Node class)"
        )

    if isinstance(expr, list):
        tree = tokens_to_tree(expr, symbol_library)
    else:
        tree = expr
    code, symbol, _, _ = _tree_to_function_rec(tree, symbol_library)

    fun_string = "\n".join(symbol_library.preamble) + "\ndef _executable_expression_(X, C, y):\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))"

    fun_assignment_dict: Dict[str, Callable] = {}
    exec(fun_string, {"np": np}, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def _tree_to_function_rec(
    tree: Node,
    symbol_library: SymbolLibrary,
    var_counter: int = 0,
    const_counter: int = 0,
) -> Tuple[List[str], str, int, int]:
    """
    Recursively convert a parse tree into lines of Python code for expression evaluation.

    This is a low-level helper used by the codegen backend of [compile_expr][SRToolkit.utils.expression_compiler.compile_expr] and
    [compile_expr_rmse][SRToolkit.utils.expression_compiler.compile_expr_rmse]. Call those functions directly unless you need
    fine-grained control over code generation.

    Args:
        tree: Root of the subtree to convert.
        symbol_library: Provides NumPy function strings for each token.
        var_counter: Running count of intermediate variables, used to generate unique
            names. Default ``0``.
        const_counter: Running count of constants consumed; used to index into the ``C``
            array. Default ``0``.

    Returns:
        A 4-tuple ``(code, symbol, var_counter, const_counter)`` where ``code`` is a list of Python assignment strings forming the expression body, ``symbol`` is the name of the variable holding this subtree's result, and ``var_counter`` / ``const_counter`` are the updated counters.

    Raises:
        Exception: If the tree contains a token that is neither a recognized symbol nor
            a numeric literal.
    """
    if tree.left is None and tree.right is None:
        if symbol_library.get_type(tree.symbol) == "var":
            return [], symbol_library.get_np_fn(tree.symbol), var_counter, const_counter
        elif symbol_library.get_type(tree.symbol) == "lit":
            output_symbol = "y_{}".format(var_counter)
            code = [f"{output_symbol} = np.full(X.shape[0], {symbol_library.get_np_fn(tree.symbol)})"]
            return code, output_symbol, var_counter + 1, const_counter
        elif symbol_library.get_type(tree.symbol) == "const":
            return (
                [],
                symbol_library.get_np_fn(tree.symbol).format(const_counter),
                var_counter,
                const_counter + 1,
            )
        else:
            if is_float(tree.symbol):
                return [], tree.symbol, var_counter, const_counter
            else:
                raise Exception(f"Encountered invalid symbol {tree.symbol} while converting to function.")

    elif tree.left is not None and tree.right is None:
        code, symbol, var_counter, const_counter = _tree_to_function_rec(
            tree.left, symbol_library, var_counter, const_counter
        )
        output_symbol = "y_{}".format(var_counter)
        code.append(f"{output_symbol} = " + symbol_library.get_np_fn(tree.symbol).format(symbol))
        return code, output_symbol, var_counter + 1, const_counter

    else:
        assert tree.right is not None, "Right child should be present in this branch."
        assert tree.left is not None, "Left child should be present if right child is present."
        left_code, left_symbol, var_counter, const_counter = _tree_to_function_rec(
            tree.left, symbol_library, var_counter, const_counter
        )
        right_code, right_symbol, var_counter, const_counter = _tree_to_function_rec(
            tree.right, symbol_library, var_counter, const_counter
        )
        output_symbol = "y_{}".format(var_counter)
        code = left_code + right_code
        code.append(f"{output_symbol} = " + symbol_library.get_np_fn(tree.symbol).format(left_symbol, right_symbol))
        return code, output_symbol, var_counter + 1, const_counter


# ── Stack evaluator backend ──────────────────────────────────────────────


def _compile_fallback_unary(np_fn: str, preamble: List[str]) -> Callable:
    """Build a Python callable ``f(x) → np.ndarray`` from a unary np_fn string."""
    src = "\n".join(preamble) + f"\ndef _fn_(_x_):\n    return {np_fn.format('_x_')}\n"
    ns: Dict[str, Callable] = {}
    exec(src, {"np": np}, ns)
    return ns["_fn_"]


def _compile_fallback_binary(np_fn: str, preamble: List[str]) -> Callable:
    """Build a Python callable ``f(a, b) → np.ndarray`` from a binary np_fn string."""
    src = "\n".join(preamble) + f"\ndef _fn_(_a_, _b_):\n    return {np_fn.format('_a_', '_b_')}\n"
    ns: Dict[str, Callable] = {}
    exec(src, {"np": np}, ns)
    return ns["_fn_"]


def _collect_maximal_const_free(node: Node, symbol_library: SymbolLibrary) -> Tuple[bool, List[Node]]:
    """Return ``(has_const, maximal_const_free_nodes)`` for the subtree rooted at *node*.

    A subtree is maximal const-free if it contains no ``C`` tokens and its parent does
    (or the whole expression is const-free). These are exactly the subtrees that can be
    pre-evaluated once against X and reused across optimizer calls.
    """
    if is_float(node.symbol):
        return False, []
    if symbol_library.get_type(node.symbol) == "const":
        return True, []

    left_has_const, left_maximal = (
        _collect_maximal_const_free(node.left, symbol_library) if node.left is not None else (False, [])
    )
    right_has_const, right_maximal = (
        _collect_maximal_const_free(node.right, symbol_library) if node.right is not None else (False, [])
    )

    if not (left_has_const or right_has_const):
        return False, []

    result: List[Node] = []
    if node.left is not None:
        result.extend([node.left] if not left_has_const else left_maximal)
    if node.right is not None:
        result.extend([node.right] if not right_has_const else right_maximal)
    return True, result


def _build_partial_instructions(
    node: Node,
    symbol_library: SymbolLibrary,
    const_free_ids: set,
    cached_arrays: Dict,
    token_ids_list: List,
    arities_list: List,
    extra_int_list: List,
    float_vals_list: List,
    python_ops_list: List,
    const_counter: List,  # one-element list used as a mutable int
) -> None:
    """Recursively emit postfix instructions, substituting CACHED leaves for
    pre-evaluated constant-free subtrees."""
    if id(node) in const_free_ids:
        token_ids_list.append(_eval.CACHED)
        arities_list.append(0)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        python_ops_list.append(cached_arrays[id(node)])
        return

    sym = node.symbol
    stype = symbol_library.get_type(sym) if not is_float(sym) else None

    if stype == "const":
        token_ids_list.append(_eval.CONST)
        arities_list.append(0)
        extra_int_list.append(const_counter[0])
        float_vals_list.append(0.0)
        python_ops_list.append(None)
        const_counter[0] += 1

    elif stype == "fn":  # unary: recurse into left child, then emit fn
        assert node.left is not None
        _build_partial_instructions(
            node.left,
            symbol_library,
            const_free_ids,
            cached_arrays,
            token_ids_list,
            arities_list,
            extra_int_list,
            float_vals_list,
            python_ops_list,
            const_counter,
        )
        cython_id = symbol_library.get_cython_id(sym)
        arities_list.append(1)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        if cython_id >= 0:
            token_ids_list.append(cython_id)
            python_ops_list.append(None)
        else:
            token_ids_list.append(_eval.PYTHON)
            python_ops_list.append(_compile_fallback_unary(symbol_library.get_np_fn(sym), symbol_library.preamble))

    elif stype == "op":  # binary: left, right, then emit op
        assert node.left is not None and node.right is not None
        _build_partial_instructions(
            node.left,
            symbol_library,
            const_free_ids,
            cached_arrays,
            token_ids_list,
            arities_list,
            extra_int_list,
            float_vals_list,
            python_ops_list,
            const_counter,
        )
        _build_partial_instructions(
            node.right,
            symbol_library,
            const_free_ids,
            cached_arrays,
            token_ids_list,
            arities_list,
            extra_int_list,
            float_vals_list,
            python_ops_list,
            const_counter,
        )
        cython_id = symbol_library.get_cython_id(sym)
        arities_list.append(2)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        if cython_id >= 0:
            token_ids_list.append(cython_id)
            python_ops_list.append(None)
        else:
            token_ids_list.append(_eval.PYTHON)
            python_ops_list.append(_compile_fallback_binary(symbol_library.get_np_fn(sym), symbol_library.preamble))

    else:
        raise ValueError(
            f"Unexpected token '{sym}' with type '{stype}' in partial instruction builder. "
            "Constant-free subtrees containing this token should have been pre-evaluated."
        )


def _prepare_cython_traversal_partial(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """Like ``_prepare_cython_traversal`` but pre-evaluates all maximal
    constant-free subtrees against *X* and replaces them with ``CACHED`` leaf
    instructions.

    This is only beneficial when the compiled callable will be called repeatedly
    with the same *X* but varying *C* — i.e. inside the L-BFGS-B parameter
    estimation loop.  The pre-evaluation cost is paid once at compile time; every
    subsequent call only executes the nodes that actually depend on *C*.

    Args:
        expr: Expression as a token list or a Node tree.
        symbol_library: Symbol library.
        X: Input data, shape ``(n_samples, n_features)``.  Must be C-contiguous
            float64; the caller is responsible for ensuring this.

    Returns:
        Same 5-tuple as ``_prepare_cython_traversal``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    tree = tokens_to_tree(expr, symbol_library) if isinstance(expr, list) else expr

    has_const, const_free_nodes = _collect_maximal_const_free(tree, symbol_library)

    # Nothing to cache: whole expression is const-free, or it is pure constants (e.g. just C).
    if not has_const or not const_free_nodes:
        return _prepare_cython_traversal(tree, symbol_library)

    # Pre-evaluate each maximal constant-free subtree against X.
    cached_arrays: Dict = {}
    for node in const_free_nodes:
        t_ids, t_aris, t_ei, t_fv, t_po = _prepare_cython_traversal(node, symbol_library)
        arr = _eval.execute(X, np.array([], dtype=np.float64), t_ids, t_aris, t_ei, t_fv, t_po)
        cached_arrays[id(node)] = np.ascontiguousarray(arr, dtype=np.float64)

    const_free_ids = set(cached_arrays.keys())

    # Build the reduced instruction sequence.
    token_ids_list: List = []
    arities_list: List = []
    extra_int_list: List = []
    float_vals_list: List = []
    python_ops_list: List = []
    const_counter = [0]

    _build_partial_instructions(
        tree,
        symbol_library,
        const_free_ids,
        cached_arrays,
        token_ids_list,
        arities_list,
        extra_int_list,
        float_vals_list,
        python_ops_list,
        const_counter,
    )

    return (
        np.array(token_ids_list, dtype=np.int32),
        np.array(arities_list, dtype=np.int32),
        np.array(extra_int_list, dtype=np.int32),
        np.array(float_vals_list, dtype=np.float64),
        python_ops_list,
    )


def _prepare_cython_traversal(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Convert an expression into the postfix instruction arrays consumed by the
    Cython evaluator.

    The const-index assignment order matches ``_tree_to_function_rec``: constants
    are numbered in left-first depth-first traversal order, which is identical
    to their order of appearance in the postfix token list.

    Args:
        expr: Expression as a token list or a Node tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.

    Returns:
        A 5-tuple ``(token_ids, arities, extra_int, float_vals, python_ops)``
        where the first four are 1-D NumPy arrays and the last is a list of
        Python callables (``None`` for C-implemented tokens).
    """
    if isinstance(expr, list):
        tree = tokens_to_tree(expr, symbol_library)
    else:
        tree = expr

    postfix: List[str] = tree.to_list(notation="postfix")
    n = len(postfix)

    token_ids = np.zeros(n, dtype=np.int32)
    arities = np.zeros(n, dtype=np.int32)
    extra_int = np.zeros(n, dtype=np.int32)
    float_vals = np.zeros(n, dtype=np.float64)
    python_ops: List = [None] * n

    const_counter = 0

    for i, token in enumerate(postfix):
        stype = symbol_library.get_type(token)
        cython_id = symbol_library.get_cython_id(token)

        if stype == "var":
            # Extract column index from np_fn of the form "X[:, k]"
            np_fn = symbol_library.get_np_fn(token)
            col = int(np_fn.split(", ")[1].rstrip("]"))
            token_ids[i] = _eval.VAR
            extra_int[i] = col

        elif stype == "const":
            token_ids[i] = _eval.CONST
            extra_int[i] = const_counter
            const_counter += 1

        elif stype == "lit":
            token_ids[i] = _eval.FLOAT
            float_vals[i] = eval(symbol_library.get_np_fn(token), {"np": np})

        elif is_float(token):
            token_ids[i] = _eval.FLOAT
            float_vals[i] = float(token)

        elif stype == "fn":
            arities[i] = 1
            if cython_id >= 0:
                token_ids[i] = cython_id
            else:
                token_ids[i] = _eval.PYTHON
                python_ops[i] = _compile_fallback_unary(symbol_library.get_np_fn(token), symbol_library.preamble)

        elif stype == "op":
            arities[i] = 2
            if cython_id >= 0:
                token_ids[i] = cython_id
            else:
                token_ids[i] = _eval.PYTHON
                python_ops[i] = _compile_fallback_binary(symbol_library.get_np_fn(token), symbol_library.preamble)

        else:
            raise ValueError(
                f"Token '{token}' has unknown type '{stype}'. Ensure all tokens are registered in the symbol library."
            )

    return token_ids, arities, extra_int, float_vals, python_ops


def _expr_to_cython_callable(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Compile an expression into a fast callable backed by the Cython evaluator.

    All operations present in the default symbol library are dispatched as tight
    C loops with no Python overhead and no intermediate array allocation.
    User-defined symbols without a Cython ID fall back to Python callables
    compiled from their ``np_fn`` string at this call site (one-time cost).

    When the compiled Cython extension is not available this function
    transparently uses the pure-Python fallback evaluator instead.

    Examples:
        >>> f = _expr_to_cython_callable(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]))
        array([2., 3., 4.])
        >>> f = _expr_to_cython_callable(["sin", "(", "X_0", ")"])
        >>> import math
        >>> abs(f(np.array([[math.pi / 2]]), np.array([]))[0] - 1.0) < 1e-10
        np.True_

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D C-contiguous float64 array of
        shape ``(n_samples, n_features)`` and ``C`` is a 1-D float64 array of
        constant values (pass an empty array or ``None`` for constant-free
        expressions). Returns a 1-D output array of shape ``(n_samples,)``.
    """
    token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)

    def _callable_(X: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        if C is None:
            C = np.array([], dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        C = np.ascontiguousarray(C, dtype=np.float64)
        return _eval.execute(X, C, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


def _expr_to_cython_error_callable(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    X: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Compile an expression into a fast RMSE callable backed by the Cython evaluator.

    Equivalent to [_expr_to_cython_callable][SRToolkit.utils.expression_compiler._expr_to_cython_callable] followed by an RMSE reduction,
    but the residual sum-of-squares is computed in C inside the Cython extension
    without creating an intermediate output array.

    When *X* is provided, all maximal constant-free subtrees are pre-evaluated
    against *X* at compile time and replaced with ``CACHED`` instructions.
    The returned callable then only evaluates the nodes that directly depend on
    *C* on each call — a significant speedup when the same *X* is used for
    repeated evaluations with varying *C* (e.g. inside the L-BFGS-B loop).

    Examples:
        >>> f = _expr_to_cython_error_callable(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]), np.array([2.0, 3.0, 4.0]))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        X: Optional input data used to pre-evaluate constant-free subtrees.
            Must be C-contiguous float64, shape ``(n_samples, n_features)``.
            When ``None`` (default), the full expression is compiled without caching.

    Returns:
        A callable ``f(X, C, y)`` returning the scalar RMSE as a float.
    """
    if X is not None:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal_partial(
            expr, symbol_library, X
        )
    else:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)

    def _callable_(X: np.ndarray, C: np.ndarray, y: np.ndarray) -> float:
        if C is None:
            C = np.array([], dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        C = np.ascontiguousarray(C, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        return _eval.execute_error(X, C, y, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


# ── Stack_py evaluator backend ──────────────────────────────────────────────


def _expr_to_python_callable(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Compile an expression into a callable backed by the pure-Python evaluator.

    Uses the same postfix instruction arrays as the Cython backend but executes
    them through the pure-Python interpreter in ``_eval_cython.py``. This avoids
    the Cython→Python boundary overhead that occurs when custom (non-default)
    symbols fall back to Python callables inside the compiled extension, making it
    faster than ``_expr_to_cython_callable`` for symbol libraries that contain
    many user-defined symbols.

    Examples:
        >>> f = _expr_to_python_callable(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]))
        array([2., 3., 4.])

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D array of shape
        ``(n_samples, n_features)`` and ``C`` is a 1-D array of constant values
        (pass ``None`` or an empty array for constant-free expressions).
        Returns a 1-D output array of shape ``(n_samples,)``.
    """
    token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)

    def _callable_(X: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        if C is None:
            C = np.array([], dtype=np.float64)
        return _eval_py.execute(X, C, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


def _expr_to_python_stack_error_callable(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    X: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Compile an expression into an RMSE callable backed by the pure-Python evaluator.

    Uses the same postfix instruction arrays as the Cython backend but executes
    them through the pure-Python interpreter in ``_eval_cython.py``.  This avoids
    the Cython→Python boundary overhead that occurs when custom (non-default)
    symbols fall back to Python callables inside the compiled extension, making it
    faster than ``_expr_to_cython_error_callable`` for symbol libraries that contain
    many user-defined symbols.

    When *X* is provided, all maximal constant-free subtrees are pre-evaluated
    against *X* at compile time and replaced with ``CACHED`` instructions,
    giving the same speedup as the Cython backend for repeated calls with
    varying *C*.

    Examples:
        >>> f = _expr_to_python_stack_error_callable(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]), np.array([2.0, 3.0, 4.0]))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        X: Optional input data used to pre-evaluate constant-free subtrees.
            Must be C-contiguous float64, shape ``(n_samples, n_features)``.
            When ``None`` (default), the full expression is compiled without caching.

    Returns:
        A callable ``f(X, C, y)`` returning the scalar RMSE as a float.
    """
    if X is not None:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal_partial(
            expr, symbol_library, X
        )
    else:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)

    def _callable_(X: np.ndarray, C: np.ndarray, y: np.ndarray) -> float:
        if C is None:
            C = np.array([], dtype=np.float64)
        return _eval_py.execute_error(X, C, y, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


# ── Interface for compiling expressions ──────────────────────────────────────────────


def compile_expr(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    backend: str = "stack",
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    Compile an expression into a callable ``f(X, C) → np.ndarray``.

    Examples:
        >>> f = compile_expr(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), None)
        array([2., 3., 4.])
        >>> f = compile_expr(["X_0", "+", "1"], backend="codegen")
        >>> f(np.array([[1], [2], [3]]), np.array([]))
        array([2, 3, 4])

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        backend: Evaluation backend. One of:

            - ``"stack"`` (default): postfix stack-machine evaluator backed by
              Cython; falls back to pure-Python when the compiled extension is
              unavailable.
            - ``"codegen"``: generates Python/NumPy source via ``exec()``;
              compatible with any custom symbol library.
            - ``"stack_py"``: pure-Python stack-machine evaluator; avoids the
              Cython→Python boundary overhead when the library contains many
              custom symbols.

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D array of shape
        ``(n_samples, n_features)`` and ``C`` is a 1-D array of constant values
        (pass ``None`` or an empty array for constant-free expressions).
        Returns a 1-D output array of shape ``(n_samples,)``.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
    """
    if backend == "stack":
        return _expr_to_cython_callable(expr, symbol_library)
    elif backend == "codegen":
        return _expr_to_executable_function(expr, symbol_library)
    elif backend == "stack_py":
        return _expr_to_python_callable(expr, symbol_library)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of: 'stack', 'codegen', 'stack_py'.")


def compile_expr_rmse(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
    backend: str = "stack",
    X: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """
    Compile an expression into an RMSE callable ``f(X, C, y) → float``.

    Examples:
        >>> f = compile_expr_rmse(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]), np.array([2.0, 3.0, 4.0]))
        0.0
        >>> f = compile_expr_rmse(["X_0", "+", "1"], backend="codegen")
        >>> print(float(f(np.array([[1], [2], [3]]), np.array([]), np.array([2, 3, 4]))))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library used to look up token types.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].
        backend: Evaluation backend. One of:

            - ``"stack"`` (default): postfix stack-machine evaluator backed by
              Cython; RMSE is computed in C without an intermediate output array.
              Falls back to pure-Python when the compiled extension is unavailable.
            - ``"codegen"``: generates Python/NumPy source via ``exec()``;
              compatible with any custom symbol library.
            - ``"stack_py"``: pure-Python stack-machine evaluator; avoids the
              Cython→Python boundary overhead when the library contains many
              custom symbols.

        X: Optional input data of shape ``(n_samples, n_features)``. When provided
            with ``backend="stack"`` or ``backend="stack_py"``, all constant-free
            subtrees are pre-evaluated against *X* at compile time — a significant
            speedup when the same *X* is reused across many calls with varying *C*
            (e.g. inside an optimiser loop). Ignored for ``backend="codegen"``.

    Returns:
        A callable ``f(X, C, y)`` returning the scalar RMSE as a float.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
    """
    if backend == "stack":
        return _expr_to_cython_error_callable(expr, symbol_library, X)
    elif backend == "codegen":
        return _expr_to_error_function(expr, symbol_library)
    elif backend == "stack_py":
        return _expr_to_python_stack_error_callable(expr, symbol_library, X)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of: 'stack', 'codegen', 'stack_py'.")
