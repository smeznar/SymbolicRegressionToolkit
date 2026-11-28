"""Functions for compiling symbolic expressions into executable Python callables.

The primary interface are two functions:

- [compile_expr][SRToolkit.utils.expression_compiler.compile_expr]:
  returns a callable ``f(X, C) → np.ndarray``; ``C`` holds one value per free parameter (see
  [parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels]).
- [compile_expr_rmse][SRToolkit.utils.expression_compiler.compile_expr_rmse]:
  returns an RMSE callable ``f(X, C, y) → float``.

Both accept a ``backend`` parameter that selects the evaluation engine:

- ``"stack"`` (default): postfix stack-machine evaluator backed by Cython; falls
  back to pure-Python automatically when the compiled extension is unavailable
  (e.g. installed from source without a C compiler). The fallback is about as fast
  as ``"codegen"`` and up to about 2x slower than the compiled extension, mostly on
  small datasets; a warning is issued the first time it is used, and
  [CYTHON_AVAILABLE][SRToolkit.utils.expression_compiler.CYTHON_AVAILABLE] tells
  which one is in use.
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
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

# Import the evaluator backend. Python prefers the compiled `_eval_cython.so`
# (or `.pyd` on Windows) over `_eval_cython.py` when both are present, so this
# single import transparently selects the fast Cython path when available and
# falls back to the pure-Python implementation otherwise.
from SRToolkit.utils import _eval_cython as _eval  # type: ignore[attr-defined]
from SRToolkit.utils.expression import Expression, _to_expression
from SRToolkit.utils.expression_tree import (
    Node,
    _as_tree,
    _constant_indices,
    is_float,
    tokens_to_tree,  # noqa: F401  (used by the doctests)
)
from SRToolkit.utils.symbol_library import SymbolLibrary

CYTHON_AVAILABLE: bool = not str(_eval.__file__).endswith(".py")
"""Whether the compiled Cython extension is used by the ``"stack"`` backend."""

_fallback_warned = False


def _warn_if_no_cython() -> None:
    """Warn once that the ``"stack"`` backend runs on the pure-Python fallback."""
    global _fallback_warned
    if not CYTHON_AVAILABLE and not _fallback_warned:
        _fallback_warned = True
        warnings.warn(
            "SRToolkit's compiled Cython extension is not available, so the 'stack' backend uses the slower "
            "pure-Python evaluator (up to about 2x slower parameter fitting). Reinstall with a C compiler available "
            "(or from a prebuilt wheel) to fix this.",
            RuntimeWarning,
            stacklevel=3,
        )


def _variable_column(token: str, symbol_library: SymbolLibrary) -> int:
    """Return the column of ``X`` that a variable symbol reads.

    Returns:
        The index ``k`` from the variable's ``python_str``, which has the form ``"X[:, k]"``.
    """
    return int(symbol_library[token].python_str.split(", ")[1].rstrip("]"))


def _check_X(X: np.ndarray, num_columns: int) -> np.ndarray:
    """Return ``X`` as a C-contiguous float64 array after checking that it is 2-D and wide enough.

    The Cython evaluator indexes columns without bounds checks, so a narrow ``X`` must be rejected
    before evaluation rather than read out of bounds.

    Returns:
        ``X`` as a C-contiguous float64 array.

    Raises:
        ValueError: If ``X`` is not 2-D or has fewer than ``num_columns`` columns.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D with shape (n_samples, n_features), got shape {X.shape}.")
    if X.shape[1] < num_columns:
        raise ValueError(
            f"X has {X.shape[1]} column(s), but the expression uses column {num_columns - 1} "
            f"(X needs at least {num_columns})."
        )
    return X


def _columns_used(token_ids: np.ndarray, extra_int: np.ndarray) -> int:
    """Return the number of columns of ``X`` an instruction array needs (highest variable column + 1)."""
    var_columns = extra_int[token_ids == _eval.VAR]
    return int(var_columns.max()) + 1 if var_columns.size else 0


def _num_parameters(token_ids: np.ndarray, extra_int: np.ndarray) -> int:
    """Return the number of parameters an instruction array reads from ``C`` (highest index + 1)."""
    indices = extra_int[token_ids == _eval.CONST]
    return int(indices.max()) + 1 if indices.size else 0


def _check_num_constants(C: np.ndarray | None, num_constants: int) -> None:
    """Raise if *C* holds fewer values than the expression has free parameters.

    Every backend calls this, so they fail the same way; the Cython evaluator does not
    bounds-check ``C`` and would otherwise read past the end of the array.
    """
    given = 0 if C is None else len(C)
    if given < num_constants:
        raise ValueError(f"The expression has {num_constants} free constant(s), but C holds only {given} value(s).")


_eval_py_spec = _ilu.spec_from_file_location(
    "_eval_cython_py",
    _pl.Path(__file__).parent / "_eval_cython.py",
)
_eval_py = _ilu.module_from_spec(_eval_py_spec)  # type: ignore[arg-type]
_eval_py_spec.loader.exec_module(_eval_py)  # type: ignore[union-attr]


# ── Codegen evaluator backend ──────────────────────────────────────────────


def _expr_to_executable_function(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Compile an expression into an executable Python function.

    The returned callable evaluates the expression over a batch of inputs and a vector
    of constant values.

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
        >>> # Custom symbols can use modules other than numpy, listed in the library's ``modules``.
        >>> from SRToolkit.utils import FN_PREFIX
        >>> symbol_library = SymbolLibrary(["+"], 1, modules={"m": "math"})
        >>> symbol_library.add_symbol("lgamma", "fn", FN_PREFIX, "np.vectorize(m.lgamma)({})")
        >>> executable_fun = _expr_to_executable_function(["lgamma", "(", "X_0", ")"], symbol_library)
        >>> executable_fun(np.array([[1.0], [3.0]]), np.array([]))
        array([0.        , 0.69314718])

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, modules).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C)`` where ``X`` is a 2-D array of shape ``(n_samples, n_features)`` and ``C`` is a 1-D array of constant values. Returns a 1-D output array of shape ``(n_samples,)``.

    Raises:
        TypeError: If ``expr`` is neither a list of token strings nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    tree = _as_tree(expr, symbol_library)
    slots = _constant_indices(tree.to_list(symbol_library, notation="postfix"), symbol_library)
    code, symbol, _ = _tree_to_function_rec(tree, symbol_library, iter(slots))
    num_constants = len(set(slots))

    fun_string = "def _executable_expression_(X, C):\n"
    if num_constants > 0:
        fun_string += f"\t_check_num_constants(C, {num_constants})\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    # An expression of numbers only evaluates to a scalar; broadcast it like the other backends.
    fun_string += f"\t_out_ = {symbol}\n\treturn _out_ if np.ndim(_out_) else np.full(X.shape[0], _out_)"

    fun_assignment_dict: dict[str, Callable] = {}
    namespace = {**symbol_library.module_namespace(), "_check_num_constants": _check_num_constants}
    exec(fun_string, namespace, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def _expr_to_error_function(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
) -> Callable[[np.ndarray, np.ndarray | None, np.ndarray], float]:
    """Compile an expression into a callable that computes the RMSE against target values.

    Examples:
        >>> executable_fun = _expr_to_error_function(["X_0", "+", "1"])
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0
        >>> tree = tokens_to_tree(["X_0", "+", "1"], SymbolLibrary.default_symbols(1))
        >>> executable_fun = _expr_to_error_function(tree)
        >>> print(float(executable_fun(np.array([[1], [2], [3], [4]]), np.array([]), np.array([2, 3, 4, 5]))))
        0.0

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Defines token semantics (NumPy function strings, modules).
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        A callable ``f(X, C, y)`` where ``X`` is a 2-D array of shape ``(n_samples, n_features)``, ``C`` is a 1-D array of constant values, and ``y`` is a 1-D target array. Returns the scalar RMSE as a float.

    Raises:
        TypeError: If ``expr`` is neither a list of token strings nor a [Node][SRToolkit.utils.expression_tree.Node].
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    tree = _as_tree(expr, symbol_library)
    slots = _constant_indices(tree.to_list(symbol_library, notation="postfix"), symbol_library)
    code, symbol, _ = _tree_to_function_rec(tree, symbol_library, iter(slots))
    num_constants = len(set(slots))

    fun_string = "def _executable_expression_(X, C, y):\n"
    if num_constants > 0:
        fun_string += f"\t_check_num_constants(C, {num_constants})\n"
    for c in code:
        fun_string += "\t" + c + "\n"
    fun_string += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))"

    fun_assignment_dict: dict[str, Callable] = {}
    namespace = {**symbol_library.module_namespace(), "_check_num_constants": _check_num_constants}
    exec(fun_string, namespace, fun_assignment_dict)
    return fun_assignment_dict["_executable_expression_"]


def _tree_to_function_rec(
    tree: Node,
    symbol_library: SymbolLibrary,
    const_slots: Iterator[int],
    var_counter: int = 0,
) -> tuple[list[str], str, int]:
    """Recursively convert a parse tree into lines of Python code for expression evaluation.

    This is a low-level helper used by the codegen backend of [compile_expr][SRToolkit.utils.expression_compiler.compile_expr] and
    [compile_expr_rmse][SRToolkit.utils.expression_compiler.compile_expr_rmse]. Call those functions directly unless you need
    fine-grained control over code generation.

    Args:
        tree: Root of the subtree to convert.
        symbol_library: Provides NumPy function strings for each token.
        const_slots: Index into ``C`` of each remaining constant, in traversal order (see
            ``_constant_indices``); one is consumed per constant leaf.
        var_counter: Running count of intermediate variables, used to generate unique
            names. Default ``0``.

    Returns:
        A 3-tuple ``(code, symbol, var_counter)`` where ``code`` is a list of Python assignment strings forming the expression body, ``symbol`` is the name of the variable holding this subtree's result, and ``var_counter`` is the updated counter.

    Raises:
        Exception: If the tree contains a token that is neither a recognized symbol nor
            a numeric literal.
    """
    if tree.left is None and tree.right is None:
        leaf = symbol_library.get(tree.symbol)
        if leaf is not None and leaf.type == "var":
            return [], leaf.python_str, var_counter
        elif leaf is not None and leaf.type == "lit":
            output_symbol = "y_{}".format(var_counter)
            code = [f"{output_symbol} = np.full(X.shape[0], {leaf.python_str})"]
            return code, output_symbol, var_counter + 1
        elif leaf is not None and leaf.type == "const":
            return [], leaf.python_str.format(next(const_slots)), var_counter
        else:
            if is_float(tree.symbol):
                return [], f"({tree.symbol})", var_counter
            else:
                raise Exception(f"Encountered invalid symbol {tree.symbol} while converting to function.")

    elif tree.left is not None and tree.right is None:
        code, symbol, var_counter = _tree_to_function_rec(tree.left, symbol_library, const_slots, var_counter)
        output_symbol = "y_{}".format(var_counter)
        code.append(f"{output_symbol} = " + symbol_library[tree.symbol].python_str.format(symbol))
        return code, output_symbol, var_counter + 1

    else:
        assert tree.right is not None, "Right child should be present in this branch."
        assert tree.left is not None, "Left child should be present if right child is present."
        left_code, left_symbol, var_counter = _tree_to_function_rec(tree.left, symbol_library, const_slots, var_counter)
        right_code, right_symbol, var_counter = _tree_to_function_rec(
            tree.right, symbol_library, const_slots, var_counter
        )
        output_symbol = "y_{}".format(var_counter)
        code = left_code + right_code
        code.append(f"{output_symbol} = " + symbol_library[tree.symbol].python_str.format(left_symbol, right_symbol))
        return code, output_symbol, var_counter + 1


# ── Stack evaluator backend ──────────────────────────────────────────────


def _compile_fallback_unary(python_str: str, namespace: dict[str, Any]) -> Callable:
    """Build a Python callable ``f(x) → np.ndarray`` from a unary python_str string, whose names are
    looked up in ``namespace`` (see [SymbolLibrary.module_namespace][SRToolkit.utils.symbol_library.SymbolLibrary.module_namespace]).
    """
    ns: dict[str, Callable] = {}
    exec(f"def _fn_(_x_):\n    return {python_str.format('_x_')}\n", dict(namespace), ns)
    return ns["_fn_"]


def _compile_fallback_binary(python_str: str, namespace: dict[str, Any]) -> Callable:
    """Build a Python callable ``f(a, b) → np.ndarray`` from a binary python_str string, whose names
    are looked up in ``namespace``, as in ``_compile_fallback_unary``.
    """
    ns: dict[str, Callable] = {}
    exec(f"def _fn_(_a_, _b_):\n    return {python_str.format('_a_', '_b_')}\n", dict(namespace), ns)
    return ns["_fn_"]


def _collect_maximal_const_free(node: Node, symbol_library: SymbolLibrary) -> tuple[bool, list[Node]]:
    """Return ``(has_const, maximal_const_free_nodes)`` for the subtree rooted at *node*.

    A subtree is maximal const-free if it contains no ``C`` tokens and its parent does
    (or the whole expression is const-free). These are exactly the subtrees that can be
    pre-evaluated once against X and reused across optimizer calls.
    """
    if is_float(node.symbol):
        return False, []
    symbol = symbol_library.get(node.symbol)
    if symbol is not None and symbol.type == "const":
        return True, []

    left_has_const, left_maximal = (
        _collect_maximal_const_free(node.left, symbol_library) if node.left is not None else (False, [])
    )
    right_has_const, right_maximal = (
        _collect_maximal_const_free(node.right, symbol_library) if node.right is not None else (False, [])
    )

    if not (left_has_const or right_has_const):
        return False, []

    result: list[Node] = []
    if node.left is not None:
        result.extend([node.left] if not left_has_const else left_maximal)
    if node.right is not None:
        result.extend([node.right] if not right_has_const else right_maximal)
    return True, result


def _build_partial_instructions(
    node: Node,
    symbol_library: SymbolLibrary,
    const_free_ids: set,
    cached_arrays: dict,
    token_ids_list: list,
    arities_list: list,
    extra_int_list: list,
    float_vals_list: list,
    python_ops_list: list,
    const_slots: Iterator[int],
) -> None:
    """Recursively emit postfix instructions, substituting CACHED leaves for
    pre-evaluated constant-free subtrees.
    """
    if id(node) in const_free_ids:
        token_ids_list.append(_eval.CACHED)
        arities_list.append(0)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        python_ops_list.append(cached_arrays[id(node)])
        return

    sym = node.symbol
    symbol = symbol_library.get(sym)
    stype = None if symbol is None else symbol.type

    if stype == "const":
        token_ids_list.append(_eval.CONST)
        arities_list.append(0)
        extra_int_list.append(next(const_slots))
        float_vals_list.append(0.0)
        python_ops_list.append(None)

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
            const_slots,
        )
        assert symbol is not None
        arities_list.append(1)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        if symbol.cython_id >= 0:
            token_ids_list.append(symbol.cython_id)
            python_ops_list.append(None)
        else:
            token_ids_list.append(_eval.PYTHON)
            python_ops_list.append(_compile_fallback_unary(symbol.python_str, symbol_library.module_namespace()))

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
            const_slots,
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
            const_slots,
        )
        assert symbol is not None
        arities_list.append(2)
        extra_int_list.append(0)
        float_vals_list.append(0.0)
        if symbol.cython_id >= 0:
            token_ids_list.append(symbol.cython_id)
            python_ops_list.append(None)
        else:
            token_ids_list.append(_eval.PYTHON)
            python_ops_list.append(_compile_fallback_binary(symbol.python_str, symbol_library.module_namespace()))

    else:
        raise ValueError(
            f"Unexpected token '{sym}' with type '{stype}' in partial instruction builder. "
            "Constant-free subtrees containing this token should have been pre-evaluated."
        )


def _prepare_cython_traversal_partial(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
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
    tree = _as_tree(expr, symbol_library)

    has_const, const_free_nodes = _collect_maximal_const_free(tree, symbol_library)

    # Nothing to cache: whole expression is const-free, or it is pure constants (e.g. just C).
    if not has_const or not const_free_nodes:
        return _prepare_cython_traversal(tree, symbol_library)

    # Pre-evaluate each maximal constant-free subtree against X.
    cached_arrays: dict = {}
    for node in const_free_nodes:
        t_ids, t_aris, t_ei, t_fv, t_po = _prepare_cython_traversal(node, symbol_library)
        arr = _eval.execute(X, np.array([], dtype=np.float64), t_ids, t_aris, t_ei, t_fv, t_po)
        cached_arrays[id(node)] = np.ascontiguousarray(arr, dtype=np.float64)

    const_free_ids = set(cached_arrays.keys())

    # Build the reduced instruction sequence.
    token_ids_list: list = []
    arities_list: list = []
    extra_int_list: list = []
    float_vals_list: list = []
    python_ops_list: list = []

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
        iter(_constant_indices(tree.to_list(symbol_library, notation="postfix"), symbol_library)),
    )

    return (
        np.array(token_ids_list, dtype=np.int32),
        np.array(arities_list, dtype=np.int32),
        np.array(extra_int_list, dtype=np.int32),
        np.array(float_vals_list, dtype=np.float64),
        python_ops_list,
    )


def _prepare_cython_traversal(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Convert an expression into the postfix instruction arrays consumed by the
    Cython evaluator.

    Constants read ``C`` at the index ``_constant_indices`` gives them (parameters by first
    appearance), as in ``_tree_to_function_rec``.

    Args:
        expr: Expression as a token list or a Node tree.
        symbol_library: Symbol library used to look up token types and Cython IDs.

    Returns:
        A 5-tuple ``(token_ids, arities, extra_int, float_vals, python_ops)``
        where the first four are 1-D NumPy arrays and the last is a list of
        Python callables (``None`` for C-implemented tokens).
    """
    tree = _as_tree(expr, symbol_library)

    postfix: list[str] = tree.to_list(notation="postfix")
    n = len(postfix)

    token_ids = np.zeros(n, dtype=np.int32)
    arities = np.zeros(n, dtype=np.int32)
    extra_int = np.zeros(n, dtype=np.int32)
    float_vals = np.zeros(n, dtype=np.float64)
    python_ops: list = [None] * n

    const_slots = iter(_constant_indices(postfix, symbol_library))

    namespace = symbol_library.module_namespace()
    for i, token in enumerate(postfix):
        symbol = symbol_library.get(token)
        stype = None if symbol is None else symbol.type
        cython_id = -1 if symbol is None else symbol.cython_id

        if stype == "var":
            token_ids[i] = _eval.VAR
            extra_int[i] = _variable_column(token, symbol_library)

        elif stype == "const":
            token_ids[i] = _eval.CONST
            extra_int[i] = next(const_slots)

        elif stype == "lit":
            token_ids[i] = _eval.FLOAT
            float_vals[i] = eval(symbol_library[token].python_str, dict(namespace))

        elif is_float(token):
            token_ids[i] = _eval.FLOAT
            float_vals[i] = float(token)

        elif stype == "fn":
            arities[i] = 1
            if cython_id >= 0:
                token_ids[i] = cython_id
            else:
                token_ids[i] = _eval.PYTHON
                python_ops[i] = _compile_fallback_unary(symbol_library[token].python_str, namespace)

        elif stype == "op":
            arities[i] = 2
            if cython_id >= 0:
                token_ids[i] = cython_id
            else:
                token_ids[i] = _eval.PYTHON
                python_ops[i] = _compile_fallback_binary(symbol_library[token].python_str, namespace)

        else:
            raise ValueError(
                f"Token '{token}' has unknown type '{stype}'. Ensure all tokens are registered in the symbol library."
            )

    return token_ids, arities, extra_int, float_vals, python_ops


def _expr_to_cython_callable(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Compile an expression into a fast callable backed by the Cython evaluator.

    All operations present in the default symbol library are dispatched as tight
    C loops with no Python overhead and no intermediate array allocation.
    User-defined symbols without a Cython ID fall back to Python callables
    compiled from their ``python_str`` string at this call site (one-time cost).

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
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)
    num_constants = _num_parameters(token_ids, extra_int)
    num_columns = _columns_used(token_ids, extra_int)

    def _callable_(X: np.ndarray, C: np.ndarray | None) -> np.ndarray:
        if num_constants > 0:
            _check_num_constants(C, num_constants)
        if C is None:
            C = np.array([], dtype=np.float64)
        X = _check_X(X, num_columns)
        C = np.ascontiguousarray(C, dtype=np.float64)
        return _eval.execute(X, C, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


def _expr_to_cython_error_callable(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
    X: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray | None, np.ndarray], float]:
    """Compile an expression into a fast RMSE callable backed by the Cython evaluator.

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
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    if X is not None:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal_partial(
            expr, symbol_library, X
        )
    else:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)
    num_constants = _num_parameters(token_ids, extra_int)
    num_columns = _columns_used(token_ids, extra_int)

    def _callable_(X: np.ndarray, C: np.ndarray | None, y: np.ndarray) -> float:
        if num_constants > 0:
            _check_num_constants(C, num_constants)
        if C is None:
            C = np.array([], dtype=np.float64)
        X = _check_X(X, num_columns)
        C = np.ascontiguousarray(C, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        return _eval.execute_error(X, C, y, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


# ── Stack_py evaluator backend ──────────────────────────────────────────────


def _expr_to_python_callable(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Compile an expression into a callable backed by the pure-Python evaluator.

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
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)
    num_constants = _num_parameters(token_ids, extra_int)

    def _callable_(X: np.ndarray, C: np.ndarray | None) -> np.ndarray:
        if num_constants > 0:
            _check_num_constants(C, num_constants)
        if C is None:
            C = np.array([], dtype=np.float64)
        return _eval_py.execute(X, C, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


def _expr_to_python_stack_error_callable(
    expr: list[str] | Node,
    symbol_library: SymbolLibrary | None = None,
    X: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray | None, np.ndarray], float]:
    """Compile an expression into an RMSE callable backed by the pure-Python evaluator.

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
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    if X is not None:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal_partial(
            expr, symbol_library, X
        )
    else:
        token_ids, arities, extra_int, float_vals, python_ops = _prepare_cython_traversal(expr, symbol_library)
    num_constants = _num_parameters(token_ids, extra_int)

    def _callable_(X: np.ndarray, C: np.ndarray | None, y: np.ndarray) -> float:
        if num_constants > 0:
            _check_num_constants(C, num_constants)
        if C is None:
            C = np.array([], dtype=np.float64)
        return _eval_py.execute_error(X, C, y, token_ids, arities, extra_int, float_vals, python_ops)

    return _callable_


# ── Interface for compiling expressions ──────────────────────────────────────────────


def _tree_and_library(expr: Any, symbol_library: SymbolLibrary | None) -> tuple[Node, SymbolLibrary]:
    # The tree of compile_expr's input and the library to compile it in: for None, an Expression's
    # own library, otherwise the active or default one.
    if symbol_library is None:
        symbol_library = expr.symbol_library if isinstance(expr, Expression) else SymbolLibrary.get_or_default()
    tree = expr if isinstance(expr, Node) else _to_expression(expr, symbol_library).tree
    return tree, symbol_library


def compile_expr(
    expr: list[str] | Expression | Node,
    symbol_library: SymbolLibrary | None = None,
    backend: str = "stack",
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Compile an expression into a callable ``f(X, C) → np.ndarray``.

    An [Expression][SRToolkit.utils.expression.Expression] is parsed once, however often it is
    compiled. Its specs are not used: ``C`` holds every parameter, fixed values included.

    Args:
        expr: The expression, as an infix token list, an
            [Expression][SRToolkit.utils.expression.Expression] or a
            [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: The library to compile ``expr`` in. ``None``: an ``Expression``'s own
            library, otherwise the active or default library.
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
        A callable ``f(X, C)`` where ``X`` has shape ``(n_samples, n_features)`` and ``C`` holds at
        least one value per parameter, in the order of
        [parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels] (every occurrence of a
        labelled constant reads the same entry). Pass ``None`` or an empty array for a
        constant-free expression; too few values raise ``ValueError`` when ``f`` is called.
        ``f`` returns an array of shape ``(n_samples,)``.

    Raises:
        TypeError: If ``expr`` is neither a token list, an ``Expression`` nor a ``Node``.
        ExpressionParseError: If ``expr`` is not a well-formed expression.
        ValueError: If ``backend`` is not one of the supported values.

    Examples:
        >>> f = compile_expr(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), None)
        array([2., 3., 4.])
        >>> f = compile_expr(["X_0", "+", "1"], backend="codegen")
        >>> f(np.array([[1], [2], [3]]), np.array([]))
        array([2, 3, 4])
    """
    expr, symbol_library = _tree_and_library(expr, symbol_library)
    if backend == "stack":
        _warn_if_no_cython()
        return _expr_to_cython_callable(expr, symbol_library)
    elif backend == "codegen":
        return _expr_to_executable_function(expr, symbol_library)
    elif backend == "stack_py":
        return _expr_to_python_callable(expr, symbol_library)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of: 'stack', 'codegen', 'stack_py'.")


def compile_expr_rmse(
    expr: list[str] | Expression | Node,
    symbol_library: SymbolLibrary | None = None,
    backend: str = "stack",
    X: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray | None, np.ndarray], float]:
    """Compile an expression into an RMSE callable ``f(X, C, y) → float``.

    As [compile_expr][SRToolkit.utils.expression_compiler.compile_expr], an ``Expression`` is
    parsed once and its specs are not used.

    Args:
        expr: The expression, as in [compile_expr][SRToolkit.utils.expression_compiler.compile_expr].
        symbol_library: As in [compile_expr][SRToolkit.utils.expression_compiler.compile_expr].
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
        A callable ``f(X, C, y)`` returning the scalar RMSE as a float, with ``X`` and ``C`` as in
        [compile_expr][SRToolkit.utils.expression_compiler.compile_expr].

    Raises:
        TypeError: If ``expr`` is neither a token list, an ``Expression`` nor a ``Node``.
        ExpressionParseError: If ``expr`` is not a well-formed expression.
        ValueError: If ``backend`` is not one of the supported values.

    Examples:
        >>> f = compile_expr_rmse(["X_0", "+", "1"])
        >>> f(np.array([[1.0], [2.0], [3.0]]), np.array([]), np.array([2.0, 3.0, 4.0]))
        0.0
        >>> f = compile_expr_rmse(["X_0", "+", "1"], backend="codegen")
        >>> print(float(f(np.array([[1], [2], [3]]), np.array([]), np.array([2, 3, 4]))))
        0.0
    """
    expr, symbol_library = _tree_and_library(expr, symbol_library)
    if backend == "stack":
        _warn_if_no_cython()
        return _expr_to_cython_error_callable(expr, symbol_library, X)
    elif backend == "codegen":
        return _expr_to_error_function(expr, symbol_library)
    elif backend == "stack_py":
        return _expr_to_python_stack_error_callable(expr, symbol_library, X)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Must be one of: 'stack', 'codegen', 'stack_py'.")
