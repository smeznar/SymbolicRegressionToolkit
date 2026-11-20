"""Algebraic simplification of symbolic expressions using SymPy.
The functions in this script are work in progress and will be improved in the future for better accuracy and performance.
"""

import itertools
from typing import List, Optional, Union

from sympy import Basic, Expr, Symbol, expand, sympify
from sympy.core import Add, Mul, Pow

from SRToolkit.utils.expression_tree import Node, _as_tokens, is_float
from SRToolkit.utils.symbol_library import SymbolLibrary


def simplify(
    expr: Union[List[str], Node],
    symbol_library: Optional[SymbolLibrary] = None,
) -> Union[List[str], Node]:
    """Simplify an expression algebraically.

    Two successive steps are applied:

    1. **SymPy simplification** — expands and reduces the expression algebraically
       (e.g. ``X_0 * X_1 / X_0`` → ``X_1``).
    2. **Constant folding** — collapses any sub-expression containing no variables
       into a single free constant ``C`` (e.g. ``C * C + C`` → ``C``).

    A labelled constant (``C_0``) is one symbol however often it occurs, so ``C_0 - C_0``
    cancels. A label that still occurs more than once is kept and not folded, since folding it
    would add parameters; every other constant folds into the bare free constant.

    Examples:
        >>> expr = ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
        >>> print("".join(simplify(expr)))
        C+X_1

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library defining variables and constants.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        The simplified expression in the same form as the input (list if a list was given, [Node][SRToolkit.utils.expression_tree.Node] if a tree was given).

    Raises:
        Exception: If simplification fails or the result contains tokens absent from
            ``symbol_library``.
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_or_default()
    is_tree = isinstance(expr, Node)
    expr = _as_tokens(expr, symbol_library)

    variables = [Symbol(v) for v in symbol_library.get_symbols_of_type("var")]
    c = Symbol(symbol_library.constant_symbol or "__C__")

    # A shared label is kept as its own symbol and never folded, like a variable. Every other
    # constant gets a symbol of its own and folds freely.
    labels = [
        t
        for t in expr
        if (s := symbol_library.get(t)) is not None and s.type == "const" and t not in symbol_library.symbols
    ]
    shared = {t for t in labels if labels.count(t) > 1}
    variables += [Symbol(t) for t in shared]
    counter = itertools.count()
    tokens = [
        f"_{next(counter)}_"
        if (s := symbol_library.get(t)) is not None and s.type == "const" and t not in shared
        else t
        for t in expr
    ]

    eq = _simplify_constants(sympify("".join(tokens)), c, variables)[2][0][1]
    eq = _simplify_constants(expand(_enumerate_constants(eq, c)), c, variables)[2][0][1]
    expr = _sympy_to_sr(eq)
    if not _check_tree(expr, symbol_library):
        raise Exception(
            "Simplified expression contains invalid symbols. Possibly skip its simplification or add symbols to the SymbolLibrary."
        )

    if is_tree:
        return expr
    else:
        return expr.to_list(symbol_library=symbol_library, notation="infix")


def _check_tree(expr: Node, symbol_library: SymbolLibrary) -> bool:
    """Return ``True`` if every token in the tree is present in ``symbol_library`` or is numeric.

    Args:
        expr: Expression tree to validate.
        symbol_library: Symbol library to check against.

    Returns:
        ``True`` if all nodes are valid, ``False`` if any token is absent and non-numeric.
    """
    if symbol_library.get(expr.symbol) is None and not is_float(expr.symbol):
        return False
    if isinstance(expr.left, Node) and not _check_tree(expr.left, symbol_library):
        return False
    if isinstance(expr.right, Node) and not _check_tree(expr.right, symbol_library):
        return False

    return True


def _sympy_to_number(expr):
    """Return the numeric value of a SymPy number node as a Python ``int`` or ``float``."""
    evaluated = float(expr.evalf())
    return int(evaluated) if evaluated.is_integer() else evaluated


def _sympy_to_sr(expr: Union[Expr, Basic]) -> Node:
    """Convert a SymPy expression into a [Node][SRToolkit.utils.expression_tree.Node] tree.

    Division is handled explicitly: SymPy represents ``x/y`` as ``Mul(x, Pow(y, -1))``,
    so any ``Pow`` factor with exponent ``-1`` inside a ``Mul`` is converted to a ``"/"`` node.

    Args:
        expr: A SymPy expression object.

    Returns:
        The corresponding [Node][SRToolkit.utils.expression_tree.Node] tree.

    Raises:
        ValueError: If ``expr`` contains a SymPy construct that has no mapping to the
            supported token types.
    """
    if expr.is_Number:
        return Node(str(_sympy_to_number(expr)))

    if expr.is_Symbol:
        return Node(str(expr))

    if expr.is_Function:
        func_name = expr.func.__name__
        arg = _sympy_to_sr(expr.args[0])
        return Node(func_name, left=arg)

    if isinstance(expr, Add):
        args = expr.as_ordered_terms()
        # Detect subtraction
        if len(args) == 2 and args[1].is_Mul and args[1].args[0] == -1:
            return Node("-", _sympy_to_sr(-args[1]), _sympy_to_sr(args[0]))
        # Handle regular addition
        root = Node("+", _sympy_to_sr(args[1]), _sympy_to_sr(args[0]))
        for term in args[2:]:
            root = Node("+", _sympy_to_sr(term), root)
        return root

    if isinstance(expr, Mul):
        # Process factors explicitly, ensuring left-to-right associativity
        factors = list(expr.args)
        root = _sympy_to_sr(factors[0])  # Start with the first factor
        for factor in factors[1:]:
            if factor.is_Pow and factor.args[1] == -1:  # Division
                divisor = _sympy_to_sr(factor.args[0])
                root = Node("/", divisor, root)  # Left-to-right division
            else:  # Multiplication
                multiplicand = _sympy_to_sr(factor)
                root = Node("*", multiplicand, root)
        return root

    if isinstance(expr, Pow):
        base, exp = expr.args
        return Node("^", _sympy_to_sr(exp), _sympy_to_sr(base))

    raise ValueError(f"{expr}")


def _simplify_constants(eq, c, var):
    """Recursively fold constant-only sub-expressions in a SymPy expression.

    Any sub-tree that contains no variables is replaced by the constant symbol ``c``.

    Args:
        eq: SymPy expression to process.
        c: SymPy symbol representing the free constant.
        var: List of SymPy symbols representing input variables.

    Returns:
        A 3-tuple ``(has_var, has_c, substitutions)`` where ``has_var`` is ``True`` if the
        sub-expression depends on a variable, ``has_c`` is ``True`` if it depends on a
        constant, and ``substitutions`` is a list of ``(original, replacement)`` pairs.
    """
    if len(eq.args) == 0:
        if eq in var:
            return True, False, [(eq, eq)]
        elif eq in eq.free_symbols:
            return False, True, [(eq, c)]
        else:
            return False, False, [(eq, eq)]
    else:
        has_var, has_c, subs = [], [], []
        for a in eq.args:
            a_rec = _simplify_constants(a, c, var)
            has_var += [a_rec[0]]
            has_c += [a_rec[1]]
            subs += [a_rec[2]]
        if sum(has_var) == 0 and True in has_c:
            return False, True, [(eq, c)]
        else:
            args = []
            if isinstance(eq, (Add, Mul, Pow)):
                has_free_c = False
                if True in [has_c[i] and not has_var[i] for i in range(len(has_c))]:
                    has_free_c = True

                for i in range(len(has_var)):
                    if has_var[i] or (not has_free_c and not has_c[i]):
                        args += [eq.args[i].subs(subs[i])]
                if has_free_c:
                    args += [c]

            else:
                for i in range(len(has_var)):
                    args += [eq.args[i].subs(subs[i])]
            return True in has_var, True in has_c, [(eq, eq.func(*args))]


def _enumerate_constants(eq: Basic, c: Symbol) -> Basic:
    """Replace each occurrence of ``c`` in ``eq`` with a new symbol, so folding treats them apart."""
    counter = itertools.count()

    def rec(e: Basic) -> Basic:
        if e == c:
            return Symbol(f"_{next(counter)}_")
        return e.func(*[rec(a) for a in e.args]) if e.args else e

    return rec(eq)
