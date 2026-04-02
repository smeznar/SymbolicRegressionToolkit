"""
Algebraic simplification of symbolic expressions using SymPy.
The functions in this script are work in progress and will be improved in the future for better accuracy and performance.
"""

import re
from typing import List, Union

import numpy as np
from sympy import Basic, Expr, expand, sympify
from sympy import symbols as sp_symbols
from sympy.core import Add, Mul, Pow

from SRToolkit.utils.expression_tree import Node, is_float
from SRToolkit.utils.symbol_library import SymbolLibrary


def simplify(
    expr: Union[List[str], Node],
    symbol_library: SymbolLibrary = SymbolLibrary.default_symbols(),
) -> Union[List[str], Node]:
    """
    Simplify an expression algebraically.

    Two successive steps are applied:

    1. **SymPy simplification** — expands and reduces the expression algebraically
       (e.g. ``X_0 * X_1 / X_0`` → ``X_1``).
    2. **Constant folding** — collapses any sub-expression containing no variables
       into a single free constant ``C`` (e.g. ``C * C + C`` → ``C``).

    Examples:
        >>> expr = ["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
        >>> print("".join(simplify(expr)))
        C+X_1

    Args:
        expr: Expression as a token list in infix notation or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library defining variables and constants.
            Defaults to [SymbolLibrary.default_symbols][SRToolkit.utils.symbol_library.SymbolLibrary.default_symbols].

    Returns:
        The simplified expression in the same form as the input (list if a list was
        given, [Node][SRToolkit.utils.expression_tree.Node] if a tree was given).

    Raises:
        Exception: If simplification fails or the result contains tokens absent from
            ``symbol_library``.
    """
    is_tree = False
    if isinstance(expr, Node):
        expr = expr.to_list(symbol_library=symbol_library, notation="infix")
        is_tree = True

    variables = symbol_library.get_symbols_of_type("var")

    # We expect only one symbol for constants
    if len(symbol_library.get_symbols_of_type("const")) > 0:
        constant = symbol_library.get_symbols_of_type("const")[0]
    else:
        # In this case constants shouldn't be problematic as they are not in the SymbolLibrary
        # Just in case and to not change other functions, I changed it to __C__.
        constant = "__C__"

    expr = _simplify_expression("".join(expr), constant, variables)
    expr = sympify(_denumerate_constants(str(expr), constant), evaluate=False)
    expr = _sympy_to_sr(expr)
    if not _check_tree(expr, symbol_library):
        raise Exception(
            "Simplified expression contains invalid symbols. Possibly skip its simplification or add symbols to the SymbolLibrary."
        )

    if is_tree:
        return expr
    else:
        return expr.to_list(symbol_library=symbol_library, notation="infix")


def _check_tree(expr: Node, symbol_library: SymbolLibrary) -> bool:
    """
    Return ``True`` if every token in the tree is present in ``symbol_library`` or is numeric.

    Args:
        expr: Expression tree to validate.
        symbol_library: Symbol library to check against.

    Returns:
        ``True`` if all nodes are valid, ``False`` if any token is absent and non-numeric.
    """
    if expr.symbol not in symbol_library.symbols and not is_float(expr.symbol):
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
    """
    Convert a SymPy expression into a [Node][SRToolkit.utils.expression_tree.Node] tree.

    Handles left-associative division explicitly to match the SRToolkit tree
    convention when SymPy reorders multiplication factors.

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

    if expr.is_Rational and expr.q != 1:
        return Node("/", _sympy_to_sr(expr.q), _sympy_to_sr(expr.p))

    raise ValueError(f"{expr}")


def _simplify_constants(eq, c, var):
    """
    Recursively fold constant-only sub-expressions in a SymPy expression.

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
                        if len(subs[i]) > 0:
                            args += [eq.args[i].subs(subs[i])]
                        else:
                            args += [eq.args[i]]
                if has_free_c:
                    args += [c]

            else:
                for i in range(len(has_var)):
                    if len(subs[i]) > 0:
                        args += [eq.args[i].subs(subs[i])]
                    else:
                        args += [eq.args[i]]
            return True in has_var, True in has_c, [(eq, eq.func(*args))]


def _enumerate_constants(expr, constant):
    """
    Replace each occurrence of the constant token with a uniquely numbered version.

    Example: ``C*x**2 + C*x + C`` → ``C0*x**2 + C1*x + C2``

    Args:
        expr: SymPy expression containing unnumbered constants.
        constant: Character string used as the constant token (e.g. ``"C"``).

    Returns:
        A 2-tuple ``(numbered_expr, constants)`` where ``numbered_expr`` is the SymPy
        expression with enumerated constants and ``constants`` is a tuple of the
        generated constant name strings.
    """

    char_list = np.array(list(str(expr)), dtype="<U16")
    constind = np.where(char_list == constant)[0]
    """ Rename all constants: c -> cn, where n is the index of the associated term"""
    constants = [constant + str(i) for i in range(len(constind))]
    char_list[constind] = constants
    return sympify("".join(char_list)), tuple(constants)


def _denumerate_constants(expr, constant):
    """
    Remove the numeric suffix from all constant tokens in a string expression.

    Inverse of ``_enumerate_constants``: ``"C0*x + C1"`` → ``"C*x + C"``.

    Args:
        expr: String representation of the expression with enumerated constants.
        constant: Base constant token string (e.g. ``"C"``).

    Returns:
        Expression string with all ``C<n>`` occurrences replaced by ``C``.
    """
    return re.sub(f"{constant}\\d", constant, expr)


def _simplify_expression(expr_str, constant, variables):
    """
    Apply full algebraic simplification to an expression string.

    Performs two rounds of constant folding around a SymPy ``expand`` call to produce
    a canonical form.

    Args:
        expr_str: Infix expression as a concatenated token string (no spaces).
        constant: Token string representing the free constant (e.g. ``"C"``).
        variables: Token strings representing the input variables.

    Returns:
        Simplified SymPy expression object.
    """
    x = [sp_symbols(s.strip("'")) for s in variables]
    c = sp_symbols(constant)

    expr, _ = _enumerate_constants(expr_str, constant)
    expr = _simplify_constants(expr, c, x)[2][0][1]
    expr, _ = _enumerate_constants(expr, constant)
    expr = expand(expr)
    expr = _simplify_constants(expr, c, x)[2][0][1]
    # expr, _ = _enumerate_constants(expr, constant)
    # expr = _simplify_constants(expr, c, x)[2][0][1]
    return expr


if __name__ == "__main__":
    # Should simplify
    expr = ["C", "+", "C*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
    print(simplify(expr))

    # Should raise an exception
    expr = ["X_0", "*", "X_0", "^2"]
    sl = SymbolLibrary.from_symbol_list(["+", "*", "-"], 1)
    print(simplify(expr, symbol_library=sl))
