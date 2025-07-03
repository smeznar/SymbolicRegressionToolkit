from typing import Union, List

import numpy as np
from sympy import sympify, expand, Expr, Basic
from sympy.core import Mul, Add, Pow, Symbol
from sympy import symbols as sp_symbols
import re

from SRToolkit.utils.symbol_library import SymbolLibrary
from SRToolkit.utils.expression_tree import Node
from SRToolkit.utils.expression_tree import is_float

def simplify(expr: Union[List[str], Node], symbol_library: SymbolLibrary=SymbolLibrary.default_symbols()) -> Union[List[str], Node]:
    """
    Simplifies a mathematical expression by:
        1. making use of sympy's simplification functions
        2. simplifying constants, e.g. C*C + C -> C

    Examples:
        >>> expr = ["C", "+", "C" "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
        >>> print("".join(simplify(expr)))
        C+X_1
    
    Args:
        expr: The expression given as a list of tokens in the infix notation or as an instance of SRToolkit.utils.expression_tree.Node
        symbol_library: The symbol library to use. Defaults to SymbolLibrary.default_symbols().

    Raises:
        Exception: If problems occur during simplification or if the expression contains invalid symbols.

    Returns:
        The simplified expression
    """
    is_tree = False
    if isinstance(expr, Node):
        expr = expr.to_list(notation="infix", symbol_library=symbol_library)
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
        raise Exception("Simplified expression contains invalid symbols. Possibly skip its simplification or add symbols to the SymbolLibrary.")

    if is_tree:
        return expr
    else:
        return expr.to_list("infix", symbol_library=symbol_library)


def _check_tree(expr: Node, symbol_library: SymbolLibrary) -> bool:
    """
    Checks if the expression tree contains only valid symbols from the symbol library.

    Args:
        expr: The expression tree to check.
        symbol_library: The symbol library to use.

    Returns:
        True if the expression tree contains only valid symbols from the symbol library, False otherwise.
    """
    if expr.symbol not in symbol_library.symbols and not is_float(expr.symbol):
        return False
    if isinstance(expr.left, Node) and not _check_tree(expr.left, symbol_library):
        return False
    if isinstance(expr.right, Node) and not _check_tree(expr.right, symbol_library):
        return False

    return True


def _sympy_to_number(expr):
    """
    Extracts the number contained in the Sympy node 
    """
    evaluated = float(expr.evalf())
    return int(evaluated) if evaluated.is_integer() else evaluated


def _sympy_to_sr(expr: Union[Expr, Basic]) -> Node:
    """
    Converts a Sympy expression into an instance of SRtoolkit.expression_tree.Node, explicitly handling left-associative division.

    Args:
        expr: The Sympy expression.

    Returns:
        An instance of SRtoolkit.expression_tree.Node that corresponds to the expression given as the input.
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
            return Node('-', _sympy_to_sr(-args[1]), _sympy_to_sr(args[0]))
        # Handle regular addition
        root = Node('+', _sympy_to_sr(args[1]), _sympy_to_sr(args[0]))
        for term in args[2:]:
            root = Node('+', _sympy_to_sr(term), root)
        return root

    if isinstance(expr, Mul):
        # Process factors explicitly, ensuring left-to-right associativity
        factors = list(expr.args)
        root = _sympy_to_sr(factors[0])  # Start with the first factor
        for factor in factors[1:]:
            if factor.is_Pow and factor.args[1] == -1:  # Division
                divisor = _sympy_to_sr(factor.args[0])
                root = Node('/', divisor, root)  # Left-to-right division
            else:  # Multiplication
                multiplicand = _sympy_to_sr(factor)
                root = Node('*', multiplicand, root)
        return root

    if isinstance(expr, Pow):
        base, exp = expr.args
        return Node('^', _sympy_to_sr(exp), _sympy_to_sr(base))

    if expr.is_Rational and expr.q != 1:
        return Node('/', _sympy_to_sr(expr.q), _sympy_to_sr(expr.p))

    raise ValueError(f"{expr}")


def _simplify_constants(eq, c, var):
    """ Simplifies the constants in a Sympy expression. output[2][0][1] is the simplified expression.

    Args:
        eq: The Sympy expression.
        c: The constant symbol.
        var: List of symbols representing variables.
    
    Returns:
        - bool: True if the expression contains a variable.
        - bool: True if the expression contains the constant.
        - list: List of tuples containing the original and simplified expressions
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
            a_rec = _simplify_constants (a, c, var)
            has_var += [a_rec[0]]; has_c += [a_rec[1]]; subs += [a_rec[2]]
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
    """ Enumerates the constants in a Sympy expression. 

    Example: C*x**2 + C*x + C -> C0*x**2 + C1*x + C2

    Input:
        expr - Sympy expression
        constant - constant symbol

    Returns:
        Sympy expression with enumerated constants
        list of enumerated constants"""
        
    char_list = np.array(list(str(expr)), dtype='<U16')
    constind = np.where(char_list == constant)[0]
    """ Rename all constants: c -> cn, where n is the index of the associated term"""
    constants = [constant+str(i) for i in range(len(constind))]
    char_list[constind] = constants
    return sympify("".join(char_list)), tuple(constants)


def _denumerate_constants(expr, constant):
    """ Removes the enumeration of constants in a Sympy expression.

    Args:
        expr: Sympy expression
        constant: constant symbol

    Returns:
        Sympy expression with denumerated constants
    """
    return re.sub(f'{constant}\\d', constant, expr)


def _simplify_expression (expr_str, constant, variables):
    """Simplifies a mathematical expression.

    Args:
        expr_str: String representing the expression.
        constant: The character representing numerical constants.
        variables: List of characters representing variables.

    Returns:
        expr: Sympy expression object in canonical form.
        symbols_params: Tuple of enumerated constants.
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
    expr = ["C", "+", "C" "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"]
    print(simplify(expr))

    # Should raise an exception
    expr = ["X_0", "*", "X_0", "^2"]
    sl = SymbolLibrary.from_symbol_list(["+", "*", "-"], 1)
    print(simplify(expr, symbol_library=sl))