"""
The module containing the expression tree data structure and functions for transforming expressions into trees and back.
"""
from typing import List
import warnings
from copy import copy

from SRToolkit.utils.symbol_library import SymbolLibrary


class Node:
    def __init__(self, symbol: str = None, right: "Node" = None, left: "Node" = None):
        """
        Initializes a Node object. We assume that nodes containing functions have only one child node, i.e. right is None.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> len(node)
            3

        Args:
            symbol: The symbol string stored in this node.
            right: The right child of this node.
            left: The left child of this node.

        Methods:
            __len__(self):
                Returns the number of nodes in the tree rooted at this node.
            __str__(self):
                Returns a string representation of the tree rooted at this node.
            to_list(self, notation: str = "infix", symbol_library: SymbolLibrary = None):
                Returns a list representation of the tree rooted at this node.

        """
        self.symbol = symbol
        self.right = right
        self.left = left

    def __len__(self) -> int:
        """
        Returns the number of nodes in the tree rooted at this node.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> len(node)
            3

        Returns:
            The number of nodes in the tree rooted at this node.
        """
        return (
            1
            + (len(self.left) if self.left is not None else 0)
            + (len(self.right) if self.right is not None else 0)
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the tree rooted at this node.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> str(node)
            'x+1'

        Returns:
            A string representation of the tree rooted at this node.
        """
        return "".join(self.to_list())

    def to_list(self, symbol_library: SymbolLibrary = None, notation: str = "infix") -> List[str]:
        """
        Transforms the tree rooted at this node into a list of tokens.

        Examples:
            >>> node = Node("+", Node("X_0"), Node("1"))
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['1', '+', 'X_0']
            >>> node.to_list(notation="postfix")
            ['1', 'X_0', '+']
            >>> node.to_list(notation="prefix")
            ['+', '1', 'X_0']
            >>> node = Node("+", Node("*", Node("X_0"), Node("X_1")), Node("1"))
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['1', '+', 'X_1', '*', 'X_0']
            >>> node.to_list(notation="infix")
            ['1', '+', '(', 'X_1', '*', 'X_0', ')']
            >>> node = Node("sin", None, Node("X_0"))
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['sin', '(', 'X_0', ')']
            >>> node = Node("^2", None, Node("X_0"))
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['X_0', '^2']
            >>> node.to_list()
            ['(', 'X_0', ')', '^2']
            >>> node = Node("*", Node("*", Node("X_0"), Node("X_0")),  Node("X_0"))
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols(),notation="infix")
            ['X_0', '*', '(', 'X_0', '*', 'X_0', ')']

        Args:
            notation: The notation to use for the resulting list of tokens. One of "prefix", "postfix", or "infix".
            symbol_library: The symbol library to use when converting the tree. This library defines the properties of the symbols in the tree.

        Returns:
            A list of tokens representing the tree rooted at this node in the specified notation.

        Raises:
             Exception: If the notation is not one of "prefix", "postfix", or "infix" or if a symbol is not in the symbol library.

        Notes:
            If the notation is "infix" and the symbol library is not provided, then the resulting list of tokens may contain unnecessary parentheses or have other issues.
        """
        left = [] if self.left is None else self.left.to_list(symbol_library, notation)
        right = [] if self.right is None else self.right.to_list(symbol_library, notation)

        if notation == "prefix":
            return [self.symbol] + left + right

        elif notation == "postfix":
            return left + right + [self.symbol]

        elif notation == "infix" and symbol_library is None:
            warnings.warn("Symbol library not provided. Generated expression may contain unnecessary parentheses and"
                          " have other issues.")
            if self.left is None and self.right is None:
                return [self.symbol]
            if self.right is None and self.left is not None:
                if self.symbol[0] == "^":
                    return ["("] + left + [")", self.symbol]
                else:
                    return [self.symbol, "("] + left + [")"]
            else:
                if len(left) > 1:
                    left = ["("] + left + [")"]
                if len(right) > 1:
                    right = ["("] + right + [")"]
                return left + [self.symbol] + right

        elif notation == "infix":
            if is_float(self.symbol):
                return [self.symbol]
            if symbol_library.get_type(self.symbol) in ["var", "const", "lit"]:
                return [self.symbol]
            elif symbol_library.get_type(self.symbol) == "fn":
                if symbol_library.get_precedence(self.symbol) > 0:
                    return [self.symbol, "("] + left + [")"]
                else:
                    if len(left) > 1:
                        left = ["("] + left + [")"]
                    return left + [self.symbol]
            elif symbol_library.get_type(self.symbol) == "op":
                if not is_float(self.left.symbol) and -1 < symbol_library.get_precedence(self.left.symbol) <= symbol_library.get_precedence(self.symbol):
                    left = ["("] + left + [")"]
                if not is_float(self.right.symbol) and -1 < symbol_library.get_precedence(self.right.symbol) <= symbol_library.get_precedence(self.symbol):
                    right = ["("] + right + [")"]
                return left + [self.symbol] + right
            else:
                raise Exception(f"Invalid symbol type for symbol {self.symbol}.")
        else:
            raise Exception("Invalid notation selected. Use 'infix', 'prefix', 'postfix', or leave blank (defaults to 'infix').")

    def to_latex(self, symbol_library: SymbolLibrary) -> str:
        r"""
        Transforms the tree rooted at this node into a LaTeX expression.

        Examples:
            >>> node = Node("+", Node("X_0"), Node("1"))
            >>> node.to_latex(symbol_library=SymbolLibrary.default_symbols())
            '$1 + X_{0}$'
            >>> node = Node("+", Node("*", Node("X_0"), Node("X_1")), Node("1"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $1 + X_{1} \cdot X_{0}$
            >>> node = Node("sin", None, Node("X_0"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $\sin X_{0}$
            >>> node = Node("+", Node("*", Node("X_0"), Node("C")), Node("C"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $C_{0} + C_{1} \cdot X_{0}$

        Args:
            symbol_library: The symbol library to use when converting the tree. This library defines the properties of the symbols in the tree.

        Returns:
            A latex string representing the tree rooted at this node.

        Raises:
             Exception: If the notation is not one of "prefix", "postfix", or "infix" or if a symbol is not in the symbol library.
        """
        assert symbol_library is not None, "[Node.to_latex] parameter symbol_library should be of type SymbolLibrary"
        return f"${self.__to_latex_rec(symbol_library)[0]}$"


    def __to_latex_rec(self, symbol_library, num_const=0) -> (str, int):
        left, num_const = ("", num_const) if self.left is None else self.left.__to_latex_rec(symbol_library, num_const)
        right, num_const = ("", num_const) if self.right is None else self.right.__to_latex_rec(symbol_library, num_const)

        if is_float(self.symbol):
            return str(self.symbol), num_const
        elif symbol_library.get_type(self.symbol) == "const":
            return symbol_library.get_latex_str(self.symbol).format(num_const), num_const + 1
        elif symbol_library.get_type(self.symbol) in ["var", "lit"]:
                return symbol_library.get_latex_str(self.symbol), num_const
        elif symbol_library.get_type(self.symbol) == "fn":
            if symbol_library.get_type(self.left.symbol) in ["fn", "op"]:
                left = f"({left})"
            return symbol_library.get_latex_str(self.symbol).format(left), num_const
        elif symbol_library.get_type(self.symbol) == "op":
            if not is_float(self.left.symbol) and -1 < symbol_library.get_precedence(self.left.symbol) < symbol_library.get_precedence(self.symbol):
                left = f"({left})"
            if not is_float(self.right.symbol) and -1 < symbol_library.get_precedence(self.right.symbol) < symbol_library.get_precedence(self.symbol):
                right = f"({right})"
            return symbol_library.get_latex_str(self.symbol).format(left, right), num_const
        else:
            raise Exception(f"Invalid symbol type for symbol {self.symbol}.")

    def __copy__(self):
        """
        Creates a copy of the expression (usefull for manipulating expressions).

        Examples:
            >>> node = Node("+", Node("X_0"), Node("1"))
            >>> new_node = copy(node)
            >>> node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['1', '+', 'X_0']
            >>> new_node.to_list(symbol_library=SymbolLibrary.default_symbols())
            ['1', '+', 'X_0']
            >>> node == node
            True
            >>> node == new_node
            False

        Returns:
            A copy of the expression (tree).
        """
        if self.left is not None:
            left = copy(self.left)
        else:
            left = None
        if self.right is not None:
            right = copy(self.right)
        else:
            right = None
        return Node(copy(self.symbol), left=left, right=right)

def is_float(element: any) -> bool:
    """
    Checks if a given element is a float.

    Examples:
        >>> is_float(1.0)
        True
        >>> is_float("1.0")
        True
        >>> is_float("1")
        True
        >>> is_float(None)
        False


    Args:
        element: The element to check.

    Returns:
        True if the element is a float, False otherwise.
    """
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens: List[str], sl: SymbolLibrary) -> Node:
    """
    Converts a list of tokens to a tree data structure. Throws an exception if the expression is invalid (check syntax
    and that all symbols are in the symbol library correctly defined).

    Examples:
        >>> tree = tokens_to_tree(["(", "x", "+", "y", ")"], SymbolLibrary.default_symbols())
        >>> len(tree)
        3

    Args:
        tokens: The list of tokens to convert.
        sl: The symbol library to use when parsing the tokens.

    Returns:
        The root of the expression tree data structure.

    Raises:
        Exception: If the expression is invalid. Usually this means that a symbol is not in the symbol library or that
                   there is a syntactic error in the expression.
    """
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    expr_str = "".join(tokens)
    tokens = ["("] + tokens + [")"]
    operator_stack = []
    out_stack = []
    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif sl.get_type(token) in ["var", "const", "lit"] or is_float(token):
            out_stack.append(Node(token))
        elif sl.get_type(token) == "fn":
            if token[0] == "^":
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif sl.get_type(token) == "op":
            while (
                len(operator_stack) > 0
                and operator_stack[-1] != "("
                and sl.get_precedence(operator_stack[-1]) > sl.get_precedence(token)
            ):
                if sl.get_type(operator_stack[-1]) == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(
                        Node(operator_stack.pop(), out_stack.pop(), out_stack.pop())
                    )
            operator_stack.append(token)
        else:
            if token != ")":
                raise Exception(f"Invalid symbol \"{token}\" in expression {expr_str}. Did you add token \"{token}\" to the symbol library?")

            while len(operator_stack) > 0 and operator_stack[-1] != "(":
                if sl.get_type(operator_stack[-1]) == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(
                        Node(operator_stack.pop(), out_stack.pop(), out_stack.pop())
                    )
            operator_stack.pop()
            if len(operator_stack) > 0 and sl.get_type(operator_stack[-1]) == "fn":
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")


if __name__ == '__main__':
    tree = tokens_to_tree(["(", "X_0", "+", "tan", "(", "X_1", "-", "5.2", ")", ")"], SymbolLibrary.default_symbols(num_variables=2))