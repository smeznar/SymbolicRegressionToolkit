"""
Binary expression tree ([Node][SRToolkit.utils.expression_tree.Node]) and conversion utilities between token
lists, trees, and LaTeX strings.
"""

import warnings
from copy import copy
from typing import Any, List, Optional, Tuple, Union

from SRToolkit.utils.symbol_library import SymbolLibrary


class Node:
    def __init__(self, symbol: str, right: Optional["Node"] = None, left: Optional["Node"] = None) -> None:
        """
        A node in a binary expression tree.

        - Binary operators (``"op"``) set both ``left`` and ``right``.
        - Unary functions (``"fn"``) set only ``left``; ``right`` is ``None``.
        - Leaves (variables, constants, literals, numeric values) have both children as ``None``.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> len(node)
            3

        Args:
            symbol: Token string stored at this node.
            right: Right operand (binary operators only).
            left: Left operand (operators and unary functions).
        """
        self.symbol = symbol
        self.right = right
        self.left = left

    def to_list(self, symbol_library: Optional[SymbolLibrary] = None, notation: str = "infix") -> List[str]:
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
            symbol_library: Symbol library used to determine token types and precedences
                during infix reconstruction. If ``None`` with ``"infix"`` notation, the
                output may contain redundant parentheses.
            notation: Output notation: ``"infix"``, ``"prefix"``, or ``"postfix"``.
                Default ``"infix"``.

        Returns:
            Token list representing the subtree rooted at this node.

        Raises:
            Exception: If ``notation`` is not one of the accepted values, or if a token's
                type cannot be resolved during infix reconstruction.
        """
        # if symbol_library is None:
        #     symbol_library = SymbolLibrary.default_symbols()

        left = [] if self.left is None else self.left.to_list(symbol_library, notation)
        right = [] if self.right is None else self.right.to_list(symbol_library, notation)

        if notation == "prefix":
            return [self.symbol] + left + right

        elif notation == "postfix":
            return left + right + [self.symbol]

        elif notation == "infix" and symbol_library is None:
            warnings.warn(
                "Symbol library not provided. Generated expression may contain unnecessary parentheses and"
                " have other issues."
            )
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
            assert symbol_library is not None, "[Node.to_list] parameter symbol_library should be of type SymbolLibrary"
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
                if (
                    self.left is not None
                    and not is_float(self.left.symbol)
                    and -1
                    < symbol_library.get_precedence(self.left.symbol)
                    <= symbol_library.get_precedence(self.symbol)
                ):
                    left = ["("] + left + [")"]
                if (
                    self.right is not None
                    and not is_float(self.right.symbol)
                    and -1
                    < symbol_library.get_precedence(self.right.symbol)
                    <= symbol_library.get_precedence(self.symbol)
                ):
                    right = ["("] + right + [")"]
                return left + [self.symbol] + right
            else:
                raise Exception(f"Invalid symbol type for symbol {self.symbol}.")
        else:
            raise Exception(
                "Invalid notation selected. Use 'infix', 'prefix', 'postfix', or leave blank (defaults to 'infix')."
            )

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
            symbol_library: Symbol library providing the LaTeX template for each token.

        Returns:
            A LaTeX string of the form ``$...$``.

        Raises:
            Exception: If the tree contains a token whose type cannot be resolved in
                ``symbol_library``.
        """
        assert symbol_library is not None, "[Node.to_latex] parameter symbol_library should be of type SymbolLibrary"
        return f"${self.__to_latex_rec(symbol_library)[0]}$"

    def __to_latex_rec(self, symbol_library, num_const=0) -> Tuple[str, int]:
        left, num_const = ("", num_const) if self.left is None else self.left.__to_latex_rec(symbol_library, num_const)
        right, num_const = (
            ("", num_const) if self.right is None else self.right.__to_latex_rec(symbol_library, num_const)
        )

        if is_float(self.symbol):
            return str(self.symbol), num_const
        elif symbol_library.get_type(self.symbol) == "const":
            return symbol_library.get_latex_str(self.symbol).format(num_const), num_const + 1
        elif symbol_library.get_type(self.symbol) in ["var", "lit"]:
            return symbol_library.get_latex_str(self.symbol), num_const
        elif symbol_library.get_type(self.symbol) == "fn":
            if self.left is not None and symbol_library.get_type(self.left.symbol) in ["fn", "op"]:
                left = f"({left})"
            return symbol_library.get_latex_str(self.symbol).format(left), num_const
        elif symbol_library.get_type(self.symbol) == "op":
            if (
                self.left is not None
                and not is_float(self.left.symbol)
                and -1 < symbol_library.get_precedence(self.left.symbol) < symbol_library.get_precedence(self.symbol)
            ):
                left = f"({left})"
            if (
                self.right is not None
                and not is_float(self.right.symbol)
                and -1 < symbol_library.get_precedence(self.right.symbol) < symbol_library.get_precedence(self.symbol)
            ):
                right = f"({right})"
            return symbol_library.get_latex_str(self.symbol).format(left, right), num_const
        else:
            raise Exception(f"Invalid symbol type for symbol {self.symbol}.")

    def height(self) -> int:
        """
        Return the height of the subtree rooted at this node.

        A single-node tree has height 1.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> node.height()
            2

        Returns:
            Height of the subtree.
        """
        return 1 + max(
            (self.left.height() if self.left is not None else 0),
            (self.right.height() if self.right is not None else 0),
        )

    def __len__(self) -> int:
        """
        Return the number of nodes in the subtree rooted at this node.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> len(node)
            3

        Returns:
            Total node count of the subtree.
        """
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)

    def __str__(self) -> str:
        """
        Return the expression as a concatenated string using default infix notation that may contain redundant parentheses.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> str(node)
            '1+x'

        Returns:
            Concatenated token string with no spaces.
        """
        return "".join(self.to_list())

    def __copy__(self) -> "Node":
        """
        Return a deep copy of the subtree rooted at this node.

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
            An independent copy of the subtree.
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


def is_float(element: Any) -> bool:
    """
    Return ``True`` if ``element`` can be interpreted as a floating-point number.

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
        element: Value to test.

    Returns:
        ``True`` if ``float(element)`` succeeds, ``False`` otherwise (including ``None``).
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
    Parse a token list into an expression tree using the shunting-yard algorithm.

    Examples:
        >>> tree = tokens_to_tree(["(", "X_0", "+", "X_1", ")"], SymbolLibrary.default_symbols())
        >>> len(tree)
        3

    Args:
        tokens: Token list in infix notation.
        sl: Symbol library used to resolve token types and precedences.

    Returns:
        Root [Node][SRToolkit.utils.expression_tree.Node] of the parsed expression tree.

    Raises:
        Exception: If a token is absent from ``sl``, or if the expression is
            syntactically invalid.
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
                and sl.get_precedence(operator_stack[-1]) >= sl.get_precedence(token)
            ):
                if sl.get_type(operator_stack[-1]) == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            if token != ")":
                raise Exception(
                    f'Invalid symbol "{token}" in expression {expr_str}. Did you add token "{token}" to the symbol library?'
                )

            while len(operator_stack) > 0 and operator_stack[-1] != "(":
                if sl.get_type(operator_stack[-1]) == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and sl.get_type(operator_stack[-1]) == "fn":
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")


def expr_to_latex(expr: Union[Node, List[str]], symbol_library: SymbolLibrary) -> str:
    """
    Convert an expression to a LaTeX string.

    Examples:
        >>> expr_to_latex(["(", "X_0", "+", "X_1", ")"], SymbolLibrary.default_symbols())
        '$X_{0} + X_{1}$'
        >>> expr = Node("+", Node("X_0"), Node("1"))
        >>> expr_to_latex(expr, SymbolLibrary.default_symbols())
        '$1 + X_{0}$'

    Args:
        expr: Expression as a token list or a [Node][SRToolkit.utils.expression_tree.Node] tree.
        symbol_library: Symbol library providing LaTeX templates.

    Returns:
        A LaTeX string of the form ``$...$``, or an empty string if conversion fails.
    """
    try:
        if isinstance(expr, Node):
            return expr.to_latex(symbol_library)
        elif isinstance(expr, list):
            return tokens_to_tree(expr, symbol_library).to_latex(symbol_library)
        else:
            raise Exception(
                f"Invalid type for expression {str(expr)}. Should be SRToolkit.utils.Node or a list of tokens."
            )
    except Exception as e:
        print(f"Error while converting expression {str(expr)} to LaTeX: {str(e)}")
        return ""


if __name__ == "__main__":
    tree = tokens_to_tree(
        ["(", "X_0", "+", "tan", "(", "X_1", "-", "5.2", ")", ")"],
        SymbolLibrary.default_symbols(num_variables=2),
    )
