from typing import List

from .symbol_library import SymbolLibrary


class Node:
    def __init__(self, symbol: str=None, right: "Node"=None, left: "Node"=None):
        """
        Initializes a Node object.

        Parameters
        ----------
        symbol : str
            The symbol string stored in this node.
        right : Node
            The right child of this node.
        left : Node
            The left child of this node.
        """
        self.symbol = symbol
        self.right = right
        self.left = left

    def __len__(self) -> int:
        """
        Returns the number of nodes in the tree rooted at this node.
        """
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)


def is_float(element: any) -> bool:
    """
    Checks if a given element is a float.

    Parameters
    ----------
    element : any
        The element to check.

    Returns
    -------
    bool
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

    Parameters
    ----------
    tokens : list[str]
        The list of tokens to convert.
    sl : SymbolLibrary
        The symbol library to use when parsing the tokens.

    Returns
    -------
    Node
        The root of the tree data structure.
    """
    num_tokens = len([t for t in tokens if t != "(" and t != ")"])
    expr_str = ''.join(tokens)
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
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and sl.get_precedence(operator_stack[-1]) > sl.get_precedence(token):
                if sl.get_type(operator_stack[-1]) == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
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