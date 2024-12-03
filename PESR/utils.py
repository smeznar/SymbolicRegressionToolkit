from symbol_library import SymbolLibrary

class Node:
    def __init__(self, symbol=None, right=None, left=None):
        self.symbol = symbol
        self.right = right
        self.left = left

    def __len__(self):
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)

    # def to_postfix(self) -> list[str]:
    #     if self.left is None and self.right is None:
    #         return [self.symbol]
    #     elif self.right is None:
    #         return self.left.to_postfix() + [self.symbol]
    #     else:
    #         return self.left.to_postfix() + self.right.to_postfix() + [self.symbol]
    #
    # def to_dict(self) -> dict:
    #     d = {'s': self.symbol}
    #     if self.left is not None:
    #         d['l'] = self.left.to_dict()
    #     if self.right is not None:
    #         d['r'] = self.right.to_dict()
    #     return d
    #
    # @staticmethod
    # def from_dict(d):
    #     left = None
    #     right = None
    #     if "l" in d:
    #         left = Node.from_dict(d["l"])
    #     if 'r' in d:
    #         right = Node.from_dict(d["r"])
    #     return Node(d["s"], right=right, left=left)
    #
    # def __len__(self):
    #     return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)



def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens: list[str], sl: SymbolLibrary) -> Node:
    """
    tokens : list of string tokens
    symbols: dictionary of possible tokens -> attributes, each token must have attributes: nargs (0-2), order
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