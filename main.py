import numpy as np
from scipy.optimize import minimize

SYMBOLS = {
    "+": {"symbol": '+', "type": "op", "precedence": 0, "np_fn": "{} = {} + {}"},
    "-": {"symbol": '-', "type": "op", "precedence": 0, "np_fn": "{} = {} - {}"},
    "*": {"symbol": '*', "type": "op", "precedence": 1, "np_fn": "{} = {} * {}"},
    "/": {"symbol": '/', "type": "op", "precedence": 1, "np_fn": "{} = {} / {}"},
    "^": {"symbol": "^", "type": "op", "precedence": 2, "np_fn": "{} = np.pow({},{})"},
    "u-": {"symbol": "u-", "type": "fn", "precedence": 5, "np_fn": "{} = -{}"},
    "sqrt": {"symbol": 'sqrt', "type": "fn", "precedence": 5, "np_fn": "{} = np.sqrt({})"},
    "sin": {"symbol": 'sin', "type": "fn", "precedence": 5, "np_fn": "{} = np.sin({})"},
    "cos": {"symbol": 'cos', "type": "fn", "precedence": 5, "np_fn": "{} = np.cos({})"},
    "exp": {"symbol": 'exp', "type": "fn", "precedence": 5, "np_fn": "{} = np.exp({})"},
    "log": {"symbol": 'log', "type": "fn", "precedence": 5, "np_fn": "{} = np.log({})"},
    "^-1": {"symbol": "^-1", "type": "fn", "precedence": -1, "np_fn": "{} = 1/{}"},
    "^2": {"symbol": '^2', "type": "fn", "precedence": -1, "np_fn": "{} = {}**2"},
    "^3": {"symbol": '^3', "type": "fn", "precedence": -1, "np_fn": "{} = {}**3"},
    "^4": {"symbol": '^4', "type": "fn", "precedence": -1, "np_fn": "{} = {}**4"},
    "^5": {"symbol": '^5', "type": "fn", "precedence": -1, "np_fn": "{} = {}**5"},
    "1": {"symbol": '1', "type": "lit", "precedence": 5, "np_fn": "1"},
    "pi": {"symbol": 'pi', "type": "lit", "precedence": 5, "np_fn": "np.pi"},
    "e": {"symbol": 'e', "type": "lit", "precedence": 5, "np_fn": "np.e"},
    "C": {"symbol": 'C', "type": "const", "precedence": 5, "np_fn": "C[{}]"},
}
for i, char in enumerate('ABDEFGHIJKLMNOPQRSTUVWXYZČŠŽ'):
    SYMBOLS[char] = {"symbol": char, "type": "var", "precedence": 5, "np_fn": "X[:, {}]".format(i)}


class Node:
    def __init__(self, symbol=None, right=None, left=None):
        self.symbol = symbol
        self.right = right
        self.left = left

    def to_postfix(self) -> list[str]:
        if self.left is None and self.right is None:
            return [self.symbol]
        elif self.right is None:
            return self.left.to_postfix() + [self.symbol]
        else:
            return self.left.to_postfix() + self.right.to_postfix() + [self.symbol]

    def to_dict(self) -> dict:
        d = {'s': self.symbol}
        if self.left is not None:
            d['l'] = self.left.to_dict()
        if self.right is not None:
            d['r'] = self.right.to_dict()
        return d

    @staticmethod
    def from_dict(d):
        left = None
        right = None
        if "l" in d:
            left = Node.from_dict(d["l"])
        if 'r' in d:
            right = Node.from_dict(d["r"])
        return Node(d["s"], right=right, left=left)

    def __len__(self):
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)


def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def tokens_to_tree(tokens: list[str]) -> Node:
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
        elif token in SYMBOLS and SYMBOLS[token]["type"] in ["var", "const", "lit"] or is_float(token):
            out_stack.append(Node(token))
        elif token in SYMBOLS and SYMBOLS[token]["type"] == "fn":
            if token[0] == "^":
                out_stack.append(Node(token, left=out_stack.pop()))
            else:
                operator_stack.append(token)
        elif token in SYMBOLS and SYMBOLS[token]["type"] == "op":
            while len(operator_stack) > 0 and operator_stack[-1] != '(' \
                    and SYMBOLS[operator_stack[-1]]["precedence"] > SYMBOLS[token]["precedence"]:
                if SYMBOLS[operator_stack[-1]]["type"] == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.append(token)
        else:
            while len(operator_stack) > 0 and operator_stack[-1] != '(':
                if SYMBOLS[operator_stack[-1]]["type"] == "fn":
                    out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
                else:
                    out_stack.append(Node(operator_stack.pop(), out_stack.pop(), out_stack.pop()))
            operator_stack.pop()
            if len(operator_stack) > 0 and operator_stack[-1] in SYMBOLS \
                    and SYMBOLS[operator_stack[-1]]["type"] == "fn":
                out_stack.append(Node(operator_stack.pop(), left=out_stack.pop()))
    if len(out_stack[-1]) == num_tokens:
        return out_stack[-1]
    else:
        raise Exception(f"Error while parsing expression {expr_str}.")


def infix_to_postfix(exprs: list[list[str]]) -> list[list[str]]:
    postfix = []
    for expr in exprs:
        tree = tokens_to_tree(expr)
        postfix.append(tree.to_postfix())
    return postfix


def tree_to_function(tree: Node, has_y=False, number=None)-> str:
    error_estimation_param = "" if not has_y else ", y"
    if number is not None:
        fun_body = f"def _executable_expression_{number}(X, C{error_estimation_param}):\n"
    else:
        fun_body = f"def _executable_expression_(X, C{error_estimation_param}):\n"
    code, symbol, var_counter, const_counter = tree_to_function_rec(tree)
    for c in code:
        fun_body += "\t" + c + "\n"

    if has_y:
        fun_body += f"\treturn np.sqrt(np.mean(({symbol}-y)**2))"
        return fun_body
    else:
        fun_body += "\treturn " + symbol
        return fun_body


def tree_to_function_rec(tree: Node, var_counter: int=0, const_counter: int=0) -> tuple[list[str], str, int, int]:
    if tree.left is None and tree.right is None:
        if tree.symbol in SYMBOLS and SYMBOLS[tree.symbol]["type"] in ["var", "lit"]:
            return [], SYMBOLS[tree.symbol]["np_fn"], var_counter, const_counter
        elif tree.symbol in SYMBOLS and SYMBOLS[tree.symbol]["type"] == "const":
            return [], SYMBOLS[tree.symbol]["np_fn"].format(const_counter), var_counter, const_counter + 1
        else:
            if is_float(tree.symbol):
                return [], tree.symbol, var_counter, const_counter
            else:
                raise Exception(f"Error while parsing expression {tree.symbol}.")

    elif tree.left is not None and tree.right is None:
        code, symbol, var_counter, const_counter = tree_to_function_rec(tree.left, const_counter)
        output_symbol = "y_{}".format(var_counter)
        code.append(SYMBOLS[tree.symbol]["np_fn"].format(output_symbol, symbol))
        return code, output_symbol, var_counter + 1, const_counter

    else:
        left_code, left_symbol, var_counter, const_counter = tree_to_function_rec(tree.left, var_counter, const_counter)
        right_code, right_symbol, var_counter, const_counter = tree_to_function_rec(tree.right, var_counter, const_counter)
        output_symbol = "y_{}".format(var_counter)
        code = left_code + right_code
        code.append(SYMBOLS[tree.symbol]["np_fn"].format(output_symbol, left_symbol, right_symbol))
        return code, output_symbol, var_counter + 1, const_counter


def estimate_parameters(expr: list[str], data: np.array, y: np.array, expr_id: int):
    num_constants = sum([1 for t in expr if t == "C"])
    tree = tokens_to_tree(expr)

    if num_constants == 0:
        function_body = tree_to_function(tree, True, expr_id)
        exec(function_body)
        rmse = locals()["_executable_expression_{}".format(expr_id)](data, np.array([]), y)
        del locals()["_executable_expression_{}".format(expr_id)]
        return rmse, np.array([])
    else:
        function_body = tree_to_function(tree, True, expr_id)
        exec(function_body)
        rmse, constants = optimize_parameters(locals()["_executable_expression_{}".format(expr_id)], data, y, num_constants)
        del locals()["_executable_expression_{}".format(expr_id)]
        return rmse, constants


def optimize_parameters(function, data: np.array, y: np.array, num_constants: int):
    constants = np.zeros(num_constants)
    res = minimize(lambda c: function(data, c, y), constants, method="L-BFGS-B")
    return res.fun, res.x

if __name__ == '__main__':
    import numpy as np

    data = np.array([[1],
                     [2],
                     [3]])
    y = np.array([2, 3, 4])
    constants = np.array([2])

    # expressions = [["C", "*", "B", "-", "A"], ["2", "*","sqrt", "(", "A", ")", "*", "B"]]
    expressions = [["A", "+", "C"]]
    for expr in expressions:
        print(estimate_parameters(expr, data, y, 0))
        # tree = tokens_to_tree(expr)
        # created_function = tree_to_function(tree)
        # print(expr)
        # print()
        # print(created_function)
        # print()
        # exec(created_function)
        # # print(adict["a"](data, constants))
        # a = 0
        print("-----------------------------------------------------------------")