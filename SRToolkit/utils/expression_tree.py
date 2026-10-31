"""
Binary expression tree ([Node][SRToolkit.utils.expression_tree.Node]) and conversion utilities between token
lists, trees, and LaTeX strings.
"""

import math
import string
import warnings
from copy import copy
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union

from SRToolkit.utils.symbol_library import SymbolLibrary


class ExpressionParseError(ValueError):
    """
    Raised by [tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree] when a token list
    is not a well-formed expression: an unknown token, a missing operand, two adjacent operands,
    or unbalanced or empty brackets.

    [SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] marks expressions that raise it
    as ``invalid``.

    Examples:
        >>> try:
        ...     tokens_to_tree(["(", "X_0", "+", "X_1"], SymbolLibrary.default_symbols())
        ... except ExpressionParseError as e:
        ...     print(e)
        Error while parsing expression (X_0+X_1: unclosed '('.
    """


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

        Warning:
            The second positional argument is ``right``, not ``left``. When passing
            children positionally (e.g. ``Node("+", Node("a"), Node("b"))``),
            ``Node("a")`` becomes the ``right`` child and ``Node("b")`` the ``left``.
            Use keyword arguments to avoid confusion: ``Node("+", right=Node("a"), left=Node("b"))``.

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
                during infix reconstruction. If ``None`` with ``"infix"`` notation, a
                ``UserWarning`` is issued and the output may contain redundant parentheses.
            notation: Output notation: ``"infix"``, ``"prefix"``, or ``"postfix"``.
                Default ``"infix"``.

        Returns:
            Token list representing the subtree rooted at this node.

        Raises:
            Exception: If ``notation`` is not one of the accepted values.
            Exception: If ``symbol_library`` is provided and a token's type cannot be
                resolved during infix reconstruction.
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
            try:
                symbol_library = SymbolLibrary.get_active()
            except RuntimeError:
                pass

        if notation == "infix" and symbol_library is None:
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

        if notation == "infix":
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

    def to_latex(self, symbol_library: Optional[SymbolLibrary] = None) -> str:
        r"""
        Transforms the tree rooted at this node into a LaTeX expression.

        Examples:
            >>> node = Node("+", right=Node("X_0"), left=Node("1"))
            >>> node.to_latex(symbol_library=SymbolLibrary.default_symbols())
            '$1 + X_{0}$'
            >>> node = Node("+", right=Node("*", right=Node("X_0"), left=Node("X_1")), left=Node("1"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $1 + X_{1} \cdot X_{0}$
            >>> node = Node("sin", None, Node("X_0"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $\sin X_{0}$
            >>> node = Node("+", right=Node("*", right=Node("X_0"), left=Node("C")), left=Node("C"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $C_{0} + C_{1} \cdot X_{0}$

        Args:
            symbol_library: Symbol library providing LaTeX templates for each token.
                If None, falls back to the currently active library set via
                'with SymbolLibrary(...) as sl:'. Defaults to None.

        Returns:
            A LaTeX string of the form ``$...$``.

        Raises:
            Exception: If the tree contains a token whose type cannot be resolved in
                ``symbol_library``.
        """
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_active()
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
            template = symbol_library.get_latex_str(self.symbol)
            grouped, _ = (_latex_placeholders(template) or ((False, False),))[0]
            if self.left is not None and not grouped and not _is_latex_atom(self.left, symbol_library):
                left = f"({left})"
            return template.format(left), num_const
        elif symbol_library.get_type(self.symbol) == "op":
            template = symbol_library.get_latex_str(self.symbol)
            placeholders = _latex_placeholders(template) + ((False, False),) * 2
            precedence = symbol_library.get_precedence(self.symbol)
            if self.left is not None and self._latex_needs_brackets(
                self.left, placeholders[0], precedence, False, symbol_library
            ):
                left = f"({left})"
            if self.right is not None and self._latex_needs_brackets(
                self.right, placeholders[1], precedence, True, symbol_library
            ):
                right = f"({right})"
            return template.format(left, right), num_const
        else:
            raise Exception(f"Invalid symbol type for symbol {self.symbol}.")

    @staticmethod
    def _latex_needs_brackets(
        child: "Node",
        placeholder: Tuple[bool, bool],
        precedence: int,
        is_right: bool,
        symbol_library: SymbolLibrary,
    ) -> bool:
        grouped, is_base = placeholder
        if grouped:
            # The template already groups this operand in braces (\frac{..}{..}, ^{..}).
            return False
        if is_base:
            # Anything but an atom must be bracketed as the base of a super- or subscript.
            return not _is_latex_atom(child, symbol_library)
        if is_float(child.symbol):
            return is_right and str(child.symbol).startswith("-")
        if symbol_library.get_type(child.symbol) != "op":
            return False
        child_precedence = symbol_library.get_precedence(child.symbol)
        # Operators are left-associative, so an operand of equal precedence on the right was
        # grouped explicitly: a - (b + c).
        return child_precedence < precedence or (is_right and child_precedence == precedence)

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


@lru_cache(maxsize=None)
def _latex_placeholders(template: str) -> Tuple[Tuple[bool, bool], ...]:
    # For each {} placeholder in a LaTeX template, in order: whether the template wraps it in
    # braces (so it is already grouped), and whether it is the base of a super- or subscript.
    # Cached, since each symbol has one fixed template; a tuple, so callers cannot change it.
    # Formatter.parse splits the literal text at every escaped brace, so join it back up:
    # texts[i] is the text before placeholder i and texts[i + 1] the text after it.
    texts = [""]
    for literal, field, _, _ in string.Formatter().parse(template):
        texts[-1] += literal
        if field is not None:
            texts.append("")
    return tuple(
        (texts[i].endswith("{") and texts[i + 1].startswith("}"), texts[i + 1].lstrip().startswith(("^", "_")))
        for i in range(len(texts) - 1)
    )


def _is_latex_atom(node: Node, symbol_library: SymbolLibrary) -> bool:
    if is_float(node.symbol):
        return not str(node.symbol).startswith("-")
    return symbol_library.get_type(node.symbol) in ["var", "const", "lit"]


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
        >>> is_float("hello")
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


def tokens_to_tree(tokens: List[str], sl: Optional[SymbolLibrary] = None, notation: str = "infix") -> Node:
    """
    Parse a token list into an expression tree.

    Infix token lists are parsed with the shunting-yard algorithm; prefix and
    postfix lists are parsed with a single stack pass using the arity implied by
    each token's type (``"op"`` is binary, ``"fn"`` is unary, and variables,
    constants, literals, and numeric literals are leaves). This is the inverse of
    [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list], so a tree
    survives a round trip through any of the three notations.

    Examples:
        >>> tree = tokens_to_tree(["(", "X_0", "+", "X_1", ")"], SymbolLibrary.default_symbols())
        >>> len(tree)
        3
        >>> tree = tokens_to_tree(["+", "1", "X_0"], SymbolLibrary.default_symbols(), notation="prefix")
        >>> tree.to_list(SymbolLibrary.default_symbols())
        ['1', '+', 'X_0']
        >>> tree = tokens_to_tree(["X_0", "X_1", "*", "1", "+"], SymbolLibrary.default_symbols(), notation="postfix")
        >>> tree.to_list(SymbolLibrary.default_symbols())
        ['X_0', '*', 'X_1', '+', '1']
        >>> tree = tokens_to_tree(["sin", "X_0"], SymbolLibrary.default_symbols(), notation="prefix")
        >>> tree.to_list(SymbolLibrary.default_symbols())
        ['sin', '(', 'X_0', ')']

    Infix conventions, some of which differ from usual mathematical notation:

    - A function with positive precedence is prefix (``sin ( X_0 )``); one with zero or negative
      precedence is postfix (``X_0 ^2``), as in
      [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list].
    - A prefix function written without brackets applies to the operand that follows it, more
      tightly than any binary operator: ``u- X_0 * X_1`` is ``(-X_0) * X_1``, and ``u- X_0 ^ 2``
      is ``(-X_0)^2``, not ``-(X_0^2)`` as in mathematics. A postfix function binds more tightly
      still: ``u- X_0 ^2`` is ``-(X_0^2)``.
    - Binary operators are left-associative, ``^`` included: ``X_0 ^ X_1 ^ X_2`` is
      ``(X_0^X_1)^X_2``, whereas mathematics reads it as ``X_0^(X_1^X_2)``.
    - Numbers are any token ``float()`` accepts (``1.5``, ``-2``, ``1e-3``, ``inf``), except
      ``nan``.

    Write brackets where these conventions differ from what you mean.

    Args:
        tokens: Token list in the notation given by ``notation``.
        sl: Symbol library used to resolve token types and precedences. If None,
            falls back to the currently active library set via
            'with SymbolLibrary(...) as sl:'. Defaults to None.
        notation: Notation of ``tokens``: ``"infix"``, ``"prefix"``, or
            ``"postfix"``. Default ``"infix"``.

    Returns:
        Root [Node][SRToolkit.utils.expression_tree.Node] of the parsed expression tree.

    Raises:
        ExpressionParseError: If a token is absent from ``sl``, or if the expression is not
            well-formed (e.g. an operator lacks operands, brackets are unbalanced or empty, or
            tokens remain unconsumed).
        Exception: If ``notation`` is not one of the accepted values.
    """
    if sl is None:
        sl = SymbolLibrary.get_active()

    if notation == "infix":
        return _infix_to_tree(tokens, sl)
    elif notation in ("prefix", "postfix"):
        return _prefix_postfix_to_tree(tokens, sl, notation)
    else:
        raise Exception(
            "Invalid notation selected. Use 'infix', 'prefix', 'postfix', or leave blank (defaults to 'infix')."
        )


def _infix_to_tree(tokens: List[str], sl: SymbolLibrary) -> Node:
    """
    Parse an infix token list into an expression tree using the shunting-yard algorithm.

    The parser tracks whether an operand or an operator comes next, so malformed input is
    rejected rather than silently repaired: brackets must be balanced and non-empty, and an
    operand cannot directly follow another operand. See
    [tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree] for the conventions.

    Args:
        tokens: Token list in infix notation.
        sl: Symbol library used to resolve token types and precedences.

    Returns:
        Root [Node][SRToolkit.utils.expression_tree.Node] of the parsed expression tree.

    Raises:
        ExpressionParseError: If a token is absent from ``sl``, or if the expression is not
            well-formed.
    """
    expr_str = "".join(tokens)

    def error(reason: str) -> ExpressionParseError:
        return ExpressionParseError(f"Error while parsing expression {expr_str}: {reason}.")

    operator_stack: List[str] = []
    out_stack: List[Node] = []

    def reduce() -> None:
        # The state checks below guarantee that the operands are on the stack.
        symbol = operator_stack.pop()
        if sl.get_type(symbol) == "fn":
            out_stack.append(Node(symbol, left=out_stack.pop()))
        else:
            right = out_stack.pop()
            out_stack.append(Node(symbol, right=right, left=out_stack.pop()))

    expect_operand = True

    def push_operand(token: str) -> None:
        nonlocal expect_operand
        if not expect_operand:
            raise error(f"operand '{token}' cannot follow an operand")
        out_stack.append(Node(token))
        expect_operand = False

    for token in tokens:
        token_type = sl.get_type(token)
        if token == "(":
            if not expect_operand:
                raise error("'(' cannot follow an operand")
            operator_stack.append(token)
        elif token == ")":
            if expect_operand:
                raise error("missing operand before ')'")
            while len(operator_stack) > 0 and operator_stack[-1] != "(":
                reduce()
            if len(operator_stack) == 0:
                raise error("unmatched ')'")
            operator_stack.pop()
            if len(operator_stack) > 0 and sl.get_type(operator_stack[-1]) == "fn":
                reduce()
        elif token_type in ["var", "const", "lit"]:
            push_operand(token)
        elif is_float(token):
            if math.isnan(float(token)):
                raise error(f"'{token}' is not a number")
            push_operand(token)
        elif token_type == "fn" and sl.get_precedence(token) <= 0:
            if expect_operand:
                raise error(f"function '{token}' is missing an operand")
            out_stack.append(Node(token, left=out_stack.pop()))
        elif token_type == "fn":
            if not expect_operand:
                raise error(f"function '{token}' cannot follow an operand")
            operator_stack.append(token)
        elif token_type == "op":
            if expect_operand:
                raise error(f"operator '{token}' is missing an operand")
            # A prefix function on the stack has its operand complete, and binds more tightly than
            # any binary operator regardless of its precedence value.
            while (
                len(operator_stack) > 0
                and operator_stack[-1] != "("
                and (
                    sl.get_type(operator_stack[-1]) == "fn"
                    or sl.get_precedence(operator_stack[-1]) >= sl.get_precedence(token)
                )
            ):
                reduce()
            operator_stack.append(token)
            expect_operand = True
        else:
            raise ExpressionParseError(
                f'Invalid symbol "{token}" in expression {expr_str}. Did you add token "{token}" to the symbol library?'
            )

    if expect_operand:
        raise error("the expression is empty" if len(tokens) == 0 else "missing operand at the end")
    while len(operator_stack) > 0:
        if operator_stack[-1] == "(":
            raise error("unclosed '('")
        reduce()
    return out_stack[0]


def _prefix_postfix_to_tree(tokens: List[str], sl: SymbolLibrary, notation: str) -> Node:
    """
    Parse a prefix or postfix token list into an expression tree with a single stack pass.

    Prefix and postfix are *mirror images* of each other, not reverses: operands keep
    their left-to-right order in both notations and only the operator moves, so postfix
    is **not** ``reversed(prefix)`` (e.g. ``(X_0 - X_1) - X_2`` is ``- - X_0 X_1 X_2`` in
    prefix but ``X_0 X_1 - X_2 -`` in postfix). The single pass handles both by flipping
    two things together:

    * **scan direction** — prefix is scanned right-to-left, postfix left-to-right, so that
      a token's operands have always been pushed by the time the token is reached;
    * **operand pop-order** — for a binary operator, prefix pops ``left`` then ``right``
      while postfix pops ``right`` then ``left``.

    That operand-order flip is what keeps non-commutative operators (``-``, ``/``, ``^``)
    correct; reversing the token list alone would silently swap their operands.

    Args:
        tokens: Token list in prefix or postfix notation.
        sl: Symbol library used to resolve token types.
        notation: Either ``"prefix"`` or ``"postfix"``.

    Returns:
        Root [Node][SRToolkit.utils.expression_tree.Node] of the parsed expression tree.

    Raises:
        ExpressionParseError: If a token is absent from ``sl``, or if the expression is
            syntactically invalid (an operator/function without enough operands, or
            more than one subtree left on the stack at the end).
    """
    expr_str = "".join(tokens)
    out_stack: List[Node] = []
    for token in reversed(tokens) if notation == "prefix" else tokens:
        if sl.get_type(token) in ["var", "const", "lit"]:
            out_stack.append(Node(token))
        elif is_float(token):
            if math.isnan(float(token)):
                raise ExpressionParseError(f"Error while parsing expression {expr_str}: '{token}' is not a number.")
            out_stack.append(Node(token))
        elif sl.get_type(token) == "fn":
            if len(out_stack) < 1:
                raise ExpressionParseError(
                    f"Error while parsing expression {expr_str}: function '{token}' is missing an operand."
                )
            out_stack.append(Node(token, left=out_stack.pop()))
        elif sl.get_type(token) == "op":
            if len(out_stack) < 2:
                raise ExpressionParseError(
                    f"Error while parsing expression {expr_str}: operator '{token}' is missing an operand."
                )
            if notation == "prefix":
                left = out_stack.pop()
                right = out_stack.pop()
            else:
                right = out_stack.pop()
                left = out_stack.pop()
            out_stack.append(Node(token, right=right, left=left))
        else:
            raise ExpressionParseError(
                f'Invalid symbol "{token}" in expression {expr_str}. Did you add token "{token}" to the symbol library?'
            )
    if len(out_stack) != 1:
        raise ExpressionParseError(f"Error while parsing expression {expr_str}.")
    return out_stack[0]


def expr_to_latex(expr: Union[Node, List[str]], symbol_library: Optional[SymbolLibrary] = None) -> str:
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
        symbol_library: Symbol library providing LaTeX templates. If None, falls
            back to the currently active library set via
            'with SymbolLibrary(...) as sl:'. Defaults to None.

    Returns:
        A LaTeX string of the form ``$...$``, or an empty string if conversion fails.
    """
    if symbol_library is None:
        symbol_library = SymbolLibrary.get_active()
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
