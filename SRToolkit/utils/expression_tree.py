"""Expression trees and conversions between token lists, trees and LaTeX.

[Node][SRToolkit.utils.expression_tree.Node] is a binary expression tree, built from tokens by
[tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree]. Functions elsewhere accept an
expression as a token list or an [Expression][SRToolkit.utils.expression.Expression], which
parses its tokens into a tree once; wrap a tree with
[Expression.from_tree][SRToolkit.utils.expression.Expression.from_tree] to pass it to them.
"""

import itertools
import math
import string
import warnings
from collections.abc import Iterator
from copy import copy
from functools import lru_cache
from typing import Any

from SRToolkit.utils.symbol_library import SymbolLibrary


class ExpressionParseError(ValueError):
    """Raised by [tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree] when a token list
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
    def __init__(self, symbol: str, right: "Node | None" = None, left: "Node | None" = None) -> None:
        """A node in a binary expression tree.

        - Binary operators (``"op"``) set both ``left`` and ``right``.
        - Unary functions (``"fn"``) set only ``left``; ``right`` is ``None``.
        - Leaves (variables, free constants, named constants, numeric values) have both children as ``None``.

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

    def to_list(self, symbol_library: SymbolLibrary | None = None, notation: str = "infix") -> list[str]:
        """Transforms the tree rooted at this node into a list of tokens.

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
            ExpressionParseError: If, in infix notation with a library (given or active), a symbol of
                the tree is neither a number nor in the library.
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
            symbol = symbol_library.get(self.symbol)
            if symbol is None:
                raise ExpressionParseError(f"Invalid symbol type for symbol {self.symbol}.")
            if symbol.type in ["var", "const", "lit"]:
                return [self.symbol]
            elif symbol.type == "fn":
                if symbol.precedence > 0:
                    return [self.symbol, "("] + left + [")"]
                else:
                    if len(left) > 1:
                        left = ["("] + left + [")"]
                    return left + [self.symbol]
            elif symbol.type == "op":
                left_child = None if self.left is None else symbol_library.get(self.left.symbol)
                if left_child is not None and -1 < left_child.precedence <= symbol.precedence:
                    left = ["("] + left + [")"]
                right_child = None if self.right is None else symbol_library.get(self.right.symbol)
                if right_child is not None and -1 < right_child.precedence <= symbol.precedence:
                    right = ["("] + right + [")"]
                return left + [self.symbol] + right
            else:
                raise ExpressionParseError(f"Invalid symbol type for symbol {self.symbol}.")
        else:
            raise Exception(
                "Invalid notation selected. Use 'infix', 'prefix', 'postfix', or leave blank (defaults to 'infix')."
            )

    def to_latex(self, symbol_library: SymbolLibrary | None = None) -> str:
        r"""Return the tree rooted at this node as a LaTeX string.

        A labelled constant shows its label's digits as written (``C_5`` → ``C_{5}``), and bare
        constants get, in order of appearance, the lowest numbers that no labelled constant in the
        expression uses, so the numbers are not parameter indices.

        Examples:
            >>> node = Node("+", right=Node("*", right=Node("X_0"), left=Node("X_1")), left=Node("1"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $1 + X_{1} \cdot X_{0}$
            >>> node = Node("sin", None, Node("X_0"))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $\sin X_{0}$
            >>> node = Node("+", right=Node("C_0"), left=Node("*", right=Node("X_0"), left=Node("C")))
            >>> print(node.to_latex(symbol_library=SymbolLibrary.default_symbols()))
            $C_{1} \cdot X_{0} + C_{0}$

        Args:
            symbol_library: Symbol library providing LaTeX templates for each token.
                If None, falls back to the currently active library set via
                'with SymbolLibrary(...) as sl:'. Defaults to None.

        Returns:
            A LaTeX string of the form ``$...$``.

        Raises:
            ExpressionParseError: If the tree contains a token whose type cannot be resolved in
                ``symbol_library``.
        """
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_active()
        used = {label.rpartition("_")[2] for label in _parameter_labels(self, symbol_library) if label is not None}
        bare_numbers = (str(n) for n in itertools.count() if str(n) not in used)
        return f"${self.__to_latex_rec(symbol_library, bare_numbers)}$"

    def __to_latex_rec(self, symbol_library: SymbolLibrary, bare_numbers: Iterator[str]) -> str:
        left = "" if self.left is None else self.left.__to_latex_rec(symbol_library, bare_numbers)
        right = "" if self.right is None else self.right.__to_latex_rec(symbol_library, bare_numbers)

        if is_float(self.symbol):
            return str(self.symbol)
        symbol = symbol_library.get(self.symbol)
        if symbol is None:
            raise ExpressionParseError(f"Invalid symbol type for symbol {self.symbol}.")
        if symbol.type == "const":
            # A labelled constant shows its label's number, a bare one the next unused number.
            labelled = self.symbol not in symbol_library.symbols
            number = self.symbol.rpartition("_")[2] if labelled else next(bare_numbers)
            return symbol.latex_str.format(number)
        elif symbol.type in ["var", "lit"]:
            return symbol.latex_str
        elif symbol.type == "fn":
            template = symbol.latex_str
            grouped, _ = (_latex_placeholders(template) or ((False, False),))[0]
            if self.left is not None and not grouped and not _is_latex_atom(self.left, symbol_library):
                left = f"({left})"
            return template.format(left)
        elif symbol.type == "op":
            template = symbol.latex_str
            placeholders = _latex_placeholders(template) + ((False, False),) * 2
            precedence = symbol.precedence
            if self.left is not None and self._latex_needs_brackets(
                self.left, placeholders[0], precedence, False, symbol_library
            ):
                left = f"({left})"
            if self.right is not None and self._latex_needs_brackets(
                self.right, placeholders[1], precedence, True, symbol_library
            ):
                right = f"({right})"
            return template.format(left, right)
        else:
            raise ExpressionParseError(f"Invalid symbol type for symbol {self.symbol}.")

    @staticmethod
    def _latex_needs_brackets(
        child: "Node",
        placeholder: tuple[bool, bool],
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
        child_symbol = symbol_library.get(child.symbol)
        if child_symbol is None or child_symbol.type != "op":
            return False
        child_precedence = child_symbol.precedence
        # Operators are left-associative, so an operand of equal precedence on the right was
        # grouped explicitly: a - (b + c).
        return child_precedence < precedence or (is_right and child_precedence == precedence)

    def height(self) -> int:
        """Return the height of the subtree rooted at this node.

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
        """Return the number of nodes in the subtree rooted at this node.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> len(node)
            3

        Returns:
            Total node count of the subtree.
        """
        return 1 + (len(self.left) if self.left is not None else 0) + (len(self.right) if self.right is not None else 0)

    def __str__(self) -> str:
        """Return the expression as a concatenated string using default infix notation that may contain redundant parentheses.

        Examples:
            >>> node = Node("+", Node("x"), Node("1"))
            >>> str(node)
            '1+x'

        Returns:
            Concatenated token string with no spaces.
        """
        return "".join(self.to_list())

    def __copy__(self) -> "Node":
        """Return a deep copy of the subtree rooted at this node.

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
def _latex_placeholders(template: str) -> tuple[tuple[bool, bool], ...]:
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
    symbol = symbol_library.get(node.symbol)
    return symbol is not None and symbol.type in ["var", "const", "lit"]


def is_float(element: Any) -> bool:
    """Return ``True`` if ``element`` can be interpreted as a floating-point number.

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
        ``True`` if ``float(element)`` succeeds, ``False`` otherwise (including ``None`` and values
        ``float`` cannot convert, such as lists or complex numbers). Numeric strings and ``bool``
        count as floats.
    """
    if element is None:
        return False
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def _check_tokens(tokens: Any, name: str) -> None:
    """Raise ``TypeError`` unless ``tokens`` is a list of strings, naming the parameter ``name``.

    Raises:
        TypeError: If ``tokens`` is not a list, or contains an element that is not a string. For a
            number, the message suggests writing it as a string.
    """
    if not isinstance(tokens, list):
        raise TypeError(f"{name} must be a list of token strings, got {type(tokens).__name__}.")
    for token in tokens:
        if not isinstance(token, str):
            hint = ""
            if not isinstance(token, bool) and is_float(token):
                hint = f" Write numbers as strings, e.g. {str(token)!r}."
            raise TypeError(
                f"{name} must contain only token strings, but contains {token!r} ({type(token).__name__}).{hint}"
            )


def _constant_indices(tokens: list[str], symbol_library: SymbolLibrary) -> list[int]:
    """Return the parameter index of each free constant in ``tokens``, in token order.

    Parameters are numbered by first appearance: every bare constant (a registered ``"const"``
    symbol, e.g. ``C``) is a new parameter, a labelled constant (``C_5``) gets one at its first
    occurrence and keeps it. Operands keep their order in infix, prefix and postfix, so the
    numbering is the same in every notation. The single definition of the order, shared by the
    compiler and ``_parameter_labels``.
    """
    slots: dict[str, int] = {}
    indices: list[int] = []
    num_parameters = 0
    for token in tokens:
        symbol = symbol_library.get(token)
        if symbol is None or symbol.type != "const":
            continue
        if token in symbol_library.symbols:
            indices.append(num_parameters)
            num_parameters += 1
        else:
            if token not in slots:
                slots[token] = num_parameters
                num_parameters += 1
            indices.append(slots[token])
    return indices


def _parameter_labels(expr: "list[str] | Node", symbol_library: SymbolLibrary) -> list[str | None]:
    """Return the label of each free parameter, or ``None`` for a bare constant.

    In the order of the parameter vector (see
    [Expression.parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels]). A
    label is one parameter however often it occurs; each bare ``C`` is its own. Reads the tokens
    (a tree is written out first) without parsing or checking them.
    """
    tokens = expr.to_list(symbol_library=symbol_library) if isinstance(expr, Node) else expr
    constants = [token for token in tokens if (s := symbol_library.get(token)) is not None and s.type == "const"]
    indices = _constant_indices(tokens, symbol_library)
    labels: list[str | None] = [None] * len(set(indices))
    for token, index in zip(constants, indices):
        if token not in symbol_library.symbols:
            labels[index] = token
    return labels


def tokens_to_tree(tokens: list[str], sl: SymbolLibrary | None = None, notation: str = "infix") -> Node:
    """Parse a token list into an expression tree.

    Infix token lists are parsed with the shunting-yard algorithm; prefix and
    postfix lists are parsed with a single stack pass using the arity implied by
    each token's type (``"op"`` is binary, ``"fn"`` is unary, and variables,
    free constants, named constants and numbers are leaves). This is the inverse of
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
        TypeError: If ``tokens`` is not a list of strings. Numbers must be written as strings
            (``"1.5"``, not ``1.5``).
        ExpressionParseError: If a token is absent from ``sl``, or if the expression is not
            well-formed (e.g. an operator lacks operands, brackets are unbalanced or empty, or
            tokens remain unconsumed).
        Exception: If ``notation`` is not one of the accepted values.
    """
    _check_tokens(tokens, "tokens")
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


def _infix_to_tree(tokens: list[str], sl: SymbolLibrary) -> Node:
    """Parse an infix token list into an expression tree using the shunting-yard algorithm.

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

    operator_stack: list[str] = []
    out_stack: list[Node] = []

    def reduce() -> None:
        # The state checks below guarantee that the operands are on the stack.
        symbol = operator_stack.pop()
        if sl[symbol].type == "fn":
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
        symbol = sl.get(token)
        token_type = None if symbol is None else symbol.type
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
            if len(operator_stack) > 0 and operator_stack[-1] != "(" and sl[operator_stack[-1]].type == "fn":
                reduce()
        elif token_type in ["var", "const", "lit"]:
            push_operand(token)
        elif is_float(token):
            if math.isnan(float(token)):
                raise error(f"'{token}' is not a number")
            push_operand(token)
        elif symbol is not None and symbol.type == "fn" and symbol.precedence <= 0:
            if expect_operand:
                raise error(f"function '{token}' is missing an operand")
            out_stack.append(Node(token, left=out_stack.pop()))
        elif token_type == "fn":
            if not expect_operand:
                raise error(f"function '{token}' cannot follow an operand")
            operator_stack.append(token)
        elif symbol is not None and symbol.type == "op":
            if expect_operand:
                raise error(f"operator '{token}' is missing an operand")
            # A prefix function on the stack has its operand complete, and binds more tightly than
            # any binary operator regardless of its precedence value.
            while (
                len(operator_stack) > 0
                and operator_stack[-1] != "("
                and (sl[operator_stack[-1]].type == "fn" or sl[operator_stack[-1]].precedence >= symbol.precedence)
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


def _prefix_postfix_to_tree(tokens: list[str], sl: SymbolLibrary, notation: str) -> Node:
    """Parse a prefix or postfix token list into an expression tree with a single stack pass.

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
    out_stack: list[Node] = []
    for token in reversed(tokens) if notation == "prefix" else tokens:
        symbol = sl.get(token)
        token_type = None if symbol is None else symbol.type
        if token_type in ["var", "const", "lit"]:
            out_stack.append(Node(token))
        elif is_float(token):
            if math.isnan(float(token)):
                raise ExpressionParseError(f"Error while parsing expression {expr_str}: '{token}' is not a number.")
            out_stack.append(Node(token))
        elif token_type == "fn":
            if len(out_stack) < 1:
                raise ExpressionParseError(
                    f"Error while parsing expression {expr_str}: function '{token}' is missing an operand."
                )
            out_stack.append(Node(token, left=out_stack.pop()))
        elif token_type == "op":
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
