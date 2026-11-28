"""The [Expression][SRToolkit.utils.expression.Expression] class: an expression with its free-constant specs.

An ``Expression`` is the form that functions accept besides token lists. It holds the tokens, the
symbol library they are written in and the specs of its labelled constants (see
[free_constants][SRToolkit.utils.free_constants]), and computes its tree and other facts once, so an
expression used several times is parsed once.
"""

from collections.abc import Sequence
from typing import Any

from .expression_tree import ExpressionParseError, Node, _check_tokens, parameter_labels, tokens_to_tree
from .free_constants import Constant, _spec_key, _specs_by_label, resolve_constants
from .symbol_library import SymbolLibrary


class Expression:
    """An immutable expression in a symbol library, with its free-constant specs, parsed on first use.

    An expression holds its infix tokens, the library they are written in, the specs of its
    labelled constants, and facts derived from them, each computed on first use and kept. Build one
    from tokens or with [from_tree][SRToolkit.utils.expression.Expression.from_tree]. Functions that
    take an expression accept a token list (an expression without specs) or an ``Expression``, so an
    expression used several times is parsed once. Do not change the library after building an
    expression.

    The tokens are type-checked and the specs' labels checked when the expression is built, but the
    tokens are parsed only on first use, so a malformed expression can be built:
    [parses][SRToolkit.utils.expression.Expression.parses] is ``False`` and
    [tree][SRToolkit.utils.expression.Expression.tree] raises.
    [parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels],
    [num_parameters][SRToolkit.utils.expression.Expression.num_parameters] and
    [variables][SRToolkit.utils.expression.Expression.variables] are read from the tokens
    without parsing, so they work on malformed expressions too.

    Args:
        tokens: The expression as a list or tuple of infix token strings.
        symbol_library: The library the tokens are written in. ``None`` uses the active or default
            library, fixed when the expression is built.
        free_constants: A sequence of [Constant][SRToolkit.utils.free_constants.Constant] specs of
            labelled constants of the expression, in any order. ``None`` or empty: none; every
            parameter is fitted under the default spec.

    Raises:
        TypeError: If ``tokens`` is not a list or tuple of token strings, or ``free_constants`` is
            not a sequence of ``Constant`` (a single ``Constant`` is not a sequence; for a dict
            element the message points to ``Constant.from_dict``).
        ValueError: If a spec has no label, two specs share a label, or a spec's label is not a
            labelled constant of the expression.

    Examples:
        >>> sl = SymbolLibrary.default_symbols(2)
        >>> g = Constant(label="C_0", value=9.81, fixed=True)
        >>> expr = Expression(["C", "*", "X_1", "+", "C_0", "*", "X_0", "-", "C_0"], sl, [g])
        >>> expr.num_parameters, expr.parameter_labels, expr.variables
        (2, (None, 'C_0'), ('X_1', 'X_0'))
        >>> expr.free_constants
        (Constant(label='C_0', prior=None, bounds=None, value=9.81, fixed=True),)
        >>> Expression(["X_0", "+"], sl).parses
        False

    Notes:
        Two expressions are equal, and hash equally, if their tokens are equal and their specs are
        equal in any order, whatever their libraries. Specs are compared by ``label``, ``prior``,
        ``bounds``, ``value`` and ``fixed``, with numbers compared as floats; an unset ``prior`` or
        ``bounds`` differs from an explicit one, even if they resolve the same. An expression never
        equals a token list.

        ``str`` joins the tokens with no separator (``sin(X_0)``). ``repr`` gives
        ``Expression([...])``, with the specs if there are any and without the library. A pickled
        expression keeps only its tokens, library and specs.
    """

    __slots__ = (
        "_tokens",
        "_symbol_library",
        "_free_constants",
        "_spec_key",
        "_tree",
        "_parse_error",
        "_parameter_labels",
        "_variables",
        "_latex",
    )

    def __init__(
        self,
        tokens: list[str] | tuple[str, ...],
        symbol_library: SymbolLibrary | None = None,
        free_constants: Sequence[Constant] | None = None,
    ) -> None:
        if isinstance(tokens, tuple):
            tokens = list(tokens)
        _check_tokens(tokens, "tokens")
        self._tokens = tuple(tokens)
        self._symbol_library = SymbolLibrary.get_or_default() if symbol_library is None else symbol_library
        if free_constants is not None and (
            isinstance(free_constants, Constant) or not isinstance(free_constants, Sequence)
        ):
            raise TypeError(f"free_constants must be a sequence of Constant, got {type(free_constants).__name__}.")
        for spec in free_constants or ():
            if not isinstance(spec, Constant):
                hint = " Build specs from dicts with Constant.from_dict." if isinstance(spec, dict) else ""
                raise TypeError(f"free_constants must contain only Constant, but contains {type(spec).__name__}.{hint}")
        self._free_constants = tuple(free_constants or ())
        self._spec_key: str | None = None
        self._tree: Node | None = None
        self._parse_error: ExpressionParseError | None = None
        self._parameter_labels: tuple[str | None, ...] | None = None
        self._variables: tuple[str, ...] | None = None
        self._latex: str | None = None
        if self._free_constants:
            _specs_by_label(self._free_constants, self.parameter_labels)

    @classmethod
    def from_tree(
        cls,
        tree: Node,
        symbol_library: SymbolLibrary | None = None,
        free_constants: Sequence[Constant] | None = None,
    ) -> "Expression":
        """Build an expression from a tree.

        The tokens are written out with [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list];
        the tree is not kept, so the expression is the same as one built from these tokens. A
        malformed tree (e.g. an operator with one operand) gives tokens that do not parse.

        Args:
            tree: The expression tree.
            symbol_library: As for the constructor.
            free_constants: As for the constructor.

        Returns:
            The expression.

        Raises:
            TypeError: If ``tree`` is not a [Node][SRToolkit.utils.expression_tree.Node], or as for
                the constructor.
            ExpressionParseError: If a symbol of the tree is neither a number nor in the library.
            ValueError: As for the constructor.

        Examples:
            >>> tree = Node("+", Node("X_0"), Node("1"))
            >>> Expression.from_tree(tree, SymbolLibrary.default_symbols(1)).tokens
            ('1', '+', 'X_0')
        """
        if not isinstance(tree, Node):
            raise TypeError(f"tree must be a Node, got {type(tree).__name__}.")
        if symbol_library is None:
            symbol_library = SymbolLibrary.get_or_default()
        return cls(tree.to_list(symbol_library), symbol_library, free_constants)

    @property
    def tokens(self) -> tuple[str, ...]:
        """The infix tokens.

        As given, or as [Node.to_list][SRToolkit.utils.expression_tree.Node.to_list] writes them for
        [from_tree][SRToolkit.utils.expression.Expression.from_tree].
        """
        return self._tokens

    @property
    def symbol_library(self) -> SymbolLibrary:
        """The symbol library the tokens are written in."""
        return self._symbol_library

    @property
    def free_constants(self) -> tuple[Constant, ...]:
        """The specs of labelled constants, in the given order, as a tuple; empty if there are none."""
        return self._free_constants

    @property
    def tree(self) -> Node:
        """The expression tree, shared by every user of the expression, so do not change it.

        The tokens are parsed on first access. A parse error is kept and raised again on every access.

        Raises:
            ExpressionParseError: If the tokens are not a well-formed expression.
        """
        if self._tree is None:
            if self._parse_error is None:
                try:
                    self._tree = tokens_to_tree(list(self._tokens), self._symbol_library)
                except ExpressionParseError as e:
                    self._parse_error = ExpressionParseError(*e.args)  # kept without traceback or context
            if self._parse_error is not None:
                # A new error each time, so one raise does not change what a later one carries.
                raise ExpressionParseError(*self._parse_error.args) from None
        return self._tree  # type: ignore[return-value]

    @property
    def parses(self) -> bool:
        """Whether the tokens are a well-formed expression."""
        return self.parse_error is None

    @property
    def parse_error(self) -> ExpressionParseError | None:
        """The ``ExpressionParseError`` the tokens raise, or ``None`` if they parse (reading it parses)."""
        try:
            self.tree
        except ExpressionParseError as e:
            return e
        return None

    @property
    def parameter_labels(self) -> tuple[str | None, ...]:
        """The label of each free parameter, or ``None`` for a bare constant.

        As [parameter_labels][SRToolkit.utils.expression_tree.parameter_labels] returns them, read
        from the tokens without parsing.
        """
        if self._parameter_labels is None:
            self._parameter_labels = tuple(parameter_labels(list(self._tokens), self._symbol_library))
        return self._parameter_labels

    @property
    def num_parameters(self) -> int:
        """The number of free parameters (fixed constants included), the length of the parameter vector.

        Read from the tokens without parsing.
        """
        return len(self.parameter_labels)

    @property
    def variables(self) -> tuple[str, ...]:
        """The variable tokens the expression uses, each once, in order of first appearance.

        Read from the tokens without parsing.
        """
        if self._variables is None:
            sl = self._symbol_library
            self._variables = tuple(
                dict.fromkeys(t for t in self._tokens if (s := sl.get(t)) is not None and s.type == "var")
            )
        return self._variables

    def resolve(self, default_constant: Constant) -> tuple[Constant, ...]:
        """Return one complete spec per parameter.

        As [resolve_constants][SRToolkit.utils.free_constants.resolve_constants] does for the
        expression's specs.

        Args:
            default_constant: The default spec, for parameters without a spec.

        Returns:
            One spec per parameter, in the order of
            [parameter_labels][SRToolkit.utils.expression.Expression.parameter_labels].

        Raises:
            ValueError: If ``default_constant`` is not a valid default spec, or a spec that is not
                fixed has a ``value`` outside the bounds it inherits from ``default_constant``.

        Examples:
            >>> from SRToolkit.utils.free_constants import uniform_default
            >>> spec = Constant(label="C_1", value=2.0, fixed=True)
            >>> expr = Expression(["C", "*", "X_0", "+", "C_1"], free_constants=[spec])
            >>> [(s.label, s.value, s.bounds) for s in expr.resolve(uniform_default((-5.0, 5.0)))]
            [(None, None, (-5.0, 5.0)), ('C_1', 2.0, (-5.0, 5.0))]
        """
        return resolve_constants(list(self._tokens), self._free_constants, default_constant, self._symbol_library)

    def to_latex(self) -> str:
        """Return the expression as a LaTeX string.

        Written by [Node.to_latex][SRToolkit.utils.expression_tree.Node.to_latex], which also numbers
        the constants.

        Returns:
            A LaTeX string of the form ``$...$``.

        Raises:
            ExpressionParseError: If the tokens are not a well-formed expression.
        """
        if self._latex is None:
            self._latex = self.tree.to_latex(self._symbol_library)
        return self._latex

    def _key(self) -> tuple[tuple[str, ...], str]:
        # Tokens and spec key: what equality and hashing compare.
        if self._spec_key is None:
            self._spec_key = _spec_key(self._free_constants)
        return self._tokens, self._spec_key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expression) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        if self._free_constants:
            return f"Expression({list(self._tokens)!r}, free_constants={self._free_constants!r})"
        return f"Expression({list(self._tokens)!r})"

    def __str__(self) -> str:
        return "".join(self._tokens)

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (list(self._tokens), self._symbol_library, self._free_constants or None)


def _as_expression(expr: Any, symbol_library: SymbolLibrary | None = None, name: str = "expr") -> Expression:
    """Return a token list or an [Expression][SRToolkit.utils.expression.Expression] as an ``Expression``.

    The single place where SRToolkit accepts "a token list or an Expression". A token list (or
    tuple) becomes an expression without specs. An ``Expression`` is returned unchanged if ``symbol_library`` is
    ``None`` or its own library; for another library object it is built again from its tokens and
    specs with ``symbol_library``, as a token list would be.

    Raises:
        TypeError: If ``expr`` is neither a list or tuple of token strings nor an ``Expression`` (the message
            names ``name``, and for a Node points to
            [Expression.from_tree][SRToolkit.utils.expression.Expression.from_tree]).
        ValueError: If ``expr`` is an ``Expression`` rebuilt with ``symbol_library``, and its specs'
            labels are not labelled constants in that library.
    """
    if isinstance(expr, Expression):
        if symbol_library is None or expr.symbol_library is symbol_library:
            return expr
        return Expression(list(expr.tokens), symbol_library, expr.free_constants)
    if isinstance(expr, Node):
        raise TypeError(
            f"{name} must be a list of token strings or an Expression; wrap a Node with Expression.from_tree."
        )
    if isinstance(expr, tuple):
        expr = list(expr)
    if not isinstance(expr, list):
        raise TypeError(f"{name} must be a list of token strings or an Expression, got {type(expr).__name__}.")
    _check_tokens(expr, name)
    return Expression(expr, symbol_library)


def _to_expression(expr: Any, symbol_library: SymbolLibrary | None = None, name: str = "expr") -> Expression:
    # Temporary until Node inputs are dropped: _as_expression that also accepts a Node, wrapped
    # with Expression.from_tree.
    if isinstance(expr, Node):
        return Expression.from_tree(expr, symbol_library)
    return _as_expression(expr, symbol_library, name)
