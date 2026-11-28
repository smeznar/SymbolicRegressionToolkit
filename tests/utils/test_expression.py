import pickle

import pytest

import SRToolkit.utils.expression as expression
from SRToolkit.utils.expression import Expression, _as_expression
from SRToolkit.utils.expression_tree import ExpressionParseError, Node, parameter_labels
from SRToolkit.utils.free_constants import Constant, resolve_constants, uniform_default
from SRToolkit.utils.symbol_library import SymbolLibrary


class _Tagged(Expression):
    __slots__ = ()


class TestExpression:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(2)

    def test_tokens_are_kept_as_given_and_copied(self):
        tokens = ["(", "X_0", "+", "C", ")"]
        expr = Expression(tokens, self.sl)
        tokens.append("+")
        assert expr.tokens == ("(", "X_0", "+", "C", ")")
        assert expr.symbol_library is self.sl

    def test_tuple_of_tokens_is_accepted(self):
        expr = Expression(("X_0", "+", "C"), self.sl)
        assert Expression(expr.tokens, self.sl) == expr
        assert expr.tokens == ("X_0", "+", "C")

    def test_empty_token_list_builds_but_does_not_parse(self):
        expr = Expression([], self.sl)
        assert expr.parses is False
        assert expr.variables == ()
        assert expr.parameter_labels == ()

    @pytest.mark.parametrize("tokens", [{"X_0"}, "X_0", ["X_0", "+", 1]], ids=["set", "str", "number"])
    def test_non_token_list_raises(self, tokens):
        with pytest.raises(TypeError, match="^tokens must"):
            Expression(tokens, self.sl)

    def test_none_library_is_fixed_when_built(self):
        SymbolLibrary.set_default(None)
        try:
            with self.sl:
                expr = Expression(["X_1"])
                from_tree = Expression.from_tree(Node("X_1"))
            assert expr.symbol_library is self.sl
            assert expr.variables == ("X_1",)
            assert from_tree.symbol_library is self.sl
        finally:
            SymbolLibrary.set_default(None)

    def test_from_tree_writes_tokens_and_copies_tree(self):
        tree = Node("*", Node("+", Node("X_1"), Node("X_0")), Node("C"))
        expr = Expression.from_tree(tree, self.sl)
        assert list(expr.tokens) == tree.to_list(self.sl)
        assert expr.tree is not tree
        tree.symbol = "-"
        tree.right.symbol = "X_1"
        assert expr.tree.symbol == "*"
        assert expr.tree.to_list(self.sl) == list(expr.tokens)

    def test_from_tree_writes_tokens_with_the_library(self):
        expr = Expression.from_tree(Node("^2", None, Node("X_0")), self.sl)
        assert expr.tokens == ("X_0", "^2")
        assert expr.parses and expr.tree.symbol == "^2"

    @pytest.mark.parametrize(
        "tree", [Node("+", None, Node("X_0")), Node("sin", Node("X_0"), None)], ids=["op-one-operand", "fn-right-child"]
    )
    def test_from_malformed_tree_is_like_its_tokens(self, tree):
        expr = Expression.from_tree(tree, self.sl)
        assert expr.parses is False
        assert expr == Expression(expr.tokens, self.sl)

    def test_from_tree_with_unknown_symbol_raises_parse_error(self):
        with pytest.raises(ExpressionParseError):
            Expression.from_tree(Node("+", Node("X_0"), Node("zz")), self.sl)

    @pytest.mark.parametrize("tree", [["X_0"], "X_0", None], ids=["list", "str", "none"])
    def test_from_tree_with_non_node_raises(self, tree):
        with pytest.raises(TypeError, match=f"^tree must be a Node, got {type(tree).__name__}"):
            Expression.from_tree(tree, self.sl)

    def test_well_formed_expression_parses(self):
        expr = Expression(["sin", "(", "X_0", ")", "*", "C"], self.sl)
        assert expr.parses is True
        assert expr.parse_error is None
        assert expr.tree.to_list(self.sl) == list(expr.tokens)

    def test_malformed_expression_is_built_but_does_not_parse(self):
        expr = Expression(["C", "*", "X_0", "+"], self.sl)
        assert expr.tokens == ("C", "*", "X_0", "+")
        # Read from the tokens, without parsing.
        assert expr.variables == ("X_0",)
        assert expr.parameter_labels == (None,)
        assert expr.parses is False
        assert isinstance(expr.parse_error, ExpressionParseError)
        for _ in range(2):
            with pytest.raises(ExpressionParseError):
                expr.tree

    @pytest.mark.parametrize("tokens", [["X_0", "+", "C"], ["X_0", "+"]], ids=["valid", "malformed"])
    def test_tokens_are_parsed_once(self, tokens, monkeypatch):
        calls = []
        parse = expression.tokens_to_tree

        def counting(*args, **kwargs):
            calls.append(args)
            return parse(*args, **kwargs)

        monkeypatch.setattr(expression, "tokens_to_tree", counting)
        expr = Expression(tokens, self.sl)
        first = expr.tree if expr.parses else None
        for _ in range(3):
            assert (expr.tree if expr.parses else None) is first
            expr.parse_error
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "tokens",
        [
            ["X_0", "+", "X_1"],
            ["C", "*", "X_0", "+", "C"],
            ["C_1", "*", "X_0", "+", "C", "*", "C_1", "+", "C_0"],
        ],
        ids=["none", "bare", "labelled"],
    )
    def test_parameters_match_parameter_labels(self, tokens):
        expr = Expression(tokens, self.sl)
        expected = parameter_labels(tokens, self.sl)
        assert expr.parameter_labels == tuple(expected)
        assert expr.num_parameters == len(expected)

    def test_variables_once_in_order_of_first_appearance(self):
        expr = Expression(["X_1", "*", "sin", "(", "X_0", ")", "+", "X_1", "*", "C", "+", "2", "+", "pi"], self.sl)
        assert expr.variables == ("X_1", "X_0")
        assert Expression(["C", "+", "1"], self.sl).variables == ()

    def test_variables_follow_the_library(self):
        tokens = ["X_0", "+", "X_1"]
        assert Expression(tokens, SymbolLibrary.default_symbols(1)).variables == ("X_0",)
        assert Expression(tokens, self.sl).variables == ("X_0", "X_1")

    def test_to_latex(self):
        expr = Expression(["C", "*", "X_0", "/", "X_1"], self.sl)
        assert expr.to_latex() == expr.tree.to_latex(self.sl)
        with pytest.raises(ExpressionParseError):
            Expression(["X_0", "+"], self.sl).to_latex()

    def test_equality_and_hash_use_the_tokens(self):
        a = Expression(["X_0", "+", "C"], self.sl)
        b = Expression(["X_0", "+", "C"], SymbolLibrary.default_symbols(1))
        assert a == b
        assert hash(a) == hash(b)
        assert a != Expression(["C", "+", "X_0"], self.sl)
        assert a != ["X_0", "+", "C"]
        assert len({a, b}) == 1

    def test_parse_error_does_not_carry_an_earlier_context(self):
        expr = Expression(["X_0", "+"], self.sl)
        try:
            raise KeyError("unrelated")
        except KeyError:
            assert expr.parses is False
        with pytest.raises(ExpressionParseError) as info:
            expr.tree
        assert info.value.__context__ is None

    def test_str_and_repr(self):
        expr = Expression(["X_0", "+", "C"], self.sl)
        assert str(expr) == "X_0+C"
        assert repr(expr) == "Expression(['X_0', '+', 'C'])"

    def test_pickle_keeps_tokens_and_library(self):
        expr = Expression.from_tree(Node("+", Node("X_0"), Node("C")), self.sl)
        expr.tree
        copied = pickle.loads(pickle.dumps(expr))
        assert copied == expr
        assert copied.symbol_library.symbols.keys() == self.sl.symbols.keys()
        assert copied.tree.to_list(self.sl) == list(expr.tokens)
        # The cached facts are not pickled: a parsed expression pickles like an unparsed one.
        assert pickle.dumps(expr) == pickle.dumps(Expression(list(expr.tokens), self.sl))

    def test_pickled_subclass_stays_a_subclass(self):
        copied = pickle.loads(pickle.dumps(_Tagged(["X_0"], self.sl)))
        assert type(copied) is _Tagged


class TestExpressionSpecs:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(2)
        self.tokens = ["C_0", "*", "X_0", "+", "C_1", "*", "C"]
        self.g = Constant(label="C_0", value=9.81, fixed=True)
        self.h = Constant(label="C_1", bounds=(0.0, 1.0))

    @pytest.mark.parametrize("given", [None, []], ids=["none", "empty"])
    def test_no_specs_give_empty_tuple(self, given):
        assert Expression(self.tokens, self.sl, given).free_constants == ()

    def test_specs_are_kept_as_a_tuple(self):
        specs = [self.g, self.h]
        expr = Expression(self.tokens, self.sl, specs)
        specs.pop()
        assert expr.free_constants == (self.g, self.h)

    @pytest.mark.parametrize(
        "specs, match",
        [
            ([Constant(value=1.0, fixed=True)], "needs a label"),
            ([Constant(label="C_0"), Constant(label="C_0", value=1.0)], "share the label"),
            ([Constant(label="C_7")], "not labelled constants"),
        ],
        ids=["no-label", "duplicate", "unknown"],
    )
    def test_specs_that_do_not_fit_raise(self, specs, match):
        with pytest.raises(ValueError, match=match):
            Expression(self.tokens, self.sl, specs)

    @pytest.mark.parametrize(
        "specs, match",
        [
            (Constant(label="C_0"), "sequence of Constant, got Constant"),
            ({"label": "C_0"}, "sequence of Constant, got dict"),
            ([{"label": "C_0"}], r"contains dict\. Build specs from dicts with Constant\.from_dict\.$"),
            (["C_0"], r"contains str\.$"),
        ],
        ids=["bare-constant", "dict", "dict-element", "str-element"],
    )
    def test_specs_that_are_not_constants_raise(self, specs, match):
        with pytest.raises(TypeError, match=match):
            Expression(self.tokens, self.sl, specs)

    def test_specs_are_checked_without_parsing(self, monkeypatch):
        monkeypatch.setattr(expression, "tokens_to_tree", pytest.fail)
        malformed = ["C_0", "*", "X_0", "+"]
        assert Expression(malformed, self.sl, [self.g]).free_constants == (self.g,)
        with pytest.raises(ValueError):
            Expression(malformed, self.sl, [self.h])

    def test_from_tree_keeps_and_checks_specs(self):
        tree = Node("*", Node("X_0"), Node("C_0"))
        assert Expression.from_tree(tree, self.sl, [self.g]).free_constants == (self.g,)
        with pytest.raises(ValueError):
            Expression.from_tree(tree, self.sl, [self.h])

    def test_equality_and_hash_include_the_specs(self):
        a = Expression(self.tokens, self.sl, [self.g, self.h])
        b = Expression(self.tokens, SymbolLibrary.default_symbols(1), [self.h, self.g])
        assert a == b
        assert hash(a) == hash(b)
        assert a != Expression(self.tokens, self.sl, [self.g])
        assert a != Expression(self.tokens, self.sl)
        assert Expression(self.tokens, self.sl, [self.g]) != Expression(
            self.tokens, self.sl, [Constant(label="C_0", value=9.8, fixed=True)]
        )

    def test_resolve_matches_resolve_constants(self):
        default = uniform_default((-5.0, 5.0))
        expr = Expression(self.tokens, self.sl, [self.g, self.h])
        assert expr.resolve(default) == resolve_constants(self.tokens, [self.g, self.h], default, self.sl)
        assert Expression(self.tokens, self.sl).resolve(default) == (default, default, default)

    def test_resolve_uses_the_expression_library(self):
        sl = SymbolLibrary.default_symbols(1, const_symbol="K")
        spec = Constant(label="K_0", value=1.0, fixed=True)
        resolved = Expression(["K_0", "*", "X_0", "+", "K"], sl, [spec]).resolve(uniform_default((-5.0, 5.0)))
        assert [s.label for s in resolved] == ["K_0", None]

    def test_resolve_rejects_value_outside_inherited_bounds(self):
        expr = Expression(self.tokens, self.sl, [Constant(label="C_0", value=9.0)])
        with pytest.raises(ValueError):
            expr.resolve(uniform_default((-5.0, 5.0)))

    def test_repr_shows_specs_only_when_given(self):
        assert repr(Expression(["C_0"], self.sl, [self.g])) == f"Expression(['C_0'], free_constants=({self.g!r},))"
        assert repr(Expression(["C_0"], self.sl)) == "Expression(['C_0'])"

    def test_pickle_keeps_specs(self):
        expr = Expression(self.tokens, self.sl, [self.g, self.h])
        copied = pickle.loads(pickle.dumps(expr))
        assert copied == expr
        assert copied.free_constants == (self.g, self.h)


class TestAsExpression:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(2)

    @pytest.mark.parametrize("library", ["same", None], ids=["same", "none"])
    def test_expression_with_same_or_no_library_is_returned(self, library):
        expr = Expression(["X_0"], self.sl)
        assert _as_expression(expr, self.sl if library == "same" else None) is expr

    def test_expression_with_other_library_is_rebuilt(self):
        other = SymbolLibrary.default_symbols(1)
        expr = Expression(["X_0", "+", "X_1"], self.sl)
        rebuilt = _as_expression(expr, other)
        assert rebuilt.symbol_library is other
        assert rebuilt.tokens == expr.tokens
        assert rebuilt.variables == ("X_0",)

    def test_tuple_is_wrapped(self):
        expr = _as_expression(("X_0", "*", "C"), self.sl)
        assert expr == Expression(["X_0", "*", "C"], self.sl)

    def test_list_is_wrapped(self):
        expr = _as_expression(["X_0", "*", "C"], self.sl)
        assert isinstance(expr, Expression)
        assert expr.tokens == ("X_0", "*", "C")
        assert expr.symbol_library is self.sl

    def test_node_raises_pointing_to_from_tree(self):
        with pytest.raises(TypeError, match="from_tree"):
            _as_expression(Node("X_0"), self.sl)

    def test_other_inputs_raise_naming_the_parameter(self):
        with pytest.raises(TypeError, match="^expr must be .* got set"):
            _as_expression({"X_0"}, self.sl)
        with pytest.raises(TypeError, match="^expr1 must be"):
            _as_expression("X_0", self.sl, "expr1")
        with pytest.raises(TypeError, match="expr1.*'1.5'"):
            _as_expression(["X_0", "+", 1.5], self.sl, "expr1")

    def test_rebuilt_expression_keeps_its_specs(self):
        spec = Constant(label="C_0", value=1.0, fixed=True)
        expr = Expression(["C_0", "*", "X_0"], self.sl, [spec])
        rebuilt = _as_expression(expr, SymbolLibrary.default_symbols(1))
        assert rebuilt.free_constants == (spec,)
        with pytest.raises(ValueError, match="not labelled constants"):
            _as_expression(expr, SymbolLibrary.default_symbols(1, const_symbol=None))

    def test_list_has_no_specs(self):
        assert _as_expression(["C_0", "*", "X_0"], self.sl).free_constants == ()
