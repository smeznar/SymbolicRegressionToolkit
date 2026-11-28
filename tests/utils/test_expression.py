import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from strategies import infix_expressions, labelled_infix_expressions

import SRToolkit.utils.expression as expression
from SRToolkit.dataset import SR_dataset, UniformSampling
from SRToolkit.evaluation.parameter_estimator import ParameterEstimator
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.expression import Expression, _as_expression, canonicalise
from SRToolkit.utils.expression_compiler import compile_expr, compile_expr_rmse
from SRToolkit.utils.expression_constraints import MaxTreeDepth
from SRToolkit.utils.expression_simplifier import simplify
from SRToolkit.utils.expression_tree import ExpressionParseError, Node
from SRToolkit.utils.free_constants import Constant, uniform_default
from SRToolkit.utils.measures import bed, edit_distance
from SRToolkit.utils.symbol_library import SymbolLibrary

_X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]])
_Y = np.array([1.0, 2.0, 3.0])


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
        "tokens, labels",
        [
            (["C", "*", "X_0", "+", "C_5", "*", "X_1", "+", "C"], (None, "C_5", None)),
            (["C_5", "+", "C_1", "*", "C_5", "+", "C_01"], ("C_5", "C_1", "C_01")),
            (["C", "+", "C"], (None, None)),
            (["X_0", "+", "pi"], ()),
        ],
        ids=["mixed", "labels-by-first-appearance", "bare-each-own", "none"],
    )
    def test_parameter_labels(self, tokens, labels):
        expr = Expression(tokens, self.sl)
        assert expr.parameter_labels == labels
        assert expr.num_parameters == len(labels)

    def test_registered_symbol_named_like_a_label_is_not_a_parameter(self):
        sl = SymbolLibrary.default_symbols(1)
        sl.add_symbol("C_1", "lit", 1000, "1.5")
        assert Expression(["C_1", "+", "C", "+", "C_2"], sl).parameter_labels == (None, "C_2")

    def test_parameter_labels_in_a_renamed_constant_library(self):
        sl = SymbolLibrary.default_symbols(2, const_symbol="K")
        assert Expression(["K", "*", "X_0", "+", "K_1", "*", "X_1", "+", "K_1"], sl).parameter_labels == (None, "K_1")

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

    # Reason: clause (public functions take a token list or an Expression; a Node raises TypeError
    # pointing to Expression.from_tree, also where it reaches _as_expression through other checks).
    @pytest.mark.parametrize(
        "call",
        [
            lambda node, sl, se: compile_expr(node, sl),
            lambda node, sl, se: compile_expr_rmse(node, sl),
            lambda node, sl, se: se.evaluate_expr(node),
            lambda node, sl, se: se.check_constraints(node),
            lambda node, sl, se: se.get_model(node),
            lambda node, sl, se: se.stored_tokens(node),
            lambda node, sl, se: ParameterEstimator(_X, _Y, symbol_library=sl).estimate_parameters(node),
            lambda node, sl, se: (c := MaxTreeDepth(3), c.setup(sl), c.pre_check(node)),
            lambda node, sl, se: simplify(node, sl),
            lambda node, sl, se: edit_distance(node, ["X_0"], symbol_library=sl),
            lambda node, sl, se: bed(node, ["X_0"], _X, symbol_library=sl),
            lambda node, sl, se: SR_dataset(_X, sl, y=_Y, ground_truth=node),
        ],
        ids=[
            "compile_expr",
            "compile_expr_rmse",
            "evaluate_expr",
            "check_constraints",
            "get_model",
            "stored_tokens",
            "estimate_parameters",
            "pre_check",
            "simplify",
            "edit_distance",
            "bed",
            "dataset-ground-truth",
        ],
    )
    def test_public_entry_points_reject_a_node(self, call):
        se = SR_evaluator(_X, _Y, symbol_library=self.sl)
        with pytest.raises(TypeError, match="Expression.from_tree"):
            call(Node("X_0"), self.sl, se)

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


INF = float("inf")


# ---------------------------------------------------------------------------------------------------
# Expression.resolve
# ---------------------------------------------------------------------------------------------------


def _lit_c1_library():
    """Default library where ``C_1`` is a registered named constant, not a labelled constant (D17)."""
    sl = SymbolLibrary.default_symbols(1)
    sl.add_symbol("C_1", "lit", 5, "1.0")
    return sl


class TestExpressionResolve:
    @pytest.mark.parametrize("specs", [None, ()], ids=["none", "empty"])
    def test_without_specs_gives_default_itself(self, specs):
        default = uniform_default((-5.0, 5.0))
        res = Expression(["C", "*", "X_0", "+", "C_1"], free_constants=specs).resolve(default)
        assert len(res) == 2
        assert [s is default for s in res] == [True, True]

    def test_expression_without_constants_is_empty(self):
        assert Expression(["X_0", "+", "pi"]).resolve(uniform_default((-5.0, 5.0))) == ()

    def test_maps_specs_to_parameter_slots(self):
        default = uniform_default((-5.0, 5.0))
        expr = ["C_2", "*", "X_0", "+", "C", "*", "C_0", "+", "C_2", "+", "C_7"]
        specs = [Constant(label="C_0", value=1.0), Constant(label="C_2", value=3.0, fixed=True)]
        res = Expression(expr, free_constants=specs).resolve(default)
        # parameters by first appearance: C_2, bare C, C_0, C_7 (C_7 has no spec)
        assert [s.label for s in res] == ["C_2", None, "C_0", None]
        assert [s.value for s in res] == [3.0, None, 1.0, None]
        assert [s.fixed for s in res] == [True, False, False, False]
        assert res[1] is default
        assert res[3] is default

    def test_fills_missing_prior_and_bounds_from_default(self):
        default = uniform_default((-5.0, 5.0))
        own_prior = UniformSampling(0.0, 1.0)
        specs = [
            Constant(label="C_0"),
            Constant(label="C_1", prior=own_prior),
            Constant(label="C_2", bounds=(0.0, INF)),
        ]
        c0, c1, c2 = Expression(["C_0", "+", "C_1", "+", "C_2"], free_constants=specs).resolve(default)
        assert c0.prior is default.prior
        assert c0.bounds == (-5.0, 5.0)
        assert c1.prior is own_prior
        assert c1.bounds == (-5.0, 5.0)
        assert c2.prior is default.prior
        assert c2.bounds == (0.0, INF)

    @pytest.mark.parametrize(
        "value, fixed",
        [(9.81, True), (5.0, False), (-5.0, False)],
        ids=["fixed_outside", "free_on_upper", "free_on_lower"],
    )
    def test_accepts_values_against_inherited_bounds(self, value, fixed):
        spec = Constant(label="C_0", value=value, fixed=fixed)
        (res,) = Expression(["C_0", "*", "X_0"], free_constants=[spec]).resolve(uniform_default((-5.0, 5.0)))
        assert res.value == value
        assert res.bounds == (-5.0, 5.0)

    @pytest.mark.parametrize(
        "expr, specs, default, symbol_library, exc, match",
        [
            (
                ["C_0"],
                None,
                dict(label="C_0", prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0)),
                None,
                ValueError,
                "default",
            ),
            (["C_0"], None, dict(bounds=(-1.0, 1.0)), None, ValueError, "default"),
            (["C_0"], None, dict(prior=UniformSampling(-1.0, 1.0)), None, ValueError, "default"),
            (
                ["C_0"],
                None,
                dict(prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0), value=0.5),
                None,
                ValueError,
                "default_constant",
            ),
            (
                ["C_0"],
                None,
                dict(prior=UniformSampling(-1.0, 1.0), bounds=(-1.0, 1.0), value=0.5, fixed=True),
                None,
                ValueError,
                "default_constant",
            ),
            (["C_0"], [dict(value=1.0)], None, None, ValueError, "label"),
            (["C_0"], [dict(label="C_0"), dict(label="C_0", value=1.0)], None, None, ValueError, "C_0"),
            (["C_0"], [dict(label="C_9")], None, None, ValueError, "C_9"),
            (["C", "+", "X_0"], [dict(label="C")], None, None, ValueError, "label"),
            (["C", "+", "X_0"], [dict(label="X_0")], None, None, ValueError, "X_0"),
            (["C_0"], [dict(label="C_0", value=6.0)], None, None, ValueError, "value|bounds"),
            (["C_1", "+", "C_0"], [dict(label="C_1")], None, _lit_c1_library(), ValueError, "C_1"),
        ],
        ids=[
            "default_has_label",
            "default_without_prior",
            "default_without_bounds",
            "default_with_value",
            "default_fixed",
            "spec_without_label",
            "duplicate_label",
            "label_not_in_expr",
            "bare_C_label",
            "variable_label",
            "free_value_outside_inherited_bounds",
            "lit_symbol_is_not_labelled_constant",
        ],
    )
    def test_rejects_invalid_input(self, expr, specs, default, symbol_library, exc, match):
        # specs and default are given as Constant kwargs; default None means uniform_default((-5, 5))
        specs = None if specs is None else [Constant(**kw) for kw in specs]
        default = uniform_default((-5.0, 5.0)) if default is None else Constant(**default)
        with pytest.raises(exc, match=match):
            Expression(expr, symbol_library, specs).resolve(default)


# ---------------------------------------------------------------------------------------------------
# canonicalise
# ---------------------------------------------------------------------------------------------------

_X_CANON = np.random.default_rng(0).uniform(0.5, 2.0, size=(5, 2))


def _relabel(tokens):
    """Rename the labels of ``labelled_infix_expressions`` by a fixed bijection."""
    renaming = {"C_0": "C_7", "C_1": "C_0", "C_7": "C_1"}
    return [renaming.get(t, t) for t in tokens]


def _canonicalise(*args):
    """The canonical tokens as a list, so results compare with plain lists."""
    return list(canonicalise(*args).tokens)


class TestCanonicalise:
    # Reason: clause (once -> bare, shared -> C_<k> with k from the parameter order, skipping
    # registered names, const symbol of the library) + boundary (empty, no constants, no const symbol,
    # malformed tokens) + implied (input not mutated).
    @pytest.mark.parametrize(
        "lib, expr, expected",
        [
            (None, "C_3 + X_0", "C + X_0"),
            (None, "C_2 * X_0 + C_9 + C_9 + C_2", "C_0 * X_0 + C_1 + C_1 + C_0"),
            # positions count bare and once-occurring constants: C_8 is parameter 2
            (None, "C_4 + C + C_8 * C_8", "C + C + C_2 * C_2"),
            # renaming is simultaneous: C_1 -> C_0 and C_0 -> C_1 do not collide
            (None, "C_1 * C_1 + C_0 * C_0", "C_0 * C_0 + C_1 * C_1"),
            (None, "X_0 + pi", "X_0 + pi"),
            (None, "", ""),
            (
                lambda: SymbolLibrary.default_symbols(1, const_symbol="K"),
                "K_3 * X_0 + K_3 + K_5",
                "K_0 * X_0 + K_0 + K",
            ),
            # D17: the named constant C_1 is not a parameter; C_5 is parameter 0
            (_lit_c1_library, "C_1 + C_5 * X_0 + C_5", "C_1 + C_0 * X_0 + C_0"),
            # D17: C_5 is parameter 1, but C_1 is registered, so it gets C_2
            (_lit_c1_library, "C_7 * C_7 + C_5 * X_0 + C_5", "C_0 * C_0 + C_2 * X_0 + C_2"),
            # labels are handed out per parameter, bare ones included: the bare C takes C_2
            (_lit_c1_library, "C_7 * C_7 + C + C_5 * C_5", "C_0 * C_0 + C + C_3 * C_3"),
            (lambda: SymbolLibrary.from_symbol_list(["+"], 1), "C_0 + C_0", "C_0 + C_0"),
            # tokens only: malformed input is rewritten without parsing
            (None, "C_3 * ( C_3", "C_0 * ( C_0"),
        ],
        ids=[
            "once",
            "renumber",
            "positions-count-bare",
            "swap",
            "no-constants",
            "empty",
            "const-symbol-K",
            "D17-registered-untouched",
            "D17-skip",
            "D17-skip-counts-bare",
            "no-const-symbol",
            "malformed",
        ],
    )
    def test_canonical_form(self, lib, expr, expected):
        tokens = expr.split()
        given_tokens = list(tokens)
        assert _canonicalise(tokens, None if lib is None else lib()) == expected.split()
        assert tokens == given_tokens

    # Reason: clause (an Expression is read in its own library).
    def test_expression_input_uses_its_library(self):
        sl = SymbolLibrary.default_symbols(1, const_symbol="K")
        assert _canonicalise(Expression(["K_5", "+", "K_5", "+", "K_1"], sl)) == ["K_0", "+", "K_0", "+", "K"]

    # Reason: clause (Returns: an Expression in the library of expr; one that does not change is
    # returned itself).
    def test_returns_an_expression_and_an_unchanged_one_itself(self):
        sl = SymbolLibrary.default_symbols(1, const_symbol="K")
        unchanged = Expression(["K", "+", "X_0"], sl)
        assert canonicalise(unchanged) is unchanged
        changed = canonicalise(Expression(["K_3", "+", "X_0"], sl))
        assert (changed.tokens, changed.symbol_library) == (("K", "+", "X_0"), sl)

    # Reason: clause (Raises ValueError for an expression with specs).
    def test_rejects_expression_with_specs(self):
        expr = Expression(["C_0", "+", "C_0"], free_constants=[Constant(label="C_0", value=1.0, fixed=True)])
        with pytest.raises(ValueError, match="specs"):
            _canonicalise(expr)

    # Reason: clause (Raises TypeError).
    @pytest.mark.parametrize("expr", ["C_0+C_0", None, [1, "+"]], ids=["str", "none", "non-str-token"])
    def test_rejects_non_expression(self, expr):
        with pytest.raises(TypeError, match="token strings"):
            _canonicalise(expr)

    # Reason: implied (only names change): idempotent, same parameter count, same function of the
    # same parameter vector.
    @settings(max_examples=60)
    @given(expr=labelled_infix_expressions())
    def test_canonical_form_is_idempotent_and_computes_the_same_function(self, expr):
        canon = _canonicalise(expr)
        assert _canonicalise(canon) == canon
        n = Expression(expr).num_parameters
        assert Expression(canon).num_parameters == n
        params = np.random.default_rng(1).uniform(0.5, 2.0, size=n)
        with np.errstate(all="ignore"):
            np.testing.assert_array_equal(compile_expr(canon)(_X_CANON, params), compile_expr(expr)(_X_CANON, params))

    # Reason: clause ("expressions that differ only in how their constants are written then have one form").
    @settings(max_examples=60)
    @given(expr=labelled_infix_expressions())
    def test_invariant_under_relabelling(self, expr):
        assert _canonicalise(_relabel(expr)) == _canonicalise(expr)

    # Reason: clause ("an expression with only bare constants is unchanged").
    @settings(max_examples=60)
    @given(expr=infix_expressions())
    def test_bare_only_expression_unchanged(self, expr):
        assert _canonicalise(expr) == expr
