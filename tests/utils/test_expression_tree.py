import re
from copy import copy

import pytest

from SRToolkit.utils.expression_tree import ExpressionParseError, Node, expr_to_latex, is_float, tokens_to_tree
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestTokensToTree:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=3)

    def test_simple_binary_op(self):
        tree = tokens_to_tree(["X_0", "+", "X_1"], self.sl)
        assert tree.symbol == "+"
        assert len(tree) == 3

    def test_operator_precedence(self):
        # X_0 + X_1 * X_2 → + at root, * as right child
        tree = tokens_to_tree(["X_0", "+", "X_1", "*", "X_2"], self.sl)
        assert tree.symbol == "+"
        assert tree.right.symbol == "*"

    def test_prefix_fn(self):
        tree = tokens_to_tree(["sin", "(", "X_0", ")"], self.sl)
        assert tree.symbol == "sin"
        assert len(tree) == 2

    def test_postfix_fn(self):
        tree = tokens_to_tree(["X_0", "^2"], self.sl)
        assert tree.symbol == "^2"
        assert tree.left.symbol == "X_0"

    def test_float_token_treated_as_leaf(self):
        tree = tokens_to_tree(["X_0", "+", "1.5"], self.sl)
        assert tree.symbol == "+"
        assert len(tree) == 3

    def test_fn_applied_after_closing_paren(self):
        # tan(X_1 - 5.2) exercises the fn-on-close-paren branch
        tokens = ["(", "X_0", "+", "tan", "(", "X_1", "-", "5.2", ")", ")"]
        tree = tokens_to_tree(tokens, self.sl)
        assert tree.symbol == "+"
        assert tree.right.symbol == "tan"

    def test_op_pops_higher_precedence_op_from_stack(self):
        # X_0 * X_1 + X_2: when + arrives, * is popped (line 371)
        tree = tokens_to_tree(["X_0", "*", "X_1", "+", "X_2"], self.sl)
        assert tree.symbol == "+"
        assert tree.left.symbol == "*"

    def test_fn_on_stack_popped_by_lower_precedence_op(self):
        # sin X_0 + X_1 (no parens): when + arrives, sin is popped from operator stack (lines 368-369)
        tree = tokens_to_tree(["sin", "X_0", "+", "X_1"], self.sl)
        assert tree.symbol == "+"
        assert tree.left.symbol == "sin"

    def test_fn_popped_in_closing_while_loop(self):
        # (sin X_0): paren comes before fn, so sin sits above ( on the stack
        # and is popped by the while-loop fn branch when ) is encountered (line 381)
        tree = tokens_to_tree(["(", "sin", "X_0", ")"], self.sl)
        assert tree.symbol == "sin"
        assert tree.left.symbol == "X_0"

    def test_postfix_fn_on_subexpression(self):
        # (X_0 + X_1)^2: ^2 wraps the compound node already on the output stack
        tree = tokens_to_tree(["(", "X_0", "+", "X_1", ")", "^2"], self.sl)
        assert tree.symbol == "^2"
        assert tree.left.symbol == "+"
        assert len(tree) == 4

    def test_const_token_treated_as_leaf(self):
        tree = tokens_to_tree(["C", "+", "X_0"], self.sl)
        assert tree.symbol == "+"
        assert len(tree) == 3

    def test_invalid_token_raises(self):
        with pytest.raises(Exception, match="Invalid symbol"):
            tokens_to_tree(["X_0", "???", "X_1"], self.sl)

    def test_malformed_expression_raises(self):
        # Two leaves with no operator produce disconnected output nodes
        with pytest.raises(Exception, match="Error while parsing"):
            tokens_to_tree(["X_0", "X_1"], self.sl)

    @pytest.mark.parametrize(
        "tokens, reason",
        [
            ("( X_0 + X_1", "unclosed '('"),
            ("( ( X_0", "unclosed '('"),
            ("X_0 + X_1 )", "unmatched ')'"),
            (") X_0 (", "missing operand before ')'"),
            ("( ) X_0", "missing operand before ')'"),
            ("X_0 + ( )", "missing operand before ')'"),
            ("X_0 ( ) + X_1", "'(' cannot follow an operand"),
            ("X_0 X_1", "operand 'X_1' cannot follow an operand"),
            ("X_0 sin ( X_1 )", "function 'sin' cannot follow an operand"),
            ("^2 X_0", "function '^2' is missing an operand"),
            ("+ X_0", "operator '+' is missing an operand"),
            ("X_0 +", "missing operand at the end"),
            ("sin", "missing operand at the end"),
            ("", "the expression is empty"),
        ],
    )
    def test_malformed_infix_raises_parse_error(self, tokens, reason):
        with pytest.raises(ExpressionParseError, match=re.escape(reason)):
            tokens_to_tree(tokens.split(), self.sl)

    def test_parse_error_is_value_error(self):
        with pytest.raises(ValueError):
            tokens_to_tree(["X_0", "???"], self.sl)

    @pytest.mark.parametrize("notation", ["prefix", "postfix"])
    def test_prefix_postfix_errors_are_parse_errors(self, notation):
        with pytest.raises(ExpressionParseError):
            tokens_to_tree(["X_0", "X_1"], self.sl, notation=notation)
        with pytest.raises(ExpressionParseError):
            tokens_to_tree(["+"], self.sl, notation=notation)

    def test_prefix_fn_without_brackets_binds_to_next_operand(self):
        # u- X_0 * X_1 is (-X_0) * X_1, as in the bundled Feynman ground truths
        tree = tokens_to_tree(["u-", "X_0", "*", "X_1"], self.sl)
        assert tree.symbol == "*"
        assert tree.left.symbol == "u-"
        # A postfix function binds tighter: u- X_0 ^2 is -(X_0^2)
        tree = tokens_to_tree(["u-", "X_0", "^2"], self.sl)
        assert tree.symbol == "u-"
        assert tree.left.symbol == "^2"

    def test_redundant_brackets_accepted(self):
        tree = tokens_to_tree(["(", "(", "X_0", ")", ")", "+", "(", "X_1", ")"], self.sl)
        assert tree.to_list(self.sl) == ["X_0", "+", "X_1"]

    def test_round_trip_of_to_list(self):
        tokens = ["sin", "(", "X_0", ")", "*", "cos", "(", "X_1", "+", "C", ")", "-", "2.5", "/", "X_0", "^", "X_2"]
        tree = tokens_to_tree(tokens, self.sl)
        assert tokens_to_tree(tree.to_list(self.sl), self.sl).to_list(self.sl) == tree.to_list(self.sl)

    @pytest.mark.parametrize("notation", ["infix", "prefix", "postfix"])
    def test_nan_literal_rejected(self, notation):
        tokens = {"infix": ["X_0", "+", "nan"], "prefix": ["+", "X_0", "nan"], "postfix": ["X_0", "nan", "+"]}
        with pytest.raises(ExpressionParseError, match="'nan' is not a number"):
            tokens_to_tree(tokens[notation], self.sl, notation=notation)

    def test_infinite_literals_accepted(self):
        tree = tokens_to_tree(["X_0", "^", "-inf"], self.sl)
        assert tree.right.symbol == "-inf"

    def test_left_associative_power(self):
        tree = tokens_to_tree(["X_0", "^", "X_1", "^", "X_2"], self.sl)
        assert tree.left.symbol == "^"

    def test_postfix_detected_by_precedence(self):
        # A postfix function whose name does not start with "^"
        self.sl.add_symbol("!", "fn", -1000, "{}", "{}!")
        tree = tokens_to_tree(["X_0", "!", "+", "X_1"], self.sl)
        assert tree.symbol == "+"
        assert tree.left.symbol == "!"
        assert tree.to_list(self.sl) == ["X_0", "!", "+", "X_1"]

    def test_prefix_fn_binds_tightly_whatever_its_precedence(self):
        # sin registered with precedence 5, below every operator
        self.sl.add_symbol("f", "fn", 5, "np.sin({})", r"f {}")
        tree = tokens_to_tree(["f", "X_0", "+", "X_1"], self.sl)
        assert tree.symbol == "+"
        assert tree.left.symbol == "f"


class TestNodeInit:
    def test_stores_symbol_and_children(self):
        left = Node("X_0")
        right = Node("1")
        node = Node("+", right=right, left=left)
        assert node.symbol == "+"
        assert node.left is left
        assert node.right is right

    def test_default_children_are_none(self):
        node = Node("X_0")
        assert node.left is None
        assert node.right is None


class TestNodeToList:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=3)

    def test_prefix_notation(self):
        node = Node("+", right=Node("X_0"), left=Node("X_1"))
        assert node.to_list(notation="prefix") == ["+", "X_1", "X_0"]

    def test_postfix_notation(self):
        node = Node("+", right=Node("X_0"), left=Node("X_1"))
        assert node.to_list(notation="postfix") == ["X_1", "X_0", "+"]

    def test_invalid_notation_raises(self):
        with pytest.raises(Exception, match="Invalid notation"):
            Node("X_0").to_list(notation="rpn")

    def test_infix_no_library_leaf(self):
        with pytest.warns(UserWarning):
            result = Node("X_0").to_list()
        assert result == ["X_0"]

    def test_infix_no_library_unary_hat(self):
        node = Node("^2", right=None, left=Node("X_0"))
        with pytest.warns(UserWarning):
            result = node.to_list()
        assert result == ["(", "X_0", ")", "^2"]

    def test_infix_no_library_unary_other(self):
        node = Node("sin", right=None, left=Node("X_0"))
        with pytest.warns(UserWarning):
            result = node.to_list()
        assert result == ["sin", "(", "X_0", ")"]

    def test_infix_no_library_binary_wraps_complex_children(self):
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        outer = Node("*", right=inner, left=Node("X_2"))
        with pytest.warns(UserWarning):
            result = outer.to_list()
        assert result == ["X_2", "*", "(", "X_1", "+", "X_0", ")"]

    def test_infix_with_library_float_and_leaf_tokens(self):
        assert Node("1.5").to_list(self.sl) == ["1.5"]
        assert Node("X_0").to_list(self.sl) == ["X_0"]
        assert Node("C").to_list(self.sl) == ["C"]
        assert Node("pi").to_list(self.sl) == ["pi"]

    def test_infix_with_library_fn_positive_precedence(self):
        node = Node("sin", right=None, left=Node("X_0"))
        assert node.to_list(self.sl) == ["sin", "(", "X_0", ")"]

    def test_infix_with_library_fn_negative_precedence(self):
        assert Node("^2", right=None, left=Node("X_0")).to_list(self.sl) == ["X_0", "^2"]
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        node = Node("^2", right=None, left=inner)
        assert node.to_list(self.sl) == ["(", "X_1", "+", "X_0", ")", "^2"]

    def test_infix_no_library_binary_wraps_complex_left_child(self):
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        outer = Node("*", right=Node("X_2"), left=inner)
        with pytest.warns(UserWarning):
            result = outer.to_list()
        assert result == ["(", "X_1", "+", "X_0", ")", "*", "X_2"]

    def test_infix_with_library_op_parenthesises_lower_precedence_children(self):
        # right child lower precedence → right wrapped
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        outer = Node("*", right=inner, left=Node("X_2"))
        assert outer.to_list(self.sl) == ["X_2", "*", "(", "X_1", "+", "X_0", ")"]

    def test_infix_with_library_op_parenthesises_lower_precedence_left_child(self):
        # left child lower precedence → left wrapped
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        outer = Node("*", right=Node("X_2"), left=inner)
        assert outer.to_list(self.sl) == ["(", "X_1", "+", "X_0", ")", "*", "X_2"]

    def test_infix_with_library_invalid_symbol_raises(self):
        with pytest.raises(Exception, match="Invalid symbol type"):
            Node("unknown_sym").to_list(self.sl)


class TestNodeToLatex:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=3)

    def test_float_node(self):
        assert Node("1.5").to_latex(self.sl) == "$1.5$"

    def test_var_and_lit(self):
        assert Node("X_0").to_latex(self.sl) == "$X_{0}$"
        assert Node("pi").to_latex(self.sl) == r"$\pi$"

    def test_const_indexed(self):
        assert Node("C").to_latex(self.sl) == "$C_{0}$"

    def test_multiple_consts_increment_counter(self):
        node = Node("+", right=Node("C"), left=Node("C"))
        assert node.to_latex(self.sl) == "$C_{0} + C_{1}$"

    def test_fn_wraps_fn_or_op_left_child(self):
        # fn with op child → extra parens
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        assert Node("sin", None, inner).to_latex(self.sl) == r"$\sin (X_{1} + X_{0})$"
        # fn with simple (var) child → no extra parens
        assert Node("sin", None, Node("X_0")).to_latex(self.sl) == r"$\sin X_{0}$"

    def test_op_wraps_lower_precedence_children(self):
        # left child lower precedence → left wrapped
        inner = Node("+", right=Node("X_0"), left=Node("X_1"))
        node = Node("*", right=Node("X_2"), left=inner)
        assert node.to_latex(self.sl) == r"$(X_{1} + X_{0}) \cdot X_{2}$"
        # right child lower precedence → right wrapped
        node2 = Node("*", right=inner, left=Node("X_2"))
        assert node2.to_latex(self.sl) == r"$X_{2} \cdot (X_{1} + X_{0})$"

    @pytest.mark.parametrize(
        "tokens, latex",
        [
            # equal precedence on the right is bracketed; on the left it is not
            ("X_0 - ( X_1 + X_2 )", r"$X_{0} - (X_{1} + X_{2})$"),
            ("X_0 - ( X_1 - X_2 )", r"$X_{0} - (X_{1} - X_{2})$"),
            ("X_0 - X_1 - X_2", r"$X_{0} - X_{1} - X_{2}$"),
            # a non-atomic base of a superscript is bracketed
            ("( X_0 ^ X_1 ) ^ X_2", r"$(X_{0}^{X_{1}})^{X_{2}}$"),
            ("u- X_0 ^ 2", r"$(- X_{0})^{2}$"),
            ("sin ( X_0 ) ^ 2", r"$(\sin X_{0})^{2}$"),
            ("( X_0 ^2 ) ^ X_1", r"$(X_{0}^2)^{X_{1}}$"),
            ("-1.5 ^ 2", r"$(-1.5)^{2}$"),
            ("u- X_0 ^2", r"$- (X_{0}^2)$"),
            # operands the template groups in braces are not bracketed
            ("X_0 ^ ( X_1 ^ X_2 )", r"$X_{0}^{X_{1}^{X_{2}}}$"),
            ("X_0 / ( X_1 * X_2 )", r"$\frac{X_{0}}{X_{1} \cdot X_{2}}$"),
            ("sqrt ( X_0 + X_1 )", r"$\sqrt {X_{0} + X_{1}}$"),
            ("exp ( X_0 * X_1 )", r"$e^{X_{0} \cdot X_{1}}$"),
            # a negative number on the right of an operator
            ("X_0 - -1.5", r"$X_{0} - (-1.5)$"),
        ],
    )
    def test_brackets(self, tokens, latex):
        assert tokens_to_tree(tokens.split(), self.sl).to_latex(self.sl) == latex

    def test_invalid_symbol_raises(self):
        with pytest.raises(Exception, match="Invalid symbol type"):
            Node("unknown").to_latex(self.sl)

    def test_to_latex_with_auto_generated_templates(self):
        lib = SymbolLibrary()
        lib.add_symbol("+", "op", 0, "{} + {}")
        lib.add_symbol("X_0", "var", 5, "X[:, 0]")
        lib.add_symbol("X_1", "var", 5, "X[:, 1]")
        node = Node("+", right=Node("X_0"), left=Node("X_1"))
        assert node.to_latex(lib) == r"$\text{{X_1}} \text{+} \text{{X_0}}$"


class TestNodeHeight:
    def test_single_leaf_has_height_one(self):
        assert Node("X_0").height() == 1

    def test_binary_op_depth_two(self):
        assert Node("+", Node("1"), Node("X_0")).height() == 2

    def test_returns_max_branch_height(self):
        # left branch is deeper: +(*(X_0, X_1), X_2) → height 3
        inner = Node("*", Node("X_0"), Node("X_1"))
        assert Node("+", Node("X_2"), inner).height() == 3

    def test_unary_fn_depth(self):
        assert Node("sin", None, Node("X_0")).height() == 2


class TestNodeLen:
    def test_single_leaf(self):
        assert len(Node("X_0")) == 1

    def test_binary_op_three_nodes(self):
        assert len(Node("+", Node("1"), Node("X_0"))) == 3

    def test_deeper_tree_counts_all_nodes(self):
        # +(*(X_0, X_1), X_2) → 5 nodes
        inner = Node("*", Node("X_0"), Node("X_1"))
        assert len(Node("+", Node("X_2"), inner)) == 5


class TestNodeStr:
    def test_single_leaf(self):
        with pytest.warns(UserWarning):
            assert str(Node("X_0")) == "X_0"

    def test_binary_op_single_token_children(self):
        # Both children are single tokens → no extra parens
        with pytest.warns(UserWarning):
            assert str(Node("+", Node("1"), Node("X_0"))) == "X_0+1"

    def test_nested_wraps_multi_token_child(self):
        # Inner node produces multiple tokens → gets wrapped in parens
        inner = Node("+", Node("X_0"), Node("X_1"))
        with pytest.warns(UserWarning):
            result = str(Node("*", Node("X_2"), inner))
        assert result == "(X_1+X_0)*X_2"


class TestNodeCopy:
    def test_copy_is_deep(self):
        original = Node("+", right=Node("X_0"), left=Node("1"))
        copied = copy(original)
        # Mutate the copy's left child symbol
        copied.left.symbol = "X_9"
        assert original.left.symbol == "1"

    def test_copy_has_same_structure(self):
        original = Node("+", right=Node("X_0"), left=Node("1"))
        copied = copy(original)
        assert copied.symbol == "+"
        assert copied.left.symbol == "1"
        assert copied.right.symbol == "X_0"

    def test_copy_is_not_same_object(self):
        original = Node("X_0")
        assert copy(original) is not original


class TestExprToLatex:
    def setup_method(self):
        self.sl = SymbolLibrary.default_symbols(num_variables=2)

    def test_node_input(self):
        node = Node("+", right=Node("X_0"), left=Node("1"))
        assert expr_to_latex(node, self.sl) == "$1 + X_{0}$"

    def test_list_input(self):
        assert expr_to_latex(["(", "X_0", "+", "X_1", ")"], self.sl) == "$X_{0} + X_{1}$"

    def test_invalid_type_returns_empty_string(self, capsys):
        result = expr_to_latex("X_0 + X_1", self.sl)
        assert result == ""
        assert "Error" in capsys.readouterr().out


class TestNodeToListContextFallback:
    def setup_method(self):
        SymbolLibrary.set_default(None)

    def teardown_method(self):
        SymbolLibrary.set_default(None)

    def test_to_list_uses_active_context_no_warning(self, recwarn):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        node = Node("sin", None, Node("X_0"))
        with sl:
            result = node.to_list()
        assert result == ["sin", "(", "X_0", ")"]
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_to_list_uses_default_no_warning(self, recwarn):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        SymbolLibrary.set_default(sl)
        node = Node("sin", None, Node("X_0"))
        result = node.to_list()
        assert result == ["sin", "(", "X_0", ")"]
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_to_list_warns_when_no_context_or_default(self):
        node = Node("sin", None, Node("X_0"))
        with pytest.warns(UserWarning):
            result = node.to_list()
        # crude fallback: prefix fn wrapped in parens
        assert result == ["sin", "(", "X_0", ")"]

    def test_str_node_uses_context_no_warning(self, recwarn):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        node = Node("+", Node("1"), Node("X_0"))
        with sl:
            result = str(node)
        assert result == "X_0+1"
        user_warnings = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


class TestContextManagerFallback:
    def test_tokens_to_tree_no_active_context_raises(self):
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            tokens_to_tree(["X_0", "+", "1"])

    def test_node_to_latex_uses_active_context(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        node = Node("+", right=Node("X_0"), left=Node("1"))
        with sl:
            result = node.to_latex()
        assert result == "$1 + X_{0}$"

    def test_node_to_latex_no_active_context_raises(self):
        node = Node("+", right=Node("X_0"), left=Node("1"))
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            node.to_latex()

    def test_expr_to_latex_uses_active_context(self):
        sl = SymbolLibrary.default_symbols(num_variables=1)
        with sl:
            result = expr_to_latex(["X_0", "+", "1"])
        assert result == "$X_{0} + 1$"

    def test_expr_to_latex_no_active_context_raises(self):
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            expr_to_latex(["X_0", "+", "1"])


class TestIsFloat:
    def test_returns_false_for_none(self):
        assert is_float(None) is False

    def test_returns_true_for_numeric_values(self):
        assert is_float(1) is True
        assert is_float(1.5) is True
        assert is_float("1") is True
        assert is_float("1.0") is True
        assert is_float("-2.5") is True

    def test_returns_false_for_non_numeric_strings(self):
        assert is_float("hello") is False
        assert is_float("+") is False
        assert is_float("X_0") is False
