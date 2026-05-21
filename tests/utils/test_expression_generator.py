import pytest

from SRToolkit.utils.expression_generator import generate_n_expressions
from SRToolkit.utils.grammar import Grammar, Rule
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestGenerateNExpressions:
    def test_returns_correct_count(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        exprs = generate_n_expressions(sl, 10, verbose=False)
        assert len(exprs) == 10

    def test_non_unique_allows_duplicates(self):
        sl = SymbolLibrary.from_symbol_list([], num_variables=1)
        exprs = generate_n_expressions(sl, 3, unique=False, verbose=False, max_expression_length=1)
        assert exprs == [["X_0"], ["X_0"], ["X_0"]]

    def test_unique_prevents_duplicates(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        exprs = generate_n_expressions(sl, 20, unique=True, verbose=False)
        expr_strings = ["".join(e) for e in exprs]
        assert len(expr_strings) == len(set(expr_strings))

    def test_accepts_grammar_object(self):
        g = Grammar([Rule("E", ["x"]), Rule("E", ["y"])], start="E")
        exprs = generate_n_expressions(g, 5, unique=False, verbose=False)
        assert len(exprs) == 5
        assert all(e in [["x"], ["y"]] for e in exprs)

    def test_accepts_string_grammar(self):
        grammar = Grammar.from_symbol_library(
            SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        ).to_grammar_string()
        exprs = generate_n_expressions(grammar, 5, verbose=False)
        assert len(exprs) == 5

    def test_invalid_type_raises(self):
        with pytest.raises(Exception, match="expression_description"):
            generate_n_expressions(42, 5, verbose=False)

    def test_max_expression_length_applied(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        exprs = generate_n_expressions(sl, 20, verbose=False, max_expression_length=5)
        assert all(len(e) <= 5 for e in exprs)

    def test_verbose_true_runs_without_error(self):
        sl = SymbolLibrary.from_symbol_list([], num_variables=1)
        exprs = generate_n_expressions(sl, 2, unique=False, verbose=True, max_expression_length=1)
        assert len(exprs) == 2

    def test_max_consecutive_generation_failures_raises(self):
        # Purely recursive grammar always fails → triggers the failure guard
        grammar = "E -> '1' '+' E [1.0]"
        with pytest.raises(Exception, match="in a row"):
            generate_n_expressions(
                grammar, 1, verbose=True, max_expression_length=1, max_consecutive_generation_failures=2
            )
