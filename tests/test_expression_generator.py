import nltk
import pytest

from SRToolkit.utils.expression_generator import (
    _expand,
    create_generic_pcfg,
    generate_from_pcfg,
    generate_n_expressions,
)
from SRToolkit.utils.symbol_library import SymbolLibrary


class TestCreateGenericPcfg:
    def test_output_matches_docstring_example(self):
        sl = SymbolLibrary.from_symbol_list(["+", "-", "*", "sin", "^2", "pi"], 2)
        expected = (
            "E -> E '+' F [0.2]\n"
            "E -> E '-' F [0.2]\n"
            "E -> F [0.6]\n"
            "F -> F '*' B [0.4]\n"
            "F -> B [0.6]\n"
            "B -> T [1.0]\n"
            "T -> R [0.2]\n"
            "T -> C [0.2]\n"
            "T -> V [0.6]\n"
            "C -> 'pi' [1.0]\n"
            "R -> 'sin' '(' E ')' [0.4]\n"
            "R -> P [0.15]\n"
            "R -> '(' E ')' [0.45]\n"
            "P -> '(' E ')' '^2' [1.0]\n"
            "V -> 'X_0' [0.5]\n"
            "V -> 'X_1' [0.5]\n"
        )
        assert create_generic_pcfg(sl) == expected

    def test_no_ops_produce_passthrough_rules(self):
        sl = SymbolLibrary.from_symbol_list([], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "E -> F [1.0]" in grammar
        assert "F -> B [1.0]" in grammar
        assert "B -> T [1.0]" in grammar

    def test_only_lits_no_consts(self):
        sl = SymbolLibrary.from_symbol_list(["pi", "e"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "T -> C [0.2]" in grammar
        assert "C -> 'pi' [0.5]" in grammar
        assert "C -> 'e' [0.5]" in grammar

    def test_only_consts_no_lits(self):
        sl = SymbolLibrary.from_symbol_list(["C"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "T -> C [0.2]" in grammar
        assert "C -> 'C' [1.0]" in grammar

    def test_both_lits_and_consts(self):
        sl = SymbolLibrary.from_symbol_list(["pi", "C"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        # lit gets weight 0.2, const gets weight 0.8
        assert "C -> 'pi' [0.2]" in grammar
        assert "C -> 'C' [0.8]" in grammar

    def test_no_const_no_lit_omits_T_to_C(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "T -> C" not in grammar
        assert "T -> V [0.7]" in grammar

    def test_prefix_fn_without_postfix(self):
        sl = SymbolLibrary.from_symbol_list(["sin"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "R -> 'sin' '(' E ')'" in grammar
        assert "R -> '(' E ')' [0.6]" in grammar
        assert "R -> P" not in grammar

    def test_postfix_fn_without_prefix(self):
        sl = SymbolLibrary.from_symbol_list(["^2"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "R -> P [0.15]" in grammar
        assert "R -> '(' E ')' [0.85]" in grammar
        assert "R -> 'sin'" not in grammar

    def test_no_fns_at_all(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "R -> '(' E ')' [1.0]" in grammar

    def test_power_op_produces_B_rules(self):
        sl = SymbolLibrary.from_symbol_list(["^"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        assert "B -> B '^' T [0.05]" in grammar
        assert "B -> T [0.95]" in grammar

    def test_multiple_prefix_fns_split_probability(self):
        sl = SymbolLibrary.from_symbol_list(["sin", "cos"], num_variables=1)
        grammar = create_generic_pcfg(sl)
        # 0.4 / 2 = 0.2 per fn
        assert "R -> 'sin' '(' E ')' [0.2]" in grammar
        assert "R -> 'cos' '(' E ')' [0.2]" in grammar

    def test_grammar_parseable_by_nltk(self):
        sl = SymbolLibrary.default_symbols(num_variables=2)
        grammar = create_generic_pcfg(sl)
        # Should not raise
        nltk.PCFG.fromstring(grammar)


class TestExpand:
    def test_nonterminal_with_no_productions_returns_symbol_string(self):
        # A grammar where B is referenced but never defined → _expand returns ["B"]
        grammar = nltk.PCFG.fromstring("E -> '1' [1.0]")
        result = _expand(grammar, nltk.grammar.Nonterminal("B"), 0, 40)
        assert result == ["B"]


class TestGenerateFromPcfg:
    def test_deterministic_grammar_returns_expected(self):
        assert generate_from_pcfg("E -> '1' [1.0]") == ["1"]

    def test_custom_start_symbol(self):
        grammar = "E -> F [1.0]\nF -> '2' [1.0]"
        assert generate_from_pcfg(grammar, start_symbol="F") == ["2"]

    def test_limit_exceeded_raises(self):
        # Purely recursive grammar always exceeds max_depth=2 → fails every attempt
        grammar = "E -> '1' '+' E [1.0]"
        with pytest.raises(Exception, match="Couldn't find an expression"):
            generate_from_pcfg(grammar, max_depth=2, limit=1)


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

    def test_accepts_string_grammar(self):
        grammar = create_generic_pcfg(SymbolLibrary.from_symbol_list(["+"], num_variables=2))
        exprs = generate_n_expressions(grammar, 5, verbose=False)
        assert len(exprs) == 5

    def test_invalid_type_raises(self):
        with pytest.raises(Exception, match="Description of expressions"):
            generate_n_expressions(42, 5, verbose=False)

    def test_max_expression_length_applied(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=2)
        exprs = generate_n_expressions(sl, 20, verbose=False, max_expression_length=5)
        assert all(len(e) <= 5 for e in exprs)

    def test_verbose_true_runs_without_error(self):
        sl = SymbolLibrary.from_symbol_list([], num_variables=1)
        exprs = generate_n_expressions(sl, 2, unique=False, verbose=True, max_expression_length=1)
        assert len(exprs) == 2

    def test_max_consecutive_failures_raises(self):
        # Purely recursive grammar always fails generate_from_pcfg → triggers the failure guard
        # verbose=True ensures pbar.close() inside the guard (line 276) is also covered
        grammar = "E -> '1' '+' E [1.0]"
        with pytest.raises(Exception, match="consecutive failures"):
            generate_n_expressions(grammar, 1, verbose=True, max_expression_length=1, max_consecutive_failures=2)


class TestCreateGenericPcfgContextManager:
    def test_uses_active_context_when_no_argument(self):
        sl = SymbolLibrary.from_symbol_list(["+"], num_variables=1)
        with sl:
            grammar = create_generic_pcfg()
        assert "E -> E '+' F" in grammar
        assert "V -> 'X_0'" in grammar

    def test_no_active_context_raises(self):
        with pytest.raises(RuntimeError, match="No active SymbolLibrary"):
            create_generic_pcfg()
