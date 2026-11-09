"""Tests for SRToolkit.approaches.ProGED: how adapt chooses the grammar."""

import logging
import warnings

import numpy as np
import pytest

from SRToolkit.approaches import ProGED
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
from SRToolkit.dataset.sr_dataset import SR_dataset
from SRToolkit.evaluation import SR_evaluator
from SRToolkit.utils.expression_constraints import MaxTokens
from SRToolkit.utils.grammar.grammar import Grammar, Rule
from SRToolkit.utils.symbol_library import SymbolLibrary

_LOGGER = "SRToolkit.dataset.background_knowledge"
_X = np.random.default_rng(0).random((10, 2)) + 0.5


def _lib():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _products_of(var):
    """A grammar that only produces products of ``var``: var, var*var, var*var*var, ..."""
    return Grammar(Rule.from_line(f"E -> E '*' '{var}' [0.5] | '{var}' [0.5]"), start="E")


def _dataset(reference_grammar=None):
    return SR_dataset(_X, _lib(), y=_X[:, 0] + _X[:, 1], max_evaluations=5, reference_grammar=reference_grammar)


def _tokens(results):
    return {t for m in results[-1].all_models for t in m.expr}


def test_adapt_uses_reference_grammar(caplog):
    lib = _lib()
    bk = BackgroundKnowledge(lib, constraints=[MaxTokens(1)], reference_grammar=_products_of("X_0"))
    required = MaxTokens(1)
    required.setup(lib)
    approach = ProGED()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        approach.adapt(_X, bk, (required,))
    se = SR_evaluator(_X, _X[:, 0], symbol_library=lib, max_evaluations=5, success_threshold=-1)
    approach.search(se, seed=0)
    assert {t for m in se.models for t in m.expr} <= {"X_0", "*"}
    assert any(len(m.expr) > 1 for m in se.models)  # the MaxTokens(1) constraints did not shape the grammar
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


@pytest.mark.parametrize(
    "second_grammar, check",
    [
        (_products_of("X_1"), lambda tokens: tokens <= {"X_1", "*"}),
        (None, lambda tokens: not tokens <= {"X_0", "*"}),
    ],
    ids=["other-reference-grammar", "generic-grammar"],
)
def test_reused_instance_rechooses_grammar_per_dataset(second_grammar, check):
    # regression: adapt stored the chosen grammar as the constructor grammar, so the scope became
    # "never" and a reused instance kept sampling dataset A's grammar on dataset B.
    approach = ProGED()
    _dataset(_products_of("X_0")).evaluate_approach(approach, initial_seed=0, verbose=False)
    results = _dataset(second_grammar).evaluate_approach(approach, initial_seed=0, verbose=False)
    assert check(_tokens(results))
    assert approach.adaptation_scope == "experiment"
    assert approach.config.to_dict()["grammar"] is None


def test_constructor_grammar_is_kept():
    approach = ProGED(_products_of("X_0"))
    results = _dataset(_products_of("X_1")).evaluate_approach(approach, initial_seed=0, verbose=False)
    assert _tokens(results) <= {"X_0", "*"}
    assert approach.adaptation_scope == "never"
    # adapt does nothing with a constructor grammar, even if the task has a reference grammar.
    approach.adapt(_X, BackgroundKnowledge(_lib(), reference_grammar=_products_of("X_1")), ())
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=5, success_threshold=-1)
    approach.search(se, seed=0)
    assert {t for m in se.models for t in m.expr} <= {"X_0", "*"}


@pytest.mark.parametrize(
    "grammar, allowed",
    [
        (_products_of("X_0"), {"X_0", "*"}),
        ("# start: E\nE -> E '*' 'X_0' [0.5] | 'X_0' [0.5]", {"X_0", "*"}),
        (SymbolLibrary.from_symbol_list(["*"], num_variables=1), {"X_0", "*", "(", ")"}),
    ],
    ids=["grammar", "grammar-string", "symbol-library"],
)
def test_constructor_grammar_survives_config_round_trip(grammar, allowed):
    # A grid rebuilds approaches from their config, so the constructor grammar must be in it.
    approach = ProGED.from_config(ProGED(grammar).config.to_dict())
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=5, success_threshold=-1)
    approach.search(se, seed=0)
    assert {t for m in se.models for t in m.expr} <= allowed


def test_search_skips_repeats_and_rejected_then_stops():
    grammar = Grammar(Rule.from_line("E -> E '+' 'X_0' [0.3] | 'X_0' [0.7]"), start="E")
    lib = _lib()
    se = SR_evaluator(
        _X, _X[:, 0], symbol_library=lib, max_evaluations=10, success_threshold=-1, required_constraints=[MaxTokens(3)]
    )
    with pytest.warns(UserWarning):
        ProGED(grammar, max_consecutive_skips=50).search(se, seed=0)
    assert se.total_evaluations == 2
    assert sorted("".join(m.expr) for m in se.models) == ["X_0", "X_0+X_0"]
    assert all(m.status != "rejected" for m in se.models)


def test_search_spends_budget_on_new_expressions():
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=20, success_threshold=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ProGED(_lib()).search(se, seed=0)
    assert se.total_evaluations == 20 == len(se.models)


@pytest.mark.parametrize(
    "skip_evaluated, skip_rejected",
    [(False, False), (True, False), (False, True)],
    ids=["no-skipping", "skip-evaluated-only", "skip-rejected-only"],
)
def test_search_skip_flags_are_independent(skip_evaluated, skip_rejected):
    grammar = Grammar(Rule.from_line("E -> E '+' 'X_0' [0.5] | 'X_0' [0.5]"), start="E")
    se = SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=5,
        success_threshold=-1,
        required_constraints=[MaxTokens(3)],
    )
    approach = ProGED(grammar, skip_evaluated=skip_evaluated, skip_rejected=skip_rejected)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        approach.search(se, seed=2)  # its first 5 samples include a repeat and rejected ones
    assert se.total_evaluations == 5
    assert (len(se.models) == 5) is skip_evaluated  # no repeats exactly when they are skipped
    assert any(m.status == "rejected" for m in se.models) is not skip_rejected


def test_config_round_trip_keeps_skip_settings():
    approach = ProGED(max_consecutive_skips=7, skip_evaluated=False, skip_rejected=False)
    rebuilt = ProGED.from_config(approach.config.to_dict())
    config = rebuilt.config
    assert (config.max_consecutive_skips, config.skip_evaluated, config.skip_rejected) == (7, False, False)
    assert (rebuilt.max_consecutive_skips, rebuilt.skip_evaluated, rebuilt.skip_rejected) == (7, False, False)


def test_search_stops_after_exactly_max_consecutive_skips():
    only_x0 = Grammar(Rule.from_line("E -> 'X_0' [1.0]"), start="E")
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1)
    with pytest.warns(UserWarning, match=r"\b3 samples"):
        ProGED(only_x0, max_consecutive_skips=3).search(se, seed=0)
    assert se.total_evaluations == 1


def test_search_is_reproducible_with_a_seed():
    def run():
        se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=15, success_threshold=-1)
        ProGED(_lib()).search(se, seed=5)
        return ["".join(m.expr) for m in se.models]

    assert run() == run()


def test_from_config_defaults_for_older_configs():
    approach = ProGED.from_config({"grammar": None})
    assert (approach.max_consecutive_skips, approach.skip_evaluated, approach.skip_rejected) == (10000, True, True)
