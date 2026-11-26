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
        _X,
        _X[:, 0],
        symbol_library=lib,
        max_evaluations=10,
        success_threshold=-1,
        required_constraints=[MaxTokens(3)],
        max_consecutive_skips=50,
    )
    ProGED(grammar).search(se, seed=0)
    assert se.stop_reason == "max_consecutive_skips"
    assert se.total_evaluations == 2
    assert sorted("".join(m.expr) for m in se.models) == ["X_0", "X_0+X_0"]
    assert all(m.status != "rejected" for m in se.models)


# Reason: clause (canonicalise_constants: samples differing only in their labels are one expression;
# off by default): the grammar yields C_0*X_0+C_0 and C_1*X_0+C_1.
@pytest.mark.parametrize(
    "canonicalise, exprs",
    [
        (True, [["C_0", "*", "X_0", "+", "C_0"]]),
        (False, [["C_0", "*", "X_0", "+", "C_0"], ["C_1", "*", "X_0", "+", "C_1"]]),
    ],
    ids=["on", "off"],
)
def test_search_canonicalise_constants(canonicalise, exprs):
    grammar = Grammar(
        Rule.from_line("E -> 'C_0' '*' 'X_0' '+' 'C_0' [0.5] | 'C_1' '*' 'X_0' '+' 'C_1' [0.5]"), start="E"
    )
    se = SR_evaluator(
        _X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1, max_consecutive_skips=20
    )
    ProGED(grammar, canonicalise_constants=canonicalise).search(se, seed=0)
    assert se.stop_reason == "max_consecutive_skips"
    assert sorted(m.expr for m in se.models) == exprs


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
    assert set(se.skips) == {r for r, on in [("evaluated", skip_evaluated), ("rejected", skip_rejected)] if on}


def test_config_round_trip_keeps_skip_settings():
    approach = ProGED(skip_evaluated=False, skip_rejected=False, canonicalise_constants=True)
    rebuilt = ProGED.from_config(approach.config.to_dict())
    c = rebuilt.config
    assert (c.skip_evaluated, c.skip_rejected, c.canonicalise_constants) == (False, False, True)
    r = rebuilt
    assert (r.skip_evaluated, r.skip_rejected, r.canonicalise_constants) == (False, False, True)


# Reason: clause (stops after exactly max_consecutive_skips skips, each counted once: as "evaluated" if both
# conditions hold, each with its flag set, otherwise as "rejected"). The grammar only gives X_0; in the
# rejecting cases the user evaluated it (as rejected) before the search.
@pytest.mark.parametrize(
    "rejects, skip_evaluated, expected",
    [
        (False, True, {"evaluated": 3}),
        (True, True, {"evaluated": 3}),
        (True, False, {"rejected": 3}),
    ],
    ids=["repeats", "evaluated-and-rejected", "rejected-skip-evaluated-off"],
)
def test_search_stops_after_exactly_max_consecutive_skips(rejects, skip_evaluated, expected):
    only_x0 = Grammar(Rule.from_line("E -> 'X_0' [1.0]"), start="E")
    constraints = [MaxTokens(0)] if rejects else None
    se = SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=10,
        success_threshold=-1,
        required_constraints=constraints,
        max_consecutive_skips=3,
    )
    if rejects:
        se.evaluate_expr(["X_0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ProGED(only_x0, skip_evaluated=skip_evaluated).search(se, seed=0)
    assert se.stop_reason == "max_consecutive_skips"
    assert se.total_evaluations == 1
    assert se.skips == expected


# Reason: clause (ask: one sample per call; a skipped sample is a lookup with its skip reason, which
# the step loop counts, not ask).
def test_ask_proposes_skipped_samples_as_lookups():
    only_x0 = Grammar(Rule.from_line("E -> 'X_0' [1.0]"), start="E")
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1)
    approach = ProGED(only_x0)
    approach.start(se, seed=0)
    (first,) = approach.ask(se)
    assert (first.expr, first.skip) == (["X_0"], None)
    se.evaluate_expr(first.expr)
    (second,) = approach.ask(se)
    assert (second.expr, second.skip) == (["X_0"], "evaluated")
    assert (se.skips, se.should_stop) == ({}, False)


def test_search_is_reproducible_with_a_seed():
    def run():
        se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=15, success_threshold=-1)
        ProGED(_lib()).search(se, seed=5)
        return ["".join(m.expr) for m in se.models]

    assert run() == run()


def test_from_config_defaults_for_older_configs():
    a = ProGED.from_config({"grammar": None})
    assert (a.skip_evaluated, a.skip_rejected, a.canonicalise_constants) == (True, True, False)
    # Unknown keys, such as the removed max_consecutive_skips, are ignored.
    b = ProGED.from_config({"grammar": None, "max_consecutive_skips": 7, "no_such_key": 1})
    assert (b.skip_evaluated, b.skip_rejected, b.canonicalise_constants) == (True, True, False)


# Reason: clause (get_search_state: the sampling grammar is the whole search state besides NumPy's
# global random state; set_search_state restores it in place of start).
def test_search_state_round_trips_the_grammar():
    grammar = _products_of("X_0")
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1)
    original = ProGED(grammar)
    original.start(se, seed=0)
    state = original.get_search_state()
    restored = ProGED()  # no grammar of its own: it can only sample from the state
    restored.set_search_state(state)
    np.random.seed(1)
    expected = ["".join(original.ask(se)[0].expr) for _ in range(5)]
    np.random.seed(1)
    assert ["".join(restored.ask(se)[0].expr) for _ in range(5)] == expected
    assert len(set(expected)) > 1  # precondition: the samples depend on the random state


def test_start_without_grammar_or_adapt_raises():
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib())
    with pytest.raises(RuntimeError, match="adapt"):
        ProGED().start(se, seed=0)
