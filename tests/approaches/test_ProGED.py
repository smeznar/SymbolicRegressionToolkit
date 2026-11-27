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
from SRToolkit.utils.settings import Settings
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
        settings=Settings(evaluator={"max_consecutive_skips": 50}),
    )
    ProGED(grammar).search(se, seed=0)
    assert se.stop_reason == "max_consecutive_skips"
    assert se.total_evaluations == 2
    assert sorted("".join(m.expr) for m in se.models) == ["X_0", "X_0+X_0"]
    assert all(m.status != "rejected" for m in se.models)


# Reason: clause (grammar: with the evaluator setting canonicalise_constants, samples differing only
# in their labels are one expression, also for the repeat check; off by default): the grammar yields
# C_0*X_0+C_0 and C_1*X_0+C_1.
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
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1, settings=Settings(evaluator={"max_consecutive_skips": 20, "canonicalise_constants": canonicalise}))  # fmt: skip
    ProGED(grammar).search(se, seed=0)
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
    approach = ProGED(skip_evaluated=False, skip_rejected=False)
    rebuilt = ProGED.from_config(approach.config.to_dict())
    c = rebuilt.config
    assert (c.skip_evaluated, c.skip_rejected) == (False, False)
    assert (rebuilt.skip_evaluated, rebuilt.skip_rejected) == (False, False)


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
        settings=Settings(evaluator={"max_consecutive_skips": 3}),
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
    state = approach.start(se, seed=0)
    (first,) = approach.ask(se, state)
    assert (first.expr, first.skip) == (["X_0"], None)
    se.evaluate_expr(first.expr)
    (second,) = approach.ask(se, state)
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
    assert (a.skip_evaluated, a.skip_rejected) == (True, True)
    # Unknown keys, such as the removed max_consecutive_skips and canonicalise_constants, are ignored.
    b = ProGED.from_config(
        {"grammar": None, "max_consecutive_skips": 7, "canonicalise_constants": True, "no_such_key": 1}
    )
    assert (b.skip_evaluated, b.skip_rejected) == (True, True)


def test_start_without_grammar_or_adapt_raises():
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib())
    with pytest.raises(RuntimeError, match="adapt"):
        ProGED().start(se, seed=0)


# Reason: regression + clause (start: "seed ... None to leave NumPy's global random state as it is";
# "Seed NumPy's global random state"; returns None). The old start called np.random.seed(seed) for
# every seed, so seed=None reseeded from entropy. Oracle: the draws right after np.random.seed(ref).
@pytest.mark.parametrize(
    "seed, reference_seed",
    [(None, 123), (7, 7)],
    ids=["seed-none-leaves-state", "seed-sets-state"],
)
def test_start_seeds_or_leaves_global_random_state(seed, reference_seed):
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib())
    np.random.seed(reference_seed)
    expected = np.random.random(3)
    np.random.seed(123)
    assert ProGED(_products_of("X_0")).start(se, seed=seed) is None
    np.testing.assert_array_equal(np.random.random(3), expected)


# Reason: regression + clause (ask: "Each call makes one derivation attempt ([generate_one]) ... there is
# no redraw"). S yields X_0 or the never-ending L, each with p = 0.5. Oracle: from the same seed, 20 calls of
# generate_one(max_retries=1), the single attempt the docstring names. Any redraw (the old code, or more
# retries) gives fewer Nones and consumes the random state differently.
def test_ask_failed_derivation_is_none_not_redrawn():
    half_failing = Grammar(Rule.from_line("S -> 'X_0' [0.5] | L [0.5]") + Rule.from_line("L -> L 'X_0' [1.0]"), "S")
    np.random.seed(0)
    expected = [half_failing.generate_one(max_retries=1) for _ in range(20)]
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), max_evaluations=10, success_threshold=-1)
    approach = ProGED(half_failing)
    state = approach.start(se, seed=0)
    proposals = [approach.ask(se, state)[0] for _ in range(20)]
    assert [c if c is None else c.expr for c in proposals] == expected
    assert {c.skip for c in proposals if c is not None} == {None}
    assert None in expected and ["X_0"] in expected  # precondition: both outcomes occur


# Reason: regression + clause (ask: "A run that finds only skips, e.g. with a grammar that never yields an
# expression, is stopped by the evaluator's max_consecutive_skips"; None counts as "not_expression").
# The old code raised a bare Exception after 100 failed derivations in a row.
def test_search_with_grammar_that_never_yields_stops_on_skip_limit():
    never_ends = Grammar(Rule.from_line("E -> E '+' 'X_0' [1.0]"), start="E")
    se = SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=10,
        success_threshold=-1,
        settings=Settings(evaluator={"max_consecutive_skips": 3}),
    )
    ProGED(never_ends).search(se, seed=0)
    assert se.stop_reason == "max_consecutive_skips"
    assert se.skips == {"not_expression": 3}
    assert se.total_evaluations == 0


_LONG_EXPR = ["X_0", "+"] * 25 + ["X_0"]  # 51 tokens
_LONG_GRAMMAR = Grammar(Rule.from_line("E -> " + " ".join(f"'{t}'" for t in _LONG_EXPR)), start="E")


# Reason: regression + clause (grammar: "Sample length is not limited by ProGED: a MaxTokens among the
# evaluator's required constraints rejects long samples"). The old code silently redrew every sample over
# 50 tokens, so this grammar's only sample was never proposed (and the search raised after 100 redraws).
@pytest.mark.parametrize(
    "required_constraints, skip",
    [(None, None), ([MaxTokens(50)], "rejected")],
    ids=["no-constraint", "max-tokens-rejects"],
)
def test_ask_proposes_long_samples(required_constraints, skip):
    se = SR_evaluator(
        _X, _X[:, 0], symbol_library=_lib(), success_threshold=-1, required_constraints=required_constraints
    )
    approach = ProGED(_LONG_GRAMMAR)
    (candidate,) = approach.ask(se, approach.start(se, seed=0))
    assert (candidate.expr, candidate.skip) == (_LONG_EXPR, skip)


def _branches(copies, depth):
    """A grammar whose only derivation takes exactly ``1 + copies * depth`` rule applications: S joins
    ``copies`` chains C0 -> C1 -> ... -> 'X_0' of ``depth`` rules with '+'. Several shallow chains instead of
    one deep chain, because a derivation tree a few hundred levels deep can hit Python's recursion limit."""
    rules = Rule.from_line("S -> " + " '+' ".join(["C0"] * copies))
    rules += [r for i in range(depth - 1) for r in Rule.from_line(f"C{i} -> C{i + 1}")]
    return Grammar(rules + Rule.from_line(f"C{depth - 1} -> 'X_0'"), start="S")


# Reason: boundary (ask: a derivation attempt fails after "more than 1000 rule applications").
# 1 + 9 * 111 = 1000 applications give X_0 + ... + X_0 (9 terms); 1 + 10 * 100 = 1001 fail.
@pytest.mark.parametrize(
    "copies, depth, expr",
    [(9, 111, ["X_0", "+"] * 8 + ["X_0"]), (10, 100, None)],
    ids=["1000-steps", "1001-steps"],
)
def test_ask_derivation_step_limit_is_1000(copies, depth, expr):
    se = SR_evaluator(_X, _X[:, 0], symbol_library=_lib(), success_threshold=-1)
    approach = ProGED(_branches(copies, depth))
    (candidate,) = approach.ask(se, approach.start(se, seed=0))
    assert (candidate if candidate is None else candidate.expr) == expr
