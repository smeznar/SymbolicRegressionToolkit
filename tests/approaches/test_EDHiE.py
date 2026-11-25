"""Tests for SRToolkit.approaches.EDHiE (tiny configs: training is expensive)."""

import copy
import logging
import pickle
import random
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pymoo")

from strategies import SignalAfter  # noqa: E402

from SRToolkit.approaches.EDHiE import EDHiE  # noqa: E402
from SRToolkit.approaches.sr_approach import Candidate  # noqa: E402
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge  # noqa: E402
from SRToolkit.dataset.sr_dataset import SR_dataset  # noqa: E402
from SRToolkit.evaluation import SR_evaluator  # noqa: E402
from SRToolkit.evaluation.model_store import ModelStore  # noqa: E402
from SRToolkit.evaluation.results import SR_results  # noqa: E402
from SRToolkit.utils.expression_constraints import MaxConstants, MaxTokens, RequiredVariables  # noqa: E402
from SRToolkit.utils.grammar.grammar import Grammar, Rule  # noqa: E402
from SRToolkit.utils.symbol_library import SymbolLibrary  # noqa: E402

_LOGGER = "SRToolkit.dataset.background_knowledge"
_X = np.random.default_rng(0).random((10, 2)) + 0.5


def _lib():
    return SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)


def _products_of_x0():
    return Grammar(Rule.from_line("E -> E '*' 'X_0' [0.5] | 'X_0' [0.5]"), start="E")


def _set_up(constraint, lib):
    constraint.setup(lib)
    return constraint


def _tiny(**kwargs):
    return EDHiE(latent_size=8, max_expression_length=10, epochs=1, population_size=10, verbose=False, **kwargs)


def test_training_expressions_from_reference_grammar_follow_constraints(caplog):
    lib = _lib()
    bk = BackgroundKnowledge(lib, reference_grammar=_products_of_x0())
    approach = _tiny(num_expressions=20, max_consecutive_skips=50)
    with caplog.at_level(logging.INFO, logger=_LOGGER), pytest.warns(UserWarning):
        expressions = approach.training_expressions(bk, (_set_up(MaxTokens(5), lib),))
    assert len(expressions) == 20
    assert {"".join(e) for e in expressions} == {"X_0", "X_0*X_0", "X_0*X_0*X_0"}
    assert [r.levelno for r in caplog.records if r.name == _LOGGER] == [logging.INFO]


def test_training_expressions_from_symbol_library_are_unique():
    lib = _lib()
    approach = _tiny(num_expressions=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        expressions = approach.training_expressions(BackgroundKnowledge(lib), ())
    assert len({"".join(e) for e in expressions}) == 30 == len(expressions)
    assert all(len(e) <= 10 for e in expressions)
    allowed = set(lib.symbols2index()) | {"(", ")"}
    assert {t for e in expressions for t in e} <= allowed


def test_training_expressions_raise_when_nothing_passes():
    lib = _lib()
    bk = BackgroundKnowledge(lib, reference_grammar=_products_of_x0())
    approach = _tiny(num_expressions=20, max_consecutive_skips=50)
    with pytest.raises(ValueError):
        approach.training_expressions(bk, (_set_up(RequiredVariables(["X_1"]), lib),))


def test_training_expressions_write_labelled_constants_bare_before_filtering():
    # Regression: labelled constants from a reference grammar reached the one-hot vocabulary (KeyError).
    # Clause: they are written as bare C before the constraint filter, so C_0*X_0+C_0 becomes
    # C*X_0+C, which has 2 parameters and fails MaxConstants(1); C_0*X_0 becomes C*X_0 and passes.
    lib = _lib()
    grammar = Grammar(Rule.from_line("E -> 'C_0' '*' 'X_0' '+' 'C_0' [0.5] | 'C_0' '*' 'X_0' [0.5]"), start="E")
    approach = _tiny(num_expressions=5, max_consecutive_skips=20)
    with pytest.warns(UserWarning):
        expressions = approach.training_expressions(
            BackgroundKnowledge(lib, reference_grammar=grammar), (_set_up(MaxConstants(1), lib),)
        )
    assert {"".join(e) for e in expressions} == {"C*X_0"}
    assert len(expressions) == 5


@pytest.mark.slow
def test_adapt_trains_and_keeps_scope_once():
    # Regression: a reused instance must train again for the next task, not keep this model.
    lib = _lib()
    approach = _tiny(num_expressions=50)
    approach.adapt(np.empty((0, 9)), BackgroundKnowledge(lib), (_set_up(MaxTokens(10), lib),))
    assert approach.model is not None
    assert approach.adaptation_scope == "once"


@pytest.mark.slow
def test_adapt_restores_global_random_states():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    before = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    _tiny(num_expressions=50, training_seed=3).adapt(_X, BackgroundKnowledge(_lib()), ())
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1])
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.slow
def test_adapt_is_reproducible_with_training_seed():
    def weights():
        approach = _tiny(num_expressions=50, training_seed=3)
        approach.adapt(_X, BackgroundKnowledge(_lib()), ())
        return [p.detach().clone() for p in approach.model.parameters()]

    first, second = weights(), weights()
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.fixture(scope="session")
def tiny_hvae_weights(tmp_path_factory):
    """Weights of the tiny HVAE, trained once and saved next to pytest's temp directories.

    Later sessions reuse the file: mutmut runs every mutant in a forked pytest session, which cannot
    train (autograd does not survive a fork). The name hashes this file, the EDHiE module and the
    torch version, so a change to any of them trains again.
    """
    import hashlib
    import inspect
    import os

    source = Path(__file__).read_bytes() + Path(inspect.getfile(EDHiE)).read_bytes() + torch.__version__.encode()
    path = tmp_path_factory.getbasetemp().parent / f"edhie-tiny-{hashlib.sha256(source).hexdigest()[:16]}.pt"
    if not path.exists():
        approach = _tiny(num_expressions=200, training_seed=0)
        approach.adapt(_X, BackgroundKnowledge(_lib()), ())
        partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
        approach.save_adapted_state(str(partial))
        os.replace(partial, path)  # atomic: a concurrent session never reads half a file
    return path


@pytest.fixture(scope="module")
def trained_model(tiny_hvae_weights):
    return _tiny(weights_path=str(tiny_hvae_weights)).model


def test_weights_path_makes_scope_never(tmp_path, trained_model):
    approach = _tiny()
    approach.model = trained_model
    path = str(tmp_path / "hvae.pt")
    approach.save_adapted_state(path)
    assert _tiny(weights_path=path).adaptation_scope == "never"


def _evaluator(**kwargs):
    return SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=30,
        success_threshold=-1,
        required_constraints=[MaxTokens(7)],
        **kwargs,
    )


def _stepper(model, **kwargs):
    approach = _tiny(**kwargs)
    approach.model = model
    return approach


def _search(model, seed=0, **kwargs):
    se = _evaluator()
    _stepper(model, **kwargs).search(se, seed=seed)
    return se


@pytest.mark.slow
def test_search_skips_evaluated_and_rejected(trained_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        se = _search(trained_model, max_consecutive_skips=500)
    assert len(se.models) == se.total_evaluations
    assert all(m.status != "rejected" for m in se.models)


@pytest.mark.slow
def test_search_without_skipping_bills_every_individual(trained_model):
    se = _search(trained_model, skip_evaluated=False, skip_rejected=False)
    assert se.total_evaluations == 30
    assert len(se.models) < 30


@pytest.mark.slow
def test_skip_stop_keeps_its_generations_evaluations(trained_model):
    # Clause (ask/tell): at max_consecutive_skips, ask proposes nothing after that individual and
    # tell stops the run with an "EDHiE" reason, so the generation's earlier evaluations are kept.
    first = _stepper(trained_model, max_consecutive_skips=1)
    se = _evaluator()
    first.start(se, seed=0)
    generation = _proposals(first.ask(se))
    assert len(generation) < 10 or not generation[-1][1]  # precondition: a skip in generation 1
    assert not se.should_stop  # the stop belongs to tell, not ask
    run = _search(trained_model, max_consecutive_skips=1)
    assert run.stop_reason.startswith("EDHiE")
    assert [tuple(run.models[k].expr) for k in run.trace] == [t for t, evaluate in generation if evaluate]


def test_config_round_trip_keeps_skip_settings():
    config = _tiny(max_consecutive_skips=7, skip_evaluated=False, skip_rejected=False, training_seed=3).config.to_dict()
    rebuilt = EDHiE.from_config(config)
    settings = (rebuilt.max_consecutive_skips, rebuilt.skip_evaluated, rebuilt.skip_rejected, rebuilt.training_seed)
    assert settings == (7, False, False, 3)


def test_from_config_defaults_for_older_configs():
    approach = EDHiE.from_config({"latent_size": 8})
    settings = (approach.max_consecutive_skips, approach.skip_evaluated, approach.skip_rejected, approach.training_seed)
    assert settings == (10000, True, True, None)


# --------------------------------------------------------------------------- #
# Step interface (checkpoint stage 4)
# --------------------------------------------------------------------------- #
def _proposals(candidates):
    """(tokens, evaluate) per proposed candidate, None for a None entry; bare token lists and Nodes are evaluated."""
    wrapped = [c if c is None or isinstance(c, Candidate) else Candidate(c) for c in candidates]
    return [
        None
        if c is None
        else (tuple(c.expr if isinstance(c.expr, list) else c.expr.to_list(symbol_library=_lib())), c.evaluate)
        for c in wrapped
    ]


def _run(approach, seed):
    se = _evaluator(seed=0)  # seeds constant fitting, so models compare exactly
    approach.search(se, seed=seed)
    return list(se.trace), [m.to_dict() for m in se.models]


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_search_with_same_seed_repeats_its_run(trained_model):
    # Regression: pymoo's generator was unseeded. One object for all runs: start resets the state.
    approach = _stepper(trained_model, max_consecutive_skips=200)
    first, again, other_seed = _run(approach, 0), _run(approach, 0), _run(approach, 1)
    assert again == first
    assert other_seed[0] != first[0]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ask_proposes_evaluated_and_repeated_expressions_as_lookups(trained_model):
    first, se = _stepper(trained_model), _evaluator()
    first.start(se, seed=0)
    proposals = _proposals(first.ask(se))
    assert any(not evaluate for _, evaluate in proposals)  # precondition: generation 1 repeats one
    # On an empty evaluator an expression is evaluated exactly at its first occurrence.
    assert [evaluate for _, evaluate in proposals] == [
        tokens not in [earlier for earlier, _ in proposals[:i]] for i, (tokens, _) in enumerate(proposals)
    ]
    # An expression evaluated before the generation becomes a lookup; nothing else changes.
    looked_up = next(tokens for tokens, evaluate in proposals if evaluate)
    se_with_model = _evaluator()
    se_with_model.evaluate_expr(list(looked_up))
    second = _stepper(trained_model)
    second.start(se_with_model, seed=0)
    assert _proposals(second.ask(se_with_model)) == [(t, e and t != looked_up) for t, e in proposals]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ask_stops_proposing_at_max_consecutive_skips(trained_model):
    # Clause: after max_consecutive_skips skips in a row ask proposes nothing more (a lookup still
    # counts and is proposed); the next ask stops the run. Without skip_rejected every decoded
    # individual is proposed, so lookups are the only skips.
    unlimited, se = _stepper(trained_model, skip_rejected=False), _evaluator()
    unlimited.start(se, seed=0)
    generation = _proposals(unlimited.ask(se))
    assert len(generation) == 10  # precondition: every individual decoded
    lookups = [not evaluate for _, evaluate in generation]
    first_pair = next(i for i in range(1, 10) if lookups[i - 1] and lookups[i])
    assert any(lookups[: first_pair - 1])  # precondition: a single lookup before, which must not stop
    limited, se = _stepper(trained_model, skip_rejected=False, max_consecutive_skips=2), _evaluator()
    limited.start(se, seed=0)
    candidates = limited.ask(se)
    assert _proposals(candidates) == generation[: first_pair + 1]
    assert not se.should_stop
    limited.tell(se, candidates, [None] * len(candidates))
    assert not se.should_stop
    assert limited.ask(se) == []
    assert se.stop_reason.startswith("EDHiE")


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_skip_stop_survives_an_interrupt_in_its_step(tmp_path, trained_model):
    # Regression: an interrupt in the step that reached max_consecutive_skips lost the skip stop, so
    # the resumed run went on past where a straight run stops.
    def session(signal_after=None, **kwargs):
        approach = _stepper(trained_model, skip_rejected=False, max_consecutive_skips=2)
        se = _evaluator(seed=0, **kwargs)
        if signal_after is not None:
            se.register_callbacks(SignalAfter(signal_after))
        with se.stop_on_signals():
            approach.run(se, seed=0)
        return se

    straight = session()
    path = tmp_path / "run.sqlite"
    first = session(signal_after=1, checkpoint=path)
    assert first.stop_reason == "interrupted"
    assert list(first.trace) == list(straight.trace)  # precondition: the limit is reached in the first step
    first.close()
    resumed = session(checkpoint=path, resume=True)
    assert resumed.stop_reason == straight.stop_reason
    assert resumed.stop_reason.startswith("EDHiE")
    assert ([m.to_dict() for m in resumed.models], list(resumed.trace)) == (
        [m.to_dict() for m in straight.models],
        list(straight.trace),
    )
    resumed.close()


def test_max_consecutive_skips_below_one_raises():
    with pytest.raises(ValueError, match="max_consecutive_skips"):
        _tiny(max_consecutive_skips=0)


def _rejecting_evaluator():
    return SR_evaluator(
        _X,
        _X[:, 0],
        symbol_library=_lib(),
        max_evaluations=30,
        success_threshold=-1,
        required_constraints=[MaxTokens(0)],  # rejects everything
    )


def test_rejected_individuals_are_proposed_as_lookups(trained_model):
    # Clause: a rejected individual is a lookup (one entry per individual), so a generation of only
    # rejected individuals is a step of its own; the skip stop still ends the run, with no evaluation.
    approach, se = _stepper(trained_model, max_consecutive_skips=25), _rejecting_evaluator()
    approach.start(se, seed=0)
    assert [evaluate for _, evaluate in _proposals(approach.ask(se))] == [False] * 10
    assert not se.should_stop
    se = _rejecting_evaluator()
    approach.search(se, seed=0)  # 25 skips in a row: in the third generation
    assert se.stop_reason.startswith("EDHiE")
    assert se.total_evaluations == 0


def _global_rng_states():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _restore_global_rng_states(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def test_search_state_round_trips_without_weights(trained_model):
    original, se = _stepper(trained_model), _evaluator()
    original.start(se, seed=0)
    candidates = original.ask(se)
    original.tell(se, candidates, [None] * len(candidates))  # all inf: survival ties use the GA's generator
    state = original.get_search_state()
    largest = max(trained_model.parameters(), key=lambda p: p.numel())
    assert largest.detach().numpy().tobytes() not in state
    assert len(state) < len(pickle.dumps(trained_model.state_dict()))
    restored = _stepper(copy.deepcopy(trained_model))  # equal weights, another object
    restored.set_search_state(state)
    rngs = _global_rng_states()
    expected = _proposals(original.ask(se))
    _restore_global_rng_states(rngs)
    assert expected
    candidates = restored.ask(se)
    assert _proposals(candidates) == expected
    restored.tell(se, candidates, [None] * len(candidates))
    assert not se.should_stop  # no skip stop was restored with the state


def _second_generation(model):
    approach, se = _stepper(model), _evaluator()
    approach.start(se, seed=0)
    candidates = approach.ask(se)
    approach.tell(se, candidates, [None] * len(candidates))  # all inf: survival ties use the GA's generator
    return _proposals(approach.ask(se))


def test_start_with_same_seed_repeats_the_second_generation(trained_model):
    # Regression (fast form of test_search_with_same_seed_repeats_its_run): pymoo's generator was unseeded.
    assert _second_generation(trained_model) == _second_generation(trained_model)


def _model_of(se, candidate):
    """Evaluate a proposal as the step loop does and return its model."""
    candidate = candidate if isinstance(candidate, Candidate) else Candidate(candidate)
    if candidate.evaluate:
        se.evaluate_expr(candidate.expr)
    return se.get_model(candidate.expr)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tell_selects_by_the_errors(trained_model):
    # Clause (tell): a proposed individual gets its model's error. Metamorphic: from the same state
    # and global random states, telling the real models or none must lead to different next generations.
    se = _evaluator(seed=0)
    with_errors, without = _stepper(trained_model), _stepper(trained_model)
    with_errors.start(se, seed=0)
    candidates = with_errors.ask(se)
    without.start(se, seed=0)
    assert _proposals(without.ask(se)) == _proposals(candidates)  # precondition: the same generation
    after_ask = _global_rng_states()
    models = [_model_of(se, c) for c in candidates]
    assert any(m is not None and np.isfinite(m.error) for m in models)  # precondition: something to select by
    _restore_global_rng_states(after_ask)
    with_errors.tell(se, candidates, models)
    next_with_errors = _proposals(with_errors.ask(se))
    _restore_global_rng_states(after_ask)
    without.tell(se, candidates, [None] * len(candidates))
    assert _proposals(without.ask(se)) != next_with_errors


def _started_state(model):
    approach = _stepper(model)
    approach.start(_evaluator(), seed=0)
    return approach.get_search_state()


def _start_without_model(model):
    _tiny().start(_evaluator(), seed=0)


def _ask_before_start(model):
    _stepper(model).ask(_evaluator())


def _state_before_start(model):
    _stepper(model).get_search_state()


def _set_state_without_model(model):
    _tiny().set_search_state(_started_state(model))


def _set_state_of_another_model(model):
    other = copy.deepcopy(model)  # same architecture, one weight changed
    with torch.no_grad():
        next(other.parameters()).add_(1.0)
    _stepper(other).set_search_state(_started_state(model))


@pytest.mark.parametrize(
    "call, error, match",
    [
        (_start_without_model, RuntimeError, "HVAE"),
        (_ask_before_start, RuntimeError, "start"),
        (_state_before_start, RuntimeError, "start"),
        (_set_state_without_model, RuntimeError, "HVAE"),
        (_set_state_of_another_model, ValueError, "HVAE"),
    ],
    ids=["start-no-model", "ask-before-start", "state-before-start", "set-state-no-model", "set-state-other-model"],
)
def test_step_interface_errors(trained_model, call, error, match):
    with pytest.raises(error, match=match):
        call(trained_model)


# Acceptance: 2N calls straight == N calls, stop, resume to 2N (class docstring: resumable runs).
_SESSION = textwrap.dedent(
    """
    import sys

    import numpy as np

    sys.path.insert(0, sys.argv[1])
    from strategies import SignalAfter

    from SRToolkit.approaches.EDHiE import EDHiE
    from SRToolkit.evaluation.sr_evaluator import SR_evaluator
    from SRToolkit.utils.symbol_library import SymbolLibrary


    def session(weights, checkpoint=None, resume=False, signals=(), snapshot_interval=300.0):
        X = np.random.default_rng(0).random((10, 2)) + 0.5
        lib = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)
        approach = EDHiE(latent_size=8, max_expression_length=10, population_size=10, verbose=False,
                         max_consecutive_skips=200, weights_path=weights)  # a fresh object each session
        # No MaxTokens: with it the tiny model runs out of new expressions before the budget.
        se = SR_evaluator(X, X[:, 0], symbol_library=lib, max_evaluations=40, seed=0, success_threshold=-1,
                          checkpoint=checkpoint, resume=resume, snapshot_interval=snapshot_interval)
        for n in signals:
            se.register_callbacks(SignalAfter(n))
        try:
            with se.stop_on_signals():
                approach.run(se, seed=0)
        except KeyboardInterrupt:  # a crash: keep the checkpoint as it is
            se.close()
            raise
        return se


    if __name__ == "__main__":
        checkpoint, mode, signal_after, weights = sys.argv[2:6]
        signals = (int(signal_after),) if int(signal_after) else ()
        session(weights, checkpoint, resume=mode == "resume", signals=signals).close()
    """
)
# Inside a step: with seed 0, steps end after evaluations ..., 21, 23, ... (most generations evaluate
# one new expression and look up the rest), so the step with the 22nd evaluation also does the 23rd.
_N = 22


def _session(*args, **kwargs):
    namespace = {"__name__": "session"}
    exec(_SESSION, namespace)
    return namespace["session"](*args, **kwargs)


def _run_session(script, weights, path, mode, signal_after):
    tests_dir = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, str(script), tests_dir, str(path), mode, str(signal_after), str(weights)],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.slow  # each case runs EDHiE three times (about 20 s)
@pytest.mark.parametrize(
    "case",
    ["interrupt", "crash", "fresh-process"],
    ids=["same-process-interrupt", "same-process-crash", "fresh-process-interrupt"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_resume_equals_straight_run(tmp_path, trained_model, case):
    weights, path = tmp_path / "hvae.pt", tmp_path / "run.sqlite"
    _stepper(trained_model).save_adapted_state(str(weights))
    if case == "interrupt":
        first = _session(str(weights), path, signals=(_N,))
        assert first.stop_reason == "interrupted"
        assert len(first.trace) > _N  # the generation holding call N was completed: N fell inside it
        first.close()
    elif case == "crash":  # a second signal aborts inside the step; resume rewinds to the snapshot before it
        with pytest.raises(KeyboardInterrupt):
            _session(str(weights), path, signals=(_N, _N + 1), snapshot_interval=1e-6)
        assert ModelStore(path, "read").snapshot.num_calls < _N
    else:
        script = tmp_path / "session.py"
        script.write_text(_SESSION)
        _run_session(script, weights, path, "new", _N)
        assert len(SR_results.from_checkpoint(path)[0].trace) > _N
    if case == "fresh-process":
        _run_session(script, weights, path, "resume", 0)
    else:
        _session(str(weights), path, resume=True).close()
    straight = _session(str(weights))
    resumed = SR_results.from_checkpoint(path)[0]
    assert [m.to_dict() for m in resumed.all_models] == [m.to_dict() for m in straight.models]
    assert resumed.trace == list(straight.trace)


def _extension_dataset(budget):
    X = np.random.default_rng(0).random((10, 2)) + 0.5
    lib = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)
    return SR_dataset(X, lib, y=X[:, 0], dataset_name="ds", max_evaluations=budget, success_threshold=-1, seed=0)


# Reason: clause (stage 6: an extension after a budget that cut a generation equals a run with the
# larger budget from the start; the cut generation is redone from the snapshot before it).
@pytest.mark.slow  # runs EDHiE three times
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_extension_after_a_cut_generation_equals_straight_run(tmp_path, trained_model):
    weights = tmp_path / "hvae.pt"
    _stepper(trained_model).save_adapted_state(str(weights))

    def approach():
        return EDHiE(
            latent_size=8,
            max_expression_length=10,
            population_size=10,
            verbose=False,
            max_consecutive_skips=200,
            weights_path=str(weights),
        )

    first = _extension_dataset(_N).evaluate_approach(
        approach(), checkpoint_dir=tmp_path, keep_checkpoint=True, verbose=False, snapshot_interval=1e-9
    )[0]
    assert first.stop_reason == "max_evaluations"
    path = first.checkpoint["path"]
    assert ModelStore(path, "read").snapshot.num_calls < _N  # the budget cut a generation
    extended = _extension_dataset(_N).extend_run(approach(), path, max_evaluations=40)[0]
    straight = _extension_dataset(40).evaluate_approach(approach(), verbose=False)[0]
    assert [m.to_dict() for m in extended.all_models] == [m.to_dict() for m in straight.all_models]
    assert extended.trace == straight.trace
