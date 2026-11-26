"""Tests for the step interface of SRToolkit.approaches.sr_approach: the default step loop,
SR_approach.run and resuming from snapshots (checkpoint stage 2c)."""

import pickle
import random
import signal
import subprocess
import sys
import textwrap
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest
from strategies import (
    ScriptedApproach,
    SignalAfter,
    SteppedApproach,
    result_dict,
    send_signal,
)

from SRToolkit.approaches import ProGED
from SRToolkit.approaches.sr_approach import Candidate, SR_approach
from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
from SRToolkit.evaluation import model_store
from SRToolkit.evaluation.model_store import ModelStore
from SRToolkit.evaluation.results import SR_results
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.expression_tree import tokens_to_tree
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.symbol_library import SymbolLibrary

_X = np.array([[1.0], [2.0], [3.0]])
_Y = 2 * _X[:, 0]


def _se(checkpoint=None, **kwargs):
    args = dict(seed=0, success_threshold=-1, initialization="mean")
    args.update(kwargs)
    return SR_evaluator(_X, _Y, checkpoint=checkpoint, **args)


def _result(se):
    return result_dict(se.get_results()[0])


@pytest.mark.parametrize(
    "approach, expected",
    [(SteppedApproach(), True), (ScriptedApproach(), False), (ProGED(), True)],
    ids=["stepped", "search-only", "ProGED"],
)
def test_supports_steps(approach, expected):
    assert approach.supports_steps is expected


def test_default_search_runs_the_step_loop():
    se, approach = _se(), SteppedApproach(stop_after_asks=3)
    approach.search(se, seed=0)
    assert len(se.trace) == 6  # three steps of two candidates
    told = [pair for candidates, results in approach.told for pair in zip(candidates, results)]
    assert told == [(se.models[i].expr, se.models[i]) for i in se.trace]  # in order, as evaluated
    assert se.get_results()[0].stop_reason == "returned"  # the empty ask ended the loop


def test_mid_batch_stop_tells_only_evaluated_prefix():
    se, approach = _se(max_evaluations=3), SteppedApproach()
    approach.search(se, seed=0)
    assert [len(exprs) for exprs, _ in approach.told] == [2, 1]
    assert (len(se.trace), se.stop_reason) == (3, "max_evaluations")


# Reason: clause (the final snapshot is taken only after a complete step; a step cut by a stop
# keeps the snapshot before it).
@pytest.mark.parametrize("budget, expected", [(4, 4), (5, 4)], ids=["complete", "cut"])
def test_final_snapshot_only_after_a_complete_step(tmp_path, monkeypatch, budget, expected):
    now = [0.0]
    monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
    path = tmp_path / "run.sqlite"
    se = _se(path, max_evaluations=budget, snapshot_interval=1.0)
    approach = SteppedApproach()

    def tick(*args):  # a snapshot is due before every step
        now[0] += 1.0
        return SteppedApproach.ask(approach, *args)

    approach.ask = tick
    approach.run(se, seed=0)
    se.close()
    assert ModelStore(path, "read").snapshot.num_calls == expected


# Regression (dev-check audit): a session that runs no step (a finished run resumed directly) kept
# its old snapshot state but saved it with the newer call count, so a later extension rewound to the
# wrong point.
def test_session_without_steps_keeps_the_snapshot(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
    path = tmp_path / "run.sqlite"
    se = _se(path, max_evaluations=5, snapshot_interval=1.0)
    approach = SteppedApproach()

    def tick(*args):  # a snapshot is due before every step
        now[0] += 1.0
        return SteppedApproach.ask(approach, *args)

    approach.ask = tick
    approach.run(se, seed=0)  # the budget cuts the third step: the snapshot stays at 4 calls
    se.close()
    se = _se(path, resume=True, max_evaluations=5)  # finished: stays stopped, runs no step
    SteppedApproach().run(se, seed=0)
    se.close()
    assert ModelStore(path, "read").snapshot.num_calls == 4


# Reason: clause (step loop: an empty ask ends the loop normally, so the final snapshot is taken);
# boundary: an empty first ask (nothing evaluated) still ran a step.
@pytest.mark.parametrize("asks, calls", [(2, 4), (0, 0)], ids=["after-two-steps", "first-ask-empty"])
def test_search_ended_by_an_empty_ask_takes_the_final_snapshot(tmp_path, asks, calls):
    path = tmp_path / "run.sqlite"
    se, approach = _se(path), SteppedApproach(stop_after_asks=asks)  # batches of 2
    approach.run(se, seed=0)
    snapshot = ModelStore(path, "read").snapshot
    assert (se.stop_reason, len(se.trace), snapshot.num_calls, pickle.loads(snapshot.state)) == (
        "returned",
        calls,
        calls,
        {"asks": asks},
    )


def test_interrupt_in_step_completes_step_and_snapshots(tmp_path):
    path = tmp_path / "run.sqlite"
    se, approach = _se(path), SteppedApproach()
    se.register_callbacks(SignalAfter(1))  # inside the first step of two candidates
    with se.stop_on_signals():
        approach.run(se, seed=0)
    assert (se.stop_reason, len(se.trace), [len(e) for e, _ in approach.told]) == ("interrupted", 2, [2])
    snapshot = ModelStore(path, "read").snapshot  # the final snapshot, at the step boundary
    assert (snapshot.num_calls, pickle.loads(snapshot.state)) == (2, {"asks": 1})


class _Log(SteppedApproach):
    # Logs (method, state) for every step-interface call, and the evaluator start and tell get.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log, self.evaluators = [], []

    def start(self, sr_evaluator, seed=None):
        self.evaluators.append(sr_evaluator)
        state = super().start(sr_evaluator, seed)
        self.log.append(("start", state))
        return state

    def ask(self, sr_evaluator, state):
        self.log.append(("ask", state))
        return super().ask(sr_evaluator, state)

    def tell(self, sr_evaluator, state, candidates, results):
        self.log.append(("tell", state))
        self.evaluators.append(sr_evaluator)
        super().tell(sr_evaluator, state, candidates, results)

    def dump_search_state(self, state):
        self.log.append(("dump", state))
        return super().dump_search_state(state)


# Reason: clause (module step loop and start: start once before the first ask; with a checkpoint one
# dump before the first step; the same state object reaches every ask, tell and dump of a session).
def test_step_loop_passes_one_state_object_in_documented_order(tmp_path):
    se = _se(tmp_path / "run.sqlite", max_evaluations=4, snapshot_interval=None)
    approach = _Log()  # batches of 2: two steps spend the budget, then the final snapshot
    approach.run(se, seed=0)
    assert [name for name, _ in approach.log] == ["start", "dump", "ask", "tell", "ask", "tell", "dump"]
    state = approach.log[0][1]
    assert all(s is state for _, s in approach.log)
    assert approach.evaluators == [se, se, se]  # start and two tells


class _Unpicklable(SteppedApproach):
    def start(self, sr_evaluator, seed=None):
        return {**super().start(sr_evaluator, seed), "lock": threading.Lock()}


# Reason: clause (dump_search_state: with a checkpoint the state is dumped once before the first step,
# so a state that cannot be saved fails before any budget is spent; without one it is never dumped).
@pytest.mark.parametrize("checkpoint", [True, False], ids=["checkpoint", "in-memory"])
def test_unpicklable_state_fails_before_first_step_only_with_checkpoint(tmp_path, checkpoint):
    path = tmp_path / "run.sqlite"
    se, approach = _se(path if checkpoint else None), _Unpicklable(stop_after_asks=2)
    if checkpoint:
        with pytest.raises(TypeError, match="pickle"):
            approach.run(se, seed=0)
        evaluations = se.total_evaluations
        se.close()
        assert (approach.told, evaluations, ModelStore(path, "read").snapshot) == ([], 0, None)
    else:
        approach.run(se, seed=0)
        assert (se.stop_reason, se.total_evaluations) == ("returned", 4)


class _ResumeOnly(_Log):
    # Fails if a resumed run starts or searches anew; load_search_state disturbs the global random
    # states, which the framework restores after it.
    def start(self, sr_evaluator, seed=None):
        raise AssertionError("start called on resume")

    def search(self, sr_evaluator, seed=None):
        raise AssertionError("search called on resume")

    def load_search_state(self, data):
        random.seed(999)
        np.random.seed(999)
        state = super().load_search_state(data)
        self.log.append(("load", state))
        return state


# Reason: clause (run: on resume neither start nor search is called; load_search_state runs before
# the global random states are restored; the loaded object reaches every ask, tell and dump).
@pytest.mark.filterwarnings("ignore:.*callbacks differ:UserWarning")  # the interrupting callback is not re-registered
def test_resume_calls_neither_start_nor_search_and_keeps_the_loaded_state(tmp_path):
    straight = _se(max_evaluations=8)
    SteppedApproach().run(straight, seed=0)
    path = tmp_path / "run.sqlite"
    first = _se(path, max_evaluations=8)
    first.register_callbacks(SignalAfter(4))  # at the end of the second step
    with first.stop_on_signals():
        SteppedApproach().run(first, seed=0)
    first.close()
    se, approach = _se(path, resume=True, max_evaluations=8), _ResumeOnly()
    approach.run(se, seed=0)
    assert _result(se) == _result(straight)
    assert [name for name, _ in approach.log[:3]] == ["load", "dump", "ask"]  # dumped before the first step
    state = approach.log[0][1]
    assert all(s is state for _, s in approach.log)


class _BadLoad(SteppedApproach):
    def load_search_state(self, data):
        raise ValueError("bad state")


# Reason: clause (run: errors of load_search_state propagate).
@pytest.mark.filterwarnings("ignore:.*callbacks differ:UserWarning")  # the interrupting callback is not re-registered
def test_run_propagates_load_search_state_errors(tmp_path):
    path = tmp_path / "run.sqlite"
    first = _se(path, max_evaluations=8)
    first.register_callbacks(SignalAfter(2))
    with first.stop_on_signals():
        SteppedApproach().run(first, seed=0)
    first.close()
    with pytest.raises(ValueError, match="bad state"):
        _BadLoad().run(_se(path, resume=True, max_evaluations=8), seed=0)


# Reason: clause (dump/load_search_state pickle by default; the approach's own generator belongs in the
# state; dump must not change the state).
def test_default_state_methods_round_trip_with_own_generator():
    approach = SteppedApproach()
    state = {"asks": 3, "rng": np.random.default_rng(0)}
    loaded = approach.load_search_state(approach.dump_search_state(state))
    assert loaded["asks"] == 3
    np.testing.assert_array_equal(loaded["rng"].random(3), state["rng"].random(3))


class _Mixed(SteppedApproach):
    # Proposes Candidates (one with a free-constant spec, one canonicalised), a Node and a token
    # list, then stops.
    def ask(self, sr_evaluator, state):
        if state["asks"]:
            return []
        state["asks"] += 1
        self.proposed = [
            Candidate(["C_0", "*", "X_0"], [_SPEC]),
            Candidate(["C_0", "+", "X_0"], canonicalise_constants=True),
            tokens_to_tree(["X_0", "*", "X_0"], sr_evaluator.symbol_library),
            ["X_0"],
        ]
        return self.proposed


_SPEC = Constant(label="C_0", bounds=(0.0, 10.0))


def test_step_loop_takes_candidates_nodes_and_token_lists():
    se, approach = _se(), _Mixed()
    approach.search(se, seed=0)
    [(candidates, results)] = approach.told
    assert all(c is p for c, p in zip(candidates, approach.proposed, strict=True))  # as ask returned them
    assert results == list(se.models)
    assert [m.expr for m in results] == [["C_0", "*", "X_0"], ["C", "+", "X_0"], ["X_0", "*", "X_0"], ["X_0"]]
    assert results[0].free_constants == (_SPEC,)


class _SkipsAndNone(SteppedApproach):
    # Step 1 evaluates X_0; step 2 only skips X_0 (evaluated) and X_1 (never), with a None entry
    # (not an expression) between them; step 3 evaluates X_1.
    _STEPS = [
        [["X_0"]],
        [Candidate(["X_0"], skip="a"), None, Candidate(["X_1"], skip="b")],
        [["X_1"]],
    ]

    def ask(self, sr_evaluator, state):
        state["asks"] += 1
        return self._STEPS[state["asks"] - 1] if state["asks"] <= len(self._STEPS) else []


# Reason: clause (Candidate.skip: a skip is looked up, spending no budget; tell gets its stored model,
# or None if it was never evaluated, and None for a None entry).
def test_skip_and_none_candidates_spend_no_budget():
    se = SR_evaluator(np.ones((3, 2)), np.ones(3), seed=0, success_threshold=-1)
    approach = _SkipsAndNone()
    approach.search(se, seed=0)
    [(_, first), (_, skipped), _] = approach.told  # the step of skips did not end the run
    assert skipped == [first[0], None, None]
    assert ([m.expr for m in se.models], list(se.trace)) == ([["X_0"], ["X_1"]], [0, 1])


class _Steps(SteppedApproach):
    # Proposes the given steps in turn, then nothing; ask_skip: a skip counted from ask in every ask.
    def __init__(self, steps, ask_skip=None):
        super().__init__()
        self.steps, self.ask_skip = steps, ask_skip

    def ask(self, sr_evaluator, state):
        if self.ask_skip is not None:
            sr_evaluator.count_skip(self.ask_skip)
        state["asks"] += 1
        return self.steps[state["asks"] - 1] if state["asks"] <= len(self.steps) else []


# Step 1: a skip, a None entry, an evaluation (resets the skips in a row) and a second skip; step 2
# evaluates X_1.
_SKIP_STEPS = [
    [Candidate(["X_0"], skip="a"), None, ["X_0"], Candidate(["X_1"], skip="b")],
    [["X_1"]],
]


# Reason: clause (the step loop counts a None entry as "not_expression" and a candidate's skip reason,
# in order; an evaluation resets the skips in a row; reaching max_consecutive_skips stops the run at
# that entry, and tell gets the entries up to it).
@pytest.mark.parametrize(
    "steps, limit, told, skips, calls, stop_reason",
    [
        (_SKIP_STEPS, 3, 4, {"a": 1, "not_expression": 1, "b": 1}, 2, "returned"),
        (_SKIP_STEPS, 2, 2, {"a": 1, "not_expression": 1}, 0, "max_consecutive_skips"),
    ],
    ids=["reset-by-evaluation", "limit-reached"],
)
def test_step_loop_counts_skips_in_order(steps, limit, told, skips, calls, stop_reason):
    se = SR_evaluator(np.ones((3, 2)), np.ones(3), seed=0, success_threshold=-1, max_consecutive_skips=limit)
    approach = _Steps(steps)
    approach.run(se, seed=0)
    assert (len(approach.told[0][1]), se.skips, len(se.trace), se.stop_reason) == (told, skips, calls, stop_reason)


# Reason: clause (Skips: a count_skip from ask is counted before the step's candidates, so it can stop
# the run before any of them is evaluated; tell then gets empty lists).
def test_count_skip_in_ask_can_stop_before_the_step():
    se = SR_evaluator(np.ones((3, 2)), np.ones(3), seed=0, success_threshold=-1, max_consecutive_skips=1)
    approach = _Steps([[["X_0"]]], ask_skip="x")
    approach.run(se, seed=0)
    assert (approach.told, len(se.trace), se.skips, se.stop_reason) == (
        [([], [])],
        0,
        {"x": 1},
        "max_consecutive_skips",
    )


@pytest.mark.parametrize("skip", ["", 3], ids=["empty", "not-str"])
def test_candidate_skip_must_be_a_non_empty_str(skip):
    with pytest.raises(ValueError, match="skip"):
        Candidate(["X_0"], skip=skip)


class _FieldLookup(SteppedApproach):
    # Step 1 evaluates X_0*X_1/X_0 simplified, C_0*X_0 with a spec and C_0+X_0 canonicalised; step 2
    # skips (looks up) each with the same fields and without them.
    _EXPR = ["X_0", "*", "X_1", "/", "X_0"]
    _STEPS = [
        [
            Candidate(_EXPR, simplify_expr=True),
            Candidate(["C_0", "*", "X_0"], [_SPEC]),
            Candidate(["C_0", "+", "X_0"], canonicalise_constants=True),
        ],
        [
            Candidate(_EXPR, simplify_expr=True, skip="s"),
            Candidate(_EXPR, skip="s"),
            Candidate(["C_0", "*", "X_0"], [_SPEC], skip="s"),
            Candidate(["C_0", "*", "X_0"], skip="s"),
            Candidate(["C_0", "+", "X_0"], canonicalise_constants=True, skip="s"),
            Candidate(["C_0", "+", "X_0"], skip="s"),
        ],
    ]

    def ask(self, sr_evaluator, state):
        state["asks"] += 1
        return self._STEPS[state["asks"] - 1] if state["asks"] <= len(self._STEPS) else []


# Reason: clause (Candidate: simplify_expr, free_constants and canonicalise_constants are used for
# evaluating; a skip is looked up "with its specs, simplified and canonicalised as above").
def test_candidate_fields_are_used_for_evaluation_and_lookup():
    se = SR_evaluator(np.ones((3, 2)), np.ones(3), seed=0, success_threshold=-1)
    approach = _FieldLookup()
    approach.search(se, seed=0)
    [(_, [simplified, with_spec, canonical]), (_, looked_up)] = approach.told
    assert [simplified.expr, with_spec.free_constants, canonical.expr] == [["X_1"], (_SPEC,), ["C", "+", "X_0"]]
    assert looked_up == [simplified, None, with_spec, None, canonical, None]


class _FailingSecondTell(SteppedApproach):
    def tell(self, sr_evaluator, state, candidates, results):
        super().tell(sr_evaluator, state, candidates, results)
        if len(self.told) == 2:
            raise RuntimeError("boom")


def test_due_snapshot_is_taken_between_steps(tmp_path, monkeypatch):
    # Snapshots due by time are taken at step boundaries; here the run then fails, so the due
    # snapshot after the first step is the one left.
    now = [0.0]
    monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
    path = tmp_path / "run.sqlite"
    se, approach = _se(path), _FailingSecondTell()
    now[0] = 300.0  # the default interval: every boundary with new calls is due
    with pytest.raises(RuntimeError, match="boom"):
        approach.run(se, seed=0)
    se.close()
    snapshot = ModelStore(path, "read").snapshot
    assert (snapshot.num_calls, pickle.loads(snapshot.state)) == (2, {"asks": 1})


class _FailingTell(SteppedApproach):
    def tell(self, sr_evaluator, state, candidates, results):
        raise RuntimeError("boom")


def test_no_final_snapshot_after_an_exception(tmp_path):
    path = tmp_path / "run.sqlite"
    se = _se(path)
    with pytest.raises(RuntimeError, match="boom"):
        _FailingTell().run(se, seed=0)
    se.close()
    assert ModelStore(path, "read").snapshot is None


@pytest.mark.parametrize("make", [SteppedApproach, ScriptedApproach], ids=["stepped", "search-only"])
def test_run_without_snapshot_searches(make):
    run, search = _se(max_evaluations=6), _se(max_evaluations=6)
    make().run(run, seed=0)
    make().search(search, seed=0)
    assert _result(run) == _result(search)


def test_run_with_snapshot_needs_the_step_interface(tmp_path):
    path = tmp_path / "run.sqlite"
    se = _se(path)
    se.evaluate_expr(["X_0"])
    se.save_snapshot(b"state")
    se.close()
    with pytest.raises(ValueError, match="snapshot"):
        ScriptedApproach().run(_se(path, resume=True), seed=0)


# --------------------------------------------------------------------------- #
# Acceptance: ProGED, 2N calls straight == N calls, interrupt, resume to 2N
# --------------------------------------------------------------------------- #
_SESSION = textwrap.dedent(
    """
    import sys

    import numpy as np

    sys.path.insert(0, sys.argv[1])
    from strategies import SignalAfter

    from SRToolkit.approaches import ProGED
    from SRToolkit.dataset.background_knowledge import BackgroundKnowledge
    from SRToolkit.evaluation.sr_evaluator import SR_evaluator
    from SRToolkit.utils.symbol_library import SymbolLibrary


    def session(checkpoint=None, resume=False, signal_after=None):
        X = np.random.default_rng(0).random((10, 2)) + 0.5
        lib = SymbolLibrary.from_symbol_list(["+", "*", "C"], num_variables=2)
        approach = ProGED()  # a fresh object each session
        approach.adapt(X, BackgroundKnowledge(lib), ())
        se = SR_evaluator(X, X[:, 0] + X[:, 1], symbol_library=lib, max_evaluations=40, seed=0,
                          success_threshold=-1, checkpoint=checkpoint, resume=resume)
        if signal_after is not None:
            se.register_callbacks(SignalAfter(signal_after))
        with se.stop_on_signals():
            approach.run(se, seed=0)
        return se


    if __name__ == "__main__":
        se = session(sys.argv[2], resume=sys.argv[3] == "resume", signal_after=int(sys.argv[4]) or None)
        se.close()
    """
)


def _session(checkpoint=None, resume=False, signal_after=None):
    namespace = {"__name__": "session"}
    exec(_SESSION, namespace)
    return namespace["session"](checkpoint, resume, signal_after)


def _run_session(script, path, mode, signal_after):
    tests_dir = str(Path(__file__).resolve().parents[1])
    done = subprocess.run(
        [sys.executable, str(script), tests_dir, str(path), mode, str(signal_after)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize(
    "fresh_process", [False, pytest.param(True, marks=pytest.mark.slow)], ids=["same-process", "fresh-process"]
)
def test_proged_resume_equals_straight_run(tmp_path, fresh_process):
    path = tmp_path / "run.sqlite"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if fresh_process:
            script = tmp_path / "session.py"
            script.write_text(_SESSION)
            _run_session(script, path, "new", 20)
            assert len(SR_results.from_checkpoint(path)[0].trace) == 20  # interrupted at N
            _run_session(script, path, "resume", 0)
        else:
            first = _session(path, signal_after=20)
            assert len(first.trace) == 20
            first.close()
            _session(path, resume=True).close()
        straight = _session()
    resumed = SR_results.from_checkpoint(path)[0]
    assert [m.to_dict() for m in resumed.all_models] == [m.to_dict() for m in straight.models]
    assert resumed.trace == list(straight.trace)
    assert resumed.skips == straight.skips


class _NoSearch(SR_approach):  # neither search nor the step interface
    def __init__(self):
        super().__init__(SteppedApproach().config)


class _NoStart(SteppedApproach):
    start = SR_approach.start


# Reason: clause (search raises if the approach implements neither search nor ask, or ask without
# start; start and ask raise unless overridden).
@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: _NoSearch().search(_se(), seed=0), "step interface"),
        (lambda: _NoStart().search(_se(), seed=0), "implements ask but not start"),
        (lambda: SR_approach.start(SteppedApproach(), _se(), seed=0), "start"),
        (lambda: SR_approach.ask(SteppedApproach(), _se(), None), None),  # the type is all the spec says
    ],
    ids=["neither", "ask-without-start", "start-default", "ask-default"],
)
def test_unimplemented_step_interface_raises(call, match):
    with pytest.raises(NotImplementedError, match=match):
        call()


def _proged_session(path=None, resume=False, signal_on_skip=False):
    # ProGED on a small grammar that repeats itself, so samples are skipped as already evaluated.
    X = np.random.default_rng(0).random((10, 2)) + 0.5
    lib = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=2)
    approach = ProGED()
    approach.adapt(X, BackgroundKnowledge(lib), ())
    se = SR_evaluator(
        X,
        X[:, 0] + X[:, 1],
        symbol_library=lib,
        max_evaluations=40,
        seed=0,
        success_threshold=-1,
        checkpoint=path,
        resume=resume,
        max_consecutive_skips=3,  # small, so the skip limit decides where the run ends
    )
    if signal_on_skip:  # interrupt in the step of the second skip in a row, one before the limit
        get_model = se.get_model

        def lookup(*args, **kwargs):
            model = get_model(*args, **kwargs)
            if model is not None and se._skips_in_a_row == 1 and not se._interrupted:
                send_signal(signal.SIGINT)
            return model

        se.get_model = lookup
    with se.stop_on_signals():
        approach.run(se, seed=0)
    return se


def test_proged_interrupted_among_skips_resumes_as_straight_run(tmp_path):
    path = tmp_path / "run.sqlite"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = _proged_session(path, signal_on_skip=True)
        assert first.stop_reason == "interrupted"
        first.close()
        assert ModelStore(path, "read").get_header("snapshot_skips")["in_a_row"] == 2  # one before the limit
        _proged_session(path, resume=True).close()
        straight = _proged_session()
    assert straight.stop_reason == "max_consecutive_skips"
    resumed = SR_results.from_checkpoint(path)[0]
    assert [m.to_dict() for m in resumed.all_models] == [m.to_dict() for m in straight.models]
    assert resumed.trace == list(straight.trace)
    assert resumed.skips == straight.skips
