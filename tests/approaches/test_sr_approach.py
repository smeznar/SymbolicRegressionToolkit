"""Tests for the step interface of SRToolkit.approaches.sr_approach: the default step loop,
SR_approach.run and resuming from snapshots (checkpoint stage 2c)."""

import json
import pickle
import signal
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
from strategies import (
    ScriptedApproach,
    SignalAfter,
    StatelessSteppedApproach,
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
    [
        (SteppedApproach(), (True, True)),
        (StatelessSteppedApproach(), (True, False)),
        (ScriptedApproach(), (False, False)),
        (ProGED(), (True, True)),
    ],
    ids=["stepped", "stateless", "search-only", "ProGED"],
)
def test_supports_steps_and_snapshots(approach, expected):
    assert (approach.supports_steps, approach.supports_snapshots) == expected


@pytest.mark.parametrize("with_checkpoint", [False, True], ids=["in-memory", "checkpoint"])
def test_stateless_step_approach_runs_without_snapshots(tmp_path, monkeypatch, with_checkpoint):
    now = [0.0]
    monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
    path = tmp_path / "run.sqlite" if with_checkpoint else None
    se = _se(path, max_evaluations=6)
    now[0] = 1e9  # a snapshot would be due at every step
    StatelessSteppedApproach().run(se, seed=0)  # get_search_state would raise NotImplementedError
    assert (se.stop_reason, se.total_evaluations) == ("max_evaluations", 6)
    se.close()
    if with_checkpoint:
        assert ModelStore(path, "read").snapshot is None


def test_default_search_runs_the_step_loop():
    se, approach = _se(), SteppedApproach(stop_after_asks=3)
    approach.search(se, seed=0)
    assert approach.started is True
    assert len(se.trace) == 6  # three steps of two candidates
    told = [pair for candidates, results in approach.told for pair in zip(candidates, results)]
    assert told == [(se.models[i].expr, se.models[i]) for i in se.trace]  # in order, as evaluated
    assert se.get_results()[0].stop_reason == "returned"  # the empty ask ended the loop


def test_mid_batch_stop_tells_only_evaluated_prefix():
    se, approach = _se(max_evaluations=3), SteppedApproach()
    approach.search(se, seed=0)
    assert [len(exprs) for exprs, _ in approach.told] == [2, 1]
    assert (len(se.trace), se.stop_reason) == (3, "max_evaluations")


def test_interrupt_in_step_completes_step_and_snapshots(tmp_path):
    path = tmp_path / "run.sqlite"
    se, approach = _se(path), SteppedApproach()
    se.register_callbacks(SignalAfter(1))  # inside the first step of two candidates
    with se.stop_on_signals():
        approach.run(se, seed=0)
    assert (se.stop_reason, len(se.trace), [len(e) for e, _ in approach.told]) == ("interrupted", 2, [2])
    snapshot = ModelStore(path, "read").snapshot  # the final snapshot, at the step boundary
    assert (snapshot.num_calls, snapshot.state) == (2, approach.get_search_state())


class _Recording(SteppedApproach):
    # Records the evaluator each step-interface method gets.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluators = []

    def start(self, sr_evaluator, seed=None):
        self.evaluators.append(sr_evaluator)
        super().start(sr_evaluator, seed)

    def tell(self, sr_evaluator, candidates, results):
        self.evaluators.append(sr_evaluator)
        super().tell(sr_evaluator, candidates, results)


def test_step_interface_methods_get_the_evaluator():
    se, approach = _se(), _Recording(stop_after_asks=2)
    approach.search(se, seed=0)
    assert approach.evaluators == [se, se, se]  # start and two tells


class _Mixed(SteppedApproach):
    # Proposes Candidates (one with a free-constant spec, one canonicalised), a Node and a token
    # list, then stops.
    def ask(self, sr_evaluator):
        if self.asks:
            return []
        self.asks += 1
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


class _Lookups(SteppedApproach):
    # Step 1 evaluates X_0; step 2 only looks up X_0 (evaluated) and X_1 (never); step 3 evaluates X_1.
    _STEPS = [
        [["X_0"]],
        [Candidate(["X_0"], evaluate=False), Candidate(["X_1"], evaluate=False)],
        [["X_1"]],
    ]

    def ask(self, sr_evaluator):
        self.asks += 1
        return self._STEPS[self.asks - 1] if self.asks <= len(self._STEPS) else []


def test_lookup_candidates_spend_no_budget():
    se = SR_evaluator(np.ones((3, 2)), np.ones(3), seed=0, success_threshold=-1)
    approach = _Lookups()
    approach.search(se, seed=0)
    [(_, first), (_, lookups), _] = approach.told  # the step of lookups did not end the run
    assert lookups == [first[0], None]
    assert ([m.expr for m in se.models], list(se.trace)) == ([["X_0"], ["X_1"]], [0, 1])


class _FailingSecondTell(SteppedApproach):
    def tell(self, sr_evaluator, candidates, results):
        super().tell(sr_evaluator, candidates, results)
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
    assert (snapshot.num_calls, snapshot.state) == (2, pickle.dumps({"asks": 1}))


class _FailingTell(SteppedApproach):
    def tell(self, sr_evaluator, candidates, results):
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


@pytest.mark.parametrize("approach", [ScriptedApproach, StatelessSteppedApproach], ids=["search-only", "stateless"])
def test_run_with_snapshot_needs_state_methods(tmp_path, approach):
    path = tmp_path / "run.sqlite"
    se = _se(path)
    se.evaluate_expr(["X_0"])
    se.save_snapshot(b"state")
    se.close()
    with pytest.raises(ValueError, match="snapshot"):
        approach().run(_se(path, resume=True), seed=0)


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


class _NoSearch(SR_approach):  # neither search nor the step interface
    def __init__(self):
        super().__init__(SteppedApproach().config)


def test_search_without_step_interface_raises():
    with pytest.raises(NotImplementedError, match="step interface"):
        _NoSearch().search(_se(), seed=0)


class _NoStart(SteppedApproach):
    start = SR_approach.start


def test_step_interface_without_start_raises():
    with pytest.raises(NotImplementedError, match="implements ask but not start"):
        _NoStart().search(_se(), seed=0)


def _proged_session(path=None, resume=False, signal_on_skip=False):
    # ProGED on a small grammar that repeats itself, so samples are skipped as already evaluated.
    X = np.random.default_rng(0).random((10, 2)) + 0.5
    lib = SymbolLibrary.from_symbol_list(["+", "*"], num_variables=2)
    approach = ProGED(max_consecutive_skips=3)  # small, so the skip count decides where the run ends
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
    )
    if signal_on_skip:  # interrupt at the second skip in a row, one before the limit, in the middle of an ask
        get_model = se.get_model

        def lookup(*args, **kwargs):
            model = get_model(*args, **kwargs)
            if model is not None and approach._skipped == 1 and not se._interrupted:
                send_signal(signal.SIGINT)
            return model

        se.get_model = lookup
    with se.stop_on_signals():
        approach.run(se, seed=0)
    return se


def test_proged_interrupted_while_sampling_resumes_as_straight_run(tmp_path):
    path = tmp_path / "run.sqlite"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = _proged_session(path, signal_on_skip=True)
        assert first.stop_reason == "interrupted"
        first.close()
        assert json.loads(ModelStore(path, "read").snapshot.state)["skipped"] == 2  # mid-ask
        _proged_session(path, resume=True).close()
        straight = _proged_session()
    assert straight.stop_reason.startswith("ProGED")  # ended by skips
    resumed = SR_results.from_checkpoint(path)[0]
    assert [m.to_dict() for m in resumed.all_models] == [m.to_dict() for m in straight.models]
    assert resumed.trace == list(straight.trace)
