import gc
import operator
import random
import sqlite3
import threading
import weakref

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from strategies import STORE_POOL, result_answers, result_dict, store_calls

from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation import model_store
from SRToolkit.evaluation.callbacks import SR_callback
from SRToolkit.evaluation.model_store import ModelStore, Snapshot
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.types import EvalResult, ModelResult, RunSummary


def _model(expr: str, error: float = 1.0, **kwargs) -> ModelResult:
    return ModelResult(expr=expr.split(), error=error, **kwargs)


def _filled(n: int, path=None) -> ModelStore:
    # A store with models "X_0" .. "X_{n-1}" with errors 0 .. n-1, each called once.
    store = ModelStore(path)
    for i in range(n):
        store.add([f"X_{i}"], "", _model(f"X_{i}", float(i)))
    return store


def _committed(path) -> tuple[int, int]:
    # (models, trace rows) visible to a separate connection, i.e. what survives a crash now.
    with sqlite3.connect(path) as con:
        return tuple(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("models", "trace"))


_X2 = np.array([[1.0], [2.0], [3.0]])
_y2 = np.array([2.0, 4.0, 6.0])


class _StopOnNextCall(SR_callback):
    def on_expr_evaluated(self, event):
        return "callback"  # a string stops with that reason


# ── Lookup and add ─────────────────────────────────────────────────────────────


class TestLookupAndAdd:
    def test_lookup_returns_index_error_status_of_added_model(self):
        store = _filled(2)
        assert store.add(["X_9"], "", _model("X_9", np.inf, status="invalid")) == 2
        assert store.lookup(["X_9"], "") == (2, np.inf, "invalid")
        assert store.lookup(["X_1"], "") == (1, 1.0, "ok")

    def test_same_tokens_with_another_spec_key_are_separate(self):
        store = ModelStore()
        store.add(["C_0"], "", _model("C_0", 1.0))
        assert store.lookup(["C_0"], "k") is None
        store.add(["C_0"], "k", _model("C_0", 2.0))
        assert (store.lookup(["C_0"], ""), store.lookup(["C_0"], "k")) == ((0, 1.0, "ok"), (1, 2.0, "ok"))

    def test_add_and_add_call_record_the_trace(self):
        store = _filled(2)
        store.add_call(0)
        assert (len(store.models), list(store.trace)) == (2, [0, 1, 0])

    def test_model_reads_back_equal(self):
        spec = Constant(label="C_0", prior=UniformSampling(0, 1), bounds=(0.0, 1.0))
        model = _model(
            "C_0 * X_0",
            np.inf,
            parameters=np.array([0.5]),
            status="undefined",
            status_detail="nan",
            ranking_data={"iterations": np.int64(3), "loss": np.float64(0.5)},
            free_constants=(spec,),
        )
        store = ModelStore()
        store.add(model.expr, "k", model)
        assert store.models[0].to_dict() == model.to_dict()
        # numpy scalars come back as Python scalars
        data = store.models[0].ranking_data
        assert (type(data["iterations"]), type(data["loss"])) == (int, float)

    def test_unsaveable_model_stores_nothing(self):
        store = _filled(1)
        with pytest.raises(TypeError, match=r"ranking_data\['bad'\]"):
            store.add(["X_5"], "", _model("X_5", ranking_data={"bad": object()}))
        with pytest.raises(sqlite3.IntegrityError):
            store.add(["X_0"], "", _model("X_0"))
        assert (len(store.models), list(store.trace), store.lookup(["X_5"], "")) == (1, [0], None)
        assert store.add(["X_5"], "", _model("X_5")) == 1  # the store is still consistent

    def test_invalid_exprs(self):
        store = _filled(1)
        store.add(["X_0", "X_1"], "", _model("X_0 X_1", np.inf, status="invalid"))
        store.add(["(", "X_1"], "", _model("( X_1", np.inf, status="invalid"))
        assert store.invalid_exprs() == ["X_0X_1", "(X_1"]


# ── Views ──────────────────────────────────────────────────────────────────────


class TestViews:
    def test_models_view_reads_like_a_list(self):
        store = _filled(4)
        models = store.models
        assert len(models) == 4
        assert (models[0].expr, models[-1].expr) == (["X_0"], ["X_3"])
        assert type(models[1:3]) is list and [m.error for m in models[1:3]] == [1.0, 2.0]
        assert [m.error for m in models[::-2]] == [3.0, 1.0]
        assert [m.error for m in models] == [0.0, 1.0, 2.0, 3.0]
        assert (models[np.int64(1)].expr, store.trace[np.intp(-1)]) == (["X_1"], 3)  # numpy integers
        for index in (4, -5):
            with pytest.raises(IndexError):
                models[index]
        for index in (1.0, "1"):
            with pytest.raises(TypeError):
                models[index]

    def test_trace_view_reads_like_a_list(self):
        store = _filled(3)
        store.add_call(1)
        trace = store.trace
        assert (len(trace), trace[0], trace[-1], trace[1:3], list(trace)) == (4, 0, 1, [1, 2], [0, 1, 2, 1])
        assert type(trace[1:3]) is list
        for index in (4, -5):
            with pytest.raises(IndexError):
                trace[index]

    def test_views_equal_sequences_with_equal_items(self):
        store = _filled(2)
        assert store.trace == [0, 1] and store.trace == (0, 1)
        assert store.trace != [0] and store.trace != [1, 0] and store.trace != "01"
        assert ModelStore().models == [] and ModelStore().models != None  # noqa: E711
        assert store.trace != 2  # a non-sequence compares unequal instead of raising
        assert store.models == [_model("X_0", 0.0), _model("X_1", 1.0)]
        assert store.models != [_model("X_0", 0.0), _model("X_1", 2.0)]

    def test_models_view_equals_models_with_array_parameters(self):
        store = ModelStore()
        model = _model("C * X_0 + C", parameters=np.array([1.0, 2.0]))
        store.add(model.expr, "", model)
        assert store.models == [_model("C * X_0 + C", parameters=np.array([1.0, 2.0]))]
        assert store.models != [_model("C * X_0 + C", parameters=np.array([1.0, 3.0]))]

    @pytest.mark.parametrize("view", ["models", "trace"])
    @pytest.mark.parametrize(
        "change",
        [lambda v: operator.setitem(v, 0, v[0]), lambda v: operator.delitem(v, 0)],
        ids=["setitem", "delitem"],
    )
    def test_views_are_read_only(self, view, change):
        store = _filled(1)
        with pytest.raises(TypeError, match="support item"):
            change(getattr(store, view))
        assert (len(store.models), list(store.trace)) == (1, [0])

    def test_views_see_later_additions(self):
        store = _filled(1)
        models, trace = store.models, store.trace
        store.add(["X_1"], "", _model("X_1"))
        store.add_call(0)
        assert (len(models), models[-1].expr, list(trace)) == (2, ["X_1"], [0, 1, 0])

    def test_changing_a_read_model_saves_nothing(self):
        store = _filled(1)
        store.models[0].error = 5.0
        assert store.models[0] is not store.models[0]
        assert (store.models[0].error, store.lookup(["X_0"], "")) == (0.0, (0, 0.0, "ok"))


# ── Checkpoint file ────────────────────────────────────────────────────────────


class TestCheckpoint:
    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ModelStore("")

    @pytest.mark.parametrize(
        "create",
        [ModelStore, lambda path: SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), checkpoint=path)],
        ids=["store", "evaluator"],
    )
    def test_existing_file_raises_and_is_kept(self, tmp_path, create):
        path = tmp_path / "run.sqlite"
        path.write_text("keep")
        with pytest.raises(FileExistsError):
            create(path)
        assert path.read_text() == "keep"

    def test_unwritable_path_raises(self, tmp_path):
        with pytest.raises(sqlite3.OperationalError):
            ModelStore(tmp_path / "missing" / "run.sqlite")

    def test_commit_writes_pending_calls_in_wal_mode(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = _filled(2, str(path))
        store.add_call(0)
        assert _committed(path) == (0, 0)
        store.commit()
        assert _committed(path) == (2, 3)
        with sqlite3.connect(path) as con:
            assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_commit_in_memory_keeps_the_store(self):
        store = _filled(2)
        store.commit()
        assert (store.models, list(store.trace), store.lookup(["X_1"], "")) == (
            [_model("X_0", 0.0), _model("X_1", 1.0)],
            [0, 1],
            (1, 1.0, "ok"),
        )

    def test_commit_due_after_1000_new_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_store.time, "monotonic", lambda: 0.0)
        path = tmp_path / "run.sqlite"
        store = _filled(999, path)
        store.add_call(0)  # cache hits do not count toward the 1000
        assert not store.commit_due
        store.add(["X_999"], "", _model("X_999"))
        assert store.commit_due
        assert len(ModelStore(path, "read").models) == 0  # the store never commits on its own
        store.commit()
        assert not store.commit_due
        for i in range(1000, 1999):  # counted again from the last commit
            store.add([f"X_{i}"], "", _model(f"X_{i}"))
        assert not store.commit_due
        store.add(["X_1999"], "", _model("X_1999"))
        assert store.commit_due

    def test_commit_due_30_seconds_after_last_commit(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        store = ModelStore(tmp_path / "run.sqlite")  # opened at t = 0
        now[0] = 30.0
        assert not store.commit_due  # 30 s since opening, but nothing new
        store.add(["X_0"], "", _model("X_0"))
        assert store.commit_due
        store.commit()
        now[0] = 59.0
        store.add_call(0)
        assert not store.commit_due  # 29 s since the last commit
        now[0] = 60.0
        assert store.commit_due
        store.commit()
        now[0] = 90.0
        assert not store.commit_due  # 30 s since the last commit, but nothing new

    @pytest.mark.parametrize(
        "budget, callback, last_call, reason",
        [
            (3, None, lambda se: se.evaluate_expr(["X_0", "*", "X_0"]), "max_evaluations"),
            (-1, None, lambda se: se.evaluate_expr(["X_0"]), "success_threshold"),
            (-1, None, lambda se: se.stop("done"), "done"),
            (-1, _StopOnNextCall, lambda se: se.evaluate_expr(["X_0", "*", "X_0"]), "callback"),
        ],
        ids=["budget", "threshold", "stop", "callback"],
    )
    def test_evaluator_commits_when_it_stops(self, tmp_path, budget, callback, last_call, reason):
        path = tmp_path / "run.sqlite"
        se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), max_evaluations=budget, checkpoint=path)
        se.evaluate_expr(["X_0", "+", "X_0"])
        se.evaluate_expr(["X_0", "*", "X_0"])
        assert _committed(path) == (0, 0)
        se.register_callbacks(None if callback is None else callback())
        last_call(se)
        assert se.stop_reason == reason
        assert _committed(path) == (len(se.models), len(se.trace))

    def test_evaluator_commits_when_results_are_collected(self, tmp_path):
        path = tmp_path / "run.sqlite"
        se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), checkpoint=path, success_threshold=-1)
        se.evaluate_expr(["X_0", "+", "X_0"])
        se.get_results()
        assert _committed(path) == (1, 1)

    # Reason: implied (the store does not tie the evaluator to the thread that created it, as the
    # old lists did not; e.g. an approach evaluating from a worker thread).
    @pytest.mark.parametrize("resume", [False, True], ids=["created", "resumed"])
    def test_evaluator_works_from_another_thread(self, tmp_path, resume):
        path = tmp_path / "run.sqlite"
        if resume:
            SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), checkpoint=path).close()
        se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), checkpoint=path, resume=resume)
        worker = threading.Thread(target=se.evaluate_expr, args=(["X_0"],))
        worker.start()
        worker.join()
        assert (len(se.models), list(se.trace), se.models[0].expr) == (1, [0], ["X_0"])

    def test_checkpoint_and_memory_runs_agree(self, tmp_path):
        def run(checkpoint):
            se = SR_evaluator(_X2, _y2, seed=0, success_threshold=-1, checkpoint=checkpoint)
            for expr in (["C", "*", "X_0"], ["X_0", "+", "C"], ["C", "*", "X_0"], ["X_0", "X_0"]):
                se.evaluate_expr(expr)
            return [m.to_dict() for m in se.models], list(se.trace), se.invalid, result_dict(se.get_results()[0])

        assert run(tmp_path / "run.sqlite") == run(None)

    def test_failing_constructor_leaves_no_file(self, tmp_path):
        path = tmp_path / "run.sqlite"
        with pytest.raises(ValueError):
            SR_evaluator(np.array([[1.0], [2.0]]), checkpoint=path)  # RMSE needs y or a ground truth
        assert not path.exists()


# Reason: implied (a dropped evaluator frees its database at once, without waiting for the garbage
# collector; grids create many evaluators).
def test_dropped_evaluator_frees_its_store():
    se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
    se.evaluate_expr(["X_0"])
    store = weakref.ref(se._store)
    gc.disable()
    try:
        del se
        assert store() is None
    finally:
        gc.enable()


# ── Modes, header, close, snapshots (checkpoint stages 2a, 2c) ─────────────────────────


def _saved(path) -> None:
    # A closed checkpoint with the calls X_0 (error 0), X_1 (error 1), bad (invalid), X_0: trace [0, 1, 2, 0].
    store = ModelStore(path)
    store.add(["X_0"], "", _model("X_0", 0.0))
    store.add(["X_1"], "", _model("X_1", 1.0))
    store.add(["bad"], "", _model("bad", np.inf, status="invalid"))
    store.add_call(0)
    store.close()


def _write_text(path):
    path.write_text("keep")


def _write_other_version(path):
    store = ModelStore(path)
    store.set_header("format_version", 999)
    store.close()


def _write_other_sqlite(path):
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE t (x)")


class TestOpenModes:
    @pytest.mark.parametrize(
        "make_file, use_path, mode, error, match",
        [
            (None, False, "resume", ValueError, "mode"),
            (None, False, "read", ValueError, "mode"),
            (_saved, True, "append", ValueError, "mode"),
            (None, True, "resume", FileNotFoundError, None),
            (None, True, "read", FileNotFoundError, None),
            (_write_text, True, "resume", ValueError, "checkpoint"),
            (_write_text, True, "read", ValueError, "checkpoint"),
            (_write_other_sqlite, True, "resume", ValueError, "checkpoint"),
            (_write_other_version, True, "read", ValueError, "format version 999"),
        ],
        ids=[
            "none-resume",
            "none-read",
            "bad-mode",
            "missing-resume",
            "missing-read",
            "text-resume",
            "text-read",
            "other-sqlite",
            "other-version",
        ],
    )
    def test_open_errors(self, tmp_path, make_file, use_path, mode, error, match):
        path = tmp_path / "run.sqlite"
        if make_file is not None:
            make_file(path)
        before = path.read_bytes() if path.exists() else None
        with pytest.raises(error, match=match):
            ModelStore(path if use_path else None, mode)
        assert (path.read_bytes() if path.exists() else None) == before  # never changes or deletes a file

    def test_read_mode_sees_committed_data_and_never_blocks_writer(self, tmp_path):
        path = tmp_path / "run.sqlite"
        writer = _filled(2, path)
        writer.commit()
        writer.add(["X_2"], "", _model("X_2", 2.0))  # pending
        reader = ModelStore(path, "read")
        assert (reader.models, list(reader.trace)) == ([_model("X_0", 0.0), _model("X_1", 1.0)], [0, 1])
        with pytest.raises(sqlite3.OperationalError):
            reader.add(["X_9"], "", _model("X_9"))
        with pytest.raises(sqlite3.OperationalError):
            reader.add_call(0)
        with pytest.raises(sqlite3.OperationalError):
            reader.set_header("a", 1)
        with pytest.raises(sqlite3.OperationalError):
            reader.save_snapshot(b"state", b"rngs", [])
        with pytest.raises(sqlite3.OperationalError):
            reader.rewind()
        writer.commit()  # not blocked by the open reader
        assert len(reader.models) == 3
        reader.commit()
        reader.close()

    def test_consistent_read_sees_one_state(self, tmp_path):
        path = tmp_path / "run.sqlite"
        writer = _filled(2, path)
        writer.commit()
        reader = ModelStore(path, "read")
        with reader.consistent_read():
            assert len(reader.models) == 2
            writer.add(["X_2"], "", _model("X_2", 2.0))
            writer.commit()  # not blocked, but not seen inside the block
            assert (len(reader.models), list(reader.trace)) == (2, [0, 1])
        assert len(reader.models) == 3
        reader.close()


class TestHeader:
    def test_header_round_trips_through_commit(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = ModelStore(path)
        reader = ModelStore(path, "read")
        assert store.get_header("a", 7) == 7
        store.set_header("a", {"x": (1, 2)})
        assert store.get_header("a") == {"x": [1, 2]}  # as json.loads gives it, before the commit too
        assert reader.get_header("a", 7) == 7
        store.commit()
        assert reader.get_header("a", 7) == {"x": [1, 2]}
        store.set_header("a", None)
        store.commit()
        assert (store.get_header("a", 7), reader.get_header("a", 7)) == (7, 7)

    def test_set_header_rejects_unsaveable_value(self):
        store = ModelStore()
        store.set_header("a", 1)
        with pytest.raises(TypeError, match="JSON"):
            store.set_header("a", object())
        assert store.get_header("a") == 1


class _FailingCommits:
    # A connection whose commits fail while fail[0] is set (a full disk); everything else is delegated.
    def __init__(self, con, fail):
        object.__setattr__(self, "_con", con)
        object.__setattr__(self, "_fail", fail)

    def execute(self, sql, *args):
        if self._fail[0] and sql.strip().upper().startswith(("COMMIT", "END")):
            raise sqlite3.OperationalError("database or disk is full")
        return self._con.execute(sql, *args)

    def commit(self):
        if self._fail[0]:
            raise sqlite3.OperationalError("database or disk is full")
        return self._con.commit()

    def __getattr__(self, name):
        return getattr(self._con, name)

    def __setattr__(self, name, value):
        setattr(self._con, name, value)

    def __enter__(self):
        return self._con.__enter__()

    def __exit__(self, *exc):
        return self._con.__exit__(*exc)


class _InterruptAfterCommit:
    # A connection that raises KeyboardInterrupt right after its next COMMIT, as a signal there would.
    def __init__(self, con):
        object.__setattr__(self, "_con", con)
        object.__setattr__(self, "armed", False)

    def execute(self, sql, *args):
        result = self._con.execute(sql, *args)
        if self.armed and sql == "COMMIT":
            object.__setattr__(self, "armed", False)
            raise KeyboardInterrupt
        return result

    def __getattr__(self, name):
        return getattr(self._con, name)


class TestCommitAndClose:
    def test_close_after_interrupt_between_commit_and_begin(self, tmp_path, monkeypatch):
        connect = sqlite3.connect
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: _InterruptAfterCommit(connect(*a, **kw)))
        path = tmp_path / "run.sqlite"
        store = _filled(2, path)
        store._con.armed = True
        with pytest.raises(KeyboardInterrupt):
            store.commit()  # committed, but no new transaction was begun
        store.close()
        assert len(ModelStore(path, "read").models) == 2

    @pytest.mark.parametrize("trigger", ["commit", "close"])
    def test_failed_commit_rolls_back_and_closes(self, tmp_path, monkeypatch, trigger):
        fail = [False]
        connect = sqlite3.connect
        monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: _FailingCommits(connect(*a, **kw), fail))
        path = tmp_path / "run.sqlite"
        store = _filled(2, path)
        store.commit()
        store.add(["X_2"], "", _model("X_2"))
        fail[0] = True
        with pytest.raises(sqlite3.Error):
            getattr(store, trigger)()
        fail[0] = False
        with pytest.raises(RuntimeError):
            store.lookup(["X_0"], "")
        with pytest.raises(RuntimeError):
            store.commit()
        assert len(ModelStore(path, "read").models) == 2  # the committed models stay
        assert len(ModelStore(path, "resume").models) == 2  # and can be resumed

    def test_close_completes_the_file_and_blocks_use(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = _filled(2, path)
        models = store.models
        store.close()
        assert not (tmp_path / "run.sqlite-wal").exists()
        assert not (tmp_path / "run.sqlite-shm").exists()
        assert (ModelStore(path, "read").models, list(ModelStore(path, "read").trace)) == (
            [_model("X_0", 0.0), _model("X_1", 1.0)],
            [0, 1],
        )
        with pytest.raises(sqlite3.ProgrammingError):
            store.lookup(["X_0"], "")
        with pytest.raises(sqlite3.ProgrammingError):
            store.add_call(0)
        with pytest.raises(sqlite3.ProgrammingError):
            len(models)
        assert store.close() is None  # closing again does nothing


class _FailingSchema:
    # A connection that fails writing the header after the schema (e.g. a full disk), when the
    # file, its -wal and its -shm exist; everything else is delegated.
    def __init__(self, con):
        self._con = con

    def execute(self, sql, *args):
        if sql.startswith("INSERT INTO header"):
            raise sqlite3.OperationalError("database or disk is full")
        return self._con.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._con, name)


def test_failed_create_removes_the_file(tmp_path, monkeypatch):
    connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: _FailingSchema(connect(*a, **kw)))
    with pytest.raises(sqlite3.OperationalError):
        ModelStore(tmp_path / "run.sqlite")
    assert list(tmp_path.iterdir()) == []  # no file, -wal or -shm


class TestSnapshots:
    def test_save_snapshot_replaces_and_commits(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = _filled(2, path)
        store.save_snapshot(b"a", b"r", [])
        store.add(["X_2"], "", _model("X_2", 2.0))
        store.save_snapshot(b"b", b"r", [{"callback_class": "c", "state": {"n": 1}}])
        expected = Snapshot(b"b", b"r", [{"callback_class": "c", "state": {"n": 1}}], 3, 3)
        assert (store.snapshot, ModelStore(path, "read").snapshot) == (expected, expected)  # committed

    def test_save_snapshot_rejects_unsaveable_callbacks(self, tmp_path):
        store = _filled(1, tmp_path / "run.sqlite")
        store.save_snapshot(b"a", b"r", [])
        with pytest.raises(TypeError):
            store.save_snapshot(b"b", b"r", [{"state": object()}])
        assert store.snapshot == Snapshot(b"a", b"r", [], 1, 1)

    @pytest.mark.parametrize(
        "with_snapshot, models, trace, next_index",
        [(True, [_model("X_0", 0.0), _model("X_1", 1.0)], [0, 1, 0], 2), (False, [], [], 0)],
        ids=["with-snapshot", "without-snapshot"],
    )
    def test_rewind_cuts_back_to_snapshot(self, tmp_path, with_snapshot, models, trace, next_index):
        path = tmp_path / "run.sqlite"
        store = _filled(2, path)
        store.add_call(0)
        if with_snapshot:
            store.save_snapshot(b"s", b"r", [])  # at 3 calls and 2 models
        store.add(["X_2"], "", _model("X_2", 2.0))
        store.add_call(1)
        store.close()
        resumed = ModelStore(path, "resume")
        resumed.rewind()
        assert (resumed.models, list(resumed.trace), resumed.lookup(["X_2"], "")) == (models, trace, None)
        reader = ModelStore(path, "read")  # rewind commits
        assert (reader.models, list(reader.trace)) == (models, trace)
        assert resumed.add(["X_9"], "", _model("X_9")) == next_index

    def test_snapshot_due_after_interval_with_something_new(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        store = _filled(1000, tmp_path / "run.sqlite")
        assert (store.commit_due, store.snapshot_due(300)) == (True, False)  # new models do not count
        store.commit()
        now[0] = 299.0
        assert not store.snapshot_due(300)
        now[0] = 300.0
        assert (store.snapshot_due(300), store.snapshot_due(301)) == (True, False)  # a commit is not a snapshot
        store.save_snapshot(b"s", b"r", [])
        now[0] = 900.0
        assert not store.snapshot_due(300)  # nothing new since the snapshot
        store.add_call(0)
        assert store.snapshot_due(300) and not store.snapshot_due(601)  # counted from the snapshot

    @pytest.mark.parametrize(
        "errors, expected",
        [([], np.inf), ([3.0, np.inf, 1.0], 1.0), ([np.inf], np.inf)],
        ids=["empty", "smallest", "only-invalid"],
    )
    def test_min_error(self, errors, expected):
        store = ModelStore()
        for i, error in enumerate(errors):
            store.add([f"X_{i}"], "", _model(f"X_{i}", error))
        assert store.min_error == expected


# ── Summary ────────────────────────────────────────────────────────────────────


def _fill(store, calls):
    # One counted call per (pool_index, error); a repeated expression is a cache hit.
    for pool_index, error in calls:
        tokens = STORE_POOL[pool_index]
        found = store.lookup(tokens, "")
        if found is None:
            store.add(tokens, "", ModelResult(list(tokens), error, status="ok" if error < np.inf else "invalid"))
        else:
            store.add_call(found[0])
    return store


class TestSummary:
    # Reason: clause (summary: top_k models ranked as by top_models, and a RunSummary of counts,
    # incumbents with their full-run index, status counts and Pareto front).
    def test_summary_fields_hand_computed(self):
        # Calls: X_0 .5 (i0) | X_1 inf (i1) | hit i0 | C*X_0 .2 (i2) | X_0+C .2 (i3) | C .5 (i4) |
        # X_0*X_0+C .05 (i5).
        store = _fill(ModelStore(), [(0, 0.5), (1, np.inf), (0, 0.5), (3, 0.2), (4, 0.2), (2, 0.5), (5, 0.05)])
        m = list(store.models)
        top, summary = store.summary(3)
        assert top == [m[5], m[2], m[3]]  # the tie at 0.2 (same length) goes to the first evaluated
        assert summary == RunSummary(
            num_evaluated=6,
            evaluation_calls=7,
            incumbents=[(1, 0, m[0]), (4, 2, m[2]), (7, 5, m[5])],
            status_counts={"ok": 5, "invalid": 1, "rejected": 0, "undefined": 0, "error": 0},
            call_status_counts={"ok": 6, "invalid": 1, "rejected": 0, "undefined": 0, "error": 0},
            pareto_front=[m[0], m[2], m[5]],  # length 1: X_0 (first of the 0.5 tie with C)
        )
        assert (
            list(summary.status_counts)
            == list(summary.call_status_counts)
            == ["ok", "invalid", "rejected", "undefined", "error"]
        )

    # Reason: clause (the summary equals what the full result derives, so a compact result answers
    # like the full one: ties, failed models, fewer finite models than top_k, empty run).
    @given(store_calls(), st.integers(1, 8))
    @example([], 3)
    def test_summary_result_answers_like_full_result(self, calls, top_k):
        store = _fill(ModelStore(), calls)
        top, summary = store.summary(top_k)
        full = EvalResult(list(store.models), list(store.trace), "", top_k=top_k, success_threshold=0.3)
        compact = EvalResult(top, [], "", top_k=top_k, success_threshold=0.3, summary=summary)
        assert top == full.top_models(top_k)
        assert compact.top_models(-1) == full.top_models(top_k)
        budgets = range(-1, len(calls) + 3)
        np.testing.assert_equal(result_answers(compact, top_k, budgets), result_answers(full, top_k, budgets))

    # Reason: clause (Raises: ValueError if top_k < 1).
    @pytest.mark.parametrize("top_k", [0, -1])
    def test_summary_rejects_top_k_below_one(self, top_k):
        with pytest.raises(ValueError, match="top_k"):
            _filled(2).summary(top_k)

    # Reason: clause ("in mode='read', what was committed").
    def test_summary_in_read_mode_sees_committed_only(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = _filled(3, path)
        store.commit()
        store.add(["X_8"], "", _model("X_8", 0.0))
        store.add_call(0)
        reader = ModelStore(path, "read")
        top, summary = reader.summary(1)
        assert (summary.num_evaluated, summary.evaluation_calls) == (3, 3)
        assert top == [_model("X_0", 0.0)]
        reader.close()
        store.close()


@pytest.mark.parametrize(
    "draw",
    [
        lambda: random.random(),
        lambda: float(np.random.random()),
        lambda: float(pytest.importorskip("torch").rand(1)),
    ],
    ids=["random", "numpy", "torch"],
)
def test_rng_round_trip(draw):
    # Private helpers, tested directly: the torch state is not reachable without a torch approach.
    draw()
    saved = model_store._capture_rngs()
    first = draw()
    model_store._restore_rngs(saved)
    assert draw() == first
