import gc
import operator
import sqlite3
import threading
import weakref

import numpy as np
import pytest

from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.evaluation import model_store
from SRToolkit.evaluation.callbacks import SRCallbacks
from SRToolkit.evaluation.model_store import ModelStore
from SRToolkit.evaluation.sr_evaluator import SR_evaluator
from SRToolkit.utils.free_constants import Constant
from SRToolkit.utils.types import ModelResult


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


class _StopOnNextCall(SRCallbacks):
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

    def test_commits_every_1000_new_models(self, tmp_path):
        path = tmp_path / "run.sqlite"
        store = _filled(999, path)
        store.add_call(0)  # cache hits do not count toward the 1000
        assert _committed(path) == (0, 0)
        store.add(["X_999"], "", _model("X_999"))
        assert _committed(path) == (1000, 1001)
        for i in range(1000, 1999):  # counted again from the last commit
            store.add([f"X_{i}"], "", _model(f"X_{i}"))
        assert _committed(path) == (1000, 1001)
        store.add(["X_1999"], "", _model("X_1999"))
        assert _committed(path) == (2000, 2001)

    def test_commits_30_seconds_after_the_last_commit(self, tmp_path, monkeypatch):
        now = [0.0]
        monkeypatch.setattr(model_store.time, "monotonic", lambda: now[0])
        path = tmp_path / "run.sqlite"
        store = _filled(1, path)
        now[0] = 29.0
        store.add_call(0)
        assert _committed(path) == (0, 0)
        now[0] = 30.0
        store.add_call(0)  # a cache hit commits too once the time is up
        assert _committed(path) == (1, 3)
        now[0] = 59.0  # 29 s after the last commit
        store.add(["X_1"], "", _model("X_1"))
        assert _committed(path) == (1, 3)
        now[0] = 60.0
        store.add(["X_2"], "", _model("X_2"))
        assert _committed(path) == (3, 5)

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
    def test_evaluator_works_from_another_thread(self, tmp_path):
        se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), checkpoint=tmp_path / "run.sqlite")
        worker = threading.Thread(target=se.evaluate_expr, args=(["X_0"],))
        worker.start()
        worker.join()
        assert (len(se.models), list(se.trace), se.models[0].expr) == (1, [0], ["X_0"])

    def test_checkpoint_and_memory_runs_agree(self, tmp_path):
        def run(checkpoint):
            se = SR_evaluator(_X2, _y2, seed=0, success_threshold=-1, checkpoint=checkpoint)
            for expr in (["C", "*", "X_0"], ["X_0", "+", "C"], ["C", "*", "X_0"], ["X_0", "X_0"]):
                se.evaluate_expr(expr)
            return [m.to_dict() for m in se.models], list(se.trace), se.invalid, se.get_results()[0].to_dict()

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
