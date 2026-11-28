"""Storage of an evaluator's models and trace in SQLite.

[Evaluator][SRToolkit.evaluation.evaluator.Evaluator] keeps every evaluated model and its
evaluation trace in a [ModelStore][SRToolkit.evaluation.model_store.ModelStore] and exposes them
as the read-only views [ModelsView][SRToolkit.evaluation.model_store.ModelsView] (``se.models``)
and [TraceView][SRToolkit.evaluation.model_store.TraceView] (``se.trace``).

Without a checkpoint the database is in memory, so nothing touches the disk and memory grows with
the number of models (about 1 KB each). With a checkpoint path it is a file, and memory stays
constant during the run: only the best error so far and a small page cache stay in RAM.
Collecting the results copies every model into RAM, unless only a summary is collected
([summary][SRToolkit.evaluation.model_store.ModelStore.summary], used by
``get_results(keep_all_models=False)``).

A model is saved as the JSON of [ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict],
with its error, status and length (number of tokens) in columns of their own, so a lookup and a
summary never parse the record of every model (a cache hit parses only the record it returns). Models
come back as new objects (numpy scalars in ``ranking_data`` as Python scalars), so changing one
saves nothing.

Only [evaluate_expr][SRToolkit.evaluation.evaluator.Evaluator.evaluate_expr] writes to the
store. Do not change the database or the file yourself: a cache hit returns the stored model and
the trace refers to models by position, so a changed or removed row makes later results wrong.

## Checkpoint file

The file layout is internal; read a checkpoint with ``mode="read"`` or
[Results.from_checkpoint][SRToolkit.evaluation.results.Results.from_checkpoint]. Besides the
models and trace it holds a header of JSON values the evaluator uses to check a resume (see
[Checkpoints][SRToolkit.evaluation.evaluator]) and at most one
[Snapshot][SRToolkit.evaluation.model_store.Snapshot].

The store never commits on its own: the evaluator commits when
[commit_due][SRToolkit.evaluation.model_store.ModelStore.commit_due] says so, when evaluation
stops, when results are collected, with each snapshot and on
[close][SRToolkit.evaluation.model_store.ModelStore.close]. A crash loses at most what came after
the last commit; after a failed commit (see [commit][SRToolkit.evaluation.model_store.ModelStore.commit])
the committed part stays and can be resumed. The file is in WAL mode, so another process can read
it during the run; keep the ``-wal`` and ``-shm`` files with it until the store is closed.

## Snapshots

[save_snapshot][SRToolkit.evaluation.model_store.ModelStore.save_snapshot] records the approach's
state between two steps with the number of calls and models at that point, replacing the previous
snapshot. A resumed store shows the whole saved run until
[rewind][SRToolkit.evaluation.model_store.ModelStore.rewind] deletes what came after the snapshot,
so the file never mixes two runs.
"""

import hashlib
import json
import math
import operator
import os
import pickle
import random
import sqlite3
import sys
import time
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast, overload

import numpy as np

from SRToolkit.utils.types import STATUSES, ModelResult, RunSummary, Status

# Version of the file layout, written to the header table.
_FORMAT_VERSION = 3
# A checkpoint is committed after this many new models or seconds, whichever comes first.
_COMMIT_MODELS = 1000
_COMMIT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE header (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE models (idx INTEGER PRIMARY KEY, key BLOB UNIQUE NOT NULL, error REAL, status TEXT, length INTEGER,
                     record TEXT);
CREATE TABLE trace (k INTEGER PRIMARY KEY, idx INTEGER NOT NULL);
CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK (id = 0), num_calls INTEGER, num_models INTEGER,
                       state BLOB, rngs BLOB, callbacks TEXT);
"""


@dataclass(frozen=True)
class Snapshot:
    """The point of a run that a resumed run continues from.

    Attributes:
        state: The approach's search state, as returned by its
            [dump_search_state][SRToolkit.approaches.approach.Approach.dump_search_state].
        rngs: The global random states (Python, NumPy and, if imported, PyTorch), as pickled bytes.
        callbacks: The callbacks' states (see
            [CallbackDispatcher.get_state][SRToolkit.evaluation.callbacks.CallbackDispatcher.get_state]).
        num_calls: The number of calls recorded at the snapshot.
        num_models: The number of models stored at the snapshot.
    """

    state: bytes
    rngs: bytes
    callbacks: list[dict]
    num_calls: int
    num_models: int


def _run_finished(path: str | os.PathLike) -> bool:
    # Whether a checkpoint holds a finished run: one stopped for any reason but an interrupt.
    store = ModelStore(path, "read")
    try:
        return store.get_header("stop_reason") not in (None, "interrupted")
    finally:
        store.close()


def _checkpoint_reference(store: "ModelStore", path: str | os.PathLike) -> dict[str, Any]:
    # The reference to the checkpoint at path that EvalResult.checkpoint holds.
    return {"path": os.path.abspath(path), "fingerprint": store.get_header("fingerprint", {})}


def _remove_checkpoint(path: str | os.PathLike) -> None:
    # Delete a checkpoint file with its -wal and -shm files, those that exist.
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(os.fspath(path) + suffix):
            os.remove(os.fspath(path) + suffix)


def _compact_checkpoint(path: str | os.PathLike) -> None:
    # Turn a closed checkpoint into one self-contained, minimal file (no -wal / -shm needed).
    con = sqlite3.connect(path, isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("VACUUM")
    finally:
        con.close()


def _capture_rngs() -> bytes:
    # The global random states: Python's random, NumPy's and, if PyTorch is imported, its CPU and
    # CUDA states, pickled. PyTorch is never imported here, so a run without it stays light.
    states: dict[str, Any] = {"random": random.getstate(), "numpy": np.random.get_state()}
    torch = sys.modules.get("torch")
    if torch is not None:
        states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            states["cuda"] = torch.cuda.get_rng_state_all()
    return pickle.dumps(states)


def _restore_rngs(rngs: bytes) -> None:
    # Restore the states returned by _capture_rngs.
    states = pickle.loads(rngs)
    random.setstate(states["random"])
    np.random.set_state(states["numpy"])
    if "torch" in states:
        import torch

        torch.set_rng_state(states["torch"])
        if "cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])


def _model_key(tokens: Sequence[str], spec_key: str) -> bytes:
    # 128-bit hash of the model's identity; collisions are negligible for any realistic run.
    return hashlib.blake2b(("\x1f".join(tokens) + "\x1e" + spec_key).encode(), digest_size=16).digest()


class ModelStore:
    """SQLite database of the models and trace of one evaluator (see the module docstring)."""

    def __init__(self, path: str | os.PathLike | None = None, mode: Literal["create", "resume", "read"] = "create"):
        """SQLite database of the models and trace of one evaluator.

        Args:
            path: The checkpoint file, or ``None`` (default) for a database in memory, which only
                ``mode="create"`` accepts.
            mode: ``"create"`` (default): create the file. ``"resume"``: open a checkpoint to
                continue its run; it shows the whole saved run until
                [rewind][SRToolkit.evaluation.model_store.ModelStore.rewind]. ``"read"``: open a
                checkpoint read-only, also while its run goes on; it sees what was committed before
                each call. ``add``, ``add_call``, ``set_header``, ``save_snapshot`` and ``rewind``
                then raise ``sqlite3.OperationalError``; ``commit`` and ``close`` work.

        A file the constructor began to create is removed again, with its ``-wal`` and ``-shm``
        files, if the constructor fails.

        Raises:
            ValueError: If ``path`` is empty, or ``None`` with a mode other than ``"create"``; if
                ``mode`` is none of the three; if the file is not a checkpoint, or one of another
                format version.
            FileExistsError: If ``mode="create"`` and a file exists at ``path``.
            FileNotFoundError: If ``mode`` is ``"resume"`` or ``"read"`` and no file exists at
                ``path``.
            sqlite3.OperationalError: If the file cannot be created or opened (e.g. its directory
                does not exist or is not writable).
        """
        if mode not in ("create", "resume", "read"):
            raise ValueError(f"[ModelStore] mode must be 'create', 'resume' or 'read', got {mode!r}.")
        if path is None and mode != "create":
            raise ValueError(f"[ModelStore] mode={mode!r} needs a path; only mode='create' accepts None.")
        if path is not None and not os.fspath(path):
            raise ValueError("[ModelStore] path must not be empty; pass None for a database in memory.")
        self._read = mode == "read"
        self._closed = False
        if mode == "create":
            if path is not None and os.path.exists(path):
                raise FileExistsError(f"[ModelStore] {os.fspath(path)!r} already exists.")
            con = None
            try:
                # Transactions are managed explicitly (isolation_level=None); one is always open.
                con = sqlite3.connect(
                    ":memory:" if path is None else path, isolation_level=None, check_same_thread=False
                )
                if path is not None:
                    con.execute("PRAGMA journal_mode=WAL")
                    con.execute("PRAGMA synchronous=NORMAL")
                con.executescript(_SCHEMA)
                con.execute("INSERT INTO header VALUES ('format_version', ?)", (json.dumps(_FORMAT_VERSION),))
            except BaseException:
                if con is not None:
                    con.close()
                if path is not None:
                    _remove_checkpoint(path)
                raise
            self._con = con
        else:
            assert path is not None  # checked above: only mode "create" accepts None
            if not os.path.exists(path):
                raise FileNotFoundError(f"[ModelStore] {os.fspath(path)!r} does not exist.")
            target = Path(path).absolute().as_uri() + "?mode=ro" if self._read else os.fspath(path)
            self._con = sqlite3.connect(target, uri=self._read, isolation_level=None, check_same_thread=False)
            try:
                row = self._con.execute("SELECT value FROM header WHERE key = 'format_version'").fetchone()
            except sqlite3.DatabaseError:
                row = None
            if row is None or json.loads(row[0]) != _FORMAT_VERSION:
                self._con.close()
                found = "no format version" if row is None else f"format version {json.loads(row[0])!r}"
                raise ValueError(
                    f"[ModelStore] {os.fspath(path)!r} is not a checkpoint of format version "
                    f"{_FORMAT_VERSION} ({found})."
                )
            if not self._read:
                self._con.execute("PRAGMA synchronous=NORMAL")
        self._num_models = 0 if mode == "create" else self._row_count("models", "idx")
        self._num_calls = 0 if mode == "create" else self._row_count("trace", "k")
        self._pending = 0
        self._dirty = False
        self._last_commit = time.monotonic()
        self._snapshot_dirty = False  # a new model or call since the last snapshot
        self._last_snapshot = self._last_commit
        if not self._read:
            self._con.execute("BEGIN")

    def _row_count(self, table: str, column: str) -> int:
        # Rows of a table numbered 0, 1, ... in column (an index lookup, unlike COUNT(*)).
        return self._con.execute(f"SELECT COALESCE(MAX({column}) + 1, 0) FROM {table}").fetchone()[0]

    @property
    def _model_count(self) -> int:
        if self._closed:
            self._con.execute("SELECT 1")  # raises: ProgrammingError if closed, RuntimeError if a commit failed
        return self._row_count("models", "idx") if self._read else self._num_models

    @property
    def _call_count(self) -> int:
        if self._closed:
            self._con.execute("SELECT 1")
        return self._row_count("trace", "k") if self._read else self._num_calls

    def _check_writable(self) -> None:
        if self._read:
            raise sqlite3.OperationalError("[ModelStore] the store was opened read-only (mode='read').")

    # The views are created on access, not stored: a stored view would form a reference cycle with
    # the store and keep a dropped evaluator's database open until the garbage collector runs.
    @property
    def models(self) -> "ModelsView":
        """[View][SRToolkit.evaluation.model_store.ModelsView] of the models, in the order added."""
        return ModelsView(self)

    @property
    def trace(self) -> "TraceView":
        """[View][SRToolkit.evaluation.model_store.TraceView] of the model index of each call, in order."""
        return TraceView(self)

    def lookup(self, tokens: Sequence[str], spec_key: str) -> tuple[int, float, Status] | None:
        """Find a stored model without parsing its record.

        Args:
            tokens: The model's expression tokens.
            spec_key: The key of its free-constant specs, as computed by the evaluator (``""``
                without specs). The same tokens with different specs are different models.

        Returns:
            ``(index, error, status)`` of the model, or ``None`` if it is not stored.
        """
        row = self._con.execute(
            "SELECT idx, error, status FROM models WHERE key = ?", (_model_key(tokens, spec_key),)
        ).fetchone()
        return None if row is None else (row[0], row[1], cast(Status, row[2]))

    def add(self, tokens: Sequence[str], spec_key: str, model: ModelResult) -> int:
        """Store a new model and record a call to it, atomically.

        Does not commit; see [commit_due][SRToolkit.evaluation.model_store.ModelStore.commit_due].

        Args:
            tokens: The model's expression tokens.
            spec_key: The key of its free-constant specs, as computed by the evaluator (``""``
                without specs). The same tokens with different specs are different models.
            model: The model; saved through
                [to_dict][SRToolkit.utils.types.ModelResult.to_dict].

        Returns:
            The model's index in ``models``.

        Raises:
            TypeError: If ``model`` cannot be saved (see
                [to_dict][SRToolkit.utils.types.ModelResult.to_dict]). Nothing is then stored.
            sqlite3.IntegrityError: If the model is already stored. Nothing is then stored.
        """
        self._check_writable()
        record = json.dumps(model.to_dict())
        index = self._num_models
        self._con.execute("SAVEPOINT add_model")
        try:
            self._con.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)",
                (index, _model_key(tokens, spec_key), float(model.error), model.status, len(model.expr), record),
            )
            self._con.execute("INSERT INTO trace VALUES (?, ?)", (self._num_calls, index))
        except BaseException:
            self._con.execute("ROLLBACK TO add_model")
            raise
        finally:
            self._con.execute("RELEASE add_model")
        self._num_models += 1
        self._num_calls += 1
        self._pending += 1
        self._dirty = self._snapshot_dirty = True
        return index

    def add_call(self, index: int) -> None:
        """Record a call to the already stored model at ``index`` (a cache hit).

        Does not commit; see [commit_due][SRToolkit.evaluation.model_store.ModelStore.commit_due].
        """
        self._check_writable()
        self._con.execute("INSERT INTO trace VALUES (?, ?)", (self._num_calls, index))
        self._num_calls += 1
        self._dirty = self._snapshot_dirty = True

    @property
    def commit_due(self) -> bool:
        """Whether a commit is due.

        It is due once at least 1000 new models were added since the last commit, or once 30
        seconds or more have passed since the last commit (or since the store was opened) and at
        least one new model or call was added. Header values and snapshots do not count.
        """
        return self._pending >= _COMMIT_MODELS or (
            self._dirty and time.monotonic() - self._last_commit >= _COMMIT_SECONDS
        )

    def snapshot_due(self, interval: float) -> bool:
        """Whether a snapshot is due: ``interval`` seconds or more have passed since the last
        snapshot saved by this store (or since it was opened) and at least one new model or call
        was added since.
        """
        return self._snapshot_dirty and time.monotonic() - self._last_snapshot >= interval

    def save_snapshot(
        self, state: bytes, rngs: bytes, callbacks: list[dict], headers: dict[str, Any] | None = None
    ) -> None:
        """Record a snapshot at the current number of calls and models, and commit.

        The snapshot replaces the previous one.

        Args:
            state: The approach's state.
            rngs: The pickled global random states.
            callbacks: The callbacks' states; must be accepted by ``json.dumps``.
            headers: Header values (key to value), written in the same commit as the snapshot: both
                are saved or neither. ``None``: no header values.

        Raises:
            TypeError: If ``callbacks`` or a header value cannot be saved as JSON. Nothing is then
                recorded.
            sqlite3.Error: If the commit fails (see
                [commit][SRToolkit.evaluation.model_store.ModelStore.commit]).
        """
        self._check_writable()
        saved_callbacks = json.dumps(callbacks)
        saved_headers = [(key, json.dumps(value)) for key, value in (headers or {}).items()]
        self._con.executemany("INSERT OR REPLACE INTO header VALUES (?, ?)", saved_headers)
        self._con.execute(
            "INSERT OR REPLACE INTO snapshot VALUES (0, ?, ?, ?, ?, ?)",
            (self._num_calls, self._num_models, state, rngs, saved_callbacks),
        )
        self.commit()
        self._snapshot_dirty = False
        self._last_snapshot = time.monotonic()

    @property
    def snapshot(self) -> Snapshot | None:
        """The last [Snapshot][SRToolkit.evaluation.model_store.Snapshot] saved, or ``None``."""
        row = self._con.execute(
            "SELECT state, rngs, callbacks, num_calls, num_models FROM snapshot WHERE id = 0"
        ).fetchone()
        if row is None:
            return None
        state, rngs, callbacks, num_calls, num_models = row
        return Snapshot(bytes(state), bytes(rngs), json.loads(callbacks), num_calls, num_models)

    @property
    def min_error(self) -> float:
        """The smallest error of the stored models, ``inf`` if there are none."""
        (error,) = self._con.execute("SELECT MIN(error) FROM models").fetchone()
        return float("inf") if error is None else error

    def summary(self, top_k: int) -> tuple[list[ModelResult], RunSummary]:
        """Return the best models and a summary of the run, without loading every model.

        What [EvalResult][SRToolkit.utils.types.EvalResult] would derive from all models and the
        trace: the best ``top_k`` models ranked as by
        [top_models][SRToolkit.utils.types.EvalResult.top_models], and a
        [RunSummary][SRToolkit.utils.types.RunSummary] equal to the counts, best-so-far history,
        status counts and Pareto front of the full result. Memory use grows with ``top_k`` and the
        sizes of the history and the front, not with the run. A model with a non-finite error
        counts as failed, as in ``EvalResult``. Sees what the store shows: in ``mode="read"``, what
        was committed.

        Args:
            top_k: The number of best models to return, at least 1.

        Returns:
            ``(top_models, summary)``: the best ``min(top_k, number of models)`` models, best first,
            and the summary.

        Raises:
            ValueError: If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"[ModelStore.summary] top_k must be at least 1, got {top_k}.")

        def model(index: int) -> ModelResult:
            (record,) = self._con.execute("SELECT record FROM models WHERE idx = ?", (index,)).fetchone()
            return ModelResult.from_dict(json.loads(record))

        # NaN is stored as NULL; a failed model sorts after every finite one, then by length (_ranking_key).
        failed = "(error IS NULL OR abs(error) = :inf)"
        inf = {"inf": float("inf")}
        top = self._con.execute(
            f"SELECT record FROM models ORDER BY {failed}, CASE WHEN {failed} THEN 0 ELSE error END, length, idx "
            "LIMIT :k",
            {**inf, "k": top_k},
        )
        top_models = [ModelResult.from_dict(json.loads(record)) for (record,) in top]

        incumbents: list[tuple[int, int, ModelResult]] = []
        best = None
        calls = self._con.execute("SELECT trace.idx, error FROM trace JOIN models USING (idx) ORDER BY k")
        for call, (index, error) in enumerate(calls, start=1):
            if error is not None and math.isfinite(error) and (best is None or error < best):
                best = error
                incumbents.append((call, index, model(index)))

        def counts(query: str) -> dict[str, int]:
            found = dict(self._con.execute(query).fetchall())
            return {status: found.get(status, 0) for status in STATUSES}

        # Per length, the lowest finite error and the first model with it; then keep strict improvements.
        front: list[ModelResult] = []
        rows = self._con.execute(
            f"SELECT error, MIN(idx) FROM models JOIN (SELECT length, MIN(error) AS best FROM models WHERE NOT "
            f"{failed} GROUP BY length) USING (length) WHERE error = best GROUP BY length ORDER BY length",
            inf,
        )
        for error, index in rows.fetchall():
            if not front or error < front[-1].error:
                front.append(model(index))

        return top_models, RunSummary(
            num_evaluated=self._model_count,
            evaluation_calls=self._call_count,
            incumbents=incumbents,
            status_counts=counts("SELECT status, COUNT(*) FROM models GROUP BY status"),
            call_status_counts=counts("SELECT status, COUNT(*) FROM trace JOIN models USING (idx) GROUP BY status"),
            pareto_front=front,
        )

    def rewind(self) -> None:
        """Delete what came after the last snapshot, and commit.

        Call it once, right after opening the store with ``mode="resume"``.

        The calls recorded after it and the models first called after it are deleted, so the run
        continues from the snapshot; without a snapshot every call and model is deleted. New
        models and calls are then numbered from the snapshot's counts.
        """
        self._check_writable()
        snapshot = self.snapshot
        num_calls, num_models = (0, 0) if snapshot is None else (snapshot.num_calls, snapshot.num_models)
        self._con.execute("DELETE FROM trace WHERE k >= ?", (num_calls,))
        self._con.execute("DELETE FROM models WHERE idx >= ?", (num_models,))
        self._num_calls, self._num_models = num_calls, num_models
        self.commit()

    @contextmanager
    def consistent_read(self) -> Generator[None, None, None]:
        """Make every read inside the ``with`` block see the file as it was at its first read.

        For a store opened with ``mode="read"`` while its run goes on: without it, each read sees
        what was committed before it, so the models and the trace could come from different
        commits. In other modes it does nothing.
        """
        if not self._read:
            yield
            return
        self._con.execute("BEGIN")
        try:
            yield
        finally:
            self._con.execute("COMMIT")

    def get_header(self, key: str, default: Any = None) -> Any:
        """Return the value stored under ``key`` in the header, or ``default`` if there is none.

        Values come back as ``json.loads`` gives them (tuples as lists, dict keys as strings),
        including values set and not yet committed.

        Args:
            key: The header key.
            default: The value to return if ``key`` is not stored.
        """
        row = self._con.execute("SELECT value FROM header WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_header(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` in the header, or delete the key if ``value`` is ``None``.

        Saved with the next commit.

        Raises:
            TypeError: If ``value`` cannot be saved as JSON (``json.dumps``).
        """
        self._check_writable()
        if value is None:
            self._con.execute("DELETE FROM header WHERE key = ?", (key,))
        else:
            self._con.execute("INSERT OR REPLACE INTO header VALUES (?, ?)", (key, json.dumps(value)))

    def invalid_exprs(self) -> list[str]:
        """Return the strings of the models with status ``"invalid"``, in the order added."""
        rows = self._con.execute("SELECT record FROM models WHERE status = 'invalid' ORDER BY idx")
        return ["".join(json.loads(record)["expr"]) for (record,) in rows]

    def commit(self) -> None:
        """Commit the pending models, calls and header values to the checkpoint file.

        For a database in memory it does nothing observable.

        Raises:
            sqlite3.Error: If the commit fails (e.g. the disk is full). The uncommitted changes are
                rolled back and the connection is closed; every later call raises
                ``RuntimeError``.
            RuntimeError: If an earlier commit failed.
        """
        self._commit(begin=True)

    def _commit(self, begin: bool) -> None:
        # Commit and, with begin, open the next transaction. A failed commit rolls back and closes.
        if self._read:
            self._con.execute("SELECT 1")  # raises if closed
            return
        try:
            # No transaction is open if a KeyboardInterrupt came between the last COMMIT and BEGIN.
            if self._con.in_transaction:
                self._con.execute("COMMIT")
        except sqlite3.Error as e:
            try:
                self._con.execute("ROLLBACK")
            except sqlite3.Error:
                pass  # the failed COMMIT may have ended the transaction already
            self._con.close()
            self._con = cast(sqlite3.Connection, _FailedConnection(e))  # raises on any use
            self._closed = True
            raise
        if begin:
            # A compacted checkpoint is switched back to WAL here, outside a transaction and only
            # once the run goes on, so a refused resume (closed without changes) leaves it untouched.
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("BEGIN")
        self._pending = 0
        self._dirty = False
        self._last_commit = time.monotonic()

    def close(self) -> None:
        """Commit and close the database.

        A checkpoint file is then complete without its ``-wal`` and ``-shm`` files (unless a reader
        still has it open). A database in memory is lost.
        Afterwards every other method and the views raise ``sqlite3.ProgrammingError``. Closing
        again does nothing.

        Raises:
            sqlite3.Error: If the final commit fails; the store is closed anyway, and later calls
                raise ``RuntimeError``.
        """
        if self._closed:
            return
        try:
            self._commit(begin=False)
        finally:
            if not self._closed:
                self._con.close()
                self._closed = True


class _FailedConnection:
    # Stands in for the connection of a store whose commit failed, so every later use says so.
    def __init__(self, error: sqlite3.Error):
        self._error = error

    def execute(self, *args: Any) -> Any:
        raise RuntimeError(
            f"[ModelStore] a commit failed earlier ({self._error}); the store is closed. The checkpoint "
            "keeps what was committed before and can be resumed."
        )

    def __getattr__(self, name: str) -> Any:  # e.g. in_transaction
        self.execute()


_T = TypeVar("_T")


class _View(Sequence[_T], Generic[_T]):
    # A read-only sequence over one table of a store; subclasses give the length, one item and
    # an iterator in order.
    def __init__(self, store: ModelStore):
        self._store = store

    def _get(self, position: int) -> _T:
        raise NotImplementedError

    @overload
    def __getitem__(self, index: int) -> _T: ...

    @overload
    def __getitem__(self, index: slice) -> list[_T]: ...

    def __getitem__(self, index: int | slice) -> _T | list[_T]:
        if isinstance(index, slice):
            return [self._get(i) for i in range(*index.indices(len(self)))]
        index = operator.index(index)  # numpy integers too
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(f"index {index} out of range")
        return self._get(position)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence) or isinstance(other, str):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))


class ModelsView(_View[ModelResult]):
    """Read-only sequence of the models in a store.

    The view of a [ModelStore][SRToolkit.evaluation.model_store.ModelStore]'s models. It is live:
    models added later appear in it. Supports ``len``, indexing (negative too), slicing (giving a
    list) and iteration. Equal to any sequence with equal models in the same order
    (``se.models == []``).

    Examples:
        >>> import numpy as np
        >>> from SRToolkit.evaluation import Evaluator
        >>> se = Evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        >>> _ = se.evaluate_expr(["X_0"])
        >>> len(se.models), se.models[-1].expr
        (1, ['X_0'])
    """

    def __len__(self) -> int:
        return self._store._model_count

    def _get(self, position: int) -> ModelResult:
        (record,) = self._store._con.execute("SELECT record FROM models WHERE idx = ?", (position,)).fetchone()
        return ModelResult.from_dict(json.loads(record))

    def __iter__(self) -> Iterator[ModelResult]:
        for (record,) in self._store._con.execute("SELECT record FROM models ORDER BY idx"):
            yield ModelResult.from_dict(json.loads(record))


class TraceView(_View[int]):
    """Read-only sequence of the model indices in a store's trace.

    The view of a [ModelStore][SRToolkit.evaluation.model_store.ModelStore]'s trace. It is live:
    calls recorded later appear in it. Supports ``len``, indexing (negative too),
    slicing (giving a list) and iteration. Equal to any sequence with the same indices in the same
    order (``se.trace == [0, 1, 0]``).
    """

    def __len__(self) -> int:
        return self._store._call_count

    def _get(self, position: int) -> int:
        return self._store._con.execute("SELECT idx FROM trace WHERE k = ?", (position,)).fetchone()[0]

    def __iter__(self) -> Iterator[int]:
        for (index,) in self._store._con.execute("SELECT idx FROM trace ORDER BY k"):
            yield index
