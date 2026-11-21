"""Storage of an evaluator's models and trace in SQLite.

[SR_evaluator][SRToolkit.evaluation.sr_evaluator.SR_evaluator] keeps every evaluated model and its
evaluation trace in a [ModelStore][SRToolkit.evaluation.model_store.ModelStore] and exposes them
as the read-only views [ModelsView][SRToolkit.evaluation.model_store.ModelsView] (``se.models``)
and [TraceView][SRToolkit.evaluation.model_store.TraceView] (``se.trace``).

Without a checkpoint the database is in memory, so nothing touches the disk and memory grows with
the number of models (about 1 KB each). With a checkpoint path it is a file, and memory stays
constant during the run: only the best error so far and a small page cache stay in RAM.
Collecting the results copies every model into RAM.

A model is saved as the JSON of [ModelResult.to_dict][SRToolkit.utils.types.ModelResult.to_dict],
with its error and status in columns of their own, so a cache hit never parses the record. Each
``se.models[i]`` therefore returns a new [ModelResult][SRToolkit.utils.types.ModelResult]:
changing it saves nothing, and numpy scalars in ``ranking_data`` come back as Python scalars.

Do not modify the database or the checkpoint file yourself: a cache hit returns the stored error
without scoring again, and the trace refers to models by position, so a changed or removed row
makes later results wrong. Only
[evaluate_expr][SRToolkit.evaluation.sr_evaluator.SR_evaluator.evaluate_expr] writes to it.

## Checkpoint file

The file layout is internal; a public reader will come later. Pending models and calls are
committed:

- after 1000 new models since the last commit (cache hits do not count);
- at the first new model or call 30 seconds or more after the last commit;
- when evaluation stops (the first [stop][SRToolkit.evaluation.sr_evaluator.SR_evaluator.stop]
  only);
- when results are collected
  ([get_results][SRToolkit.evaluation.sr_evaluator.SR_evaluator.get_results]).

A crash loses at most the models and calls since the last commit. There is no ``close()`` yet: the
connection closes when the store is garbage collected, and an uncommitted batch is rolled back at
exit, so only committed data is safe. The file is in WAL mode, so another process can read it while
the run continues; committed data may be only in the ``-wal`` file, so keep the ``-wal`` and
``-shm`` files with the checkpoint while it is open.
"""

import hashlib
import json
import operator
import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from typing import Generic, TypeVar, cast, overload

from SRToolkit.utils.types import ModelResult, Status

# Version of the file layout, written to the header table.
_FORMAT_VERSION = 1
# A checkpoint is committed after this many new models or seconds, whichever comes first.
_COMMIT_MODELS = 1000
_COMMIT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE header (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE models (idx INTEGER PRIMARY KEY, key BLOB UNIQUE NOT NULL, error REAL, status TEXT, record TEXT);
CREATE TABLE trace (k INTEGER PRIMARY KEY, idx INTEGER NOT NULL);
"""


def _key(tokens: Sequence[str], spec_key: str) -> bytes:
    # 128-bit hash of the model's identity; collisions are negligible for any realistic run.
    return hashlib.blake2b(("\x1f".join(tokens) + "\x1e" + spec_key).encode(), digest_size=16).digest()


class ModelStore:
    def __init__(self, path: str | os.PathLike | None = None):
        """SQLite database of the models and trace of one evaluator.

        Args:
            path: The checkpoint file to create, or ``None`` (default) for a database in memory.

        Raises:
            ValueError: If ``path`` is empty.
            FileExistsError: If a file exists at ``path``.
            sqlite3.OperationalError: If the file cannot be created (e.g. its directory does not
                exist or is not writable).
        """
        if path is not None and not os.fspath(path):
            raise ValueError("[ModelStore] path must not be empty; pass None for a database in memory.")
        if path is not None and os.path.exists(path):
            raise FileExistsError(f"[ModelStore] {os.fspath(path)!r} already exists.")
        # Transactions are managed explicitly (isolation_level=None); one is always open.
        self._con = sqlite3.connect(":memory:" if path is None else path, isolation_level=None, check_same_thread=False)
        if path is not None:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.executescript(_SCHEMA)
        self._con.execute("INSERT INTO header VALUES ('format_version', ?)", (json.dumps(_FORMAT_VERSION),))
        self._num_models = 0
        self._num_calls = 0
        self._pending = 0
        self._last_commit = time.monotonic()
        self._con.execute("BEGIN")

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
            "SELECT idx, error, status FROM models WHERE key = ?", (_key(tokens, spec_key),)
        ).fetchone()
        return None if row is None else (row[0], row[1], cast(Status, row[2]))

    def add(self, tokens: Sequence[str], spec_key: str, model: ModelResult) -> int:
        """Store a new model and record a call to it, atomically.

        May commit (see the module docstring).

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
        record = json.dumps(model.to_dict())
        index = self._num_models
        self._con.execute("SAVEPOINT add_model")
        try:
            self._con.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?)",
                (index, _key(tokens, spec_key), float(model.error), model.status, record),
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
        self._commit_if_due()
        return index

    def add_call(self, index: int) -> None:
        """Record a call to the already stored model at ``index`` (a cache hit).

        May commit (see the module docstring).
        """
        self._con.execute("INSERT INTO trace VALUES (?, ?)", (self._num_calls, index))
        self._num_calls += 1
        self._commit_if_due()

    def _commit_if_due(self) -> None:
        if self._pending >= _COMMIT_MODELS or time.monotonic() - self._last_commit >= _COMMIT_SECONDS:
            self.commit()

    def invalid_exprs(self) -> list[str]:
        """Return the strings of the models with status ``"invalid"``, in the order they were added."""
        rows = self._con.execute("SELECT record FROM models WHERE status = 'invalid' ORDER BY idx")
        return ["".join(json.loads(record)["expr"]) for (record,) in rows]

    def commit(self) -> None:
        """Commit the pending models and calls to the checkpoint file.

        For a database in memory it does nothing observable.
        """
        self._con.execute("COMMIT")
        self._con.execute("BEGIN")
        self._pending = 0
        self._last_commit = time.monotonic()


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
    list) and iteration. Each access returns a new object, so changing it saves nothing. Equal to
    any sequence with equal models in the same order (``se.models == []``).

    Examples:
        >>> import numpy as np
        >>> from SRToolkit.evaluation import SR_evaluator
        >>> se = SR_evaluator(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        >>> _ = se.evaluate_expr(["X_0"])
        >>> len(se.models), se.models[-1].expr
        (1, ['X_0'])
    """

    def __len__(self) -> int:
        return self._store._num_models

    def _get(self, position: int) -> ModelResult:
        (record,) = self._store._con.execute("SELECT record FROM models WHERE idx = ?", (position,)).fetchone()
        return ModelResult.from_dict(json.loads(record))

    def __iter__(self) -> Iterator[ModelResult]:
        for (record,) in self._store._con.execute("SELECT record FROM models ORDER BY idx"):
            yield ModelResult.from_dict(json.loads(record))


class TraceView(_View[int]):
    """Read-only sequence of the model indices in a store's trace.

    The view of a [ModelStore][SRToolkit.evaluation.model_store.ModelStore]'s trace. It is live:
    calls recorded later appear in it. Supports ``len``, indexing (negative too), slicing (giving a list) and iteration. Equal to any
    sequence with the same indices in the same order (``se.trace == [0, 1, 0]``).
    """

    def __len__(self) -> int:
        return self._store._num_calls

    def _get(self, position: int) -> int:
        return self._store._con.execute("SELECT idx FROM trace WHERE k = ?", (position,)).fetchone()[0]

    def __iter__(self) -> Iterator[int]:
        for (index,) in self._store._con.execute("SELECT idx FROM trace ORDER BY k"):
            yield index
