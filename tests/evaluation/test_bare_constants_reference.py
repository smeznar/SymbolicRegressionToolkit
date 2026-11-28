"""Regression test of the constants rework: expressions with only bare ``C`` keep their scores.

``tests/data/bare_constants_reference.json`` was generated at commit 86cff56 (before the constants
rework) with ``python tests/evaluation/test_bare_constants_reference.py``.
Every value must stay bit-identical: RMSE scores and fitted parameters on all backends and both
initializations, fits stopped after the first step (``max_iter=0``; these depend directly on the
starting point, so they pin the per-expression seed), BED scores and behaviour matrices.

Exact floats depend on the platform and on the Python, NumPy and SciPy versions, so the test is
skipped elsewhere. Delete it after stage 3 of the rework.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy

from SRToolkit.dataset import Nguyen
from SRToolkit.evaluation.evaluator import Evaluator
from SRToolkit.utils.measures import create_behavior_matrix
from SRToolkit.utils.settings import Settings
from SRToolkit.utils.symbol_library import SymbolLibrary

REFERENCE = Path(__file__).parent.parent / "data" / "bare_constants_reference.json"
SEED = 7
ROWS = 500
BACKENDS = ["stack", "stack_py", "codegen"]
EXPRESSIONS = [
    "C",
    "C * X_0",
    "X_0 + C",
    "C * X_0 + C",
    "C * X_0 ^2 + C * X_0 + C",
    "sin ( C * X_0 ) + C",
    "log ( C * X_0 )",
    "C / ( X_0 + C )",
    "pi * C * X_0",
    "C + C + C + C + C + C + C + C + C",  # over max_constants=8
    "C * sin ( X_0 ) + C * cos ( X_1 )",
    "exp ( C * X_0 ) * C + X_1 * C",
]


def _environment() -> dict[str, str]:
    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _hex(values: np.ndarray | None) -> list[str] | None:
    return None if values is None else [float(v).hex() for v in np.asarray(values, dtype=np.float64)]


def compute() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ["NG-1", "NG-10"]:
        dataset = Nguyen().create_dataset(name)
        X, y = dataset.X[:ROWS], dataset.y[:ROWS]
        sl = SymbolLibrary.default_symbols(X.shape[1])
        exprs = [e.split() for e in EXPRESSIONS if X.shape[1] > 1 or "X_1" not in e]
        for backend in BACKENDS:
            for init in ["random", "mean"]:
                for max_iter in [100, 0]:
                    settings = Settings(
                        evaluator={"backend": backend},
                        estimation={"initialization": init},
                        minimize={"max_iter": max_iter},
                    )
                    se = Evaluator(X, y, symbol_library=sl, seed=SEED, settings=settings)
                    for tokens in exprs:
                        error = se.evaluate_expr(tokens).error
                        model = se.models[-1]
                        out[f"rmse|{name}|{backend}|{init}|{max_iter}|{' '.join(tokens)}"] = {
                            "error": float(error).hex(),
                            "parameters": _hex(model.parameters),
                        }
            se = Evaluator(
                X, ground_truth=dataset.ground_truth, ranking_function="bed", symbol_library=sl, seed=SEED,
                settings=Settings(evaluator={"backend": backend}),
            )  # fmt: skip
            for tokens in exprs:
                key = f"{name}|{backend}|{' '.join(tokens)}"
                out[f"bed|{key}"] = float(se.evaluate_expr(tokens).error).hex()
                matrix = create_behavior_matrix(tokens, X[:64], symbol_library=sl, seed=SEED, backend=backend)
                out[f"matrix|{key}"] = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
    return out


def test_bare_constant_results_match_reference():
    reference = json.loads(REFERENCE.read_text())
    if reference["environment"] != _environment():
        pytest.skip(f"reference was generated on {reference['environment']}")
    expected = reference["values"]
    actual = compute()
    assert actual.keys() == expected.keys()
    mismatched = [k for k in expected if actual[k] != expected[k]]
    assert mismatched == []


if __name__ == "__main__":
    REFERENCE.write_text(
        json.dumps({"environment": _environment(), "values": compute()}, indent=1, sort_keys=True) + "\n"
    )
