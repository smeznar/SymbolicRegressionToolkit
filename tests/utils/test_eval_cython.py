"""
Tests for _eval_cython.py (pure-Python fallback) and the compiled
_eval_cython.so (Cython extension).

Both modules expose an identical execute() / execute_error() interface.
The same parametrised suite runs against both backends so numerical
correctness is verified for each.

Coverage note
-------------
100% line coverage of _eval_cython.py is achieved by hitting:
  - every leaf token type  : VAR, CONST, FLOAT, CACHED, Python fallback
  - every unary op         : ids 10–30 plus Python fallback
  - every binary op        : ids 0–4 plus Python fallback
  - execute_error

The compiled .so cannot be measured by coverage.py without rebuilding
the extension with ``# cython: linetrace=True``, but the same tests
verify correctness for both backends.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

# ── Load both backends ─────────────────────────────────────────────────────────

_PY_PATH = pathlib.Path(__file__).parent.parent.parent / "SRToolkit/utils/_eval_cython.py"
_py_spec = importlib.util.spec_from_file_location("_eval_cython_py", _PY_PATH)
_eval_py = importlib.util.module_from_spec(_py_spec)
_py_spec.loader.exec_module(_eval_py)

from SRToolkit.utils import _eval_cython as _eval_cy  # noqa: E402


@pytest.fixture(
    params=[
        pytest.param(_eval_py, id="python"),
        pytest.param(_eval_cy, id="cython"),
    ]
)
def ev(request):
    return request.param


# ── Array construction helpers ─────────────────────────────────────────────────


def mk(tids, aris, eis=None, fvs=None, ops=None):
    """Build the five postfix-instruction arrays for execute()."""
    n = len(tids)
    return (
        np.array(tids, dtype=np.int32),
        np.array(aris, dtype=np.int32),
        np.array(eis if eis is not None else [0] * n, dtype=np.int32),
        np.array(fvs if fvs is not None else [0.0] * n, dtype=np.float64),
        ops if ops is not None else [None] * n,
    )


def col(*vals):
    """Return a C-contiguous (n, 1) float64 column array."""
    return np.ascontiguousarray(np.array(vals, dtype=np.float64)[:, None])


def xcols(*col_vals):
    """Return a C-contiguous (n, m) float64 array from m column sequences."""
    return np.ascontiguousarray(np.column_stack([np.array(c, dtype=np.float64) for c in col_vals]))


C0 = np.array([], dtype=np.float64)  # empty constants vector


# ── Constants ──────────────────────────────────────────────────────────────────

_CONSTANT_NAMES = [
    ("ADD", 0),
    ("SUB", 1),
    ("MUL", 2),
    ("DIV", 3),
    ("POW", 4),
    ("SIN", 10),
    ("COS", 11),
    ("TAN", 12),
    ("EXP", 13),
    ("SQRT", 14),
    ("LN", 15),
    ("LOG", 16),
    ("ARCSIN", 17),
    ("ARCCOS", 18),
    ("ARCTAN", 19),
    ("SINH", 20),
    ("COSH", 21),
    ("TANH", 22),
    ("FLOOR", 23),
    ("CEIL", 24),
    ("NEG", 25),
    ("INV", 26),
    ("SQ", 27),
    ("CUBE", 28),
    ("POW4", 29),
    ("POW5", 30),
    ("VAR", 50),
    ("CONST", 51),
    ("LIT_PI", 52),
    ("LIT_E", 53),
    ("FLOAT", 54),
    ("CACHED", 55),
    ("PYTHON", -1),
]


class TestConstants:
    """Token-ID constants must be identical in both backends."""

    @pytest.mark.parametrize("name,expected", _CONSTANT_NAMES)
    def test_value(self, ev, name, expected):
        assert getattr(ev, name) == expected


# ── Leaf tokens ────────────────────────────────────────────────────────────────


class TestLeaves:
    def test_var_first_column(self, ev):
        X = xcols([1.0, 2.0, 3.0], [9.0, 9.0, 9.0])
        tids, aris, eis, fvs, ops = mk([50], [0], eis=[0])
        np.testing.assert_array_equal(ev.execute(X, C0, tids, aris, eis, fvs, ops), [1.0, 2.0, 3.0])

    def test_var_second_column(self, ev):
        X = xcols([9.0, 9.0], [4.0, 5.0])
        tids, aris, eis, fvs, ops = mk([50], [0], eis=[1])
        np.testing.assert_array_equal(ev.execute(X, C0, tids, aris, eis, fvs, ops), [4.0, 5.0])

    def test_const(self, ev):
        X = col(0.0, 0.0)
        C = np.array([7.5], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([51], [0], eis=[0])
        np.testing.assert_allclose(ev.execute(X, C, tids, aris, eis, fvs, ops), [7.5, 7.5])

    def test_float_literal(self, ev):
        X = col(0.0, 0.0, 0.0)
        tids, aris, eis, fvs, ops = mk([54], [0], fvs=[3.14])
        np.testing.assert_allclose(ev.execute(X, C0, tids, aris, eis, fvs, ops), [3.14, 3.14, 3.14])

    def test_cached(self, ev):
        X = col(0.0, 0.0)
        cached = np.ascontiguousarray([8.0, 9.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([55], [0], ops=[cached])
        np.testing.assert_allclose(ev.execute(X, C0, tids, aris, eis, fvs, ops), [8.0, 9.0])

    def test_python_fallback_leaf(self, ev):
        X = col(1.0, 2.0)

        def my_leaf(x, c):
            return np.full(x.shape[0], 42.0)

        tids, aris, eis, fvs, ops = mk([-1], [0], ops=[my_leaf])
        np.testing.assert_allclose(ev.execute(X, C0, tids, aris, eis, fvs, ops), [42.0, 42.0])


# ── Unary ops ─────────────────────────────────────────────────────────────────

_UNARY_CASES = [
    (10, "sin", 0.5, np.sin(0.5)),
    (11, "cos", 0.5, np.cos(0.5)),
    (12, "tan", 0.5, np.tan(0.5)),
    (13, "exp", 1.0, np.exp(1.0)),
    (14, "sqrt", 4.0, 2.0),
    (15, "ln", 1.0, 0.0),
    (16, "log10", 10.0, 1.0),
    (17, "arcsin", 0.5, np.arcsin(0.5)),
    (18, "arccos", 0.5, np.arccos(0.5)),
    (19, "arctan", 1.0, np.pi / 4),
    (20, "sinh", 1.0, np.sinh(1.0)),
    (21, "cosh", 1.0, np.cosh(1.0)),
    (22, "tanh", 1.0, np.tanh(1.0)),
    (23, "floor", 1.7, 1.0),
    (24, "ceil", 1.2, 2.0),
    (25, "neg", 3.0, -3.0),
    (26, "inv", 2.0, 0.5),
    (27, "sq", 3.0, 9.0),
    (28, "cube", 2.0, 8.0),
    (29, "pow4", 2.0, 16.0),
    (30, "pow5", 2.0, 32.0),
]


class TestUnaryOps:
    @pytest.mark.parametrize("tid,name,x_val,expected", _UNARY_CASES)
    def test_unary_op(self, ev, tid, name, x_val, expected):
        X = col(x_val)
        tids, aris, eis, fvs, ops = mk([50, tid], [0, 1])
        result = ev.execute(X, C0, tids, aris, eis, fvs, ops)
        np.testing.assert_allclose(result, [expected], rtol=1e-12, atol=1e-12)

    def test_python_fallback_unary(self, ev):
        X = col(5.0, 6.0)

        def my_unary(a):
            return a + 100.0

        tids, aris, eis, fvs, ops = mk([50, -1], [0, 1], ops=[None, my_unary])
        result = ev.execute(X, C0, tids, aris, eis, fvs, ops)
        np.testing.assert_allclose(result, [105.0, 106.0])


# ── Binary ops ────────────────────────────────────────────────────────────────

_BINARY_CASES = [
    pytest.param(0, "add", lambda a, b: a + b, id="add"),
    pytest.param(1, "sub", lambda a, b: a - b, id="sub"),
    pytest.param(2, "mul", lambda a, b: a * b, id="mul"),
    pytest.param(3, "div", lambda a, b: a / b, id="div"),
    pytest.param(4, "pow", lambda a, b: np.power(a, b), id="pow"),
]


class TestBinaryOps:
    @pytest.mark.parametrize("tid,name,fn", _BINARY_CASES)
    def test_binary_op(self, ev, tid, name, fn):
        # X_0 OP X_1, postfix: [VAR_0, VAR_1, OP]
        X = xcols([2.0, 4.0], [3.0, 5.0])
        tids, aris, eis, fvs, ops = mk([50, 50, tid], [0, 0, 2], [0, 1, 0])
        result = ev.execute(X, C0, tids, aris, eis, fvs, ops)
        np.testing.assert_allclose(result, fn(X[:, 0], X[:, 1]), rtol=1e-12)

    def test_python_fallback_binary(self, ev):
        X = xcols([1.0, 2.0], [3.0, 4.0])

        def my_binary(a, b):
            return a * b + 1000.0

        tids, aris, eis, fvs, ops = mk([50, 50, -1], [0, 0, 2], [0, 1, 0], ops=[None, None, my_binary])
        result = ev.execute(X, C0, tids, aris, eis, fvs, ops)
        np.testing.assert_allclose(result, [1.0 * 3.0 + 1000.0, 2.0 * 4.0 + 1000.0])


# ── execute_error ─────────────────────────────────────────────────────────────


class TestExecuteError:
    def test_zero_rmse_when_prediction_equals_target(self, ev):
        X = col(1.0, 2.0, 3.0)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([50], [0])
        assert ev.execute_error(X, C0, y, tids, aris, eis, fvs, ops) == pytest.approx(0.0)

    def test_known_rmse(self, ev):
        # expression = constant 0; targets = [1, 2, 3, 4]
        # RMSE = sqrt(mean([1, 4, 9, 16])) = sqrt(7.5)
        X = col(0.0, 0.0, 0.0, 0.0)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([54], [0], fvs=[0.0])
        result = ev.execute_error(X, C0, y, tids, aris, eis, fvs, ops)
        assert result == pytest.approx(float(np.sqrt(np.mean(y**2))), rel=1e-10)

    def test_returns_python_float(self, ev):
        X = col(1.0)
        y = np.array([1.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([50], [0])
        assert isinstance(ev.execute_error(X, C0, y, tids, aris, eis, fvs, ops), float)


# ── Integration ───────────────────────────────────────────────────────────────


class TestIntegration:
    def test_compound_expression(self, ev):
        # sin(X_0) * X_1 + C_0
        # postfix: VAR_0, SIN, VAR_1, MUL, CONST_0, ADD
        X = xcols([0.0, np.pi / 2], [2.0, 3.0])
        C = np.array([10.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk(
            [50, 10, 50, 2, 51, 0],
            [0, 1, 0, 2, 0, 2],
            [0, 0, 1, 0, 0, 0],
        )
        result = ev.execute(X, C, tids, aris, eis, fvs, ops)
        np.testing.assert_allclose(result, np.sin(X[:, 0]) * X[:, 1] + 10.0, rtol=1e-12)

    def test_two_constants(self, ev):
        # C_0 + C_1, postfix: CONST_0, CONST_1, ADD
        X = col(0.0, 0.0)
        C = np.array([3.0, 7.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([51, 51, 0], [0, 0, 2], [0, 1, 0])
        np.testing.assert_allclose(ev.execute(X, C, tids, aris, eis, fvs, ops), [10.0, 10.0])

    def test_execute_error_with_constant(self, ev):
        # expression = X_0 + C_0, target = X_0 + 5 → RMSE = 0 when C_0 = 5
        X = col(1.0, 2.0, 3.0)
        C = np.array([5.0], dtype=np.float64)
        y = np.array([6.0, 7.0, 8.0], dtype=np.float64)
        tids, aris, eis, fvs, ops = mk([50, 51, 0], [0, 0, 2], [0, 0, 0])
        assert ev.execute_error(X, C, y, tids, aris, eis, fvs, ops) == pytest.approx(0.0)
