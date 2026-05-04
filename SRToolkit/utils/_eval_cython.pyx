# cython: language_level=3
"""
Cython stack machine for evaluating symbolic expressions.

Expressions are represented as postfix token sequences with pre-built integer
dispatch arrays. Known operations (all default symbol library ops) are executed
as tight C loops with no Python overhead and no intermediate array allocation.
Unknown operations fall back to calling a Python callable.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt as c_sqrt, pow as c_pow

# ── Token ID constants (exported to Python) ──────────────────────────────────
# Binary operators
ADD   = 0
SUB   = 1
MUL   = 2
DIV   = 3
POW   = 4
# Unary functions
SIN   = 10
COS   = 11
TAN   = 12
EXP   = 13
SQRT  = 14
LN    = 15
LOG   = 16
ARCSIN = 17
ARCCOS = 18
ARCTAN = 19
SINH  = 20
COSH  = 21
TANH  = 22
FLOOR = 23
CEIL  = 24
NEG   = 25
INV   = 26
SQ    = 27
CUBE  = 28
POW4  = 29
POW5  = 30
# Leaf types
VAR    = 50
CONST  = 51
LIT_PI = 52
LIT_E  = 53
FLOAT  = 54
CACHED = 55
# Python fallback
PYTHON = -1

# ── Stack depth (sufficient for any expression the library generates) ─────────
cdef int MAX_STACK_DEPTH = 128


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _eval_stack(
    double[:, ::1] X,
    double[::1] C,
    int[::1] token_ids,
    int[::1] arities,
    int[::1] extra_int,
    double[::1] float_vals,
    list python_ops,
    object stack_arr,
):
    cdef double[:, ::1] s = stack_arr
    cdef int n = X.shape[0]
    cdef int n_tokens = token_ids.shape[0]
    cdef int sp = -1
    cdef int i, j, tid, ari, ei
    cdef double fv, t
    cdef double[::1] cached_mv

    for i in range(n_tokens):
        tid = token_ids[i]
        ari = arities[i]
        ei  = extra_int[i]

        # ── Leaf (push onto stack) ────────────────────────────────────────────
        if ari == 0:
            sp += 1
            if tid == 50:           # VAR: copy column from X
                for j in range(n): s[sp, j] = X[j, ei]
            elif tid == 51:         # CONST: broadcast scalar
                fv = C[ei]
                for j in range(n): s[sp, j] = fv
            elif tid == 54:         # FLOAT literal
                fv = float_vals[i]
                for j in range(n): s[sp, j] = fv
            elif tid == 55:         # CACHED: push pre-evaluated constant-free subtree
                cached_mv = python_ops[i]
                for j in range(n): s[sp, j] = cached_mv[j]
            else:                   # Python fallback leaf: callable(X, C) → array
                py_r = np.asarray(python_ops[i](np.asarray(X), np.asarray(C)), dtype=np.float64)
                for j in range(n): s[sp, j] = py_r[j]

        # ── Unary (apply in-place to top of stack) ───────────────────────────
        # NumPy ufuncs with out= dispatch to the system's vectorised libm
        # (AVX2/AVX-512 where available) without requiring -ffast-math.
        elif ari == 1:
            top = stack_arr[sp]
            if tid == 10:    np.sin(top,    out=top)
            elif tid == 11:  np.cos(top,    out=top)
            elif tid == 12:  np.tan(top,    out=top)
            elif tid == 13:  np.exp(top,    out=top)
            elif tid == 14:  np.sqrt(top,   out=top)
            elif tid == 15:  np.log(top,    out=top)
            elif tid == 16:  np.log10(top,  out=top)
            elif tid == 17:  np.arcsin(top, out=top)
            elif tid == 18:  np.arccos(top, out=top)
            elif tid == 19:  np.arctan(top, out=top)
            elif tid == 20:  np.sinh(top,   out=top)
            elif tid == 21:  np.cosh(top,   out=top)
            elif tid == 22:  np.tanh(top,   out=top)
            elif tid == 23:  np.floor(top,  out=top)
            elif tid == 24:  np.ceil(top,   out=top)
            elif tid == 25:         # u- (unary minus)
                for j in range(n): s[sp, j] = -s[sp, j]
            elif tid == 26:         # ^-1
                for j in range(n): s[sp, j] = 1.0 / s[sp, j]
            elif tid == 27:         # ^2
                for j in range(n): s[sp, j] = s[sp, j] * s[sp, j]
            elif tid == 28:         # ^3
                for j in range(n): s[sp, j] = s[sp, j] * s[sp, j] * s[sp, j]
            elif tid == 29:         # ^4
                for j in range(n):
                    t = s[sp, j] * s[sp, j]
                    s[sp, j] = t * t
            elif tid == 30:         # ^5
                for j in range(n):
                    t = s[sp, j] * s[sp, j]
                    s[sp, j] = t * t * s[sp, j]
            else:                   # Python fallback unary: callable(array) → array
                py_r = np.asarray(python_ops[i](np.asarray(s[sp])), dtype=np.float64)
                for j in range(n): s[sp, j] = py_r[j]

        # ── Binary (apply to top two; pop one) ───────────────────────────────
        else:
            if tid == 0:
                for j in range(n): s[sp - 1, j] = s[sp - 1, j] + s[sp, j]
            elif tid == 1:
                for j in range(n): s[sp - 1, j] = s[sp - 1, j] - s[sp, j]
            elif tid == 2:
                for j in range(n): s[sp - 1, j] = s[sp - 1, j] * s[sp, j]
            elif tid == 3:
                for j in range(n): s[sp - 1, j] = s[sp - 1, j] / s[sp, j]
            elif tid == 4:
                for j in range(n): s[sp - 1, j] = c_pow(s[sp - 1, j], s[sp, j])
            else:                   # Python fallback binary: callable(a, b) → array
                py_r = np.asarray(
                    python_ops[i](np.asarray(s[sp - 1]), np.asarray(s[sp])),
                    dtype=np.float64,
                )
                for j in range(n): s[sp - 1, j] = py_r[j]
            sp -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
def execute(
    double[:, ::1] X,
    double[::1] C,
    int[::1] token_ids,
    int[::1] arities,
    int[::1] extra_int,
    double[::1] float_vals,
    list python_ops,
):
    """
    Evaluate a symbolic expression over input data X using a pre-built
    postfix instruction sequence.

    Args:
        X: Input data, shape ``(n_samples, n_features)``. Must be C-contiguous float64.
        C: Constant values, shape ``(n_constants,)``. Pass an empty array when the
            expression has no free constants.
        token_ids: Integer operation code for each postfix token.
        arities: Arity (0=leaf, 1=unary, 2=binary) for each token.
        extra_int: Per-token integer payload: column index for variables,
            constant index for constants, 0 otherwise.
        float_vals: Per-token float payload: literal float value for
            ``FLOAT`` tokens, 0.0 otherwise.
        python_ops: Per-token Python callable used when ``token_ids[i] == PYTHON``.
            ``None`` for all C-implemented tokens.

    Returns:
        1-D float64 array of shape ``(n_samples,)`` with the expression output.
    """
    cdef int n = X.shape[0]
    stack_arr = np.empty((MAX_STACK_DEPTH, n), dtype=np.float64)
    _eval_stack(X, C, token_ids, arities, extra_int, float_vals, python_ops, stack_arr)
    return np.array(stack_arr[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def execute_error(
    double[:, ::1] X,
    double[::1] C,
    double[::1] y,
    int[::1] token_ids,
    int[::1] arities,
    int[::1] extra_int,
    double[::1] float_vals,
    list python_ops,
):
    """
    Evaluate an expression and return the RMSE against target values ``y``.

    Equivalent to ``sqrt(mean((execute(...) - y)^2))``, but computes the
    residual sum-of-squares directly in C without creating an intermediate
    output array.

    Args:
        X: Input data, shape ``(n_samples, n_features)``. Must be C-contiguous float64.
        C: Constant values, shape ``(n_constants,)``.
        y: Target values, shape ``(n_samples,)``.
        token_ids: Integer operation code for each postfix token.
        arities: Arity for each token.
        extra_int: Per-token integer payload.
        float_vals: Per-token float payload.
        python_ops: Per-token Python fallback callables.

    Returns:
        Scalar RMSE as a Python float.
    """
    cdef int n = X.shape[0]
    cdef double diff, rss
    cdef double[:, ::1] s
    cdef int j

    stack_arr = np.empty((MAX_STACK_DEPTH, n), dtype=np.float64)
    s = stack_arr
    _eval_stack(X, C, token_ids, arities, extra_int, float_vals, python_ops, stack_arr)

    rss = 0.0
    for j in range(n):
        diff = s[0, j] - y[j]
        rss += diff * diff
    return c_sqrt(rss / n)
