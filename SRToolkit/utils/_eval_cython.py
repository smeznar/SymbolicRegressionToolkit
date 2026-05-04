"""
Pure-Python fallback for the Cython stack machine.

When the compiled ``_eval_cython`` extension is not available this module is
imported instead (Python prefers ``.so``/``.pyd`` over ``.py`` for the same
module name, so the Cython build automatically shadows this file).

The interface is identical to the Cython module — same function signatures,
same exported constants — so callers in ``expression_compiler`` don't need to
know which backend is active.
"""

import numpy as np

# ── Token ID constants (must match _eval_cython.pyx) ─────────────────────────
ADD = 0
SUB = 1
MUL = 2
DIV = 3
POW = 4
SIN = 10
COS = 11
TAN = 12
EXP = 13
SQRT = 14
LN = 15
LOG = 16
ARCSIN = 17
ARCCOS = 18
ARCTAN = 19
SINH = 20
COSH = 21
TANH = 22
FLOOR = 23
CEIL = 24
NEG = 25
INV = 26
SQ = 27
CUBE = 28
POW4 = 29
POW5 = 30
VAR = 50
CONST = 51
LIT_PI = 52
LIT_E = 53
FLOAT = 54
CACHED = 55
PYTHON = -1


def execute(X, C, token_ids, arities, extra_int, float_vals, python_ops):
    """
    Evaluate a symbolic expression over input data X.

    This pure-Python implementation has the same interface as the Cython
    ``execute`` function. It uses NumPy for element-wise operations and is
    therefore slower than the Cython version, but requires no compilation.

    Args:
        X: Input data, shape ``(n_samples, n_features)``.
        C: Constant values, shape ``(n_constants,)``.
        token_ids: Integer operation code for each postfix token.
        arities: Arity (0=leaf, 1=unary, 2=binary) for each token.
        extra_int: Per-token integer payload.
        float_vals: Per-token float payload for ``FLOAT`` tokens.
        python_ops: Per-token Python callables for ``PYTHON`` tokens.

    Returns:
        1-D float64 array of shape ``(n_samples,)``.
    """
    n = X.shape[0]
    n_tokens = len(token_ids)
    stack = [None] * 64
    sp = -1

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for i in range(n_tokens):
            tid = int(token_ids[i])
            ari = int(arities[i])
            ei = int(extra_int[i])

            if ari == 0:
                sp += 1
                if tid == 50:
                    stack[sp] = X[:, ei].copy()
                elif tid == 51:
                    stack[sp] = np.full(n, C[ei])
                elif tid == 54:
                    stack[sp] = np.full(n, float_vals[i])
                elif tid == 55:  # CACHED: push pre-evaluated array
                    stack[sp] = python_ops[i]
                else:
                    stack[sp] = np.asarray(python_ops[i](X, C), dtype=np.float64)

            elif ari == 1:
                a = stack[sp]
                if tid == 10:
                    stack[sp] = np.sin(a)
                elif tid == 11:
                    stack[sp] = np.cos(a)
                elif tid == 12:
                    stack[sp] = np.tan(a)
                elif tid == 13:
                    stack[sp] = np.exp(a)
                elif tid == 14:
                    stack[sp] = np.sqrt(a)
                elif tid == 15:
                    stack[sp] = np.log(a)
                elif tid == 16:
                    stack[sp] = np.log10(a)
                elif tid == 17:
                    stack[sp] = np.arcsin(a)
                elif tid == 18:
                    stack[sp] = np.arccos(a)
                elif tid == 19:
                    stack[sp] = np.arctan(a)
                elif tid == 20:
                    stack[sp] = np.sinh(a)
                elif tid == 21:
                    stack[sp] = np.cosh(a)
                elif tid == 22:
                    stack[sp] = np.tanh(a)
                elif tid == 23:
                    stack[sp] = np.floor(a)
                elif tid == 24:
                    stack[sp] = np.ceil(a)
                elif tid == 25:
                    stack[sp] = -a
                elif tid == 26:
                    stack[sp] = 1.0 / a
                elif tid == 27:
                    stack[sp] = a * a
                elif tid == 28:
                    stack[sp] = a * a * a
                elif tid == 29:
                    t = a * a
                    stack[sp] = t * t
                elif tid == 30:
                    t = a * a
                    stack[sp] = t * t * a
                else:
                    stack[sp] = np.asarray(python_ops[i](a), dtype=np.float64)

            else:  # binary
                left, right = stack[sp - 1], stack[sp]
                if tid == 0:
                    stack[sp - 1] = left + right
                elif tid == 1:
                    stack[sp - 1] = left - right
                elif tid == 2:
                    stack[sp - 1] = left * right
                elif tid == 3:
                    stack[sp - 1] = left / right
                elif tid == 4:
                    stack[sp - 1] = np.power(left, right)
                else:
                    stack[sp - 1] = np.asarray(python_ops[i](left, right), dtype=np.float64)
                sp -= 1

    return np.asarray(stack[0], dtype=np.float64)


def execute_error(X, C, y, token_ids, arities, extra_int, float_vals, python_ops):
    """
    Evaluate an expression and return the RMSE against target values ``y``.

    Args:
        X: Input data, shape ``(n_samples, n_features)``.
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
    result = execute(X, C, token_ids, arities, extra_int, float_vals, python_ops)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt(np.mean((result - y) ** 2)))
