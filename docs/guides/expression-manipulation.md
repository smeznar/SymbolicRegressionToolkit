---
title: Expression Manipulation
---

# Expression Manipulation

Expressions in SRToolkit are represented as **infix token lists** — plain Python lists of strings. This format is the common currency between the symbol library, the parser, the compiler, and the SR approaches.

## Token lists

A token list is an ordered sequence of strings in infix notation:

```python
expr = ["X_0", "+", "C", "*", "sin", "(", "X_1", ")"]
```

Token types:

| Type | Examples | Description |
|---|---|---|
| `op` | `+` `-` `*` `/` `^` | Binary operators |
| `fn` | `sin` `cos` `exp` `sqrt` `ln` `^2` `^3` | Unary functions and postfix powers |
| `var` | `X_0` `X_1` … | Input variables, mapped to columns of `X` in order |
| `const` | `C` | Free constant optimised during parameter estimation |
| `lit` | `pi` `e` | Fixed numeric literals |

Postfix power tokens (`^2`, `^3`, `^4`, `^5`, `^-1`) are written after their operand:

```python
["X_0", "^2", "+", "X_1", "^3"]   # x0² + x1³
```

## Symbol library

A [SymbolLibrary][SRToolkit.utils.symbol_library.SymbolLibrary] defines which tokens are valid in an expression and how they compile to NumPy. Most use cases are covered by the two factory methods:

```python
from SRToolkit.utils import SymbolLibrary

# Full default set (all operators, functions, C, pi, e) with 2 variables
sl = SymbolLibrary.default_symbols(num_variables=2)

# Restrict to a specific subset
sl = SymbolLibrary.from_symbol_list(
    ["+", "-", "*", "/", "sin", "cos", "exp", "sqrt", "^2", "C"],
    num_variables=3,
)
```

Custom symbols can be added for non-standard backends:

```python
sl = SymbolLibrary(preamble=["import numpy as np", "import scipy.special as sp"])
sl.add_symbol("erf", "fn", precedence=5, np_fn="sp.erf({})", latex_str=r"\mathrm{erf}\,{}")
```

## Symbol library context

Passing `sl` to every function call is verbose. Two alternatives let you set it once.

**Context manager** — active for the duration of the `with` block:

```python
sl = SymbolLibrary.default_symbols(num_variables=2)

with sl:
    tree  = tokens_to_tree(["X_0", "+", "X_1", "*", "C"])
    f     = compile_expr(["X_0", "*", "C"])
    latex = expr_to_latex(["sin", "(", "X_0", ")", "+", "X_1"])
    d     = edit_distance(["X_0", "+", "1"], ["X_0", "-", "1"])
```

**Module-level default** — persists for the whole session, useful in scripts and notebooks:

```python
SymbolLibrary.set_default(SymbolLibrary.default_symbols(num_variables=3))

# No sl argument needed anywhere below this point
tree = tokens_to_tree(["X_0", "+", "X_1"])
f    = compile_expr(["X_0", "*", "C"])

SymbolLibrary.set_default(None)   # clear when done
```

The resolution order is: **explicit argument → context manager → module default → `default_symbols()`**. Functions that parse token vocabularies (`tokens_to_tree`, `create_generic_pcfg`) raise `RuntimeError` if no library is available; all other functions fall back to `default_symbols()`.

## Expression trees

[tokens_to_tree][SRToolkit.utils.expression_tree.tokens_to_tree] parses a token list into a binary [Node][SRToolkit.utils.expression_tree.Node] tree:

```python
from SRToolkit.utils import SymbolLibrary, tokens_to_tree

sl = SymbolLibrary.default_symbols(num_variables=2)
tree = tokens_to_tree(["X_0", "+", "X_1", "*", "C"], sl)
```

Convert back to a token list in any notation:

```python
tree.to_list(sl, notation="infix")    # ['X_0', '+', 'X_1', '*', 'C']
tree.to_list(notation="prefix")       # ['+', 'X_0', '*', 'X_1', 'C']
tree.to_list(notation="postfix")      # ['X_0', 'X_1', 'C', '*', '+']
```

## Executable functions

[compile_expr][SRToolkit.utils.expression_compiler.compile_expr] compiles an expression into a fast callable `f(X, C)`:

```python
import numpy as np
from SRToolkit.utils import compile_expr

f = compile_expr(["X_0", "*", "C", "+", "X_1"])

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])
C = np.array([2.0])       # one free constant

print(f(X, C))            # [4.  10.  16.]
```

`X` must have shape `(n_samples, n_features)`. `C` is a 1-D array with one entry per `C` token in the expression. Pass an empty array (`np.array([])`) when there are no constants.

## LaTeX rendering

```python
from SRToolkit.utils import SymbolLibrary, tokens_to_tree
from SRToolkit.utils.expression_compiler import expr_to_latex

sl = SymbolLibrary.default_symbols(num_variables=2)
latex = expr_to_latex(["sin", "(", "X_0", ")", "+", "X_1", "^2"], sl)
print(latex)  # $\sin X_{0} + X_{1}^2$
```

## Simplification

[simplify][SRToolkit.utils.expression_simplifier.simplify] applies algebraic simplification followed by constant folding:

```python
from SRToolkit.utils import SymbolLibrary
from SRToolkit.utils.expression_simplifier import simplify

sl = SymbolLibrary.default_symbols(num_variables=2)

# Algebraic reduction + constant folding
simplified = simplify(["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"], sl)
print(simplified)  # ['C', '+', 'X_1']
```

!!! note
    Simplification requires SymPy. It may fail for expressions containing tokens outside the default symbol set — wrap calls in a `try/except` when batch-processing large result sets.
