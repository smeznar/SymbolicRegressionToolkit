import numpy as np

from SRToolkit.utils import (
    SymbolLibrary,
    compile_expr,
    expr_to_latex,
    simplify,
    tokens_to_tree,
)

if __name__ == "__main__":
    # ---------------------------------------- Token lists -----------------------------------------------------
    # Expressions in SRToolkit are plain Python lists of string tokens. By default they are written in infix
    # notation, exactly as you would write the expression on paper (parentheses and all).
    expr = ["X_0", "+", "C", "*", "sin", "(", "X_1", ")"]
    print("Token list:", expr)

    # SRToolkit recognises five token types: binary operators (op: + - * / ^), unary functions and postfix
    # powers (fn: sin cos exp sqrt ln ^2 ^3), input variables (var: X_0 X_1 ...), free constants optimised
    # during parameter estimation (const: C), and fixed numeric literals (lit: pi e).

    # ------------------------------------- The symbol library -------------------------------------------------
    # A SymbolLibrary defines which tokens are valid and how they compile to NumPy. The two factory methods
    # cover most use cases: the full default set, or a restricted subset.
    sl = SymbolLibrary.default_symbols(num_variables=2)
    sl_restricted = SymbolLibrary.from_symbol_list(
        ["+", "-", "*", "/", "sin", "cos", "exp", "sqrt", "^2", "C"],
        num_variables=3,
    )
    print("Default library variables:", [s for s in ["X_0", "X_1"] if sl.get_type(s) == "var"])

    # Passing `sl` to every call is verbose. A context manager sets it for the duration of the block, so the
    # functions below need no explicit `sl` argument.
    with sl:
        tree = tokens_to_tree(["X_0", "+", "X_1", "*", "C"])
        f = compile_expr(["X_0", "*", "C"])

    # ----------------------------------------- Expression trees -----------------------------------------------
    # tokens_to_tree parses a token list into a binary Node tree, and Node.to_list converts it back into a
    # token list in any of the three notations.
    tree = tokens_to_tree(["X_0", "+", "X_1", "*", "C"], sl)
    print("\nInfix   :", tree.to_list(sl, notation="infix"))
    print("Prefix  :", tree.to_list(sl, notation="prefix"))
    print("Postfix :", tree.to_list(sl, notation="postfix"))

    # tokens_to_tree is the inverse of to_list: pass notation="prefix"/"postfix" to parse a list written in
    # that notation. Infix lists are parsed with the shunting-yard algorithm; prefix and postfix with a
    # single stack pass that reads each token's arity from the symbol library (no parentheses needed).
    # All three reconstruct the same tree, so an expression survives a round trip through any notation.
    infix = tokens_to_tree(["X_0", "+", "X_1", "*", "C"], sl)
    prefix = tokens_to_tree(["+", "X_0", "*", "X_1", "C"], sl, notation="prefix")
    postfix = tokens_to_tree(["X_0", "X_1", "C", "*", "+"], sl, notation="postfix")
    assert infix.to_list(sl) == prefix.to_list(sl) == postfix.to_list(sl)
    print("Round trip across all three notations agrees:", infix.to_list(sl))

    # ---------------------------------------- Executable functions --------------------------------------------
    # compile_expr turns an expression into a callable f(X, C): X is a (num_samples, num_variables) array and
    # C is a 1-D array with one entry per `C` token (pass np.array([]) when there are no constants).
    f = compile_expr(["X_0", "*", "C", "+", "X_1"], sl)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    C = np.array([2.0])  # one free constant
    print("\nf(X, C) =", f(X, C))  # [ 4. 10. 16.]

    # ------------------------------------------- LaTeX rendering -----------------------------------------------
    # expr_to_latex renders an expression as a LaTeX string, handy for figures and papers.
    latex = expr_to_latex(["sin", "(", "X_0", ")", "+", "X_1", "^2"], sl)
    print("\nLaTeX:", latex)

    # ------------------------------------------- Simplification ------------------------------------------------
    # simplify applies algebraic simplification followed by constant folding (requires SymPy). Wrap calls in
    # try/except when batch-processing, as it may fail for tokens outside the default symbol set.
    simplified = simplify(["C", "+", "C", "*", "C", "+", "X_0", "*", "X_1", "/", "X_0"], sl)
    print("Simplified:", simplified)  # ['C', '+', 'X_1']
