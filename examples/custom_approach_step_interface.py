"""A custom approach written with the step interface.

Instead of writing the whole search loop yourself (see custom_approach_random_search.py), you write three
small methods and the framework runs the loop:

- start: create the search state, the memory of the run;
- ask: propose a few expressions;
- tell: look at their errors and update the state.

The framework evaluates the expressions between ask and tell. Because it also sees the state between two
steps, it can save it, so a stopped run can be continued later instead of starting again.

The approach here, Grow, builds an expression one piece at a time:

1. The first step tries each variable and the constant on its own (X_0, X_1, C).
2. Every later step takes the best expression so far and tries a few random ways of extending it with an
   operator and a variable or the constant, e.g. C*X_1 -> C*X_1*X_1 or C*X_1+X_0.
3. If an extension has a lower error, it becomes the new best expression.

For example, on Feynman I.14.4 (0.5*X_0*X_1^2), Grow goes C -> C*X_1 -> C*X_1*X_1 -> C*X_1*X_1*X_0.
It is a toy: it is greedy and only appends to the end, so it cannot find most expressions.
"""

from dataclasses import dataclass

import numpy as np

from SRToolkit.approaches import Approach, ApproachConfig
from SRToolkit.dataset import Feynman

BATCH_SIZE = 4  # how many extensions one step tries


# The search state: everything the run learns. It is kept here, not on the approach, so the framework can save
# it. The random generator belongs here too, so a continued run draws the same numbers as one never stopped.
@dataclass
class GrowState:
    rng: np.random.Generator
    best_expr: list[str] | None = None  # None until the first step
    best_error: float = np.inf


class Grow(Approach):
    def __init__(self):
        super().__init__(ApproachConfig(name="Grow"))

    def start(self, evaluator, seed=None):
        return GrowState(rng=np.random.default_rng(seed))

    def ask(self, evaluator, state):
        library = evaluator.symbol_library
        operands = library.get_symbols_of_type("var") + [library.constant_symbol]  # X_0, X_1, ..., C

        if state.best_expr is None:  # first step
            return [[operand] for operand in operands]

        extensions = [[op, operand] for op in library.get_symbols_of_type("op") for operand in operands]
        chosen = state.rng.choice(len(extensions), size=BATCH_SIZE, replace=False)
        return [state.best_expr + extensions[i] for i in chosen]

    def tell(self, evaluator, state, candidates, results):
        for result in results:
            if result.error < state.best_error:
                state.best_expr = result.expr
                state.best_error = result.error


if __name__ == "__main__":
    dataset = Feynman().create_dataset("I.14.4")
    dataset.max_evaluations = 10  # too small on purpose: the run stops before it finds the expression

    # checkpoint_dir saves the run to a file, so it can be continued later.
    results = dataset.evaluate_approach(
        Grow(), initial_seed=0, checkpoint_dir="checkpoints", keep_checkpoint=True, verbose=False
    )
    print(f"After {results[0].evaluation_calls} evaluations: {results[0].best_expr}")

    # Continue the same run with a larger budget. It picks up where it stopped, and the result is the same as
    # if the budget had been 200 from the start. (After Ctrl-C, evaluate_approach(..., resume=True) continues
    # a run in the same way.)
    checkpoint_file = results[0].checkpoint["path"]
    results = dataset.extend_run(Grow(), checkpoint_file, max_evaluations=200)
    print(f"After {results[0].evaluation_calls} evaluations: {results[0].best_expr}")
