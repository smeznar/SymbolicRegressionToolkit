# SRToolkit: Symbolic Regression / Equation Discovery Benchmark Toolkit

![SRToolkit logo](https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/refs/heads/master/docs/assets/imgs/logo.png)

**Documentation:**: [https://smeznar.github.io/SymbolicRegressionToolkit](https://smeznar.github.io/SymbolicRegressionToolkit/)

## What is SRToolkit?

The **SRToolkit** is a **comprehensive Python toolkit** designed to accelerate research and development in 
**Symbolic Regression (SR)** / **Equation Discovery (ED)**. It provides a robust, easy-to-use framework for 
**benchmarking, rapid prototyping, and mathematical expression manipulation**.

## Core Features

SRToolkit provides a straightforward interface for:

* **Implementing** new Symbolic Regression approaches and evaluating their performance against other approaches.

* **Benchmarking** Symbolic Regression approaches using built-in benchmarks (currently **Feynman** and **Nguyen**) or **custom data**.

* **Experiment organization** and postprocessing of results.

* **Converting expressions** into **expression trees** or **fast, callable NumPy functions**.

* **Generating random expressions** by defining the symbol space or a grammar.

* **Estimating constant parameters** of expressions against real-world data.

* **Comparing** and **measuring the distance** between expressions.

## Installation

To install the latest stable release of the package, run the following command in your terminal:
```
pip install symbolic-regression-toolkit
```

Alternatively, you can install the latest build directly from the repository with the command (Recommended):

```
pip install git+https://github.com/smeznar/SymbolicRegressionToolkit
```

## Examples

### 1. Expression Manipulation (The Toolkit Core)

SRToolkit offers fundamental utilities for working with mathematical expressions as tokens, trees, and executable code—the building blocks for any SR approach.

```python
import numpy as np
from SRToolkit.utils import expr_to_executable_function, tokens_to_tree, SymbolLibrary, expr_to_latex

# Create an executable function from the expression
expr = expr_to_executable_function(["X_0", "+", "X_1", "*", "C"])

# Calculate the output at two points (1, 2) and (2, 5) with C=3
data_points = np.array([[1, 2], [2, 5]])
constants = [3]
output = expr(data_points, constants)
# Variable "output" should now contain np.array([7, 17])

# Create a SymbolLibrary defining the symbol space for 2 variables
sl = SymbolLibrary.default_symbols(num_variables=2)

# Create an expression tree from the token list
expr_tree = tokens_to_tree(["X_0", "+", "X_1", "*", "C"], sl)

# Transform the expression into a list of symbols in postfix notation
postfix_expr = expr_tree.to_list(notation="postfix")

# Create a LaTeX string of the expression for clear presentation
expr_latex = expr_to_latex(expr_tree, sl)
```

### 2. Benchmarking and Evaluation (The Main Use Case)

The primary advantage of SRToolkit is its robust benchmarking framework, allowing you to quickly evaluate and compare different Symbolic Regression approaches.

```python
from SRToolkit.approaches import EDHiE, ProGED
from SRToolkit.dataset import Feynman
from SRToolkit.evaluation import LoggingCallback
from SRToolkit.experiments import ExperimentGrid

# Load the Feynman benchmark and pick two 2-variable datasets to run on.
bm = Feynman()
ds_names = bm.list_datasets(num_variables=2, verbose=False)
dataset1 = bm.create_dataset(ds_names[0])
dataset2 = bm.create_dataset(ds_names[1])

# Define the SR approaches to benchmark.
# EDHiE requires a pre-trained/adapted model state; ProGED needs no time consuming adaptation.
edhie = EDHiE()
proged = ProGED()

# Map each (approach, dataset) pair to a file where the adapted model state will
# be saved. Both datasets reuse the same file here because they share the same
# number of variables, so one adapted state covers both.
adapted_states = {edhie.name: {ds_names[0]: "adapted_state_2_vars.pt", ds_names[1]: "adapted_state_2_vars.pt"}}

# Build the experiment grid: every combination of approach × dataset will be run
# num_experiments times (with different random seeds). Results are written under
# results_dir, specifically to "results_dir/{dataset}/{approach}/exp_{seed}.json".
eg = ExperimentGrid(
    approaches=[proged, edhie],
    datasets=[dataset1, dataset2],
    num_experiments=2,
    results_dir="../results/",
    adapted_states=adapted_states,
)

# Write a shell script of CLI commands that can be executed in parallel, e.g.:
#   cat commands.sh | parallel -j 4
eg.save_commands("commands.sh")

# Run adaptation for any approach/dataset pair whose adapted state file is missing.
# This is a no-op if all state files already exist.
eg.adapt_if_missing()

# Collect all pending jobs (skip any whose result file already exists on disk).
jobs = eg.create_jobs(skip_completed=True)

# Run each job sequentially in this process. To parallelize, use the generated
# commands.sh instead.
for job in jobs:
    job.run()

# See how many jobs are completed 
eg.progress()

```

Additional examples can be found in the `examples` folder or in the official documentation.

## Roadmap 🗺️

In future releases, our primary focus will:

* **Expanded Library of Approaches:** Add more Symbolic Regression approaches to the toolkit.

* **Result Visualization:** Implement a robust visualization and result aggregation framework for SR results.

* **Simplification:** Implement a better (more accurate, efficient, and stable) simplification system for expressions. 

* **Constraints:** Implement more robust expression generation constraints using techniques like attribute grammars.

* **Improved Benchmarking:** Improve the robustness and efficiency of the benchmarking framework (continuous).

* **Advanced Expressions (Distant Plan):** Implement support for different types of expressions, such as **ODEs and PDEs**.

## Contributing 🤝

We welcome contributions! Whether you're adding a new benchmark, implementing an SR approach, fixing a bug, or improving the documentation, please feel free to 
open a issue on the Github page or submit a **Pull Request (PR)** with a clear description of your changes.

We are especially looking for contributions of:

* New **Benchmarks** and **Datasets**.

* Implementations of additional **Symbolic Regression Approaches** (once the core framework for comparison is finalized).

Instructions on how to contribute can be found in the [Contribution Guide](https://github.com/smeznar/SymbolicRegressionToolkit/blob/master/CONTRIBUTING.md).