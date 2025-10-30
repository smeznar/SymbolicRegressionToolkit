# SRToolkit: Symbolic Regression / Equation Discovery Benchmark Toolkit

![SRToolkit logo](https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/refs/heads/master/docs/assets/imgs/logo.png)

**Documentation:**: [https://smeznar.github.io/SymbolicRegressionToolkit](https://smeznar.github.io/SymbolicRegressionToolkit/)

## What is SRToolkit?

The **SRToolkit** is a **comprehensive Python toolkit** designed to accelerate research and development in 
**Symbolic Regression (SR)** / **Equation Discovery (ED)**. It provides a robust, easy-to-use framework for 
**benchmarking, rapid prototyping, and mathematical expression manipulation**.

## Core Features

SRToolkit provides a straightforward interface for:

* **Benchmarking** Symbolic Regression algorithms using built-in datasets (currently **Feynman** and **Nguyen**) or **custom data**.

* **Converting expressions** into **expression trees** or **fast, callable NumPy functions**.

* **Generating random expressions** by defining the symbol space or a grammar.

* **Estimating constant parameters** of expressions against real-world data.

* **Comparing** and **measuring the distance** between expressions.

## Installation

To install the latest stable release of the package, run the following command in your terminal:
```
pip install symbolic-regression-toolkit
```

Alternatively, you can install the latest build directly from the repository with the command:

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
from SRToolkit.dataset import SR_benchmark
from SRToolkit.utils import generate_n_expressions

# Create the Feynman benchmark suite
feynman = SR_benchmark.feynman("./data/feynman")

# List datasets in the benchmark and select the first 2-variable one
dataset_name = feynman.list_datasets(verbose=False, num_variables=2)[0]

# Create the dataset and the dedicated evaluator object
dataset = feynman.create_dataset(dataset_name)
evaluator = dataset.create_evaluator()

# Generate 100 random expressions for a baseline evaluation
expressions = generate_n_expressions(dataset.symbol_library, 100)

# Evaluate the expressions and print their error
for expr in expressions:
    rmse = evaluator.evaluate_expr(expr)
    print(f"Expr: {''.join(expr)}, Error: {rmse}")

# Get structured results of the evaluation, focusing on the 20 best expressions
results = evaluator.get_results(top_k=20)
```

Additional examples can be found in the `examples` folder or in the official documentation.

## Roadmap 🗺️

In future releases, our primary focus will be on benchmarking and comparability:

* **Benchmarking Core:** Add the ability to save/load benchmark runs and automatically evaluate multiple ED/SR approaches.

* **SR Library:** Create a library of easy-to-use and comparable ED/SR approach implementations.

* **Advanced Expressions (Distant Plan):** Implement support for different types of expressions, such as **ODEs and PDEs**.

* **Constraints:** Implement more robust expression generation constraints using techniques like attribute grammars.

## Contributing 🤝

We welcome contributions! Whether you're adding a new benchmark, implementing an SR approach, fixing a bug, or improving the documentation, please feel free to submit a **Pull Request (PR)** with a clear description of your changes.

We are especially looking for contributions of:

* New **Benchmarks** and **Datasets** (e.g., datasets from physics, finance, etc.).

* Implementations of additional **Symbolic Regression Approaches** (once the core framework for comparison is finalized).