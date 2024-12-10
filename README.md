**Symbolic Regression/Equation Discovery Toolkit**
--------------------------------------------------------

Documentation: [https://smeznar.github.io/SymbolicRegressionToolkit](https://smeznar.github.io/SymbolicRegressionToolkit/)

This repository provides a Python-based toolkit for equation discovery/symbolic regression. Currently, the toolkit contains
code for transforming infix expressions into trees, parameter estimation, and performance evaluation for symbolic regression models.

Currently, we only support (vanilla) mathematical expressions, however, we provide a simple interface for adding custom symbols.
In the future, we might extend our functionality to support more advanced expressions (differential equations, PDEs, ...).

A simple example of how to use the toolkit can be found in the `examples` folder. Script `examples/SR_evaluation_minimal_example.py`
contains a minimal example of how to use the toolkit for evaluating Symbolic Regression models. Script `examples/parameter_estimation_minimal_example.py`
contains a minimal example of how to use the toolkit for parameter estimation. Lastly, script `examples/customization.py` shows
how we can customize various parts of the toolkit and create executable python functions from infix expressions.

*Installation*
--------------

To install the package, run the following command in your terminal:
```
pip install symbolic-regression-toolkit
```

**Contributing**
------------

Contributions are welcome! If you'd like to contribute to the project, please submit a pull request with a clear description of your changes.
