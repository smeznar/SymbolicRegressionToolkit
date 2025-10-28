**Symbolic Regression/Equation Discovery Toolkit**
--------------------------------------------------------

![SRToolkit logo](https://raw.githubusercontent.com/smeznar/SymbolicRegressionToolkit/refs/heads/master/docs/assets/imgs/logo.png)

Documentation: [https://smeznar.github.io/SymbolicRegressionToolkit](https://smeznar.github.io/SymbolicRegressionToolkit/)

This repository provides a Python-based toolkit for equation discovery (ED)/symbolic regression (SR). SRToolkit provides an easy-to-use
interface for:
- Transforming expressions into expression trees or callable functions
- Generating random expressions by describing the space of expressions as a list of symbols or a grammar
- Measuring the distance between expressions
- Estimating parameters of expressions against the data
- Evaluating ED/SR approaches either on build-in benchmarks (currently "Feynman" and "Nguyen") or custom data


[//]: # (Additional examples can be found in the `examples` folder or in the documentation.)

[//]: # (A simple example of how to use the toolkit can be found in the `examples` folder. Script `examples/SR_evaluation_minimal_example.py`)

[//]: # (contains a minimal example of how to use the toolkit for evaluating Symbolic Regression models. Script `examples/parameter_estimation_minimal_example.py`)

[//]: # (contains a minimal example of how to use the toolkit for parameter estimation. Lastly, script `examples/customization.py` shows)

[//]: # (how we can customize various parts of the toolkit and create executable python functions from infix expressions.)

[//]: # ()

*Installation*
--------------

To install the lastest release of the package, run the following command in your terminal:
```
pip install symbolic-regression-toolkit
```

Otherwise, you can install the latest build with the command:

```
pip install git+https://github.com/smeznar/SymbolicRegressionToolkit
```

**Roadmap**
------------
In future releases, our primary focus will be on benchmarking. We want to add the ability to save/load benchmarks and
automatically evaluate ED/SR approaches. Additionally, we want to create a library of ED/SR approaches that are easy
to use and compare against.

More distant plans include the ability to use different types of expressions such as ODEs and PDEs, implement more 
constraints during expression generation with help of attribute grammars, ...

**Contributing**
------------

Contributions are welcome! If you'd like to contribute to the project, please submit a pull request with a clear 
description of your changes. Once the framework for benchmark saving/loading and SR_approaches is setup, we will
be happy for any contribution of additional benchmarks and approaches.
