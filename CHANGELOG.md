### SymbolicRegressionToolkit-1.5.0 (2026-04-14)
- Reworked how SR_results and ResultAugmenters work
- Improved documentation
- Fixed some bugs
- Made the interface more consistent and easier to use
- Two instances of SR_benchmarks are now in separate files and allow resampling of data
- All (or at least all relevant) classes can now be saved and loaded/transformed into and from a dictionary
- Added some tests (a lot more to go)
- Added new examples

**New Features:**

- Added a base class for approaches and implemented ProGED and EDHiE
- Added a way to run generate and run experiments easily
- Added the callback system to allow custom events during experiments


### SymbolicRegressionToolkit-1.4.0 (2025-10-28)

- Updated documentation
- Rewrote the SR_dataset and SR_benchmark classes
- New readme and logo

**New Features:**

- Added ResultAugmenter to augment result of evaluation with additional measures, simplification of best expressions, etc.
- Added distance measures
- Added BED as a possible ranking measure for evaluation


### SymbolicRegressionToolkit-1.3.2 (2025-07-07)

- Updated documentation

**New Feature:**

- Added a way to generate expressions
- Expression simplification now more or less works

**Bug Fixes:**

- Feynman dataset and Nguyen datasets are now almost done
- Small fixes to different functionalities


### SymbolicRegressionToolkit-1.2.6 (2025-04-26)

- Updated documentation
- Added the change log

**New Features:**

- Expressions can now be transformed into latex code
- Added Dataset and benchmark objects that create evaluators for Symbolic Regression models
- Added modified versions of feynman and nguyen benchmarks

**Bug Fixes:**

- Fixed expressions with constants only evaluating to one value instead of an array


### SymbolicRegressionToolkit-1.1.0 (2024-12-10)

- Project restructure

**New Features:**

- Added documentation
- Expanded upon examples

### SymbolicRegressionToolkit-1.0.0 (2024-12-06)

- Initial release
- Expression compilation
- Parameter estimation
- Model evaluation
- Examples for parameter estimation and performance evaluation