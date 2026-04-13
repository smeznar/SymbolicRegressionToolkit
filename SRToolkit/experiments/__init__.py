"""
Job-based experiment runner for multi-dataset, multi-approach symbolic regression experiments.

Exports three public classes:

- [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] — seed, result path, and adaptation-state
  path for a single run.
- [ExperimentJob][SRToolkit.experiments.ExperimentJob] — one atomic experiment (dataset × approach ×
  seed); can be run in-process or dispatched via the CLI.
- [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] — full cross-product grid with serialization,
  HPC command-file generation, progress tracking, and result loading.
"""

from .experiment_grid import ExperimentGrid, ExperimentInfo, ExperimentJob

__all__ = [
    "ExperimentGrid",
    "ExperimentInfo",
    "ExperimentJob",
]
