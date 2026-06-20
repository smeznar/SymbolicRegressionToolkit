"""
Job-based experiment runner for multi-dataset, multi-approach symbolic regression experiments.

Exports three public classes:

- [ExperimentInfo][SRToolkit.experiments.ExperimentInfo] — seed, result path, and adaptation-state
  path for a single run.
- [ExperimentJob][SRToolkit.experiments.ExperimentJob] — one atomic experiment (dataset × approach ×
  seed); can be run in-process or rebuilt from a grid by the CLI.
- [ExperimentGrid][SRToolkit.experiments.ExperimentGrid] — full cross-product grid with local
  persistence (``save``/``load``), shareable serialisation (``to_dict``/``from_dict`` and the
  self-contained ``export``/``from_export`` folder), HPC command-file generation, progress
  tracking, and result loading.
"""

from .experiment_grid import ExperimentGrid, ExperimentInfo, ExperimentJob

__all__ = [
    "ExperimentGrid",
    "ExperimentInfo",
    "ExperimentJob",
]
