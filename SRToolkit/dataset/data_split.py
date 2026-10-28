"""
A set of named arrays held by an [SR_dataset][SRToolkit.dataset.sr_dataset.SR_dataset].

A dataset keeps its data as [DataSplit][SRToolkit.dataset.data_split.DataSplit]s: the data the
search runs on, an optional held-out test split
([SR_dataset.test][SRToolkit.dataset.sr_dataset.SR_dataset]), and any extra data for your own
measures ([SR_dataset.aux_data][SRToolkit.dataset.sr_dataset.SR_dataset]). All of them are saved,
cached and shared together with the dataset.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from SRToolkit.dataset.data_source import DataSource


@dataclass
class DataSplit:
    """
    Named arrays (usually ``X`` and, if there are targets, ``y``), optionally with the
    [DataSource][SRToolkit.dataset.data_source.DataSource] they can be regenerated from.

    Examples:
        >>> split = DataSplit(arrays={"X": np.zeros((3, 2)), "y": np.zeros(3)})
        >>> split.X.shape
        (3, 2)
        >>> split.y.shape
        (3,)
        >>> DataSplit(arrays={"X": np.zeros((3, 2))}).y is None
        True

    Attributes:
        arrays: Mapping of array name to array, conventionally ``"X"`` and ``"y"``. Extra data
            for your own measures may hold any other arrays.
        data_source: Where the arrays can be regenerated from (e.g. a seeded
            [SampleSource][SRToolkit.dataset.data_source.SampleSource]). With a source, the
            arrays are regenerated when the dataset is loaded instead of being stored; without
            one, they are stored with the dataset.
    """

    arrays: Dict[str, np.ndarray]
    data_source: Optional[DataSource] = field(default=None)

    @property
    def X(self) -> np.ndarray:
        """The ``"X"`` array."""
        return self.arrays["X"]

    @property
    def y(self) -> Optional[np.ndarray]:
        """The ``"y"`` array, or ``None`` if this split has no targets."""
        return self.arrays.get("y")
