"""
This module contains the SR_approach class, which is the base class for all symbolic regression approaches.
"""
from typing import Optional

from SRToolkit.evaluation import SR_evaluator


class SR_approach:
    def __init__(self, name: str):
        """
        The base class for all symbolic regression approaches. Any symbolic regression approach should inherit from
        this class.

        Args:
            name: The name of the approach.
        """
        self.name = name

    def search(self, sr_evaluator: SR_evaluator, seed: Optional[int] = None):
        """
        Run the symbolic regression search.

        Implementations should use the provided evaluator to score generated expressions.
        All evaluation results are stored inside `sr_evaluator`, so nothing is returned.

        Args:
            sr_evaluator: Evaluator used for scoring expressions.
            seed: Random seed used for generating expressions.
        """
        raise NotImplementedError

    def clone(self) -> "SR_approach":
        """
        Clones the SR_approach instance. This is used to make multiple independent copies of the approach and making
        multiple independent evaluations/parallel evaluations of the approach possible. If the approach is stateless,
        returning self is sufficient, otherwise a deep copy of the approach should be returned. This allows us to do
        pretraining and finetuning of the approach independently.
        """
        raise NotImplementedError