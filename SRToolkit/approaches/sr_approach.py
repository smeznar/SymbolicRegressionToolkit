from typing import Optional

from SRToolkit.evaluation import SR_evaluator


class SR_approach:
    def __init__(self, sample_size: Optional[int] = None):
        if sample_size is None:
            raise ValueError("Sample size (number of expressions generated in each iteration) must be specified.")
        self.sample_size = sample_size

    def search(self, sr_evaluator: SR_evaluator):
        raise NotImplementedError