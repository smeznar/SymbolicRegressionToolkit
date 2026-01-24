from SRToolkit.evaluation import SR_evaluator


class SR_approach:
    def __init__(self, sample_size: int, name: str):
        self.sample_size = sample_size
        self.name = name

    def search(self, sr_evaluator: SR_evaluator):
        raise NotImplementedError