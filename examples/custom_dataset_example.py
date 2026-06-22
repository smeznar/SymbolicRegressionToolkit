import json

import numpy as np

from SRToolkit.dataset import SR_benchmark, SR_dataset
from SRToolkit.dataset.sampling import UniformSampling
from SRToolkit.utils import SymbolLibrary

_SYMBOL_LIST = ["+", "-", "*", "/", "sin", "cos", "^2", "^3", "C"]


# ----------------------------------- A reusable benchmark via SR_benchmark ------------------------------------
# Group multiple datasets into a reusable benchmark by subclassing SR_benchmark. add_from_samplers attaches a
# SampleSource so each dataset's data is generated lazily (and reproducibly, given a seed) from its samplers
# and ground truth the first time create_dataset() is called, then cached locally for subsequent runs.
class MyBenchmark(SR_benchmark):
    __data_version__ = "1.0.0"

    def __init__(self):
        super().__init__("my_benchmark", version="1.0.0")
        self._populate()

    def _populate(self):
        sl_1v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=1)
        sl_2v = SymbolLibrary.from_symbol_list(_SYMBOL_LIST, num_variables=2)

        self.add_from_samplers(
            ground_truth=["X_0", "^2", "+", "1"],
            samplers=[UniformSampling(0.5, 5.0, uses_negative=False)],
            symbol_library=sl_1v,
            dataset_name="eq1",
            original_equation="x^2 + 1",
            n_samples=10000,
            seed=42,
            success_threshold=1e-6,
        )
        self.add_from_samplers(
            ground_truth=["sin", "(", "X_0", ")", "+", "X_1", "^2"],
            samplers=[UniformSampling(0.5, 5.0), UniformSampling(0.5, 5.0)],
            symbol_library=sl_2v,
            dataset_name="eq2",
            original_equation="sin(x0) + x1^2",
            n_samples=10000,
            seed=42,
            success_threshold=1e-6,
        )


if __name__ == "__main__":
    # ------------------------------------ A standalone SR_dataset ---------------------------------------------
    # SR_dataset wraps input data and evaluation settings for a single problem. Here we build one directly from
    # arrays we already have in memory.
    rng = np.random.default_rng(0)
    X = rng.uniform(0.5, 5.0, size=(1000, 2))
    y = X[:, 0] ** 2 + np.sin(X[:, 1])

    sl = SymbolLibrary.from_symbol_list(["+", "*", "sin", "^2", "C"], num_variables=2)
    dataset = SR_dataset(
        X=X,
        y=y,
        symbol_library=sl,
        dataset_name="my_equation",
        original_equation="x0^2 + sin(x1)",
        ground_truth=["X_0", "^2", "+", "sin", "(", "X_1", ")"],
        max_evaluations=50000,
        success_threshold=1e-6,
        constant_bounds=(-10.0, 10.0),
        max_expr_length=20,
    )
    print("Standalone dataset:", dataset.X.shape, dataset.original_equation)

    # ------------------------- A dataset from an expression + samplers (no arrays yet) ------------------------
    # A common case is having only the target expression and a sampling spec. from_samplers draws X from the
    # samplers, evaluates the ground truth for y, and records a SampleSource so the data can be regenerated.
    sampled = SR_dataset.from_samplers(
        ground_truth=["X_0", "^2", "+", "sin", "(", "X_1", ")"],
        samplers=[UniformSampling(0.5, 5.0), UniformSampling(0.5, 5.0)],
        n_samples=1000,
        seed=42,
    )
    print("Sampled dataset:", sampled.X.shape)

    # ------------------------------------- Using the custom benchmark -----------------------------------------
    bm = MyBenchmark()
    bm.list_datasets()

    # Standard load materialises (and caches) the data from the SampleSource on first use.
    ds = bm.create_dataset("eq1")
    print("\neq1 loaded:", ds.X.shape)

    # Pass n_samples (and optionally seed) for a fresh draw; the cache is not modified.
    ds_small = bm.create_dataset("eq1", n_samples=500, seed=7)
    print("eq1 resampled:", ds_small.X.shape)

    # ----------------------------------- Serialising the benchmark to JSON ------------------------------------
    # to_dict() returns a portable, data-free config: each dataset keeps only its samplers and a data_source
    # pointer, so the arrays are regenerated on the recipient's machine (reproducibly, thanks to the seed).
    config = bm.to_dict()
    with open("my_benchmark.json", "w") as fh:
        json.dump(config, fh, indent=2)

    bm2 = SR_benchmark.from_dict("my_benchmark.json")
    ds2 = bm2.create_dataset("eq2")
    print("\nReconstructed from config:", ds2.X.shape, ds2.original_equation)
