"""
Rebuild the hosted data archives of the built-in benchmarks.

Each archive holds ``benchmark.json`` (the benchmark's current config) and ``data/<key>.npz``
(the canonical arrays). [UrlSource][SRToolkit.dataset.data_source.UrlSource] reads this layout,
in this release and in SRToolkit 1.6.0, so keep it: the files are served from GitHub master to
every installed version.

What it does, per benchmark:
  1. Clears any stale cache for the benchmark.
  2. Loads every dataset, which downloads the *canonical* data archive once into the cache
     (so the archived arrays are the authoritative, cross-machine-identical bytes, NOT a fresh
     local sample).
  3. Writes ``benchmark.json`` and every cached data file of the benchmark into the zip.

Usage:
    python scripts/build_archives.py            # writes data/{nguyen,feynman,srsd}.zip
"""

import json
import zipfile

from SRToolkit.dataset import Feynman, Nguyen, SRSD_Feynman, data_cache

# (benchmark class, output path) for each hosted archive.
TARGETS = [
    (Nguyen, "data/nguyen.zip"),
    (Feynman, "data/feynman.zip"),
    (SRSD_Feynman, "data/srsd.zip"),
]


def build(bm_cls, out_path: str) -> None:
    bm = bm_cls()

    # 1. Start from a clean cache so no stale or locally-sampled arrays leak in.
    data_cache.remove(bm.benchmark_name)

    # 2. Warm the cache with the canonical data (one archive download).
    names = bm.list_datasets(verbose=False)
    datasets = [bm.create_dataset(name) for name in names]

    # 3. Write the config and the cached files (train, test and aux splits) in the hosted layout.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("benchmark.json", json.dumps(bm.to_dict(), indent=2))
        for ds in datasets:
            for key, _ in ds._splits():
                zf.write(data_cache.dataset_path(bm.benchmark_name, bm.version, key), f"data/{key}.npz")
    print(f"wrote {out_path} ({len(names)} datasets)")


def main() -> None:
    for bm_cls, out_path in TARGETS:
        build(bm_cls, out_path)


if __name__ == "__main__":
    main()
