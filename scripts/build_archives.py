"""
Rebuild the hosted benchmark archives in the self-contained ``to_archive`` format
(``benchmark.json`` + ``data/*.npz``).

Why this exists: the benchmark *configuration* (symbol library, entry keys, samplers,
data sources, ...) lives in the Python ``_populate`` code, while the hosted archives only
ever carried raw ``X``/``y`` arrays. When the config dicts change but the data does not,
the archives still need refreshing **only if** you want them to carry the new
``benchmark.json`` (so ``from_url`` / ``from_archive`` reconstruct the current format).

What it does, per benchmark:
  1. Clears any stale cache for the benchmark.
  2. Loads every dataset, which triggers ``_ensure_data`` to download the *canonical*
     data archive once into the cache (so the embedded arrays are the authoritative,
     cross-machine-identical bytes -- NOT a fresh local sample).
  3. Writes a self-contained ``to_archive`` zip whose ``benchmark.json`` reflects the
     current config and whose ``data/*.npz`` are the canonical arrays.

``UrlSource`` accepts this layout too (it strips the ``data/`` prefix and ignores
``benchmark.json``), so the same file serves both the built-in download path and
``SR_benchmark.from_url`` / ``from_archive``.

Usage:
    python scripts/build_archives.py            # writes data/{nguyen,feynman,srsd}.zip
"""

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

    # 2. Warm the cache with the canonical data (one archive download via _ensure_data).
    for name in bm.list_datasets(verbose=False):
        bm.create_dataset(name)

    # 3. Embed the current config + canonical data into a self-contained archive.
    bm.to_archive(out_path)
    print(f"wrote {out_path} ({len(bm.list_datasets(verbose=False))} datasets)")


def main() -> None:
    for bm_cls, out_path in TARGETS:
        build(bm_cls, out_path)


if __name__ == "__main__":
    main()
