"""
Managed storage for installed bundles.

All bundles live under ``<user_data_dir>/SRToolkit/srtk_bundles/``. Each installed
bundle is stored in a sub-directory whose name is ``<safe_name>_<version_slug>``
(hyphens and dots in the name/version are replaced with underscores). The parent
directory ``<user_data_dir>/SRToolkit`` is added to ``sys.path``, so the bundle is
importable as ``srtk_bundles.<safe_name>_<version_slug>`` — i.e. the on-disk
directory name is the last segment of the import path (see
[BundleManifest.import_prefix][SRToolkit.bundle._manifest.BundleManifest.import_prefix]).
An ``index.json`` file in the bundles root tracks installed bundles so they can
be listed and uninstalled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from packaging.version import Version
from platformdirs import user_data_dir

from ._manifest import BundleManifest


def bundles_root() -> Path:
    return Path(user_data_dir("SRToolkit")) / "srtk_bundles"


def bundle_path(name: str, version: str) -> Path:
    safe_name = name.replace("-", "_")
    version_slug = version.replace(".", "_").replace("-", "_")
    return bundles_root() / f"{safe_name}_{version_slug}"


def _index_path() -> Path:
    return bundles_root() / "index.json"


def _read_index() -> Dict[str, dict]:
    p = _index_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_index(index: Dict[str, dict]) -> None:
    _index_path().write_text(json.dumps(index, indent=2), encoding="utf-8")


def _index_key(name: str, version: str) -> str:
    return f"{name}@{version}"


def enable_bundle_imports() -> None:
    """
    Make installed bundles importable in the current Python process.

    Adds the bundles-root parent directory to ``sys.path`` (if not already
    present) so that ``import srtk_bundles.<safe_name>_<version_slug>`` works.
    Idempotent and safe to call repeatedly.

    Called automatically at the end of [install][SRToolkit.bundle.install] and
    by [bind_config][SRToolkit.bundle.bind_config]. Call it directly only when
    you want to ``import`` a bundle's classes by hand in a fresh Python session
    without going through ``from_dict`` / ``bind_config``.
    """
    bundles_parent = str(bundles_root().parent)
    if bundles_parent not in sys.path:
        sys.path.insert(0, bundles_parent)


def register(manifest: BundleManifest, install_path: Path) -> None:
    index = _read_index()
    index[_index_key(manifest.name, manifest.version)] = {
        "name": manifest.name,
        "version": manifest.version,
        "author": manifest.author,
        "path": str(install_path),
        "import_prefix": manifest.import_prefix,
    }
    _write_index(index)


def deregister(name: str, version: str) -> None:
    index = _read_index()
    key = _index_key(name, version)
    if key not in index:
        raise KeyError(f"Bundle {name!r} version {version!r} is not installed.")
    del index[key]
    _write_index(index)


def lookup(name: str, version: Optional[str] = None) -> dict:
    """Return the index entry for the bundle. If version is None, return the latest."""
    index = _read_index()
    if version is not None:
        key = _index_key(name, version)
        if key not in index:
            raise KeyError(f"Bundle {name!r} version {version!r} is not installed.")
        return index[key]
    matches = [v for k, v in index.items() if v["name"] == name]
    if not matches:
        raise KeyError(f"Bundle {name!r} is not installed.")
    return max(matches, key=lambda e: Version(e["version"]))


def all_entries() -> list:
    return list(_read_index().values())
