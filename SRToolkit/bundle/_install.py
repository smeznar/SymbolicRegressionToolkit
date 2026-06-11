"""
Bundle install, uninstall, list operations.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

from packaging.requirements import Requirement
from packaging.version import Version

from . import _store
from ._manifest import BundleManifest


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _check_srtk_version(required: str) -> None:
    if not required:
        return
    from SRToolkit import __version__

    if Version(__version__) < Version(required):
        raise RuntimeError(f"Bundle requires SRToolkit>={required} but {__version__} is installed.")


def _check_deps(python_deps: List[str]) -> List[str]:
    """Return the subset of ``python_deps`` that cannot be imported."""
    missing = []
    for spec in python_deps:
        pkg_name = Requirement(spec).name
        try:
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
    return missing


def _extract_and_verify(srtk_path: Path, tmp_dir: Path) -> BundleManifest:
    with zipfile.ZipFile(srtk_path) as zf:
        zf.extractall(tmp_dir)
    manifest = BundleManifest.from_dict(json.loads((tmp_dir / "manifest.json").read_text(encoding="utf-8")))
    manifest.verify(tmp_dir)
    _check_srtk_version(manifest.srtk_min_version)
    return manifest


def install(srtk_path: Path) -> None:
    """
    Install a ``.srtk`` bundle onto this machine.

    Steps:

    1. Unzip to a temporary directory and load the manifest.
    2. Verify per-file checksums declared in the manifest.
    3. Check the required ``SRToolkit`` version.
    4. Check Python dependencies — list missing ones and suggest a ``pip install``
       command; the user decides whether to continue.
    5. Prompt the user to confirm that arbitrary user code will be executable.
    6. Copy ``src/`` to the managed bundle directory.
    7. Register the bundle in the local index.

    Args:
        srtk_path: Path to a ``.srtk`` bundle file produced by
            [pack][SRToolkit.bundle.pack].

    Raises:
        ValueError: On checksum mismatch.
        RuntimeError: If the installed SRToolkit is too old.
    """
    srtk_path = Path(srtk_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest = _extract_and_verify(srtk_path, tmp_dir)

        missing_deps = _check_deps(manifest.python_deps)
        if missing_deps:
            print(f"Bundle '{manifest.name}' declares missing dependencies:")
            for dep in missing_deps:
                print(f"  {dep}")
            print(f"\nInstall them with:\n  pip install {' '.join(missing_deps)}\n")
            if not _confirm("Continue installation without them?"):
                print("Installation cancelled.")
                return

        print(
            f"\nBundle '{manifest.name}' v{manifest.version}"
            + (f" by {manifest.author!r}" if manifest.author else "")
            + " contains user-defined code that will run when loaded."
        )
        if not _confirm("Install?"):
            print("Installation cancelled.")
            return

        install_path = _store.bundle_path(manifest.name, manifest.version)
        if install_path.exists():
            shutil.rmtree(install_path)
        install_path.mkdir(parents=True)

        src_tmp = tmp_dir / "src"
        if src_tmp.exists():
            shutil.copytree(src_tmp, install_path, dirs_exist_ok=True)

        _store.register(manifest, install_path)
        _store.enable_bundle_imports()
        print(f"Installed '{manifest.name}' v{manifest.version} → {install_path}")


def uninstall(name: str, version: Optional[str] = None) -> None:
    """
    Remove an installed bundle from this machine.

    Args:
        name: Bundle name.
        version: Version to remove. If ``None``, removes the latest installed version.
    """
    entry = _store.lookup(name, version)
    install_path = Path(entry["path"])
    if install_path.exists():
        shutil.rmtree(install_path)
    _store.deregister(entry["name"], entry["version"])
    print(f"Uninstalled '{entry['name']}' v{entry['version']}")


def list_installed() -> list:
    """
    Return a list of all installed bundle index entries.

    Each entry is a dict with keys ``name``, ``version``, ``author``, ``path``,
    and ``import_prefix``.
    """
    return _store.all_entries()
