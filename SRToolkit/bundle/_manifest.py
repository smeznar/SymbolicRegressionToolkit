"""
BundleManifest dataclass and checksum verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@dataclass
class BundleManifest:
    name: str
    version: str
    author: str = ""
    srtk_min_version: str = ""
    python_deps: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)

    @property
    def version_slug(self) -> str:
        return self.version.replace(".", "_").replace("-", "_")

    @property
    def import_prefix(self) -> str:
        safe_name = self.name.replace("-", "_")
        return f"srtk_bundles.{safe_name}_{self.version_slug}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "srtk_min_version": self.srtk_min_version,
            "python_deps": self.python_deps,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BundleManifest":
        return cls(
            name=d["name"],
            version=d["version"],
            author=d.get("author", ""),
            srtk_min_version=d.get("srtk_min_version", ""),
            python_deps=d.get("python_deps", []),
            files=d.get("files", {}),
        )

    def verify(self, extracted_dir: Path) -> None:
        """Raise ValueError if any declared file is missing or has a wrong checksum."""
        for rel, expected in self.files.items():
            p = extracted_dir / rel
            if not p.exists():
                raise ValueError(f"Bundle file missing: {rel}")
            actual = _sha256(p)
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {rel}: expected {expected[:12]}…, got {actual[:12]}…")
