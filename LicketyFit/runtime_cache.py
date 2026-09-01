"""Portable, source-guarded runtime-cache identities."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Iterable


RUNTIME_IDENTITY_FILES = (
    "LicketyFit/runtime_cache.py",
    "scripts/batch_fit_driver.py",
    "LicketyFit/Emitter.py",
    "LicketyFit/PMT.py",
    "LicketyFit/photon_scattering_native.py",
    "LicketyFit/photon_scattering_native.cpp",
)


def runtime_cache_content_tag(
    project_root: str | Path,
    *,
    source_files: Iterable[str] = RUNTIME_IDENTITY_FILES,
    python_identity: str | None = None,
) -> str:
    """Return a path-independent tag for compiled/runtime physics artifacts.

    The absolute extraction path is deliberately excluded.  Every source that
    owns a cached binary or optical transfer participates, along with Python's
    ABI cache tag.  Missing files are represented explicitly, so an incomplete
    installation can never alias a complete release.
    """
    root = Path(project_root).resolve()
    if python_identity is None:
        python_identity = (
            f"{sys.implementation.name}:{sys.version_info.major}."
            f"{sys.version_info.minor}:{getattr(sys.implementation, 'cache_tag', '')}"
        )
    digest = hashlib.sha256(str(python_identity).encode("utf-8"))
    for relative in source_files:
        normalized = Path(relative).as_posix()
        digest.update(normalized.encode("utf-8"))
        try:
            digest.update((root / normalized).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]

