"""File helpers shared by the source adapters."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str) -> str | None:
    """The SHA-256 of a local file, or None when no such file exists."""

    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
