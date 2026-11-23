"""The one place report artifacts are written: text, JSON, CSV and checksums.

Both the fixed-design report and the generative report write through these helpers, so the
line endings, encoding, JSON layout and checksum format of every artifact are decided here
once (CONTRIBUTING section 6). pandas is not imported: a data frame is written through its
own ``to_csv``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from inspect_safety_stats.config.canonicalize import canonical_value

CHECKSUMS = "checksums.json"


class _Frame(Protocol):
    def to_csv(self, path: Path, *, index: bool, lineterminator: str) -> Any: ...


def sha256(path: Path) -> str:
    """The SHA-256 of a file, read in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    """Write text as UTF-8 with ``\\n`` line endings on every platform."""

    path.write_text(text, encoding="utf-8", newline="\n")


def json_text(value: Any, *, ensure_ascii: bool = False, allow_nan: bool = False) -> str:
    """A JSON document with sorted keys, two-space indentation and a final newline."""

    return (
        json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_json(
    path: Path, value: Any, *, ensure_ascii: bool = False, allow_nan: bool = False
) -> None:
    write_text(path, json_text(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan))


def csv_text(rows: list[dict[str, Any]]) -> str:
    """Rows as CSV text, with the columns in first-seen order and csv's own line endings.

    Nested values are written as canonical JSON, other values in canonical form.
    """

    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(canonical_value(value), ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else canonical_value(value)
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def write_csv_text(path: Path, text: str) -> None:
    """Write CSV text unchanged: it carries its own line endings (csv's default CRLF)."""

    path.write_text(text, encoding="utf-8", newline="")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_text(path, csv_text(rows))


def write_frame(path: Path, frame: _Frame) -> None:
    """Write a data frame as CSV without its index, with ``\\n`` line endings."""

    frame.to_csv(path, index=False, lineterminator="\n")


def write_checksums(directory: Path, names: Iterable[str]) -> dict[str, str]:
    """Write checksums.json: the SHA-256 of each named file in the directory, by name."""

    checksums = {name: sha256(directory / name) for name in sorted(names)}
    write_json(directory / CHECKSUMS, checksums)
    return checksums
