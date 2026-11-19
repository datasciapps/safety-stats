"""Execution ledger schemas.

Every attempted sampling unit is recorded before and after evaluation. An Inspect log
alone is not sufficient for inference: it shows what was evaluated, not what was
selected, so a unit whose rewrite failed validation would simply be absent and the
realized design would silently differ from the declared one.

Prompt content is deliberately absent from these rows. The ledger carries identifiers
and hashes; the prompt text lives in a separate restricted materialization artifact.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from inspect_safety_stats.generative.kernels import ValidationStatus


class LedgerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class SelectionRecord(LedgerModel):
    """One selected structural path, written before its outcome is observed."""

    sampling_unit_id: str = Field(min_length=1)
    stage_id: Literal["pilot", "main"] = "main"
    selection_order: int = Field(ge=0)
    path_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_id: str = Field(min_length=1)
    semantic_seed_id: str = Field(min_length=1)
    stratum: str | None = None
    target_path_probability: float = Field(gt=0.0, le=1.0)
    proposal_path_probability: float = Field(gt=0.0, le=1.0)
    proposal_id: str = Field(min_length=1)
    selection_seed: int = Field(ge=0)


class RewriteRecord(LedgerModel):
    """The outcome of the rewrite procedure for one sampling unit."""

    sampling_unit_id: str = Field(min_length=1)
    rewrite_draw_id: str = Field(min_length=1)
    rewrite_kernel_id: str = Field(min_length=1)
    rewrite_kernel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rewrite_attempts: int = Field(ge=1)
    validation_status: ValidationStatus
    prompt_id: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_semantic_seed_id: str = Field(min_length=1)

    def is_evaluable(self) -> bool:
        return self.validation_status == "valid"


AttemptValidatorStatus = Literal["valid", "invalid", "error", "not_run"]
DuplicateKind = Literal["none", "exact", "near"]


class AttemptRecord(LedgerModel):
    """One call of the kernel for one sampling unit, and what became of its rewrite.

    Written only when the plan asks the package to validate rewrites or to detect near
    duplicates, so that fidelity can be reported per arm: the rewrite ledger keeps one row
    per unit, which says how many attempts a unit took but not why each one failed.

    ``validator_status`` is ``not_run`` when the kernel itself rejected the attempt, or when
    it was rejected as a duplicate first: the validator is the costlier check, so it runs
    last. ``accepted`` marks the attempt whose rewrite the unit kept.
    """

    sampling_unit_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    kernel_status: ValidationStatus
    duplicate: DuplicateKind = "none"
    validator_id: str | None = None
    validator_status: AttemptValidatorStatus = "not_run"
    accepted: bool = False
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


SELECTION_COLUMNS = tuple(SelectionRecord.model_fields)
REWRITE_COLUMNS = tuple(RewriteRecord.model_fields)
ATTEMPT_COLUMNS = tuple(AttemptRecord.model_fields)


class IncrementalCsvWriter:
    """Append rows as they are produced, flushing each one.

    Materialization writes its ledgers this way rather than in one batch at the end. A run
    killed partway -- by a rate limit, a dropped connection, a spend cap -- then leaves a
    usable record of the units it did complete, which is what makes resumption possible at
    all. Writing only on success loses every completed unit.

    Every row ends with LF, as every text artifact does (CONTRIBUTING section 6), and on
    every platform: the file is opened with ``newline=""``, so nothing translates it. A field
    holding a line break is quoted, so a prompt's own line endings are kept as they are.
    """

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self._path = Path(path)
        self._columns = list(columns)
        self._handle: Any = None
        self._writer: Any = None

    def __enter__(self) -> IncrementalCsvWriter:
        existing = self._path.exists() and self._path.stat().st_size > 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=self._columns, lineterminator="\n"
        )
        if not existing:
            self._writer.writeheader()
            self._handle.flush()
        return self

    def append(self, row: Mapping[str, object]) -> None:
        self._writer.writerow({key: row.get(key) for key in self._columns})
        self._handle.flush()

    def replace(self, rows: Iterable[Mapping[str, object]]) -> None:
        """Rewrite the file to hold only ``rows``, as :meth:`append` would have written them.

        The rows go to a sibling file first, which then takes the ledger's place in one
        step, so an interruption leaves the old ledger or the new one, never a truncated
        one. A failure that the process survives removes the sibling; one it does not is
        overwritten by the next replace. Call it while the writer is closed: Windows cannot
        replace an open file.

        The sibling's rows reach the disk before the rename does. Otherwise a power loss
        soon after could leave the ledger's name on an empty file, on file systems that may
        commit a rename before the data of the file renamed. On POSIX the directory is then
        synced as well, so that the rename itself survives one.
        """

        partial = self._path.with_name(self._path.name + ".partial")
        try:
            with partial.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=self._columns, lineterminator="\n"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key) for key in self._columns})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, self._path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        _sync_directory(self._path.parent)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _sync_directory(directory: Path) -> None:
    """Commit a rename in ``directory`` to disk, where the platform can.

    Windows opens no directory as a file, and a file system that cannot sync a directory
    says so with EINVAL or ENOTSUP; the rename then stands as the file system keeps it.
    """

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
    finally:
        os.close(descriptor)


def rewrite_ledger_writer(path: str | Path) -> IncrementalCsvWriter:
    return IncrementalCsvWriter(path, REWRITE_COLUMNS)


def attempt_ledger_writer(path: str | Path) -> IncrementalCsvWriter:
    return IncrementalCsvWriter(path, ATTEMPT_COLUMNS)


def write_attempts(records: list[AttemptRecord], path: str | Path) -> None:
    frame = pd.DataFrame(
        [record.model_dump(mode="json") for record in records], columns=list(ATTEMPT_COLUMNS)
    )
    frame.to_csv(path, index=False, lineterminator="\n")


def read_attempts(path: str | Path) -> list[AttemptRecord]:
    """Read an attempt ledger (``attempts.csv``), refusing a unit's attempt recorded twice."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    records = []
    for row in frame.to_dict("records"):
        cleaned = _clean(row)
        if cleaned.get("accepted") is not None:
            text = str(cleaned["accepted"])
            if text not in {"True", "False"}:
                raise ValueError(f"attempt ledger {path} has accepted={text!r}")
            cleaned["accepted"] = text == "True"
        records.append(AttemptRecord.model_validate(cleaned))
    keys = [(record.sampling_unit_id, record.attempt) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"attempt ledger {path} repeats an attempt of a sampling unit")
    return records


def prompt_sha256(prompt: str) -> str:
    """Content hash used for provenance without disclosing the prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_selections(records: list[SelectionRecord], path: str | Path) -> None:
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    frame.reindex(columns=list(SELECTION_COLUMNS)).to_csv(
        path, index=False, lineterminator="\n"
    )


def write_rewrites(records: list[RewriteRecord], path: str | Path) -> None:
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    frame.reindex(columns=list(REWRITE_COLUMNS)).to_csv(
        path, index=False, lineterminator="\n"
    )


def read_selections(path: str | Path) -> list[SelectionRecord]:
    """Read and validate a selection ledger CSV, one ``SelectionRecord`` per row.

    Every cell is read as text, as written, and the record's fields convert it: an identifier
    such as ``007`` stays ``007``. Empty cells are read as missing values. Raises
    ``ValueError`` when two rows share a ``selection_order`` or a ``sampling_unit_id``, and
    pydantic's ``ValidationError`` for an invalid row.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    records = [SelectionRecord.model_validate(_clean(row)) for row in frame.to_dict("records")]
    orders = [record.selection_order for record in records]
    if len(set(orders)) != len(orders):
        raise ValueError(f"selection ledger {path} repeats a selection_order")
    identifiers = [record.sampling_unit_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"selection ledger {path} repeats a sampling_unit_id")
    return records


def read_rewrite_rows(path: str | Path) -> list[RewriteRecord]:
    """Every row of a rewrite ledger CSV, validated, in file order, a repeated unit included.

    :func:`read_rewrites` is the reader to use. A resume reads with this one, because it
    drops the rows of errored units, and any repeat they make, before it checks for repeats.
    Every cell is read as text, as written, and the record's fields convert it.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [RewriteRecord.model_validate(_clean(row)) for row in frame.to_dict("records")]


def read_rewrites(path: str | Path) -> list[RewriteRecord]:
    """Read and validate a rewrite ledger CSV, one ``RewriteRecord`` per row.

    Every cell is read as text, as written, and the record's fields convert it. Empty cells
    are read as missing values. Raises ``ValueError`` when two rows share a
    ``sampling_unit_id``, and pydantic's ``ValidationError`` for an invalid row.
    """

    records = read_rewrite_rows(path)
    identifiers = [record.sampling_unit_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"rewrite ledger {path} repeats a sampling_unit_id")
    return records


def _clean(row: Mapping[Hashable, object]) -> dict[str, object]:
    return {str(key): None if value == "" else value for key, value in row.items()}
