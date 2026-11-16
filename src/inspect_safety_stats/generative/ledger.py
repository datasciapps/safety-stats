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
import hashlib
from collections.abc import Hashable, Mapping, Sequence
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
        self._writer = csv.DictWriter(self._handle, fieldnames=self._columns)
        if not existing:
            self._writer.writeheader()
            self._handle.flush()
        return self

    def append(self, row: Mapping[str, object]) -> None:
        self._writer.writerow({key: row.get(key) for key in self._columns})
        self._handle.flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


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
    frame = pd.read_csv(path, keep_default_na=False)
    records = [SelectionRecord.model_validate(_clean(row)) for row in frame.to_dict("records")]
    orders = [record.selection_order for record in records]
    if len(set(orders)) != len(orders):
        raise ValueError(f"selection ledger {path} repeats a selection_order")
    identifiers = [record.sampling_unit_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"selection ledger {path} repeats a sampling_unit_id")
    return records


def read_rewrites(path: str | Path) -> list[RewriteRecord]:
    frame = pd.read_csv(path, keep_default_na=False)
    records = [RewriteRecord.model_validate(_clean(row)) for row in frame.to_dict("records")]
    identifiers = [record.sampling_unit_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"rewrite ledger {path} repeats a sampling_unit_id")
    return records


def _clean(row: Mapping[Hashable, object]) -> dict[str, object]:
    return {str(key): None if value == "" else value for key, value in row.items()}
