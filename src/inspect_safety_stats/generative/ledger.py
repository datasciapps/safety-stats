"""Execution ledger schemas.

Every attempted sampling unit is recorded before and after evaluation. An Inspect log
alone is not sufficient for inference: it shows what was evaluated, not what was
selected, so a unit whose rewrite failed validation would simply be absent and the
realized design would silently differ from the declared one.

Prompt content is deliberately absent from these rows. The ledger carries identifiers
and hashes; the prompt text lives in a separate restricted materialization artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Literal

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


SELECTION_COLUMNS = tuple(SelectionRecord.model_fields)
REWRITE_COLUMNS = tuple(RewriteRecord.model_fields)


def prompt_sha256(prompt: str) -> str:
    """Content hash used for provenance without disclosing the prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def write_selections(records: list[SelectionRecord], path: str | Path) -> None:
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    frame.reindex(columns=list(SELECTION_COLUMNS)).to_csv(path, index=False)


def write_rewrites(records: list[RewriteRecord], path: str | Path) -> None:
    frame = pd.DataFrame([record.model_dump(mode="json") for record in records])
    frame.reindex(columns=list(REWRITE_COLUMNS)).to_csv(path, index=False)


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
