"""The records ingestion produces, free of any evaluation tool."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from inspect_safety_stats.ingest.manifest import ResolvedGroup
    from inspect_safety_stats.schema.observations import Observation
    from inspect_safety_stats.schema.reports import Diagnostic
    from inspect_safety_stats.stats.independence import BlockIndex


@dataclass
class NormalizedData:
    observations: list[Observation]
    manifest: pd.DataFrame
    groups: list[ResolvedGroup]
    diagnostics: list[Diagnostic]
    cohorts: list[dict[str, Any]]
    source_hashes: dict[str, str | None]
    judge_verdicts: dict[str, dict[str, bool | None]] = dataclass_field(default_factory=dict)
    # Built by the analysis on first use, from these observations and this manifest.
    block_index: BlockIndex | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )


@dataclass(frozen=True)
class ErrorContext:
    stage: Literal["target_generation", "post_generation", "unknown"]
    content_delivered: bool | None
    provider_error_code: str | None
