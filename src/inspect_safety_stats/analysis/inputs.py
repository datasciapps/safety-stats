"""The plan, the inputs per-judge runs reuse, and the integrity they summarise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.manifest import ResolvedGroup
from inspect_safety_stats.ingest.records import SourceData
from inspect_safety_stats.schema.reports import Diagnostic, IntegritySummary


def _coerce_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def _integrity_status(diagnostics: list[Diagnostic]) -> Literal["valid", "degraded", "invalid"]:
    if any(item.severity == "ERROR" for item in diagnostics):
        return "invalid"
    if any(bool(item.details.get("integrity_degraded")) for item in diagnostics):
        return "degraded"
    return "valid"


@dataclass
class _JudgeInputs:
    """The manifest, groups and source as read, before the primary run normalized them.

    Each declared judge's run normalizes them again, so the logs are read once.
    """

    manifest: pd.DataFrame
    groups: list[ResolvedGroup]
    source: SourceData


def _sorted_integrity(diagnostics: list[Diagnostic]) -> IntegritySummary:
    return IntegritySummary(
        status=_integrity_status(diagnostics),
        diagnostics=sorted(
            diagnostics, key=lambda item: (item.severity, item.code, item.scope or "")
        ),
    )
