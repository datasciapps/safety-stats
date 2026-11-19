"""The plan, the inputs per-judge runs reuse, and the integrity they summarise."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.config.registration import RegistrationReceipt
from inspect_safety_stats.ingest.manifest import ResolvedGroup
from inspect_safety_stats.ingest.records import SourceData
from inspect_safety_stats.schema.reports import Diagnostic, IntegritySummary
from inspect_safety_stats.validate.plan import validate_plan_references


def _coerce_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def _integrity_status(diagnostics: list[Diagnostic]) -> Literal["valid", "degraded", "invalid"]:
    """What the diagnostics do to integrity, as the registry's effect for their codes says.

    An ERROR makes it invalid: ``tests/unit/test_registry.py`` gives a code the effect
    ``invalid`` exactly when its entry lists ERROR. A diagnostic whose code has the effect
    ``degraded`` degrades it. The sites that emit those codes also write
    ``integrity_degraded`` into the details, for a reader of the report, and the registry test
    holds the codes with that detail to the ``degraded`` ones.
    """

    if any(item.severity == "ERROR" for item in diagnostics):
        return "invalid"
    if any(item.effect == "degraded" for item in diagnostics):
        return "degraded"
    return "valid"


@dataclass
class _JudgeInputs:
    """The manifest, groups and source as read, before the primary run normalized them.

    Each declared judge's run, and each outcome mapping sensitivity's, normalizes them again,
    so each log is read once, and the source remembers each log's hash, so it is hashed once
    (``read_inputs``). ``receipt`` is the registration receipt the primary run checked timing
    with, when it matched the plan.
    """

    manifest: pd.DataFrame
    groups: list[ResolvedGroup]
    source: SourceData
    receipt: RegistrationReceipt | None = None


def _derived_run_diagnostics(
    derived: AnalysisPlan, plan_diagnostics: Sequence[Diagnostic]
) -> list[Diagnostic]:
    """What a run of a plan derived from the analysed one starts from, before its input.

    A judge's run and an outcome mapping sensitivity's change only the primary outcome, so
    they share the plan's own checks and those of its decision policy and registration
    receipt: ``plan_diagnostics``, as the primary run recorded them. A receipt for another plan
    or a decision policy that fails its checks is then an error in every such run, as it is in
    the primary's; the derived plan's own checks could not see either, since both compare the
    frozen plan. The derived plan's checks add what they find beyond them.
    """

    carried = list(plan_diagnostics)
    carried.extend(
        item for item in validate_plan_references(derived) if item not in plan_diagnostics
    )
    return carried


def _sorted_integrity(diagnostics: list[Diagnostic]) -> IntegritySummary:
    return IntegritySummary(
        status=_integrity_status(diagnostics),
        diagnostics=sorted(
            diagnostics, key=lambda item: (item.severity, item.code, item.scope or "")
        ),
    )
