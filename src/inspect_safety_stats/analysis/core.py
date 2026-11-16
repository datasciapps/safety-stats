"""The analysis core over one plan and one normalized source, and per-judge runs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from inspect_safety_stats.analysis.claims import _claims
from inspect_safety_stats.analysis.comparisons import _comparison_set_results, _comparisons
from inspect_safety_stats.analysis.direct_mean import _direct_mean_context, _hierarchy_nodes
from inspect_safety_stats.analysis.estimates import _prompt_estimates
from inspect_safety_stats.analysis.inputs import _integrity_status, _JudgeInputs
from inspect_safety_stats.analysis.judge_correction import _apply_judge_correction
from inspect_safety_stats.analysis.judge_panel import _apply_panel_derived_correction
from inspect_safety_stats.analysis.regions import _project_hierarchy_regions
from inspect_safety_stats.analysis.results import RiskRows
from inspect_safety_stats.analysis.risks import _group_risks, _hierarchy_risks
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.models import AnalysisPlan, JudgeSpec
from inspect_safety_stats.ingest.normalize import normalize
from inspect_safety_stats.ingest.records import NormalizedData, SourceData
from inspect_safety_stats.schema.reports import (
    ClaimResult,
    ComparisonResult,
    ComparisonSetResult,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    PromptEstimate,
)
from inspect_safety_stats.validate.plan import validate_plan_references


@dataclass
class _AnalysisCore:
    """What one plan decides over one normalized source, before the report is assembled."""

    estimates: list[PromptEstimate]
    risks: list[GroupRiskResult]
    hierarchy_risks: list[HierarchyRiskResult]
    comparisons: list[ComparisonResult]
    judge_calibration_summary: dict[str, Any] | None
    panel_correction: dict[str, Any] | None
    comparison_sets: list[ComparisonSetResult]
    integrity_status: Literal["valid", "degraded", "invalid"]
    claims: list[ClaimResult]


def _analysis_core(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> _AnalysisCore:
    estimates = _prompt_estimates(plan, data, diagnostics)
    hierarchy_nodes = _hierarchy_nodes(plan, data)
    direct_context = _direct_mean_context(
        plan,
        data,
        nodes=hierarchy_nodes,
    )
    risks = _group_risks(
        plan,
        data,
        estimates,
        diagnostics,
        direct_context=direct_context,
    )
    hierarchy_risks = _hierarchy_risks(
        plan,
        data,
        hierarchy_nodes,
        estimates,
        diagnostics,
        direct_context,
    )
    # Comparisons read the prompt estimates only, and project their own regions.
    comparisons = _comparisons(plan, data, estimates, diagnostics)
    computed = RiskRows(risks=risks, hierarchy_risks=hierarchy_risks, comparisons=comparisons)
    # Each adjusting stage returns new rows and passes them on, so the judge correction
    # corrects the bounds that are displayed: the projected ones.
    projected = _project_hierarchy_regions(plan, data, hierarchy_nodes, computed, diagnostics)
    judge_correction = _apply_judge_correction(
        plan, data, hierarchy_nodes, projected, diagnostics
    )
    panel_correction = _apply_panel_derived_correction(plan, judge_correction.rows, diagnostics)
    final = panel_correction.rows
    risks, hierarchy_risks, comparisons = final.risks, final.hierarchy_risks, final.comparisons
    judge_calibration_summary = judge_correction.summary
    comparison_sets = _comparison_set_results(plan, comparisons, risks)
    integrity_status = _integrity_status(diagnostics)
    claims = _claims(
        plan,
        data,
        risks,
        comparisons,
        integrity_status != "invalid",
        diagnostics,
        judge_calibration_summary,
    )
    return _AnalysisCore(
        estimates=estimates,
        risks=risks,
        hierarchy_risks=hierarchy_risks,
        comparisons=comparisons,
        judge_calibration_summary=judge_calibration_summary,
        panel_correction=panel_correction.summary,
        comparison_sets=comparison_sets,
        integrity_status=integrity_status,
        claims=claims,
    )


def _judge_plan(plan: AnalysisPlan, judge: JudgeSpec) -> AnalysisPlan:
    """The plan as it would read with this judge as its only, primary outcome.

    The plan's judge calibration validates its primary outcome, not this judge, so it is
    left out, and so is the panel, which a single judge's run has no use for.
    """

    inputs = plan.inputs.model_copy(
        update={
            "score": judge.score,
            "judges": [],
            "composites": [],
            "judge_calibration": None,
            "panel_groups": None,
        }
    )
    outputs = plan.outputs.model_copy(update={"judge_claims": False})
    return plan.model_copy(update={"inputs": inputs, "outputs": outputs})


def _add_judge_claims(panel: dict[str, Any], plan: AnalysisPlan, inputs: _JudgeInputs) -> None:
    """Decide every claim again on each declared judge's verdicts, beside the primary's.

    Each judge's run is the analysis the plan would give with that judge as its primary
    outcome, over the same rollouts, so comparing judge configurations is one analysis.
    """

    by_id = {judge.id: judge for judge in plan.inputs.judges}
    for entry in panel["judges"]:
        judge_plan = _judge_plan(plan, by_id[entry["id"]])
        diagnostics = validate_plan_references(judge_plan)
        data = normalize(
            judge_plan, inputs.manifest.copy(), copy.deepcopy(inputs.groups), inputs.source
        )
        diagnostics.extend(data.diagnostics)
        core = _analysis_core(judge_plan, data, diagnostics)
        entry["integrity"] = core.integrity_status
        entry["claims"] = [
            {
                "claim_id": claim.claim_id,
                "status": claim.status,
                "estimate": claim.estimate,
                "lower": claim.lower,
                "upper": claim.upper,
                "method": claim.method,
            }
            for claim in sorted(core.claims, key=lambda item: item.claim_id)
        ]


def _scorer_configurations(value: Any, score_name: str) -> list[Any] | None:
    """The configurations a run records for the scorer named like a judge's score.

    None when the run records no scorers at all, an empty list when none has that name.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    scorers = json.loads(value) if isinstance(value, str) else value
    if not isinstance(scorers, list):
        return None
    return [
        {key: item[key] for key in sorted(item) if key != "name"}
        for item in scorers
        if isinstance(item, dict) and item.get("name") == score_name
    ]


def _add_judge_configurations(
    panel: dict[str, Any], plan: AnalysisPlan, data: NormalizedData, source_data: SourceData
) -> None:
    """Record, per judge and cohort, the scorer configurations and model roles of its runs.

    The values are the distinct ones the cohort's runs record, sorted by their canonical
    form; a cohort whose runs record none says so with null.
    """

    runs_by_key = {run.run_key: run for run in source_data.runs}
    by_id = {judge.id: judge for judge in plan.inputs.judges}
    for entry in panel["judges"]:
        score_name = by_id[entry["id"]].score.name
        configuration: dict[str, Any] = {}
        for summary in sorted(data.cohorts, key=lambda item: item["cohort_id"]):
            scorers: dict[str, Any] = {}
            roles: dict[str, Any] = {}
            for run_key in summary["run_keys"]:
                attributes = runs_by_key[run_key].attributes
                found = _scorer_configurations(attributes.get("scorers"), score_name)
                if found is not None:
                    for item in found:
                        scorers[sha256_value(item)] = item
                role_value = attributes.get("model_roles")
                if isinstance(role_value, str):
                    role_value = json.loads(role_value)
                if isinstance(role_value, dict):
                    roles[sha256_value(role_value)] = role_value
            configuration[summary["cohort_id"]] = {
                "scorer": [scorers[key] for key in sorted(scorers)] or None,
                "model_roles": [roles[key] for key in sorted(roles)] or None,
            }
        entry["configuration"] = configuration
