"""The analysis core over one plan and one normalized source, and per-judge runs."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from inspect_safety_stats.analysis.adaptive_claims import _adaptive_claims
from inspect_safety_stats.analysis.claims import _claims
from inspect_safety_stats.analysis.comparisons import _comparison_set_results, _comparisons
from inspect_safety_stats.analysis.direct_mean import _direct_mean_context, _hierarchy_nodes
from inspect_safety_stats.analysis.estimates import _prompt_estimates
from inspect_safety_stats.analysis.inputs import _integrity_status, _JudgeInputs
from inspect_safety_stats.analysis.judge_correction import apply_judge_correction
from inspect_safety_stats.analysis.judge_panel import apply_panel_derived_correction
from inspect_safety_stats.analysis.regions import project_hierarchy_regions
from inspect_safety_stats.analysis.results import RiskRows
from inspect_safety_stats.analysis.risks import _group_risks, _hierarchy_risks
from inspect_safety_stats.analysis.tail_precision import single_rollout_tail_claims
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.models import AnalysisPlan, JudgeSpec
from inspect_safety_stats.decision.engine import mandatory_statuses
from inspect_safety_stats.ingest.normalize import normalize
from inspect_safety_stats.ingest.records import NormalizedData, SourceData
from inspect_safety_stats.schema.reports import (
    AdaptiveClaimResult,
    ClaimResult,
    ComparisonResult,
    ComparisonSetResult,
    Decision,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    PromptEstimate,
)
from inspect_safety_stats.validate.plan import noisy_valid_test_supported, validate_plan_references


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
    adaptive_claims: list[AdaptiveClaimResult]

    def mandatory_statuses(self) -> list[Decision]:
        """The statuses the overall decision rolls up: every mandatory claim's, of both kinds."""

        return mandatory_statuses(self.claims, self.adaptive_claims)


def _analysis_core(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> _AnalysisCore:
    estimates = _prompt_estimates(plan, data, diagnostics)
    hierarchy_nodes = _hierarchy_nodes(plan, data)
    direct_context = _direct_mean_context(
        plan,
        data,
        nodes=hierarchy_nodes,
        diagnostics=diagnostics,
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
    projected = project_hierarchy_regions(plan, data, hierarchy_nodes, computed, diagnostics)
    judge_correction = apply_judge_correction(
        plan, data, hierarchy_nodes, projected, diagnostics
    )
    panel_correction = apply_panel_derived_correction(plan, judge_correction.rows, diagnostics)
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
        adaptive_claims=_adaptive_claims(plan, data, integrity_status != "invalid"),
    )


def _judge_plan(plan: AnalysisPlan, judge: JudgeSpec) -> AnalysisPlan:
    """The plan as it would read with this judge as its only, primary outcome.

    The plan's judge calibration validates its primary outcome, not this judge, so it is
    left out, and so is the panel, which a single judge's run has no use for. A
    ``noisy_valid_test`` claim's test needs that calibration's labels, so where the plan can
    run the test the judge's run bounds the claim as a ``corrected_interval`` claim instead
    (its row decides nothing under the validated calibration the test requires). Where the
    plan cannot, the claim is kept, so the judge's run reports the primary's plan error.
    """

    claims = plan.claims
    if noisy_valid_test_supported(plan):
        claims = [
            claim.model_copy(update={"method": "corrected_interval"})
            if claim.method == "noisy_valid_test"
            else claim
            for claim in plan.claims
        ]
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
    return plan.model_copy(update={"inputs": inputs, "outputs": outputs, "claims": claims})


_UNCALIBRATED_JUDGE_REASON = (
    "The plan declares a validated judge calibration, so its thresholds are about true risk. "
    "The calibration's labels validate the primary outcome, not this judge, so this judge's "
    "bounds are on its reported risk only and the claim is not decided on them."
)
_NOISY_VALID_TEST_REASON = (
    "The claim's noisy_valid_test needs labels of this judge's verdicts, which the plan does "
    "not have, so this judge's run bounds it as a corrected_interval claim."
)


def _judge_claim_row(
    claim: ClaimResult, *, validated_calibration: bool, noisy_valid_test: bool
) -> dict[str, Any]:
    """One claim of a judge's run, as its row in the judge panel.

    A judge's run has no calibration, so its bounds are on judge-reported risk. Under a
    validated calibration the plan's thresholds are about true risk, where the primary refuses
    to decide on judge-reported bounds; so does the row, which stays INCONCLUSIVE.
    """

    status, reasons = claim.status, list(claim.reasons)
    if validated_calibration:
        status, reasons = "INCONCLUSIVE", [_UNCALIBRATED_JUDGE_REASON]
        if noisy_valid_test:
            reasons.append(_NOISY_VALID_TEST_REASON)
    return {
        "claim_id": claim.claim_id,
        "status": status,
        "estimate": claim.estimate,
        "lower": claim.lower,
        "upper": claim.upper,
        "method": claim.method,
        "reasons": reasons,
    }


def _add_judge_claims(panel: dict[str, Any], plan: AnalysisPlan, inputs: _JudgeInputs) -> None:
    """Decide every claim again on each declared judge's verdicts, beside the primary's.

    Each judge's run is the analysis the plan would give with that judge as its primary
    outcome, over the same rollouts, so comparing judge configurations is one analysis. Its
    bounds are on that judge's reported risk; under a validated calibration its rows decide
    nothing (``_judge_claim_row``). Beside its integrity status, each judge records how often
    each diagnostic code occurred in its run, so the report says why a run is not valid.
    """

    validated_calibration = (
        plan.inputs.judge_calibration is not None
        and plan.inputs.judge_calibration.mode == "validated"
    )
    noisy_claims = {claim.id for claim in plan.claims if claim.method == "noisy_valid_test"}
    by_id = {judge.id: judge for judge in plan.inputs.judges}
    for entry in panel["judges"]:
        judge_plan = _judge_plan(plan, by_id[entry["id"]])
        diagnostics = validate_plan_references(judge_plan)
        data = normalize(
            judge_plan,
            inputs.manifest.copy(),
            copy.deepcopy(inputs.groups),
            inputs.source,
            inputs.receipt,
        )
        diagnostics.extend(data.diagnostics)
        diagnostics.extend(single_rollout_tail_claims(judge_plan, data))
        core = _analysis_core(judge_plan, data, diagnostics)
        entry["integrity"] = core.integrity_status
        counts: dict[str, int] = {}
        for item in diagnostics:
            counts[item.code] = counts.get(item.code, 0) + 1
        entry["diagnostic_counts"] = dict(sorted(counts.items()))
        entry["claims"] = [
            _judge_claim_row(
                claim,
                validated_calibration=validated_calibration,
                noisy_valid_test=claim.claim_id in noisy_claims,
            )
            for claim in sorted(core.claims, key=lambda item: item.claim_id)
        ]


def scorer_configurations(value: Any, score_name: str) -> list[dict[str, str]] | None:
    """Fingerprints of the configurations a run records for the scorer named like a judge's score.

    Each is ``{"name": <score name>, "sha256": <digest>}``, the digest being the SHA-256 of
    the scorer's whole recorded entry (name, options, metrics and metadata) as canonical JSON.
    The entry itself is never copied: scorer options and metadata can hold credentials,
    endpoints and local paths, such as a request header, a token or a template file, and
    Inspect redacts only some of them. Equal fingerprints mean equal recorded configurations.
    None when the run records no scorers at all, an empty list when none has that name.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    scorers = json.loads(value) if isinstance(value, str) else value
    if not isinstance(scorers, list):
        return None
    return [
        {"name": score_name, "sha256": sha256_value(item)}
        for item in scorers
        if isinstance(item, dict) and item.get("name") == score_name
    ]


def _role_models(roles: dict[str, Any], run_key: str) -> dict[str, Any]:
    """Each model role's name and model id, without the rest of its configuration.

    A run's roles also record ``base_url``, ``args`` and the generate config with its
    ``extra_headers``, which can hold endpoints and credentials; none of it reaches a report.
    """

    named: dict[str, Any] = {}
    for name in sorted(roles):
        role = roles[name]
        model = role.get("model") if isinstance(role, dict) else None
        if not isinstance(model, str):
            raise ValueError(
                f"run {run_key} records model role {name!r} without a model id; "
                "eval.model_roles must map each role to a configuration with a 'model'"
            )
        named[name] = {"model": model}
    return named


def _add_judge_configurations(
    panel: dict[str, Any], plan: AnalysisPlan, data: NormalizedData, source_data: SourceData
) -> None:
    """Record, per judge and cohort, the scorer fingerprints and model roles of its runs.

    The values are the distinct ones the cohort's runs record, sorted by their canonical
    form; a cohort whose runs record none says so with null. Neither copies a configuration
    value other than a role's model id, so no credential, endpoint or path reaches a report.
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
                found = scorer_configurations(attributes.get("scorers"), score_name)
                if found is not None:
                    for item in found:
                        scorers[item["sha256"]] = item
                role_value = attributes.get("model_roles")
                if isinstance(role_value, str):
                    role_value = json.loads(role_value)
                if isinstance(role_value, dict):
                    named = _role_models(role_value, run_key)
                    roles[sha256_value(named)] = named
            configuration[summary["cohort_id"]] = {
                "scorer": [scorers[key] for key in sorted(scorers)] or None,
                "model_roles": [roles[key] for key in sorted(roles)] or None,
            }
        entry["configuration"] = configuration
