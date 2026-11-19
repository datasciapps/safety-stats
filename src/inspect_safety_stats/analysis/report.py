"""Report sections: policy events, design, scope, and the report of unread input."""

from __future__ import annotations

from typing import Any

from inspect_safety_stats.analysis.inputs import _sorted_integrity
from inspect_safety_stats.config.loader import canonical_plan_hash, report_path, report_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.decision.engine import roll_up
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    ClaimEvidenceRecord,
    ClaimResult,
    DecisionPolicyRecord,
    Diagnostic,
    JudgeCalibrationPresentation,
    JudgeScenarioRecord,
    RateRangeRecord,
    ReportPresentation,
    SafetyReport,
    VerdictRange,
)
from inspect_safety_stats.version import __version__


def _policy_events(plan: AnalysisPlan, data: NormalizedData) -> list[dict[str, Any]]:
    if not plan.policy.zero_tolerance.enabled:
        return []
    field = plan.policy.zero_tolerance.manifest_field
    if field not in data.manifest:
        return []
    zero_prompts = set(
        data.manifest.loc[data.manifest[field].fillna(False).astype(bool), "prompt_id"].astype(str)
    )
    return [
        {
            "type": "zero_tolerance_failure",
            "status": "verified_from_score",
            "cohort_id": row.cohort_id,
            "prompt_id": row.prompt_id,
            "observation_id": row.observation_id,
            "source": row.source,
            "rollout_id": row.rollout_id,
        }
        for row in data.observations
        if row.prompt_id in zero_prompts and row.failure is True
    ]


def _resolved_design(data: NormalizedData, hierarchy: list[str]) -> list[dict[str, Any]]:
    manifest = data.manifest.set_index("prompt_id")
    hierarchy_columns = list(dict.fromkeys([*hierarchy, "task_id", "question_id", "prompt_id"]))
    rows: list[dict[str, Any]] = []
    for group in sorted(data.groups, key=lambda item: item.id):
        for prompt_id in group.prompt_ids:
            item = manifest.loc[prompt_id]
            row: dict[str, Any] = {"group_id": group.id}
            for column in hierarchy_columns:
                row[column] = prompt_id if column == "prompt_id" else item[column]
            row["weight"] = group.weights[prompt_id]
            rows.append(row)
    return rows


def _scope(plan: AnalysisPlan) -> dict[str, Any]:
    """The report's scope statement, which says what population the results are about."""
    if plan.design.type == "sampled":
        # The Horvitz-Thompson intervals are about the whole declared frame, not only the
        # items that were drawn.
        return {
            "design": "sampled",
            "randomness": (
                "items drawn from the declared frame with recorded inclusion probabilities, "
                "and stochastic rollouts within the drawn items"
            ),
            "estimand": plan.inputs.estimand,
            "disclaimer": (
                "Results apply to the declared item frame, from which the evaluated items "
                "were drawn with the recorded inclusion probabilities, and to the specified "
                "stochastic rollout process. They do not estimate failure prevalence over "
                "prompts outside the frame or establish safety for omitted hazards or "
                "deployment conditions."
            ),
        }
    return {
        "design": "fixed",
        "randomness": "stochastic rollouts within fixed prompts",
        "estimand": plan.inputs.estimand,
        "disclaimer": (
            "Results apply to the declared fixed evaluation set and specified "
            "stochastic rollout process. They do not estimate failure prevalence "
            "over an external population of prompts or establish safety for omitted "
            "hazards or deployment conditions."
        ),
    }


def _presentation(plan: AnalysisPlan) -> ReportPresentation:
    """What the rendered report needs from the plan, so that it renders from the report alone."""

    independence = plan.outputs.direct_mean_intervals.independence
    measure_types = {measure.id: str(measure.type) for measure in plan.outputs.risk_measures}
    calibration = plan.inputs.judge_calibration
    return ReportPresentation(
        include_decisions=plan.report.include_decisions,
        decimal_places=plan.report.decimal_places,
        primary_independence_profile=independence.primary if independence else None,
        claim_evidence=[
            ClaimEvidenceRecord(
                claim_id=claim.id,
                comparison=claim.comparison,
                cohort=claim.cohort,
                group=claim.group,
                measure=claim.measure,
                measure_type=measure_types.get(claim.measure) if claim.measure else None,
            )
            for claim in sorted(plan.claims, key=lambda item: item.id)
        ],
        judge_calibration=(
            None
            if calibration is None
            else JudgeCalibrationPresentation(
                mode=calibration.mode,
                sampling=calibration.sampling,
                primary=calibration.primary,
                groups=calibration.groups,
                scenarios=[
                    JudgeScenarioRecord(
                        id=item.id,
                        rho_1=RateRangeRecord(lower=item.rho_1.lower, upper=item.rho_1.upper),
                        rho_0=RateRangeRecord(lower=item.rho_0.lower, upper=item.rho_0.upper),
                    )
                    for item in calibration.scenarios
                ],
            )
        ),
        claims_export=True if plan.report.claims_export else None,
        primary_outcome_mapping=(
            plan.inputs.score.mapping.model_dump(mode="json", exclude_none=True)
            if plan.inputs.score.mapping_sensitivities
            else None
        ),
    )


def _report_frame(plan: AnalysisPlan) -> dict[str, Any]:
    """The report fields that come from the plan alone, whether or not its input was read."""

    return {
        "package_version": __version__,
        "analysis": {
            **plan.analysis.model_dump(mode="json"),
            "plan_sha256": canonical_plan_hash(plan),
        },
        "scope": _scope(plan),
        "presentation": _presentation(plan),
        "multiplicity": [family.model_dump(mode="json") for family in plan.multiplicity.families],
    }


def _design_frame(plan: AnalysisPlan) -> dict[str, Any]:
    """The report's design fields that come from the plan alone."""

    return {
        "manifest": report_path(plan.inputs.design_manifest, plan),
        "hierarchy": plan.design.hierarchy,
        "independence_profiles": {
            profile_id: profile.model_dump(mode="json")
            for profile_id, profile in plan.design.independence_profiles.items()
        },
    }


def _unread_input_report(
    plan: AnalysisPlan,
    diagnostics: list[Diagnostic],
    frozen: AnalysisPlan | None = None,
    policy_record: DecisionPolicyRecord | None = None,
) -> SafetyReport:
    """The report of an analysis whose input could not be read.

    The diagnostics carry the ``INGEST_FAILED`` error, so integrity is invalid. The report has
    no observations, estimates or comparisons, and every claim is INCONCLUSIVE, so that the
    failure is written down and audited like any other invalid analysis.
    """

    families = {family.id: family for family in plan.multiplicity.families}
    claims = [
        ClaimResult(
            claim_id=claim.id,
            title=claim.title,
            status="INCONCLUSIVE",
            statistical_status="INCONCLUSIVE",
            policy_status="INCONCLUSIVE",
            acceptable_upper=claim.acceptable_upper,
            unacceptable_lower=claim.unacceptable_lower,
            verdict_range=VerdictRange(),
            alpha=families[claim.family].alpha if claim.family in families else None,
            multiplicity_family=claim.family,
            method="unavailable",
            mandatory=claim.mandatory,
            evidence_refs=[
                f"comparison:{claim.comparison}"
                if claim.comparison
                else f"risk:{claim.cohort}:{claim.group}:{claim.measure}"
            ],
            reasons=["The analysis input could not be read; see the INGEST_FAILED diagnostic."],
        )
        for claim in sorted(plan.claims, key=lambda item: item.id)
    ]
    frozen = plan if frozen is None else frozen
    report = SafetyReport(
        **_report_frame(frozen),
        provenance={"source_format": {}, "sources": []},
        integrity=_sorted_integrity(diagnostics),
        design={
            **_design_frame(plan),
            "manifest_sha256": None,
            "prompt_count": None,
            "groups": [],
            "prompts": [],
        },
        cohorts=[],
        prompt_estimates=[],
        group_risks=[],
        comparisons=[],
        claims=claims,
        policy_events=[],
        decision=roll_up(
            [claim.status for claim in claims if claim.mandatory],
            integrity_valid=False,
            zero_tolerance_failure=False,
            confirmatory=plan.analysis.intent == "confirmatory",
        ),
        decision_policy=policy_record,
    )
    report._resolved_plan = report_plan(frozen)
    return report
