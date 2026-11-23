"""claims.json, rendered from the report model alone (experimental, #11)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from safety_stats.schema.claims import (
    ClaimBound,
    ClaimEstimand,
    ClaimEvidence,
    ClaimRecord,
    ClaimsAnalysis,
    ClaimsExport,
    ClaimSources,
    ClaimsProvenance,
    ClaimsSource,
)
from safety_stats.schema.reports import (
    ClaimEvidenceRecord,
    ComparisonResult,
    GroupRiskResult,
    SafetyReport,
)

CLAIMS_JSON = "claims.json"


def _named_claims(report: SafetyReport, code: str) -> set[str]:
    """The claims the diagnostics with ``code`` name."""

    return {
        str(claim_id)
        for diagnostic in report.integrity.diagnostics
        if diagnostic.code == code
        for claim_id in diagnostic.details.get("claim_ids") or []
    }


def _named(refs: Sequence[str], kind: str) -> str | None:
    """What the first ``<kind>:<name>`` evidence ref names, if there is one."""

    prefix = f"{kind}:"
    return next((ref.removeprefix(prefix) for ref in refs if ref.startswith(prefix)), None)


def _evidence(
    report: SafetyReport, record: ClaimEvidenceRecord
) -> tuple[str | None, GroupRiskResult | ComparisonResult | None]:
    """The result a claim was decided on and its JSON pointer, found as the analysis finds it."""

    if record.comparison is not None:
        for index, comparison in enumerate(report.comparisons):
            if comparison.comparison_id == record.comparison and comparison.level is None:
                return f"/comparisons/{index}", comparison
        return None, None
    for index, risk in enumerate(report.group_risks):
        if (risk.cohort_id, risk.group_id, risk.measure_id) == (
            record.cohort,
            record.group,
            record.measure,
        ):
            return f"/group_risks/{index}", risk
    return None, None


def claims_export(report: SafetyReport) -> ClaimsExport:
    """Every claim of the report as one record, each value copied from the report."""

    evidence_records = {
        item.claim_id: (index, item)
        for index, item in enumerate(report.presentation.claim_evidence or [])
    }
    corrected = _named_claims(report, "JUDGE_CORRECTED_CLAIMS")
    uncorrected = _named_claims(report, "JUDGE_CORRECTION_UNAVAILABLE")
    records: list[ClaimRecord] = []
    for index, claim in enumerate(report.claims):
        if claim.claim_id not in evidence_records:
            raise ValueError(
                f"claim {claim.claim_id!r} has no record in presentation.claim_evidence, so "
                "claims.json cannot say what it is about"
            )
        evidence_index, estimand = evidence_records[claim.claim_id]
        evidence_pointer, evidence = _evidence(report, estimand)
        test = _named(claim.evidence_refs, "test")
        interval_pointer = None
        interval = None
        if evidence is not None and evidence_pointer is not None:
            # A test's bound is the test's own, computed from the result's primary interval.
            # A noisy-valid claim also names the validated correction, which the analysis
            # records before deciding the claim by the test; its bound does not come from it.
            named = None if test is not None else _named(claim.evidence_refs, "interval")
            interval_id = named if named is not None else evidence.primary_interval_id
            for position, item in enumerate(evidence.intervals):
                if item.interval_id == interval_id:
                    interval_pointer = f"{evidence_pointer}/intervals/{position}"
                    interval = item
                    break
        records.append(
            ClaimRecord(
                claim_id=claim.claim_id,
                title=claim.title,
                status=claim.status,
                descriptive_status=claim.descriptive_status,
                mandatory=claim.mandatory,
                judge_corrected=claim.claim_id in corrected,
                judge_correction_unavailable=claim.claim_id in uncorrected,
                estimand=ClaimEstimand(
                    cohort=estimand.cohort,
                    group=estimand.group,
                    measure=estimand.measure,
                    measure_type=estimand.measure_type,
                    comparison=estimand.comparison,
                    acceptable_upper=claim.acceptable_upper,
                    unacceptable_lower=claim.unacceptable_lower,
                ),
                bound=ClaimBound(
                    estimate=claim.estimate,
                    lower=claim.lower,
                    upper=claim.upper,
                    alpha=claim.alpha,
                    method=claim.method,
                    multiplicity_family=claim.multiplicity_family,
                ),
                evidence=ClaimEvidence(
                    refs=list(claim.evidence_refs),
                    test=test,
                    eligibility=evidence.eligibility if evidence else None,
                    interval_id=interval.interval_id if interval else None,
                    interval_method=interval.method if interval else None,
                    coverage_mode=interval.coverage_mode if interval else None,
                    coverage_scope=interval.coverage_scope if interval else None,
                    family_alpha=interval.family_alpha if interval else None,
                    independence_profile=interval.independence_profile if interval else None,
                    assumptions=list(interval.assumptions) if interval else [],
                ),
                sources=ClaimSources(
                    claim=f"/claims/{index}",
                    estimand=f"/presentation/claim_evidence/{evidence_index}",
                    evidence=evidence_pointer,
                    evidence_interval=interval_pointer,
                ),
                reasons=list(claim.reasons),
            )
        )
    analysis: dict[str, Any] = report.analysis
    return ClaimsExport(
        report_schema_version=report.schema_version,
        package_version=report.package_version,
        analysis=ClaimsAnalysis(
            id=analysis.get("id"),
            intent=analysis.get("intent"),
            plan_sha256=analysis.get("plan_sha256"),
            plan_frozen_at=analysis.get("plan_frozen_at"),
        ),
        provenance=ClaimsProvenance(
            sources=[
                ClaimsSource(path=item["path"], sha256=item.get("sha256"))
                for item in report.provenance.get("sources") or []
            ],
            manifest_sha256=report.design.get("manifest_sha256"),
        ),
        integrity_status=report.integrity.status,
        decision=report.decision,
        claims=records,
    )
