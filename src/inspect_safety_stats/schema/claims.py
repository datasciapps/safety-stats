"""The schema of claims.json: a report's claims as one record each (experimental, #11).

Every value is copied from report.json, and each record names, as JSON pointers (RFC 6901), the
places in report.json it was copied from. ``inspect-safety-stats schema claims`` prints the JSON
schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from inspect_safety_stats.schema.reports import Decision, Eligibility, ResultModel


class ClaimEstimand(ResultModel):
    """What the claim is about, and the thresholds it is decided at."""

    cohort: str | None = None
    group: str | None = None
    measure: str | None = None
    measure_type: str | None = None
    comparison: str | None = None
    acceptable_upper: float
    unacceptable_lower: float


class ClaimBound(ResultModel):
    """The bound the claim was decided on.

    For a claim decided on a validated judge correction this is the corrected bound, and
    ``alpha`` is the error budget of the claim family and the judge calibration together.
    """

    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    alpha: float | None = None
    method: str
    multiplicity_family: str


class ClaimEvidence(ResultModel):
    """The result the claim was decided on, and the interval its bound comes from.

    The interval is the one the claim's evidence refs name (``interval:<id>``, as for a
    judge-corrected claim), else the result's primary interval. The interval fields are None
    when that result, or that interval, is not in the report.
    """

    refs: list[str] = Field(default_factory=list)
    eligibility: Eligibility | None = None
    interval_id: str | None = None
    interval_method: str | None = None
    coverage_mode: Literal["two_sided", "per_direction"] | None = None
    coverage_scope: str | None = None
    family_alpha: float | None = None
    independence_profile: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class ClaimSources(ResultModel):
    """JSON pointers into report.json to the values a record copies."""

    claim: str
    estimand: str
    evidence: str | None = None
    evidence_interval: str | None = None


class ClaimRecord(ResultModel):
    claim_id: str
    title: str
    status: Decision
    # The status a confirmatory analysis would give, in an exploratory one; it certifies nothing.
    descriptive_status: Decision | None = None
    mandatory: bool
    judge_corrected: bool
    estimand: ClaimEstimand
    bound: ClaimBound
    evidence: ClaimEvidence
    sources: ClaimSources


class ClaimsAnalysis(ResultModel):
    id: str | None = None
    intent: str | None = None
    plan_sha256: str | None = None
    plan_frozen_at: str | None = None


class ClaimsSource(ResultModel):
    path: str
    # None when the log could not be read to hash it, as in report.json.
    sha256: str | None = None


class ClaimsProvenance(ResultModel):
    sources: list[ClaimsSource] = Field(default_factory=list)
    manifest_sha256: str | None = None


class ClaimsExport(ResultModel):
    """claims.json: every claim of a report, with its bound, evidence and provenance.

    Experimental. Written when the plan sets ``report.claims_export``. The provenance is the
    report's and holds for every record.
    """

    schema_version: Literal["1.0"] = "1.0"
    experimental: Literal[True] = True
    report_schema_version: str
    package_version: str
    analysis: ClaimsAnalysis
    provenance: ClaimsProvenance
    integrity_status: Literal["valid", "degraded", "invalid"]
    decision: Decision
    claims: list[ClaimRecord]
