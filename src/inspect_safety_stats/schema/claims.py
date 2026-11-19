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
    """The bound the claim was decided on, as the claim in report.json gives it.

    ``alpha`` is the claim's error budget, which is not ``evidence.family_alpha``, the budget
    of the multiplicity family the evidence interval was computed in. For a claim decided on a
    validated judge correction the bound is the corrected one, and ``alpha`` is the claim
    family's alpha plus the correction's ``evidence.family_alpha``. For a claim decided by the
    noisy-but-valid test, when the test runs, the bound is the test's set of true risks, and
    ``alpha`` is the claim family's alpha plus the judge family's whole alpha. When the test
    cannot run, ``lower`` and ``upper`` are None and ``alpha`` is the claim family's alone.

    A claim can be INCONCLUSIVE without a decision on its bound and still carry the bound
    report.json gives it. Among the cases: its multiplicity family is undeclared (``alpha`` is
    None), or the plan's validated judge calibration has no certifying correction for it
    (``judge_correction_unavailable``). In the second the bound is the uncorrected
    judge-reported one, not a bound on the true risk the claim is about. A claim whose family
    is undeclared carries that uncorrected bound too under a validated calibration, but is not
    flagged: its None ``alpha`` shows it. The record's ``reasons`` say why.
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
    judge-corrected claim), else the result's primary interval. A claim decided by a test
    (``test:<name>``, now only ``noisy_valid_certification``) has the test's bound, which the
    test computes from the result's primary interval and the judge calibration: ``test`` names
    the test, and the interval is that primary interval, whatever interval the refs name. When
    the test cannot run, ``test`` still names it, as the test the claim was to be decided by.
    When the test runs, the record's ``reasons`` describe the test and its calibration; when it
    cannot, they say why. Either way ``assumptions`` are the interval's alone, and claims.json
    does not list the test's assumptions. The interval fields are None when that result, or
    that interval, is not in the report.

    ``family_alpha`` is the interval's own: the error budget of the multiplicity family it was
    computed in. It is not the claim's error budget, ``bound.alpha``. For a judge correction it
    is the share of the judge family's alpha that the correction spends. The calibration
    divides that alpha equally over its scopes, one per declared cohort, or per cohort and
    group when it is scoped to groups, so with several scopes it is a share, not the whole
    budget; a comparison's correction spends the shares of the two it spans. A
    judge-corrected claim's ``bound.alpha`` is the claim family's alpha plus this
    ``family_alpha``.
    """

    refs: list[str] = Field(default_factory=list)
    # The test the claim was decided by, or was to be decided by when the test cannot run, as
    # its ``test:<name>`` ref names it; None when the claim was decided on an interval.
    test: str | None = None
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
    # The plan declares a validated judge calibration, but the claim's evidence has no
    # certifying correction: the claim is INCONCLUSIVE, and its bound is the uncorrected
    # judge-reported one, on which it was not decided.
    judge_correction_unavailable: bool
    estimand: ClaimEstimand
    bound: ClaimBound
    evidence: ClaimEvidence
    sources: ClaimSources
    # Why the claim has its status, as report.json gives them.
    reasons: list[str]


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
