from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

Decision = Literal["PASS", "FAIL", "INCONCLUSIVE"]
Eligibility = Literal["certifying", "diagnostic", "invalid"]


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Diagnostic(ResultModel):
    code: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    message: str
    scope: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class IntegritySummary(ResultModel):
    status: Literal["valid", "degraded", "invalid"]
    diagnostics: list[Diagnostic] = Field(default_factory=list)


class PromptEstimate(ResultModel):
    cohort_id: str
    prompt_id: str
    question_id: str
    task_id: str
    n_total: int
    n_expected: int | None = None
    n_valid: int
    n_failure: int
    n_safe: int
    n_error: int
    n_unclassified: int = 0
    n_policy_blocked_safe: int = 0
    p_hat: float | None = None
    lower: float | None = None
    upper: float | None = None
    alpha_lower: float | None = None
    alpha_upper: float | None = None
    method: str | None = None
    missingness_robust: bool = False
    eligibility: Eligibility = "invalid"
    warnings: list[str] = Field(default_factory=list)


class RiskIntervalResult(ResultModel):
    interval_id: str
    lower: float | None = None
    upper: float | None = None
    method: str
    role: Literal["primary", "reference", "sensitivity"]
    # What a sensitivity interval varies: the independence profile its blocks come from, or
    # the judge's error rates (an assumed calibration scenario or an uncertified correction).
    # None for primary and reference intervals, and in reports written before the field.
    varied_assumption: Literal["independence", "judge_error"] | None = None
    coverage_scope: str | None = None
    coverage_mode: Literal["two_sided", "per_direction"] | None = None
    family: str | None = None
    family_alpha: float | None = None
    directional_family_alpha: float | None = None
    alpha_lower: float | None = None
    alpha_upper: float | None = None
    simultaneous_coverage: float | None = None
    lower_one_sided_coverage: float | None = None
    upper_one_sided_coverage: float | None = None
    completion_estimate_lower: float | None = None
    completion_estimate_upper: float | None = None
    half_width_lower: float | None = None
    half_width_upper: float | None = None
    effective_sample_size: float | None = None
    concentration_variance: float | None = None
    variance_upper_lower: float | None = None
    variance_upper_upper: float | None = None
    maximum_increment: float | None = None
    chernoff_kernel: str | None = None
    raw_lower: float | None = None
    raw_upper: float | None = None
    projection_method: str | None = None
    projection_constraint_count: int | None = None
    calibration_method: str | None = None
    calibration_frontier: str | None = None
    calibration_frontier_alpha: float | None = None
    calibration_frontier_size: int | None = None
    candidate_lower: float | None = None
    candidate_upper: float | None = None
    baseline_lower: float | None = None
    baseline_upper: float | None = None
    independence_profile: str | None = None
    independent_block_count: int | None = None
    missingness_method: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    eligibility: Eligibility
    warnings: list[str] = Field(default_factory=list)
    # Experimental (#127), set only when the plan asks for it.
    single_valued_bound: float | None = None
    single_valued_bound_delta: float | None = None
    single_valued_bound_guarantee: str | None = None

    @model_validator(mode="after")
    def _varied_assumption_only_on_sensitivity(self) -> RiskIntervalResult:
        if self.varied_assumption is not None and self.role != "sensitivity":
            raise ValueError(
                f"interval {self.interval_id!r} has role {self.role!r}; only a sensitivity "
                "interval names a varied_assumption"
            )
        return self

    @model_serializer(mode="wrap")
    def leave_out_absent_single_valued_bound(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Left out unless requested, so a report that does not use the experimental bound
        # keeps the bytes it had before the bound existed.
        data: dict[str, Any] = handler(self)
        if self.single_valued_bound is None:
            for name in _SINGLE_VALUED_FIELDS:
                data.pop(name, None)
        return data


_SINGLE_VALUED_FIELDS = (
    "single_valued_bound",
    "single_valued_bound_delta",
    "single_valued_bound_guarantee",
)


class GroupRiskResult(ResultModel):
    cohort_id: str
    group_id: str
    measure_id: str
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    method: str
    primary_interval_id: str | None = None
    intervals: list[RiskIntervalResult] = Field(default_factory=list)
    interval_method: str | None = None
    raw_lower: float | None = None
    raw_upper: float | None = None
    projection_method: str | None = None
    coverage_scope: str | None = None
    coverage_mode: Literal["two_sided", "per_direction"] | None = None
    family_alpha: float | None = None
    directional_family_alpha: float | None = None
    simultaneous_coverage: float | None = None
    lower_one_sided_coverage: float | None = None
    upper_one_sided_coverage: float | None = None
    prompt_count: int
    valid_prompt_count: int
    resolved_weight: float
    unresolved_weight: float
    parameters: dict[str, Any] = Field(default_factory=dict)
    eligibility: Eligibility
    certified_safe_weight: float | None = None
    certified_unsafe_weight: float | None = None
    warnings: list[str] = Field(default_factory=list)


class HierarchyRiskResult(ResultModel):
    cohort_id: str
    group_id: str
    level: str
    path: dict[str, str]
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    primary_interval_id: str | None = None
    intervals: list[RiskIntervalResult] = Field(default_factory=list)
    interval_method: str | None = None
    raw_lower: float | None = None
    raw_upper: float | None = None
    projection_method: str | None = None
    coverage_scope: str | None = None
    simultaneous_coverage: float | None = None
    lower_one_sided_coverage: float | None = None
    upper_one_sided_coverage: float | None = None
    design_weight: float
    prompt_count: int
    valid_prompt_count: int
    eligibility: Eligibility
    warnings: list[str] = Field(default_factory=list)


class ComparisonResult(ResultModel):
    comparison_id: str
    comparison_set_id: str | None = None
    result_id: str | None = None
    candidate_cohort: str
    baseline_cohort: str
    group_id: str
    measure_id: str
    estimand: str
    level: str | None = None
    path: dict[str, str] = Field(default_factory=dict)
    candidate_estimate: float | None = None
    baseline_estimate: float | None = None
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    method: str
    primary_interval_id: str | None = None
    intervals: list[RiskIntervalResult] = Field(default_factory=list)
    interval_method: str | None = None
    raw_lower: float | None = None
    raw_upper: float | None = None
    projection_method: str | None = None
    coverage_scope: str | None = None
    coverage_mode: Literal["two_sided", "per_direction"] | None = None
    family: str | None = None
    family_alpha: float | None = None
    directional_family_alpha: float | None = None
    simultaneous_coverage: float | None = None
    lower_one_sided_coverage: float | None = None
    upper_one_sided_coverage: float | None = None
    independence_profile: str | None = None
    prompt_count: int = 0
    candidate_rollout_count: int = 0
    baseline_rollout_count: int = 0
    resolved_weight: float = 1.0
    effect_summaries: dict[str, float | None] = Field(default_factory=dict)
    pair_count: int | None = None
    discordant_counts: dict[str, int] = Field(default_factory=dict)
    p_value: float | None = None
    cluster_count: int | None = None
    diagnostic_outputs: dict[str, Any] = Field(default_factory=dict)
    eligibility: Eligibility
    warnings: list[str] = Field(default_factory=list)


class ComparisonSetResult(ResultModel):
    comparison_set_id: str
    contrast_type: Literal["versus_reference", "all_pairs"]
    cohorts: list[str]
    reference_cohort: str | None = None
    group_id: str
    measure_id: str
    estimands: list[str]
    comparison_ids: list[str]
    arm_estimates: dict[str, float | None] = Field(default_factory=dict)
    arm_intervals: dict[str, list[float]] = Field(default_factory=dict)
    rank_intervals: dict[str, list[int]] = Field(default_factory=dict)
    independence_profile: str
    simultaneous_coverage: float
    warnings: list[str] = Field(default_factory=list)


class ClaimProfileVerdict(ResultModel):
    """A claim's verdict under one declared independence profile.

    The primary profile's row repeats the claim's own status. A sensitivity row decides the
    same thresholds on that profile's interval of the same method; it is shown for
    robustness and never replaces the primary verdict.
    """

    independence_profile: str
    role: Literal["primary", "sensitivity"]
    status: Decision
    interval_id: str
    lower: float | None = None
    upper: float | None = None
    independent_block_count: int | None = None
    effective_sample_size: float | None = None


class JudgeErrorStressResult(ResultModel):
    """How good the judge must be for a threshold claim's verdict to hold on true risk.

    Experimental and diagnostic: it never moves a claim. ``applicable`` is False, with the
    reason, when the claim was not decided on judge-reported bounds this can stress.
    """

    applicable: bool
    reason: str | None = None
    judge_reported_lower: float | None = None
    judge_reported_upper: float | None = None
    unclassified_share: float | None = None
    pass_sensitivity_at_perfect_specificity: float | None = None
    pass_specificity_at_perfect_sensitivity: float | None = None
    fail_sensitivity_at_perfect_specificity: float | None = None
    fail_specificity_at_perfect_sensitivity: float | None = None
    accuracy_source: str | None = None
    sensitivity_lower: float | None = None
    sensitivity_upper: float | None = None
    specificity_lower: float | None = None
    specificity_upper: float | None = None
    stressed_status: Decision | None = None
    true_risk_lower: float | None = None
    true_risk_upper: float | None = None
    identified: bool | None = None
    robust: bool | None = None


class ClaimResult(ResultModel):
    claim_id: str
    title: str
    status: Decision
    statistical_status: Decision
    policy_status: Decision
    # In an exploratory analysis, the status a confirmatory analysis of the same data would
    # give. It certifies nothing and never enters the decision. None in a confirmatory
    # analysis, and when there are no certifying bounds to compare with the thresholds.
    descriptive_status: Decision | None = None
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    acceptable_upper: float
    unacceptable_lower: float
    # None when the claim's family is undeclared: such a claim has no error budget.
    alpha: float | None
    multiplicity_family: str
    method: str
    mandatory: bool
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    profile_verdicts: list[ClaimProfileVerdict] = Field(default_factory=list)
    # Present only when the plan opts in with report.judge_error_stress.
    judge_error_stress: JudgeErrorStressResult | None = None

    @model_serializer(mode="wrap")
    def leave_out_absent_judge_error_stress(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.judge_error_stress is None:
            data.pop("judge_error_stress", None)
        return data


class SafetyReport(ResultModel):
    """The result of ``analyze``: estimates, bounds, comparisons, claims and the decision.

    Its results apply to the declared fixed evaluation set. ``write`` saves it as a report
    directory, and ``inspect-safety-stats schema report`` prints the JSON schema of its
    ``report.json``.
    """

    schema_version: Literal["3.0"] = "3.0"
    package_version: str
    analysis: dict[str, Any]
    provenance: dict[str, Any]
    scope: dict[str, Any]
    presentation: dict[str, Any] = Field(default_factory=dict)
    integrity: IntegritySummary
    design: dict[str, Any]
    cohorts: list[dict[str, Any]]
    prompt_estimates: list[PromptEstimate]
    group_risks: list[GroupRiskResult]
    hierarchy_risks: list[HierarchyRiskResult] = Field(default_factory=list)
    comparisons: list[ComparisonResult]
    comparison_sets: list[ComparisonSetResult] = Field(default_factory=list)
    multiplicity: list[dict[str, Any]]
    claims: list[ClaimResult]
    policy_events: list[dict[str, Any]]
    decision: Decision
    _observations: list[Any] = PrivateAttr(default_factory=list)
    _resolved_design: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _resolved_plan: dict[str, Any] = PrivateAttr(default_factory=dict)

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        from inspect_safety_stats.report.writer import write_report

        return write_report(self, output, overwrite=overwrite)
