from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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


class ClaimResult(ResultModel):
    claim_id: str
    title: str
    status: Decision
    statistical_status: Decision
    policy_status: Decision
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    acceptable_upper: float
    unacceptable_lower: float
    alpha: float
    multiplicity_family: str
    method: str
    mandatory: bool
    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SafetyReport(ResultModel):
    schema_version: Literal["2.0"] = "2.0"
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
