from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    computed_field,
    model_serializer,
    model_validator,
)

from inspect_safety_stats.registry import CODES, Effect, SourceKind
from inspect_safety_stats.schema.pointers import ItemPointers
from inspect_safety_stats.schema.scopes import scope_segments

Decision = Literal["PASS", "FAIL", "INCONCLUSIVE"]
Eligibility = Literal["certifying", "diagnostic", "invalid"]

# Validation context key: the model is read from a stored report, not built by an analysis.
STORED_REPORT = "stored_report"


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Diagnostic(ResultModel):
    """One finding of an analysis.

    ``source`` and ``effect`` are what the registry says of the code
    (``inspect_safety_stats.registry.CODES``): the kind of input the finding concerns and what
    it does to the results. They are filled from the registry, so a reader of ``report.json``
    can act on a diagnostic without looking its code up. A code the registry does not list is
    rejected, and so is a ``source`` or ``effect`` that differs from its entry.

    ``scope`` is a path that says where the finding sits, as ``inspect_safety_stats.schema.scopes``
    describes: ``plan/claims/mean_safe``, ``cohorts/candidate/prompts/p1``. A scope that does
    not follow it is rejected, except when a stored report is read back (``STORED_REPORT``).

    ``item`` is a JSON pointer into ``report.json`` to the record the scope names, such as
    ``/claims/0`` or ``/prompt_estimates/12``, and null when the report holds no record of it
    (``inspect_safety_stats.schema.pointers``). The report fills it from the scope, also when
    it is read back, so it always points into the report that holds the diagnostic.
    """

    code: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    source: SourceKind | None = None
    effect: Effect | None = None
    message: str
    scope: str | None = None
    item: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_source_and_effect_from_the_registry(self, info: ValidationInfo) -> Diagnostic:
        entry = CODES.get(self.code)
        if entry is None:
            raise ValueError(
                f"diagnostic code {self.code!r} has no entry in inspect_safety_stats.registry"
            )
        for name, registered in (("source", entry.source), ("effect", entry.effect)):
            given = getattr(self, name)
            if given is None:
                setattr(self, name, registered)
            elif given != registered:
                raise ValueError(
                    f"diagnostic {self.code} gives {name} {given!r}; the registry says "
                    f"{registered!r}"
                )
        # A report written before scopes were paths keeps its free-text scopes; reading it back
        # (``STORED_REPORT``) must not reject them.
        stored = bool(info.context and info.context.get(STORED_REPORT))
        if self.scope is not None and not stored:
            scope_segments(self.scope)
        return self


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
    # Rollouts whose answer stopped at the generation token limit without a score or a sample
    # error, counted as the plan's score.token_limit says; None when it is "score", which does
    # not look.
    n_token_limit: int | None = None
    p_hat: float | None = None
    lower: float | None = None
    upper: float | None = None
    alpha_lower: float | None = None
    alpha_upper: float | None = None
    method: str | None = None
    missingness_robust: bool = False
    eligibility: Eligibility = "invalid"
    warnings: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def leave_out_absent_token_limit_count(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Left out unless the plan's score.token_limit looks, so a report that does not keeps
        # the bytes it had before the field existed.
        data: dict[str, Any] = handler(self)
        if self.n_token_limit is None:
            data.pop("n_token_limit", None)
        return data


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


class VerdictRange(ResultModel):
    """The thresholds at which a claim would PASS or FAIL, read off the bounds it was decided on.

    The claim is PASS for every ``acceptable_upper`` at or above ``pass_acceptable_upper``
    (strictly above when ``pass_strict``), and FAIL for every ``unacceptable_lower`` at or below
    ``fail_unacceptable_lower`` (strictly below when ``fail_strict``). Each holds whatever the
    other threshold of a valid pair is. A ``None`` boundary means that no threshold the plan
    allows gives that verdict. Every other pair of thresholds gives INCONCLUSIVE.

    The noisy-but-valid test takes thresholds in [0, 1] only, so its range speaks only of pairs
    in [0, 1]: with a negative ``acceptable_upper`` the test does not run.
    """

    pass_acceptable_upper: float | None = None
    pass_strict: bool = False
    fail_unacceptable_lower: float | None = None
    fail_strict: bool = False


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
    # None when the plan's thresholds are ones its test cannot take, which leaves the range
    # unknown, and in reports written before verdict ranges were reported.
    verdict_range: VerdictRange | None = None
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


class AdaptiveClaimResult(ResultModel):
    """Experimental (#125): a claim that can only FAIL, on an attacker that adapts across runs.

    ``lower`` bounds the attacker's running mean of conditional success probabilities over the
    ``runs`` behaviours in ``order_field`` order, anytime-valid at ``alpha`` and with no
    independence between runs. It describes this attacker at this budget only. The status is
    FAIL or INCONCLUSIVE, never PASS.
    """

    claim_id: str
    title: str
    status: Literal["FAIL", "INCONCLUSIVE"]
    # In an exploratory analysis, what a confirmatory analysis of the same data would give.
    descriptive_status: Literal["FAIL", "INCONCLUSIVE"] | None = None
    cohort_id: str
    group_id: str
    order_field: str
    # The behaviours in the stream, up to the last one attacked, and the successes among them.
    runs: int | None = None
    successes: int | None = None
    lower: float | None = None
    unacceptable_lower: float
    alpha: float | None
    multiplicity_family: str
    method: str
    mandatory: bool
    reasons: list[str] = Field(default_factory=list)


class PolicyThresholdsRecord(ResultModel):
    acceptable_upper: float
    unacceptable_lower: float


class DecisionPolicyVersionRecord(ResultModel):
    id: str
    set_at: str
    reason: str | None = None
    thresholds: dict[str, PolicyThresholdsRecord]
    # Whether the version was set after the earliest start time of the selected evaluations;
    # None when no evaluation reports a start time.
    set_after_first_evaluation: bool | None = None


class ClaimThresholdsRecord(ResultModel):
    """The thresholds a claim was decided at, and where they came from."""

    claim_id: str
    acceptable_upper: float
    unacceptable_lower: float
    # "plan", or the id of the policy version that set them last.
    set_by: str
    plan_acceptable_upper: float
    plan_unacceptable_lower: float


class DecisionPolicyRecord(ResultModel):
    """The decision policy an analysis applied (experimental).

    Its thresholds replace the plan's for the claims it sets, while ``analysis.plan_sha256``
    stays the hash of the plan as frozen. ``applied`` is false when the policy failed a check
    (see the ``DECISION_POLICY_*`` diagnostics); the claims then keep the plan's thresholds.
    """

    inference_plan_sha256: str
    policy_inference_plan_sha256: str
    policy_sha256: str
    applied: bool
    first_evaluation_started_at: str | None = None
    versions: list[DecisionPolicyVersionRecord]
    claims: list[ClaimThresholdsRecord]


class RateRangeRecord(ResultModel):
    lower: float
    upper: float


class JudgeCohortBoxRecord(ResultModel):
    """A declared scenario's box for one cohort."""

    rho_1: RateRangeRecord
    rho_0: RateRangeRecord


class JudgeScenarioRecord(ResultModel):
    """A declared judge-accuracy scenario: ``rho_1`` is P(truly unsafe | judge said unsafe)
    and ``rho_0`` is P(truly unsafe | judge said safe)."""

    id: str
    rho_1: RateRangeRecord
    rho_0: RateRangeRecord
    # The boxes the scenario gives cohorts of their own, by cohort; None when it gives every
    # cohort the box above, which keeps report.json as it was before the field existed.
    cohorts: dict[str, JudgeCohortBoxRecord] | None = None

    @model_serializer(mode="wrap")
    def leave_out_absent_cohorts(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.cohorts is None:
            data.pop("cohorts", None)
        return data


class JudgeCalibrationPresentation(ResultModel):
    """The plan's judge calibration, as far as the report's text depends on it."""

    mode: Literal["assumed", "validated", "transported"]
    sampling: str | None = None
    primary: str | None = None
    # The groups the calibration is scoped to; None when it applies to every group.
    groups: list[str] | None = None
    scenarios: list[JudgeScenarioRecord] = Field(default_factory=list)


class ClaimEvidenceRecord(ResultModel):
    """The result a claim was decided on: a comparison, or a cohort's measure on a group."""

    claim_id: str
    comparison: str | None = None
    cohort: str | None = None
    group: str | None = None
    measure: str | None = None
    # The plan's type of that measure, such as ``weighted_mean``.
    measure_type: str | None = None


class ReportPresentation(ResultModel):
    """What the rendered report (Markdown, HTML, CSV) needs beyond the results.

    A report is rendered from ``report.json`` alone. Fields that are None were not recorded,
    in reports written before they existed.

    ``include_decisions`` hides decisions; it does not withhold them (#58). Every rendering
    leaves out the overall decision and the claims' statuses when it is false, while
    ``report.json`` keeps both. A report that does not record it reads as the plan's default,
    false, here and in ``report.html``.

    ``SafetyReport.presentation`` used to be a dictionary. Reading it as one
    (``presentation["include_decisions"]``, ``presentation.get(...)``, ``in``) is deprecated:
    it still works and raises a ``DeprecationWarning``. A field that is None reads as a
    missing key, as ``report.json`` leaves it out.
    """

    include_decisions: bool = False
    decimal_places: int | None = None
    primary_independence_profile: str | None = None
    claim_evidence: list[ClaimEvidenceRecord] | None = None
    judge_calibration: JudgeCalibrationPresentation | None = None
    # The plan's primary outcome mapping; recorded only when the plan declares outcome mapping
    # sensitivities (experimental), which show it beside the alternatives.
    primary_outcome_mapping: dict[str, Any] | None = None
    # True when the plan asks for claims.json (report.claims_export, experimental); None
    # otherwise, which keeps report.json as it was before the export existed.
    claims_export: bool | None = None
    # report.precision_target when the plan sets it; None at the default, which keeps
    # report.json as it was before the summary page read it.
    precision_target: float | None = None

    @model_serializer(mode="wrap")
    def leave_out_absent_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # A field a report does not record stays out of report.json, rather than null.
        data: dict[str, Any] = handler(self)
        return {key: value for key, value in data.items() if value is not None}

    def _deprecated_read(self, key: str) -> Any:
        # stacklevel 3 names the code that called __getitem__, get or __contains__.
        advice = (
            f"read report.presentation.{key} instead"
            if key in type(self).model_fields
            else f"it has no field {key!r}"
        )
        warnings.warn(
            "SafetyReport.presentation is a ReportPresentation, not a dictionary; "
            f"{advice}. Reading it as a dictionary will be removed in a later release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return getattr(self, key) if key in type(self).model_fields else None

    def __getitem__(self, key: str) -> Any:
        value = self._deprecated_read(key)
        if value is None:
            raise KeyError(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._deprecated_read(key)
        return default if value is None else value

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._deprecated_read(key) is not None


class MappingSensitivityResult(ResultModel):
    """One displayed result as an alternative outcome mapping reads it."""

    kind: Literal["group_risk", "comparison", "claim"]
    result_id: str
    cohort_id: str | None = None
    group_id: str | None = None
    measure_id: str | None = None
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    # A claim's status under this mapping. It certifies nothing and never enters the decision.
    status: Decision | None = None


class OutcomeMappingSensitivity(ResultModel):
    """Experimental: the analysis repeated under one declared alternative outcome mapping.

    Each mapping's intervals carry the coverage the plan declares for that mapping alone; they
    are not simultaneous across mappings. Nothing here enters a claim's status or the decision.
    """

    id: str
    title: str | None = None
    mapping: dict[str, Any]
    integrity_status: Literal["valid", "degraded", "invalid"]
    # How many diagnostics of each code this mapping's analysis raised.
    diagnostic_counts: dict[str, int] = Field(default_factory=dict)
    # The decision this mapping alone would give. Descriptive only.
    decision: Decision
    results: list[MappingSensitivityResult] = Field(default_factory=list)


class PlanDefault(ResultModel):
    """A plan field the author left out, and the value its default gave the analysis."""

    # Dotted field names, ``[i]`` for a list item and the key for a mapping entry.
    path: str
    value: Any


class PlanSetting(ResultModel):
    """The value a plan field had in the analysis, and whether it came from a default."""

    path: str
    value: Any
    default: bool


class AssumptionUse(ResultModel):
    """An assumption of the registry the results rest on, with the plan settings it took.

    ``id``, ``statement`` and ``if_wrong`` are the registry entry's
    (``inspect_safety_stats.registry.ASSUMPTIONS``). ``settings`` holds the value of each plan
    field that sets the assumption, empty when no plan field does.
    """

    id: str
    statement: str
    if_wrong: str
    settings: list[PlanSetting] = Field(default_factory=list)


def design_levels(hierarchy: Sequence[str]) -> list[str]:
    """The identifier columns of a row of ``design.prompts``: the design's hierarchy levels,
    then ``task_id``, ``question_id`` and ``prompt_id``, each once."""

    return list(dict.fromkeys([*hierarchy, "task_id", "question_id", "prompt_id"]))


def design_columns(hierarchy: Sequence[str]) -> list[str]:
    """The columns of a row of ``design.prompts``, in the order of ``design.resolved.csv``:
    ``group_id``, the identifier columns (``design_levels``), then ``weight``.

    ``report.json`` stores each row with its keys sorted, so this is the one place their order
    is defined.
    """

    return ["group_id", *design_levels(hierarchy), "weight"]


class SafetyReport(ResultModel):
    """The result of ``analyze``: estimates, bounds, comparisons, claims and the decision.

    Its results apply to the declared fixed evaluation set or, under a sampled design, to the
    declared item frame; ``scope`` says which. ``write`` saves it as a report directory, and
    ``inspect-safety-stats schema report`` prints the JSON schema of its ``report.json``.
    """

    schema_version: Literal["3.0"] = "3.0"
    package_version: str
    analysis: dict[str, Any]
    # Every default the analysis applied: the plan fields its author left out, which
    # analysis.plan_sha256 does not cover (#56). Empty in a report written before the field.
    plan_defaults: list[PlanDefault] = Field(default_factory=list)
    # The registry's assumptions this analysis rests on, in registry order. Empty in a report
    # written before the field.
    assumptions_used: list[AssumptionUse] = Field(default_factory=list)
    provenance: dict[str, Any]
    scope: dict[str, Any]
    presentation: ReportPresentation = Field(default_factory=ReportPresentation)
    integrity: IntegritySummary
    # The manifest and its hash, the hierarchy, the independence profiles, the groups and, in
    # design.prompts, one row per prompt of each group with the keys of design_columns. The
    # hierarchy's levels are the plan's own column names, so the rows are not a fixed model.
    design: dict[str, Any]
    cohorts: list[dict[str, Any]]
    prompt_estimates: list[PromptEstimate]
    group_risks: list[GroupRiskResult]
    hierarchy_risks: list[HierarchyRiskResult] = Field(default_factory=list)
    comparisons: list[ComparisonResult]
    comparison_sets: list[ComparisonSetResult] = Field(default_factory=list)
    multiplicity: list[dict[str, Any]]
    claims: list[ClaimResult]
    # Experimental (#125); left out of report.json when the plan declares none.
    adaptive_claims: list[AdaptiveClaimResult] = Field(default_factory=list)
    policy_events: list[dict[str, Any]]
    decision: Decision
    # Present only when the analysis was given a decision policy.
    decision_policy: DecisionPolicyRecord | None = None
    # Experimental; left out of report.json when the plan declares no alternative mapping.
    outcome_mapping_sensitivities: list[OutcomeMappingSensitivity] = Field(default_factory=list)
    # Copies of the analysis input, written beside the report as observations.csv and
    # plan.resolved.yaml. No rendering reads them: everything report.md, report.html and the
    # other CSVs show is in the fields above.
    _observations: list[Any] = PrivateAttr(default_factory=list)
    _resolved_plan: dict[str, Any] = PrivateAttr(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> ReportSummary:
        """What to read first, with pointers to the rest (``inspect_safety_stats.schema.summary``).

        Written to ``report.json``; derived from the other fields, so it is dropped when a
        saved report is read back.
        """

        from inspect_safety_stats.schema.summary import report_summary

        return report_summary(self)

    @model_validator(mode="before")
    @classmethod
    def drop_the_derived_summary(cls, data: Any) -> Any:
        if isinstance(data, dict) and "summary" in data:
            data = {key: value for key, value in data.items() if key != "summary"}
        return data

    @model_serializer(mode="wrap")
    def leave_out_absent_optional_sections(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # A report of an analysis without a decision policy or outcome mapping sensitivities
        # keeps the bytes it had before either existed.
        data: dict[str, Any] = handler(self)
        if self.decision_policy is None:
            data.pop("decision_policy", None)
        if not self.outcome_mapping_sensitivities:
            data.pop("outcome_mapping_sensitivities", None)
        if not self.adaptive_claims:
            data.pop("adaptive_claims", None)
        return data

    @model_validator(mode="after")
    def point_each_diagnostic_at_its_item(self) -> SafetyReport:
        # Derived from the scope and the results, so a stored value is recomputed, not trusted.
        pointers = ItemPointers(self)
        for diagnostic in self.integrity.diagnostics:
            diagnostic.item = pointers.pointer(diagnostic.scope)
        return self

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        from inspect_safety_stats.report.writer import write_report

        return write_report(self, output, overwrite=overwrite)

def mapping_results(report: SafetyReport) -> list[MappingSensitivityResult]:
    """The displayed results that outcome mapping sensitivities put side by side."""

    return result_readings(report.group_risks, report.comparisons, report.claims)


def result_readings(
    group_risks: list[GroupRiskResult],
    comparisons: list[ComparisonResult],
    claims: list[ClaimResult],
) -> list[MappingSensitivityResult]:
    """The rows ``mapping_results`` gives, from an analysis's results before a report holds them.

    Each kind is sorted as ``analyze`` sorts it in the report, so an alternative mapping's rows
    line up with the primary's.
    """

    rows = [
        MappingSensitivityResult(
            kind="group_risk",
            result_id=f"{risk.cohort_id}:{risk.group_id}:{risk.measure_id}",
            cohort_id=risk.cohort_id,
            group_id=risk.group_id,
            measure_id=risk.measure_id,
            estimate=risk.estimate,
            lower=risk.lower,
            upper=risk.upper,
        )
        for risk in sorted(
            group_risks, key=lambda item: (item.cohort_id, item.group_id, item.measure_id)
        )
    ]
    rows.extend(
        MappingSensitivityResult(
            kind="comparison",
            result_id=comparison.result_id or comparison.comparison_id,
            group_id=comparison.group_id,
            measure_id=comparison.measure_id,
            estimate=comparison.estimate,
            lower=comparison.lower,
            upper=comparison.upper,
        )
        for comparison in sorted(comparisons, key=lambda item: item.comparison_id)
    )
    rows.extend(
        MappingSensitivityResult(
            kind="claim",
            result_id=claim.claim_id,
            estimate=claim.estimate,
            lower=claim.lower,
            upper=claim.upper,
            status=claim.status,
        )
        for claim in sorted(claims, key=lambda item: item.claim_id)
    )
    return rows


# ORDER MATTERS: schema/summary.py imports this module's models, so it can only be imported once
# they exist; SafetyReport.summary's type is resolved by the rebuild that follows.
from inspect_safety_stats.schema.summary import ReportSummary  # noqa: E402

SafetyReport.model_rebuild()
