from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


def _reject_boolean(value: object) -> object:
    # Lax validation would read `true` as 1. A plan is a pre-registered document, so a typo in
    # an integer field must fail rather than run with a guessed value.
    if isinstance(value, bool):
        raise ValueError("must be an integer, not a boolean")
    return value


# Every integer field in a plan, analysis or generative, uses this type.
PlanInt = Annotated[int, BeforeValidator(_reject_boolean)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class RegistrationSpec(StrictModel):
    uri: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class AnalysisMetadata(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None
    intent: Literal["confirmatory", "exploratory"] = "confirmatory"
    plan_frozen_at: datetime | None = None
    registration: RegistrationSpec | None = None

    @model_validator(mode="after")
    def confirmatory_has_freeze_time(self) -> AnalysisMetadata:
        if self.intent == "confirmatory" and self.plan_frozen_at is None:
            raise ValueError("confirmatory analyses require analysis.plan_frozen_at")
        return self


# Schema 3.0 names run and rollout fields neutrally: a few fields every source has, and the
# source's own fields under ``attributes.``. Schema 2.0 names the fields of the source verbatim.
RUN_FIELDS = ("run_key", "model", "status", "run_id")
ROLLOUT_FIELDS = ("run_key", "item_id", "repeat", "rollout_id")
ATTRIBUTE_PREFIX = "attributes."
LEGACY_ROLLOUT_KEY = ["eval_id", "id", "epoch"]
NEUTRAL_ROLLOUT_KEY = ["run_key", "item_id", "repeat"]


def neutral_field_error(name: str, reserved: tuple[str, ...]) -> str | None:
    """Why ``name`` is not a schema 3.0 field name, or None when it is one."""

    if name in reserved:
        return None
    if name.startswith(ATTRIBUTE_PREFIX) and len(name) > len(ATTRIBUTE_PREFIX):
        return None
    return (
        f"{name!r} is neither one of {list(reserved)} nor a source field written "
        f"{ATTRIBUTE_PREFIX}<name>"
    )


class SampleJoinSpec(StrictModel):
    inspect_field: Literal["id"] = "id"
    manifest_field: Literal["prompt_id"] = "prompt_id"


class FailureIfSpec(StrictModel):
    operator: Literal[
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
    ]
    value: float


class OutcomeMappingSpec(StrictModel):
    failure_values: list[Any] | None = None
    safe_values: list[Any] | None = None
    failure_if: FailureIfSpec | None = None

    @model_validator(mode="after")
    def exactly_one_mapping_style(self) -> OutcomeMappingSpec:
        categorical = self.failure_values is not None or self.safe_values is not None
        numeric = self.failure_if is not None
        if categorical == numeric:
            raise ValueError("provide either failure_values/safe_values or failure_if")
        if categorical and (self.failure_values is None or self.safe_values is None):
            raise ValueError("categorical mapping requires both failure_values and safe_values")
        return self


class ScoreSpec(StrictModel):
    name: str = Field(min_length=1)
    mapping: OutcomeMappingSpec
    missing: Literal["conservative_bounds", "block", "count_as_failure", "exclude"] = (
        "conservative_bounds"
    )
    errors: Literal["end_to_end", "conservative_bounds", "block", "count_as_failure", "exclude"] = (
        "end_to_end"
    )


class JudgeRateRangeSpec(StrictModel):
    """A declared closed range for one conditional judge accuracy."""

    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> JudgeRateRangeSpec:
        if self.lower > self.upper:
            raise ValueError("judge rate range lower endpoint cannot exceed its upper endpoint")
        return self


class JudgeCalibrationScenarioSpec(StrictModel):
    """One named assumption about the judge's conditional accuracy.

    ``rho_1`` is P(truly unsafe | judge said unsafe) and ``rho_0`` is
    P(truly unsafe | judge said safe).
    """

    id: str = Field(min_length=1)
    rho_1: JudgeRateRangeSpec
    rho_0: JudgeRateRangeSpec

    @model_validator(mode="after")
    def strictly_informative(self) -> JudgeCalibrationScenarioSpec:
        if self.rho_1.lower <= self.rho_0.upper:
            raise ValueError(
                f"judge calibration scenario {self.id!r} is not strictly informative: "
                "the rho_1 lower endpoint must exceed the rho_0 upper endpoint"
            )
        return self


class JudgeCalibrationSpec(StrictModel):
    """Judge-misclassification calibration, either declared or measured.

    ``mode: assumed`` carries no evidence: every corrected interval it produces is a
    diagnostic sensitivity result. ``mode: validated`` estimates the same two
    conditional accuracies from human labels drawn within each judge stratum, and its
    corrected intervals are certifying at the combined error budget.
    """

    mode: Literal["assumed", "validated"] = "assumed"
    scenarios: list[JudgeCalibrationScenarioSpec] = Field(default_factory=list)
    primary: str | None = None
    validation_manifest: Path | None = None
    family: str | None = None
    sampling: Literal["simple_random_within_judge_stratum", "simple_random"] = (
        "simple_random_within_judge_stratum"
    )

    @model_validator(mode="after")
    def validate_mode(self) -> JudgeCalibrationSpec:
        if self.mode == "assumed":
            if not self.scenarios:
                raise ValueError("assumed judge calibration requires at least one scenario")
            if self.validation_manifest is not None or self.family is not None:
                raise ValueError(
                    "validation_manifest and family require mode: validated"
                )
            ids = [item.id for item in self.scenarios]
            if len(set(ids)) != len(ids):
                raise ValueError("judge calibration scenario identifiers must be unique")
            if self.primary is None:
                raise ValueError("assumed judge calibration requires a primary scenario")
            if self.primary not in ids:
                raise ValueError(
                    f"judge calibration primary {self.primary!r} is not a declared scenario"
                )
            return self
        if self.validation_manifest is None:
            raise ValueError("validated judge calibration requires a validation_manifest")
        if not self.family:
            raise ValueError("validated judge calibration requires a multiplicity family")
        if self.scenarios or self.primary is not None:
            raise ValueError(
                "validated judge calibration estimates its accuracies from labels; "
                "remove the assumed scenarios and primary"
            )
        return self


class JudgeAccuracySpec(StrictModel):
    """Declared prevalence-free accuracy ranges for one judge."""

    sensitivity: JudgeRateRangeSpec
    specificity: JudgeRateRangeSpec


class JudgeSpec(StrictModel):
    """One declared binary judge over every rollout.

    ``block`` names the judge's dependence block: judges sharing a block are one
    opinion, not independent votes. It defaults to the judge's own identifier, so a
    panel is independent only where the plan says nothing else. ``kind: monitor``
    marks a chain-of-thought or trajectory monitor, whose relevant error is its miss
    rate rather than its false-alarm rate.
    """

    id: str = Field(min_length=1)
    score: ScoreSpec
    block: str | None = None
    kind: Literal["judge", "monitor"] = "judge"
    accuracy: JudgeAccuracySpec | None = None

    @model_validator(mode="after")
    def resolve_block(self) -> JudgeSpec:
        if self.block is None:
            return self.model_copy(update={"block": self.id})
        if not self.block.strip():
            raise ValueError("judge block must be a non-empty identifier")
        return self


class CompositeJudgeSpec(StrictModel):
    """A derived verdict combining declared judges; available as a score column."""

    id: str = Field(min_length=1)
    rule: Literal["any", "all", "majority"]
    judges: list[str] = Field(min_length=2)


class InputsSpec(StrictModel):
    design_manifest: Path
    estimand: Literal["end_to_end_system", "underlying_model"] = "end_to_end_system"
    # Schema 2.0 only; None, and left out of the plan, under schema 3.0.
    sample_join: SampleJoinSpec | None = Field(default_factory=SampleJoinSpec)
    score: ScoreSpec
    judge_calibration: JudgeCalibrationSpec | None = None
    judges: list[JudgeSpec] = Field(default_factory=list)
    composites: list[CompositeJudgeSpec] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def leave_out_absent_sample_join(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if self.sample_join is None:
            data.pop("sample_join", None)
        return data

    @model_validator(mode="after")
    def validate_judge_panel(self) -> InputsSpec:
        identifiers = [judge.id for judge in self.judges] + [item.id for item in self.composites]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("judge and composite identifiers must be unique")
        declared = {judge.id for judge in self.judges}
        for composite in self.composites:
            unknown = sorted(set(composite.judges) - declared)
            if unknown:
                raise ValueError(
                    f"composite {composite.id!r} references unknown judge(s): "
                    + ", ".join(unknown)
                )
            if len(set(composite.judges)) != len(composite.judges):
                raise ValueError(f"composite {composite.id!r} lists a judge more than once")
        return self


class WeightSpec(StrictModel):
    strategy: Literal["equal_prompt", "nested_equal", "explicit"] = "nested_equal"
    hierarchy: list[str] = Field(default_factory=lambda: ["task_id", "question_id", "prompt_id"])
    column: str | None = None

    @model_validator(mode="after")
    def explicit_has_column(self) -> WeightSpec:
        if self.strategy == "explicit" and not self.column:
            raise ValueError("explicit weights require a manifest column")
        return self


class IndependenceProfileSpec(StrictModel):
    """A predeclared partition of observations into independent blocks."""

    block_by: list[str] = Field(min_length=1)
    comparison_coupling: Literal["independent", "joint"] = "independent"

    @model_validator(mode="after")
    def validate_block_keys(self) -> IndependenceProfileSpec:
        if len(set(self.block_by)) != len(self.block_by):
            raise ValueError("independence profile block_by fields must be unique")
        if "rollout" in self.block_by and self.block_by != ["rollout"]:
            raise ValueError("the reserved rollout block cannot be combined with other fields")
        if self.block_by == ["rollout"] and self.comparison_coupling != "independent":
            raise ValueError("rollout blocks require comparison_coupling=independent")
        return self


class IndependenceSelectionSpec(StrictModel):
    primary: str = Field(min_length=1)
    sensitivities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profiles(self) -> IndependenceSelectionSpec:
        if self.primary in self.sensitivities:
            raise ValueError("the primary independence profile cannot be a sensitivity")
        if len(set(self.sensitivities)) != len(self.sensitivities):
            raise ValueError("independence sensitivity profiles must be unique")
        return self


class DesignSpec(StrictModel):
    type: Literal["fixed", "sampled"] = "fixed"
    sampling: Literal["poisson", "fixed_size"] = "poisson"
    inclusion_probability_column: str | None = None
    stratum_column: str | None = None
    hierarchy: list[str] = Field(default_factory=lambda: ["task_id", "question_id", "prompt_id"])
    # The schema 2.0 default; a schema 3.0 plan defaults to NEUTRAL_ROLLOUT_KEY.
    rollout_key: list[str] = Field(default_factory=lambda: list(LEGACY_ROLLOUT_KEY))
    require_complete_manifest: bool = True
    require_expected_rollouts: bool = True
    weights: WeightSpec = Field(default_factory=WeightSpec)
    independence_profiles: dict[str, IndependenceProfileSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sampling(self) -> DesignSpec:
        if self.type == "sampled":
            if not self.inclusion_probability_column:
                raise ValueError(
                    "a sampled design requires design.inclusion_probability_column"
                )
            if self.stratum_column and self.sampling != "fixed_size":
                raise ValueError("design.stratum_column requires sampling: fixed_size")
            return self
        if self.inclusion_probability_column is not None:
            raise ValueError(
                "inclusion_probability_column requires design.type: sampled"
            )
        if self.stratum_column is not None:
            raise ValueError("stratum_column requires design.type: sampled")
        if self.sampling != "poisson":
            raise ValueError("design.sampling requires design.type: sampled")
        return self

    @model_validator(mode="after")
    def validate_independence_profile_ids(self) -> DesignSpec:
        invalid = [
            profile_id
            for profile_id in self.independence_profiles
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_id) is None
        ]
        if invalid:
            raise ValueError(
                "independence profile identifiers must contain only letters, numbers, "
                "periods, underscores, and hyphens"
            )
        return self


class CohortSpec(StrictModel):
    match: dict[str, Any]
    require_uniform: list[str] = Field(default_factory=list)


class GroupSpec(StrictModel):
    id: str = Field(min_length=1)
    title: str | None = None
    filter: dict[str, Any] = Field(default_factory=dict)
    weights: WeightSpec | None = None


class RiskMeasureSpec(StrictModel):
    id: str = Field(min_length=1)
    type: Literal[
        "weighted_mean",
        "maximum",
        "cvar",
        "top_k_mean",
        "worst_of_k",
        "entropic",
        "threshold_exceedance",
        "threshold_excess",
    ]
    beta: float | None = None
    lambda_: float | None = Field(default=None, alias="lambda")
    k: PlanInt | None = None
    threshold: float | None = None
    threshold_column: str | None = None
    diagnostic_only: bool = False

    @model_validator(mode="after")
    def validate_parameters(self) -> RiskMeasureSpec:
        if self.type == "cvar" and (self.beta is None or not 0 <= self.beta < 1):
            raise ValueError("cvar requires beta in [0, 1)")
        if self.type == "entropic" and (self.lambda_ is None or self.lambda_ <= 0):
            raise ValueError("entropic risk requires lambda > 0")
        if self.type == "top_k_mean" and (self.k is None or self.k <= 0):
            raise ValueError("top_k_mean requires k > 0")
        if self.type == "worst_of_k" and (self.k is None or self.k <= 0):
            raise ValueError("worst_of_k requires k > 0")
        if self.type in {"threshold_exceedance", "threshold_excess"}:
            if (self.threshold is None) == (self.threshold_column is None):
                raise ValueError(f"{self.type} requires exactly one threshold source")
        return self


class PromptBoundsSpec(StrictModel):
    enabled: bool = True
    simultaneous: bool = True
    family: str | None = None
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    alpha_allocation: Literal["equal_bonferroni", "weighted_bonferroni", "explicit"] = (
        "equal_bonferroni"
    )
    explicit_alpha: dict[str, float] | None = None

    @model_validator(mode="after")
    def explicit_has_alpha_map(self) -> PromptBoundsSpec:
        if self.alpha_allocation == "explicit" and not self.explicit_alpha:
            raise ValueError("explicit alpha allocation requires explicit_alpha")
        if self.alpha_allocation != "explicit" and self.explicit_alpha is not None:
            raise ValueError("explicit_alpha is only valid with explicit allocation")
        if self.explicit_alpha and any(not 0 < value < 1 for value in self.explicit_alpha.values()):
            raise ValueError("every explicit alpha must be in (0, 1)")
        return self


class DirectMeanIntervalsSpec(StrictModel):
    enabled: bool = False
    method: Literal[
        "weighted_hoeffding",
        "weighted_empirical_bernstein",
        "weighted_kl_chernoff",
    ] = "weighted_empirical_bernstein"
    reference_methods: list[
        Literal[
            "weighted_hoeffding",
            "weighted_empirical_bernstein",
            "weighted_kl_chernoff",
        ]
    ] = Field(default_factory=list)
    family: str | None = None
    groups: list[str] | None = None
    simultaneous: bool = True
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec | None = None
    hierarchy_nodes: bool = False
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "flat_bonferroni"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"

    @model_validator(mode="after")
    def enabled_has_family(self) -> DirectMeanIntervalsSpec:
        if self.enabled and not self.family:
            raise ValueError("enabled direct mean intervals require a multiplicity family")
        if self.enabled and self.independence is None:
            raise ValueError("enabled direct mean intervals require an independence selection")
        if self.groups is not None:
            if not self.groups:
                raise ValueError("direct mean interval groups cannot be empty")
            if len(set(self.groups)) != len(self.groups):
                raise ValueError("direct mean interval groups must be unique")
        if self.method in self.reference_methods:
            raise ValueError("the primary direct mean method cannot also be a reference method")
        if len(set(self.reference_methods)) != len(self.reference_methods):
            raise ValueError("direct mean reference methods must be unique")
        if self.hierarchy_calibration == "frontier_sidak" and not self.hierarchy_nodes:
            raise ValueError("frontier_sidak hierarchy calibration requires hierarchy_nodes=true")
        return self


class OutputsSpec(StrictModel):
    prompt_bounds: PromptBoundsSpec = Field(default_factory=PromptBoundsSpec)
    direct_mean_intervals: DirectMeanIntervalsSpec = Field(default_factory=DirectMeanIntervalsSpec)
    risk_measures: list[RiskMeasureSpec] = Field(default_factory=list)


class ComparisonSpec(StrictModel):
    id: str = Field(min_length=1)
    candidate: str
    baseline: str
    group: str
    measure: str
    estimand: Literal[
        "candidate_minus_baseline",
        "risk_difference",
        "risk_ratio",
        "relative_risk_reduction",
    ] = "risk_difference"
    method: Literal[
        "weighted_hoeffding",
        "weighted_empirical_bernstein",
        "weighted_kl_chernoff",
    ] = "weighted_kl_chernoff"
    reference_methods: list[
        Literal[
            "weighted_hoeffding",
            "weighted_empirical_bernstein",
            "weighted_kl_chernoff",
        ]
    ] = Field(default_factory=list)
    family: str | None = None
    simultaneous: bool = True
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec | None = None
    hierarchy_nodes: bool = False
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "flat_bonferroni"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"
    allocation_weight: float = Field(default=1.0, gt=0)
    pair_key: list[str] | None = None
    diagnostic_methods: list[
        Literal["exact_mcnemar", "newcombe_paired", "fixed_cluster_bootstrap"]
    ] = Field(default_factory=list)
    cluster_key: str | None = None
    bootstrap_replicates: PlanInt = Field(default=2000, ge=100)
    bootstrap_seed: PlanInt = 0

    _comparison_set_id: str | None = PrivateAttr(default=None)
    _joint_arm_region: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def validate_comparison_method(self) -> ComparisonSpec:
        if not self.family:
            raise ValueError("comparisons require a multiplicity family")
        if self.independence is None:
            raise ValueError("comparisons require an independence selection")
        if self.method in self.reference_methods:
            raise ValueError("the primary comparison method cannot be a reference method")
        if len(set(self.reference_methods)) != len(self.reference_methods):
            raise ValueError("comparison reference methods must be unique")
        if self.hierarchy_calibration == "frontier_sidak" and not self.hierarchy_nodes:
            raise ValueError("frontier_sidak comparison calibration requires hierarchy_nodes=true")
        if self.estimand == "candidate_minus_baseline":
            object.__setattr__(self, "estimand", "risk_difference")
        if self.pair_key is not None and not self.pair_key:
            raise ValueError("pair_key cannot be empty")
        return self


class ComparisonContrastsSpec(StrictModel):
    type: Literal["versus_reference", "all_pairs"] = "versus_reference"
    reference: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> ComparisonContrastsSpec:
        if self.type == "versus_reference" and not self.reference:
            raise ValueError("versus_reference contrasts require reference")
        if self.type == "all_pairs" and self.reference is not None:
            raise ValueError("all_pairs contrasts do not accept reference")
        return self


class ComparisonSetSpec(StrictModel):
    """A jointly calibrated comparison family over two or more cohorts."""

    id: str = Field(min_length=1)
    cohorts: list[str] = Field(min_length=2)
    contrasts: ComparisonContrastsSpec
    estimands: list[Literal["risk_difference", "risk_ratio", "relative_risk_reduction"]] = [
        "risk_difference"
    ]
    group: str
    measure: str
    method: Literal[
        "weighted_hoeffding",
        "weighted_empirical_bernstein",
        "weighted_kl_chernoff",
    ] = "weighted_kl_chernoff"
    reference_methods: list[
        Literal[
            "weighted_hoeffding",
            "weighted_empirical_bernstein",
            "weighted_kl_chernoff",
        ]
    ] = Field(default_factory=list)
    family: str
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec
    hierarchy_nodes: bool = True
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "frontier_sidak"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"
    allocation_weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_set(self) -> ComparisonSetSpec:
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError("comparison-set cohorts must be unique")
        if not self.estimands:
            raise ValueError("comparison-set estimands cannot be empty")
        if len(set(self.estimands)) != len(self.estimands):
            raise ValueError("comparison-set estimands must be unique")
        if self.contrasts.reference is not None and self.contrasts.reference not in self.cohorts:
            raise ValueError("comparison-set reference must be one of its cohorts")
        if self.method in self.reference_methods:
            raise ValueError("the primary comparison-set method cannot also be a reference")
        if len(set(self.reference_methods)) != len(self.reference_methods):
            raise ValueError("comparison-set reference methods must be unique")
        if self.hierarchy_calibration == "frontier_sidak" and not self.hierarchy_nodes:
            raise ValueError(
                "frontier_sidak comparison-set calibration requires hierarchy_nodes=true"
            )
        return self

    def pairs(self) -> list[tuple[str, str]]:
        if self.contrasts.type == "versus_reference":
            reference = self.contrasts.reference
            if reference is None:
                raise ValueError("versus_reference contrasts require reference")
            return [(cohort, reference) for cohort in self.cohorts if cohort != reference]
        return [
            (candidate, baseline)
            for candidate_index, candidate in enumerate(self.cohorts)
            for baseline in self.cohorts[:candidate_index]
        ]

    def expand(self) -> list[ComparisonSpec]:
        result: list[ComparisonSpec] = []
        for candidate, baseline in self.pairs():
            for estimand in self.estimands:
                spec = ComparisonSpec(
                    id=f"{self.id}::{candidate}_vs_{baseline}::{estimand}",
                    candidate=candidate,
                    baseline=baseline,
                    group=self.group,
                    measure=self.measure,
                    estimand=estimand,
                    method=self.method,
                    reference_methods=self.reference_methods,
                    family=self.family,
                    simultaneous=True,
                    coverage_mode=self.coverage_mode,
                    independence=self.independence,
                    hierarchy_nodes=self.hierarchy_nodes,
                    hierarchy_calibration=self.hierarchy_calibration,
                    hierarchy_region=self.hierarchy_region,
                    allocation_weight=self.allocation_weight,
                )
                spec._comparison_set_id = self.id
                spec._joint_arm_region = True
                result.append(spec)
        return result


class MultiplicityFamilySpec(StrictModel):
    id: str = Field(min_length=1)
    purpose: Literal["all_mandatory", "simultaneous_intervals", "separate_claims", "exploratory"]
    method: Literal[
        "intersection_union", "bonferroni", "weighted_bonferroni"
    ]
    alpha: float = Field(gt=0, lt=1)


class MultiplicitySpec(StrictModel):
    families: list[MultiplicityFamilySpec] = Field(default_factory=list)


class ClaimSpec(StrictModel):
    id: str = Field(min_length=1)
    title: str
    cohort: str | None = None
    group: str | None = None
    measure: str | None = None
    comparison: str | None = None
    family: str
    acceptable_upper: float = Field(ge=-1, le=1)
    unacceptable_lower: float = Field(ge=-1, le=1)
    mandatory: bool = True
    method: Literal["corrected_interval", "noisy_valid_test"] = "corrected_interval"

    @model_validator(mode="after")
    def validate_source_and_thresholds(self) -> ClaimSpec:
        if self.method == "noisy_valid_test" and self.comparison is not None:
            raise ValueError("noisy_valid_test applies to aggregate claims, not comparisons")
        aggregate = self.cohort is not None or self.group is not None or self.measure is not None
        if self.comparison is not None:
            if aggregate:
                raise ValueError("comparison claim cannot also declare cohort/group/measure")
        elif not (self.cohort and self.group and self.measure):
            raise ValueError("aggregate claim requires cohort, group, and measure")
        if self.acceptable_upper >= self.unacceptable_lower:
            raise ValueError("acceptable_upper must be less than unacceptable_lower")
        return self


class ZeroToleranceSpec(StrictModel):
    enabled: bool = True
    manifest_field: str = "zero_tolerance"


class PolicySpec(StrictModel):
    zero_tolerance: ZeroToleranceSpec = Field(default_factory=ZeroToleranceSpec)
    data_quality_failure: Literal["inconclusive"] = "inconclusive"


class ReportSpec(StrictModel):
    include_prompt_text: bool = False
    include_observations: bool = True
    include_decisions: bool = False
    decimal_places: PlanInt = Field(default=6, ge=0, le=15)


class AnalysisPlan(StrictModel):
    schema_version: Literal["2.0", "3.0"] = "2.0"
    analysis: AnalysisMetadata
    inputs: InputsSpec
    design: DesignSpec = Field(default_factory=DesignSpec)
    cohorts: dict[str, CohortSpec]
    groups: list[GroupSpec]
    outputs: OutputsSpec
    comparisons: list[ComparisonSpec] = Field(default_factory=list)
    comparison_sets: list[ComparisonSetSpec] = Field(default_factory=list)
    multiplicity: MultiplicitySpec
    claims: list[ClaimSpec]
    policy: PolicySpec = Field(default_factory=PolicySpec)
    report: ReportSpec = Field(default_factory=ReportSpec)

    _source_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def apply_schema_version_defaults(cls, data: Any) -> Any:
        """Fill the defaults that differ between schema versions, before validation."""

        if not isinstance(data, dict):
            return data
        inputs = data.get("inputs")
        if data.get("schema_version") != "3.0":
            if isinstance(inputs, dict) and "sample_join" in inputs:
                if inputs["sample_join"] is None:
                    raise ValueError("inputs.sample_join cannot be null")
            return data
        data = dict(data)
        if isinstance(inputs, dict):
            if "sample_join" in inputs:
                raise ValueError(
                    "inputs.sample_join was removed in schema 3.0: rollouts join the design "
                    "manifest's prompt_id by their item_id"
                )
            data["inputs"] = {**inputs, "sample_join": None}
        design = data.get("design", {})
        if isinstance(design, dict) and "rollout_key" not in design:
            data["design"] = {**design, "rollout_key": list(NEUTRAL_ROLLOUT_KEY)}
        return data

    @model_validator(mode="after")
    def neutral_field_names(self) -> AnalysisPlan:
        if self.schema_version != "3.0":
            return self
        problems = [
            f"design.rollout_key: {error}"
            for name in self.design.rollout_key
            if (error := neutral_field_error(name, ROLLOUT_FIELDS))
        ]
        for cohort_id, cohort in self.cohorts.items():
            problems += [
                f"cohorts.{cohort_id}.{where}: {error}"
                for where, names in (
                    ("match", list(cohort.match)),
                    ("require_uniform", cohort.require_uniform),
                )
                for name in names
                if (error := neutral_field_error(name, RUN_FIELDS))
            ]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @model_validator(mode="after")
    def independence_references_exist(self) -> AnalysisPlan:
        profiles = set(self.design.independence_profiles)
        selections: list[tuple[str, IndependenceSelectionSpec | None]] = [
            ("outputs.direct_mean_intervals", self.outputs.direct_mean_intervals.independence),
            *[(f"comparisons.{item.id}", item.independence) for item in self.comparisons],
            *[(f"comparison_sets.{item.id}", item.independence) for item in self.comparison_sets],
        ]
        for location, selection in selections:
            if selection is None:
                continue
            referenced = {selection.primary, *selection.sensitivities}
            unknown = sorted(referenced - profiles)
            if unknown:
                raise ValueError(
                    f"{location} references unknown independence profiles: " + ", ".join(unknown)
                )
        return self

    @model_validator(mode="after")
    def comparison_identifiers_are_unique(self) -> AnalysisPlan:
        set_ids = [item.id for item in self.comparison_sets]
        if len(set(set_ids)) != len(set_ids):
            raise ValueError("comparison-set identifiers must be unique")
        expanded_ids = [item.id for item in self.expanded_comparisons()]
        if len(set(expanded_ids)) != len(expanded_ids):
            raise ValueError("expanded comparison identifiers must be unique")
        return self

    def source_path(self) -> Path | None:
        return self._source_path

    def expanded_comparisons(self) -> list[ComparisonSpec]:
        return [
            *self.comparisons,
            *(comparison for item in self.comparison_sets for comparison in item.expand()),
        ]
