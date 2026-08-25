"""Strict plan schema for generative hierarchy evaluations.

A generative plan is deliberately a separate top-level document from the fixed
analysis plan. The fixed plan conditions on a materialized design and treats prompt
weights as given; a generative plan declares the process that produces prompts. A
single file carrying both would let an author combine them into an estimand neither
engine can state.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inspect_safety_stats.config.models import JudgeCalibrationScenarioSpec

RESERVED_CHARACTERS = ("|", "=")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


def reject_reserved(label: str, value: str) -> str:
    """Reject identifiers containing the canonical node-id separators."""

    if not value:
        raise ValueError(f"{label} must not be empty")
    for character in RESERVED_CHARACTERS:
        if character in value:
            raise ValueError(
                f"{label} {value!r} contains reserved character {character!r}; "
                "canonical node identifiers use it as a separator"
            )
    return value


class GenerativeMetadata(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None
    intent: Literal["confirmatory", "exploratory"] = "confirmatory"
    plan_frozen_at: datetime | None = None
    selection_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def confirmatory_has_freeze_time(self) -> GenerativeMetadata:
        if self.intent == "confirmatory" and self.plan_frozen_at is None:
            raise ValueError("confirmatory evaluations require evaluation.plan_frozen_at")
        return self


class ArmValueSpec(StrictModel):
    id: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def identifier_is_canonical(self) -> ArmValueSpec:
        reject_reserved("arm value id", self.id)
        return self


class ContextWeightSpec(StrictModel):
    """Weights that replace an arm's global weights in a matching context."""

    when: dict[str, str] = Field(min_length=1)
    values: list[ArmValueSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def values_are_unique(self) -> ContextWeightSpec:
        ids = [value.id for value in self.values]
        if len(set(ids)) != len(ids):
            raise ValueError("context_weights entry repeats an arm value id")
        return self


class ArmSpec(StrictModel):
    values: list[ArmValueSpec] | None = None
    values_from: str | None = None
    weight_column: str | None = None
    context_weights: list[ContextWeightSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_value_source(self) -> ArmSpec:
        if (self.values is None) == (self.values_from is None):
            raise ValueError("an arm declares exactly one of 'values' or 'values_from'")
        if self.values_from is None and self.weight_column is not None:
            raise ValueError("'weight_column' applies only to 'values_from'")
        if self.values_from is not None and self.weight_column is None:
            raise ValueError("'values_from' requires 'weight_column'")
        if self.values is not None:
            ids = [value.id for value in self.values]
            if len(set(ids)) != len(ids):
                raise ValueError("an arm repeats a value id")
        return self


class FixedLevelSpec(StrictModel):
    source: Literal["seed_manifest"] = "seed_manifest"
    weights: Literal["explicit", "equal_within_parent"]
    weight_column: str | None = None

    @model_validator(mode="after")
    def explicit_needs_column(self) -> FixedLevelSpec:
        if self.weights == "explicit" and self.weight_column is None:
            raise ValueError("fixed level with weights: explicit requires a weight_column")
        if self.weights == "equal_within_parent" and self.weight_column is not None:
            raise ValueError("weight_column applies only to weights: explicit")
        return self


class ConstraintSpec(StrictModel):
    """Prune structurally invalid arm combinations below a matched context."""

    id: str = Field(min_length=1)
    when: dict[str, str] = Field(min_length=1)
    allow: dict[str, list[str]] = Field(min_length=1)

    @model_validator(mode="after")
    def allow_lists_are_populated(self) -> ConstraintSpec:
        for level, values in self.allow.items():
            if not values:
                raise ValueError(
                    f"constraint {self.id!r} allows nothing at level {level!r}; "
                    "a positive-weight context must retain at least one valid child"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"constraint {self.id!r} repeats a value at level {level!r}")
        return self


class TaxonomySpec(StrictModel):
    seed_manifest: str = Field(min_length=1)
    seed_text_column: str = "seed_text"
    root_id: str = "all"
    hierarchy: list[str] = Field(min_length=1)
    semantic_seed_level: str = Field(min_length=1)
    fixed_levels: dict[str, FixedLevelSpec] = Field(default_factory=dict)
    arms: dict[str, ArmSpec] = Field(default_factory=dict)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    renormalization: Literal["renormalize_valid_children", "explicit_context_weights"] | None = None
    maximum_paths: int = Field(default=500_000, gt=0)

    @model_validator(mode="after")
    def hierarchy_is_well_formed(self) -> TaxonomySpec:
        reject_reserved("root_id", self.root_id)
        for level in self.hierarchy:
            reject_reserved("hierarchy level", level)
        if len(set(self.hierarchy)) != len(self.hierarchy):
            raise ValueError("taxonomy.hierarchy repeats a level")
        if self.semantic_seed_level not in self.hierarchy:
            raise ValueError("taxonomy.semantic_seed_level must appear in taxonomy.hierarchy")

        declared = set(self.fixed_levels) | set(self.arms)
        overlap = set(self.fixed_levels) & set(self.arms)
        if overlap:
            raise ValueError(
                f"levels declared as both fixed and arm: {', '.join(sorted(overlap))}"
            )
        missing = [level for level in self.hierarchy if level not in declared]
        if missing:
            raise ValueError(f"levels with no declaration: {', '.join(missing)}")
        unknown = sorted(declared - set(self.hierarchy))
        if unknown:
            raise ValueError(f"declared levels absent from the hierarchy: {', '.join(unknown)}")

        # Semantic seeds are a mandatory stratification frontier: everything above them
        # is fixed and read from the manifest, everything below is sampled.
        seed_index = self.hierarchy.index(self.semantic_seed_level)
        expected_fixed = set(self.hierarchy[: seed_index + 1])
        if set(self.fixed_levels) != expected_fixed:
            raise ValueError(
                "fixed levels must be exactly the hierarchy prefix ending at "
                f"{self.semantic_seed_level!r}; got {sorted(self.fixed_levels)}"
            )
        return self

    @model_validator(mode="after")
    def constraints_are_resolvable(self) -> TaxonomySpec:
        if self.constraints and self.renormalization is None:
            raise ValueError(
                "taxonomy.constraints prune valid children, so taxonomy.renormalization "
                "must select 'renormalize_valid_children' or 'explicit_context_weights'"
            )
        order = {level: index for index, level in enumerate(self.hierarchy)}
        for constraint in self.constraints:
            for level in constraint.when:
                if level not in order:
                    raise ValueError(
                        f"constraint {constraint.id!r} conditions on unknown level {level!r}"
                    )
            latest_condition = max(order[level] for level in constraint.when)
            for level in constraint.allow:
                if level not in self.arms:
                    raise ValueError(
                        f"constraint {constraint.id!r} constrains {level!r}, which is not an arm; "
                        "fixed levels come from the seed manifest and cannot be pruned"
                    )
                if order[level] <= latest_condition:
                    raise ValueError(
                        f"constraint {constraint.id!r} constrains {level!r} at or above the "
                        "level it conditions on; a context must be known before it applies"
                    )
        return self

    @model_validator(mode="after")
    def context_weights_reference_known_levels(self) -> TaxonomySpec:
        order = {level: index for index, level in enumerate(self.hierarchy)}
        for level, arm in self.arms.items():
            for entry in arm.context_weights:
                for condition in entry.when:
                    if condition not in order:
                        raise ValueError(
                            f"arm {level!r} has context_weights on unknown level {condition!r}"
                        )
                    if order[condition] >= order[level]:
                        raise ValueError(
                            f"arm {level!r} has context_weights conditioning on {condition!r}, "
                            "which is not strictly above it"
                        )
        return self


class RewriteValidationSpec(StrictModel):
    validator_id: str = Field(min_length=1)
    mode: Literal["deterministic"] = "deterministic"
    on_invalid: Literal["retry", "fail"] = "retry"
    maximum_attempts: int = Field(default=5, ge=1)


class RewriteDeduplicationSpec(StrictModel):
    scope: Literal["semantic_seed_path", "structural_path", "none"] = "semantic_seed_path"
    on_duplicate: Literal["retry", "accept"] = "retry"


class RewriteSpec(StrictModel):
    """The frozen rewrite procedure whose induced distribution is the estimand's kernel."""

    kernel_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_template: str = Field(min_length=1)
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(gt=0)
    seed_replayable: bool = False
    validation: RewriteValidationSpec
    deduplication: RewriteDeduplicationSpec = Field(default_factory=RewriteDeduplicationSpec)


class JudgeCalibrationSpec(StrictModel):
    """Judge-misclassification calibration for a generative evaluation.

    ``mode: assumed`` carries no evidence: every corrected interval it produces is a
    diagnostic sensitivity result. ``mode: validated`` estimates the same two conditional
    accuracies from human labels drawn within each judge stratum, and its corrected
    intervals are certifying at the combined error budget.

    Human labels key on the sample identifier materialization wrote, since a generative
    run has no prompt manifest to key on.
    """

    mode: Literal["assumed", "validated"] = "assumed"
    scenarios: list[JudgeCalibrationScenarioSpec] = Field(default_factory=list)
    primary: str | None = None
    validation_manifest: str | None = None
    sampling: Literal["simple_random_within_judge_stratum"] = (
        "simple_random_within_judge_stratum"
    )
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def coherent_mode(self) -> JudgeCalibrationSpec:
        if self.mode == "assumed":
            if not self.scenarios:
                raise ValueError("assumed judge calibration requires at least one scenario")
            identifiers = [scenario.id for scenario in self.scenarios]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("judge calibration scenarios repeat an id")
            if self.primary is None:
                if len(self.scenarios) > 1:
                    raise ValueError(
                        "assumed judge calibration with several scenarios must name a primary"
                    )
                self.__dict__["primary"] = self.scenarios[0].id
            elif self.primary not in identifiers:
                raise ValueError(f"primary scenario {self.primary!r} is not declared")
            if self.validation_manifest is not None:
                raise ValueError("validation_manifest applies only to mode: validated")
        else:
            if self.validation_manifest is None:
                raise ValueError("validated judge calibration requires a validation_manifest")
            if self.scenarios:
                raise ValueError("validated judge calibration does not declare scenarios")
        return self


class ScoreMappingSpec(StrictModel):
    failure_values: list[str] = Field(min_length=1)
    safe_values: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def mappings_are_disjoint(self) -> ScoreMappingSpec:
        overlap = set(self.failure_values) & set(self.safe_values)
        if overlap:
            raise ValueError(f"score values mapped both ways: {', '.join(sorted(overlap))}")
        return self


class ScoreSpec(StrictModel):
    name: str = Field(min_length=1)
    mapping: ScoreMappingSpec


class CohortSpec(StrictModel):
    model: str = Field(min_length=1)
    generate_config: dict[str, Any] = Field(default_factory=dict)


class TargetSpec(StrictModel):
    cohorts: dict[str, CohortSpec] = Field(min_length=1)
    shared_rewrites: bool = True
    rollouts_per_rewrite: int = Field(default=1, ge=1)
    score: ScoreSpec
    estimand: Literal["end_to_end_system", "underlying_model"] = "end_to_end_system"

    @model_validator(mode="after")
    def cohort_ids_are_canonical(self) -> TargetSpec:
        for cohort_id in self.cohorts:
            reject_reserved("cohort id", cohort_id)
        return self


class SamplingSpec(StrictModel):
    """Direct target-distribution sampling, optionally with guaranteed stratum coverage.

    The adaptive, Neyman-allocated and importance-sampling methods in the original plan
    are deliberately absent: measurement showed they widen the bound this package
    reports even while they reduce estimator variance.
    """

    method: Literal["direct", "proportional_stratified"] = "direct"
    total_rewrites: int = Field(gt=0)
    minimum_rewrites_per_stratum: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def minimum_requires_stratification(self) -> SamplingSpec:
        if self.method == "direct" and self.minimum_rewrites_per_stratum:
            raise ValueError(
                "minimum_rewrites_per_stratum requires method: proportional_stratified"
            )
        return self


class InferenceSpec(StrictModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    coverage_mode: Literal["one_sided_upper", "two_sided"] = "one_sided_upper"
    report_all_nodes: bool = True
    simultaneous_tree: bool = False


class GenerativePlan(StrictModel):
    schema_version: Literal["1.0"]
    evaluation: GenerativeMetadata
    taxonomy: TaxonomySpec
    rewrite: RewriteSpec
    target: TargetSpec
    sampling: SamplingSpec
    judge_calibration: JudgeCalibrationSpec | None = None
    inference: InferenceSpec = Field(default_factory=InferenceSpec)


def load_generative_plan(path: str | Path) -> GenerativePlan:
    """Parse and strictly validate a generative evaluation plan."""

    text = Path(path).read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"generative plan {path} does not contain a YAML mapping")
    return GenerativePlan.model_validate(document)
