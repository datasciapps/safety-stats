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

from inspect_safety_stats.config.models import (
    JudgeCalibrationScenarioSpec,
    PlanFloat,
    PlanInt,
)

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
    selection_seed: PlanInt = Field(ge=0)

    @model_validator(mode="after")
    def confirmatory_has_freeze_time(self) -> GenerativeMetadata:
        if self.intent == "confirmatory" and self.plan_frozen_at is None:
            raise ValueError("confirmatory evaluations require evaluation.plan_frozen_at")
        return self


class ArmValueSpec(StrictModel):
    id: str = Field(min_length=1)
    weight: PlanFloat = Field(ge=0.0, le=1.0)

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
    maximum_paths: PlanInt = Field(default=500_000, gt=0)

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
    """How a rewrite is judged acceptable.

    ``invoke`` is experimental. Left ``false``, ``validator_id`` is only declared and
    fingerprinted, and the kernel's own verdict is trusted. Set ``true``, materialization
    needs a validator whose ``validator_id`` is this one, calls it on every rewrite the
    kernel returns as valid, and records every attempt in ``attempts.csv`` so the analysis
    can report fidelity by arm.
    """

    validator_id: str = Field(min_length=1)
    mode: Literal["deterministic"] = "deterministic"
    on_invalid: Literal["retry", "fail"] = "retry"
    maximum_attempts: PlanInt = Field(default=5, ge=1)
    invoke: bool = False


class RewriteDeduplicationSpec(StrictModel):
    """Which rewrites within a scope count as the same prompt.

    By default only an exact repeat of the text does. ``near_duplicate: normalized_text``
    (experimental) also counts two rewrites that agree after Unicode NFKC normalization,
    case folding, removal of punctuation and symbols, and whitespace collapsing, and makes
    materialization record every attempt in ``attempts.csv``. A near duplicate is handled
    by ``on_duplicate`` like an exact one.
    """

    scope: Literal["semantic_seed_path", "structural_path", "none"] = "semantic_seed_path"
    on_duplicate: Literal["retry", "accept"] = "retry"
    near_duplicate: Literal["off", "normalized_text"] = "off"


def turn_count(level: str, value: str) -> int:
    """The user-turn count a value of the conversation-turns arm declares."""

    if not value.isdigit() or int(value) < 1:
        raise ValueError(
            f"rewrite.conversation_turns arm {level!r} has value {value!r}; its values are "
            "user-turn counts, positive integers such as '1' or '3'"
        )
    return int(value)


class SystemPromptSpec(StrictModel):
    """An arm over system prompts (experimental).

    The rewrite procedure attaches ``texts[value]`` as the system turn of every rewrite on
    a path whose ``level`` takes that value. An empty text declares that value's paths to
    have no system prompt. A system turn the kernel writes stops the run on every path, an
    empty text's included, so each arm value's system prompt is exactly the declared text.
    """

    level: str = Field(min_length=1)
    texts: dict[str, str] = Field(min_length=1)


class RewriteSpec(StrictModel):
    """The frozen rewrite procedure whose induced distribution is the estimand's kernel."""

    kernel_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_template: str = Field(min_length=1)
    temperature: PlanFloat = Field(ge=0.0)
    max_tokens: PlanInt = Field(gt=0)
    seed_replayable: bool = False
    validation: RewriteValidationSpec
    deduplication: RewriteDeduplicationSpec = Field(default_factory=RewriteDeduplicationSpec)
    # Experimental. ``conversation`` makes the kernel return whole conversations (system,
    # user and generator-written assistant turns), materialized as Inspect message lists.
    output: Literal["prompt", "conversation"] = "prompt"
    system_prompts: SystemPromptSpec | None = None
    # Experimental. An arm whose values are user-turn counts: each conversation on a path
    # must have exactly that many user turns, or the attempt is rejected as invalid.
    conversation_turns: str | None = None

    @model_validator(mode="after")
    def turns_need_conversations(self) -> RewriteSpec:
        if self.conversation_turns is not None and self.output != "conversation":
            raise ValueError("rewrite.conversation_turns requires rewrite.output: conversation")
        return self

    def stores_conversations(self) -> bool:
        """Whether accepted rewrites are conversations rather than bare prompts."""

        return self.output == "conversation" or self.system_prompts is not None

    @property
    def records_attempts(self) -> bool:
        """Whether materialization writes the per-attempt ledger, ``attempts.csv``."""

        return self.validation.invoke or self.deduplication.near_duplicate != "off"


class JudgeCalibrationSpec(StrictModel):
    """Judge-misclassification calibration for a generative evaluation.

    ``mode: assumed`` carries no evidence: every corrected interval it produces is a
    diagnostic sensitivity result. ``mode: validated`` estimates the same two conditional
    accuracies from human labels drawn within each judge stratum. Its corrected root is
    certifying at the combined error budget when the root weights every unit equally; its
    corrected nodes and slices are sensitivities, since the pooled labels do not measure
    the judge's accuracy within them.

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
    alpha: PlanFloat = Field(default=0.05, gt=0.0, lt=1.0)

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
            # The generative correction reads one box per scenario; a per-cohort box would
            # be ignored, so it is refused.
            per_cohort = [scenario.id for scenario in self.scenarios if scenario.cohorts]
            if per_cohort:
                raise ValueError(
                    "per-cohort judge calibration boxes are not supported in generative "
                    "plans; remove cohorts from scenarios " + ", ".join(map(repr, per_cohort))
                )
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
    rollouts_per_rewrite: PlanInt = Field(default=1, ge=1)
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
    total_rewrites: PlanInt = Field(gt=0)
    minimum_rewrites_per_stratum: PlanInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def minimum_requires_stratification(self) -> SamplingSpec:
        if self.method == "direct" and self.minimum_rewrites_per_stratum:
            raise ValueError(
                "minimum_rewrites_per_stratum requires method: proportional_stratified"
            )
        return self


class ReweightingSpec(StrictModel):
    """An alternative marginal mix at one hierarchy level, declared in advance.

    Design weights are a judgement, not a measurement, and on a realistic taxonomy
    changing them moves the root risk by about as much as the interval is wide. Declaring
    the alternatives in the frozen plan is what separates a sensitivity analysis from
    picking the weighting that gives the answer you wanted.
    """

    id: str = Field(min_length=1)
    level: str = Field(min_length=1)
    weights: dict[str, PlanFloat] = Field(min_length=1)

    @model_validator(mode="after")
    def weights_are_a_distribution(self) -> ReweightingSpec:
        if any(weight < 0.0 for weight in self.weights.values()):
            raise ValueError(f"reweighting {self.id!r} has a negative weight")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reweighting {self.id!r} weights sum to {total}, not 1")
        return self


class FamilySliceSpec(StrictModel):
    """Every path carrying ``level=value``, as a member of the declared family."""

    level: str = Field(min_length=1)
    value: str = Field(min_length=1)


class ArmContrastSpec(StrictModel):
    """Two values of one arm level, compared on the seeds that hold both."""

    id: str = Field(min_length=1)
    level: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    baseline: str = Field(min_length=1)

    @model_validator(mode="after")
    def values_differ(self) -> ArmContrastSpec:
        if self.candidate == self.baseline:
            raise ValueError(f"contrast {self.id!r} compares {self.candidate!r} with itself")
        return self


class SliceFamilySpec(StrictModel):
    """Experimental: slices and arm contrasts covered simultaneously at ``1 - alpha``.

    Declared in the frozen plan for the same reason as design sensitivity: the family the
    error budget is spent on must not be chosen after seeing the results.
    """

    slices: list[FamilySliceSpec] = Field(default_factory=list)
    contrasts: list[ArmContrastSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def family_is_well_formed(self) -> SliceFamilySpec:
        if not self.slices and not self.contrasts:
            raise ValueError("inference.slice_family declares no slice and no contrast")
        keys = [(item.level, item.value) for item in self.slices]
        if len(set(keys)) != len(keys):
            raise ValueError("inference.slice_family declares the same slice twice")
        ids = [item.id for item in self.contrasts]
        if len(set(ids)) != len(ids):
            raise ValueError("inference.slice_family declares the same contrast id twice")
        return self


class InferenceSpec(StrictModel):
    alpha: PlanFloat = Field(default=0.05, gt=0.0, lt=1.0)
    coverage_mode: Literal["one_sided_upper", "two_sided"] = "one_sided_upper"
    report_all_nodes: bool = True
    simultaneous_tree: bool = False
    design_sensitivity: list[ReweightingSpec] = Field(default_factory=list)
    slice_family: SliceFamilySpec | None = None


class GenerativePlan(StrictModel):
    """A parsed and strictly validated generative evaluation plan.

    Built by :func:`load_generative_plan`. Unknown keys are rejected, and ``inference``
    falls back to its defaults when the plan omits it.
    """

    schema_version: Literal["1.0"]
    evaluation: GenerativeMetadata
    taxonomy: TaxonomySpec
    rewrite: RewriteSpec
    target: TargetSpec
    sampling: SamplingSpec
    judge_calibration: JudgeCalibrationSpec | None = None
    inference: InferenceSpec = Field(default_factory=InferenceSpec)

    @model_validator(mode="after")
    def system_prompts_name_an_arm(self) -> GenerativePlan:
        turns_level = self.rewrite.conversation_turns
        if turns_level is not None:
            turns_arm = self.taxonomy.arms.get(turns_level)
            if turns_arm is None:
                raise ValueError(
                    f"rewrite.conversation_turns {turns_level!r} is not an arm in taxonomy.arms"
                )
            for value in turns_arm.values or []:
                turn_count(turns_level, value.id)
        spec = self.rewrite.system_prompts
        if spec is None:
            return self
        if spec.level == turns_level:
            raise ValueError(
                "rewrite.system_prompts and rewrite.conversation_turns name the same arm"
            )
        arm = self.taxonomy.arms.get(spec.level)
        if arm is None:
            raise ValueError(
                f"rewrite.system_prompts.level {spec.level!r} is not an arm in "
                "taxonomy.arms; a system prompt varies only over a sampled level"
            )
        if arm.values is not None:
            declared = {value.id for value in arm.values}
            missing = sorted(declared - set(spec.texts))
            unknown = sorted(set(spec.texts) - declared)
            if missing:
                raise ValueError(
                    f"rewrite.system_prompts.texts has no text for {spec.level} values: "
                    f"{', '.join(missing)}"
                )
            if unknown:
                raise ValueError(
                    f"rewrite.system_prompts.texts names values {spec.level!r} does not "
                    f"declare: {', '.join(unknown)}"
                )
        return self

    @model_validator(mode="after")
    def slice_family_is_over_arm_levels(self) -> GenerativePlan:
        family = self.inference.slice_family
        if family is None:
            return self
        for level in [item.level for item in family.slices] + [
            item.level for item in family.contrasts
        ]:
            if level not in self.taxonomy.arms:
                raise ValueError(
                    f"inference.slice_family names {level!r}, which is not an arm level; "
                    "slices of fixed levels are nodes of the tree"
                )
        return self


def load_generative_plan(path: str | Path) -> GenerativePlan:
    """Parse and strictly validate a generative evaluation plan."""

    text = Path(path).read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"generative plan {path} does not contain a YAML mapping")
    return GenerativePlan.model_validate(document)
