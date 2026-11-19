from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

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

from inspect_safety_stats.config.methods import (
    DEFAULT_COMPARISON_METHOD,
    DEFAULT_DIRECT_MEAN_INTERVAL_METHOD,
    ComparisonMethod,
    DirectMeanMethod,
)


def _refuse_booleans(kind: str) -> BeforeValidator:
    def refuse(value: object) -> object:
        # Lax validation would read `true` as 1. A plan is a pre-registered document, so a
        # typo in a numeric field must fail rather than run with a guessed value.
        if isinstance(value, bool):
            raise ValueError(f"must be {kind}, not a boolean")
        return value

    return BeforeValidator(refuse)


# Every integer and every float in a plan, analysis or generative, uses one of these types.
PlanInt = Annotated[int, _refuse_booleans("an integer")]
PlanFloat = Annotated[float, _refuse_booleans("a number")]

# report.precision_target when a plan leaves it out.
DEFAULT_PRECISION_TARGET = 0.05

# The thresholds a claim takes, from its plan or a decision policy: acceptable_upper below
# unacceptable_lower, both in [THRESHOLD_MIN, THRESHOLD_MAX]. Verdict ranges read the same limits.
THRESHOLD_MIN = -1
THRESHOLD_MAX = 1


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class OmitsAbsentFields(StrictModel):
    """A model that leaves the fields named in ``absent_when_none`` out of its dump while None.

    A field added after plans were frozen is left out until a plan sets it, so the resolved plan
    a report records, and the plan's hash, stay as they were for every plan that does not.
    """

    absent_when_none: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def leave_out_absent_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for name in self.absent_when_none:
            if getattr(self, name) is None:
                data.pop(name, None)
        return data


def _group_scope(groups: list[str] | None, where: str) -> None:
    """Reject an empty or repeated group scope; unknown groups are a plan-reference error."""

    if groups is None:
        return
    if not groups:
        raise ValueError(
            f"{where} groups cannot be empty; leave the field out to cover every group"
        )
    if len(set(groups)) != len(groups):
        raise ValueError(f"{where} groups must be unique")


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
    value: PlanFloat


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


class OutcomeMappingSensitivitySpec(StrictModel):
    """Experimental: an alternative reading of the same score, reported beside the primary one.

    The whole analysis is repeated under ``mapping``; its results are shown side by side with
    the primary results and never enter a claim's status or the decision.
    """

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str | None = None
    mapping: OutcomeMappingSpec


# The id the primary mapping has in the side-by-side outputs.
PRIMARY_MAPPING_ID = "primary"


class ScoreSpec(StrictModel):
    name: str = Field(min_length=1)
    mapping: OutcomeMappingSpec
    missing: Literal["conservative_bounds", "block", "count_as_failure", "exclude"] = (
        "conservative_bounds"
    )
    errors: Literal["end_to_end", "conservative_bounds", "block", "count_as_failure", "exclude"] = (
        "end_to_end"
    )
    # Experimental. How a rollout counts whose output a provider's content filter stopped
    # before it delivered any content, without a sample error: `score` leaves it to the score,
    # as before; `safe` counts it as a target policy block, safe for a harm estimand; `failure`
    # counts it as a refusal, for an over-refusal estimand on safe prompts.
    content_filter_stops: Literal["score", "safe", "failure"] = "score"
    # Experimental. How a rollout counts whose answer stopped at the generation token limit
    # (Inspect's stop reason `max_tokens`) and has no score and no sample error: `score` does
    # not look, and leaves it to `missing`, as before; `missing` also leaves it to `missing`,
    # but counts it on its own; `safe` and `failure` classify it, and count it on its own. A
    # truncated answer that was scored keeps its score.
    token_limit: Literal["score", "missing", "safe", "failure"] = "score"
    # Experimental. Left out of the plan when empty, so that plans without it keep their hash.
    mapping_sensitivities: list[OutcomeMappingSensitivitySpec] = Field(
        default_factory=list,
        description=(
            "Experimental. Alternative outcome mappings of the same score, analysed and "
            "reported beside the primary mapping; they never enter claims or the decision."
        ),
    )

    @model_serializer(mode="wrap")
    def leave_out_defaults(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # The experimental fields are left out at their defaults, so that plans written before
        # them keep their hash.
        data: dict[str, Any] = handler(self)
        if self.content_filter_stops == "score":
            data.pop("content_filter_stops", None)
        if self.token_limit == "score":
            data.pop("token_limit", None)
        if not self.mapping_sensitivities:
            data.pop("mapping_sensitivities", None)
        return data

    @model_validator(mode="after")
    def distinct_mapping_sensitivities(self) -> ScoreSpec:
        identifiers = [item.id for item in self.mapping_sensitivities]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("outcome mapping sensitivity identifiers must be unique")
        if PRIMARY_MAPPING_ID in identifiers:
            raise ValueError(
                f"{PRIMARY_MAPPING_ID!r} names the primary mapping and cannot name a sensitivity"
            )
        for item in self.mapping_sensitivities:
            if item.mapping == self.mapping:
                raise ValueError(
                    f"outcome mapping sensitivity {item.id!r} repeats the primary mapping"
                )
        return self


def judge_run_score(primary: ScoreSpec, judge: ScoreSpec) -> ScoreSpec:
    """The score a judge's run under ``outputs.judge_claims`` reads: the judge's own, except
    the stop reasons the judge leaves at ``score``, which it takes from the primary score.

    ``content_filter_stops`` and ``token_limit`` say how rollouts count that the target's
    provider stopped, which is the same for every judge of the same rollouts. Left at
    ``score`` the judge's score says nothing about them, so its run counts them as the plan's
    primary analysis does, and differs from it in the judge's verdicts alone.
    """

    update: dict[str, Any] = {}
    if judge.content_filter_stops == "score":
        update["content_filter_stops"] = primary.content_filter_stops
    if judge.token_limit == "score":
        update["token_limit"] = primary.token_limit
    return judge.model_copy(update=update) if update else judge


def _content_filter_stops_problem(score: ScoreSpec, estimand: str) -> str | None:
    """Why ``score.content_filter_stops`` does not fit the estimand or ``score.errors``, if it
    does not."""

    stops = score.content_filter_stops
    if stops != "score" and estimand != "end_to_end_system":
        return (
            "score.content_filter_stops counts a provider's block as the outcome of the "
            "end-to-end system; it needs estimand end_to_end_system"
        )
    if stops == "failure" and score.errors == "end_to_end":
        return (
            "score.content_filter_stops: failure counts a block as a refusal, but "
            "score.errors: end_to_end counts a policy-blocked sample error as safe; "
            "choose another score.errors"
        )
    return None


class JudgeRateRangeSpec(StrictModel):
    """A declared closed range for one conditional judge accuracy."""

    lower: PlanFloat = Field(ge=0, le=1)
    upper: PlanFloat = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> JudgeRateRangeSpec:
        if self.lower > self.upper:
            raise ValueError("judge rate range lower endpoint cannot exceed its upper endpoint")
        return self


class JudgeCohortBoxSpec(StrictModel):
    """A scenario's declared accuracy box for one cohort, replacing its default box there."""

    rho_1: JudgeRateRangeSpec
    rho_0: JudgeRateRangeSpec


class JudgeCalibrationScenarioSpec(OmitsAbsentFields):
    """One named assumption about the judge's conditional accuracy.

    ``rho_1`` is P(truly unsafe | judge said unsafe) and ``rho_0`` is
    P(truly unsafe | judge said safe).

    ``cohorts`` gives some cohorts a box of their own, for a judge whose accuracy depends on
    the model that produced the output. A cohort it does not list takes ``rho_1`` and
    ``rho_0``. Each arm of a corrected comparison takes its own cohort's box.
    """

    # Left out of the serialised plan when absent, so the hash and the resolved plan of a
    # plan without it are what they were before the field existed.
    absent_when_none = ("cohorts",)

    id: str = Field(min_length=1)
    rho_1: JudgeRateRangeSpec
    rho_0: JudgeRateRangeSpec
    cohorts: dict[str, JudgeCohortBoxSpec] | None = None

    @model_validator(mode="after")
    def strictly_informative(self) -> JudgeCalibrationScenarioSpec:
        if self.rho_1.lower <= self.rho_0.upper:
            raise ValueError(
                f"judge calibration scenario {self.id!r} is not strictly informative: "
                "the rho_1 lower endpoint must exceed the rho_0 upper endpoint"
            )
        if self.cohorts is not None and not self.cohorts:
            raise ValueError(
                f"judge calibration scenario {self.id!r} cohorts cannot be empty; leave the "
                "field out to give every cohort the scenario's box"
            )
        for cohort_id, box in (self.cohorts or {}).items():
            if box.rho_1.lower <= box.rho_0.upper:
                raise ValueError(
                    f"judge calibration scenario {self.id!r} is not strictly informative for "
                    f"cohort {cohort_id!r}: the rho_1 lower endpoint must exceed the rho_0 "
                    "upper endpoint"
                )
        return self

    def box(self, cohort_id: str) -> tuple[JudgeRateRangeSpec, JudgeRateRangeSpec]:
        """The (rho_1, rho_0) box this scenario declares for one cohort."""

        own = (self.cohorts or {}).get(cohort_id)
        return (self.rho_1, self.rho_0) if own is None else (own.rho_1, own.rho_0)


class PublishedValidationSetSpec(StrictModel):
    """A published human-labelled judge validation set, and the judge's sensitivity on it.

    Experimental. Only the sensitivity is transported to the analysed run: the judge's
    false-positive rate on the published set does not carry over to another model's
    outputs, so it is not declared here at all. ``name``, ``source``, ``judge`` and
    ``rubric`` are the provenance the transport assumption rests on: the same judge and
    rubric, applied to outputs of other generators.
    """

    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    judge: str = Field(min_length=1)
    rubric: str = Field(min_length=1)
    human_unsafe: PlanInt = Field(ge=1)
    judge_flagged_unsafe: PlanInt = Field(ge=0)
    sensitivity_alpha: PlanFloat = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def flagged_within_unsafe(self) -> PublishedValidationSetSpec:
        if self.judge_flagged_unsafe > self.human_unsafe:
            raise ValueError("judge_flagged_unsafe cannot exceed human_unsafe")
        return self


class LabellingProtocolSpec(StrictModel):
    """Experimental: how the human labels of a validated calibration were collected.

    ``queue`` is a CSV whose ``observation_id`` column lists the rollouts in the order they
    are to be labelled; a queue that orders each judge stratum by a seeded random
    permutation makes every prefix a simple random sample within each stratum. ``tier``
    keeps only the queue rows whose ``tier`` column has that value, for a queue that also
    holds rows the calibration does not use. ``seed`` records the permutation's seed for an
    audit; the package cannot recompute the queue from it.

    The analysis checks that the validation manifest labels exactly the first rows of the
    queue. ``stopping`` declares why labelling stopped there: ``fixed_size`` at ``size`` rows,
    which the analysis checks; ``label_independent`` at a point chosen without looking at the
    labels, such as when time ran out; ``data_dependent`` possibly because of the labels,
    which only anytime-valid bounds allow, so the calibration then uses confidence sequences.
    """

    queue: Path
    tier: str | None = Field(default=None, min_length=1)
    seed: PlanInt | None = None
    stopping: Literal["fixed_size", "label_independent", "data_dependent"]
    size: PlanInt | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def size_only_when_fixed(self) -> LabellingProtocolSpec:
        if self.stopping == "fixed_size" and self.size is None:
            raise ValueError("labelling protocol stopping: fixed_size requires a size")
        if self.stopping != "fixed_size" and self.size is not None:
            raise ValueError("labelling protocol size applies only to stopping: fixed_size")
        return self

    @model_validator(mode="after")
    def tier_without_padding(self) -> LabellingProtocolSpec:
        # The queue's tier cells are read without surrounding whitespace, so a padded tier
        # would select no row, and every label would read as outside the queue.
        if self.tier is not None and self.tier != self.tier.strip():
            raise ValueError(
                f"labelling protocol tier {self.tier!r} begins or ends with whitespace; "
                "write the tier name without it"
            )
        return self


class JudgeCalibrationSpec(OmitsAbsentFields):
    """Judge-misclassification calibration, either declared, transported or measured.

    ``mode: assumed`` carries no evidence: every corrected interval it produces is a
    diagnostic sensitivity result. ``mode: transported`` (experimental) bounds true risk
    from the judge's sensitivity on a published human-labelled set and is likewise a
    diagnostic sensitivity result. ``mode: validated`` estimates the two conditional
    accuracies from human labels, and its corrected intervals are certifying at the combined
    error budget. ``sampling`` says how the labels were drawn: a simple random sample within
    each judge stratum (the default); a simple random sample of the classified rollouts
    (``simple_random``), which also estimates the judge's sensitivity and false-positive
    rate by true class; or, experimental, a stratified design whose declared strata, given
    in ``validation_strata``, nest within the judge strata
    (``stratified_within_judge_stratum``).

    ``groups`` scopes an assumed or validated calibration to declared groups. Without it
    there is one calibration per cohort, for every row of the cohort. With it the assumed
    scenarios apply only to rows of the listed groups, and validated mode estimates one
    calibration per cohort and listed group, from the labels on that group's rollouts.
    """

    # Left out of the serialised plan when absent, so the hash and the resolved plan of a
    # plan without them are what they were before the fields existed.
    absent_when_none = ("published_set", "groups", "validation_strata", "protocol")

    mode: Literal["assumed", "validated", "transported"] = "assumed"
    scenarios: list[JudgeCalibrationScenarioSpec] = Field(default_factory=list)
    primary: str | None = None
    validation_manifest: Path | None = None
    family: str | None = None
    sampling: Literal[
        "simple_random_within_judge_stratum", "simple_random", "stratified_within_judge_stratum"
    ] = "simple_random_within_judge_stratum"
    published_set: PublishedValidationSetSpec | None = None
    groups: list[str] | None = None
    # Experimental: the declared fine stratum of every classified observation, for
    # sampling: stratified_within_judge_stratum.
    validation_strata: Path | None = None
    # Experimental: how the labels were collected, which the analysis checks.
    protocol: LabellingProtocolSpec | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> JudgeCalibrationSpec:
        _group_scope(self.groups, "judge calibration")
        if self.protocol is not None and self.mode != "validated":
            raise ValueError("a labelling protocol requires mode: validated")
        if (
            self.protocol is not None
            and self.protocol.stopping == "data_dependent"
            and self.sampling != "simple_random_within_judge_stratum"
        ):
            # The noisy-but-valid test's bounds by true class, and the stratified design's
            # fine-stratum bounds, have no anytime-valid form here yet.
            raise ValueError(
                "labelling protocol stopping: data_dependent requires sampling: "
                "simple_random_within_judge_stratum"
            )
        if self.mode != "transported" and self.published_set is not None:
            raise ValueError("published_set requires mode: transported")
        stratified = self.sampling == "stratified_within_judge_stratum"
        if stratified and self.mode != "validated":
            raise ValueError("sampling: stratified_within_judge_stratum requires mode: validated")
        if stratified != (self.validation_strata is not None):
            raise ValueError(
                "validation_strata and sampling: stratified_within_judge_stratum go together; "
                "declare both or neither"
            )
        if self.mode == "transported":
            if self.published_set is None:
                raise ValueError("transported judge calibration requires a published_set")
            if (
                self.scenarios
                or self.primary is not None
                or self.validation_manifest is not None
                or self.family is not None
                or self.groups is not None
            ):
                raise ValueError(
                    "transported judge calibration takes only a published_set; remove the "
                    "scenarios, primary, validation_manifest, family and groups"
                )
            return self
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


class JudgeAccuracySpec(OmitsAbsentFields):
    """Declared prevalence-free accuracy ranges for one judge.

    ``groups`` scopes the declaration to declared groups: the ranges hold on those groups'
    rollouts only. Without it they hold on every rollout.
    """

    absent_when_none = ("groups",)

    sensitivity: JudgeRateRangeSpec
    specificity: JudgeRateRangeSpec
    groups: list[str] | None = None

    @model_validator(mode="after")
    def valid_scope(self) -> JudgeAccuracySpec:
        _group_scope(self.groups, "judge accuracy")
        return self


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
    def no_mapping_sensitivities(self) -> JudgeSpec:
        if self.score.mapping_sensitivities:
            raise ValueError(
                f"judge {self.id!r}: mapping_sensitivities applies to inputs.score only"
            )
        return self

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


class JudgeFlipSpec(StrictModel):
    """Experimental: the judge's flip rate under content-preserving wrappers (#122).

    ``manifest`` is a CSV with columns ``observation_id``, ``wrapper`` and ``verdict``: the
    primary judge's verdict (``safe`` or ``unsafe``) on a judge-positive completion of the run
    after it was rescored inside one wrapper, one row per completion and wrapper. Every rescored
    completion carries a row for every declared wrapper. With ``sampling: all`` every
    judge-positive completion of the run was rescored; with ``simple_random`` a simple random
    sample of them was. The flip rate is a diagnostic: it never changes a claim status, and a
    rate above ``warn_above`` gives a warning. ``alpha`` is the two-sided error level of its
    Clopper-Pearson interval.
    """

    manifest: Path
    wrappers: list[str] = Field(min_length=1)
    sampling: Literal["all", "simple_random"] = "all"
    warn_above: PlanFloat = Field(gt=0, lt=1)
    alpha: PlanFloat = Field(default=0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def distinct_wrappers(self) -> JudgeFlipSpec:
        if any(not item.strip() for item in self.wrappers):
            raise ValueError("judge_flip wrappers must be non-empty names")
        if len(set(self.wrappers)) != len(self.wrappers):
            raise ValueError("judge_flip wrappers must be unique")
        return self


class InputsSpec(OmitsAbsentFields):
    # sample_join is None only under schema 3.0, which leaves it out of the plan.
    absent_when_none = ("sample_join", "panel_groups", "judge_flip")

    design_manifest: Path
    estimand: Literal["end_to_end_system", "underlying_model"] = "end_to_end_system"
    # Schema 2.0 only; None, and left out of the plan, under schema 3.0.
    sample_join: SampleJoinSpec | None = Field(default_factory=SampleJoinSpec)
    score: ScoreSpec
    judge_calibration: JudgeCalibrationSpec | None = None
    judges: list[JudgeSpec] = Field(default_factory=list)
    composites: list[CompositeJudgeSpec] = Field(default_factory=list)
    # The groups the judge panel section also breaks its counts and agreement down by.
    panel_groups: list[str] | None = None
    # Experimental: the primary judge's flip rate under content-preserving wrappers.
    judge_flip: JudgeFlipSpec | None = None

    @model_validator(mode="after")
    def validate_judge_panel(self) -> InputsSpec:
        _group_scope(self.panel_groups, "judge panel")
        if self.panel_groups is not None and not self.judges:
            raise ValueError("panel_groups requires declared judges")
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

    @model_validator(mode="after")
    def content_filter_stops_fit_the_estimand(self) -> InputsSpec:
        problem = _content_filter_stops_problem(self.score, self.estimand)
        if problem is not None:
            raise ValueError(problem)
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
    beta: PlanFloat | None = None
    lambda_: PlanFloat | None = Field(default=None, alias="lambda")
    k: PlanInt | None = None
    threshold: PlanFloat | None = None
    threshold_column: str | None = None
    diagnostic_only: bool = False
    # Experimental, worst_of_k only: a direct interval from the unbiased per-prompt shares.
    direct_interval: bool = False

    @model_serializer(mode="wrap")
    def leave_out_absent_direct_interval(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Left out unless requested, so plans without it keep their hash and rows their
        # parameters.
        data: dict[str, Any] = handler(self)
        if not self.direct_interval:
            data.pop("direct_interval", None)
        return data

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
        if self.direct_interval and self.type != "worst_of_k":
            raise ValueError(
                "direct_interval applies to worst_of_k measures only; weighted means get "
                "direct intervals from outputs.direct_mean_intervals"
            )
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
    explicit_alpha: dict[str, PlanFloat] | None = None

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
    method: DirectMeanMethod = DEFAULT_DIRECT_MEAN_INTERVAL_METHOD
    reference_methods: list[DirectMeanMethod] = Field(default_factory=list)
    family: str | None = None
    groups: list[str] | None = None
    simultaneous: bool = True
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec | None = None
    hierarchy_nodes: bool = False
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "flat_bonferroni"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"
    # Experimental (#127): report M = inf_delta [U_delta + delta] beside each direct interval.
    single_valued_bound: bool = False

    @model_serializer(mode="wrap")
    def leave_out_default_single_valued_bound(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Plans hash their defaults (#56). Left out while off, a plan that does not use the
        # option keeps its hash and its resolved plan from before the option existed.
        data: dict[str, Any] = handler(self)
        if not self.single_valued_bound:
            data.pop("single_valued_bound", None)
        return data

    @model_validator(mode="after")
    def enabled_has_family(self) -> DirectMeanIntervalsSpec:
        if self.single_valued_bound and not self.enabled:
            raise ValueError("single_valued_bound requires enabled direct mean intervals")
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


class AnytimeBoundsSpec(StrictModel):
    """Experimental: anytime-valid upper bounds for runs stopped early (#126).

    For each cohort and group, a one-sided upper bound on the weighted mean failure
    probability that holds at every prefix of the run's sample order at once, and so wherever
    a rule such as optstop's stopped the run. The order must be Inspect's ``sample_shuffle``
    with a recorded integer seed; the stream is the first ``rollouts_per_item`` epochs of each
    item, in that order. ``independence_profile`` names a profile of
    ``design.independence_profiles``: ``[rollout]`` treats an item's rollouts as independent,
    and any profile that blocks by ``prompt_id`` treats only items as independent. The bounds
    split the ``family``'s alpha equally. They are reported as diagnostics beside the claims,
    and never change a claim's status.
    """

    order: Literal["inspect_sample_shuffle"] = "inspect_sample_shuffle"
    rollouts_per_item: PlanInt = Field(default=1, ge=1)
    independence_profile: str = Field(min_length=1)
    family: str = Field(min_length=1)
    groups: list[str] | None = None

    @model_validator(mode="after")
    def validate_groups(self) -> AnytimeBoundsSpec:
        _group_scope(self.groups, "anytime bound")
        return self


class OutputsSpec(StrictModel):
    prompt_bounds: PromptBoundsSpec = Field(default_factory=PromptBoundsSpec)
    direct_mean_intervals: DirectMeanIntervalsSpec = Field(default_factory=DirectMeanIntervalsSpec)
    risk_measures: list[RiskMeasureSpec] = Field(default_factory=list)
    # Experimental. Decide every claim again on each declared judge's own verdicts, as if the
    # plan named that judge as its primary outcome, and record the judges' configurations.
    judge_claims: bool = False
    # Experimental (#126): anytime-valid upper bounds for runs stopped early. Left out of the
    # dump while absent, so plans without it keep their hash.
    anytime_bounds: AnytimeBoundsSpec | None = None

    @model_serializer(mode="wrap")
    def leave_out_judge_claims_when_off(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Off, the field is left out, so plans written before it keep their hash and the
        # resolved plan a report records keeps its bytes.
        data: dict[str, Any] = handler(self)
        if not self.judge_claims:
            data.pop("judge_claims", None)
        if self.anytime_bounds is None:
            data.pop("anytime_bounds", None)
        return data


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
    method: ComparisonMethod = DEFAULT_COMPARISON_METHOD
    reference_methods: list[ComparisonMethod] = Field(default_factory=list)
    family: str | None = None
    simultaneous: bool = True
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec | None = None
    hierarchy_nodes: bool = False
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "flat_bonferroni"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"
    allocation_weight: PlanFloat = Field(default=1.0, gt=0)
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
    method: ComparisonMethod = DEFAULT_COMPARISON_METHOD
    reference_methods: list[ComparisonMethod] = Field(default_factory=list)
    family: str
    coverage_mode: Literal["two_sided", "per_direction"] = "two_sided"
    independence: IndependenceSelectionSpec
    hierarchy_nodes: bool = True
    hierarchy_calibration: Literal["flat_bonferroni", "frontier_sidak"] = "frontier_sidak"
    hierarchy_region: Literal["none", "linear_program"] = "linear_program"
    allocation_weight: PlanFloat = Field(default=1.0, gt=0)

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
    alpha: PlanFloat = Field(gt=0, lt=1)


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
    acceptable_upper: PlanFloat = Field(ge=THRESHOLD_MIN, le=THRESHOLD_MAX)
    unacceptable_lower: PlanFloat = Field(ge=THRESHOLD_MIN, le=THRESHOLD_MAX)
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


class AdaptiveClaimSpec(StrictModel):
    """Experimental (#125): a claim that an attacker adapting across behaviours fails the gate.

    The cohort's runs on the group's behaviours, taken in the order of the design manifest's
    ``order_field``, form one stream. The claim bounds the attacker's running mean of
    conditional success probabilities from below, with an anytime-valid bound that needs no
    independence between runs, and is FAIL when that bound reaches ``unacceptable_lower``.

    It can only FAIL. An upper bound on one adaptive attacker is no evidence about other
    attackers or budgets, so the claim takes no ``acceptable_upper`` and is never PASS.
    """

    id: str = Field(min_length=1)
    title: str
    cohort: str
    group: str
    order_field: str = Field(min_length=1)
    family: str
    unacceptable_lower: PlanFloat = Field(gt=0, le=1)
    mandatory: bool = True

    @model_validator(mode="before")
    @classmethod
    def refuse_pass(cls, data: Any) -> Any:
        if isinstance(data, dict) and "acceptable_upper" in data:
            raise ValueError(
                "an adaptive claim can only FAIL, so it takes no acceptable_upper: its lower "
                "bound describes one attacker at one budget, and an upper bound on that "
                "attacker is no evidence about other attackers or budgets. Put a PASS claim "
                "on a fixed set of attacks in claims instead"
            )
        return data


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
    # Experimental. Left out of the plan dump while off, so a plan that does not opt in keeps
    # its hash and every artifact it produced before the field existed.
    judge_error_stress: bool = False
    # Experimental (#11): write claims.json beside report.json. Left out of the plan dump while
    # off, like judge_error_stress.
    claims_export: bool = False
    # The widest a bound may lie from its estimate and still count as precise. Left out of
    # the plan dump at its default, so plans that do not set it keep their hash (#56).
    precision_target: PlanFloat = Field(default=DEFAULT_PRECISION_TARGET, gt=0, lt=0.5)

    @model_validator(mode="after")
    def claims_export_needs_decisions(self) -> ReportSpec:
        # claims.json carries every claim's status. With include_decisions false the rendered
        # report leaves statuses out, and whether that hides or redacts them is open (#58), so
        # the export is refused rather than deciding it here.
        if self.claims_export and not self.include_decisions:
            raise ValueError(
                "report.claims_export needs report.include_decisions: true, because claims.json "
                "carries each claim's status"
            )
        return self

    @model_serializer(mode="wrap")
    def leave_out_judge_error_stress_when_off(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not self.judge_error_stress:
            data.pop("judge_error_stress", None)
        if not self.claims_export:
            data.pop("claims_export", None)
        if self.precision_target == DEFAULT_PRECISION_TARGET:
            data.pop("precision_target", None)
        return data


class AnalysisPlan(StrictModel):
    """A validated analysis plan: the design, cohorts, groups, comparisons and claims.

    Read one from a file with ``load_plan``. Unknown fields are rejected. The fields are
    documented in docs/plan-reference.md, and ``inspect-safety-stats schema plan`` prints the
    JSON schema.
    """

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
    # Experimental (#125). Left out of the plan dump while empty, so a plan without it keeps
    # its hash and every artifact it produced before the field existed.
    adaptive_claims: list[AdaptiveClaimSpec] = Field(default_factory=list)
    policy: PolicySpec = Field(default_factory=PolicySpec)
    report: ReportSpec = Field(default_factory=ReportSpec)

    _source_path: Path | None = PrivateAttr(default=None)

    @model_serializer(mode="wrap")
    def leave_out_absent_adaptive_claims(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if not self.adaptive_claims:
            data.pop("adaptive_claims", None)
        return data

    @model_validator(mode="after")
    def adaptive_claims_stay_out_of_claims_export(self) -> AnalysisPlan:
        # claims.json follows the claim schema of #11, which has an acceptable_upper for every
        # claim. Until it has a form for a claim that can only FAIL, the two are refused
        # together rather than leaving the adaptive claims out of the export.
        if self.adaptive_claims and self.report.claims_export:
            raise ValueError(
                "report.claims_export cannot export adaptive_claims yet: claims.json has no "
                "form for a claim that can only FAIL. Turn one of them off"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def check_schema_version_fields(cls, data: Any) -> Any:
        """Refuse a null sample_join, and any sample_join under schema 3.0."""

        if not isinstance(data, dict):
            return data
        inputs = data.get("inputs")
        if not isinstance(inputs, dict) or "sample_join" not in inputs:
            return data
        if data.get("schema_version") == "3.0":
            raise ValueError(
                "inputs.sample_join was removed in schema 3.0: rollouts join the design "
                "manifest's prompt_id by their item_id"
            )
        if inputs["sample_join"] is None:
            raise ValueError("inputs.sample_join cannot be null")
        return data

    # ORDER MATTERS: this validator comes before neutral_field_names, which reads the
    # rollout key it fills.
    @model_validator(mode="after")
    def apply_schema_version_defaults(self) -> AnalysisPlan:
        """Fill the defaults that differ between schema versions.

        They are set without marking the fields as written (``model_fields_set``), so the
        plan's hash, which covers only what the author wrote, leaves them out, and the report
        lists them among the defaults it applied.
        """

        if self.schema_version != "3.0":
            return self
        object.__setattr__(self.inputs, "sample_join", None)
        if "rollout_key" not in self.design.model_fields_set:
            object.__setattr__(self.design, "rollout_key", list(NEUTRAL_ROLLOUT_KEY))
        return self

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

    @model_validator(mode="after")
    def judge_claims_need_judges(self) -> AnalysisPlan:
        if self.outputs.judge_claims and not self.inputs.judges:
            raise ValueError(
                "outputs.judge_claims decides the claims on each declared judge's verdicts, "
                "but inputs.judges declares no judge. Declare the judges, or set "
                "outputs.judge_claims: false"
            )
        return self

    @model_validator(mode="after")
    def judge_runs_count_content_filter_stops_consistently(self) -> AnalysisPlan:
        # Each judge's run reads the judge's score with the stop reasons it inherits
        # (judge_run_score), which must fit as the primary score must. Without judge_claims
        # no run reads them.
        if not self.outputs.judge_claims:
            return self
        for judge in self.inputs.judges:
            score = judge_run_score(self.inputs.score, judge.score)
            problem = _content_filter_stops_problem(score, self.inputs.estimand)
            if problem is None:
                continue
            inherited = (
                " The judge leaves its content_filter_stops at score, so its run takes "
                f"{score.content_filter_stops} from inputs.score."
                if judge.score.content_filter_stops == "score"
                else ""
            )
            raise ValueError(
                f"outputs.judge_claims runs judge {judge.id!r} on its own score, where "
                f"{problem}.{inherited}"
            )
        return self

    def source_path(self) -> Path | None:
        return self._source_path

    def expanded_comparisons(self) -> list[ComparisonSpec]:
        return [
            *self.comparisons,
            *(comparison for item in self.comparison_sets for comparison in item.expand()),
        ]
