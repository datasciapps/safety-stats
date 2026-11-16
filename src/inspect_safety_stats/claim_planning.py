"""Plan an analysis plan's claims and intervals the way ``analyze`` will compute them.

``analyze`` decides a ``weighted_mean`` claim on a group with direct-mean intervals on the
primary direct-mean interval, and reports such an interval for every selected group, every
hierarchy node and every declared independence profile. Their endpoint alphas, weights and
independence blocks all come from the plan and the design manifest. The planner does not
recompute them: it gives analyze's own functions the rollouts the manifest declares, as if
already observed, and reads the calibration and the blocks off the result. The interval
method is the plan's, and the detectable rates invert that method's interval. Claims decided
any other way are reported as not planned, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import AnalysisPlan, ClaimSpec
from inspect_safety_stats.ingest.manifest import ResolvedGroup, load_manifest, resolve_groups
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.observations import Observation, observation_field
from inspect_safety_stats.stats.independence import (
    BoundedBlocks,
    direct_bounded_blocks,
    validate_profile_fields,
)
from inspect_safety_stats.stats.planning import (
    DirectMeanMethod,
    FixedDesignDetectableRates,
    MethodDetectableRates,
    detectable_failure_rates,
    fixed_design_detectable_failure_rates,
)

if TYPE_CHECKING:
    from inspect_safety_stats.api import _EndpointCalibration

# Fields a block may be formed by before any rollout exists: the manifest's, and the
# observation fields that the manifest or the plan fix.
_PLANNABLE_OBSERVATION_FIELDS = frozenset({"prompt_id", "question_id", "task_id", "cohort_id"})

Calibration = Literal["pointwise", "flat_bonferroni", "frontier_sidak"]


@dataclass(frozen=True)
class EndpointAlpha:
    """The alpha of one endpoint of a direct-mean interval, as analyze calibrates it."""

    endpoint_alpha: float
    family_id: str
    family_alpha: float
    directional_family_alpha: float
    target_count: int
    cohort_count: int
    calibration: Calibration
    frontier_id: str | None = None
    frontier_alpha: float | None = None
    frontier_size: int | None = None


@dataclass(frozen=True)
class PlannedInterval:
    """One direct-mean interval analyze will report: a profile, a cohort and a target.

    ``certification_floor`` is the interval's upper endpoint when every rollout is safe.
    ``acceptable_upper`` is the smallest of the group's planned claims in the cohort, and
    ``resolves`` whether the floor is at or below it, that is whether any outcome can certify
    this target at the group's threshold. Both are ``None`` without such a claim.
    """

    independence_profile: str
    cohort: str
    group: str
    level: str | None
    path: dict[str, str]
    target_id: str
    independent_blocks: int
    effective_sample_size: float
    alpha: EndpointAlpha
    method: DirectMeanMethod
    certification_floor: float
    acceptable_upper: float | None
    resolves: bool | None


@dataclass(frozen=True)
class PlannedClaim:
    claim_id: str
    cohort: str
    group: str
    measure: str
    prompts: int
    total_rollouts: int
    independence_profile: str
    independent_blocks: int | None
    alpha: EndpointAlpha
    acceptable_upper: float
    unacceptable_lower: float
    # Under the plan's interval method, then under each of its reference methods.
    planned: MethodDetectableRates
    references: tuple[MethodDetectableRates, ...]
    # The Hoeffding closed form the planner used before it read the method; None when the
    # threshold cannot be crossed by a failure rate.
    hoeffding_support: FixedDesignDetectableRates | None
    hoeffding_rejection: FixedDesignDetectableRates | None


@dataclass(frozen=True)
class UnplannedClaim:
    claim_id: str
    reason: str


ClaimPlan = PlannedClaim | UnplannedClaim


@dataclass(frozen=True)
class DesignPlan:
    claims: list[ClaimPlan]
    intervals: list[PlannedInterval]
    method: DirectMeanMethod | None


def _unplanned_reason(plan: AnalysisPlan, claim: ClaimSpec) -> str | None:
    """Why ``analyze`` will not decide ``claim`` on a primary direct-mean interval."""

    if claim.comparison is not None:
        return f"comparison claim on {claim.comparison!r}; the planner covers aggregate claims"
    if claim.method == "noisy_valid_test":
        return "decided by the noisy-but-valid test, which the planner does not model"
    calibration = plan.inputs.judge_calibration
    if calibration is not None and calibration.mode == "validated":
        return "decided on the validated judge correction, which the planner does not model"
    if claim.family not in {family.id for family in plan.multiplicity.families}:
        return f"its multiplicity family {claim.family!r} is not declared"
    if claim.cohort not in plan.cohorts:
        return f"its cohort {claim.cohort!r} is not declared"
    measures = {measure.id: measure for measure in plan.outputs.risk_measures}
    measure = measures.get(str(claim.measure))
    if measure is None:
        return f"its measure {claim.measure!r} is not declared"
    if measure.type != "weighted_mean":
        return (
            f"measure {measure.id!r} is {measure.type}, decided on propagated prompt bounds; "
            "the planner covers weighted_mean claims decided on direct-mean intervals"
        )
    spec = plan.outputs.direct_mean_intervals
    if not spec.enabled:
        return (
            "outputs.direct_mean_intervals is disabled, so the claim is decided on propagated "
            "prompt bounds, which the planner does not model"
        )
    if claim.group not in (spec.groups or [group.id for group in plan.groups]):
        return (
            f"group {claim.group!r} has no direct-mean interval, so the claim is decided on "
            "propagated prompt bounds, which the planner does not model"
        )
    return None


def _rollout_counts(frame: pd.DataFrame, rollouts: int | None) -> dict[str, int]:
    if rollouts is not None:
        return {str(prompt_id): rollouts for prompt_id in frame["prompt_id"]}
    if "expected_rollouts" not in frame:
        raise ValueError(
            "the design manifest has no expected_rollouts column; "
            "pass --rollouts for a uniform count"
        )
    counts: dict[str, int] = {}
    for prompt_id, value in zip(frame["prompt_id"], frame["expected_rollouts"], strict=True):
        if isinstance(value, bool) or pd.isna(value) or float(value) != int(value) or value < 1:
            raise ValueError(
                f"expected_rollouts of prompt {prompt_id!r} is {value!r}; "
                "the planner needs a positive whole number for every prompt"
            )
        counts[str(prompt_id)] = int(value)
    return counts


def _planned_observations(
    plan: AnalysisPlan, frame: pd.DataFrame, counts: dict[str, int]
) -> list[Observation]:
    """The rollouts the manifest declares, for every cohort, all safe.

    Their outcomes do not matter: calibration and blocks depend on the design only.
    """

    observations: list[Observation] = []
    for cohort_id in sorted(plan.cohorts):
        for row in frame.itertuples(index=False):
            prompt_id = str(row.prompt_id)
            for repeat in range(1, counts[prompt_id] + 1):
                observation_id = f"planned:{cohort_id}:{prompt_id}:{repeat}"
                observations.append(
                    Observation(
                        observation_id=observation_id,
                        source="planned",
                        run_key=f"planned:{cohort_id}",
                        rollout_id=observation_id,
                        item_id=prompt_id,
                        repeat=repeat,
                        prompt_id=prompt_id,
                        question_id=str(row.question_id),
                        task_id=str(row.task_id),
                        cohort_id=cohort_id,
                        score_name="planned",
                        failure=False,
                        outcome_status="classified_safe",
                        model="planned",
                        configuration_fingerprint="planned",
                        collection_order=len(observations),
                    )
                )
    return observations


def _check_plannable_profiles(plan: AnalysisPlan, frame: pd.DataFrame, profiles: list[str]) -> None:
    for profile_id in profiles:
        for field in plan.design.independence_profiles[profile_id].block_by:
            if field == "rollout" or field in frame.columns:
                continue
            if observation_field(field) not in _PLANNABLE_OBSERVATION_FIELDS:
                raise ValueError(
                    f"independence profile {profile_id!r} blocks by {field!r}, which the logs "
                    "supply and the design manifest does not; the planner resolves blocks "
                    "from the manifest only"
                )


def _endpoint_alpha(
    calibration: _EndpointCalibration,
    *,
    family_id: str,
    family_alpha: float,
    directional: float,
    target_count: int,
    cohort_count: int,
) -> EndpointAlpha:
    method = calibration.method
    if method not in ("pointwise", "flat_bonferroni", "frontier_sidak"):
        raise RuntimeError(f"analyze calibrated an interval by an unknown method {method!r}")
    return EndpointAlpha(
        endpoint_alpha=calibration.endpoint_alpha,
        family_id=family_id,
        family_alpha=family_alpha,
        directional_family_alpha=directional,
        target_count=target_count,
        cohort_count=cohort_count,
        calibration=method,  # type: ignore[arg-type]  # checked above
        frontier_id=calibration.frontier_id,
        frontier_alpha=calibration.frontier_alpha,
        frontier_size=calibration.frontier_size,
    )


def plan_design(
    plan: AnalysisPlan,
    *,
    power: float,
    manifest: str | Path | None = None,
    rollouts: int | None = None,
) -> DesignPlan:
    """Detectable rates for every claim of ``plan``, and every interval analyze reports."""

    # Private helpers of the analysis, so that nodes, calibration and blocks are analyze's
    # own; a copy here could drift from it.
    from inspect_safety_stats.api import (
        _block_index,
        _calibration_key,
        _coverage_floors,
        _direct_mean_context,
        _hierarchy_nodes,
        _hierarchy_region_target_id,
        _independence,
    )

    if plan.design.type != "fixed":
        raise ValueError(
            "the plan declares a sampled design; plan fixed-design plans fixed designs only"
        )
    frame = load_manifest(manifest if manifest is not None else plan.inputs.design_manifest)
    counts = _rollout_counts(frame, rollouts)
    groups = {group.id: group for group in resolve_groups(plan, frame)}
    spec = plan.outputs.direct_mean_intervals
    reasons = {claim.id: _unplanned_reason(plan, claim) for claim in plan.claims}
    family = next((item for item in plan.multiplicity.families if item.id == spec.family), None)
    if not spec.enabled or family is None:
        if spec.enabled:
            reasons = {
                claim_id: reason
                or f"outputs.direct_mean_intervals names the undeclared family {spec.family!r}"
                for claim_id, reason in reasons.items()
            }
        return DesignPlan(
            claims=[UnplannedClaim(claim.id, str(reasons[claim.id])) for claim in plan.claims],
            intervals=[],
            method=None,
        )
    independence = _independence(spec)
    profile_ids = [independence.primary, *independence.sensitivities]
    _check_plannable_profiles(plan, frame, profile_ids)
    validate_profile_fields(plan.design.independence_profiles, frame)
    data = NormalizedData(
        observations=_planned_observations(plan, frame, counts),
        manifest=frame,
        groups=list(groups.values()),
        diagnostics=[],
        cohorts=[],
        source_hashes={},
    )
    nodes = _hierarchy_nodes(plan, data)
    context = _direct_mean_context(plan, data, nodes=nodes)
    directional, _, _ = _coverage_floors(family.alpha, spec.coverage_mode)
    selected = sorted(spec.groups or [group.id for group in data.groups])
    targets: list[tuple[ResolvedGroup, str | None, dict[str, str], tuple[str, ...], np.ndarray]]
    targets = [
        (
            groups[group_id],
            None,
            {},
            tuple(groups[group_id].prompt_ids),
            np.asarray([groups[group_id].weights[item] for item in groups[group_id].prompt_ids]),
        )
        for group_id in selected
    ]
    targets += [
        (groups[node.group_id], node.level, node.path, node.prompt_ids, np.asarray(node.weights))
        for node in nodes
    ]

    def alpha_of(cohort_id: str, target_id: str, profile_id: str) -> EndpointAlpha:
        return _endpoint_alpha(
            context.calibrations[_calibration_key(cohort_id, target_id, profile_id)],
            family_id=family.id,
            family_alpha=family.alpha,
            directional=directional,
            target_count=len(targets),
            cohort_count=len(plan.cohorts),
        )

    def blocks_of(
        cohort_id: str, prompt_ids: tuple[str, ...], weights: np.ndarray, profile: str
    ) -> BoundedBlocks:
        return direct_bounded_blocks(
            data.observations,
            data.manifest,
            plan.design.independence_profiles[profile],
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            prompt_weights=weights,
            index=_block_index(data),
        )

    method: DirectMeanMethod = spec.method
    claims: list[ClaimPlan] = []
    strictest: dict[tuple[str, str], float] = {}
    for claim in plan.claims:
        reason = reasons[claim.id]
        if reason is not None:
            claims.append(UnplannedClaim(claim.id, reason))
            continue
        cohort_id, group = str(claim.cohort), groups[str(claim.group)]
        key = (cohort_id, group.id)
        strictest[key] = min(strictest.get(key, 1.0), claim.acceptable_upper)
        prompt_ids = tuple(group.prompt_ids)
        weights = np.asarray([group.weights[item] for item in prompt_ids], dtype=float)
        target_id = _hierarchy_region_target_id(group.id, None, {})
        blocks = blocks_of(cohort_id, prompt_ids, weights, independence.primary)
        alpha = alpha_of(cohort_id, target_id, independence.primary)
        # A threshold at or beyond 0 or 1 can never be crossed by a failure rate.
        acceptable = claim.acceptable_upper if 0 < claim.acceptable_upper < 1 else None
        unacceptable = claim.unacceptable_lower if 0 < claim.unacceptable_lower < 1 else None
        planned, *references = [
            detectable_failure_rates(
                blocks.weights,
                method=item,
                alpha=alpha.endpoint_alpha,
                power=power,
                acceptable_upper=acceptable,
                unacceptable_lower=unacceptable,
            )
            for item in [method, *spec.reference_methods]
        ]
        block_by = plan.design.independence_profiles[independence.primary].block_by
        rollout_profile = block_by == ["rollout"]
        hoeffding: dict[str, FixedDesignDetectableRates | None] = {}
        for side, threshold in (("support", acceptable), ("rejection", unacceptable)):
            hoeffding[side] = (
                None
                if threshold is None
                else fixed_design_detectable_failure_rates(
                    (
                        [counts[item] for item in prompt_ids]
                        if rollout_profile
                        else [1] * blocks.count
                    ),
                    weights if rollout_profile else blocks.weights,
                    threshold=threshold,
                    support_alpha=alpha.endpoint_alpha,
                    rejection_alpha=alpha.endpoint_alpha,
                    power=power,
                    independence_unit="rollout" if rollout_profile else "prompt_cluster",
                )
            )
        claims.append(
            PlannedClaim(
                claim_id=claim.id,
                cohort=cohort_id,
                group=group.id,
                measure=str(claim.measure),
                prompts=len(prompt_ids),
                total_rollouts=sum(counts[item] for item in prompt_ids),
                independence_profile=independence.primary,
                independent_blocks=None if rollout_profile else blocks.count,
                alpha=alpha,
                acceptable_upper=claim.acceptable_upper,
                unacceptable_lower=claim.unacceptable_lower,
                planned=planned,
                references=tuple(references),
                hoeffding_support=hoeffding["support"],
                hoeffding_rejection=hoeffding["rejection"],
            )
        )

    intervals: list[PlannedInterval] = []
    for profile_id in profile_ids:
        for cohort_id in sorted(plan.cohorts):
            for group, level, path, prompt_ids, weights in targets:
                target_id = _hierarchy_region_target_id(group.id, level, path)
                blocks = blocks_of(cohort_id, prompt_ids, weights, profile_id)
                alpha = alpha_of(cohort_id, target_id, profile_id)
                floor = detectable_failure_rates(
                    blocks.weights, method=method, alpha=alpha.endpoint_alpha, power=power
                ).certification_floor
                threshold = strictest.get((cohort_id, group.id))
                intervals.append(
                    PlannedInterval(
                        independence_profile=profile_id,
                        cohort=cohort_id,
                        group=group.id,
                        level=level,
                        path=dict(path),
                        target_id=target_id,
                        independent_blocks=blocks.count,
                        effective_sample_size=1 / float(np.sum(np.square(blocks.weights))),
                        alpha=alpha,
                        method=method,
                        certification_floor=floor,
                        acceptable_upper=threshold,
                        resolves=None if threshold is None else floor <= threshold,
                    )
                )
    return DesignPlan(claims=claims, intervals=intervals, method=method)


def plan_claims(
    plan: AnalysisPlan,
    *,
    power: float,
    manifest: str | Path | None = None,
    rollouts: int | None = None,
) -> list[ClaimPlan]:
    """Detectable rates for every claim of ``plan``, in the plan's order."""

    return plan_design(plan, power=power, manifest=manifest, rollouts=rollouts).claims
