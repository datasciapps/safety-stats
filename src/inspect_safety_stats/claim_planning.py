"""Plan every claim of an analysis plan the way ``analyze`` will decide it.

``analyze`` decides a ``weighted_mean`` claim on a group with direct-mean intervals on the
primary direct-mean interval. That interval's endpoint alpha, weights, hierarchy nodes and
independence blocks all come from the plan, so the planner reads them from the same plan
rather than from command-line options that can drift from it. Claims decided any other way
are reported as not planned, with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import AnalysisPlan, ClaimSpec
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_groups
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.stats.planning import (
    FixedDesignDetectableRates,
    fixed_design_detectable_failure_rates,
)


@dataclass(frozen=True)
class EndpointAlpha:
    """The alpha of one endpoint of a direct-mean interval, and how it was derived."""

    endpoint_alpha: float
    family_id: str
    family_alpha: float
    directional_family_alpha: float
    target_count: int
    cohort_count: int
    calibration: Literal["pointwise", "flat_bonferroni", "frontier_sidak"]


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
    # The rates are guaranteed for a Hoeffding test; analyze decides on this method.
    primary_method: str
    alpha: EndpointAlpha
    acceptable_upper: float
    unacceptable_lower: float
    support: FixedDesignDetectableRates | None
    rejection: FixedDesignDetectableRates | None


@dataclass(frozen=True)
class UnplannedClaim:
    claim_id: str
    reason: str


ClaimPlan = PlannedClaim | UnplannedClaim


def _endpoint_alpha(plan: AnalysisPlan, data: NormalizedData) -> EndpointAlpha:
    """The endpoint alpha ``analyze`` gives every primary direct-mean interval.

    Under ``frontier_sidak`` the analysis spends at least the flat Bonferroni share, so the
    share returned here is a conservative stand-in for it.
    """

    # Private helpers of the analysis, so that targets are counted exactly as analyze counts
    # them; a copy here could drift from it.
    from inspect_safety_stats.api import _coverage_floors, _hierarchy_nodes

    spec = plan.outputs.direct_mean_intervals
    family = next((item for item in plan.multiplicity.families if item.id == spec.family), None)
    if family is None:
        raise ValueError(
            f"outputs.direct_mean_intervals names the undeclared multiplicity family "
            f"{spec.family!r}"
        )
    directional, _, _ = _coverage_floors(family.alpha, spec.coverage_mode)
    selected = spec.groups or [group.id for group in data.groups]
    # analyze's targets: each selected group, each hierarchy node, and each selected group's
    # direct worst_of_k interval, which shares the family.
    worst_of_k_direct = sum(1 for measure in plan.outputs.risk_measures if measure.direct_interval)
    target_count = (
        len(selected) * (1 + worst_of_k_direct) + len(_hierarchy_nodes(plan, data))
    )
    cohort_count = len(plan.cohorts)
    calibration: Literal["pointwise", "flat_bonferroni", "frontier_sidak"]
    if not spec.simultaneous:
        endpoint, calibration = directional, "pointwise"
    else:
        endpoint = directional / target_count / cohort_count
        calibration = (
            "frontier_sidak"
            if spec.hierarchy_calibration == "frontier_sidak" and spec.hierarchy_nodes
            else "flat_bonferroni"
        )
    return EndpointAlpha(
        endpoint_alpha=endpoint,
        family_id=family.id,
        family_alpha=family.alpha,
        directional_family_alpha=directional,
        target_count=target_count,
        cohort_count=cohort_count,
        calibration=calibration,
    )


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
    if measure.direct_interval:
        return (
            f"measure {measure.id!r} is worst_of_k with a direct interval, which the planner "
            "does not model; the planner covers weighted_mean claims"
        )
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


def _block_weights(
    frame: pd.DataFrame, weights: np.ndarray, block_by: list[str]
) -> np.ndarray:
    """Total design weight of each independent block of manifest rows."""

    unknown = [column for column in block_by if column not in frame]
    if unknown:
        raise ValueError(
            f"independence blocks by {block_by} name fields absent from the design manifest "
            f"({', '.join(unknown)}); the planner resolves blocks from the manifest only"
        )
    missing = [column for column in block_by if frame[column].isna().any()]
    if missing:
        raise ValueError(
            f"independence block column(s) {', '.join(missing)} have missing values"
        )
    keys = frame[block_by].astype(str).agg("\x1f".join, axis=1)
    totals = pd.Series(weights, index=frame.index).groupby(keys.to_numpy(), sort=True).sum()
    return totals.to_numpy(dtype=float)


def plan_claims(
    plan: AnalysisPlan,
    *,
    power: float,
    manifest: str | Path | None = None,
    rollouts: int | None = None,
) -> list[ClaimPlan]:
    """Detectable rates for every claim of ``plan``, in the plan's order."""

    if plan.design.type != "fixed":
        raise ValueError(
            "the plan declares a sampled design; plan fixed-design plans fixed designs only"
        )
    frame = load_manifest(manifest if manifest is not None else plan.inputs.design_manifest)
    if rollouts is None and "expected_rollouts" not in frame:
        raise ValueError(
            "the design manifest has no expected_rollouts column; "
            "pass --rollouts for a uniform count"
        )
    groups = {group.id: group for group in resolve_groups(plan, frame)}
    data = NormalizedData(
        observations=[],
        manifest=frame,
        groups=list(groups.values()),
        diagnostics=[],
        cohorts=[],
        source_hashes={},
    )
    spec = plan.outputs.direct_mean_intervals
    alpha = _endpoint_alpha(plan, data) if spec.enabled else None
    rows = frame.set_index("prompt_id", drop=False)
    results: list[ClaimPlan] = []
    for claim in plan.claims:
        reason = _unplanned_reason(plan, claim)
        if reason is not None or alpha is None or spec.independence is None:
            results.append(UnplannedClaim(claim.id, reason or "no direct-mean interval"))
            continue
        group = groups[str(claim.group)]
        prompt_ids = list(group.prompt_ids)
        weights = np.asarray([group.weights[item] for item in prompt_ids], dtype=float)
        selected = rows.loc[prompt_ids]
        trials = (
            [rollouts] * len(prompt_ids)
            if rollouts is not None
            else [int(value) for value in selected["expected_rollouts"]]
        )
        profile_id = spec.independence.primary
        block_by = plan.design.independence_profiles[profile_id].block_by
        unit: Literal["rollout", "prompt_cluster"] = "rollout"
        unit_trials, unit_weights, blocks = trials, weights, None
        if block_by != ["rollout"]:
            # Within a block any dependence is allowed, so a block counts as one bounded
            # contribution with its total weight, whatever its rollout count.
            unit = "prompt_cluster"
            unit_weights = _block_weights(selected.reset_index(drop=True), weights, block_by)
            unit_trials = [1] * len(unit_weights)
            blocks = len(unit_weights)
        planned: dict[str, FixedDesignDetectableRates | None] = {}
        for side, threshold in (
            ("support", claim.acceptable_upper),
            ("rejection", claim.unacceptable_lower),
        ):
            # A threshold at or beyond 0 or 1 can never be crossed by a failure rate.
            planned[side] = (
                fixed_design_detectable_failure_rates(
                    unit_trials,
                    unit_weights,
                    threshold=threshold,
                    support_alpha=alpha.endpoint_alpha,
                    rejection_alpha=alpha.endpoint_alpha,
                    power=power,
                    independence_unit=unit,
                )
                if 0 < threshold < 1
                else None
            )
        results.append(
            PlannedClaim(
                claim_id=claim.id,
                cohort=str(claim.cohort),
                group=group.id,
                measure=str(claim.measure),
                prompts=len(prompt_ids),
                total_rollouts=sum(trials),
                independence_profile=profile_id,
                independent_blocks=blocks,
                primary_method=spec.method,
                alpha=alpha,
                acceptable_upper=claim.acceptable_upper,
                unacceptable_lower=claim.unacceptable_lower,
                support=planned["support"],
                rejection=planned["rejection"],
            )
        )
    return results
