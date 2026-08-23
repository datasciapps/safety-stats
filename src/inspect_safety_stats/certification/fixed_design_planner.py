from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from inspect_safety_stats.certification.config import (
    CertificationClaimSpec,
    CertificationPlan,
    canonical_certification_plan_hash,
    load_certification_plan,
)
from inspect_safety_stats.certification.evidence import normalized_weights
from inspect_safety_stats.certification.planning import (
    certification_sample_size,
    fixed_design_detectable_failure_rates,
    maximum_certifiable_failure_rate,
    minimum_detectable_failure_rate,
    rejection_sample_size,
)
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_groups
from inspect_safety_stats.version import __version__

PlanningDirection = Literal["support", "rejection"]


class PlanningResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FixedDesignClaimPlan(PlanningResultModel):
    claim_id: str
    title: str
    target: dict[str, Any]
    direction: PlanningDirection
    method: Literal["exact_binomial", "weighted_hoeffding"]
    threshold: float
    anticipated_rate: float
    target_power: float
    family: str
    family_alpha: float
    local_alpha: float
    prompt_count: int
    maximum_concentration_variance: float | None = None
    required_equal_rollouts_per_prompt: int
    required_single_claim_total_rollouts: int
    planned_min_rollouts: int
    planned_max_rollouts: int
    planned_total_rollouts: int
    planned_effective_size: float
    planned_detectable_rate: float | None = None
    binding_equal_allocation: bool = False
    within_rollout_limit: bool = True


class FixedDesignPromptAllocation(PlanningResultModel):
    prompt_id: str
    hierarchy: dict[str, str]
    rollouts: int
    existing_expected_rollouts: int | None = None
    maximum_claim_weight: float


class FixedDesignPlanningReport(PlanningResultModel):
    schema_version: Literal["2.0"] = "2.0"
    package_version: str
    analysis: dict[str, Any]
    certification: dict[str, Any]
    settings: dict[str, Any]
    design: dict[str, Any]
    claims: list[FixedDesignClaimPlan]
    allocation: list[FixedDesignPromptAllocation]
    summary: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    _resolved_analysis_plan: dict[str, Any] = PrivateAttr(default_factory=dict)
    _resolved_certification_plan: dict[str, Any] = PrivateAttr(default_factory=dict)

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        from inspect_safety_stats.certification.planning_artifacts import (
            write_fixed_design_planning_report,
        )

        return write_fixed_design_planning_report(self, output, overwrite=overwrite)


@dataclass(frozen=True)
class _ResolvedPlanningClaim:
    result_id: str
    title: str
    spec: CertificationClaimSpec
    target: dict[str, Any]
    prompt_ids: tuple[str, ...]
    weights: np.ndarray


@dataclass(frozen=True)
class _MeanConstraint:
    claim_id: str
    prompt_ids: tuple[str, ...]
    coefficients: np.ndarray
    maximum_variance: float


def _coerce_analysis_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def _coerce_certification_plan(plan: CertificationPlan | str | Path) -> CertificationPlan:
    return plan if isinstance(plan, CertificationPlan) else load_certification_plan(plan)


def _resolved_group_frames(
    analysis: AnalysisPlan, manifest: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    indexed = manifest.set_index("prompt_id", drop=False)
    for group in resolve_groups(analysis, manifest):
        frame = indexed.loc[list(group.prompt_ids)].copy()
        frame["weight"] = [group.weights[prompt_id] for prompt_id in group.prompt_ids]
        frames[group.id] = frame.reset_index(drop=True)
    return frames


def _hierarchy_prefix(analysis: AnalysisPlan, level: str) -> list[str]:
    hierarchy = list(analysis.design.hierarchy)
    if level not in hierarchy or level == "prompt_id":
        raise ValueError(f"hierarchy planning level {level!r} is not a non-prompt level")
    return hierarchy[: hierarchy.index(level) + 1]


def _claim_from_frame(
    claim: CertificationClaimSpec,
    *,
    result_id: str,
    title: str,
    target: dict[str, Any],
    frame: pd.DataFrame,
) -> _ResolvedPlanningClaim:
    weights = frame["weight"].to_numpy(dtype=float, copy=True)
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError(f"claim {result_id!r} has no positive design weight")
    weights /= total
    return _ResolvedPlanningClaim(
        result_id=result_id,
        title=title,
        spec=claim,
        target=target,
        prompt_ids=tuple(frame["prompt_id"].astype(str)),
        weights=weights,
    )


def _expand_claims(
    analysis: AnalysisPlan,
    certification: CertificationPlan,
    manifest: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
) -> list[_ResolvedPlanningClaim]:
    resolved: list[_ResolvedPlanningClaim] = []
    manifest_index = manifest.set_index("prompt_id", drop=False)
    for claim in certification.claims:
        target = claim.target
        if target.cohort not in analysis.cohorts:
            raise ValueError(f"claim {claim.id!r} references unknown cohort {target.cohort!r}")
        base = target.model_dump(mode="json", exclude_none=True)
        if target.type == "prompt_failure_rate":
            assert target.prompt_id is not None
            if target.prompt_id not in manifest_index.index:
                raise ValueError(
                    f"claim {claim.id!r} references absent prompt {target.prompt_id!r}"
                )
            resolved.append(
                _ResolvedPlanningClaim(
                    result_id=claim.id,
                    title=claim.title,
                    spec=claim,
                    target=base,
                    prompt_ids=(target.prompt_id,),
                    weights=np.asarray([1.0]),
                )
            )
            continue

        assert target.group is not None
        group_frame = groups.get(target.group)
        if group_frame is None:
            raise ValueError(f"claim {claim.id!r} references unknown group {target.group!r}")
        if target.type == "group_mean":
            resolved.append(
                _claim_from_frame(
                    claim,
                    result_id=claim.id,
                    title=claim.title,
                    target=base,
                    frame=group_frame,
                )
            )
            continue

        assert target.level is not None
        prefix = _hierarchy_prefix(analysis, target.level)
        missing = [field for field in prefix if field not in group_frame]
        if missing:
            raise ValueError(f"manifest is missing hierarchy fields: {', '.join(missing)}")
        paths: list[dict[str, str]]
        if target.all_paths:
            unique = (
                group_frame[prefix].astype(str).drop_duplicates().sort_values(prefix, kind="stable")
            )
            paths = [
                dict(zip(prefix, values, strict=True))
                for values in unique.itertuples(index=False, name=None)
            ]
        else:
            assert target.path is not None
            if set(target.path) != set(prefix):
                raise ValueError(
                    f"claim {claim.id!r} path must contain exactly: {', '.join(prefix)}"
                )
            paths = [{field: str(target.path[field]) for field in prefix}]
        for path in paths:
            selected = group_frame
            for field, value in path.items():
                selected = selected.loc[selected[field].astype(str) == value]
            if selected.empty:
                raise ValueError(f"claim {claim.id!r} path selects no prompts")
            suffix = sha256_value(path)[:10]
            path_text = " → ".join(path.values())
            expanded_target = {**base, "path": path, "all_paths": False}
            resolved.append(
                _claim_from_frame(
                    claim,
                    result_id=f"{claim.id}:{target.level}:{suffix}",
                    title=f"{claim.title} — {path_text}",
                    target=expanded_target,
                    frame=selected,
                )
            )
    return resolved


def _local_alphas(
    certification: CertificationPlan,
    claims: list[_ResolvedPlanningClaim],
    direction: PlanningDirection,
) -> dict[str, tuple[str, float, float]]:
    allocations: dict[str, tuple[str, float, float]] = {}
    purpose = "simultaneous_support" if direction == "support" else "simultaneous_rejection"
    for family in certification.multiplicity.families:
        if family.purpose != purpose:
            continue
        selected = [
            claim
            for claim in claims
            if getattr(claim.spec, direction).family == family.id
            and (direction == "support" or claim.spec.rejection.enabled)
        ]
        raw = [getattr(claim.spec, direction).allocation_weight for claim in selected]
        weights = normalized_weights(raw, weighted=family.method == "weighted_bonferroni")
        for claim, weight in zip(selected, weights, strict=True):
            allocations[claim.result_id] = (family.id, family.alpha, family.alpha * weight)
    missing = [claim.result_id for claim in claims if claim.result_id not in allocations]
    if missing:
        raise ValueError(
            f"claims have no {direction} multiplicity allocation: {', '.join(missing)}"
        )
    return allocations


def _maximum_variance(
    *, alpha: float, power: float, threshold: float, anticipated_rate: float
) -> float:
    gap = abs(threshold - anticipated_rate)
    denominator = (math.sqrt(math.log(1 / alpha)) + math.sqrt(math.log(1 / (1 - power)))) ** 2
    return 2 * gap**2 / denominator


def _ceil_stable(value: float) -> int:
    return max(1, int(math.ceil(value - 1e-12)))


def _initial_allocation(
    prompt_ids: list[str],
    constraints: list[_MeanConstraint],
    exact_minimums: dict[str, int],
    minimum_rollouts: int,
) -> dict[str, int]:
    allocation = {prompt_id: minimum_rollouts for prompt_id in prompt_ids}
    for prompt_id, required in exact_minimums.items():
        allocation[prompt_id] = max(allocation[prompt_id], required)
    for constraint in constraints:
        weights = np.sqrt(constraint.coefficients)
        for prompt_id, weight in zip(constraint.prompt_ids, weights, strict=True):
            required = _ceil_stable(float(weight) / constraint.maximum_variance)
            allocation[prompt_id] = max(allocation[prompt_id], required)
    return allocation


def _constraint_value(constraint: _MeanConstraint, allocation: dict[str, int]) -> float:
    trials = np.asarray([allocation[prompt_id] for prompt_id in constraint.prompt_ids])
    return float(np.sum(constraint.coefficients / trials))


def _reduce_allocation(
    allocation: dict[str, int],
    constraints: list[_MeanConstraint],
    exact_minimums: dict[str, int],
    minimum_rollouts: int,
) -> None:
    by_prompt: dict[str, list[_MeanConstraint]] = {prompt_id: [] for prompt_id in allocation}
    for constraint in constraints:
        for prompt_id in constraint.prompt_ids:
            by_prompt[prompt_id].append(constraint)
    for prompt_id in sorted(allocation):
        current = allocation[prompt_id]
        required = max(minimum_rollouts, exact_minimums.get(prompt_id, minimum_rollouts))
        for constraint in by_prompt[prompt_id]:
            index = constraint.prompt_ids.index(prompt_id)
            coefficient = float(constraint.coefficients[index])
            total = _constraint_value(constraint, allocation)
            residual = constraint.maximum_variance - (total - coefficient / current)
            if residual <= 0:
                required = current
                break
            required = max(required, _ceil_stable(coefficient / residual))
        allocation[prompt_id] = min(current, required)


def plan_fixed_design(
    *,
    analysis_plan: AnalysisPlan | str | Path,
    certification_plan: CertificationPlan | str | Path,
    anticipated_rate: float,
    power: float = 0.9,
    direction: PlanningDirection = "support",
    minimum_rollouts: int = 1,
    maximum_rollouts: int | None = None,
) -> FixedDesignPlanningReport:
    """Plan rollout counts for a fixed weighted certification design."""

    analysis = _coerce_analysis_plan(analysis_plan)
    certification = _coerce_certification_plan(certification_plan)
    if not 0 <= anticipated_rate <= 1:
        raise ValueError("anticipated_rate must be in [0, 1]")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if (
        isinstance(minimum_rollouts, bool)
        or not isinstance(minimum_rollouts, int)
        or minimum_rollouts <= 0
    ):
        raise ValueError("minimum_rollouts must be a positive integer")
    if maximum_rollouts is not None and (
        isinstance(maximum_rollouts, bool)
        or not isinstance(maximum_rollouts, int)
        or maximum_rollouts < minimum_rollouts
    ):
        raise ValueError("maximum_rollouts must be an integer at least minimum_rollouts")
    if direction not in {"support", "rejection"}:
        raise ValueError("direction must be support or rejection")

    analysis_hash = canonical_plan_hash(analysis)
    required_hash = certification.source.require_plan_sha256
    if required_hash is not None and required_hash.lower() != analysis_hash.lower():
        raise ValueError("analysis plan SHA-256 does not match certification plan")
    manifest = load_manifest(analysis.inputs.design_manifest)
    mean_profiles = {
        cast(str, getattr(claim, direction).independence_profile)
        for claim in certification.claims
        if claim.target.type != "prompt_failure_rate"
        and (direction == "support" or claim.rejection.enabled)
    }
    for profile_id in mean_profiles:
        profile = analysis.design.independence_profiles.get(profile_id)
        if profile is None:
            raise ValueError(
                f"certification planning references unknown independence profile {profile_id!r}"
            )
        if profile.block_by != ["rollout"]:
            raise ValueError(
                "rollout-count optimization requires mean claims to use a profile with "
                "block_by: [rollout]"
            )
    group_frames = _resolved_group_frames(analysis, manifest)
    claims = _expand_claims(analysis, certification, manifest, group_frames)
    allocations = _local_alphas(certification, claims, direction)

    constraints: list[_MeanConstraint] = []
    exact_minimums: dict[str, int] = {}
    requirements: dict[str, tuple[int, int, float | None, str, float, float]] = {}
    for claim in claims:
        threshold = claim.spec.threshold
        if direction == "support" and not anticipated_rate < threshold:
            raise ValueError(
                f"support planning requires anticipated_rate below threshold for {claim.result_id}"
            )
        if direction == "rejection" and not anticipated_rate > threshold:
            raise ValueError(
                "rejection planning requires anticipated_rate above threshold for "
                f"{claim.result_id}"
            )
        family_id, family_alpha, local_alpha = allocations[claim.result_id]
        if claim.spec.target.type == "prompt_failure_rate":
            required = (
                certification_sample_size(
                    true_rate=anticipated_rate,
                    threshold=threshold,
                    alpha=local_alpha,
                    power=power,
                )
                if direction == "support"
                else rejection_sample_size(
                    true_rate=anticipated_rate,
                    threshold=threshold,
                    alpha=local_alpha,
                    power=power,
                )
            )
            exact_minimums[claim.prompt_ids[0]] = max(
                exact_minimums.get(claim.prompt_ids[0], 0), required
            )
            requirements[claim.result_id] = (
                required,
                required,
                None,
                family_id,
                family_alpha,
                local_alpha,
            )
            continue
        maximum_variance = _maximum_variance(
            alpha=local_alpha,
            power=power,
            threshold=threshold,
            anticipated_rate=anticipated_rate,
        )
        equal = _ceil_stable(float(np.sum(np.square(claim.weights))) / maximum_variance)
        single = int(
            np.sum(
                [
                    max(minimum_rollouts, _ceil_stable(float(weight) / maximum_variance))
                    for weight in claim.weights
                ]
            )
        )
        constraints.append(
            _MeanConstraint(
                claim_id=claim.result_id,
                prompt_ids=claim.prompt_ids,
                coefficients=np.square(claim.weights),
                maximum_variance=maximum_variance,
            )
        )
        requirements[claim.result_id] = (
            max(minimum_rollouts, equal),
            single,
            maximum_variance,
            family_id,
            family_alpha,
            local_alpha,
        )

    prompt_ids = sorted(manifest["prompt_id"].astype(str))
    allocation = _initial_allocation(prompt_ids, constraints, exact_minimums, minimum_rollouts)
    _reduce_allocation(allocation, constraints, exact_minimums, minimum_rollouts)
    equal_rollouts = max(item[0] for item in requirements.values())
    max_planned = max(allocation.values())
    within_limits = maximum_rollouts is None or max_planned <= maximum_rollouts
    binding_equal = {
        claim_id for claim_id, item in requirements.items() if item[0] == equal_rollouts
    }

    claim_results: list[FixedDesignClaimPlan] = []
    constraint_index = {item.claim_id: item for item in constraints}
    for claim in claims:
        equal, single_total, claim_maximum_variance, family_id, family_alpha, local_alpha = (
            requirements[claim.result_id]
        )
        planned = np.asarray([allocation[prompt_id] for prompt_id in claim.prompt_ids])
        if claim.spec.target.type == "prompt_failure_rate":
            n = int(planned[0])
            detectable = (
                maximum_certifiable_failure_rate(
                    trials=n,
                    threshold=claim.spec.threshold,
                    alpha=local_alpha,
                    power=power,
                )
                if direction == "support"
                else minimum_detectable_failure_rate(
                    trials=n,
                    threshold=claim.spec.threshold,
                    alpha=local_alpha,
                    power=power,
                )
            )
            effective_size = float(n)
            method: Literal["exact_binomial", "weighted_hoeffding"] = "exact_binomial"
        else:
            rates = fixed_design_detectable_failure_rates(
                planned,
                claim.weights,
                threshold=claim.spec.threshold,
                support_alpha=local_alpha,
                rejection_alpha=local_alpha,
                power=power,
                independence_unit="rollout",
            )
            detectable = (
                rates.maximum_certifiable_failure_rate
                if direction == "support"
                else rates.minimum_detectable_failure_rate
            )
            effective_size = rates.effective_sample_size
            method = "weighted_hoeffding"
            assert claim_maximum_variance is not None
            assert (
                _constraint_value(constraint_index[claim.result_id], allocation)
                <= claim_maximum_variance + 1e-12
            )
        claim_results.append(
            FixedDesignClaimPlan(
                claim_id=claim.result_id,
                title=claim.title,
                target=claim.target,
                direction=direction,
                method=method,
                threshold=claim.spec.threshold,
                anticipated_rate=anticipated_rate,
                target_power=power,
                family=family_id,
                family_alpha=family_alpha,
                local_alpha=local_alpha,
                prompt_count=len(claim.prompt_ids),
                maximum_concentration_variance=claim_maximum_variance,
                required_equal_rollouts_per_prompt=equal,
                required_single_claim_total_rollouts=single_total,
                planned_min_rollouts=int(np.min(planned)),
                planned_max_rollouts=int(np.max(planned)),
                planned_total_rollouts=int(np.sum(planned)),
                planned_effective_size=effective_size,
                planned_detectable_rate=detectable,
                binding_equal_allocation=claim.result_id in binding_equal,
                within_rollout_limit=(
                    maximum_rollouts is None or int(np.max(planned)) <= maximum_rollouts
                ),
            )
        )

    hierarchy = list(analysis.design.hierarchy)
    claim_prompt_ids = {prompt_id for claim in claims for prompt_id in claim.prompt_ids}
    manifest_index = manifest.set_index("prompt_id", drop=False)
    maximum_weights = {prompt_id: 0.0 for prompt_id in prompt_ids}
    for claim in claims:
        for prompt_id, weight in zip(claim.prompt_ids, claim.weights, strict=True):
            maximum_weights[prompt_id] = max(maximum_weights[prompt_id], float(weight))
    allocation_rows = [
        FixedDesignPromptAllocation(
            prompt_id=prompt_id,
            hierarchy={
                field: str(manifest_index.loc[prompt_id, field])
                for field in hierarchy
                if field in manifest_index
            },
            rollouts=allocation[prompt_id],
            existing_expected_rollouts=(
                int(cast(Any, manifest_index.loc[prompt_id, "expected_rollouts"]))
                if "expected_rollouts" in manifest_index
                else None
            ),
            maximum_claim_weight=maximum_weights[prompt_id],
        )
        for prompt_id in prompt_ids
    ]
    report = FixedDesignPlanningReport(
        package_version=__version__,
        analysis={
            "id": analysis.analysis.id,
            "title": analysis.analysis.title,
            "plan_sha256": analysis_hash,
            "manifest": str(analysis.inputs.design_manifest),
        },
        certification={
            **certification.certification.model_dump(mode="json"),
            "plan_sha256": canonical_certification_plan_hash(certification),
        },
        settings={
            "direction": direction,
            "anticipated_rate": anticipated_rate,
            "power": power,
            "independence_profiles": sorted(mean_profiles),
            "method": "weighted_hoeffding",
            "minimum_rollouts": minimum_rollouts,
            "maximum_rollouts": maximum_rollouts,
        },
        design={
            "type": "fixed",
            "hierarchy": hierarchy,
            "prompt_count": len(prompt_ids),
            "claim_prompt_count": len(claim_prompt_ids),
            "hierarchy_counts": {
                field: int(manifest[field].astype(str).nunique())
                for field in hierarchy
                if field in manifest
            },
        },
        claims=claim_results,
        allocation=allocation_rows,
        summary={
            "feasible": within_limits,
            "required_equal_rollouts_per_prompt": equal_rollouts,
            "common_total_rollouts": equal_rollouts * len(prompt_ids),
            "common_claim_total_rollouts": equal_rollouts * len(claim_prompt_ids),
            "adaptive_total_rollouts": int(sum(allocation.values())),
            "adaptive_min_rollouts": min(allocation.values()),
            "adaptive_max_rollouts": max_planned,
            "binding_equal_claims": sorted(binding_equal),
            "claim_count": len(claim_results),
        },
        warnings=(
            []
            if within_limits
            else ["The required allocation exceeds maximum_rollouts for at least one prompt."]
        ),
    )
    report._resolved_analysis_plan = analysis.model_dump(mode="json", by_alias=True)
    report._resolved_certification_plan = certification.model_dump(mode="json")
    return report
