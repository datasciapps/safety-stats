from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from inspect_safety_stats.certification.config import (
    CertificationClaimSpec,
    CertificationPlan,
    DirectionSpec,
    SensitivitySpec,
    canonical_certification_plan_hash,
    load_certification_plan,
)
from inspect_safety_stats.certification.evidence import (
    adjusted_bonferroni_p_value,
    block_mean_bounds,
    block_mean_rejection_p_value,
    block_mean_support_p_value,
    exact_rejection_p_value,
    exact_support_p_value,
    exact_threshold_bounds,
    normalized_weights,
)
from inspect_safety_stats.certification.planning import (
    maximum_certifiable_failure_rate,
    minimum_detectable_failure_rate,
)
from inspect_safety_stats.certification.schema import (
    CertificationClaimResult,
    CertificationReport,
    DetectabilityResult,
    DirectionalEvidence,
    EvidenceStatus,
    SensitivityResult,
)
from inspect_safety_stats.certification.sources import (
    EmpiricalCertificationSource,
    load_empirical_source,
)
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.models import IndependenceProfileSpec
from inspect_safety_stats.schema.reports import PromptEstimate
from inspect_safety_stats.stats.independence import BoundedBlocks, direct_bounded_blocks
from inspect_safety_stats.version import __version__


@dataclass(frozen=True)
class _ResolvedClaim:
    result_id: str
    title: str
    spec: CertificationClaimSpec
    target: dict[str, Any]


@dataclass(frozen=True)
class _TargetData:
    estimates: tuple[PromptEstimate, ...]
    weights: np.ndarray
    source: EmpiricalCertificationSource

    @property
    def failures(self) -> np.ndarray:
        return np.asarray([item.n_failure for item in self.estimates], dtype=int)

    @property
    def unclassified(self) -> np.ndarray:
        return np.asarray([item.n_unclassified for item in self.estimates], dtype=int)

    @property
    def trials(self) -> np.ndarray:
        return np.asarray([item.n_total for item in self.estimates], dtype=int)

    @property
    def estimate(self) -> float | None:
        if any(item.p_hat is None for item in self.estimates):
            return None
        return float(
            np.dot(
                self.weights,
                np.asarray([cast(float, item.p_hat) for item in self.estimates]),
            )
        )

    def blocks(self, profile_id: str) -> tuple[BoundedBlocks, IndependenceProfileSpec]:
        if not self.source.observations:
            raise ValueError(
                "mean certification with named independence profiles requires observations.csv"
            )
        raw_profiles = self.source.report.design.get("independence_profiles", {})
        if profile_id not in raw_profiles:
            raise ValueError(
                f"certification references unknown source independence profile {profile_id!r}"
            )
        profile = IndependenceProfileSpec.model_validate(raw_profiles[profile_id])
        prompt_ids = [item.prompt_id for item in self.estimates]
        manifest = self.source.design.loc[
            self.source.design["prompt_id"].astype(str).isin(prompt_ids)
        ].drop_duplicates("prompt_id")
        return (
            direct_bounded_blocks(
                self.source.observations,
                manifest,
                profile,
                cohort_id=self.estimates[0].cohort_id,
                prompt_ids=prompt_ids,
                prompt_weights=self.weights,
            ),
            profile,
        )


def _coerce_plan(plan: CertificationPlan | str | Path) -> CertificationPlan:
    return plan if isinstance(plan, CertificationPlan) else load_certification_plan(plan)


def _path_prefix(source: EmpiricalCertificationSource, level: str) -> list[str]:
    hierarchy = [str(item) for item in source.report.design.get("hierarchy", [])]
    if level not in hierarchy or level == "prompt_id":
        raise ValueError(f"hierarchy certification level {level!r} is not a non-prompt level")
    return hierarchy[: hierarchy.index(level) + 1]


def _group_design(source: EmpiricalCertificationSource, group: str) -> pd.DataFrame:
    frame = source.design.loc[source.design["group_id"].astype(str) == group].copy()
    if frame.empty:
        raise ValueError(f"certification group {group!r} is absent from the resolved design")
    if frame["prompt_id"].duplicated().any():
        raise ValueError(f"certification group {group!r} contains duplicate prompts")
    return frame


def _expand_claims(
    plan: CertificationPlan, source: EmpiricalCertificationSource
) -> list[_ResolvedClaim]:
    resolved: list[_ResolvedClaim] = []
    for claim in plan.claims:
        target = claim.target
        base = target.model_dump(mode="json", exclude_none=True)
        if target.type != "hierarchy_mean" or not target.all_paths:
            if target.type == "hierarchy_mean":
                assert target.group is not None and target.level is not None
                prefix = _path_prefix(source, target.level)
                assert target.path is not None
                if set(target.path) != set(prefix):
                    raise ValueError(
                        f"claim {claim.id!r} path must contain exactly: {', '.join(prefix)}"
                    )
                base["path"] = {field: str(target.path[field]) for field in prefix}
            resolved.append(_ResolvedClaim(claim.id, claim.title, claim, base))
            continue

        assert target.group is not None and target.level is not None
        frame = _group_design(source, target.group)
        prefix = _path_prefix(source, target.level)
        missing = [field for field in prefix if field not in frame]
        if missing:
            raise ValueError(f"resolved design is missing hierarchy fields: {', '.join(missing)}")
        paths = frame[prefix].astype(str).drop_duplicates().sort_values(prefix)
        for values in paths.itertuples(index=False, name=None):
            path = dict(zip(prefix, values, strict=True))
            suffix = sha256_value(path)[:10]
            expanded_target = {**base, "all_paths": False, "path": path}
            resolved.append(
                _ResolvedClaim(
                    result_id=f"{claim.id}:{target.level}:{suffix}",
                    title=f"{claim.title} — {' → '.join(path.values())}",
                    spec=claim,
                    target=expanded_target,
                )
            )
    ids = [item.result_id for item in resolved]
    if len(set(ids)) != len(ids):
        raise ValueError("expanded certification claim ids are not unique")
    return resolved


def _family_allocations(
    plan: CertificationPlan, claims: list[_ResolvedClaim]
) -> dict[tuple[str, Literal["support", "rejection"]], tuple[float, float, float]]:
    allocations: dict[tuple[str, Literal["support", "rejection"]], tuple[float, float, float]] = {}
    families = {item.id: item for item in plan.multiplicity.families}
    for direction in ("support", "rejection"):
        family_ids = sorted(
            {
                getattr(item.spec, direction).family
                for item in claims
                if direction == "support" or item.spec.rejection.enabled
            }
        )
        for family_id in family_ids:
            family = families[family_id]
            selected = [
                item
                for item in claims
                if getattr(item.spec, direction).family == family_id
                and (direction == "support" or item.spec.rejection.enabled)
            ]
            raw_weights = [getattr(item.spec, direction).allocation_weight for item in selected]
            weights = normalized_weights(
                raw_weights, weighted=family.method == "weighted_bonferroni"
            )
            for claim, weight in zip(selected, weights, strict=True):
                allocations[(claim.result_id, direction)] = (
                    family.alpha,
                    family.alpha * weight,
                    weight,
                )
    return allocations


def _target_data(resolved: _ResolvedClaim, source: EmpiricalCertificationSource) -> _TargetData:
    target = resolved.target
    cohort = str(target["cohort"])
    estimate_index = {
        (item.cohort_id, item.prompt_id): item for item in source.report.prompt_estimates
    }
    if target["type"] == "prompt_failure_rate":
        prompt_id = str(target["prompt_id"])
        estimate = estimate_index.get((cohort, prompt_id))
        if estimate is None:
            raise ValueError(
                f"claim {resolved.result_id!r} references absent prompt {cohort}:{prompt_id}"
            )
        return _TargetData((estimate,), np.asarray([1.0]), source)

    group = str(target["group"])
    frame = _group_design(source, group)
    if target["type"] == "hierarchy_mean":
        path = cast(dict[str, str], target["path"])
        for field, value in path.items():
            if field not in frame:
                raise ValueError(f"resolved design has no hierarchy field {field!r}")
            frame = frame.loc[frame[field].astype(str) == str(value)]
        if frame.empty:
            raise ValueError(f"claim {resolved.result_id!r} hierarchy path selects no prompts")
    prompt_ids = tuple(frame["prompt_id"].astype(str))
    estimates: list[PromptEstimate] = []
    for prompt_id in prompt_ids:
        item = estimate_index.get((cohort, prompt_id))
        if item is None:
            raise ValueError(
                f"claim {resolved.result_id!r} is missing prompt estimate {cohort}:{prompt_id}"
            )
        estimates.append(item)
    weights = frame["weight"].to_numpy(dtype=float, copy=True)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise ValueError(f"claim {resolved.result_id!r} has no positive design weight")
    weights /= total_weight
    return _TargetData(tuple(estimates), weights, source)


def _assumptions(method: str, independence_profile: str | None) -> list[str]:
    if method == "clopper_pearson":
        return [
            "IID Bernoulli rollouts within the fixed prompt.",
            "The rollout process and scoring rule are stable across attempted trials.",
        ]
    assumptions = [
        f"Blocks in the named {independence_profile!r} profile are mutually independent.",
        "Arbitrary dependence is allowed within a declared block.",
        "The block partition and design weights are fixed independently of outcomes.",
    ]
    if method == "weighted_kl_chernoff":
        assumptions.append(
            "Bernoulli-KL inversion is used for equal atomic weights and a "
            "maximum-increment weighted Chernoff bound otherwise."
        )
    return assumptions


def _directional_evidence(
    *,
    direction: Literal["support", "rejection"],
    spec: DirectionSpec,
    threshold: float,
    family_alpha: float,
    local_alpha: float,
    normalized_weight: float,
    raw_p_value: float,
    bound: float,
) -> DirectionalEvidence:
    supports = bound <= threshold if direction == "support" else bound > threshold
    margin = threshold - bound if direction == "support" else bound - threshold
    return DirectionalEvidence(
        direction=direction,
        family=spec.family,
        family_alpha=family_alpha,
        local_alpha=local_alpha,
        allocation_weight=normalized_weight,
        raw_p_value=raw_p_value,
        adjusted_p_value=adjusted_bonferroni_p_value(raw_p_value, normalized_weight),
        bound=bound,
        threshold=threshold,
        margin=margin,
        crosses_threshold=supports and raw_p_value <= local_alpha,
        method=spec.method,
        independence_profile=spec.independence_profile,
        assumptions=_assumptions(spec.method, spec.independence_profile),
    )


def _sensitivity(
    spec: SensitivitySpec,
    data: _TargetData,
    *,
    threshold: float,
    support_alpha: float,
    rejection_alpha: float,
) -> SensitivityResult:
    blocks, _ = data.blocks(spec.independence_profile)
    bounds = block_mean_bounds(
        spec.method,
        blocks,
        support_alpha=support_alpha,
        rejection_alpha=rejection_alpha,
        independence_profile=spec.independence_profile,
    )
    support_p = block_mean_support_p_value(
        spec.method,
        blocks,
        threshold=threshold,
        independence_profile=spec.independence_profile,
    )
    rejection_p = block_mean_rejection_p_value(
        spec.method,
        blocks,
        threshold=threshold,
        independence_profile=spec.independence_profile,
    )
    if bounds.upper <= threshold and support_p <= support_alpha:
        evidence: EvidenceStatus = "SUPPORTED"
    elif bounds.lower > threshold and rejection_p <= rejection_alpha:
        evidence = "REJECTED"
    else:
        evidence = "INCONCLUSIVE"
    return SensitivityResult(
        method=spec.method,
        independence_profile=spec.independence_profile,
        lower=bounds.lower,
        upper=bounds.upper,
        support_raw_p_value=support_p,
        rejection_raw_p_value=rejection_p,
        evidence=evidence,
    )


def _detectability(
    spec: CertificationClaimSpec,
    data: _TargetData,
    *,
    threshold: float,
    support_alpha: float,
    rejection_alpha: float,
    power: float | None,
) -> list[DetectabilityResult]:
    if power is None:
        return []
    if spec.target.type == "prompt_failure_rate":
        trials = int(data.trials[0])
        maximum_rate = maximum_certifiable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=support_alpha,
            power=power,
        )
        minimum_rate = (
            minimum_detectable_failure_rate(
                trials=trials,
                threshold=threshold,
                alpha=rejection_alpha,
                power=power,
            )
            if spec.rejection.enabled
            else None
        )
        return [
            DetectabilityResult(
                method="exact_binomial",
                independence_profile=None,
                target_power=power,
                maximum_certifiable_failure_rate=maximum_rate,
                minimum_detectable_failure_rate=minimum_rate,
                support_gap=threshold - maximum_rate if maximum_rate is not None else None,
                rejection_gap=minimum_rate - threshold if minimum_rate is not None else None,
                effective_sample_size=float(trials),
                guarantee="exact",
                assumptions=_assumptions("clopper_pearson", None),
            )
        ]

    profile_id = cast(str, spec.support.independence_profile)
    blocks, _ = data.blocks(profile_id)
    concentration_variance = float(np.sum(np.square(blocks.weights)))
    power_radius = math.sqrt(0.5 * concentration_variance * math.log(1 / (1 - power)))
    support_gap = (
        math.sqrt(0.5 * concentration_variance * math.log(1 / support_alpha)) + power_radius
    )
    rejection_gap = (
        math.sqrt(0.5 * concentration_variance * math.log(1 / rejection_alpha)) + power_radius
    )
    maximum_rate = threshold - support_gap
    minimum_rate = threshold + rejection_gap
    return [
        DetectabilityResult(
            method="weighted_hoeffding_guarantee",
            independence_profile=profile_id,
            target_power=power,
            maximum_certifiable_failure_rate=(
                max(0.0, maximum_rate) if maximum_rate >= 0 else None
            ),
            minimum_detectable_failure_rate=(
                min(1.0, minimum_rate) if spec.rejection.enabled and minimum_rate <= 1 else None
            ),
            support_gap=support_gap if maximum_rate >= 0 else None,
            rejection_gap=(rejection_gap if spec.rejection.enabled and minimum_rate <= 1 else None),
            effective_sample_size=1 / concentration_variance,
            guarantee="conservative",
            assumptions=_assumptions("weighted_hoeffding", profile_id),
        )
    ]


def _claim_result(
    resolved: _ResolvedClaim,
    source: EmpiricalCertificationSource,
    allocations: dict[tuple[str, Literal["support", "rejection"]], tuple[float, float, float]],
    *,
    detectable_power: float | None,
) -> CertificationClaimResult:
    data = _target_data(resolved, source)
    spec = resolved.spec
    threshold = spec.threshold
    support_family_alpha, support_alpha, support_weight = allocations[
        (resolved.result_id, "support")
    ]
    rejection_allocation = allocations.get((resolved.result_id, "rejection"))
    rejection_alpha = rejection_allocation[1] if rejection_allocation else 0.5

    if spec.target.type == "prompt_failure_rate":
        item = data.estimates[0]
        lower, upper = exact_threshold_bounds(
            item.n_failure,
            item.n_unclassified,
            item.n_total,
            support_alpha=support_alpha,
            rejection_alpha=rejection_alpha,
        )
        support_p = exact_support_p_value(
            item.n_failure + item.n_unclassified, item.n_total, threshold
        )
        rejection_p = exact_rejection_p_value(item.n_failure, item.n_total, threshold)
        completion_lower = item.n_failure / item.n_total
        completion_upper = (item.n_failure + item.n_unclassified) / item.n_total
    else:
        support_profile = cast(str, spec.support.independence_profile)
        support_blocks, _ = data.blocks(support_profile)
        support_bounds = block_mean_bounds(
            spec.support.method,
            support_blocks,
            support_alpha=support_alpha,
            rejection_alpha=0.5,
            independence_profile=support_profile,
        )
        support_p = block_mean_support_p_value(
            spec.support.method,
            support_blocks,
            threshold=threshold,
            independence_profile=support_profile,
        )
        upper = support_bounds.upper
        completion_lower = support_bounds.completion_estimate_lower
        completion_upper = support_bounds.completion_estimate_upper
        if spec.rejection.enabled:
            rejection_profile = cast(str, spec.rejection.independence_profile)
            rejection_blocks, _ = data.blocks(rejection_profile)
            rejection_bounds = block_mean_bounds(
                spec.rejection.method,
                rejection_blocks,
                support_alpha=0.5,
                rejection_alpha=rejection_alpha,
                independence_profile=rejection_profile,
            )
            rejection_p = block_mean_rejection_p_value(
                spec.rejection.method,
                rejection_blocks,
                threshold=threshold,
                independence_profile=rejection_profile,
            )
            lower = rejection_bounds.lower
        else:
            lower = support_bounds.lower
            rejection_p = 1.0

    support = _directional_evidence(
        direction="support",
        spec=spec.support,
        threshold=threshold,
        family_alpha=support_family_alpha,
        local_alpha=support_alpha,
        normalized_weight=support_weight,
        raw_p_value=support_p,
        bound=upper,
    )
    rejection: DirectionalEvidence | None = None
    if rejection_allocation is not None:
        rejection_family_alpha, rejection_local_alpha, rejection_weight = rejection_allocation
        rejection = _directional_evidence(
            direction="rejection",
            spec=spec.rejection,
            threshold=threshold,
            family_alpha=rejection_family_alpha,
            local_alpha=rejection_local_alpha,
            normalized_weight=rejection_weight,
            raw_p_value=rejection_p,
            bound=lower,
        )

    numerical_evidence: EvidenceStatus
    if support.crosses_threshold:
        numerical_evidence = "SUPPORTED"
    elif rejection is not None and rejection.crosses_threshold:
        numerical_evidence = "REJECTED"
    else:
        numerical_evidence = "INCONCLUSIVE"
    if support.crosses_threshold and rejection is not None and rejection.crosses_threshold:
        raise RuntimeError(f"claim {resolved.result_id!r} supports contradictory directions")

    sensitivities = (
        [
            _sensitivity(
                sensitivity,
                data,
                threshold=threshold,
                support_alpha=support_alpha,
                rejection_alpha=rejection_alpha,
            )
            for sensitivity in spec.sensitivity
        ]
        if spec.target.type != "prompt_failure_rate"
        else []
    )
    detectability = _detectability(
        spec,
        data,
        threshold=threshold,
        support_alpha=support_alpha,
        rejection_alpha=rejection_alpha,
        power=detectable_power,
    )
    robustness: Literal["stable", "sensitive", "not_evaluated"]
    if not sensitivities:
        robustness = "not_evaluated"
    elif all(item.evidence == numerical_evidence for item in sensitivities):
        robustness = "stable"
    else:
        robustness = "sensitive"

    warnings: list[str] = []
    if robustness == "sensitive":
        warnings.append("The evidence direction changes under a declared sensitivity method.")
    return CertificationClaimResult(
        claim_id=resolved.result_id,
        title=resolved.title,
        target=resolved.target,
        threshold=threshold,
        estimate=data.estimate,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        support=support,
        rejection=rejection,
        evidence=numerical_evidence,
        sensitivities=sensitivities,
        detectability=detectability,
        robustness=robustness,
        prompt_count=len(data.estimates),
        attempted_rollouts=int(np.sum(data.trials)),
        unclassified_outcomes=int(np.sum(data.unclassified)),
        warnings=warnings,
    )


def certify(
    *,
    empirical_report: str | Path,
    plan: CertificationPlan | str | Path,
) -> CertificationReport:
    resolved_plan = _coerce_plan(plan)
    source = load_empirical_source(empirical_report)
    required_hash = resolved_plan.source.require_plan_sha256
    empirical_plan_hash = str(source.report.analysis.get("plan_sha256", ""))
    if required_hash is not None and required_hash.lower() != empirical_plan_hash.lower():
        raise ValueError("empirical analysis plan SHA-256 does not match certification plan")
    if source.report.integrity.status == "invalid":
        raise ValueError("empirical report integrity is invalid")

    claims = _expand_claims(resolved_plan, source)
    allocations = _family_allocations(resolved_plan, claims)
    results = [
        _claim_result(
            claim,
            source,
            allocations,
            detectable_power=resolved_plan.report.detectable_power,
        )
        for claim in claims
    ]
    summary = {
        "supported": sum(item.evidence == "SUPPORTED" for item in results),
        "rejected": sum(item.evidence == "REJECTED" for item in results),
        "inconclusive": sum(item.evidence == "INCONCLUSIVE" for item in results),
        "total": len(results),
    }
    report = CertificationReport(
        package_version=__version__,
        analysis={
            **resolved_plan.certification.model_dump(mode="json"),
            "plan_sha256": canonical_certification_plan_hash(resolved_plan),
        },
        source={
            "empirical_report": str(source.report_path),
            "empirical_report_sha256": source.report_sha256,
            "resolved_design_sha256": source.design_sha256,
            "empirical_analysis_id": source.report.analysis.get("id"),
            "empirical_plan_sha256": empirical_plan_hash,
            "empirical_integrity": source.report.integrity.status,
        },
        scope={
            "design": "fixed",
            "claim": "risk_at_or_below_specified_threshold",
            "support_rule": "one_sided_upper_bound_at_or_below_threshold",
            "rejection_rule": "one_sided_lower_bound_strictly_above_threshold",
            "disclaimer": (
                "Certification evidence applies only to the declared fixed evaluation design "
                "and rollout process under each method's assumptions."
            ),
        },
        families=[item.model_dump(mode="json") for item in resolved_plan.multiplicity.families],
        claims=results,
        summary=summary,
        warnings=[],
    )
    report._resolved_plan = resolved_plan.model_dump(mode="json")
    return report
