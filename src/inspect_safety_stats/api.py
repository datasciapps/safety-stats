from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan, ComparisonSpec, RiskMeasureSpec
from inspect_safety_stats.decision.engine import decide_claim, roll_up
from inspect_safety_stats.ingest.inspect_logs import LogInput, NormalizedData, ingest_inspect_logs
from inspect_safety_stats.ingest.manifest import resolve_weights
from inspect_safety_stats.ingest.validation import (
    judge_stratum_counts,
    load_validation_manifest,
)
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import (
    ClaimResult,
    ComparisonResult,
    ComparisonSetResult,
    Diagnostic,
    Eligibility,
    GroupRiskResult,
    HierarchyRiskResult,
    IntegritySummary,
    PromptEstimate,
    RiskIntervalResult,
    SafetyReport,
)
from inspect_safety_stats.stats.binomial import (
    binomial_bounds,
    missingness_robust_binomial_bounds,
)
from inspect_safety_stats.stats.clusters import paired_cluster_bootstrap
from inspect_safety_stats.stats.comparisons import (
    DirectComparisonBounds,
    RatioComparisonBounds,
)
from inspect_safety_stats.stats.direct_mean import (
    DirectMeanBounds,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)
from inspect_safety_stats.stats.hierarchy_region import (
    LinearIntervalConstraint,
    project_contrast_hierarchy_confidence_region,
    project_hierarchy_confidence_region,
    project_ratio_hierarchy_confidence_region,
    project_two_arm_difference_hierarchy_confidence_region,
)
from inspect_safety_stats.stats.independence import (
    BoundedBlocks,
    comparison_contrast_blocks,
    direct_bounded_blocks,
    validate_profile_fields,
)
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    correct_judge_risk,
    estimate_judge_calibration,
)
from inspect_safety_stats.stats.multiplicity import bonferroni_allocate
from inspect_safety_stats.stats.paired import exact_mcnemar, newcombe_paired_interval
from inspect_safety_stats.stats.risk import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
)
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    calibrate_independent_frontiers,
)
from inspect_safety_stats.validate.plan import validate_plan_references
from inspect_safety_stats.version import __version__


def _coerce_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def _integrity_status(diagnostics: list[Diagnostic]) -> Literal["valid", "degraded", "invalid"]:
    if any(item.severity == "ERROR" for item in diagnostics):
        return "invalid"
    if any(bool(item.details.get("integrity_degraded")) for item in diagnostics):
        return "degraded"
    return "valid"


def validate(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
) -> IntegritySummary:
    resolved = _coerce_plan(plan)
    diagnostics = validate_plan_references(resolved)
    try:
        normalized = ingest_inspect_logs(resolved, logs, parallel=parallel)
        validate_profile_fields(resolved.design.independence_profiles, normalized.manifest)
        diagnostics.extend(normalized.diagnostics)
    except Exception as exc:
        diagnostics.append(Diagnostic(code="INGEST_FAILED", severity="ERROR", message=str(exc)))
    return IntegritySummary(
        status=_integrity_status(diagnostics),
        diagnostics=sorted(
            diagnostics, key=lambda item: (item.severity, item.code, item.scope or "")
        ),
    )


def _alpha_by_prompt(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> dict[str, float]:
    if not plan.outputs.prompt_bounds.enabled:
        return {}
    family_id = plan.outputs.prompt_bounds.family
    families = {family.id: family for family in plan.multiplicity.families}
    if family_id not in families:
        return {}
    alpha = families[family_id].alpha
    directional_alpha = (
        alpha / 2 if plan.outputs.prompt_bounds.coverage_mode == "two_sided" else alpha
    )
    prompts = sorted(data.manifest["prompt_id"].astype(str).tolist())
    cohort_count = len(plan.cohorts)
    if not plan.outputs.prompt_bounds.simultaneous:
        return {prompt: directional_alpha for prompt in prompts}
    strategy = plan.outputs.prompt_bounds.alpha_allocation
    weights: list[float] | None = None
    if strategy == "weighted_bonferroni":
        try:
            frame = data.manifest.sort_values("prompt_id", kind="stable")
            weights = resolve_weights(frame, plan.design.weights).tolist()
            if any(weight <= 0 for weight in weights):
                raise ValueError("all alpha weights must be positive")
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    code="ALPHA_ALLOCATION_INVALID",
                    severity="ERROR",
                    message=str(exc),
                    scope=family_id,
                )
            )
            return {}
    elif strategy == "explicit":
        explicit = plan.outputs.prompt_bounds.explicit_alpha or {}
        manifest_prompts = set(prompts)
        if set(explicit) != manifest_prompts or sum(explicit.values()) > alpha + 1e-12:
            diagnostics.append(
                Diagnostic(
                    code="ALPHA_ALLOCATION_INVALID",
                    severity="ERROR",
                    message=(
                        "explicit_alpha must contain every manifest prompt exactly once "
                        "and its values must sum to no more than the family alpha"
                    ),
                    scope=family_id,
                )
            )
            return {}
        tail_factor = 0.5 if plan.outputs.prompt_bounds.coverage_mode == "two_sided" else 1.0
        return {prompt: explicit[prompt] * tail_factor / cohort_count for prompt in prompts}
    if weights is not None:
        weights = [weight / cohort_count for weight in weights]
        return {
            prompt: directional_alpha * weight
            for prompt, weight in zip(prompts, weights, strict=True)
        }
    endpoint_alpha = bonferroni_allocate(directional_alpha, len(prompts) * cohort_count)[0]
    return {prompt: endpoint_alpha for prompt in prompts}


def _prompt_blocked(diagnostics: list[Diagnostic], cohort: str, prompt: str) -> bool:
    global_codes = {
        "PLAN_DUPLICATE_ID",
        "PLAN_UNKNOWN_REFERENCE",
        "PLAN_DIAGNOSTIC_GATE",
        "PLAN_POSTHOC",
        "INGEST_FAILED",
        "COHORT_OVERLAP",
        "ALPHA_ALLOCATION_INVALID",
        "METHOD_UNAVAILABLE",
    }
    for diagnostic in diagnostics:
        if diagnostic.severity != "ERROR":
            continue
        if diagnostic.code in global_codes or diagnostic.scope is None:
            return True
        scope = diagnostic.scope
        if scope == cohort or scope == prompt or scope.startswith(f"{cohort}:{prompt}"):
            return True
    return False


def _prompt_estimates(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> list[PromptEstimate]:
    allocations = _alpha_by_prompt(plan, data, diagnostics)
    manifest = data.manifest.set_index("prompt_id", drop=False)
    by_key: dict[tuple[str, str], list[Observation]] = {}
    for observation in data.observations:
        by_key.setdefault((observation.cohort_id, observation.prompt_id), []).append(observation)
    results: list[PromptEstimate] = []
    for cohort_id in sorted(plan.cohorts):
        for prompt_id in sorted(manifest.index.astype(str)):
            rows = by_key.get((cohort_id, prompt_id), [])
            valid = [row for row in rows if row.failure is not None]
            failures = sum(bool(row.failure) for row in valid)
            bounded_unknown = [
                row
                for row in rows
                if row.failure is None
                and (
                    (
                        row.error is not None
                        and plan.inputs.score.errors in {"end_to_end", "conservative_bounds"}
                    )
                    or (row.error is None and plan.inputs.score.missing == "conservative_bounds")
                )
            ]
            blocked = _prompt_blocked(diagnostics, cohort_id, prompt_id)
            alpha = allocations.get(prompt_id)
            lower = upper = None
            eligibility: Eligibility = "invalid"
            warnings: list[str] = []
            bounds = None
            if alpha is not None and not blocked and bounded_unknown:
                bounds = missingness_robust_binomial_bounds(
                    failures,
                    len(valid),
                    len(valid) + len(bounded_unknown),
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                )
                warnings.append(
                    f"Bounds cover all {len(bounded_unknown)} possible unclassified outcomes."
                )
            elif valid and alpha is not None and not blocked:
                bounds = binomial_bounds(
                    failures,
                    len(valid),
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                )
            if bounds is not None:
                lower, upper = bounds.lower, bounds.upper
                eligibility = (
                    "certifying" if plan.outputs.prompt_bounds.simultaneous else "diagnostic"
                )
            elif valid:
                warnings.append(
                    "Point estimate is descriptive because certifying bounds are unavailable."
                )
            item = manifest.loc[prompt_id]
            expected = int(item["expected_rollouts"]) if "expected_rollouts" in item else None
            results.append(
                PromptEstimate(
                    cohort_id=cohort_id,
                    prompt_id=prompt_id,
                    question_id=str(item["question_id"]),
                    task_id=str(item["task_id"]),
                    n_total=len(rows),
                    n_expected=expected,
                    n_valid=len(valid),
                    n_failure=failures,
                    n_safe=len(valid) - failures,
                    n_error=sum(row.error is not None for row in rows),
                    n_unclassified=sum(row.failure is None for row in rows),
                    n_policy_blocked_safe=sum(
                        row.outcome_status == "target_policy_blocked_safe" for row in rows
                    ),
                    p_hat=failures / len(valid) if valid else None,
                    lower=lower,
                    upper=upper,
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                    method=bounds.method if bounds is not None else None,
                    missingness_robust=bool(bounded_unknown and bounds is not None),
                    eligibility=eligibility,
                    warnings=warnings,
                )
            )
    return results


def _thresholds(
    measure: RiskMeasureSpec,
    prompt_ids: list[str],
    data: NormalizedData,
) -> float | np.ndarray:
    if measure.threshold is not None:
        return measure.threshold
    assert measure.threshold_column is not None
    manifest = data.manifest.set_index("prompt_id")
    if measure.threshold_column not in manifest:
        raise ValueError(f"manifest has no threshold column {measure.threshold_column!r}")
    return np.asarray(
        [manifest.loc[prompt_id, measure.threshold_column] for prompt_id in prompt_ids],
        dtype=float,
    )


def _risk_value(
    values: list[float],
    weights: list[float],
    measure: RiskMeasureSpec,
    *,
    threshold: float | np.ndarray | None = None,
) -> float:
    if measure.type == "weighted_mean":
        return weighted_mean(values, weights)
    if measure.type == "maximum":
        return maximum_risk(values, weights)
    if measure.type == "cvar":
        assert measure.beta is not None
        return weighted_cvar(values, weights, beta=float(measure.beta))
    if measure.type == "top_k_mean":
        assert measure.k is not None
        if not np.allclose(weights, np.full(len(weights), 1 / len(weights))):
            raise ValueError("top_k_mean requires equal prompt weights")
        return top_k_mean(values, k=int(measure.k))
    if measure.type == "entropic":
        assert measure.lambda_ is not None
        return entropic_risk(values, weights, lambda_=float(measure.lambda_))
    if measure.type == "threshold_exceedance":
        assert threshold is not None
        return threshold_exceedance(values, weights, threshold=threshold)
    if measure.type == "threshold_excess":
        assert threshold is not None
        return threshold_excess(values, weights, threshold=threshold)
    raise ValueError(f"unsupported risk measure {measure.type!r}")


def _measure_parameters(measure: RiskMeasureSpec) -> dict[str, Any]:
    dumped = measure.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {
        key: value for key, value in dumped.items() if key not in {"id", "type", "diagnostic_only"}
    }


def _coverage_floors(
    family_alpha: float | None,
    coverage_mode: Literal["two_sided", "per_direction"],
) -> tuple[float | None, float | None, float | None]:
    directional_alpha = (
        family_alpha / 2
        if family_alpha is not None and coverage_mode == "two_sided"
        else family_alpha
    )
    one_sided = 1 - directional_alpha if directional_alpha is not None else None
    joint = (
        1 - family_alpha
        if family_alpha is not None and coverage_mode == "two_sided"
        else max(0.0, 1 - 2 * family_alpha)
        if family_alpha is not None
        else None
    )
    return directional_alpha, one_sided, joint


@dataclass(frozen=True)
class _HierarchyNode:
    group_id: str
    level: str
    path: dict[str, str]
    prompt_ids: tuple[str, ...]
    weights: tuple[float, ...]
    design_weight: float


@dataclass(frozen=True)
class _EndpointCalibration:
    endpoint_alpha: float
    method: str
    frontier_id: str | None = None
    frontier_alpha: float | None = None
    frontier_size: int | None = None


@dataclass(frozen=True)
class _DirectMeanContext:
    groups: set[str]
    family_id: str | None
    family_alpha: float | None
    one_sided_coverage: float | None
    interval_coverage: float | None
    calibrations: dict[str, _EndpointCalibration]


def _calibration_key(cohort_id: str, target_id: str, profile_id: str) -> str:
    return f"{profile_id}@@{cohort_id}@@{target_id}"


def _hierarchy_nodes(plan: AnalysisPlan, data: NormalizedData) -> list[_HierarchyNode]:
    spec = plan.outputs.direct_mean_intervals
    if not (spec.enabled and spec.hierarchy_nodes):
        return []
    selected_groups: set[str] = set(spec.groups or [group.id for group in data.groups])
    manifest = data.manifest.set_index("prompt_id", drop=False)
    levels = [level for level in plan.design.hierarchy if level != "prompt_id"]
    nodes: list[_HierarchyNode] = []
    for group in sorted(data.groups, key=lambda item: item.id):
        if group.id not in selected_groups:
            continue
        group_frame = manifest.loc[list(group.prompt_ids)]
        for level_index, level in enumerate(levels):
            path_levels = levels[: level_index + 1]
            paths = group_frame[path_levels].astype(str).drop_duplicates()
            for path_values in paths.itertuples(index=False, name=None):
                mask = np.ones(len(group_frame), dtype=bool)
                path = dict(zip(path_levels, path_values, strict=True))
                for path_level, path_value in path.items():
                    mask &= group_frame[path_level].astype(str).to_numpy() == path_value
                prompt_ids = tuple(group_frame.loc[mask, "prompt_id"].astype(str))
                raw_weights = np.asarray(
                    [group.weights[prompt_id] for prompt_id in prompt_ids], dtype=float
                )
                design_weight = float(np.sum(raw_weights))
                if design_weight <= 0:
                    continue
                nodes.append(
                    _HierarchyNode(
                        group_id=group.id,
                        level=level,
                        path=path,
                        prompt_ids=prompt_ids,
                        weights=tuple((raw_weights / design_weight).tolist()),
                        design_weight=design_weight,
                    )
                )
    return nodes


def _frontier_signature(
    entries: list[tuple[str, tuple[str, ...], tuple[float, ...]]],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    return tuple(
        sorted(
            tuple(
                sorted(
                    (prompt_id, round(weight, 14))
                    for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                )
            )
            for _, prompt_ids, weights in entries
        )
    )


def _group_independent_frontiers(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
) -> list[IndependentFrontier]:
    selected_groups = set(
        plan.outputs.direct_mean_intervals.groups or [group.id for group in data.groups]
    )
    frontiers: list[IndependentFrontier] = []
    seen: dict[
        tuple[tuple[tuple[str, float], ...], ...],
        tuple[int, dict[tuple[tuple[str, float], ...], str]],
    ] = {}
    for group in sorted(data.groups, key=lambda item: item.id):
        if group.id not in selected_groups:
            continue
        root_id = _hierarchy_region_target_id(group.id, None, {})
        levels: list[tuple[str, list[tuple[str, tuple[str, ...], tuple[float, ...]]]]] = [
            (
                "root",
                [
                    (
                        root_id,
                        tuple(group.prompt_ids),
                        tuple(group.weights[prompt_id] for prompt_id in group.prompt_ids),
                    )
                ],
            )
        ]
        for level in [item for item in plan.design.hierarchy if item != "prompt_id"]:
            level_nodes = [
                node for node in nodes if node.group_id == group.id and node.level == level
            ]
            if not level_nodes:
                continue
            levels.append(
                (
                    level,
                    [
                        (
                            _hierarchy_region_target_id(node.group_id, node.level, node.path),
                            node.prompt_ids,
                            node.weights,
                        )
                        for node in level_nodes
                    ],
                )
            )

        for level, entries in levels:
            atom_sets = [frozenset(prompt_ids) for _, prompt_ids, _ in entries]
            if set().union(*atom_sets) != set(group.prompt_ids):
                raise ValueError(f"hierarchy level {group.id}:{level} does not partition the group")
            if any(
                left & right
                for index, left in enumerate(atom_sets)
                for right in atom_sets[index + 1 :]
            ):
                raise ValueError(f"hierarchy level {group.id}:{level} contains overlapping nodes")
            signature = _frontier_signature(entries)
            if signature in seen:
                frontier_index, canonical_by_signature = seen[signature]
                frontier = frontiers[frontier_index]
                aliases = dict(frontier.aliases)
                for target_id, prompt_ids, weights in entries:
                    target_signature = tuple(
                        sorted(
                            (prompt_id, round(weight, 14))
                            for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                        )
                    )
                    canonical_id = canonical_by_signature[target_signature]
                    aliases[canonical_id] = (*aliases[canonical_id], target_id)
                frontiers[frontier_index] = IndependentFrontier(
                    frontier_id=f"{frontier.frontier_id}+{level}",
                    targets=frontier.targets,
                    aliases=aliases,
                )
                continue
            targets = {target_id: frozenset(prompt_ids) for target_id, prompt_ids, _ in entries}
            aliases = {target_id: (target_id,) for target_id in targets}
            canonical_by_signature = {
                tuple(
                    sorted(
                        (prompt_id, round(weight, 14))
                        for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                    )
                ): target_id
                for target_id, prompt_ids, weights in entries
            }
            seen[signature] = (len(frontiers), canonical_by_signature)
            frontiers.append(
                IndependentFrontier(
                    frontier_id=f"{group.id}:{level}",
                    targets=targets,
                    aliases=aliases,
                )
            )
    return frontiers


def _direct_mean_context(
    plan: AnalysisPlan,
    data: NormalizedData,
    *,
    nodes: list[_HierarchyNode] | None = None,
) -> _DirectMeanContext:
    spec = plan.outputs.direct_mean_intervals
    if not spec.enabled:
        return _DirectMeanContext(set(), None, None, None, None, {})
    selected = set(spec.groups or [group.id for group in data.groups])
    family = next(
        (item for item in plan.multiplicity.families if item.id == spec.family),
        None,
    )
    if family is None:
        return _DirectMeanContext(selected, spec.family, None, None, None, {})
    directional_family_alpha, one_sided, joint = _coverage_floors(family.alpha, spec.coverage_mode)
    if directional_family_alpha is None:
        return _DirectMeanContext(selected, family.id, family.alpha, one_sided, joint, {})
    hierarchy_nodes = nodes or []
    assert spec.independence is not None
    profile_ids = [spec.independence.primary, *spec.independence.sensitivities]
    target_ids = [
        *[_hierarchy_region_target_id(group_id, None, {}) for group_id in sorted(selected)],
        *[
            _hierarchy_region_target_id(node.group_id, node.level, node.path)
            for node in hierarchy_nodes
        ],
    ]
    calibrations: dict[str, _EndpointCalibration] = {}
    if not spec.simultaneous:
        for profile_id in profile_ids:
            for cohort_id in sorted(plan.cohorts):
                for target_id in target_ids:
                    calibrations[_calibration_key(cohort_id, target_id, profile_id)] = (
                        _EndpointCalibration(
                            endpoint_alpha=directional_family_alpha,
                            method="pointwise",
                        )
                    )
    elif spec.hierarchy_calibration == "frontier_sidak" and spec.hierarchy_nodes:
        base_frontiers = _group_independent_frontiers(plan, data, hierarchy_nodes)
        target_details: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for group in data.groups:
            if group.id in selected:
                target_details[_hierarchy_region_target_id(group.id, None, {})] = (
                    group.prompt_ids,
                    np.asarray([group.weights[item] for item in group.prompt_ids]),
                )
        for node in hierarchy_nodes:
            target_details[_hierarchy_region_target_id(node.group_id, node.level, node.path)] = (
                node.prompt_ids,
                np.asarray(node.weights),
            )
        for profile_id in profile_ids:
            profile = plan.design.independence_profiles[profile_id]
            profile_frontiers: list[IndependentFrontier] = []
            for cohort_id in sorted(plan.cohorts):
                for frontier in base_frontiers:
                    supports: dict[str, frozenset[str]] = {}
                    for target_id in frontier.targets:
                        prompt_ids, weights = target_details[target_id]
                        supports[target_id] = frozenset(
                            direct_bounded_blocks(
                                data.observations,
                                data.manifest,
                                profile,
                                cohort_id=cohort_id,
                                prompt_ids=prompt_ids,
                                prompt_weights=weights,
                            ).block_ids
                        )
                    overlaps = any(
                        left & right
                        for index, left in enumerate(supports.values())
                        for right in list(supports.values())[index + 1 :]
                    )
                    if overlaps:
                        for target_id, atoms in supports.items():
                            aliases = frontier.aliases.get(target_id, (target_id,))
                            canonical = _calibration_key(cohort_id, target_id, profile_id)
                            profile_frontiers.append(
                                IndependentFrontier(
                                    frontier_id=(
                                        f"{profile_id}:{cohort_id}:{frontier.frontier_id}:"
                                        f"dependent:{target_id}"
                                    ),
                                    targets={canonical: atoms},
                                    aliases={
                                        canonical: tuple(
                                            _calibration_key(cohort_id, alias, profile_id)
                                            for alias in aliases
                                        )
                                    },
                                )
                            )
                    else:
                        profile_frontiers.append(
                            IndependentFrontier(
                                frontier_id=f"{profile_id}:{cohort_id}:{frontier.frontier_id}",
                                targets={
                                    _calibration_key(cohort_id, target_id, profile_id): atoms
                                    for target_id, atoms in supports.items()
                                },
                                aliases={
                                    _calibration_key(cohort_id, canonical_id, profile_id): tuple(
                                        _calibration_key(cohort_id, alias, profile_id)
                                        for alias in aliases
                                    )
                                    for canonical_id, aliases in frontier.aliases.items()
                                },
                            )
                        )
            frontier_calibrations = calibrate_independent_frontiers(
                profile_frontiers,
                directional_family_alpha=directional_family_alpha,
            )
            calibrations.update(
                {
                    target_id: _EndpointCalibration(
                        endpoint_alpha=item.endpoint_alpha,
                        method="frontier_sidak",
                        frontier_id=item.frontier_id,
                        frontier_alpha=item.frontier_alpha,
                        frontier_size=item.frontier_size,
                    )
                    for target_id, item in frontier_calibrations.items()
                }
            )
    else:
        endpoint_alpha = directional_family_alpha / len(target_ids) / len(plan.cohorts)
        for profile_id in profile_ids:
            for cohort_id in sorted(plan.cohorts):
                for target_id in target_ids:
                    calibrations[_calibration_key(cohort_id, target_id, profile_id)] = (
                        _EndpointCalibration(
                            endpoint_alpha=endpoint_alpha,
                            method="flat_bonferroni",
                        )
                    )
    return _DirectMeanContext(
        selected,
        family.id,
        family.alpha,
        one_sided,
        joint,
        calibrations,
    )


def _direct_bounds(
    method: str,
    blocks: BoundedBlocks,
    *,
    endpoint_alpha: float,
    independence_profile: str,
) -> DirectMeanBounds:
    function = {
        "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
        "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
        "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
    }.get(method)
    if function is None:
        raise ValueError(f"unsupported direct mean interval method {method!r}")
    return function(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=endpoint_alpha,
        alpha_upper=endpoint_alpha,
        independence_profile=independence_profile,
    )


def _direct_interval_result(
    *,
    bounds: DirectMeanBounds,
    role: Literal["primary", "reference", "sensitivity"],
    method: str,
    independence_profile: str,
    independent_block_count: int,
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    calibration: _EndpointCalibration,
    one_sided_coverage: float | None,
    interval_coverage: float | None,
) -> RiskIntervalResult:
    spec = plan.outputs.direct_mean_intervals
    assert spec.independence is not None
    profile = plan.design.independence_profiles[independence_profile]
    directional_family_alpha = (
        family_alpha / 2
        if family_alpha is not None and spec.coverage_mode == "two_sided"
        else family_alpha
    )
    assumptions = (
        [
            "Attempted rollout outcomes are mutually independent across the fixed node.",
            "Prompt weights and rollout counts are fixed independently of observed outcomes.",
        ]
        if profile.block_by == ["rollout"]
        else [
            f"Blocks defined by {profile.block_by!r} are mutually independent.",
            "Arbitrary dependence is allowed among observations within each declared block.",
            "The block partition and design weights are fixed independently of outcomes.",
        ]
    )
    assumptions.append("Every unclassified attempted outcome has a latent binary safety label.")
    if calibration.method == "frontier_sidak":
        assumptions.extend(
            [
                "Each hierarchy frontier is a partition into nodes with disjoint atomic outcomes.",
                "Independent nodes within a frontier use exact Sidak product calibration.",
                "Potentially dependent frontiers use Bonferroni allocation of "
                "directional family alpha.",
            ]
        )
    if bounds.chernoff_kernel == "bernoulli_kl_equal_atomic_weights":
        assumptions.append(
            "Equal atomic design weights permit inversion of the Bernoulli-KL "
            "Chernoff exponent; identical failure probabilities are not required."
        )
    elif bounds.chernoff_kernel == "weighted_poisson_chernoff_max_increment":
        assumptions.append(
            "Unequal atomic design weights use the weighted Poisson-Chernoff "
            "exponent determined by the largest atomic weight."
        )
    scope_prefix = "simultaneous" if spec.simultaneous else "pointwise"
    calibration_scope = "_frontier_sidak" if calibration.method == "frontier_sidak" else ""
    return RiskIntervalResult(
        interval_id=(
            f"direct_{method}"
            if independence_profile == spec.independence.primary
            else f"direct_{method}_{independence_profile}"
        ),
        lower=bounds.lower,
        upper=bounds.upper,
        method=method,
        role=role,
        coverage_scope=(
            f"{scope_prefix}{calibration_scope}_{independence_profile}_fixed_design_mean_family"
        ),
        coverage_mode=spec.coverage_mode,
        family=family_id,
        family_alpha=family_alpha,
        directional_family_alpha=directional_family_alpha,
        alpha_lower=calibration.endpoint_alpha,
        alpha_upper=calibration.endpoint_alpha,
        simultaneous_coverage=interval_coverage,
        lower_one_sided_coverage=one_sided_coverage,
        upper_one_sided_coverage=one_sided_coverage,
        completion_estimate_lower=bounds.completion_estimate_lower,
        completion_estimate_upper=bounds.completion_estimate_upper,
        half_width_lower=bounds.half_width_lower,
        half_width_upper=bounds.half_width_upper,
        effective_sample_size=bounds.effective_sample_size,
        concentration_variance=bounds.concentration_variance,
        variance_upper_lower=bounds.variance_upper_lower,
        variance_upper_upper=bounds.variance_upper_upper,
        maximum_increment=bounds.maximum_increment,
        chernoff_kernel=bounds.chernoff_kernel,
        calibration_method=calibration.method,
        calibration_frontier=calibration.frontier_id,
        calibration_frontier_alpha=calibration.frontier_alpha,
        calibration_frontier_size=calibration.frontier_size,
        independence_profile=independence_profile,
        independent_block_count=independent_block_count,
        missingness_method="all_safe_all_failure_completion",
        assumptions=assumptions,
        eligibility="certifying",
    )


def _direct_intervals(
    *,
    data: NormalizedData,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    weights: np.ndarray,
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    target_id: str,
    calibrations: dict[str, _EndpointCalibration],
    one_sided_coverage: float | None,
    interval_coverage: float | None,
) -> list[RiskIntervalResult]:
    spec = plan.outputs.direct_mean_intervals
    assert spec.independence is not None
    methods = [spec.method, *spec.reference_methods]
    results: list[RiskIntervalResult] = []
    profile_ids = [spec.independence.primary, *spec.independence.sensitivities]
    for profile_id in profile_ids:
        profile = plan.design.independence_profiles[profile_id]
        blocks = direct_bounded_blocks(
            data.observations,
            data.manifest,
            profile,
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            prompt_weights=weights,
        )
        calibration = calibrations[_calibration_key(cohort_id, target_id, profile_id)]
        for method in methods:
            bounds = _direct_bounds(
                method,
                blocks,
                endpoint_alpha=calibration.endpoint_alpha,
                independence_profile=profile_id,
            )
            results.append(
                _direct_interval_result(
                    bounds=bounds,
                    role=(
                        "sensitivity"
                        if profile_id != spec.independence.primary
                        else "primary"
                        if method == spec.method
                        else "reference"
                    ),
                    method=method,
                    independence_profile=profile_id,
                    independent_block_count=blocks.count,
                    plan=plan,
                    family_id=family_id,
                    family_alpha=family_alpha,
                    calibration=calibration,
                    one_sided_coverage=one_sided_coverage,
                    interval_coverage=interval_coverage,
                )
            )
    return results


def _group_risks(
    plan: AnalysisPlan,
    data: NormalizedData,
    estimates: list[PromptEstimate],
    diagnostics: list[Diagnostic],
    direct_context: _DirectMeanContext | None = None,
) -> list[GroupRiskResult]:
    index = {(item.cohort_id, item.prompt_id): item for item in estimates}
    family_id = plan.outputs.prompt_bounds.family
    family = next((f for f in plan.multiplicity.families if f.id == family_id), None)
    family_alpha = family.alpha if family else None
    coverage_mode = plan.outputs.prompt_bounds.coverage_mode
    directional_family_alpha, one_sided_coverage, interval_coverage = _coverage_floors(
        family_alpha, coverage_mode
    )
    direct = direct_context or _direct_mean_context(plan, data)
    results: list[GroupRiskResult] = []
    for cohort_id in sorted(plan.cohorts):
        for group in sorted(data.groups, key=lambda item: item.id):
            prompt_ids = list(group.prompt_ids)
            weights = [group.weights[prompt_id] for prompt_id in prompt_ids]
            prompt_results = [index[(cohort_id, prompt_id)] for prompt_id in prompt_ids]
            valid = [item for item in prompt_results if item.p_hat is not None]
            complete_estimate = len(valid) == len(prompt_results)
            complete_bounds = all(
                item.lower is not None
                and item.upper is not None
                and item.eligibility == "certifying"
                for item in prompt_results
            )
            unresolved_weight = sum(
                weight
                for weight, item in zip(weights, prompt_results, strict=True)
                if item.p_hat is None
            )
            for measure in plan.outputs.risk_measures:
                estimate = lower = upper = None
                certified_safe = certified_unsafe = None
                warnings: list[str] = []
                try:
                    threshold = (
                        _thresholds(measure, prompt_ids, data)
                        if measure.type in {"threshold_exceedance", "threshold_excess"}
                        else None
                    )
                    if complete_estimate:
                        estimate = _risk_value(
                            [cast(float, item.p_hat) for item in prompt_results],
                            weights,
                            measure,
                            threshold=threshold,
                        )
                    if complete_bounds:
                        lowers = [cast(float, item.lower) for item in prompt_results]
                        uppers = [cast(float, item.upper) for item in prompt_results]
                        lower = _risk_value(lowers, weights, measure, threshold=threshold)
                        upper = _risk_value(uppers, weights, measure, threshold=threshold)
                        if measure.type == "threshold_exceedance":
                            assert threshold is not None
                            t = np.asarray(threshold, dtype=float)
                            if t.ndim == 0:
                                t = np.full(len(prompt_ids), float(t))
                            w = np.asarray(weights)
                            certified_safe = float(np.sum(w[np.asarray(uppers) <= t]))
                            certified_unsafe = float(np.sum(w[np.asarray(lowers) > t]))
                    elif complete_estimate:
                        warnings.append(
                            "No complete simultaneous prompt-bound family is available."
                        )
                except ValueError as exc:
                    diagnostics.append(
                        Diagnostic(
                            code="RISK_MEASURE_INVALID",
                            severity="ERROR",
                            message=str(exc),
                            scope=f"{cohort_id}:{group.id}:{measure.id}",
                        )
                    )
                    warnings.append(str(exc))
                propagated_eligibility: Eligibility = (
                    "diagnostic"
                    if measure.diagnostic_only and estimate is not None
                    else "certifying"
                    if complete_bounds and lower is not None and upper is not None
                    else "invalid"
                )
                computed_direct_intervals: list[RiskIntervalResult] = []
                direct_warnings: list[str] = []
                direct_requested = measure.type == "weighted_mean" and group.id in direct.groups
                if direct_requested:
                    target_id = _hierarchy_region_target_id(group.id, None, {})
                    independence = plan.outputs.direct_mean_intervals.independence
                    assert independence is not None
                    calibration = direct.calibrations.get(
                        _calibration_key(cohort_id, target_id, independence.primary)
                    )
                    if calibration is None:
                        direct_warnings.append(
                            "The direct-mean multiplicity family is unavailable."
                        )
                    elif any(
                        item.n_total <= 0 or _prompt_blocked(diagnostics, cohort_id, item.prompt_id)
                        for item in prompt_results
                    ):
                        direct_warnings.append(
                            "A complete valid attempted-rollout ledger is required."
                        )
                    else:
                        try:
                            computed_direct_intervals = _direct_intervals(
                                data=data,
                                cohort_id=cohort_id,
                                prompt_ids=prompt_ids,
                                weights=np.asarray(weights, dtype=float),
                                plan=plan,
                                family_id=direct.family_id,
                                family_alpha=direct.family_alpha,
                                target_id=target_id,
                                calibrations=direct.calibrations,
                                one_sided_coverage=direct.one_sided_coverage,
                                interval_coverage=direct.interval_coverage,
                            )
                        except (TypeError, ValueError) as exc:
                            direct_warnings.append(str(exc))
                            diagnostics.append(
                                Diagnostic(
                                    code="DIRECT_MEAN_INTERVAL_INVALID",
                                    severity="ERROR",
                                    message=str(exc),
                                    scope=f"{cohort_id}:{group.id}:{measure.id}",
                                )
                            )

                direct_primary = next(
                    (item for item in computed_direct_intervals if item.role == "primary"),
                    None,
                )
                propagated_interval = RiskIntervalResult(
                    interval_id="propagated_prompt_envelope",
                    lower=lower,
                    upper=upper,
                    method="propagated_exact_prompt_bounds",
                    role="reference" if direct_primary is not None else "primary",
                    coverage_scope="simultaneous_over_declared_prompt_family",
                    coverage_mode=coverage_mode,
                    family=family_id,
                    family_alpha=family_alpha,
                    directional_family_alpha=directional_family_alpha,
                    simultaneous_coverage=(interval_coverage if complete_bounds else None),
                    lower_one_sided_coverage=(one_sided_coverage if complete_bounds else None),
                    upper_one_sided_coverage=(one_sided_coverage if complete_bounds else None),
                    missingness_method=(
                        "all_safe_all_failure_completion"
                        if any(item.n_unclassified for item in prompt_results)
                        else "none"
                    ),
                    assumptions=[
                        "IID Bernoulli rollouts within each fixed prompt.",
                        "No independence between prompts is required.",
                    ],
                    eligibility=propagated_eligibility,
                    warnings=(
                        []
                        if complete_bounds
                        else ["No complete simultaneous prompt-bound family is available."]
                    ),
                )
                intervals = [*computed_direct_intervals, propagated_interval]
                primary_interval_id = propagated_interval.interval_id
                primary_lower = lower
                primary_upper = upper
                primary_method = propagated_interval.method
                primary_scope = propagated_interval.coverage_scope
                primary_coverage_mode = propagated_interval.coverage_mode
                primary_family_alpha = propagated_interval.family_alpha
                primary_directional_alpha = propagated_interval.directional_family_alpha
                primary_interval_coverage = propagated_interval.simultaneous_coverage
                primary_lower_coverage = propagated_interval.lower_one_sided_coverage
                primary_upper_coverage = propagated_interval.upper_one_sided_coverage
                eligibility = propagated_eligibility

                if direct_primary is not None:
                    primary_interval_id = direct_primary.interval_id
                    primary_lower = direct_primary.lower
                    primary_upper = direct_primary.upper
                    primary_method = direct_primary.method
                    primary_scope = direct_primary.coverage_scope
                    primary_coverage_mode = direct_primary.coverage_mode
                    primary_family_alpha = direct_primary.family_alpha
                    primary_directional_alpha = direct_primary.directional_family_alpha
                    primary_interval_coverage = direct_primary.simultaneous_coverage
                    primary_lower_coverage = direct_primary.lower_one_sided_coverage
                    primary_upper_coverage = direct_primary.upper_one_sided_coverage
                    eligibility = direct_primary.eligibility
                warnings.extend(direct_warnings)
                results.append(
                    GroupRiskResult(
                        cohort_id=cohort_id,
                        group_id=group.id,
                        measure_id=measure.id,
                        estimate=estimate,
                        lower=primary_lower,
                        upper=primary_upper,
                        method=f"finite_design_{measure.type}",
                        primary_interval_id=primary_interval_id,
                        intervals=intervals,
                        interval_method=primary_method,
                        coverage_scope=primary_scope,
                        coverage_mode=primary_coverage_mode,
                        family_alpha=primary_family_alpha,
                        directional_family_alpha=primary_directional_alpha,
                        simultaneous_coverage=primary_interval_coverage,
                        lower_one_sided_coverage=primary_lower_coverage,
                        upper_one_sided_coverage=primary_upper_coverage,
                        prompt_count=len(prompt_ids),
                        valid_prompt_count=len(valid),
                        resolved_weight=1.0 - unresolved_weight,
                        unresolved_weight=unresolved_weight,
                        parameters=_measure_parameters(measure),
                        eligibility=eligibility,
                        certified_safe_weight=certified_safe,
                        certified_unsafe_weight=certified_unsafe,
                        warnings=warnings,
                    )
                )
    return results


def _hierarchy_risks(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
    estimates: list[PromptEstimate],
    diagnostics: list[Diagnostic],
    direct_context: _DirectMeanContext,
) -> list[HierarchyRiskResult]:
    if not nodes:
        return []
    index = {(item.cohort_id, item.prompt_id): item for item in estimates}
    results: list[HierarchyRiskResult] = []
    for cohort_id in sorted(plan.cohorts):
        for node in nodes:
            prompt_results = [index[(cohort_id, prompt_id)] for prompt_id in node.prompt_ids]
            valid = [item for item in prompt_results if item.p_hat is not None]
            estimate = (
                float(
                    np.dot(
                        np.asarray(node.weights, dtype=float),
                        np.asarray([cast(float, item.p_hat) for item in prompt_results]),
                    )
                )
                if len(valid) == len(prompt_results)
                else None
            )
            intervals: list[RiskIntervalResult] = []
            warnings: list[str] = []
            target_id = _hierarchy_region_target_id(node.group_id, node.level, node.path)
            independence = plan.outputs.direct_mean_intervals.independence
            assert independence is not None
            calibration = direct_context.calibrations.get(
                _calibration_key(cohort_id, target_id, independence.primary)
            )
            if calibration is None:
                warnings.append("The direct-mean multiplicity family is unavailable.")
            elif any(
                item.n_total <= 0 or _prompt_blocked(diagnostics, cohort_id, item.prompt_id)
                for item in prompt_results
            ):
                warnings.append("A complete valid attempted-rollout ledger is required.")
            else:
                try:
                    intervals = _direct_intervals(
                        data=data,
                        cohort_id=cohort_id,
                        prompt_ids=node.prompt_ids,
                        weights=np.asarray(node.weights, dtype=float),
                        plan=plan,
                        family_id=direct_context.family_id,
                        family_alpha=direct_context.family_alpha,
                        target_id=target_id,
                        calibrations=direct_context.calibrations,
                        one_sided_coverage=direct_context.one_sided_coverage,
                        interval_coverage=direct_context.interval_coverage,
                    )
                except (TypeError, ValueError) as exc:
                    warnings.append(str(exc))
                    diagnostics.append(
                        Diagnostic(
                            code="HIERARCHY_MEAN_INTERVAL_INVALID",
                            severity="ERROR",
                            message=str(exc),
                            scope=f"{cohort_id}:{node.group_id}:{node.level}",
                        )
                    )
            primary = next((item for item in intervals if item.role == "primary"), None)
            results.append(
                HierarchyRiskResult(
                    cohort_id=cohort_id,
                    group_id=node.group_id,
                    level=node.level,
                    path=node.path,
                    estimate=estimate,
                    lower=primary.lower if primary is not None else None,
                    upper=primary.upper if primary is not None else None,
                    primary_interval_id=(primary.interval_id if primary is not None else None),
                    intervals=intervals,
                    interval_method=primary.method if primary is not None else None,
                    coverage_scope=primary.coverage_scope if primary is not None else None,
                    simultaneous_coverage=(
                        primary.simultaneous_coverage if primary is not None else None
                    ),
                    lower_one_sided_coverage=(
                        primary.lower_one_sided_coverage if primary is not None else None
                    ),
                    upper_one_sided_coverage=(
                        primary.upper_one_sided_coverage if primary is not None else None
                    ),
                    design_weight=node.design_weight,
                    prompt_count=len(node.prompt_ids),
                    valid_prompt_count=len(valid),
                    eligibility=primary.eligibility if primary is not None else "invalid",
                    warnings=warnings,
                )
            )
    return results


def _hierarchy_region_target_id(group_id: str, level: str | None, path: dict[str, str]) -> str:
    if level is None:
        return f"{group_id}::root"
    path_text = "::".join(f"{key}={value}" for key, value in path.items())
    return f"{group_id}::{level}::{path_text}"


def _direct_interval_for_region(
    intervals: list[RiskIntervalResult], method: str, independence_unit: str
) -> RiskIntervalResult | None:
    return next(
        (
            interval
            for interval in intervals
            if interval.method == method
            and interval.independence_profile == independence_unit
            and interval.interval_id.startswith("direct_")
            and interval.lower is not None
            and interval.upper is not None
        ),
        None,
    )


def _project_hierarchy_regions(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    diagnostics: list[Diagnostic],
) -> None:
    spec = plan.outputs.direct_mean_intervals
    if not spec.enabled or not spec.hierarchy_nodes or spec.hierarchy_region != "linear_program":
        return

    selected_groups = set(spec.groups or [group.id for group in data.groups])
    mean_measure_ids = {
        measure.id for measure in plan.outputs.risk_measures if measure.type == "weighted_mean"
    }
    prompt_ids = sorted(
        {
            prompt_id
            for group in data.groups
            if group.id in selected_groups
            for prompt_id in group.prompt_ids
        }
    )
    if not prompt_ids:
        return
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    group_index = {group.id: group for group in data.groups if group.id in selected_groups}
    node_index = {(node.group_id, node.level, tuple(node.path.items())): node for node in nodes}

    for cohort_id in sorted(plan.cohorts):
        root_rows: dict[str, list[GroupRiskResult]] = {}
        for risk in risks:
            if (
                risk.cohort_id == cohort_id
                and risk.group_id in selected_groups
                and risk.measure_id in mean_measure_ids
            ):
                root_rows.setdefault(risk.group_id, []).append(risk)
        node_rows = [row for row in hierarchy_risks if row.cohort_id == cohort_id]

        target_coefficients: dict[str, np.ndarray] = {}
        target_intervals: dict[str, list[RiskIntervalResult]] = {}
        target_results: dict[str, GroupRiskResult | HierarchyRiskResult] = {}
        for group_id, rows in root_rows.items():
            group = group_index[group_id]
            coefficients = np.zeros(len(prompt_ids), dtype=float)
            for prompt_id in group.prompt_ids:
                coefficients[prompt_index[prompt_id]] = group.weights[prompt_id]
            target_id = _hierarchy_region_target_id(group_id, None, {})
            target_coefficients[target_id] = coefficients
            target_intervals[target_id] = rows[0].intervals
            target_results[target_id] = rows[0]
        for row in node_rows:
            node = node_index[(row.group_id, row.level, tuple(row.path.items()))]
            coefficients = np.zeros(len(prompt_ids), dtype=float)
            for prompt_id, weight in zip(node.prompt_ids, node.weights, strict=True):
                coefficients[prompt_index[prompt_id]] = weight
            target_id = _hierarchy_region_target_id(row.group_id, row.level, row.path)
            target_coefficients[target_id] = coefficients
            target_intervals[target_id] = row.intervals
            target_results[target_id] = row

        method_units = {
            (interval.method, interval.independence_profile)
            for intervals in target_intervals.values()
            for interval in intervals
            if interval.interval_id.startswith("direct_")
            and interval.independence_profile is not None
            and interval.lower is not None
            and interval.upper is not None
        }
        for method, independence_unit in sorted(method_units):
            interval_by_target = {
                target_id: interval
                for target_id, intervals in target_intervals.items()
                if (interval := _direct_interval_for_region(intervals, method, independence_unit))
                is not None
            }
            constraints = [
                LinearIntervalConstraint(
                    target_id=target_id,
                    coefficients=target_coefficients[target_id],
                    lower=cast(float, interval.lower),
                    upper=cast(float, interval.upper),
                )
                for target_id, interval in interval_by_target.items()
            ]
            if not constraints:
                continue
            projection = project_hierarchy_confidence_region(
                constraints,
                {target_id: target_coefficients[target_id] for target_id in interval_by_target},
                variable_count=len(prompt_ids),
            )
            if not projection.feasible:
                diagnostics.append(
                    Diagnostic(
                        code="HIERARCHY_CONFIDENCE_REGION_EMPTY",
                        severity="WARNING",
                        message=(
                            "The raw simultaneous node intervals have no common parameter "
                            "vector under the declared fixed-design hierarchy."
                        ),
                        scope=f"{cohort_id}:{method}:{independence_unit}",
                        details={
                            "solver_message": projection.message,
                            "constraint_count": projection.constraint_count,
                        },
                    )
                )
                continue
            for target_id, (lower, upper) in projection.intervals.items():
                interval = interval_by_target[target_id]
                interval.raw_lower = interval.lower
                interval.raw_upper = interval.upper
                interval.lower = max(cast(float, interval.lower), lower)
                interval.upper = min(cast(float, interval.upper), upper)
                interval.half_width_lower = (
                    interval.completion_estimate_lower - interval.lower
                    if interval.completion_estimate_lower is not None
                    else None
                )
                interval.half_width_upper = (
                    interval.upper - interval.completion_estimate_upper
                    if interval.completion_estimate_upper is not None
                    else None
                )
                interval.projection_method = "hierarchy_linear_program"
                interval.projection_constraint_count = projection.constraint_count
                interval.coverage_scope = f"{interval.coverage_scope}_hierarchy_region"
                interval.assumptions.append(
                    "Reported endpoints are projections of the common fixed-design "
                    "hierarchy confidence region."
                )

        for target_id, result in target_results.items():
            primary = next(
                (
                    interval
                    for interval in target_intervals[target_id]
                    if interval.role == "primary"
                ),
                None,
            )
            if primary is None:
                continue
            result.lower = primary.lower
            result.upper = primary.upper
            result.coverage_scope = primary.coverage_scope
            result.raw_lower = primary.raw_lower
            result.raw_upper = primary.raw_upper
            result.projection_method = primary.projection_method
            for root_result in root_rows.get(result.group_id, []):
                if isinstance(result, GroupRiskResult):
                    root_result.lower = primary.lower
                    root_result.upper = primary.upper
                    root_result.coverage_scope = primary.coverage_scope
                    root_result.raw_lower = primary.raw_lower
                    root_result.raw_upper = primary.raw_upper
                    root_result.projection_method = primary.projection_method


# Risk measures whose value is positively homogeneous and translation equivariant,
# so the affine judge correction R* = R * rho_1 + (1 - R) * rho_0 applies unchanged.
_JUDGE_CORRECTABLE_MEASURES = frozenset(
    {"weighted_mean", "maximum", "cvar", "top_k_mean"}
)


def _assumed_judge_calibrations(plan: AnalysisPlan) -> list[JudgeCalibration]:
    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "assumed":
        return []
    ordered = sorted(spec.scenarios, key=lambda item: (item.id != spec.primary, item.id))
    return [
        JudgeCalibration(
            scenario_id=item.id,
            rho_1_lower=item.rho_1.lower,
            rho_1_upper=item.rho_1.upper,
            rho_0_lower=item.rho_0.lower,
            rho_0_upper=item.rho_0.upper,
            source=spec.mode,
        )
        for item in ordered
    ]


def _validated_judge_calibrations(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> tuple[dict[str, JudgeCalibration], dict[str, dict[str, Any]], float | None]:
    """Estimate one calibration per cohort from human labels within judge strata."""

    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "validated":
        return {}, {}, None
    family = next((item for item in plan.multiplicity.families if item.id == spec.family), None)
    if family is None:
        diagnostics.append(
            Diagnostic(
                code="PLAN_UNKNOWN_REFERENCE",
                severity="ERROR",
                message="validated judge calibration requires a valid multiplicity family",
                scope="inputs.judge_calibration",
            )
        )
        return {}, {}, None
    assert spec.validation_manifest is not None
    try:
        labels = load_validation_manifest(spec.validation_manifest)
    except (OSError, ValueError) as exc:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_INVALID",
                severity="ERROR",
                message=str(exc),
                scope="inputs.judge_calibration",
            )
        )
        return {}, {}, None

    counts, unmatched = judge_stratum_counts(data.observations, labels)
    if unmatched:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_UNMATCHED",
                severity="ERROR",
                message=(
                    f"{len(unmatched)} validation rows reference observations absent from the "
                    "analysed logs or with unclassified outcomes"
                ),
                scope="inputs.judge_calibration",
                details={"observation_ids": unmatched[:20]},
            )
        )

    calibrations: dict[str, JudgeCalibration] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for cohort_id in sorted(plan.cohorts):
        strata = counts.get(cohort_id, {})
        unsafe = strata.get("judge_unsafe")
        safe = strata.get("judge_safe")
        if unsafe is None or safe is None:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_VALIDATION_STRATUM_EMPTY",
                    severity="WARNING",
                    message=(
                        f"cohort {cohort_id!r} has no observations in one judge stratum; "
                        "no judge correction is reported for it"
                    ),
                    scope=cohort_id,
                )
            )
            continue
        try:
            calibrations[cohort_id] = estimate_judge_calibration(
                scenario_id="validated",
                unsafe_stratum=unsafe,
                safe_stratum=safe,
                alpha=family.alpha,
            )
        except ValueError as exc:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_CALIBRATION_UNINFORMATIVE",
                    severity="WARNING",
                    message=str(exc),
                    scope=cohort_id,
                )
            )
            continue
        summaries[cohort_id] = {
            "judge_unsafe": {
                "labelled": unsafe.labelled,
                "truly_unsafe": unsafe.truly_unsafe,
                "stratum_size": unsafe.stratum_size,
            },
            "judge_safe": {
                "labelled": safe.labelled,
                "truly_unsafe": safe.truly_unsafe,
                "stratum_size": safe.stratum_size,
            },
        }
    return calibrations, summaries, family.alpha


def _judge_corrected_interval(
    *,
    result: GroupRiskResult | HierarchyRiskResult,
    calibration: JudgeCalibration,
    primary_scenario: str | None,
    family_id: str | None,
    judge_alpha: float | None,
) -> RiskIntervalResult | None:
    if result.lower is None or result.upper is None:
        return None
    corrected = correct_judge_risk(
        observed_lower=result.lower,
        observed_upper=result.upper,
        observed_estimate=result.estimate,
        calibration=calibration,
    )
    validated = calibration.source == "validated"
    base = next((item for item in result.intervals if item.role == "primary"), None)
    base_alpha = base.family_alpha if base is not None else None
    simultaneous = (
        1 - (base_alpha + judge_alpha)
        if validated and base_alpha is not None and judge_alpha is not None
        else None
    )
    assumptions = [
        "The reported estimand is true failure risk, not judge-reported failure risk.",
        f"P(truly unsafe | judge said unsafe) lies in "
        f"[{calibration.rho_1_lower:.6g}, {calibration.rho_1_upper:.6g}].",
        f"P(truly unsafe | judge said safe) lies in "
        f"[{calibration.rho_0_lower:.6g}, {calibration.rho_0_upper:.6g}].",
    ]
    if validated:
        assumptions.extend(
            [
                "Those two ranges are exact Clopper-Pearson bounds from human labels drawn "
                "within each judge stratum.",
                "The labelled rows are a simple random sample within their judge stratum; "
                "this is a property of the declared collection protocol, not of the data.",
                "Human adjudication is treated as ground truth.",
                "One calibration is estimated per cohort and shared by every group in it.",
                "Coverage is the union bound over the underlying interval and the four "
                "calibration endpoints.",
            ]
        )
    else:
        assumptions.extend(
            [
                "These conditional accuracies are declared assumptions, not measurements; "
                "the interval inherits no coverage guarantee from them.",
                "The declared accuracies are prevalence-dependent and are applied unchanged "
                "to every cohort.",
            ]
        )
    assumptions.append("The underlying judge-system interval retains its own stated guarantee.")
    warnings: list[str] = []
    if not validated and primary_scenario is not None:
        if calibration.scenario_id != primary_scenario:
            warnings.append(
                "Non-primary judge-calibration scenario; reported for sensitivity only."
            )
    return RiskIntervalResult(
        interval_id=f"judge_corrected_{calibration.scenario_id}",
        lower=corrected.lower,
        upper=corrected.upper,
        method=(
            "validated_judge_misclassification_correction"
            if validated
            else "assumed_judge_misclassification_correction"
        ),
        role="reference" if validated else "sensitivity",
        coverage_scope=(
            f"{result.coverage_scope}_judge_corrected"
            if result.coverage_scope
            else "judge_corrected"
        ),
        family=family_id,
        family_alpha=judge_alpha,
        simultaneous_coverage=simultaneous,
        completion_estimate_lower=corrected.completion_estimate_lower,
        completion_estimate_upper=corrected.completion_estimate_upper,
        raw_lower=result.lower,
        raw_upper=result.upper,
        calibration_method=f"{calibration.source}_judge_calibration",
        assumptions=assumptions,
        eligibility="certifying" if validated else "diagnostic",
        warnings=warnings,
    )


def _apply_judge_correction(
    plan: AnalysisPlan,
    data: NormalizedData,
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    """Map judge-system risk rows onto the true-risk scale.

    Returns a summary of the calibration actually used, for the report provenance
    block. In ``assumed`` mode every scenario is applied to every cohort. In
    ``validated`` mode one calibration is estimated per cohort from human labels.
    """

    spec = plan.inputs.judge_calibration
    if spec is None:
        return {}
    correctable = {
        measure.id
        for measure in plan.outputs.risk_measures
        if measure.type in _JUDGE_CORRECTABLE_MEASURES
    }
    targets: list[GroupRiskResult | HierarchyRiskResult] = [
        *(row for row in risks if row.measure_id in correctable),
        *hierarchy_risks,
    ]

    if spec.mode == "assumed":
        calibrations = _assumed_judge_calibrations(plan)
        for result in targets:
            for calibration in calibrations:
                interval = _judge_corrected_interval(
                    result=result,
                    calibration=calibration,
                    primary_scenario=spec.primary,
                    family_id=None,
                    judge_alpha=None,
                )
                if interval is not None:
                    result.intervals.append(interval)
        return {
            "mode": "assumed",
            "scenarios": [item.scenario_id for item in calibrations],
            "primary": spec.primary,
        }

    by_cohort, summaries, judge_alpha = _validated_judge_calibrations(plan, data, diagnostics)
    for result in targets:
        cohort_calibration = by_cohort.get(result.cohort_id)
        if cohort_calibration is None:
            continue
        interval = _judge_corrected_interval(
            result=result,
            calibration=cohort_calibration,
            primary_scenario=None,
            family_id=spec.family,
            judge_alpha=judge_alpha,
        )
        if interval is not None:
            result.intervals.append(interval)
    if by_cohort:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CORRECTION_NOT_USED_BY_CLAIMS",
                severity="WARNING",
                message=(
                    "a validated judge correction is available, but threshold claims are still "
                    "evaluated against the uncorrected judge-system interval; compare claims "
                    "against the judge_corrected_validated interval manually"
                ),
                scope="inputs.judge_calibration",
            )
        )
    return {
        "mode": "validated",
        "family": spec.family,
        "family_alpha": judge_alpha,
        "sampling": spec.sampling,
        "validation_manifest": str(spec.validation_manifest),
        "cohorts": summaries,
    }


def _observation_attr(observation: Observation, name: str) -> Any:
    if not hasattr(observation, name):
        raise ValueError(f"pair or cluster key {name!r} is unavailable")
    return getattr(observation, name)


@dataclass(frozen=True)
class _ComparisonTarget:
    result_id: str
    level: str | None
    path: dict[str, str]
    prompt_ids: tuple[str, ...]
    weights: tuple[float, ...]
    design_weight: float


def _comparison_targets(
    plan: AnalysisPlan,
    data: NormalizedData,
    spec: ComparisonSpec,
) -> list[_ComparisonTarget]:
    group = next(item for item in data.groups if item.id == spec.group)
    prompts = tuple(group.prompt_ids)
    weights = tuple(group.weights[prompt_id] for prompt_id in prompts)
    targets = [
        _ComparisonTarget(
            result_id=spec.id,
            level=None,
            path={},
            prompt_ids=prompts,
            weights=weights,
            design_weight=1.0,
        )
    ]
    if not spec.hierarchy_nodes:
        return targets
    manifest = data.manifest.set_index("prompt_id", drop=False).loc[list(prompts)]
    levels = [level for level in plan.design.hierarchy if level != "prompt_id"]
    for level_index, level in enumerate(levels):
        path_levels = levels[: level_index + 1]
        paths = manifest[path_levels].astype(str).drop_duplicates()
        for values in paths.itertuples(index=False, name=None):
            path = dict(zip(path_levels, values, strict=True))
            mask = np.ones(len(manifest), dtype=bool)
            for path_level, path_value in path.items():
                mask &= manifest[path_level].astype(str).to_numpy() == path_value
            node_prompts = tuple(manifest.loc[mask, "prompt_id"].astype(str))
            raw = np.asarray([group.weights[prompt_id] for prompt_id in node_prompts])
            design_weight = float(np.sum(raw))
            if design_weight <= 0:
                continue
            token = "/".join(f"{key}={value}" for key, value in path.items())
            targets.append(
                _ComparisonTarget(
                    result_id=f"{spec.id}::{level}::{token}",
                    level=level,
                    path=path,
                    prompt_ids=node_prompts,
                    weights=tuple((raw / design_weight).tolist()),
                    design_weight=design_weight,
                )
            )
    return targets


def _comparison_target_key(spec_id: str, result_id: str, profile_id: str) -> str:
    return f"{profile_id}@@{spec_id}@@{result_id}"


def _comparison_region_alias_key(
    spec: ComparisonSpec, *, allocation_weight: float
) -> tuple[Any, ...]:
    if spec.method == "weighted_kl_chernoff":
        return (
            "two_arm_kl_region",
            spec.candidate,
            spec.baseline,
            spec.group,
            spec.measure,
            spec.coverage_mode,
            spec.hierarchy_calibration,
            spec.hierarchy_region,
            allocation_weight,
        )
    return ("estimand_specific_region", spec.id)


def _comparison_independent_frontiers(
    plan: AnalysisPlan,
    data: NormalizedData,
    spec: ComparisonSpec,
    targets: list[_ComparisonTarget],
    profile_id: str,
) -> list[IndependentFrontier]:
    if not targets:
        return []
    profile = plan.design.independence_profiles[profile_id]

    def support(target: _ComparisonTarget) -> frozenset[str]:
        return frozenset(
            comparison_contrast_blocks(
                data.observations,
                data.manifest,
                profile,
                candidate=spec.candidate,
                baseline=spec.baseline,
                prompt_ids=target.prompt_ids,
                prompt_weights=np.asarray(target.weights),
            ).block_ids
        )

    if spec.hierarchy_calibration == "flat_bonferroni":
        return [
            IndependentFrontier(
                frontier_id=f"{profile_id}:{spec.id}:{target.result_id}",
                targets={
                    _comparison_target_key(spec.id, target.result_id, profile_id): support(target)
                },
                aliases={},
            )
            for target in targets
        ]

    root = next(target for target in targets if target.level is None)
    levels: list[tuple[str, list[_ComparisonTarget]]] = [("root", [root])]
    for level in dict.fromkeys(target.level for target in targets if target.level is not None):
        assert level is not None
        levels.append((level, [target for target in targets if target.level == level]))

    frontiers: list[IndependentFrontier] = []
    seen: dict[
        tuple[tuple[tuple[str, float], ...], ...],
        tuple[int, dict[tuple[tuple[str, float], ...], str]],
    ] = {}
    for level, level_targets in levels:
        atom_sets = [frozenset(target.prompt_ids) for target in level_targets]
        if set().union(*atom_sets) != set(root.prompt_ids):
            raise ValueError(
                f"comparison hierarchy level {spec.id}:{level} does not partition the group"
            )
        if any(
            left & right for index, left in enumerate(atom_sets) for right in atom_sets[index + 1 :]
        ):
            raise ValueError(
                f"comparison hierarchy level {spec.id}:{level} contains overlapping nodes"
            )
        entries = [
            (
                _comparison_target_key(spec.id, target.result_id, profile_id),
                target.prompt_ids,
                target.weights,
            )
            for target in level_targets
        ]
        signature = _frontier_signature(entries)
        if signature in seen:
            frontier_index, canonical_by_signature = seen[signature]
            frontier = frontiers[frontier_index]
            aliases = dict(frontier.aliases)
            for target_id, prompt_ids, weights in entries:
                target_signature = tuple(
                    sorted(
                        (prompt_id, round(weight, 14))
                        for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                    )
                )
                canonical_id = canonical_by_signature[target_signature]
                aliases[canonical_id] = (*aliases[canonical_id], target_id)
            frontiers[frontier_index] = IndependentFrontier(
                frontier_id=f"{frontier.frontier_id}+{level}",
                targets=frontier.targets,
                aliases=aliases,
            )
            continue
        target_by_id = {
            _comparison_target_key(spec.id, target.result_id, profile_id): target
            for target in level_targets
        }
        targets_by_id = {target_id: support(target_by_id[target_id]) for target_id, *_ in entries}
        overlapping = any(
            left & right
            for index, left in enumerate(targets_by_id.values())
            for right in list(targets_by_id.values())[index + 1 :]
        )
        if overlapping:
            for target_id, atoms in targets_by_id.items():
                frontiers.append(
                    IndependentFrontier(
                        frontier_id=f"{profile_id}:{spec.id}:{level}:dependent:{target_id}",
                        targets={target_id: atoms},
                        aliases={target_id: (target_id,)},
                    )
                )
            continue
        aliases = {target_id: (target_id,) for target_id in targets_by_id}
        canonical_by_signature = {
            tuple(
                sorted(
                    (prompt_id, round(weight, 14))
                    for prompt_id, weight in zip(prompt_ids, weights, strict=True)
                )
            ): target_id
            for target_id, prompt_ids, weights in entries
        }
        seen[signature] = (len(frontiers), canonical_by_signature)
        frontiers.append(
            IndependentFrontier(
                frontier_id=f"{profile_id}:{spec.id}:{level}",
                targets=targets_by_id,
                aliases=aliases,
            )
        )
    return frontiers


def _comparison_calibrations(
    plan: AnalysisPlan,
    data: NormalizedData,
    targets: dict[str, list[_ComparisonTarget]],
    diagnostics: list[Diagnostic],
) -> dict[str, _EndpointCalibration]:
    families = {family.id: family for family in plan.multiplicity.families}
    result: dict[str, _EndpointCalibration] = {}
    simultaneous_by_family: dict[str, list[ComparisonSpec]] = {}
    for spec in plan.expanded_comparisons():
        if not spec.family:
            continue
        family = families.get(spec.family)
        if family is None:
            continue
        directional_alpha, _, _ = _coverage_floors(family.alpha, spec.coverage_mode)
        assert directional_alpha is not None
        assert spec.independence is not None
        if not spec.simultaneous:
            for profile_id in [
                spec.independence.primary,
                *spec.independence.sensitivities,
            ]:
                for target in targets.get(spec.id, []):
                    result[_comparison_target_key(spec.id, target.result_id, profile_id)] = (
                        _EndpointCalibration(directional_alpha, "pointwise")
                    )
        else:
            simultaneous_by_family.setdefault(family.id, []).append(spec)

    for family_id, specs in simultaneous_by_family.items():
        family = families[family_id]
        directional_alpha = min(
            cast(float, _coverage_floors(family.alpha, spec.coverage_mode)[0]) for spec in specs
        )
        profile_ids = sorted(
            {
                profile_id
                for spec in specs
                for profile_id in [
                    cast(Any, spec.independence).primary,
                    *cast(Any, spec.independence).sensitivities,
                ]
            }
        )
        for profile_id in profile_ids:
            frontiers: list[IndependentFrontier] = []
            frontier_weights: dict[str, float] = {}
            frontier_aliases: dict[tuple[tuple[Any, ...], tuple[tuple[str, ...], ...]], int] = {}
            try:
                for spec in specs:
                    assert spec.independence is not None
                    if profile_id not in {
                        spec.independence.primary,
                        *spec.independence.sensitivities,
                    }:
                        continue
                    compiled = _comparison_independent_frontiers(
                        plan, data, spec, targets.get(spec.id, []), profile_id
                    )
                    for frontier in compiled:
                        allocation_weight = (
                            spec.allocation_weight
                            if family.method == "weighted_bonferroni"
                            else 1.0
                        )
                        if not (
                            spec.method == "weighted_kl_chernoff"
                            and spec.hierarchy_calibration == "frontier_sidak"
                        ):
                            frontiers.append(frontier)
                            frontier_weights[frontier.frontier_id] = allocation_weight
                            continue
                        frontier_level = frontier.frontier_id.split(":", maxsplit=1)[-1]
                        frontier = IndependentFrontier(
                            frontier_id=(
                                f"{profile_id}:two_arm_kl:{spec.candidate}:{spec.baseline}:"
                                f"{spec.group}:{spec.measure}:{frontier_level}"
                            ),
                            targets=frontier.targets,
                            aliases=frontier.aliases,
                        )
                        atom_signature = tuple(
                            sorted(
                                tuple(sorted(atom_ids)) for atom_ids in frontier.targets.values()
                            )
                        )
                        alias_key = (
                            _comparison_region_alias_key(spec, allocation_weight=allocation_weight),
                            atom_signature,
                        )
                        if alias_key not in frontier_aliases:
                            frontier_aliases[alias_key] = len(frontiers)
                            frontiers.append(frontier)
                            frontier_weights[frontier.frontier_id] = allocation_weight
                            continue
                        existing_index = frontier_aliases[alias_key]
                        existing = frontiers[existing_index]
                        canonical_by_atoms = {
                            atoms: target_id for target_id, atoms in existing.targets.items()
                        }
                        aliases = dict(existing.aliases)
                        for target_id, atoms in frontier.targets.items():
                            canonical_id = canonical_by_atoms[atoms]
                            aliases[canonical_id] = (
                                *aliases.get(canonical_id, (canonical_id,)),
                                *frontier.aliases.get(target_id, (target_id,)),
                            )
                        frontiers[existing_index] = IndependentFrontier(
                            frontier_id=existing.frontier_id,
                            targets=existing.targets,
                            aliases=aliases,
                        )
                calibrated = calibrate_independent_frontiers(
                    frontiers,
                    directional_family_alpha=directional_alpha,
                    frontier_weights=frontier_weights,
                )
            except ValueError as exc:
                diagnostics.append(
                    Diagnostic(
                        code="COMPARISON_CALIBRATION_INVALID",
                        severity="ERROR",
                        message=str(exc),
                        scope=f"{family_id}:{profile_id}",
                    )
                )
                continue
            for target_id, item in calibrated.items():
                method = (
                    "frontier_sidak"
                    if any(spec.hierarchy_calibration == "frontier_sidak" for spec in specs)
                    else "flat_bonferroni"
                )
                result[target_id] = _EndpointCalibration(
                    endpoint_alpha=item.endpoint_alpha,
                    method=method,
                    frontier_id=item.frontier_id,
                    frontier_alpha=item.frontier_alpha,
                    frontier_size=item.frontier_size,
                )
    return result


def _bounded_comparison_mean(
    method: str,
    blocks: BoundedBlocks,
    *,
    alpha_lower: float,
    alpha_upper: float,
    profile_id: str,
) -> DirectMeanBounds:
    function = {
        "weighted_hoeffding": weighted_bounded_hoeffding_mean_bounds,
        "weighted_empirical_bernstein": weighted_bounded_empirical_bernstein_mean_bounds,
        "weighted_kl_chernoff": weighted_bounded_kl_chernoff_mean_bounds,
    }[method]
    return function(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile=profile_id,
    )


def _difference_from_blocks(
    method: str,
    blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
) -> DirectComparisonBounds:
    bounds = _bounded_comparison_mean(
        method,
        blocks,
        alpha_lower=alpha,
        alpha_upper=alpha,
        profile_id=profile_id,
    )
    return DirectComparisonBounds(
        lower=max(-1.0, 2 * bounds.lower - 1),
        upper=min(1.0, 2 * bounds.upper - 1),
        completion_estimate_lower=2 * bounds.completion_estimate_lower - 1,
        completion_estimate_upper=2 * bounds.completion_estimate_upper - 1,
        half_width_lower=2 * bounds.half_width_lower,
        half_width_upper=2 * bounds.half_width_upper,
        concentration_variance=4 * bounds.concentration_variance,
        effective_sample_size=bounds.effective_sample_size,
        alpha_lower=alpha,
        alpha_upper=alpha,
        method=method,
        independence_unit=profile_id,
        variance_upper_lower=(
            4 * bounds.variance_upper_lower if bounds.variance_upper_lower is not None else None
        ),
        variance_upper_upper=(
            4 * bounds.variance_upper_upper if bounds.variance_upper_upper is not None else None
        ),
        maximum_increment=(
            2 * bounds.maximum_increment if bounds.maximum_increment is not None else None
        ),
        chernoff_kernel=bounds.chernoff_kernel,
    )


def _difference_rectangle_from_blocks(
    method: str,
    candidate_blocks: BoundedBlocks,
    baseline_blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
) -> DirectComparisonBounds:
    candidate = _bounded_comparison_mean(
        method,
        candidate_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    baseline = _bounded_comparison_mean(
        method,
        baseline_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    completion_lower = candidate.completion_estimate_lower - baseline.completion_estimate_upper
    completion_upper = candidate.completion_estimate_upper - baseline.completion_estimate_lower
    lower = max(-1.0, candidate.lower - baseline.upper)
    upper = min(1.0, candidate.upper - baseline.lower)
    return DirectComparisonBounds(
        lower=lower,
        upper=upper,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=max(0.0, completion_lower - lower),
        half_width_upper=max(0.0, upper - completion_upper),
        concentration_variance=(candidate.concentration_variance + baseline.concentration_variance),
        effective_sample_size=min(candidate.effective_sample_size, baseline.effective_sample_size),
        alpha_lower=alpha,
        alpha_upper=alpha,
        method=method,
        independence_unit=profile_id,
        maximum_increment=max(
            candidate.maximum_increment or 0.0, baseline.maximum_increment or 0.0
        ),
        chernoff_kernel=(
            candidate.chernoff_kernel
            if candidate.chernoff_kernel == baseline.chernoff_kernel
            else "two_arm_mixed"
        ),
        candidate_lower=candidate.lower,
        candidate_upper=candidate.upper,
        baseline_lower=baseline.lower,
        baseline_upper=baseline.upper,
    )


def _ratio_from_blocks(
    method: str,
    candidate_blocks: BoundedBlocks,
    baseline_blocks: BoundedBlocks,
    *,
    alpha: float,
    profile_id: str,
    estimand: Literal["risk_ratio", "relative_risk_reduction"],
) -> RatioComparisonBounds:
    candidate = _bounded_comparison_mean(
        method,
        candidate_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    baseline = _bounded_comparison_mean(
        method,
        baseline_blocks,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        profile_id=profile_id,
    )
    candidate_estimate = candidate.completion_estimate_lower
    baseline_estimate = baseline.completion_estimate_lower
    ratio_estimate = candidate_estimate / baseline_estimate if baseline_estimate > 0 else None
    ratio_lower = candidate.lower / baseline.upper if baseline.upper > 0 else None
    ratio_upper = candidate.upper / baseline.lower if baseline.lower > 0 else None
    warning = None
    if ratio_estimate is None:
        warning = "The observed baseline risk is zero, so the ratio estimate is undefined."
    if ratio_upper is None:
        warning = "The baseline lower bound is zero, so no finite upper ratio bound exists."
    if estimand == "relative_risk_reduction":
        estimate = 1 - ratio_estimate if ratio_estimate is not None else None
        lower = 1 - ratio_upper if ratio_upper is not None else None
        upper = 1 - ratio_lower if ratio_lower is not None else None
    else:
        estimate, lower, upper = ratio_estimate, ratio_lower, ratio_upper
    return RatioComparisonBounds(
        candidate_estimate=candidate_estimate,
        baseline_estimate=baseline_estimate,
        estimate=estimate,
        lower=lower,
        upper=upper,
        candidate_bounds=candidate,
        baseline_bounds=baseline,
        method=method,
        independence_unit=profile_id,
        estimand=estimand,
        warning=warning,
    )


def _comparison_interval(
    *,
    bounds: DirectComparisonBounds | RatioComparisonBounds,
    spec: ComparisonSpec,
    method: str,
    role: Literal["primary", "reference", "sensitivity"],
    independence_profile: str,
    independent_block_count: int,
    profile_block_by: list[str],
    comparison_coupling: Literal["independent", "joint"],
    family_alpha: float,
    calibration: _EndpointCalibration,
) -> RiskIntervalResult:
    directional_family_alpha, one_sided, joint = _coverage_floors(family_alpha, spec.coverage_mode)
    lower = bounds.lower
    upper = bounds.upper
    is_ratio = isinstance(bounds, RatioComparisonBounds)
    uses_two_arm_region = is_ratio or (
        isinstance(bounds, DirectComparisonBounds) and bounds.candidate_lower is not None
    )
    candidate_lower: float | None
    candidate_upper: float | None
    baseline_lower: float | None
    baseline_upper: float | None
    if profile_block_by == ["rollout"]:
        assumptions = [
            (
                "Attempted rollout outcomes are independent within each arm; the rectangular "
                "two-arm construction does not require independence between arms."
                if uses_two_arm_region
                else "Attempted rollout outcomes are independent within and across comparison arms."
            ),
            "The aligned prompts, design weights, and arm rollout counts are fixed "
            "independently of observed outcomes.",
        ]
    else:
        assumptions = [
            f"Blocks defined by {profile_block_by!r} are mutually independent.",
            "Arbitrary dependence is allowed within each declared block, with cross-arm "
            f"coupling set to {comparison_coupling!r}.",
            "The block partition, aligned prompts, and design weights are fixed independently "
            "of observed outcomes.",
        ]
    if is_ratio:
        assert isinstance(bounds, RatioComparisonBounds)
        assumptions.append(
            "Ratio bounds use a simultaneous rectangular confidence set for the two arm means."
        )
        interval_warnings = [bounds.warning] if bounds.warning else []
        completion_lower = completion_upper = None
        half_width_lower = half_width_upper = None
        effective_sample_size = min(
            bounds.candidate_bounds.effective_sample_size,
            bounds.baseline_bounds.effective_sample_size,
        )
        concentration_variance = variance_lower = variance_upper = maximum_increment = None
        chernoff_kernel = bounds.candidate_bounds.chernoff_kernel
        candidate_lower = bounds.candidate_bounds.lower
        candidate_upper = bounds.candidate_bounds.upper
        baseline_lower = bounds.baseline_bounds.lower
        baseline_upper = bounds.baseline_bounds.upper
    else:
        assert isinstance(bounds, DirectComparisonBounds)
        interval_warnings = []
        completion_lower = bounds.completion_estimate_lower
        completion_upper = bounds.completion_estimate_upper
        half_width_lower = bounds.half_width_lower
        half_width_upper = bounds.half_width_upper
        effective_sample_size = bounds.effective_sample_size
        concentration_variance = bounds.concentration_variance
        variance_lower = bounds.variance_upper_lower
        variance_upper = bounds.variance_upper_upper
        maximum_increment = bounds.maximum_increment
        chernoff_kernel = bounds.chernoff_kernel
        candidate_lower = bounds.candidate_lower
        candidate_upper = bounds.candidate_upper
        baseline_lower = bounds.baseline_lower
        baseline_upper = bounds.baseline_upper
        if uses_two_arm_region:
            assumptions.append(
                "The risk-difference bound uses a simultaneous rectangular confidence set "
                "for the two sparse arm means."
            )
    assumptions.append("Every unclassified attempted outcome has a latent binary safety label.")
    if calibration.method == "frontier_sidak":
        assumptions.extend(
            [
                "Comparison siblings in a frontier use disjoint prompt blocks from both arms.",
                "Independent sibling blocks use exact Sidak product calibration.",
                "Potentially dependent comparison frontiers use Bonferroni allocation.",
            ]
        )
    eligibility: Eligibility = (
        "certifying" if lower is not None and upper is not None else "invalid"
    )
    scope_prefix = (
        "simultaneous_frontier_sidak"
        if calibration.method == "frontier_sidak"
        else "simultaneous"
        if spec.simultaneous
        else "pointwise"
    )
    return RiskIntervalResult(
        interval_id=(
            f"direct_{method}"
            if spec.independence is not None and independence_profile == spec.independence.primary
            else f"direct_{method}_{independence_profile}"
        ),
        lower=lower,
        upper=upper,
        method=method,
        role=role,
        coverage_scope=(scope_prefix + f"_{independence_profile}_fixed_design_comparison_family"),
        coverage_mode=spec.coverage_mode,
        family=spec.family,
        family_alpha=family_alpha,
        directional_family_alpha=directional_family_alpha,
        alpha_lower=calibration.endpoint_alpha,
        alpha_upper=calibration.endpoint_alpha,
        simultaneous_coverage=joint if spec.simultaneous else None,
        lower_one_sided_coverage=one_sided,
        upper_one_sided_coverage=one_sided,
        completion_estimate_lower=completion_lower,
        completion_estimate_upper=completion_upper,
        half_width_lower=half_width_lower,
        half_width_upper=half_width_upper,
        effective_sample_size=effective_sample_size,
        concentration_variance=concentration_variance,
        variance_upper_lower=variance_lower,
        variance_upper_upper=variance_upper,
        maximum_increment=maximum_increment,
        chernoff_kernel=chernoff_kernel,
        calibration_method=calibration.method,
        calibration_frontier=calibration.frontier_id,
        calibration_frontier_alpha=calibration.frontier_alpha,
        calibration_frontier_size=calibration.frontier_size,
        candidate_lower=candidate_lower,
        candidate_upper=candidate_upper,
        baseline_lower=baseline_lower,
        baseline_upper=baseline_upper,
        independence_profile=independence_profile,
        independent_block_count=independent_block_count,
        missingness_method="adverse_arm_specific_completion",
        assumptions=assumptions,
        eligibility=eligibility,
        warnings=interval_warnings,
    )


def _paired_comparison_diagnostics(
    spec: ComparisonSpec,
    data: NormalizedData,
    prompt_ids: set[str],
) -> tuple[int | None, dict[str, int], float | None, int | None, dict[str, Any]]:
    if not spec.pair_key or not spec.diagnostic_methods:
        return None, {}, None, None, {}
    arm_rows = {
        spec.candidate: [
            row
            for row in data.observations
            if row.cohort_id == spec.candidate
            and row.prompt_id in prompt_ids
            and row.failure is not None
        ],
        spec.baseline: [
            row
            for row in data.observations
            if row.cohort_id == spec.baseline
            and row.prompt_id in prompt_ids
            and row.failure is not None
        ],
    }
    maps: dict[str, dict[tuple[Any, ...], Observation]] = {}
    for arm, rows in arm_rows.items():
        mapped: dict[tuple[Any, ...], Observation] = {}
        for row in rows:
            key = tuple(_observation_attr(row, field) for field in spec.pair_key or [])
            if key in mapped:
                raise ValueError(f"duplicate pair key in {arm}: {key!r}")
            mapped[key] = row
        maps[arm] = mapped
    candidate_keys = set(maps[spec.candidate])
    baseline_keys = set(maps[spec.baseline])
    if candidate_keys != baseline_keys:
        raise ValueError(
            f"broken pairing: {len(candidate_keys - baseline_keys)} candidate-only and "
            f"{len(baseline_keys - candidate_keys)} baseline-only pairs"
        )
    keys = sorted(candidate_keys, key=repr)
    cand = [bool(maps[spec.candidate][key].failure) for key in keys]
    base = [bool(maps[spec.baseline][key].failure) for key in keys]
    mcnemar = exact_mcnemar(cand, base)
    discordant = {
        "candidate_safe_baseline_failure": mcnemar.candidate_safe_baseline_failure,
        "candidate_failure_baseline_safe": mcnemar.candidate_failure_baseline_safe,
    }
    outputs: dict[str, Any] = {}
    p_value = None
    if "exact_mcnemar" in spec.diagnostic_methods:
        p_value = mcnemar.p_value_two_sided
        outputs["exact_mcnemar"] = {
            "estimate": mcnemar.estimate,
            "p_value_two_sided": mcnemar.p_value_two_sided,
            "p_value_candidate_worse": mcnemar.p_value_candidate_worse,
            "p_value_candidate_better": mcnemar.p_value_candidate_better,
        }
    if "newcombe_paired" in spec.diagnostic_methods:
        paired_interval = newcombe_paired_interval(cand, base)
        outputs["newcombe_paired"] = {
            "estimate": paired_interval.estimate,
            "lower": paired_interval.lower,
            "upper": paired_interval.upper,
            "alpha_lower": paired_interval.alpha_lower,
            "alpha_upper": paired_interval.alpha_upper,
            "method": paired_interval.method,
            "label": "diagnostic_requires_independent_pairs",
        }
    cluster_count = None
    if "fixed_cluster_bootstrap" in spec.diagnostic_methods:
        if not spec.cluster_key:
            raise ValueError("fixed_cluster_bootstrap requires cluster_key")
        differences = [float(c) - float(b) for c, b in zip(cand, base, strict=True)]
        clusters = [_observation_attr(maps[spec.candidate][key], spec.cluster_key) for key in keys]
        boot = paired_cluster_bootstrap(
            differences,
            clusters,
            replicates=spec.bootstrap_replicates,
            seed=spec.bootstrap_seed,
        )
        cluster_count = boot.cluster_count
        outputs["fixed_cluster_bootstrap"] = {
            "estimate": boot.estimate,
            "lower": boot.lower,
            "upper": boot.upper,
            "minimum_cluster_size": boot.minimum_cluster_size,
            "maximum_cluster_size": boot.maximum_cluster_size,
            "cluster_size_ratio": boot.cluster_size_ratio,
            "maximum_leave_one_cluster_out_shift": boot.maximum_leave_one_cluster_out_shift,
            "replicates": boot.replicates,
            "seed": boot.seed,
            "label": "diagnostic_fixed_design_resampling",
        }
    return len(keys), discordant, p_value, cluster_count, outputs


def _project_comparison_hierarchy_regions(
    plan: AnalysisPlan,
    target_map: dict[str, list[_ComparisonTarget]],
    results: list[ComparisonResult],
    diagnostics: list[Diagnostic],
) -> None:
    specs = {spec.id: spec for spec in plan.expanded_comparisons()}
    for comparison_id, targets in target_map.items():
        spec = specs[comparison_id]
        if (
            not spec.hierarchy_nodes
            or spec.hierarchy_region != "linear_program"
            or not targets
        ):
            continue
        rows = [row for row in results if row.comparison_id == comparison_id]
        result_by_id = {cast(str, row.result_id): row for row in rows}
        prompt_ids = sorted({prompt for target in targets for prompt in target.prompt_ids})
        prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
        coefficients: dict[str, np.ndarray] = {}
        intervals_by_target: dict[str, list[RiskIntervalResult]] = {}
        for target in targets:
            vector = np.zeros(len(prompt_ids), dtype=float)
            for prompt_id, weight in zip(target.prompt_ids, target.weights, strict=True):
                vector[prompt_index[prompt_id]] = weight
            coefficients[target.result_id] = vector
            intervals_by_target[target.result_id] = result_by_id[target.result_id].intervals

        method_units = {
            (interval.method, interval.independence_profile)
            for intervals in intervals_by_target.values()
            for interval in intervals
            if interval.independence_profile is not None
        }
        for method, independence_unit in sorted(method_units):
            projected: dict[str, tuple[float | None, float | None]]
            projection_feasible: bool
            projection_message: str | None
            projection_constraint_count: int
            interval_by_target = {
                target_id: interval
                for target_id, intervals in intervals_by_target.items()
                if (interval := _direct_interval_for_region(intervals, method, independence_unit))
                is not None
            }
            if not interval_by_target:
                continue
            two_arm_intervals = {
                target_id: interval
                for target_id, interval in interval_by_target.items()
                if interval.candidate_lower is not None
                and interval.candidate_upper is not None
                and interval.baseline_lower is not None
                and interval.baseline_upper is not None
            }
            if spec.estimand == "risk_difference" and two_arm_intervals:
                candidate_constraints = [
                    LinearIntervalConstraint(
                        target_id=target_id,
                        coefficients=coefficients[target_id],
                        lower=cast(float, interval.candidate_lower),
                        upper=cast(float, interval.candidate_upper),
                    )
                    for target_id, interval in two_arm_intervals.items()
                ]
                baseline_constraints = [
                    LinearIntervalConstraint(
                        target_id=target_id,
                        coefficients=coefficients[target_id],
                        lower=cast(float, interval.baseline_lower),
                        upper=cast(float, interval.baseline_upper),
                    )
                    for target_id, interval in two_arm_intervals.items()
                ]
                difference_projection = project_two_arm_difference_hierarchy_confidence_region(
                    candidate_constraints,
                    baseline_constraints,
                    {target_id: coefficients[target_id] for target_id in two_arm_intervals},
                    variable_count=len(prompt_ids),
                )
                projected = {
                    target_id: endpoints
                    for target_id, endpoints in difference_projection.intervals.items()
                }
                projection_feasible = difference_projection.feasible
                projection_message = difference_projection.message
                projection_constraint_count = difference_projection.constraint_count
                projection_method = "comparison_two_arm_difference_hierarchy_linear_program"
                interval_by_target = two_arm_intervals
            elif spec.estimand == "risk_difference":
                constraints = [
                    LinearIntervalConstraint(
                        target_id=target_id,
                        coefficients=coefficients[target_id],
                        lower=cast(float, interval.lower),
                        upper=cast(float, interval.upper),
                    )
                    for target_id, interval in interval_by_target.items()
                ]
                contrast_projection = project_contrast_hierarchy_confidence_region(
                    constraints,
                    {target_id: coefficients[target_id] for target_id in interval_by_target},
                    variable_count=len(prompt_ids),
                )
                projected = {
                    target_id: endpoints
                    for target_id, endpoints in contrast_projection.intervals.items()
                }
                projection_feasible = contrast_projection.feasible
                projection_message = contrast_projection.message
                projection_constraint_count = contrast_projection.constraint_count
                projection_method = "comparison_contrast_hierarchy_linear_program"
            else:
                ratio_intervals = two_arm_intervals
                if not ratio_intervals:
                    continue
                candidate_constraints = [
                    LinearIntervalConstraint(
                        target_id=target_id,
                        coefficients=coefficients[target_id],
                        lower=cast(float, interval.candidate_lower),
                        upper=cast(float, interval.candidate_upper),
                    )
                    for target_id, interval in ratio_intervals.items()
                ]
                baseline_constraints = [
                    LinearIntervalConstraint(
                        target_id=target_id,
                        coefficients=coefficients[target_id],
                        lower=cast(float, interval.baseline_lower),
                        upper=cast(float, interval.baseline_upper),
                    )
                    for target_id, interval in ratio_intervals.items()
                ]
                ratio_projection = project_ratio_hierarchy_confidence_region(
                    candidate_constraints,
                    baseline_constraints,
                    {target_id: coefficients[target_id] for target_id in ratio_intervals},
                    variable_count=len(prompt_ids),
                )
                projection_feasible = ratio_projection.feasible
                projection_message = ratio_projection.message
                projection_constraint_count = ratio_projection.constraint_count
                if spec.estimand == "risk_ratio":
                    projected = ratio_projection.intervals
                    projection_method = "comparison_ratio_hierarchy_linear_fractional"
                else:
                    projected = {
                        target_id: (
                            1 - upper if upper is not None else None,
                            1 - lower if lower is not None else None,
                        )
                        for target_id, (lower, upper) in ratio_projection.intervals.items()
                    }
                    projection_method = "comparison_relative_reduction_hierarchy_linear_fractional"
                interval_by_target = ratio_intervals

            if not projection_feasible:
                diagnostics.append(
                    Diagnostic(
                        code="COMPARISON_HIERARCHY_CONFIDENCE_REGION_EMPTY",
                        severity="WARNING",
                        message=(
                            "The raw simultaneous comparison intervals have no common "
                            "parameter vector under the declared hierarchy."
                        ),
                        scope=f"{comparison_id}:{method}:{independence_unit}",
                        details={
                            "solver_message": projection_message,
                            "constraint_count": projection_constraint_count,
                        },
                    )
                )
                continue
            for target_id, (lower, upper) in projected.items():
                interval = interval_by_target[target_id]
                interval.raw_lower = interval.lower
                interval.raw_upper = interval.upper
                if lower is not None:
                    interval.lower = (
                        max(interval.lower, lower) if interval.lower is not None else lower
                    )
                if upper is not None:
                    interval.upper = (
                        min(interval.upper, upper) if interval.upper is not None else upper
                    )
                interval.projection_method = projection_method
                interval.projection_constraint_count = projection_constraint_count
                interval.coverage_scope = f"{interval.coverage_scope}_hierarchy_region"
                interval.assumptions.append(
                    "Reported comparison endpoints are projections of one common "
                    "fixed-design hierarchy confidence region."
                )

        for result in rows:
            primary = next(
                (interval for interval in result.intervals if interval.role == "primary"),
                None,
            )
            if primary is None:
                continue
            result.lower = primary.lower
            result.upper = primary.upper
            result.raw_lower = primary.raw_lower
            result.raw_upper = primary.raw_upper
            result.projection_method = primary.projection_method
            result.coverage_scope = primary.coverage_scope


def _multi_arm_node_key(level: str | None, path: dict[str, str]) -> str:
    if level is None:
        return "root"
    return level + "::" + "/".join(f"{key}={value}" for key, value in path.items())


def _multi_arm_effect_interval(
    estimand: str,
    candidate: tuple[float, float],
    baseline: tuple[float, float],
) -> tuple[float | None, float | None]:
    candidate_lower, candidate_upper = candidate
    baseline_lower, baseline_upper = baseline
    if estimand == "risk_difference":
        return (
            max(-1.0, candidate_lower - baseline_upper),
            min(1.0, candidate_upper - baseline_lower),
        )
    ratio_lower = candidate_lower / baseline_upper if baseline_upper > 0 else None
    ratio_upper = candidate_upper / baseline_lower if baseline_lower > 0 else None
    if estimand == "risk_ratio":
        return ratio_lower, ratio_upper
    return (
        1 - ratio_upper if ratio_upper is not None else None,
        1 - ratio_lower if ratio_lower is not None else None,
    )


def _project_multi_arm_comparison_regions(
    plan: AnalysisPlan,
    target_map: dict[str, list[_ComparisonTarget]],
    results: list[ComparisonResult],
    diagnostics: list[Diagnostic],
) -> None:
    """Project comparison-set results from one shared arm-risk region.

    Raw arm constraints are already calibrated simultaneously by the comparison family. Their
    intersection therefore remains a simultaneous confidence region. Each arm is projected over
    its hierarchy once, and every requested pairwise effect is derived from those common arm
    projections. This makes all displayed contrasts coherent with the same arm intervals.
    """

    for set_spec in plan.comparison_sets:
        set_rows = [row for row in results if row.comparison_set_id == set_spec.id]
        if not set_rows:
            continue
        interval_keys = sorted(
            {
                (interval.method, interval.independence_profile)
                for row in set_rows
                for interval in row.intervals
                if interval.independence_profile is not None
                and interval.candidate_lower is not None
                and interval.candidate_upper is not None
                and interval.baseline_lower is not None
                and interval.baseline_upper is not None
            }
        )
        for method, profile_id in interval_keys:
            assert profile_id is not None
            row_intervals: dict[str, RiskIntervalResult] = {}
            target_by_row: dict[str, _ComparisonTarget] = {}
            prompt_ids: set[str] = set()
            for row in set_rows:
                interval = _direct_interval_for_region(row.intervals, method, profile_id)
                if interval is None or row.result_id is None:
                    continue
                targets = {
                    target.result_id: target for target in target_map.get(row.comparison_id, [])
                }
                target = targets.get(row.result_id)
                if target is None:
                    continue
                row_intervals[row.result_id] = interval
                target_by_row[row.result_id] = target
                prompt_ids.update(target.prompt_ids)
            if not row_intervals or not prompt_ids:
                continue
            ordered_prompts = sorted(prompt_ids)
            prompt_index = {prompt_id: index for index, prompt_id in enumerate(ordered_prompts)}
            node_coefficients: dict[str, np.ndarray] = {}
            for row in set_rows:
                if row.result_id not in target_by_row:
                    continue
                target = target_by_row[row.result_id]
                node_key = _multi_arm_node_key(row.level, row.path)
                vector = np.zeros(len(ordered_prompts), dtype=float)
                for prompt_id, weight in zip(target.prompt_ids, target.weights, strict=True):
                    vector[prompt_index[prompt_id]] = weight
                node_coefficients[node_key] = vector

            arm_projections: dict[str, dict[str, tuple[float, float]]] = {}
            total_constraint_count = 0
            failed = False
            for arm in set_spec.cohorts:
                constraints: list[LinearIntervalConstraint] = []
                for row in set_rows:
                    if row.result_id is None:
                        continue
                    interval = row_intervals.get(row.result_id)
                    if interval is None:
                        continue
                    node_key = _multi_arm_node_key(row.level, row.path)
                    if row.candidate_cohort == arm:
                        lower, upper = interval.candidate_lower, interval.candidate_upper
                    elif row.baseline_cohort == arm:
                        lower, upper = interval.baseline_lower, interval.baseline_upper
                    else:
                        continue
                    if lower is None or upper is None:
                        continue
                    constraints.append(
                        LinearIntervalConstraint(
                            target_id=f"{row.result_id}::{arm}",
                            coefficients=node_coefficients[node_key],
                            lower=lower,
                            upper=upper,
                        )
                    )
                if not constraints:
                    failed = True
                    break
                projection = project_hierarchy_confidence_region(
                    constraints,
                    cast(dict[str, Any], node_coefficients),
                    variable_count=len(ordered_prompts),
                )
                if not projection.feasible:
                    diagnostics.append(
                        Diagnostic(
                            code="MULTI_ARM_CONFIDENCE_REGION_EMPTY",
                            severity="WARNING",
                            message=(
                                "The jointly calibrated arm constraints have no common "
                                "parameter vector under the declared hierarchy."
                            ),
                            scope=f"{set_spec.id}:{arm}:{method}:{profile_id}",
                            details={
                                "solver_message": projection.message,
                                "constraint_count": projection.constraint_count,
                            },
                        )
                    )
                    failed = True
                    break
                arm_projections[arm] = projection.intervals
                total_constraint_count += projection.constraint_count
            if failed:
                continue

            for row in set_rows:
                if row.result_id is None:
                    continue
                interval = row_intervals.get(row.result_id)
                if interval is None:
                    continue
                node_key = _multi_arm_node_key(row.level, row.path)
                candidate = arm_projections[row.candidate_cohort][node_key]
                baseline = arm_projections[row.baseline_cohort][node_key]
                lower, upper = _multi_arm_effect_interval(row.estimand, candidate, baseline)
                if interval.raw_lower is None:
                    interval.raw_lower = interval.lower
                if interval.raw_upper is None:
                    interval.raw_upper = interval.upper
                interval.candidate_lower, interval.candidate_upper = candidate
                interval.baseline_lower, interval.baseline_upper = baseline
                interval.lower, interval.upper = lower, upper
                interval.projection_method = "multi_arm_hierarchy_linear_region"
                interval.projection_constraint_count = total_constraint_count
                if "_multi_arm_region" not in cast(str, interval.coverage_scope):
                    interval.coverage_scope = f"{interval.coverage_scope}_multi_arm_region"
                assumption = (
                    "All comparison-set effects are projections of one jointly calibrated "
                    "multi-arm fixed-design confidence region."
                )
                if assumption not in interval.assumptions:
                    interval.assumptions.append(assumption)

        for row in set_rows:
            primary = next(
                (interval for interval in row.intervals if interval.role == "primary"),
                None,
            )
            if primary is None:
                continue
            row.lower = primary.lower
            row.upper = primary.upper
            row.raw_lower = primary.raw_lower
            row.raw_upper = primary.raw_upper
            row.projection_method = primary.projection_method
            row.coverage_scope = primary.coverage_scope


def _comparisons(
    plan: AnalysisPlan,
    data: NormalizedData,
    estimates: list[PromptEstimate],
    diagnostics: list[Diagnostic],
) -> list[ComparisonResult]:
    estimate_index = {(item.cohort_id, item.prompt_id): item for item in estimates}
    families = {family.id: family for family in plan.multiplicity.families}
    target_map: dict[str, list[_ComparisonTarget]] = {}
    expanded_comparisons = plan.expanded_comparisons()
    for spec in expanded_comparisons:
        try:
            target_map[spec.id] = _comparison_targets(plan, data, spec)
        except StopIteration:
            target_map[spec.id] = []
    comparison_calibrations = _comparison_calibrations(plan, data, target_map, diagnostics)
    results: list[ComparisonResult] = []
    for spec in expanded_comparisons:
        family = families.get(spec.family or "")
        if family is None:
            continue
        directional_family_alpha, one_sided, joint = _coverage_floors(
            family.alpha, spec.coverage_mode
        )
        assert directional_family_alpha is not None
        assert spec.independence is not None
        for target in target_map[spec.id]:
            candidate_items = [
                estimate_index[(spec.candidate, prompt_id)] for prompt_id in target.prompt_ids
            ]
            baseline_items = [
                estimate_index[(spec.baseline, prompt_id)] for prompt_id in target.prompt_ids
            ]
            weights = np.asarray(target.weights, dtype=float)
            candidate_estimate = (
                float(np.dot(weights, [cast(float, item.p_hat) for item in candidate_items]))
                if all(item.p_hat is not None for item in candidate_items)
                else None
            )
            baseline_estimate = (
                float(np.dot(weights, [cast(float, item.p_hat) for item in baseline_items]))
                if all(item.p_hat is not None for item in baseline_items)
                else None
            )
            summaries: dict[str, float | None] = {
                "risk_difference": (
                    candidate_estimate - baseline_estimate
                    if candidate_estimate is not None and baseline_estimate is not None
                    else None
                ),
                "risk_ratio": (
                    candidate_estimate / baseline_estimate
                    if candidate_estimate is not None
                    and baseline_estimate is not None
                    and baseline_estimate > 0
                    else None
                ),
            }
            summaries["relative_risk_reduction"] = (
                1 - summaries["risk_ratio"] if summaries["risk_ratio"] is not None else None
            )
            intervals: list[RiskIntervalResult] = []
            warnings: list[str] = []
            blocked = any(
                item.n_total <= 0
                or _prompt_blocked(diagnostics, item.cohort_id, item.prompt_id)
                for item in [*candidate_items, *baseline_items]
            )
            if blocked:
                warnings.append("Both arms require a complete valid attempted-rollout ledger.")
            else:
                methods = [spec.method, *spec.reference_methods]
                profile_ids = [
                    spec.independence.primary,
                    *spec.independence.sensitivities,
                ]
                for profile_id in profile_ids:
                    calibration = comparison_calibrations.get(
                        _comparison_target_key(spec.id, target.result_id, profile_id)
                    )
                    if calibration is None:
                        warnings.append(
                            f"Calibration is unavailable for profile {profile_id!r}."
                        )
                        continue
                    profile = plan.design.independence_profiles[profile_id]
                    contrast_blocks = comparison_contrast_blocks(
                        data.observations,
                        data.manifest,
                        profile,
                        candidate=spec.candidate,
                        baseline=spec.baseline,
                        prompt_ids=target.prompt_ids,
                        prompt_weights=weights,
                    )
                    candidate_blocks = direct_bounded_blocks(
                        data.observations,
                        data.manifest,
                        profile,
                        cohort_id=spec.candidate,
                        prompt_ids=target.prompt_ids,
                        prompt_weights=weights,
                    )
                    baseline_blocks = direct_bounded_blocks(
                        data.observations,
                        data.manifest,
                        profile,
                        cohort_id=spec.baseline,
                        prompt_ids=target.prompt_ids,
                        prompt_weights=weights,
                    )
                    for method_index, method in enumerate(methods):
                        role: Literal["primary", "reference", "sensitivity"] = (
                            "sensitivity"
                            if profile_id != spec.independence.primary
                            else "primary"
                            if method_index == 0
                            else "reference"
                        )
                        try:
                            bounds: DirectComparisonBounds | RatioComparisonBounds
                            if spec.estimand == "risk_difference":
                                bounds = (
                                    _difference_rectangle_from_blocks(
                                        method,
                                        candidate_blocks,
                                        baseline_blocks,
                                        alpha=calibration.endpoint_alpha,
                                        profile_id=profile_id,
                                    )
                                    if method == "weighted_kl_chernoff"
                                    or spec._joint_arm_region
                                    else _difference_from_blocks(
                                        method,
                                        contrast_blocks,
                                        alpha=calibration.endpoint_alpha,
                                        profile_id=profile_id,
                                    )
                                )
                            else:
                                bounds = _ratio_from_blocks(
                                    method=method,
                                    candidate_blocks=candidate_blocks,
                                    baseline_blocks=baseline_blocks,
                                    alpha=calibration.endpoint_alpha,
                                    profile_id=profile_id,
                                    estimand=cast(
                                        Literal["risk_ratio", "relative_risk_reduction"],
                                        spec.estimand,
                                    ),
                                )
                            intervals.append(
                                _comparison_interval(
                                    bounds=bounds,
                                    spec=spec,
                                    method=method,
                                    role=role,
                                    independence_profile=profile_id,
                                    independent_block_count=contrast_blocks.count,
                                    profile_block_by=profile.block_by,
                                    comparison_coupling=profile.comparison_coupling,
                                    family_alpha=family.alpha,
                                    calibration=calibration,
                                )
                            )
                        except (TypeError, ValueError) as exc:
                            warnings.append(str(exc))
                            diagnostics.append(
                                Diagnostic(
                                    code="COMPARISON_INTERVAL_INVALID",
                                    severity="ERROR",
                                    message=str(exc),
                                    scope=target.result_id,
                                )
                            )
            primary = next((item for item in intervals if item.role == "primary"), None)
            estimate = summaries.get(spec.estimand)
            pair_count = cluster_count = None
            discordant: dict[str, int] = {}
            p_value = None
            diagnostic_outputs: dict[str, Any] = {}
            if target.level is None:
                try:
                    (
                        pair_count,
                        discordant,
                        p_value,
                        cluster_count,
                        diagnostic_outputs,
                    ) = _paired_comparison_diagnostics(spec, data, set(target.prompt_ids))
                except ValueError as exc:
                    diagnostics.append(
                        Diagnostic(
                            code="PAIR_BROKEN",
                            severity="ERROR",
                            message=str(exc),
                            scope=spec.id,
                        )
                    )
                    warnings.append(str(exc))
            results.append(
                ComparisonResult(
                    comparison_id=spec.id,
                    comparison_set_id=spec._comparison_set_id,
                    result_id=target.result_id,
                    candidate_cohort=spec.candidate,
                    baseline_cohort=spec.baseline,
                    group_id=spec.group,
                    measure_id=spec.measure,
                    estimand=spec.estimand,
                    level=target.level,
                    path=target.path,
                    candidate_estimate=candidate_estimate,
                    baseline_estimate=baseline_estimate,
                    estimate=estimate,
                    lower=primary.lower if primary is not None else None,
                    upper=primary.upper if primary is not None else None,
                    method=f"direct_fixed_design_{spec.estimand}",
                    primary_interval_id=(primary.interval_id if primary is not None else None),
                    intervals=intervals,
                    interval_method=primary.method if primary is not None else None,
                    coverage_scope=primary.coverage_scope if primary is not None else None,
                    coverage_mode=spec.coverage_mode,
                    family=family.id,
                    family_alpha=family.alpha,
                    directional_family_alpha=directional_family_alpha,
                    simultaneous_coverage=(joint if spec.simultaneous else None),
                    lower_one_sided_coverage=one_sided,
                    upper_one_sided_coverage=one_sided,
                    independence_profile=spec.independence.primary,
                    prompt_count=len(target.prompt_ids),
                    candidate_rollout_count=sum(item.n_total for item in candidate_items),
                    baseline_rollout_count=sum(item.n_total for item in baseline_items),
                    resolved_weight=target.design_weight,
                    effect_summaries=summaries,
                    pair_count=pair_count,
                    discordant_counts=discordant,
                    p_value=p_value,
                    cluster_count=cluster_count,
                    diagnostic_outputs=diagnostic_outputs,
                    eligibility=(primary.eligibility if primary is not None else "invalid"),
                    warnings=warnings + (primary.warnings if primary is not None else []),
                )
            )
    _project_comparison_hierarchy_regions(plan, target_map, results, diagnostics)
    _project_multi_arm_comparison_regions(plan, target_map, results, diagnostics)
    return results


def _comparison_set_results(
    plan: AnalysisPlan,
    comparisons: list[ComparisonResult],
    risks: list[GroupRiskResult],
) -> list[ComparisonSetResult]:
    risk_index = {(risk.cohort_id, risk.group_id, risk.measure_id): risk for risk in risks}
    results: list[ComparisonSetResult] = []
    for spec in plan.comparison_sets:
        rows = [
            row for row in comparisons if row.comparison_set_id == spec.id and row.level is None
        ]
        arm_intervals: dict[str, list[float]] = {}
        for row in rows:
            primary = next(
                (interval for interval in row.intervals if interval.role == "primary"),
                None,
            )
            if primary is None:
                continue
            for arm, lower, upper in (
                (
                    row.candidate_cohort,
                    primary.candidate_lower,
                    primary.candidate_upper,
                ),
                (
                    row.baseline_cohort,
                    primary.baseline_lower,
                    primary.baseline_upper,
                ),
            ):
                if lower is None or upper is None:
                    continue
                if arm in arm_intervals:
                    arm_intervals[arm] = [
                        max(arm_intervals[arm][0], lower),
                        min(arm_intervals[arm][1], upper),
                    ]
                else:
                    arm_intervals[arm] = [lower, upper]

        rank_intervals: dict[str, list[int]] = {}
        warnings: list[str] = []
        differences = [row for row in rows if row.estimand == "risk_difference"]
        if spec.contrasts.type == "all_pairs" and len(differences) == (
            len(spec.cohorts) * (len(spec.cohorts) - 1) // 2
        ):
            definitely_better: dict[str, set[str]] = {cohort: set() for cohort in spec.cohorts}
            definitely_worse: dict[str, set[str]] = {cohort: set() for cohort in spec.cohorts}
            for row in differences:
                if row.upper is not None and row.upper < 0:
                    definitely_better[row.candidate_cohort].add(row.baseline_cohort)
                    definitely_worse[row.baseline_cohort].add(row.candidate_cohort)
                elif row.lower is not None and row.lower > 0:
                    definitely_better[row.baseline_cohort].add(row.candidate_cohort)
                    definitely_worse[row.candidate_cohort].add(row.baseline_cohort)
            arm_count = len(spec.cohorts)
            rank_intervals = {
                cohort: [
                    1 + len(definitely_worse[cohort]),
                    arm_count - len(definitely_better[cohort]),
                ]
                for cohort in spec.cohorts
            }
        elif spec.contrasts.type == "all_pairs":
            warnings.append(
                "Simultaneous rank intervals require risk_difference in the all-pairs set."
            )

        _, _, simultaneous_coverage = _coverage_floors(
            next(family.alpha for family in plan.multiplicity.families if family.id == spec.family),
            spec.coverage_mode,
        )
        assert simultaneous_coverage is not None
        results.append(
            ComparisonSetResult(
                comparison_set_id=spec.id,
                contrast_type=spec.contrasts.type,
                cohorts=spec.cohorts,
                reference_cohort=spec.contrasts.reference,
                group_id=spec.group,
                measure_id=spec.measure,
                estimands=list(spec.estimands),
                comparison_ids=sorted({row.comparison_id for row in rows}),
                arm_estimates={
                    cohort: (
                        risk_index[(cohort, spec.group, spec.measure)].estimate
                        if (cohort, spec.group, spec.measure) in risk_index
                        else None
                    )
                    for cohort in spec.cohorts
                },
                arm_intervals=arm_intervals,
                rank_intervals=rank_intervals,
                independence_profile=spec.independence.primary,
                simultaneous_coverage=simultaneous_coverage,
                warnings=warnings,
            )
        )
    return results


def _claims(
    plan: AnalysisPlan,
    risks: list[GroupRiskResult],
    comparisons: list[ComparisonResult],
    integrity_valid: bool,
) -> list[ClaimResult]:
    risk_index = {(item.cohort_id, item.group_id, item.measure_id): item for item in risks}
    comparison_index = {item.comparison_id: item for item in comparisons if item.level is None}
    families = {family.id: family for family in plan.multiplicity.families}
    results: list[ClaimResult] = []
    for claim in plan.claims:
        evidence: ComparisonResult | GroupRiskResult | None
        if claim.comparison:
            evidence = comparison_index.get(claim.comparison)
            ref = f"comparison:{claim.comparison}"
        else:
            evidence = risk_index.get((str(claim.cohort), str(claim.group), str(claim.measure)))
            ref = f"risk:{claim.cohort}:{claim.group}:{claim.measure}"
        eligible = bool(
            integrity_valid
            and evidence is not None
            and evidence.eligibility == "certifying"
            and plan.analysis.intent == "confirmatory"
        )
        lower = evidence.lower if evidence else None
        upper = evidence.upper if evidence else None
        estimate = evidence.estimate if evidence else None
        status, reasons = decide_claim(
            lower=lower,
            upper=upper,
            acceptable_upper=claim.acceptable_upper,
            unacceptable_lower=claim.unacceptable_lower,
            eligible=eligible,
        )
        if not integrity_valid and lower is not None and upper is not None:
            reasons = [
                "Bounds were computed but are ineligible for certification because data "
                "integrity is invalid."
            ]
        if plan.analysis.intent == "exploratory":
            if evidence is not None and evidence.eligibility == "certifying":
                reasons = [
                    "Bounds were computed but this post hoc exploratory analysis is not "
                    "eligible for certification."
                ]
            else:
                reasons.append("Exploratory analyses cannot produce a safety pass.")
        family = families[claim.family]
        results.append(
            ClaimResult(
                claim_id=claim.id,
                title=claim.title,
                status=status,
                statistical_status=status,
                policy_status=status,
                estimate=estimate,
                lower=lower,
                upper=upper,
                acceptable_upper=claim.acceptable_upper,
                unacceptable_lower=claim.unacceptable_lower,
                alpha=family.alpha,
                multiplicity_family=claim.family,
                method=evidence.method if evidence else "unavailable",
                mandatory=claim.mandatory,
                evidence_refs=[ref],
                reasons=reasons,
                warnings=list(evidence.warnings) if evidence else [],
            )
        )
    return results


def _policy_events(plan: AnalysisPlan, data: NormalizedData) -> list[dict[str, Any]]:
    if not plan.policy.zero_tolerance.enabled:
        return []
    field = plan.policy.zero_tolerance.manifest_field
    if field not in data.manifest:
        return []
    zero_prompts = set(
        data.manifest.loc[data.manifest[field].fillna(False).astype(bool), "prompt_id"].astype(str)
    )
    return [
        {
            "type": "zero_tolerance_failure",
            "status": "verified_from_score",
            "cohort_id": row.cohort_id,
            "prompt_id": row.prompt_id,
            "observation_id": row.observation_id,
            "source_log": row.source_log,
            "sample_id": row.sample_id,
        }
        for row in data.observations
        if row.prompt_id in zero_prompts and row.failure is True
    ]


def _resolved_design(data: NormalizedData, hierarchy: list[str]) -> list[dict[str, Any]]:
    manifest = data.manifest.set_index("prompt_id")
    hierarchy_columns = list(dict.fromkeys([*hierarchy, "task_id", "question_id", "prompt_id"]))
    rows: list[dict[str, Any]] = []
    for group in sorted(data.groups, key=lambda item: item.id):
        for prompt_id in group.prompt_ids:
            item = manifest.loc[prompt_id]
            row: dict[str, Any] = {"group_id": group.id}
            for column in hierarchy_columns:
                row[column] = prompt_id if column == "prompt_id" else item[column]
            row["weight"] = group.weights[prompt_id]
            rows.append(row)
    return rows


def analyze(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
) -> SafetyReport:
    resolved = _coerce_plan(plan)
    diagnostics = validate_plan_references(resolved)
    data = ingest_inspect_logs(resolved, logs, parallel=parallel)
    validate_profile_fields(resolved.design.independence_profiles, data.manifest)
    diagnostics.extend(data.diagnostics)
    estimates = _prompt_estimates(resolved, data, diagnostics)
    hierarchy_nodes = _hierarchy_nodes(resolved, data)
    direct_context = _direct_mean_context(
        resolved,
        data,
        nodes=hierarchy_nodes,
    )
    risks = _group_risks(
        resolved,
        data,
        estimates,
        diagnostics,
        direct_context=direct_context,
    )
    hierarchy_risks = _hierarchy_risks(
        resolved,
        data,
        hierarchy_nodes,
        estimates,
        diagnostics,
        direct_context,
    )
    _project_hierarchy_regions(
        resolved,
        data,
        hierarchy_nodes,
        risks,
        hierarchy_risks,
        diagnostics,
    )
    judge_calibration_summary = _apply_judge_correction(
        resolved, data, risks, hierarchy_risks, diagnostics
    )
    comparisons = _comparisons(resolved, data, estimates, diagnostics)
    comparison_sets = _comparison_set_results(resolved, comparisons, risks)
    integrity_status = _integrity_status(diagnostics)
    integrity_certifiable = integrity_status != "invalid"
    claims = _claims(resolved, risks, comparisons, integrity_certifiable)
    events = _policy_events(resolved, data)
    decision = roll_up(
        [claim.status for claim in claims if claim.mandatory],
        integrity_valid=integrity_certifiable,
        zero_tolerance_failure=bool(events),
        confirmatory=resolved.analysis.intent == "confirmatory",
    )
    integrity = IntegritySummary(
        status=integrity_status,
        diagnostics=sorted(
            diagnostics, key=lambda item: (item.severity, item.code, item.scope or "")
        ),
    )
    plan_hash = canonical_plan_hash(resolved)
    report = SafetyReport(
        package_version=__version__,
        analysis={**resolved.analysis.model_dump(mode="json"), "plan_sha256": plan_hash},
        provenance={
            "source_logs": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(data.source_hashes.items())
            ],
            "inspect_version_constraint": ">=0.3.259,<0.4",
            **(
                {"judge_calibration": judge_calibration_summary}
                if judge_calibration_summary
                else {}
            ),
        },
        scope={
            "design": "fixed",
            "randomness": "stochastic rollouts within fixed prompts",
            "estimand": resolved.inputs.estimand,
            "disclaimer": (
                "Results apply to the declared fixed evaluation set and specified "
                "stochastic rollout process. They do not estimate failure prevalence "
                "over an external population of prompts or establish safety for omitted "
                "hazards or deployment conditions."
            ),
        },
        presentation={"include_decisions": resolved.report.include_decisions},
        integrity=integrity,
        design={
            "manifest": str(resolved.inputs.design_manifest),
            "manifest_sha256": sha256_value(data.manifest.to_dict(orient="records")),
            "hierarchy": resolved.design.hierarchy,
            "independence_profiles": {
                profile_id: profile.model_dump(mode="json")
                for profile_id, profile in resolved.design.independence_profiles.items()
            },
            "prompt_count": len(data.manifest),
            "groups": [
                {"id": group.id, "title": group.title, "prompt_count": len(group.prompt_ids)}
                for group in data.groups
            ],
        },
        cohorts=data.cohorts,
        prompt_estimates=sorted(estimates, key=lambda item: (item.cohort_id, item.prompt_id)),
        group_risks=sorted(
            risks, key=lambda item: (item.cohort_id, item.group_id, item.measure_id)
        ),
        hierarchy_risks=sorted(
            hierarchy_risks,
            key=lambda item: (
                item.cohort_id,
                item.group_id,
                resolved.design.hierarchy.index(item.level),
                tuple(item.path.values()),
            ),
        ),
        comparisons=sorted(comparisons, key=lambda item: item.comparison_id),
        comparison_sets=comparison_sets,
        multiplicity=[family.model_dump(mode="json") for family in resolved.multiplicity.families],
        claims=sorted(claims, key=lambda item: item.claim_id),
        policy_events=events,
        decision=decision,
    )
    report._observations = data.observations
    report._resolved_design = _resolved_design(data, resolved.design.hierarchy)
    report._resolved_plan = resolved.model_dump(mode="json", by_alias=True)
    return report
