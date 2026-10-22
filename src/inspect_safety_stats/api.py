from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import (
    AnalysisPlan,
    ClaimSpec,
    ComparisonSpec,
    RiskMeasureSpec,
)
from inspect_safety_stats.decision.engine import decide_claim, roll_up
from inspect_safety_stats.ingest.inspect_logs import LogInput, NormalizedData, ingest_inspect_logs
from inspect_safety_stats.ingest.manifest import (
    load_inclusion_probabilities,
    resolve_weights,
)
from inspect_safety_stats.ingest.validation import (
    judge_stratum_counts,
    load_validation_manifest,
    truth_stratum_counts,
)
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import (
    ClaimResult,
    ComparisonResult,
    ComparisonSetResult,
    Decision,
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
from inspect_safety_stats.stats.design_based import (
    DesignBasedBounds,
    FixedSizeDesign,
    InclusionDesign,
    fixed_size_mean_bounds,
    horvitz_thompson_mean_bounds,
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
    BlockIndex,
    BoundedBlocks,
    block_index,
    comparison_contrast_blocks,
    direct_bounded_blocks,
    validate_profile_fields,
)
from inspect_safety_stats.stats.judges import (
    JudgeAccuracyBox,
    compose_verdicts,
    composite_accuracy_box,
    effective_votes,
    identified_risk_set,
    pairwise_agreement,
)
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    JudgeStratumCounts,
    TruthStratumCounts,
    correct_judge_risk,
    delta_method_corrected_risk,
    estimate_judge_calibration,
    noisy_valid_certification,
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
    worst_of_k_risk,
)
from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    any_overlap,
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


def _block_index(data: NormalizedData) -> BlockIndex:
    """The block index for this analysis's observations, built on first use."""
    if data.block_index is None:
        data.block_index = block_index(data.observations, data.manifest)
    return data.block_index


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
        # An error about one prompt names it in its details, with its cohort when it concerns
        # only one. Scope strings are for people. Matching their prefixes let an error about
        # p10 block p1, and an error about group g1 block a prompt named g1.
        details = diagnostic.details
        if details.get("prompt_id") is not None:
            if details["prompt_id"] == prompt and details.get("cohort_id") in (None, cohort):
                return True
            continue
        if diagnostic.scope == cohort:
            return True
    return False


def _bounds_carry_unknown_outcome(plan: AnalysisPlan, row: Observation) -> bool:
    """Whether an unclassified rollout stays in its prompt's bounds as an unknown outcome.

    Under the other policies it is blocked or excluded, and leaves the prompt's denominator.
    """
    if row.failure is not None:
        return False
    if row.error is not None:
        return plan.inputs.score.errors in {"end_to_end", "conservative_bounds"}
    return plan.inputs.score.missing == "conservative_bounds"


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
            bounded_unknown = [row for row in rows if _bounds_carry_unknown_outcome(plan, row)]
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


# Measures that are probabilities whenever every value is. Threshold excess is not one: a
# negative threshold lets it exceed 1.
_PROBABILITY_MEASURES = frozenset(
    {
        "weighted_mean",
        "maximum",
        "cvar",
        "top_k_mean",
        "entropic",
        "worst_of_k",
        "threshold_exceedance",
    }
)


def _risk_value(
    values: list[float],
    weights: list[float],
    measure: RiskMeasureSpec,
    *,
    threshold: float | np.ndarray | None = None,
) -> float:
    value = _unclipped_risk_value(values, weights, measure, threshold=threshold)
    if measure.type in _PROBABILITY_MEASURES:
        # Floating-point sums can land just outside [0, 1]: nine weights of 1/9 sum to
        # 1.0000000000000002. A judge correction then rejects the bound as no probability.
        return min(1.0, max(0.0, value))
    return value


def _unclipped_risk_value(
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
    if measure.type == "worst_of_k":
        assert measure.k is not None
        return worst_of_k_risk(values, weights, k=int(measure.k))
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


def _path_rows(frame: pd.DataFrame, path_levels: list[str]) -> dict[tuple[str, ...], list[int]]:
    """Row positions of each distinct hierarchy path in ``frame``.

    ORDER MATTERS: paths come in order of first appearance and rows in frame order, as
    drop_duplicates() and one boolean mask per path gave them before. Nodes, target ids,
    frontiers and report rows are all emitted in this order.
    """
    rows: dict[tuple[str, ...], list[int]] = {}
    values = frame[path_levels].astype(str).itertuples(index=False, name=None)
    for position, path_values in enumerate(values):
        rows.setdefault(path_values, []).append(position)
    return rows


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
        group_prompts = group_frame["prompt_id"].astype(str).tolist()
        for level_index, level in enumerate(levels):
            path_levels = levels[: level_index + 1]
            for path_values, rows in _path_rows(group_frame, path_levels).items():
                path = dict(zip(path_levels, path_values, strict=True))
                prompt_ids = tuple(group_prompts[row] for row in rows)
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
            if any_overlap(atom_sets):
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
                                index=_block_index(data),
                            ).block_ids
                        )
                    overlaps = any_overlap(supports.values())
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


def _sampled_frame_intervals(
    *,
    data: NormalizedData,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    weights: np.ndarray,
    estimates: dict[tuple[str, str], PromptEstimate],
    plan: AnalysisPlan,
    family_id: str | None,
    family_alpha: float | None,
    calibration: _EndpointCalibration,
    one_sided_coverage: float | None,
    interval_coverage: float | None,
) -> list[RiskIntervalResult]:
    """Horvitz-Thompson intervals for a frame mean estimated from an item sample.

    The declared independence profiles do not apply here. Inclusion randomness lives at
    the frame item, so the frame item is the independent block regardless of how rollouts
    within it are grouped.
    """

    spec = plan.outputs.direct_mean_intervals
    column = str(plan.design.inclusion_probability_column)
    stratified = plan.design.sampling == "fixed_size"
    manifest = data.manifest.set_index("prompt_id", drop=False)

    # Poisson inclusion is item-by-item, so only the target's own items matter. A
    # fixed-size draw happens at stratum level, so the whole frame is needed: an item
    # outside the target still consumed one of its stratum's draws.
    scope = list(manifest["prompt_id"].astype(str)) if stratified else list(prompt_ids)
    frame = manifest.loc[scope]
    probabilities = load_inclusion_probabilities(frame, column)
    target_weight = dict(zip(prompt_ids, np.asarray(weights, dtype=float), strict=True))

    sampled = np.zeros(len(scope), dtype=bool)
    lower_rates = np.zeros(len(scope), dtype=float)
    upper_rates = np.zeros(len(scope), dtype=float)
    frame_weights = np.zeros(len(scope), dtype=float)
    for index, prompt_id in enumerate(scope):
        frame_weights[index] = target_weight.get(prompt_id, 0.0)
        item = estimates.get((cohort_id, prompt_id))
        if item is None or item.n_total <= 0:
            continue
        sampled[index] = True
        lower_rates[index] = item.n_failure / item.n_total
        upper_rates[index] = (item.n_failure + item.n_unclassified) / item.n_total

    design: FixedSizeDesign | InclusionDesign
    if stratified:
        labels = (
            frame[str(plan.design.stratum_column)].astype(str).tolist()
            if plan.design.stratum_column
            else ["_all"] * len(scope)
        )
        order = {label: position for position, label in enumerate(sorted(set(labels)))}
        stratum_index = np.asarray([order[label] for label in labels], dtype=int)
        draws = np.bincount(stratum_index[sampled], minlength=len(order))
        design = FixedSizeDesign(stratum_index, frame_weights, draws, sampled)
        estimator = "stratified_without_replacement"
    else:
        design = InclusionDesign(frame_weights, probabilities, sampled)
        estimator = "horvitz_thompson"

    results: list[RiskIntervalResult] = []
    for method in [spec.method, *spec.reference_methods]:
        bounds: DesignBasedBounds
        if isinstance(design, FixedSizeDesign):
            bounds = fixed_size_mean_bounds(
                design,
                lower_rates,
                upper_rates,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                method=method,
            )
        else:
            bounds = horvitz_thompson_mean_bounds(
                design,
                lower_rates,
                upper_rates,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                method=method,
            )
        results.append(
            RiskIntervalResult(
                interval_id=f"{estimator}_{method}",
                lower=bounds.lower,
                upper=bounds.upper,
                method=method,
                role="primary" if method == spec.method else "reference",
                coverage_scope=(
                    f"{'simultaneous' if spec.simultaneous else 'pointwise'}"
                    "_sampled_frame_mean_family"
                ),
                coverage_mode=spec.coverage_mode,
                family=family_id,
                family_alpha=family_alpha,
                alpha_lower=calibration.endpoint_alpha,
                alpha_upper=calibration.endpoint_alpha,
                simultaneous_coverage=interval_coverage,
                lower_one_sided_coverage=one_sided_coverage,
                upper_one_sided_coverage=one_sided_coverage,
                completion_estimate_lower=bounds.completion_estimate_lower,
                completion_estimate_upper=bounds.completion_estimate_upper,
                effective_sample_size=bounds.effective_sample_size,
                maximum_increment=bounds.maximum_weight,
                calibration_method=calibration.method,
                calibration_frontier=calibration.frontier_id,
                calibration_frontier_alpha=calibration.frontier_alpha,
                calibration_frontier_size=calibration.frontier_size,
                independence_profile="frame_item",
                independent_block_count=bounds.sampled_count,
                missingness_method="all_safe_all_failure_completion",
                assumptions=[
                    (
                        "Each stratum drew exactly its planned number of frame items "
                        "without replacement; the bound uses Hoeffding's reduction, under "
                        "which a convex function of a without-replacement sum is dominated "
                        "by the with-replacement one, so it is conservative."
                        if stratified
                        else "Frame items were included independently with the declared "
                        "probabilities (Poisson sampling)."
                    ),
                    "Inclusion was determined before, and independently of, the outcomes.",
                    "Design weights are declared over the whole frame and are not "
                    "renormalized to the realized sample.",
                    f"Inverse-probability weighting widens every endpoint by a factor of "
                    f"{bounds.inflation:.6g}.",
                    f"The largest single frame item carries {bounds.maximum_weight:.6g} of "
                    "the estimator's weight.",
                    "Every unclassified attempted outcome has a latent binary safety label.",
                ],
                eligibility="certifying",
            )
        )
    return results


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
    estimates: dict[tuple[str, str], PromptEstimate] | None = None,
) -> list[RiskIntervalResult]:
    spec = plan.outputs.direct_mean_intervals
    assert spec.independence is not None
    if plan.design.type == "sampled":
        assert estimates is not None
        return _sampled_frame_intervals(
            data=data,
            cohort_id=cohort_id,
            prompt_ids=prompt_ids,
            weights=weights,
            estimates=estimates,
            plan=plan,
            family_id=family_id,
            family_alpha=family_alpha,
            calibration=calibrations[
                _calibration_key(cohort_id, target_id, spec.independence.primary)
            ],
            one_sided_coverage=one_sided_coverage,
            interval_coverage=interval_coverage,
        )
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
            index=_block_index(data),
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
                        (item.n_total <= 0 and plan.design.type != "sampled")
                        or _prompt_blocked(diagnostics, cohort_id, item.prompt_id)
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
                                estimates=index,
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
    unsampled_nodes: list[str] = []
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
            elif plan.design.type == "sampled" and not any(
                item.n_total > 0 for item in prompt_results
            ):
                # The node's estimand is simply not identified by this sample.
                unsampled_nodes.append(target_id)
                warnings.append(
                    "No descendant frame item was drawn into the sample, so this node's "
                    "risk is not identified."
                )
            elif any(
                (item.n_total <= 0 and plan.design.type != "sampled")
                or _prompt_blocked(diagnostics, cohort_id, item.prompt_id)
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
                        estimates=index,
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
    if unsampled_nodes:
        diagnostics.append(
            Diagnostic(
                code="HIERARCHY_NODE_UNSAMPLED",
                severity="WARNING",
                message=(
                    f"{len(unsampled_nodes)} hierarchy nodes had no frame item drawn into "
                    "the sample; their conditional risk is not identified and no interval "
                    "is reported for them"
                ),
                scope="hierarchy",
                details={"nodes": sorted(unsampled_nodes)[:20], "count": len(unsampled_nodes)},
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
                _keep_only_joint_coverage(interval)

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
            result.lower_one_sided_coverage = primary.lower_one_sided_coverage
            result.upper_one_sided_coverage = primary.upper_one_sided_coverage
            for root_result in root_rows.get(result.group_id, []):
                if isinstance(result, GroupRiskResult):
                    root_result.lower = primary.lower
                    root_result.upper = primary.upper
                    root_result.coverage_scope = primary.coverage_scope
                    root_result.raw_lower = primary.raw_lower
                    root_result.raw_upper = primary.raw_upper
                    root_result.projection_method = primary.projection_method
                    root_result.lower_one_sided_coverage = primary.lower_one_sided_coverage
                    root_result.upper_one_sided_coverage = primary.upper_one_sided_coverage


def _keep_only_joint_coverage(interval: RiskIntervalResult) -> None:
    """Restate a projected interval's one-sided coverage as its simultaneous floor.

    A projected endpoint optimises over a region that every raw endpoint shapes, lower and
    upper alike. It therefore covers on the joint event only: a raw lower endpoint that misses
    can pull a projected upper endpoint below the truth. Each direction keeps the
    simultaneous floor, not the one-sided level of a raw endpoint.
    """
    interval.lower_one_sided_coverage = interval.simultaneous_coverage
    interval.upper_one_sided_coverage = interval.simultaneous_coverage


# Risk measures whose value is positively homogeneous and translation equivariant,
# so the affine judge correction R* = R * rho_1 + (1 - R) * rho_0 applies unchanged.
# A validated calibration certifies only the weighted mean: see _pooled_calibration_gap.
_JUDGE_CORRECTABLE_MEASURES = frozenset(
    {"weighted_mean", "maximum", "cvar", "top_k_mean"}
)


@dataclass(frozen=True)
class _PromptRollouts:
    """One prompt's rollouts in one cohort, counted the ways its rates divide by them."""

    classified: int
    # Prompt bounds divide by the classified rollouts plus the unknown outcomes the missing
    # and error policies keep. Direct-mean intervals divide by every rollout, and complete
    # each unclassified one, so the two counts differ only where a policy excludes some.
    bounded: int
    attempted: int


def _labelled_population(
    plan: AnalysisPlan, data: NormalizedData
) -> dict[str, dict[str, _PromptRollouts]]:
    """Per cohort and prompt, the rollouts behind its rates.

    A cohort's classified rollouts are the population its judge labels are drawn from.
    """

    counts: dict[str, dict[str, list[int]]] = {}
    for row in data.observations:
        tally = counts.setdefault(row.cohort_id, {}).setdefault(row.prompt_id, [0, 0, 0])
        tally[2] += 1
        if row.failure is not None:
            tally[0] += 1
            tally[1] += 1
        elif _bounds_carry_unknown_outcome(plan, row):
            tally[1] += 1
    return {
        cohort_id: {prompt_id: _PromptRollouts(*tally) for prompt_id, tally in prompts.items()}
        for cohort_id, prompts in counts.items()
    }


def _pooled_calibration_gap(
    population: dict[str, _PromptRollouts],
    *,
    design_type: str,
    measure_type: str,
    weights: dict[str, float],
) -> str | None:
    """Why a cohort's pooled labels do not measure the judge accuracies a row needs.

    The labels are drawn uniformly within judge strata over the cohort's classified
    rollouts, so they estimate P(truly unsafe | verdict) with each of those rollouts
    weighted equally. R* = R rho_1 + (1 - R) rho_0 needs the accuracies weighted the way the
    row weights its rollouts. That holds exactly only for a weighted mean, in a fixed
    design, that gives every classified rollout of the cohort the same weight. Returns
    None when the row is such a mean.
    """

    if design_type != "fixed":
        return (
            "a sampled design estimates a frame mean, and labels on the drawn items do not "
            "bound the judge's accuracy over the frame"
        )
    if measure_type != "weighted_mean":
        return (
            f"a {measure_type} measure concentrates on the riskiest prompts, where the "
            "judge's accuracy can differ from the pooled labels' average"
        )
    labelled = [(prompt_id, item) for prompt_id, item in population.items() if item.classified]
    if not labelled:
        return "the cohort has no classified rollouts"
    if any(weights.get(prompt_id, 0.0) <= 0.0 for prompt_id, _ in labelled):
        return "the row covers only part of the cohort's labelled rollouts"
    # A rollout weighs its prompt's weight over the rollouts the prompt's rate divides by.
    # Either kind of interval can be the row's primary, so both counts must agree.
    for per_rollout in (
        [weights[prompt_id] / item.bounded for prompt_id, item in labelled],
        [weights[prompt_id] / item.attempted for prompt_id, item in labelled],
    ):
        lightest, heaviest = min(per_rollout), max(per_rollout)
        if heaviest > lightest * (1 + 1e-9):
            return (
                f"the row gives some of the cohort's rollouts {heaviest / lightest:.3g} times "
                "the weight of others"
            )
    return None


def _uncertified_correction_warning(gap: str) -> str:
    return (
        f"Sensitivity analysis, not a certified bound: {gap}. The human labels weight every "
        "classified rollout of the cohort equally, so they measure the judge's accuracy only "
        "for a weighted mean that does the same, over the whole cohort in a fixed design."
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
) -> tuple[
    dict[str, JudgeCalibration], dict[str, dict[str, Any]], float | None, dict[str, Any]
]:
    """Estimate one calibration per cohort from human labels within judge strata."""

    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "validated":
        return {}, {}, None, {}
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
        return {}, {}, None, {}
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
        return {}, {}, None, {}

    counts, unmatched = judge_stratum_counts(data.observations, labels)
    truth_strata: dict[str, Any] = {}
    if spec.sampling == "simple_random":
        # A simple random sample of the design also estimates sensitivity and the
        # false-positive rate by true class, which the noisy-but-valid test needs.
        for cohort_id, by_class in truth_stratum_counts(data.observations, labels).items():
            truth_strata[cohort_id] = {
                name: {"labelled": item.labelled, "judge_positive": item.judge_positive}
                for name, item in by_class.items()
            }
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

    # Every declared cohort gets an equal share of the judge family budget. A single
    # risk row consumes one share; a comparison consumes the two cohorts it spans. A
    # union over all of them therefore never exceeds the declared family alpha.
    per_cohort_alpha = family.alpha / len(plan.cohorts)
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
                alpha=per_cohort_alpha,
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
    return calibrations, summaries, family.alpha, truth_strata


def _judge_corrected_interval(
    *,
    result: GroupRiskResult | HierarchyRiskResult,
    calibration: JudgeCalibration,
    primary_scenario: str | None,
    family_id: str | None,
    judge_alpha: float | None,
    gap: str | None = None,
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
    # Measured accuracies certify only a row they were measured for; elsewhere the
    # corrected interval is a sensitivity, as an assumed one always is.
    certified = validated and gap is None
    base = next((item for item in result.intervals if item.role == "primary"), None)
    base_alpha = base.family_alpha if base is not None else None
    spent = calibration.alpha if calibration.alpha is not None else judge_alpha
    simultaneous = (
        1 - (base_alpha + spent)
        if certified and base_alpha is not None and spent is not None
        else None
    )
    # Each corrected endpoint uses the base endpoint plus two of the four calibration
    # endpoints, so a one-sided claim spends half the calibration budget.
    base_directional = base.directional_family_alpha if base is not None else None
    if base is not None and base.projection_method is not None:
        # A projected base endpoint covers only at the simultaneous floor.
        base_directional = (
            1 - base.simultaneous_coverage if base.simultaneous_coverage is not None else None
        )
    one_sided = (
        1 - (base_directional + spent / 2)
        if certified and base_directional is not None and spent is not None
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
            ]
        )
        if certified:
            assumptions.extend(
                [
                    "The labels weight every classified rollout of the cohort equally, and so "
                    "does this row, so they measure the judge's accuracy for its estimand.",
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
    if validated and gap is not None:
        warnings.append(_uncertified_correction_warning(gap))
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
        role="reference" if certified else "sensitivity",
        coverage_scope=(
            f"{result.coverage_scope}_judge_corrected"
            if result.coverage_scope
            else "judge_corrected"
        ),
        family=family_id,
        family_alpha=spent,
        simultaneous_coverage=simultaneous,
        lower_one_sided_coverage=one_sided,
        upper_one_sided_coverage=one_sided,
        completion_estimate_lower=corrected.completion_estimate_lower,
        completion_estimate_upper=corrected.completion_estimate_upper,
        raw_lower=result.lower,
        raw_upper=result.upper,
        calibration_method=f"{calibration.source}_judge_calibration",
        assumptions=assumptions,
        eligibility="certifying" if certified else "diagnostic",
        warnings=warnings,
    )


def _judge_panel_summary(plan: AnalysisPlan, data: NormalizedData) -> dict[str, Any] | None:
    """Describe the declared judge panel: per-judge counts, composites, and agreement.

    Bookkeeping only. Effective votes is the Kish count over declared dependence
    blocks, and agreement is descriptive; nothing here corrects a risk estimate.
    """

    if not plan.inputs.judges:
        return None
    cohorts = sorted(plan.cohorts)
    by_cohort: dict[str, list[str]] = {cohort_id: [] for cohort_id in cohorts}
    for observation in data.observations:
        by_cohort.setdefault(observation.cohort_id, []).append(observation.observation_id)

    def verdicts(cohort_id: str, judge_id: str) -> list[bool | None]:
        return [
            data.judge_verdicts.get(observation_id, {}).get(judge_id)
            for observation_id in by_cohort[cohort_id]
        ]

    def counts(values: list[bool | None]) -> dict[str, int]:
        positives = sum(value is True for value in values)
        negatives = sum(value is False for value in values)
        return {
            "classified": positives + negatives,
            "failures": positives,
            "unclassified": len(values) - positives - negatives,
        }

    blocks = {judge.id: judge.block or judge.id for judge in plan.inputs.judges}
    judges = [
        {
            "id": judge.id,
            "kind": judge.kind,
            "block": blocks[judge.id],
            "score": judge.score.name,
            "cohorts": {cohort_id: counts(verdicts(cohort_id, judge.id)) for cohort_id in cohorts},
        }
        for judge in plan.inputs.judges
    ]
    composites = []
    for composite in plan.inputs.composites:
        per_cohort = {}
        for cohort_id in cohorts:
            combined = [
                compose_verdicts(
                    {
                        judge_id: data.judge_verdicts.get(observation_id, {}).get(judge_id)
                        for judge_id in composite.judges
                    },
                    composite.rule,
                )
                for observation_id in by_cohort[cohort_id]
            ]
            per_cohort[cohort_id] = counts(combined)
        composites.append(
            {
                "id": composite.id,
                "rule": composite.rule,
                "judges": list(composite.judges),
                "effective_votes": effective_votes([blocks[j] for j in composite.judges]),
                "cohorts": per_cohort,
            }
        )
    agreement = []
    for first, second in itertools.combinations([judge.id for judge in plan.inputs.judges], 2):
        pair_cohorts = {}
        for cohort_id in cohorts:
            first_votes = verdicts(cohort_id, first)
            second_votes = verdicts(cohort_id, second)
            summary = pairwise_agreement(first_votes, second_votes)
            paired = [
                (bool(a), bool(b))
                for a, b in zip(first_votes, second_votes, strict=True)
                if a is not None and b is not None
            ]
            first_only = second_only = 0
            p_value: float | None = None
            if paired:
                mcnemar = exact_mcnemar([a for a, _ in paired], [b for _, b in paired])
                first_only = mcnemar.candidate_failure_baseline_safe
                second_only = mcnemar.candidate_safe_baseline_failure
                p_value = mcnemar.p_value_two_sided
            pair_cohorts[cohort_id] = {
                "compared": summary.compared,
                "disagreements": summary.disagreements,
                "agreement": summary.agreement,
                "kappa": summary.kappa,
                "discordant_first_only": first_only,
                "discordant_second_only": second_only,
                "mcnemar_p_value": p_value,
            }
        agreement.append({"first": first, "second": second, "cohorts": pair_cohorts})
    return {
        "judges": judges,
        "composites": composites,
        "agreement": agreement,
        "effective_votes_all": effective_votes(list(blocks.values())),
    }


def _apply_panel_derived_correction(
    plan: AnalysisPlan,
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    diagnostics: list[Diagnostic],
) -> dict[str, Any] | None:
    """Partial identification of true risk when the primary outcome is a composite.

    The composite's accuracy box is derived from its judges' declared boxes under the
    declared dependence blocks, and the judge-reported interval is inverted over that
    box. Declared boxes are assumptions, so every interval this adds is a diagnostic
    sensitivity that can never move a claim.
    """

    composite = next(
        (item for item in plan.inputs.composites if item.id == plan.inputs.score.name), None
    )
    if composite is None:
        return None
    judges = {judge.id: judge for judge in plan.inputs.judges}
    members = [judges[judge_id] for judge_id in composite.judges]
    if any(member.accuracy is None for member in members):
        return None
    if composite.rule == "majority":
        diagnostics.append(
            Diagnostic(
                code="PANEL_CORRECTION_UNSUPPORTED_RULE",
                severity="INFO",
                message=(
                    "a majority composite has no closed-form accuracy box under partial "
                    "dependence; validate the composite's own verdict instead"
                ),
                scope=f"inputs.composites.{composite.id}",
            )
        )
        return None
    boxes = []
    for member in members:
        assert member.accuracy is not None
        boxes.append(
            JudgeAccuracyBox(
                sensitivity_lower=member.accuracy.sensitivity.lower,
                sensitivity_upper=member.accuracy.sensitivity.upper,
                specificity_lower=member.accuracy.specificity.lower,
                specificity_upper=member.accuracy.specificity.upper,
            )
        )
    box = composite_accuracy_box(
        boxes, [member.block or member.id for member in members], composite.rule
    )
    correctable = {
        measure.id
        for measure in plan.outputs.risk_measures
        if measure.type in _JUDGE_CORRECTABLE_MEASURES
    }
    targets: list[GroupRiskResult | HierarchyRiskResult] = [
        *(row for row in risks if row.measure_id in correctable),
        *hierarchy_risks,
    ]
    for result in targets:
        if result.lower is None or result.upper is None:
            continue
        identified = identified_risk_set(result.lower, result.upper, box)
        estimate_range = (
            identified_risk_set(result.estimate, result.estimate, box)
            if result.estimate is not None
            else None
        )
        result.intervals.append(
            RiskIntervalResult(
                interval_id="judge_corrected_panel_derived",
                lower=identified.lower,
                upper=identified.upper,
                method="derived_panel_misclassification_correction",
                role="sensitivity",
                coverage_scope=(
                    f"{result.coverage_scope}_panel_derived"
                    if result.coverage_scope
                    else "panel_derived"
                ),
                completion_estimate_lower=(
                    estimate_range.lower if estimate_range is not None else None
                ),
                completion_estimate_upper=(
                    estimate_range.upper if estimate_range is not None else None
                ),
                raw_lower=result.lower,
                raw_upper=result.upper,
                calibration_method="derived_panel_accuracy_box",
                assumptions=[
                    "The reported estimand is true failure risk, not judge-reported failure risk.",
                    f"The composite {composite.id!r} ({composite.rule}) accuracy box is derived "
                    "from its judges' declared sensitivity/specificity ranges: Frechet bounds "
                    "within each declared dependence block, independence across blocks.",
                    f"Derived sensitivity lies in [{box.sensitivity_lower:.6g}, "
                    f"{box.sensitivity_upper:.6g}] and specificity in "
                    f"[{box.specificity_lower:.6g}, {box.specificity_upper:.6g}].",
                    "Judge accuracies are declared, prevalence-free, and applied unchanged to "
                    "every prompt and cohort; they carry no coverage guarantee.",
                    "True risk is the Rogan-Gladen inversion over the box; extrema sit at the box "
                    "vertices because the inversion is separately monotone in each argument.",
                ],
                eligibility="diagnostic",
                warnings=(
                    []
                    if identified.identified
                    else [
                        "The judge-reported rate is outside what any accuracy in the declared "
                        "boxes can produce, so the data contradict the boxes; true risk is "
                        "unidentified and the reported set is the whole unit interval."
                    ]
                    if identified.unidentified_reason == "contradicted"
                    else [
                        "The declared boxes admit an uninformative panel (sensitivity + "
                        "specificity <= 1); true risk is unidentified and the reported set is "
                        "the whole unit interval."
                    ]
                ),
            )
        )
    return {
        "composite": composite.id,
        "rule": composite.rule,
        "judges": list(composite.judges),
        "derived_box": {
            "sensitivity": [box.sensitivity_lower, box.sensitivity_upper],
            "specificity": [box.specificity_lower, box.specificity_upper],
        },
    }


def _judge_reported_variance(data: NormalizedData, cohort_id: str, group_id: str) -> float | None:
    """Binomial variance of the design-weighted judge-reported mean under rollout independence."""

    group = next((item for item in data.groups if item.id == group_id), None)
    if group is None:
        return None
    counts: dict[str, list[int]] = {prompt_id: [0, 0] for prompt_id in group.prompt_ids}
    for observation in data.observations:
        if observation.cohort_id != cohort_id or observation.failure is None:
            continue
        tally = counts.get(observation.prompt_id)
        if tally is None:
            continue
        tally[0] += int(observation.failure)
        tally[1] += 1
    variance = 0.0
    for prompt_id, (failures, classified) in counts.items():
        if classified == 0:
            return None
        rate = failures / classified
        variance += group.weights[prompt_id] ** 2 * rate * (1 - rate) / classified
    return variance


def _delta_reference_interval(
    *,
    result: GroupRiskResult,
    data: NormalizedData,
    strata: dict[str, Any],
    alpha: float,
    family_id: str | None,
    gap: str | None = None,
) -> RiskIntervalResult | None:
    """Asymptotic delta-method reference beside the exact validated correction."""

    unsafe = strata.get("judge_unsafe")
    safe = strata.get("judge_safe")
    if result.estimate is None or result.lower is None or result.upper is None:
        return None
    if not unsafe or not safe:
        return None
    variance = _judge_reported_variance(data, result.cohort_id, result.group_id)
    if variance is None:
        return None
    try:
        delta = delta_method_corrected_risk(
            observed_estimate=result.estimate,
            observed_variance=variance,
            unsafe_stratum=JudgeStratumCounts(**unsafe),
            safe_stratum=JudgeStratumCounts(**safe),
            alpha=alpha,
        )
    except ValueError:
        return None
    return RiskIntervalResult(
        interval_id="judge_corrected_validated_delta",
        lower=delta.lower,
        upper=delta.upper,
        method="delta_method_judge_misclassification_correction",
        role="sensitivity",
        coverage_scope=(
            f"{result.coverage_scope}_judge_corrected_delta"
            if result.coverage_scope
            else "judge_corrected_delta"
        ),
        family=family_id,
        family_alpha=alpha,
        completion_estimate_lower=delta.estimate,
        completion_estimate_upper=delta.estimate,
        half_width_lower=delta.estimate - delta.lower,
        half_width_upper=delta.upper - delta.estimate,
        raw_lower=result.lower,
        raw_upper=result.upper,
        calibration_method="validated_judge_calibration_delta_method",
        assumptions=[
            "The reported estimand is true failure risk, not judge-reported failure risk.",
            "Asymptotic Wald interval from a first-order (delta-method) expansion of "
            "R* = R rho_1 + (1 - R) rho_0 with a normal quantile; it carries no "
            "finite-sample coverage guarantee and cannot support a claim.",
            "Judge-reported variance is the rollout-independence binomial variance "
            "sum_j w_j^2 p_j (1 - p_j) / n_j of the design-weighted mean.",
            "Stratum proportions are Agresti-Coull stabilised, (x + 2) / (n + 4).",
            "Shown as the asymptotic reference for the exact vertex box: the gap between "
            "them is the price of the finite-sample guarantee.",
        ],
        eligibility="diagnostic",
        warnings=[_uncertified_correction_warning(gap)] if gap is not None else [],
    )


def _apply_judge_correction(
    plan: AnalysisPlan,
    data: NormalizedData,
    hierarchy_nodes: list[_HierarchyNode],
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    comparisons: list[ComparisonResult],
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

    by_cohort, summaries, judge_alpha, truth_strata = _validated_judge_calibrations(
        plan, data, diagnostics
    )
    population = _labelled_population(plan, data)
    group_weights = {group.id: group.weights for group in data.groups}
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    node_weights = {
        (node.group_id, node.level, tuple(node.path.items())): dict(
            zip(node.prompt_ids, node.weights, strict=True)
        )
        for node in hierarchy_nodes
    }
    for result in targets:
        cohort_calibration = by_cohort.get(result.cohort_id)
        if cohort_calibration is None:
            continue
        if isinstance(result, GroupRiskResult):
            weights = group_weights[result.group_id]
            measure_type = measure_types[result.measure_id]
        else:
            # Hierarchy nodes are weighted means over their own prompts.
            weights = node_weights[(result.group_id, result.level, tuple(result.path.items()))]
            measure_type = "weighted_mean"
        gap = _pooled_calibration_gap(
            population.get(result.cohort_id, {}),
            design_type=plan.design.type,
            measure_type=measure_type,
            weights=weights,
        )
        interval = _judge_corrected_interval(
            result=result,
            calibration=cohort_calibration,
            primary_scenario=None,
            family_id=spec.family,
            judge_alpha=judge_alpha,
            gap=gap,
        )
        if interval is not None:
            result.intervals.append(interval)
        if isinstance(result, GroupRiskResult) and judge_alpha is not None:
            reference = _delta_reference_interval(
                result=result,
                data=data,
                strata=summaries.get(result.cohort_id, {}),
                alpha=judge_alpha / len(plan.cohorts),
                family_id=spec.family,
                gap=gap,
            )
            if reference is not None:
                result.intervals.append(reference)
    _apply_judge_correction_to_comparisons(
        plan, data, population, comparisons, by_cohort, diagnostics
    )
    return {
        "mode": "validated",
        "family": spec.family,
        "family_alpha": judge_alpha,
        "per_cohort_alpha": (
            judge_alpha / len(plan.cohorts) if judge_alpha is not None else None
        ),
        "sampling": spec.sampling,
        "validation_manifest": str(spec.validation_manifest),
        "cohorts": summaries,
        "truth_strata": truth_strata,
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
    manifest_prompts = manifest["prompt_id"].astype(str).tolist()
    for level_index, level in enumerate(levels):
        path_levels = levels[: level_index + 1]
        for values, rows in _path_rows(manifest, path_levels).items():
            path = dict(zip(path_levels, values, strict=True))
            node_prompts = tuple(manifest_prompts[row] for row in rows)
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
                index=_block_index(data),
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
        if any_overlap(atom_sets):
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
        overlapping = any_overlap(targets_by_id.values())
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
        if not rows:
            # No results, for example because the comparison's family is undeclared, which
            # plan validation already reports. There is nothing to project.
            continue
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
                _keep_only_joint_coverage(interval)

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
            result.lower_one_sided_coverage = primary.lower_one_sided_coverage
            result.upper_one_sided_coverage = primary.upper_one_sided_coverage


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
        # The same switch already gates the pairwise projection of the set's comparisons.
        if set_spec.hierarchy_region != "linear_program":
            continue
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


def _judge_corrected_comparison_interval(
    *,
    result: ComparisonResult,
    candidate_calibration: JudgeCalibration,
    baseline_calibration: JudgeCalibration,
    family_id: str | None,
    gap: str | None = None,
) -> RiskIntervalResult | None:
    """Correct a comparison by correcting each arm, then recombining.

    rho is prevalence-dependent, so the two arms carry different calibrations and the
    effect cannot be corrected from the contrast alone: rescaling a difference by a
    single shared factor is only valid under prevalence-free parameters. This therefore
    requires an interval that exposes both arm bounds -- the two-arm rectangle used by
    weighted KL/Chernoff and by every comparison set. A direct contrast interval, which
    bounds only the difference, is skipped.
    """

    primary = next((item for item in result.intervals if item.role == "primary"), None)
    if primary is None:
        return None
    arms = (
        primary.candidate_lower,
        primary.candidate_upper,
        primary.baseline_lower,
        primary.baseline_upper,
    )
    if any(value is None for value in arms):
        return None
    candidate = correct_judge_risk(
        observed_lower=cast(float, primary.candidate_lower),
        observed_upper=cast(float, primary.candidate_upper),
        observed_estimate=result.candidate_estimate,
        calibration=candidate_calibration,
    )
    baseline = correct_judge_risk(
        observed_lower=cast(float, primary.baseline_lower),
        observed_upper=cast(float, primary.baseline_upper),
        observed_estimate=result.baseline_estimate,
        calibration=baseline_calibration,
    )
    lower, upper = _multi_arm_effect_interval(
        result.estimand,
        (candidate.lower, candidate.upper),
        (baseline.lower, baseline.upper),
    )
    spent = sum(
        item.alpha or 0.0 for item in (candidate_calibration, baseline_calibration)
    )
    base_alpha = primary.family_alpha
    certified = gap is None
    return RiskIntervalResult(
        interval_id="judge_corrected_validated",
        lower=lower,
        upper=upper,
        method="validated_judge_misclassification_correction",
        role="reference" if certified else "sensitivity",
        coverage_scope=(
            f"{primary.coverage_scope}_judge_corrected"
            if primary.coverage_scope
            else "judge_corrected"
        ),
        family=family_id,
        family_alpha=spent,
        simultaneous_coverage=(
            1 - (base_alpha + spent) if certified and base_alpha is not None else None
        ),
        raw_lower=primary.lower,
        raw_upper=primary.upper,
        candidate_lower=candidate.lower,
        candidate_upper=candidate.upper,
        baseline_lower=baseline.lower,
        baseline_upper=baseline.upper,
        calibration_method="validated_judge_calibration",
        independence_profile=primary.independence_profile,
        assumptions=[
            "The reported effect is a contrast of true failure risks, not of "
            "judge-reported risks.",
            "Each arm is corrected with its own cohort's calibration, because the "
            "conditional judge accuracies are prevalence-dependent.",
            f"Candidate P(truly unsafe | judge said unsafe) lies in "
            f"[{candidate_calibration.rho_1_lower:.6g}, "
            f"{candidate_calibration.rho_1_upper:.6g}]; P(truly unsafe | judge said safe) "
            f"in [{candidate_calibration.rho_0_lower:.6g}, "
            f"{candidate_calibration.rho_0_upper:.6g}].",
            f"Baseline P(truly unsafe | judge said unsafe) lies in "
            f"[{baseline_calibration.rho_1_lower:.6g}, "
            f"{baseline_calibration.rho_1_upper:.6g}]; P(truly unsafe | judge said safe) "
            f"in [{baseline_calibration.rho_0_lower:.6g}, "
            f"{baseline_calibration.rho_0_upper:.6g}].",
            "The effect is recombined from the two corrected arm intervals, so it makes "
            "no independence assumption between arms beyond the one the underlying "
            "two-arm region already makes.",
            *(
                [
                    "Each cohort's labels weight its classified rollouts equally, and so does "
                    "the comparison, so they measure the judge's accuracy for both arms.",
                    "Coverage is the union bound over that region and both arm calibrations.",
                ]
                if certified
                else []
            ),
        ],
        eligibility="certifying" if certified else "diagnostic",
        warnings=[_uncertified_correction_warning(gap)] if gap is not None else [],
    )


def _apply_judge_correction_to_comparisons(
    plan: AnalysisPlan,
    data: NormalizedData,
    population: dict[str, dict[str, _PromptRollouts]],
    comparisons: list[ComparisonResult],
    by_cohort: dict[str, JudgeCalibration],
    diagnostics: list[Diagnostic],
) -> None:
    """Append validated corrections to comparisons that expose both arm bounds."""

    if not by_cohort:
        return
    spec = plan.inputs.judge_calibration
    assert spec is not None
    target_weights: dict[tuple[str, str], dict[str, float]] = {}
    for comparison in plan.expanded_comparisons():
        try:
            targets = _comparison_targets(plan, data, comparison)
        except StopIteration:
            continue
        for target in targets:
            target_weights[(comparison.id, target.result_id)] = dict(
                zip(target.prompt_ids, target.weights, strict=True)
            )
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    skipped: list[str] = []
    for result in comparisons:
        candidate = by_cohort.get(result.candidate_cohort)
        baseline = by_cohort.get(result.baseline_cohort)
        if candidate is None or baseline is None:
            skipped.append(result.result_id or result.comparison_id)
            continue
        weights = target_weights[(result.comparison_id, result.result_id or result.comparison_id)]
        # Each arm is corrected with its own cohort's labels, so each must be identified.
        gaps: list[str] = []
        for arm, cohort_id in (
            ("candidate", result.candidate_cohort),
            ("baseline", result.baseline_cohort),
        ):
            gap = _pooled_calibration_gap(
                population.get(cohort_id, {}),
                design_type=plan.design.type,
                measure_type=measure_types[result.measure_id],
                weights=weights,
            )
            if gap is not None:
                gaps.append(f"{arm} cohort {cohort_id!r}: {gap}")
        interval = _judge_corrected_comparison_interval(
            result=result,
            candidate_calibration=candidate,
            baseline_calibration=baseline,
            family_id=spec.family,
            gap="; ".join(gaps) or None,
        )
        if interval is None:
            skipped.append(result.result_id or result.comparison_id)
            continue
        result.intervals.append(interval)
    if skipped:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON",
                severity="WARNING",
                message=(
                    "some comparisons could not be judge-corrected because their primary "
                    "interval does not expose both arm bounds, or a cohort has no "
                    "calibration; those effects remain on the judge-reported scale"
                ),
                scope="comparisons",
                details={"result_ids": sorted(skipped)[:20], "count": len(skipped)},
            )
        )


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
                        index=_block_index(data),
                    )
                    candidate_blocks = direct_bounded_blocks(
                        data.observations,
                        data.manifest,
                        profile,
                        cohort_id=spec.candidate,
                        prompt_ids=target.prompt_ids,
                        prompt_weights=weights,
                        index=_block_index(data),
                    )
                    baseline_blocks = direct_bounded_blocks(
                        data.observations,
                        data.manifest,
                        profile,
                        cohort_id=spec.baseline,
                        prompt_ids=target.prompt_ids,
                        prompt_weights=weights,
                        index=_block_index(data),
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

        family = next(
            (family for family in plan.multiplicity.families if family.id == spec.family), None
        )
        if family is None:
            # Plan validation reports the undeclared family. Without its budget the set has no
            # joint coverage to report.
            continue
        _, _, simultaneous_coverage = _coverage_floors(family.alpha, spec.coverage_mode)
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


def _certifying_judge_correction(
    evidence: ComparisonResult | GroupRiskResult | None,
) -> RiskIntervalResult | None:
    """Return the judge correction a claim may be decided from, if one exists.

    Only a ``validated`` correction qualifies. An ``assumed`` correction is a declared
    sensitivity carrying no coverage guarantee, and must never move a safety decision;
    the eligibility check enforces that structurally rather than by naming modes.
    """

    if evidence is None:
        return None
    return next(
        (
            interval
            for interval in evidence.intervals
            if interval.interval_id.startswith("judge_corrected_")
            and interval.eligibility == "certifying"
            and interval.lower is not None
            and interval.upper is not None
        ),
        None,
    )


def _noisy_valid_claim(
    *,
    claim: ClaimSpec,
    evidence: ComparisonResult | GroupRiskResult | None,
    eligible: bool,
    family_alpha: float,
    judge_summary: dict[str, Any] | None,
    gap: str | None,
) -> tuple[Decision, list[str], float | None, float | None, str, float]:
    """Decide an aggregate claim with the finite-sample noisy-but-valid test."""

    method = "noisy_valid_certification_test"
    if not isinstance(evidence, GroupRiskResult) or not eligible:
        return (
            "INCONCLUSIVE",
            [
                "No complete certifying judge-reported bound is available for the "
                "noisy-but-valid test."
            ],
            None,
            None,
            method,
            family_alpha,
        )
    if gap is not None:
        # The truth strata pool the cohort's labels exactly as the judge strata do.
        return (
            "INCONCLUSIVE",
            [
                f"The noisy-but-valid test cannot certify this row: {gap}. Its labels are a "
                "simple random sample of the cohort's classified rollouts, so they measure "
                "the judge's sensitivity and false-positive rate only for a weighted mean "
                "that weights those rollouts equally, over the whole cohort in a fixed design."
            ],
            None,
            None,
            method,
            family_alpha,
        )
    summary = judge_summary or {}
    strata = (summary.get("truth_strata") or {}).get(evidence.cohort_id) or {}
    judge_alpha = summary.get("family_alpha")
    per_cohort_alpha = summary.get("per_cohort_alpha")
    base = next((item for item in evidence.intervals if item.role == "primary"), None)
    if (
        summary.get("mode") != "validated"
        or summary.get("sampling") != "simple_random"
        or "truly_unsafe" not in strata
        or "truly_safe" not in strata
        or judge_alpha is None
        or per_cohort_alpha is None
        or base is None
        or base.alpha_upper is None
        or evidence.lower is None
        or evidence.upper is None
    ):
        return (
            "INCONCLUSIVE",
            [
                "The noisy-but-valid test requires a validated judge calibration from a simple "
                "random sample of labelled rollouts with both true classes represented."
            ],
            None,
            None,
            method,
            family_alpha,
        )
    try:
        # A projected upper endpoint misses whenever the joint event fails, so its error is the
        # simultaneous budget, not the alpha its raw endpoint was allocated.
        reported_alpha = (
            1 - base.simultaneous_coverage
            if base.projection_method is not None and base.simultaneous_coverage is not None
            else base.alpha_upper
        )
        result = noisy_valid_certification(
            judge_reported_lower=evidence.lower,
            judge_reported_upper=evidence.upper,
            judge_reported_alpha=reported_alpha,
            truly_unsafe=TruthStratumCounts(**strata["truly_unsafe"]),
            truly_safe=TruthStratumCounts(**strata["truly_safe"]),
            calibration_alpha=float(per_cohort_alpha),
            acceptable_upper=claim.acceptable_upper,
            unacceptable_lower=claim.unacceptable_lower,
        )
    except ValueError as exc:
        return "INCONCLUSIVE", [str(exc)], None, None, method, family_alpha
    reasons = [
        "Noisy-but-valid certification on true failure risk: judge-reported interval "
        f"[{result.judge_reported_lower:.12g}, {result.judge_reported_upper:.12g}]; PASS "
        "requires its upper endpoint below the critical threshold "
        f"{result.critical_threshold_pass:.12g}, FAIL requires its lower endpoint above "
        f"{result.critical_threshold_fail:.12g}.",
        "Calibration from a simple random sample of labelled rollouts: sensitivity in "
        f"[{result.sensitivity_lower:.12g}, {result.sensitivity_upper:.12g}], false-positive "
        f"rate in [{result.false_positive_lower:.12g}, {result.false_positive_upper:.12g}]; "
        f"identified true-risk set [{result.true_risk_lower:.12g}, "
        f"{result.true_risk_upper:.12g}].",
        f"Type-I error for a one-sided decision is at most {result.alpha_pass:.12g}: the "
        "judge-reported endpoint's alpha plus half the calibration budget.",
    ]
    if result.unidentified_reason == "contradicted":
        reasons.append(
            "The judge-reported rate is outside what the calibrated sensitivity and "
            "false-positive rate can produce, so the calibration and the data disagree and "
            "nothing identifies true risk."
        )
    elif not result.identified:
        reasons.append(
            "The calibration does not establish an informative judge (sensitivity plus "
            "specificity may not exceed one), so nothing identifies true risk."
        )
    return (
        result.decision,
        reasons,
        result.true_risk_lower,
        result.true_risk_upper,
        method,
        family_alpha + float(judge_alpha),
    )


def _claims(
    plan: AnalysisPlan,
    data: NormalizedData,
    risks: list[GroupRiskResult],
    comparisons: list[ComparisonResult],
    integrity_valid: bool,
    diagnostics: list[Diagnostic],
    judge_summary: dict[str, Any] | None = None,
) -> list[ClaimResult]:
    risk_index = {(item.cohort_id, item.group_id, item.measure_id): item for item in risks}
    comparison_index = {item.comparison_id: item for item in comparisons if item.level is None}
    families = {family.id: family for family in plan.multiplicity.families}
    validated_judge = (
        plan.inputs.judge_calibration is not None
        and plan.inputs.judge_calibration.mode == "validated"
    )
    population = (
        _labelled_population(plan, data)
        if any(claim.method == "noisy_valid_test" for claim in plan.claims)
        else {}
    )
    group_weights = {group.id: group.weights for group in data.groups}
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    results: list[ClaimResult] = []
    corrected_claims: list[str] = []
    uncorrected_claims: list[str] = []
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
        method = evidence.method if evidence else "unavailable"
        evidence_refs = [ref]
        family = families.get(claim.family)
        if family is None:
            # Plan validation reports the undeclared family. Without its error budget the claim
            # cannot be decided, so it stays INCONCLUSIVE and the report is still written.
            results.append(
                ClaimResult(
                    claim_id=claim.id,
                    title=claim.title,
                    status="INCONCLUSIVE",
                    statistical_status="INCONCLUSIVE",
                    policy_status="INCONCLUSIVE",
                    estimate=estimate,
                    lower=lower,
                    upper=upper,
                    acceptable_upper=claim.acceptable_upper,
                    unacceptable_lower=claim.unacceptable_lower,
                    alpha=None,
                    multiplicity_family=claim.family,
                    method=method,
                    mandatory=claim.mandatory,
                    evidence_refs=evidence_refs,
                    reasons=[
                        f"The claim's multiplicity family {claim.family!r} is not declared, so "
                        "it has no error budget."
                    ],
                    warnings=list(evidence.warnings) if evidence else [],
                )
            )
            continue
        alpha = family.alpha
        correction_note: str | None = None

        # A validated correction retargets the claim from judge-reported risk onto true
        # risk. The threshold in the plan is about the model, not about the judge.
        correction = _certifying_judge_correction(evidence)
        if correction is not None:
            correction_note = (
                f"Decided on true failure risk using the validated judge correction "
                f"[{cast(float, correction.lower):.12g}, "
                f"{cast(float, correction.upper):.12g}]; the uncorrected judge-system "
                f"interval was [{cast(float, correction.raw_lower):.12g}, "
                f"{cast(float, correction.raw_upper):.12g}]."
            )
            lower = correction.lower
            upper = correction.upper
            # Under a calibration box there is a range of corrected point estimates,
            # not a single one, so no scalar estimate is reported.
            estimate = None
            method = correction.method
            evidence_refs.append(f"interval:{correction.interval_id}")
            if correction.family_alpha is not None:
                alpha = family.alpha + correction.family_alpha
            if claim.method != "noisy_valid_test":
                corrected_claims.append(claim.id)
        elif evidence is not None and claim.method != "noisy_valid_test":
            uncorrected_claims.append(claim.id)

        if claim.method == "noisy_valid_test":
            gap = (
                _pooled_calibration_gap(
                    population.get(evidence.cohort_id, {}),
                    design_type=plan.design.type,
                    measure_type=measure_types[evidence.measure_id],
                    weights=group_weights[evidence.group_id],
                )
                if isinstance(evidence, GroupRiskResult)
                else None
            )
            status, reasons, lower, upper, method, alpha = _noisy_valid_claim(
                claim=claim,
                evidence=evidence,
                eligible=eligible,
                family_alpha=family.alpha,
                judge_summary=judge_summary,
                gap=gap,
            )
            estimate = None
            correction_note = None
            evidence_refs.append("test:noisy_valid_certification")
        elif validated_judge and correction is None:
            # A validated calibration makes every claim a claim about true risk. Without a
            # certifying correction only judge-reported risk remains, and the labels may show
            # the judge missing failures: a refused calibration used to pass claims that way.
            status = "INCONCLUSIVE"
            # A correction the labels cannot certify for this estimand says why itself.
            why = [
                warning
                for interval in (evidence.intervals if evidence is not None else [])
                if interval.interval_id == "judge_corrected_validated"
                for warning in interval.warnings
            ]
            reasons = [
                "The plan declares a validated judge calibration, but this claim's evidence has "
                "no certifying correction, so it is not decided on judge-reported risk."
                + ("" if why else " The judge diagnostics say why."),
                *why,
            ]
        else:
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
        if correction_note is not None:
            reasons.insert(0, correction_note)
            if alpha != family.alpha:
                reasons.append(
                    f"The error budget is the union of the claim family ({family.alpha:.12g}) "
                    f"and the judge calibration, giving {alpha:.12g}."
                )
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
                alpha=alpha,
                multiplicity_family=claim.family,
                method=method,
                mandatory=claim.mandatory,
                evidence_refs=evidence_refs,
                reasons=reasons,
                warnings=list(evidence.warnings) if evidence else [],
            )
        )
    if corrected_claims:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CORRECTED_CLAIMS",
                severity="INFO",
                message=(
                    "threshold claims were decided on true failure risk using the validated "
                    "judge correction rather than the uncorrected judge-system interval"
                ),
                scope="claims",
                details={"claim_ids": sorted(corrected_claims)},
            )
        )
    if validated_judge and uncorrected_claims:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CORRECTION_UNAVAILABLE",
                severity="WARNING",
                message=(
                    "the plan declares a validated judge calibration, but these claims have no "
                    "certifying correction for their evidence, so they are INCONCLUSIVE rather "
                    "than decided on judge-reported risk"
                ),
                scope="claims",
                details={"claim_ids": sorted(uncorrected_claims)},
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
    comparisons = _comparisons(resolved, data, estimates, diagnostics)
    # Applied last so it corrects the bounds that are actually displayed, after both
    # the hierarchy and comparison region projections have settled.
    judge_calibration_summary = _apply_judge_correction(
        resolved, data, hierarchy_nodes, risks, hierarchy_risks, comparisons, diagnostics
    )
    panel_correction = _apply_panel_derived_correction(
        resolved, risks, hierarchy_risks, diagnostics
    )
    comparison_sets = _comparison_set_results(resolved, comparisons, risks)
    integrity_status = _integrity_status(diagnostics)
    integrity_certifiable = integrity_status != "invalid"
    claims = _claims(
        resolved,
        data,
        risks,
        comparisons,
        integrity_certifiable,
        diagnostics,
        judge_calibration_summary,
    )
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
    judge_panel = _judge_panel_summary(resolved, data)
    if judge_panel is not None and panel_correction is not None:
        judge_panel["derived_correction"] = panel_correction
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
            **({"judge_panel": judge_panel} if judge_panel else {}),
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
