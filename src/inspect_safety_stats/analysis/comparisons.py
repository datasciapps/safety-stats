"""The comparison stage and comparison sets."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from inspect_safety_stats.analysis.comparison_intervals import (
    _comparison_calibrations,
    _comparison_interval,
    _comparison_target_key,
    _comparison_targets,
    _ComparisonTarget,
    _difference_from_blocks,
    _difference_rectangle_from_blocks,
    _paired_comparison_diagnostics,
    _ratio_from_blocks,
)
from inspect_safety_stats.analysis.comparison_regions import (
    _project_comparison_hierarchy_regions,
    _project_multi_arm_comparison_regions,
)
from inspect_safety_stats.analysis.estimates import _block_index, prompt_blocked
from inspect_safety_stats.analysis.measures import _coverage_floors, _independence
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    ComparisonResult,
    ComparisonSetResult,
    Diagnostic,
    GroupRiskResult,
    PromptEstimate,
    RiskIntervalResult,
)
from inspect_safety_stats.schema.scopes import scope_path
from inspect_safety_stats.stats.comparisons import DirectComparisonBounds, RatioComparisonBounds
from inspect_safety_stats.stats.independence import (
    comparison_contrast_blocks,
    direct_bounded_blocks,
)


def _target_scope(comparison_id: str, target: _ComparisonTarget) -> str:
    """``comparisons/<id>``, or for a hierarchy node ``.../levels/<level>/nodes/<path>``."""
    if target.level is None:
        return scope_path("comparisons", comparison_id)
    node = "/".join(f"{key}={value}" for key, value in target.path.items())
    return scope_path("comparisons", comparison_id, "levels", target.level, "nodes", node)


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
        independence = _independence(spec)
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
                or prompt_blocked(diagnostics, item.cohort_id, item.prompt_id)
                for item in [*candidate_items, *baseline_items]
            )
            if blocked:
                warnings.append("Both arms require a complete valid attempted-rollout ledger.")
            else:
                methods = [spec.method, *spec.reference_methods]
                profile_ids = [
                    independence.primary,
                    *independence.sensitivities,
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
                            if profile_id != independence.primary
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
                                    scope=_target_scope(spec.id, target),
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
                        newcombe_withheld,
                    ) = _paired_comparison_diagnostics(spec, data, set(target.prompt_ids))
                    partial = diagnostic_outputs.get("partial_pairing")
                    if partial is not None:
                        message = (
                            f"{partial['pair_count'] - partial['scored_pair_count']} of "
                            f"{partial['pair_count']} pairs have an unscored arm; the paired "
                            "diagnostics bound every completion of the unscored outcomes"
                        )
                        diagnostics.append(
                            Diagnostic(
                                code="PAIR_UNSCORED_COMPLETED",
                                severity="WARNING",
                                message=message,
                                scope=scope_path("comparisons", spec.id),
                            )
                        )
                        warnings.append(message)
                    if newcombe_withheld is not None:
                        diagnostics.append(
                            Diagnostic(
                                code="PAIR_NEWCOMBE_UNAVAILABLE",
                                severity="WARNING",
                                message=newcombe_withheld,
                                scope=scope_path("comparisons", spec.id),
                            )
                        )
                        warnings.append(newcombe_withheld)
                except ValueError as exc:
                    diagnostics.append(
                        Diagnostic(
                            code="PAIR_BROKEN",
                            severity="ERROR",
                            message=str(exc),
                            scope=scope_path("comparisons", spec.id),
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
                    independence_profile=independence.primary,
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
