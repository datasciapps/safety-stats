"""Projection of comparison intervals onto hierarchy confidence regions."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from inspect_safety_stats.analysis.comparison_intervals import _ComparisonTarget
from inspect_safety_stats.analysis.regions import (
    _direct_interval_for_region,
    _keep_only_joint_coverage,
)
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.schema.reports import ComparisonResult, Diagnostic, RiskIntervalResult
from inspect_safety_stats.stats.hierarchy_region import (
    LinearIntervalConstraint,
    project_contrast_hierarchy_confidence_region,
    project_hierarchy_confidence_region,
    project_ratio_hierarchy_confidence_region,
    project_two_arm_difference_hierarchy_confidence_region,
)


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
