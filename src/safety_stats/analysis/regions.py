"""Projection of hierarchy intervals onto their common confidence region."""

from __future__ import annotations

from typing import cast

import numpy as np

from inspect_safety_stats.analysis.direct_mean import _hierarchy_region_target_id, _HierarchyNode
from inspect_safety_stats.analysis.results import RiskRows
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
)
from inspect_safety_stats.schema.scopes import cohort_scope
from inspect_safety_stats.stats.hierarchy_region import (
    LinearIntervalConstraint,
    project_hierarchy_confidence_region,
)


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


def project_hierarchy_regions(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
    rows: RiskRows,
    diagnostics: list[Diagnostic],
) -> RiskRows:
    """The rows with their direct intervals projected onto the common hierarchy region.

    The rows given are left as they were.
    """

    projected = rows.copy()
    _project_onto_regions(
        plan, data, nodes, projected.risks, projected.hierarchy_risks, diagnostics
    )
    return projected


def _project_onto_regions(
    plan: AnalysisPlan,
    data: NormalizedData,
    nodes: list[_HierarchyNode],
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    diagnostics: list[Diagnostic],
) -> None:
    """Project the rows' direct intervals in place: only on rows the stage has copied."""

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
                        scope=cohort_scope(
                            cohort_id, "methods", method, "units", independence_unit
                        ),
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
