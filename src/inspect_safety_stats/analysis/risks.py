"""Group and hierarchy-node risk results."""

from __future__ import annotations

from typing import cast

import numpy as np

from inspect_safety_stats.analysis.direct_mean import (
    _calibration_key,
    _direct_intervals,
    _direct_mean_context,
    _DirectMeanContext,
    _hierarchy_region_target_id,
    _HierarchyNode,
    _worst_of_k_direct_interval,
)
from inspect_safety_stats.analysis.estimates import prompt_blocked
from inspect_safety_stats.analysis.measures import (
    _coverage_floors,
    _independence,
    _measure_parameters,
    _risk_value,
    _thresholds,
)
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    Diagnostic,
    Eligibility,
    GroupRiskResult,
    HierarchyRiskResult,
    PromptEstimate,
    RiskIntervalResult,
)
from inspect_safety_stats.schema.scopes import cohort_scope, node_id, plan_scope
from inspect_safety_stats.stats.risk import worst_of_k_estimate


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
                    if complete_estimate and measure.type == "worst_of_k":
                        # Plugging p_hat into 1 - (1 - p) ** k understates the risk, so the
                        # estimate is the unbiased U-statistic instead.
                        if measure.k is None:
                            raise ValueError(f"risk measure {measure.id!r} (worst_of_k) requires k")
                        classified = [item.n_valid for item in prompt_results]
                        if min(classified) >= measure.k:
                            estimate = min(
                                1.0,
                                worst_of_k_estimate(
                                    [item.n_failure for item in prompt_results],
                                    classified,
                                    weights,
                                    k=measure.k,
                                ),
                            )
                        else:
                            warnings.append(
                                f"No unbiased estimate of {measure.k} attempts: a prompt has only "
                                f"{min(classified)} classified rollouts."
                            )
                    elif complete_estimate:
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
                        # _risk_value has already refused a threshold measure without one.
                        if measure.type == "threshold_exceedance" and threshold is not None:
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
                            scope=cohort_scope(
                                cohort_id, "groups", group.id, "measures", measure.id
                            ),
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
                    independence = _independence(plan.outputs.direct_mean_intervals)
                    calibration = direct.calibrations.get(
                        _calibration_key(cohort_id, target_id, independence.primary)
                    )
                    if calibration is None:
                        direct_warnings.append(
                            "The direct-mean multiplicity family is unavailable."
                        )
                    elif any(
                        (item.n_total <= 0 and plan.design.type != "sampled")
                        or prompt_blocked(diagnostics, cohort_id, item.prompt_id)
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
                                    scope=cohort_scope(
                                cohort_id, "groups", group.id, "measures", measure.id
                            ),
                                )
                            )

                if measure.direct_interval and group.id in direct.groups:
                    worst_interval, worst_warnings = _worst_of_k_direct_interval(
                        plan=plan,
                        data=data,
                        measure=measure,
                        cohort_id=cohort_id,
                        group_id=group.id,
                        prompt_ids=prompt_ids,
                        weights=weights,
                        prompt_results=prompt_results,
                        direct=direct,
                        diagnostics=diagnostics,
                    )
                    direct_warnings.extend(worst_warnings)
                    if worst_interval is not None:
                        computed_direct_intervals.append(worst_interval)

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
            independence = _independence(plan.outputs.direct_mean_intervals)
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
                or prompt_blocked(diagnostics, cohort_id, item.prompt_id)
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
                            scope=cohort_scope(
                                cohort_id,
                                "groups",
                                node.group_id,
                                "levels",
                                node.level,
                                "nodes",
                                node_id(node.path),
                            ),
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
                scope=plan_scope("design", "hierarchy"),
                details={"nodes": sorted(unsampled_nodes)[:20], "count": len(unsampled_nodes)},
            )
        )
    return results
