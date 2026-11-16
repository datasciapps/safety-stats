"""The judge correction stage applied to risks, nodes and comparisons."""

from __future__ import annotations

from typing import Any

from inspect_safety_stats.analysis.comparison_judges import _apply_judge_correction_to_comparisons
from inspect_safety_stats.analysis.direct_mean import _HierarchyNode
from inspect_safety_stats.analysis.judge_calibration import (
    _JUDGE_CORRECTABLE_MEASURES,
    _assumed_judge_calibrations,
    _judge_corrected_interval,
    _judge_scope,
    _judge_scopes,
    _JudgeScope,
    _labelled_population,
    _pooled_calibration_gap,
    _scope_population,
    _transported_judge_interval,
    _uncertified_correction_warning,
    _unclassified_share,
    _validated_judge_calibrations,
)
from inspect_safety_stats.config.loader import report_path
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    ComparisonResult,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
)
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    delta_method_corrected_risk,
    transported_sensitivity_lower,
)


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
    unclassified_share: float = 0.0,
    scoped_group: str | None = None,
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
    warnings = [_uncertified_correction_warning(gap, scoped_group)] if gap is not None else []
    if unclassified_share > 0:
        warnings.append(
            f"It covers the classified rollouts only: {unclassified_share:.6g} of the row's "
            "weight rests on unclassified rollouts, which the exact box keeps at their worst "
            "case."
        )
    return RiskIntervalResult(
        interval_id="judge_corrected_validated_delta",
        lower=delta.lower,
        upper=delta.upper,
        method="delta_method_judge_misclassification_correction",
        role="sensitivity",
        varied_assumption="judge_error",
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
        warnings=warnings,
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
    ``validated`` mode one calibration is estimated per cohort from human labels. A
    calibration scoped to groups corrects only the rows of those groups, and validated mode
    then estimates one calibration per cohort and group.
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
    population = _labelled_population(plan, data)
    group_weights = {group.id: group.weights for group in data.groups}
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    node_weights = {
        (node.group_id, node.level, tuple(node.path.items())): dict(
            zip(node.prompt_ids, node.weights, strict=True)
        )
        for node in hierarchy_nodes
    }

    def weighting(result: GroupRiskResult | HierarchyRiskResult) -> tuple[dict[str, float], str]:
        """The prompt weights behind a row, and its measure type."""
        if isinstance(result, GroupRiskResult):
            return group_weights[result.group_id], measure_types[result.measure_id]
        # Hierarchy nodes are weighted means over their own prompts.
        key = (result.group_id, result.level, tuple(result.path.items()))
        return node_weights[key], "weighted_mean"

    if spec.mode == "transported":
        published = spec.published_set
        if published is None:
            raise ValueError("a transported inputs.judge_calibration requires a published_set")
        sensitivity_lower = transported_sensitivity_lower(
            judge_flagged_unsafe=published.judge_flagged_unsafe,
            human_unsafe=published.human_unsafe,
            alpha=published.sensitivity_alpha,
        )
        for result in targets:
            interval = _transported_judge_interval(
                result=result, published=published, sensitivity_lower=sensitivity_lower
            )
            if interval is not None:
                result.intervals.append(interval)
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CALIBRATION_TRANSPORTED",
                severity="INFO",
                message=(
                    "Experimental: judge-corrected rows use a sensitivity transported from "
                    f"{published.name!r}; they are sensitivity analyses and decide no claim"
                ),
                scope="inputs.judge_calibration",
            )
        )
        return {
            "mode": "transported",
            "experimental": True,
            "published_set": {
                "name": published.name,
                "source": published.source,
                "judge": published.judge,
                "rubric": published.rubric,
                "human_unsafe": published.human_unsafe,
                "judge_flagged_unsafe": published.judge_flagged_unsafe,
                "sensitivity_alpha": published.sensitivity_alpha,
            },
            "sensitivity_lower": sensitivity_lower,
        }

    if spec.mode == "assumed":
        calibrations = _assumed_judge_calibrations(plan)
        for result in targets:
            if _judge_scope(plan, result.cohort_id, result.group_id) is None:
                continue
            weights, measure_type = weighting(result)
            share = _unclassified_share(
                population.get(result.cohort_id, {}), weights, measure_type
            )
            for calibration in calibrations:
                interval = _judge_corrected_interval(
                    result=result,
                    calibration=calibration,
                    primary_scenario=spec.primary,
                    family_id=None,
                    judge_alpha=None,
                    unclassified_share=share,
                    scoped_groups=spec.groups,
                )
                if interval is not None:
                    result.intervals.append(interval)
        return {
            "mode": "assumed",
            "scenarios": [item.scenario_id for item in calibrations],
            "primary": spec.primary,
            **({"groups": list(spec.groups)} if spec.groups is not None else {}),
        }

    by_scope, summaries, judge_alpha, truth_strata = _validated_judge_calibrations(
        plan, data, diagnostics
    )
    scope_count = len(_judge_scopes(plan))
    stratified = spec.sampling == "stratified_without_replacement"
    for result in targets:
        scope = _judge_scope(plan, result.cohort_id, result.group_id)
        measured = by_scope.get(scope) if scope is not None else None
        if scope is None or measured is None:
            continue
        weights, measure_type = weighting(result)
        scope_population = _scope_population(population, data, scope)
        gap = _pooled_calibration_gap(
            scope_population,
            design_type=plan.design.type,
            measure_type=measure_type,
            weights=weights,
            group_id=scope[1],
        )
        share = _unclassified_share(scope_population, weights, measure_type)
        interval = _judge_corrected_interval(
            result=result,
            calibration=measured,
            primary_scenario=None,
            family_id=spec.family,
            judge_alpha=judge_alpha,
            gap=gap,
            unclassified_share=share,
            scoped_group=scope[1],
            stratified=stratified,
        )
        if interval is not None:
            result.intervals.append(interval)
        # The reference propagates the weighted mean's variance, so only a weighted-mean
        # row has one: on a tail row it would describe a different estimand. Its stratum
        # proportions assume simple random samples within the judge strata, so a stratified
        # design has none.
        if (
            isinstance(result, GroupRiskResult)
            and measure_type == "weighted_mean"
            and judge_alpha is not None
            and not stratified
        ):
            reference = _delta_reference_interval(
                result=result,
                data=data,
                strata=summaries.get(scope, {}),
                alpha=judge_alpha / scope_count,
                family_id=spec.family,
                gap=gap,
                unclassified_share=share,
                scoped_group=scope[1],
            )
            if reference is not None:
                result.intervals.append(reference)
    _apply_judge_correction_to_comparisons(
        plan, data, population, comparisons, by_scope, spec.family, diagnostics
    )
    per_calibration_alpha = judge_alpha / scope_count if judge_alpha is not None else None
    summary: dict[str, Any]
    if spec.groups is None:
        summary = {
            "mode": "validated",
            "family": spec.family,
            "family_alpha": judge_alpha,
            "per_cohort_alpha": per_calibration_alpha,
            "sampling": spec.sampling,
            "validation_manifest": str(spec.validation_manifest),
            "cohorts": {cohort_id: item for (cohort_id, _), item in summaries.items()},
            "truth_strata": {cohort_id: item for (cohort_id, _), item in truth_strata.items()},
        }
    else:

        def nested(by_scope: dict[_JudgeScope, Any]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for (cohort_id, group_id), item in by_scope.items():
                out.setdefault(cohort_id, {})[str(group_id)] = item
            return out

        summary = {
            "mode": "validated",
            "family": spec.family,
            "family_alpha": judge_alpha,
            "groups": list(spec.groups),
            "per_calibration_alpha": per_calibration_alpha,
            "sampling": spec.sampling,
            "validation_manifest": str(spec.validation_manifest),
            "cohorts": nested(summaries),
            "truth_strata": nested(truth_strata),
        }
    if stratified:
        # Only a stratified design records it, so other reports keep their bytes.
        summary["experimental"] = True
        if spec.validation_strata is not None:
            summary["validation_strata"] = report_path(spec.validation_strata, plan)
    return summary
