"""The judge correction stage applied to risks, nodes and comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inspect_safety_stats.analysis.comparison_judges import _apply_judge_correction_to_comparisons
from inspect_safety_stats.analysis.direct_mean import _HierarchyNode
from inspect_safety_stats.analysis.judge_calibration import (
    JUDGE_CORRECTABLE_MEASURES,
    PromptRollouts,
    _assumed_judge_calibrations,
    _judge_corrected_interval,
    _judge_scope,
    _judge_scopes,
    _JudgeScope,
    _labelled_population,
    _per_cohort_scenarios,
    _policy_rollouts,
    _row_weights,
    _scope_population,
    _transported_judge_interval,
    _uncertified_correction_warning,
    _unclassified_share,
    _validated_judge_calibrations,
    pooled_calibration_gap,
)
from inspect_safety_stats.analysis.results import RiskRows
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
from inspect_safety_stats.schema.scopes import cohort_scope, plan_scope, scope_path
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    TruthStratumCounts,
    delta_method_corrected_risk,
    lang_reiczigel_corrected_risk,
    transported_sensitivity_lower,
)


def _judge_reported_counts(
    data: NormalizedData, cohort_id: str, group_id: str
) -> list[tuple[float, int, int]] | None:
    """Each prompt's ``(weight, judge-reported failures, classified rollouts)`` in a group.

    ``None`` when the group is unknown or a prompt has no classified rollout.
    """

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
    if any(classified == 0 for _, classified in counts.values()):
        return None
    return [
        (group.weights[prompt_id], failures, classified)
        for prompt_id, (failures, classified) in counts.items()
    ]


def _judge_reported_variance(data: NormalizedData, cohort_id: str, group_id: str) -> float | None:
    """Binomial variance of the design-weighted judge-reported mean under rollout independence."""

    counts = _judge_reported_counts(data, cohort_id, group_id)
    if counts is None:
        return None
    variance = 0.0
    for weight, failures, classified in counts:
        rate = failures / classified
        variance += weight**2 * rate * (1 - rate) / classified
    return variance


def _judge_reported_effective_size(
    data: NormalizedData, cohort_id: str, group_id: str
) -> float | None:
    """Rollouts an unweighted binomial mean would need for the weighted mean's variance.

    Under rollout independence and a common rate ``p`` the design-weighted mean has variance
    ``p (1 - p) sum_j w_j^2 / n_j`` for weights summing to one, so the effective size is
    ``(sum_j w_j)^2 / sum_j (w_j^2 / n_j)``: the classified count when every rollout weighs
    the same.
    """

    counts = _judge_reported_counts(data, cohort_id, group_id)
    if counts is None:
        return None
    total = sum(weight for weight, _, _ in counts)
    spread = sum(weight**2 / classified for weight, _, classified in counts)
    if total <= 0 or spread <= 0:
        return None
    return total**2 / spread


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


def _lang_reiczigel_interval(
    *,
    result: GroupRiskResult,
    data: NormalizedData,
    truth: dict[str, Any],
    alpha: float,
    family_id: str | None,
    gap: str | None = None,
    unclassified_share: float = 0.0,
    scoped_group: str | None = None,
    policy_classified: tuple[int, int] = (0, 0),
) -> RiskIntervalResult | str:
    """Lang-Reiczigel limits beside the exact validated correction, or why there are none."""

    if result.estimate is None or result.lower is None or result.upper is None:
        return "the row has no judge-reported estimate"
    if any(policy_classified):
        return (
            f"{sum(policy_classified)} of the row's rollouts were classified by a plan policy, "
            "not by the judge, and the limits would read them as judge verdicts"
        )
    unsafe = truth.get("truly_unsafe")
    safe = truth.get("truly_safe")
    if not unsafe or not safe:
        missing = "truly unsafe" if not unsafe else "truly safe"
        return f"no labelled rollout is {missing}, so the judge's accuracy is not estimated"
    size = _judge_reported_effective_size(data, result.cohort_id, result.group_id)
    if size is None:
        return "a prompt in the row has no classified rollout"
    try:
        limits = lang_reiczigel_corrected_risk(
            apparent_rate=result.estimate,
            apparent_size=size,
            truly_unsafe=TruthStratumCounts(**unsafe),
            truly_safe=TruthStratumCounts(**safe),
            alpha=alpha,
        )
    except ValueError as exc:
        return str(exc)
    warnings = [_uncertified_correction_warning(gap, scoped_group)] if gap is not None else []
    if unclassified_share > 0:
        warnings.append(
            f"It covers the classified rollouts only: {unclassified_share:.6g} of the row's "
            "weight rests on unclassified rollouts, which the exact box keeps at their worst "
            "case."
        )
    if limits.sensitivity_estimate < 0.5 or limits.specificity_estimate < 0.5:
        warnings.append(
            f"Estimated sensitivity {limits.sensitivity_estimate:.6g} or specificity "
            f"{limits.specificity_estimate:.6g} is below 0.5; the method was developed for "
            "both at or above 0.5."
        )
    if limits.upper == 0:
        warnings.append(
            "The judge reported fewer failures than its estimated false-positive rate "
            "implies, so the whole interval fell below zero and was clipped to the single "
            "point 0. The data contradict the estimated accuracies; this is not evidence "
            "of zero risk."
        )
    return RiskIntervalResult(
        interval_id="judge_corrected_validated_lang_reiczigel",
        lower=limits.lower,
        upper=limits.upper,
        method="lang_reiczigel_judge_misclassification_correction",
        role="sensitivity",
        varied_assumption="judge_error",
        coverage_scope=(
            f"{result.coverage_scope}_judge_corrected_lang_reiczigel"
            if result.coverage_scope
            else "judge_corrected_lang_reiczigel"
        ),
        family=family_id,
        family_alpha=alpha,
        completion_estimate_lower=limits.estimate,
        completion_estimate_upper=limits.estimate,
        effective_sample_size=size,
        raw_lower=result.lower,
        raw_upper=result.upper,
        calibration_method="validated_judge_calibration_lang_reiczigel",
        assumptions=[
            "The reported estimand is true failure risk, not judge-reported failure risk.",
            "Approximate limits of Lang and Reiczigel (2014) for prevalence adjusted for "
            "estimated sensitivity and specificity: Agresti-Coull-adjusted counts, a bias "
            "shift and a Wald half-width. They carry no coverage guarantee and cannot "
            "support a claim.",
            "Sensitivity and specificity are estimated from the labelled rollouts by true "
            "class, which requires the labels to be a simple random sample of the cohort's "
            "classified rollouts.",
            "The judge-reported rate is the row's design-weighted mean over an effective "
            f"{size:.6g} rollouts, (sum_j w_j)^2 / sum_j (w_j^2 / n_j), under rollout "
            "independence.",
            "In simulation (research/judge_correction_lang_reiczigel.py) their coverage "
            "falls to 91.4% at nominal 95% where prevalence is 2% or below and labels "
            "number 300 or fewer.",
        ],
        eligibility="diagnostic",
        warnings=warnings,
    )


def _policy_classified_diagnostics(
    population: dict[str, dict[str, PromptRollouts]],
    policy_rows: dict[str, set[str]],
    diagnostics: list[Diagnostic],
) -> None:
    """Say, per cohort, that corrected rows hold rollouts a plan policy classified."""

    for cohort_id in sorted(policy_rows):
        prompts = population.get(cohort_id, {}).values()
        safe = sum(item.policy_safe for item in prompts)
        failures = sum(item.policy_failures for item in prompts)
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CORRECTION_POLICY_CLASSIFIED",
                severity="INFO",
                message=(
                    f"{safe + failures} rollouts of cohort {cohort_id!r} were classified by a "
                    f"plan policy, not by the judge ({safe} counted safe, {failures} counted as "
                    "failures); the judge corrections of the rows that hold them credit the "
                    "judge with no false alarm at the upper bound and no missed failure at the "
                    "lower, and those rows have no asymptotic reference"
                ),
                scope=cohort_scope(cohort_id),
                details={
                    "groups": sorted(policy_rows[cohort_id]),
                    "policy_safe": safe,
                    "policy_failures": failures,
                },
            )
        )


@dataclass(frozen=True)
class JudgeCorrection:
    """The rows with their judge-corrected intervals, and the calibration they used."""

    rows: RiskRows
    # The calibration actually used, for the report's provenance; empty when there is none.
    summary: dict[str, Any]


def apply_judge_correction(
    plan: AnalysisPlan,
    data: NormalizedData,
    hierarchy_nodes: list[_HierarchyNode],
    rows: RiskRows,
    diagnostics: list[Diagnostic],
) -> JudgeCorrection:
    """The rows with judge-corrected intervals added. The rows given are left as they were."""

    corrected = rows.copy()
    summary = _add_judge_corrections(
        plan,
        data,
        hierarchy_nodes,
        corrected.risks,
        corrected.hierarchy_risks,
        corrected.comparisons,
        diagnostics,
    )
    return JudgeCorrection(rows=corrected, summary=summary)


def _add_judge_corrections(
    plan: AnalysisPlan,
    data: NormalizedData,
    hierarchy_nodes: list[_HierarchyNode],
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    comparisons: list[ComparisonResult],
    diagnostics: list[Diagnostic],
) -> dict[str, Any]:
    """Map judge-system risk rows onto the true-risk scale, in place: on copied rows only.

    Returns a summary of the calibration actually used, for the report provenance
    block. In ``assumed`` mode every scenario is applied to every cohort, and to both arms
    of every comparison. In
    ``validated`` mode one calibration is estimated per cohort from human labels. A
    calibration scoped to groups corrects only the rows of those groups, and validated mode
    then estimates one calibration per cohort and group. ``transported`` mode corrects risk
    rows and hierarchy nodes only, and names the comparisons it leaves uncorrected in
    ``JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON``.
    """

    spec = plan.inputs.judge_calibration
    if spec is None:
        return {}
    correctable = {
        measure.id
        for measure in plan.outputs.risk_measures
        if measure.type in JUDGE_CORRECTABLE_MEASURES
    }
    targets: list[GroupRiskResult | HierarchyRiskResult] = [
        *(row for row in risks if row.measure_id in correctable),
        *hierarchy_risks,
    ]
    population = _labelled_population(plan, data)
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    row_weights = _row_weights(data, hierarchy_nodes)

    def weighting(result: GroupRiskResult | HierarchyRiskResult) -> tuple[dict[str, float], str]:
        """The prompt weights behind a row, and its measure type."""
        # Hierarchy nodes are weighted means over their own prompts.
        measure_type = (
            measure_types[result.measure_id]
            if isinstance(result, GroupRiskResult)
            else "weighted_mean"
        )
        return row_weights(result), measure_type

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
                scope=plan_scope("inputs", "judge_calibration"),
            )
        )
        # Comparisons stay on the judge-reported scale, and the report says which.
        uncorrected = sorted({result.result_id or result.comparison_id for result in comparisons})
        if uncorrected:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON",
                    severity="WARNING",
                    message=(
                        "a transported calibration corrects no comparison: with a lower end of "
                        "zero on each arm, a corrected effect would only span minus the "
                        "baseline's corrected upper bound to the candidate's, so these effects "
                        "remain on the judge-reported scale"
                    ),
                    scope=scope_path("comparisons"),
                    details={"result_ids": uncorrected[:20], "count": len(uncorrected)},
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

    # Per cohort, the groups of corrected rows that hold rollouts a plan policy classified.
    policy_rows: dict[str, set[str]] = {}
    if spec.mode == "assumed":
        per_cohort = _per_cohort_scenarios(plan)
        for result in targets:
            if _judge_scope(plan, result.cohort_id, result.group_id) is None:
                continue
            weights, measure_type = weighting(result)
            cohort_population = population.get(result.cohort_id, {})
            share = _unclassified_share(cohort_population, weights, measure_type)
            by_policy = _policy_rollouts(cohort_population, weights)
            if any(by_policy):
                policy_rows.setdefault(result.cohort_id, set()).add(result.group_id)
            for calibration in _assumed_judge_calibrations(plan, result.cohort_id):
                interval = _judge_corrected_interval(
                    result=result,
                    calibration=calibration,
                    primary_scenario=spec.primary,
                    family_id=None,
                    judge_alpha=None,
                    unclassified_share=share,
                    scoped_groups=spec.groups,
                    per_cohort=calibration.scenario_id in per_cohort,
                    policy_classified=by_policy,
                )
                if interval is not None:
                    result.intervals.append(interval)
        # Every scope carries every declared scenario, each with its cohort's box, so each
        # arm of a comparison takes its own cohort's box: the same box on both arms unless
        # the scenario declares boxes by cohort.
        _apply_judge_correction_to_comparisons(
            plan,
            data,
            population,
            comparisons,
            {scope: _assumed_judge_calibrations(plan, scope[0]) for scope in _judge_scopes(plan)},
            None,
            diagnostics,
            primary_scenario=spec.primary,
            per_cohort=per_cohort,
            policy_rows=policy_rows,
        )
        _policy_classified_diagnostics(population, policy_rows, diagnostics)
        return {
            "mode": "assumed",
            "scenarios": sorted(
                (item.id for item in spec.scenarios), key=lambda item: (item != spec.primary, item)
            ),
            "primary": spec.primary,
            **({"groups": list(spec.groups)} if spec.groups is not None else {}),
        }

    by_scope, summaries, judge_alpha, truth_strata, protocol = _validated_judge_calibrations(
        plan, data, diagnostics
    )
    scope_count = len(_judge_scopes(plan))
    stratified = spec.sampling == "stratified_within_judge_stratum"
    for result in targets:
        scope = _judge_scope(plan, result.cohort_id, result.group_id)
        measured = by_scope.get(scope) if scope is not None else None
        if scope is None or measured is None:
            continue
        weights, measure_type = weighting(result)
        scope_population = _scope_population(population, data, scope)
        gap = pooled_calibration_gap(
            scope_population,
            design_type=plan.design.type,
            measure_type=measure_type,
            weights=weights,
            group_id=scope[1],
        )
        share = _unclassified_share(scope_population, weights, measure_type)
        by_policy = _policy_rollouts(scope_population, weights)
        if any(by_policy):
            policy_rows.setdefault(result.cohort_id, set()).add(result.group_id)
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
            policy_classified=by_policy,
        )
        if interval is not None:
            result.intervals.append(interval)
        # The reference propagates the weighted mean's variance, so only a weighted-mean
        # row has one: on a tail row it would describe a different estimand. Its stratum
        # proportions assume simple random samples within the judge strata, so a stratified
        # design has none. It would read policy-classified rollouts as judge verdicts, so a
        # row holding one has none either; JUDGE_CORRECTION_POLICY_CLASSIFIED says so.
        if (
            isinstance(result, GroupRiskResult)
            and measure_type == "weighted_mean"
            and judge_alpha is not None
            and not stratified
        ):
            reference = (
                None
                if any(by_policy)
                else _delta_reference_interval(
                    result=result,
                    data=data,
                    strata=summaries.get(scope, {}),
                    alpha=judge_alpha / scope_count,
                    family_id=spec.family,
                    gap=gap,
                    unclassified_share=share,
                    scoped_group=scope[1],
                )
            )
            if reference is not None:
                result.intervals.append(reference)
            # Lang-Reiczigel needs sensitivity and specificity, which only labels drawn
            # as a simple random sample estimate without bias.
            if spec.sampling == "simple_random":
                limits = _lang_reiczigel_interval(
                    result=result,
                    data=data,
                    truth=truth_strata.get(scope, {}),
                    alpha=judge_alpha / scope_count,
                    family_id=spec.family,
                    gap=gap,
                    unclassified_share=share,
                    scoped_group=scope[1],
                    policy_classified=by_policy,
                )
                if isinstance(limits, str):
                    diagnostics.append(
                        Diagnostic(
                            code="JUDGE_LANG_REICZIGEL_UNAVAILABLE",
                            severity="INFO",
                            message=(
                                f"no Lang-Reiczigel limits for group {result.group_id!r}: "
                                f"{limits}"
                            ),
                            scope=cohort_scope(result.cohort_id),
                            details={"group_id": result.group_id},
                        )
                    )
                else:
                    result.intervals.append(limits)
    _apply_judge_correction_to_comparisons(
        plan,
        data,
        population,
        comparisons,
        {scope: [calibration] for scope, calibration in by_scope.items()},
        spec.family,
        diagnostics,
        stratified=stratified,
        policy_rows=policy_rows,
    )
    _policy_classified_diagnostics(population, policy_rows, diagnostics)
    per_calibration_alpha = judge_alpha / scope_count if judge_alpha is not None else None
    summary: dict[str, Any]
    if spec.groups is None:
        summary = {
            "mode": "validated",
            "family": spec.family,
            "family_alpha": judge_alpha,
            "per_cohort_alpha": per_calibration_alpha,
            "sampling": spec.sampling,
            # As the report names every input: relative to the plan, never by its absolute path.
            "validation_manifest": None
            if spec.validation_manifest is None
            else report_path(spec.validation_manifest, plan),
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
            # As the report names every input: relative to the plan, never by its absolute path.
            "validation_manifest": None
            if spec.validation_manifest is None
            else report_path(spec.validation_manifest, plan),
            "cohorts": nested(summaries),
            "truth_strata": nested(truth_strata),
        }
    if protocol is not None:
        # Only a plan that declares a protocol gets the key, so other reports keep their bytes.
        summary["protocol"] = protocol
    if stratified:
        # Only a stratified design records it, so other reports keep their bytes.
        summary["experimental"] = True
        if spec.validation_strata is not None:
            summary["validation_strata"] = report_path(spec.validation_strata, plan)
    if spec.groups is not None:
        # A scope to groups is experimental too, and only a scoped plan records it.
        summary["experimental"] = True
    return summary
