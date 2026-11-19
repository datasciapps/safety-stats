"""Judge correction of comparison intervals."""

from __future__ import annotations

from typing import cast

from inspect_safety_stats.analysis.comparison_intervals import _comparison_targets
from inspect_safety_stats.analysis.comparison_regions import _multi_arm_effect_interval
from inspect_safety_stats.analysis.judge_calibration import (
    PromptRollouts,
    _judge_scope,
    _JudgeScope,
    _scope_population,
    _uncertified_base_warning,
    _uncertified_correction_warning,
    _unclassified_assumption,
    _unclassified_share,
    pooled_calibration_gap,
)
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import ComparisonResult, Diagnostic, RiskIntervalResult
from inspect_safety_stats.schema.scopes import scope_path
from inspect_safety_stats.stats.misclassification import JudgeCalibration, correct_judge_risk


def _judge_corrected_comparison_interval(
    *,
    result: ComparisonResult,
    candidate_calibration: JudgeCalibration,
    baseline_calibration: JudgeCalibration,
    family_id: str | None,
    gap: str | None = None,
    candidate_share: float = 0.0,
    baseline_share: float = 0.0,
    tail: bool = False,
    scoped_group: str | None = None,
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
        unclassified_share=candidate_share,
        tail=tail,
    )
    baseline = correct_judge_risk(
        observed_lower=cast(float, primary.baseline_lower),
        observed_upper=cast(float, primary.baseline_upper),
        observed_estimate=result.baseline_estimate,
        calibration=baseline_calibration,
        unclassified_share=baseline_share,
        tail=tail,
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
    certified = gap is None and result.eligibility == "certifying"
    warnings: list[str] = []
    if gap is not None:
        warnings.append(_uncertified_correction_warning(gap, scoped_group))
    elif not certified:
        warnings.append(_uncertified_base_warning(result.eligibility))
    return RiskIntervalResult(
        interval_id="judge_corrected_validated",
        lower=lower,
        upper=upper,
        method="validated_judge_misclassification_correction",
        role="reference" if certified else "sensitivity",
        varied_assumption=None if certified else "judge_error",
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
            (
                "Each arm is corrected with its own cohort's calibration, because the "
                "conditional judge accuracies are prevalence-dependent."
                if scoped_group is None
                else f"Each arm is corrected with its own cohort's calibration for group "
                f"{scoped_group!r}, from the labels on that group's rollouts, because the "
                "conditional judge accuracies are prevalence-dependent."
            ),
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
                    (
                        "Each cohort's labels weight its classified rollouts equally"
                        if scoped_group is None
                        else f"Each cohort's labels weight the classified rollouts of group "
                        f"{scoped_group!r} equally"
                    )
                    + ", and so does "
                    "the comparison, so they measure the judge's accuracy for both arms.",
                    "Coverage is the union bound over that region and both arm calibrations.",
                ]
                if certified
                else []
            ),
            *(
                f"{arm.capitalize()} arm: {_unclassified_assumption(share)}"
                for arm, share in (("candidate", candidate_share), ("baseline", baseline_share))
                if share > 0
            ),
        ],
        eligibility="certifying" if certified else "diagnostic",
        warnings=warnings,
    )


def _apply_judge_correction_to_comparisons(
    plan: AnalysisPlan,
    data: NormalizedData,
    population: dict[str, dict[str, PromptRollouts]],
    comparisons: list[ComparisonResult],
    by_scope: dict[_JudgeScope, JudgeCalibration],
    family_id: str | None,
    diagnostics: list[Diagnostic],
) -> None:
    """Append validated corrections to comparisons that expose both arm bounds.

    A comparison of a group the calibration is not scoped to stays uncorrected, silently:
    the plan scoped the calibration away from it.
    """

    if not by_scope:
        return
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
        candidate_scope = _judge_scope(plan, result.candidate_cohort, result.group_id)
        baseline_scope = _judge_scope(plan, result.baseline_cohort, result.group_id)
        if candidate_scope is None or baseline_scope is None:
            continue
        candidate = by_scope.get(candidate_scope)
        baseline = by_scope.get(baseline_scope)
        if candidate is None or baseline is None:
            skipped.append(result.result_id or result.comparison_id)
            continue
        weights = target_weights[(result.comparison_id, result.result_id or result.comparison_id)]
        # Each arm is corrected with its own cohort's labels, so each must be identified.
        gaps: list[str] = []
        for arm, scope in (("candidate", candidate_scope), ("baseline", baseline_scope)):
            gap = pooled_calibration_gap(
                _scope_population(population, data, scope),
                design_type=plan.design.type,
                measure_type=measure_types[result.measure_id],
                weights=weights,
                group_id=scope[1],
            )
            if gap is not None:
                gaps.append(f"{arm} cohort {scope[0]!r}: {gap}")
        measure_type = measure_types[result.measure_id]
        interval = _judge_corrected_comparison_interval(
            result=result,
            candidate_calibration=candidate,
            baseline_calibration=baseline,
            family_id=family_id,
            gap="; ".join(gaps) or None,
            candidate_share=_unclassified_share(
                _scope_population(population, data, candidate_scope), weights, measure_type
            ),
            baseline_share=_unclassified_share(
                _scope_population(population, data, baseline_scope), weights, measure_type
            ),
            tail=measure_type != "weighted_mean",
            scoped_group=candidate_scope[1],
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
                scope=scope_path("comparisons"),
                details={"result_ids": sorted(skipped)[:20], "count": len(skipped)},
            )
        )
