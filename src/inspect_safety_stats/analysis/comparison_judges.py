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
    primary_scenario: str | None = None,
    scoped_groups: list[str] | None = None,
    per_cohort: bool = False,
) -> RiskIntervalResult | None:
    """Correct a comparison by correcting each arm, then recombining.

    A validated calibration gives each arm its own cohort's measured box. An assumed
    scenario gives both arms the same declared box, or with ``per_cohort`` each arm the box
    it declares for that arm's cohort; each arm is free within its box independently of the
    other, and the result is a sensitivity analysis like every assumed correction.

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
    if candidate_calibration.source != "validated":
        return _assumed_comparison_interval(
            result=result,
            primary=primary,
            calibrations=(candidate_calibration, baseline_calibration),
            bounds=(lower, upper),
            arms=(candidate.lower, candidate.upper, baseline.lower, baseline.upper),
            candidate_share=candidate_share,
            baseline_share=baseline_share,
            primary_scenario=primary_scenario,
            scoped_groups=scoped_groups,
            per_cohort=per_cohort,
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


def _assumed_comparison_interval(
    *,
    result: ComparisonResult,
    primary: RiskIntervalResult,
    calibrations: tuple[JudgeCalibration, JudgeCalibration],
    bounds: tuple[float | None, float | None],
    arms: tuple[float, float, float, float],
    candidate_share: float,
    baseline_share: float,
    primary_scenario: str | None,
    scoped_groups: list[str] | None,
    per_cohort: bool,
) -> RiskIntervalResult:
    """One declared scenario's correction of a comparison, applied to both arms.

    ``calibrations`` are the candidate's and the baseline's boxes, which differ only when
    the scenario declares boxes by cohort (``per_cohort``)."""

    calibration, baseline_calibration = calibrations
    scope = (
        ""
        if scoped_groups is None
        else ", on the rows of groups " + ", ".join(repr(item) for item in scoped_groups)
        + " only"
    )
    boxes = (
        [
            f"{arm.capitalize()} P(truly unsafe | judge said unsafe) lies in "
            f"[{box.rho_1_lower:.6g}, {box.rho_1_upper:.6g}]; P(truly unsafe | judge said "
            f"safe) in [{box.rho_0_lower:.6g}, {box.rho_0_upper:.6g}]."
            for arm, box in (("candidate", calibration), ("baseline", baseline_calibration))
        ]
        if per_cohort
        else [
            f"On both arms P(truly unsafe | judge said unsafe) lies in "
            f"[{calibration.rho_1_lower:.6g}, {calibration.rho_1_upper:.6g}] and "
            f"P(truly unsafe | judge said safe) in "
            f"[{calibration.rho_0_lower:.6g}, {calibration.rho_0_upper:.6g}]."
        ]
    )
    warnings: list[str] = []
    if primary_scenario is not None and calibration.scenario_id != primary_scenario:
        warnings.append("Non-primary judge-calibration scenario; reported for sensitivity only.")
    return RiskIntervalResult(
        interval_id=f"judge_corrected_{calibration.scenario_id}",
        lower=bounds[0],
        upper=bounds[1],
        method="assumed_judge_misclassification_correction",
        role="sensitivity",
        varied_assumption="judge_error",
        coverage_scope=(
            f"{primary.coverage_scope}_judge_corrected"
            if primary.coverage_scope
            else "judge_corrected"
        ),
        raw_lower=primary.lower,
        raw_upper=primary.upper,
        candidate_lower=arms[0],
        candidate_upper=arms[1],
        baseline_lower=arms[2],
        baseline_upper=arms[3],
        calibration_method="assumed_judge_calibration",
        independence_profile=primary.independence_profile,
        assumptions=[
            "The reported effect is a contrast of true failure risks, not of "
            "judge-reported risks.",
            *boxes,
            "These conditional accuracies are declared assumptions, not measurements; "
            "the interval inherits no coverage guarantee from them.",
            (
                "The declared accuracies are prevalence-dependent; the scenario declares "
                f"each arm's box for that arm's cohort{scope}."
                if per_cohort
                else "The declared accuracies are prevalence-dependent and are applied "
                f"unchanged to both arms' cohorts{scope}."
            ),
            (
                "Each arm is corrected over its own declared box independently of the other, "
                "so a judge whose accuracy on each arm stays within that arm's box is covered; "
                "one whose accuracy leaves its box on either arm is not."
                if per_cohort
                else "Each arm is corrected over the whole declared box independently of the "
                "other, so a judge whose accuracy differs between the arms within that box is "
                "covered; one whose accuracy leaves the box on either arm is not."
            ),
            "The effect is recombined from the two corrected arm intervals, so it makes "
            "no independence assumption between arms beyond the one the underlying "
            "two-arm region already makes.",
            *(
                f"{arm.capitalize()} arm: {_unclassified_assumption(share)}"
                for arm, share in (("candidate", candidate_share), ("baseline", baseline_share))
                if share > 0
            ),
            "The underlying judge-system interval retains its own stated guarantee.",
        ],
        eligibility="diagnostic",
        warnings=warnings,
    )


def _apply_judge_correction_to_comparisons(
    plan: AnalysisPlan,
    data: NormalizedData,
    population: dict[str, dict[str, PromptRollouts]],
    comparisons: list[ComparisonResult],
    by_scope: dict[_JudgeScope, list[JudgeCalibration]],
    family_id: str | None,
    diagnostics: list[Diagnostic],
    primary_scenario: str | None = None,
    per_cohort: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Append judge corrections to comparisons that expose both arm bounds.

    ``by_scope`` gives each calibration scope its calibrations: one measured box in validated
    mode, every declared scenario in assumed mode, with the box it declares for the scope's
    cohort. A comparison gets one corrected interval per scenario both its arms carry, each
    arm corrected with its own scope's box. ``per_cohort`` names the assumed scenarios that
    declare boxes by cohort.

    A comparison of a group the calibration is not scoped to stays uncorrected, silently:
    the plan scoped the calibration away from it. Any other comparison left uncorrected is
    reported in ``JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON``.
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
    spec = plan.inputs.judge_calibration
    scoped_groups = spec.groups if spec is not None else None
    skipped: list[str] = []
    for result in comparisons:
        candidate_scope = _judge_scope(plan, result.candidate_cohort, result.group_id)
        baseline_scope = _judge_scope(plan, result.baseline_cohort, result.group_id)
        if candidate_scope is None or baseline_scope is None:
            continue
        candidates = by_scope.get(candidate_scope) or []
        baselines = {item.scenario_id: item for item in by_scope.get(baseline_scope) or []}
        pairs = [
            (item, baselines[item.scenario_id])
            for item in candidates
            if item.scenario_id in baselines
        ]
        if not pairs:
            skipped.append(result.result_id or result.comparison_id)
            continue
        weights = target_weights[(result.comparison_id, result.result_id or result.comparison_id)]
        measure_type = measure_types[result.measure_id]
        # Each arm is corrected with its own cohort's labels, so each must be identified.
        # Declared accuracies are not measured from labels, so they have no such gap.
        measured = all(item.source == "validated" for item, _ in pairs)
        gaps: list[str] = []
        for arm, scope in (("candidate", candidate_scope), ("baseline", baseline_scope)):
            if not measured:
                continue
            gap = pooled_calibration_gap(
                _scope_population(population, data, scope),
                design_type=plan.design.type,
                measure_type=measure_type,
                weights=weights,
                group_id=scope[1],
            )
            if gap is not None:
                gaps.append(f"{arm} cohort {scope[0]!r}: {gap}")
        candidate_share = _unclassified_share(
            _scope_population(population, data, candidate_scope), weights, measure_type
        )
        baseline_share = _unclassified_share(
            _scope_population(population, data, baseline_scope), weights, measure_type
        )
        corrected = False
        for candidate, baseline in pairs:
            interval = _judge_corrected_comparison_interval(
                result=result,
                candidate_calibration=candidate,
                baseline_calibration=baseline,
                family_id=family_id,
                gap="; ".join(gaps) or None,
                candidate_share=candidate_share,
                baseline_share=baseline_share,
                tail=measure_type != "weighted_mean",
                scoped_group=candidate_scope[1],
                primary_scenario=primary_scenario,
                scoped_groups=scoped_groups,
                per_cohort=candidate.scenario_id in per_cohort,
            )
            if interval is not None:
                result.intervals.append(interval)
                corrected = True
        if not corrected:
            skipped.append(result.result_id or result.comparison_id)
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
