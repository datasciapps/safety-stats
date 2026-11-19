"""Claim decisions over risks and comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from inspect_safety_stats.analysis.judge_calibration import (
    JUDGE_CORRECTABLE_MEASURES,
    PromptRollouts,
    _judge_scope,
    _judge_scope_excludes,
    _JudgeScope,
    _labelled_population,
    _out_of_scope_reason,
    _scope_population,
    _scope_rollouts,
    _unclassified_share,
    pooled_calibration_gap,
)
from inspect_safety_stats.analysis.judge_panel import _DeclaredBox, _primary_accuracy_box
from inspect_safety_stats.config.models import AnalysisPlan, ClaimSpec
from inspect_safety_stats.decision.engine import (
    claim_verdict_range,
    decide_claim,
    noisy_valid_verdict_range,
)
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    ClaimProfileVerdict,
    ClaimResult,
    ComparisonResult,
    Decision,
    Diagnostic,
    GroupRiskResult,
    JudgeErrorStressResult,
    RiskIntervalResult,
    VerdictRange,
)
from inspect_safety_stats.schema.scopes import plan_scope
from inspect_safety_stats.stats.judge_stress import stress_verdict
from inspect_safety_stats.stats.misclassification import (
    TruthStratumCounts,
    noisy_valid_certification,
)


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
    scope: _JudgeScope | None = None,
) -> tuple[Decision, list[str], float | None, float | None, str, float, VerdictRange | None]:
    """Decide an aggregate claim with the finite-sample noisy-but-valid test.

    ``scope`` is the calibration the claim's row is corrected with; None when the
    calibration is scoped to groups that leave the row out.

    The verdict range is None when the test could not run at the plan's thresholds, which
    leaves the thresholds at which it would decide unknown.
    """

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
            VerdictRange(),
        )
    if scope is None:
        return (
            "INCONCLUSIVE",
            [_out_of_scope_reason(evidence.group_id)],
            None,
            None,
            method,
            family_alpha,
            VerdictRange(),
        )
    if gap is not None:
        # The truth strata pool the cohort's labels exactly as the judge strata do.
        return (
            "INCONCLUSIVE",
            [
                f"The noisy-but-valid test cannot certify this row: {gap}. Its labels are a "
                f"simple random sample of {_scope_rollouts(scope[1], 'classified rollouts')}, "
                "so they measure "
                "the judge's sensitivity and false-positive rate only for a weighted mean "
                "that weights those rollouts equally, over the whole "
                + ("cohort" if scope[1] is None else "group")
                + " in a fixed design."
            ],
            None,
            None,
            method,
            family_alpha,
            VerdictRange(),
        )
    summary = judge_summary or {}
    strata = (summary.get("truth_strata") or {}).get(evidence.cohort_id) or {}
    if scope[1] is not None:
        strata = strata.get(scope[1]) or {}
    judge_alpha = summary.get("family_alpha")
    per_cohort_alpha = summary.get(
        "per_cohort_alpha" if scope[1] is None else "per_calibration_alpha"
    )
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
            VerdictRange(),
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
        return "INCONCLUSIVE", [str(exc)], None, None, method, family_alpha, None
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
        noisy_valid_verdict_range(result),
    )


def verdicts_by_profile(
    claim: Any,
    evidence: ComparisonResult | GroupRiskResult | None,
    *,
    status: Decision,
    lower: float | None,
    upper: float | None,
    confirmatory_and_valid: bool,
) -> list[ClaimProfileVerdict]:
    """Return the claim's verdict under its primary and each sensitivity profile.

    The primary row repeats the claim's decision. Each sensitivity row decides the same
    thresholds on that profile's interval computed with the primary interval's method.
    Evidence whose primary interval assumes no independence profile, such as the propagated
    prompt envelope, has no rows.
    """
    if evidence is None:
        return []
    primary = next(
        (item for item in evidence.intervals if item.interval_id == evidence.primary_interval_id),
        None,
    )
    if primary is None or primary.independence_profile is None:
        return []
    rows = [
        ClaimProfileVerdict(
            independence_profile=primary.independence_profile,
            role="primary",
            status=status,
            interval_id=primary.interval_id,
            lower=lower,
            upper=upper,
            independent_block_count=primary.independent_block_count,
            effective_sample_size=primary.effective_sample_size,
        )
    ]
    for interval in evidence.intervals:
        if (
            interval.role != "sensitivity"
            or interval.method != primary.method
            or interval.independence_profile is None
            or interval.independence_profile == primary.independence_profile
        ):
            continue
        sensitivity_status, _ = decide_claim(
            lower=interval.lower,
            upper=interval.upper,
            acceptable_upper=claim.acceptable_upper,
            unacceptable_lower=claim.unacceptable_lower,
            eligible=confirmatory_and_valid and interval.eligibility == "certifying",
        )
        rows.append(
            ClaimProfileVerdict(
                independence_profile=interval.independence_profile,
                role="sensitivity",
                status=sensitivity_status,
                interval_id=interval.interval_id,
                lower=interval.lower,
                upper=interval.upper,
                independent_block_count=interval.independent_block_count,
                effective_sample_size=interval.effective_sample_size,
            )
        )
    return rows


def _has_sensitivity_profiles(evidence: ComparisonResult | GroupRiskResult | None) -> bool:
    """Whether the evidence has intervals under a profile other than its primary one."""
    if evidence is None:
        return False
    primary_profile = next(
        (
            item.independence_profile
            for item in evidence.intervals
            if item.interval_id == evidence.primary_interval_id
        ),
        None,
    )
    return any(
        item.role == "sensitivity"
        and item.independence_profile is not None
        and item.independence_profile != primary_profile
        for item in evidence.intervals
    )


def _claim_judge_error_stress(
    *,
    claim: ClaimSpec,
    evidence: ComparisonResult | GroupRiskResult | None,
    status: Decision,
    why_not: str | None,
    measure_types: Mapping[str, str],
    population: dict[str, dict[str, PromptRollouts]],
    group_weights: dict[str, dict[str, float]],
    accuracy: _DeclaredBox | None,
) -> JudgeErrorStressResult:
    """Where a claim decided on judge-reported bounds tips, and whether a declared box holds it.

    Diagnostic only: the result is reported beside the claim and never changes its status. A
    box declared for groups that leave the claim's group out is not applied to it.
    """

    def skipped(reason: str) -> JudgeErrorStressResult:
        return JudgeErrorStressResult(applicable=False, reason=reason)

    if claim.comparison:
        return skipped(
            "comparison claims are not stressed: the judge's accuracy may differ between arms"
        )
    if (
        not isinstance(evidence, GroupRiskResult)
        or evidence.lower is None
        or evidence.upper is None
    ):
        return skipped("the claim has no complete bounds")
    if why_not is not None:
        return skipped(why_not)
    measure_type = measure_types[evidence.measure_id]
    if measure_type not in JUDGE_CORRECTABLE_MEASURES:
        return skipped(
            f"a {measure_type} measure does not carry the judge's per-prompt error model over "
            "to the row"
        )
    share = _unclassified_share(
        population.get(evidence.cohort_id, {}), group_weights[evidence.group_id], measure_type
    )
    if accuracy is not None and not accuracy.holds_on(evidence.group_id):
        accuracy = None
    box, source = (accuracy.box, accuracy.source) if accuracy is not None else (None, None)
    stress = stress_verdict(
        judge_reported_lower=evidence.lower,
        judge_reported_upper=evidence.upper,
        acceptable_upper=claim.acceptable_upper,
        unacceptable_lower=claim.unacceptable_lower,
        box=box,
        unclassified_share=share,
    )
    stressed = stress.stressed
    return JudgeErrorStressResult(
        applicable=True,
        judge_reported_lower=evidence.lower,
        judge_reported_upper=evidence.upper,
        unclassified_share=share,
        pass_sensitivity_at_perfect_specificity=(
            stress.pass_line.sensitivity_at_perfect_specificity
        ),
        pass_specificity_at_perfect_sensitivity=(
            stress.pass_line.specificity_at_perfect_sensitivity
        ),
        fail_sensitivity_at_perfect_specificity=(
            stress.fail_line.sensitivity_at_perfect_specificity
        ),
        fail_specificity_at_perfect_sensitivity=(
            stress.fail_line.specificity_at_perfect_sensitivity
        ),
        accuracy_source=source,
        sensitivity_lower=box.sensitivity_lower if box else None,
        sensitivity_upper=box.sensitivity_upper if box else None,
        specificity_lower=box.specificity_lower if box else None,
        specificity_upper=box.specificity_upper if box else None,
        stressed_status=stressed.verdict if stressed else None,
        true_risk_lower=stressed.true_risk_lower if stressed else None,
        true_risk_upper=stressed.true_risk_upper if stressed else None,
        identified=stressed.identified if stressed else None,
        robust=stressed.verdict == status if stressed else None,
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
    stress_population = (
        population or _labelled_population(plan, data) if plan.report.judge_error_stress else {}
    )
    stress_box = _primary_accuracy_box(plan) if plan.report.judge_error_stress else None
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
        certifying = evidence is not None and evidence.eligibility == "certifying"
        exploratory = plan.analysis.intent == "exploratory"
        # An exploratory analysis certifies nothing, but its reader still needs what the bounds
        # say at the plan's thresholds. Such a claim is decided as a confirmatory analysis would
        # decide it, the result is kept as its descriptive status, and the claim stays
        # INCONCLUSIVE below.
        eligible = bool(integrity_valid and certifying)
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
                    verdict_range=VerdictRange(),
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
        # Filled only where the claim is decided directly on the evidence intervals: a
        # judge-corrected or noisy-valid decision has no per-profile counterpart.
        profile_verdicts: list[ClaimProfileVerdict] = []

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
            scope = (
                _judge_scope(plan, evidence.cohort_id, evidence.group_id)
                if isinstance(evidence, GroupRiskResult)
                else None
            )
            gap = (
                pooled_calibration_gap(
                    _scope_population(population, data, scope),
                    design_type=plan.design.type,
                    measure_type=measure_types[evidence.measure_id],
                    weights=group_weights[evidence.group_id],
                    group_id=scope[1],
                )
                if isinstance(evidence, GroupRiskResult) and scope is not None
                else None
            )
            status, reasons, lower, upper, method, alpha, verdict_range = _noisy_valid_claim(
                claim=claim,
                evidence=evidence,
                eligible=eligible,
                family_alpha=family.alpha,
                judge_summary=judge_summary,
                gap=gap,
                scope=scope,
            )
            estimate = None
            correction_note = None
            evidence_refs.append("test:noisy_valid_certification")
        elif validated_judge and correction is None:
            # A validated calibration makes every claim a claim about true risk. Without a
            # certifying correction only judge-reported risk remains, and the labels may show
            # the judge missing failures: a refused calibration used to pass claims that way.
            status = "INCONCLUSIVE"
            verdict_range = VerdictRange()
            # A correction the labels cannot certify for this estimand says why itself.
            why = [
                warning
                for interval in (evidence.intervals if evidence is not None else [])
                if interval.interval_id == "judge_corrected_validated"
                for warning in interval.warnings
            ]
            if evidence is not None and _judge_scope_excludes(plan, evidence.group_id):
                why.append(_out_of_scope_reason(evidence.group_id))
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
            profile_verdicts = verdicts_by_profile(
                claim,
                evidence,
                status=status,
                lower=lower,
                upper=upper,
                confirmatory_and_valid=bool(
                    integrity_valid and plan.analysis.intent == "confirmatory"
                ),
            )
            verdict_range = claim_verdict_range(lower=lower, upper=upper, eligible=eligible)
        if not integrity_valid and lower is not None and upper is not None:
            reasons = [
                "Bounds were computed but are ineligible for certification because data "
                "integrity is invalid."
            ]
        descriptive_status: Decision | None = None
        if exploratory:
            if certifying:
                decided = reasons
                reasons = [
                    "Bounds were computed but this post hoc exploratory analysis is not "
                    "eligible for certification."
                ]
                if eligible:
                    descriptive_status = status
                    reasons.append(
                        f"Descriptive only, not a certification: a confirmatory analysis of "
                        f"these bounds at the plan's thresholds would give {status}."
                    )
                    reasons.extend(decided)
            else:
                reasons.append("Exploratory analyses cannot produce a safety pass.")
            status = "INCONCLUSIVE"
        # ORDER MATTERS: after the exploratory demotion above, so `status` is the final one.
        stress = None
        if plan.report.judge_error_stress:
            stress = _claim_judge_error_stress(
                claim=claim,
                evidence=evidence,
                status=status,
                why_not=(
                    "the analysis is exploratory, so the claim is not decided on its bounds"
                    if exploratory
                    else "the claim was not decided on its bounds: they are not certifying, or "
                    "data integrity is invalid"
                    if not eligible
                    else "the noisy-but-valid test already accounts for judge error"
                    if claim.method == "noisy_valid_test"
                    else "the claim is decided on a validated judge correction, which already "
                    "accounts for judge error"
                    if correction is not None
                    else "the plan's validated calibration leaves the claim INCONCLUSIVE rather "
                    "than deciding it on judge-reported risk"
                    if validated_judge
                    else None
                ),
                measure_types=measure_types,
                population=stress_population,
                group_weights=group_weights,
                accuracy=stress_box,
            )
        if correction_note is not None:
            reasons.insert(0, correction_note)
            if alpha != family.alpha:
                reasons.append(
                    f"The error budget is the union of the claim family ({family.alpha:.12g}) "
                    f"and the judge calibration, giving {alpha:.12g}."
                )
        if not profile_verdicts and _has_sensitivity_profiles(evidence):
            reasons.append(
                "Verdicts under the sensitivity independence profiles are not reported, "
                "because this claim is not decided directly on the profile intervals."
            )
        results.append(
            ClaimResult(
                claim_id=claim.id,
                title=claim.title,
                status=status,
                statistical_status=status,
                policy_status=status,
                descriptive_status=descriptive_status,
                estimate=estimate,
                lower=lower,
                upper=upper,
                acceptable_upper=claim.acceptable_upper,
                unacceptable_lower=claim.unacceptable_lower,
                verdict_range=verdict_range,
                alpha=alpha,
                multiplicity_family=claim.family,
                method=method,
                mandatory=claim.mandatory,
                evidence_refs=evidence_refs,
                reasons=reasons,
                warnings=list(evidence.warnings) if evidence else [],
                profile_verdicts=profile_verdicts,
                judge_error_stress=stress,
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
                scope=plan_scope("claims"),
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
                scope=plan_scope("claims"),
                details={"claim_ids": sorted(uncorrected_claims)},
            )
        )
    return results
