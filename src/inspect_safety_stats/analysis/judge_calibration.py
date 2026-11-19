"""Judge calibration, and judge-corrected and transported intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inspect_safety_stats.analysis.estimates import _bounds_carry_unknown_outcome
from inspect_safety_stats.config.models import (
    AnalysisPlan,
    JudgeCalibrationSpec,
    PublishedValidationSetSpec,
)
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.ingest.validation import (
    check_labelled_prefix,
    fine_stratum_counts,
    judge_stratum_counts,
    load_validation_manifest,
    load_validation_strata,
    truth_stratum_counts,
)
from inspect_safety_stats.schema.reports import (
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
)
from inspect_safety_stats.schema.scopes import cohort_scope, plan_scope
from inspect_safety_stats.stats.judge_calibration import (
    NO_STRATA,
    FineStrata,
    JudgeStrata,
    calibrate_cohort,
    cohort_alpha_share,
    completion_diagnostic,
    fine_strata_summary,
    unlabelled_fine_strata_diagnostic,
)
from inspect_safety_stats.stats.misclassification import (
    CalibrationBound,
    JudgeCalibration,
    correct_judge_risk,
    transport_judge_risk,
)

# Risk measures whose value is positively homogeneous and translation equivariant,
# so the affine judge correction R* = R * rho_1 + (1 - R) * rho_0 applies unchanged.
# A validated calibration certifies only the weighted mean: see pooled_calibration_gap.
JUDGE_CORRECTABLE_MEASURES = frozenset(
    {"weighted_mean", "maximum", "cvar", "top_k_mean"}
)


@dataclass(frozen=True)
class PromptRollouts:
    """One prompt's rollouts in one cohort, counted the ways its rates divide by them."""

    classified: int
    # Prompt bounds divide by the classified rollouts plus the unknown outcomes the missing
    # and error policies keep. Direct-mean intervals divide by every rollout, and complete
    # each unclassified one, so the two counts differ only where a policy excludes some.
    bounded: int
    attempted: int


# A judge calibration's key: its cohort, and its group when the calibration is scoped to groups.
_JudgeScope = tuple[str, str | None]


def _judge_scope(plan: AnalysisPlan, cohort_id: str, group_id: str) -> _JudgeScope | None:
    """The calibration a row of this cohort and group is corrected with, or None if none is."""

    spec = plan.inputs.judge_calibration
    if spec is None or spec.groups is None:
        return (cohort_id, None)
    return None if _judge_scope_excludes(plan, group_id) else (cohort_id, group_id)


def _judge_scope_excludes(plan: AnalysisPlan, group_id: str) -> bool:
    """Whether the judge calibration is scoped to groups that leave this one out."""

    spec = plan.inputs.judge_calibration
    return spec is not None and spec.groups is not None and group_id not in spec.groups


def _judge_scopes(plan: AnalysisPlan) -> list[_JudgeScope]:
    spec = plan.inputs.judge_calibration
    groups: list[str | None] = [None] if spec is None or spec.groups is None else list(spec.groups)
    return [(cohort_id, group_id) for cohort_id in sorted(plan.cohorts) for group_id in groups]


def _scope_population(
    population: dict[str, dict[str, PromptRollouts]],
    data: NormalizedData,
    scope: _JudgeScope,
) -> dict[str, PromptRollouts]:
    """The labelled population of one calibration: its cohort's, cut to its group's prompts."""

    cohort_id, group_id = scope
    rollouts = population.get(cohort_id, {})
    if group_id is None:
        return rollouts
    prompts = set(next(group for group in data.groups if group.id == group_id).prompt_ids)
    return {prompt_id: item for prompt_id, item in rollouts.items() if prompt_id in prompts}


def _out_of_scope_reason(group_id: str) -> str:
    return (
        "The judge calibration is scoped to groups that do not include this row's group "
        f"{group_id!r}, so no labels calibrate it."
    )


def _scope_rollouts(group_id: str | None, noun: str = "rollouts") -> str:
    """Name the rollouts a calibration's labels are drawn from, as the wording of a sentence."""

    if group_id is None:
        return f"the cohort's {noun}"
    return f"the {noun} of group {group_id!r} in the cohort"


def _labelled_population(
    plan: AnalysisPlan, data: NormalizedData
) -> dict[str, dict[str, PromptRollouts]]:
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
        cohort_id: {prompt_id: PromptRollouts(*tally) for prompt_id, tally in prompts.items()}
        for cohort_id, prompts in counts.items()
    }


def pooled_calibration_gap(
    population: dict[str, PromptRollouts],
    *,
    design_type: str,
    measure_type: str,
    weights: dict[str, float],
    group_id: str | None = None,
) -> str | None:
    """Why a cohort's pooled labels do not measure the judge accuracies a row needs.

    With ``group_id`` the labels, and ``population``, are those of one group in the cohort.

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
        if group_id is not None:
            return f"group {group_id!r} has no classified rollouts in the cohort"
        return "the cohort has no classified rollouts"
    if any(weights.get(prompt_id, 0.0) <= 0.0 for prompt_id, _ in labelled):
        return "the row covers only part of " + _scope_rollouts(group_id, "labelled rollouts")
    # A rollout weighs its prompt's weight over the rollouts the prompt's rate divides by.
    # Either kind of interval can be the row's primary, so both counts must agree.
    for per_rollout in (
        [weights[prompt_id] / item.bounded for prompt_id, item in labelled],
        [weights[prompt_id] / item.attempted for prompt_id, item in labelled],
    ):
        lightest, heaviest = min(per_rollout), max(per_rollout)
        if heaviest > lightest * (1 + 1e-9):
            return (
                f"the row gives some of {_scope_rollouts(group_id)} "
                f"{heaviest / lightest:.3g} times the weight of others"
            )
    return None


def _unclassified_share(
    population: dict[str, PromptRollouts], weights: dict[str, float], measure_type: str
) -> float:
    """The weight a row's bounds put on rollouts they complete as unknown outcomes.

    A tail measure can rest on a single prompt, so for one this is the largest share in
    any prompt it weights. Each prompt takes the larger of its two counts' shares: they
    differ only under ``exclude``, which direct-mean intervals complete and prompt bounds
    drop, and the larger share is conservative for either.
    """

    shares: list[tuple[float, float]] = []
    for prompt_id, weight in weights.items():
        item = population.get(prompt_id)
        if weight <= 0 or item is None or item.attempted == 0:
            continue
        share = (item.attempted - item.classified) / item.attempted
        if item.bounded:
            share = max(share, (item.bounded - item.classified) / item.bounded)
        shares.append((weight, share))
    if not shares:
        return 0.0
    if measure_type == "weighted_mean":
        return min(1.0, sum(weight * share for weight, share in shares))
    return max(share for _, share in shares)


def _unclassified_assumption(share: float) -> str:
    return (
        f"{share:.6g} of the row's weight rests on unclassified rollouts. They keep their "
        "completion on the true-risk scale, truly safe at the lower bound and truly unsafe at "
        "the upper, and only the classified mass is mapped through rho."
    )


def _uncertified_base_warning(eligibility: str) -> str:
    return (
        f"Sensitivity analysis, not a certified bound: the judge-system interval it corrects "
        f"is {eligibility}, and a correction cannot support more than its base."
    )


def _uncertified_correction_warning(gap: str, group_id: str | None = None) -> str:
    if group_id is not None:
        return (
            f"Sensitivity analysis, not a certified bound: {gap}. The human labels weight every "
            f"classified rollout of group {group_id!r} in the cohort equally, so they measure "
            "the judge's accuracy only for a weighted mean that does the same, over the whole "
            "group in a fixed design."
        )
    return (
        f"Sensitivity analysis, not a certified bound: {gap}. The human labels weight every "
        "classified rollout of the cohort equally, so they measure the judge's accuracy only "
        "for a weighted mean that does the same, over the whole cohort in a fixed design."
    )


def _assumed_judge_calibrations(plan: AnalysisPlan, cohort_id: str) -> list[JudgeCalibration]:
    """Every declared scenario's box for one cohort: its own box where the scenario gives
    the cohort one, the scenario's default box otherwise."""

    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "assumed":
        return []
    ordered = sorted(spec.scenarios, key=lambda item: (item.id != spec.primary, item.id))
    calibrations = []
    for item in ordered:
        rho_1, rho_0 = item.box(cohort_id)
        calibrations.append(
            JudgeCalibration(
                scenario_id=item.id,
                rho_1_lower=rho_1.lower,
                rho_1_upper=rho_1.upper,
                rho_0_lower=rho_0.lower,
                rho_0_upper=rho_0.upper,
                source=spec.mode,
            )
        )
    return calibrations


def _per_cohort_scenarios(plan: AnalysisPlan) -> set[str]:
    """The assumed scenarios that give some cohort a box of its own."""

    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "assumed":
        return set()
    return {item.id for item in spec.scenarios if item.cohorts is not None}


def _validated_judge_calibrations(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
) -> tuple[
    dict[_JudgeScope, JudgeCalibration],
    dict[_JudgeScope, dict[str, Any]],
    float | None,
    dict[_JudgeScope, Any],
    dict[str, Any] | None,
]:
    """Estimate one calibration per cohort, or per cohort and scoped group, from human labels.

    A scoped calibration counts only the labels, and the stratum sizes, of its group's rollouts.
    The last item records the declared labelling protocol and what checking it found, or is
    ``None`` when the plan declares none.
    """

    spec = plan.inputs.judge_calibration
    if spec is None or spec.mode != "validated":
        return {}, {}, None, {}, None
    family = next((item for item in plan.multiplicity.families if item.id == spec.family), None)
    if family is None:
        diagnostics.append(
            Diagnostic(
                code="PLAN_UNKNOWN_REFERENCE",
                severity="ERROR",
                message="validated judge calibration requires a valid multiplicity family",
                scope=plan_scope("inputs", "judge_calibration"),
            )
        )
        return {}, {}, None, {}, None
    if spec.validation_manifest is None:
        raise ValueError("a validated inputs.judge_calibration requires a validation_manifest")
    try:
        labels = load_validation_manifest(spec.validation_manifest)
    except (OSError, ValueError) as exc:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_INVALID",
                severity="ERROR",
                message=str(exc),
                scope=plan_scope("inputs", "judge_calibration"),
            )
        )
        return {}, {}, None, {}, None

    protocol = _labelling_protocol(spec, labels, diagnostics)
    if spec.protocol is not None and protocol is None:
        return {}, {}, None, {}, None
    bound: CalibrationBound = (
        "confidence_sequence"
        if spec.protocol is not None and spec.protocol.stopping == "data_dependent"
        else "clopper_pearson"
    )

    declared: dict[str, str] | None = None
    if spec.sampling == "stratified_without_replacement":
        if spec.validation_strata is None:
            raise ValueError("stratified judge-calibration sampling requires validation_strata")
        try:
            declared = load_validation_strata(spec.validation_strata)
        except (OSError, ValueError) as exc:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_VALIDATION_INVALID",
                    severity="ERROR",
                    message=str(exc),
                    scope=plan_scope("inputs", "judge_calibration"),
                )
            )
            return {}, {}, None, {}, None
        design = fine_stratum_counts(data.observations, labels, declared)
        # The strata file is the design: a classified row it does not place, or a key that
        # names no classified row, means the sizes it implies are not the design's.
        for code, keys, what in (
            (
                "JUDGE_VALIDATION_STRATA_INCOMPLETE",
                design.unstratified,
                "classified observations have no declared stratum",
            ),
            (
                "JUDGE_VALIDATION_STRATA_UNMATCHED",
                design.unknown_strata,
                "validation strata rows reference observations absent from the analysed "
                "logs or with unclassified outcomes",
            ),
        ):
            if keys:
                diagnostics.append(
                    Diagnostic(
                        code=code,
                        severity="ERROR",
                        message=f"{len(keys)} {what}; no judge correction is reported",
                        scope=plan_scope("inputs", "judge_calibration"),
                        details={"observation_ids": keys[:20]},
                    )
                )
        if design.unstratified or design.unknown_strata:
            return {}, {}, None, {}, None

    _, unmatched = judge_stratum_counts(data.observations, labels)
    scopes = _judge_scopes(plan)
    counts: dict[_JudgeScope, JudgeStrata] = {}
    fine: dict[_JudgeScope, FineStrata] = {}
    truth_strata: dict[_JudgeScope, Any] = {}
    in_scope: set[str] = set()
    for scope in scopes:
        cohort_id, group_id = scope
        observations = data.observations
        if group_id is not None:
            prompts = set(next(item for item in data.groups if item.id == group_id).prompt_ids)
            observations = [row for row in observations if row.prompt_id in prompts]
            in_scope.update(row.observation_id for row in observations)
        if declared is not None:
            # A scoped calibration's fine strata are the declared strata within its group.
            scoped_fine = fine_stratum_counts(observations, labels, declared)
            if cohort_id in scoped_fine.by_cohort:
                counts[scope] = scoped_fine.judge_strata(cohort_id)
                fine[scope] = scoped_fine.by_cohort[cohort_id]
        elif cohort_id in (scoped := judge_stratum_counts(observations, labels)[0]):
            counts[scope] = scoped[cohort_id]
        if spec.sampling == "simple_random":
            # A simple random sample of the design also estimates sensitivity and the
            # false-positive rate by true class, which the noisy-but-valid test needs.
            by_class = truth_stratum_counts(observations, labels).get(cohort_id)
            if by_class is not None:
                truth_strata[scope] = {
                    name: {"labelled": item.labelled, "judge_positive": item.judge_positive}
                    for name, item in by_class.items()
                }
    if spec.groups is not None:
        outside = sorted(set(labels) - set(unmatched) - in_scope)
        if outside:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_VALIDATION_OUT_OF_SCOPE",
                    severity="WARNING",
                    message=(
                        f"{len(outside)} validation rows label rollouts outside the groups the "
                        "calibration is scoped to; they calibrate nothing"
                    ),
                    scope=plan_scope("inputs", "judge_calibration"),
                    details={"observation_ids": outside[:20]},
                )
            )
    if unmatched:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_UNMATCHED",
                severity="ERROR",
                message=(
                    f"{len(unmatched)} validation rows reference observations absent from the "
                    "analysed logs or with unclassified outcomes"
                ),
                scope=plan_scope("inputs", "judge_calibration"),
                details={"observation_ids": unmatched[:20]},
            )
        )

    # Every calibration (one per declared cohort, or per cohort and scoped group) gets an
    # equal share of the judge family budget. A single risk row consumes one share; a
    # comparison consumes the two it spans. A union over all of them therefore never
    # exceeds the declared family alpha.
    per_calibration_alpha = cohort_alpha_share(family.alpha, len(scopes))
    calibrations: dict[_JudgeScope, JudgeCalibration] = {}
    summaries: dict[_JudgeScope, dict[str, Any]] = {}
    for scope in scopes:
        cohort_id, group_id = scope
        where = (
            cohort_scope(cohort_id)
            if group_id is None
            else cohort_scope(cohort_id, "groups", group_id)
        )
        outcome = calibrate_cohort(
            counts.get(scope, NO_STRATA),
            scenario_id="validated",
            alpha=per_calibration_alpha,
            fine=None if declared is None else fine.get(scope, {}),
            bound=bound,
        )
        if outcome.empty_strata:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_VALIDATION_STRATUM_EMPTY",
                    severity="WARNING",
                    message=(
                        f"cohort {cohort_id!r} has no observations in one judge stratum; "
                        "no judge correction is reported for it"
                        if group_id is None
                        else f"group {group_id!r} of cohort {cohort_id!r} has no observations "
                        "in one judge stratum; no judge correction is reported for it"
                    ),
                    scope=where,
                )
            )
            continue
        if outcome.calibration is None:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_CALIBRATION_UNINFORMATIVE",
                    severity="WARNING",
                    message=str(outcome.refusal),
                    scope=where,
                )
            )
            continue
        calibrations[scope] = outcome.calibration
        completed = completion_diagnostic(outcome.calibration, where)
        if completed is not None:
            diagnostics.append(completed)
        summary: dict[str, Any] = dict(outcome.strata.summary())
        if declared is not None:
            partial = unlabelled_fine_strata_diagnostic(outcome.fine_strata, where)
            if partial is not None:
                diagnostics.append(partial)
            summary["fine_strata"] = fine_strata_summary(outcome.fine_strata)
        summaries[scope] = summary
    if declared is not None and calibrations:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CALIBRATION_STRATIFIED",
                severity="INFO",
                message=(
                    "Experimental: the judge calibration reads its labels as a stratified "
                    "sample without replacement within declared fine strata, with exact "
                    "hypergeometric bounds; the asymptotic delta-method reference, which "
                    "assumes simple random samples within the judge strata, is not reported"
                ),
                scope=plan_scope("inputs", "judge_calibration"),
            )
        )
    return calibrations, summaries, family.alpha, truth_strata, protocol


def _labelling_protocol(
    spec: JudgeCalibrationSpec,
    labels: dict[str, bool],
    diagnostics: list[Diagnostic],
) -> dict[str, Any] | None:
    """Check the labels against the declared labelling protocol, and record both.

    ``None`` with an ERROR diagnostic when the labels break the protocol, and ``None``
    without one when the plan declares no protocol.
    """

    protocol = spec.protocol
    if protocol is None:
        return None
    try:
        prefix = check_labelled_prefix(protocol.queue, labels, tier=protocol.tier)
        if protocol.stopping == "fixed_size" and prefix.labelled != protocol.size:
            raise ValueError(
                f"the labelling protocol fixes {protocol.size} labels in advance, but the "
                f"validation manifest has {prefix.labelled}"
            )
    except (OSError, ValueError) as exc:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_LABELLING_PROTOCOL_VIOLATED",
                severity="ERROR",
                message=str(exc),
                scope=plan_scope("inputs", "judge_calibration", "protocol"),
            )
        )
        return None
    return {
        "queue_sha256": prefix.queue_sha256,
        "queue_rows": prefix.queue_rows,
        "labelled_prefix": prefix.labelled,
        "tier": protocol.tier,
        "seed": protocol.seed,
        "stopping": protocol.stopping,
        "size": protocol.size,
        "bound": (
            "confidence_sequence"
            if protocol.stopping == "data_dependent"
            else "clopper_pearson"
        ),
    }


def _judge_corrected_interval(
    *,
    result: GroupRiskResult | HierarchyRiskResult,
    calibration: JudgeCalibration,
    primary_scenario: str | None,
    family_id: str | None,
    judge_alpha: float | None,
    gap: str | None = None,
    unclassified_share: float = 0.0,
    scoped_group: str | None = None,
    scoped_groups: list[str] | None = None,
    stratified: bool = False,
    per_cohort: bool = False,
) -> RiskIntervalResult | None:
    """Correct one row; ``scoped_group`` names a validated calibration's group, and
    ``scoped_groups`` the groups an assumed calibration is scoped to. ``per_cohort`` says
    that the assumed scenario declares boxes by cohort, so ``calibration`` is the row's
    cohort's box."""

    if result.lower is None or result.upper is None:
        return None
    corrected = correct_judge_risk(
        observed_lower=result.lower,
        observed_upper=result.upper,
        observed_estimate=result.estimate,
        calibration=calibration,
        unclassified_share=unclassified_share,
    )
    validated = calibration.source == "validated"
    # Measured accuracies certify only a row they were measured for, and never more than
    # the row's own bounds support. Elsewhere the corrected interval is a sensitivity, as an
    # assumed one always is.
    certified = validated and gap is None and result.eligibility == "certifying"
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
    if validated and stratified:
        if calibration.completed_strata:
            assumptions.append(
                "The "
                + " and ".join(calibration.completed_strata)
                + " judge stratum has no human label, so its range is [0, 1], which holds "
                "with certainty."
            )
        assumptions.extend(
            [
                "Experimental: each range sums exact hypergeometric bounds on the truly unsafe "
                "count of every declared fine stratum within its judge stratum, weighted by "
                "the fine stratum's size; a census is exact and a fine stratum without labels "
                "is completed from none to all of its rollouts.",
                "Each endpoint's budget is split equally over the fine strata that were "
                "sampled but not censused.",
                "The labelled rows of each fine stratum are a simple random sample without "
                "replacement of it, of a size fixed independently of the labels; this is a "
                "property of the declared collection protocol, not of the data.",
                "The declared validation strata place every classified rollout of the cohort.",
                "Human adjudication is treated as ground truth.",
            ]
        )
        if scoped_group is not None:
            assumptions.append(
                f"The calibration is scoped to group {scoped_group!r}: its fine strata are the "
                "declared strata of the group's rollouts in the cohort."
            )
        if certified:
            population = (
                "the cohort" if scoped_group is None else f"group {scoped_group!r} in the cohort"
            )
            assumptions.extend(
                [
                    f"The fine strata weight back to every classified rollout of {population} "
                    "equally, and so does this row, so they measure the judge's accuracy for "
                    "its estimand.",
                    "Coverage is the union bound over the underlying interval and every "
                    "calibration count bound.",
                ]
            )
    elif validated:
        sequence = calibration.bound == "confidence_sequence"
        method = (
            "anytime-valid hypergeometric confidence sequence (Waudby-Smith and Ramdas, 2020)"
            if sequence
            else "exact Clopper-Pearson bound"
        )
        if calibration.completed_strata:
            assumptions.append(
                "The "
                + " and ".join(calibration.completed_strata)
                + " judge stratum has no human label, so its range is [0, 1], which holds "
                f"with certainty; any other range is an {method} from "
                "human labels drawn within its judge stratum."
            )
        else:
            assumptions.append(
                f"Each of those two ranges is an {method} from human labels drawn "
                "within its judge stratum."
                if sequence
                else "Those two ranges are exact Clopper-Pearson bounds from human labels "
                "drawn within each judge stratum."
            )
        assumptions.extend(
            [
                (
                    "The labelled rows were drawn one at a time, without replacement and in a "
                    "random order, within their judge stratum, as the declared labelling "
                    "queue orders them; the ranges hold at every number of labels at once, "
                    "so labelling may have stopped because of the labels seen."
                    if sequence
                    else "The labelled rows are a simple random sample within their judge "
                    "stratum; this is a property of the declared collection protocol, not "
                    "of the data."
                ),
                "Human adjudication is treated as ground truth.",
            ]
        )
        if scoped_group is not None:
            assumptions.append(
                f"The calibration is scoped to group {scoped_group!r}: its labels and judge "
                "strata are those of the group's rollouts in the cohort."
            )
        if certified:
            labels = (
                "The labels weight every classified rollout of the cohort equally"
                if scoped_group is None
                else f"The labels weight every classified rollout of group {scoped_group!r} "
                "in the cohort equally"
            )
            assumptions.extend(
                [
                    f"{labels}, and so "
                    "does this row, so they measure the judge's accuracy for its estimand.",
                    "Coverage is the union bound over the underlying interval and the four "
                    "calibration endpoints.",
                ]
            )
    else:
        scope = (
            ""
            if scoped_groups is None
            else ", on the rows of groups " + ", ".join(repr(item) for item in scoped_groups)
            + " only"
        )
        assumptions.extend(
            [
                "These conditional accuracies are declared assumptions, not measurements; "
                "the interval inherits no coverage guarantee from them.",
                (
                    f"The declared accuracies are prevalence-dependent; the scenario declares "
                    f"this box for cohort {result.cohort_id!r}{scope}."
                    if per_cohort
                    else "The declared accuracies are prevalence-dependent and are applied "
                    f"unchanged to every cohort{scope}."
                ),
            ]
        )
    if unclassified_share > 0:
        assumptions.append(_unclassified_assumption(unclassified_share))
    assumptions.append("The underlying judge-system interval retains its own stated guarantee.")
    warnings: list[str] = []
    if validated and gap is not None:
        warnings.append(_uncertified_correction_warning(gap, scoped_group))
    elif validated and not certified:
        warnings.append(_uncertified_base_warning(result.eligibility))
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
        varied_assumption=None if certified else "judge_error",
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


def _transported_judge_interval(
    *,
    result: GroupRiskResult | HierarchyRiskResult,
    published: PublishedValidationSetSpec,
    sensitivity_lower: float,
) -> RiskIntervalResult | None:
    """A true-risk sensitivity from a judge sensitivity measured on a published set.

    Experimental. Only the sensitivity is transported; the false-positive rate is bounded
    by the run's own judged rate, so the interval's lower end is zero.
    """

    if result.upper is None:
        return None
    corrected = transport_judge_risk(
        observed_upper=result.upper,
        observed_estimate=result.estimate,
        sensitivity_lower=sensitivity_lower,
    )
    return RiskIntervalResult(
        interval_id="judge_corrected_transported",
        lower=corrected.lower,
        upper=corrected.upper,
        method="transported_judge_sensitivity_correction",
        role="sensitivity",
        varied_assumption="judge_error",
        coverage_scope=(
            f"{result.coverage_scope}_judge_corrected"
            if result.coverage_scope
            else "judge_corrected"
        ),
        completion_estimate_lower=corrected.completion_estimate_lower,
        completion_estimate_upper=corrected.completion_estimate_upper,
        raw_lower=result.lower,
        raw_upper=result.upper,
        calibration_method="transported_judge_calibration",
        assumptions=[
            "The reported estimand is true failure risk, not judge-reported failure risk.",
            f"The judge's sensitivity is at least {sensitivity_lower:.6g}: the one-sided "
            f"Clopper-Pearson lower bound at alpha {published.sensitivity_alpha:.6g} from "
            f"{published.judge_flagged_unsafe} of {published.human_unsafe} human-unsafe items "
            f"flagged on the published set {published.name!r} ({published.source}).",
            "The Clopper-Pearson bound treats the published set's human-unsafe items as "
            "independent draws that share one sensitivity. Several completions of the same "
            "behaviour are not independent, so counts that include them overstate the bound's "
            "precision.",
            f"Transport assumption: the same judge ({published.judge}) and rubric "
            f"({published.rubric}), applied to this run's outputs, is at least as sensitive "
            "on every prompt as on the published set, whose outputs came from other "
            "generators. Nothing in this run checks it.",
            "The published set's false-positive rate is not transported. This run's false "
            "positives are bounded only by its own judged failures, so every judged failure "
            "may be a false positive and the lower bound is zero.",
            "The upper bound is the judge-system upper bound divided by the sensitivity "
            "bound; unclassified rollouts stay completed as failures.",
            "Human adjudication on the published set is treated as ground truth.",
        ],
        eligibility="diagnostic",
        warnings=[
            "Experimental transported judge calibration: a sensitivity analysis with no "
            "coverage guarantee, which never decides a claim."
        ],
    )
