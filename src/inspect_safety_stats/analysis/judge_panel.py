"""The judge panel summary, and the correction derived from declared accuracies."""

from __future__ import annotations

import itertools
from typing import Any

from inspect_safety_stats.analysis.judge_calibration import _JUDGE_CORRECTABLE_MEASURES
from inspect_safety_stats.config.models import AnalysisPlan, JudgeAccuracySpec, JudgeSpec
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import (
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    RiskIntervalResult,
)
from inspect_safety_stats.stats.judges import (
    JudgeAccuracyBox,
    compose_verdicts,
    composite_accuracy_box,
    effective_votes,
    identified_risk_set,
    pairwise_agreement,
)
from inspect_safety_stats.stats.paired import exact_mcnemar


def _judge_panel_summary(plan: AnalysisPlan, data: NormalizedData) -> dict[str, Any] | None:
    """Describe the declared judge panel: per-judge counts, composites, and agreement.

    Bookkeeping only. Effective votes is the Kish count over declared dependence
    blocks, and agreement is descriptive; nothing here corrects a risk estimate.

    With ``inputs.panel_groups`` every judge, composite and pair also carries ``groups``:
    the same counts and agreement for each listed group's rollouts, per cohort, so a
    cohort that holds several benchmarks does not mix them into one figure.
    """

    if not plan.inputs.judges:
        return None
    cohorts = sorted(plan.cohorts)
    blocks = {judge.id: judge.block or judge.id for judge in plan.inputs.judges}
    pairs = list(itertools.combinations([judge.id for judge in plan.inputs.judges], 2))

    def counts(values: list[bool | None]) -> dict[str, int]:
        positives = sum(value is True for value in values)
        negatives = sum(value is False for value in values)
        return {
            "classified": positives + negatives,
            "failures": positives,
            "unclassified": len(values) - positives - negatives,
        }

    def agreement_stats(
        first_votes: list[bool | None], second_votes: list[bool | None]
    ) -> dict[str, Any]:
        summary = pairwise_agreement(first_votes, second_votes)
        paired = [
            (bool(a), bool(b))
            for a, b in zip(first_votes, second_votes, strict=True)
            if a is not None and b is not None
        ]
        first_only = second_only = 0
        p_value: float | None = None
        if paired:
            mcnemar = exact_mcnemar([a for a, _ in paired], [b for _, b in paired])
            first_only = mcnemar.candidate_failure_baseline_safe
            second_only = mcnemar.candidate_safe_baseline_failure
            p_value = mcnemar.p_value_two_sided
        return {
            "compared": summary.compared,
            "disagreements": summary.disagreements,
            "agreement": summary.agreement,
            "kappa": summary.kappa,
            "discordant_first_only": first_only,
            "discordant_second_only": second_only,
            "mcnemar_p_value": p_value,
        }

    def section(prompts: set[str] | None) -> tuple[
        dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]
    ]:
        """Per-cohort figures for each judge, composite and pair, over these prompts' rollouts.

        ``prompts`` None means every rollout.
        """

        by_cohort: dict[str, list[str]] = {cohort_id: [] for cohort_id in cohorts}
        for observation in data.observations:
            if prompts is None or observation.prompt_id in prompts:
                by_cohort.setdefault(observation.cohort_id, []).append(
                    observation.observation_id
                )

        def verdicts(cohort_id: str, judge_id: str) -> list[bool | None]:
            return [
                data.judge_verdicts.get(observation_id, {}).get(judge_id)
                for observation_id in by_cohort[cohort_id]
            ]

        judge_figures = {
            judge.id: {cohort_id: counts(verdicts(cohort_id, judge.id)) for cohort_id in cohorts}
            for judge in plan.inputs.judges
        }
        composite_figures = {
            composite.id: {
                cohort_id: counts(
                    [
                        compose_verdicts(
                            {
                                judge_id: data.judge_verdicts.get(observation_id, {}).get(judge_id)
                                for judge_id in composite.judges
                            },
                            composite.rule,
                        )
                        for observation_id in by_cohort[cohort_id]
                    ]
                )
                for cohort_id in cohorts
            }
            for composite in plan.inputs.composites
        }
        pair_figures = [
            {
                cohort_id: agreement_stats(verdicts(cohort_id, first), verdicts(cohort_id, second))
                for cohort_id in cohorts
            }
            for first, second in pairs
        ]
        return judge_figures, composite_figures, pair_figures

    judge_figures, composite_figures, pair_figures = section(None)
    judges: list[dict[str, Any]] = [
        {
            "id": judge.id,
            "kind": judge.kind,
            "block": blocks[judge.id],
            "score": judge.score.name,
            "cohorts": judge_figures[judge.id],
        }
        for judge in plan.inputs.judges
    ]
    composites: list[dict[str, Any]] = [
        {
            "id": composite.id,
            "rule": composite.rule,
            "judges": list(composite.judges),
            "effective_votes": effective_votes([blocks[j] for j in composite.judges]),
            "cohorts": composite_figures[composite.id],
        }
        for composite in plan.inputs.composites
    ]
    agreement: list[dict[str, Any]] = [
        {"first": first, "second": second, "cohorts": figures}
        for (first, second), figures in zip(pairs, pair_figures, strict=True)
    ]
    panel: dict[str, Any] = {
        "judges": judges,
        "composites": composites,
        "agreement": agreement,
        "effective_votes_all": effective_votes(list(blocks.values())),
    }
    if plan.inputs.panel_groups is not None:
        prompts_by_group = {group.id: set(group.prompt_ids) for group in data.groups}
        panel["groups"] = list(plan.inputs.panel_groups)
        for group_id in plan.inputs.panel_groups:
            by_judge, by_composite, by_pair = section(prompts_by_group[group_id])
            for entry in judges:
                entry.setdefault("groups", {})[group_id] = by_judge[entry["id"]]
            for entry in composites:
                entry.setdefault("groups", {})[group_id] = by_composite[entry["id"]]
            for entry, figures in zip(agreement, by_pair, strict=True):
                entry.setdefault("groups", {})[group_id] = figures
    return panel


def _declared_box(accuracy: JudgeAccuracySpec) -> JudgeAccuracyBox:
    return JudgeAccuracyBox(
        sensitivity_lower=accuracy.sensitivity.lower,
        sensitivity_upper=accuracy.sensitivity.upper,
        specificity_lower=accuracy.specificity.lower,
        specificity_upper=accuracy.specificity.upper,
    )


def _composite_box(
    members: list[JudgeSpec], accuracies: list[JudgeAccuracySpec], rule: str
) -> JudgeAccuracyBox:
    """An ``any`` or ``all`` composite's box from its members' declared boxes and blocks."""
    return composite_accuracy_box(
        [_declared_box(accuracy) for accuracy in accuracies],
        [member.block or member.id for member in members],
        rule,
    )


def _declared_scope(plan: AnalysisPlan, accuracies: list[JudgeAccuracySpec]) -> list[str] | None:
    """The groups, in plan order, that every one of these declarations covers.

    None when no declaration is scoped to groups, so they all hold on every rollout. A
    declaration scoped to groups holds on those groups' rows only, so a box built from
    several declarations holds only where all of them do; the list can then be empty.
    """

    if all(accuracy.groups is None for accuracy in accuracies):
        return None
    return [
        group.id
        for group in plan.groups
        if all(accuracy.groups is None or group.id in accuracy.groups for accuracy in accuracies)
    ]


# A declared accuracy box, the verdict it is declared for, and the groups it holds on (None: all).
_DeclaredBox = tuple[JudgeAccuracyBox, str, list[str] | None]


def _primary_accuracy_box(plan: AnalysisPlan) -> _DeclaredBox | None:
    """The declared accuracy box of the verdict that is the primary outcome, if it has one.

    That is a declared judge whose score is the primary score, or an ``any`` or ``all``
    composite whose judges all declare accuracy ranges. A ``majority`` composite has none.
    """

    name = plan.inputs.score.name
    composite = next((item for item in plan.inputs.composites if item.id == name), None)
    judges = {judge.id: judge for judge in plan.inputs.judges}
    if composite is not None:
        members = [judges[judge_id] for judge_id in composite.judges]
        accuracies = [member.accuracy for member in members if member.accuracy is not None]
        if composite.rule == "majority" or len(accuracies) != len(members):
            return None
        return (
            _composite_box(members, accuracies, composite.rule),
            f"composite:{composite.id}",
            _declared_scope(plan, accuracies),
        )
    judge = next((item for item in plan.inputs.judges if item.score.name == name), None)
    if judge is None or judge.accuracy is None:
        return None
    return (
        _declared_box(judge.accuracy),
        f"judge:{judge.id}",
        _declared_scope(plan, [judge.accuracy]),
    )


def _apply_panel_derived_correction(
    plan: AnalysisPlan,
    risks: list[GroupRiskResult],
    hierarchy_risks: list[HierarchyRiskResult],
    diagnostics: list[Diagnostic],
) -> dict[str, Any] | None:
    """Partial identification of true risk when the primary outcome is a composite.

    The composite's accuracy box is derived from its judges' declared boxes under the
    declared dependence blocks, and the judge-reported interval is inverted over that
    box. Declared boxes are assumptions, so every interval this adds is a diagnostic
    sensitivity that can never move a claim.

    A judge whose accuracy declaration is scoped to groups vouches only for those groups'
    rows, so the box holds, and a row is corrected, only in the groups every member's
    declaration covers.
    """

    composite = next(
        (item for item in plan.inputs.composites if item.id == plan.inputs.score.name), None
    )
    if composite is None:
        return None
    judges = {judge.id: judge for judge in plan.inputs.judges}
    members = [judges[judge_id] for judge_id in composite.judges]
    accuracies = [member.accuracy for member in members if member.accuracy is not None]
    if len(accuracies) != len(members):
        return None
    if composite.rule == "majority":
        diagnostics.append(
            Diagnostic(
                code="PANEL_CORRECTION_UNSUPPORTED_RULE",
                severity="INFO",
                message=(
                    "a majority composite has no closed-form accuracy box under partial "
                    "dependence; validate the composite's own verdict instead"
                ),
                scope=f"inputs.composites.{composite.id}",
            )
        )
        return None
    scoped = _declared_scope(plan, accuracies)
    if scoped is not None and not scoped:
        diagnostics.append(
            Diagnostic(
                code="PANEL_CORRECTION_NO_COMMON_GROUP",
                severity="WARNING",
                message=(
                    f"the judges of composite {composite.id!r} declare their accuracies for "
                    "no group in common, so no panel-derived correction is reported"
                ),
                scope=f"inputs.composites.{composite.id}",
            )
        )
        return None
    box = _composite_box(members, accuracies, composite.rule)
    correctable = {
        measure.id
        for measure in plan.outputs.risk_measures
        if measure.type in _JUDGE_CORRECTABLE_MEASURES
    }
    targets: list[GroupRiskResult | HierarchyRiskResult] = [
        *(row for row in risks if row.measure_id in correctable),
        *hierarchy_risks,
    ]
    where = (
        "every prompt and cohort"
        if scoped is None
        else "every cohort, on the rows of groups "
        + ", ".join(repr(item) for item in scoped)
        + " only, the groups every judge's declaration covers"
    )
    for result in targets:
        if result.lower is None or result.upper is None:
            continue
        if scoped is not None and result.group_id not in scoped:
            continue
        identified = identified_risk_set(result.lower, result.upper, box)
        estimate_range = (
            identified_risk_set(result.estimate, result.estimate, box)
            if result.estimate is not None
            else None
        )
        result.intervals.append(
            RiskIntervalResult(
                interval_id="judge_corrected_panel_derived",
                lower=identified.lower,
                upper=identified.upper,
                method="derived_panel_misclassification_correction",
                role="sensitivity",
                varied_assumption="judge_error",
                coverage_scope=(
                    f"{result.coverage_scope}_panel_derived"
                    if result.coverage_scope
                    else "panel_derived"
                ),
                completion_estimate_lower=(
                    estimate_range.lower if estimate_range is not None else None
                ),
                completion_estimate_upper=(
                    estimate_range.upper if estimate_range is not None else None
                ),
                raw_lower=result.lower,
                raw_upper=result.upper,
                calibration_method="derived_panel_accuracy_box",
                assumptions=[
                    "The reported estimand is true failure risk, not judge-reported failure risk.",
                    f"The composite {composite.id!r} ({composite.rule}) accuracy box is derived "
                    "from its judges' declared sensitivity/specificity ranges: Frechet bounds "
                    "within each declared dependence block, independence across blocks.",
                    f"Derived sensitivity lies in [{box.sensitivity_lower:.6g}, "
                    f"{box.sensitivity_upper:.6g}] and specificity in "
                    f"[{box.specificity_lower:.6g}, {box.specificity_upper:.6g}].",
                    "Judge accuracies are declared, prevalence-free, and applied unchanged to "
                    f"{where}; they carry no coverage guarantee.",
                    "True risk is the Rogan-Gladen inversion over the box; extrema sit at the box "
                    "vertices because the inversion is separately monotone in each argument.",
                ],
                eligibility="diagnostic",
                warnings=(
                    []
                    if identified.identified
                    else [
                        "The judge-reported rate is outside what any accuracy in the declared "
                        "boxes can produce, so the data contradict the boxes; true risk is "
                        "unidentified and the reported set is the whole unit interval."
                    ]
                    if identified.unidentified_reason == "contradicted"
                    else [
                        "The declared boxes admit an uninformative panel (sensitivity + "
                        "specificity <= 1); true risk is unidentified and the reported set is "
                        "the whole unit interval."
                    ]
                ),
            )
        )
    return {
        "composite": composite.id,
        "rule": composite.rule,
        "judges": list(composite.judges),
        "derived_box": {
            "sensitivity": [box.sensitivity_lower, box.sensitivity_upper],
            "specificity": [box.specificity_lower, box.specificity_upper],
        },
        **({"groups": scoped} if scoped is not None else {}),
    }
