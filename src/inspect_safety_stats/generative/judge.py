"""Judge misclassification for a generative evaluation.

Every generative interval elsewhere estimates a property of the **judge-model system**:
the probability that the judge reports a failure under the declared generative design. That
is not the same as the probability the model fails, and at low prevalence the gap is large --
at 1% true prevalence and 95% specificity most of the reported number is judge error.

The correction is the same identity the fixed-design engine uses,
``R* = R*rho_1 + (1 - R)*rho_0``, which is affine in ``R`` with positive slope. Generative
root, node and slice estimates are all weighted means, so it applies to each of them
unchanged. See [judge misclassification](../../../docs/judge-misclassification.md) for the
derivation and for why this parameterisation is preferred to Rogan--Gladen.

Comparisons are deliberately not corrected -- see :func:`comparison_refusal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inspect_safety_stats.generative.inference import NodeEstimate
from inspect_safety_stats.generative.ledger import RewriteRecord
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.ingest.validation import load_validation_manifest
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    JudgeStratumCounts,
    completion_warning,
    correct_judge_risk,
    estimate_judge_calibration,
)

Eligibility = Literal["diagnostic", "certifying"]


@dataclass(frozen=True)
class CorrectedEstimate:
    """One estimate mapped from the judge-reported scale onto true risk."""

    node_id: str
    level: str
    scenario_id: str
    source: str
    eligibility: Eligibility
    observed_lower: float
    observed_upper: float
    lower: float
    upper: float
    estimate_lower: float | None
    estimate_upper: float | None
    alpha: float


def load_labels(path: str | Path) -> dict[str, bool]:
    """Read human adjudications keyed by materialized sample identifier.

    The file format is the fixed-design one -- ``observation_id,human_label`` with labels
    ``safe`` or ``unsafe`` -- but the identifier column carries the sample id that
    materialization wrote, since a generative run has no prompt manifest to key on.
    """

    return load_validation_manifest(path)


def judge_strata(
    rewrites: list[RewriteRecord],
    outcomes: list[tuple[str, int, int, int]],
    labels: dict[str, bool],
) -> tuple[JudgeStratumCounts, JudgeStratumCounts, list[str]]:
    """Cross human labels against judge verdicts, per judge stratum.

    ``outcomes`` are ``(sampling_unit_id, trials, failures, unclassified)`` tuples. A unit
    with an unclassified rollout belongs to neither stratum: it is already carried by the
    conservative completion inside the interval.
    """

    prompt_of = {
        record.sampling_unit_id: str(record.prompt_id)
        for record in rewrites
        if record.is_evaluable() and record.prompt_id is not None
    }
    sizes = {True: 0, False: 0}
    labelled = {True: 0, False: 0}
    unsafe = {True: 0, False: 0}
    seen: set[str] = set()

    for unit_id, trials, failures, unclassified in outcomes:
        if trials != 1:
            raise ValueError(
                "judge calibration needs one rollout per rewrite, so that a unit carries a "
                f"single binary judge verdict; unit {unit_id} has {trials}"
            )
        if unclassified:
            continue
        verdict = failures > 0
        sizes[verdict] += 1
        prompt_id = prompt_of.get(unit_id)
        if prompt_id is None:
            continue
        human = labels.get(prompt_id)
        if human is None:
            continue
        seen.add(prompt_id)
        labelled[verdict] += 1
        if human:
            unsafe[verdict] += 1

    return (
        JudgeStratumCounts(labelled[True], unsafe[True], sizes[True]),
        JudgeStratumCounts(labelled[False], unsafe[False], sizes[False]),
        sorted(set(labels) - seen),
    )


def declared_calibration(plan: GenerativePlan, scenario_id: str | None = None) -> JudgeCalibration:
    """Build the box a declared scenario asserts. Produces no evidence."""

    spec = plan.judge_calibration
    if spec is None or spec.mode != "assumed":
        raise ValueError("the plan declares no assumed judge calibration")
    wanted = scenario_id or spec.primary
    for scenario in spec.scenarios:
        if scenario.id == wanted:
            return JudgeCalibration(
                scenario_id=scenario.id,
                rho_1_lower=scenario.rho_1.lower,
                rho_1_upper=scenario.rho_1.upper,
                rho_0_lower=scenario.rho_0.lower,
                rho_0_upper=scenario.rho_0.upper,
                source="assumed",
            )
    raise ValueError(f"scenario {wanted!r} is not declared")


def measured_calibration(
    plan: GenerativePlan,
    rewrites: list[RewriteRecord],
    outcomes: list[tuple[str, int, int, int]],
    labels: dict[str, bool],
    *,
    cohort_id: str,
) -> tuple[JudgeCalibration | None, list[Diagnostic]]:
    """Estimate the judge's conditional accuracies from human labels for one cohort.

    The calibration budget is divided equally among the declared cohorts, because rho is
    prevalence-dependent and each cohort needs its own. Within a cohort's share,
    :func:`estimate_judge_calibration` splits between the two parameters and again between
    their endpoints, so a union over all four spends no more than the share.
    """

    spec = plan.judge_calibration
    if spec is None or spec.mode != "validated":
        raise ValueError("the plan declares no validated judge calibration")

    diagnostics: list[Diagnostic] = []
    unsafe, safe, unmatched = judge_strata(rewrites, outcomes, labels)
    if unmatched:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_UNMATCHED",
                severity="ERROR",
                message=(
                    f"{len(unmatched)} human labels reference samples this run did not "
                    f"evaluate, starting with {unmatched[0]!r}"
                ),
                scope=cohort_id,
            )
        )
    for name, stratum in (("judge_unsafe", unsafe), ("judge_safe", safe)):
        if stratum.stratum_size == 0:
            diagnostics.append(
                Diagnostic(
                    code="JUDGE_VALIDATION_STRATUM_EMPTY",
                    severity="WARNING",
                    message=f"the {name} stratum is empty, so rho cannot be estimated",
                    scope=cohort_id,
                )
            )
            return None, diagnostics

    share = spec.alpha / max(1, len(plan.target.cohorts))
    try:
        calibration = estimate_judge_calibration(
            scenario_id=cohort_id, unsafe_stratum=unsafe, safe_stratum=safe, alpha=share
        )
    except ValueError as exc:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CALIBRATION_UNINFORMATIVE",
                severity="WARNING",
                message=str(exc),
                scope=cohort_id,
                details={
                    "judge_unsafe_labelled": unsafe.labelled,
                    "judge_safe_labelled": safe.labelled,
                },
            )
        )
        return None, diagnostics
    warning = completion_warning(calibration)
    if warning is not None:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_STRATUM_UNLABELLED_COMPLETED",
                severity="WARNING",
                message=warning,
                scope=cohort_id,
                details={"completed_strata": list(calibration.completed_strata)},
            )
        )
    return calibration, diagnostics


def correct(
    estimates: list[NodeEstimate],
    calibration: JudgeCalibration,
    *,
    interval_alpha: float,
) -> list[CorrectedEstimate]:
    """Map estimates onto the true-risk scale.

    The reported ``alpha`` is the union of the interval's own budget and whatever the
    calibration spent, because that is the corrected statement's real error probability.
    """

    certifying = calibration.source == "validated"
    combined = interval_alpha + (calibration.alpha or 0.0)
    corrected = []
    for estimate in estimates:
        mapped = correct_judge_risk(
            observed_lower=estimate.lower,
            observed_upper=estimate.upper,
            observed_estimate=estimate.estimate,
            calibration=calibration,
        )
        corrected.append(
            CorrectedEstimate(
                node_id=estimate.node_id,
                level=estimate.level,
                scenario_id=mapped.scenario_id,
                source=mapped.source,
                eligibility="certifying" if certifying else "diagnostic",
                observed_lower=estimate.lower,
                observed_upper=estimate.upper,
                lower=mapped.lower,
                upper=mapped.upper,
                estimate_lower=mapped.completion_estimate_lower,
                estimate_upper=mapped.completion_estimate_upper,
                alpha=combined if certifying else interval_alpha,
            )
        )
    return corrected


def comparison_refusal(cohort_pair: str) -> Diagnostic:
    """Comparisons stay on the judge-reported scale, and say so.

    Correcting a paired comparison would mean bounding each arm separately and
    recombining, which discards the concordance the paired construction depends on --
    typically a two- to ten-fold loss. Scaling the paired difference by a single
    ``rho_1 - rho_0`` instead would be tighter, but only under a prevalence-free judge,
    which this package does not otherwise assume. Neither trade is made silently.
    """

    return Diagnostic(
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON",
        severity="WARNING",
        message=(
            f"comparison {cohort_pair} stays on the judge-reported scale: correcting it "
            "would either discard the paired construction or assume a prevalence-free judge"
        ),
        scope=cohort_pair,
    )


def slice_family_refusal() -> Diagnostic:
    """The declared slice family stays on the judge-reported scale, and says so.

    Correcting it would spend part of the family's error budget on the judge calibration
    and map each member separately; the family's simultaneous calibration does not cover
    that yet, so no member is corrected rather than some.
    """

    return Diagnostic(
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_SLICE_FAMILY",
        severity="WARNING",
        message=(
            "the declared slice family stays on the judge-reported scale: its simultaneous "
            "calibration does not include judge correction"
        ),
        scope="slice_family",
    )
