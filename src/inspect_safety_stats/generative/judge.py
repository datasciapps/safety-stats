"""Judge misclassification for a generative evaluation.

Every generative interval elsewhere estimates a property of the **judge-model system**:
the probability that the judge reports a failure under the declared generative design. That
is not the same as the probability the model fails, and at low prevalence the gap is large --
at 1% true prevalence and 95% specificity most of the reported number is judge error.

The correction is the same identity the fixed-design engine uses,
``R* = R*rho_1 + (1 - R)*rho_0``, which is affine in ``R`` with positive slope. Generative
root, node and slice estimates are all weighted means, so the identity holds for each of
them, but only with rho weighted the way the estimate weights its units. Measured rho pools
every classified unit of the cohort with equal weight, so it certifies only the root of a
design that weights its units equally; every other corrected estimate is a sensitivity. See
[judge misclassification](../../../docs/judge-misclassification.md) for the derivation and
for why this parameterisation is preferred to Rogan--Gladen.

Comparisons are deliberately not corrected -- see :func:`comparison_refusal`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inspect_safety_stats.generative.inference import NodeEstimate
from inspect_safety_stats.generative.ledger import RewriteRecord
from inspect_safety_stats.generative.schema import GenerativePlan
from inspect_safety_stats.ingest.validation import load_validation_manifest
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.judge_calibration import (
    NO_STRATA,
    JudgeStrata,
    calibrate_cohort,
    cohort_alpha_share,
    completion_diagnostic,
    count_judge_strata,
)
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    JudgeStratumCounts,
    correct_judge_risk,
)

Eligibility = Literal["diagnostic", "certifying"]
# A generative analysis calibrates one cohort per run, so its rows share one cohort key.
_COHORT = "run"


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
    warnings: tuple[str, ...] = ()


def pooled_calibration_gap(estimate: NodeEstimate, *, root_id: str) -> str | None:
    """Why a cohort's pooled labels do not measure the judge accuracies an estimate needs.

    The labels are drawn uniformly within judge strata over the cohort's classified units,
    so they estimate P(truly unsafe | verdict) with each of those units weighted equally.
    ``R* = R rho_1 + (1 - R) rho_0`` needs the accuracies weighted the way the estimate
    weights its units. rho depends on prevalence, so a node or slice where the judge misses
    more failures than average would get a corrected upper bound below its true risk. That
    holds exactly only for the root when every unit carries the same coefficient in it.
    Returns None for such a root.
    """

    if estimate.node_id != root_id:
        kind = "slice" if estimate.node_id.startswith("slice:") else "node"
        return (
            f"the {kind} covers only part of the cohort's labelled units, where the judge's "
            "accuracy can differ from the pooled labels' average"
        )
    if not estimate.equal_coefficients:
        return (
            f"the root gives some units {estimate.coefficient_ratio:.3g} times the weight of "
            "others"
        )
    return None


def uncertified_correction_warning(gap: str) -> str:
    return (
        f"Sensitivity analysis, not a certified bound: {gap}. The human labels weight every "
        "classified unit of the cohort equally, so they measure the judge's accuracy only for "
        "the root of a design that does the same."
    )


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

    ``outcomes`` are ``(sampling_unit_id, trials, failures, unclassified)`` tuples, and
    labels are keyed by the sample id a unit's rewrite was materialized under. A unit with
    an unclassified rollout belongs to neither stratum; see :func:`count_judge_strata`,
    which does the counting. The labels say nothing about such a unit, so :func:`correct`
    keeps its conservative completion rather than mapping it through rho.
    """

    prompt_of = {
        record.sampling_unit_id: str(record.prompt_id)
        for record in rewrites
        if record.is_evaluable() and record.prompt_id is not None
    }

    def rows() -> Iterator[tuple[str, str | None, bool | None]]:
        for unit_id, trials, failures, unclassified in outcomes:
            if trials != 1:
                raise ValueError(
                    "judge calibration needs one rollout per rewrite, so that a unit carries "
                    f"a single binary judge verdict; unit {unit_id} has {trials}"
                )
            verdict = None if unclassified else failures > 0
            yield _COHORT, prompt_of.get(unit_id), verdict

    counts, unmatched = count_judge_strata(rows(), labels)
    strata = counts.get(_COHORT, NO_STRATA)
    return strata.judge_unsafe, strata.judge_safe, unmatched


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
    outcome = calibrate_cohort(
        JudgeStrata(unsafe, safe),
        scenario_id=cohort_id,
        alpha=cohort_alpha_share(spec.alpha, len(plan.target.cohorts)),
    )
    if outcome.empty_strata:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_VALIDATION_STRATUM_EMPTY",
                severity="WARNING",
                message=(
                    f"the {outcome.empty_strata[0]} stratum is empty, so rho cannot be estimated"
                ),
                scope=cohort_id,
            )
        )
        return None, diagnostics
    if outcome.calibration is None:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_CALIBRATION_UNINFORMATIVE",
                severity="WARNING",
                message=str(outcome.refusal),
                scope=cohort_id,
                details={
                    "judge_unsafe_labelled": unsafe.labelled,
                    "judge_safe_labelled": safe.labelled,
                },
            )
        )
        return None, diagnostics
    completed = completion_diagnostic(outcome.calibration, cohort_id)
    if completed is not None:
        diagnostics.append(completed)
    return outcome.calibration, diagnostics


def correct(
    estimates: list[NodeEstimate],
    calibration: JudgeCalibration,
    *,
    interval_alpha: float,
    root_id: str,
) -> list[CorrectedEstimate]:
    """Map estimates onto the true-risk scale.

    A measured calibration certifies only the estimates :func:`pooled_calibration_gap`
    finds no gap for; the rest are sensitivities that say why. The reported ``alpha`` of a
    certifying estimate is the union of the interval's own budget and whatever the
    calibration spent, because that is the corrected statement's real error probability.

    Units with no judge verdict keep the completion the observed bounds gave them, safe at
    the lower end and failing at the upper: only the classified mass is mapped through rho.
    """

    validated = calibration.source == "validated"
    combined = interval_alpha + (calibration.alpha or 0.0)
    corrected = []
    for estimate in estimates:
        gap = pooled_calibration_gap(estimate, root_id=root_id) if validated else None
        certifying = validated and gap is None
        mapped = correct_judge_risk(
            observed_lower=estimate.lower,
            observed_upper=estimate.upper,
            observed_estimate=estimate.estimate,
            calibration=calibration,
            unclassified_share=estimate.unclassified_share,
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
                warnings=(uncertified_correction_warning(gap),) if gap is not None else (),
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
