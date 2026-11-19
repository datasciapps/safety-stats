"""Judge-error stress for threshold verdicts: where a verdict tips, and whether it survives.

A threshold claim decided on judge-reported risk says nothing about how good the judge has
to be for the verdict to hold for true risk. Under the prevalence-free model, a judge with
sensitivity ``Se`` and false-positive rate ``FPR = 1 - Sp`` reports failures at

    q = pi * Se + (1 - pi) * FPR

for true risk ``pi``, applied per prompt. For an informative judge (``Se > FPR``) the map is
affine and increasing in ``pi``, so it carries over to every monotone, positively homogeneous,
translation-equivariant measure: weighted mean, maximum, CVaR and top-k mean. With
judge-reported bounds ``[L, U]``, true risk lies in

    [(L - FPR) / (Se - FPR), (U - FPR) / (Se - FPR)],

the Rogan-Gladen inversion of ``identified_risk_set``, which the panel-derived correction
uses too.

Rollouts the judge never classified need no share of their own. The bounds complete them as
judge-safe at the lower end and judge-unsafe at the upper, and inverting that completion is
conservative on both sides. A rollout completed as judge-unsafe adds
``(1 - FPR) / (Se - FPR) >= 1`` of its weight to the upper bound, at least the 1 it can add to
true risk, and one completed as judge-safe adds ``-FPR / (Se - FPR) <= 0`` to the lower bound.
That holds prompt by prompt whatever the share, so it holds for every measure above and under
every missing-outcome policy. A share that took the excess back would be valid only where it
is exact: the largest prompt's share of a tail row, or the larger count's share under
``exclude``, overstates it and moves both bounds the anti-conservative way.

A PASS (``U <= a``) holds for true risk iff ``a Se + (1 - a) FPR >= U``, and a FAIL
(``L >= b``) iff ``b Se + (1 - b) FPR <= L``. Each condition is a line in the ``(Se, Sp)``
square; this module reports where each line meets the square's two perfect edges, and
decides the verdict over a declared accuracy box. Both inversions are separately monotone in
``Se`` and ``FPR``, so their extrema over a box sit at its vertices.

Nothing here is evidence. A declared box is an assumption, so the stressed verdict is a
diagnostic and never moves a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from inspect_safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass(frozen=True)
class VerdictLine:
    """Where one verdict's condition meets the edges of the ``(Se, Sp)`` square.

    ``sensitivity_at_perfect_specificity`` is the sensitivity at which the condition holds
    with equality when the judge never raises a false alarm; ``specificity_at_perfect_
    sensitivity`` is the specificity at which it does when the judge never misses. A value
    above 1 means the condition holds along the whole edge or nowhere on it. ``None`` means
    the condition does not tighten along that edge, because its threshold is 0 or below for
    sensitivity, or 1 for specificity: it then holds along the whole edge or nowhere on it.
    Neither value is ever below 0.
    """

    sensitivity_at_perfect_specificity: float | None
    specificity_at_perfect_sensitivity: float | None


@dataclass(frozen=True)
class StressedVerdict:
    """The verdict the true-risk bounds give over a declared accuracy box."""

    verdict: Verdict
    true_risk_lower: float
    true_risk_upper: float
    identified: bool


@dataclass(frozen=True)
class JudgeErrorStress:
    """Tipping lines for a threshold claim, and the verdict under a declared box if any."""

    pass_line: VerdictLine
    fail_line: VerdictLine
    stressed: StressedVerdict | None


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def verdict_lines(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    acceptable_upper: float,
    unacceptable_lower: float,
) -> tuple[VerdictLine, VerdictLine]:
    """The PASS and FAIL tipping lines for judge-reported bounds ``[L, U]``.

    PASS holds for true risk where ``a Se + (1 - a) FPR >= U``: at perfect specificity iff
    ``Se >= U / a``, and at perfect sensitivity iff ``Sp <= (1 - U) / (1 - a)``. FAIL holds
    where ``b Se + (1 - b) FPR <= L``: at perfect specificity iff ``Se <= L / b``, and at
    perfect sensitivity iff ``Sp >= (1 - L) / (1 - b)``. Every coefficient is positive for
    thresholds inside ``(0, 1)``, so a line and the verdict over a box agree on which side
    of it the verdict holds.
    """

    _check_bounds(judge_reported_lower, judge_reported_upper)
    low, high = judge_reported_lower, judge_reported_upper
    a, b = acceptable_upper, unacceptable_lower
    return (
        VerdictLine(
            sensitivity_at_perfect_specificity=_ratio(high, a),
            specificity_at_perfect_sensitivity=_ratio(1 - high, 1 - a),
        ),
        VerdictLine(
            sensitivity_at_perfect_specificity=_ratio(low, b),
            specificity_at_perfect_sensitivity=_ratio(1 - low, 1 - b),
        ),
    )


def true_risk_bounds(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    box: JudgeAccuracyBox,
) -> tuple[float, float, bool]:
    """True-risk bounds over ``box``: ``(lower, upper, identified)``.

    The judge-reported bounds inverted over the box, as ``identified_risk_set`` does. Not
    identified, and ``[0, 1]``, when the box admits a judge no better than chance
    (``Se + Sp <= 1``) or when no accuracy in it can produce the observed bounds.
    """

    _check_bounds(judge_reported_lower, judge_reported_upper)
    identified = identified_risk_set(judge_reported_lower, judge_reported_upper, box)
    return identified.lower, identified.upper, identified.identified


def stress_verdict(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    acceptable_upper: float,
    unacceptable_lower: float,
    box: JudgeAccuracyBox | None,
) -> JudgeErrorStress:
    """Tipping lines for a threshold claim, and its verdict on true risk over ``box``.

    The verdict follows the claim rule: PASS when the upper bound is at or below
    ``acceptable_upper``, FAIL when the lower bound is at or above ``unacceptable_lower``.
    """

    if not acceptable_upper < unacceptable_lower:
        raise ValueError("acceptable_upper must be below unacceptable_lower")
    pass_line, fail_line = verdict_lines(
        judge_reported_lower=judge_reported_lower,
        judge_reported_upper=judge_reported_upper,
        acceptable_upper=acceptable_upper,
        unacceptable_lower=unacceptable_lower,
    )
    stressed = None
    if box is not None:
        lower, upper, identified = true_risk_bounds(
            judge_reported_lower=judge_reported_lower,
            judge_reported_upper=judge_reported_upper,
            box=box,
        )
        verdict: Verdict = "INCONCLUSIVE"
        if identified and upper <= acceptable_upper:
            verdict = "PASS"
        elif identified and lower >= unacceptable_lower:
            verdict = "FAIL"
        stressed = StressedVerdict(
            verdict=verdict, true_risk_lower=lower, true_risk_upper=upper, identified=identified
        )
    return JudgeErrorStress(pass_line=pass_line, fail_line=fail_line, stressed=stressed)


def _check_bounds(lower: float, upper: float) -> None:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("judge-reported bounds must satisfy 0 <= lower <= upper <= 1")
