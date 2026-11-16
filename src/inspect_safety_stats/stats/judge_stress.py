"""Judge-error stress for threshold verdicts: where a verdict tips, and whether it survives.

A threshold claim decided on judge-reported risk says nothing about how good the judge has
to be for the verdict to hold for true risk. Under the prevalence-free model, a judge with
sensitivity ``Se`` and false-positive rate ``FPR = 1 - Sp`` reports failures at

    q = pi * Se + (1 - pi) * FPR

for true risk ``pi``, applied per prompt. The map is affine and increasing in ``pi`` for an
informative judge (``Se > FPR``), so it carries over to every positively homogeneous,
translation-equivariant measure: weighted mean, maximum, CVaR and top-k mean.

Rollouts the judge never classified carry a share ``s`` of a row's weight. The bounds
complete them as judge-safe at the lower end and judge-unsafe at the upper; here they keep
that completion on the true-risk scale (truly safe at the lower end, truly unsafe at the
upper), and only the classified mass is inverted. With judge-reported bounds ``[L, U]``:

    true upper = s + (U - s - FPR (1 - s)) / (Se - FPR)
    true lower =     (L     - FPR (1 - s)) / (Se - FPR)

A PASS (``U <= a``) holds for true risk iff ``(a - s) Se + (1 - a) FPR >= U - s``, and a
FAIL (``L >= b``) iff ``b Se + (1 - s - b) FPR <= L``. Each condition is a line in the
``(Se, Sp)`` square; this module reports where each line meets the square's two perfect
edges, and decides the verdict over a declared accuracy box. Both inversions are separately
monotone in ``Se`` and ``FPR``, so their extrema over a box sit at its vertices.

Nothing here is evidence. A declared box is an assumption, so the stressed verdict is a
diagnostic and never moves a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from inspect_safety_stats.stats.judges import JudgeAccuracyBox

Verdict = Literal["PASS", "FAIL", "INCONCLUSIVE"]


@dataclass(frozen=True)
class VerdictLine:
    """Where one verdict's condition meets the edges of the ``(Se, Sp)`` square.

    ``sensitivity_at_perfect_specificity`` is the sensitivity at which the condition holds
    with equality when the judge never raises a false alarm; ``specificity_at_perfect_
    sensitivity`` is the specificity at which it does when the judge never misses. A value
    outside ``[0, 1]`` means the condition holds along the whole edge or nowhere on it;
    ``None`` means the line never meets that edge (its coefficient there is not positive).
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
    unclassified_share: float = 0.0,
) -> tuple[VerdictLine, VerdictLine]:
    """The PASS and FAIL tipping lines for judge-reported bounds ``[L, U]``.

    PASS holds for true risk where ``(a - s) Se + (1 - a) FPR >= U - s``: at perfect
    specificity iff ``Se >= (U - s) / (a - s)``, and at perfect sensitivity iff
    ``Sp <= (1 - U) / (1 - a)``. FAIL holds where ``b Se + (1 - s - b) FPR <= L``: at
    perfect specificity iff ``Se <= L / b``, and at perfect sensitivity iff
    ``Sp >= 1 - (L - b) / (1 - s - b)``.
    """

    _check_bounds(judge_reported_lower, judge_reported_upper, unclassified_share)
    s, low, high = unclassified_share, judge_reported_lower, judge_reported_upper
    a, b = acceptable_upper, unacceptable_lower
    pass_fpr = _ratio(high - a, 1 - a)
    fail_fpr = _ratio(low - b, 1 - s - b)
    return (
        VerdictLine(
            sensitivity_at_perfect_specificity=_ratio(high - s, a - s),
            specificity_at_perfect_sensitivity=None if pass_fpr is None else 1 - pass_fpr,
        ),
        VerdictLine(
            sensitivity_at_perfect_specificity=_ratio(low, b),
            specificity_at_perfect_sensitivity=None if fail_fpr is None else 1 - fail_fpr,
        ),
    )


def true_risk_bounds(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    box: JudgeAccuracyBox,
    unclassified_share: float = 0.0,
) -> tuple[float, float, bool]:
    """True-risk bounds over ``box``: ``(lower, upper, identified)``.

    Not identified, and ``[0, 1]``, when the box admits a judge no better than chance
    (``Se + Sp <= 1``) or when no accuracy in it can produce the observed bounds.
    """

    _check_bounds(judge_reported_lower, judge_reported_upper, unclassified_share)
    if box.sensitivity_lower + box.specificity_lower <= 1:
        return 0.0, 1.0, False
    s = unclassified_share
    # The classified judge-unsafe mass cannot exceed the classified share; as in
    # correct_judge_risk, a tail row's share is its largest prompt's, which may exceed U.
    mass_lower = min(judge_reported_lower, 1 - s)
    mass_upper = max(mass_lower, judge_reported_upper - s)
    corners = [
        (se, 1 - sp)
        for se in (box.sensitivity_lower, box.sensitivity_upper)
        for sp in (box.specificity_lower, box.specificity_upper)
    ]
    lower = min((mass_lower - fpr * (1 - s)) / (se - fpr) for se, fpr in corners)
    upper = max(s + (mass_upper - fpr * (1 - s)) / (se - fpr) for se, fpr in corners)
    if upper < 0 or lower > 1:
        return 0.0, 1.0, False
    return max(0.0, lower), min(1.0, upper), True


def stress_verdict(
    *,
    judge_reported_lower: float,
    judge_reported_upper: float,
    acceptable_upper: float,
    unacceptable_lower: float,
    box: JudgeAccuracyBox | None,
    unclassified_share: float = 0.0,
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
        unclassified_share=unclassified_share,
    )
    stressed = None
    if box is not None:
        lower, upper, identified = true_risk_bounds(
            judge_reported_lower=judge_reported_lower,
            judge_reported_upper=judge_reported_upper,
            box=box,
            unclassified_share=unclassified_share,
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


def _check_bounds(lower: float, upper: float, share: float) -> None:
    if not 0 <= lower <= upper <= 1:
        raise ValueError("judge-reported bounds must satisfy 0 <= lower <= upper <= 1")
    if not 0 <= share <= 1:
        raise ValueError("the unclassified share must be in [0, 1]")
