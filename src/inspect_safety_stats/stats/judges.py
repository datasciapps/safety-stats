"""Bookkeeping for panels of binary judges: composites, effective votes, agreement.

Nothing here corrects a risk estimate. These functions make a judge panel's
structure explicit -- which verdicts a composite can actually reach, how many
independent opinions the panel carries under its declared dependence blocks, and
how often judges agree -- so that agreement is never mistaken for correctness.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

CompositeRule = Literal["any", "all", "majority"]
_RULES = {"any", "all", "majority"}


def compose_verdicts(verdicts: Mapping[str, bool | None], rule: str) -> bool | None:
    """Combine binary judge verdicts under ``rule``.

    ``None`` means the rule cannot be decided from the votes present: a missing
    vote that could still change the outcome, or a tie under ``majority``. A
    composite is therefore never more decisive than its inputs allow, and an
    undecided composite flows through the ordinary unclassified-outcome
    handling downstream.
    """

    if not verdicts:
        raise ValueError("a composite needs at least one judge verdict")
    if rule not in _RULES:
        raise ValueError(f"unsupported composite rule {rule!r}")
    values = list(verdicts.values())
    positives = sum(value is True for value in values)
    negatives = sum(value is False for value in values)
    missing = len(values) - positives - negatives
    if rule == "any":
        if positives:
            return True
        return None if missing else False
    if rule == "all":
        if negatives:
            return False
        return None if missing else True
    half = len(values) / 2
    if positives > half:
        return True
    if negatives > half:
        return False
    return None


def effective_votes(blocks: Sequence[str]) -> float:
    """Kish effective number of independent votes over declared dependence blocks.

    Each judge casts one vote; judges sharing a block are one opinion. With block
    sizes ``s_b`` the effective count is ``(sum s_b)^2 / sum s_b^2``: all judges
    independent gives the panel size, all in one block gives one.
    """

    if not blocks:
        raise ValueError("effective votes need at least one judge")
    sizes = Counter(blocks).values()
    total = sum(sizes)
    return float(total * total / sum(size * size for size in sizes))


@dataclass(frozen=True)
class AgreementSummary:
    """Descriptive agreement between two judges over the rollouts both classified."""

    compared: int
    disagreements: int
    agreement: float | None
    kappa: float | None


def pairwise_agreement(
    first: Sequence[bool | None], second: Sequence[bool | None]
) -> AgreementSummary:
    """Raw agreement and Cohen's kappa over rollouts where both judges gave a verdict.

    Kappa is undefined (``None``) when the expected agreement is one, i.e. when
    neither judge shows any variation. These are diagnostics: two judges that
    share a blind spot agree perfectly and are still both wrong.
    """

    if len(first) != len(second):
        raise ValueError("both judges must cover the same rollouts")
    pairs = [(a, b) for a, b in zip(first, second, strict=True) if a is not None and b is not None]
    compared = len(pairs)
    if compared == 0:
        return AgreementSummary(compared=0, disagreements=0, agreement=None, kappa=None)
    agree = sum(a == b for a, b in pairs)
    disagreements = compared - agree
    observed = agree / compared
    first_positive = sum(a for a, _ in pairs) / compared
    second_positive = sum(b for _, b in pairs) / compared
    expected = first_positive * second_positive + (1 - first_positive) * (1 - second_positive)
    kappa = None if expected >= 1.0 else (observed - expected) / (1 - expected)
    return AgreementSummary(
        compared=compared, disagreements=disagreements, agreement=observed, kappa=kappa
    )


@dataclass(frozen=True)
class JudgeAccuracyBox:
    """Declared ranges for a judge's sensitivity ``P(flag | unsafe)`` and specificity
    ``P(clear | safe)``.

    Unlike the posterior accuracies ``rho_0, rho_1`` used for a single judge, these are
    prevalence-free, which is what lets boxes for several judges be combined.
    """

    sensitivity_lower: float
    sensitivity_upper: float
    specificity_lower: float
    specificity_upper: float

    def __post_init__(self) -> None:
        for name, low, high in (
            ("sensitivity", self.sensitivity_lower, self.sensitivity_upper),
            ("specificity", self.specificity_lower, self.specificity_upper),
        ):
            if not 0 <= low <= high <= 1:
                raise ValueError(f"{name} must satisfy 0 <= lower <= upper <= 1")


def _conjunction(intervals: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Frechet bounds on ``P(all events)`` from each event's probability interval.

    Valid under arbitrary dependence: the lower bound is attained when the events
    overlap as little as possible, the upper when the rarest event implies the rest.
    """

    lower = max(0.0, sum(low for low, _ in intervals) - (len(intervals) - 1))
    upper = min(high for _, high in intervals)
    return lower, min(upper, 1.0)


def _product(intervals: Sequence[tuple[float, float]]) -> tuple[float, float]:
    lower = upper = 1.0
    for low, high in intervals:
        lower *= low
        upper *= high
    return lower, upper


def composite_accuracy_box(
    judges: Sequence[JudgeAccuracyBox], blocks: Sequence[str], rule: str
) -> JudgeAccuracyBox:
    """Accuracy box of an ``any`` or ``all`` composite from its judges' boxes.

    Within a declared dependence block the judges' errors may be arbitrarily
    dependent, so the block's joint behaviour is bounded by Frechet inequalities;
    across blocks the judges are independent, so block probabilities multiply. Every
    step is monotone interval arithmetic, so the result is sharp for the declared
    structure and widens as the declared boxes widen. ``majority`` has no closed form
    under partial dependence and is refused; calibrate such a composite directly.
    """

    if len(judges) != len(blocks):
        raise ValueError("judge boxes and blocks must be aligned")
    if not judges:
        raise ValueError("a composite needs at least one judge")
    if rule == "majority":
        raise ValueError(
            "majority composites have no closed-form accuracy box under partial dependence; "
            "calibrate the composite's own verdict instead"
        )
    if rule not in {"any", "all"}:
        raise ValueError(f"unsupported composite rule {rule!r}")
    grouped: dict[str, list[JudgeAccuracyBox]] = {}
    for judge, block in zip(judges, blocks, strict=True):
        grouped.setdefault(block, []).append(judge)
    if rule == "any":
        # Sensitivity is 1 - P(every judge clears | unsafe); specificity is
        # P(every judge clears | safe).
        clear_given_unsafe = _product(
            [
                _conjunction([(1 - j.sensitivity_upper, 1 - j.sensitivity_lower) for j in members])
                for members in grouped.values()
            ]
        )
        clear_given_safe = _product(
            [
                _conjunction([(j.specificity_lower, j.specificity_upper) for j in members])
                for members in grouped.values()
            ]
        )
        return JudgeAccuracyBox(
            sensitivity_lower=1 - clear_given_unsafe[1],
            sensitivity_upper=1 - clear_given_unsafe[0],
            specificity_lower=clear_given_safe[0],
            specificity_upper=clear_given_safe[1],
        )
    # all: sensitivity is P(every judge flags | unsafe); specificity is 1 - P(every flags | safe).
    flag_given_unsafe = _product(
        [
            _conjunction([(j.sensitivity_lower, j.sensitivity_upper) for j in members])
            for members in grouped.values()
        ]
    )
    flag_given_safe = _product(
        [
            _conjunction([(1 - j.specificity_upper, 1 - j.specificity_lower) for j in members])
            for members in grouped.values()
        ]
    )
    return JudgeAccuracyBox(
        sensitivity_lower=flag_given_unsafe[0],
        sensitivity_upper=flag_given_unsafe[1],
        specificity_lower=1 - flag_given_safe[1],
        specificity_upper=1 - flag_given_safe[0],
    )


@dataclass(frozen=True)
class IdentifiedRiskSet:
    """The set of true risks consistent with a judge-reported interval and an accuracy box."""

    lower: float
    upper: float
    identified: bool


def identified_risk_set(
    observed_lower: float, observed_upper: float, box: JudgeAccuracyBox
) -> IdentifiedRiskSet:
    """Partial identification of true risk from judge-reported risk under an accuracy box.

    With judge-reported rate ``q``, sensitivity ``Se`` and specificity ``Sp``,
    ``q = pi * Se + (1 - pi) * (1 - Sp)``, so ``pi = (q - (1 - Sp)) / (Se + Sp - 1)``
    (Rogan-Gladen). The inversion is separately monotone in each argument, so its
    extrema over the box are attained at vertices. If the box admits
    ``Se + Sp <= 1`` the panel may be uninformative and nothing identifies ``pi``:
    the honest answer is the whole unit interval, and ``identified`` is False.
    """

    if not 0 <= observed_lower <= observed_upper <= 1:
        raise ValueError("observed bounds must satisfy 0 <= lower <= upper <= 1")
    if box.sensitivity_lower + box.specificity_lower <= 1:
        return IdentifiedRiskSet(lower=0.0, upper=1.0, identified=False)
    vertices = [
        (q - (1 - sp)) / (se + sp - 1)
        for q in (observed_lower, observed_upper)
        for se in (box.sensitivity_lower, box.sensitivity_upper)
        for sp in (box.specificity_lower, box.specificity_upper)
    ]
    return IdentifiedRiskSet(
        lower=max(0.0, min(vertices)), upper=min(1.0, max(vertices)), identified=True
    )
