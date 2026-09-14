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
