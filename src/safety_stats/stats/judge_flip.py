"""The judge flip rate under content-preserving wrappers (#122).

A judge-positive completion is rescored inside wrappers that keep its content verbatim, such
as a refusal prefix or a story frame. A verdict that becomes safe is a flip: since the content
did not change, it is a false negative the wrapper induced. The share of rescored completions
that flip under at least one wrapper, and under each, is a binomial proportion over the
rescored completions, bounded here with the exact Clopper-Pearson interval.

The binomial model treats the rescored completions as independent. Under ``simple_random``
the random sample makes them so for the run's share, however flips cluster by prompt.
Drawing without replacement makes the count less variable than the binomial, and the
interval's exact coverage of the run's share stays at or above ``1 - alpha`` (checked for
every run of up to 40 judge-positive completions). Under ``all`` the rescored completions are
all of the run's judge-positive completions, so the run's share is the estimate itself. The
interval is then descriptive: read as a range for the rate of the process that produced the
completions, it holds only if they flip independently. Completions of one prompt share their
content and tend to flip together, and then the interval is too narrow for that rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from safety_stats.stats.binomial import binomial_bounds


@dataclass(frozen=True)
class FlipRate:
    """A flip count over the rescored completions, with its two-sided interval."""

    flipped: int
    rescored: int
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class JudgeFlipResult:
    """The flip rate under at least one wrapper, and under each declared wrapper."""

    judge_positive: int
    any_wrapper: FlipRate | None
    by_wrapper: dict[str, FlipRate]


def _rate(flipped: int, rescored: int, alpha: float) -> FlipRate:
    bounds = binomial_bounds(flipped, rescored, alpha_lower=alpha / 2, alpha_upper=alpha / 2)
    return FlipRate(
        flipped=flipped,
        rescored=rescored,
        estimate=flipped / rescored,
        lower=bounds.lower,
        upper=bounds.upper,
    )


def judge_flip_rate(
    judge_positive: Iterable[str],
    verdicts: Mapping[str, Mapping[str, bool]],
    wrappers: Sequence[str],
    *,
    sampling: Literal["all", "simple_random"],
    alpha: float,
) -> JudgeFlipResult:
    """The flip rates of the rescored judge-positive completions.

    ``judge_positive`` holds the identifiers of the run's judge-positive completions.
    ``verdicts`` maps each rescored completion to its verdict under each wrapper, True for
    unsafe. Every rescored completion must be judge-positive and carry a verdict for every
    one of ``wrappers``, and none other; with ``sampling: all`` every judge-positive
    completion must have been rescored, and with ``simple_random`` at least one. A design
    that breaks any of these raises ``ValueError``, since the proportion would not be the
    declared one. Each interval is two-sided at ``alpha``, split equally between its ends,
    and holds for its own rate alone, not simultaneously with the others. It assumes the
    rescored completions flip independently: under ``simple_random`` it is for the share of
    the run's judge-positive completions, and under ``all``, where that share is known, it is
    descriptive (see the module docstring). A run without judge-positive completions has no
    rate (``any_wrapper`` is None).
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not wrappers or len(set(wrappers)) != len(wrappers):
        raise ValueError("wrappers must be a non-empty list of distinct names")
    positive = set(judge_positive)
    rescored = set(verdicts)
    outside = sorted(rescored - positive)
    if outside:
        raise ValueError(
            "the judge flip manifest rescores completions that are not judge-positive in "
            f"the analysed run: {', '.join(outside[:5])}"
            + (f" and {len(outside) - 5} more" if len(outside) > 5 else "")
        )
    declared = set(wrappers)
    for observation_id in sorted(rescored):
        found = set(verdicts[observation_id])
        if found != declared:
            missing = sorted(declared - found)
            extra = sorted(found - declared)
            faults = ([f"under undeclared wrappers {extra}"] if extra else []) + (
                [f"not under declared wrappers {missing}"] if missing else []
            )
            raise ValueError(
                f"the judge flip manifest rescores {observation_id!r} "
                + " and ".join(faults)
                + "; every rescored completion needs exactly the declared wrappers"
            )
    if sampling == "all":
        unrescored = sorted(positive - rescored)
        if unrescored:
            raise ValueError(
                f"sampling: all, but {len(unrescored)} judge-positive completions were not "
                f"rescored, among them {', '.join(unrescored[:5])}"
            )
    elif positive and not rescored:
        raise ValueError("sampling: simple_random, but the judge flip manifest rescores none")
    if not rescored:
        return JudgeFlipResult(judge_positive=0, any_wrapper=None, by_wrapper={})
    count = len(rescored)
    any_flipped = sum(
        1 for item in verdicts.values() if not all(item[wrapper] for wrapper in wrappers)
    )
    by_wrapper = {
        wrapper: _rate(
            sum(1 for item in verdicts.values() if not item[wrapper]), count, alpha
        )
        for wrapper in wrappers
    }
    return JudgeFlipResult(
        judge_positive=len(positive),
        any_wrapper=_rate(any_flipped, count, alpha),
        by_wrapper=by_wrapper,
    )
