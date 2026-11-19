"""The share of rollouts a judge never classified, bounded for judge corrections.

A judge correction keeps the rollouts with no judge verdict at their worst case and maps only
the classified mass through rho (:func:`~inspect_safety_stats.stats.misclassification.
correct_judge_risk`). The share those rollouts hold is a parameter of the population the row
estimates, and the run's own share only estimates it: with the run's share in its place, a
corrected upper bound falls below true risk whenever the run saw fewer unclassified rollouts
than the population holds. The correction therefore takes a one-sided upper confidence bound on
the population share, under the row's own independence assumption, and adds the bound's alpha to
its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from inspect_safety_stats.analysis.estimates import _block_index
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import RiskIntervalResult
from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.direct_mean import bounded_kl_chernoff_upper_bound
from inspect_safety_stats.stats.independence import direct_bounded_blocks

# The only policy whose prompt bounds drop an unclassified rollout and keep the prompt. The
# others keep it as an unknown outcome (``conservative_bounds``, ``end_to_end``), classify it
# (``count_as_failure``), or block its prompt (``block``), so that no bound rests on it.
_DROPPING_POLICY = "exclude"

PER_PROMPT_METHOD = "per_prompt_clopper_pearson_bonferroni"
BLOCK_METHOD = "block_kl_chernoff"
UNBOUNDED_METHOD = "not_bounded_sampled_frame"


class PromptCounts(Protocol):
    """A prompt's rollouts in one cohort: those with a judge verdict, and all of them."""

    @property
    def classified(self) -> int: ...

    @property
    def attempted(self) -> int: ...


@dataclass(frozen=True)
class UnclassifiedShare:
    """The share of a row's weight on unclassified rollouts: observed, and bounded.

    ``observed`` is the run's own share, weighted as the row weights its rollouts (for a tail
    measure, the largest of any prompt it weights). ``upper`` bounds the population share at
    one-sided level ``alpha`` by ``method``. ``in_bounds`` says whether the row's judge-system
    bounds complete every unclassified rollout, which decides how the correction places the
    classified mass (see ``correct_judge_risk``).
    """

    observed: float
    upper: float
    alpha: float
    method: str
    in_bounds: bool
    prompts: int = 0
    independence_profile: str | None = None


def bounds_complete_unclassified(plan: AnalysisPlan) -> bool:
    """Whether a row's bounds hold every unclassified rollout they rest on at its worst case.

    ``conservative_bounds`` and ``end_to_end`` keep such a rollout in its prompt's bounds as
    an unknown outcome, and ``count_as_failure`` classifies it. ``block`` blocks its prompt,
    which then has no bounds of its own: a row that still has bounds completes the prompt's
    whole weight, and the judge-system interval, like the correction, describes only runs
    that such a rollout did not block. ``exclude`` drops the rollout from its prompt's
    denominator, so the prompt's bounds cover only the rate among the rest. Direct-mean
    intervals complete every rollout under any policy, but a row's primary interval may be
    either kind, so a plan that excludes is read the conservative way.
    """

    score = plan.inputs.score
    return _DROPPING_POLICY not in {score.missing, score.errors}


def share_alpha(spent: float | None, base: RiskIntervalResult | None) -> float:
    """The one-sided alpha of a row's share bound.

    A measured calibration spends a quarter of its budget on each of its four endpoints, and
    the share is a fifth parameter of the same judge model, bounded at the same level and
    added to the budget. A declared or transported calibration spends nothing, so the share
    is bounded at the row's own one-sided level.
    """

    if spent is not None and spent > 0:
        return spent / 4
    if base is not None and base.directional_family_alpha is not None:
        return base.directional_family_alpha
    if base is not None and base.family_alpha is not None:
        return base.family_alpha / 2
    return 0.025


def _observed_share(
    rollouts: Mapping[str, PromptCounts], weights: list[tuple[str, float]], tail: bool
) -> float:
    shares: list[tuple[float, float]] = []
    for prompt_id, weight in weights:
        item = rollouts.get(prompt_id)
        if item is None or item.attempted == 0:
            continue
        shares.append((weight, (item.attempted - item.classified) / item.attempted))
    if not shares:
        return 0.0
    if tail:
        return max(share for _, share in shares)
    return min(1.0, sum(weight * share for weight, share in shares))


def unclassified_share_bound(
    plan: AnalysisPlan,
    data: NormalizedData,
    rollouts: Mapping[str, PromptCounts],
    *,
    cohort_id: str,
    weights: Mapping[str, float],
    measure_type: str,
    independence_profile: str | None,
    alpha: float,
) -> UnclassifiedShare:
    """Bound the population share of a row's weight on unclassified rollouts.

    A weighted mean whose primary interval assumes a declared independence profile gets the
    KL/Chernoff bound over that profile's independent blocks, each block's unclassified share
    a [0, 1] value: the row's own assumption, and nothing more. Any other row of a fixed design
    gets an exact Clopper-Pearson bound on each prompt's share at ``alpha`` over the number of
    prompts, so that all hold at once by Bonferroni: the weighted sum bounds a mean, and the
    largest bounds a tail measure. That needs only independent rollouts within each prompt,
    as the prompt bounds do. A weighted prompt with no rollout counts as wholly unclassified.

    A sampled design estimates a frame mean, and bounding the frame's share would need the
    design's own estimator, so its share is taken at the largest value, 1. Its corrections are
    sensitivity analyses in every mode.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    in_bounds = bounds_complete_unclassified(plan)
    weighted = [(prompt_id, weight) for prompt_id, weight in weights.items() if weight > 0]
    tail = measure_type != "weighted_mean"
    observed = _observed_share(rollouts, weighted, tail)
    if plan.design.type != "fixed":
        return UnclassifiedShare(observed, 1.0, alpha, UNBOUNDED_METHOD, in_bounds)
    if not weighted:
        return UnclassifiedShare(observed, observed, alpha, PER_PROMPT_METHOD, in_bounds)
    profile = (
        plan.design.independence_profiles.get(independence_profile)
        if independence_profile is not None
        else None
    )
    if not tail and profile is not None:
        try:
            blocks = direct_bounded_blocks(
                data.observations,
                data.manifest,
                profile,
                cohort_id=cohort_id,
                prompt_ids=[prompt_id for prompt_id, _ in weighted],
                prompt_weights=np.asarray([weight for _, weight in weighted], dtype=float),
                index=_block_index(data),
            )
        except ValueError:
            # No blocks can be formed, for example when a prompt has no rollout: fall back
            # to the per-prompt bounds, which need no independence between prompts.
            pass
        else:
            # A block's upper completion exceeds its lower one by its unclassified share.
            block_shares = np.clip(blocks.upper - blocks.lower, 0.0, 1.0)
            estimate = min(1.0, max(0.0, float(np.dot(blocks.weights, block_shares))))
            upper, _ = bounded_kl_chernoff_upper_bound(estimate, blocks.weights, alpha=alpha)
            return UnclassifiedShare(
                observed,
                min(1.0, upper),
                alpha,
                BLOCK_METHOD,
                in_bounds,
                prompts=len(weighted),
                independence_profile=independence_profile,
            )
    endpoint = alpha / len(weighted)
    bounds: list[tuple[float, float]] = []
    for prompt_id, weight in weighted:
        item = rollouts.get(prompt_id)
        if item is None or item.attempted == 0:
            bounds.append((weight, 1.0))
            continue
        unclassified = item.attempted - item.classified
        bounds.append((weight, binomial_upper_bound(unclassified, item.attempted, alpha=endpoint)))
    if tail:
        upper = max(bound for _, bound in bounds)
    else:
        upper = min(1.0, sum(weight * bound for weight, bound in bounds))
    return UnclassifiedShare(
        observed, upper, alpha, PER_PROMPT_METHOD, in_bounds, prompts=len(weighted)
    )


def unclassified_share_text(share: UnclassifiedShare) -> str:
    """One sentence on the row's unclassified share: observed, bounded, and how."""

    if share.method == UNBOUNDED_METHOD:
        how = (
            "a sampled design's frame share is not bounded here, so it is taken at its "
            "largest, 1"
        )
    elif share.method == BLOCK_METHOD:
        how = (
            f"a one-sided KL/Chernoff bound at alpha {share.alpha:.6g} over the independent "
            f"blocks of independence profile {share.independence_profile!r}, the row's own "
            "assumption"
        )
    else:
        how = (
            f"one-sided exact Clopper-Pearson bounds on each of its {share.prompts} prompts' "
            f"shares, at alpha {share.alpha:.6g} in all by Bonferroni, which assume "
            "independent rollouts within each prompt and none between prompts"
        )
    return (
        f"Unclassified rollouts hold {share.observed:.6g} of the row's weight in this run and "
        f"at most {share.upper:.6g} in the population: {how}."
    )


def unclassified_assumption(share: UnclassifiedShare) -> str:
    """The assumption a corrected interval states about its unclassified rollouts."""

    text = (
        f"{unclassified_share_text(share)} They keep their completion on the true-risk scale, "
        "truly safe at the lower bound and truly unsafe at the upper, and only the classified "
        "mass is mapped through rho."
    )
    if not share.in_bounds:
        text += (
            " The plan's missing or error policy excludes some of them from the judge-system "
            "bounds, which then cover only the classified rollouts' rate, so that rate is "
            "corrected on its own and the unclassified share is mixed in at its worst case."
        )
    return text
