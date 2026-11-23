"""Whether the rollouts per prompt can make tail and threshold measures informative."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from safety_stats.analysis.measures import _risk_value, _thresholds
from safety_stats.config.models import AnalysisPlan, RiskMeasureSpec
from safety_stats.ingest.records import NormalizedData
from safety_stats.schema.reports import Diagnostic, PromptEstimate
from safety_stats.schema.scopes import cohort_scope, plan_scope
from safety_stats.stats.binomial import binomial_upper_bound

_THRESHOLD_TYPES = frozenset({"threshold_exceedance", "threshold_excess"})
_PROPAGATED_TYPES = frozenset({"maximum", "cvar", "top_k_mean", "entropic", *_THRESHOLD_TYPES})
# The measures nonlinear in each prompt's failure probability, which docs/statistical-methods.md
# (section 2.4.9) calls misleading or vacuous when each prompt has one rollout, whose empirical
# rate is then 0 or 1: at one rollout a top-k mean is an equal-weight CVaR, and a threshold
# measure counts each prompt that failed its one trial as above any threshold below 1. The
# weighted mean is linear, and worst_of_k gives no estimate below k rollouts per prompt and is
# the weighted mean at k = 1.
_SINGLE_ROLLOUT_TYPES = _PROPAGATED_TYPES


def single_rollout_tail_claims(plan: AnalysisPlan, data: NormalizedData) -> list[Diagnostic]:
    """Degrade integrity when a claim rests on a nonlinear measure of single-rollout prompts.

    For each claim on a ``cvar``, ``maximum``, ``top_k_mean``, ``entropic``,
    ``threshold_exceedance`` or ``threshold_excess`` measure, the classified rollouts of each
    prompt in the claim's cohort and group are counted. When the lower median count, over the
    prompts with any, is one, so that at least half of those prompts have a single rollout,
    their rates are 0 or 1 and the plug-in measure is driven by single trials, so the claim
    rests on an assumption the logs do not support. The lower median flags an even split, the
    conservative reading. Data only: ``validate`` reports it as ``analyze`` does.
    """

    measures = {measure.id: measure for measure in plan.outputs.risk_measures}
    groups = {group.id: group for group in data.groups}
    classified = Counter(
        (item.cohort_id, item.prompt_id) for item in data.observations if item.failure is not None
    )
    found: list[Diagnostic] = []
    for claim in plan.claims:
        measure = measures.get(claim.measure or "")
        group = groups.get(claim.group or "")
        cohort = claim.cohort
        # A claim without a measure, group or cohort is reported by the plan checks.
        if measure is None or group is None or cohort is None:
            continue
        if measure.type not in _SINGLE_ROLLOUT_TYPES:
            continue
        counts = [
            count
            for count in (classified[(cohort, prompt)] for prompt in group.prompt_ids)
            if count > 0
        ]
        if not counts or _lower_median(counts) > 1:
            continue
        single = sum(count == 1 for count in counts)
        found.append(
            Diagnostic(
                code="TAIL_MEASURE_SINGLE_ROLLOUT",
                severity="WARNING",
                message=(
                    f"claim {claim.id!r} uses {measure.type} measure {measure.id!r}, but "
                    f"{single} of the {len(counts)} prompts with a classified rollout in "
                    f"cohort {claim.cohort!r}, group {group.id!r} have only one, so their "
                    "rates are 0 or 1 and the measure is misleading or vacuous; run "
                    "repeated rollouts per prompt, or claim on a weighted mean."
                ),
                scope=plan_scope("claims", claim.id),
                details={
                    "integrity_degraded": True,
                    "measure": measure.id,
                    "measure_type": measure.type,
                    "cohort": claim.cohort,
                    "group": group.id,
                    "prompts_with_rollouts": len(counts),
                    "single_rollout_prompts": single,
                    "median_rollouts_per_prompt": _lower_median(counts),
                },
            )
        )
    return found


def _propagated(measure: RiskMeasureSpec) -> bool:
    """Whether a measure's interval comes only from the per-prompt bounds."""

    if measure.type == "worst_of_k":
        return not measure.direct_interval
    return measure.type in _PROPAGATED_TYPES


# The most rollouts per prompt the search for the count a measure needs tries.
_MOST_ROLLOUTS = 10**9


def _lower_median(counts: Sequence[int]) -> int:
    """The lower median of nonempty ``counts``: at least half of them are at most it."""

    ordered = sorted(counts)
    return ordered[(len(ordered) - 1) // 2]


def _zero_failure_width(
    measure: RiskMeasureSpec,
    weights: list[float],
    threshold: float | np.ndarray | None,
    alpha: float,
) -> Callable[[int], float]:
    """How far above its estimate a measure's upper bound lies when no prompt fails.

    With ``n`` rollouts and no failure on any prompt, each prompt's estimate is 0 and its
    upper bound is ``binomial_upper_bound(0, n, alpha)``. The measure's estimate is then its
    value with every prompt at 0, and its propagated upper bound its value with every prompt at
    that bound; the returned function gives the difference at each ``n``. The difference never
    grows with ``n``: the bound falls, and every propagated measure is monotone in each
    prompt's value. A measure that cannot be computed on these weights or thresholds raises
    ``ValueError``.
    """

    zeros = [0.0] * len(weights)
    estimate = _risk_value(zeros, weights, measure, threshold=threshold)

    def width(rollouts: int) -> float:
        bound = binomial_upper_bound(0, rollouts, alpha=alpha)
        uppers = [bound] * len(weights)
        return _risk_value(uppers, weights, measure, threshold=threshold) - estimate

    return width


def _rollouts_needed(width: Callable[[int], float], target: float, rollouts: int) -> int | None:
    """The fewest rollouts per prompt above ``rollouts`` at which ``width`` meets ``target``.

    ``width(rollouts)`` exceeds ``target``, and ``width`` never grows. None when not even
    ``_MOST_ROLLOUTS`` meets it, as for a threshold of zero, which every bound exceeds.
    """

    if width(_MOST_ROLLOUTS) > target:
        return None
    # Double until the target is met, then bisect: low misses it and high meets it.
    low, high = rollouts, min(2 * rollouts, _MOST_ROLLOUTS)
    while width(high) > target:
        low, high = high, min(2 * high, _MOST_ROLLOUTS)
    while high - low > 1:
        middle = (low + high) // 2
        if width(middle) > target:
            low = middle
        else:
            high = middle
    return high


def _names(measures: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""

    if len(measures) == 1:
        return measures[0]
    return ", ".join(measures[:-1]) + " and " + measures[-1]


def _needed_text(needed: dict[str, int | None]) -> str:
    """The sentence that says how many rollouts per prompt each named measure needs."""

    counts = sorted(set(needed.values()), key=lambda count: (count is None, count or 0))
    if len(counts) == 1:
        (count,) = counts
        if count is None:
            return "No number of rollouts per prompt up to 10^9 would be enough."
        return f"About {count} rollouts per prompt would be needed."
    parts = [
        ("about " + str(count) if count is not None else "none up to 10^9")
        + " for "
        + _names([measure for measure, value in needed.items() if value == count])
        for count in counts
    ]
    return "Rollouts per prompt needed: " + "; ".join(parts) + "."


def tail_measure_precision(
    plan: AnalysisPlan,
    data: NormalizedData,
    estimates: list[PromptEstimate],
    diagnostics: list[Diagnostic],
) -> None:
    """Warn when propagated measures cannot be informative at this rollout count.

    Tail and threshold measures are built from simultaneous per-prompt bounds, so their
    precision is set by the rollouts on each prompt, not the total. For each cohort and group,
    over the prompts whose upper bound certifies, ``n`` is the lower median of their classified
    rollouts and ``alpha`` the smallest of their upper bounds' levels. A prompt with no failures
    in ``n`` rollouts has the Clopper-Pearson upper bound ``1 - alpha ** (1 / n)``. Were every
    prompt of the group such a prompt, each measure's estimate would be its value with every
    prompt at 0 and its upper bound its value with every prompt at that bound. When the
    difference exceeds ``report.precision_target``, the measure can only be wide there, and one
    warning per cohort and group names each such measure with the fewest rollouts per prompt
    that would bring the difference within the target. For a maximum, CVaR, top-k mean or
    entropic risk the difference is the prompt's bound itself; for ``worst_of_k`` it is
    ``1 - (1 - bound) ** k``; a threshold measure widens only through the prompts whose
    threshold lies below the bound. The warning leaves integrity as it is.
    """

    measures = sorted(
        (measure for measure in plan.outputs.risk_measures if _propagated(measure)),
        key=lambda measure: measure.id,
    )
    if not measures:
        return
    target = plan.report.precision_target
    index = {(item.cohort_id, item.prompt_id): item for item in estimates}
    for cohort in sorted(plan.cohorts):
        for group in sorted(data.groups, key=lambda item: item.id):
            # A measure built from the prompt bounds gets an interval only when every prompt of
            # the group has a certifying bound (complete_bounds in risks.py). One withheld or
            # non-certifying bound leaves the measure without an interval, so there is no width
            # to warn about.
            results = [index.get((cohort, prompt)) for prompt in group.prompt_ids]
            complete = [
                item
                for item in results
                if item is not None
                and item.eligibility == "certifying"
                and item.lower is not None
                and item.upper is not None
            ]
            if not results or len(complete) < len(results):
                continue
            bounded = [
                item for item in complete if item.alpha_upper is not None and item.n_valid > 0
            ]
            if not bounded:
                continue
            rollouts = _lower_median([item.n_valid for item in bounded])
            alpha = min(item.alpha_upper for item in bounded if item.alpha_upper is not None)
            weights = [group.weights[prompt] for prompt in group.prompt_ids]
            needed: dict[str, int | None] = {}
            for measure in measures:
                try:
                    threshold = (
                        _thresholds(measure, list(group.prompt_ids), data)
                        if measure.type in _THRESHOLD_TYPES
                        else None
                    )
                    width = _zero_failure_width(measure, weights, threshold, alpha)
                    wide = width(rollouts) > target
                except ValueError:
                    # The group's risks report the measure as RISK_MEASURE_INVALID.
                    continue
                if wide:
                    needed[measure.id] = _rollouts_needed(width, target, rollouts)
            if not needed:
                continue
            upper = binomial_upper_bound(0, rollouts, alpha=alpha)
            named = _names(list(needed))
            one = len(needed) == 1
            reachable = [count for count in needed.values() if count is not None]
            diagnostics.append(
                Diagnostic(
                    code="TAIL_MEASURES_UNINFORMATIVE",
                    severity="WARNING",
                    message=(
                        f"In cohort {cohort!r}, group {group.id!r}, {named} "
                        f"{'is' if one else 'are'} built from per-prompt bounds; with about "
                        f"{rollouts} rollouts per prompt a prompt with no failures is still "
                        f"bounded only at {upper:.3f}, so even if no prompt failed "
                        f"{'its upper bound' if one else 'their upper bounds'} would lie more "
                        f"than the precision target {target:g} above "
                        f"{'its estimate' if one else 'their estimates'}. "
                        + _needed_text(needed)
                    ),
                    scope=cohort_scope(cohort, "groups", group.id),
                    details={
                        "cohort": cohort,
                        "group": group.id,
                        "measures": list(needed),
                        "median_rollouts_per_prompt": rollouts,
                        "zero_failure_prompt_upper": upper,
                        "rollouts_per_prompt_needed": (
                            max(reachable) if len(reachable) == len(needed) else None
                        ),
                        "rollouts_per_prompt_needed_by_measure": needed,
                    },
                )
            )
