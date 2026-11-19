"""Whether the rollouts per prompt can make tail and threshold measures informative."""

from __future__ import annotations

from collections import Counter

from inspect_safety_stats.config.models import AnalysisPlan, RiskMeasureSpec
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import Diagnostic, PromptEstimate
from inspect_safety_stats.schema.scopes import plan_scope
from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.planning import zero_failure_sample_size

_PROPAGATED_TYPES = frozenset(
    {"maximum", "cvar", "top_k_mean", "entropic", "threshold_exceedance", "threshold_excess"}
)
# The nonlinear measures docs/statistical-methods.md (section 2.4.9) calls misleading or
# vacuous when each prompt has one rollout, whose empirical rate is then 0 or 1.
_SINGLE_ROLLOUT_TYPES = frozenset({"cvar", "maximum", "entropic"})


def single_rollout_tail_claims(plan: AnalysisPlan, data: NormalizedData) -> list[Diagnostic]:
    """Degrade integrity when a claim rests on a nonlinear measure of single-rollout prompts.

    For each claim on a ``cvar``, ``maximum`` or ``entropic`` measure, the classified rollouts
    of each prompt in the claim's cohort and group are counted. When the median count, over
    the prompts with any, is one, every typical per-prompt rate is 0 or 1 and the plug-in
    measure is driven by single trials, so the claim rests on an assumption the logs do not
    support. Data only: ``validate`` reports it as ``analyze`` does.
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
        counts = sorted(
            count
            for count in (classified[(cohort, prompt)] for prompt in group.prompt_ids)
            if count > 0
        )
        if not counts or counts[len(counts) // 2] > 1:
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
                    "median_rollouts_per_prompt": counts[len(counts) // 2],
                },
            )
        )
    return found


def _propagated(measure: RiskMeasureSpec) -> bool:
    """Whether a measure's interval comes only from the per-prompt bounds."""

    if measure.type == "worst_of_k":
        return not measure.direct_interval
    return measure.type in _PROPAGATED_TYPES


def tail_measure_precision(
    plan: AnalysisPlan,
    estimates: list[PromptEstimate],
    diagnostics: list[Diagnostic],
) -> None:
    """Warn when propagated measures cannot be informative at this rollout count.

    Tail and threshold measures are built from simultaneous per-prompt bounds, so their
    precision is set by the rollouts on each prompt, not the total. A prompt with no failures
    and ``n`` rollouts has the Clopper-Pearson upper bound ``1 - alpha ** (1 / n)`` at its
    simultaneous level. When that alone exceeds ``report.precision_target`` at the median
    rollout count, the measures can only be wide. The warning leaves integrity as it is.
    """

    requested = sorted(
        measure.id for measure in plan.outputs.risk_measures if _propagated(measure)
    )
    if not requested:
        return
    usable = [item for item in estimates if item.alpha_upper is not None and item.n_valid > 0]
    if not usable:
        return
    rollouts = sorted(item.n_valid for item in usable)[len(usable) // 2]
    alpha = min(item.alpha_upper for item in usable if item.alpha_upper is not None)
    zero_failure_upper = binomial_upper_bound(0, rollouts, alpha=alpha)
    target = plan.report.precision_target
    if zero_failure_upper <= target:
        return
    needed = zero_failure_sample_size(threshold=target, alpha=alpha)
    diagnostics.append(
        Diagnostic(
            code="TAIL_MEASURES_UNINFORMATIVE",
            severity="WARNING",
            message=(
                f"{', '.join(requested)} are built from per-prompt bounds; with about "
                f"{rollouts} rollouts per prompt a prompt with no failures is still bounded "
                f"only at {zero_failure_upper:.3f}, above the precision target {target:g}. "
                f"About {needed} rollouts per prompt would be needed."
            ),
            scope=plan_scope("outputs", "risk_measures"),
            details={
                "measures": requested,
                "median_rollouts_per_prompt": rollouts,
                "zero_failure_prompt_upper": zero_failure_upper,
                "rollouts_per_prompt_needed": needed,
            },
        )
    )
