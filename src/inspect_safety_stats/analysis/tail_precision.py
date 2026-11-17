"""Whether the rollouts per prompt can make tail and threshold measures informative."""

from __future__ import annotations

from inspect_safety_stats.config.models import AnalysisPlan, RiskMeasureSpec
from inspect_safety_stats.schema.reports import Diagnostic, PromptEstimate
from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.planning import zero_failure_sample_size

_PROPAGATED_TYPES = frozenset(
    {"maximum", "cvar", "top_k_mean", "entropic", "threshold_exceedance", "threshold_excess"}
)


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
            scope="outputs.risk_measures",
            details={
                "measures": requested,
                "median_rollouts_per_prompt": rollouts,
                "zero_failure_prompt_upper": zero_failure_upper,
                "rollouts_per_prompt_needed": needed,
            },
        )
    )
