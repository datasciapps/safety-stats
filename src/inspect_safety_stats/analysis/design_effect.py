"""How much the grouping of rollouts into an independence profile's blocks shrinks their evidence.

A profile that blocks by a column above the item, such as a conversation or a shared context,
puts rollouts of several prompts in one block. When those rollouts tend to fail together, a
profile that treats them as independent overstates how much evidence they carry. For each such
profile and cohort, this module estimates the empirical intra-cluster correlation of the scored
rollouts' outcomes within the profile's blocks, the design effect it implies and the effective
number of rollouts, so that a user sees how much the choice between the profiles matters.

The figures describe the data and change no bound: the bounds still rest on the profiles the
plan declares.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from inspect_safety_stats.config.models import AnalysisPlan, IndependenceProfileSpec
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import plan_scope
from inspect_safety_stats.stats.independence import _serialized_key, block_index, block_key

# A design effect of 2 halves the evidence an item-level profile counts on. At or above it, a
# primary profile that splits the blocks makes CLUSTER_DESIGN_EFFECT a warning.
DESIGN_EFFECT_WARNING = 2.0


def intracluster_correlation(sizes: Sequence[int], failures: Sequence[int]) -> float | None:
    """The one-way ANOVA estimate of the intra-cluster correlation of binary outcomes.

    ``sizes`` and ``failures`` give each cluster's outcome count and failure count. With
    ``N`` outcomes in ``G`` clusters of sizes ``m_g`` and failure rates ``p_g`` around the
    overall rate ``p``, the between- and within-cluster mean squares are
    ``MSB = sum m_g (p_g - p)^2 / (G - 1)`` and ``MSW = sum m_g p_g (1 - p_g) / (N - G)``, and
    the estimate is ``(MSB - MSW) / (MSB + (m0 - 1) MSW)`` with
    ``m0 = (N - sum m_g^2 / N) / (G - 1)``. It can be negative. None when it is undefined:
    fewer than two clusters, no cluster with two outcomes, or no variation in the outcomes.
    """

    if len(sizes) != len(failures):
        raise ValueError("cluster sizes and failure counts must be aligned")
    if any(size < 1 for size in sizes) or any(
        failed < 0 or failed > size for size, failed in zip(sizes, failures, strict=True)
    ):
        raise ValueError("every cluster needs an outcome, and at most one failure per outcome")
    total = sum(sizes)
    clusters = len(sizes)
    if clusters < 2 or total == clusters:
        return None
    overall = sum(failures) / total
    between = math.fsum(
        size * (failed / size - overall) ** 2
        for size, failed in zip(sizes, failures, strict=True)
    ) / (clusters - 1)
    within = math.fsum(
        failed * (1 - failed / size) for size, failed in zip(sizes, failures, strict=True)
    ) / (total - clusters)
    typical = (total - math.fsum(size * size for size in sizes) / total) / (clusters - 1)
    denominator = between + (typical - 1) * within
    if denominator <= 0:
        return None
    return (between - within) / denominator


def design_effect(sizes: Sequence[int], correlation: float) -> float:
    """``1 + (m - 1) * max(correlation, 0)``, with ``m = sum m_g^2 / N``.

    ``m`` is the size of the cluster an outcome sits in, averaged over outcomes, which is the
    cluster size of the classical approximation when clusters differ in size. A negative
    correlation is read as no clustering, so the design effect is never below 1.
    """

    total = sum(sizes)
    typical = math.fsum(size * size for size in sizes) / total
    return 1 + (typical - 1) * max(correlation, 0.0)


def _primary_profiles(plan: AnalysisPlan) -> list[str]:
    """Every profile the plan names as the primary of a result, sorted."""

    selections = [
        plan.outputs.direct_mean_intervals.independence,
        *(item.independence for item in plan.comparisons),
        *(item.independence for item in plan.comparison_sets),
    ]
    primaries = {selection.primary for selection in selections if selection is not None}
    if plan.outputs.anytime_bounds is not None:
        primaries.add(plan.outputs.anytime_bounds.independence_profile)
    return sorted(primaries)


def _cohort_figures(
    rows: list[Observation],
    profile: IndependenceProfileSpec,
    primaries: dict[str, IndependenceProfileSpec],
    manifest_values: Any,
) -> tuple[dict[str, Any], bool, list[str]]:
    """One cohort's figures, whether a block holds several prompts, and the primaries that
    split a block."""

    blocks: dict[tuple[Any, ...], list[int]] = {}
    prompts: dict[tuple[Any, ...], set[str]] = {}
    primary_blocks: dict[str, dict[tuple[Any, ...], set[str]]] = {name: {} for name in primaries}
    for row in rows:
        key = block_key(row, profile, manifest_values, comparison=False)
        counts = blocks.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += row.failure is True
        prompts.setdefault(key, set()).add(row.prompt_id)
        for name, spec in primaries.items():
            inner = _serialized_key(block_key(row, spec, manifest_values, comparison=False))
            primary_blocks[name].setdefault(key, set()).add(inner)
    ordered = sorted(blocks.items(), key=lambda item: _serialized_key(item[0]))
    sizes = [counts[0] for _, counts in ordered]
    failures = [counts[1] for _, counts in ordered]
    correlation = intracluster_correlation(sizes, failures)
    effect = None if correlation is None else design_effect(sizes, correlation)
    if correlation is None and len(sizes) > 1 and max(sizes) == 1:
        effect = 1.0
    figures = {
        "rollouts": len(rows),
        "blocks": len(sizes),
        "largest_block": max(sizes),
        "intracluster_correlation": correlation,
        "design_effect": effect,
        "effective_rollouts": None if effect is None else len(rows) / effect,
    }
    grouped = any(len(names) > 1 for names in prompts.values())
    splitting = sorted(
        name
        for name, inner in primary_blocks.items()
        if any(len(keys) > 1 for keys in inner.values())
    )
    return figures, grouped, splitting


def _describe(cohort_id: str, figures: dict[str, Any]) -> str:
    head = (
        f"cohort {cohort_id!r}: {figures['rollouts']} scored rollouts in "
        f"{figures['blocks']} blocks"
    )
    if figures["blocks"] == 1:
        return f"{head}, so no correlation between blocks can be estimated"
    if figures["intracluster_correlation"] is None:
        return f"{head}, all with the same outcome, so no correlation can be estimated"
    return (
        f"{head}, intra-cluster correlation {figures['intracluster_correlation']:.3f}, "
        f"design effect {figures['design_effect']:.2f}, about "
        f"{figures['effective_rollouts']:.0f} effective rollouts"
    )


def cluster_design_effects(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> None:
    """Report the design effect of each profile whose blocks group rollouts of several prompts.

    For every profile of ``design.independence_profiles`` and every cohort, the scored
    rollouts (those with a failure or safe outcome) are grouped into the profile's blocks.
    A profile whose blocks never hold more than one prompt gets no diagnostic: its grouping is
    at the item level. For the others, one ``CLUSTER_DESIGN_EFFECT`` per profile gives each
    cohort's intra-cluster correlation, design effect and effective rollout count. It is a
    warning when, in some cohort, the design effect is at least ``DESIGN_EFFECT_WARNING`` and
    a primary profile of the plan puts rollouts of one of the profile's blocks in different
    blocks of its own, which it treats as independent; otherwise it is informational.
    """

    profiles = plan.design.independence_profiles
    if not profiles:
        return
    if data.block_index is None:
        data.block_index = block_index(data.observations, data.manifest)
    manifest_values = data.block_index.manifest_values
    by_cohort: dict[str, list[Observation]] = {}
    for row in data.observations:
        if row.failure is not None:
            by_cohort.setdefault(row.cohort_id, []).append(row)
    primary_ids = [name for name in _primary_profiles(plan) if name in profiles]
    for profile_id in sorted(profiles):
        profile = profiles[profile_id]
        if profile.block_by == ["rollout"]:
            continue
        primaries = {name: profiles[name] for name in primary_ids if name != profile_id}
        cohorts: list[dict[str, Any]] = []
        grouped = False
        warned: set[str] = set()
        for cohort_id in sorted(by_cohort):
            figures, cohort_grouped, splitting = _cohort_figures(
                by_cohort[cohort_id], profile, primaries, manifest_values
            )
            grouped = grouped or cohort_grouped
            cohorts.append({"cohort_id": cohort_id, **figures})
            effect = figures["design_effect"]
            if effect is not None and effect >= DESIGN_EFFECT_WARNING:
                warned.update(splitting)
        if not grouped:
            continue
        message = (
            f"Independence profile {profile_id!r} blocks by {profile.block_by!r}, putting "
            "rollouts of several prompts in one block; "
            + "; ".join(_describe(item["cohort_id"], item) for item in cohorts)
            + "."
        )
        if warned:
            message += (
                f" The primary profile(s) {', '.join(repr(name) for name in sorted(warned))} "
                "treat rollouts that share one of these blocks as independent, so the results "
                "resting on them count more independent rollouts than the effective number "
                f"above. Read the bounds under {profile_id!r}, declared as an independence "
                "sensitivity or as the primary, beside theirs."
            )
        diagnostics.append(
            Diagnostic(
                code="CLUSTER_DESIGN_EFFECT",
                severity="WARNING" if warned else "INFO",
                message=message,
                scope=plan_scope("design", "independence_profiles", profile_id),
                details={
                    "profile": profile_id,
                    "block_by": list(profile.block_by),
                    "design_effect_warning": DESIGN_EFFECT_WARNING,
                    "primaries_splitting_blocks": sorted(warned),
                    "cohorts": cohorts,
                },
            )
        )
