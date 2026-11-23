"""How much the grouping of rollouts into an independence profile's blocks shrinks their evidence.

A profile that blocks by a column above the item, such as a conversation or a shared context,
puts rollouts of several prompts in one block. When those rollouts tend to fail together, a
profile that treats them as independent overstates how much evidence they carry. For each such
profile and cohort, this module estimates the empirical intra-cluster correlation of the scored
rollouts' outcomes within the profile's blocks, the design effect it implies and the effective
number of rollouts, so that a user sees how much the choice between the profiles matters.

A primary profile that splits those blocks counts its own blocks as the independent units, not
the rollouts: a prompt-level primary already treats a prompt's rollouts as one block. For each
such primary the same figures are computed on its units, the scored rollouts of one of its
blocks within one of the profile's blocks, with each unit's failure rate as its outcome and the
correlation taken from pairs of units in one block, which units with few rollouts do not raise.
These figures decide whether the diagnostic warns. Figures on rollouts would not do: whenever
prompts differ in failure rate, the rollouts of one prompt are correlated, and with repeated
rollouts that correlation alone would make the warning fire about a prompt-level primary,
however little the grouping adds. For a primary whose units are single rollouts, such as
``[rollout]``, the figures on its units are the figures on rollouts, computed by the same
arithmetic.

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

# A design effect of 2 on a primary profile's units means they carry about half the evidence of
# as many independent units, which the primary counts them as. At or above it, in some cohort,
# that primary makes CLUSTER_DESIGN_EFFECT a warning.
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


def unit_intracluster_correlation(blocks: Sequence[Sequence[tuple[int, int]]]) -> float | None:
    """The pairwise estimate of the intra-cluster correlation of units' failure rates.

    Each block is a sequence of units, and each unit a pair ``(rollouts, failures)`` whose
    outcome is its failure rate ``y_u = failures / rollouts``. Every unit counts once,
    whatever its number of rollouts. With ``K`` units in ``G`` blocks of ``k_g`` units, overall
    mean ``ybar`` and deviations ``d_u = y_u - ybar``, the estimate is the covariance of two
    distinct units of one block over the variance of all units:
    ``C = sum_g [(sum_{u in g} d_u)^2 - sum_{u in g} d_u^2] / sum_g k_g (k_g - 1)`` over
    ``S^2 = sum_u d_u^2 / K``. It can be negative. None when it is undefined: fewer than two
    blocks, no block with two units, or every unit with the same failure rate.

    Units with fewer rollouts have noisier rates. The one-way ANOVA estimator compares the
    spread of block means with the spread within blocks, so small blocks of noisy units beside
    a large block of well-replicated ones raise it with no dependence at all. The products of
    two independent units' deviations have mean about zero, however noisy each unit is, so the
    pairwise estimate does not.

    When every unit is one rollout, the outcomes are binary, every unit is as noisy as the
    others, and the estimate is the one-way ANOVA estimate of :func:`intracluster_correlation`
    on the blocks' rollout and failure counts, so that a ``[rollout]`` primary's figures keep
    the bits of the figures on rollouts.
    """

    if any(not units for units in blocks) or any(
        rollouts < 1 or failed < 0 or failed > rollouts
        for units in blocks
        for rollouts, failed in units
    ):
        raise ValueError(
            "every block needs a unit, every unit a rollout, and at most one failure per rollout"
        )
    if all(rollouts == 1 for units in blocks for rollouts, _ in units):
        return intracluster_correlation(
            [len(units) for units in blocks],
            [sum(failed for _, failed in units) for units in blocks],
        )
    counts = [len(units) for units in blocks]
    total = sum(counts)
    clusters = len(counts)
    if clusters < 2 or total == clusters:
        return None
    # Equal rates, compared exactly as fractions: their float mean could differ from them in
    # the last bit and leave a ratio of rounding errors. Rates that differ leave a variance
    # above zero.
    first_rollouts, first_failed = blocks[0][0]
    if all(
        failed * first_rollouts == first_failed * rollouts
        for units in blocks
        for rollouts, failed in units
    ):
        return None
    rates = [[failed / rollouts for rollouts, failed in units] for units in blocks]
    overall = math.fsum(rate for block in rates for rate in block) / total
    deviations = [[rate - overall for rate in block] for block in rates]
    variance = math.fsum(value * value for block in deviations for value in block) / total
    pairs = sum(count * (count - 1) for count in counts)
    covariance = (
        math.fsum(
            math.fsum(block) ** 2 - math.fsum(value * value for value in block)
            for block in deviations
        )
        / pairs
    )
    return covariance / variance


def design_effect(sizes: Sequence[int], correlation: float) -> float:
    """``1 + (m - 1) * max(correlation, 0)``, with ``m = sum m_g^2 / N``.

    ``m`` is the size of the cluster an outcome sits in, averaged over outcomes, which is the
    cluster size of the classical approximation when clusters differ in size. A negative
    correlation is read as no clustering, so the design effect is never below 1.
    """

    total = sum(sizes)
    typical = math.fsum(size * size for size in sizes) / total
    return 1 + (typical - 1) * max(correlation, 0.0)


def unit_design_effect(
    blocks: Sequence[Sequence[tuple[int, int]]],
) -> tuple[float | None, float | None]:
    """The intra-cluster correlation of units' failure rates within blocks, and the design
    effect on the units.

    ``blocks`` is as for :func:`unit_intracluster_correlation`. The design effect is
    :func:`design_effect` with the blocks' sizes counted in units, so ``m`` is the number of
    units in the block a unit sits in, averaged over units. Both are None when the
    correlation is undefined.

    With the pairwise correlation at 0 or above, the design effect is
    ``sum_g (sum_{u in g} d_u)^2 / sum_u d_u^2``: the variance of the units' total with the
    blocks as the independent units, over its variance with the units as independent. It is
    at most the number of units in the largest block, though the correlation can pass 1 when
    blocks differ in size.
    """

    correlation = unit_intracluster_correlation(blocks)
    if correlation is None:
        return None, None
    return correlation, design_effect([len(units) for units in blocks], correlation)


def _primary_profiles(plan: AnalysisPlan) -> list[str]:
    """Every profile the plan names as the primary of a result it computes, sorted.

    Disabled direct mean intervals compute no result, so their selection names no primary.
    """

    direct = plan.outputs.direct_mean_intervals
    selections = [
        direct.independence if direct.enabled else None,
        *(item.independence for item in plan.comparisons),
        *(item.independence for item in plan.comparison_sets),
    ]
    primaries = {selection.primary for selection in selections if selection is not None}
    if plan.outputs.anytime_bounds is not None:
        primaries.add(plan.outputs.anytime_bounds.independence_profile)
    return sorted(primaries)


def _unit_figures(
    profile_id: str, own_blocks: int, blocks: list[list[tuple[int, int]]]
) -> dict[str, Any]:
    """The figures on one primary's units, given as (rollouts, failures) within each block,
    with the number of the primary's own blocks they come from."""

    correlation, effect = unit_design_effect(blocks)
    units = sum(len(block) for block in blocks)
    return {
        "profile": profile_id,
        "blocks": own_blocks,
        "units": units,
        "intracluster_correlation": correlation,
        "design_effect": effect,
        "effective_units": None if effect is None else units / effect,
    }


def _cohort_figures(
    rows: list[Observation],
    profile: IndependenceProfileSpec,
    primaries: dict[str, IndependenceProfileSpec],
    manifest_values: Any,
) -> tuple[dict[str, Any], bool]:
    """One cohort's figures, with those on the units of each primary that splits a block, and
    whether a block holds several prompts."""

    blocks: dict[tuple[Any, ...], list[int]] = {}
    prompts: dict[tuple[Any, ...], set[str]] = {}
    # A primary's units within each block: the scored rollouts and failures of each of its
    # blocks, by the block's serialized key.
    primary_units: dict[str, dict[tuple[Any, ...], dict[str, list[int]]]] = {
        name: {} for name in primaries
    }
    for row in rows:
        key = block_key(row, profile, manifest_values, comparison=False)
        counts = blocks.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += row.failure is True
        prompts.setdefault(key, set()).add(row.prompt_id)
        for name, spec in primaries.items():
            inner = _serialized_key(block_key(row, spec, manifest_values, comparison=False))
            unit = primary_units[name].setdefault(key, {}).setdefault(inner, [0, 0])
            unit[0] += 1
            unit[1] += row.failure is True
    ordered = sorted(blocks.items(), key=lambda item: _serialized_key(item[0]))
    sizes = [counts[0] for _, counts in ordered]
    failures = [counts[1] for _, counts in ordered]
    correlation = intracluster_correlation(sizes, failures)
    effect = None if correlation is None else design_effect(sizes, correlation)
    if correlation is None and len(sizes) > 1 and max(sizes) == 1:
        effect = 1.0
    splitting: list[dict[str, Any]] = []
    for name in sorted(primaries):
        units = primary_units[name]
        if not any(len(inner) > 1 for inner in units.values()):
            continue
        splitting.append(
            _unit_figures(
                name,
                len({inner for inner_units in units.values() for inner in inner_units}),
                [
                    [(unit[0], unit[1]) for _, unit in sorted(units[key].items())]
                    for key, _ in ordered
                ],
            )
        )
    figures = {
        "rollouts": len(rows),
        "blocks": len(sizes),
        "largest_block": max(sizes),
        "intracluster_correlation": correlation,
        "design_effect": effect,
        "effective_rollouts": None if effect is None else len(rows) / effect,
        "primaries": splitting,
    }
    grouped = any(len(names) > 1 for names in prompts.values())
    return figures, grouped


def _spans_blocks(figures: dict[str, Any]) -> bool:
    """Whether some block of the primary spans several of the profile's blocks, giving more
    units than the primary has blocks."""

    return bool(figures["units"] != figures["blocks"])


def _describe_units(figures: dict[str, Any]) -> str:
    own = (
        f"from its {figures['blocks']} blocks, some of which span several of these blocks"
        if _spans_blocks(figures)
        else f"its {figures['blocks']} blocks"
    )
    head = f"for the {figures['units']} units of primary {figures['profile']!r} in them ({own})"
    if figures["intracluster_correlation"] is None:
        return f"{head}, all with the same failure rate, so no correlation can be estimated"
    return (
        f"{head}, intra-cluster correlation {figures['intracluster_correlation']:.3f}, "
        f"design effect {figures['design_effect']:.2f}, about "
        f"{figures['effective_units']:.0f} effective units"
    )


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
        + "".join(f", and {_describe_units(item)}" for item in figures["primaries"])
    )


def _warning(profile_id: str, warned: list[str], spanning: list[str]) -> str:
    """What the warning adds to the message, for the primaries warned about."""

    def names(profiles: list[str]) -> str:
        return ", ".join(repr(name) for name in profiles)

    threshold = f"{DESIGN_EFFECT_WARNING:g}"
    text = (
        f" The primary profile(s) {names(warned)} treat their units in one of these blocks as "
        f"independent, though where the design effect on their units is {threshold} or more, "
        "those units carry the evidence of at most half as many independent ones."
    )
    nested = [name for name in warned if name not in spanning]
    if nested:
        text += (
            f" The units of {names(nested)} are their own blocks, so there the results resting "
            f"on them count at least {threshold} times as many independent blocks as the "
            "effective number of their units."
        )
    if spanning:
        text += (
            f" The blocks of {names(spanning)} span several of these blocks: the results "
            "resting on them count their own blocks, given above, as independent, though "
            "blocks that meet in one of these blocks share its dependence, and the effective "
            "number of their units does not say how many independent blocks those are worth."
        )
    return text + (
        f" Read the bounds under {profile_id!r}, declared as an independence sensitivity or as "
        "the primary, beside theirs."
    )


def cluster_design_effects(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> None:
    """Report the design effect of each profile whose blocks group rollouts of several prompts.

    For every profile of ``design.independence_profiles`` and every cohort, the scored
    rollouts (those with a failure or safe outcome) are grouped into the profile's blocks.
    A profile whose blocks never hold more than one prompt gets no diagnostic: its grouping is
    at the item level. For the others, one ``CLUSTER_DESIGN_EFFECT`` per profile gives each
    cohort's intra-cluster correlation, design effect and effective rollout count. For each
    primary profile of a result the plan computes that puts rollouts of one of the profile's
    blocks in different blocks of its own, which it treats as independent, it gives the same
    figures on that primary's units, from :func:`unit_design_effect`: the scored rollouts of
    one of its blocks within one of the profile's blocks, each with its failure rate as the
    outcome, with the number of its own blocks they come from. The diagnostic is a warning
    when, in some cohort, the design effect on a primary's units is at least
    ``DESIGN_EFFECT_WARNING``, so that their effective number is at most half of them;
    otherwise it is informational. The message says that a warned primary whose units are its
    own blocks counts at least that many times as many independent blocks as their effective
    number, and that one whose blocks span several of the profile's blocks counts its own
    blocks, which share the dependence, so that the effective number of units does not count
    them.
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
        # Primaries warned about in a cohort where some of their blocks span several of the
        # profile's blocks: their units are not their blocks.
        spanning: set[str] = set()
        for cohort_id in sorted(by_cohort):
            figures, cohort_grouped = _cohort_figures(
                by_cohort[cohort_id], profile, primaries, manifest_values
            )
            grouped = grouped or cohort_grouped
            cohorts.append({"cohort_id": cohort_id, **figures})
            for item in figures["primaries"]:
                if (
                    item["design_effect"] is not None
                    and item["design_effect"] >= DESIGN_EFFECT_WARNING
                ):
                    warned.add(item["profile"])
                    if _spans_blocks(item):
                        spanning.add(item["profile"])
        if not grouped:
            continue
        message = (
            f"Independence profile {profile_id!r} blocks by {profile.block_by!r}, putting "
            "rollouts of several prompts in one block; "
            + "; ".join(_describe(item["cohort_id"], item) for item in cohorts)
            + "."
        )
        if warned:
            message += _warning(profile_id, sorted(warned), sorted(spanning))
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
