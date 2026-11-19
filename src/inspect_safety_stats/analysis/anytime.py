"""Anytime-valid upper bounds for runs stopped early (#126, experimental).

A run that a rule such as optstop's stopped once its outcomes looked precise enough cannot be
certified by the fixed-horizon bounds. When the run's items ran in a random order fixed before
the run, the one-sided bound of ``stats/sequential.py`` holds at every prefix of that order at
once, and so wherever the rule stopped. This module builds that stream for each cohort and
group from the run's recorded sample order and reports the bound as a diagnostic beside the
claims, without changing any claim's status.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from inspect_safety_stats.config.models import AnalysisPlan, AnytimeBoundsSpec
from inspect_safety_stats.ingest.records import NormalizedData, SampleOrder
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.sequential import AnytimeUpperBound, anytime_upper_bound
from inspect_safety_stats.validate.plan import anytime_bound_problems

SCOPE = "outputs.anytime_bounds"

OrderReader = Callable[[Mapping[str, str]], Mapping[str, SampleOrder]]


class _Refused(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _log_location(plan: AnalysisPlan, source: str) -> str:
    """The log an observation was read from: observations record it relative to the plan."""

    if "://" in source or plan.source_path() is None or Path(source).is_absolute():
        return source
    return str((cast(Path, plan.source_path()).parent / source).resolve())


def _recorded_order(
    plan: AnalysisPlan, observations: list[Observation], read_orders: OrderReader | None
) -> SampleOrder:
    runs = sorted({row.run_key for row in observations})
    if not runs:
        raise _Refused("ANYTIME_BOUND_INVALID", "it has no observations, so no run to read")
    if len(runs) > 1:
        raise _Refused(
            "ANYTIME_BOUND_INVALID",
            f"its observations come from {len(runs)} runs, each with its own order; the bound "
            "reads the order of one run",
        )
    if read_orders is None:
        raise _Refused(
            "ANYTIME_ORDER_UNRECORDED",
            "the evaluation source records no sample order; only Inspect logs do",
        )
    order = read_orders({runs[0]: _log_location(plan, observations[0].source)})[runs[0]]
    shuffle = order.shuffle
    if isinstance(shuffle, bool) or not isinstance(shuffle, int):
        why = (
            "it was shuffled without a recorded seed (sample_shuffle: true)"
            if shuffle is True
            else "its samples were not shuffled; Inspect does not shuffle by default"
        )
        raise _Refused(
            "ANYTIME_ORDER_UNRECORDED",
            f"run {order.run_key!r} has no recorded random order: {why}. Run it with "
            "sample_shuffle set to an integer seed",
        )
    if not order.shuffled or not order.sample_ids:
        raise _Refused(
            "ANYTIME_ORDER_UNRECORDED",
            f"run {order.run_key!r} records sample_shuffle {shuffle} but not the shuffled "
            "sample ids",
        )
    return order


def _group_order(
    order: SampleOrder, observations: list[Observation], prompts: tuple[str, ...]
) -> list[str]:
    """The group's prompts in draw order: a uniformly random order of them."""

    item_prompts: dict[str, str] = {}
    for row in observations:
        if item_prompts.setdefault(row.item_id, row.prompt_id) != row.prompt_id:
            raise _Refused(
                "ANYTIME_BOUND_INVALID",
                f"sample {row.item_id!r} joins more than one prompt, so its place is ambiguous",
            )
    members = set(prompts)
    # A sample never observed has no join; under schema 3.0 its id is its prompt id.
    drawn = [item_prompts.get(item, item) for item in order.sample_ids or ()]
    in_group = [prompt for prompt in drawn if prompt in members]
    missing = sorted(members - set(in_group))
    if missing:
        raise _Refused(
            "ANYTIME_BOUND_INVALID",
            f"the recorded order of run {order.run_key!r} does not place {len(missing)} of its "
            f"prompts (first: {missing[0]!r}), so it is not an order of the whole group",
        )
    if len(in_group) != len(members):
        raise _Refused(
            "ANYTIME_BOUND_INVALID",
            f"the recorded order of run {order.run_key!r} lists a prompt of the group twice",
        )
    return in_group


def _stream(
    in_order: list[str], observations: list[Observation], rollouts: int
) -> tuple[np.ndarray, int, int]:
    """Each drawn prompt's failure share over its first ``rollouts`` epochs.

    The stream runs to the last prompt with any of those epochs observed. Before it, an epoch
    that is missing or unclassified counts as a failure. That only raises the outcomes, and the
    bound's bets are fixed in advance, so it stays valid for the upper bound. Returns the
    shares, the number of complete prompts, and the number of rollouts counted as failures
    for being missing or unclassified.
    """

    outcomes: dict[str, dict[int, bool | None]] = {}
    for row in observations:
        if 1 <= row.repeat <= rollouts:
            outcomes.setdefault(row.prompt_id, {})[row.repeat] = row.failure
    observed = [index for index, prompt in enumerate(in_order) if prompt in outcomes]
    length = observed[-1] + 1 if observed else 0
    shares = np.zeros(length)
    complete = 0
    filled = 0
    for index, prompt in enumerate(in_order[:length]):
        epochs = outcomes.get(prompt, {})
        known = [epochs.get(epoch) for epoch in range(1, rollouts + 1)]
        unknown = sum(value is None for value in known)
        shares[index] = (sum(value is True for value in known) + unknown) / rollouts
        complete += unknown == 0
        filled += unknown
    return shares, complete, filled


def _bound(
    plan: AnalysisPlan,
    spec: AnytimeBoundsSpec,
    data: NormalizedData,
    cohort: str,
    group_id: str,
    observations: list[Observation],
    read_orders: OrderReader | None,
    alpha: float,
) -> tuple[AnytimeUpperBound, dict[str, Any]]:
    group = next(item for item in data.groups if item.id == group_id)
    scoped = [row for row in observations if row.prompt_id in group.weights]
    order = _recorded_order(plan, observations, read_orders)
    in_order = _group_order(order, observations, group.prompt_ids)
    shares, complete, filled = _stream(in_order, scoped, spec.rollouts_per_item)
    position = {prompt: index for index, prompt in enumerate(group.prompt_ids)}
    weights = np.array([group.weights[prompt] for prompt in group.prompt_ids], dtype=float)
    profile = plan.design.independence_profiles[spec.independence_profile]
    independent_rollouts = spec.rollouts_per_item if profile.block_by == ["rollout"] else 1
    bound = anytime_upper_bound(
        shares,
        np.array([position[prompt] for prompt in in_order[: len(shares)]], dtype=int),
        weights / weights.sum(),
        alpha=alpha,
        independent_rollouts=independent_rollouts,
    )
    details = {
        "run_key": order.run_key,
        "sample_shuffle": order.shuffle,
        "rollouts_per_item": spec.rollouts_per_item,
        "independence_profile": spec.independence_profile,
        "independent_rollouts": independent_rollouts,
        "population": bound.population,
        "draws": bound.draws,
        "complete_draws": complete,
        "rollouts_counted_as_failures": filled,
    }
    return bound, details


def anytime_bound_diagnostics(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
    read_orders: OrderReader | None,
) -> None:
    """Add the anytime-valid bounds the plan's ``outputs.anytime_bounds`` asks for, if any.

    Each cohort and group gets an ``ANYTIME_UPPER_BOUND`` diagnostic (INFO) that carries the
    bound and, for each weighted-mean claim on it, whether the bound lies at or below the
    claim's ``acceptable_upper``. A run without a recorded random order is refused with an
    ``ANYTIME_ORDER_UNRECORDED`` error, and any other reason the stream cannot be built with
    an ``ANYTIME_BOUND_INVALID`` error. ``read_orders`` reads the runs' recorded orders, and is
    None for a source that records none.
    """

    spec = plan.outputs.anytime_bounds
    if spec is None:
        return
    if anytime_bound_problems(plan):
        # validate_plan_references has reported them as PLAN_ANYTIME_BOUNDS_INVALID.
        return
    family = next(item for item in plan.multiplicity.families if item.id == spec.family)
    group_ids = spec.groups or [group.id for group in plan.groups]
    cohorts = sorted(plan.cohorts)
    alpha = family.alpha / (len(cohorts) * len(group_ids))
    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    score = plan.inputs.score.name
    for cohort in cohorts:
        observations = [
            row for row in data.observations if row.cohort_id == cohort and row.score_name == score
        ]
        for group_id in group_ids:
            scope = f"{SCOPE}:{cohort}:{group_id}"
            try:
                bound, details = _bound(
                    plan, spec, data, cohort, group_id, observations, read_orders, alpha
                )
            except _Refused as refusal:
                diagnostics.append(
                    Diagnostic(
                        code=refusal.code,
                        severity="ERROR",
                        message=f"no anytime-valid bound for cohort {cohort!r}, group "
                        f"{group_id!r}: {refusal}",
                        scope=scope,
                    )
                )
                continue
            weighted = [
                claim
                for claim in plan.claims
                if claim.cohort == cohort
                and claim.group == group_id
                and measure_types.get(str(claim.measure)) == "weighted_mean"
            ]
            certified = [
                claim.id for claim in weighted if bound.upper <= claim.acceptable_upper
            ]
            claims = [
                {
                    "claim_id": claim.id,
                    "acceptable_upper": claim.acceptable_upper,
                    "certified": claim.id in certified,
                }
                for claim in weighted
            ]
            diagnostics.append(
                Diagnostic(
                    code="ANYTIME_UPPER_BOUND",
                    severity="INFO",
                    message=(
                        f"anytime-valid upper bound {bound.upper:.6g} on the weighted mean "
                        f"failure probability at one-sided alpha {alpha:.6g}, after "
                        f"{bound.draws} of {bound.population} prompts in the recorded order"
                        + (
                            f"; at or below acceptable_upper for {', '.join(certified)}"
                            if certified
                            else ""
                        )
                        + ". It holds wherever the run stopped; claims keep their "
                        "fixed-horizon status"
                    ),
                    scope=scope,
                    details={
                        "cohort": cohort,
                        "group": group_id,
                        **details,
                        "alpha": alpha,
                        "alpha_sequence": bound.alpha_sequence,
                        "alpha_population": bound.alpha_population,
                        "estimate": bound.estimate,
                        "realized_upper": bound.realized_upper,
                        "upper": bound.upper,
                        "sequence_kernel": bound.sequence_kernel,
                        "population_kernel": bound.population_kernel,
                        "claims": claims,
                    },
                )
            )
