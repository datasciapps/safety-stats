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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from safety_stats.config.models import AnalysisPlan, AnytimeBoundsSpec
from safety_stats.ingest.records import NormalizedData, SampleOrder
from safety_stats.schema.observations import Observation
from safety_stats.schema.reports import Diagnostic
from safety_stats.schema.scopes import cohort_scope
from safety_stats.stats.sequential import AnytimeUpperBound, anytime_upper_bound
from safety_stats.validate.plan import anytime_bound_problems

OrderReader = Callable[[Mapping[str, str]], Mapping[str, SampleOrder]]


class _Refused(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Stream:
    """A bounded cohort and group whose run's recorded order places every prompt once.

    ``in_order`` holds the group's prompts in draw order, and ``observations`` the cohort's
    observations of the plan's score.
    """

    cohort: str
    group_id: str
    scope: str
    order: SampleOrder
    in_order: list[str]
    observations: list[Observation]


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
            "sample_shuffle set to a nonzero integer seed",
        )
    if shuffle == 0 and not order.shuffled:
        # Inspect records the seed 0 but, as it shuffles only under a truthy seed, leaves the
        # samples in dataset order.
        raise _Refused(
            "ANYTIME_ORDER_UNRECORDED",
            f"run {order.run_key!r} has no recorded random order: its samples were not "
            "shuffled, as Inspect does not shuffle under sample_shuffle 0. Run it with "
            "sample_shuffle set to a nonzero integer seed",
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
    stream: _Stream,
    alpha: float,
) -> tuple[AnytimeUpperBound, dict[str, Any]]:
    group = next(item for item in data.groups if item.id == stream.group_id)
    scoped = [row for row in stream.observations if row.prompt_id in group.weights]
    shares, complete, filled = _stream(stream.in_order, scoped, spec.rollouts_per_item)
    position = {prompt: index for index, prompt in enumerate(group.prompt_ids)}
    weights = np.array([group.weights[prompt] for prompt in group.prompt_ids], dtype=float)
    profile = plan.design.independence_profiles[spec.independence_profile]
    independent_rollouts = spec.rollouts_per_item if profile.block_by == ["rollout"] else 1
    bound = anytime_upper_bound(
        shares,
        np.array([position[prompt] for prompt in stream.in_order[: len(shares)]], dtype=int),
        weights / weights.sum(),
        alpha=alpha,
        independent_rollouts=independent_rollouts,
    )
    details = {
        "run_key": stream.order.run_key,
        "sample_shuffle": stream.order.shuffle,
        "rollouts_per_item": spec.rollouts_per_item,
        "independence_profile": spec.independence_profile,
        "independent_rollouts": independent_rollouts,
        "population": bound.population,
        "draws": bound.draws,
        "complete_draws": complete,
        "rollouts_counted_as_failures": filled,
    }
    return bound, details


def _requested(plan: AnalysisPlan) -> AnytimeBoundsSpec | None:
    """The plan's ``outputs.anytime_bounds``, unless it is absent or the plan cannot serve it.

    validate_plan_references reports what keeps a plan from serving it as
    PLAN_ANYTIME_BOUNDS_INVALID.
    """

    spec = plan.outputs.anytime_bounds
    if spec is None or anytime_bound_problems(plan):
        return None
    return spec


def _bounded_groups(plan: AnalysisPlan, spec: AnytimeBoundsSpec) -> list[str]:
    return spec.groups or [group.id for group in plan.groups]


def _streams(
    plan: AnalysisPlan,
    spec: AnytimeBoundsSpec,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
    read_orders: OrderReader | None,
) -> list[_Stream]:
    """The stream of every bounded cohort and group, and an error for each that has none.

    A run's order is read once per cohort, from its log's header.
    """

    score = plan.inputs.score.name
    groups = {group.id: group for group in data.groups}
    streams: list[_Stream] = []
    for cohort in sorted(plan.cohorts):
        observations = [
            row for row in data.observations if row.cohort_id == cohort and row.score_name == score
        ]
        order: SampleOrder | _Refused
        try:
            order = _recorded_order(plan, observations, read_orders)
        except _Refused as refusal:
            order = refusal
        for group_id in _bounded_groups(plan, spec):
            scope = cohort_scope(cohort, "groups", group_id, "outputs", "anytime_bounds")
            try:
                if isinstance(order, _Refused):
                    raise order
                in_order = _group_order(order, observations, groups[group_id].prompt_ids)
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
            streams.append(_Stream(cohort, group_id, scope, order, in_order, observations))
    return streams


def anytime_order_diagnostics(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
    read_orders: OrderReader | None,
) -> None:
    """For ``validate``: the errors ``anytime_bounds`` would add, without computing a bound.

    The same checks of the runs' recorded orders run, so ``validate`` refuses what
    ``analyze`` refuses (CONTRIBUTING section 5.2).
    """

    spec = _requested(plan)
    if spec is not None:
        _streams(plan, spec, data, diagnostics, read_orders)


@dataclass(frozen=True)
class ComputedAnytimeBound:
    """The anytime-valid bound of one cohort and group, before it is reported."""

    cohort: str
    group_id: str
    scope: str
    alpha: float
    bound: AnytimeUpperBound
    details: dict[str, Any]


def anytime_bounds(
    plan: AnalysisPlan,
    data: NormalizedData,
    diagnostics: list[Diagnostic],
    read_orders: OrderReader | None,
) -> list[ComputedAnytimeBound]:
    """The anytime-valid bounds the plan's ``outputs.anytime_bounds`` asks for, if any.

    A run without a recorded random order is refused with an ``ANYTIME_ORDER_UNRECORDED``
    error, and any other reason the stream cannot be built with an ``ANYTIME_BOUND_INVALID``
    error, each added to ``diagnostics``. ``read_orders`` reads the runs' recorded orders, and
    is None for a source that records none. ``anytime_bound_diagnostics`` reports the bounds.
    """

    spec = _requested(plan)
    if spec is None:
        return []
    group_ids = _bounded_groups(plan, spec)
    cohorts = sorted(plan.cohorts)
    family = next(item for item in plan.multiplicity.families if item.id == spec.family)
    alpha = family.alpha / (len(cohorts) * len(group_ids))
    computed: list[ComputedAnytimeBound] = []
    for stream in _streams(plan, spec, data, diagnostics, read_orders):
        bound, details = _bound(plan, spec, data, stream, alpha)
        computed.append(
            ComputedAnytimeBound(
                stream.cohort, stream.group_id, stream.scope, alpha, bound, details
            )
        )
    return computed


def _withheld(plan: AnalysisPlan, integrity_valid: bool) -> str | None:
    """Why no claim may be certified by the bounds, as none may pass: None when they may."""

    reasons = []
    if not integrity_valid:
        reasons.append("data integrity is invalid")
    if plan.analysis.intent == "exploratory":
        reasons.append("the analysis is exploratory")
    calibration = plan.inputs.judge_calibration
    # The plan refuses a validated calibration that covers a bounded group. One scoped to other
    # groups still decides every claim on the corrected risk, and a claim it gives no certifying
    # correction, as on the groups it leaves out, is INCONCLUSIVE: a bound on the judge's
    # reported risk can certify none of them.
    if calibration is not None and calibration.mode == "validated":
        reasons.append(
            "the validated judge calibration decides every claim on corrected risk, not on "
            "the judge's reported risk"
        )
    return " and ".join(reasons) if reasons else None


def anytime_bound_diagnostics(
    plan: AnalysisPlan,
    computed: list[ComputedAnytimeBound],
    diagnostics: list[Diagnostic],
    *,
    integrity_valid: bool,
) -> None:
    """Report each computed bound as an ``ANYTIME_UPPER_BOUND`` diagnostic (INFO).

    It carries the bound and, for each weighted-mean claim on its cohort and group, whether
    the bound certifies the claim's ``acceptable_upper``: lies at or below it. A claim can
    pass on judge-reported risk only in a confirmatory analysis whose integrity is valid and
    whose plan has no validated judge calibration, and the bound certifies nothing elsewhere
    either: under invalid integrity (``integrity_valid`` false, as the claims were decided),
    exploratory intent or a validated calibration, each claim's ``certified`` is None, and
    ``certification_withheld`` says why.
    """

    measure_types = {measure.id: measure.type for measure in plan.outputs.risk_measures}
    withheld = _withheld(plan, integrity_valid)
    for item in computed:
        bound, alpha = item.bound, item.alpha
        weighted = [
            claim
            for claim in plan.claims
            if claim.cohort == item.cohort
            and claim.group == item.group_id
            and measure_types.get(str(claim.measure)) == "weighted_mean"
        ]
        certified = [
            claim.id
            for claim in weighted
            if withheld is None and bound.upper <= claim.acceptable_upper
        ]
        claims = [
            {
                "claim_id": claim.id,
                "acceptable_upper": claim.acceptable_upper,
                "certified": None if withheld is not None else claim.id in certified,
            }
            for claim in weighted
        ]
        if withheld is not None:
            outcome = f"; it certifies no claim, because {withheld}"
        elif certified:
            outcome = f"; at or below acceptable_upper for {', '.join(certified)}"
        else:
            outcome = ""
        diagnostics.append(
            Diagnostic(
                code="ANYTIME_UPPER_BOUND",
                severity="INFO",
                message=(
                    f"experimental anytime-valid upper bound {bound.upper:.6g} on the "
                    f"weighted mean failure probability at one-sided alpha {alpha:.6g}, after "
                    f"{bound.draws} of {bound.population} prompts in the recorded order"
                    + outcome
                    + ". It holds wherever the run stopped; claims keep their "
                    "fixed-horizon status"
                ),
                scope=item.scope,
                details={
                    "experimental": True,
                    "cohort": item.cohort,
                    "group": item.group_id,
                    **item.details,
                    "alpha": alpha,
                    "alpha_sequence": bound.alpha_sequence,
                    "alpha_population": bound.alpha_population,
                    "estimate": bound.estimate,
                    "realized_upper": bound.realized_upper,
                    "upper": bound.upper,
                    "sequence_kernel": bound.sequence_kernel,
                    "population_kernel": bound.population_kernel,
                    "claims": claims,
                    **(
                        {"certification_withheld": withheld} if withheld is not None else {}
                    ),
                },
            )
        )
