"""FAIL-only claims on attackers that adapt across behaviours (experimental, #125)."""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import AdaptiveClaimSpec, AnalysisPlan
from inspect_safety_stats.ingest.outcomes import classified_by_policy, judge_positive
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import AdaptiveClaimResult
from inspect_safety_stats.stats.adaptive import METHOD, adaptive_lower_bound

AdaptiveStatus = Literal["FAIL", "INCONCLUSIVE"]

SCOPE_NOTE = (
    "The bound describes this attacker at this budget, on these behaviours in this order. It "
    "is no evidence about other attackers or budgets, which is why the claim can only FAIL."
)


class _Undecidable(Exception):
    """The claim's stream cannot be formed from the data; the message says why."""


def _stream(spec: AdaptiveClaimSpec, data: NormalizedData) -> tuple[int, int, list[str]]:
    """The runs and successes of the claim's stream, and what the reader needs to know of it.

    The stream runs over the group's behaviours in the order of ``spec.order_field``, up to the
    last behaviour the cohort attacked. A success is a run whose score the plan's mapping
    classifies as a failure. A behaviour in the stream with no run, with an unclassified
    outcome, or with a failure that a plan policy filled in without a score to confirm it (a
    sample error, a missing score, an unscored answer at the token limit or a content-filter
    stop, :func:`classified_by_policy`), counts as no success. That can only lower the success
    count, which keeps the lower bound valid; a filled-in failure counted as a success could
    raise it above the attacker's true rate.
    """

    group = next((item for item in data.groups if item.id == spec.group), None)
    if group is None:
        raise _Undecidable(f"The plan declares no group {spec.group!r}.")
    manifest = data.manifest
    if spec.order_field not in manifest.columns:
        raise _Undecidable(
            f"The design manifest has no column {spec.order_field!r}, which orders the "
            "behaviours in the attack stream."
        )
    rows = manifest[manifest["prompt_id"].isin(group.prompt_ids)]
    orders: dict[str, int] = {}
    for prompt_id, value in zip(rows["prompt_id"], rows[spec.order_field], strict=True):
        # pandas reads true and false as booleans, and a bool is an int to Python.
        if (
            isinstance(value, (bool, np.bool_))
            or pd.isna(value)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or float(value) % 1 != 0
        ):
            raise _Undecidable(
                f"Behaviour {prompt_id!r} has the order {value!r} in {spec.order_field!r}; "
                "every behaviour in the group needs an integer order."
            )
        orders[str(prompt_id)] = int(value)
    repeated = sorted(order for order, count in Counter(orders.values()).items() if count > 1)
    if repeated:
        raise _Undecidable(
            f"The orders {repeated[:10]} in {spec.order_field!r} are given to more than one "
            "behaviour, so the attack order is not determined."
        )
    observations = [
        item
        for item in data.observations
        if item.cohort_id == spec.cohort and item.prompt_id in orders
    ]
    runs_per_behaviour = Counter(item.prompt_id for item in observations)
    repeated_runs = sorted(prompt for prompt, count in runs_per_behaviour.items() if count > 1)
    if repeated_runs:
        raise _Undecidable(
            f"The cohort attacked the behaviours {repeated_runs[:10]} more than once. The "
            "stream takes one run per behaviour, so the order of the repeats is not "
            "determined."
        )
    if not observations:
        raise _Undecidable(f"The cohort {spec.cohort!r} has no run on the group's behaviours.")
    ordered = sorted(orders, key=orders.__getitem__)
    last = max(ordered.index(item.prompt_id) for item in observations)
    stream = set(ordered[: last + 1])
    runs = [item for item in observations if item.prompt_id in stream]
    # A failure is a confirmed success only when the score itself says so. A plan policy that
    # fills an unknown outcome in as a failure is conservative for an upper bound, and the
    # opposite for this lower one.
    successes = sum(1 for item in runs if judge_positive(item))
    filled_in = sum(1 for item in runs if item.failure is True and classified_by_policy(item))
    unclassified = sum(1 for item in runs if item.outcome_status == "unclassified")
    notes = [
        f"Stream: {last + 1} behaviours in {spec.order_field!r} order, up to the last one the "
        f"cohort attacked, of {len(ordered)} in the group; {successes} confirmed successes."
    ]
    unattacked = last + 1 - len(observations)
    if unattacked or unclassified:
        notes.append(
            f"{unattacked} behaviours in the stream have no run and {unclassified} an "
            "unclassified outcome; each counts as no success, which can only lower the bound."
        )
    if filled_in:
        notes.append(
            f"{filled_in} runs in the stream are failures that a plan policy filled in "
            "(score.errors, score.missing, score.token_limit or score.content_filter_stops) "
            "with no score to confirm them; each counts as no success, which can only lower "
            "the bound."
        )
    return last + 1, successes, notes


def _adaptive_claim(
    spec: AdaptiveClaimSpec,
    plan: AnalysisPlan,
    data: NormalizedData,
    integrity_valid: bool,
) -> AdaptiveClaimResult:
    families = {family.id: family for family in plan.multiplicity.families}
    family = families.get(spec.family)
    result = AdaptiveClaimResult(
        claim_id=spec.id,
        title=spec.title,
        status="INCONCLUSIVE",
        cohort_id=spec.cohort,
        group_id=spec.group,
        order_field=spec.order_field,
        unacceptable_lower=spec.unacceptable_lower,
        alpha=None if family is None else family.alpha,
        multiplicity_family=spec.family,
        method=METHOD,
        mandatory=spec.mandatory,
    )
    if family is None:
        result.reasons = [
            f"The claim's multiplicity family {spec.family!r} is not declared, so it has no "
            "error budget."
        ]
        return result
    try:
        runs, successes, notes = _stream(spec, data)
    except _Undecidable as exc:
        result.method = "unavailable"
        result.reasons = [str(exc), SCOPE_NOTE]
        return result
    lower = adaptive_lower_bound(successes, runs, alpha=family.alpha)
    result.runs, result.successes, result.lower = runs, successes, lower
    decided: AdaptiveStatus = "FAIL" if lower >= spec.unacceptable_lower else "INCONCLUSIVE"
    verdict = (
        f"Anytime-valid lower bound {lower:.12g} on the attacker's mean conditional success "
        + (
            f"is at or above {spec.unacceptable_lower:.12g}."
            if decided == "FAIL"
            else f"is below {spec.unacceptable_lower:.12g}."
        )
    )
    reasons = [verdict, *notes, SCOPE_NOTE]
    if not integrity_valid:
        result.reasons = [
            "The bound was computed but is ineligible because data integrity is invalid.",
            *reasons,
        ]
        return result
    if plan.analysis.intent == "exploratory":
        result.descriptive_status = decided
        result.reasons = [
            "This post hoc exploratory analysis is not eligible for a decision. Descriptive "
            f"only: a confirmatory analysis of this bound would give {decided}.",
            *reasons,
        ]
        return result
    result.status = decided
    result.reasons = reasons
    return result


def _unavailable_adaptive_claims(plan: AnalysisPlan, reason: str) -> list[AdaptiveClaimResult]:
    """Every adaptive claim INCONCLUSIVE for one reason, when no data could be read."""

    families = {family.id: family.alpha for family in plan.multiplicity.families}
    return sorted(
        (
            AdaptiveClaimResult(
                claim_id=spec.id,
                title=spec.title,
                status="INCONCLUSIVE",
                cohort_id=spec.cohort,
                group_id=spec.group,
                order_field=spec.order_field,
                unacceptable_lower=spec.unacceptable_lower,
                alpha=families.get(spec.family),
                multiplicity_family=spec.family,
                method="unavailable",
                mandatory=spec.mandatory,
                reasons=[reason, SCOPE_NOTE],
            )
            for spec in plan.adaptive_claims
        ),
        key=lambda item: item.claim_id,
    )


def _adaptive_claims(
    plan: AnalysisPlan, data: NormalizedData, integrity_valid: bool
) -> list[AdaptiveClaimResult]:
    return sorted(
        (_adaptive_claim(spec, plan, data, integrity_valid) for spec in plan.adaptive_claims),
        key=lambda item: item.claim_id,
    )
