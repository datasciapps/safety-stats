"""The overall decision when a decision policy's thresholds may depend on the data.

With thresholds fixed before the data, a wrong overall PASS needs one claim whose true risk is
above its threshold to pass, and which claim that is does not depend on the data: the chance
is at most that claim's error budget (the intersection-union bound). A threshold read off the
data breaks that. Set every acceptable_upper at its own upper bound, and every claim passes;
the overall PASS is wrong whenever any bound misses, which across separate coverage events
happens with up to the sum of their budgets. Moving one threshold does as much when the claim
is chosen from the data: tighten the claim with the lowest bound to that bound. Only mandatory
claims that one simultaneous event covers keep the overall PASS at one budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from inspect_safety_stats.analysis.claims import coverage_event
from inspect_safety_stats.analysis.policy import _policy_roll_up, _PolicyApplication
from inspect_safety_stats.config.decision_policy import DecisionPolicy
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.decision.engine import decide_claim, roll_up
from inspect_safety_stats.schema.reports import Decision, Diagnostic, RiskIntervalResult
from inspect_safety_stats.stats.binomial import binomial_lower_bound, binomial_upper_bound

ALPHA = 0.05
CLAIMS = 10
ROLLOUTS = 200
TRUE_RISK = 0.10
SIMULATIONS = 4000
SEED = 20261119


def _plan(claims: int = CLAIMS, acceptable_upper: float = 0.05) -> AnalysisPlan:
    """A plan with one mandatory claim per group, in an intersection-union family."""

    groups = [f"g{index}" for index in range(claims)]
    return AnalysisPlan.model_validate(
        {
            "schema_version": "3.0",
            "analysis": {
                "id": "gate",
                "title": "Gate",
                "intent": "confirmatory",
                "plan_frozen_at": "2026-01-01T00:00:00Z",
            },
            "inputs": {
                "design_manifest": "design.csv",
                "score": {
                    "name": "harmful",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
            },
            "cohorts": {"candidate": {"match": {"model": "m"}}},
            "groups": [{"id": group} for group in groups],
            "outputs": {
                "prompt_bounds": {"enabled": True, "family": "displayed"},
                "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
            },
            "multiplicity": {
                "families": [
                    {
                        "id": "displayed",
                        "purpose": "simultaneous_intervals",
                        "method": "bonferroni",
                        "alpha": ALPHA,
                    },
                    {
                        "id": "gate",
                        "purpose": "all_mandatory",
                        "method": "intersection_union",
                        "alpha": ALPHA,
                    },
                ]
            },
            "claims": [
                {
                    "id": group,
                    "title": group,
                    "cohort": "candidate",
                    "group": group,
                    "measure": "mean",
                    "family": "gate",
                    "acceptable_upper": acceptable_upper,
                    "unacceptable_lower": 0.9,
                }
                for group in groups
            ],
        }
    )


def _interval(
    scope: str, *, role: str = "primary", interval_id: str = "direct"
) -> RiskIntervalResult:
    return RiskIntervalResult.model_validate(
        {
            "interval_id": interval_id,
            "method": "weighted_kl_chernoff",
            "role": role,
            "coverage_scope": scope,
            "family": "direct",
            "eligibility": "certifying",
        }
    )


def _ref(group: str) -> str:
    return f"risk:candidate:{group}:mean"


def _policy(thresholds: dict[str, float], set_at: str = "2025-01-01T00:00:00Z") -> DecisionPolicy:
    # Dated before any run: the date is the author's word, and the roll-up does not read it.
    return DecisionPolicy.model_validate(
        {
            "inference_plan_sha256": "0" * 64,
            "versions": [
                {
                    "id": "at-the-bounds",
                    "set_at": set_at,
                    "thresholds": {
                        claim: {"acceptable_upper": value, "unacceptable_lower": 0.99}
                        for claim, value in thresholds.items()
                    },
                }
            ],
        }
    )


def _overall(
    plan: AnalysisPlan,
    thresholds: dict[str, float],
    events: dict[str, str],
    statuses: list[Decision],
    diagnostics: list[Diagnostic] | None = None,
) -> Decision:
    application = _PolicyApplication(
        frozen=plan, policy=_policy(thresholds), applied=True, effective=plan
    )
    decision = roll_up(
        statuses, integrity_valid=True, zero_tolerance_failure=False, confirmatory=True
    )
    written = [] if diagnostics is None else diagnostics
    return _policy_roll_up(application, events, decision, written)


def _simulate(simultaneous: bool, *, one: bool = False) -> tuple[float, float]:
    """The rates of a wrong overall PASS and of any overall PASS, thresholds at the bounds.

    Each claim's upper bound is a one-sided Clopper-Pearson bound on its own independent
    rollouts: at the full alpha for pointwise bounds, each its own coverage event, or at
    alpha / CLAIMS for a simultaneous family, which one event covers. The policy sets every
    claim's acceptable_upper at its bound, or with ``one`` only that of the claim with the
    lowest bound, the others keeping the plan's, a loose 0.5 then.
    """

    rng = np.random.default_rng(SEED)
    endpoint_alpha = ALPHA / CLAIMS if simultaneous else ALPHA
    counts = range(ROLLOUTS + 1)
    upper = [binomial_upper_bound(count, ROLLOUTS, alpha=endpoint_alpha) for count in counts]
    lower = [binomial_lower_bound(count, ROLLOUTS, alpha=endpoint_alpha) for count in counts]
    plan = _plan(acceptable_upper=0.5 if one else 0.05)
    scope = (
        "simultaneous_rollout_fixed_design_mean_family"
        if simultaneous
        else "pointwise_rollout_fixed_design_mean_family"
    )
    events = {claim.id: coverage_event(_ref(claim.id), _interval(scope)) for claim in plan.claims}
    wrong = passed = 0
    for _ in range(SIMULATIONS):
        failures = rng.binomial(ROLLOUTS, TRUE_RISK, size=CLAIMS)
        # The most data-dependent threshold: each claim's own upper bound, rounded up.
        at_bounds = {
            claim.id: math.ceil(upper[count] * 1e6) / 1e6
            for claim, count in zip(plan.claims, failures, strict=True)
        }
        chosen = min(at_bounds, key=at_bounds.__getitem__)
        # What the policy sets, and the thresholds in force.
        thresholds = {chosen: at_bounds[chosen]} if one else at_bounds
        in_force = {
            claim.id: thresholds.get(claim.id, claim.acceptable_upper) for claim in plan.claims
        }
        statuses = [
            decide_claim(
                lower=lower[count],
                upper=upper[count],
                acceptable_upper=in_force[claim.id],
                unacceptable_lower=0.99,
                eligible=True,
            )[0]
            for claim, count in zip(plan.claims, failures, strict=True)
        ]
        if _overall(plan, thresholds, events, statuses) == "PASS":
            passed += 1
            wrong += any(TRUE_RISK > value for value in in_force.values())
    return wrong / SIMULATIONS, passed / SIMULATIONS


@pytest.mark.parametrize("one", [False, True], ids=["every-claim", "one-chosen-claim"])
@pytest.mark.parametrize("simultaneous", [False, True], ids=["pointwise", "simultaneous"])
def test_thresholds_at_the_bounds_keep_a_wrong_overall_pass_within_alpha(
    simultaneous: bool, one: bool
) -> None:
    wrong, passed = _simulate(simultaneous, one=one)
    # Three standard errors of a rate at alpha, so a correct procedure fails this rarely.
    margin = 3 * math.sqrt(ALPHA * (1 - ALPHA) / SIMULATIONS)
    assert wrong <= ALPHA + margin
    if simultaneous:
        # One event covers every bound, so the overall PASS still certifies, and it is right
        # whenever that event holds.
        assert passed == 1.0
    else:
        # Without the roll-up's check about 0.27 of these PASSes would be wrong, and as many
        # with one chosen claim moved.
        assert passed == 0.0


def test_coverage_events_are_shared_only_within_one_simultaneous_family() -> None:
    simultaneous = _interval("simultaneous_rollout_fixed_design_mean_family")
    projected = _interval("simultaneous_rollout_fixed_design_mean_family_hierarchy_region")
    pointwise = _interval("pointwise_rollout_fixed_design_mean_family")
    corrected = _interval(
        "simultaneous_rollout_fixed_design_mean_family_judge_corrected",
        role="reference",
        interval_id="judge_corrected_validated",
    )
    assert coverage_event(_ref("a"), simultaneous) == coverage_event(_ref("b"), simultaneous)
    # A projection covers on the joint event of the family it was projected from.
    assert coverage_event(_ref("a"), projected) == coverage_event(_ref("b"), simultaneous)
    assert coverage_event(_ref("a"), pointwise) != coverage_event(_ref("b"), pointwise)
    assert coverage_event(_ref("a"), pointwise) == coverage_event(_ref("a"), pointwise)
    # A judge correction adds the calibration's own event, so it is never shared.
    assert coverage_event(_ref("a"), corrected) != coverage_event(_ref("b"), corrected)
    assert coverage_event(_ref("a"), None) != coverage_event(_ref("b"), None)
    other = simultaneous.model_copy(update={"family": "other"})
    assert coverage_event(_ref("a"), simultaneous) != coverage_event(_ref("a"), other)


def test_the_roll_up_reads_moved_acceptable_uppers_of_mandatory_claims() -> None:
    plan = _plan(3)
    pointwise = _interval("pointwise_rollout_fixed_design_mean_family")
    events = {claim.id: coverage_event(_ref(claim.id), pointwise) for claim in plan.claims}
    passes: list[Decision] = ["PASS"] * 3
    # The plan's own acceptable_upper, restated, is fixed in advance.
    assert _overall(plan, {"g0": 0.05, "g1": 0.05}, events, passes) == "PASS"
    # An optional claim's threshold does not enter the overall decision.
    optional = plan.model_copy(
        update={
            "claims": [
                plan.claims[0],
                plan.claims[1].model_copy(update={"mandatory": False}),
                plan.claims[2],
            ]
        }
    )
    assert _overall(optional, {"g1": 0.2}, events, passes) == "PASS"
    # One moved mandatory threshold is enough, and every mandatory claim's event is listed.
    diagnostics: list[Diagnostic] = []
    assert _overall(plan, {"g2": 0.2}, events, passes, diagnostics) == "INCONCLUSIVE"
    assert [item.code for item in diagnostics] == ["DECISION_POLICY_OVERALL_UNCERTIFIED"]
    assert diagnostics[0].details["claims"] == ["g2"]
    assert sorted(diagnostics[0].details["coverage_events"]) == ["g0", "g1", "g2"]
    # One simultaneous event over every mandatory claim keeps the PASS.
    simultaneous = _interval("simultaneous_rollout_fixed_design_mean_family")
    shared = {claim.id: coverage_event(_ref(claim.id), simultaneous) for claim in plan.claims}
    assert _overall(plan, {"g0": 0.2, "g2": 0.2}, shared, passes) == "PASS"
    # An optional claim on an event of its own does not count either.
    apart = {**shared, "g1": coverage_event(_ref("g1"), pointwise)}
    assert _overall(optional, {"g0": 0.2}, apart, passes) == "PASS"
    assert _overall(plan, {"g0": 0.2}, apart, passes) == "INCONCLUSIVE"
    # A noisy_valid_test claim, whose thresholds no policy can move, still counts with an event
    # of its own, so the rule does not rest on the policy check that refuses to move them.
    tested = plan.model_copy(
        update={
            "claims": [
                plan.claims[0].model_copy(update={"method": "noisy_valid_test"}),
                *plan.claims[1:],
            ]
        }
    )
    with_test = {**shared, "g0": coverage_event(_ref("g0"), None)}
    assert _overall(tested, {"g1": 0.2}, with_test, passes) == "INCONCLUSIVE"
    # FAIL is the conservative verdict, and stays.
    fails: list[Decision] = ["PASS", "FAIL", "PASS"]
    assert _overall(plan, {"g0": 0.2, "g1": 0.2}, events, fails) == "FAIL"
    not_applied = _PolicyApplication(
        frozen=plan, policy=_policy({"g0": 0.2, "g1": 0.2}), applied=False, effective=plan
    )
    assert _policy_roll_up(not_applied, events, "INCONCLUSIVE", []) == "INCONCLUSIVE"
    assert _policy_roll_up(None, events, "PASS", []) == "PASS"
