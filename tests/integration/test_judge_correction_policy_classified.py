"""Judge corrections of rows that hold rollouts a plan policy classified, end to end.

A plan policy, not the judge, classifies a provider's policy block counted safe, a sample error
or missing score counted as a failure, and a content-filter stop or unscored truncated answer
the plan classifies. Its classification is the truth the estimand defines, and the row's bounds
do not carry the share of such rollouts, so the correction of such a row credits the judge with
no false alarm at the upper bound and no missed failure at the lower.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from tests.diagnostic_codes.harness import Inputs, Outcome, claim
from tests.diagnostic_codes.judge_analysis_cases import (
    DECLARED,
    RHO_0,
    RHO_1,
    assumed,
    comparison,
    judged,
    policy_blocked,
    validated,
)


def _domains(inputs: Inputs) -> None:
    """p1 in domain a and p2 in domain b, with a group b of p2 only."""

    inputs.manifest[0]["domain"], inputs.manifest[1]["domain"] = "a", "b"
    inputs.plan["groups"].append({"id": "b", "filter": {"domain": "b"}})


def _classify_first_by_policy(policy: str) -> Callable[[Inputs], None]:
    """Have a plan policy classify p1's first rollout: safe, or as a failure."""

    def data(inputs: Inputs) -> None:
        _domains(inputs)
        first = inputs.rollouts[0]
        assert first["item_id"] == "p1"
        if policy == "blocked_safe":
            policy_blocked(inputs)
        elif policy == "missing_as_failure":
            inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"
            first["score_harmful"] = None
        else:
            inputs.plan["inputs"]["score"]["errors"] = "count_as_failure"
            policy_blocked(inputs)

    return data


def _row(outcome: Outcome, group: str = "all", measure: str = "mean") -> Any:
    (found,) = [
        item
        for item in outcome.report.group_risks
        if (item.cohort_id, item.group_id, item.measure_id) == ("candidate", group, measure)
    ]
    return found


def _interval(row: Any, interval_id: str) -> Any:
    found = [item for item in row.intervals if item.interval_id == interval_id]
    return found[0] if found else None


@pytest.mark.parametrize("policy", ["blocked_safe", "missing_as_failure", "error_as_failure"])
def test_an_assumed_correction_credits_no_judge_error_on_a_row_a_policy_classified(
    tmp_path: Any, policy: str
) -> None:
    outcome = judged(assumed, _classify_first_by_policy(policy))(tmp_path)

    # Group all holds p1's policy-classified rollout: rho_1 may reach 1 at the upper bound and
    # rho_0 fall to 0 at the lower.
    row = _row(outcome)
    widened = _interval(row, DECLARED)
    assert widened.lower == pytest.approx(row.lower * RHO_1[0], abs=1e-12)
    assert widened.upper == pytest.approx(row.upper + (1 - row.upper) * RHO_0[1], abs=1e-12)
    safe, failures = (1, 0) if policy == "blocked_safe" else (0, 1)
    assert any(
        item.startswith(
            f"1 of the row's rollouts were classified by a plan policy, not by the judge "
            f"({safe} counted safe, {failures} counted as failures)"
        )
        for item in widened.assumptions
    )
    # Group b does not, so it keeps the declared box.
    row = _row(outcome, "b")
    plain = _interval(row, DECLARED)
    assert plain.lower == pytest.approx(
        row.lower * RHO_1[0] + (1 - row.lower) * RHO_0[0], abs=1e-12
    )
    assert plain.upper == pytest.approx(
        row.upper * RHO_1[1] + (1 - row.upper) * RHO_0[1], abs=1e-12
    )
    assert not any("plan policy" in item for item in plain.assumptions)
    (found,) = outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")
    assert found.details == {"groups": ["all"], "policy_safe": safe, "policy_failures": failures}


def _judge_labels(inputs: Inputs, observations: list[Any]) -> None:
    """Label every rollout the judge classified, as the judge says, by a simple random sample."""

    labels = [
        (item.observation_id, "unsafe" if item.failure else "safe")
        for item in observations
        if item.failure is not None and item.error is None
    ]
    validated(inputs, labels, sampling="simple_random")


def _box(interval: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """The rho_1 and rho_0 ranges a corrected interval states, to six significant digits."""

    def ranges(given: str) -> tuple[float, float]:
        (text,) = [item for item in interval.assumptions if item.startswith(given)]
        low, high = re.findall(r"\[([^,]+), ([^\]]+)\]", text)[0]
        return float(low), float(high)

    return (
        ranges("P(truly unsafe | judge said unsafe)"),
        ranges("P(truly unsafe | judge said safe)"),
    )


def test_a_validated_correction_of_such_a_row_still_certifies_and_has_no_reference(
    tmp_path: Any,
) -> None:
    outcome = judged(_judge_labels, _classify_first_by_policy("blocked_safe"))(tmp_path)

    row = _row(outcome)
    corrected = _interval(row, "judge_corrected_validated")
    (rho_1, rho_0) = _box(corrected)
    assert corrected.lower == pytest.approx(row.lower * rho_1[0], rel=1e-5, abs=0)
    assert corrected.upper == pytest.approx(
        row.upper + (1 - row.upper) * rho_0[1], rel=1e-5, abs=0
    )
    # The widened box holds whatever the policy's share, so the correction still certifies,
    # and the claim is decided on it.
    assert (corrected.role, corrected.eligibility) == ("reference", "certifying")
    decided = claim(outcome)
    assert (decided.lower, decided.upper) == (corrected.lower, corrected.upper)
    # The asymptotic references would read the blocked rollout as a judge verdict.
    assert _interval(row, "judge_corrected_validated_delta") is None
    assert _interval(row, "judge_corrected_validated_lang_reiczigel") is None
    reasons = [
        item.message
        for item in outcome.with_code("JUDGE_LANG_REICZIGEL_UNAVAILABLE")
        if item.details.get("group_id") == "all"
    ]
    assert reasons and "classified by a plan policy" in reasons[0]
    # Group b holds no such rollout and keeps both.
    other = _row(outcome, "b")
    assert _interval(other, "judge_corrected_validated_delta") is not None
    plain = _interval(other, corrected.interval_id)
    assert not any("plan policy" in item for item in plain.assumptions)


def test_each_arm_of_a_comparison_is_corrected_by_its_own_rollouts(tmp_path: Any) -> None:
    def data(inputs: Inputs) -> None:
        comparison("weighted_kl_chernoff")(inputs)
        # Only the candidate's run r1 has the blocked rollout; the baseline's r2 does not.
        policy_blocked(inputs)

    outcome = judged(assumed, data)(tmp_path)

    (result,) = [item for item in outcome.report.comparisons if item.level is None]
    primary = next(item for item in result.intervals if item.role == "primary")
    corrected = next(item for item in result.intervals if item.interval_id == DECLARED)
    assert corrected.candidate_lower == pytest.approx(
        primary.candidate_lower * RHO_1[0], abs=1e-12
    )
    assert corrected.candidate_upper == pytest.approx(
        primary.candidate_upper + (1 - primary.candidate_upper) * RHO_0[1], abs=1e-12
    )
    assert corrected.baseline_lower == pytest.approx(
        primary.baseline_lower * RHO_1[0] + (1 - primary.baseline_lower) * RHO_0[0], abs=1e-12
    )
    assert corrected.baseline_upper == pytest.approx(
        primary.baseline_upper * RHO_1[1] + (1 - primary.baseline_upper) * RHO_0[1], abs=1e-12
    )
    assert any(
        item.startswith("Candidate arm: 1 of the row's rollouts were classified by a plan policy")
        for item in corrected.assumptions
    )
    assert not any(item.startswith("Baseline arm:") for item in corrected.assumptions)
    (found,) = outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")
    assert found.details["groups"] == ["all"]
