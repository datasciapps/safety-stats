from __future__ import annotations

import pytest

from inspect_safety_stats.stats.planning import (
    exact_half_width,
    rollout_scale_for_half_width,
    rollouts_for_half_width,
    zero_failure_sample_size,
)


def test_zero_rate_matches_zero_failure_planning() -> None:
    for threshold, alpha in [(0.01, 0.05), (0.05, 0.01), (0.002, 0.1)]:
        assert rollouts_for_half_width(
            rate=0.0, half_width=threshold, alpha=alpha
        ) == zero_failure_sample_size(threshold=threshold, alpha=alpha)


def test_returned_count_is_the_smallest_that_meets_the_target() -> None:
    needed = rollouts_for_half_width(rate=0.1, half_width=0.03, alpha=0.01)
    assert needed is not None
    assert exact_half_width(0.1, needed, 0.01) <= 0.03
    assert exact_half_width(0.1, needed - 1, 0.01) > 0.03


def test_unreachable_targets_return_none() -> None:
    assert rollouts_for_half_width(rate=0.5, half_width=1e-5, alpha=0.05, maximum=1000) is None


def test_scale_is_one_when_the_interval_already_meets_the_target() -> None:
    assert rollout_scale_for_half_width(
        rate=0.2, observed_half_width=0.02, observed_trials=500,
        target_half_width=0.05, alpha=0.05,
    ) == 1.0


def test_halving_a_central_interval_needs_about_four_times_the_rollouts() -> None:
    observed = exact_half_width(0.5, 400, 0.025)
    scale = rollout_scale_for_half_width(
        rate=0.5, observed_half_width=observed, observed_trials=400,
        target_half_width=observed / 2, alpha=0.025,
    )
    assert scale == pytest.approx(4.0, rel=0.03)


def test_scale_is_anchored_on_the_observed_width() -> None:
    reference = exact_half_width(0.1, 200, 0.05)
    narrow = rollout_scale_for_half_width(
        rate=0.1, observed_half_width=reference, observed_trials=200,
        target_half_width=reference / 2, alpha=0.05,
    )
    wide = rollout_scale_for_half_width(
        rate=0.1, observed_half_width=2 * reference, observed_trials=200,
        target_half_width=reference / 2, alpha=0.05,
    )
    assert narrow is not None and wide is not None
    assert wide > 3 * narrow


def test_judge_label_plan_meets_the_target_with_minimal_judge_safe_labels() -> None:
    from inspect_safety_stats.stats.planning import _planned_upper, judge_label_plan

    plan = judge_label_plan(judge_risk_upper=0.05, target_upper=0.08, alpha=0.05, cohorts=2)
    assert plan is not None
    assert plan.endpoint_alpha == pytest.approx(0.05 / 2 / 4)
    assert plan.corrected_upper <= 0.08
    assert plan.rho_1_lower > plan.rho_0_upper
    fewer = _planned_upper(0.0, plan.judge_safe_labels - 1, plan.endpoint_alpha)
    assert fewer + 0.05 * (plan.rho_1_upper - fewer) > 0.08


def test_judge_label_plan_is_none_below_the_floor() -> None:
    from inspect_safety_stats.stats.planning import judge_label_plan

    assert judge_label_plan(judge_risk_upper=0.10, target_upper=0.08, alpha=0.05) is None
    imperfect = judge_label_plan(
        judge_risk_upper=0.10, target_upper=0.08, alpha=0.05, anticipated_rho_1=0.5
    )
    assert imperfect is not None
    assert imperfect.floor == pytest.approx(0.05)
