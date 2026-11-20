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
    assert scale == pytest.approx(4.0, rel=0.03, abs=0)


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
