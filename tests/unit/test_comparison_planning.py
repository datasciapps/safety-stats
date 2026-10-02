from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds
from inspect_safety_stats.stats.planning import (
    comparison_power,
    comparison_rollouts,
    kl_rate_bounds,
)


@pytest.mark.parametrize(("successes", "trials", "alpha"), [(0, 50, 0.05), (7, 200, 0.0125),
                                                           (200, 200, 0.01), (31, 64, 0.2)])
def test_rate_bounds_match_the_analysis_engine(successes: int, trials: int, alpha: float) -> None:
    values = np.asarray([1.0] * successes + [0.0] * (trials - successes))
    engine = weighted_bounded_kl_chernoff_mean_bounds(
        values, values, np.full(trials, 1 / trials), alpha_lower=alpha, alpha_upper=alpha,
        independence_profile="rollout",
    )
    assert kl_rate_bounds(successes, trials, alpha) == pytest.approx((engine.lower, engine.upper))


def _simulated_power(
    candidate: float, baseline: float, trials: int, alpha: float, margin: float | None
) -> float:
    rng = np.random.default_rng(7)
    replicates = 4000
    successes = 0
    for x, y in zip(rng.binomial(trials, candidate, replicates),
                    rng.binomial(trials, baseline, replicates), strict=True):
        candidate_lower, candidate_upper = kl_rate_bounds(int(x), trials, alpha)
        baseline_lower, baseline_upper = kl_rate_bounds(int(y), trials, alpha)
        if margin is not None:
            successes += candidate_upper - baseline_lower < margin
        elif candidate > baseline:
            successes += candidate_lower - baseline_upper > 0
        else:
            successes += candidate_upper - baseline_lower < 0
    return successes / replicates


@pytest.mark.parametrize(
    ("candidate", "baseline", "trials", "margin"),
    [(0.10, 0.04, 400, None), (0.02, 0.07, 500, None), (0.05, 0.05, 1500, 0.04)],
)
def test_exact_power_agrees_with_simulation(
    candidate: float, baseline: float, trials: int, margin: float | None
) -> None:
    alpha = 0.0125
    exact = comparison_power(
        candidate_rate=candidate, baseline_rate=baseline, rollouts_per_arm=trials,
        endpoint_alpha=alpha, margin=margin,
    )
    simulated = _simulated_power(candidate, baseline, trials, alpha, margin)
    standard_error = max(np.sqrt(exact * (1 - exact) / 4000), 0.002)
    assert abs(exact - simulated) < 4 * standard_error
    assert 0.05 < exact < 0.999


def test_planned_rollouts_reach_the_target_power() -> None:
    plan = comparison_rollouts(
        candidate_rate=0.08, baseline_rate=0.05, power=0.8, endpoint_alpha=0.0125
    )
    assert plan is not None
    assert plan.achieved_power >= 0.8
    smaller = comparison_power(
        candidate_rate=0.08, baseline_rate=0.05,
        rollouts_per_arm=plan.rollouts_per_arm // 2, endpoint_alpha=0.0125,
    )
    assert smaller < 0.8


def test_planning_refuses_goals_that_are_false_at_the_anticipated_rates() -> None:
    with pytest.raises(ValueError, match="not true at the anticipated rates"):
        comparison_power(candidate_rate=0.08, baseline_rate=0.05, rollouts_per_arm=100,
                         endpoint_alpha=0.01, margin=0.02)
    with pytest.raises(ValueError, match="no difference to detect"):
        comparison_power(candidate_rate=0.05, baseline_rate=0.05, rollouts_per_arm=100,
                         endpoint_alpha=0.01)


def test_plan_comparison_command() -> None:
    result = CliRunner().invoke(
        app,
        ["plan", "comparison", "--baseline-rate", "0.05", "--candidate-rate", "0.05",
         "--margin", "0.03", "--power", "0.8", "--prompts", "100"],
    )
    assert result.exit_code == 0, result.output
    assert "Arm endpoint alpha: 0.0125" in result.output
    assert "per prompt over 100 prompts" in result.output
    bad = CliRunner().invoke(
        app, ["plan", "comparison", "--baseline-rate", "0.05", "--candidate-rate", "0.05"]
    )
    assert bad.exit_code != 0
