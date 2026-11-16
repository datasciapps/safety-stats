"""Tests for the resolvable risk differences of a two-arm comparison design."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from inspect_safety_stats.config.models import IndependenceProfileSpec
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.stats.comparisons import direct_risk_difference_bounds
from inspect_safety_stats.stats.direct_mean import weighted_bounded_hoeffding_mean_bounds
from inspect_safety_stats.stats.independence import comparison_contrast_blocks
from inspect_safety_stats.stats.planning import comparison_resolvable_differences

ALPHA = 0.025
POWER = 0.8


def _radius(sum_of_squared_ranges: float, error: float) -> float:
    # Hoeffding (1963) for a sum of independent terms with the given ranges.
    return math.sqrt(0.5 * sum_of_squared_ranges * math.log(1 / error))


def test_unpaired_gaps_match_hoeffding_on_the_difference() -> None:
    # Four prompts of weight 1/4 and 100 rollouts per arm: the difference is a sum of 800
    # independent terms, each with range 0.25 / 100.
    plan = comparison_resolvable_differences(
        [100] * 4,
        [100] * 4,
        [0.25] * 4,
        margin=0.05,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        power=POWER,
    )
    ranges = 800 * (0.25 / 100) ** 2
    gap = _radius(ranges, ALPHA) + _radius(ranges, 1 - POWER)
    assert plan.support_gap == pytest.approx(gap, rel=1e-12, abs=0)
    assert plan.rejection_gap == pytest.approx(gap, rel=1e-12, abs=0)
    assert plan.maximum_supportable_difference == pytest.approx(0.05 - gap, rel=1e-12, abs=0)
    assert plan.minimum_detectable_difference == pytest.approx(0.05 + gap, rel=1e-12, abs=0)
    assert plan.concentration_variance == pytest.approx(ranges, rel=1e-12, abs=0)
    assert plan.block_count == 800
    assert plan.independence_unit == "rollout"


def test_pairs_cost_a_factor_of_two_in_variance_at_equal_counts() -> None:
    # A pair's difference ranges over [-1, 1] times its weight; two independent arms each
    # range over [0, 1]. Hoeffding cannot use the pair's correlation, only its range.
    shared = dict(margin=0.0, alpha_lower=ALPHA, alpha_upper=ALPHA, power=POWER)
    unpaired = comparison_resolvable_differences([50, 50], [50, 50], [0.5, 0.5], **shared)
    paired = comparison_resolvable_differences(
        [50, 50], [50, 50], [0.5, 0.5], independence_unit="pair", **shared
    )
    assert paired.concentration_variance == pytest.approx(
        2 * unpaired.concentration_variance, rel=1e-12, abs=0
    )
    assert paired.block_count == 100
    assert paired.support_gap is not None and unpaired.support_gap is not None
    assert paired.support_gap == pytest.approx(
        math.sqrt(2) * unpaired.support_gap, rel=1e-12, abs=0
    )


def test_prompt_clusters_ignore_rollout_counts() -> None:
    shared = dict(margin=0.1, alpha_lower=ALPHA, alpha_upper=ALPHA, power=POWER)
    many = comparison_resolvable_differences(
        [40, 7], [3, 90], [0.5, 0.5], independence_unit="prompt_cluster", **shared
    )
    single = comparison_resolvable_differences(
        [1, 1], [1, 1], [0.5, 0.5], independence_unit="prompt_cluster", **shared
    )
    assert many.concentration_variance == pytest.approx(4 * 0.5, rel=1e-12, abs=0)
    assert many.support_gap == single.support_gap
    assert many.block_count == 2
    assert many.maximum_supportable_difference is None
    assert many.minimum_detectable_difference is None


def test_unattainable_directions_are_none() -> None:
    plan = comparison_resolvable_differences(
        [20], [20], [1.0], margin=-0.9, alpha_lower=ALPHA, alpha_upper=ALPHA, power=POWER
    )
    assert plan.maximum_supportable_difference is None
    assert plan.support_gap is None
    assert plan.minimum_detectable_difference is not None
    assert plan.rejection_gap is not None


def _observation(cohort: str, prompt: str, repeat: int) -> Observation:
    return Observation(
        observation_id=f"{cohort}-{prompt}-{repeat}",
        source="test.eval",
        run_key=f"eval-{cohort}",
        rollout_id=f"{cohort}-{prompt}-{repeat}",
        item_id=prompt,
        repeat=repeat,
        prompt_id=prompt,
        question_id=prompt,
        task_id="task",
        cohort_id=cohort,
        score_name="safety",
        score_raw="PASS",
        failure=False,
        model=cohort,
        configuration_fingerprint=cohort,
        collection_order=repeat,
    )


@pytest.mark.parametrize(
    ("profile", "unit"),
    [
        (IndependenceProfileSpec(block_by=["rollout"]), "rollout"),
        (
            IndependenceProfileSpec(block_by=["prompt_id", "repeat"], comparison_coupling="joint"),
            "pair",
        ),
        (
            IndependenceProfileSpec(block_by=["prompt_id"], comparison_coupling="joint"),
            "prompt_cluster",
        ),
    ],
)
def test_radii_are_those_of_the_analysis_hoeffding_difference(
    profile: IndependenceProfileSpec, unit: str
) -> None:
    # Unequal and uneven counts, as analyze would see them, through analyze's own blocks.
    candidate_counts = {"p1": 3, "p2": 5, "p3": 2}
    baseline_counts = {"p1": 3, "p2": 2, "p3": 4}
    weights = np.asarray([0.5, 0.3, 0.2])
    observations = [
        _observation(cohort, prompt, repeat)
        for cohort, counts in (("candidate", candidate_counts), ("baseline", baseline_counts))
        for prompt, count in counts.items()
        for repeat in range(1, count + 1)
    ]
    manifest = pd.DataFrame(
        [{"prompt_id": p, "question_id": p, "task_id": "task"} for p in candidate_counts]
    )
    blocks = comparison_contrast_blocks(
        observations,
        manifest,
        profile,
        candidate="candidate",
        baseline="baseline",
        prompt_ids=list(candidate_counts),
        prompt_weights=weights,
    )
    analysis = weighted_bounded_hoeffding_mean_bounds(
        blocks.lower,
        blocks.upper,
        blocks.weights,
        alpha_lower=0.01,
        alpha_upper=ALPHA,
        independence_profile=unit,
    )
    plan = comparison_resolvable_differences(
        list(candidate_counts.values()),
        list(baseline_counts.values()),
        weights,
        margin=0.0,
        alpha_lower=0.01,
        alpha_upper=ALPHA,
        power=POWER,
        independence_unit=unit,  # type: ignore[arg-type]
    )
    # The difference is 2Q - 1, so its variance proxy is four times that of Q, and every
    # radius on it is twice the analysis's half-width on Q.
    assert plan.concentration_variance == pytest.approx(
        4 * analysis.concentration_variance, rel=1e-12, abs=0
    )
    assert 2 * _radius(plan.concentration_variance / 4, ALPHA) == pytest.approx(
        2 * analysis.half_width_upper, rel=1e-12, abs=0
    )
    assert plan.block_count == blocks.count


@pytest.mark.parametrize("unit", ["rollout", "prompt_cluster"])
def test_radii_match_the_direct_hoeffding_difference(unit: str) -> None:
    candidate = np.asarray([4, 9, 2])
    baseline = np.asarray([6, 1, 7])
    weights = np.asarray([0.2, 0.5, 0.3])
    analysis = direct_risk_difference_bounds(
        np.zeros(3, dtype=int),
        np.zeros(3, dtype=int),
        candidate,
        np.zeros(3, dtype=int),
        np.zeros(3, dtype=int),
        baseline,
        weights,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        method="weighted_hoeffding",
        independence_unit=unit,  # type: ignore[arg-type]
    )
    plan = comparison_resolvable_differences(
        candidate,
        baseline,
        weights,
        margin=0.0,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        power=POWER,
        independence_unit=unit,  # type: ignore[arg-type]
    )
    assert plan.concentration_variance == pytest.approx(
        analysis.concentration_variance, rel=1e-12, abs=0
    )
    assert plan.support_gap is None or plan.support_gap >= 2 * analysis.half_width_upper
    assert plan.effective_sample_size == pytest.approx(
        analysis.effective_sample_size, rel=1e-12, abs=0
    )


def test_monte_carlo_power_at_the_resolvable_differences() -> None:
    # Seeded simulation of the analysis's Hoeffding difference at the planned true differences.
    # 4000 replicates: the standard error of an empirical power near 0.8 is at most 0.0064, so
    # an empirical power below 0.8 - 4 * 0.0064 would contradict the guarantee.
    rng = np.random.default_rng(20261116)
    candidate = np.asarray([60, 60, 60, 60])
    baseline = np.asarray([60, 60, 60, 60])
    weights = np.asarray([0.4, 0.3, 0.2, 0.1])
    margin = 0.05
    plan = comparison_resolvable_differences(
        candidate, baseline, weights, margin=margin, alpha_lower=ALPHA, alpha_upper=ALPHA,
        power=POWER,
    )
    assert plan.maximum_supportable_difference is not None
    assert plan.minimum_detectable_difference is not None
    baseline_rates = np.asarray([0.3, 0.2, 0.25, 0.3])
    replicates = 4000
    for true_difference, supported in (
        (plan.maximum_supportable_difference, True),
        (plan.minimum_detectable_difference, False),
    ):
        # Heterogeneous prompt differences whose weighted mean is the planned difference.
        candidate_rates = np.clip(baseline_rates + true_difference, 0, 1)
        realised = float(np.dot(weights, candidate_rates - baseline_rates))
        assert realised == pytest.approx(true_difference, rel=1e-12, abs=1e-12)
        hits = 0
        for _ in range(replicates):
            bounds = direct_risk_difference_bounds(
                rng.binomial(candidate, candidate_rates),
                np.zeros(4, dtype=int),
                candidate,
                rng.binomial(baseline, baseline_rates),
                np.zeros(4, dtype=int),
                baseline,
                weights,
                alpha_lower=ALPHA,
                alpha_upper=ALPHA,
                method="weighted_hoeffding",
            )
            hits += bounds.upper <= margin if supported else bounds.lower >= margin
        assert hits / replicates >= POWER - 4 * 0.0064


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (dict(candidate_trials=[10], baseline_trials=[10, 10]), "aligned"),
        (dict(candidate_trials=[0], baseline_trials=[10]), "positive integers"),
        (dict(candidate_trials=[True], baseline_trials=[10]), "positive integers"),
        (dict(margin=1.0), "margin"),
        (dict(margin=True), "margin"),
        (dict(weights=[0.9]), "sum to one"),
        (dict(power=1.0), "power"),
        (dict(independence_unit="prompt"), "independence_unit"),
    ],
)
def test_invalid_inputs_are_rejected(arguments: dict[str, object], message: str) -> None:
    call: dict[str, object] = dict(
        candidate_trials=[10],
        baseline_trials=[10],
        weights=[1.0],
        margin=0.0,
        alpha_lower=ALPHA,
        alpha_upper=ALPHA,
        power=POWER,
    )
    call.update(arguments)
    with pytest.raises(ValueError, match=message):
        comparison_resolvable_differences(**call)  # type: ignore[arg-type]
