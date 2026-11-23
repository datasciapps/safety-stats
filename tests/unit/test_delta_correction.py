"""Tests for the asymptotic (delta-method) judge-corrected interval and its use as a reference."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from safety_stats.stats import binomial_bounds
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    correct_judge_risk,
    delta_method_corrected_risk,
    estimate_judge_calibration,
)


def test_delta_method_matches_the_hand_computed_formula() -> None:
    unsafe = JudgeStratumCounts(labelled=20, truly_unsafe=18, stratum_size=100)
    safe = JudgeStratumCounts(labelled=30, truly_unsafe=0, stratum_size=900)
    result = delta_method_corrected_risk(
        observed_estimate=0.1,
        observed_variance=0.001,
        unsafe_stratum=unsafe,
        safe_stratum=safe,
        alpha=0.05,
    )
    # Agresti-Coull stabilised stratum proportions: (x + 2) / (n + 4).
    rho_1, rho_0 = 20 / 24, 2 / 34
    var_1, var_0 = rho_1 * (1 - rho_1) / 24, rho_0 * (1 - rho_0) / 34
    estimate = 0.1 * rho_1 + 0.9 * rho_0
    variance = (rho_1 - rho_0) ** 2 * 0.001 + 0.1**2 * var_1 + 0.9**2 * var_0
    assert result.estimate == pytest.approx(estimate)
    assert result.standard_error == pytest.approx(math.sqrt(variance))
    z = norm.ppf(0.975)
    assert result.lower == pytest.approx(estimate - z * math.sqrt(variance))
    assert result.upper == pytest.approx(estimate + z * math.sqrt(variance))
    assert result.rho_1_estimate == pytest.approx(rho_1)
    assert result.alpha == 0.05


def test_delta_method_validates_inputs_and_clips_to_the_unit_interval() -> None:
    unsafe = JudgeStratumCounts(labelled=5, truly_unsafe=5, stratum_size=10)
    safe = JudgeStratumCounts(labelled=5, truly_unsafe=0, stratum_size=10)
    with pytest.raises(ValueError, match="alpha"):
        delta_method_corrected_risk(
            observed_estimate=0.5,
            observed_variance=0.01,
            unsafe_stratum=unsafe,
            safe_stratum=safe,
            alpha=1.5,
        )
    with pytest.raises(ValueError, match="variance"):
        delta_method_corrected_risk(
            observed_estimate=0.5,
            observed_variance=-1.0,
            unsafe_stratum=unsafe,
            safe_stratum=safe,
            alpha=0.05,
        )
    with pytest.raises(ValueError, match="labelled"):
        delta_method_corrected_risk(
            observed_estimate=0.5,
            observed_variance=0.01,
            unsafe_stratum=JudgeStratumCounts(labelled=0, truly_unsafe=0, stratum_size=10),
            safe_stratum=safe,
            alpha=0.05,
        )
    tiny = delta_method_corrected_risk(
        observed_estimate=0.0,
        observed_variance=0.0,
        unsafe_stratum=unsafe,
        safe_stratum=safe,
        alpha=0.05,
    )
    assert tiny.lower >= 0.0 and tiny.upper <= 1.0


def test_delta_method_is_narrower_than_the_exact_box_on_a_realistic_design() -> None:
    rollouts, failures = 1000, 150
    unsafe = JudgeStratumCounts(labelled=100, truly_unsafe=80, stratum_size=failures)
    safe = JudgeStratumCounts(labelled=100, truly_unsafe=2, stratum_size=rollouts - failures)
    observed = binomial_bounds(failures, rollouts, alpha_lower=0.025, alpha_upper=0.025)
    exact = correct_judge_risk(
        observed_lower=observed.lower,
        observed_upper=observed.upper,
        observed_estimate=failures / rollouts,
        calibration=estimate_judge_calibration(
            scenario_id="v", unsafe_stratum=unsafe, safe_stratum=safe, alpha=0.05
        ),
    )
    q = failures / rollouts
    delta = delta_method_corrected_risk(
        observed_estimate=q,
        observed_variance=q * (1 - q) / rollouts,
        unsafe_stratum=unsafe,
        safe_stratum=safe,
        alpha=0.05,
    )
    assert exact.lower <= delta.lower
    assert delta.upper <= exact.upper
    assert (delta.upper - delta.lower) < 0.7 * (exact.upper - exact.lower)


def test_delta_method_coverage_is_close_to_nominal_with_moderate_calibration() -> None:
    # Asymptotic, so this pins "close to nominal", not a finite-sample guarantee.
    rng = np.random.default_rng(17)
    prevalence, sensitivity, specificity = 0.05, 0.90, 0.97
    q = prevalence * sensitivity + (1 - prevalence) * (1 - specificity)
    rho_1 = prevalence * sensitivity / q
    rho_0 = prevalence * (1 - sensitivity) / (1 - q)
    rollouts, labels = 500, 100
    covered = 0
    replications = 1000
    for _ in range(replications):
        positives = int(rng.binomial(rollouts, q))
        q_hat = positives / rollouts
        m1 = min(labels, positives)
        m0 = min(labels, rollouts - positives)
        unsafe = JudgeStratumCounts(
            labelled=m1, truly_unsafe=int(rng.binomial(m1, rho_1)), stratum_size=positives
        )
        safe = JudgeStratumCounts(
            labelled=m0,
            truly_unsafe=int(rng.binomial(m0, rho_0)),
            stratum_size=rollouts - positives,
        )
        result = delta_method_corrected_risk(
            observed_estimate=q_hat,
            observed_variance=q_hat * (1 - q_hat) / rollouts,
            unsafe_stratum=unsafe,
            safe_stratum=safe,
            alpha=0.05,
        )
        covered += result.lower <= prevalence <= result.upper
    assert covered / replications >= 0.95 - 0.04
