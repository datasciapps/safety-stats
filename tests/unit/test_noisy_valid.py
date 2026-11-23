"""Tests for the finite-sample noisy-but-valid certification test."""

from __future__ import annotations

import numpy as np
import pytest

from safety_stats.stats import binomial_bounds
from safety_stats.stats.misclassification import (
    TruthStratumCounts,
    noisy_valid_certification,
)


def _certify(  # type: ignore[no-untyped-def]
    q_lower: float, q_upper: float, unsafe: TruthStratumCounts, safe: TruthStratumCounts, **kw
):
    defaults = dict(
        judge_reported_alpha=0.025,
        calibration_alpha=0.05,
        acceptable_upper=0.10,
        unacceptable_lower=0.30,
    )
    defaults.update(kw)
    return noisy_valid_certification(
        judge_reported_lower=q_lower,
        judge_reported_upper=q_upper,
        truly_unsafe=unsafe,
        truly_safe=safe,
        **defaults,
    )


def test_truth_stratum_counts_validate() -> None:
    with pytest.raises(ValueError, match="judge_positive"):
        TruthStratumCounts(labelled=5, judge_positive=6)
    with pytest.raises(ValueError, match="nonnegative"):
        TruthStratumCounts(labelled=-1, judge_positive=0)


def test_critical_thresholds_follow_the_exact_calibration_bounds() -> None:
    unsafe = TruthStratumCounts(labelled=40, judge_positive=36)
    safe = TruthStratumCounts(labelled=200, judge_positive=4)
    result = _certify(0.02, 0.06, unsafe, safe)
    endpoint = 0.05 / 4
    sensitivity = binomial_bounds(36, 40, alpha_lower=endpoint, alpha_upper=endpoint)
    false_positive = binomial_bounds(4, 200, alpha_lower=endpoint, alpha_upper=endpoint)
    assert result.sensitivity_lower == pytest.approx(sensitivity.lower)
    assert result.false_positive_lower == pytest.approx(false_positive.lower)
    # PASS threshold: judge-reported rate implied by pi = acceptable under the least
    # favourable calibration; FAIL threshold mirrors it with the upper endpoints.
    assert result.critical_threshold_pass == pytest.approx(
        0.10 * sensitivity.lower + 0.90 * false_positive.lower
    )
    assert result.critical_threshold_fail == pytest.approx(
        0.30 * sensitivity.upper + 0.70 * false_positive.upper
    )
    assert result.alpha_pass == pytest.approx(0.025 + 0.05 / 2)
    assert result.alpha_fail == pytest.approx(0.025 + 0.05 / 2)


def test_decisions_follow_the_thresholds_and_the_identified_set() -> None:
    unsafe = TruthStratumCounts(labelled=200, judge_positive=190)
    safe = TruthStratumCounts(labelled=1000, judge_positive=20)
    # A judge-reported rate comfortably below the PASS threshold certifies.
    passing = _certify(0.01, 0.04, unsafe, safe)
    assert passing.decision == "PASS"
    assert passing.judge_reported_upper < passing.critical_threshold_pass
    assert passing.true_risk_upper < 0.10
    # Above the FAIL threshold establishes unacceptable risk.
    failing = _certify(0.45, 0.55, unsafe, safe)
    assert failing.decision == "FAIL"
    assert failing.judge_reported_lower > failing.critical_threshold_fail
    assert failing.true_risk_lower >= 0.30
    # In between, neither.
    middle = _certify(0.12, 0.20, unsafe, safe)
    assert middle.decision == "INCONCLUSIVE"
    assert middle.identified is True


def test_uninformative_calibration_is_inconclusive_not_passing() -> None:
    weak_unsafe = TruthStratumCounts(labelled=4, judge_positive=3)
    weak_safe = TruthStratumCounts(labelled=4, judge_positive=1)
    result = _certify(0.0, 0.01, weak_unsafe, weak_safe)
    assert result.identified is False
    assert result.decision == "INCONCLUSIVE"
    assert (result.true_risk_lower, result.true_risk_upper) == (0.0, 1.0)


def test_pass_is_equivalent_to_the_identified_set_clearing_the_threshold() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        unsafe = TruthStratumCounts(labelled=60, judge_positive=int(rng.integers(45, 61)))
        safe = TruthStratumCounts(labelled=300, judge_positive=int(rng.integers(0, 15)))
        lower = float(rng.uniform(0.0, 0.15))
        upper = float(min(1.0, lower + rng.uniform(0.0, 0.1)))
        result = _certify(lower, upper, unsafe, safe)
        if result.identified:
            assert (result.decision == "PASS") == (result.true_risk_upper < 0.10)
            assert (result.decision == "FAIL") == (result.true_risk_lower > 0.30)


def test_type_one_error_is_controlled_at_the_boundary() -> None:
    # True risk sits exactly on the acceptable threshold: a PASS is a false pass.
    rng = np.random.default_rng(11)
    prevalence, sensitivity, specificity = 0.10, 0.90, 0.97
    q = prevalence * sensitivity + (1 - prevalence) * (1 - specificity)
    rollouts, labels = 2000, 300
    false_passes = 0
    replications = 800
    for _ in range(replications):
        positives = int(rng.binomial(rollouts, q))
        observed = binomial_bounds(positives, rollouts, alpha_lower=0.025, alpha_upper=0.025)
        truly_unsafe = int(rng.binomial(labels, prevalence))
        unsafe = TruthStratumCounts(
            labelled=truly_unsafe, judge_positive=int(rng.binomial(truly_unsafe, sensitivity))
        )
        safe = TruthStratumCounts(
            labelled=labels - truly_unsafe,
            judge_positive=int(rng.binomial(labels - truly_unsafe, 1 - specificity)),
        )
        result = _certify(observed.lower, observed.upper, unsafe, safe)
        false_passes += result.decision == "PASS"
    # Union bound: 0.025 (judge-reported endpoint) + 0.05 / 2 (two calibration endpoints).
    assert false_passes / replications <= 0.05 + 0.02


def test_the_test_has_power_when_risk_is_well_below_the_threshold() -> None:
    rng = np.random.default_rng(12)
    prevalence, sensitivity, specificity = 0.01, 0.90, 0.97
    q = prevalence * sensitivity + (1 - prevalence) * (1 - specificity)
    rollouts, labels = 4000, 600
    passes = 0
    replications = 200
    for _ in range(replications):
        positives = int(rng.binomial(rollouts, q))
        observed = binomial_bounds(positives, rollouts, alpha_lower=0.025, alpha_upper=0.025)
        truly_unsafe = int(rng.binomial(labels, prevalence))
        unsafe = TruthStratumCounts(
            labelled=truly_unsafe, judge_positive=int(rng.binomial(truly_unsafe, sensitivity))
        )
        safe = TruthStratumCounts(
            labelled=labels - truly_unsafe,
            judge_positive=int(rng.binomial(labels - truly_unsafe, 1 - specificity)),
        )
        passes += _certify(observed.lower, observed.upper, unsafe, safe).decision == "PASS"
    assert passes / replications >= 0.5
