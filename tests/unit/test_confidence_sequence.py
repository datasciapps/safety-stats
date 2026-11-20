"""The anytime-valid range for a judge stratum's error rate (labelling that may stop on results).

The range bounds rho, the probability that a rollout of the judge stratum is truly unsafe, not
the share K / N of the run's N rollouts in it that are. The run's rollouts only sample rho:
given N, K is binomial with rate rho, so a census knows K but not rho.
"""

import math
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from inspect_safety_stats.stats import misclassification
from inspect_safety_stats.stats.binomial import binomial_bounds
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    JudgeStratumCounts,
    correct_judge_risk,
    estimate_judge_calibration,
)


def _range(labelled: int, unsafe: int, size: int, alpha: float) -> tuple[float, float]:
    """The judge-unsafe range a calibration with the sequence gives, at endpoint ``alpha``.

    The judge-safe stratum has no label, so it is completed as [0, 1] and spends nothing.
    """

    calibration = estimate_judge_calibration(
        scenario_id="validated",
        unsafe_stratum=JudgeStratumCounts(labelled, unsafe, size),
        safe_stratum=JudgeStratumCounts(0, 0, 1),
        alpha=4 * alpha,
        bound="confidence_sequence",
    )
    return calibration.rho_1_lower, calibration.rho_1_upper


def _posterior_density(unsafe: int, labelled: int, rate: Fraction) -> Fraction:
    """The Beta(unsafe + 1, labelled - unsafe + 1) density at ``rate``, in exact rationals."""

    safe = labelled - unsafe
    return (labelled + 1) * comb(labelled, unsafe) * rate**unsafe * (1 - rate) ** safe


@pytest.mark.parametrize(
    ("unsafe", "labelled"),
    [(0, 1), (1, 1), (3, 10), (0, 25), (25, 25), (7, 30), (12, 50), (1, 400), (399, 400)],
)
def test_the_endpoints_are_where_the_posterior_density_crosses_alpha(
    unsafe: int, labelled: int
) -> None:
    alpha = Fraction(1, 20)
    lower, upper = _range(labelled, unsafe, 10 * labelled, float(alpha))
    # Exact rationals and no SciPy: just inside each finite endpoint the density exceeds
    # alpha and just outside it does not, which pins the endpoint to a relative 1e-9.
    step = Fraction(1, 10**9)
    if unsafe == 0:
        assert lower == 0.0
    else:
        low = Fraction(lower)
        assert _posterior_density(unsafe, labelled, low * (1 - step)) < alpha
        assert _posterior_density(unsafe, labelled, low * (1 + step)) > alpha
    if unsafe == labelled:
        assert upper == 1.0
    else:
        gap = 1 - Fraction(upper)
        assert _posterior_density(unsafe, labelled, 1 - gap * (1 - step)) < alpha
        assert _posterior_density(unsafe, labelled, 1 - gap * (1 + step)) > alpha


def test_a_census_knows_the_share_but_not_the_judge_s_accuracy() -> None:
    sequence = misclassification.judge_accuracy_confidence_sequence
    assert sequence(JudgeStratumCounts(0, 0, 50), alpha=0.01) == (0.0, 1.0)
    # Labelling all 50 rollouts of a stratum gives the share 13 / 50 exactly, but the judge's
    # accuracy only as well as 50 labels from a stratum of any size do.
    census = _range(50, 13, 50, 0.01)
    assert census == _range(50, 13, 5000, 0.01)
    assert census[0] < 0.09 and census[1] > 0.5


def test_contains_the_sample_share_and_narrows_with_more_labels() -> None:
    widths = []
    for labelled in (10, 40, 160):
        unsafe = labelled // 10
        lower, upper = _range(labelled, unsafe, 1000, 0.05)
        assert lower <= unsafe / labelled <= upper
        widths.append(upper - lower)
    assert widths == sorted(widths, reverse=True)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_rejects_an_alpha_outside_the_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        misclassification.judge_accuracy_confidence_sequence(
            JudgeStratumCounts(1, 0, 5), alpha=alpha
        )


def test_calibration_uses_the_sequence_at_the_endpoint_budget() -> None:
    unsafe = JudgeStratumCounts(labelled=40, truly_unsafe=36, stratum_size=300)
    safe = JudgeStratumCounts(labelled=60, truly_unsafe=1, stratum_size=2000)
    calibration = estimate_judge_calibration(
        scenario_id="validated",
        unsafe_stratum=unsafe,
        safe_stratum=safe,
        alpha=0.08,
        bound="confidence_sequence",
    )
    sequence = misclassification.judge_accuracy_confidence_sequence
    assert (calibration.rho_1_lower, calibration.rho_1_upper) == sequence(unsafe, alpha=0.02)
    assert (calibration.rho_0_lower, calibration.rho_0_upper) == sequence(safe, alpha=0.02)


def test_the_range_is_never_narrower_than_clopper_pearson() -> None:
    # The range pays for holding at every number of labels. A sweep of every count up to 400
    # labels at 0.0125, 0.05, 0.1 and 0.4 found it never narrower; this checks up to 60.
    for alpha in (0.0125, 0.1):
        for labelled in range(1, 61):
            for unsafe in range(labelled + 1):
                lower, upper = _range(labelled, unsafe, labelled, alpha)
                fixed = binomial_bounds(unsafe, labelled, alpha_lower=alpha, alpha_upper=alpha)
                assert lower <= fixed.lower and upper >= fixed.upper, (alpha, labelled, unsafe)


def _allowed(level: float, trials: int) -> float:
    """A miss rate the level admits: the level plus three binomial standard errors."""

    return level + 3 * math.sqrt(level * (1 - level) / trials)


def _range_table(size: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """The range after every number of labels and unsafe count in a stratum, indexed [n, S]."""

    lower = np.full((size + 1, size + 1), np.nan)
    upper = np.full((size + 1, size + 1), np.nan)
    for labelled in range(1, size + 1):
        for unsafe in range(labelled + 1):
            lower[labelled, unsafe], upper[labelled, unsafe] = _range(
                labelled, unsafe, size, alpha
            )
    return lower, upper


def _ever_missed(
    *, seed: int, trials: int, size: int, rate: float, alpha: float
) -> tuple[float, float]:
    """How often the range ever lies above, and ever below, the judge's accuracy ``rate``.

    Each run draws a stratum of ``size`` rollouts anew, each truly unsafe with probability
    ``rate``, and labels them in a random order up to a census. A run that misses on a side
    at any number of labels counts as a miss on that side: this is the most adversarial
    stopping rule, which stops as soon as the range excludes the truth.
    """

    rng = np.random.default_rng(seed)
    lower, upper = _range_table(size, alpha)
    # The rollouts are independent, so their labelling order is a random order already.
    unsafe = np.cumsum(rng.random((trials, size)) < rate, axis=1)
    labelled = np.broadcast_to(np.arange(1, size + 1), unsafe.shape)
    above = (lower[labelled, unsafe] > rate).any(axis=1).mean()
    below = (upper[labelled, unsafe] < rate).any(axis=1).mean()
    return float(above), float(below)


@pytest.mark.parametrize(("rate", "seed"), [(0.3, 20261119), (0.04, 20261120)])
def test_stopping_on_results_keeps_each_side_of_the_range_at_its_level(
    rate: float, seed: int
) -> None:
    # 2,000 strata of 60 rollouts drawn anew, labelled up to a census, each side counted as
    # missed if it ever excluded the truth. Each side may miss alpha = 0.1; it missed 0% to
    # 5.3% (simulation standard error at 0.1 about 0.7%). The share K / N, which the
    # calibration used to take as rho, missed 43.6% to 58.5%: a census lands it on one side of
    # rho unless K / N equals rho. Fixed-sample Clopper-Pearson on the labels missed up to
    # 31.8%, and the equal-tailed Beta credible interval up to 100%.
    alpha, trials = 0.1, 2000
    above, below = _ever_missed(seed=seed, trials=trials, size=60, rate=rate, alpha=alpha)
    assert max(above, below) <= _allowed(alpha, trials), (above, below)


def _corrected_misses(
    *, seed: int = 20261122, trials: int = 2000, step: int = 2
) -> tuple[float, float]:
    """How often a corrected interval ever lies above, and ever below, the true risk.

    Each run draws 120 rollouts anew: the judge calls each unsafe with probability 0.5, and
    each is truly unsafe with probability 0.6 if called unsafe and 0.1 if called safe. The
    judge-reported risk gets a two-sided Clopper-Pearson interval at 0.05. Both judge strata
    are labelled in a random order, ``step`` rows of each at a time, up to a census, and after
    each round the calibration at 0.05 corrects that interval. A run counts as a miss on a
    side if any round's interval missed on it; rounds whose box is not strictly informative
    are refused, as the calibration refuses them, and cannot miss.
    """

    rollouts, judge_rate, rates = 120, 0.5, (0.6, 0.1)
    alpha_observed, alpha_calibration = 0.05, 0.05
    true_risk = judge_rate * rates[0] + (1 - judge_rate) * rates[1]
    endpoint = alpha_calibration / 4
    cache: dict[tuple[int, int, int], tuple[float, float]] = {}

    def accuracy(labelled: int, unsafe: int, size: int) -> tuple[float, float]:
        key = (labelled, unsafe, size)
        if key not in cache:
            cache[key] = _range(labelled, unsafe, size, endpoint)
        return cache[key]

    rng = np.random.default_rng(seed)
    above = below = 0
    for _ in range(trials):
        judge_unsafe = int(rng.binomial(rollouts, judge_rate))
        sizes = (judge_unsafe, rollouts - judge_unsafe)
        # The rollouts are independent, so their labelling order is a random order already.
        unsafe = [
            np.cumsum(rng.random(size) < rate) for size, rate in zip(sizes, rates, strict=True)
        ]
        observed = binomial_bounds(
            judge_unsafe,
            rollouts,
            alpha_lower=alpha_observed / 2,
            alpha_upper=alpha_observed / 2,
        )
        missed_above = missed_below = False
        for done in range(step, max(sizes) + step, step):
            ranges = [
                accuracy(min(done, size), int(found[min(done, size) - 1]), size)
                if size
                else (0.0, 1.0)
                for size, found in zip(sizes, unsafe, strict=True)
            ]
            if ranges[0][0] <= ranges[1][1]:
                continue
            calibration = JudgeCalibration(
                scenario_id="validated",
                rho_1_lower=ranges[0][0],
                rho_1_upper=ranges[0][1],
                rho_0_lower=ranges[1][0],
                rho_0_upper=ranges[1][1],
                source="validated",
                alpha=alpha_calibration,
                bound="confidence_sequence",
            )
            corrected = correct_judge_risk(
                observed_lower=observed.lower,
                observed_upper=observed.upper,
                observed_estimate=judge_unsafe / rollouts,
                calibration=calibration,
            )
            missed_above |= corrected.lower > true_risk
            missed_below |= corrected.upper < true_risk
        above += missed_above
        below += missed_below
    return above / trials, below / trials


def test_stopping_on_results_keeps_the_corrected_interval_at_its_stated_coverage() -> None:
    # Each corrected endpoint uses one endpoint of the judge-reported interval and two of the
    # four calibration endpoints, so each side claims to miss at most 0.025 + 0.025 = 0.05,
    # at any round. Neither side ever missed in 2,000 runs. With the share K / N as the range,
    # the interval lay above the true risk in 12.25% of them and below it in 16.25%.
    above, below = _corrected_misses()
    assert max(above, below) <= _allowed(0.05, 2000), (above, below)
