"""The anytime-valid range for a judge stratum's error rate (labelling that may stop on results)."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest

from inspect_safety_stats.stats.binomial import binomial_bounds
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_judge_calibration,
    hypergeometric_confidence_sequence,
)


def _exact(unsafe: int, labelled: int, size: int, alpha: Fraction) -> tuple[Fraction, Fraction]:
    """The same set, computed with exact rationals and no SciPy."""

    kept = [
        k
        for k in range(size + 1)
        if Fraction(
            (labelled + 1) * comb(k, unsafe) * comb(size - k, labelled - unsafe),
            comb(size, labelled),
        )
        > alpha
    ]
    return Fraction(kept[0], size), Fraction(kept[-1], size)


@pytest.mark.parametrize(
    ("unsafe", "labelled", "size"),
    [(0, 1, 10), (3, 10, 40), (0, 25, 200), (25, 25, 200), (7, 30, 31), (12, 50, 500)],
)
def test_matches_an_exact_rational_computation(unsafe: int, labelled: int, size: int) -> None:
    lower, upper = hypergeometric_confidence_sequence(
        JudgeStratumCounts(labelled, unsafe, size), alpha=0.05
    )
    expected_lower, expected_upper = _exact(unsafe, labelled, size, Fraction(1, 20))
    assert lower == pytest.approx(float(expected_lower), rel=0, abs=0)
    assert upper == pytest.approx(float(expected_upper), rel=0, abs=0)


def test_no_label_is_the_whole_range_and_a_census_is_exact() -> None:
    assert hypergeometric_confidence_sequence(JudgeStratumCounts(0, 0, 50), alpha=0.01) == (
        0.0,
        1.0,
    )
    assert hypergeometric_confidence_sequence(JudgeStratumCounts(50, 13, 50), alpha=0.01) == (
        0.26,
        0.26,
    )


def test_contains_the_sample_share_and_narrows_with_more_labels() -> None:
    widths = []
    for labelled in (10, 40, 160):
        unsafe = labelled // 10
        lower, upper = hypergeometric_confidence_sequence(
            JudgeStratumCounts(labelled, unsafe, 1000), alpha=0.05
        )
        assert lower <= unsafe / labelled <= upper
        widths.append(upper - lower)
    assert widths == sorted(widths, reverse=True)


def test_rejects_an_alpha_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha"):
        hypergeometric_confidence_sequence(JudgeStratumCounts(1, 0, 5), alpha=1.0)


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
    assert (calibration.rho_1_lower, calibration.rho_1_upper) == (
        hypergeometric_confidence_sequence(unsafe, alpha=0.02)
    )
    assert (calibration.rho_0_lower, calibration.rho_0_upper) == (
        hypergeometric_confidence_sequence(safe, alpha=0.02)
    )


def _misses(
    rng: np.random.Generator, size: int, unsafe: int, alpha: float, method: str, horizon: int
) -> bool:
    """Whether a labeller who stops on the first miss of ``method`` ever sees the truth excluded.

    This is the most adversarial stopping rule: stop as soon as the range excludes the
    truth. A fixed-sample interval misses under it far more often than ``alpha``; an
    anytime-valid range must not.
    """

    truth = unsafe / size
    population = np.zeros(size, dtype=int)
    population[:unsafe] = 1
    order = rng.permutation(population)
    for labelled in range(1, horizon + 1):
        seen = int(order[:labelled].sum())
        if method == "sequence":
            lower, upper = hypergeometric_confidence_sequence(
                JudgeStratumCounts(labelled, seen, size), alpha=alpha
            )
        else:
            interval = binomial_bounds(
                seen, labelled, alpha_lower=alpha / 2, alpha_upper=alpha / 2
            )
            lower, upper = interval.lower, interval.upper
        if not lower <= truth <= upper:
            return True
    return False


def test_stopping_on_results_keeps_the_sequence_at_nominal_coverage() -> None:
    # Seeded Monte Carlo: 600 labelling paths of up to 120 labels through a stratum of 400
    # rows, 60 unsafe, each stopped the first time the range excludes the truth. The miss
    # rate must stay at or below alpha = 0.1; its simulation standard error at 0.1 is about
    # 0.012, and the bound allows three of them. Fixed-sample Clopper-Pearson under the same
    # rule misses far more than alpha (about 0.3 at this seed), which is why data-dependent
    # stopping needs the sequence.
    rng = np.random.default_rng(20261116)
    trials, alpha = 600, 0.1
    sequence = sum(_misses(rng, 400, 60, alpha, "sequence", 120) for _ in range(trials)) / trials
    standard_error = (alpha * (1 - alpha) / trials) ** 0.5
    assert sequence <= alpha + 3 * standard_error
    clopper = sum(_misses(rng, 400, 60, alpha, "clopper", 120) for _ in range(trials)) / trials
    assert clopper > alpha + 3 * standard_error
