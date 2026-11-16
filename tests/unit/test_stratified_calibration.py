"""The stratified, finite-population validated judge calibration."""

import math

import numpy as np
import pytest
from scipy.stats import hypergeom

from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_judge_calibration,
    estimate_stratified_judge_calibration,
    hypergeometric_count_bounds,
)


def _counts(labelled: int, truly_unsafe: int, size: int) -> JudgeStratumCounts:
    return JudgeStratumCounts(labelled=labelled, truly_unsafe=truly_unsafe, stratum_size=size)


@pytest.mark.parametrize(
    ("unsafe", "labelled", "size"),
    [(0, 5, 20), (5, 5, 20), (3, 10, 40), (1, 1, 2), (7, 30, 31)],
)
def test_count_bounds_are_the_extreme_counts_the_tails_admit(
    unsafe: int, labelled: int, size: int
) -> None:
    level = 0.05
    lower, upper = hypergeometric_count_bounds(
        truly_unsafe=unsafe,
        labelled=labelled,
        stratum_size=size,
        alpha_lower=level,
        alpha_upper=level,
    )
    # Brute force over every count, from the hypergeometric probabilities themselves.
    admitted_low = [
        k for k in range(size + 1) if hypergeom.sf(unsafe - 1, size, k, labelled) > level
    ]
    admitted_high = [
        k for k in range(size + 1) if hypergeom.cdf(unsafe, size, k, labelled) > level
    ]
    assert lower == min(admitted_low)
    assert upper == max(admitted_high)
    # The labels already hold `unsafe` truly unsafe rows and `labelled - unsafe` safe ones.
    assert unsafe <= lower <= upper <= size - (labelled - unsafe)


def test_a_census_gives_the_count_exactly() -> None:
    assert hypergeometric_count_bounds(
        truly_unsafe=4, labelled=9, stratum_size=9, alpha_lower=0.01, alpha_upper=0.01
    ) == (4, 4)
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={"all": _counts(12, 9, 12)},
        safe_strata={"all": _counts(40, 0, 40)},
        alpha=0.05,
    )
    calibration = stratified.calibration
    assert (calibration.rho_1_lower, calibration.rho_1_upper) == (0.75, 0.75)
    assert (calibration.rho_0_lower, calibration.rho_0_upper) == (0.0, 0.0)
    assert all(item.alpha == 0 for item in stratified.fine_strata)


def test_an_unlabelled_stratum_is_completed_without_spending_alpha() -> None:
    assert hypergeometric_count_bounds(
        truly_unsafe=0, labelled=0, stratum_size=7, alpha_lower=0.0, alpha_upper=0.0
    ) == (0, 7)


def test_one_well_labelled_fine_stratum_is_narrower_than_the_binomial_calibration() -> None:
    unsafe = _counts(40, 34, 60)
    safe = _counts(200, 2, 900)
    binomial = estimate_judge_calibration(
        scenario_id="validated", unsafe_stratum=unsafe, safe_stratum=safe, alpha=0.05
    )
    finite = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={"all": unsafe},
        safe_strata={"all": safe},
        alpha=0.05,
    ).calibration
    assert finite.rho_1_lower >= binomial.rho_1_lower
    assert finite.rho_1_upper <= binomial.rho_1_upper
    assert finite.rho_0_upper <= binomial.rho_0_upper
    # Two thirds of the judge-unsafe stratum is labelled, which the binomial ignores.
    assert finite.rho_1_upper - finite.rho_1_lower < binomial.rho_1_upper - binomial.rho_1_lower


def test_each_fine_stratum_is_weighted_by_its_own_size() -> None:
    # The judges' disagreements are a small stratum labelled in full; the agreements are
    # large and sampled thinly. Pooling the labels would weight them 20 : 20.
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={
            "disagree": _counts(20, 4, 20),
            "agree": _counts(20, 20, 380),
        },
        safe_strata={"all": _counts(100, 0, 2000)},
        alpha=0.05,
    )
    calibration = stratified.calibration
    by_name = {item.stratum: item for item in stratified.fine_strata}
    # The census is exact and spends nothing; the sampled stratum takes the whole endpoint.
    assert (by_name["disagree"].unsafe_lower, by_name["disagree"].unsafe_upper) == (4, 4)
    assert by_name["disagree"].alpha == 0
    assert by_name["agree"].alpha == pytest.approx(0.05 / 4, rel=0, abs=0)
    assert calibration.rho_1_lower == pytest.approx(
        (4 + by_name["agree"].unsafe_lower) / 400, rel=0, abs=0
    )
    assert calibration.rho_1_upper == pytest.approx(4 / 400 + 380 / 400, rel=0, abs=1e-15)
    # Pooled as one simple random sample, the oversampled disagreements drag the rate below
    # every value the design admits.
    pooled = (4 + 20) / 40
    assert pooled < calibration.rho_1_lower


def test_the_endpoint_budget_is_split_over_the_sampled_fine_strata_only() -> None:
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={
            "a": _counts(10, 9, 50),
            "b": _counts(10, 8, 50),
            "census": _counts(5, 5, 5),
            "unlabelled": _counts(0, 0, 5),
        },
        safe_strata={"all": _counts(100, 0, 1000)},
        alpha=0.1,
    )
    spent = {item.stratum: item.alpha for item in stratified.fine_strata}
    assert spent["a"] == spent["b"] == pytest.approx(0.1 / 8, rel=0, abs=0)
    assert spent["census"] == spent["unlabelled"] == 0
    assert [item.stratum for item in stratified.unlabelled_fine_strata] == ["unlabelled"]
    calibration = stratified.calibration
    # The unlabelled stratum is completed as [0, 5] of 110 rollouts.
    by_name = {item.stratum: item for item in stratified.fine_strata}
    assert (by_name["unlabelled"].unsafe_lower, by_name["unlabelled"].unsafe_upper) == (0, 5)
    assert calibration.completed_strata == ()


def test_a_judge_stratum_without_any_label_is_completed_and_both_are_refused() -> None:
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={"x": _counts(0, 0, 3), "y": _counts(0, 0, 4)},
        safe_strata={"all": _counts(300, 0, 900)},
        alpha=0.05,
    )
    calibration = stratified.calibration
    assert calibration.completed_strata == ("judge_unsafe",)
    assert (calibration.rho_1_lower, calibration.rho_1_upper) == (0.0, 1.0)
    with pytest.raises(ValueError, match="at least one judge stratum"):
        estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata={"x": _counts(0, 0, 3)},
            safe_strata={"all": _counts(0, 0, 9)},
            alpha=0.05,
        )


def test_an_uninformative_box_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly informative"):
        estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata={"all": _counts(5, 2, 50)},
            safe_strata={"all": _counts(5, 2, 50)},
            alpha=0.05,
        )


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_alpha_must_be_a_probability(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata={"all": _counts(5, 5, 50)},
            safe_strata={"all": _counts(5, 0, 50)},
            alpha=alpha,
        )


def test_an_empty_judge_stratum_is_rejected() -> None:
    with pytest.raises(ValueError, match="holds no rollout"):
        estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata={},
            safe_strata={"all": _counts(5, 0, 50)},
            alpha=0.05,
        )


def test_the_box_covers_both_parameters_at_the_declared_level() -> None:
    # A fixed finite population, drawn again and again by the stratified design.
    rng = np.random.default_rng(20261116)
    unsafe_sizes, unsafe_truth = {"disagree": 15, "agree": 200}, {"disagree": 6, "agree": 190}
    safe_sizes, safe_truth = {"disagree": 25, "agree": 600}, {"disagree": 9, "agree": 3}
    labels = {"disagree": 10, "agree": 25}
    rho_1 = sum(unsafe_truth.values()) / sum(unsafe_sizes.values())
    rho_0 = sum(safe_truth.values()) / sum(safe_sizes.values())
    alpha = 0.3
    trials = 1500

    def draw(sizes: dict[str, int], truth: dict[str, int]) -> dict[str, JudgeStratumCounts]:
        return {
            name: _counts(
                labels[name],
                int(rng.hypergeometric(truth[name], sizes[name] - truth[name], labels[name])),
                sizes[name],
            )
            for name in sizes
        }

    misses = 0
    for _ in range(trials):
        calibration = estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata=draw(unsafe_sizes, unsafe_truth),
            safe_strata=draw(safe_sizes, safe_truth),
            alpha=alpha,
        ).calibration
        covered = (
            calibration.rho_1_lower <= rho_1 <= calibration.rho_1_upper
            and calibration.rho_0_lower <= rho_0 <= calibration.rho_0_upper
        )
        misses += not covered
    # A binomial standard error above the level still fails a box that undercovers.
    assert misses / trials <= alpha + 3 * math.sqrt(alpha * (1 - alpha) / trials)
