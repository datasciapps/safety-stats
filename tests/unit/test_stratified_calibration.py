"""The stratified, finite-population validated judge calibration."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import binom

from inspect_safety_stats.stats.binomial import binomial_lower_bound, binomial_upper_bound
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    correct_judge_risk,
    estimate_judge_calibration,
    estimate_stratified_judge_calibration,
    hypergeometric_count_bounds,
)


def _counts(labelled: int, truly_unsafe: int, size: int) -> JudgeStratumCounts:
    return JudgeStratumCounts(labelled=labelled, truly_unsafe=truly_unsafe, stratum_size=size)


def _hypergeometric_tails(
    unsafe: int, labelled: int, size: int, count: int
) -> tuple[Fraction, Fraction]:
    """``P(X >= unsafe)`` and ``P(X <= unsafe)`` for ``labelled`` draws without replacement
    from ``size`` rows of which ``count`` are unsafe, exactly, from binomial coefficients."""

    total = math.comb(size, labelled)
    pmf = [
        Fraction(math.comb(count, k) * math.comb(size - count, labelled - k), total)
        for k in range(labelled + 1)
    ]
    return sum(pmf[unsafe:], Fraction(0)), sum(pmf[: unsafe + 1], Fraction(0))


@pytest.mark.parametrize(
    ("unsafe", "labelled", "size"),
    [(0, 5, 20), (5, 5, 20), (3, 10, 40), (1, 1, 2), (7, 30, 31)],
)
def test_count_bounds_are_the_extreme_counts_the_tails_admit(
    unsafe: int, labelled: int, size: int
) -> None:
    level = Fraction(1, 20)
    lower, upper = hypergeometric_count_bounds(
        truly_unsafe=unsafe,
        labelled=labelled,
        stratum_size=size,
        alpha_lower=float(level),
        alpha_upper=float(level),
    )
    # Brute force over every count, in exact rational arithmetic rather than with scipy.
    tails = [_hypergeometric_tails(unsafe, labelled, size, count) for count in range(size + 1)]
    admitted_low = [count for count, (upper_tail, _) in enumerate(tails) if upper_tail > level]
    admitted_high = [count for count, (_, lower_tail) in enumerate(tails) if lower_tail > level]
    assert lower == min(admitted_low)
    assert upper == max(admitted_high)
    # The labels already hold `unsafe` truly unsafe rows and `labelled - unsafe` safe ones.
    assert unsafe <= lower <= upper <= size - (labelled - unsafe)


def test_a_census_knows_the_count_but_still_samples_the_rate() -> None:
    assert hypergeometric_count_bounds(
        truly_unsafe=4, labelled=9, stratum_size=9, alpha_lower=0.01, alpha_upper=0.01
    ) == (4, 4)
    unsafe, safe = _counts(12, 9, 12), _counts(40, 0, 40)
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={"all": unsafe},
        safe_strata={"all": safe},
        alpha=0.05,
    )
    assert all(item.alpha == 0 for item in stratified.fine_strata)
    assert [(item.unsafe_lower, item.unsafe_upper) for item in stratified.fine_strata] == [
        (9, 9),
        (0, 0),
    ]
    # The twelve judge-unsafe rollouts only sample P(truly unsafe | judge said unsafe), so the
    # range is the Clopper-Pearson range of 9 in 12 at half an endpoint's quarter of alpha.
    # Each end is where the binomial tail beyond the count equals that level.
    calibration = stratified.calibration
    level = 0.05 / 8
    assert stratified.rate_alpha == pytest.approx(level, rel=0, abs=0)
    assert binom.sf(8, 12, calibration.rho_1_lower) == pytest.approx(level, rel=1e-9, abs=0)
    assert binom.cdf(9, 12, calibration.rho_1_upper) == pytest.approx(level, rel=1e-9, abs=0)
    assert calibration.rho_0_lower == 0.0
    assert binom.cdf(0, 40, calibration.rho_0_upper) == pytest.approx(level, rel=1e-9, abs=0)
    assert calibration.rho_1_lower < 0.75 < calibration.rho_1_upper
    # The same labels read as a simple random sample within each judge stratum spend the whole
    # quarter on that bound, so the stratified range is the wider one.
    binomial = estimate_judge_calibration(
        scenario_id="validated", unsafe_stratum=unsafe, safe_stratum=safe, alpha=0.05
    )
    assert calibration.rho_1_lower < binomial.rho_1_lower
    assert calibration.rho_1_upper > binomial.rho_1_upper
    assert calibration.rho_0_upper > binomial.rho_0_upper


def test_an_unlabelled_stratum_is_completed_without_spending_alpha() -> None:
    assert hypergeometric_count_bounds(
        truly_unsafe=0, labelled=0, stratum_size=7, alpha_lower=0.0, alpha_upper=0.0
    ) == (0, 7)


def test_one_fine_stratum_is_never_narrower_than_the_binomial_calibration() -> None:
    # A single fine stratum is a simple random sample within its judge stratum, which the
    # binomial calibration reads at the whole endpoint budget, while the stratified one splits
    # it between the count bound and the rate bound. The stratified range once narrowed as the
    # labelled share grew, to a single point for a census. A large clean census of the
    # judge-safe stratum keeps every box with an unsafe label informative.
    safe = _counts(100_000, 0, 100_000)
    for size in range(1, 21):
        for labelled in range(1, size + 1):
            for unsafe in range(1, labelled + 1):
                stratum = _counts(labelled, unsafe, size)
                stratified = estimate_stratified_judge_calibration(
                    scenario_id="validated",
                    unsafe_strata={"all": stratum},
                    safe_strata={"all": safe},
                    alpha=0.05,
                ).calibration
                binomial = estimate_judge_calibration(
                    scenario_id="validated", unsafe_stratum=stratum, safe_stratum=safe, alpha=0.05
                )
                assert stratified.rho_1_lower <= binomial.rho_1_lower, (size, labelled, unsafe)
                assert stratified.rho_1_upper >= binomial.rho_1_upper, (size, labelled, unsafe)


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
    # The census is exact and spends nothing; the sampled stratum's count bounds take half of
    # each endpoint, and the Clopper-Pearson bounds on the summed counts the other half.
    assert (by_name["disagree"].unsafe_lower, by_name["disagree"].unsafe_upper) == (4, 4)
    assert by_name["disagree"].alpha == 0
    assert by_name["agree"].alpha == pytest.approx(0.05 / 8, rel=0, abs=0)
    assert by_name["agree"].unsafe_upper == 380
    unsafe_lower = 4 + by_name["agree"].unsafe_lower
    assert binom.sf(unsafe_lower - 1, 400, calibration.rho_1_lower) == pytest.approx(
        0.05 / 8, rel=1e-9, abs=0
    )
    assert binom.cdf(384, 400, calibration.rho_1_upper) == pytest.approx(
        0.05 / 8, rel=1e-9, abs=0
    )
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
    # Each endpoint keeps 0.1 / 4: half for the rate bound, and half split over "a" and "b".
    assert stratified.rate_alpha == pytest.approx(0.1 / 8, rel=0, abs=0)
    assert spent["a"] == spent["b"] == pytest.approx(0.1 / 16, rel=0, abs=0)
    assert spent["census"] == spent["unlabelled"] == 0
    # The judge-safe stratum has one sampled fine stratum, which takes the whole count half.
    assert spent["all"] == pytest.approx(0.1 / 8, rel=0, abs=0)
    assert [item.stratum for item in stratified.unlabelled_fine_strata] == ["unlabelled"]
    calibration = stratified.calibration
    # The unlabelled stratum is completed as [0, 5] of 110 rollouts.
    by_name = {item.stratum: item for item in stratified.fine_strata}
    assert (by_name["unlabelled"].unsafe_lower, by_name["unlabelled"].unsafe_upper) == (0, 5)
    assert calibration.completed_strata == ()


def test_a_census_keeps_only_half_of_each_endpoint_for_the_rate() -> None:
    # The halves do not move when no fine stratum is sampled: which strata a design censuses
    # can depend on their sizes, and the rate bound holds only over those sizes.
    stratified = estimate_stratified_judge_calibration(
        scenario_id="validated",
        unsafe_strata={"x": _counts(30, 25, 30), "y": _counts(10, 2, 10)},
        safe_strata={"all": _counts(200, 3, 200)},
        alpha=0.2,
    )
    calibration = stratified.calibration
    assert calibration.rho_1_lower == binomial_lower_bound(27, 40, alpha=0.2 / 8)
    assert calibration.rho_1_upper == binomial_upper_bound(27, 40, alpha=0.2 / 8)
    assert calibration.rho_0_lower == binomial_lower_bound(3, 200, alpha=0.2 / 8)
    assert calibration.rho_0_upper == binomial_upper_bound(3, 200, alpha=0.2 / 8)


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


# The Monte Carlo tests below draw every rollout anew in each trial: its judge verdict, whether
# a second judge agrees with it, and its truth. The calibration must cover the conditional
# probabilities rho_1 and rho_0, which a run's rollouts only sample, not the share of them that
# is truly unsafe.

_JUDGE_STRATA = ("judge_unsafe", "judge_safe")
_ALL_LABELLED: dict[tuple[str, str], int | None] = {
    (judge, fine): None for judge in _JUDGE_STRATA for fine in ("disagree", "agree")
}


def _rates(
    disagree: dict[str, float], unsafe_given: dict[tuple[str, str], float]
) -> dict[str, float]:
    return {
        judge: disagree[judge] * unsafe_given[(judge, "disagree")]
        + (1 - disagree[judge]) * unsafe_given[(judge, "agree")]
        for judge in _JUDGE_STRATA
    }


def _draw_run(
    rng: np.random.Generator,
    *,
    rollouts: int,
    judge_rate: float,
    disagree: dict[str, float],
    unsafe_given: dict[tuple[str, str], float],
    labels: dict[tuple[str, str], int | None],
) -> tuple[int, dict[str, dict[str, JudgeStratumCounts]]]:
    """One run's judge-unsafe count and its labelled fine strata.

    ``labels`` gives the number of rollouts labelled in each fine stratum, at most all of
    them; ``None`` labels all of them.
    """

    judge_unsafe = int(rng.binomial(rollouts, judge_rate))
    sizes = {"judge_unsafe": judge_unsafe, "judge_safe": rollouts - judge_unsafe}
    strata: dict[str, dict[str, JudgeStratumCounts]] = {name: {} for name in _JUDGE_STRATA}
    for judge in _JUDGE_STRATA:
        disagreements = int(rng.binomial(sizes[judge], disagree[judge]))
        for fine, size in (("disagree", disagreements), ("agree", sizes[judge] - disagreements)):
            if size == 0:
                continue
            truly_unsafe = int(rng.binomial(size, unsafe_given[(judge, fine)]))
            wanted = labels[(judge, fine)]
            labelled = size if wanted is None else min(wanted, size)
            found = int(rng.hypergeometric(truly_unsafe, size - truly_unsafe, labelled))
            strata[judge][fine] = _counts(labelled, found, size)
    return judge_unsafe, strata


def _endpoint_misses(
    *,
    seed: int,
    trials: int,
    alpha: float,
    rollouts: int,
    judge_rate: float,
    disagree: dict[str, float],
    unsafe_given: dict[tuple[str, str], float],
    labels: dict[tuple[str, str], int | None],
) -> list[float]:
    """How often each of the four endpoints misses its rate: rho_1 low and high, then rho_0."""

    rng = np.random.default_rng(seed)
    rates = _rates(disagree, unsafe_given)
    misses = np.zeros(4)
    for _ in range(trials):
        _, strata = _draw_run(
            rng,
            rollouts=rollouts,
            judge_rate=judge_rate,
            disagree=disagree,
            unsafe_given=unsafe_given,
            labels=labels,
        )
        calibration = estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata=strata["judge_unsafe"],
            safe_strata=strata["judge_safe"],
            alpha=alpha,
        ).calibration
        misses += [
            rates["judge_unsafe"] < calibration.rho_1_lower,
            rates["judge_unsafe"] > calibration.rho_1_upper,
            rates["judge_safe"] < calibration.rho_0_lower,
            rates["judge_safe"] > calibration.rho_0_upper,
        ]
    return [float(item) for item in misses / trials]


def _allowed(level: float, trials: int) -> float:
    """A miss rate the level admits: the level plus three binomial standard errors."""

    return level + 3 * math.sqrt(level * (1 - level) / trials)


def test_a_census_misses_each_rate_at_most_at_the_rate_bound_s_level() -> None:
    # With every rollout labelled the count bounds are exact, so an endpoint can miss only
    # through its Clopper-Pearson bound, at half its quarter of alpha: 0.05 here. That bound is
    # near exact at these sizes, so this checks the budget: each endpoint missed 3.65% to 4.2%
    # of 2,000 runs (standard error about 0.5%). The finite-population range K / N missed each
    # about half of the time, and a rate bound given the whole quarter 7.7% to 8.9%.
    alpha, trials = 0.4, 2000
    misses = _endpoint_misses(
        seed=20261119,
        trials=trials,
        alpha=alpha,
        rollouts=400,
        judge_rate=0.3,
        disagree={"judge_unsafe": 0.2, "judge_safe": 0.1},
        unsafe_given={
            ("judge_unsafe", "disagree"): 0.5,
            ("judge_unsafe", "agree"): 0.85,
            ("judge_safe", "disagree"): 0.4,
            ("judge_safe", "agree"): 0.05,
        },
        labels=_ALL_LABELLED,
    )
    assert max(misses) <= _allowed(alpha / 8, trials), misses


def test_a_design_that_samples_the_agreements_covers_each_rate() -> None:
    # The documented design: every disagreement of the two judges is labelled, and a fixed
    # number of the agreements. Each endpoint may miss a quarter of alpha, 0.1 here; each
    # missed 1.2% to 2.15% of 2,000 runs. The finite-population range K / N missed 28.5% to
    # 36.9%, and Clopper-Pearson bounds taken at the opposite count bounds (the lower rate
    # bound at the upper count, and the reverse) up to 20%.
    alpha, trials = 0.4, 2000
    misses = _endpoint_misses(
        seed=20261120,
        trials=trials,
        alpha=alpha,
        rollouts=600,
        judge_rate=0.3,
        disagree={"judge_unsafe": 0.3, "judge_safe": 0.3},
        unsafe_given={
            ("judge_unsafe", "disagree"): 0.4,
            ("judge_unsafe", "agree"): 0.8,
            ("judge_safe", "disagree"): 0.4,
            ("judge_safe", "agree"): 0.05,
        },
        labels={
            ("judge_unsafe", "disagree"): None,
            ("judge_unsafe", "agree"): 90,
            ("judge_safe", "disagree"): None,
            ("judge_safe", "agree"): 220,
        },
    )
    assert max(misses) <= _allowed(alpha / 4, trials), misses


def test_the_corrected_interval_covers_the_true_risk_of_iid_rollouts() -> None:
    # One equal-weight row of 400 iid rollouts, every one labelled, with a two-sided
    # Clopper-Pearson interval on the judge-reported risk R at 0.05 and the calibration at
    # 0.05. The corrected interval claims to miss R* = R rho_1 + (1 - R) rho_0 at most 10% of
    # the time, and each side at most 5%. It never missed in 2,000 runs. With the
    # finite-population range K / N it missed 18.4%: 9.0% below and 9.4% above.
    alpha_observed, alpha_calibration, trials = 0.05, 0.05, 2000
    rollouts, judge_rate = 400, 0.3
    disagree = {"judge_unsafe": 0.2, "judge_safe": 0.1}
    unsafe_given = {
        ("judge_unsafe", "disagree"): 0.5,
        ("judge_unsafe", "agree"): 0.8,
        ("judge_safe", "disagree"): 0.4,
        ("judge_safe", "agree"): 0.06,
    }
    rates = _rates(disagree, unsafe_given)
    true_risk = judge_rate * rates["judge_unsafe"] + (1 - judge_rate) * rates["judge_safe"]
    rng = np.random.default_rng(20261121)
    low = high = 0
    for _ in range(trials):
        judge_unsafe, strata = _draw_run(
            rng,
            rollouts=rollouts,
            judge_rate=judge_rate,
            disagree=disagree,
            unsafe_given=unsafe_given,
            labels=_ALL_LABELLED,
        )
        calibration = estimate_stratified_judge_calibration(
            scenario_id="validated",
            unsafe_strata=strata["judge_unsafe"],
            safe_strata=strata["judge_safe"],
            alpha=alpha_calibration,
        ).calibration
        corrected = correct_judge_risk(
            observed_lower=binomial_lower_bound(
                judge_unsafe, rollouts, alpha=alpha_observed / 2
            ),
            observed_upper=binomial_upper_bound(
                judge_unsafe, rollouts, alpha=alpha_observed / 2
            ),
            observed_estimate=judge_unsafe / rollouts,
            calibration=calibration,
        )
        low += true_risk < corrected.lower
        high += true_risk > corrected.upper
    # Each corrected endpoint uses one endpoint of R and two of the four calibration endpoints.
    one_sided = alpha_observed / 2 + alpha_calibration / 2
    assert low / trials <= _allowed(one_sided, trials)
    assert high / trials <= _allowed(one_sided, trials)
    assert (low + high) / trials <= _allowed(alpha_observed + alpha_calibration, trials)
