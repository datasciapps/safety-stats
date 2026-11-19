"""A judge correction bounds the population share of unclassified rollouts, not the run's.

``correct_judge_risk`` keeps the rollouts with no judge verdict at their worst case and maps
only the classified mass through rho. How much weight those rollouts carry is a parameter of
the population: the run's own share only estimates it, and with the run's share in its place
a corrected upper bound fell below true risk whenever the run saw fewer unclassified rollouts
than the population holds. The Monte Carlo tests here simulate that population and check
coverage at the alpha the correction reports; the others pin each form against an independent
computation. The coverage of a certified generative root through ``correct`` is in
``test_generative_judge_coverage.py``.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import yaml

from inspect_safety_stats.analysis.unclassified_share import (
    PER_PROMPT_METHOD,
    UNBOUNDED_METHOD,
    bounds_complete_unclassified,
    share_alpha,
    unclassified_share_bound,
)
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.generative.inference import unclassified_upper_bound
from inspect_safety_stats.generative.judge import correct
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.stats import JudgeCalibration, correct_judge_risk
from inspect_safety_stats.stats.binomial import binomial_bounds, binomial_upper_bound
from inspect_safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_judge_calibration,
    transport_judge_risk,
)
from tests.generative_fixtures import root_estimate

SEED = 20261119


@pytest.mark.parametrize(
    "kernel",
    ["exact", "equal_coefficients", "unequal_coefficients", "concentrated_coefficients"],
)
def test_the_generative_share_bound_covers_the_design_share(kernel: str) -> None:
    """Seeded Monte Carlo of each kernel ``unclassified_upper_bound`` chooses.

    Draws are independent; each carries one rollout (exact Clopper-Pearson), three with
    equal coefficients (Bernoulli KL over draws), or unequal coefficients (Poisson deviance
    at the largest coefficient). In ``unequal_coefficients`` unclassified rollouts are rarer
    in the heavier draws. In ``concentrated_coefficients`` ten draws hold half the weight
    and most of the unclassified share, so the share varies as over about twenty draws: a
    Bernoulli KL bound over all 200 draws misses in about 15% of runs.
    """

    rng = np.random.default_rng(SEED)
    draws, alpha, replications = 200, 0.05, 3000
    trials = 1 if kernel in {"exact", "concentrated_coefficients"} else 3
    if kernel == "unequal_coefficients":
        raw = np.where(np.arange(draws) < draws // 4, 4.0, 1.0)
        rates = np.where(np.arange(draws) < draws // 4, 0.02, 0.20)
    elif kernel == "concentrated_coefficients":
        raw = np.where(np.arange(draws) < 10, 20.0, 1.0)
        rates = np.where(np.arange(draws) < 10, 0.30, 0.02)
    else:
        raw = np.ones(draws)
        rates = np.full(draws, 0.08)
    coefficients = raw / raw.sum()
    design_share = float(np.dot(coefficients, rates))
    misses = 0
    for _ in range(replications):
        unclassified = rng.binomial(trials, rates)
        share = float(np.dot(coefficients, unclassified / trials))
        root = root_estimate(
            draws=draws,
            trials=draws * trials,
            unclassified=int(unclassified.sum()),
            failures=0,
            upper=1.0,
            share=share,
            equal=kernel in {"exact", "equal_coefficients"},
            maximum_coefficient=float(coefficients.max()),
        )
        bound = unclassified_upper_bound(root, alpha=alpha)
        assert share <= bound <= 1.0
        misses += bound < design_share
    standard_error = math.sqrt(alpha * (1 - alpha) / replications)
    assert misses / replications <= alpha + 3 * standard_error


def test_the_generative_share_bound_is_exact_clopper_pearson_for_single_rollout_draws() -> None:
    root = root_estimate(draws=400, unclassified=7, failures=3, upper=0.1)
    assert unclassified_upper_bound(root, alpha=0.0125) == pytest.approx(
        binomial_upper_bound(7, 400, alpha=0.0125), rel=1e-12, abs=0
    )
    # None observed is not none in the design.
    clean = root_estimate(draws=400, unclassified=0, failures=3, upper=0.1)
    assert unclassified_upper_bound(clean, alpha=0.05) == pytest.approx(
        1 - 0.05 ** (1 / 400), rel=1e-9, abs=0
    )
    with pytest.raises(ValueError, match="alpha"):
        unclassified_upper_bound(root, alpha=0.0)


def test_a_declared_generative_calibration_bounds_the_share_at_the_root_alpha() -> None:
    root = root_estimate(draws=1000, unclassified=100, failures=50, upper=0.2)
    declared = JudgeCalibration("declared", 0.5, 0.7, 0.0, 0.01)
    (corrected,) = correct([root], declared, interval_alpha=0.05, root_id="all")
    assert corrected.eligibility == "diagnostic"
    assert corrected.alpha == 0.05
    assert corrected.unclassified_alpha == 0.05
    share = binomial_upper_bound(100, 1000, alpha=0.05)
    assert corrected.unclassified_upper == pytest.approx(share, rel=1e-12, abs=0)
    # Upper end: the classified judge-unsafe mass at its largest, the share truly unsafe.
    assert corrected.upper == pytest.approx(
        max(
            (0.2 - share) * rho_1 + (1 - 0.2) * rho_0 + share
            for rho_1 in (0.5, 0.7)
            for rho_0 in (0.0, 0.01)
        ),
        rel=1e-12,
        abs=0,
    )


def test_an_excluded_share_is_mixed_in_at_its_worst_case() -> None:
    """Seeded Monte Carlo of the form for bounds that cover only the classified rate.

    One prompt under ``exclude``: 30% of rollouts unclassified, half the rest judge-unsafe,
    60% of those truly unsafe, the judge-safe ones all safe, every classified rollout
    labelled. The judge-system interval is two-sided Clopper-Pearson on the classified
    rollouts. True risk lies between 0.35 * 0.6 (every unclassified rollout safe) and that
    plus 0.30 (every one unsafe), and the corrected interval must hold both. Reading the
    bounds as if they completed the unclassified rollouts puts the classified mass at the
    classified rate, 0.5 instead of 0.35, and its lower end missed in every run.
    """

    rng = np.random.default_rng(SEED)
    rollouts, unclassified_rate, judge_unsafe_rate, rho_1 = 1000, 0.30, 0.35, 0.6
    safe_end = judge_unsafe_rate * rho_1
    unsafe_end = safe_end + unclassified_rate
    interval_alpha = calibration_alpha = 0.05
    shared = calibration_alpha / 4
    replications = 2000
    misses = completion_misses = 0
    for _ in range(replications):
        unclassified, judge_unsafe, judge_safe = (
            int(value)
            for value in rng.multinomial(
                rollouts,
                [unclassified_rate, judge_unsafe_rate, 1 - unclassified_rate - judge_unsafe_rate],
            )
        )
        truly_unsafe = int(rng.binomial(judge_unsafe, rho_1))
        calibration = estimate_judge_calibration(
            scenario_id="labels",
            unsafe_stratum=JudgeStratumCounts(judge_unsafe, truly_unsafe, judge_unsafe),
            safe_stratum=JudgeStratumCounts(judge_safe, 0, judge_safe),
            alpha=calibration_alpha,
        )
        base = binomial_bounds(
            judge_unsafe,
            judge_unsafe + judge_safe,
            alpha_lower=interval_alpha / 2,
            alpha_upper=interval_alpha / 2,
        )
        share = binomial_upper_bound(unclassified, rollouts, alpha=shared)
        for in_bounds in (False, True):
            corrected = correct_judge_risk(
                observed_lower=base.lower,
                observed_upper=base.upper,
                observed_estimate=None,
                calibration=calibration,
                unclassified_share=share,
                unclassified_in_bounds=in_bounds,
            )
            missed = corrected.lower > safe_end or corrected.upper < unsafe_end
            if in_bounds:
                completion_misses += missed
            else:
                misses += missed
    reported = interval_alpha + calibration_alpha + shared
    standard_error = math.sqrt(reported * (1 - reported) / replications)
    assert misses / replications <= reported + 3 * standard_error
    # The completing form is unsound here, which is why the exclude policy must not use it.
    assert completion_misses / replications > 0.5


def test_the_two_forms_match_an_independent_computation() -> None:
    calibration = JudgeCalibration("box", 0.6, 0.9, 0.01, 0.05)
    lower, upper, share = 0.10, 0.30, 0.12
    rho = [(r1, r0) for r1 in (0.6, 0.9) for r0 in (0.01, 0.05)]

    # Completing bounds: M in [L, U - s], s at most each end's own endpoint allows.
    completing = correct_judge_risk(
        observed_lower=lower,
        observed_upper=upper,
        observed_estimate=None,
        calibration=calibration,
        unclassified_share=share,
    )
    assert completing.lower == pytest.approx(
        min(m * r1 + (1 - share - m) * r0 for m in (lower, upper - share) for r1, r0 in rho),
        rel=1e-12,
        abs=0,
    )
    assert completing.upper == pytest.approx(
        max(
            m * r1 + (1 - share - m) * r0 + share
            for m in (lower, upper - share)
            for r1, r0 in rho
        ),
        rel=1e-12,
        abs=0,
    )

    # Bounds on the classified rate Q: (1 - s) A + s u with A = Q rho_1 + (1 - Q) rho_0.
    excluding = correct_judge_risk(
        observed_lower=lower,
        observed_upper=upper,
        observed_estimate=None,
        calibration=calibration,
        unclassified_share=share,
        unclassified_in_bounds=False,
    )
    plain = [q * r1 + (1 - q) * r0 for q in (lower, upper) for r1, r0 in rho]
    assert excluding.lower == pytest.approx((1 - share) * min(plain), rel=1e-12, abs=0)
    assert excluding.upper == pytest.approx(
        (1 - share) * max(plain) + share, rel=1e-12, abs=0
    )
    # The classified mass can sit below the classified rate's lower bound, so the excluding
    # form reaches lower than the completing one.
    assert excluding.lower < completing.lower


def test_a_transported_upper_bound_mixes_in_an_excluded_share() -> None:
    result = transport_judge_risk(
        observed_upper=0.3548,
        observed_estimate=None,
        sensitivity_lower=0.7671,
        unclassified_share=0.5,
    )
    assert result.upper == pytest.approx(0.5 * (0.3548 / 0.7671) + 0.5, rel=1e-12, abs=0)
    # Every unclassified rollout may be truly unsafe, so the bound is at least the share.
    assert result.upper >= 0.5
    assert transport_judge_risk(
        observed_upper=0.9, observed_estimate=None, sensitivity_lower=0.5, unclassified_share=0.3
    ).upper == 1.0
    with pytest.raises(ValueError, match="unclassified share"):
        transport_judge_risk(
            observed_upper=0.3, observed_estimate=None, sensitivity_lower=0.5, unclassified_share=2
        )


PLAN = """
analysis: {id: share, title: Share, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
outputs: {prompt_bounds: {enabled: false}}
multiplicity: {families: []}
claims: []
"""


def _plan(**score: str) -> AnalysisPlan:
    data = yaml.safe_load(PLAN)
    data["inputs"]["score"].update(score)
    return AnalysisPlan.model_validate(data)


class _Counts:
    def __init__(self, classified: int, attempted: int) -> None:
        self.classified = classified
        self.attempted = attempted


@pytest.mark.parametrize(
    ("missing", "errors", "complete"),
    [
        ("conservative_bounds", "end_to_end", True),
        ("count_as_failure", "count_as_failure", True),
        ("block", "block", True),
        ("exclude", "block", False),
        ("conservative_bounds", "exclude", False),
    ],
)
def test_only_an_excluding_policy_reads_the_bounds_as_the_classified_rate(
    missing: str, errors: str, complete: bool
) -> None:
    assert bounds_complete_unclassified(_plan(missing=missing, errors=errors)) is complete


def test_a_fixed_design_share_is_bounded_prompt_by_prompt() -> None:
    """Without a declared independence profile, each prompt gets an exact bound.

    The bounds hold together by Bonferroni at the share alpha; a weighted mean sums them and
    a tail measure takes the largest.
    """

    plan = _plan()
    data = cast(NormalizedData, None)  # Only a declared profile reads the observations.
    rollouts = {"p1": _Counts(18, 20), "p2": _Counts(40, 40), "p3": _Counts(0, 0)}
    weights = {"p1": 0.5, "p2": 0.3, "p3": 0.2, "p4": 0.0}
    mean = unclassified_share_bound(
        plan,
        data,
        rollouts,
        cohort_id="candidate",
        weights=weights,
        measure_type="weighted_mean",
        independence_profile=None,
        alpha=0.03,
    )
    p1 = binomial_upper_bound(2, 20, alpha=0.01)
    p2 = binomial_upper_bound(0, 40, alpha=0.01)
    assert mean.method == PER_PROMPT_METHOD
    assert mean.prompts == 3
    assert mean.in_bounds
    assert mean.observed == pytest.approx(0.5 * 0.1, rel=1e-12, abs=0)
    # A weighted prompt with no rollout counts as wholly unclassified.
    assert mean.upper == pytest.approx(0.5 * p1 + 0.3 * p2 + 0.2, rel=1e-12, abs=0)

    tail = unclassified_share_bound(
        plan,
        data,
        {"p1": _Counts(18, 20), "p2": _Counts(40, 40)},
        cohort_id="candidate",
        weights={"p1": 0.5, "p2": 0.5},
        measure_type="maximum",
        independence_profile=None,
        alpha=0.02,
    )
    assert tail.observed == pytest.approx(0.1, rel=1e-12, abs=0)
    assert tail.upper == pytest.approx(
        max(binomial_upper_bound(2, 20, alpha=0.01), binomial_upper_bound(0, 40, alpha=0.01)),
        rel=1e-12,
        abs=0,
    )


def test_a_sampled_design_takes_its_share_at_one() -> None:
    data = yaml.safe_load(PLAN)
    data["design"] = {"type": "sampled", "inclusion_probability_column": "pi"}
    plan = AnalysisPlan.model_validate(data)
    share = unclassified_share_bound(
        plan,
        cast(NormalizedData, None),
        {"p1": _Counts(20, 20)},
        cohort_id="candidate",
        weights={"p1": 1.0},
        measure_type="weighted_mean",
        independence_profile=None,
        alpha=0.05,
    )
    assert (share.observed, share.upper, share.method) == (0.0, 1.0, UNBOUNDED_METHOD)


def test_a_fixed_design_share_bound_covers_the_population_share() -> None:
    """Seeded Monte Carlo of the prompt-by-prompt bound on a weighted mean and a maximum."""

    rng = np.random.default_rng(SEED)
    plan = _plan()
    rates = {"p1": 0.02, "p2": 0.10, "p3": 0.25}
    sizes = {"p1": 30, "p2": 60, "p3": 15}
    weights = {"p1": 0.5, "p2": 0.3, "p3": 0.2}
    alpha, replications = 0.05, 3000
    population = sum(weights[prompt] * rates[prompt] for prompt in rates)
    misses = {"weighted_mean": 0, "maximum": 0}
    for _ in range(replications):
        rollouts = {
            prompt: _Counts(sizes[prompt] - int(rng.binomial(sizes[prompt], rate)), sizes[prompt])
            for prompt, rate in rates.items()
        }
        for measure, target in (("weighted_mean", population), ("maximum", max(rates.values()))):
            bound = unclassified_share_bound(
                plan,
                cast(NormalizedData, None),
                rollouts,
                cohort_id="candidate",
                weights=weights,
                measure_type=measure,
                independence_profile=None,
                alpha=alpha,
            )
            misses[measure] += bound.upper < target
    standard_error = math.sqrt(alpha * (1 - alpha) / replications)
    for count in misses.values():
        assert count / replications <= alpha + 3 * standard_error


def test_the_share_alpha_is_a_quarter_of_a_measured_budget_or_the_row_level() -> None:
    from inspect_safety_stats.schema.reports import RiskIntervalResult

    assert share_alpha(0.05, None) == 0.0125
    base = RiskIntervalResult(
        interval_id="primary",
        lower=0.0,
        upper=0.1,
        method="m",
        role="primary",
        eligibility="certifying",
        family_alpha=0.06,
    )
    assert share_alpha(None, base) == 0.03
    directional = base.model_copy(update={"directional_family_alpha": 0.02})
    assert share_alpha(None, directional) == 0.02
    assert share_alpha(None, None) == 0.025
    assert share_alpha(0.0, None) == 0.025
