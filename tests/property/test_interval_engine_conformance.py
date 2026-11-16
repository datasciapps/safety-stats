"""Conformance checks every direct-mean and comparison engine must pass.

The engines are registered in ``tests/conformance.py`` and every check below runs over every
registered engine, so a new engine meets the same bar by adding one registry entry. The
checks fall into three groups.

- **Coverage.** Exact miss probabilities by enumerating every outcome of small designs,
  exact miss probabilities at a few hundred rollouts where the bound depends only on the
  total failure count, dominance over Clopper-Pearson, and a seeded simulation for unequal
  weights.
- **Structure.** Intervals are ordered, nest as alpha falls, rise with failures, ignore the
  order of prompts, reflect when failures and successes are relabelled, and do not change
  when a prompt's rollouts are split into two prompts of proportional weight.
- **Agreement.** The count engines, which comparisons and the generative path call, equal
  the bounded engines, which the fixed-design pipeline calls, on the same data.

Coverage is checked under the dependence each unit admits. The rollout unit assumes
independent rollouts. The prompt-cluster unit allows any dependence within a prompt, so it is
also checked with every rollout of a prompt sharing one outcome, and, for comparisons, with
the two arms of a prompt as negatively dependent as possible.
"""

from __future__ import annotations

import math
from typing import Any, get_args

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom

import inspect_safety_stats.stats as stats
from inspect_safety_stats.config.models import (
    ComparisonSetSpec,
    ComparisonSpec,
    DirectMeanIntervalsSpec,
)
from inspect_safety_stats.stats import (
    binomial_bounds,
    direct_ratio_bounds,
    direct_risk_difference_bounds,
)
from tests.conformance import (
    BOUNDED_ENGINES,
    COMPARISON_METHODS,
    COUNT_ENGINES,
    METHODS,
    MISS_TOLERANCE,
    CountEngine,
    Unit,
    WithinPrompt,
    call_bounded_engine,
    call_count_engine,
    count_engine,
    count_outcomes,
    countermonotone_arm_probabilities,
    outcome_probabilities,
    poisson_binomial_pmf,
    rate_grid,
    spread,
    worst_misses,
)

TOLERANCE = 1e-12

# Unequal pairs catch an engine that spends one direction's alpha on the other.
ALPHA_PAIRS = ((0.05, 0.05), (0.25, 0.25), (0.02, 0.2))

# Each unit under the dependence it admits.
UNIT_SETTINGS: tuple[tuple[Unit, WithinPrompt], ...] = (
    ("rollout", "independent"),
    ("prompt_cluster", "independent"),
    ("prompt_cluster", "identical"),
)


def _engine_id(engine: Any) -> str:
    return str(engine.method)


# --- Registry ------------------------------------------------------------------------------

# Public engines for a different estimand, with their own coverage tests.
_OTHER_ESTIMANDS = {
    # The mean of a sampled frame under declared inclusion probabilities: tests/unit/
    # test_design_based.py.
    "horvitz_thompson_mean_bounds",
}


def _literal_values(model: Any, field: str) -> set[str]:
    values = get_args(model.model_fields[field].annotation)
    # list[Literal[...]] wraps the literal once more.
    if len(values) == 1 and get_args(values[0]):
        values = get_args(values[0])
    return set(values)


def test_the_registry_lists_every_engine_a_plan_can_select() -> None:
    """A method a plan accepts, or a public engine, must be registered to be checked."""

    def selectable(*models: Any) -> set[str]:
        values: set[str] = set()
        for model in models:
            values |= _literal_values(model, "method") | _literal_values(model, "reference_methods")
        return values

    assert selectable(DirectMeanIntervalsSpec) == set(METHODS)
    assert selectable(ComparisonSpec, ComparisonSetSpec) == set(COMPARISON_METHODS)
    assert {engine.method for engine in BOUNDED_ENGINES} == set(METHODS)

    registered = {engine.function for engine in (*COUNT_ENGINES, *BOUNDED_ENGINES)}
    public = {
        name
        for name in stats.__all__
        if name.endswith("_mean_bounds") and name not in _OTHER_ESTIMANDS
    }
    unregistered = {name for name in public if getattr(stats, name) not in registered}
    assert not unregistered, f"register these engines in tests/conformance.py: {unregistered}"


# --- Exact coverage ------------------------------------------------------------------------

# One prompt; equal weights, where KL/Chernoff uses its Bernoulli kernel and is tightest;
# equal prompt weights with unequal rollout counts, and unequal weights, where it falls back
# to the Poisson kernel.
SMALL_DESIGNS: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = (
    ((6,), (1.0,)),
    ((3, 3, 3), (1 / 3, 1 / 3, 1 / 3)),
    ((1, 5), (0.5, 0.5)),
    ((4, 2), (0.8, 0.2)),
    ((2, 2, 3), (0.2, 0.3, 0.5)),
)
# Dense near both ends, where sparse failures and near-certain failures live.
RATE_LEVELS = (0.0, 0.002, 0.02, 0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.92, 0.98, 1.0)
# The grid has one axis per prompt, so beyond three prompts it is coarser.
COARSE_RATE_LEVELS = (0.0, 0.05, 0.35, 0.65, 0.95, 1.0)


def _levels(prompts: int) -> tuple[float, ...]:
    return RATE_LEVELS if prompts <= 3 else COARSE_RATE_LEVELS


@pytest.mark.parametrize(("unit", "within"), UNIT_SETTINGS)
@pytest.mark.parametrize("engine", COUNT_ENGINES, ids=_engine_id)
def test_count_engines_cover_exactly_on_small_designs(
    engine: CountEngine, unit: Unit, within: WithinPrompt
) -> None:
    """Each endpoint misses with probability at most its alpha, at every rate on the grid.

    The miss probabilities are exact: every outcome of the design is enumerated. KL/Chernoff
    on three equal prompts of three rollouts reaches 86% of its alpha, so a bound only
    slightly too narrow fails here.
    """

    for trials, weights in SMALL_DESIGNS:
        outcomes = count_outcomes(trials, within)
        rates = rate_grid(_levels(len(trials)), len(trials))
        probabilities = outcome_probabilities(outcomes, trials, rates, within)
        truth = rates @ np.asarray(weights)
        for alpha_lower, alpha_upper in ALPHA_PAIRS:
            bounds = [
                call_count_engine(
                    engine,
                    row,
                    trials,
                    weights,
                    alpha_lower=alpha_lower,
                    alpha_upper=alpha_upper,
                    unit=unit,
                )
                for row in outcomes
            ]
            worst = worst_misses(
                np.asarray([bound.lower for bound in bounds]),
                np.asarray([bound.upper for bound in bounds]),
                probabilities,
                truth,
                rates,
            )
            context = (trials, weights, alpha_lower, alpha_upper, worst)
            assert worst.lower <= alpha_lower + MISS_TOLERANCE, context
            assert worst.upper <= alpha_upper + MISS_TOLERANCE, context


# Prompts of equal rollout weight but different failure rates: the fixed-design setting, in
# which rollouts are independent but not identically distributed.
SCALE_DESIGNS: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...] = (
    ((200,), (0.03,)),
    ((40, 40, 40, 40, 40), (0.0, 0.01, 0.02, 0.05, 0.12)),
    ((100, 100), (0.001, 0.05)),
    ((50, 50, 50, 50), (0.0, 0.0, 0.0, 0.04)),
    ((30, 30, 30, 30), (0.3, 0.5, 0.6, 0.9)),
)


@pytest.mark.parametrize("engine", COUNT_ENGINES, ids=_engine_id)
def test_count_engines_cover_heterogeneous_rates_exactly_at_scale(engine: CountEngine) -> None:
    """Exact coverage at a few hundred rollouts, where the variance terms decide the width.

    On the small designs the empirical-Bernstein variance term is capped and the engine is
    far from its alpha. Here it is not. With equal weight per rollout every engine depends on
    the total failure count only, which this test also checks, so the exact distribution of
    the total, a convolution of binomials, gives the exact miss probabilities.
    """

    for trials, rates in SCALE_DESIGNS:
        total_trials = sum(trials)
        weights = [n / total_trials for n in trials]
        truth = float(np.dot(weights, rates))
        pmf = poisson_binomial_pmf(trials, rates)
        for alpha_lower, alpha_upper in ALPHA_PAIRS:
            lower_miss = upper_miss = 0.0
            for total, probability in enumerate(pmf):
                bounds = [
                    call_count_engine(
                        engine,
                        spread(total, trials, from_front=from_front),
                        trials,
                        weights,
                        alpha_lower=alpha_lower,
                        alpha_upper=alpha_upper,
                        unit="rollout",
                    )
                    for from_front in (True, False)
                ]
                assert bounds[0].lower == pytest.approx(bounds[1].lower, rel=0, abs=TOLERANCE)
                assert bounds[0].upper == pytest.approx(bounds[1].upper, rel=0, abs=TOLERANCE)
                if bounds[0].lower > truth + MISS_TOLERANCE:
                    lower_miss += probability
                if bounds[0].upper < truth - MISS_TOLERANCE:
                    upper_miss += probability
            context = (trials, rates, alpha_lower, alpha_upper, lower_miss, upper_miss)
            assert lower_miss <= alpha_lower + MISS_TOLERANCE, context
            assert upper_miss <= alpha_upper + MISS_TOLERANCE, context


@pytest.mark.parametrize("engine", COUNT_ENGINES, ids=_engine_id)
@pytest.mark.parametrize("trials", [1, 2, 7, 30, 250, 2000])
def test_count_engines_never_sit_inside_clopper_pearson(engine: CountEngine, trials: int) -> None:
    """On one prompt, no valid bound that rises with the failure count is tighter than exact.

    If an upper bound ``U`` rises with the count and ``U(x) < CP(x)`` for some ``x``, take a
    rate ``p`` between them. Every count up to ``x`` then misses ``p``, and those counts have
    probability above alpha because ``p`` is below the Clopper-Pearson bound, so ``U`` does
    not cover. Monotonicity is checked separately, so this is a necessary condition for
    coverage that holds at any number of rollouts, far beyond what enumeration reaches.
    """

    for alpha in (1e-4, 0.01, 0.05, 0.3):
        for failures in range(trials + 1):
            exact = binomial_bounds(failures, trials, alpha_lower=alpha, alpha_upper=alpha)
            bounds = call_count_engine(
                engine, [failures], [trials], [1.0], alpha_lower=alpha, alpha_upper=alpha,
                unit="rollout",
            )
            context = (failures, trials, alpha, bounds.lower, bounds.upper, exact)
            assert bounds.upper >= exact.upper - 1e-9, context
            assert bounds.lower <= exact.lower + 1e-9, context


# Two blocks are the most dispersed values a [0, 1] block can take, and the unequal weights
# exercise the KL/Chernoff Poisson kernel.
BLOCK_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (1.0,),
    (1 / 3, 1 / 3, 1 / 3),
    (0.8, 0.2),
    (0.2, 0.3, 0.5),
    (0.5, 0.25, 0.125, 0.125),
    (1 / 6,) * 6,
)


@pytest.mark.parametrize("engine", BOUNDED_ENGINES, ids=_engine_id)
def test_bounded_engines_cover_exactly_on_two_point_blocks(engine: Any) -> None:
    """The engines the fixed-design pipeline calls, checked exactly on independent blocks."""

    for weights in BLOCK_WEIGHTS:
        trials = (1,) * len(weights)
        outcomes = count_outcomes(trials, "independent")
        rates = rate_grid(_levels(len(weights)), len(weights))
        probabilities = outcome_probabilities(outcomes, trials, rates, "independent")
        truth = rates @ np.asarray(weights)
        for alpha_lower, alpha_upper in ALPHA_PAIRS:
            bounds = [
                call_bounded_engine(
                    engine, row, row, weights, alpha_lower=alpha_lower, alpha_upper=alpha_upper
                )
                for row in outcomes.astype(float)
            ]
            worst = worst_misses(
                np.asarray([bound.lower for bound in bounds]),
                np.asarray([bound.upper for bound in bounds]),
                probabilities,
                truth,
                rates,
            )
            context = (weights, alpha_lower, alpha_upper, worst)
            assert worst.lower <= alpha_lower + MISS_TOLERANCE, context
            assert worst.upper <= alpha_upper + MISS_TOLERANCE, context


COMPARISON_DESIGNS: tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]], ...] = (
    ((3,), (4,), (1.0,)),
    ((2, 3), (3, 2), (0.4, 0.6)),
    ((2, 2), (2, 2), (0.5, 0.5)),
)
ARM_RATE_LEVELS = (0.0, 0.05, 0.3, 0.6, 0.95, 1.0)
COMPARISON_ALPHA_PAIRS = ((0.05, 0.05), (0.02, 0.2))
# The prompt-cluster unit allows any dependence between the two arms of a prompt, and the
# most negative dependence spreads the difference widest.
COMPARISON_SETTINGS: tuple[tuple[Unit, str], ...] = (
    *UNIT_SETTINGS,
    ("prompt_cluster", "countermonotone_arms"),
)


def _comparison_probabilities(
    outcomes: np.ndarray,
    candidate_trials: tuple[int, ...],
    baseline_trials: tuple[int, ...],
    rates: np.ndarray,
    within: str,
) -> np.ndarray:
    if within == "countermonotone_arms":
        return countermonotone_arm_probabilities(outcomes, candidate_trials, baseline_trials, rates)
    trials = (*candidate_trials, *baseline_trials)
    return outcome_probabilities(outcomes, trials, rates, within)  # type: ignore[arg-type]


@pytest.mark.parametrize(("unit", "within"), COMPARISON_SETTINGS)
@pytest.mark.parametrize("method", COMPARISON_METHODS)
def test_comparisons_cover_exactly_on_small_designs(method: str, unit: Unit, within: str) -> None:
    """Risk differences, risk ratios and relative risk reductions, each endpoint exactly.

    A ratio endpoint with no finite bound is an infinite one, so it never misses. Rates with
    a zero baseline risk have no ratio and are left out of the ratio checks.
    """

    for candidate_trials, baseline_trials, weights in COMPARISON_DESIGNS:
        prompts = len(weights)
        trials = (*candidate_trials, *baseline_trials)
        outcomes = count_outcomes(trials, "independent" if within == "independent" else "identical")
        rates = rate_grid(ARM_RATE_LEVELS, 2 * prompts)
        probabilities = _comparison_probabilities(
            outcomes, candidate_trials, baseline_trials, rates, within
        )
        candidate_risk = rates[:, :prompts] @ np.asarray(weights)
        baseline_risk = rates[:, prompts:] @ np.asarray(weights)
        defined = baseline_risk > 0
        ratio = np.divide(
            candidate_risk, baseline_risk, where=defined, out=np.zeros_like(candidate_risk)
        )
        zeros = np.zeros(prompts, dtype=int)
        for alpha_lower, alpha_upper in COMPARISON_ALPHA_PAIRS:
            endpoints: dict[str, list[tuple[float, float]]] = {
                "risk_difference": [],
                "risk_ratio": [],
                "relative_risk_reduction": [],
            }
            for row in outcomes:
                arms = (
                    row[:prompts], zeros, np.asarray(candidate_trials),
                    row[prompts:], zeros, np.asarray(baseline_trials),
                    np.asarray(weights),
                )
                options: dict[str, Any] = {
                    "alpha_lower": alpha_lower,
                    "alpha_upper": alpha_upper,
                    "method": method,
                    "independence_unit": unit,
                }
                difference = direct_risk_difference_bounds(*arms, **options)
                endpoints["risk_difference"].append((difference.lower, difference.upper))
                for estimand in ("risk_ratio", "relative_risk_reduction"):
                    bounds = direct_ratio_bounds(*arms, **options, estimand=estimand)
                    endpoints[estimand].append(
                        (
                            -math.inf if bounds.lower is None else bounds.lower,
                            math.inf if bounds.upper is None else bounds.upper,
                        )
                    )
            truths = {
                "risk_difference": (candidate_risk - baseline_risk, np.ones_like(defined)),
                "risk_ratio": (ratio, defined),
                "relative_risk_reduction": (1 - ratio, defined),
            }
            for estimand, (truth, rows) in truths.items():
                lower, upper = (np.asarray(side) for side in zip(*endpoints[estimand], strict=True))
                worst = worst_misses(lower, upper, probabilities[rows], truth[rows], rates[rows])
                context = (estimand, candidate_trials, baseline_trials, weights, alpha_lower,
                           alpha_upper, worst)
                assert worst.lower <= alpha_lower + MISS_TOLERANCE, context
                assert worst.upper <= alpha_upper + MISS_TOLERANCE, context


# --- Simulated coverage under unequal weights ----------------------------------------------

SIMULATION_TRIALS = (20, 5, 40, 10, 3, 25, 8, 15)
SIMULATION_WEIGHTS = (0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05)
SIMULATION_RATES = (0.02, 0.4, 0.0, 0.1, 0.9, 0.05, 0.3, 0.01)
SIMULATION_REPLICATIONS = 1500
SIMULATION_ALPHA = 0.05


@pytest.mark.parametrize(("unit", "within"), UNIT_SETTINGS)
@pytest.mark.parametrize("engine", COUNT_ENGINES, ids=_engine_id)
def test_count_engines_cover_unequal_weights_in_simulation(
    engine: CountEngine, unit: Unit, within: WithinPrompt
) -> None:
    """Seeded simulation for a design too large to enumerate, with heavy and light prompts.

    At 1,500 replications the standard error of a miss rate of 0.05 is 0.0056. The limit is
    the 99.9th percentile of the miss count when an endpoint misses at exactly its alpha, so
    an engine that misses noticeably more often than alpha fails, and the seed keeps the
    result the same on every run.
    """

    generator = np.random.default_rng(20261111)
    trials = np.asarray(SIMULATION_TRIALS)
    rates = np.asarray(SIMULATION_RATES)
    truth = float(np.dot(SIMULATION_WEIGHTS, rates))
    lower_misses = upper_misses = 0
    for _ in range(SIMULATION_REPLICATIONS):
        if within == "independent":
            failures = generator.binomial(trials, rates)
        else:
            failures = trials * (generator.random(len(trials)) < rates)
        bounds = call_count_engine(
            engine,
            failures.tolist(),
            SIMULATION_TRIALS,
            SIMULATION_WEIGHTS,
            alpha_lower=SIMULATION_ALPHA,
            alpha_upper=SIMULATION_ALPHA,
            unit=unit,
        )
        lower_misses += bounds.lower > truth + MISS_TOLERANCE
        upper_misses += bounds.upper < truth - MISS_TOLERANCE
    limit = binom.ppf(0.999, SIMULATION_REPLICATIONS, SIMULATION_ALPHA)
    assert lower_misses <= limit, (lower_misses, limit)
    assert upper_misses <= limit, (upper_misses, limit)


# --- Structure -----------------------------------------------------------------------------

ALPHAS = st.floats(min_value=1e-4, max_value=0.3, allow_nan=False)
UNITS = st.sampled_from(["rollout", "prompt_cluster"])


@st.composite
def count_designs(draw: st.DrawFn, size: int | None = None) -> dict[str, list[Any]]:
    """Prompts with failures, unclassified outcomes and trials, and weights summing to one.

    Weights are log-uniform, so a few heavy prompts next to many light ones are common.
    """

    if size is None:
        size = draw(st.integers(min_value=1, max_value=5))
    trials = draw(st.lists(st.sampled_from([1, 2, 3, 7, 20, 150]), min_size=size, max_size=size))
    failures: list[int] = []
    unclassified: list[int] = []
    for n in trials:
        unknown = draw(st.integers(min_value=0, max_value=min(2, n)))
        failures.append(draw(st.integers(min_value=0, max_value=n - unknown)))
        unclassified.append(unknown)
    exponents = st.floats(min_value=-3.0, max_value=0.0, allow_nan=False)
    raw = [10**exponent for exponent in draw(st.lists(exponents, min_size=size, max_size=size))]
    total = math.fsum(raw)
    return {
        "failures": failures,
        "unclassified": unclassified,
        "trials": trials,
        "weights": [value / total for value in raw],
    }


def _call(
    engine: CountEngine, design: dict[str, list[Any]], alpha_lower: float, alpha_upper: float,
    unit: Unit,
) -> Any:
    return call_count_engine(
        engine,
        design["failures"],
        design["trials"],
        design["weights"],
        unclassified=design["unclassified"],
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        unit=unit,
    )


ENGINES = st.sampled_from(COUNT_ENGINES)


@settings(deadline=None)
@given(engine=ENGINES, design=count_designs(), alpha=ALPHAS, unit=UNITS)
def test_intervals_are_ordered_around_their_completion_estimates(
    engine: CountEngine, design: dict[str, list[Any]], alpha: float, unit: Unit
) -> None:
    bounds = _call(engine, design, alpha, alpha, unit)
    assert 0.0 <= bounds.lower <= bounds.completion_estimate_lower + TOLERANCE
    assert bounds.completion_estimate_lower <= bounds.completion_estimate_upper + TOLERANCE
    assert bounds.completion_estimate_upper <= bounds.upper + TOLERANCE
    assert bounds.upper <= 1.0


@settings(deadline=None)
@given(
    engine=ENGINES,
    design=count_designs(),
    alpha=ALPHAS,
    factor=st.floats(min_value=0.01, max_value=1.0),
    unit=UNITS,
)
def test_intervals_widen_as_alpha_falls(
    engine: CountEngine, design: dict[str, list[Any]], alpha: float, factor: float, unit: Unit
) -> None:
    wide = _call(engine, design, alpha * factor, alpha * factor, unit)
    narrow = _call(engine, design, alpha, alpha, unit)
    assert wide.lower <= narrow.lower + TOLERANCE
    assert wide.upper >= narrow.upper - TOLERANCE


@settings(deadline=None)
@given(engine=ENGINES, design=count_designs(), alpha=ALPHAS, unit=UNITS, data=st.data())
def test_both_bounds_rise_with_a_failure(
    engine: CountEngine,
    design: dict[str, list[Any]],
    alpha: float,
    unit: Unit,
    data: st.DataObject,
) -> None:
    """Turning one classified success into a failure never lowers either bound.

    This is what lets a missing label be completed adversarially, and what keeps a verdict
    from improving when more failures are observed.
    """

    room = [
        prompt
        for prompt, n in enumerate(design["trials"])
        if design["failures"][prompt] + design["unclassified"][prompt] < n
    ]
    if not room:
        return
    prompt = data.draw(st.sampled_from(room))
    worse = {**design, "failures": list(design["failures"])}
    worse["failures"][prompt] += 1
    before = _call(engine, design, alpha, alpha, unit)
    after = _call(engine, worse, alpha, alpha, unit)
    assert after.lower >= before.lower - TOLERANCE
    assert after.upper >= before.upper - TOLERANCE


@settings(deadline=None)
@given(engine=ENGINES, design=count_designs(), alpha=ALPHAS, unit=UNITS, data=st.data())
def test_the_order_of_prompts_does_not_matter(
    engine: CountEngine,
    design: dict[str, list[Any]],
    alpha: float,
    unit: Unit,
    data: st.DataObject,
) -> None:
    order = data.draw(st.permutations(range(len(design["trials"]))))
    shuffled = {key: [values[index] for index in order] for key, values in design.items()}
    before = _call(engine, design, alpha, alpha, unit)
    after = _call(engine, shuffled, alpha, alpha, unit)
    assert after.lower == pytest.approx(before.lower, rel=0, abs=TOLERANCE)
    assert after.upper == pytest.approx(before.upper, rel=0, abs=TOLERANCE)


@settings(deadline=None)
@given(
    engine=ENGINES,
    design=count_designs(),
    alpha_lower=ALPHAS,
    alpha_upper=ALPHAS,
    unit=UNITS,
)
def test_relabelling_failures_as_successes_reflects_the_interval(
    engine: CountEngine,
    design: dict[str, list[Any]],
    alpha_lower: float,
    alpha_upper: float,
    unit: Unit,
) -> None:
    """``lower(x) = 1 - upper(n - x - u)`` with the alphas swapped, and the same for upper.

    The check catches a sign or direction slip that coverage tests can miss when the slip
    only makes one side conservative. KL/Chernoff is symmetric only under equal atomic
    weights, so it is checked with those.
    """

    if not engine.always_mirror_symmetric:
        trials = design["trials"]
        atoms = [n / sum(trials) for n in trials] if unit == "rollout" else [1 / len(trials)] * len(
            trials
        )
        design = {**design, "weights": atoms}
    mirrored = {
        **design,
        "failures": [
            n - x - u
            for x, u, n in zip(
                design["failures"], design["unclassified"], design["trials"], strict=True
            )
        ],
    }
    bounds = _call(engine, design, alpha_lower, alpha_upper, unit)
    reflected = _call(engine, mirrored, alpha_upper, alpha_lower, unit)
    assert bounds.lower == pytest.approx(1 - reflected.upper, rel=0, abs=1e-9)
    assert bounds.upper == pytest.approx(1 - reflected.lower, rel=0, abs=1e-9)


@settings(deadline=None)
@given(engine=ENGINES, design=count_designs(), alpha=ALPHAS, data=st.data())
def test_splitting_a_prompts_rollouts_into_two_prompts_changes_nothing(
    engine: CountEngine, design: dict[str, list[Any]], alpha: float, data: st.DataObject
) -> None:
    """Under the rollout unit only the rollouts and their weights matter, not the grouping.

    Splitting a prompt's rollouts into two prompts, each with the weight of its share of the
    rollouts, leaves every rollout's weight and outcome unchanged, so the interval must not
    move. A per-prompt term that ought to be per-rollout breaks this.
    """

    splittable = [prompt for prompt, n in enumerate(design["trials"]) if n >= 2]
    if not splittable:
        return
    prompt = data.draw(st.sampled_from(splittable))
    n, x, u = (design[key][prompt] for key in ("trials", "failures", "unclassified"))
    first_trials = data.draw(st.integers(min_value=1, max_value=n - 1))
    second_trials = n - first_trials
    first_failures = data.draw(
        st.integers(min_value=max(0, x - second_trials), max_value=min(x, first_trials))
    )
    first_unknown = data.draw(
        st.integers(
            min_value=max(0, u - (second_trials - (x - first_failures))),
            max_value=min(u, first_trials - first_failures),
        )
    )
    weight = design["weights"][prompt]
    split = {key: list(values) for key, values in design.items()}
    split["trials"][prompt : prompt + 1] = [first_trials, second_trials]
    split["failures"][prompt : prompt + 1] = [first_failures, x - first_failures]
    split["unclassified"][prompt : prompt + 1] = [first_unknown, u - first_unknown]
    split["weights"][prompt : prompt + 1] = [weight * first_trials / n, weight * second_trials / n]
    before = _call(engine, design, alpha, alpha, "rollout")
    after = _call(engine, split, alpha, alpha, "rollout")
    assert after.lower == pytest.approx(before.lower, rel=0, abs=1e-9)
    assert after.upper == pytest.approx(before.upper, rel=0, abs=1e-9)


ZERO_WEIGHT_INVARIANT = [engine for engine in COUNT_ENGINES if engine.zero_weight_invariant]


@settings(deadline=None)
@given(
    engine=st.sampled_from(ZERO_WEIGHT_INVARIANT),
    design=count_designs(),
    alpha=ALPHAS,
    unit=UNITS,
    extra_trials=st.integers(min_value=1, max_value=20),
    data=st.data(),
)
def test_a_prompt_without_weight_changes_nothing(
    engine: CountEngine,
    design: dict[str, list[Any]],
    alpha: float,
    unit: Unit,
    extra_trials: int,
    data: st.DataObject,
) -> None:
    extra_failures = data.draw(st.integers(min_value=0, max_value=extra_trials))
    padded = {
        "failures": [*design["failures"], extra_failures],
        "unclassified": [*design["unclassified"], 0],
        "trials": [*design["trials"], extra_trials],
        "weights": [*design["weights"], 0.0],
    }
    before = _call(engine, design, alpha, alpha, unit)
    after = _call(engine, padded, alpha, alpha, unit)
    assert after.lower == pytest.approx(before.lower, rel=0, abs=TOLERANCE)
    assert after.upper == pytest.approx(before.upper, rel=0, abs=TOLERANCE)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A zero-weight prompt fails the equal-atomic-weight check, so KL/Chernoff switches "
        "to its wider Poisson kernel. Conservative, not invalid."
    ),
)
@pytest.mark.parametrize("unit", ["rollout", "prompt_cluster"])
def test_a_prompt_without_weight_does_not_change_the_kl_chernoff_kernel(unit: Unit) -> None:
    engine = count_engine("weighted_kl_chernoff")
    options: dict[str, Any] = {"alpha_lower": 0.05, "alpha_upper": 0.05, "unit": unit}
    alone = call_count_engine(engine, [1], [4], [1.0], **options)
    padded = call_count_engine(engine, [1, 2], [4, 3], [1.0, 0.0], **options)
    assert padded.chernoff_kernel == alone.chernoff_kernel
    assert padded.upper == pytest.approx(alone.upper, rel=0, abs=TOLERANCE)


# --- Agreement between the count and bounded engines ---------------------------------------


@settings(deadline=None)
@given(
    method=st.sampled_from(METHODS),
    design=count_designs(),
    alpha_lower=ALPHAS,
    alpha_upper=ALPHAS,
)
def test_count_engines_on_prompt_clusters_equal_bounded_engines_on_prompt_rates(
    method: str, design: dict[str, list[Any]], alpha_lower: float, alpha_upper: float
) -> None:
    """The two implementations of each engine agree, so a fix to one cannot miss the other.

    The fixed-design pipeline calls the bounded engines and comparisons call the count
    engines. A prompt cluster is a block whose value lies between its failure rate and its
    rate with every unknown counted as a failure.
    """

    trials = np.asarray(design["trials"], dtype=float)
    failures = np.asarray(design["failures"], dtype=float)
    unknown = np.asarray(design["unclassified"], dtype=float)
    counted = call_count_engine(
        count_engine(method),
        design["failures"],
        design["trials"],
        design["weights"],
        unclassified=design["unclassified"],
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        unit="prompt_cluster",
    )
    bounded = call_bounded_engine(
        next(engine for engine in BOUNDED_ENGINES if engine.method == method),
        (failures / trials).tolist(),
        ((failures + unknown) / trials).tolist(),
        design["weights"],
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    assert counted.lower == pytest.approx(bounded.lower, rel=0, abs=1e-12)
    assert counted.upper == pytest.approx(bounded.upper, rel=0, abs=1e-12)


@settings(deadline=None)
@given(
    method=st.sampled_from(METHODS),
    design=count_designs().filter(lambda design: sum(design["trials"]) <= 60),
    alpha_lower=ALPHAS,
    alpha_upper=ALPHAS,
)
def test_count_engines_on_rollouts_equal_bounded_engines_on_one_block_per_rollout(
    method: str, design: dict[str, list[Any]], alpha_lower: float, alpha_upper: float
) -> None:
    """``block_by: [rollout]`` in the pipeline and the rollout unit must give one interval."""

    lower: list[float] = []
    upper: list[float] = []
    weights: list[float] = []
    for x, u, n, weight in zip(
        design["failures"], design["unclassified"], design["trials"], design["weights"],
        strict=True,
    ):
        lower += [1.0] * x + [0.0] * (n - x)
        upper += [1.0] * (x + u) + [0.0] * (n - x - u)
        weights += [weight / n] * n
    total = math.fsum(weights)
    counted = call_count_engine(
        count_engine(method),
        design["failures"],
        design["trials"],
        design["weights"],
        unclassified=design["unclassified"],
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        unit="rollout",
    )
    bounded = call_bounded_engine(
        next(engine for engine in BOUNDED_ENGINES if engine.method == method),
        lower,
        upper,
        [weight / total for weight in weights],
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )
    assert counted.lower == pytest.approx(bounded.lower, rel=0, abs=1e-12)
    assert counted.upper == pytest.approx(bounded.upper, rel=0, abs=1e-12)


# 180 prompts of mostly sparse failures. On the small random designs above the
# empirical-Bernstein variance term is usually capped at C2 / 4, where its padding plays no
# part, so a drift in the padding of one implementation would pass. Here the proxy decides.
_SPARSE_FAILURES = ([0] * 48 + [1] * 8 + [2] * 3 + [6]) * 3
_SPARSE_UNKNOWN = ([0] * 40 + [1] * 20) * 3
_SKEW_TOTAL = sum(1 + prompt % 7 for prompt in range(180))
PROXY_DESIGNS = {
    "equal_weights": {
        "failures": _SPARSE_FAILURES,
        "unclassified": _SPARSE_UNKNOWN,
        "trials": [20] * 180,
        "weights": [1 / 180] * 180,
    },
    "skewed_weights": {
        "failures": _SPARSE_FAILURES,
        "unclassified": _SPARSE_UNKNOWN,
        "trials": [10, 30] * 90,
        "weights": [(1 + prompt % 7) / _SKEW_TOTAL for prompt in range(180)],
    },
}


@pytest.mark.parametrize("unit", ["rollout", "prompt_cluster"])
@pytest.mark.parametrize("design", list(PROXY_DESIGNS.values()), ids=list(PROXY_DESIGNS))
@pytest.mark.parametrize("method", METHODS)
def test_count_and_bounded_engines_agree_where_the_variance_proxy_decides(
    method: str, design: dict[str, list[Any]], unit: Unit
) -> None:
    trials = design["trials"]
    if unit == "prompt_cluster":
        lower = [x / n for x, n in zip(design["failures"], trials, strict=True)]
        upper = [
            (x + u) / n
            for x, u, n in zip(design["failures"], design["unclassified"], trials, strict=True)
        ]
        weights = design["weights"]
    else:
        lower, upper, weights = [], [], []
        for x, u, n, weight in zip(
            design["failures"], design["unclassified"], trials, design["weights"], strict=True
        ):
            lower += [1.0] * x + [0.0] * (n - x)
            upper += [1.0] * (x + u) + [0.0] * (n - x - u)
            weights += [weight / n] * n
        total = math.fsum(weights)
        weights = [weight / total for weight in weights]
    for alpha_lower, alpha_upper in ((0.05, 0.05), (0.001, 0.02)):
        counted = call_count_engine(
            count_engine(method),
            design["failures"],
            trials,
            design["weights"],
            unclassified=design["unclassified"],
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
            unit=unit,
        )
        bounded = call_bounded_engine(
            next(engine for engine in BOUNDED_ENGINES if engine.method == method),
            lower,
            upper,
            weights,
            alpha_lower=alpha_lower,
            alpha_upper=alpha_upper,
        )
        if method == "weighted_empirical_bernstein":
            assert counted.variance_upper_upper is not None
            assert counted.variance_upper_upper < 0.25 * counted.concentration_variance * 0.9
        assert counted.lower == pytest.approx(bounded.lower, rel=0, abs=1e-12)
        assert counted.upper == pytest.approx(bounded.upper, rel=0, abs=1e-12)


# --- Comparison structure ------------------------------------------------------------------


@st.composite
def two_arm_designs(draw: st.DrawFn) -> dict[str, list[Any]]:
    size = draw(st.integers(min_value=1, max_value=4))
    arms: dict[str, list[Any]] = {}
    for arm in ("candidate", "baseline"):
        arm_design = draw(count_designs(size))
        for key in ("failures", "unclassified", "trials"):
            arms[f"{arm}_{key}"] = arm_design[key]
        arms["weights"] = arm_design["weights"]
    return arms


def _arm_arguments(design: dict[str, list[Any]], *, swapped: bool) -> tuple[np.ndarray, ...]:
    first, second = ("baseline", "candidate") if swapped else ("candidate", "baseline")
    return tuple(
        np.asarray(design[f"{arm}_{key}"], dtype=int)
        for arm in (first, second)
        for key in ("failures", "unclassified", "trials")
    ) + (np.asarray(design["weights"], dtype=float),)


@settings(deadline=None)
@given(
    method=st.sampled_from(COMPARISON_METHODS),
    design=two_arm_designs(),
    alpha_lower=ALPHAS,
    alpha_upper=ALPHAS,
    unit=UNITS,
)
def test_swapping_the_arms_negates_the_difference_and_inverts_the_ratio(
    method: str,
    design: dict[str, list[Any]],
    alpha_lower: float,
    alpha_upper: float,
    unit: Unit,
) -> None:
    options: dict[str, Any] = {"method": method, "independence_unit": unit}
    forward = _arm_arguments(design, swapped=False)
    backward = _arm_arguments(design, swapped=True)
    difference = direct_risk_difference_bounds(
        *forward, alpha_lower=alpha_lower, alpha_upper=alpha_upper, **options
    )
    reversed_difference = direct_risk_difference_bounds(
        *backward, alpha_lower=alpha_upper, alpha_upper=alpha_lower, **options
    )
    assert difference.lower == pytest.approx(-reversed_difference.upper, rel=0, abs=1e-9)
    assert difference.upper == pytest.approx(-reversed_difference.lower, rel=0, abs=1e-9)

    ratio = direct_ratio_bounds(
        *forward, alpha_lower=alpha_lower, alpha_upper=alpha_upper, **options
    )
    inverse = direct_ratio_bounds(
        *backward, alpha_lower=alpha_upper, alpha_upper=alpha_lower, **options
    )
    if ratio.lower is not None and ratio.lower > 0:
        assert inverse.upper is not None
        assert ratio.lower == pytest.approx(1 / inverse.upper, rel=1e-9, abs=0)
    if ratio.upper is not None and ratio.upper > 0:
        assert inverse.lower is not None
        assert ratio.upper == pytest.approx(1 / inverse.lower, rel=1e-9, abs=0)


@settings(deadline=None)
@given(
    method=st.sampled_from(COMPARISON_METHODS),
    design=two_arm_designs(),
    alpha=ALPHAS,
    factor=st.floats(min_value=0.01, max_value=1.0),
    unit=UNITS,
)
def test_comparison_intervals_widen_as_alpha_falls(
    method: str, design: dict[str, list[Any]], alpha: float, factor: float, unit: Unit
) -> None:
    arguments = _arm_arguments(design, swapped=False)
    options: dict[str, Any] = {"method": method, "independence_unit": unit}
    wide = direct_risk_difference_bounds(
        *arguments, alpha_lower=alpha * factor, alpha_upper=alpha * factor, **options
    )
    narrow = direct_risk_difference_bounds(
        *arguments, alpha_lower=alpha, alpha_upper=alpha, **options
    )
    assert wide.lower <= narrow.lower + TOLERANCE
    assert wide.upper >= narrow.upper - TOLERANCE
