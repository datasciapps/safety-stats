"""Registries and exact-coverage machinery shared by the conformance tests.

Every interval engine and risk measure a plan can select is registered here, and the
conformance tests run the same checks over every registered entry. A new engine or measure
is held to the existing bar by adding one entry, and
``test_the_registry_lists_every_engine_a_plan_can_select`` and
``test_the_registry_lists_every_measure_a_plan_can_select`` fail until it is added.

Exact coverage rests on one observation: an engine's bounds depend on the observed outcome
and never on the true rates. The bounds are therefore computed once per possible outcome,
and the probability that each endpoint misses is then a weighted sum over outcomes, for as
many true-rate vectors as the grid holds, at the cost of one matrix product.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import binom

from safety_stats.config.models import RiskMeasureSpec
from safety_stats.stats import (
    DirectMeanBounds,
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
)

# Experimental, so not in safety_stats.stats.__all__ until it graduates.
from safety_stats.stats.direct_mean import (
    weighted_betting_mean_bounds,
    weighted_bounded_betting_mean_bounds,
)

Unit = Literal["rollout", "prompt_cluster"]

# How the rollouts of one prompt depend on each other. "independent" makes a prompt's failure
# count binomial. "identical" makes every rollout of a prompt share one outcome, the strongest
# dependence the prompt-cluster unit admits and the one that puts the most mass at the ends.
WithinPrompt = Literal["independent", "identical"]

# A bound that misses by less than this is rounding at a boundary, such as a true rate of
# zero against a lower bound of zero, not a miss.
MISS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CountEngine:
    """A direct-mean engine on failure counts, as comparisons and the generative path call it.

    ``always_mirror_symmetric`` declares that relabelling failures as successes reflects the
    interval: ``lower(x) = 1 - upper(n - x)`` with the two alphas swapped. KL/Chernoff has
    that only under equal atomic weights, where it inverts the symmetric Bernoulli KL; its
    unequal-weight Poisson kernel is asymmetric by construction.

    ``zero_weight_invariant`` declares that adding a prompt of zero weight leaves the
    interval alone. KL/Chernoff does not have it; see
    ``test_a_prompt_without_weight_does_not_change_the_kl_chernoff_kernel``.
    """

    method: str
    function: Callable[..., DirectMeanBounds]
    always_mirror_symmetric: bool
    zero_weight_invariant: bool


@dataclass(frozen=True)
class BoundedEngine:
    """A direct-mean engine on [0, 1]-valued blocks, as the fixed-design pipeline calls it."""

    method: str
    function: Callable[..., DirectMeanBounds]


COUNT_ENGINES = (
    CountEngine(
        "weighted_hoeffding",
        weighted_hoeffding_mean_bounds,
        always_mirror_symmetric=True,
        zero_weight_invariant=True,
    ),
    CountEngine(
        "weighted_empirical_bernstein",
        weighted_empirical_bernstein_mean_bounds,
        always_mirror_symmetric=True,
        zero_weight_invariant=True,
    ),
    CountEngine(
        "weighted_kl_chernoff",
        weighted_kl_chernoff_mean_bounds,
        always_mirror_symmetric=False,
        zero_weight_invariant=False,
    ),
    CountEngine(
        "weighted_betting",
        weighted_betting_mean_bounds,
        always_mirror_symmetric=True,
        zero_weight_invariant=True,
    ),
)
BOUNDED_ENGINES = (
    BoundedEngine("weighted_hoeffding", weighted_bounded_hoeffding_mean_bounds),
    BoundedEngine("weighted_empirical_bernstein", weighted_bounded_empirical_bernstein_mean_bounds),
    BoundedEngine("weighted_kl_chernoff", weighted_bounded_kl_chernoff_mean_bounds),
    BoundedEngine("weighted_betting", weighted_bounded_betting_mean_bounds),
)
METHODS = tuple(engine.method for engine in COUNT_ENGINES)
# Comparisons accept only these. weighted_betting is a direct-mean engine only for now, and a
# plan that selects it for a comparison is rejected by the plan schema.
COMPARISON_METHODS = ("weighted_hoeffding", "weighted_empirical_bernstein", "weighted_kl_chernoff")


def count_engine(method: str) -> CountEngine:
    return next(engine for engine in COUNT_ENGINES if engine.method == method)


def call_count_engine(
    engine: CountEngine,
    failures: Sequence[int],
    trials: Sequence[int],
    weights: Sequence[float],
    *,
    alpha_lower: float,
    alpha_upper: float,
    unit: Unit,
    unclassified: Sequence[int] | None = None,
) -> DirectMeanBounds:
    return engine.function(
        np.asarray(failures, dtype=int),
        np.asarray(unclassified if unclassified is not None else [0] * len(failures), dtype=int),
        np.asarray(trials, dtype=int),
        np.asarray(weights, dtype=float),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_unit=unit,
    )


def call_bounded_engine(
    engine: BoundedEngine,
    lower: Sequence[float],
    upper: Sequence[float],
    weights: Sequence[float],
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> DirectMeanBounds:
    return engine.function(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        np.asarray(weights, dtype=float),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        independence_profile="block",
    )


# --- Exact coverage ------------------------------------------------------------------------


def count_outcomes(trials: Sequence[int], within: WithinPrompt) -> np.ndarray:
    """Every possible vector of per-prompt failure counts, one row each."""

    support = [range(n + 1) if within == "independent" else (0, n) for n in trials]
    return np.asarray(list(itertools.product(*support)), dtype=int).reshape(-1, len(trials))


def outcome_probabilities(
    outcomes: np.ndarray, trials: Sequence[int], rates: np.ndarray, within: WithinPrompt
) -> np.ndarray:
    """The probability of each outcome row under each row of true rates, as (rates, outcomes)."""

    probabilities = np.ones((rates.shape[0], outcomes.shape[0]))
    for prompt, n in enumerate(trials):
        counts = outcomes[None, :, prompt]
        rate = rates[:, prompt, None]
        if within == "independent":
            probabilities *= binom.pmf(counts, n, rate)
        else:
            probabilities *= np.where(counts == n, rate, 1 - rate)
    return probabilities


def countermonotone_arm_probabilities(
    outcomes: np.ndarray,
    candidate_trials: Sequence[int],
    baseline_trials: Sequence[int],
    rates: np.ndarray,
) -> np.ndarray:
    """Outcome probabilities when each prompt's two arms are as negatively dependent as possible.

    Within an arm every rollout of a prompt shares one outcome, and across arms the two
    outcomes fail together only as often as they must, ``max(0, p_c + p_b - 1)``. Columns of
    ``outcomes`` are the candidate prompts followed by the baseline prompts, each 0 or n.
    """

    prompts = len(candidate_trials)
    probabilities = np.ones((rates.shape[0], outcomes.shape[0]))
    for prompt in range(prompts):
        candidate = outcomes[None, :, prompt] == candidate_trials[prompt]
        baseline = outcomes[None, :, prompts + prompt] == baseline_trials[prompt]
        candidate_rate = rates[:, prompt, None]
        baseline_rate = rates[:, prompts + prompt, None]
        both = np.maximum(0.0, candidate_rate + baseline_rate - 1)
        probabilities *= np.where(
            candidate & baseline,
            both,
            np.where(
                candidate,
                candidate_rate - both,
                np.where(baseline, baseline_rate - both, 1 - candidate_rate - baseline_rate + both),
            ),
        )
    return probabilities


def rate_grid(levels: Sequence[float], prompts: int) -> np.ndarray:
    return np.asarray(list(itertools.product(levels, repeat=prompts)), dtype=float)


@dataclass(frozen=True)
class WorstMiss:
    """The largest miss probability of each endpoint over a grid, and where it occurs."""

    lower: float
    lower_rates: tuple[float, ...]
    upper: float
    upper_rates: tuple[float, ...]


def worst_misses(
    lower: np.ndarray,
    upper: np.ndarray,
    probabilities: np.ndarray,
    truth: np.ndarray,
    rates: np.ndarray,
) -> WorstMiss:
    """Exact miss probabilities of per-outcome bounds, maximised over the true-rate grid.

    ``lower`` and ``upper`` hold one bound per outcome, with an absent bound as -inf or inf.
    ``probabilities`` is (grid, outcomes) and ``truth`` holds the estimand at each grid row.
    """

    lower_by_rate = np.sum(
        probabilities * (lower[None, :] > truth[:, None] + MISS_TOLERANCE), axis=1
    )
    upper_by_rate = np.sum(
        probabilities * (upper[None, :] < truth[:, None] - MISS_TOLERANCE), axis=1
    )
    worst_lower = int(np.argmax(lower_by_rate))
    worst_upper = int(np.argmax(upper_by_rate))
    return WorstMiss(
        lower=float(lower_by_rate[worst_lower]),
        lower_rates=tuple(float(value) for value in rates[worst_lower]),
        upper=float(upper_by_rate[worst_upper]),
        upper_rates=tuple(float(value) for value in rates[worst_upper]),
    )


def poisson_binomial_pmf(trials: Sequence[int], rates: Sequence[float]) -> np.ndarray:
    """The exact distribution of the total failure count over independent binomial prompts."""

    pmf = np.ones(1)
    for n, rate in zip(trials, rates, strict=True):
        pmf = np.convolve(pmf, binom.pmf(np.arange(n + 1), n, rate))
    return pmf


def spread(total: int, trials: Sequence[int], *, from_front: bool) -> list[int]:
    """One per-prompt split of a total failure count, filling prompts from one end."""

    order = range(len(trials)) if from_front else reversed(range(len(trials)))
    failures = [0] * len(trials)
    remaining = total
    for prompt in order:
        failures[prompt] = min(trials[prompt], remaining)
        remaining -= failures[prompt]
    return failures


# --- Risk measures -------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureCase:
    """How to build a plan's risk-measure spec of one type for the conformance checks.

    ``equal_weights_only`` declares a measure the pipeline evaluates only under equal prompt
    weights, so weighted checks such as replication do not apply. ``replication_invariant``
    declares that splitting an item's weight among identical copies leaves the value alone.
    ``zero_weight_invariant`` declares that an item of zero weight does not count.
    """

    spec: RiskMeasureSpec
    threshold: float | None = None
    equal_weights_only: bool = False
    replication_invariant: bool = True
    zero_weight_invariant: bool = True


def _spec(type_: str, **parameters: Any) -> RiskMeasureSpec:
    return RiskMeasureSpec.model_validate(
        {"id": f"conformance-{type_}", "type": type_, **parameters}
    )


MEASURE_CASES: dict[str, tuple[MeasureCase, ...]] = {
    "weighted_mean": (MeasureCase(_spec("weighted_mean")),),
    # The worst item over every prompt in the group, weighted or not (docs/statistical-methods.md,
    # section 2.4.2). CVaR and entropic risk, whose limits the docs call the maximum, ignore an
    # item of zero weight; whether the maximum should too is open.
    "maximum": (MeasureCase(_spec("maximum"), zero_weight_invariant=False),),
    "cvar": (
        MeasureCase(_spec("cvar", beta=0.0)),
        MeasureCase(_spec("cvar", beta=0.7)),
        MeasureCase(_spec("cvar", beta=0.95)),
    ),
    "top_k_mean": (
        MeasureCase(_spec("top_k_mean", k=1), equal_weights_only=True, replication_invariant=False),
        MeasureCase(_spec("top_k_mean", k=2), equal_weights_only=True, replication_invariant=False),
    ),
    "worst_of_k": (
        MeasureCase(_spec("worst_of_k", k=1)),
        MeasureCase(_spec("worst_of_k", k=5)),
    ),
    "entropic": (
        MeasureCase(_spec("entropic", **{"lambda": 0.5})),
        MeasureCase(_spec("entropic", **{"lambda": 12.0})),
    ),
    "threshold_exceedance": (
        MeasureCase(_spec("threshold_exceedance", threshold=0.1), threshold=0.1),
        MeasureCase(_spec("threshold_exceedance", threshold=0.6), threshold=0.6),
    ),
    "threshold_excess": (
        MeasureCase(_spec("threshold_excess", threshold=0.1), threshold=0.1),
        MeasureCase(_spec("threshold_excess", threshold=-0.5), threshold=-0.5),
    ),
}
