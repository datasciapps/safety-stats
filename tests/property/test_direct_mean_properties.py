"""Properties the direct-mean bounds must satisfy to cover every completion of unknowns.

Unclassified outcomes are never imputed. The reported interval has to contain the interval
of every possible completion, because then it covers whenever the interval of the true,
unobserved labels covers. Checking that only the point estimate moves the right way is not
enough: for empirical Bernstein, completing an unknown as a failure also changes the
variance bound, and the upper endpoint once fell below that of the true labels.
"""

import itertools
import math

from hypothesis import given
from hypothesis import strategies as st

from safety_stats.stats import (
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
)

TOLERANCE = 1e-12

COUNT_METHODS = (
    weighted_empirical_bernstein_mean_bounds,
    weighted_hoeffding_mean_bounds,
    weighted_kl_chernoff_mean_bounds,
)
BOUNDED_METHODS = (
    weighted_bounded_empirical_bernstein_mean_bounds,
    weighted_bounded_hoeffding_mean_bounds,
    weighted_bounded_kl_chernoff_mean_bounds,
)
# Small endpoint alphas matter: Bonferroni allocation over a few dozen nodes reaches 1e-3,
# where the empirical-Bernstein failure appeared.
ALPHAS = st.floats(min_value=1e-4, max_value=0.3, allow_nan=False)


def _normalised(draw: st.DrawFn, size: int) -> list[float]:
    # Log-uniform, so a few heavy weights next to many light ones are common: that skew is
    # where a variance bound that moves with the completion went wrong.
    exponents = st.floats(min_value=-4.0, max_value=0.0, allow_nan=False)
    raw = [10**exponent for exponent in draw(st.lists(exponents, min_size=size, max_size=size))]
    total = math.fsum(raw)
    return [value / total for value in raw]


@st.composite
def count_designs(draw: st.DrawFn) -> tuple[list[int], list[int], list[int], list[float]]:
    """Prompts with failures, unclassified outcomes and trials, plus weights summing to one.

    Trial counts span magnitudes. With only a few rollouts the C2/4 cap dominates every
    variance bound, which hides how the bound depends on the completion.
    """
    size = draw(st.integers(min_value=1, max_value=4))
    counts = st.sampled_from([1, 2, 3, 5, 50, 500, 20_000])
    trials = draw(st.lists(counts, min_size=size, max_size=size))
    failures: list[int] = []
    unclassified: list[int] = []
    for n in trials:
        unknown = draw(st.integers(min_value=0, max_value=min(2, n)))
        failures.append(draw(st.integers(min_value=0, max_value=n - unknown)))
        unclassified.append(unknown)
    return failures, unclassified, trials, _normalised(draw, size)


@given(
    design=count_designs(),
    alpha=ALPHAS,
    unit=st.sampled_from(["rollout", "prompt_cluster"]),
    method=st.sampled_from(COUNT_METHODS),
)
def test_count_bounds_contain_the_bounds_of_every_completion(
    design: tuple[list[int], list[int], list[int], list[float]],
    alpha: float,
    unit: str,
    method: object,
) -> None:
    failures, unclassified, trials, weights = design
    options = {"alpha_lower": alpha, "alpha_upper": alpha, "independence_unit": unit}
    reported = method(failures, unclassified, trials, weights, **options)  # type: ignore[operator]
    none = [0] * len(trials)
    for extra in itertools.product(*(range(unknown + 1) for unknown in unclassified)):
        labels = [count + added for count, added in zip(failures, extra, strict=True)]
        completed = method(labels, none, trials, weights, **options)  # type: ignore[operator]
        assert reported.upper >= completed.upper - TOLERANCE, (labels, completed.upper)
        assert reported.lower <= completed.lower + TOLERANCE, (labels, completed.lower)


@st.composite
def bounded_designs(draw: st.DrawFn) -> tuple[list[float], list[float], list[float], list[float]]:
    """Blocks with an adverse lower and upper completion, a value between them, and weights."""
    size = draw(st.integers(min_value=1, max_value=4))
    unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    ends = [sorted(draw(st.lists(unit, min_size=2, max_size=2))) for _ in range(size)]
    positions = draw(st.lists(unit, min_size=size, max_size=size))
    lower = [low for low, _ in ends]
    upper = [high for _, high in ends]
    between = [low + t * (high - low) for (low, high), t in zip(ends, positions, strict=True)]
    return lower, upper, between, _normalised(draw, size)


@given(design=bounded_designs(), alpha=ALPHAS, method=st.sampled_from(BOUNDED_METHODS))
def test_bounded_bounds_contain_the_bounds_of_every_value_between_the_completions(
    design: tuple[list[float], list[float], list[float], list[float]],
    alpha: float,
    method: object,
) -> None:
    lower, upper, between, weights = design
    options = {"alpha_lower": alpha, "alpha_upper": alpha, "independence_profile": "block"}
    reported = method(lower, upper, weights, **options)  # type: ignore[operator]
    corners = itertools.product(*zip(lower, upper, strict=True))
    for values in [between, *map(list, corners)]:
        completed = method(values, values, weights, **options)  # type: ignore[operator]
        assert reported.upper >= completed.upper - TOLERANCE, (values, completed.upper)
        assert reported.lower <= completed.lower + TOLERANCE, (values, completed.lower)
