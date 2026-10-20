"""Properties every risk measure must satisfy for the bound construction to be valid.

The package reports an interval for a risk measure by pushing the per-prompt
Clopper-Pearson endpoints through that measure. That is only a bound because each
measure is coordinatewise monotone, so ``test_measures_are_coordinatewise_monotone``
is not a nicety: it is the assumption the reported coverage rests on, checked for
every measure the package exposes rather than for the two it used to cover.

Weights are drawn rather than held equal, because equal weights hide exactly the
mistakes a weighted measure makes -- a sort that loses its companion weights, a
normalisation applied to one array and not the other.
"""

import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from inspect_safety_stats.stats import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
    worst_of_k_risk,
)

TOLERANCE = 1e-9

# Risks are probabilities, and the endpoints pushed through these measures are
# always in the unit interval.
RISKS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def designs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """A vector of risks with matching weights summing to one.

    Zero weights are permitted: ``entropic_risk`` restricts to the support and the
    others rely on a zero contribution, so an unweighted item is a real input rather
    than a degenerate one.
    """

    size = draw(st.integers(min_value=1, max_value=12))
    values = draw(st.lists(RISKS, min_size=size, max_size=size))
    raw = draw(st.lists(RISKS, min_size=size, max_size=size))
    total = math.fsum(raw)
    # Guards a normalisation that would otherwise divide by a subnormal total.
    assume(total > 1e-6)
    return values, [weight / total for weight in raw]


@given(
    design=designs(),
    shifts=st.lists(RISKS, min_size=1, max_size=12),
    beta=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    lambda_=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
    threshold=RISKS,
    k_seed=st.integers(min_value=0, max_value=11),
)
def test_measures_are_coordinatewise_monotone(
    design: tuple[list[float], list[float]],
    shifts: list[float],
    beta: float,
    lambda_: float,
    threshold: float,
    k_seed: int,
) -> None:
    """Raising any prompt's risk never lowers the reported measure.

    This is what makes endpoint propagation a bound: with each per-prompt interval
    covering its own risk, the measure of the upper endpoints covers the measure of
    the truth from above, and the lower endpoints from below.
    """

    lower, weights = design
    upper = [min(1.0, value + shifts[index % len(shifts)]) for index, value in enumerate(lower)]
    k = k_seed % len(lower) + 1

    assert weighted_mean(lower, weights) <= weighted_mean(upper, weights) + TOLERANCE
    assert maximum_risk(lower, weights) <= maximum_risk(upper, weights) + TOLERANCE
    assert weighted_cvar(lower, weights, beta=beta) <= (
        weighted_cvar(upper, weights, beta=beta) + TOLERANCE
    )
    assert top_k_mean(lower, k=k) <= top_k_mean(upper, k=k) + TOLERANCE
    assert entropic_risk(lower, weights, lambda_=lambda_) <= (
        entropic_risk(upper, weights, lambda_=lambda_) + TOLERANCE
    )
    assert threshold_exceedance(lower, weights, threshold=threshold) <= (
        threshold_exceedance(upper, weights, threshold=threshold) + TOLERANCE
    )
    assert threshold_excess(lower, weights, threshold=threshold) <= (
        threshold_excess(upper, weights, threshold=threshold) + TOLERANCE
    )


@given(
    design=designs(),
    beta=st.floats(min_value=0.0, max_value=0.99, allow_nan=False),
    lambda_=st.floats(min_value=0.1, max_value=20.0, allow_nan=False),
)
def test_tail_measures_sit_between_the_mean_and_the_worst_item(
    design: tuple[list[float], list[float]], beta: float, lambda_: float
) -> None:
    """CVaR and entropic risk are tail-sensitive but never leave the range.

    Both dominate the weighted mean -- CVaR by averaging only the worst tail mass,
    entropic risk by Jensen -- and neither can exceed the worst item. A measure
    reporting outside that range is reading its weights wrongly, which no
    monotonicity check would catch.
    """

    values, weights = design
    mean = weighted_mean(values, weights)
    worst = maximum_risk(values, weights)

    for reported in (
        weighted_cvar(values, weights, beta=beta),
        entropic_risk(values, weights, lambda_=lambda_),
    ):
        assert mean <= reported + TOLERANCE
        assert reported <= worst + TOLERANCE


@given(design=designs(), unweighted=RISKS)
def test_a_zero_weight_item_does_not_move_a_weighted_measure(
    design: tuple[list[float], list[float]], unweighted: float
) -> None:
    """An item carrying no policy weight contributes nothing.

    Nested-equal and explicit policy weights both admit zero, so this is a reachable
    configuration rather than a pathological one, and it is the cheapest way to catch
    a measure that iterates over values while ignoring their weights.
    """

    values, weights = design
    extended = [*values, unweighted]
    padded = [*weights, 0.0]

    assert weighted_mean(extended, padded) == pytest.approx(weighted_mean(values, weights))
    assert weighted_cvar(extended, padded, beta=0.8) == pytest.approx(
        weighted_cvar(values, weights, beta=0.8)
    )
    assert entropic_risk(extended, padded, lambda_=5) == pytest.approx(
        entropic_risk(values, weights, lambda_=5)
    )
    assert threshold_exceedance(extended, padded, threshold=0.1) == pytest.approx(
        threshold_exceedance(values, weights, threshold=0.1)
    )
    assert threshold_excess(extended, padded, threshold=0.1) == pytest.approx(
        threshold_excess(values, weights, threshold=0.1)
    )


@given(
    design=designs(),
    lambda_=st.floats(min_value=1e-3, max_value=20.0, allow_nan=False),
)
def test_entropic_risk_obeys_the_hoeffding_bound(
    design: tuple[list[float], list[float]], lambda_: float
) -> None:
    """Entropic risk exceeds the mean by at most ``lambda * range**2 / 8``.

    Hoeffding's lemma, and the reason this test exists rather than a looser "between
    the mean and the worst item" claim: a measure that simply returned the worst item
    would satisfy the looser claim at its boundary. The bound collapses onto the mean
    as lambda vanishes, which is the behaviour that pins the tail sensitivity down.
    """

    values, weights = design
    supported = [value for value, weight in zip(values, weights, strict=True) if weight > 0]
    mean = weighted_mean(values, weights)
    spread = max(supported) - min(supported)

    for tested in (lambda_, 1e-3):
        assert entropic_risk(values, weights, lambda_=tested) <= (
            mean + tested * spread**2 / 8 + TOLERANCE
        )


@given(design=designs())
def test_top_k_mean_is_anchored_at_both_ends_and_falls_as_k_grows(
    design: tuple[list[float], list[float]],
) -> None:
    """Averaging the k worst items interpolates between the worst and the mean.

    Monotonicity alone does not identify which k items are taken -- averaging the k
    *best* is monotone too -- so the anchors are what fix the direction.
    """

    values, _ = design
    count = len(values)

    assert top_k_mean(values, k=1) == pytest.approx(max(values))
    assert top_k_mean(values, k=count) == pytest.approx(sum(values) / count)
    for k in range(1, count):
        assert top_k_mean(values, k=k) >= top_k_mean(values, k=k + 1) - TOLERANCE


@given(design=designs(), threshold=RISKS)
def test_threshold_measures_stay_within_their_own_range(
    design: tuple[list[float], list[float]], threshold: float
) -> None:
    """Exceedance is a probability; excess is a nonnegative expected overshoot.

    Excess clips each item at zero before weighting, so items *below* the threshold
    contribute nothing rather than subtracting. Dropping that clip leaves a measure
    that is still monotone and still reads its weights, so only the sign catches it.
    """

    values, weights = design
    exceedance = threshold_exceedance(values, weights, threshold=threshold)
    excess = threshold_excess(values, weights, threshold=threshold)

    assert 0.0 <= exceedance <= 1.0 + TOLERANCE
    assert excess >= 0.0
    # Nothing exceeds a threshold at or above the worst item.
    assert threshold_exceedance(values, weights, threshold=1.0) == 0.0
    assert threshold_excess(values, weights, threshold=1.0) == 0.0


@given(
    values=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=20),
    shift=st.floats(min_value=0, max_value=1, allow_nan=False),
    k=st.integers(min_value=1, max_value=16),
)
def test_worst_of_k_is_monotone_and_sandwiched(
    values: list[float], shift: float, k: int
) -> None:
    weights = [1 / len(values)] * len(values)
    upper = [min(1.0, value + shift) for value in values]
    base = worst_of_k_risk(values, weights, k=k)
    shifted = worst_of_k_risk(upper, weights, k=k)
    mean = weighted_mean(values, weights)
    # Coordinatewise monotone (envelope propagation is valid) and sandwiched
    # between the single-attempt mean and the k-scaled union bound.
    assert base <= shifted + 1e-12
    assert mean - 1e-12 <= base <= min(1.0, k * mean) + 1e-12
