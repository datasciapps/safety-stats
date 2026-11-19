from __future__ import annotations

import numpy as np
import pytest

from inspect_safety_stats.analysis.design_effect import (
    DESIGN_EFFECT_WARNING,
    design_effect,
    intracluster_correlation,
    unit_design_effect,
    unit_intracluster_correlation,
)


def test_clusters_that_fail_as_one_have_correlation_one() -> None:
    assert intracluster_correlation([4, 4], [4, 0]) == pytest.approx(1.0, rel=1e-12, abs=0)
    assert design_effect([4, 4], 1.0) == pytest.approx(4.0, rel=1e-12, abs=0)


def test_pairs_that_always_split_have_correlation_minus_one() -> None:
    # Each pair holds one failure and one success: -1 / (m - 1) with m = 2.
    correlation = intracluster_correlation([2, 2], [1, 1])
    assert correlation == pytest.approx(-1.0, rel=1e-12, abs=0)
    # A negative correlation is read as no clustering.
    assert design_effect([2, 2], -1.0) == 1.0


def test_unequal_clusters_follow_the_anova_estimator() -> None:
    # N = 10, G = 3, failure rates 3/4, 1/4, 0; overall rate 2/5.
    # MSB = (4 (7/20)^2 + 4 (3/20)^2 + 2 (2/5)^2) / 2 = 0.45
    # MSW = (3 (1/4) + 1 (3/4) + 0) / 7 = 3/14
    # m0 = (10 - 36/10) / 2 = 3.2
    between, within, typical = 0.45, 3 / 14, 3.2
    expected = (between - within) / (between + (typical - 1) * within)
    correlation = intracluster_correlation([4, 4, 2], [3, 1, 0])
    assert correlation == pytest.approx(expected, rel=1e-12, abs=0)
    # m = (16 + 16 + 4) / 10 = 3.6
    assert design_effect([4, 4, 2], expected) == pytest.approx(
        1 + 2.6 * expected, rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    ("sizes", "failures"),
    [
        ([5], [2]),  # one cluster: no between-cluster variation to compare
        ([1, 1, 1], [1, 0, 1]),  # no cluster with two outcomes: no within variation
        ([3, 3], [0, 0]),  # every outcome the same
        ([3, 3], [3, 3]),
    ],
)
def test_correlation_is_undefined_without_the_variation_it_compares(
    sizes: list[int], failures: list[int]
) -> None:
    assert intracluster_correlation(sizes, failures) is None


def test_inconsistent_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned"):
        intracluster_correlation([2, 2], [1])
    with pytest.raises(ValueError, match="at most one failure"):
        intracluster_correlation([2, 2], [3, 0])
    with pytest.raises(ValueError, match="at most one failure"):
        intracluster_correlation([0, 2], [0, 1])


def test_units_follow_the_anova_estimator_on_their_failure_rates() -> None:
    # Rates 1/2, 1 | 0, 1, 0 | 2/5: K = 6, G = 3, block means 3/4, 1/3, 2/5 around 29/60.
    # MSB = (2 (16/60)^2 + 3 (9/60)^2 + (5/60)^2) / 2 = 13/120
    # MSW = (1/16 + 1/16 + 1/9 + 4/9 + 1/9 + 0) / 3 = 19/72
    # k0 = (6 - 14/6) / 2 = 11/6
    blocks = [[(2, 1), (4, 4)], [(3, 0), (1, 1), (2, 0)], [(5, 2)]]
    expected = (13 / 120 - 19 / 72) / (13 / 120 + (11 / 6 - 1) * 19 / 72)
    assert expected == pytest.approx(-336 / 709, rel=1e-12, abs=0)
    assert unit_intracluster_correlation(blocks) == pytest.approx(expected, rel=1e-12, abs=0)


def test_units_of_one_rollout_give_the_binary_estimate_to_the_bit() -> None:
    blocks = [[(1, 1), (1, 1), (1, 1), (1, 0)], [(1, 0), (1, 1), (1, 0), (1, 0)], [(1, 0), (1, 0)]]
    assert unit_intracluster_correlation(blocks) == intracluster_correlation([4, 4, 2], [3, 1, 0])


def test_units_that_fail_wholly_or_not_at_all_read_as_single_rollouts() -> None:
    # Rates of 0 and 1 over ten rollouts give the binary estimate, through the rates.
    as_units = unit_intracluster_correlation([[(10, 10), (10, 10), (10, 0)], [(10, 0)] * 3])
    assert as_units == pytest.approx(
        intracluster_correlation([3, 3], [2, 0]), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    "blocks",
    [
        [[(10, 3), (10, 5)]],  # one block
        [[(10, 3)], [(10, 5)]],  # no block with two units
        [[(10, 5), (4, 2)], [(2, 1), (6, 3)]],  # every unit fails half its rollouts
    ],
)
def test_unit_correlation_is_undefined_without_the_variation_it_compares(
    blocks: list[list[tuple[int, int]]],
) -> None:
    assert unit_intracluster_correlation(blocks) is None


def test_inconsistent_units_are_rejected() -> None:
    for blocks in ([[(2, 1)], []], [[(0, 0), (2, 1)]], [[(2, 3), (2, 1)]], [[(2, -1)]]):
        with pytest.raises(ValueError, match="every block needs a unit"):
            unit_intracluster_correlation(blocks)


CONVERSATIONS, PROMPTS, ROLLOUTS = 10, 8, 10


def _warning_rate(rng: np.random.Generator, runs: int, conversation_sd: float) -> float:
    """How often the design effect on the units of a prompt-level primary reaches the warning
    level, for 10 conversations of 8 prompts with 10 rollouts each.

    A prompt's failure probability is the logistic of -1 plus a prompt effect N(0, 1) plus,
    shared by the prompts of a conversation, a conversation effect N(0, conversation_sd).
    """

    warned = 0
    for _ in range(runs):
        shift = rng.normal(0.0, conversation_sd, size=(CONVERSATIONS, 1))
        prompts = rng.normal(0.0, 1.0, size=(CONVERSATIONS, PROMPTS))
        probability = 1 / (1 + np.exp(1.0 - shift - prompts))
        failures = rng.binomial(ROLLOUTS, probability)
        _, effect = unit_design_effect(
            [[(ROLLOUTS, int(count)) for count in row] for row in failures]
        )
        warned += effect is not None and effect >= DESIGN_EFFECT_WARNING
    return warned / runs


def test_prompt_differences_without_a_conversation_effect_seldom_warn() -> None:
    # Prompts differ widely and each has ten rollouts; conversations add nothing. On rollouts,
    # the design effect reaches 2 in about 60% of such runs, and on the units with m counted
    # in rollouts in about 35%; on the units with m counted in units, in about 2.5%. With 600
    # runs the standard error is about 0.6%.
    rate = _warning_rate(np.random.default_rng(20261119), runs=600, conversation_sd=0.0)
    assert rate <= 0.06


def test_a_conversation_effect_warns() -> None:
    # A conversation effect with standard deviation 1.5 on the logit scale, beside prompt
    # effects of 1: the design effect on the units reaches 2 in about 99% of runs. With the
    # correlation taken on rollouts and m in units, it does in about 88%.
    rate = _warning_rate(np.random.default_rng(20261120), runs=200, conversation_sd=1.5)
    assert rate >= 0.95
