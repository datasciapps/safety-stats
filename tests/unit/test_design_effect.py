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


def test_units_follow_the_pairwise_estimator_on_their_failure_rates() -> None:
    # Rates 1/2, 1 | 0, 1, 0 | 2/5: K = 6 around the mean 29/60, so in 60ths the deviations
    # are 1, 31 | -29, 31, -29 | -5.
    # S^2 = (1 + 961 + 841 + 961 + 841 + 25) / 3600 / 6 = 121/720
    # Pairs within blocks, in 3600ths: 32^2 - 962 = 62 and (-27)^2 - 2643 = -1914, and the
    # single unit gives none, over 2 + 6 = 8 ordered pairs: C = -1852 / 3600 / 8.
    blocks = [[(2, 1), (4, 4)], [(3, 0), (1, 1), (2, 0)], [(5, 2)]]
    expected = (-1852 / 3600 / 8) / (121 / 720)
    assert expected == pytest.approx(-463 / 1210, rel=1e-12, abs=0)
    assert unit_intracluster_correlation(blocks) == pytest.approx(expected, rel=1e-12, abs=0)
    assert unit_design_effect(blocks) == (unit_intracluster_correlation(blocks), 1.0)


def test_the_design_effect_on_units_is_the_variance_ratio_of_block_sums() -> None:
    # 1 + (m - 1) rho with m = sum k_g^2 / K is sum_g (sum_u d_u)^2 / sum_u d_u^2: the
    # variance of the units' total with blocks as independent, over that with units as
    # independent. It is at most the largest block's number of units, though the correlation
    # can pass 1 when blocks differ in size. Rates 1, 1, 1/3, 1/3, 1, 1 around 7/9: deviations
    # 2/9, 2/9, -4/9, -4/9, 2/9, 2/9 in blocks of 1, 1, 2, 1, 1 units. Block sums squared
    # (4 + 4 + 64 + 4 + 4) / 81 over (4 + 4 + 16 + 16 + 4 + 4) / 81 is 5/3; m = 8/6, so
    # rho = 2.
    blocks = [[(3, 3)], [(2, 2)], [(3, 1), (3, 1)], [(2, 2)], [(1, 1)]]
    correlation, effect = unit_design_effect(blocks)
    assert correlation == pytest.approx(2.0, rel=1e-12, abs=0)
    assert effect == pytest.approx(5 / 3, rel=1e-12, abs=0)
    rng = np.random.default_rng(20261121)
    checked = 0
    for _ in range(1000):
        layout = [
            [(int(rollouts), 0) for rollouts in rng.integers(1, 6, size=rng.integers(1, 7))]
            for _ in range(int(rng.integers(2, 6)))
        ]
        layout = [[(n, int(rng.integers(0, n + 1))) for n, _ in units] for units in layout]
        correlation, effect = unit_design_effect(layout)
        if correlation is None or effect is None or correlation <= 0:
            continue
        if all(n == 1 for units in layout for n, _ in units):
            continue
        rates = [failed / n for units in layout for n, failed in units]
        mean = sum(rates) / len(rates)
        sums = [sum(failed / n - mean for n, failed in units) for units in layout]
        ratio = sum(value * value for value in sums) / sum((rate - mean) ** 2 for rate in rates)
        assert effect == pytest.approx(ratio, rel=1e-9, abs=0)
        assert effect <= max(len(units) for units in layout) * (1 + 1e-12)
        checked += 1
    assert checked > 100


@pytest.mark.parametrize(
    "failed",
    [
        [[1, 1, 1, 0], [0, 1, 0, 0], [0, 0]],
        # Without the branch to the binary estimate, the pairwise one gives -0.0556 here. Even
        # the one-way ANOVA estimate on the rates, which are the outcomes, differs from the
        # binary one in the last bits for this layout: -0.06449938355719945 against ...938.
        [[1, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1], [0] * 7, [0, 0, 0, 0, 1, 0, 1], [1, 0]],
    ],
)
def test_units_of_one_rollout_give_the_binary_estimate_to_the_bit(failed: list[list[int]]) -> None:
    blocks = [[(1, outcome) for outcome in block] for block in failed]
    assert unit_intracluster_correlation(blocks) == intracluster_correlation(
        [len(block) for block in failed], [sum(block) for block in failed]
    )


@pytest.mark.parametrize(
    "blocks",
    [
        [[(10, 3), (10, 5)]],  # one block
        [[(10, 3)], [(10, 5)]],  # no block with two units
        [[(10, 5), (4, 2)], [(2, 1), (6, 3)]],  # every unit fails half its rollouts
        # Every unit fails a third of its rollouts. The rates are one float, but their mean is
        # one bit below it, so every deviation is that bit: computed in floats, the
        # correlation would be 1 and the design effect 5, a ratio of rounding errors.
        [[(3, 1), (6, 2), (9, 3), (12, 4), (15, 5)]] * 4 + [[(3, 1)] * 5],
    ],
)
def test_unit_correlation_is_undefined_without_the_variation_it_compares(
    blocks: list[list[tuple[int, int]]],
) -> None:
    assert unit_intracluster_correlation(blocks) is None
    assert unit_design_effect(blocks) == (None, None)


def test_inconsistent_units_are_rejected() -> None:
    for blocks in ([[(2, 1)], []], [[(0, 0), (2, 1)]], [[(2, 3), (2, 1)]], [[(2, -1)]]):
        with pytest.raises(ValueError, match="every block needs a unit"):
            unit_intracluster_correlation(blocks)


# 10 conversations of 8 prompts with 10 rollouts each.
GRID = [[10] * 8] * 10


def _warning_rate(
    rng: np.random.Generator, runs: int, layout: list[list[int]], conversation_sd: float
) -> float:
    """How often the design effect on the units of a prompt-level primary reaches the warning
    level. ``layout`` gives each conversation's prompts by their numbers of rollouts.

    A prompt's failure probability is the logistic of -1 plus a prompt effect N(0, 1) plus,
    shared by the prompts of a conversation, a conversation effect N(0, conversation_sd).
    """

    warned = 0
    for _ in range(runs):
        blocks = []
        for rollouts in layout:
            shift = rng.normal(0.0, conversation_sd)
            prompts = rng.normal(0.0, 1.0, size=len(rollouts))
            probability = 1 / (1 + np.exp(1.0 - shift - prompts))
            blocks.append(
                [(n, int(rng.binomial(n, p))) for n, p in zip(rollouts, probability, strict=True)]
            )
        _, effect = unit_design_effect(blocks)
        warned += effect is not None and effect >= DESIGN_EFFECT_WARNING
    return warned / runs


def test_prompt_differences_without_a_conversation_effect_seldom_warn() -> None:
    # Prompts differ widely and each has ten rollouts; conversations add nothing. On rollouts,
    # the design effect reaches 2 in about 60% of such runs, and on the units with m counted
    # in rollouts in about 29%; on the units with m counted in units, in about 1%. With 600
    # runs the standard error is about 0.4%.
    rate = _warning_rate(np.random.default_rng(20261119), 600, GRID, conversation_sd=0.0)
    assert rate <= 0.04


def test_a_conversation_effect_warns() -> None:
    # A conversation effect with standard deviation 1.5 on the logit scale, beside prompt
    # effects of 1: the design effect on the units reaches 2 in about 99% of runs. With the
    # correlation taken on rollouts and m in units, it does in about 85%.
    rate = _warning_rate(np.random.default_rng(20261120), 200, GRID, conversation_sd=1.5)
    assert rate >= 0.95


@pytest.mark.parametrize(
    ("layout", "seed"),
    [
        # Five conversations of two prompts with one rollout each, beside one of 40 prompts
        # with 20 rollouts each: the one-way ANOVA estimate on the units' rates warns in about
        # 95% of runs, the pairwise one in about 5%.
        ([[1, 1]] * 5 + [[20] * 40], 20261121),
        # One conversation of 40 prompts with 10 rollouts each, beside 20 of two prompts with
        # one rollout each: about 89% and 3%.
        ([[10] * 40] + [[1, 1]] * 20, 20261122),
    ],
)
def test_noisy_units_in_small_blocks_seldom_warn(layout: list[list[int]], seed: int) -> None:
    # Single rollouts have rates of 0 or 1, far noisier than the rates of 20 rollouts. Small
    # blocks of them give block means that spread widely, beside a large block of quiet units
    # whose spread within it is small. Conversations add nothing. With 500 runs the standard
    # error at 5% is about 1%.
    rate = _warning_rate(np.random.default_rng(seed), 500, layout, conversation_sd=0.0)
    assert rate <= 0.08


def test_few_blocks_can_warn_with_no_grouping_effect() -> None:
    # The limit docs/independence-profiles.md states: with two conversations of 40 prompts and
    # no conversation effect, the warning fires in about 17% of runs when each prompt has one
    # rollout, which take the one-way ANOVA estimate, and in about 5.5% with ten, which take
    # the pairwise one. With 600 runs the standard errors are about 1.5% and 0.9%.
    single = _warning_rate(np.random.default_rng(20261123), 600, [[1] * 40] * 2, 0.0)
    repeated = _warning_rate(np.random.default_rng(20261124), 600, [[10] * 40] * 2, 0.0)
    assert 0.11 <= single <= 0.23
    assert repeated <= 0.09
