from __future__ import annotations

import pytest

from inspect_safety_stats.analysis.design_effect import design_effect, intracluster_correlation


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
