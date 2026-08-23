import pytest

from inspect_safety_stats.stats import benjamini_hochberg, bonferroni_allocate, holm_adjust


def test_bonferroni_allocations_sum_to_alpha() -> None:
    equal = bonferroni_allocate(0.05, 4)
    weighted = bonferroni_allocate(0.05, 2, weights=[1, 3])
    assert sum(equal) == pytest.approx(0.05)
    assert weighted == pytest.approx([0.0125, 0.0375])


def test_holm_known_values() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bh_known_values() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
