import pytest

from inspect_safety_stats.stats import bonferroni_allocate


def test_bonferroni_allocations_sum_to_alpha() -> None:
    equal = bonferroni_allocate(0.05, 4)
    weighted = bonferroni_allocate(0.05, 2, weights=[1, 3])
    assert sum(equal) == pytest.approx(0.05)
    assert weighted == pytest.approx([0.0125, 0.0375])
