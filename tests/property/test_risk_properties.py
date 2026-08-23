from hypothesis import given
from hypothesis import strategies as st

from inspect_safety_stats.stats import entropic_risk, weighted_cvar


@given(
    values=st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=20),
    shift=st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_entropic_and_cvar_are_coordinatewise_monotone(values: list[float], shift: float) -> None:
    upper = [min(1.0, value + shift) for value in values]
    weights = [1 / len(values)] * len(values)
    assert (
        entropic_risk(values, weights, lambda_=5)
        <= entropic_risk(upper, weights, lambda_=5) + 1e-12
    )
    assert (
        weighted_cvar(values, weights, beta=0.8) <= weighted_cvar(upper, weights, beta=0.8) + 1e-12
    )
