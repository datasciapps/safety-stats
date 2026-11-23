"""Conformance checks every risk measure must pass, through the pipeline's own dispatch.

``tests/property/test_risk_properties.py`` checks the measure functions one by one. This file
runs the same checks over every measure type a plan accepts, through
``unclipped_risk_value``, the function the pipeline itself calls, and with unequal weights for
every measure that admits them. ``test_the_registry_lists_every_measure_a_plan_can_select``
fails until a new measure type is registered in ``tests/conformance.py``.

It also checks the two properties the analysis declares by hand, against what the measures do:

- ``PROBABILITY_MEASURES``: the value is a probability whenever every item's risk is.
- ``JUDGE_CORRECTABLE_MEASURES``: the measure commutes with the judge correction
  ``r -> rho_0 + (rho_1 - rho_0) r``, so correcting the measure's value is the same as
  correcting every item first.

Each declaration is checked in both directions. Every member must have the property, and
every other measure must visibly lack it, so neither list can drift from the code.
"""

from __future__ import annotations

import math
from typing import Any, get_args

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from safety_stats.analysis.judge_calibration import JUDGE_CORRECTABLE_MEASURES
from safety_stats.analysis.measures import PROBABILITY_MEASURES, unclipped_risk_value
from safety_stats.config.models import RiskMeasureSpec
from tests.conformance import MEASURE_CASES, MeasureCase

TOLERANCE = 1e-9

CASES = [case for cases in MEASURE_CASES.values() for case in cases]


def _case_id(case: MeasureCase) -> str:
    parameters = case.spec.model_dump(
        exclude_none=True, exclude={"id", "type", "diagnostic_only", "threshold_column"}
    )
    return "-".join([case.spec.type, *(f"{key}={value}" for key, value in parameters.items())])


def _cases(keep: Any = lambda case: True) -> Any:
    return pytest.mark.parametrize("case", [c for c in CASES if keep(c)], ids=_case_id)
RISKS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _value(case: MeasureCase, values: list[float], weights: list[float]) -> float:
    return unclipped_risk_value(values, weights, case.spec, threshold=case.threshold)


def _applies(case: MeasureCase, values: list[float]) -> bool:
    return case.spec.k is None or case.spec.type != "top_k_mean" or len(values) >= case.spec.k


@st.composite
def designs(draw: st.DrawFn, case: MeasureCase) -> tuple[list[float], list[float]]:
    """Risks and weights summing to one; equal weights for a measure that requires them."""

    size = draw(st.integers(min_value=1, max_value=10))
    values = draw(st.lists(RISKS, min_size=size, max_size=size))
    if case.equal_weights_only:
        return values, [1 / size] * size
    exponents = st.floats(min_value=-3.0, max_value=0.0, allow_nan=False)
    raw = [10**exponent for exponent in draw(st.lists(exponents, min_size=size, max_size=size))]
    total = math.fsum(raw)
    return values, [weight / total for weight in raw]


def _draw(data: st.DataObject, case: MeasureCase) -> tuple[list[float], list[float]]:
    values, weights = data.draw(designs(case), label="design")
    assume(_applies(case, values))
    return values, weights


def test_the_registry_lists_every_measure_a_plan_can_select() -> None:
    selectable = set(get_args(RiskMeasureSpec.model_fields["type"].annotation))
    assert selectable == set(MEASURE_CASES), "register new measure types in tests/conformance.py"
    assert PROBABILITY_MEASURES <= selectable
    assert JUDGE_CORRECTABLE_MEASURES <= selectable


@_cases()
@settings(deadline=None)
@given(data=st.data())
def test_raising_any_items_risk_never_lowers_the_measure(
    case: MeasureCase, data: st.DataObject
) -> None:
    """Coordinatewise monotonicity, the property that makes endpoint propagation a bound."""

    values, weights = _draw(data, case)
    raised = [
        min(1.0, value + data.draw(RISKS, label=f"raise {index}"))
        for index, value in enumerate(values)
    ]
    assert _value(case, values, weights) <= _value(case, raised, weights) + TOLERANCE


@_cases()
@settings(deadline=None)
@given(data=st.data())
def test_the_order_of_items_does_not_matter(case: MeasureCase, data: st.DataObject) -> None:
    values, weights = _draw(data, case)
    order = data.draw(st.permutations(range(len(values))))
    shuffled = _value(case, [values[i] for i in order], [weights[i] for i in order])
    assert shuffled == pytest.approx(_value(case, values, weights), rel=0, abs=TOLERANCE)


@_cases(lambda case: case.replication_invariant)
@settings(deadline=None)
@given(
    data=st.data(),
    item=st.integers(min_value=0),
    share=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_splitting_an_items_weight_among_copies_changes_nothing(
    case: MeasureCase, data: st.DataObject, item: int, share: float
) -> None:
    """Replication invariance: the docs advise dividing a duplicated item's weight.

    That advice only preserves the measure if the measure has this property; measures
    evaluated under equal weights only do not take part.
    """

    values, weights = _draw(data, case)
    item %= len(values)
    copied = [*values, values[item]]
    split = [*weights, weights[item] * (1 - share)]
    split[item] = weights[item] * share
    assert _value(case, copied, split) == pytest.approx(
        _value(case, values, weights), rel=0, abs=TOLERANCE
    )


@_cases(lambda case: case.zero_weight_invariant and not case.equal_weights_only)
@settings(deadline=None)
@given(data=st.data(), extra=RISKS)
def test_an_item_without_weight_changes_nothing(
    case: MeasureCase, data: st.DataObject, extra: float
) -> None:
    values, weights = _draw(data, case)
    assert _value(case, [*values, extra], [*weights, 0.0]) == pytest.approx(
        _value(case, values, weights), rel=0, abs=TOLERANCE
    )


# Fixed inputs on which a measure without a property shows it. Each non-member of a declared
# set must fail on at least one of them, so the declaration cannot omit a measure that has
# the property, and cannot list one that lacks it without the first check failing.
_BATTERY: tuple[tuple[list[float], list[float]], ...] = (
    ([1.0], [1.0]),
    ([0.0, 1.0], [0.5, 0.5]),
    ([0.2, 0.9, 0.4], [1 / 3, 1 / 3, 1 / 3]),
    ([0.05, 0.6], [0.5, 0.5]),
)


@_cases(lambda case: case.spec.type in PROBABILITY_MEASURES)
@settings(deadline=None)
@given(data=st.data())
def test_declared_probability_measures_stay_in_the_unit_interval(
    case: MeasureCase, data: st.DataObject
) -> None:
    values, weights = _draw(data, case)
    value = _value(case, values, weights)
    assert -TOLERANCE <= value <= 1 + TOLERANCE


@pytest.mark.parametrize(
    "measure", sorted(set(MEASURE_CASES) - PROBABILITY_MEASURES)
)
def test_undeclared_probability_measures_do_leave_the_unit_interval(measure: str) -> None:
    values = [
        _value(case, battery_values, battery_weights)
        for case in MEASURE_CASES[measure]
        for battery_values, battery_weights in _BATTERY
        if _applies(case, battery_values)
    ]
    assert any(not 0 <= value <= 1 for value in values), (
        f"{measure} stayed in [0, 1] on every input: add it to PROBABILITY_MEASURES"
    )


def _corrected(values: list[float], rho_0: float, rho_1: float) -> list[float]:
    return [rho_0 + (rho_1 - rho_0) * value for value in values]


@_cases(lambda case: case.spec.type in JUDGE_CORRECTABLE_MEASURES)
@settings(deadline=None)
@given(
    data=st.data(),
    rho_0=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    sensitivity=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
)
def test_judge_correctable_measures_commute_with_the_correction(
    case: MeasureCase, data: st.DataObject, rho_0: float, sensitivity: float
) -> None:
    """Correcting each item's risk and then measuring equals measuring and then correcting.

    ``rho_1 > rho_0``, a judge better than chance, keeps the correction increasing.
    """

    values, weights = _draw(data, case)
    rho_1 = max(rho_0, sensitivity)
    measured = _value(case, values, weights)
    assert _value(case, _corrected(values, rho_0, rho_1), weights) == pytest.approx(
        rho_0 + (rho_1 - rho_0) * measured, rel=0, abs=TOLERANCE
    )


@pytest.mark.parametrize(
    "measure", sorted(set(MEASURE_CASES) - JUDGE_CORRECTABLE_MEASURES)
)
def test_measures_not_declared_correctable_do_not_commute_with_the_correction(
    measure: str,
) -> None:
    gaps: list[float] = []
    for case in MEASURE_CASES[measure]:
        for battery_values, battery_weights in _BATTERY:
            if not _applies(case, battery_values):
                continue
            for rho_0, rho_1 in ((0.1, 0.9), (0.25, 0.75)):
                corrected = _value(case, _corrected(battery_values, rho_0, rho_1), battery_weights)
                measured = _value(case, battery_values, battery_weights)
                gaps.append(abs(corrected - (rho_0 + (rho_1 - rho_0) * measured)))
    assert max(gaps) > 1e-6, (
        f"{measure} commuted with the correction on every input: "
        "add it to JUDGE_CORRECTABLE_MEASURES"
    )

