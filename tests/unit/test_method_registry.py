"""The direct-mean methods are named once, in ``config.methods``, and dispatched once."""

from __future__ import annotations

from typing import get_args

import pytest

from inspect_safety_stats.config.methods import COMPARISON_METHODS, DIRECT_MEAN_METHODS
from inspect_safety_stats.config.models import (
    ComparisonSetSpec,
    ComparisonSpec,
    DirectMeanIntervalsSpec,
)
from inspect_safety_stats.stats.direct_mean import DIRECT_MEAN_ENGINES


def test_the_dispatch_table_covers_exactly_the_registered_methods() -> None:
    assert tuple(DIRECT_MEAN_ENGINES) == DIRECT_MEAN_METHODS


def test_the_comparison_methods_are_direct_mean_methods() -> None:
    assert COMPARISON_METHODS == DIRECT_MEAN_METHODS[: len(COMPARISON_METHODS)]
    assert "weighted_betting" in DIRECT_MEAN_METHODS
    assert "weighted_betting" not in COMPARISON_METHODS


@pytest.mark.parametrize("method", DIRECT_MEAN_METHODS)
def test_each_engine_reports_its_own_method(method: str) -> None:
    engine = DIRECT_MEAN_ENGINES[method]  # type: ignore[index]
    bounded = engine.bounded(
        [0.1, 0.2],
        [0.1, 0.3],
        [0.5, 0.5],
        alpha_lower=0.05,
        alpha_upper=0.05,
        independence_profile="rollout",
    )
    counts = engine.counts(
        [1, 2], [0, 1], [10, 10], [0.5, 0.5], alpha_lower=0.05, alpha_upper=0.05
    )
    assert bounded.method == counts.method == method


@pytest.mark.parametrize(
    ("spec", "methods"),
    [
        (DirectMeanIntervalsSpec, DIRECT_MEAN_METHODS),
        (ComparisonSpec, COMPARISON_METHODS),
        (ComparisonSetSpec, COMPARISON_METHODS),
    ],
)
@pytest.mark.parametrize("field", ["method", "reference_methods"])
def test_plan_specs_accept_exactly_their_registered_methods(
    spec: type, methods: tuple[str, ...], field: str
) -> None:
    annotation = spec.model_fields[field].annotation  # type: ignore[attr-defined]
    if field == "reference_methods":
        (annotation,) = get_args(annotation)
    assert get_args(annotation) == methods
