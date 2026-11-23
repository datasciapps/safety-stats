"""Risk measure values, parameters, thresholds and coverage floors."""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np

from safety_stats.config.models import (
    ComparisonSpec,
    DirectMeanIntervalsSpec,
    IndependenceSelectionSpec,
    RiskMeasureSpec,
)
from safety_stats.ingest.records import NormalizedData
from safety_stats.stats.risk import (
    entropic_risk,
    maximum_risk,
    threshold_exceedance,
    threshold_excess,
    top_k_mean,
    weighted_cvar,
    weighted_mean,
    worst_of_k_risk,
)


def _thresholds(
    measure: RiskMeasureSpec,
    prompt_ids: list[str],
    data: NormalizedData,
) -> float | np.ndarray:
    if measure.threshold is not None:
        return measure.threshold
    if measure.threshold_column is None:
        raise ValueError(
            f"risk measure {measure.id!r} ({measure.type}) needs threshold or threshold_column"
        )
    manifest = data.manifest.set_index("prompt_id")
    if measure.threshold_column not in manifest:
        raise ValueError(f"manifest has no threshold column {measure.threshold_column!r}")
    return np.asarray(
        [manifest.loc[prompt_id, measure.threshold_column] for prompt_id in prompt_ids],
        dtype=float,
    )


# Measures that are probabilities whenever every value is. Threshold excess is not one: a
# negative threshold lets it exceed 1.
PROBABILITY_MEASURES = frozenset(
    {
        "weighted_mean",
        "maximum",
        "cvar",
        "top_k_mean",
        "entropic",
        "worst_of_k",
        "threshold_exceedance",
    }
)


def _risk_value(
    values: list[float],
    weights: list[float],
    measure: RiskMeasureSpec,
    *,
    threshold: float | np.ndarray | None = None,
) -> float:
    value = unclipped_risk_value(values, weights, measure, threshold=threshold)
    if measure.type in PROBABILITY_MEASURES:
        # Floating-point sums can land just outside [0, 1]: nine weights of 1/9 sum to
        # 1.0000000000000002. A judge correction then rejects the bound as no probability.
        return min(1.0, max(0.0, value))
    return value


def unclipped_risk_value(
    values: list[float],
    weights: list[float],
    measure: RiskMeasureSpec,
    *,
    threshold: float | np.ndarray | None = None,
) -> float:
    if measure.type == "weighted_mean":
        return weighted_mean(values, weights)
    if measure.type == "maximum":
        return maximum_risk(values, weights)
    if measure.type == "cvar":
        return weighted_cvar(values, weights, beta=_parameter(measure, "beta", measure.beta))
    if measure.type == "top_k_mean":
        k = _parameter(measure, "k", measure.k)
        if not np.allclose(weights, np.full(len(weights), 1 / len(weights))):
            raise ValueError("top_k_mean requires equal prompt weights")
        return top_k_mean(values, k=int(k))
    if measure.type == "entropic":
        lambda_ = _parameter(measure, "lambda", measure.lambda_)
        return entropic_risk(values, weights, lambda_=lambda_)
    if measure.type == "worst_of_k":
        return worst_of_k_risk(values, weights, k=int(_parameter(measure, "k", measure.k)))
    if measure.type in {"threshold_exceedance", "threshold_excess"}:
        if threshold is None:
            raise ValueError(
                f"risk measure {measure.id!r} ({measure.type}) was given no threshold"
            )
        if measure.type == "threshold_exceedance":
            return threshold_exceedance(values, weights, threshold=threshold)
        return threshold_excess(values, weights, threshold=threshold)
    raise ValueError(f"unsupported risk measure {measure.type!r}")


def _parameter(measure: RiskMeasureSpec, name: str, value: float | None) -> float:
    """A parameter the measure's type requires.

    Plan validation guarantees it. Unlike an assert, this check survives python -O.
    """
    if value is None:
        raise ValueError(f"risk measure {measure.id!r} ({measure.type}) requires {name}")
    return float(value)


def _measure_parameters(measure: RiskMeasureSpec) -> dict[str, Any]:
    dumped = measure.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {
        key: value for key, value in dumped.items() if key not in {"id", "type", "diagnostic_only"}
    }


def _independence(spec: DirectMeanIntervalsSpec | ComparisonSpec) -> IndependenceSelectionSpec:
    """The independence profiles an interval declares.

    Plan validation requires them wherever they are used. Unlike an assert, this check
    survives python -O.
    """
    if spec.independence is None:
        where = (
            f"comparison {spec.id!r}"
            if isinstance(spec, ComparisonSpec)
            else "outputs.direct_mean_intervals"
        )
        raise ValueError(f"{where} declares no independence profiles")
    return spec.independence


@overload
def _coverage_floors(
    family_alpha: float, coverage_mode: Literal["two_sided", "per_direction"]
) -> tuple[float, float, float]: ...


@overload
def _coverage_floors(
    family_alpha: float | None, coverage_mode: Literal["two_sided", "per_direction"]
) -> tuple[float | None, float | None, float | None]: ...


def _coverage_floors(
    family_alpha: float | None,
    coverage_mode: Literal["two_sided", "per_direction"],
) -> tuple[float | None, float | None, float | None]:
    directional_alpha = (
        family_alpha / 2
        if family_alpha is not None and coverage_mode == "two_sided"
        else family_alpha
    )
    one_sided = 1 - directional_alpha if directional_alpha is not None else None
    joint = (
        1 - family_alpha
        if family_alpha is not None and coverage_mode == "two_sided"
        else max(0.0, 1 - 2 * family_alpha)
        if family_alpha is not None
        else None
    )
    return directional_alpha, one_sided, joint
