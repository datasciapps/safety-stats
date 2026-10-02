from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import OutcomeMappingSpec


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if missing is pd.NA:
            return True
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _tag(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ("float", str(value))
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    return (type(value).__name__, value)


def map_outcome(value: Any, mapping: OutcomeMappingSpec) -> bool | None:
    if is_missing(value):
        return None
    if mapping.failure_if is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"numeric score mapping received nonnumeric value {value!r}")
        threshold = mapping.failure_if.value
        operations = {
            "greater_than": value > threshold,
            "greater_than_or_equal": value >= threshold,
            "less_than": value < threshold,
            "less_than_or_equal": value <= threshold,
        }
        return bool(operations[mapping.failure_if.operator])
    tagged = _tag(value)
    failures = {_tag(item) for item in mapping.failure_values or []}
    safe = {_tag(item) for item in mapping.safe_values or []}
    if failures & safe:
        raise ValueError("failure_values and safe_values overlap under type-sensitive matching")
    if tagged in failures:
        return True
    if tagged in safe:
        return False
    raise ValueError(f"unmapped score value {value!r} ({tagged[0]})")


def map_harm(value: Any, mapping: OutcomeMappingSpec) -> float | None:
    """Map a raw score onto harm in [0, 1].

    Binary mappings give 0.0 or 1.0. A ``scale`` mapping places a numeric score
    linearly on the declared range; a value outside the range is an error rather than
    being clipped, because it means the declared scale does not describe the scorer.
    """

    if mapping.scale is None:
        failure = map_outcome(value, mapping)
        return None if failure is None else float(failure)
    if is_missing(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"graded score mapping received nonnumeric value {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"graded score mapping received nonfinite value {value!r}")
    scale = mapping.scale
    tolerance = 1e-9 * (scale.maximum - scale.minimum)
    if numeric < scale.minimum - tolerance or numeric > scale.maximum + tolerance:
        raise ValueError(
            f"score value {value!r} lies outside the declared scale "
            f"[{scale.minimum:g}, {scale.maximum:g}]"
        )
    fraction = (numeric - scale.minimum) / (scale.maximum - scale.minimum)
    fraction = min(1.0, max(0.0, fraction))
    return fraction if scale.harmful == "high" else 1.0 - fraction
