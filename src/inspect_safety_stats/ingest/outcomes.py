from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import OutcomeMappingSpec, _tag


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
