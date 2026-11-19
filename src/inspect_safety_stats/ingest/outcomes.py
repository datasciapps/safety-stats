from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import OutcomeMappingSpec, _tag

if TYPE_CHECKING:
    from inspect_safety_stats.schema.observations import Observation


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


def classified_by_policy(row: Observation) -> bool:
    """Whether a plan policy, not the primary score's mapping, classified the rollout.

    That is a policy block counted safe, a sample error counted as a failure, a content-filter
    stop or an unscored truncated answer the plan classifies, or a missing score counted as a
    failure. Each leaves its mark on the observation: a sample error, a content-filter stop's
    undelivered content, the token-limit flag, or a missing score.
    """

    return row.failure is not None and (
        row.outcome_status == "target_policy_blocked_safe"
        or row.error is not None
        or row.content_delivered is False
        or bool(row.token_limit_unscored)
        or is_missing(row.score_raw)
    )


def judge_positive(row: Observation) -> bool:
    """Whether the primary judge scored the rollout unsafe: its score mapped to a failure.

    A failure that a plan policy classified, such as a sample error or a missing score counted
    as a failure (:func:`classified_by_policy`), is not judge-positive: no judge scored it.
    """

    return row.failure is True and not classified_by_policy(row)


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
