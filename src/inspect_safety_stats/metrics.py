"""Inspect metrics: finite-sample bounds on the failure rate, reported as the eval runs.

Add them to a scorer or task and the Inspect log carries the interval alongside the usual
metrics::

    from inspect_safety_stats.metrics import failure_rate, failure_rate_lower, failure_rate_upper

    @task
    def refusal_eval():
        return Task(
            dataset=...,
            solver=generate(),
            scorer=model_graded_qa(),
            metrics=[
                failure_rate(failure_values=["I"]),
                failure_rate_lower(failure_values=["I"]),
                failure_rate_upper(failure_values=["I"]),
            ],
            epochs=Epochs(8),
        )

The score mapping is always explicit: name the failure values, or describe a graded scale.

The metrics receive every epoch's score rather than one reduced score per sample (Inspect's
``scores="unreduced"``), so they bound the same estimand as ``analyze``: the mean, over the
fixed samples with equal weight, of each sample's per-rollout failure probability, with
rollouts independent. The bound is the weighted KL/Chernoff interval (Bernoulli KL when every
sample has the same number of epochs), valid at any sample size and for graded harm in
[0, 1]. ``alpha`` is two-sided: ``failure_rate_lower`` and ``failure_rate_upper`` with the same
``alpha`` form one ``1 - alpha`` interval, each endpoint at ``alpha / 2``.

They are a quick view, not a replacement for ``analyze``. Each metric covers one scorer on one
task with no multiplicity across models, groups or other metrics, no judge correction and no
prompt-level sensitivity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from inspect_ai.scorer import Metric, SampleScore, Value, metric

from inspect_safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds

ScalarValue = str | int | float | bool

__all__ = ["failure_rate", "failure_rate_lower", "failure_rate_upper", "rate_bounds"]


def _matches(expected: ScalarValue, value: Any) -> bool:
    """Whether a score value equals a declared value, tolerating string spellings."""

    argument = str(expected)
    if isinstance(expected, bool) or isinstance(value, bool):
        return argument.strip().lower() == str(value).strip().lower()
    if str(value) == argument:
        return True
    if isinstance(value, (int, float)):
        try:
            return float(argument) == float(value)
        except ValueError:
            return False
    return False


def _harm_function(
    failure_values: Sequence[ScalarValue] | None,
    safe_values: Sequence[ScalarValue] | None,
    harm_scale: tuple[float, float] | None,
    lower_is_harmful: bool,
) -> Callable[[Any], float | None]:
    """Map a score value to harm in [0, 1], or ``None`` when it cannot be classified."""

    if (failure_values is None) == (harm_scale is None):
        raise ValueError(
            "declare the score mapping: pass failure_values for a verdict, or harm_scale for a "
            "graded score (exactly one). The package never infers whether a value such as C, I, "
            "1 or 0 means a failure."
        )
    if harm_scale is not None:
        if safe_values is not None:
            raise ValueError("safe_values applies to failure_values, not to a harm_scale")
        minimum, maximum = (float(value) for value in harm_scale)
        if not (math.isfinite(minimum) and math.isfinite(maximum) and minimum < maximum):
            raise ValueError("harm_scale must be two finite numbers, minimum below maximum")

        def graded(value: Any) -> float | None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number) or number < minimum or number > maximum:
                return None
            fraction = (number - minimum) / (maximum - minimum)
            return 1.0 - fraction if lower_is_harmful else fraction

        return graded

    failures = list(failure_values or [])
    if not failures:
        raise ValueError("failure_values must name at least one value")
    safes = None if safe_values is None else list(safe_values)
    overlap = [value for value in failures if safes and any(_matches(value, s) for s in safes)]
    if overlap:
        raise ValueError(f"values named as both failure and safe: {overlap}")

    def verdict(value: Any) -> float | None:
        if isinstance(value, (list, dict)) or value is None:
            return None
        if any(_matches(expected, value) for expected in failures):
            return 1.0
        if safes is None or any(_matches(expected, value) for expected in safes):
            return 0.0
        return None

    return verdict


@dataclass(frozen=True)
class RateBounds:
    """The estimate and interval the metrics report."""

    estimate: float
    lower: float
    upper: float
    samples: int
    rollouts: int
    unclassified: int


def rate_bounds(
    scores: Sequence[SampleScore],
    *,
    failure_values: Sequence[ScalarValue] | None = None,
    safe_values: Sequence[ScalarValue] | None = None,
    harm_scale: tuple[float, float] | None = None,
    lower_is_harmful: bool = False,
    key: str | None = None,
    alpha: float = 0.05,
) -> RateBounds:
    """Failure-rate estimate and two-sided ``1 - alpha`` KL/Chernoff interval from scores.

    Scores are grouped into samples by ``sample_id``; each sample carries equal weight and
    each of its rollouts an equal share of it. Unclassifiable values (outside the declared
    mapping, off the scale, or missing ``key``) are completed as harmless for the estimate
    and the lower bound and as most harmful for the upper bound, so they widen the interval.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    harm = _harm_function(failure_values, safe_values, harm_scale, lower_is_harmful)
    by_sample: dict[Any, list[float | None]] = {}
    for index, item in enumerate(scores):
        value: Any = item.score.value
        if key is not None:
            value = value.get(key) if isinstance(value, dict) else None
        sample = item.sample_id if item.sample_id is not None else ("__position__", index)
        by_sample.setdefault(sample, []).append(harm(value))
    if not by_sample:
        return RateBounds(math.nan, math.nan, math.nan, 0, 0, 0)

    low: list[float] = []
    high: list[float] = []
    weights: list[float] = []
    unclassified = 0
    for outcomes in by_sample.values():
        share = 1.0 / (len(by_sample) * len(outcomes))
        for outcome in outcomes:
            unclassified += outcome is None
            low.append(0.0 if outcome is None else outcome)
            high.append(1.0 if outcome is None else outcome)
            weights.append(share)
    bounds = weighted_bounded_kl_chernoff_mean_bounds(
        low,
        high,
        weights,
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        independence_profile="rollout",
    )
    return RateBounds(
        estimate=bounds.completion_estimate_lower,
        lower=bounds.lower,
        upper=bounds.upper,
        samples=len(by_sample),
        rollouts=len(low),
        unclassified=unclassified,
    )


def _metric_function(
    pick: Callable[[RateBounds], float],
    failure_values: Sequence[ScalarValue] | None,
    safe_values: Sequence[ScalarValue] | None,
    harm_scale: tuple[float, float] | Sequence[float] | None,
    lower_is_harmful: bool,
    key: str | None,
    alpha: float,
) -> Metric:
    scale = None if harm_scale is None else (float(harm_scale[0]), float(harm_scale[1]))
    if harm_scale is not None and len(harm_scale) != 2:
        raise ValueError("harm_scale must be two numbers: minimum and maximum")
    # Validate the mapping and alpha when the task is defined, not when the eval finishes.
    _harm_function(failure_values, safe_values, scale, lower_is_harmful)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    def compute(scores: list[SampleScore]) -> Value:
        return pick(
            rate_bounds(
                scores,
                failure_values=failure_values,
                safe_values=safe_values,
                harm_scale=scale,
                lower_is_harmful=lower_is_harmful,
                key=key,
                alpha=alpha,
            )
        )

    return compute


@metric(scores="unreduced")
def failure_rate(
    failure_values: list[ScalarValue] | None = None,
    safe_values: list[ScalarValue] | None = None,
    harm_scale: list[float] | None = None,
    lower_is_harmful: bool = False,
    key: str | None = None,
) -> Metric:
    """Mean failure rate (mean harm for a graded scale) over samples, equally weighted.

    Unclassifiable values count as harmless here; ``failure_rate_upper`` counts them as
    failures.

    Args:
      failure_values: Score values that are safety failures. Every other value is safe
        unless ``safe_values`` is given, in which case unlisted values are unclassified.
      safe_values: Score values that are safe, when not every other value is.
      harm_scale: ``[minimum, maximum]`` of a graded score, mapped linearly onto harm in
        [0, 1]. Use instead of ``failure_values``.
      lower_is_harmful: For ``harm_scale``, whether the minimum is the most harmful grade.
      key: For a scorer whose value is a dict, the entry to read.
    """

    return _metric_function(
        lambda result: result.estimate,
        failure_values, safe_values, harm_scale, lower_is_harmful, key, 0.05,
    )


@metric(scores="unreduced")
def failure_rate_lower(
    failure_values: list[ScalarValue] | None = None,
    safe_values: list[ScalarValue] | None = None,
    harm_scale: list[float] | None = None,
    lower_is_harmful: bool = False,
    key: str | None = None,
    alpha: float = 0.05,
) -> Metric:
    """Lower endpoint of the two-sided ``1 - alpha`` KL/Chernoff interval on the failure rate.

    Takes the arguments of ``failure_rate``, plus ``alpha``.
    """

    return _metric_function(
        lambda result: result.lower,
        failure_values, safe_values, harm_scale, lower_is_harmful, key, alpha,
    )


@metric(scores="unreduced")
def failure_rate_upper(
    failure_values: list[ScalarValue] | None = None,
    safe_values: list[ScalarValue] | None = None,
    harm_scale: list[float] | None = None,
    lower_is_harmful: bool = False,
    key: str | None = None,
    alpha: float = 0.05,
) -> Metric:
    """Upper endpoint of the two-sided ``1 - alpha`` KL/Chernoff interval on the failure rate.

    Takes the arguments of ``failure_rate``, plus ``alpha``. When every sample ran the same
    number of epochs and none failed, this is the exact zero-failure bound
    ``1 - (alpha / 2) ** (1 / rollouts)``.
    """

    return _metric_function(
        lambda result: result.upper,
        failure_values, safe_values, harm_scale, lower_is_harmful, key, alpha,
    )
