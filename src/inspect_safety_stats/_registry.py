"""Entry point that registers this package's Inspect extensions (its metrics)."""

from inspect_safety_stats.metrics import failure_rate, failure_rate_lower, failure_rate_upper

__all__ = ["failure_rate", "failure_rate_lower", "failure_rate_upper"]
