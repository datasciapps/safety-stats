from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, norm


@dataclass(frozen=True)
class McNemarResult:
    candidate_safe_baseline_safe: int
    candidate_safe_baseline_failure: int
    candidate_failure_baseline_safe: int
    candidate_failure_baseline_failure: int
    discordant: int
    estimate: float
    p_value_two_sided: float
    p_value_candidate_worse: float
    p_value_candidate_better: float


@dataclass(frozen=True)
class PairedDifferenceInterval:
    estimate: float
    lower: float
    upper: float
    alpha_lower: float
    alpha_upper: float
    method: str = "newcombe_method_10_one_sided"


def _wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def newcombe_paired_interval(
    candidate: list[bool],
    baseline: list[bool],
    *,
    alpha_lower: float = 0.05,
    alpha_upper: float = 0.05,
) -> PairedDifferenceInterval:
    """Newcombe method 10 score bounds for a paired proportion difference.

    The outcome ``True`` denotes failure and the estimand is candidate minus
    baseline. Separate one-sided error levels are used for the two bounds.
    """
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline must be nonempty and equal length")
    if not all(isinstance(value, bool) for value in candidate + baseline):
        raise TypeError("paired outcomes must be booleans")
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")

    trials = len(candidate)
    candidate_count = sum(candidate)
    baseline_count = sum(baseline)
    candidate_rate = candidate_count / trials
    baseline_rate = baseline_count / trials
    estimate = candidate_rate - baseline_rate
    both_failure = sum(cand and base for cand, base in zip(candidate, baseline, strict=True))
    candidate_only = sum(cand and not base for cand, base in zip(candidate, baseline, strict=True))
    baseline_only = sum(not cand and base for cand, base in zip(candidate, baseline, strict=True))
    both_safe = trials - both_failure - candidate_only - baseline_only
    correlation_numerator = float(both_failure * both_safe - candidate_only * baseline_only)
    if correlation_numerator > 0:
        correlation_numerator = max(correlation_numerator - trials / 2, 0.0)
    correlation_denominator = np.sqrt(
        (both_failure + candidate_only)
        * (baseline_only + both_safe)
        * (both_failure + baseline_only)
        * (candidate_only + both_safe)
    )
    correlation = (
        correlation_numerator / correlation_denominator if correlation_denominator > 0 else 0.0
    )
    correlation = float(np.clip(correlation, -1.0, 1.0))

    def bound(alpha: float, *, lower: bool) -> float:
        z = float(norm.ppf(1 - alpha))
        cand_low, cand_high = _wilson_interval(candidate_count, trials, z)
        base_low, base_high = _wilson_interval(baseline_count, trials, z)
        cand_down = candidate_rate - cand_low
        cand_up = cand_high - candidate_rate
        base_down = baseline_rate - base_low
        base_up = base_high - baseline_rate
        if lower:
            distance_squared = cand_down**2 + base_up**2 - 2 * correlation * cand_down * base_up
            return max(-1.0, estimate - float(np.sqrt(max(0.0, distance_squared))))
        distance_squared = cand_up**2 + base_down**2 - 2 * correlation * cand_up * base_down
        return min(1.0, estimate + float(np.sqrt(max(0.0, distance_squared))))

    return PairedDifferenceInterval(
        estimate=estimate,
        lower=bound(alpha_lower, lower=True),
        upper=bound(alpha_upper, lower=False),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )


def exact_mcnemar(candidate: list[bool], baseline: list[bool]) -> McNemarResult:
    """Exact McNemar test of paired failure outcomes, candidate against baseline.

    ``candidate`` and ``baseline`` are equal-length, nonempty lists of booleans, ``True``
    meaning failure, paired by position. The test is a binomial test with probability one half
    on the discordant pairs. It returns the matched-pairs table, the estimated difference in
    failure rates (candidate minus baseline), and two-sided and both one-sided p-values. All
    p-values are 1 when no pair is discordant. Derived in docs/statistical-methods.md,
    section 7.3.
    """

    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline must be nonempty and equal length")
    if not all(isinstance(x, bool) for x in candidate + baseline):
        raise TypeError("paired outcomes must be booleans")
    n00 = n01 = n10 = n11 = 0
    for cand, base in zip(candidate, baseline, strict=True):
        if not cand and not base:
            n00 += 1
        elif not cand and base:
            n01 += 1
        elif cand and not base:
            n10 += 1
        else:
            n11 += 1
    discordant = n01 + n10
    if discordant == 0:
        two_sided = worse = better = 1.0
    else:
        two_sided = float(binomtest(n10, discordant, 0.5, alternative="two-sided").pvalue)
        worse = float(binomtest(n10, discordant, 0.5, alternative="greater").pvalue)
        better = float(binomtest(n10, discordant, 0.5, alternative="less").pvalue)
    return McNemarResult(
        candidate_safe_baseline_safe=n00,
        candidate_safe_baseline_failure=n01,
        candidate_failure_baseline_safe=n10,
        candidate_failure_baseline_failure=n11,
        discordant=discordant,
        estimate=(n10 - n01) / len(candidate),
        p_value_two_sided=two_sided,
        p_value_candidate_worse=worse,
        p_value_candidate_better=better,
    )
