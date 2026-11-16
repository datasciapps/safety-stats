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


# Partial pairing: pairs in which one or both arms are unscored (outcome None).
# The unknown outcomes are bounded by their completions, never imputed.

_PARTIAL_NEWCOMBE_GRID_LIMIT = 50_000_000


@dataclass(frozen=True)
class PartialPairing:
    pair_count: int
    scored_pair_count: int
    candidate_unscored_pair_count: int
    baseline_unscored_pair_count: int
    both_unscored_pair_count: int


@dataclass(frozen=True)
class PartialMcNemarResult:
    scored: McNemarResult | None
    estimate_lower: float
    estimate_upper: float
    p_value_two_sided: float
    p_value_candidate_worse: float
    p_value_candidate_better: float


@dataclass(frozen=True)
class PartialPairedDifferenceInterval:
    estimate_lower: float
    estimate_upper: float
    lower: float
    upper: float
    alpha_lower: float
    alpha_upper: float
    completion_count: int
    method: str = "newcombe_method_10_one_sided"


def _check_partial_pairs(candidate: list[bool | None], baseline: list[bool | None]) -> None:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("candidate and baseline must be nonempty and equal length")
    if not all(value is None or isinstance(value, bool) for value in candidate + baseline):
        raise TypeError("paired outcomes must be booleans or None for unscored")


def partial_pairing(candidate: list[bool | None], baseline: list[bool | None]) -> PartialPairing:
    """Count the pairs by which of their arms are unscored (``None``)."""

    _check_partial_pairs(candidate, baseline)
    candidate_only = sum(
        cand is None and base is not None for cand, base in zip(candidate, baseline, strict=True)
    )
    baseline_only = sum(
        cand is not None and base is None for cand, base in zip(candidate, baseline, strict=True)
    )
    both = sum(
        cand is None and base is None for cand, base in zip(candidate, baseline, strict=True)
    )
    return PartialPairing(
        pair_count=len(candidate),
        scored_pair_count=len(candidate) - candidate_only - baseline_only - both,
        candidate_unscored_pair_count=candidate_only,
        baseline_unscored_pair_count=baseline_only,
        both_unscored_pair_count=both,
    )


def favourable_and_adverse_completions(
    candidate: list[bool | None], baseline: list[bool | None]
) -> tuple[tuple[list[bool], list[bool]], tuple[list[bool], list[bool]]]:
    """The two extreme completions of the unscored outcomes (``True`` is failure).

    The favourable completion scores every unscored candidate outcome safe and every unscored
    baseline outcome a failure, so each pair's difference (candidate minus baseline) is as low as
    any completion makes it. The adverse completion does the opposite, so each difference is as
    high as any completion makes it.
    """

    _check_partial_pairs(candidate, baseline)
    favourable = (
        [bool(value) if value is not None else False for value in candidate],
        [bool(value) if value is not None else True for value in baseline],
    )
    adverse = (
        [bool(value) if value is not None else True for value in candidate],
        [bool(value) if value is not None else False for value in baseline],
    )
    return favourable, adverse


def exact_mcnemar_partial(
    candidate: list[bool | None], baseline: list[bool | None]
) -> PartialMcNemarResult:
    """Exact McNemar p-values that hold for every completion of the unscored outcomes.

    A one-sided p-value P(Binomial(d, 1/2) >= n10) does not decrease when a pair's difference
    decreases: moving a pair out of n10 or into n01 can only raise it. So its largest value over
    all completions is at the favourable completion for ``candidate_worse`` and at the adverse
    completion for ``candidate_better``. The two-sided p-value, min(1, 2 * min of the two tails)
    for this symmetric test, is bounded by the same expression of the two maxima.
    """

    _check_partial_pairs(candidate, baseline)
    (fav_cand, fav_base), (adv_cand, adv_base) = favourable_and_adverse_completions(
        candidate, baseline
    )
    favourable = exact_mcnemar(fav_cand, fav_base)
    adverse = exact_mcnemar(adv_cand, adv_base)
    scored_pairs = [
        (cand, base)
        for cand, base in zip(candidate, baseline, strict=True)
        if cand is not None and base is not None
    ]
    scored = (
        exact_mcnemar(
            [bool(cand) for cand, _ in scored_pairs], [bool(base) for _, base in scored_pairs]
        )
        if scored_pairs
        else None
    )
    worse = favourable.p_value_candidate_worse
    better = adverse.p_value_candidate_better
    return PartialMcNemarResult(
        scored=scored,
        estimate_lower=favourable.estimate,
        estimate_upper=adverse.estimate,
        p_value_two_sided=min(1.0, 2.0 * min(worse, better)),
        p_value_candidate_worse=worse,
        p_value_candidate_better=better,
    )


def _newcombe_bounds_from_counts(
    both_failure: np.ndarray,
    candidate_only: np.ndarray,
    baseline_only: np.ndarray,
    both_safe: np.ndarray,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised newcombe_paired_interval over arrays of the four cell counts."""

    trials = both_failure + candidate_only + baseline_only + both_safe
    candidate_count = both_failure + candidate_only
    baseline_count = both_failure + baseline_only
    candidate_rate = candidate_count / trials
    baseline_rate = baseline_count / trials
    estimate = candidate_rate - baseline_rate
    numerator = (both_failure * both_safe - candidate_only * baseline_only).astype(float)
    numerator = np.where(numerator > 0, np.maximum(numerator - trials / 2, 0.0), numerator)
    denominator = np.sqrt(
        (
            (both_failure + candidate_only)
            * (baseline_only + both_safe)
            * (both_failure + baseline_only)
            * (candidate_only + both_safe)
        ).astype(float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(denominator > 0, numerator / denominator, 0.0)
    correlation = np.clip(correlation, -1.0, 1.0)

    def wilson(successes: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
        proportion = successes / trials
        z_squared = z * z
        scale = 1 + z_squared / trials
        center = (proportion + z_squared / (2 * trials)) / scale
        radius = (
            z * np.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        ) / scale
        return np.maximum(0.0, center - radius), np.minimum(1.0, center + radius)

    z_lower = float(norm.ppf(1 - alpha_lower))
    cand_low, _ = wilson(candidate_count, z_lower)
    _, base_high = wilson(baseline_count, z_lower)
    cand_down = candidate_rate - cand_low
    base_up = base_high - baseline_rate
    lower = np.maximum(
        -1.0,
        estimate
        - np.sqrt(
            np.maximum(0.0, cand_down**2 + base_up**2 - 2 * correlation * cand_down * base_up)
        ),
    )
    z_upper = float(norm.ppf(1 - alpha_upper))
    _, cand_high = wilson(candidate_count, z_upper)
    base_low, _ = wilson(baseline_count, z_upper)
    cand_up = cand_high - candidate_rate
    base_down = baseline_rate - base_low
    upper = np.minimum(
        1.0,
        estimate
        + np.sqrt(
            np.maximum(0.0, cand_up**2 + base_down**2 - 2 * correlation * cand_up * base_down)
        ),
    )
    return estimate, lower, upper


def _reachable_cell_counts(
    candidate: list[bool | None], baseline: list[bool | None]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every (n11, n10, n01, n00) some completion of the unscored outcomes produces.

    Each unscored pair moves between two cells (one arm unknown) or among all four (both
    unknown), so the reachable set is a Minkowski sum of M-convex sets. Such a sum is M-convex,
    hence the integer points of a base polytope whose subset-sum bounds are the sums of the
    parts' bounds. The function enumerates those integer points.
    """

    cells = [(True, True), (True, False), (False, True), (False, False)]
    fixed = [0, 0, 0, 0]
    options: list[list[int]] = []
    for cand, base in zip(candidate, baseline, strict=True):
        allowed = [
            index
            for index, (cell_cand, cell_base) in enumerate(cells)
            if (cand is None or cand == cell_cand) and (base is None or base == cell_base)
        ]
        if len(allowed) == 1:
            fixed[allowed[0]] += 1
        else:
            options.append(allowed)
    subsets = [
        [index for index in range(4) if mask >> index & 1] for mask in range(1, 15)
    ]
    low = []
    high = []
    for subset in subsets:
        inside = sum(fixed[index] for index in subset)
        always = sum(all(index in subset for index in allowed) for allowed in options)
        sometimes = sum(any(index in subset for index in allowed) for allowed in options)
        low.append(inside + always)
        high.append(inside + sometimes)
    total = len(candidate)
    bound = {tuple(subset): (lo, hi) for subset, lo, hi in zip(subsets, low, high, strict=True)}
    ranges = [range(bound[(index,)][0], bound[(index,)][1] + 1) for index in range(3)]
    grid = len(ranges[0]) * len(ranges[1]) * len(ranges[2])
    if grid > _PARTIAL_NEWCOMBE_GRID_LIMIT:
        raise ValueError(
            f"newcombe_paired cannot bound {len(options)} unscored pairs: their completions "
            f"span {grid} cell-count candidates, more than {_PARTIAL_NEWCOMBE_GRID_LIMIT}"
        )
    n11, n10, n01 = np.meshgrid(
        np.asarray(ranges[0]), np.asarray(ranges[1]), np.asarray(ranges[2]), indexing="ij"
    )
    n11, n10, n01 = n11.ravel(), n10.ravel(), n01.ravel()
    n00 = total - n11 - n10 - n01
    counts = (n11, n10, n01, n00)
    keep = np.ones(n11.shape, dtype=bool)
    for subset in subsets:
        lo, hi = bound[tuple(subset)]
        subset_sum = sum(counts[index] for index in subset)
        keep &= (subset_sum >= lo) & (subset_sum <= hi)
    return n11[keep], n10[keep], n01[keep], n00[keep]


def newcombe_paired_interval_partial(
    candidate: list[bool | None],
    baseline: list[bool | None],
    *,
    alpha_lower: float = 0.05,
    alpha_upper: float = 0.05,
) -> PartialPairedDifferenceInterval:
    """Newcombe method 10 bounds that hold for every completion of the unscored outcomes.

    The bound is not monotone in the cell counts, so the extreme completions do not suffice:
    the lower bound is the minimum, and the upper bound the maximum, over every cell-count
    vector a completion can produce.
    """

    _check_partial_pairs(candidate, baseline)
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    n11, n10, n01, n00 = _reachable_cell_counts(candidate, baseline)
    estimate, lower, upper = _newcombe_bounds_from_counts(
        n11, n10, n01, n00, alpha_lower=alpha_lower, alpha_upper=alpha_upper
    )
    return PartialPairedDifferenceInterval(
        estimate_lower=float(np.min(estimate)),
        estimate_upper=float(np.max(estimate)),
        lower=float(np.min(lower)),
        upper=float(np.max(upper)),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        completion_count=len(n11),
    )
