from __future__ import annotations

from collections.abc import Iterator
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
    # The product of the four margins passes 2 ** 64 from 131,072 pairs, and NumPy cannot take
    # the square root of a larger Python integer. Each product of two margins is below 2 ** 53
    # under about 1.9e8 pairs, so it is exact as a float, and the float product of the two is
    # the exact product rounded once, as the integer's conversion rounded it before.
    candidate_spread = float((both_failure + candidate_only) * (baseline_only + both_safe))
    baseline_spread = float((both_failure + baseline_only) * (candidate_only + both_safe))
    correlation_denominator = np.sqrt(candidate_spread * baseline_spread)
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


# Partial pairing: pairs in which one or both arms are unscored (outcome None).
# The unknown outcomes are bounded by their completions, never imputed.

# newcombe_paired_interval_partial evaluates its bound on every distinct table the
# completions produce, a block at a time, so its memory is that of a block and its time grows
# with the number of tables. It refuses more tables than the limit. k pairs unscored in both
# arms produce C(k + 3, 3) tables, past the limit from k = 668. Pairs unscored in one arm, a
# of one kind and b of another, a kind being the unscored arm and the other arm's outcome,
# produce (a + 1)(b + 1), past the limit from a = b = 7071.
_PARTIAL_NEWCOMBE_BLOCK = 1 << 18
_PARTIAL_NEWCOMBE_COMPLETION_LIMIT = 50_000_000


class CompletionLimitError(ValueError):
    """The completions of the unscored outcomes produce more tables than a method evaluates."""


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

    # Float counts: products of counts overflow an integer array, int64 from about 110,000
    # pairs and a narrower integer much sooner. Every count is exact as a float. A product of
    # two is exact below 2 ** 53 (two margins under about 1.9e8 pairs, trials ** 2 under about
    # 9.5e7) and otherwise rounded once, as converting the integer product rounded it, and
    # multiplying by 4 is exact, so each result below is rounded as in the scalar interval.
    both_failure, candidate_only, baseline_only, both_safe = (
        np.asarray(counts, dtype=float)
        for counts in (both_failure, candidate_only, baseline_only, both_safe)
    )
    trials = both_failure + candidate_only + baseline_only + both_safe
    candidate_count = both_failure + candidate_only
    baseline_count = both_failure + baseline_only
    candidate_rate = candidate_count / trials
    baseline_rate = baseline_count / trials
    estimate = candidate_rate - baseline_rate
    numerator = both_failure * both_safe - candidate_only * baseline_only
    numerator = np.where(numerator > 0, np.maximum(numerator - trials / 2, 0.0), numerator)
    # The two exact products of two margins are multiplied last, so the four-margin product
    # is rounded once.
    denominator = np.sqrt(
        ((both_failure + candidate_only) * (baseline_only + both_safe))
        * ((both_failure + baseline_only) * (candidate_only + both_safe))
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


# The cells of a 2x2 table of paired outcomes, (candidate, baseline) with True for failure, in
# the order (n11, n10, n01, n00), and every nonempty proper subset of them.
_CELLS = ((True, True), (True, False), (False, True), (False, False))
_CELL_SUBSETS = tuple(
    tuple(index for index in range(4) if mask >> index & 1) for mask in range(1, 15)
)


@dataclass(frozen=True)
class _CompletionTables:
    """The tables (n11, n10, n01, n00) that the completions of the unscored outcomes produce.

    Each unscored pair moves between two cells (one arm unknown) or among all four (both
    unknown), so the reachable set is a Minkowski sum of M-convex sets. Such a sum is M-convex,
    hence the integer points of a base polytope whose subset-sum bounds are the sums of the
    parts' bounds. ``bounds`` maps each subset of ``_CELL_SUBSETS`` to them; every table has
    ``pairs`` pairs, ``unscored`` of which have an unscored arm.

    Given n10 and n01, n00 is ``pairs - n11 - n10 - n01``, so a bound on a subset that holds
    one of n11 and n00 but not the other bounds n11, and every other bound holds or fails for
    the row as a whole. The tables are therefore rows of consecutive n11 values, one row for
    each (n10, n01) that some completion reaches.
    """

    pairs: int
    unscored: int
    bounds: dict[tuple[int, ...], tuple[int, int]]


def _completion_tables(
    candidate: list[bool | None], baseline: list[bool | None]
) -> _CompletionTables:
    fixed = [0, 0, 0, 0]
    # Unscored pairs by the cells they can fall in: there are five such sets at most.
    options: dict[tuple[int, ...], int] = {}
    for cand, base in zip(candidate, baseline, strict=True):
        allowed = tuple(
            index
            for index, (cell_cand, cell_base) in enumerate(_CELLS)
            if (cand is None or cand == cell_cand) and (base is None or base == cell_base)
        )
        if len(allowed) == 1:
            fixed[allowed[0]] += 1
        else:
            options[allowed] = options.get(allowed, 0) + 1
    bounds = {}
    for subset in _CELL_SUBSETS:
        inside = sum(fixed[index] for index in subset)
        always = sum(
            count
            for allowed, count in options.items()
            if all(index in subset for index in allowed)
        )
        sometimes = sum(
            count
            for allowed, count in options.items()
            if any(index in subset for index in allowed)
        )
        bounds[subset] = (inside + always, inside + sometimes)
    return _CompletionTables(
        pairs=len(candidate), unscored=sum(options.values()), bounds=bounds
    )


def _table_rows(
    tables: _CompletionTables, n10_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """n10, n01 and the lowest and highest n11 of each row whose n10 is in ``n10_values``."""

    low01, high01 = tables.bounds[(2,)]
    n10, n01 = (
        values.ravel()
        for values in np.meshgrid(
            n10_values, np.arange(low01, high01 + 1, dtype=np.int64), indexing="ij"
        )
    )
    low11, high11 = tables.bounds[(0,)]
    low = np.full(n10.shape, low11, dtype=np.int64)
    high = np.full(n10.shape, high11, dtype=np.int64)
    keep = np.ones(n10.shape, dtype=bool)
    n11_and_n00 = tables.pairs - n10 - n01
    for subset, (lo, hi) in tables.bounds.items():
        rest = (n10 if 1 in subset else 0) + (n01 if 2 in subset else 0)
        if 0 in subset and 3 not in subset:
            # The subset sum is n11 + rest.
            low = np.maximum(low, lo - rest)
            high = np.minimum(high, hi - rest)
        elif 3 in subset and 0 not in subset:
            # The subset sum is n00 + rest, that is n11_and_n00 - n11 + rest.
            low = np.maximum(low, n11_and_n00 + rest - hi)
            high = np.minimum(high, n11_and_n00 + rest - lo)
        else:
            value = rest + (n11_and_n00 if 0 in subset else 0)
            keep &= (value >= lo) & (value <= hi)
    keep &= low <= high
    return n10[keep], n01[keep], low[keep], high[keep]


def _table_row_blocks(
    tables: _CompletionTables,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """The rows of ``tables`` as ``_table_rows`` gives them, a few n10 values at a time."""

    low10, high10 = tables.bounds[(1,)]
    low01, high01 = tables.bounds[(2,)]
    step = max(1, _PARTIAL_NEWCOMBE_BLOCK // (high01 - low01 + 1))
    for start in range(low10, high10 + 1, step):
        yield _table_rows(
            tables, np.arange(start, min(start + step, high10 + 1), dtype=np.int64)
        )


def _table_blocks(
    tables: _CompletionTables,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Every table (n11, n10, n01, n00) once, about ``_PARTIAL_NEWCOMBE_BLOCK`` at a time.

    A block holds whole rows, so a row longer than the block size makes a block of its own.
    """

    for n10, n01, low, high in _table_row_blocks(tables):
        sizes = high - low + 1
        ends = np.cumsum(sizes)
        first = 0
        while first < len(sizes):
            done = int(ends[first - 1]) if first else 0
            last = max(
                first + 1,
                int(np.searchsorted(ends, done + _PARTIAL_NEWCOMBE_BLOCK, side="right")),
            )
            size = sizes[first:last]
            starts = np.cumsum(size) - size
            n11 = np.repeat(low[first:last] - starts, size) + np.arange(
                int(size.sum()), dtype=np.int64
            )
            block10 = np.repeat(n10[first:last], size)
            block01 = np.repeat(n01[first:last], size)
            yield n11, block10, block01, tables.pairs - n11 - block10 - block01
            first = last


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
    vector a completion can produce. They are evaluated a block at a time, so memory does not
    grow with their number, and when there are more than 50,000,000 of them, as from 668
    pairs unscored in both arms, the function raises ``CompletionLimitError``, a
    ``ValueError``, before it evaluates any.
    """

    _check_partial_pairs(candidate, baseline)
    if not 0 < alpha_lower < 1 or not 0 < alpha_upper < 1:
        raise ValueError("alpha levels must be in (0, 1)")
    tables = _completion_tables(candidate, baseline)
    count = 0
    for _, _, low, high in _table_row_blocks(tables):
        count += int(np.sum(high - low + 1))
        if count > _PARTIAL_NEWCOMBE_COMPLETION_LIMIT:
            raise CompletionLimitError(
                f"newcombe_paired cannot bound the {tables.unscored} pairs with an unscored "
                "arm: the completions of their outcomes produce more than "
                f"{_PARTIAL_NEWCOMBE_COMPLETION_LIMIT} distinct 2x2 tables"
            )
    # np.minimum and np.maximum carry a NaN through, as np.min and np.max over all tables do.
    estimate_lower = lower = np.float64(np.inf)
    estimate_upper = upper = np.float64(-np.inf)
    for n11, n10, n01, n00 in _table_blocks(tables):
        estimate, block_lower, block_upper = _newcombe_bounds_from_counts(
            n11, n10, n01, n00, alpha_lower=alpha_lower, alpha_upper=alpha_upper
        )
        estimate_lower = np.minimum(estimate_lower, np.min(estimate))
        estimate_upper = np.maximum(estimate_upper, np.max(estimate))
        lower = np.minimum(lower, np.min(block_lower))
        upper = np.maximum(upper, np.max(block_upper))
    return PartialPairedDifferenceInterval(
        estimate_lower=float(estimate_lower),
        estimate_upper=float(estimate_upper),
        lower=float(lower),
        upper=float(upper),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        completion_count=count,
    )
