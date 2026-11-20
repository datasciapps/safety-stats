from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from scipy.special import betaincinv

# The endpoints are inverse regularized incomplete beta functions. scipy.stats.beta.ppf
# evaluates this same inverse for 0 < q < 1 (and adds nothing for unit scale and zero
# location), but its per-call argument handling costs about twenty times the inverse
# itself. tests/unit/test_binomial.py pins the two to be bit-identical.


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float
    alpha_lower: float
    alpha_upper: float
    method: str = "clopper_pearson_one_sided"


def _validate(failures: int, trials: int, alpha: float) -> None:
    if isinstance(failures, bool) or isinstance(trials, bool):
        raise TypeError("counts must be integers, not booleans")
    # Integral, not int: NumPy's integers, such as a DataFrame column's sum, are counts too.
    if not isinstance(failures, Integral) or not isinstance(trials, Integral):
        raise TypeError("counts must be integers")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if failures < 0 or failures > trials:
        raise ValueError("failures must be between zero and trials")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def binomial_upper_bound(failures: int, trials: int, *, alpha: float) -> float:
    """One-sided exact Clopper-Pearson upper bound on a binomial failure rate.

    ``failures`` and ``trials`` are integers with ``0 <= failures <= trials`` and
    ``trials > 0``. ``alpha`` in ``(0, 1)`` is the one-sided error level: the bound lies at or
    above the true rate with probability at least ``1 - alpha``. Returns 1 when every trial
    failed. Assumes independent trials with one common failure probability. Derived in
    docs/statistical-methods.md, section 4.2.
    """

    _validate(failures, trials, alpha)
    if failures == trials:
        return 1.0
    return float(betaincinv(failures + 1, trials - failures, 1.0 - alpha))


def binomial_lower_bound(failures: int, trials: int, *, alpha: float) -> float:
    """One-sided exact Clopper-Pearson lower bound on a binomial failure rate.

    Takes the same arguments as ``binomial_upper_bound``, with ``alpha`` the one-sided error
    level of this bound alone. Returns 0 when no trial failed. Assumes independent trials with
    one common failure probability. Derived in docs/statistical-methods.md, section 4.2.
    """

    _validate(failures, trials, alpha)
    if failures == 0:
        return 0.0
    return float(betaincinv(failures, trials - failures + 1, alpha))


def binomial_bounds(
    failures: int,
    trials: int,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> Bounds:
    """Both one-sided Clopper-Pearson bounds, each at its own error level.

    ``alpha_lower`` and ``alpha_upper`` are the one-sided error levels of the lower and upper
    bound. Together they form an interval with coverage at least
    ``1 - alpha_lower - alpha_upper``. Counts are as in ``binomial_upper_bound``. Derived in
    docs/statistical-methods.md, section 4.2.
    """

    return Bounds(
        lower=binomial_lower_bound(failures, trials, alpha=alpha_lower),
        upper=binomial_upper_bound(failures, trials, alpha=alpha_upper),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
    )


def missingness_robust_binomial_bounds(
    failures: int,
    classified: int,
    total: int,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> Bounds:
    """Exact bounds robust to arbitrary labels on attempted, unclassified trials.

    The lower endpoint completes every unclassified trial as safe, while the
    upper endpoint completes every unclassified trial as a failure. Clopper-
    Pearson endpoints are monotone in the completed failure count, so these
    endpoints cover every possible completion of the missing binary labels.
    """

    if isinstance(classified, bool) or isinstance(total, bool):
        raise TypeError("counts must be integers, not booleans")
    if not isinstance(classified, Integral) or not isinstance(total, Integral):
        raise TypeError("counts must be integers")
    if total <= 0:
        raise ValueError("total must be positive")
    if classified < 0 or classified > total:
        raise ValueError("classified must be between zero and total")
    if failures < 0 or failures > classified:
        raise ValueError("failures must be between zero and classified")
    unclassified = total - classified
    return Bounds(
        lower=binomial_lower_bound(failures, total, alpha=alpha_lower),
        upper=binomial_upper_bound(
            failures + unclassified,
            total,
            alpha=alpha_upper,
        ),
        alpha_lower=alpha_lower,
        alpha_upper=alpha_upper,
        method="clopper_pearson_one_sided_missingness_robust",
    )
