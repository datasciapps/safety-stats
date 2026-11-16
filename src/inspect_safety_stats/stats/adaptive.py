"""An anytime-valid lower bound on the success rate of an attacker that adapts across runs.

Experimental (#125). An attacker that carries what it learned on one behaviour into the next
makes its runs dependent, so the binomial bounds of section 4 do not apply. What still has a
guarantee is a one-sided lower bound on the attacker's running average of conditional success
probabilities,

    m̄_t = (1/t) Σ_{i≤t} P(run i succeeds | runs 1, …, i−1).

For a fixed λ > 0, with ψ_λ(m) = log(1 − m + m·e^λ), the process
exp(λ S_t − Σ_{i≤t} ψ_λ(m_i)) is a nonnegative martingale, where S_t counts the successes among
the first t runs and m_i is run i's conditional success probability. ψ_λ is concave, so
Σ_i ψ_λ(m_i) ≤ t ψ_λ(m̄_t), and exp(λ S_t − t ψ_λ(m̄_t)) is dominated by that martingale. A
mixture over λ fixed in advance keeps both properties, and Ville's inequality gives

    P(∃t: L_t > m̄_t) ≤ α,  L_t = inf{m : Σ_j w_j exp(λ_j S_t − t ψ_{λ_j}(m)) < 1/α}.

Each term falls in m, so the set is an interval [L_t, 1]. Derived in
docs/statistical-methods.md, section 11.1.

The bound is one-sided by construction: an upper bound on one adaptive attacker is no evidence
about other attackers, so nothing here gives one.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

METHOD = "adaptive_mixture_lower_bound"

# The mixture over λ, fixed before any data: 20 points spaced geometrically from 0.05 to 5,
# with equal weights. Small λ serve long streams with a low success rate, large λ short streams
# with a high one. Changing the grid changes every bound this module gives, so it is part of
# the method, and research/adaptive_attacker_coverage.py checks its coverage.
MIXTURE_LAMBDAS = np.geomspace(0.05, 5.0, 20)
MIXTURE_LOG_WEIGHTS = np.full(MIXTURE_LAMBDAS.size, -math.log(MIXTURE_LAMBDAS.size))

# The bound is a root found to this absolute tolerance, and then lowered by it, so that
# rounding in the root finder can only make the bound more conservative.
_TOLERANCE = 1e-12


def _validate(successes: int, runs: int, alpha: float) -> None:
    if isinstance(successes, bool) or isinstance(runs, bool):
        raise TypeError("counts must be integers, not booleans")
    if not isinstance(successes, int) or not isinstance(runs, int):
        raise TypeError("counts must be integers")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if successes < 0 or successes > runs:
        raise ValueError("successes must be between zero and runs")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def log_mixture_e_value(successes: int, runs: int, mean: float) -> float:
    """The log of the mixture process at ``mean``: Σ_j w_j exp(λ_j S − t ψ_{λ_j}(mean)).

    The bound excludes ``mean`` when this is at least log(1/α).
    """

    psi = np.log1p(mean * np.expm1(MIXTURE_LAMBDAS))
    return float(logsumexp(MIXTURE_LOG_WEIGHTS + MIXTURE_LAMBDAS * successes - runs * psi))


def adaptive_lower_bound(successes: int, runs: int, *, alpha: float) -> float:
    """One-sided anytime-valid lower bound on an adaptive attacker's mean conditional success.

    ``successes`` counts the successful runs among the first ``runs``, taken in the order the
    behaviours were attacked, which must be fixed in advance or randomised. With probability
    at least 1 − ``alpha`` the bound stays at or below m̄_t at every t at once, whatever the
    attacker carries from one run to the next, so it may be recomputed after every run.
    Returns 0 when the data exclude no mean.
    """

    _validate(successes, runs, alpha)
    threshold = math.log(1.0 / alpha)

    def excess(mean: float) -> float:
        return log_mixture_e_value(successes, runs, mean) - threshold

    if excess(0.0) < 0:
        return 0.0
    # At mean 1 the process is exp(−λ (t − S)) ≤ 1 < 1/α, so the root lies below 1.
    root = brentq(excess, 0.0, 1.0, xtol=_TOLERANCE, rtol=4 * np.finfo(float).eps)
    return max(0.0, float(root) - _TOLERANCE)
