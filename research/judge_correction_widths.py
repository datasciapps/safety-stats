"""Width and coverage of the exact vertex box versus the delta-method reference.

Simulates a single judge with fixed sensitivity and specificity over a range of true
prevalences and calibration sizes, labels a simple random sample within each judge
stratum, and compares the exact Clopper-Pearson vertex box (certifying) with the
asymptotic delta-method interval (diagnostic) on interval width and Monte-Carlo
coverage of the true prevalence.

    python research/judge_correction_widths.py
"""

from __future__ import annotations

import numpy as np

from safety_stats.stats import binomial_bounds
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    correct_judge_risk,
    delta_method_corrected_risk,
    estimate_judge_calibration,
)

SENSITIVITY, SPECIFICITY = 0.90, 0.97
ROLLOUTS, ALPHA, REPLICATIONS = 1000, 0.05, 400


def simulate(
    prevalence: float, labels: int, rng: np.random.Generator
) -> tuple[list[float], list[float], int, int]:
    q = prevalence * SENSITIVITY + (1 - prevalence) * (1 - SPECIFICITY)
    rho_1 = prevalence * SENSITIVITY / q
    rho_0 = prevalence * (1 - SENSITIVITY) / (1 - q)
    exact_widths: list[float] = []
    delta_widths: list[float] = []
    exact_covered = delta_covered = 0
    for _ in range(REPLICATIONS):
        positives = int(rng.binomial(ROLLOUTS, q))
        m1, m0 = min(labels, positives), min(labels, ROLLOUTS - positives)
        if m1 == 0 or m0 == 0:
            continue
        unsafe = JudgeStratumCounts(m1, int(rng.binomial(m1, rho_1)), positives)
        safe = JudgeStratumCounts(m0, int(rng.binomial(m0, rho_0)), ROLLOUTS - positives)
        observed = binomial_bounds(
            positives, ROLLOUTS, alpha_lower=ALPHA / 2, alpha_upper=ALPHA / 2
        )
        q_hat = positives / ROLLOUTS
        try:
            exact = correct_judge_risk(
                observed_lower=observed.lower,
                observed_upper=observed.upper,
                observed_estimate=q_hat,
                calibration=estimate_judge_calibration(
                    scenario_id="v", unsafe_stratum=unsafe, safe_stratum=safe, alpha=ALPHA
                ),
            )
            exact_widths.append(exact.upper - exact.lower)
            exact_covered += exact.lower <= prevalence <= exact.upper
        except ValueError:
            pass  # uninformative calibration: the exact path refuses rather than corrects
        delta = delta_method_corrected_risk(
            observed_estimate=q_hat,
            observed_variance=q_hat * (1 - q_hat) / ROLLOUTS,
            unsafe_stratum=unsafe,
            safe_stratum=safe,
            alpha=ALPHA,
        )
        delta_widths.append(delta.upper - delta.lower)
        delta_covered += delta.lower <= prevalence <= delta.upper
    return exact_widths, delta_widths, exact_covered, delta_covered


def main() -> None:
    rng = np.random.default_rng(2026)
    print(
        f"{'prevalence':>10} {'labels':>7} {'exact width':>12} {'delta width':>12} "
        f"{'exact cov':>10} {'delta cov':>10} {'exact refusals':>15}"
    )
    for prevalence in (0.005, 0.02, 0.05, 0.20):
        for labels in (30, 100, 300):
            exact_w, delta_w, exact_c, delta_c = simulate(prevalence, labels, rng)
            refusals = REPLICATIONS - len(exact_w)
            exact_width = float(np.median(exact_w)) if exact_w else float("nan")
            exact_cov = exact_c / len(exact_w) if exact_w else float("nan")
            print(
                f"{prevalence:>10.3f} {labels:>7d} {exact_width:>12.4f} "
                f"{float(np.median(delta_w)):>12.4f} {exact_cov:>10.3f} "
                f"{delta_c / REPLICATIONS:>10.3f} {refusals:>15d}"
            )


if __name__ == "__main__":
    main()
