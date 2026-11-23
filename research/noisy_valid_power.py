"""Power and Type-I error of the noisy-but-valid test versus the corrected-interval decision.

Both paths use the same simple random sample of human labels: the noisy-but-valid test
estimates sensitivity and false-positive rate by true class, the corrected-interval path
estimates the posterior accuracies within judge strata. Each row reports the PASS rate for
the claim "true risk <= acceptable" at a given true prevalence; the boundary row
(prevalence equal to the threshold) is the Type-I error.

    python research/noisy_valid_power.py
"""

from __future__ import annotations

import numpy as np

from safety_stats.stats import binomial_bounds
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    TruthStratumCounts,
    correct_judge_risk,
    estimate_judge_calibration,
    noisy_valid_certification,
)

SENSITIVITY, SPECIFICITY = 0.90, 0.97
ROLLOUTS, ACCEPTABLE, UNACCEPTABLE = 2000, 0.10, 0.30
JUDGE_ALPHA, CALIBRATION_ALPHA, REPLICATIONS = 0.025, 0.05, 400


def pass_rates(prevalence: float, labels: int, rng: np.random.Generator) -> tuple[float, float]:
    noisy_passes = interval_passes = 0
    for _ in range(REPLICATIONS):
        unsafe = rng.random(ROLLOUTS) < prevalence
        flagged = np.where(
            unsafe, rng.random(ROLLOUTS) < SENSITIVITY, rng.random(ROLLOUTS) >= SPECIFICITY
        )
        observed = binomial_bounds(
            int(flagged.sum()), ROLLOUTS, alpha_lower=JUDGE_ALPHA, alpha_upper=JUDGE_ALPHA
        )
        sample = rng.choice(ROLLOUTS, size=labels, replace=False)
        s_unsafe, s_flag = unsafe[sample], flagged[sample]
        if s_unsafe.sum() == 0 or (~s_unsafe).sum() == 0 or s_flag.sum() == 0:
            continue
        noisy = noisy_valid_certification(
            judge_reported_lower=observed.lower,
            judge_reported_upper=observed.upper,
            judge_reported_alpha=JUDGE_ALPHA,
            truly_unsafe=TruthStratumCounts(int(s_unsafe.sum()), int((s_unsafe & s_flag).sum())),
            truly_safe=TruthStratumCounts(int((~s_unsafe).sum()), int((~s_unsafe & s_flag).sum())),
            calibration_alpha=CALIBRATION_ALPHA,
            acceptable_upper=ACCEPTABLE,
            unacceptable_lower=UNACCEPTABLE,
        )
        noisy_passes += noisy.decision == "PASS"
        if (~s_flag).sum() == 0:
            continue
        try:
            calibration = estimate_judge_calibration(
                scenario_id="v",
                unsafe_stratum=JudgeStratumCounts(
                    int(s_flag.sum()), int((s_flag & s_unsafe).sum()), int(flagged.sum())
                ),
                safe_stratum=JudgeStratumCounts(
                    int((~s_flag).sum()), int((~s_flag & s_unsafe).sum()), int((~flagged).sum())
                ),
                alpha=CALIBRATION_ALPHA,
            )
        except ValueError:
            continue
        corrected = correct_judge_risk(
            observed_lower=observed.lower,
            observed_upper=observed.upper,
            observed_estimate=float(flagged.mean()),
            calibration=calibration,
        )
        interval_passes += corrected.upper <= ACCEPTABLE
    return noisy_passes / REPLICATIONS, interval_passes / REPLICATIONS


def main() -> None:
    rng = np.random.default_rng(2026)
    print(f"{'prevalence':>10} {'labels':>7} {'noisy-valid PASS':>17} {'interval PASS':>14}")
    for prevalence in (0.02, 0.05, 0.08, ACCEPTABLE):
        for labels in (100, 300):
            noisy, interval = pass_rates(prevalence, labels, rng)
            tag = "  <- boundary (Type-I)" if prevalence == ACCEPTABLE else ""
            print(f"{prevalence:>10.2f} {labels:>7d} {noisy:>17.3f} {interval:>14.3f}{tag}")


if __name__ == "__main__":
    main()
