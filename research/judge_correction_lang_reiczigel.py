"""Coverage and width of the Lang-Reiczigel limits against the exact vertex box.

Simulates a binary judge with fixed sensitivity and specificity over a cohort of rollouts,
across true prevalences from 0.5% to 20% and label budgets from 30 upward, and compares
three intervals for the true prevalence at the same nominal level (95%):

- ``box``: the exact Clopper-Pearson vertex box (certifying), with the label budget split
  evenly between the two judge strata, as ``sampling: simple_random_within_judge_stratum``
  draws it. Half of alpha goes to the judge-reported interval and half to the calibration.
- ``box_srs``: the same box computed from a simple random sample of the rollouts, which is
  also a simple random sample within each judge stratum.
- ``lr``: the Lang-Reiczigel limits from that same simple random sample, which give the
  sensitivity and specificity counts the method needs (``sampling: simple_random``).

For each it reports how often the method refuses (an empty or uninformative stratum), the
coverage among the intervals it produced, the rate at which the true prevalence lies above
the upper limit (the miss that lets an unsafe model pass a threshold claim; 2.5% nominal for
a symmetric interval), and the median width.

    python research/judge_correction_lang_reiczigel.py [--replications N]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from safety_stats.stats import binomial_bounds
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    TruthStratumCounts,
    correct_judge_risk,
    estimate_judge_calibration,
    lang_reiczigel_corrected_risk,
)

JUDGES = ((0.90, 0.97), (0.80, 0.90))
PREVALENCES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
LABEL_BUDGETS = (30, 100, 300, 1000)
ROLLOUTS, ALPHA = 2000, 0.05


@dataclass
class Tally:
    produced: int = 0
    covered: int = 0
    above_upper: int = 0
    widths: list[float] = field(default_factory=list)

    def add(self, lower: float, upper: float, truth: float) -> None:
        self.produced += 1
        self.covered += lower <= truth <= upper
        self.above_upper += truth > upper
        self.widths.append(upper - lower)

    def row(self, replications: int) -> str:
        if not self.produced:
            return f"{replications:>6d} {'-':>6} {'-':>6} {'-':>7}"
        return (
            f"{replications - self.produced:>6d} {self.covered / self.produced:>6.3f} "
            f"{self.above_upper / self.produced:>6.3f} {float(np.median(self.widths)):>7.4f}"
        )


def box(
    positives: int, unsafe: JudgeStratumCounts, safe: JudgeStratumCounts
) -> tuple[float, float] | None:
    if unsafe.labelled == 0 or safe.labelled == 0:
        return None
    half = ALPHA / 2
    observed = binomial_bounds(positives, ROLLOUTS, alpha_lower=half / 2, alpha_upper=half / 2)
    try:
        calibration = estimate_judge_calibration(
            scenario_id="v", unsafe_stratum=unsafe, safe_stratum=safe, alpha=half
        )
    except ValueError:
        return None  # uninformative calibration: the exact path refuses
    corrected = correct_judge_risk(
        observed_lower=observed.lower,
        observed_upper=observed.upper,
        observed_estimate=None,
        calibration=calibration,
    )
    return corrected.lower, corrected.upper


def simulate(
    sensitivity: float,
    specificity: float,
    prevalence: float,
    budget: int,
    replications: int,
    rng: np.random.Generator,
) -> dict[str, Tally]:
    tallies = {"box": Tally(), "box_srs": Tally(), "lr": Tally()}
    for _ in range(replications):
        truth = rng.random(ROLLOUTS) < prevalence
        verdict = np.where(
            truth, rng.random(ROLLOUTS) < sensitivity, rng.random(ROLLOUTS) >= specificity
        )
        positives = int(verdict.sum())

        # Labels drawn within judge strata, half the budget in each.
        strata = []
        for flag in (True, False):
            members = np.flatnonzero(verdict == flag)
            drawn = rng.choice(members, size=min(budget // 2, members.size), replace=False)
            strata.append(
                JudgeStratumCounts(int(drawn.size), int(truth[drawn].sum()), int(members.size))
            )
        interval = box(positives, strata[0], strata[1])
        if interval is not None:
            tallies["box"].add(*interval, prevalence)

        # A simple random sample of the rollouts with the whole budget.
        drawn = rng.choice(ROLLOUTS, size=budget, replace=False)
        sample_truth, sample_verdict = truth[drawn], verdict[drawn]
        srs_strata = [
            JudgeStratumCounts(
                int((sample_verdict == flag).sum()),
                int((sample_truth & (sample_verdict == flag)).sum()),
                int((verdict == flag).sum()),
            )
            for flag in (True, False)
        ]
        interval = box(positives, srs_strata[0], srs_strata[1])
        if interval is not None:
            tallies["box_srs"].add(*interval, prevalence)
        try:
            lr = lang_reiczigel_corrected_risk(
                apparent_rate=positives / ROLLOUTS,
                apparent_size=ROLLOUTS,
                truly_unsafe=TruthStratumCounts(
                    int(sample_truth.sum()), int((sample_truth & sample_verdict).sum())
                ),
                truly_safe=TruthStratumCounts(
                    int((~sample_truth).sum()), int((~sample_truth & sample_verdict).sum())
                ),
                alpha=ALPHA,
            )
        except ValueError:
            continue  # no truly unsafe label, or an uninformative judge
        tallies["lr"].add(lr.lower, lr.upper, prevalence)
    return tallies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replications", type=int, default=1000)
    replications = parser.parse_args().replications
    rng = np.random.default_rng(2026)
    columns = f"{'refuse':>6} {'cover':>6} {'above':>6} {'width':>7}"
    print(f"{ROLLOUTS} rollouts, {replications} replications per cell, nominal 95%")
    print(f"{'Se':>4} {'Sp':>4} {'prev':>6} {'labels':>6} | box: {columns} | "
          f"box_srs: {columns} | lr: {columns}")
    for sensitivity, specificity in JUDGES:
        for prevalence in PREVALENCES:
            for budget in LABEL_BUDGETS:
                tallies = simulate(
                    sensitivity, specificity, prevalence, budget, replications, rng
                )
                print(
                    f"{sensitivity:>4.2f} {specificity:>4.2f} {prevalence:>6.3f} {budget:>6d} | "
                    f"{tallies['box'].row(replications)} | "
                    f"{tallies['box_srs'].row(replications)} | "
                    f"{tallies['lr'].row(replications)}"
                )


if __name__ == "__main__":
    main()
