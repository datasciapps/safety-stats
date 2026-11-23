"""A certified generative root covers true risk when some units have no judge verdict.

Units with no judge verdict keep their worst case through the judge correction, and how much
weight they carry in the design is a parameter that the run's own share only estimates. This
simulates such a design through the real ``correct`` and checks coverage at the alpha the
correction reports. It imports only what existed before the correction bounded the share, so
that it fails on its coverage assertion, not on an import, against code that used the run's
share.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safety_stats.generative.judge import correct
from safety_stats.stats.binomial import binomial_upper_bound
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_judge_calibration,
)
from tests.generative_fixtures import root_estimate

SEED = 20261119


def test_a_certified_generative_root_covers_true_risk_with_unclassified_units() -> None:
    """Seeded Monte Carlo through the real ``correct`` for an equal-weight direct root.

    Per draw: 30% never classified, 5% judge-unsafe of which 10% truly unsafe, the rest
    judge-safe and safe; every classified unit is labelled. In the worst case every
    unclassified unit is truly unsafe, so true risk is 0.05 * 0.10 + 0.30 = 0.305. With the
    run's own share in place of a bound, the corrected upper bound missed it in about 23% of
    runs against a reported alpha of 0.10; with no share at all, in every run.
    """

    rng = np.random.default_rng(SEED)
    draws, unclassified_rate, judge_unsafe_rate, rho_1 = 5000, 0.30, 0.05, 0.10
    truth = judge_unsafe_rate * rho_1 + unclassified_rate
    interval_alpha = calibration_alpha = 0.05
    replications = 2000
    reported = interval_alpha + calibration_alpha + calibration_alpha / 4
    misses = 0
    corrected_runs = []
    for _ in range(replications):
        unclassified, judge_unsafe, judge_safe = (
            int(value)
            for value in rng.multinomial(
                draws,
                [unclassified_rate, judge_unsafe_rate, 1 - unclassified_rate - judge_unsafe_rate],
            )
        )
        truly_unsafe = int(rng.binomial(judge_unsafe, rho_1))
        calibration = estimate_judge_calibration(
            scenario_id="labels",
            unsafe_stratum=JudgeStratumCounts(judge_unsafe, truly_unsafe, judge_unsafe),
            safe_stratum=JudgeStratumCounts(judge_safe, 0, judge_safe),
            alpha=calibration_alpha,
        )
        root = root_estimate(
            draws=draws,
            unclassified=unclassified,
            failures=judge_unsafe,
            upper=binomial_upper_bound(
                judge_unsafe + unclassified, draws, alpha=interval_alpha
            ),
        )
        (corrected,) = correct([root], calibration, interval_alpha=interval_alpha, root_id="all")
        misses += corrected.upper < truth
        corrected_runs.append((corrected, unclassified / draws))
    standard_error = math.sqrt(reported * (1 - reported) / replications)
    assert misses / replications <= reported + 3 * standard_error
    for corrected, share in corrected_runs:
        assert corrected.eligibility == "certifying"
        # The share bound spends a quarter of the calibration's budget, on top of it, and
        # that is the alpha the coverage above is checked against.
        assert corrected.alpha == pytest.approx(reported, rel=1e-12, abs=0)
        assert corrected.unclassified_alpha == pytest.approx(calibration_alpha / 4, abs=0)
        assert corrected.unclassified_upper >= share
