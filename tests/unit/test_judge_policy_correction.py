"""Judge corrections of a row holding rollouts a plan policy classified: seeded Monte Carlo.

The rho corrections come first, then the panel-derived correction.

Each prompt's rollouts fall into four kinds: a plan policy counts some as failures and some
safe, which is their true outcome; some go unclassified, with a true outcome of ``unknown``;
and the judge classifies the rest, flagging each with probability ``flagged``. A judge-unsafe
rollout is truly unsafe with probability RHO_1 and a judge-safe one with probability RHO_0.
Each prompt gets the pipeline's missingness-robust Clopper-Pearson bounds on its judge-system
rate, which counts the policy's failures beside the judge's; the row's bounds are the measure
of the prompt bounds. With ALPHA_SIDE per prompt and side, a side of the row misses with
probability at most PROMPTS * ALPHA_SIDE, whichever prompt the row rests on. The correction is
taken over a point box at the judge's true accuracies, the sharpest box there is.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safety_stats.analysis.judge_calibration import _policy_free_box
from safety_stats.analysis.judge_panel import _policy_free_accuracy_box
from safety_stats.stats.binomial import missingness_robust_binomial_bounds
from safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set
from safety_stats.stats.misclassification import JudgeCalibration, correct_judge_risk
from safety_stats.stats.risk import maximum_risk, top_k_mean, weighted_cvar, weighted_mean

_PROMPTS = 3
_ALPHA_SIDE = 0.10 / (2 * _PROMPTS)
_NOMINAL = _PROMPTS * _ALPHA_SIDE
_REPLICATIONS = 2000
_ROLLOUTS = 300

_MEASURES = {
    "weighted_mean": lambda values: weighted_mean(values, [1 / _PROMPTS] * _PROMPTS),
    "maximum": maximum_risk,
    "cvar": lambda values: weighted_cvar(values, [1 / _PROMPTS] * _PROMPTS, beta=0.5),
    "top_k_mean": lambda values: top_k_mean(values, k=2),
}

# Per prompt: (flagged, policy failures, policy safe, unclassified, unknown), the last the
# probability that an unclassified rollout is truly unsafe. The upper side has a judge that
# over-flags (RHO_1 0.5) and policy failures on the riskiest prompt: read as judge verdicts they
# are mapped to half a failure each. The lower side has a judge that misses failures (RHO_0 0.3)
# and policy-safe rollouts: read as judge verdicts each is mapped to 0.3 of a failure.
_UPPER_SIDE = ((0.2, 0.4, 0.0, 0.0, 0.0), (0.05, 0.0, 0.0, 0.1, 1.0), (0.1, 0.1, 0.0, 0.0, 0.0))
_LOWER_SIDE = ((0.3, 0.0, 0.4, 0.0, 0.0), (0.5, 0.0, 0.4, 0.0, 0.0), (0.2, 0.0, 0.2, 0.2, 0.0))
_ACCURACY = {"upper": (0.5, 0.02), "lower": (0.9, 0.3)}

Prompt = tuple[float, float, float, float, float]


def _true_risk(prompt: Prompt, rho_1: float, rho_0: float) -> float:
    flagged, failures, safe, unclassified, unknown = prompt
    judged = 1 - failures - safe - unclassified
    return judged * (flagged * rho_1 + (1 - flagged) * rho_0) + failures + unclassified * unknown


def _misses(
    measure: str, prompts: tuple[Prompt, ...], side: str, seed: int
) -> tuple[float, float]:
    """The rate at which the correction's bound on ``side`` misses true risk, and the rate at
    which the correction over the box itself, reading the policy's rollouts as judge verdicts,
    does."""

    rng = np.random.default_rng(seed)
    aggregate = _MEASURES[measure]
    rho_1, rho_0 = _ACCURACY[side]
    truth = aggregate([_true_risk(prompt, rho_1, rho_0) for prompt in prompts])
    box = JudgeCalibration("point", rho_1, rho_1, rho_0, rho_0)
    widened = _policy_free_box(box)
    misses = naive_misses = 0
    for _ in range(_REPLICATIONS):
        lowers, uppers, shares = [], [], []
        for flagged, failures, safe, unclassified, _ in prompts:
            judged = 1 - failures - safe - unclassified
            counts = rng.multinomial(_ROLLOUTS, [judged, failures, safe, unclassified])
            judge_unsafe = int(rng.binomial(int(counts[0]), flagged))
            classified = int(counts[0] + counts[1] + counts[2])
            bounds = missingness_robust_binomial_bounds(
                judge_unsafe + int(counts[1]),
                classified,
                _ROLLOUTS,
                alpha_lower=_ALPHA_SIDE,
                alpha_upper=_ALPHA_SIDE,
            )
            lowers.append(bounds.lower)
            uppers.append(bounds.upper)
            shares.append((_ROLLOUTS - classified) / _ROLLOUTS)
        low, high = aggregate(lowers), aggregate(uppers)
        corrected = correct_judge_risk(
            observed_lower=low, observed_upper=high, observed_estimate=None, calibration=widened
        )
        # What the correction did before: the box itself, with the unclassified share taken
        # out as the pipeline takes it, the mean share or the largest.
        share = sum(shares) / _PROMPTS if measure == "weighted_mean" else max(shares)
        naive = correct_judge_risk(
            observed_lower=low,
            observed_upper=high,
            observed_estimate=None,
            calibration=box,
            unclassified_share=share,
        )
        if side == "upper":
            misses += corrected.upper < truth
            naive_misses += naive.upper < truth
        else:
            misses += corrected.lower > truth
            naive_misses += naive.lower > truth
    return misses / _REPLICATIONS, naive_misses / _REPLICATIONS


_SCENARIOS = [
    (measure, side, prompts)
    for side, prompts in (("upper", _UPPER_SIDE), ("lower", _LOWER_SIDE))
    for measure in sorted(_MEASURES)
]


@pytest.mark.parametrize(("measure", "side", "prompts"), _SCENARIOS)
def test_the_correction_of_a_row_a_policy_classified_covers_at_its_nominal_rate(
    measure: str, side: str, prompts: tuple[Prompt, ...]
) -> None:
    seed = 300 + _SCENARIOS.index((measure, side, prompts))
    misses, naive = _misses(measure, prompts, side, seed)
    # Three Monte Carlo standard errors above the nominal one-sided rate.
    allowance = 3 * math.sqrt(_NOMINAL * (1 - _NOMINAL) / _REPLICATIONS)
    assert misses <= _NOMINAL + allowance, (measure, side, misses)
    # The scenario is sharp: corrected over the box itself, the bound misses far more often
    # than nominal, so a correction that kept the box would fail above.
    assert naive >= 0.5, (measure, side, naive)


def test_the_widened_box_frees_only_the_ends_that_credit_the_judge_with_an_error() -> None:
    box = JudgeCalibration("declared", 0.8, 0.95, 0.01, 0.05, source="validated", alpha=0.05)
    widened = _policy_free_box(box)
    assert (widened.rho_1_lower, widened.rho_1_upper) == (0.8, 1.0)
    assert (widened.rho_0_lower, widened.rho_0_upper) == (0.0, 0.05)
    assert (widened.scenario_id, widened.source, widened.alpha) == ("declared", "validated", 0.05)
    corrected = correct_judge_risk(
        observed_lower=0.2, observed_upper=0.4, observed_estimate=0.3, calibration=widened
    )
    assert corrected.lower == pytest.approx(0.2 * 0.8, abs=1e-15)
    assert corrected.upper == pytest.approx(0.4 + 0.6 * 0.05, abs=1e-15)


# --- The panel-derived correction ----------------------------------------------------------
#
# The panel classifies a rollout as unsafe with probability SE when it is truly unsafe and
# 1 - SP when it is truly safe; ``prevalence`` is the share of truly unsafe rollouts among those
# it classifies. The row is inverted over a point box at the panel's true accuracies. A policy-
# safe rollout read as a panel verdict has a false alarm taken off it, which lowers the upper
# bound: the upper side has a panel with many false alarms and many policy-safe rollouts. A
# policy failure read as a verdict is scaled up for missed failures, which raises the lower
# bound: the lower side has a panel that misses failures and many policy failures.
_PANEL_UPPER = ((0.2, 0.0, 0.5, 0.0, 0.0), (0.3, 0.0, 0.5, 0.05, 1.0), (0.1, 0.0, 0.6, 0.0, 0.0))
_PANEL_LOWER = ((0.1, 0.3, 0.0, 0.0, 0.0), (0.2, 0.4, 0.0, 0.0, 0.0), (0.05, 0.3, 0.1, 0.1, 0.0))
_PANEL_ACCURACY = {"upper": (0.9, 0.8), "lower": (0.6, 0.98)}


def _panel_true_risk(prompt: Prompt) -> float:
    prevalence, failures, safe, unclassified, unknown = prompt
    return (1 - failures - safe - unclassified) * prevalence + failures + unclassified * unknown


def _panel_misses(
    measure: str, prompts: tuple[Prompt, ...], side: str, seed: int
) -> tuple[float, float]:
    """As :func:`_misses`, for the Rogan-Gladen inversion of the panel-derived correction."""

    rng = np.random.default_rng(seed)
    aggregate = _MEASURES[measure]
    sensitivity, specificity = _PANEL_ACCURACY[side]
    truth = aggregate([_panel_true_risk(prompt) for prompt in prompts])
    box = JudgeAccuracyBox(sensitivity, sensitivity, specificity, specificity)
    widened = _policy_free_accuracy_box(box)
    misses = naive_misses = 0
    for _ in range(_REPLICATIONS):
        lowers, uppers = [], []
        for prevalence, failures, safe, unclassified, _ in prompts:
            judged = 1 - failures - safe - unclassified
            counts = rng.multinomial(_ROLLOUTS, [judged, failures, safe, unclassified])
            flag = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
            flagged = int(rng.binomial(int(counts[0]), flag))
            bounds = missingness_robust_binomial_bounds(
                flagged + int(counts[1]),
                int(counts[0] + counts[1] + counts[2]),
                _ROLLOUTS,
                alpha_lower=_ALPHA_SIDE,
                alpha_upper=_ALPHA_SIDE,
            )
            lowers.append(bounds.lower)
            uppers.append(bounds.upper)
        low, high = aggregate(lowers), aggregate(uppers)
        corrected = identified_risk_set(low, high, widened)
        naive = identified_risk_set(low, high, box)
        if side == "upper":
            misses += corrected.upper < truth
            naive_misses += naive.upper < truth
        else:
            misses += corrected.lower > truth
            naive_misses += naive.lower > truth
    return misses / _REPLICATIONS, naive_misses / _REPLICATIONS


_PANEL_SCENARIOS = [
    (measure, side, prompts)
    for side, prompts in (("upper", _PANEL_UPPER), ("lower", _PANEL_LOWER))
    for measure in sorted(_MEASURES)
]


@pytest.mark.parametrize(("measure", "side", "prompts"), _PANEL_SCENARIOS)
def test_the_panel_derived_correction_of_a_row_a_policy_classified_covers_at_its_nominal_rate(
    measure: str, side: str, prompts: tuple[Prompt, ...]
) -> None:
    seed = 400 + _PANEL_SCENARIOS.index((measure, side, prompts))
    misses, naive = _panel_misses(measure, prompts, side, seed)
    allowance = 3 * math.sqrt(_NOMINAL * (1 - _NOMINAL) / _REPLICATIONS)
    assert misses <= _NOMINAL + allowance, (measure, side, misses)
    # Sharp: inverted over the box itself, the bound misses far more often than nominal.
    assert naive >= 0.5, (measure, side, naive)


def test_the_widened_panel_box_frees_only_the_ends_that_credit_the_panel_with_an_error() -> None:
    widened = _policy_free_accuracy_box(JudgeAccuracyBox(0.8, 0.95, 0.9, 0.99))
    assert widened == JudgeAccuracyBox(0.8, 1.0, 0.9, 1.0)
    corrected = identified_risk_set(0.2, 0.4, widened)
    assert corrected.lower == pytest.approx((0.2 - 0.1) / 0.9, abs=1e-15)
    assert corrected.upper == pytest.approx(0.4 / 0.8, abs=1e-15)
