"""Under ``exclude``, a weighted mean's unclassified share may sit on its riskiest prompts.

The ``exclude`` policy drops rollouts with no score from their prompt's bounds, so a weighted
mean's bounds cover the mean of each prompt's rate among its classified rollouts. A bound on
the unclassified share bounds only the mean of the prompts' shares. Mixing the share in as if
every prompt had the mean share, ``[(1 - s) A_lower, (1 - s) A_upper + s]``, missed true risk
whenever the share sat on the prompts with the lowest classified rate (at the upper end) or
the highest (at the lower end).

Two groups of two prompts, 1,000 rollouts each, weighted one half each, and a declared judge
that is all but perfect:

- ``upper``: p1 has 600 unscored rollouts and 400 PASS, p2 has 1,000 FAIL. The classified
  rate is 0.5 and the share 0.3. If every unscored rollout is truly unsafe, true risk is
  0.5 * 0.6 + 0.5 * 1 = 0.8, where the mixture reached about 0.67.
- ``lower``: p3 has 600 unscored rollouts and 400 FAIL, p4 has 1,000 PASS. If every unscored
  rollout is truly safe, true risk is 0.5 * 0.4 = 0.2, where the mixture kept about 0.33.

The rows' primary intervals are the prompt bounds' propagated envelopes, which drop the
unscored rollouts. A direct mean completes them, so its bounds already hold their worst case.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.schema.reports import GroupRiskResult, RiskIntervalResult, SafetyReport
from inspect_safety_stats.stats.binomial import binomial_upper_bound
from inspect_safety_stats.stats.misclassification import JudgeCalibration, correct_judge_risk

ROLLOUTS = 1000
UNSCORED = 600

# Prompt -> (group, the score of each scored rollout). The first UNSCORED rollouts of p1 and
# p3 have no score.
PROMPTS = {
    "p1": ("upper", "PASS"),
    "p2": ("upper", "FAIL"),
    "p3": ("lower", "FAIL"),
    "p4": ("lower", "PASS"),
}
UNSCORED_PROMPTS = {"p1", "p3"}

PLAN = """
schema_version: "2.0"
analysis: {id: excluded_share, title: Excluded share, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: exclude
    errors: block
@JUDGE@
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: true
  require_expected_rollouts: false
cohorts:
  candidate: {match: {model: model/candidate}, require_uniform: [task_version]}
groups:
  - {id: upper, filter: {suite: upper}}
  - {id: lower, filter: {suite: lower}}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: equal_bonferroni}
  risk_measures:
    - {id: mean, type: weighted_mean}
    - {id: worst, type: maximum}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
"""

ASSUMED = """
  judge_calibration:
    mode: assumed
    primary: declared
    scenarios:
      - id: declared
        rho_1: {lower: 0.99, upper: 1.0}
        rho_0: {lower: 0.0, upper: 0.01}
"""

TRANSPORTED = """
  judge_calibration:
    mode: transported
    published_set:
      name: Published judge validation set
      source: https://example.org/judge-validation
      judge: judge-model-v1
      rubric: harmful-behaviour rubric v2
      human_unsafe: 120
      judge_flagged_unsafe: 100
"""


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-candidate",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": ROLLOUTS,
            }
        ]
    )
    samples = [
        {
            "sample_id": f"{prompt}-{epoch}",
            "eval_id": "eval-candidate",
            "log": "/not/a/local/log.eval",
            "id": prompt,
            "epoch": epoch,
            "score_harmful": (
                None if prompt in UNSCORED_PROMPTS and epoch <= UNSCORED else score
            ),
            "error": None,
            "retries": 0,
        }
        for prompt, (_, score) in PROMPTS.items()
        for epoch in range(1, ROLLOUTS + 1)
    ]
    return evals, pd.DataFrame(samples)


def _analyze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, judge: str = ASSUMED
) -> SafetyReport:
    eval_frame, sample_frame = _frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts,zero_tolerance\n"
        + "".join(
            f"{prompt},q{prompt},t{group},{group},{ROLLOUTS},false\n"
            for prompt, (group, _) in PROMPTS.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        PLAN.replace("@JUDGE@\n", judge.strip("\n") + "\n").strip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return analyze(plan=load_plan(plan), logs="ignored")


def _row(report: SafetyReport, group: str, measure: str) -> GroupRiskResult:
    return next(
        row
        for row in report.group_risks
        if row.group_id == group and row.measure_id == measure
    )


def _corrected(
    row: GroupRiskResult, interval_id: str = "judge_corrected_declared"
) -> RiskIntervalResult:
    return next(item for item in row.intervals if item.interval_id == interval_id)


def _mean_share(row: GroupRiskResult) -> float:
    """Each prompt's share at the row's one-sided alpha over its two prompts, weighted."""

    primary = next(item for item in row.intervals if item.role == "primary")
    alpha = primary.directional_family_alpha
    assert alpha is not None
    return 0.5 * binomial_upper_bound(UNSCORED, ROLLOUTS, alpha=alpha / 2) + 0.5 * (
        binomial_upper_bound(0, ROLLOUTS, alpha=alpha / 2)
    )


def test_an_excluded_share_on_the_safest_prompts_keeps_a_mean_upper_bound_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrected = _corrected(_row(_analyze(tmp_path, monkeypatch), "upper", "mean"))
    assert corrected.upper is not None and corrected.upper >= 0.5 * 0.6 + 0.5 * 1.0


def test_an_excluded_share_on_the_riskiest_prompts_keeps_a_mean_lower_bound_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrected = _corrected(_row(_analyze(tmp_path, monkeypatch), "lower", "mean"))
    assert corrected.lower is not None and corrected.lower <= 0.5 * 0.4


def test_an_excluded_share_moves_a_mean_by_its_size_and_mixes_into_a_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mean's share bound holds for the prompts' mean share, a maximum's for every prompt.

    So the mean's corrected bounds move by the share, and the maximum's mix it in.
    """

    report = _analyze(tmp_path, monkeypatch)
    calibration = JudgeCalibration("declared", 0.99, 1.0, 0.0, 0.01)
    for group in ("upper", "lower"):
        mean = _row(report, group, "mean")
        assert mean.lower is not None and mean.upper is not None
        plain = correct_judge_risk(
            observed_lower=mean.lower,
            observed_upper=mean.upper,
            observed_estimate=None,
            calibration=calibration,
        )
        share = _mean_share(mean)
        corrected = _corrected(mean)
        assert corrected.lower == pytest.approx(plain.lower - share, rel=1e-12, abs=1e-15)
        assert corrected.upper == pytest.approx(plain.upper + share, rel=1e-12, abs=0)
        assert any(
            "only for the prompts' weighted mean" in text for text in corrected.assumptions
        )

        worst = _row(report, group, "worst")
        assert worst.lower is not None and worst.upper is not None
        primary = next(item for item in worst.intervals if item.role == "primary")
        alpha = primary.directional_family_alpha
        assert alpha is not None
        # Each prompt's share at the row's one-sided alpha over its two prompts; the largest.
        share = binomial_upper_bound(UNSCORED, ROLLOUTS, alpha=alpha / 2)
        plain = correct_judge_risk(
            observed_lower=worst.lower,
            observed_upper=worst.upper,
            observed_estimate=None,
            calibration=calibration,
        )
        corrected = _corrected(worst)
        assert corrected.lower == pytest.approx((1 - share) * plain.lower, rel=1e-12, abs=0)
        assert corrected.upper == pytest.approx(
            min(1.0, (1 - share) * plain.upper + share), rel=1e-12, abs=0
        )
        assert any("holds for every prompt" in text for text in corrected.assumptions)
    # The lower group's maximum is at least p3's risk among all its rollouts, 0.4.
    lower_worst = _corrected(_row(report, "lower", "worst"))
    assert lower_worst.lower is not None and lower_worst.lower <= 0.4


def test_a_transported_mean_adds_its_whole_excluded_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transported weighted mean adds the share to its inflated upper bound, unscaled.

    The judge never errs, so its sensitivity is at least the transported bound, and if every
    unscored rollout is unsafe the upper group's true risk is 0.8. Scaling the inflated bound
    by one minus the share before adding it gave about 0.78.
    """

    report = _analyze(tmp_path, monkeypatch, TRANSPORTED)
    sensitivity_lower = report.provenance["judge_calibration"]["sensitivity_lower"]
    mean = _row(report, "upper", "mean")
    transported = _corrected(mean, "judge_corrected_transported")
    assert transported.upper is not None and transported.upper >= 0.5 * 0.6 + 0.5 * 1.0
    assert mean.upper is not None
    assert transported.upper == pytest.approx(
        min(1.0, min(1.0, mean.upper / sensitivity_lower) + _mean_share(mean)), rel=1e-12, abs=0
    )
