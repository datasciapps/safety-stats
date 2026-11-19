"""Tail and threshold measures warn when the rollouts per prompt cannot make them informative.

Their intervals are propagated from simultaneous per-prompt bounds, so a large total of
rollouts spread thinly over many prompts still leaves them wide.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.schema.reports import Diagnostic, SafetyReport
from inspect_safety_stats.stats.binomial import binomial_upper_bound

CODE = "TAIL_MEASURES_UNINFORMATIVE"

PLAN = """
schema_version: "2.0"
analysis: {id: tail, title: Tail, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
outputs:
  prompt_bounds: {family: displayed}
  risk_measures: [@MEASURES@]
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
@REPORT@
"""

ROLLOUTS = 10


def _plan(tmp_path: Path, measures: str, report: str = "") -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts\n"
        f"p1,q1,t1,s,{ROLLOUTS}\np2,q2,t1,s,{ROLLOUTS}\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        PLAN.replace("@MEASURES@", measures).replace("@REPORT@", report).strip() + "\n",
        encoding="utf-8",
    )
    return plan


@pytest.fixture(autouse=True)
def _logs(monkeypatch: pytest.MonkeyPatch) -> None:
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "tail",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "tail",
                "epochs": ROLLOUTS,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt,
                "epoch": epoch,
                "score_harmful": "FAIL" if (prompt, epoch) == ("p1", 1) else "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt in ("p1", "p2")
            for epoch in range(1, ROLLOUTS + 1)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def _warnings(report: SafetyReport) -> list[Diagnostic]:
    return [item for item in report.integrity.diagnostics if item.code == CODE]


def test_tail_measures_warn_when_rollouts_cannot_support_them(tmp_path: Path) -> None:
    measures = "{id: mean, type: weighted_mean}, {id: cvar_90, type: cvar, beta: 0.9}"
    report = analyze(plan=_plan(tmp_path, measures), logs="ignored")

    (warning,) = _warnings(report)
    assert warning.severity == "WARNING"
    assert warning.scope == "plan/outputs/risk_measures"
    # Only the tail measure is named. With direct_mean_intervals off, as here, the mean's
    # interval is propagated from the same per-prompt bounds, but the warning covers tail and
    # threshold measures only; the summary marks a wide interval of any measure with a dagger.
    assert warning.details["measures"] == ["cvar_90"]
    assert warning.details["median_rollouts_per_prompt"] == ROLLOUTS
    alpha = report.prompt_estimates[0].alpha_upper
    assert alpha is not None
    upper = binomial_upper_bound(0, ROLLOUTS, alpha=alpha)
    assert warning.details["zero_failure_prompt_upper"] == pytest.approx(upper, rel=1e-12, abs=0)
    needed = warning.details["rollouts_per_prompt_needed"]
    # The fewest rollouts at which a prompt with no failures meets the 0.05 target.
    assert binomial_upper_bound(0, needed, alpha=alpha) <= 0.05
    assert binomial_upper_bound(0, needed - 1, alpha=alpha) > 0.05
    # A warning only: the analysis is still valid.
    assert report.integrity.status == "valid"


def test_mean_only_plans_do_not_warn_about_tail_measures(tmp_path: Path) -> None:
    report = analyze(plan=_plan(tmp_path, "{id: mean, type: weighted_mean}"), logs="ignored")
    assert _warnings(report) == []


def test_worst_of_k_warns_unless_it_has_a_direct_interval(tmp_path: Path) -> None:
    propagated = analyze(
        plan=_plan(tmp_path, "{id: worst_3, type: worst_of_k, k: 3}"), logs="ignored"
    )
    assert [item.details["measures"] for item in _warnings(propagated)] == [["worst_3"]]
    direct = analyze(
        plan=_plan(tmp_path, "{id: worst_3, type: worst_of_k, k: 3, direct_interval: true}"),
        logs="ignored",
    )
    assert _warnings(direct) == []


def test_a_loose_enough_precision_target_does_not_warn(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        "{id: max, type: maximum}",
        report="report: {precision_target: 0.49}",
    )
    assert _warnings(analyze(plan=plan, logs="ignored")) == []


def test_the_default_precision_target_leaves_the_plan_hash_alone(tmp_path: Path) -> None:
    default = load_plan(_plan(tmp_path, "{id: max, type: maximum}"))
    assert "precision_target" not in default.model_dump(mode="json")["report"]
    stated = load_plan(
        _plan(tmp_path, "{id: max, type: maximum}", report="report: {precision_target: 0.1}")
    )
    assert stated.model_dump(mode="json")["report"]["precision_target"] == 0.1
