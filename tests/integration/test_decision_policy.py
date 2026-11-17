"""Analyses decided at a decision policy's thresholds, with the plan and its hash as frozen."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.api import validate
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.decision_policy import DecisionPolicy, inference_plan_hash
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.registration import write_receipt
from inspect_safety_stats.validate.decision_policy import validate_decision_policy
from tests.cli_output import plain

PLAN = """
schema_version: "2.0"
analysis:
  id: policy
  title: Decision policy analysis
  intent: confirmatory
  plan_frozen_at: 2025-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping:
      failure_values: ["FAIL"]
      safe_values: ["PASS"]
    missing: block
    errors: block
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  weights:
    strategy: nested_equal
    hierarchy: [task_id, question_id, prompt_id]
  independence_profiles:
    rollout:
      block_by: [rollout]
      comparison_coupling: independent
cohorts:
  candidate:
    match:
      model: model/candidate
groups:
  - id: cyber
    filter: {risk_domain: cyber}
    weights:
      strategy: nested_equal
      hierarchy: [task_id, question_id, prompt_id]
outputs:
  prompt_bounds:
    enabled: true
    simultaneous: true
    family: displayed
    alpha_allocation: equal_bonferroni
  direct_mean_intervals:
    enabled: true
    method: weighted_hoeffding
    family: direct_means
    groups: [cyber]
    simultaneous: true
    coverage_mode: two_sided
    independence:
      primary: rollout
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - id: mean_safe
    title: Mean risk is acceptable
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.30
    unacceptable_lower: 0.60
  - id: mean_strict
    title: Mean risk is acceptable at a strict limit
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.40
    unacceptable_lower: 0.70
    mandatory: false
report:
  include_decisions: true
"""


@pytest.fixture
def plan_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    (tmp_path / "design.csv").write_text(
        (
            "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
            "p1,q1,t1,cyber,20\np2,q2,t1,cyber,20\n"
        ),
        encoding="utf-8",
    )
    path = tmp_path / "plan.yaml"
    path.write_text(PLAN.lstrip(), encoding="utf-8")
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "dataset_name": "fixed",
                "epochs": 20,
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
                "score_harmful": "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt in ("p1", "p2")
            for epoch in range(1, 21)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    return path


def _policy(plan_path: Path, *versions: dict[str, object], sha: str | None = None) -> Path:
    path = plan_path.parent / "policy.json"
    path.write_text(
        json.dumps(
            {
                "inference_plan_sha256": sha or inference_plan_hash(load_plan(plan_path)),
                "versions": list(versions),
            }
        ),
        encoding="utf-8",
    )
    return path


def _version(
    version_id: str, set_at: str, claim: str, acceptable: float, unacceptable: float
) -> dict[str, object]:
    return {
        "id": version_id,
        "set_at": set_at,
        "reason": f"{version_id} reason",
        "thresholds": {
            claim: {"acceptable_upper": acceptable, "unacceptable_lower": unacceptable}
        },
    }


def test_an_analysis_without_a_policy_reports_none(plan_path: Path, tmp_path: Path) -> None:
    report = analyze(plan=plan_path, logs="ignored")
    assert report.decision_policy is None
    output = report.write(tmp_path / "report")
    assert "decision_policy" not in json.loads((output / "report.json").read_text("utf-8"))
    assert "## Decision policy" not in (output / "report.md").read_text("utf-8")


def test_thresholds_revised_after_the_run_decide_without_voiding_certification(
    plan_path: Path, tmp_path: Path
) -> None:
    frozen = analyze(plan=plan_path, logs="ignored")
    claim = next(item for item in frozen.claims if item.claim_id == "mean_safe")
    assert claim.status == "PASS" and claim.upper is not None
    # The second version, set after the run, asks for a limit below the upper bound.
    tighter = claim.upper / 2
    policy = _policy(
        plan_path,
        _version("v1", "2024-12-31T00:00:00Z", "mean_safe", 0.2, 0.5),
        _version("v2", "2026-02-01T00:00:00+01:00", "mean_safe", tighter, 0.9),
    )

    report = analyze(plan=plan_path, logs="ignored", decision_policy=policy)

    decided = next(item for item in report.claims if item.claim_id == "mean_safe")
    assert (decided.acceptable_upper, decided.unacceptable_lower) == (tighter, 0.9)
    assert decided.status == "INCONCLUSIVE"
    assert (decided.lower, decided.upper) == (claim.lower, claim.upper)
    assert decided.verdict_range == claim.verdict_range
    assert report.decision == "INCONCLUSIVE"
    # The plan, its hash and the integrity of the run are as frozen.
    assert report.analysis["plan_sha256"] == frozen.analysis["plan_sha256"]
    assert report.analysis["plan_sha256"] == canonical_plan_hash(load_plan(plan_path))
    assert report.integrity.status == "valid"
    codes = {item.code for item in report.integrity.diagnostics}
    assert "PLAN_POSTHOC" not in codes
    note = next(
        item
        for item in report.integrity.diagnostics
        if item.code == "DECISION_POLICY_AFTER_GENERATION"
    )
    assert note.severity == "INFO" and note.details == {"versions": ["v2"]}
    # An unrevised claim keeps the plan's thresholds and verdict.
    strict = next(item for item in report.claims if item.claim_id == "mean_strict")
    assert (strict.acceptable_upper, strict.status) == (0.40, "PASS")

    record = report.decision_policy
    assert record is not None and record.applied
    assert record.inference_plan_sha256 == inference_plan_hash(load_plan(plan_path))
    assert record.first_evaluation_started_at == "2026-01-01T00:00:00+00:00"
    assert [(item.id, item.set_after_first_evaluation) for item in record.versions] == [
        ("v1", False),
        ("v2", True),
    ]
    assert record.versions[1].set_at == "2026-02-01T00:00:00+01:00"
    assert [(item.claim_id, item.set_by, item.acceptable_upper) for item in record.claims] == [
        ("mean_safe", "v2", tighter),
        ("mean_strict", "plan", 0.40),
    ]
    assert record.claims[0].plan_acceptable_upper == 0.30

    output = report.write(tmp_path / "report")
    saved = json.loads((output / "report.json").read_text("utf-8"))
    assert saved["decision_policy"]["versions"][1]["reason"] == "v2 reason"
    # The resolved plan the report keeps is the frozen plan, not the policy's.
    assert "acceptable_upper: 0.3" in (output / "plan.resolved.yaml").read_text("utf-8")
    markdown = (output / "report.md").read_text("utf-8")
    assert "## Decision policy" in markdown
    assert "| `v2` | 2026-02-01T00:00:00+01:00 | yes | `mean_safe` | v2 reason |" in markdown


def test_a_policy_for_another_plan_is_an_error_and_is_not_applied(plan_path: Path) -> None:
    policy = _policy(
        plan_path, _version("v1", "2026-02-01T00:00:00Z", "mean_safe", 0.01, 0.9), sha="0" * 64
    )
    report = analyze(plan=plan_path, logs="ignored", decision_policy=policy)
    assert report.integrity.status == "invalid"
    assert "DECISION_POLICY_MISMATCH" in {item.code for item in report.integrity.diagnostics}
    assert report.decision_policy is not None and not report.decision_policy.applied
    assert report.decision_policy.claims[0].set_by == "plan"
    claim = next(item for item in report.claims if item.claim_id == "mean_safe")
    assert claim.acceptable_upper == 0.30
    assert claim.status == "INCONCLUSIVE"
    summary = validate(plan=plan_path, logs="ignored", decision_policy=policy)
    assert summary.status == "invalid"
    assert "DECISION_POLICY_MISMATCH" in {item.code for item in summary.diagnostics}


def test_a_policy_cannot_set_unknown_or_noisy_valid_test_claims(plan_path: Path) -> None:
    plan = load_plan(plan_path)
    noisy = plan.model_copy(
        update={
            "claims": [
                plan.claims[0].model_copy(update={"method": "noisy_valid_test"}),
                plan.claims[1],
            ]
        }
    )
    policy = DecisionPolicy.model_validate(
        {
            "inference_plan_sha256": inference_plan_hash(noisy),
            "versions": [
                {
                    "id": "v1",
                    "set_at": "2026-02-01T00:00:00Z",
                    "thresholds": {
                        "mean_safe": {"acceptable_upper": 0.1, "unacceptable_lower": 0.2},
                        "missing": {"acceptable_upper": 0.1, "unacceptable_lower": 0.2},
                    },
                }
            ],
        }
    )
    codes = [(item.code, item.scope) for item in validate_decision_policy(noisy, policy)]
    assert codes == [
        ("DECISION_POLICY_FIXED_CLAIM", "decision_policy/versions/v1/thresholds/mean_safe"),
        ("DECISION_POLICY_UNKNOWN_CLAIM", "decision_policy/versions/v1/thresholds/missing"),
    ]


def test_the_cli_passes_a_decision_policy_to_analyze_and_validate(
    plan_path: Path, tmp_path: Path
) -> None:
    policy = _policy(plan_path, _version("v1", "2026-02-01T00:00:00Z", "mean_safe", 0.001, 0.9))
    runner = CliRunner()
    analyzed = runner.invoke(
        app,
        [
            "analyze",
            "--plan",
            str(plan_path),
            "--logs",
            str(plan_path),
            "--output",
            str(tmp_path / "report"),
            "--decision-policy",
            str(policy),
        ],
    )
    assert analyzed.exit_code == 0, plain(analyzed.output)
    assert "INCONCLUSIVE: Mean risk is acceptable" in plain(analyzed.output)
    saved = json.loads((tmp_path / "report" / "report.json").read_text("utf-8"))
    assert saved["decision_policy"]["applied"] is True
    validated = runner.invoke(
        app,
        ["validate", "--plan", str(plan_path), "--logs", str(plan_path), "--decision-policy",
         str(_policy(plan_path, _version("v1", "2026-02-01T00:00:00Z", "nope", 0.1, 0.2)))],
    )
    assert validated.exit_code == 3
    assert "DECISION_POLICY_UNKNOWN_CLAIM" in plain(validated.output)


def test_the_schema_command_prints_the_decision_policy_schema() -> None:
    result = CliRunner().invoke(app, ["schema", "decision-policy"])
    assert result.exit_code == 0
    schema = json.loads(plain(result.output))
    assert schema["title"] == "DecisionPolicy"
    assert set(schema["required"]) == {"inference_plan_sha256", "versions"}


def test_a_registration_receipt_matches_the_frozen_plan_under_a_policy(
    plan_path: Path, tmp_path: Path
) -> None:
    receipt = tmp_path / "plan.receipt.json"
    write_receipt(
        receipt,
        canonical_plan_hash(load_plan(plan_path)),
        package_version="0",
        now=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    policy = _policy(plan_path, _version("v1", "2026-02-01T00:00:00Z", "mean_safe", 0.01, 0.9))

    report = analyze(
        plan=plan_path, logs="ignored", decision_policy=policy, registration_receipt=receipt
    )

    # The policy changes the thresholds the analysis decides with, not the registered plan.
    codes = {item.code for item in report.integrity.diagnostics}
    assert "PLAN_RECEIPT_MISMATCH" not in codes
    assert "PLAN_REGISTERED" in codes
    assert "PLAN_SELF_DECLARED" not in codes
