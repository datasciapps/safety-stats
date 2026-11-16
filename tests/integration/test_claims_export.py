"""claims.json (experimental, #11), end to end through ``analyze`` and ``write``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.report.artifacts import sha256
from inspect_safety_stats.schema.claims import ClaimsExport
from tests.claims_export import round_trip_differences
from tests.cli_output import plain

_PLAN = """
schema_version: "2.0"
analysis:
  id: claims-export
  title: Claims export
  intent: {intent}
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
    filter: {{risk_domain: cyber}}
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
    - {{id: mean, type: weighted_mean}}
    - {{id: worst_of_2, type: worst_of_k, k: 2}}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}}
claims:
  - id: mean_safe
    title: Mean risk is acceptable
    cohort: candidate
    group: cyber
    measure: mean
    family: gate
    acceptable_upper: 0.30
    unacceptable_lower: 0.60
  - id: worst_of_2_safe
    title: Worst of two is acceptable
    cohort: candidate
    group: cyber
    measure: worst_of_2
    family: gate
    acceptable_upper: 0.50
    unacceptable_lower: 0.90
report:
  include_decisions: {include_decisions}
{export}"""


def _write(
    tmp_path: Path,
    *,
    export: str = "  claims_export: true\n",
    include_decisions: bool = True,
    intent: str = "confirmatory",
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,risk_domain,expected_rollouts\n"
        "p1,q1,t1,cyber,20\np2,q2,t1,cyber,20\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        _PLAN.format(
            export=export,
            include_decisions=str(include_decisions).lower(),
            intent=intent,
        ),
        encoding="utf-8",
    )
    return plan


@pytest.fixture(autouse=True)
def _offline_logs(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": 20,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt_id}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": "FAIL" if prompt_id == "p1" and epoch <= 2 else "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt_id in ("p1", "p2")
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


@pytest.mark.parametrize("intent", ["confirmatory", "exploratory"])
def test_every_exported_value_is_the_report_json_value(tmp_path: Path, intent: str) -> None:
    report = analyze(plan=_write(tmp_path, intent=intent), logs="ignored")
    output = report.write(tmp_path / "report")

    assert round_trip_differences(output) == []
    exported = ClaimsExport.model_validate_json(
        (output / "claims.json").read_text(encoding="utf-8")
    )
    assert [record.claim_id for record in exported.claims] == ["mean_safe", "worst_of_2_safe"]
    mean = exported.claims[0]
    # The claim is decided on the group risk's primary interval, and points at both.
    assert mean.sources.evidence is not None
    assert mean.sources.evidence.startswith("/group_risks/")
    assert mean.sources.evidence_interval == f"{mean.sources.evidence}/intervals/0"
    assert mean.evidence.interval_id == report.group_risks[0].primary_interval_id
    assert mean.evidence.independence_profile == "rollout"
    assert mean.evidence.assumptions
    assert not mean.judge_corrected
    if intent == "exploratory":
        assert mean.status == "INCONCLUSIVE"
        assert mean.descriptive_status == "PASS"
    else:
        assert mean.status == "PASS"


def test_claims_json_is_in_checksums(tmp_path: Path) -> None:
    output = analyze(plan=_write(tmp_path), logs="ignored").write(tmp_path / "report")

    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["claims.json"] == sha256(output / "claims.json")


def test_the_same_analysis_writes_the_same_bytes(tmp_path: Path) -> None:
    plan = _write(tmp_path)
    first = analyze(plan=plan, logs="ignored").write(tmp_path / "first")
    second = analyze(plan=plan, logs="ignored").write(tmp_path / "second")

    assert (first / "claims.json").read_bytes() == (second / "claims.json").read_bytes()


def test_a_plan_that_does_not_opt_in_writes_what_it_wrote_before(tmp_path: Path) -> None:
    off = _write(tmp_path / "off", export="")
    explicit = _write(tmp_path / "explicit", export="  claims_export: false\n")
    assert canonical_plan_hash(load_plan(off)) == canonical_plan_hash(load_plan(explicit))

    output = analyze(plan=off, logs="ignored").write(tmp_path / "report")
    assert not (output / "claims.json").exists()
    for name in ("report.json", "plan.resolved.yaml", "checksums.json"):
        assert "claims" + "_export" not in (output / name).read_text(encoding="utf-8")
        assert "claims.json" not in (output / name).read_text(encoding="utf-8")


def test_rewriting_without_the_export_removes_a_stale_claims_json(tmp_path: Path) -> None:
    output = tmp_path / "report"
    analyze(plan=_write(tmp_path / "on"), logs="ignored").write(output)
    assert (output / "claims.json").exists()

    analyze(plan=_write(tmp_path / "off", export=""), logs="ignored").write(output, overwrite=True)
    assert not (output / "claims.json").exists()


def test_the_export_is_refused_without_decisions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="claims_export needs report.include_decisions"):
        load_plan(_write(tmp_path, include_decisions=False))


def test_the_schema_command_prints_the_claims_schema() -> None:
    result = CliRunner().invoke(app, ["schema", "claims"])

    assert result.exit_code == 0, plain(result.output)
    assert json.loads(result.output) == ClaimsExport.model_json_schema()
