"""Analyses decided at a decision policy's thresholds, with the plan and its hash as frozen."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.api import validate
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.decision_policy import DecisionPolicy, inference_plan_hash
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.registration import write_receipt
from inspect_safety_stats.schema.reports import Diagnostic, SafetyReport
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
    assert "## Decision policy" not in (output / "report-full.md").read_text("utf-8")


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
    # v1 states a time before the run, which only the policy's author vouches for.
    self_declared = next(
        item
        for item in report.integrity.diagnostics
        if item.code == "DECISION_POLICY_SELF_DECLARED"
    )
    assert self_declared.severity == "WARNING"
    assert self_declared.details == {"versions": ["v1"]}
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
    markdown = (output / "report-full.md").read_text("utf-8")
    assert "## Decision policy" in markdown
    assert "| `v2` | 2026-02-01T00:00:00+01:00 | yes | `mean_safe` | v2 reason |" in markdown
    assert "Each version's time is as the policy states it; nothing in the analysis checks it." in (
        markdown
    )


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


# --- The overall decision when a policy moves several mandatory claims' thresholds --------

# A run that started on 2026-02-01; policies are set before or after it.
_RUN = {"run_key": "r1", "model": "m", "status": "success", "started_at": "2026-02-01T00:00:00Z"}


def _gate_plan(
    groups: list[str], *, direct: dict[str, object] | None, measures: list[str] | None = None
) -> dict[str, object]:
    """One claim per group (and measure), each mandatory, in an intersection-union family."""

    measures = measures or ["mean"]
    families: list[dict[str, object]] = [
        {"id": "displayed", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": 0.05},
        {"id": "gate", "purpose": "all_mandatory", "method": "intersection_union",
         "alpha": 0.05},
    ]
    outputs: dict[str, object] = {
        "prompt_bounds": {"enabled": True, "family": "displayed"},
        "risk_measures": [
            {"id": "mean", "type": "weighted_mean"},
            {"id": "worst", "type": "maximum"},
        ],
    }
    if direct is not None:
        families.append(
            {"id": "direct", "purpose": "simultaneous_intervals", "method": "bonferroni",
             "alpha": 0.05}
        )
        outputs["direct_mean_intervals"] = {
            "enabled": True,
            "family": "direct",
            "independence": {"primary": "rollout"},
            **direct,
        }
    return {
        "schema_version": "3.0",
        "analysis": {
            "id": "gate",
            "title": "Gate",
            "intent": "confirmatory",
            "plan_frozen_at": "2026-01-01T00:00:00Z",
        },
        "inputs": {
            "design_manifest": "design.csv",
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "design": {"independence_profiles": {"rollout": {"block_by": ["rollout"]}}},
        "cohorts": {"candidate": {"match": {"model": "m"}}},
        "groups": [{"id": group, "filter": {"domain": group}} for group in groups],
        "outputs": outputs,
        "multiplicity": {"families": families},
        "claims": [
            {
                "id": f"{group}_{measure}",
                "title": f"{group} {measure} is acceptable",
                "cohort": "candidate",
                "group": group,
                "measure": measure,
                "family": "gate",
                "acceptable_upper": 0.05,
                "unacceptable_lower": 0.95,
            }
            for group in groups
            for measure in measures
        ],
    }


def _gate_inputs(tmp_path: Path, plan: dict[str, object], groups: list[str]) -> Path:
    """Two prompts per group, 40 rollouts each, 4 of them failures."""

    (tmp_path / "plan.yaml").write_text(json.dumps(plan), encoding="utf-8")
    prompts = [(group, f"{group}-p{index}") for group in groups for index in (1, 2)]
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,domain,expected_rollouts\n"
        + "".join(f"{prompt},{prompt},t1,{group},40\n" for group, prompt in prompts),
        encoding="utf-8",
    )
    (tmp_path / "runs.jsonl").write_text(json.dumps(_RUN) + "\n", encoding="utf-8")
    (tmp_path / "rollouts.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "run_key": "r1",
                    "item_id": prompt,
                    "repeat": repeat,
                    "score_harmful": "FAIL" if repeat <= 4 else "PASS",
                }
            )
            + "\n"
            for _, prompt in prompts
            for repeat in range(1, 41)
        ),
        encoding="utf-8",
    )
    return tmp_path / "plan.yaml"


def _gate_analysis(plan_path: Path, policy: Path | None = None) -> SafetyReport:
    from inspect_safety_stats.ingest.rollout_files import RolloutFiles

    directory = plan_path.parent
    return analyze(
        plan=plan_path,
        source=RolloutFiles(runs=directory / "runs.jsonl", rollouts=directory / "rollouts.jsonl"),
        decision_policy=policy,
    )


def _at_the_bounds(
    plan_path: Path, report: SafetyReport, claims: list[str] | None = None, *,
    set_at: str = "2026-03-01T00:00:00Z",
) -> Path:
    """A policy that sets each claim's acceptable_upper at its own upper bound, rounded up.

    A threshold read off the bound it is compared with passes by construction: it is the
    most data-dependent threshold there is.
    """

    chosen = [item for item in report.claims if claims is None or item.claim_id in claims]
    version = {
        "id": "at-the-bounds",
        "set_at": set_at,
        "thresholds": {
            item.claim_id: {
                "acceptable_upper": math.ceil(cast(float, item.upper) * 1e6) / 1e6,
                "unacceptable_lower": 0.99,
            }
            for item in chosen
        },
    }
    path = plan_path.parent / "policy.json"
    path.write_text(
        json.dumps(
            {
                "inference_plan_sha256": inference_plan_hash(load_plan(plan_path)),
                "versions": [version],
            }
        ),
        encoding="utf-8",
    )
    return path


def _overall_warning(report: SafetyReport) -> Diagnostic | None:
    return next(
        (
            item
            for item in report.integrity.diagnostics
            if item.code == "DECISION_POLICY_OVERALL_UNCERTIFIED"
        ),
        None,
    )


def test_thresholds_moved_on_separate_coverage_events_do_not_certify_an_overall_pass(
    tmp_path: Path,
) -> None:
    groups = ["g1", "g2", "g3"]
    plan = _gate_plan(groups, direct={"simultaneous": False})
    plan_path = _gate_inputs(tmp_path, plan, groups)
    frozen = _gate_analysis(plan_path)
    assert {item.status for item in frozen.claims} == {"INCONCLUSIVE"}
    # Loosened after the run, each to its own pointwise upper bound.
    report = _gate_analysis(plan_path, _at_the_bounds(plan_path, frozen))

    # Every claim passes, and keeps that verdict: its error bound holds at any threshold.
    assert {item.status for item in report.claims} == {"PASS"}
    # The overall PASS would rest on three separate coverage events, so it is not certified.
    assert report.decision == "INCONCLUSIVE"
    assert report.integrity.status == "valid"
    warning = _overall_warning(report)
    assert warning is not None and warning.severity == "WARNING"
    assert warning.details["claims"] == ["g1_mean", "g2_mean", "g3_mean"]
    assert len(set(warning.details["coverage_events"].values())) == 3


def test_a_backdated_policy_version_does_not_restore_the_overall_pass(tmp_path: Path) -> None:
    # set_at is the policy author's word, so a version dated before the run counts as well.
    groups = ["g1", "g2"]
    plan_path = _gate_inputs(tmp_path, _gate_plan(groups, direct={"simultaneous": False}), groups)
    frozen = _gate_analysis(plan_path)
    report = _gate_analysis(
        plan_path, _at_the_bounds(plan_path, frozen, set_at="2025-12-01T00:00:00Z")
    )
    codes = {item.code for item in report.integrity.diagnostics}
    assert "DECISION_POLICY_AFTER_GENERATION" not in codes
    assert {item.status for item in report.claims} == {"PASS"}
    assert report.decision == "INCONCLUSIVE"
    assert _overall_warning(report) is not None


@pytest.mark.parametrize(
    "direct",
    [{"simultaneous": True}, None],
    ids=["simultaneous-direct-means", "simultaneous-prompt-bounds"],
)
def test_thresholds_moved_on_one_simultaneous_family_keep_the_overall_pass(
    tmp_path: Path, direct: dict[str, object] | None
) -> None:
    groups = ["g1", "g2", "g3"]
    plan_path = _gate_inputs(tmp_path, _gate_plan(groups, direct=direct), groups)
    frozen = _gate_analysis(plan_path)
    report = _gate_analysis(plan_path, _at_the_bounds(plan_path, frozen))
    assert {item.status for item in report.claims} == {"PASS"}
    # One event covers every bound, so a wrong overall PASS is as rare as one wrong claim.
    assert report.decision == "PASS"
    assert _overall_warning(report) is None


def test_thresholds_moved_on_two_simultaneous_families_do_not_certify_an_overall_pass(
    tmp_path: Path,
) -> None:
    # The means are decided on the direct-mean family, the worst prompts on the prompt family.
    groups = ["g1"]
    plan = _gate_plan(groups, direct={"simultaneous": True}, measures=["mean", "worst"])
    plan_path = _gate_inputs(tmp_path, plan, groups)
    frozen = _gate_analysis(plan_path)
    report = _gate_analysis(plan_path, _at_the_bounds(plan_path, frozen))
    assert {item.status for item in report.claims} == {"PASS"}
    assert report.decision == "INCONCLUSIVE"
    warning = _overall_warning(report)
    assert warning is not None
    events = warning.details["coverage_events"]
    assert events["g1_mean"].endswith("of family direct")
    assert events["g1_worst"].endswith("of family displayed")


@pytest.mark.parametrize("simultaneous", [False, True], ids=["pointwise", "simultaneous"])
def test_one_moved_threshold_keeps_an_overall_pass_only_on_one_coverage_event(
    tmp_path: Path, simultaneous: bool
) -> None:
    groups = ["g1", "g2"]
    plan = _gate_plan(groups, direct={"simultaneous": simultaneous})
    for claim in cast(list[dict[str, object]], plan["claims"]):
        claim["acceptable_upper"] = 0.5
    plan_path = _gate_inputs(tmp_path, plan, groups)
    frozen = _gate_analysis(plan_path)
    assert frozen.decision == "PASS"

    # g1 is tightened to its own bound, g2 keeps the plan's threshold. Which claim to move
    # may itself be chosen from the bounds, as the one with the lowest bound, which is the
    # likeliest to have missed; only one event covering both bounds keeps that in check.
    report = _gate_analysis(plan_path, _at_the_bounds(plan_path, frozen, ["g1_mean"]))
    assert {item.status for item in report.claims} == {"PASS"}
    warning = _overall_warning(report)
    if simultaneous:
        assert report.decision == "PASS" and warning is None
    else:
        assert report.decision == "INCONCLUSIVE"
        assert warning is not None and warning.details["claims"] == ["g1_mean"]
        # Every mandatory claim's event is listed, the unmoved one's too.
        assert sorted(warning.details["coverage_events"]) == ["g1_mean", "g2_mean"]


def test_thresholds_an_overall_pass_does_not_read_keep_it(tmp_path: Path) -> None:
    groups = ["g1", "g2"]
    plan = _gate_plan(groups, direct={"simultaneous": False})
    for claim in cast(list[dict[str, object]], plan["claims"]):
        claim["acceptable_upper"] = 0.5
    cast(list[dict[str, object]], plan["claims"])[1]["mandatory"] = False
    plan_path = _gate_inputs(tmp_path, plan, groups)
    frozen = _gate_analysis(plan_path)
    assert frozen.decision == "PASS"
    policy = plan_path.parent / "restated.json"
    policy.write_text(
        json.dumps(
            {
                "inference_plan_sha256": inference_plan_hash(load_plan(plan_path)),
                "versions": [
                    {
                        "id": "restated",
                        "set_at": "2026-03-01T00:00:00Z",
                        "thresholds": {
                            # The plan's acceptable_upper, restated; unacceptable_lower moves,
                            # and no PASS depends on it.
                            "g1_mean": {"acceptable_upper": 0.5, "unacceptable_lower": 0.8},
                            # An optional claim, which the overall decision does not read.
                            "g2_mean": {"acceptable_upper": 0.45, "unacceptable_lower": 0.9},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = _gate_analysis(plan_path, policy)
    assert report.decision == "PASS" and _overall_warning(report) is None


def test_a_failing_claim_keeps_the_overall_fail(tmp_path: Path) -> None:
    groups = ["g1", "g2", "g3"]
    plan_path = _gate_inputs(tmp_path, _gate_plan(groups, direct={"simultaneous": False}), groups)
    frozen = _gate_analysis(plan_path)
    policy = _at_the_bounds(plan_path, frozen)
    value = json.loads(policy.read_text(encoding="utf-8"))
    # g3's limits are both at or below its lower bound, so it fails.
    lower = next(item for item in frozen.claims if item.claim_id == "g3_mean").lower
    value["versions"][0]["thresholds"]["g3_mean"] = {
        "acceptable_upper": -0.5,
        "unacceptable_lower": math.floor(cast(float, lower) * 1e6) / 1e6,
    }
    policy.write_text(json.dumps(value), encoding="utf-8")
    report = _gate_analysis(plan_path, policy)
    assert next(item for item in report.claims if item.claim_id == "g3_mean").status == "FAIL"
    assert report.decision == "FAIL"
    assert _overall_warning(report) is not None


def test_outcome_mapping_sensitivities_are_held_to_the_same_overall_decision(
    tmp_path: Path,
) -> None:
    groups = ["g1", "g2"]
    plan = _gate_plan(groups, direct={"simultaneous": False})
    score = cast(dict[str, object], cast(dict[str, object], plan["inputs"])["score"])
    score["mapping_sensitivities"] = [
        {
            "id": "unsure_fails",
            "title": "Unsure answers fail",
            "mapping": {"failure_values": ["FAIL", "UNSURE"], "safe_values": ["PASS"]},
        }
    ]
    plan_path = _gate_inputs(tmp_path, plan, groups)
    frozen = _gate_analysis(plan_path)
    report = _gate_analysis(plan_path, _at_the_bounds(plan_path, frozen))
    assert report.decision == "INCONCLUSIVE"
    (sensitivity,) = report.outcome_mapping_sensitivities
    assert sensitivity.decision == "INCONCLUSIVE"
    assert sensitivity.diagnostic_counts["DECISION_POLICY_OVERALL_UNCERTIFIED"] == 1


def test_a_version_is_not_reported_before_the_run_when_a_start_is_unknown(
    tmp_path: Path,
) -> None:
    groups = ["g1"]
    plan_path = _gate_inputs(tmp_path, _gate_plan(groups, direct=None), groups)
    # Half the rollouts come from a second run of the same model that reports no start time,
    # so it may have started before r1.
    unstarted = {key: value for key, value in _RUN.items() if key != "started_at"}
    (tmp_path / "runs.jsonl").write_text(
        json.dumps(_RUN) + "\n" + json.dumps({**unstarted, "run_key": "r2"}) + "\n",
        encoding="utf-8",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    records = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["repeat"] > 20:
            record["run_key"] = "r2"
    rollouts.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    thresholds = {"g1_mean": {"acceptable_upper": 0.5, "unacceptable_lower": 0.9}}
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "inference_plan_sha256": inference_plan_hash(load_plan(plan_path)),
                "versions": [
                    {"id": "early", "set_at": "2026-01-15T00:00:00Z", "thresholds": thresholds},
                    {"id": "late", "set_at": "2026-03-01T00:00:00Z", "thresholds": thresholds},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = _gate_analysis(plan_path, policy)

    record = report.decision_policy
    assert record is not None and record.applied
    assert record.first_evaluation_started_at == "2026-02-01T00:00:00+00:00"
    # "late" follows the earliest recorded start, so it follows the first evaluation's start.
    # "early" precedes r1's, but r2 may have started before it: that cannot be told.
    assert [(item.id, item.set_after_first_evaluation) for item in record.versions] == [
        ("early", None),
        ("late", True),
    ]
