"""Judge misclassification correction on the generative path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from inspect_safety_stats.generative.judge import judge_strata
from inspect_safety_stats.generative.ledger import read_rewrites
from inspect_safety_stats.generative.runner import analyze_run, materialize_run
from inspect_safety_stats.generative.schema import load_generative_plan
from tests.integration.test_generative_cli import stub_logs, write_plan

ASSUMED = {
    "mode": "assumed",
    "primary": "declared",
    "scenarios": [
        {"id": "declared", "rho_1": {"lower": 0.85, "upper": 1.0},
         "rho_0": {"lower": 0.0, "upper": 0.02}},
        {"id": "pessimistic", "rho_1": {"lower": 0.70, "upper": 1.0},
         "rho_0": {"lower": 0.0, "upper": 0.05}},
    ],
}


def with_judge(tmp_path: Path, block: dict, total: int = 200) -> Path:
    plan_path = write_plan(tmp_path, total=total)
    edited = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    edited["judge_calibration"] = block
    plan_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
    return plan_path


def write_labels(tmp_path: Path, artifacts: Path, *, unsafe_of_flagged: int,
                 unsafe_of_clean: int, flagged: int = 20) -> Path:
    """Adjudicate the first `flagged` judge-unsafe samples and a slice of the safe ones."""

    records = [r for r in read_rewrites(artifacts / "rewrites.csv") if r.is_evaluable()]
    rows = ["observation_id,human_label"]
    for index, record in enumerate(records):
        if index < flagged:                                   # judge said FAIL on these
            label = "unsafe" if index < unsafe_of_flagged else "safe"
        elif index < flagged + 120:                           # judge said PASS on these
            label = "unsafe" if index - flagged < unsafe_of_clean else "safe"
        else:
            continue
        rows.append(f"{record.prompt_id},{label}")
    path = tmp_path / "labels.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_assumed_mode_produces_sensitivity_not_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = with_judge(tmp_path, ASSUMED)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert report.corrected
    assert {item.eligibility for item in report.corrected} == {"diagnostic"}
    assert {item.source for item in report.corrected} == {"assumed"}
    assert report.evaluation["judge_calibration"]["mode"] == "assumed"

    # Both declared scenarios reach the root; only the primary reaches the nodes.
    root = [item for item in report.corrected if item.node_id == "all"]
    assert {item.scenario_id for item in root} == {"declared", "pessimistic"}
    deeper = [item for item in report.corrected if item.node_id != "all"]
    assert {item.scenario_id for item in deeper} == {"declared"}


def test_correction_lowers_the_estimate_and_widens_the_band(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = with_judge(tmp_path, ASSUMED)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=40)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    root = next(i for i in report.corrected if i.node_id == "all" and i.scenario_id == "declared")
    # False positives are being removed, so the corrected estimate sits lower...
    assert root.estimate_lower is not None
    assert root.estimate_lower < report.tree.root.estimate
    # ...while uncertainty about rho is real uncertainty added.
    assert (root.upper - root.lower) > (report.tree.root.upper - report.tree.root.lower)


def test_a_pessimistic_scenario_is_never_tighter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = with_judge(tmp_path, ASSUMED)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=40)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    root = {i.scenario_id: i for i in report.corrected if i.node_id == "all"}
    assert root["pessimistic"].upper >= root["declared"].upper


def test_validated_mode_is_certifying_and_spends_its_own_alpha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    plan = with_judge(tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"})
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    write_labels(tmp_path, artifacts, unsafe_of_flagged=18, unsafe_of_clean=0)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert report.corrected
    assert {item.eligibility for item in report.corrected} == {"certifying"}
    block = report.evaluation["judge_calibration"]
    assert block["mode"] == "validated" and block["available"] is True
    assert block["rho_1"][0] > block["rho_0"][1]        # strictly informative
    root = next(i for i in report.corrected if i.node_id == "all")
    assert root.alpha > report.tree.alpha               # the union, not the interval alone


def test_an_uninformative_label_set_refuses_rather_than_correcting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    plan = with_judge(tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"})
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    # A judge whose flagged items are mostly wrong establishes nothing.
    write_labels(tmp_path, artifacts, unsafe_of_flagged=2, unsafe_of_clean=30)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert not report.corrected
    codes = [item.code for item in report.diagnostics]
    assert "JUDGE_CALIBRATION_UNINFORMATIVE" in codes
    assert report.evaluation["judge_calibration"]["available"] is False


def test_labels_for_samples_this_run_never_saw_are_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    plan = with_judge(tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"})
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    path = write_labels(tmp_path, artifacts, unsafe_of_flagged=18, unsafe_of_clean=0)
    path.write_text(path.read_text(encoding="utf-8") + "p-not-ours,unsafe\n", encoding="utf-8")

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert any(i.code == "JUDGE_VALIDATION_UNMATCHED" for i in report.diagnostics)


def test_comparisons_are_refused_rather_than_silently_corrected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = with_judge(tmp_path, ASSUMED)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts)

    report = analyze_run(
        plan, artifacts, "logs", cohort="candidate", comparisons=("candidate:baseline",)
    )
    assert any(
        i.code == "JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON" for i in report.diagnostics
    )
    assert all(item.node_id != "comparison" for item in report.corrected)


def test_multiple_rollouts_per_rewrite_cannot_be_calibrated() -> None:
    """A unit must carry a single binary judge verdict."""

    with pytest.raises(ValueError, match="one rollout per rewrite"):
        judge_strata([], [("u-0", 4, 1, 0)], {})


def test_the_report_carries_the_corrected_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = with_judge(tmp_path, ASSUMED)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    payload = json.loads(
        (report.write(tmp_path / "report") / "report.json").read_text(encoding="utf-8")
    )
    assert payload["corrected"]
    assert payload["evaluation"]["judge_calibration"]["mode"] == "assumed"


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"mode": "assumed"}, "requires at least one scenario"),
        ({"mode": "validated"}, "requires a validation_manifest"),
        (
            {"mode": "assumed", "scenarios": [
                {"id": "a", "rho_1": {"lower": 0.1, "upper": 0.4},
                 "rho_0": {"lower": 0.0, "upper": 0.5}}]},
            "strictly informative",
        ),
        (
            {"mode": "validated", "validation_manifest": "x.csv",
             "scenarios": [{"id": "a", "rho_1": {"lower": 0.9, "upper": 1.0},
                            "rho_0": {"lower": 0.0, "upper": 0.1}}]},
            "does not declare scenarios",
        ),
    ],
)
def test_incoherent_calibration_blocks_are_rejected(
    tmp_path: Path, block: dict, message: str
) -> None:
    plan = with_judge(tmp_path, block)
    with pytest.raises(ValidationError, match=message):
        load_generative_plan(plan)
