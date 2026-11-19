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
from inspect_safety_stats.ingest import inspect_logs
from inspect_safety_stats.stats.binomial import binomial_upper_bound
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


def test_validated_mode_certifies_the_root_and_spends_its_own_alpha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "run"
    plan = with_judge(tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"})
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    write_labels(tmp_path, artifacts, unsafe_of_flagged=18, unsafe_of_clean=0)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    assert report.corrected
    # The pooled labels measure the judge for the equal-weight root only (#109).
    eligibility = {item.node_id: item.eligibility for item in report.corrected}
    assert eligibility.pop("all") == "certifying"
    assert eligibility and set(eligibility.values()) == {"diagnostic"}
    block = report.evaluation["judge_calibration"]
    assert block["mode"] == "validated" and block["available"] is True
    assert block["rho_1"][0] > block["rho_0"][1]        # strictly informative
    root = next(i for i in report.corrected if i.node_id == "all")
    assert root.alpha > report.tree.alpha               # the union, not the interval alone


def test_unclassified_units_keep_their_worst_case_through_the_correction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The labels say nothing about units with no verdict, so rho must not shrink them."""

    artifacts = tmp_path / "run"
    plan = with_judge(
        tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"}, total=300
    )
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    # The last 160 of 300 units, none of them labelled, never come back from the judge.
    prompt_ids = [
        str(record.prompt_id)
        for record in read_rewrites(artifacts / "rewrites.csv")
        if record.is_evaluable()
    ]
    missing = set(prompt_ids[-160:])
    samples = inspect_logs.samples_df
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: (
            lambda frame: frame[~frame["id"].isin(missing)]
        )(samples(logs, parallel=parallel)),
    )
    write_labels(tmp_path, artifacts, unsafe_of_flagged=8, unsafe_of_clean=0)

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    tree_root = report.tree.root
    assert tree_root.unclassified == 160
    assert tree_root.unclassified_share == pytest.approx(160 / 300, abs=1e-12)
    root = next(i for i in report.corrected if i.node_id == "all")
    assert root.eligibility == "certifying"
    # Every unit with no verdict being unsafe, with the 8 flagged units the labels found
    # unsafe, is a true risk these labels allow; the certified upper bound must cover it.
    # Mapping the whole interval through rho_1 < 1 put it below the share alone.
    assert root.upper >= (160 + 8) / 300
    # The run's 160 in 300 only estimate the design's share of such units. The correction
    # takes a one-sided upper confidence bound on it, at a quarter of the calibration's
    # alpha, which joins the root's own: the units keep their worst case at that bound.
    block = report.evaluation["judge_calibration"]
    share_alpha = block["alpha_spent"] / 4
    share = binomial_upper_bound(160, 300, alpha=share_alpha)
    upper, at_most = tree_root.upper, min(share, tree_root.upper)
    assert root.upper == pytest.approx(
        min(
            1.0,
            max(
                (upper - at_most) * rho_1 + (1 - upper) * rho_0 + at_most
                for rho_1 in block["rho_1"]
                for rho_0 in block["rho_0"]
            ),
        ),
        rel=1e-12,
        abs=0,
    )
    assert root.alpha == pytest.approx(
        report.tree.alpha + block["alpha_spent"] + share_alpha, rel=1e-12, abs=0
    )
    assert root.unclassified_upper == pytest.approx(share, rel=1e-12, abs=0)
    assert root.unclassified_alpha == pytest.approx(share_alpha, rel=1e-12, abs=0)


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


def test_an_unlabelled_judge_stratum_is_completed_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """L49: labels only among judge-safe samples still give a conservative correction."""

    artifacts = tmp_path / "run"
    plan = with_judge(tmp_path, {"mode": "validated", "validation_manifest": "labels.csv"})
    materialize_run(plan, artifacts)
    stub_logs(monkeypatch, artifacts, candidate_failures=20)
    # The first 20 evaluable samples are the judge's failures; label none of them.
    records = [r for r in read_rewrites(artifacts / "rewrites.csv") if r.is_evaluable()]
    rows = ["observation_id,human_label"]
    rows.extend(f"{record.prompt_id},safe" for record in records[20:140])
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = analyze_run(plan, artifacts, "logs", cohort="candidate")
    warnings = [i for i in report.diagnostics if i.code == "JUDGE_STRATUM_UNLABELLED_COMPLETED"]
    assert len(warnings) == 1 and warnings[0].severity == "WARNING"
    block = report.evaluation["judge_calibration"]
    assert block["available"] is True
    assert block["rho_1"] == [0.0, 1.0]
    assert block["completed_strata"] == ["judge_unsafe"]
    # The pooled labels measure the judge for the equal-weight root only (#109).
    eligibility = {item.node_id: item.eligibility for item in report.corrected}
    assert eligibility.pop("all") == "certifying"
    assert eligibility and set(eligibility.values()) == {"diagnostic"}
    root = next(i for i in report.corrected if i.node_id == "all")
    assert root.lower == 0.0
    assert root.upper >= report.tree.root.upper


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
    # A comparison belongs to no one cohort; its scope is the candidate:baseline it was asked for.
    assert [
        item.scope
        for item in report.diagnostics
        if item.code == "JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON"
    ] == ["comparisons/candidate:baseline"]
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
