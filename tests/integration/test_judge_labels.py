from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.cli import app
from inspect_safety_stats.judge_labels import draw_label_sample
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


@pytest.fixture(scope="module")
def analysed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("judge")
    logs = write_mock_logs(root / "logs", epochs=40)
    plan = scaffold_plan(logs, root / "plan", failure_values=["FAIL"]).plan_path
    report = analyze(plan=plan, logs=str(logs)).write(root / "report")
    return logs, plan, report


def test_sample_is_blinded_stratified_and_reproducible(
    analysed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, _, report = analysed
    first = draw_label_sample(report, tmp_path / "a", judge_unsafe=5, judge_safe=30, seed=11)
    again = draw_label_sample(report, tmp_path / "b", judge_unsafe=5, judge_safe=30, seed=11)
    other = draw_label_sample(report, tmp_path / "c", judge_unsafe=5, judge_safe=30, seed=12)

    sheet = pd.read_csv(first.sheet_path, dtype=str, keep_default_na=False)
    assert len(sheet) == first.rows == 2 * (5 + 30)
    assert {"failure", "cohort_id", "model", "score_raw"}.isdisjoint(sheet.columns)
    assert (sheet["human_label"] == "").all()
    assert first.strata["alpha"]["judge_unsafe"] == {
        "stratum_size": 20, "requested": 5, "drawn": 5,
    }

    record = json.loads(first.record_path.read_text(encoding="utf-8"))
    assert record["seed"] == 11
    assert record["design"] == "simple_random_within_judge_stratum"
    repeat = json.loads(again.record_path.read_text(encoding="utf-8"))
    different = json.loads(other.record_path.read_text(encoding="utf-8"))
    assert record["selected_observation_ids"] == repeat["selected_observation_ids"]
    assert record["selected_observation_ids"] != different["selected_observation_ids"]


def test_oversized_requests_take_the_whole_stratum_and_say_so(
    analysed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    _, _, report = analysed
    result = draw_label_sample(report, tmp_path, judge_unsafe=50, judge_safe=10, seed=1)
    assert result.strata["beta"]["judge_unsafe"]["drawn"] == 20
    assert any("census" in note for note in result.notes)


def test_filled_sheet_drives_a_certifying_validated_correction(
    analysed: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    logs, plan_path, report = analysed
    result = draw_label_sample(report, tmp_path, judge_unsafe=20, judge_safe=120, seed=3)
    observations = pd.read_csv(report / "observations.csv", dtype=str, keep_default_na=False)
    verdicts = dict(zip(observations["observation_id"], observations["failure"], strict=True))
    sheet = pd.read_csv(result.sheet_path, dtype=str, keep_default_na=False)
    sheet["human_label"] = [
        "unsafe" if verdicts[item] == "True" else "safe" for item in sheet["observation_id"]
    ]
    sheet.to_csv(result.sheet_path, index=False)

    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    plan["inputs"]["judge_calibration"] = {
        "mode": "validated",
        "validation_manifest": str(result.sheet_path),
        "family": "judge",
    }
    plan["multiplicity"]["families"].append(
        {"id": "judge", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": 0.05}
    )
    validated_plan = tmp_path / "validated.yaml"
    validated_plan.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    (tmp_path / "design.csv").write_text(
        (plan_path.parent / "design.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )

    corrected = analyze(plan=validated_plan, logs=str(logs))
    assert corrected.integrity.status != "invalid"
    root = [risk for risk in corrected.group_risks if risk.measure_id == "mean"]
    assert len(root) == 2
    for risk in root:
        judged = [
            item for item in risk.intervals if item.interval_id == "judge_corrected_validated"
        ]
        assert judged and judged[0].eligibility == "certifying"
        assert judged[0].lower is not None and judged[0].upper is not None
        assert judged[0].lower <= 0.1 <= judged[0].upper


def test_sampling_requires_both_strata_and_an_observations_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both judge strata"):
        draw_label_sample(tmp_path, tmp_path, judge_unsafe=0, judge_safe=5, seed=0)
    with pytest.raises(FileNotFoundError, match="observations.csv"):
        draw_label_sample(tmp_path, tmp_path, judge_unsafe=1, judge_safe=5, seed=0)


def test_judge_commands(analysed: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, _, report = analysed
    sample = CliRunner().invoke(
        app,
        ["judge", "sample", "--report", str(report), "--judge-unsafe", "3",
         "--judge-safe", "10", "--seed", "5", "--output", str(tmp_path)],
    )
    assert sample.exit_code == 0, sample.output
    assert "alpha: 3 of 20 judge-unsafe, 10 of 180 judge-safe" in sample.output

    planned = CliRunner().invoke(
        app, ["plan", "judge-labels", "--judge-risk-upper", "0.05", "--target", "0.08"]
    )
    assert planned.exit_code == 0, planned.output
    assert "judge-safe observations" in planned.output

    unreachable = CliRunner().invoke(
        app, ["plan", "judge-labels", "--judge-risk-upper", "0.10", "--target", "0.08"]
    )
    assert unreachable.exit_code == 1
    assert "Unreachable" in unreachable.output
