from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.cli import app
from inspect_safety_stats.registry import ASSUMPTIONS
from inspect_safety_stats.scaffold import scaffold_plan
from tests.cli_output import plain
from tests.mock_inspect_logs import mock_task, write_mock_logs


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("mock") / "logs")


def test_init_refuses_to_guess_which_values_are_failures(mock_logs: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--failure-values"):
        scaffold_plan(mock_logs, tmp_path)
    assert not (tmp_path / "safety-plan.yaml").exists()


@pytest.fixture(scope="module")
def three_valued_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One model whose score takes C, I and P: p0 is graded I, P, C and C in turn."""

    metadata = [{"verdicts": ["I", "P", "C", "C"] if index == 0 else ["C"]} for index in range(5)]
    return write_mock_logs(
        tmp_path_factory.mktemp("three-valued") / "logs",
        models=("mockllm/alpha",),
        tasks=[mock_task(metadata=metadata)],
    )


def test_init_takes_no_unnamed_value_of_a_score_with_more_than_two_as_safe(
    three_valued_logs: Path, tmp_path: Path
) -> None:
    # Naming only I as a failure used to make P, and any other value, safe with a note.
    for failures in (["I"], ["I", "P"]):
        with pytest.raises(ValueError, match="--safe-values") as refused:
            scaffold_plan(three_valued_logs, tmp_path, failure_values=failures)
        unnamed = str(refused.value).rsplit(":", 1)[1]
        assert "'C'" in unnamed and "'I'" not in unnamed
        assert ("'P'" in unnamed) is (failures == ["I"])
    assert not (tmp_path / "safety-plan.yaml").exists()

    result = scaffold_plan(
        three_valued_logs, tmp_path, failure_values=["I", "P"], safe_values=["C"]
    )
    assert (result.failure_values, result.safe_values) == (["I", "P"], ["C"])
    assert not any("taken as safe" in note for note in result.notes)
    every = scaffold_plan(
        three_valued_logs, tmp_path, failure_values=["I", "P", "C"], overwrite=True
    )
    assert every.safe_values == []


def test_init_takes_the_other_value_of_a_two_valued_score_as_safe_and_says_so(
    mock_logs: Path, tmp_path: Path
) -> None:
    result = scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"])
    assert result.safe_values == ["PASS"]
    assert any("'PASS', the one --failure-values does not name" in note for note in result.notes)


def test_init_plan_analyzes_end_to_end(mock_logs: Path, tmp_path: Path) -> None:
    result = scaffold_plan(
        mock_logs,
        tmp_path,
        failure_values=["FAIL"],
        group_by="category",
        baseline="mockllm/alpha",
    )
    assert result.cohorts == {"alpha": "mockllm/alpha", "beta": "mockllm/beta"}
    assert result.reference == "alpha"
    assert result.safe_values == ["PASS"]
    assert (result.prompts, result.rollouts_per_prompt) == (5, 4)

    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    assert plan["analysis"]["intent"] == "exploratory"
    assert plan["design"]["hierarchy"] == ["category", "prompt_id"]
    assert plan["comparison_sets"][0]["contrasts"] == {
        "type": "versus_reference",
        "reference": "alpha",
    }
    assert "&id" not in result.plan_path.read_text(encoding="utf-8")

    report = analyze(plan=result.plan_path, logs=str(mock_logs))
    assert report.integrity.status == "valid"
    root = {
        risk.cohort_id: risk
        for risk in report.group_risks
        if risk.group_id == "all" and risk.measure_id == "mean"
    }
    assert set(root) == {"alpha", "beta"}
    for risk in root.values():
        assert risk.estimate == pytest.approx(0.1)
        assert risk.lower is not None and risk.upper is not None
        assert risk.lower <= 0.1 <= risk.upper
    levels = {item.level for item in report.hierarchy_risks}
    assert levels == {"category"}
    overall = [item for item in report.comparisons if item.level is None]
    assert {item.estimand for item in overall} == {"risk_difference", "risk_ratio"}


def test_init_without_baseline_picks_a_reference_and_says_so(
    mock_logs: Path, tmp_path: Path
) -> None:
    result = scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"])
    assert result.reference == "alpha"
    assert any("reference" in note for note in result.notes)
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    assert plan["design"]["hierarchy"] == ["prompt_id"]
    assert plan["outputs"]["direct_mean_intervals"]["hierarchy_nodes"] is False


def test_init_rejects_unknown_inputs(mock_logs: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="never occurs"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["UNSAFE"])
    with pytest.raises(ValueError, match="no field 'colour'"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], group_by="colour")
    with pytest.raises(ValueError, match="baseline 'gamma'"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], baseline="gamma")
    with pytest.raises(ValueError, match="no Inspect evaluation logs"):
        scaffold_plan(tmp_path, tmp_path / "out", failure_values=["FAIL"])


def test_init_refuses_two_runs_of_one_model(mock_logs: Path, tmp_path: Path) -> None:
    second = write_mock_logs(tmp_path / "again", models=("mockllm/alpha",), epochs=2)
    with pytest.raises(ValueError, match="more than one successful evaluation per model"):
        scaffold_plan([mock_logs, second], tmp_path / "out", failure_values=["FAIL"])


def test_init_command_writes_files_and_refuses_to_overwrite(
    mock_logs: Path, tmp_path: Path
) -> None:
    arguments = ["init", "--logs", str(mock_logs), "--failure-values", "FAIL",
                 "--output", str(tmp_path)]
    first = CliRunner().invoke(app, arguments)
    assert first.exit_code == 0, first.output
    assert "Cohort alpha: mockllm/alpha (reference)" in plain(first.output)
    assert (tmp_path / "design.csv").is_file()
    second = CliRunner().invoke(app, arguments)
    assert second.exit_code == 3
    assert "--overwrite" in plain(second.output)


def _header_text(plan_path: Path) -> str:
    """The plan's opening comment as one line of text, with its wrapping undone."""

    lines = plan_path.read_text(encoding="utf-8").splitlines()
    comment = [line.removeprefix("#").strip() for line in lines if line.startswith("#")]
    return " ".join(line for line in comment if line)


def test_init_notes_every_assumption_it_rests_on_independence_first(
    mock_logs: Path, tmp_path: Path
) -> None:
    result = scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"])
    ids = [note.assumption for note in result.assumptions]
    assert ids[0] == "rollout_independence"
    assert sorted(ids) == sorted(ASSUMPTIONS)
    header = _header_text(result.plan_path)
    for note in result.assumptions:
        entry = ASSUMPTIONS[note.assumption]
        assert f"- {note.assumption}: {note.choice}" in header
        assert f"Assumes: {entry.statement}" in header
        assert f"If wrong: {entry.if_wrong}" in header
        assert f"Remedy: {entry.remedy}" in header
    assert header.index("- rollout_independence:") < header.index("- arm_independence:")


def test_init_leaves_out_the_fields_its_notes_say_it_leaves_out(
    mock_logs: Path, tmp_path: Path
) -> None:
    result = scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"])
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    checked = []
    for note in result.assumptions:
        left_out = re.search(r"([a-z_.]+) is left out", note.choice)
        if left_out is None:
            continue
        *parents, leaf = left_out.group(1).split(".")
        node = plan
        for key in parents:
            node = node[key]
        assert leaf not in node, note
        checked.append(note.assumption)
        if "takes its default" in note.choice:
            assert note.choice.endswith(f", {ASSUMPTIONS[note.assumption].default}.")
    assert sorted(checked) == [
        "complete_design",
        "content_filter_stops",
        "judge_accuracy",
        "missing_scores",
        "sample_errors",
        "token_limit",
    ]


def test_init_for_one_model_has_no_comparison_note(tmp_path: Path) -> None:
    logs = write_mock_logs(tmp_path / "logs", models=("mockllm/alpha",), epochs=2)
    result = scaffold_plan(logs, tmp_path / "out", failure_values=["FAIL"])
    ids = [note.assumption for note in result.assumptions]
    assert ids[0] == "rollout_independence"
    assert sorted(ids) == sorted(set(ASSUMPTIONS) - {"arm_independence"})


def test_init_refuses_a_confirmatory_plan_and_points_to_exploratory(
    mock_logs: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="cannot be confirmatory") as refused:
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], intent="confirmatory")
    assert "--intent exploratory" in str(refused.value)
    assert ASSUMPTIONS["plan_fixed_in_advance"].remedy in str(refused.value)
    assert not (tmp_path / "safety-plan.yaml").exists()
    assert not (tmp_path / "design.csv").exists()
    with pytest.raises(ValueError, match="intent must be exploratory"):
        scaffold_plan(mock_logs, tmp_path, failure_values=["FAIL"], intent="pilot")


def test_init_command_lists_the_assumptions_independence_first(
    mock_logs: Path, tmp_path: Path
) -> None:
    arguments = ["init", "--logs", str(mock_logs), "--failure-values", "FAIL",
                 "--output", str(tmp_path)]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    output = plain(result.output)
    heading = "Assumptions (the plan's header says what each means and how to change it):"
    listed = output[output.index(heading) + len(heading):]
    positions = {name: listed.find(f" {name}: ") for name in ASSUMPTIONS}
    assert all(position >= 0 for position in positions.values()), positions
    assert min(positions, key=positions.__getitem__) == "rollout_independence"
    refused = CliRunner().invoke(app, [*arguments, "--overwrite", "--intent", "confirmatory"])
    assert refused.exit_code == 3
    assert "--intent exploratory" in plain(refused.output)
