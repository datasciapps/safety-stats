from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.cli import app
from inspect_safety_stats.scaffold import scaffold_plan
from inspect_safety_stats.tradeoff import combine_tradeoff, write_tradeoff
from tests.mock_inspect_logs import write_profile_logs

CAREFUL = "mockllm/careful"
EAGER = "mockllm/eager"


def _report(
    root: Path, name: str, failing: dict[str, frozenset[str]], baseline: str | None = CAREFUL
) -> Path:
    logs = write_profile_logs(root / name / "logs", failing, name=name)
    plan = scaffold_plan(
        logs, root / name / "plan", failure_values=["FAIL"], baseline=baseline
    ).plan_path
    return analyze(plan=plan, logs=str(logs)).write(root / name / "report")


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("tradeoff")
    worst = frozenset({f"p{index}" for index in range(6)})
    # The careful model never complies with harmful prompts but refuses six benign ones;
    # the eager model complies with six harmful prompts and refuses no benign one.
    harm = _report(root, "harm", {CAREFUL: frozenset(), EAGER: worst})
    refusal = _report(root, "refusal", {CAREFUL: worst, EAGER: frozenset()})
    return harm, refusal


def test_tradeoff_reads_a_resolved_trade_off(reports: tuple[Path, Path]) -> None:
    harm, refusal = reports
    result = combine_tradeoff(harm, refusal)
    by_model = {row.model: row for row in result.models}
    assert by_model[CAREFUL].harm.estimate == pytest.approx(0.0)
    assert by_model[CAREFUL].over_refusal.estimate == pytest.approx(0.6)
    assert by_model[EAGER].harm.estimate == pytest.approx(0.6)
    assert by_model[EAGER].over_refusal.estimate == pytest.approx(0.0)

    (pair,) = result.pairs
    assert (pair.candidate, pair.baseline) == (EAGER, CAREFUL)
    assert pair.harm.estimate == pytest.approx(0.6)
    assert pair.over_refusal.estimate == pytest.approx(-0.6)
    assert pair.reading == f"trade-off: {EAGER} is more harmful but refuses less"


def test_joint_coverage_is_the_union_bound(reports: tuple[Path, Path]) -> None:
    result = combine_tradeoff(*reports)
    # Each report spends 0.05 on its rate family and 0.05 on its comparison family. Both
    # plans name their families identically; they are still separate budgets.
    assert result.rate_coverage == pytest.approx(0.90)
    assert result.difference_coverage == pytest.approx(0.90)
    assert result.overall_coverage == pytest.approx(0.80)


def test_outputs_are_written(reports: tuple[Path, Path], tmp_path: Path) -> None:
    result = combine_tradeoff(*reports)
    directory = write_tradeoff(result, tmp_path)
    markdown = (directory / "tradeoff.md").read_text(encoding="utf-8")
    assert "| mockllm/careful | 0.0% |" in markdown
    assert "more harmful but refuses less" in markdown
    assert "at least 80.0%" in markdown
    data = json.loads((directory / "tradeoff.json").read_text(encoding="utf-8"))
    assert data["overall_coverage"] == pytest.approx(0.8)
    html = (directory / "tradeoff.html").read_text(encoding="utf-8")
    assert "<svg" in html and "mockllm/eager" in html
    with pytest.raises(FileExistsError):
        write_tradeoff(result, tmp_path)


def test_reports_without_a_shared_model_are_refused(tmp_path: Path) -> None:
    harm = _report(
        tmp_path, "a", {"mockllm/one": frozenset({"p0"}), "mockllm/two": frozenset()}, baseline=None
    )
    refusal = _report(
        tmp_path, "b", {"mockllm/three": frozenset({"p0"}), "mockllm/four": frozenset()},
        baseline=None,
    )
    with pytest.raises(ValueError, match="share no model"):
        combine_tradeoff(harm, refusal)


def test_tradeoff_command(reports: tuple[Path, Path], tmp_path: Path) -> None:
    harm, refusal = reports
    result = CliRunner().invoke(
        app,
        ["tradeoff", "--harm", str(harm), "--over-refusal", str(refusal),
         "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "mockllm/eager vs mockllm/careful: trade-off" in result.output
    assert (tmp_path / "out" / "tradeoff.html").is_file()
