from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


@pytest.fixture(scope="module")
def graded_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("graded") / "logs", graded=True)


def test_init_refuses_values_outside_the_scale(graded_logs: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        scaffold_plan(graded_logs, tmp_path, harm_scale=(2, 5))
    with pytest.raises(ValueError, match="on its own"):
        scaffold_plan(graded_logs, tmp_path, harm_scale=(1, 5), failure_values=["4"])
    with pytest.raises(ValueError, match="--harm-scale"):
        scaffold_plan(graded_logs, tmp_path)


def test_graded_scores_are_analyzed_as_mean_harm(graded_logs: Path, tmp_path: Path) -> None:
    result = scaffold_plan(
        graded_logs, tmp_path / "plan", harm_scale=(1, 5), group_by="category",
        baseline="mockllm/alpha",
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    assert plan["inputs"]["score"]["mapping"] == {
        "scale": {"minimum": 1, "maximum": 5, "harmful": "high"}
    }
    report = analyze(plan=result.plan_path, logs=str(graded_logs))
    assert report.integrity.status == "valid"

    root = {
        risk.cohort_id: risk
        for risk in report.group_risks
        if risk.group_id == "all" and risk.measure_id == "mean"
    }
    for risk in root.values():
        assert risk.estimate == pytest.approx(0.1)
        assert risk.lower is not None and risk.upper is not None
        assert risk.lower < 0.1 < risk.upper

    p0 = next(
        item for item in report.prompt_estimates
        if item.cohort_id == "alpha" and item.prompt_id == "p0"
    )
    assert p0.p_hat == pytest.approx(0.5)
    assert p0.method == "bernoulli_kl_bounded_mean"
    assert p0.lower is not None and p0.upper is not None
    assert p0.lower < 0.5 < p0.upper
    assert p0.n_failure == 4

    difference = next(
        item for item in report.comparisons
        if item.level is None and item.estimand == "risk_difference"
    )
    assert difference.estimate == pytest.approx(0.0)
    assert difference.lower is not None and difference.upper is not None
    assert difference.lower < 0 < difference.upper

    summary = (report.write(tmp_path / "report") / "report.md").read_text(encoding="utf-8")
    assert "## Mean harm" in summary
    assert "| alpha | all | 10.0% |" in summary


def test_lower_is_harmful_reverses_the_scale(graded_logs: Path, tmp_path: Path) -> None:
    result = scaffold_plan(
        graded_logs, tmp_path, harm_scale=(1, 5), lower_is_harmful=True,
    )
    report = analyze(plan=result.plan_path, logs=str(graded_logs))
    root = next(
        risk for risk in report.group_risks
        if risk.cohort_id == "alpha" and risk.measure_id == "mean"
    )
    assert root.estimate == pytest.approx(0.9)
