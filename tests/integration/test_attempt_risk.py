from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.api import validate_plan_references
from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


@pytest.fixture(scope="module")
def binary_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("attempts") / "logs")


@pytest.fixture(scope="module")
def graded_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("attempts-graded") / "logs", graded=True)


def _plan(logs: Path, directory: Path, *, k: int, graded: bool = False, **extra: object) -> Path:
    result = scaffold_plan(
        logs, directory,
        **({"harm_scale": (1.0, 5.0)} if graded else {"failure_values": ["FAIL"]}),  # type: ignore[arg-type]
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"].append(
        {"id": f"worst_of_{k}", "type": "k_attempt_risk", "k": k, "family": "attempts", **extra}
    )
    plan["multiplicity"]["families"].append(
        {"id": "attempts", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": 0.05}
    )
    result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return result.plan_path


def _risk(report: object, cohort: str, measure: str) -> object:
    return next(
        item for item in report.group_risks  # type: ignore[attr-defined]
        if item.cohort_id == cohort and item.measure_id == measure
    )


def test_worst_of_k_risk_is_estimated_without_bias_and_bounded(
    binary_logs: Path, tmp_path: Path
) -> None:
    report = analyze(plan=_plan(binary_logs, tmp_path, k=2), logs=str(binary_logs))
    assert report.integrity.status == "valid"
    risk = _risk(report, "alpha", "worst_of_2")
    # p0 failed on 2 of 4 rollouts: 1 - C(2,2)/C(4,2) = 5/6; the other four prompts never.
    assert risk.estimate == pytest.approx((5 / 6) / 5)  # type: ignore[attr-defined]
    primary = next(item for item in risk.intervals if item.role == "primary")  # type: ignore[attr-defined]
    assert primary.interval_id == "attempt_weighted_kl_chernoff"
    assert primary.family == "attempts"
    assert primary.alpha_upper == pytest.approx(0.05 / 2 / (2 * 1 * 1))
    assert primary.lower <= risk.estimate <= primary.upper  # type: ignore[attr-defined]
    sensitivity = next(item for item in risk.intervals if item.role == "sensitivity")  # type: ignore[attr-defined]
    assert sensitivity.independence_profile == "prompt"
    assert sensitivity.upper >= primary.upper
    propagated = next(
        item for item in risk.intervals  # type: ignore[attr-defined]
        if item.interval_id == "propagated_prompt_envelope"
    )
    assert propagated.role == "reference"

    summary = (report.write(tmp_path / "report") / "report.md").read_text(encoding="utf-8")
    assert "## Risk over repeated attempts" in summary
    assert "| alpha | all | 2 | 16.7% |" in summary
    assert "## Tail and threshold measures" not in summary


def test_adding_an_attempt_measure_leaves_other_intervals_unchanged(
    binary_logs: Path, tmp_path: Path
) -> None:
    with_attempts = analyze(plan=_plan(binary_logs, tmp_path / "a", k=2), logs=str(binary_logs))
    result = scaffold_plan(binary_logs, tmp_path / "b", failure_values=["FAIL"])
    without = analyze(plan=result.plan_path, logs=str(binary_logs))
    left = _risk(with_attempts, "alpha", "mean")
    right = _risk(without, "alpha", "mean")
    assert (left.lower, left.upper) == (right.lower, right.upper)  # type: ignore[attr-defined]


def test_too_few_rollouts_leave_the_measure_unavailable(
    binary_logs: Path, tmp_path: Path
) -> None:
    report = analyze(plan=_plan(binary_logs, tmp_path, k=8), logs=str(binary_logs))
    risk = _risk(report, "alpha", "worst_of_8")
    assert risk.estimate is None  # type: ignore[attr-defined]
    assert any(
        item.code == "ATTEMPT_RISK_INTERVAL_INVALID" for item in report.integrity.diagnostics
    )


def test_graded_scores_give_the_expected_worst_harm(graded_logs: Path, tmp_path: Path) -> None:
    report = analyze(plan=_plan(graded_logs, tmp_path, k=2, graded=True), logs=str(graded_logs))
    risk = _risk(report, "alpha", "worst_of_2")
    # p0 scores 0.75, 0.25, 0.75, 0.25: the mean of pairwise maxima is 4/6.
    assert risk.estimate == pytest.approx((4 / 6) / 5)  # type: ignore[attr-defined]
    assert risk.lower <= risk.estimate <= risk.upper  # type: ignore[attr-defined]


def test_attempt_families_must_be_their_own(binary_logs: Path, tmp_path: Path) -> None:
    path = _plan(binary_logs, tmp_path, k=2)
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"][-1]["family"] = "risk"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    codes = {item.code for item in validate_plan_references(load_plan(path))}
    assert "PLAN_INVALID_MULTIPLICITY" in codes


def test_attempt_measures_need_k_and_a_family() -> None:
    from pydantic import ValidationError

    from inspect_safety_stats.config.models import RiskMeasureSpec

    with pytest.raises(ValidationError, match="k >= 1"):
        RiskMeasureSpec.model_validate({"id": "a", "type": "k_attempt_risk", "family": "f"})
    with pytest.raises(ValidationError, match="multiplicity family"):
        RiskMeasureSpec.model_validate({"id": "a", "type": "k_attempt_risk", "k": 2})
    with pytest.raises(ValidationError, match="only k_attempt_risk"):
        RiskMeasureSpec.model_validate({"id": "a", "type": "weighted_mean", "family": "f"})
