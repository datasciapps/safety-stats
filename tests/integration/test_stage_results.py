"""The stages that adjust risk rows return new rows and leave the ones they were given alone.

A stage that copied each row but shared what the rows hold, such as their interval lists, would
change the rows it was given whenever it adds to them. So the check runs on analyses in which
every stage adjusts every kind of row it can adjust, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.analysis import core
from inspect_safety_stats.analysis.results import RiskRows
from tests.diagnostic_codes.harness import Inputs
from tests.diagnostic_codes.judge_analysis_cases import (
    census,
    comparison,
    composite_primary,
    judged,
    with_labels,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa"

PARTS = ("risks", "hierarchy_risks", "comparisons")
# The stages in the order the core runs them, by the name the core calls them under, with the
# kinds of row each one adjusts. Comparisons project their own regions.
ADJUSTS = {
    "project_hierarchy_regions": ("risks", "hierarchy_risks"),
    "apply_judge_correction": ("risks", "hierarchy_risks", "comparisons"),
    "apply_panel_derived_correction": ("risks", "hierarchy_risks"),
}


@dataclass(frozen=True)
class Call:
    """One call of a stage: the rows it was given, before and after it ran, and its result."""

    before: dict[str, list[Any]]
    after: dict[str, list[Any]]
    returned: dict[str, list[Any]]


def _dump(rows: RiskRows) -> dict[str, list[Any]]:
    return {part: [row.model_dump(mode="json") for row in getattr(rows, part)] for part in PARTS}


def _rows_of(result: Any) -> RiskRows:
    return result if isinstance(result, RiskRows) else result.rows


def _spy_on_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Call]]:
    """Record every call of the adjusting stages, each of which must return new row objects."""

    calls: dict[str, list[Call]] = {name: [] for name in ADJUSTS}

    def spy(name: str) -> Any:
        stage = getattr(core, name)

        def wrapped(*args: Any) -> Any:
            given = next(arg for arg in args if isinstance(arg, RiskRows))
            before = _dump(given)
            result = stage(*args)
            returned = _rows_of(result)
            assert returned is not given
            for part in PARTS:
                given_ids = {id(row) for row in getattr(given, part)}
                assert not {id(row) for row in getattr(returned, part)} & given_ids
            calls[name].append(Call(before, _dump(given), _dump(returned)))
            return result

        return wrapped

    for name in ADJUSTS:
        monkeypatch.setattr(core, name, spy(name))
    return calls


def _assert_given_rows_unchanged(calls: dict[str, list[Call]]) -> None:
    for name, made in calls.items():
        assert made, f"{name} did not run"
        for call in made:
            assert call.after == call.before, f"{name} changed the rows it was given"


def test_adjusting_stages_return_new_rows_and_leave_their_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_on_stages(monkeypatch)

    report = analyze(plan=EXAMPLE / "safety-plan.yaml", logs=str(EXAMPLE / "simpleqa-8-real.eval"))

    _assert_given_rows_unchanged(calls)
    # The projection changes the rows it returns, so the check above is not vacuous for it.
    (projection,) = calls["project_hierarchy_regions"]
    assert projection.returned != projection.before
    assert any(
        row.projection_method == "hierarchy_linear_program" for row in report.hierarchy_risks
    )


def _judged_panel(inputs: Inputs) -> None:
    """Two cohorts and their comparison, hierarchy nodes projected onto a common region, and a
    composite of two judges with declared accuracies as the primary outcome."""

    comparison("weighted_kl_chernoff")(inputs)
    inputs.plan["design"]["hierarchy"] = ["task_id", "question_id", "prompt_id"]
    inputs.plan["outputs"]["direct_mean_intervals"] = {
        "enabled": True,
        "method": "weighted_kl_chernoff",
        "family": "direct",
        "groups": ["all"],
        "simultaneous": True,
        "coverage_mode": "two_sided",
        "independence": {"primary": "rollout"},
        "hierarchy_nodes": True,
        "hierarchy_region": "linear_program",
    }
    inputs.plan["multiplicity"]["families"].append(
        {"id": "direct", "purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}
    )
    composite_primary("any", None, None)(inputs)


def test_judge_and_panel_corrections_leave_the_rows_they_were_given_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _spy_on_stages(monkeypatch)

    # A validated calibration on labels for every classified rollout. judged() reads the
    # rollouts' identifiers from a first analysis of the same inputs without it.
    outcome = judged(with_labels(census), _judged_panel)(tmp_path)

    _assert_given_rows_unchanged(calls)
    # In the calibrated analysis every stage changes every kind of row it adjusts, so the check
    # above is not vacuous for any of them.
    for name, parts in ADJUSTS.items():
        _first, calibrated = calls[name]
        for part in parts:
            assert calibrated.returned[part] != calibrated.before[part], (
                f"{name} returned its {part} as they were"
            )
    report = outcome.report
    (versus,) = [row for row in report.comparisons if row.level is None]
    assert "judge_corrected_validated" in {item.interval_id for item in versus.intervals}
    assert report.hierarchy_risks
    for row in report.hierarchy_risks:
        assert row.projection_method == "hierarchy_linear_program"
        added = {item.interval_id for item in row.intervals}
        assert {"judge_corrected_validated", "judge_corrected_panel_derived"} <= added
