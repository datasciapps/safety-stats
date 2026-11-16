"""The stages that adjust risk rows return new rows and leave the ones they were given alone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.analysis import core
from inspect_safety_stats.analysis.results import RiskRows

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa"

# The stages in the order the core runs them, by the name the core calls them under.
STAGES = (
    "project_hierarchy_regions",
    "apply_judge_correction",
    "apply_panel_derived_correction",
)


def _dump(rows: RiskRows) -> list[Any]:
    return [
        [row.model_dump(mode="json") for row in part]
        for part in (rows.risks, rows.hierarchy_risks, rows.comparisons)
    ]


def _rows_of(result: Any) -> RiskRows:
    return result if isinstance(result, RiskRows) else result.rows


def test_adjusting_stages_return_new_rows_and_leave_their_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[list[Any], list[Any], list[Any]]] = {}

    def spy(name: str) -> Any:
        stage = getattr(core, name)

        def wrapped(*args: Any) -> Any:
            given = next(arg for arg in args if isinstance(arg, RiskRows))
            before = _dump(given)
            result = stage(*args)
            returned = _rows_of(result)
            assert returned is not given
            assert not {id(row) for row in returned.risks} & {id(row) for row in given.risks}
            calls[name] = (before, _dump(given), _dump(returned))
            return result

        return wrapped

    for name in STAGES:
        monkeypatch.setattr(core, name, spy(name))

    report = analyze(plan=EXAMPLE / "safety-plan.yaml", logs=str(EXAMPLE / "simpleqa-8-real.eval"))

    assert sorted(calls) == sorted(STAGES)
    for name, (before, after, _) in calls.items():
        assert after == before, f"{name} changed the rows it was given"
    # The projection changes the rows it returns, so the check above is not vacuous.
    before, _, returned = calls["project_hierarchy_regions"]
    assert returned != before
    assert any(
        row.projection_method == "hierarchy_linear_program" for row in report.hierarchy_risks
    )
