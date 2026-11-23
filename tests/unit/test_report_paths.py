"""How a report names an input path (#63).

Relative to the plan's directory and in POSIX form, so the same analysis writes the same bytes
in any checkout, and a shared report does not reveal the local directory layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from safety_stats.config.loader import report_path
from safety_stats.config.models import AnalysisPlan


def _plan(source: Path | None) -> AnalysisPlan:
    # report_path asks the plan only where it was loaded from, resolved as load_plan does.
    resolved = source.resolve() if source is not None else None
    return cast(AnalysisPlan, SimpleNamespace(source_path=lambda: resolved))


def test_a_path_inside_the_plan_directory_is_relative(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "plan.yaml")
    assert report_path(tmp_path / "logs" / "run.eval", plan) == "logs/run.eval"
    assert report_path(str(tmp_path / "design.csv"), plan) == "design.csv"


def test_a_path_beside_the_plan_directory_climbs_to_it(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "evaluation" / "plan.yaml")
    assert report_path(tmp_path / "logs" / "run.eval", plan) == "../logs/run.eval"


def test_a_uri_keeps_its_form(tmp_path: Path) -> None:
    plan = _plan(tmp_path / "plan.yaml")
    assert report_path("s3://bucket/logs/run.eval", plan) == "s3://bucket/logs/run.eval"


def test_a_plan_without_a_file_keeps_the_path_with_forward_slashes() -> None:
    assert report_path(Path("logs") / "run.eval", _plan(None)) == "logs/run.eval"


@pytest.mark.skipif(os.name != "nt", reason="only Windows paths can lack a relative form")
def test_a_path_on_another_drive_stays_absolute(tmp_path: Path) -> None:
    drive = "Y:" if str(tmp_path).upper().startswith("Z:") else "Z:"
    plan = _plan(tmp_path / "plan.yaml")
    assert report_path(f"{drive}\\logs\\run.eval", plan) == f"{drive}/logs/run.eval"
