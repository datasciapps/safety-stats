"""report.html's page script runs to its end, so every section and the scope footer render.

The page script builds the whole page on load. An uncaught error in one section leaves every
later one empty: a reference to the resolved plan, which the page no longer embeds, once left
the audit trail without its plan panel and the page without its scope footer. These tests run
the whole script under Node (``tests/page_script.py``) and are skipped without Node 18 or newer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.report.visualization import render_visualization
from inspect_safety_stats.schema.reports import STORED_REPORT, SafetyReport
from tests.page_script import PageRun, node_or_skip, run_page

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_REPORTS = {
    "simpleqa": ROOT / "examples" / "real_simpleqa" / "report" / "report.json",
    "xstest": ROOT / "examples" / "real_xstest" / "report-current" / "report.json",
}
SCOPE_CAVEAT = (
    "Values are fixed-design summaries; hierarchy filters do not create population-level claims."
)


def _assert_the_page_rendered(run: PageRun, *, decisions: bool) -> None:
    assert run.error is None, run.error
    assert run.event_errors == []
    assert SCOPE_CAVEAT in run.footer
    titles = [panel.title for panel in run.audit]
    assert titles[0].startswith("Integrity diagnostics (")
    assert any(title.startswith("Policy events (") for title in titles) is decisions
    assert titles[-2:] == ["Provenance", "Resolved plan"]
    # The page does not embed the plan; the panel says where the plan is.
    assert "plan.resolved.yaml" in run.audit[-1].text


@pytest.mark.parametrize("decisions", [False, True], ids=["decisions-hidden", "decisions-shown"])
@pytest.mark.parametrize("name", sorted(EXAMPLE_REPORTS))
def test_the_page_script_of_an_example_report_runs_to_its_end(
    name: str, decisions: bool, tmp_path: Path
) -> None:
    node = node_or_skip()
    report = SafetyReport.model_validate_json(
        EXAMPLE_REPORTS[name].read_text(encoding="utf-8"), context={STORED_REPORT: True}
    )
    report.presentation.include_decisions = decisions

    run = run_page(node, render_visualization(report), tmp_path)

    _assert_the_page_rendered(run, decisions=decisions)


def test_the_page_script_of_a_report_without_results_runs_to_its_end(tmp_path: Path) -> None:
    node = node_or_skip()
    report = analyze(
        plan=ROOT / "examples" / "real_simpleqa" / "safety-plan.yaml",
        logs=str(tmp_path / "missing.eval"),
    )
    assert [item.code for item in report.integrity.diagnostics] == ["INGEST_FAILED"]

    run = run_page(node, render_visualization(report), tmp_path)

    _assert_the_page_rendered(run, decisions=False)
