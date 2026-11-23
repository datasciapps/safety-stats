"""`report.include_decisions` hides decisions, and every reader defaults it to false (#58).

The option hides: report.json keeps the decision and every claim's status. A report whose
presentation does not record the option is read as the plan's default, false, by the Python
model and by the HTML viewer alike.
"""

from __future__ import annotations

import json
from pathlib import Path

from safety_stats.report.summary import summary_markdown
from safety_stats.report.visualization import render_visualization
from safety_stats.schema.reports import STORED_REPORT, SafetyReport

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa" / "report"


def _report_without_the_option() -> SafetyReport:
    document = json.loads((EXAMPLE / "report.json").read_text(encoding="utf-8"))
    document["presentation"].pop("include_decisions")
    return SafetyReport.model_validate(document, context={STORED_REPORT: True})


def test_a_report_that_does_not_record_include_decisions_hides_decisions() -> None:
    report = _report_without_the_option()

    assert report.presentation.include_decisions is False
    assert "## Claims" not in summary_markdown(report)


def test_the_html_viewer_defaults_include_decisions_to_false() -> None:
    html = render_visualization(_report_without_the_option())

    assert "include_decisions ?? false" in html
    assert "include_decisions ?? true" not in html
