"""Every report links to the page on what results establish and states the sandbagging condition.

A report is read away from the documentation, so its scope carries an absolute link to
docs/what-results-establish.md, which report.md and report.html both show. Every bound assumes
that the evaluated model is not strategically underperforming (#12), whatever the design.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from inspect_safety_stats.analysis.report import RESULTS_ESTABLISH_URL
from inspect_safety_stats.report.summary import summary_markdown
from inspect_safety_stats.report.writer import _markdown
from inspect_safety_stats.schema.reports import IntegritySummary, SafetyReport

ROOT = Path(__file__).resolve().parents[2]
SANDBAGGING = "not strategically underperforming"


def _report(scope: dict[str, str]) -> SafetyReport:
    return SafetyReport(
        package_version="0",
        analysis={"id": "a"},
        provenance={},
        scope=scope,
        integrity=IntegritySummary(status="valid"),
        design={},
        cohorts=[],
        prompt_estimates=[],
        group_risks=[],
        comparisons=[],
        multiplicity=[],
        claims=[],
        policy_events=[],
        decision="INCONCLUSIVE",
    )


def test_the_link_names_a_tracked_page_on_main() -> None:
    prefix = "https://github.com/aduncan001/safety_stats/blob/main/"
    assert RESULTS_ESTABLISH_URL.startswith(prefix)
    page = RESULTS_ESTABLISH_URL.removeprefix(prefix)
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--", page],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git checkout
        # As in test_repository_consistency: a copy of the tree without .git has no index.
        pytest.skip("the tracked-page check requires a git checkout")
    assert tracked == [page], f"{page} is not a tracked file"


@pytest.mark.parametrize(
    "report",
    ["examples/real_simpleqa/report", "examples/real_xstest/report-current"],
)
def test_a_fixed_design_report_links_the_page_and_states_the_condition(report: str) -> None:
    payload = json.loads((ROOT / report / "report.json").read_text(encoding="utf-8"))
    assert payload["scope"]["design"] == "fixed"
    assert payload["scope"]["what_results_establish"] == RESULTS_ESTABLISH_URL
    assert SANDBAGGING in payload["scope"]["disclaimer"]
    markdown = (ROOT / report / "report.md").read_text(encoding="utf-8")
    assert f"<{RESULTS_ESTABLISH_URL}>" in markdown
    assert SANDBAGGING in markdown


def test_report_md_shows_the_link_under_scope() -> None:
    markdown = _markdown(
        _report({"disclaimer": "Fixed design.", "what_results_establish": RESULTS_ESTABLISH_URL})
    )
    scope = markdown.split("## Scope", 1)[1].split("\n## ", 1)[0]
    assert f"What these results establish, and what they do not: <{RESULTS_ESTABLISH_URL}>" in (
        scope
    )


def test_the_summary_shows_the_link_under_scope() -> None:
    markdown = summary_markdown(
        _report({"disclaimer": "Fixed design.", "what_results_establish": RESULTS_ESTABLISH_URL})
    )
    scope = markdown.split("## Scope", 1)[1].split("\n## ", 1)[0]
    assert f"What these results establish, and what they do not: <{RESULTS_ESTABLISH_URL}>" in (
        scope
    )


def test_a_report_written_before_the_link_renders_without_it() -> None:
    markdown = _markdown(_report({"disclaimer": "Fixed design."}))
    assert "What these results establish" not in markdown
    summary = summary_markdown(_report({"disclaimer": "Fixed design."}))
    assert "What these results establish" not in summary


def test_report_html_shows_the_link() -> None:
    template = (
        ROOT / "src/inspect_safety_stats/report/visualization_template.html"
    ).read_text(encoding="utf-8")
    assert "report.scope?.what_results_establish" in template
    assert "link.href = report.scope.what_results_establish" in template
