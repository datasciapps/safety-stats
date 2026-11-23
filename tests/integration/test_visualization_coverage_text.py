"""The HTML coverage line must describe the interval it summarizes.

The template computes this text in JavaScript, so the test runs the template's own functions
under Node against a real report. Without Node the test is skipped.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from safety_stats import analyze
from safety_stats.report import visualization
from tests.template_functions import function_source

SIMPLEQA = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa"

_FUNCTIONS = ("humanize", "pct", "selectedInterval", "hasProfileIntervals", "coverageSummary")

# For every group risk and every independence-profile selection the page offers, evaluate the
# same expressions renderRisk() uses to pick the row whose coverage it describes.
_DRIVER = """
const fs = require("fs");
const [functionsPath, reportPath] = process.argv.slice(2);
const state = { independenceProfile: null };
const api = new Function(
  "state",
  fs.readFileSync(functionsPath, "utf8")
    + "\\nreturn {selectedInterval, hasProfileIntervals, coverageSummary};"
)(state);
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
const out = [];
for (const row of report.group_risks) {
  const profiles = [...new Set((row.intervals || []).map(i => i.independence_profile))];
  for (const profile of profiles) {
    state.independenceProfile = profile;
    const coverageRow = api.selectedInterval(row) || (api.hasProfileIntervals(row) ? null : row);
    out.push({
      measure: row.measure_id,
      profile,
      method: coverageRow ? (coverageRow.interval_method || coverageRow.method) : null,
      assumptions: coverageRow ? (coverageRow.assumptions || []) : [],
      text: api.coverageSummary(coverageRow),
    });
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _humanize(value: str) -> str:
    return re.sub(r"\b\w", lambda match: match.group(0).upper(), re.sub(r"[_-]+", " ", value))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_coverage_line_states_the_selected_intervals_method_and_assumptions(
    tmp_path: Path,
) -> None:
    report = analyze(
        plan=SIMPLEQA / "safety-plan.yaml", logs=str(SIMPLEQA / "simpleqa-8-real.eval")
    )
    template = visualization._TEMPLATE.read_text(encoding="utf-8")
    functions = tmp_path / "functions.js"
    functions.write_text(
        "\n".join(function_source(template, name) for name in _FUNCTIONS), encoding="utf-8"
    )
    payload = tmp_path / "report.json"
    payload.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")
    driver = tmp_path / "driver.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    node = shutil.which("node")
    assert node is not None
    completed = subprocess.run(
        [node, str(driver), str(functions), str(payload)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    cases = json.loads(completed.stdout)

    direct = [
        case
        for case in cases
        if case["method"] and case["method"] != "propagated_exact_prompt_bounds"
    ]
    assert direct, "the example must exercise at least one direct interval"
    for case in cases:
        assert case["method"] is not None, case
        assert f"· {_humanize(case['method'])}." in case["text"], case
        assert case["assumptions"], case
        for assumption in case["assumptions"]:
            assert assumption in case["text"], (case["measure"], case["profile"], assumption)
    for case in direct:
        # A direct bound assumes independent blocks; it must never be described otherwise.
        assert "does not require independence" not in case["text"], case
        assert "No independence between prompts is required" not in case["text"], case
