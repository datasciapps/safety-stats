from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from inspect_safety_stats import analyze
from inspect_safety_stats.scaffold import scaffold_plan
from tests.mock_inspect_logs import write_mock_logs


def _payload(html: str) -> dict[str, object]:
    match = re.search(
        r'<script id="inspect-safety-stats-data" type="application/json">(.*?)</script>',
        html,
        flags=re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


@pytest.mark.parametrize("graded", [False, True])
def test_html_carries_what_its_labels_need(tmp_path: Path, graded: bool) -> None:
    logs = write_mock_logs(tmp_path / "logs", graded=graded)
    result = scaffold_plan(
        logs, tmp_path / "plan",
        **({"harm_scale": (1.0, 5.0)} if graded else {"failure_values": ["FAIL"]}),  # type: ignore[arg-type]
    )
    plan = yaml.safe_load(result.plan_path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"].append(
        {"id": "worst_of_2", "type": "k_attempt_risk", "k": 2, "family": "attempts"}
    )
    plan["multiplicity"]["families"].append(
        {"id": "attempts", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": 0.05}
    )
    result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    written = analyze(plan=result.plan_path, logs=str(logs)).write(tmp_path / "report")
    html = (written / "report.html").read_text(encoding="utf-8")
    payload = _payload(html)
    resolved = payload["resolved_plan"]
    assert isinstance(resolved, dict)
    measures = {item["id"]: item for item in resolved["outputs"]["risk_measures"]}
    assert measures["worst_of_2"]["type"] == "k_attempt_risk"
    assert bool(resolved["inputs"]["score"]["mapping"].get("scale")) is graded
    assert "function measureLabel" in html
    risks = payload["report"]["group_risks"]  # type: ignore[index]
    attempt = next(item for item in risks if item["measure_id"] == "worst_of_2")
    assert "k_attempt" in attempt["coverage_scope"]
