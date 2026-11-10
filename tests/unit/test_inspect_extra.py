"""Without the inspect extra, rollout files are analysed and Inspect logs fail with a clear error.

Each check runs in a fresh interpreter in which ``inspect_ai`` and ``pyarrow`` cannot be
imported, as in an installation without the extra.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BLOCK = "import sys\nsys.modules['inspect_ai'] = None\nsys.modules['pyarrow'] = None\n"

PLAN = """
schema_version: "3.0"
analysis: {id: extra, title: Without the inspect extra, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: m}}
groups:
  - id: all
outputs:
  prompt_bounds: {enabled: true, family: displayed}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
"""


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", BLOCK + code], capture_output=True, text=True, cwd=cwd
    )


def test_reading_inspect_logs_without_the_extra_names_it(tmp_path: Path) -> None:
    completed = _run("import inspect_safety_stats.ingest.inspect_logs", tmp_path)
    assert completed.returncode != 0
    assert "pip install 'inspect-safety-stats[inspect]'" in completed.stderr


def test_rollout_files_are_analysed_without_the_extra(tmp_path: Path) -> None:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "plan.yaml").write_text(PLAN, encoding="utf-8", newline="\n")
    (tmp_path / "runs.jsonl").write_text(
        json.dumps({"run_key": "r1", "model": "m", "status": "success"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "rollouts.jsonl").write_text(
        "".join(
            json.dumps(
                {"run_key": "r1", "item_id": "p1", "repeat": repeat, "score_harmful": "PASS"}
            )
            + "\n"
            for repeat in (1, 2)
        ),
        encoding="utf-8",
        newline="\n",
    )
    completed = _run(
        "from inspect_safety_stats import analyze\n"
        "from inspect_safety_stats.ingest import RolloutFiles\n"
        "report = analyze(plan='plan.yaml',\n"
        "                 source=RolloutFiles(runs='runs.jsonl', rollouts='rollouts.jsonl'))\n"
        "print(type(report).__name__, sys.modules['inspect_ai'] is not None)",
        tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["SafetyReport", "False"]
