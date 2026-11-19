"""examples/judge_flip/flip_manifest.py exports the completions the analysis rates (#122).

The analysis rates only the failures the primary judge scored. The export must pick the same
rows of observations.csv, or a census of its output is rejected. The observations here are
written by the package's own CSV writer, and the run's logs are faked.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inspect_safety_stats.ingest.outcomes import judge_positive
from inspect_safety_stats.report.artifacts import write_csv
from inspect_safety_stats.schema.observations import Observation

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "judge_flip"
sys.path.insert(0, str(EXAMPLE_DIR))

import flip_manifest  # noqa: E402

# Each observation's classification, by epoch.
CLASSIFIED: dict[int, dict[str, Any]] = {
    1: {"score_raw": "FAIL", "failure": True, "token_limit_unscored": False},
    2: {"score_raw": "PASS", "failure": False, "token_limit_unscored": False},
    # A sample error and a missing score under count_as_failure, a content-filter stop counted
    # as a refusal, an unscored truncated answer as a failure. Each row but the missing score's
    # carries a score, so that only its own mark, read back from its column, leaves it out. A
    # truncated answer counted by token_limit never has a score; its row is made up.
    3: {"score_raw": "FAIL", "failure": True, "error": "the provider failed"},
    4: {"score_raw": None, "failure": True},
    5: {"score_raw": "PASS", "failure": True, "content_delivered": False},
    6: {"score_raw": "FAIL", "failure": True, "token_limit_unscored": True},
    7: {"score_raw": "FAIL", "failure": True, "token_limit_unscored": False},
}


def _observation(epoch: int) -> Observation:
    return Observation.model_validate(
        {
            "observation_id": f"obs-{epoch}",
            "source": "run.eval",
            "run_key": "eval-1",
            "rollout_id": f"sample-{epoch}",
            "item_id": "p1",
            "repeat": epoch,
            "prompt_id": "p1",
            "question_id": "q1",
            "task_id": "t1",
            "cohort_id": "candidate",
            "score_name": "harmful",
            "model": "model/candidate",
            "configuration_fingerprint": "f",
            "collection_order": epoch,
            **CLASSIFIED[epoch],
        }
    )


def test_the_export_leaves_out_failures_a_plan_policy_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observations = [_observation(epoch) for epoch in sorted(CLASSIFIED)]
    report = tmp_path / "report"
    report.mkdir()
    write_csv(report / "observations.csv", [row.model_dump(mode="json") for row in observations])
    log = SimpleNamespace(
        eval=SimpleNamespace(eval_id="eval-1"),
        samples=[
            SimpleNamespace(
                input="the request",
                id="p1",
                epoch=epoch,
                output=SimpleNamespace(completion=f"completion {epoch}"),
            )
            for epoch in sorted(CLASSIFIED)
        ],
    )
    monkeypatch.setattr(flip_manifest, "list_eval_logs", lambda logs: ["run.eval"])
    monkeypatch.setattr(flip_manifest, "read_eval_log", lambda info: log)

    count = flip_manifest.export_positives(report, tmp_path / "logs", tmp_path / "positives.csv")

    with (tmp_path / "positives.csv").open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert count == 2
    assert [row["observation_id"] for row in exported] == ["obs-1", "obs-7"]
    assert [row["completion"] for row in exported] == ["completion 1", "completion 7"]
    # The simulated rescoring is seeded by the rollout's item and epoch.
    assert [(row["item_id"], row["repeat"]) for row in exported] == [("p1", "1"), ("p1", "7")]
    # The same rows the analysis rates.
    assert {row["observation_id"] for row in exported} == {
        row.observation_id for row in observations if judge_positive(row)
    }
