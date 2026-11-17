"""Plan schema 3.0 names run and rollout fields neutrally; schema 2.0 plans keep their meaning."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from inspect_safety_stats import validate
from inspect_safety_stats.config.loader import canonical_plan_hash, canonical_plan_value
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.rollout_files import RolloutFiles

BASE: dict[str, Any] = {
    "schema_version": "2.0",
    "analysis": {"id": "schema", "title": "Schema versions", "intent": "exploratory"},
    "inputs": {
        "design_manifest": "design.csv",
        "score": {
            "name": "harmful",
            "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
        },
    },
    "cohorts": {"candidate": {"match": {"model": "m"}}},
    "groups": [{"id": "all"}],
    "outputs": {
        "prompt_bounds": {"enabled": True, "family": "displayed"},
        "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
    },
    "multiplicity": {
        "families": [
            {
                "id": "displayed",
                "purpose": "simultaneous_intervals",
                "method": "bonferroni",
                "alpha": 0.05,
            }
        ]
    },
    "claims": [],
}


def _plan(version: str = "3.0", **changes: Any) -> dict[str, Any]:
    plan = copy.deepcopy(BASE)
    plan["schema_version"] = version
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        target = plan
        for parent in parents:
            target = target.setdefault(parent, {})
        target[leaf] = value
    return plan


def test_a_2_0_plan_keeps_its_defaults_and_hashes_only_what_it_wrote() -> None:
    plan = AnalysisPlan.model_validate(_plan("2.0"))
    dumped = plan.model_dump(mode="json", by_alias=True)
    assert dumped["inputs"]["sample_join"] == {"inspect_field": "id", "manifest_field": "prompt_id"}
    assert dumped["design"]["rollout_key"] == ["eval_id", "id", "epoch"]
    # The hash covers what the author wrote (#56): the defaults stay out of it, and writing
    # them out is a different plan text with a different hash.
    canonical = canonical_plan_value(plan)
    assert "sample_join" not in canonical["inputs"]
    assert "rollout_key" not in canonical.get("design", {})
    assert canonical_plan_hash(plan) != canonical_plan_hash(
        AnalysisPlan.model_validate(
            _plan(
                "2.0",
                inputs__sample_join={"inspect_field": "id", "manifest_field": "prompt_id"},
                design__rollout_key=["eval_id", "id", "epoch"],
            )
        )
    )


def test_the_schema_3_0_defaults_stay_out_of_the_plan_hash() -> None:
    plan = AnalysisPlan.model_validate(_plan())
    assert plan.design.rollout_key == ["run_key", "item_id", "repeat"]
    canonical = canonical_plan_value(plan)
    assert "sample_join" not in canonical["inputs"]
    assert "rollout_key" not in canonical.get("design", {})
    # Nothing the author left out is hashed: the value is the plan as written.
    assert canonical == _plan()


def test_a_2_0_plan_cannot_set_sample_join_to_null() -> None:
    with pytest.raises(ValidationError, match="sample_join cannot be null"):
        AnalysisPlan.model_validate(_plan("2.0", inputs__sample_join=None))


def test_a_3_0_plan_has_no_sample_join_and_a_neutral_rollout_key() -> None:
    plan = AnalysisPlan.model_validate(_plan())
    dumped = plan.model_dump(mode="json", by_alias=True)
    assert "sample_join" not in dumped["inputs"]
    assert dumped["design"]["rollout_key"] == ["run_key", "item_id", "repeat"]
    assert AnalysisPlan.model_validate(dumped) == plan


def test_a_3_0_plan_refuses_sample_join() -> None:
    with pytest.raises(ValidationError, match="sample_join was removed in schema 3.0"):
        AnalysisPlan.model_validate(
            _plan(inputs__sample_join={"inspect_field": "id", "manifest_field": "prompt_id"})
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"design__rollout_key": ["eval_id", "id", "epoch"]}, "design.rollout_key: 'eval_id'"),
        ({"design__rollout_key": ["attributes."]}, "design.rollout_key: 'attributes.'"),
        ({"cohorts": {"c": {"match": {"task_name": "x"}}}}, "cohorts.c.match: 'task_name'"),
        (
            {"cohorts": {"c": {"match": {"model": "m"}, "require_uniform": ["epochs"]}}},
            "cohorts.c.require_uniform: 'epochs'",
        ),
    ],
)
def test_a_3_0_plan_refuses_source_fields_outside_attributes(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AnalysisPlan.model_validate(_plan(**changes))


def test_a_3_0_plan_accepts_neutral_and_attribute_fields() -> None:
    plan = AnalysisPlan.model_validate(
        _plan(
            design__rollout_key=["run_key", "rollout_id", "attributes.seed"],
            cohorts={
                "c": {
                    "match": {"model": "m", "status": "success", "attributes.task_name": "x"},
                    "require_uniform": ["run_id", "attributes.task_version"],
                }
            },
        )
    )
    assert plan.schema_version == "3.0"


def _write(tmp_path: Path, plan: dict[str, Any]) -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\np2,q2,t1,2\n",
        encoding="utf-8",
    )
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    return path


def _jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_a_3_0_plan_reads_rollout_files_with_its_defaults(tmp_path: Path) -> None:
    plan = _write(
        tmp_path,
        _plan(
            cohorts={
                "red": {
                    "match": {"model": "m", "attributes.team": "red"},
                    "require_uniform": ["attributes.temperature"],
                },
                "blue": {"match": {"model": "m", "attributes.team": "blue"}},
            }
        ),
    )
    runs = _jsonl(
        tmp_path / "runs.jsonl",
        [
            {"run_key": key, "model": "m", "status": "success", "team": team, "temperature": 1}
            for key, team in (("r1", "red"), ("r2", "red"), ("r3", "blue"))
        ],
    )
    rollouts = _jsonl(
        tmp_path / "rollouts.jsonl",
        [
            # r1 and r2 both hold repeat 1 of p1: distinct rollouts under the neutral key.
            {"run_key": key, "item_id": item, "repeat": repeat, "score_harmful": "PASS"}
            for key, item, repeat in [
                ("r1", "p1", 1),
                ("r2", "p1", 1),
                ("r1", "p2", 1),
                ("r1", "p2", 2),
                ("r3", "p1", 1),
                ("r3", "p1", 2),
                ("r3", "p2", 1),
                ("r3", "p2", 2),
            ]
        ],
    )
    summary = validate(plan=plan, source=RolloutFiles(runs=runs, rollouts=rollouts))
    assert [item.code for item in summary.diagnostics] == []
    assert summary.status == "valid"


def test_a_3_0_plan_reads_inspect_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = _write(
        tmp_path,
        _plan(
            cohorts={
                "c": {
                    "match": {"model": "m", "attributes.task_name": "t"},
                    "require_uniform": ["attributes.task_version"],
                }
            }
        ),
    )
    evals = pd.DataFrame(
        [
            {
                "eval_id": "e1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "model": "m",
                "task_name": "t",
                "task_version": "1",
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"s-{item}-{epoch}",
                "eval_id": "e1",
                "id": item,
                "epoch": epoch,
                "score_harmful": "PASS",
                "error": None,
                "retries": 0,
            }
            for item in ("p1", "p2")
            for epoch in (1, 2)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    summary = validate(plan=plan, logs="ignored")
    assert [item.code for item in summary.diagnostics] == []
