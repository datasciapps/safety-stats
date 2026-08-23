from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.structural.data import build_model_frame


def _plan(tmp_path: Path) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "analysis": {
                "id": "structural-test",
                "title": "Structural test",
                "intent": "exploratory",
            },
            "inputs": {
                "design_manifest": str(tmp_path / "manifest.csv"),
                "score": {
                    "name": "harmful",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
            },
            "cohorts": {"candidate": {"match": {"model": "model/test"}}},
            "groups": [{"id": "all", "filter": {}}],
            "outputs": {"risk_measures": [{"id": "mean", "type": "weighted_mean"}]},
            "multiplicity": {"families": []},
            "claims": [],
            "structural_model": {
                "enabled": True,
                "cohort": "candidate",
                "group": "all",
                "inference": {"cross_validation_folds": 0},
            },
        }
    )


def _report_directory(tmp_path: Path, *, unclassified: bool = False) -> Path:
    directory = tmp_path / "report"
    directory.mkdir()
    report = {
        "schema_version": "2.0",
        "package_version": "0.5.0",
        "analysis": {"id": "structural-test"},
        "provenance": {},
        "scope": {},
        "presentation": {},
        "integrity": {"status": "valid", "diagnostics": []},
        "design": {"groups": [{"id": "all"}]},
        "cohorts": [{"id": "candidate"}],
        "prompt_estimates": [],
        "group_risks": [
            {
                "cohort_id": "candidate",
                "group_id": "all",
                "measure_id": "mean",
                "estimate": 0.25,
                "method": "weighted_mean",
                "prompt_count": 2,
                "valid_prompt_count": 2,
                "resolved_weight": 1.0,
                "unresolved_weight": 0.0,
                "eligibility": "diagnostic",
            }
        ],
        "comparisons": [],
        "multiplicity": [],
        "claims": [],
        "policy_events": [],
        "decision": "INCONCLUSIVE",
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "design.resolved.csv").write_text(
        "group_id,task_id,question_id,prompt_id,weight\nall,t1,q1,p1,0.5\nall,t1,q2,p2,0.5\n",
        encoding="utf-8",
    )
    failure = "" if unclassified else "False"
    (directory / "observations.csv").write_text(
        "prompt_id,cohort_id,failure,configuration_fingerprint\n"
        "p1,candidate,True,config-1\n"
        f"p1,candidate,{failure},config-1\n"
        "p2,candidate,False,config-1\n"
        "p2,candidate,False,config-1\n",
        encoding="utf-8",
    )
    return directory


def test_build_model_frame_aggregates_prompt_binomials(tmp_path: Path) -> None:
    result = build_model_frame(plan=_plan(tmp_path), empirical_report=_report_directory(tmp_path))
    assert result.prompt_ids == ("p1", "p2")
    assert result.failures.tolist() == [1, 0]
    assert result.trials.tolist() == [2, 2]
    assert result.weights.tolist() == [0.5, 0.5]
    assert result.declared_hierarchy == ("task_id", "question_id", "prompt_id")
    assert result.hierarchy == ("task_id",)
    assert result.prompt_frailty_levels == ("question_id", "prompt_id")
    assert result.category_levels == {"task_id": ("t1",)}
    assert "failure" not in result.frame.columns


def test_build_model_frame_rejects_unclassified_rollouts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="complete classified outcomes"):
        build_model_frame(
            plan=_plan(tmp_path),
            empirical_report=_report_directory(tmp_path, unclassified=True),
        )


def test_structural_config_rejects_hierarchy_different_from_design(tmp_path: Path) -> None:
    plan = _plan(tmp_path).model_dump(mode="python", by_alias=True)
    plan["structural_model"]["hierarchy"] = ["task_id", "prompt_id"]
    with pytest.raises(ValidationError, match="exactly match design.hierarchy"):
        AnalysisPlan.model_validate(plan)
