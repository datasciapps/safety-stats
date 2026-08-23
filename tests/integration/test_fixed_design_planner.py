from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.certification import plan_fixed_design
from inspect_safety_stats.cli import app


def _write_plans(tmp_path: Path) -> tuple[Path, Path]:
    pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "question_id": "q1",
                "task_id": "t1",
                "policy_weight": 0.9,
            },
            {
                "prompt_id": "p2",
                "question_id": "q2",
                "task_id": "t1",
                "policy_weight": 0.1,
            },
        ]
    ).to_csv(tmp_path / "design.csv", index=False)
    analysis = {
        "schema_version": "2.0",
        "analysis": {"id": "planner-fixture", "title": "Planner fixture", "intent": "exploratory"},
        "inputs": {
            "design_manifest": "design.csv",
            "score": {
                "name": "safety",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "design": {
            "independence_profiles": {
                "rollout": {
                    "block_by": ["rollout"],
                    "comparison_coupling": "independent",
                }
            },
            "weights": {
                "strategy": "explicit",
                "hierarchy": ["task_id", "question_id", "prompt_id"],
                "column": "policy_weight",
            },
        },
        "cohorts": {"candidate": {"match": {"model": "fixture"}}},
        "groups": [{"id": "all", "filter": {}}],
        "outputs": {"prompt_bounds": {"enabled": False}},
        "multiplicity": {"families": []},
        "claims": [],
    }
    certification = {
        "schema_version": "2.0",
        "certification": {"id": "planner-cert", "title": "Planner certification"},
        "source": {},
        "claims": [
            {
                "id": "overall",
                "title": "Overall risk is at most 10%",
                "target": {"type": "group_mean", "cohort": "candidate", "group": "all"},
                "threshold": 0.1,
                "support": {
                    "family": "support",
                    "method": "weighted_hoeffding",
                    "independence_profile": "rollout",
                },
                "rejection": {
                    "family": "rejection",
                    "method": "weighted_hoeffding",
                    "independence_profile": "rollout",
                },
            }
        ],
        "multiplicity": {
            "families": [
                {
                    "id": "support",
                    "purpose": "simultaneous_support",
                    "method": "bonferroni",
                    "alpha": 0.05,
                },
                {
                    "id": "rejection",
                    "purpose": "simultaneous_rejection",
                    "method": "bonferroni",
                    "alpha": 0.05,
                },
            ]
        },
    }
    analysis_path = tmp_path / "analysis.yaml"
    certification_path = tmp_path / "certification.yaml"
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8")
    certification_path.write_text(yaml.safe_dump(certification, sort_keys=False), encoding="utf-8")
    return analysis_path, certification_path


def test_fixed_design_planner_produces_feasible_weight_adaptive_allocation(
    tmp_path: Path,
) -> None:
    analysis, certification = _write_plans(tmp_path)
    report = plan_fixed_design(
        analysis_plan=analysis,
        certification_plan=certification,
        anticipated_rate=0.02,
        power=0.8,
    )

    claim = report.claims[0]
    allocation = {item.prompt_id: item.rollouts for item in report.allocation}
    maximum_variance = claim.maximum_concentration_variance
    assert maximum_variance is not None
    achieved_variance = 0.9**2 / allocation["p1"] + 0.1**2 / allocation["p2"]
    assert achieved_variance <= maximum_variance + 1e-12
    equal_variance = (0.9**2 + 0.1**2) / claim.required_equal_rollouts_per_prompt
    assert equal_variance <= maximum_variance + 1e-12
    if claim.required_equal_rollouts_per_prompt > 1:
        previous = (0.9**2 + 0.1**2) / (claim.required_equal_rollouts_per_prompt - 1)
        assert previous > maximum_variance
    assert report.summary["adaptive_total_rollouts"] < 2 * int(
        report.summary["required_equal_rollouts_per_prompt"]
    )
    assert claim.planned_detectable_rate is not None
    assert claim.planned_detectable_rate >= 0.02
    expected_maximum = (
        2 * (0.1 - 0.02) ** 2 / (math.sqrt(math.log(20)) + math.sqrt(math.log(5))) ** 2
    )
    assert maximum_variance == pytest.approx(expected_maximum)

    output = report.write(tmp_path / "planning-output")
    assert (output / "planning-report.html").exists()
    assert (output / "planning-report.json").exists()
    assert (output / "planning-claims.csv").exists()
    assert (output / "rollout-allocation.csv").exists()
    payload = json.loads((output / "planning-report.json").read_text(encoding="utf-8"))
    assert payload["summary"]["feasible"] is True


def test_fixed_design_planner_reports_rollout_limit(tmp_path: Path) -> None:
    analysis, certification = _write_plans(tmp_path)
    report = plan_fixed_design(
        analysis_plan=analysis,
        certification_plan=certification,
        anticipated_rate=0.02,
        power=0.8,
        maximum_rollouts=2,
    )
    assert report.summary["feasible"] is False
    assert report.warnings


def test_fixed_design_planner_cli(tmp_path: Path) -> None:
    analysis, certification = _write_plans(tmp_path)
    output = tmp_path / "cli-output"
    result = CliRunner().invoke(
        app,
        [
            "plan",
            "fixed-design",
            "--analysis-plan",
            str(analysis),
            "--certification-plan",
            str(certification),
            "--anticipated-rate",
            "0.02",
            "--power",
            "0.8",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Required common rollouts per prompt" in result.stdout
    assert (output / "planning-report.html").exists()
