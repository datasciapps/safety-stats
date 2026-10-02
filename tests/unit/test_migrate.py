from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.config.migrate import PlanSchemaError, migrate_plan
from inspect_safety_stats.config.models import AnalysisPlan


def _legacy_plan() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "analysis": {"id": "legacy", "title": "Legacy", "intent": "exploratory"},
        "inputs": {
            "design_manifest": "design.csv",
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "design": {"hierarchy": ["task_id", "question_id", "prompt_id"]},
        "cohorts": {
            "candidate": {"match": {"model": "m/candidate"}},
            "baseline": {"match": {"model": "m/baseline"}},
        },
        "groups": [{"id": "all", "filter": {}}],
        "outputs": {
            "direct_mean_intervals": {
                "enabled": True,
                "method": "weighted_kl_chernoff",
                "family": "means",
                "independence_unit": "rollout",
                "cluster_sensitivity": True,
            },
            "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
        },
        "comparisons": [
            {
                "id": "difference",
                "candidate": "candidate",
                "baseline": "baseline",
                "group": "all",
                "measure": "mean",
                "family": "comparisons",
                "independence_unit": "rollout",
                "cluster_sensitivity": False,
                "pair_key": ["prompt_id", "epoch"],
                "diagnostic_methods": ["exact_mcnemar"],
            }
        ],
        "multiplicity": {
            "families": [
                {"id": "means", "purpose": "simultaneous_intervals", "method": "bonferroni",
                 "alpha": 0.05},
                {"id": "comparisons", "purpose": "simultaneous_intervals",
                 "method": "bonferroni", "alpha": 0.05},
            ]
        },
        "claims": [],
    }


def test_migration_maps_legacy_independence_and_drops_removed_diagnostics() -> None:
    upgraded, notes = migrate_plan(_legacy_plan())
    plan = AnalysisPlan.model_validate(upgraded)

    direct = plan.outputs.direct_mean_intervals.independence
    assert direct is not None
    assert (direct.primary, direct.sensitivities) == ("rollout", ["prompt"])
    comparison = plan.comparisons[0].independence
    assert comparison is not None
    assert (comparison.primary, comparison.sensitivities) == ("rollout", [])
    assert plan.design.independence_profiles["rollout"].block_by == ["rollout"]
    assert plan.design.independence_profiles["prompt"].block_by == ["prompt_id"]
    assert any("pair_key" in note for note in notes)
    assert upgraded["schema_version"] == "2.0"


def test_loader_refuses_a_legacy_plan_with_a_migration_hint(tmp_path: Path) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(_legacy_plan()), encoding="utf-8")
    with pytest.raises(PlanSchemaError, match="migrate-plan"):
        load_plan(path)


def test_loader_refuses_an_unknown_schema_version(tmp_path: Path) -> None:
    raw = _legacy_plan()
    raw["schema_version"] = "0.3"
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(PlanSchemaError, match="unsupported plan schema_version"):
        load_plan(path)


def test_migrate_plan_command_writes_a_loadable_plan(tmp_path: Path) -> None:
    source = tmp_path / "old.yaml"
    target = tmp_path / "new.yaml"
    source.write_text(yaml.safe_dump(_legacy_plan()), encoding="utf-8")
    (tmp_path / "design.csv").write_text("prompt_id,question_id,task_id\n", encoding="utf-8")

    result = CliRunner().invoke(
        app, ["migrate-plan", "--plan", str(source), "--output", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert "Upgraded plan" in result.output
    assert load_plan(target).schema_version == "2.0"

    again = CliRunner().invoke(
        app, ["migrate-plan", "--plan", str(source), "--output", str(target)]
    )
    assert again.exit_code != 0


def test_loader_accepts_an_unquoted_current_schema_version(tmp_path: Path) -> None:
    upgraded, _ = migrate_plan(_legacy_plan())
    text = yaml.safe_dump(upgraded).replace("schema_version: '2.0'", "schema_version: 2.0")
    assert "schema_version: 2.0" in text
    path = tmp_path / "plan.yaml"
    path.write_text(text, encoding="utf-8")
    assert load_plan(path).schema_version == "2.0"
