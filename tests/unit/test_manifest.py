from pathlib import Path

import pytest
import yaml

from inspect_safety_stats.config.models import AnalysisPlan, WeightSpec
from inspect_safety_stats.ingest.manifest import (
    load_manifest,
    plan_manifest_columns,
    resolve_weights,
)


def test_nested_equal_prevents_variant_count_weighting(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text(
        "prompt_id,question_id,task_id,expected_rollouts\np1,q1,t1,2\np2,q1,t1,2\np3,q2,t1,2\n",
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    weights = resolve_weights(manifest, WeightSpec(strategy="nested_equal"))
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.5])


def test_duplicate_prompt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,question_id,task_id\np1,q1,t1\np1,q1,t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)


def test_an_omitted_level_takes_the_prompt_id(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,task_id\np1,t1\np2,t1\n", encoding="utf-8")
    manifest = load_manifest(path, ["prompt_id", "task_id"])
    assert manifest["question_id"].tolist() == ["p1", "p2"]
    spec = WeightSpec(strategy="nested_equal", hierarchy=["task_id", "prompt_id"])
    weights = resolve_weights(manifest, spec)
    assert weights.tolist() == pytest.approx([0.5, 0.5], abs=0)


def test_by_default_every_identifier_column_is_required(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,task_id\np1,t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns: question_id"):
        load_manifest(path)


def test_prompt_id_cannot_be_made_optional(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id\np1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must include prompt_id"):
        load_manifest(path, ["task_id"])


_PLAN = """
analysis: {id: levels, title: Levels, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
design:
  type: fixed
  hierarchy: [task_id, prompt_id]
  weights: {strategy: nested_equal, hierarchy: [task_id, prompt_id]}
cohorts:
  candidate: {match: {model: model/candidate}}
groups: []
outputs: {prompt_bounds: {enabled: false}}
multiplicity: {families: []}
claims: []
"""


def _plan(edit: str = "") -> AnalysisPlan:
    data = yaml.safe_load(_PLAN)
    if edit == "default_hierarchy":
        del data["design"]["hierarchy"]
        del data["design"]["weights"]
    elif edit == "group_filter":
        data["groups"] = [{"id": "q1", "filter": {"question_id": "q1"}}]
    elif edit == "block_by":
        data["design"]["independence_profiles"] = {"question": {"block_by": ["question_id"]}}
    return AnalysisPlan.model_validate(data)


def test_a_plan_requires_only_the_identifier_columns_it_names() -> None:
    assert plan_manifest_columns(_plan()) == ("prompt_id", "task_id")


@pytest.mark.parametrize("edit", ["default_hierarchy", "group_filter", "block_by"])
def test_a_plan_that_names_question_id_anywhere_requires_it(edit: str) -> None:
    assert plan_manifest_columns(_plan(edit)) == ("prompt_id", "question_id", "task_id")
