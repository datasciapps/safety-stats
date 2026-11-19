from pathlib import Path

import pytest
import yaml

from inspect_safety_stats.config.models import AnalysisPlan, WeightSpec
from inspect_safety_stats.ingest.manifest import (
    check_zero_tolerance_field,
    load_manifest,
    plan_manifest_columns,
    resolve_weights,
)
from inspect_safety_stats.ingest.normalize import read_inputs
from inspect_safety_stats.ingest.records import SourceData


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


def _comparison(**keys: object) -> dict[str, object]:
    return {
        "id": "c",
        "group": "all",
        "candidate": "candidate",
        "baseline": "candidate",
        "family": "f",
        "independence": {"primary": "p"},
        "measure": "m",
        **keys,
    }


def _plan(edit: str = "") -> AnalysisPlan:
    data = yaml.safe_load(_PLAN)
    if edit == "default_hierarchy":
        del data["design"]["hierarchy"]
        del data["design"]["weights"]
    elif edit == "group_filter":
        data["groups"] = [{"id": "q1", "filter": {"question_id": "q1"}}]
    elif edit == "design_weights":
        data["design"]["weights"] = {"strategy": "nested_equal"}
    elif edit == "group_weights":
        data["groups"] = [
            {
                "id": "all",
                "filter": {},
                "weights": {
                    "strategy": "nested_equal",
                    "hierarchy": ["task_id", "question_id", "prompt_id"],
                },
            }
        ]
    elif edit in {"pair_key", "cluster_key"}:
        data["design"]["independence_profiles"] = {"p": {"block_by": ["prompt_id"]}}
        data["groups"] = [{"id": "all", "filter": {}}]
        key = {"pair_key": ["question_id", "prompt_id"], "cluster_key": "question_id"}
        data["comparisons"] = [_comparison(**{edit: key[edit]})]
    elif edit == "block_by":
        data["design"]["independence_profiles"] = {"question": {"block_by": ["question_id"]}}
    return AnalysisPlan.model_validate(data)


def test_a_plan_requires_only_the_identifier_columns_it_names() -> None:
    assert plan_manifest_columns(_plan()) == ("prompt_id", "task_id")


@pytest.mark.parametrize(
    "edit",
    [
        "default_hierarchy",
        "group_filter",
        "block_by",
        "design_weights",
        "group_weights",
        "pair_key",
        "cluster_key",
    ],
)
def test_a_plan_that_names_question_id_anywhere_requires_it(edit: str) -> None:
    assert plan_manifest_columns(_plan(edit)) == ("prompt_id", "question_id", "task_id")


def _zero_tolerance_plan(policy: dict[str, object]) -> AnalysisPlan:
    data = yaml.safe_load(_PLAN)
    data["policy"] = {"zero_tolerance": policy}
    return AnalysisPlan.model_validate(data)


def test_a_named_zero_tolerance_column_must_be_in_the_manifest(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,task_id\np1,t1\n", encoding="utf-8")
    manifest = load_manifest(path, ["prompt_id", "task_id"])
    plan = _zero_tolerance_plan({"manifest_field": "must_not_fail"})
    with pytest.raises(ValueError, match="no column 'must_not_fail'"):
        check_zero_tolerance_field(plan, manifest)
    # The default field is optional, and a disabled gate needs no column.
    check_zero_tolerance_field(_zero_tolerance_plan({}), manifest)
    check_zero_tolerance_field(
        _zero_tolerance_plan({"enabled": False, "manifest_field": "must_not_fail"}), manifest
    )
    manifest["must_not_fail"] = False
    check_zero_tolerance_field(plan, manifest)


class _UnreadSource:
    """A source that fails the test if anything reads it."""

    def read(self) -> SourceData:
        raise AssertionError("the source was read before the manifest was checked")


def test_reading_inputs_stops_at_a_manifest_without_the_named_zero_tolerance_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "design.csv"
    path.write_text("prompt_id,task_id\np1,t1\n", encoding="utf-8", newline="\n")
    data = yaml.safe_load(_PLAN)
    data["inputs"]["design_manifest"] = str(path)
    data["policy"] = {"zero_tolerance": {"manifest_field": "must_not_fail"}}
    plan = AnalysisPlan.model_validate(data)
    with pytest.raises(ValueError, match="no column 'must_not_fail'"):
        read_inputs(plan, _UnreadSource())
