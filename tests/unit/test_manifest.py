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


def test_identifiers_are_read_as_written(tmp_path: Path) -> None:
    """pandas' inference read `0001` as the number 1 and `1.0` as a float."""
    path = tmp_path / "design.csv"
    path.write_text(
        "prompt_id,question_id,task_id\n0001,1.0,01\n0002,1,true\n", encoding="utf-8"
    )
    manifest = load_manifest(path)
    assert manifest["prompt_id"].tolist() == ["0001", "0002"]
    assert manifest["question_id"].tolist() == ["1.0", "1"]
    assert manifest["task_id"].tolist() == ["01", "true"]


def _labelled_plan() -> AnalysisPlan:
    data = yaml.safe_load(_PLAN)
    data["design"]["weights"] = {"strategy": "explicit", "column": "w"}
    data["design"]["independence_profiles"] = {
        "cluster": {"block_by": ["cluster"]},
        "repeat": {"block_by": ["prompt_id", "repeat"]},
    }
    data["groups"] = [
        {"id": "tier", "filter": {"tier": "01"}},
        {"id": "domain", "filter": {}, "weights": {"hierarchy": ["domain", "prompt_id"]}},
    ]
    data["outputs"]["risk_measures"] = [
        {"id": "over", "type": "threshold_exceedance", "threshold_column": "limit"}
    ]
    return AnalysisPlan.model_validate(data)


def test_a_plan_groups_by_its_hierarchy_filters_and_blocks_but_not_its_numbers() -> None:
    # Imported here, so that the module's other tests still run against a source without it.
    from inspect_safety_stats.ingest.manifest import plan_grouping_columns

    assert plan_grouping_columns(_labelled_plan()) == (
        "cluster",
        "domain",
        "prompt_id",
        "task_id",
        "tier",
    )


def test_numeric_columns_stay_numbers_beside_text_labels(tmp_path: Path) -> None:
    from inspect_safety_stats.ingest.manifest import plan_grouping_columns

    path = tmp_path / "design.csv"
    path.write_text(
        "prompt_id,task_id,cluster,tier,domain,w,limit,expected_rollouts,mandatory\n"
        "0001,01,1.0,01,1,0.25,0.5,2,true\n"
        "0002,01,1,1,01,0.75,1,2,false\n",
        encoding="utf-8",
    )
    plan = _labelled_plan()
    manifest = load_manifest(path, plan_manifest_columns(plan), plan_grouping_columns(plan))
    assert manifest["cluster"].tolist() == ["1.0", "1"]
    assert manifest["tier"].tolist() == ["01", "1"]
    assert manifest["domain"].tolist() == ["1", "01"]
    assert manifest["w"].dtype == "float64"
    assert manifest["limit"].dtype == "float64"
    assert manifest["expected_rollouts"].tolist() == [2, 2]
    assert manifest["mandatory"].tolist() == [True, False]
    weights = resolve_weights(manifest, plan.design.weights)
    assert weights.tolist() == pytest.approx([0.25, 0.75], rel=1e-12, abs=0)


def test_a_blank_label_names_its_column_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "design.csv"
    rows = "".join(f"p{index},{'' if index % 2 else 'c'}\n" for index in range(1, 15))
    path.write_text("prompt_id,cluster\n" + rows, encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        load_manifest(path, ["prompt_id"], ["cluster"])
    assert str(raised.value).startswith(
        "design manifest column 'cluster' is blank in row 2 (prompt_id 'p1'), row 4 "
        "(prompt_id 'p3'), row 6 (prompt_id 'p5'), row 8 (prompt_id 'p7'), row 10 "
        "(prompt_id 'p9') and 2 more, counting the header as row 1."
    )
