"""How load_plan treats a plan's schema_version.

An unquoted 2.0 is the current schema, not an error, and an older schema is refused with what
changed, not with a list of unknown fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan

BODY = """
analysis: {id: versioned, title: Versioned, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
outputs: {prompt_bounds: {enabled: false}}
multiplicity: {families: []}
claims: []
"""


def _plan(directory: Path, version: str | None, *, suffix: str = ".yaml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite\np1,q1,t1,s\n", encoding="utf-8"
    )
    plan = directory / f"plan{suffix}"
    header = "" if version is None else f"schema_version: {version}\n"
    plan.write_text(header + BODY.lstrip(), encoding="utf-8")
    return plan


def test_an_unquoted_two_point_zero_is_the_current_schema(tmp_path: Path) -> None:
    quoted = load_plan(_plan(tmp_path / "quoted", '"2.0"'))
    unquoted = load_plan(_plan(tmp_path / "unquoted", "2.0"))
    assert unquoted.schema_version == "2.0"
    # The same plan, so the same registration hash.
    assert canonical_plan_hash(unquoted) == canonical_plan_hash(quoted)
    assert load_plan(_plan(tmp_path / "default", None)).schema_version == "2.0"


@pytest.mark.parametrize("version", ['"1.0"', "1.0"])
def test_a_schema_one_plan_is_told_what_changed(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="declares schema_version '1.0'") as caught:
        load_plan(_plan(tmp_path, version))
    message = str(caught.value)
    assert "replaced independence_unit and cluster_sensitivity" in message
    assert "docs/independence-profiles.md" in message
    assert "the Inspect logs can be reused" in message


def test_an_unknown_schema_is_refused_with_the_supported_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported plan schema_version 3") as caught:
        load_plan(_plan(tmp_path, "3"))
    assert 'declared as schema_version: "2.0"' in str(caught.value)
