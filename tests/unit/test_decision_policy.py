"""Decision policies: the policy file, and the inference hash that binds a policy to a plan."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from inspect_safety_stats.cli import app
from inspect_safety_stats.config.decision_policy import (
    DecisionPolicy,
    inference_plan_hash,
    load_decision_policy,
    plan_with_thresholds,
    policy_thresholds,
)
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from tests.cli_output import plain

EXAMPLE_PLAN = Path(__file__).parents[2] / "examples" / "fixed_design" / "safety-plan.yaml"
HASH = "a" * 64


def _plan_text() -> str:
    return EXAMPLE_PLAN.read_text(encoding="utf-8")


def _copy_plan(tmp_path: Path, text: str, name: str = "plan.yaml") -> Path:
    (tmp_path / "design.csv").write_bytes((EXAMPLE_PLAN.parent / "design.csv").read_bytes())
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _policy(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "1.0",
        "inference_plan_sha256": HASH,
        "versions": [
            {
                "id": "v1",
                "set_at": "2026-09-01T00:00:00Z",
                "thresholds": {"a": {"acceptable_upper": 0.05, "unacceptable_lower": 0.1}},
            },
            {
                "id": "v2",
                "set_at": "2026-09-02T00:00:00+02:00",
                "reason": "the review board asked for a stricter limit",
                "thresholds": {
                    "a": {"acceptable_upper": 0.02, "unacceptable_lower": 0.1},
                    "b": {"acceptable_upper": 0.1, "unacceptable_lower": 0.2},
                },
            },
        ],
    }
    value.update(changes)
    return value


def test_later_versions_set_the_thresholds_in_force() -> None:
    policy = DecisionPolicy.model_validate(_policy())
    in_force = policy_thresholds(policy)
    summary = {claim: (item.acceptable_upper, source) for claim, (item, source) in in_force.items()}
    assert summary == {"a": (0.02, "v2"), "b": (0.1, "v2")}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"inference_plan_sha256": "abc"}, "String should match pattern"),
        ({"versions": []}, "at least 1 item"),
        (
            {
                "versions": [
                    {"id": "v1", "set_at": "2026-09-01T00:00:00", "thresholds": {"a": {
                        "acceptable_upper": 0.05, "unacceptable_lower": 0.1}}}
                ]
            },
            "timezone",
        ),
        (
            {
                "versions": [
                    {"id": "v1", "set_at": "2026-09-01T00:00:00Z", "thresholds": {"a": {
                        "acceptable_upper": 0.1, "unacceptable_lower": 0.1}}}
                ]
            },
            "acceptable_upper must be less than unacceptable_lower",
        ),
        (
            {
                "versions": [
                    {"id": "v1", "set_at": "2026-09-01T00:00:00Z", "thresholds": {"a": {
                        "acceptable_upper": True, "unacceptable_lower": 0.1}}}
                ]
            },
            "not a boolean",
        ),
    ],
)
def test_invalid_policies_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DecisionPolicy.model_validate(_policy(**changes))


def test_versions_must_be_listed_in_the_order_they_were_set() -> None:
    value = _policy()
    versions = value["versions"]
    assert isinstance(versions, list)
    value["versions"] = list(reversed(versions))
    with pytest.raises(ValidationError, match="set at or before version"):
        DecisionPolicy.model_validate(value)
    same = _policy()
    same_versions = same["versions"]
    assert isinstance(same_versions, list)
    same_versions[1]["id"] = "v1"
    with pytest.raises(ValidationError, match="ids must be unique"):
        DecisionPolicy.model_validate(same)


def test_policy_files_are_read_from_yaml_and_json(tmp_path: Path) -> None:
    yaml_path = tmp_path / "policy.yaml"
    yaml_path.write_text(
        f"""
schema_version: 1.0
inference_plan_sha256: {HASH}
versions:
  - id: v1
    set_at: 2026-09-01T00:00:00Z
    thresholds:
      a: {{acceptable_upper: 0.05, unacceptable_lower: 0.1}}
""",
        encoding="utf-8",
    )
    assert load_decision_policy(yaml_path).versions[0].id == "v1"
    json_path = tmp_path / "policy.json"
    json_path.write_text(DecisionPolicy.model_validate(_policy()).model_dump_json(), "utf-8")
    assert len(load_decision_policy(json_path).versions) == 2
    with pytest.raises(ValueError, match="YAML or JSON"):
        load_decision_policy(tmp_path / "policy.txt")
    other = tmp_path / "other.yaml"
    other.write_text('schema_version: "2.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported decision policy schema_version"):
        load_decision_policy(other)


def test_the_inference_hash_ignores_only_the_thresholds_a_policy_may_set(tmp_path: Path) -> None:
    text = _plan_text()
    assert "acceptable_upper: 0.25" in text
    base = load_plan(_copy_plan(tmp_path, text))
    moved_text = text.replace("acceptable_upper: 0.25", "acceptable_upper: 0.01")
    moved = load_plan(_copy_plan(tmp_path, moved_text, "b.yaml"))
    # The plan's own hash changes with any threshold, the inference hash does not.
    assert canonical_plan_hash(base) != canonical_plan_hash(moved)
    assert inference_plan_hash(base) == inference_plan_hash(moved)
    assert inference_plan_hash(base) != canonical_plan_hash(base)
    # Anything else that shapes the bounds or the family changes the inference hash.
    reweighted = load_plan(
        _copy_plan(tmp_path, text.replace("alpha: 0.05", "alpha: 0.04", 1), "c.yaml")
    )
    assert inference_plan_hash(reweighted) != inference_plan_hash(base)


def test_noisy_valid_test_thresholds_stay_in_the_inference_hash(tmp_path: Path) -> None:
    plan = load_plan(_copy_plan(tmp_path, _plan_text()))
    noisy = plan.model_copy(
        update={"claims": [plan.claims[0].model_copy(update={"method": "noisy_valid_test"})]}
    )
    moved = noisy.model_copy(
        update={"claims": [noisy.claims[0].model_copy(update={"acceptable_upper": 0.001})]}
    )
    assert inference_plan_hash(noisy) != inference_plan_hash(moved)


def test_plan_with_thresholds_leaves_the_plan_and_its_hash_alone(tmp_path: Path) -> None:
    plan = load_plan(_copy_plan(tmp_path, _plan_text()))
    before = canonical_plan_hash(plan)
    claim = plan.claims[0]
    policy = DecisionPolicy.model_validate(
        _policy(
            versions=[
                {
                    "id": "v1",
                    "set_at": "2026-09-01T00:00:00Z",
                    "thresholds": {claim.id: {"acceptable_upper": 0.2, "unacceptable_lower": 0.3}},
                }
            ]
        )
    )
    effective = plan_with_thresholds(plan, policy_thresholds(policy))
    assert effective.claims[0].acceptable_upper == 0.2
    assert effective.claims[0].unacceptable_lower == 0.3
    assert plan.claims[0] is claim and claim.acceptable_upper != 0.2
    assert canonical_plan_hash(plan) == before
    assert effective.source_path() == plan.source_path()
    assert inference_plan_hash(effective) == inference_plan_hash(plan)


def test_hash_plan_prints_the_inference_hash_on_request(tmp_path: Path) -> None:
    path = _copy_plan(tmp_path, _plan_text())
    runner = CliRunner()
    full = runner.invoke(app, ["hash-plan", "--plan", str(path)])
    inference = runner.invoke(app, ["hash-plan", "--plan", str(path), "--inference"])
    assert full.exit_code == 0 and inference.exit_code == 0
    plan = load_plan(path)
    assert plain(full.output).strip() == canonical_plan_hash(plan)
    assert plain(inference.output).strip() == inference_plan_hash(plan)
