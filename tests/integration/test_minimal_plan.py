"""A minimal plan says exactly what it is missing, and following it gives an analysis (#57).

A plan that leaned on defaults analysed nothing. Prompt bounds are on by default, but take their
alpha from a family that has no default, and the old diagnostic did not say which field to set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from safety_stats import analyze, load_plan
from safety_stats.schema.reports import Diagnostic
from safety_stats.validate.plan import validate_plan_references

MINIMAL = """
schema_version: "2.0"
analysis: {id: minimal, title: Minimal, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
outputs:
  prompt_bounds: {@PROMPT_BOUNDS@}
  risk_measures: [@MEASURES@]
multiplicity: {families: [@FAMILIES@]}
claims: []
"""


def _plan(
    tmp_path: Path,
    *,
    prompt_bounds: str = "",
    measures: str = "",
    families: str = "",
    expected_rollouts: bool = False,
) -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts\np1,q1,t1,s,10\np2,q2,t1,s,10\n"
        if expected_rollouts
        else "prompt_id,question_id,task_id,suite\np1,q1,t1,s\np2,q2,t1,s\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        MINIMAL.replace("@PROMPT_BOUNDS@", prompt_bounds)
        .replace("@MEASURES@", measures)
        .replace("@FAMILIES@", families)
        .strip()
        + "\n",
        encoding="utf-8",
    )
    return plan


def _diagnostics(plan: Path, scope: str) -> list[Diagnostic]:
    return [item for item in validate_plan_references(load_plan(plan)) if item.scope == scope]


def test_a_minimal_plan_says_exactly_what_it_is_missing(tmp_path: Path) -> None:
    (family,) = _diagnostics(_plan(tmp_path), "plan/outputs/prompt_bounds")
    assert (family.code, family.severity) == ("PLAN_UNKNOWN_REFERENCE", "ERROR")
    assert "outputs.prompt_bounds.family, which is not set" in family.message
    assert "{id: displayed, purpose: simultaneous_intervals" in family.message
    assert "outputs.prompt_bounds.enabled: false" in family.message
    (measures,) = _diagnostics(_plan(tmp_path), "plan/outputs/risk_measures")
    assert (measures.code, measures.severity) == ("PLAN_NO_RISK_MEASURES", "WARNING")
    assert "{id: mean, type: weighted_mean}" in measures.message


def test_following_the_guidance_gives_a_working_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def guided(*, expected_rollouts: bool) -> Path:
        # The plan after the two fixes the diagnostics above suggest.
        return _plan(
            tmp_path,
            prompt_bounds="family: displayed",
            measures="{id: mean, type: weighted_mean}",
            families=(
                "{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}"
            ),
            expected_rollouts=expected_rollouts,
        )

    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "minimal",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "minimal",
                "epochs": 10,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt,
                "epoch": epoch,
                "score_harmful": "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt in ("p1", "p2")
            for epoch in range(1, 11)
        ]
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )

    # The next default the plan meets: rollout counts are required, and the manifest has none.
    report = analyze(plan=guided(expected_rollouts=False), logs="ignored")
    code = "PLAN_EXPECTED_ROLLOUTS_MISSING"
    (missing,) = [item for item in report.integrity.diagnostics if item.code == code]
    assert "Add it, with the number of rollouts" in missing.message
    assert "design.require_expected_rollouts: false" in missing.message

    report = analyze(plan=guided(expected_rollouts=True), logs="ignored")

    assert report.integrity.status != "invalid"
    (mean,) = report.group_risks
    assert mean.lower is not None and mean.upper is not None


def test_pointwise_prompt_bounds_need_a_family_too(tmp_path: Path) -> None:
    # Without one they used to be silently absent, with no diagnostic at all.
    plan = _plan(tmp_path, prompt_bounds="simultaneous: false")
    (family,) = _diagnostics(plan, "plan/outputs/prompt_bounds")
    assert family.code == "PLAN_UNKNOWN_REFERENCE"


def test_disabled_prompt_bounds_need_no_family(tmp_path: Path) -> None:
    plan = _plan(tmp_path, prompt_bounds="enabled: false")
    assert _diagnostics(plan, "plan/outputs/prompt_bounds") == []


@pytest.mark.parametrize("simultaneous", ["true", "false"])
@pytest.mark.parametrize(
    ("family", "families", "enabled_code"),
    [
        ("", "", "PLAN_UNKNOWN_REFERENCE"),
        ("family: undeclared", "", "PLAN_UNKNOWN_REFERENCE"),
        (
            "family: mandatory",
            "{id: mandatory, purpose: all_mandatory, method: intersection_union, alpha: 0.05}",
            "PLAN_INVALID_MULTIPLICITY",
        ),
    ],
    ids=["unset", "undeclared", "not_bonferroni"],
)
def test_disabled_prompt_bounds_ignore_simultaneous(
    tmp_path: Path, simultaneous: str, family: str, families: str, enabled_code: str
) -> None:
    # Validation used to test simultaneous before enabled, so {enabled: false} alone still
    # failed with PLAN_UNKNOWN_REFERENCE, or PLAN_INVALID_MULTIPLICITY with a family that is
    # not bonferroni, unless simultaneous: false was set too.
    def prompt_bound_codes(enabled: str) -> tuple[list[str], set[str]]:
        prompt_bounds = ", ".join(
            part
            for part in (f"enabled: {enabled}", f"simultaneous: {simultaneous}", family)
            if part
        )
        plan = _plan(tmp_path, prompt_bounds=prompt_bounds, families=families)
        diagnostics = validate_plan_references(load_plan(plan))
        # Matched on any scope naming prompt_bounds: that old code scoped it
        # outputs.prompt_bounds. Every code is returned too, so that the check does not rest
        # on the scope alone.
        return (
            [item.code for item in diagnostics if "prompt_bounds" in str(item.scope)],
            {item.code for item in diagnostics},
        )

    scoped, codes = prompt_bound_codes("false")
    assert scoped == []
    assert not codes & {"PLAN_UNKNOWN_REFERENCE", "PLAN_INVALID_MULTIPLICITY"}
    # The control: enabled, the same plan raises what a disabled one must not, and the scope
    # filter above finds it. Pointwise bounds need a family, but not a bonferroni one.
    raised = enabled_code == "PLAN_UNKNOWN_REFERENCE" or simultaneous == "true"
    assert prompt_bound_codes("true")[0] == ([enabled_code] if raised else [])


def test_an_undeclared_family_is_named(tmp_path: Path) -> None:
    plan = _plan(tmp_path, prompt_bounds="family: shown")
    (family,) = _diagnostics(plan, "plan/outputs/prompt_bounds")
    assert "names 'shown', not a declared multiplicity family" in family.message
