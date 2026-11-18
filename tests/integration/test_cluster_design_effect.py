"""CLUSTER_DESIGN_EFFECT on a clustered synthetic set: 80 single-rollout prompts in 10
conversations of 8, with the conversation in a manifest column."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats import validate
from inspect_safety_stats.ingest.rollout_files import RolloutFiles
from inspect_safety_stats.schema.reports import Diagnostic
from tests.diagnostic_codes.harness import Inputs, base_inputs, rollout, run_analysis, write_inputs

CODE = "CLUSTER_DESIGN_EFFECT"

# Conversations c0-c2 fail on 7 of their 8 prompts, c5 on one, the rest on none.
CO_FAILING = {(conversation, item) for conversation in range(3) for item in range(1, 8)} | {
    (5, 2)
}
# The same 22 failures spread over the conversations, at most three in each.
SPREAD = {(index % 10, index // 10) for index in range(22)}


def _inputs(failures: set[tuple[int, int]], primary: str = "item") -> Inputs:
    inputs = base_inputs()
    inputs.manifest = []
    inputs.rollouts = []
    for conversation in range(10):
        for item in range(8):
            prompt = f"c{conversation}p{item}"
            inputs.manifest.append(
                {
                    "prompt_id": prompt,
                    "question_id": prompt,
                    "task_id": "t1",
                    "expected_rollouts": 1,
                    "conversation": f"c{conversation}",
                }
            )
            outcome = "FAIL" if (conversation, item) in failures else "PASS"
            inputs.rollouts.append(rollout(prompt, 1, outcome))
    profiles = {"item": {"block_by": ["prompt_id"]}, "conversation": {"block_by": ["conversation"]}}
    sensitivities = [name for name in profiles if name != primary]
    inputs.plan["design"] = {"independence_profiles": profiles}
    inputs.plan["multiplicity"]["families"].append(
        {"id": "direct", "purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}
    )
    inputs.plan["outputs"]["direct_mean_intervals"] = {
        "enabled": True,
        "method": "weighted_kl_chernoff",
        "family": "direct",
        "independence": {"primary": primary, "sensitivities": sensitivities},
    }
    return inputs


def _found(diagnostics: Any) -> list[Diagnostic]:
    return [item for item in diagnostics if item.code == CODE]


def test_co_failing_conversations_warn_with_their_design_effect(tmp_path: Path) -> None:
    (found,) = _found(run_analysis(_inputs(CO_FAILING), tmp_path).diagnostics)
    assert found.severity == "WARNING"
    assert found.scope == "plan/design/independence_profiles/conversation"
    (cohort,) = found.details["cohorts"]
    # Sizes 8 x 10, overall rate 22/80: MSB = 83/60, MSW = 1/20, m0 = 8, so
    # rho = (80/60) / (83/60 + 7/20) = 10/13 and the design effect is 1 + 7 rho = 83/13.
    assert cohort["intracluster_correlation"] == pytest.approx(10 / 13, rel=1e-12, abs=0)
    assert cohort["design_effect"] == pytest.approx(83 / 13, rel=1e-12, abs=0)
    assert cohort["effective_rollouts"] == pytest.approx(80 * 13 / 83, rel=1e-12, abs=0)
    assert (cohort["rollouts"], cohort["blocks"], cohort["largest_block"]) == (80, 10, 8)
    assert found.details["primaries_splitting_blocks"] == ["item"]
    assert "about 13 effective rollouts" in found.message
    assert "'item'" in found.message


def test_spread_failures_are_only_recorded(tmp_path: Path) -> None:
    (found,) = _found(run_analysis(_inputs(SPREAD), tmp_path).diagnostics)
    assert found.severity == "INFO"
    (cohort,) = found.details["cohorts"]
    assert cohort["intracluster_correlation"] < 0
    assert cohort["design_effect"] == 1.0
    assert cohort["effective_rollouts"] == 80
    assert found.details["primaries_splitting_blocks"] == []


def test_a_conversation_primary_is_not_warned_about_its_own_blocks(tmp_path: Path) -> None:
    (found,) = _found(
        run_analysis(_inputs(CO_FAILING, primary="conversation"), tmp_path).diagnostics
    )
    # The primary already blocks by conversation; the item sensitivity is item-level and
    # gets no diagnostic of its own.
    assert found.severity == "INFO"
    assert found.details["profile"] == "conversation"
    assert found.details["cohorts"][0]["design_effect"] == pytest.approx(83 / 13, abs=1e-12)


def test_validate_reports_what_analyze_reports(tmp_path: Path) -> None:
    inputs = _inputs(CO_FAILING)
    analyzed = _found(run_analysis(inputs, tmp_path / "analyze").diagnostics)
    plan = write_inputs(inputs, tmp_path / "validate")
    summary = validate(
        plan=plan,
        source=RolloutFiles(
            runs=plan.parent / "runs.jsonl", rollouts=plan.parent / "rollouts.jsonl"
        ),
    )
    assert _found(summary.diagnostics) == analyzed
    # The diagnostic is a warning: integrity is as valid as without it.
    assert summary.status == "valid"


def test_item_level_profiles_get_no_diagnostic(tmp_path: Path) -> None:
    inputs = _inputs(CO_FAILING)
    inputs.plan["design"]["independence_profiles"] = {
        "item": {"block_by": ["prompt_id"]},
        "rollout": {"block_by": ["rollout"]},
    }
    inputs.plan["outputs"]["direct_mean_intervals"]["independence"] = {
        "primary": "item",
        "sensitivities": ["rollout"],
    }
    assert _found(run_analysis(inputs, tmp_path).diagnostics) == []
