"""CLUSTER_DESIGN_EFFECT on clustered synthetic sets, with the conversation in a manifest
column: mostly 80 prompts in 10 conversations of 8, with one rollout per prompt or ten."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
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

# Ten rollouts per prompt. Prompts differ widely in how many of their rollouts fail; the
# conversations do not, beyond their prompts: the even ones hold prompts that fail 0, 0, 0, 1, 2,
# 3, 5 and 10 times, 21 failures in all, the odd ones prompts that fail 0, 1, 2, 3, 5, 8, 8 and
# 10 times, 37 in all. That gap between conversations is smaller than the spread of their own
# prompts makes likely by chance.
PROMPT_SPREAD = {
    (conversation, item): (
        (0, 0, 0, 1, 2, 3, 5, 10) if conversation % 2 == 0 else (0, 1, 2, 3, 5, 8, 8, 10)
    )[item]
    for conversation in range(10)
    for item in range(8)
}


def _inputs(
    failures: Collection[tuple[int, int]] | Mapping[tuple[int, int], int],
    primary: str = "item",
    rollouts: int = 1,
) -> Inputs:
    """10 conversations of 8 prompts, each with ``rollouts`` rollouts. ``failures`` gives the
    (conversation, item) pairs whose rollouts all fail, or each prompt's number of failing
    rollouts."""

    failing = (
        failures
        if isinstance(failures, Mapping)
        else {key: rollouts for key in failures}
    )
    return _layout_inputs(
        [
            [(rollouts, failing.get((conversation, item), 0)) for item in range(8)]
            for conversation in range(10)
        ],
        primary,
    )


def _layout_inputs(
    conversations: Sequence[Sequence[tuple[int, int]]], primary: str = "item"
) -> Inputs:
    """Prompt ``p`` of conversation ``c`` is ``c{c}p{p}``, with the (rollouts, failures) at
    ``conversations[c][p]``: its first ``failures`` rollouts fail. Every prompt is its own
    question."""

    inputs = base_inputs()
    inputs.manifest = []
    inputs.rollouts = []
    for conversation, prompts in enumerate(conversations):
        for item, (rollouts, failures) in enumerate(prompts):
            prompt = f"c{conversation}p{item}"
            inputs.manifest.append(
                {
                    "prompt_id": prompt,
                    "question_id": prompt,
                    "task_id": "t1",
                    "expected_rollouts": rollouts,
                    "conversation": f"c{conversation}",
                }
            )
            for repeat in range(1, rollouts + 1):
                failed = repeat <= failures
                inputs.rollouts.append(rollout(prompt, repeat, "FAIL" if failed else "PASS"))
    profiles = {"item": {"block_by": ["prompt_id"]}, "conversation": {"block_by": ["conversation"]}}
    if primary == "rollout":
        profiles["rollout"] = {"block_by": ["rollout"]}
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
    # With one rollout per prompt, the item primary's units are the rollouts.
    (units,) = cohort["primaries"]
    assert units == {
        "profile": "item",
        "units": 80,
        "intracluster_correlation": cohort["intracluster_correlation"],
        "design_effect": cohort["design_effect"],
        "effective_units": cohort["effective_rollouts"],
    }
    assert "for the 80 units of primary 'item' in them" in found.message


def test_co_failing_conversations_warn_on_the_units_of_repeated_rollouts(
    tmp_path: Path,
) -> None:
    (found,) = _found(run_analysis(_inputs(CO_FAILING, rollouts=10), tmp_path).diagnostics)
    assert found.severity == "WARNING"
    assert found.details["primaries_splitting_blocks"] == ["item"]
    (cohort,) = found.details["cohorts"]
    assert (cohort["rollouts"], cohort["blocks"], cohort["largest_block"]) == (800, 10, 80)
    # Every rollout of a failing prompt fails, so the prompts' failure rates are 0 or 1, as the
    # single rollouts were, and the figures are now of the item primary's 80 units, not of the
    # 800 rollouts. Units of several rollouts take the pairwise estimate: around the mean rate
    # 11/40, deviations 29/40 and -11/40, so S^2 = (11/40)(29/40) = 319/1600. The products of
    # deviations of two units in one conversation sum to 3 (23.04 - 3.755) + (1.44 - 1.055) +
    # 6 (4.84 - 0.605) = 1673/20 over 560 ordered pairs, so rho = 239/319, close to the 10/13
    # of single rollouts, and the design effect is 1 + 7 rho = 1992/319.
    (units,) = cohort["primaries"]
    assert units["units"] == 80
    assert units["intracluster_correlation"] == pytest.approx(239 / 319, rel=1e-12, abs=0)
    assert units["design_effect"] == pytest.approx(1992 / 319, rel=1e-12, abs=0)
    assert units["effective_units"] == pytest.approx(80 * 319 / 1992, rel=1e-12, abs=0)
    assert "about 13 effective units" in found.message


def test_prompt_differences_alone_do_not_warn_an_item_primary(tmp_path: Path) -> None:
    (found,) = _found(
        run_analysis(_inputs(PROMPT_SPREAD, rollouts=10), tmp_path).diagnostics
    )
    assert found.severity == "INFO"
    assert found.details["primaries_splitting_blocks"] == []
    (cohort,) = found.details["cohorts"]
    # On rollouts, the ten rollouts of one prompt are correlated because prompts differ, and
    # the design effect passes 2. Sizes 80 x 10, rates 21/80 and 37/80 around 29/80:
    # MSB = 8/9, MSW = 283/1264 and m0 = 80.
    on_rollouts = (8 / 9 - 283 / 1264) / (8 / 9 + 79 * 283 / 1264)
    assert cohort["intracluster_correlation"] == pytest.approx(on_rollouts, rel=1e-12, abs=0)
    assert cohort["design_effect"] == pytest.approx(1 + 79 * on_rollouts, rel=1e-12, abs=0)
    assert cohort["design_effect"] > 3.8
    # The item primary already counts each prompt's rollouts as one block. On its 80 units,
    # the prompts' failure rates around 29/80 have S^2 = 783/6400, and the products of two
    # units' deviations within a conversation sum to -271/80 over 560 ordered pairs: a
    # negative correlation, so no design effect.
    (units,) = cohort["primaries"]
    assert units["units"] == 80
    assert units["intracluster_correlation"] == pytest.approx(
        -271 / 5481, rel=1e-12, abs=0
    )
    assert (units["design_effect"], units["effective_units"]) == (1.0, 80.0)
    assert "about 80 effective units." in found.message


def test_noisy_units_in_small_conversations_do_not_warn_an_item_primary(
    tmp_path: Path,
) -> None:
    # One conversation of 12 prompts with 20 rollouts each, failing 1 to 12 times, beside six
    # conversations of two prompts with one rollout each. The single rollouts' rates of 0 or 1
    # are far noisier than the rates of 20 rollouts, and nothing ties a conversation's prompts
    # together beyond chance: three of the six pairs agree.
    quiet = [(20, failures) for failures in range(1, 13)]
    noisy = [[(1, 1), (1, 0)], [(1, 0), (1, 0)], [(1, 1), (1, 1)], [(1, 0), (1, 1)]]
    noisy += [[(1, 0), (1, 0)], [(1, 1), (1, 0)]]
    (found,) = _found(run_analysis(_layout_inputs([quiet, *noisy]), tmp_path).diagnostics)
    # The one-way ANOVA estimate on the units' rates sets the six pairs' spread of block means
    # against the quiet units' small spread within their conversation, and gives a design
    # effect of about 2.8. The pairwise estimate does not: around the mean rate 89/240, with
    # S^2 = 7979/57600, the products of two units' deviations within a conversation sum to
    # -83/600 over 144 ordered pairs.
    assert found.severity == "INFO"
    assert found.details["primaries_splitting_blocks"] == []
    (cohort,) = found.details["cohorts"]
    (units,) = cohort["primaries"]
    assert units["units"] == 24
    assert units["intracluster_correlation"] == pytest.approx(
        -166 / 23937, rel=1e-12, abs=0
    )
    assert (units["design_effect"], units["effective_units"]) == (1.0, 24.0)


def test_a_rollout_primary_gets_the_figures_on_rollouts(tmp_path: Path) -> None:
    (found,) = _found(
        run_analysis(_inputs(PROMPT_SPREAD, primary="rollout", rollouts=10), tmp_path).diagnostics
    )
    # A [rollout] primary's units are the rollouts: its figures are those on rollouts, to the
    # bit, and it is warned about them as before.
    (cohort,) = found.details["cohorts"]
    (units,) = cohort["primaries"]
    assert units == {
        "profile": "rollout",
        "units": 800,
        "intracluster_correlation": cohort["intracluster_correlation"],
        "design_effect": cohort["design_effect"],
        "effective_units": cohort["effective_rollouts"],
    }
    assert found.severity == "WARNING"
    assert found.details["primaries_splitting_blocks"] == ["rollout"]


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


def test_disabled_direct_intervals_name_no_primary(tmp_path: Path) -> None:
    inputs = _inputs(CO_FAILING)
    inputs.plan["outputs"]["direct_mean_intervals"]["enabled"] = False
    (found,) = _found(run_analysis(inputs, tmp_path).diagnostics)
    # The plan computes no result under 'item': no comparison, no anytime bound and no direct
    # interval, so no primary splits the conversations, however much they fail together.
    assert found.severity == "INFO"
    assert found.details["primaries_splitting_blocks"] == []
    assert "'item'" not in found.message
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
