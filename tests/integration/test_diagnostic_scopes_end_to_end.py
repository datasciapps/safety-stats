"""A diagnostic's scope names the item it concerns, as the analysis emits it.

An ERROR scoped to a whole cohort (``cohorts/<id>``) withholds every prompt of that cohort from
the results built on prompt bounds: hierarchy nodes, direct means and comparisons
(``analysis.estimates.prompt_blocked``). An error about one group's measure, node or anytime
bound must therefore be scoped below the group, or it withholds them all. These tests run the
triggers in ``tests/diagnostic_codes``, which reach each emitting site, and check the scope each
site writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from inspect_safety_stats.analysis.estimates import prompt_blocked
from inspect_safety_stats.schema.scopes import scope_segments
from tests.diagnostic_codes import interval_cases, judge_analysis_cases, logs_cases, plan_cases
from tests.diagnostic_codes.harness import Inputs, base_inputs, run_analysis

CASES = {
    case.code: case
    for module in (interval_cases, judge_analysis_cases, logs_cases, plan_cases)
    for case in module.CASES
}

# Every ERROR the analysis scopes to one group of a cohort, or to an item below it.
GROUP_ERRORS = (
    "ANYTIME_BOUND_INVALID",
    "ANYTIME_ORDER_UNRECORDED",
    "DIRECT_MEAN_INTERVAL_INVALID",
    "HIERARCHY_MEAN_INTERVAL_INVALID",
    "RISK_MEASURE_INVALID",
)


@pytest.mark.parametrize("code", GROUP_ERRORS)
def test_an_error_about_a_group_withholds_none_of_its_cohort_s_prompts(
    code: str, tmp_path: Path
) -> None:
    triggered = CASES[code].trigger(tmp_path)
    found = triggered.with_code(code)
    assert found and {item.severity for item in found} == {"ERROR"}
    for item in found:
        assert item.scope is not None
        root, cohort, collection, *_ = scope_segments(item.scope)
        assert (root, collection) == ("cohorts", "groups"), item.scope
        prompts = [
            estimate.prompt_id
            for estimate in triggered.report.prompt_estimates
            if estimate.cohort_id == cohort
        ]
        assert prompts
        diagnostics = list(triggered.diagnostics)
        assert [p for p in prompts if prompt_blocked(diagnostics, cohort, p)] == []


def _nodes_and_measures(*measures: dict[str, object]) -> Inputs:
    """The base analysis with direct-mean intervals for its hierarchy nodes, and ``measures``."""

    inputs = base_inputs()
    inputs.plan["multiplicity"]["families"].append(
        {
            "id": "direct",
            "purpose": "simultaneous_intervals",
            "method": "bonferroni",
            "alpha": 0.05,
        }
    )
    inputs.plan["outputs"]["direct_mean_intervals"] = {
        "enabled": True,
        "method": "weighted_kl_chernoff",
        "family": "direct",
        "independence": {"primary": "rollout"},
        "hierarchy_nodes": True,
    }
    inputs.plan["design"] = {"independence_profiles": {"rollout": {"block_by": ["rollout"]}}}
    inputs.plan["outputs"]["risk_measures"] = [dict(measure) for measure in measures]
    return inputs


def _refuse(*args: object, **kwargs: object) -> NoReturn:
    raise ValueError("refused for the test")


MEAN = {"id": "mean", "type": "weighted_mean"}


@pytest.mark.parametrize(
    ("code", "measure", "patch"),
    [
        # The mean of the top three prompts, of two.
        ("RISK_MEASURE_INVALID", {"id": "top", "type": "top_k_mean", "k": 3}, None),
        # No input reaches it: the direct interval of two attempts refuses.
        (
            "DIRECT_MEAN_INTERVAL_INVALID",
            {"id": "top", "type": "worst_of_k", "k": 2, "direct_interval": True},
            "inspect_safety_stats.analysis.direct_mean.worst_of_k_kl_chernoff_bounds",
        ),
    ],
)
def test_an_invalid_measure_leaves_the_node_intervals_built_on_the_cohort_s_prompts(
    code: str,
    measure: dict[str, object],
    patch: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The control is the plan without the invalid measure, or, where a patch makes the measure
    # invalid, the plan unpatched: a measure with a direct interval shares the nodes' family.
    control_measures = [MEAN] if patch is None else [MEAN, measure]
    control = run_analysis(_nodes_and_measures(*control_measures), tmp_path / "control")
    if patch is not None:
        monkeypatch.setattr(patch, _refuse)
    triggered = run_analysis(_nodes_and_measures(MEAN, measure), tmp_path / "trigger")

    (error,) = triggered.with_code(code)
    assert error.scope == "cohorts/candidate/groups/all/measures/top"
    assert error.item == "/group_risks/1"
    assert control.status == "valid"
    nodes = triggered.report.hierarchy_risks
    assert nodes and all(node.lower is not None for node in nodes)
    assert nodes == control.report.hierarchy_risks


@pytest.mark.parametrize(
    ("code", "scopes"),
    [
        (
            # One per corrected weighted-mean row; the case has one, the candidate's mean.
            "JUDGE_LANG_REICZIGEL_UNAVAILABLE",
            ["cohorts/candidate/groups/all/measures/mean"],
        ),
        (
            # One per comparison target, method and profile.
            "COMPARISON_INTERVAL_INVALID",
            [
                "comparisons/versus/levels/question_id/nodes/task_id=t1%2Fquestion_id=q1"
                "/methods/weighted_kl_chernoff/profiles/rollout",
                "comparisons/versus/levels/question_id/nodes/task_id=t1%2Fquestion_id=q2"
                "/methods/weighted_kl_chernoff/profiles/rollout",
                "comparisons/versus/levels/question_id/nodes/task_id=t2%2Fquestion_id=q3"
                "/methods/weighted_kl_chernoff/profiles/rollout",
                "comparisons/versus/levels/task_id/nodes/task_id=t1"
                "/methods/weighted_kl_chernoff/profiles/rollout",
                "comparisons/versus/levels/task_id/nodes/task_id=t2"
                "/methods/weighted_kl_chernoff/profiles/rollout",
                "comparisons/versus/methods/weighted_kl_chernoff/profiles/rollout",
            ],
        ),
        (
            # One per hierarchy node.
            "HIERARCHY_MEAN_INTERVAL_INVALID",
            [
                "cohorts/candidate/groups/all/levels/question_id/nodes/task_id=t1%2Fquestion_id=q1",
                "cohorts/candidate/groups/all/levels/question_id/nodes/task_id=t1%2Fquestion_id=q2",
                "cohorts/candidate/groups/all/levels/task_id/nodes/task_id=t1",
            ],
        ),
    ],
)
def test_a_scope_names_the_row_method_and_profile_it_concerns(
    code: str, scopes: list[str], tmp_path: Path
) -> None:
    found = CASES[code].trigger(tmp_path).with_code(code)
    assert sorted(item.scope or "" for item in found) == sorted(scopes)
