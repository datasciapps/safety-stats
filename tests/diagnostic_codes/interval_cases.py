"""Cases for the codes of direct-mean, hierarchy and comparison intervals and their regions."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

from safety_stats.registry import SourceKind
from tests.diagnostic_codes.harness import BASE_RUN, Case, Inputs, Outcome, Run, analysis, rollout

FAMILY = {"purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}


def _family(inputs: Inputs, family_id: str) -> None:
    inputs.plan["multiplicity"]["families"].append({"id": family_id, **FAMILY})


def _prompt(inputs: Inputs, prompt_id: str, task: str, drawn: bool, **columns: Any) -> None:
    """Add a frame item, with four safe rollouts when it was drawn."""

    number = prompt_id[1:]
    inputs.manifest.append(
        {
            "prompt_id": prompt_id,
            "question_id": f"q{number}",
            "task_id": task,
            "expected_rollouts": 4,
            **columns,
        }
    )
    if drawn:
        inputs.rollouts += [rollout(prompt_id, repeat) for repeat in range(1, 5)]


def _direct_means(inputs: Inputs, **spec: Any) -> None:
    _family(inputs, "direct")
    inputs.plan["outputs"]["direct_mean_intervals"] = {
        "enabled": True,
        "method": "weighted_kl_chernoff",
        "family": "direct",
        "independence": {"primary": "rollout"},
        **spec,
    }
    inputs.plan.setdefault("design", {}).setdefault(
        "independence_profiles", {"rollout": {"block_by": ["rollout"]}}
    )


# --- Sampled designs ----------------------------------------------------------------------


def _sampled(
    p3_drawn: bool, *, nodes: bool = False, domain_group: bool = False
) -> Callable[[Inputs], None]:
    """A Poisson sample from a frame of p1, p2 (task t1) and p3 (task t2), each at 0.5."""

    def change(inputs: Inputs) -> None:
        for row in inputs.manifest:
            row["inclusion"] = 0.5
            row["domain"] = "a"
        _prompt(inputs, "p3", "t2", p3_drawn, inclusion=0.5, domain="b")
        inputs.plan["design"] = {"type": "sampled", "inclusion_probability_column": "inclusion"}
        _direct_means(inputs, hierarchy_nodes=nodes)
        if domain_group:
            inputs.plan["groups"].append({"id": "b", "filter": {"domain": "b"}})

    return change


def _node(outcome: Outcome, level: str, value: str) -> Any:
    (found,) = [
        row
        for row in outcome.report.hierarchy_risks
        if row.level == level and row.path.get(level) == value
    ]
    return found


def _group(outcome: Outcome, group_id: str) -> Any:
    (found,) = [item for item in outcome.report.group_risks if item.group_id == group_id]
    return found


def _node_unsampled(triggered: Outcome, control: Outcome) -> None:
    # Withheld: task t2, of which nothing was drawn, has no interval; task t1 and the group
    # are reported as usual.
    assert _node(triggered, "task_id", "t2").lower is None
    assert _node(control, "task_id", "t2").lower is not None
    assert _node(triggered, "task_id", "t1").lower is not None
    assert _group(triggered, "all").lower is not None


def _no_direct_interval(triggered: Outcome, control: Outcome) -> None:
    # Invalid: group b has no direct interval, so its primary is the propagated envelope.
    assert _group(triggered, "b").primary_interval_id == "propagated_prompt_envelope"
    assert _group(control, "b").primary_interval_id != "propagated_prompt_envelope"


# --- Comparisons --------------------------------------------------------------------------


def _second_cohort(inputs: Inputs) -> None:
    inputs.plan["cohorts"]["baseline"] = {"match": {"model": "m2"}}
    inputs.runs.append({**BASE_RUN, "run_key": "r2", "model": "m2"})
    inputs.rollouts += [{**item, "run_key": "r2"} for item in list(inputs.rollouts)]


def _comparison(p3_weight: float) -> Callable[[Inputs], None]:
    """A frontier_sidak comparison over task nodes, p3 (task t2) weighted ``p3_weight``."""

    def change(inputs: Inputs) -> None:
        for row in inputs.manifest:
            row["weight"] = 1
        _prompt(inputs, "p3", "t2", True, weight=p3_weight)
        _second_cohort(inputs)
        _family(inputs, "comparisons")
        inputs.plan["design"] = {
            "weights": {"strategy": "explicit", "column": "weight"},
            "independence_profiles": {"rollout": {"block_by": ["rollout"]}},
        }
        inputs.plan["comparisons"] = [
            {
                "id": "versus",
                "candidate": "candidate",
                "baseline": "baseline",
                "group": "all",
                "measure": "mean",
                "estimand": "risk_difference",
                "family": "comparisons",
                "independence": {"primary": "rollout"},
                "hierarchy_nodes": True,
                "hierarchy_calibration": "frontier_sidak",
            }
        ]

    return change


def _frontier_direct_means(p3_weight: float) -> Callable[[Inputs], None]:
    """frontier_sidak direct means over task nodes, p3 (task t2) weighted ``p3_weight``."""

    def change(inputs: Inputs) -> None:
        for row in inputs.manifest:
            row["weight"] = 1
        _prompt(inputs, "p3", "t2", True, weight=p3_weight)
        inputs.plan["design"] = {"weights": {"strategy": "explicit", "column": "weight"}}
        _direct_means(inputs, hierarchy_nodes=True, hierarchy_calibration="frontier_sidak")

    return change


def _no_calibrated_interval(triggered: Outcome, control: Outcome) -> None:
    # Invalid: no target is calibrated, so the group keeps only the propagated envelope and
    # the remaining task node has no interval.
    assert _group(triggered, "all").primary_interval_id == "propagated_prompt_envelope"
    assert _group(control, "all").primary_interval_id != "propagated_prompt_envelope"
    node = _node(triggered, "task_id", "t1")
    assert (node.lower, node.eligibility) == (None, "invalid")
    assert _node(control, "task_id", "t1").lower is not None


def _comparison_root(outcome: Outcome) -> Any:
    (found,) = [item for item in outcome.report.comparisons if item.level is None]
    return found


def _no_node_interval(triggered: Outcome, control: Outcome) -> None:
    # Invalid: every node has no interval and its row is invalid; the group is reported.
    node, kept = _node(triggered, "task_id", "t1"), _node(control, "task_id", "t1")
    assert (node.lower, node.eligibility) == (None, "invalid")
    assert kept.lower is not None
    assert _direct(triggered) is not None


def _no_comparison_interval(triggered: Outcome, control: Outcome) -> None:
    # Invalid: the comparison gets no interval under the profile.
    assert _comparison_root(triggered).lower is None
    assert _comparison_root(control).lower is not None


# --- Codes no input reaches ---------------------------------------------------------------
#
# Every direct interval contains the completion estimate of its own blocks, and every target
# of a hierarchy is built from the same blocks, so the observed prompt risks lie in all raw
# intervals at once and no input empties a region. The intervals of a fixed design, and of a
# comparison, are computed only after checks that reject every input their engines refuse.
# These cases therefore break the assumption inside the analysis, with a patch that is
# active only while the trigger runs.


def patched(target: str, replacement: Any, run: Run) -> Run:
    def patched_run(directory: Path) -> Outcome:
        with mock.patch(target, replacement):
            return run(directory)

    return patched_run


def _disjoint(constraints: list[Any]) -> list[Any]:
    """The constraints, the first pinned to 1 and the others to 0, so that none can all hold."""

    return [
        dataclasses.replace(item, lower=float(index == 0), upper=float(index == 0))
        for index, item in enumerate(constraints)
    ]


def _empty_region(name: str) -> Callable[..., Any]:
    """The projection ``name`` of ``stats.hierarchy_region``, on disjoint constraints."""

    def projection(constraints: list[Any], *args: Any, **kwargs: Any) -> Any:
        from safety_stats.stats import hierarchy_region

        return getattr(hierarchy_region, name)(_disjoint(constraints), *args, **kwargs)

    return projection


def _refuse_nodes(**kwargs: Any) -> Any:
    """Refuse the direct interval of every hierarchy node, and compute the groups' as usual."""

    from safety_stats.analysis.direct_mean import _direct_intervals

    if not kwargs["target_id"].endswith("::root"):
        raise ValueError("refused for the case")
    return _direct_intervals(**kwargs)


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise ValueError("refused for the case")


def _raw_and_projected(triggered: Any, control: Any) -> None:
    """Withheld: the trigger's interval is the control's raw one, not projected."""

    assert triggered.projection_method is None
    assert control.projection_method is not None
    assert (triggered.lower, triggered.upper) == (control.raw_lower, control.raw_upper)


def _direct(outcome: Outcome) -> Any:
    (found,) = [
        item for item in _group(outcome, "all").intervals if item.interval_id.startswith("direct_")
    ]
    return found


def _hierarchy_raw(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the group's direct interval is reported raw rather than projected.
    _raw_and_projected(_direct(triggered), _direct(control))


def _comparison_primary(outcome: Outcome) -> Any:
    root = _comparison_root(outcome)
    return next(item for item in root.intervals if item.role == "primary")


def _comparison_raw(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the comparison's interval is reported raw rather than projected.
    _raw_and_projected(_comparison_primary(triggered), _comparison_primary(control))


def _comparison_set(inputs: Inputs) -> None:
    for row in inputs.manifest:
        row["weight"] = 1
    _second_cohort(inputs)
    _family(inputs, "comparisons")
    inputs.plan["design"] = {"independence_profiles": {"rollout": {"block_by": ["rollout"]}}}
    inputs.plan["comparison_sets"] = [
        {
            "id": "arms",
            "cohorts": ["candidate", "baseline"],
            "contrasts": {"type": "versus_reference", "reference": "baseline"},
            "group": "all",
            "measure": "mean",
            "family": "comparisons",
            "independence": {"primary": "rollout"},
        }
    ]


def _set_root(outcome: Outcome) -> Any:
    (found,) = [
        item
        for item in outcome.report.comparisons
        if item.comparison_set_id == "arms" and item.level is None
    ]
    return next(item for item in found.intervals if item.role == "primary")


def _multi_arm_unprojected(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the set's effect is not derived from the common arm region, so the row keeps
    # the interval it had.
    trigger, derived = _set_root(triggered), _set_root(control)
    assert trigger.projection_method == "comparison_two_arm_difference_hierarchy_linear_program"
    assert derived.projection_method == "multi_arm_hierarchy_linear_region"


def _nodes(inputs: Inputs) -> None:
    _direct_means(inputs, hierarchy_nodes=True)


def _paired_unscored(inputs: Inputs) -> None:
    """A comparison of the two cohorts with paired diagnostics, one baseline rollout unscored."""

    _second_cohort(inputs)
    _family(inputs, "comparisons")
    inputs.plan["design"] = {"independence_profiles": {"rollout": {"block_by": ["rollout"]}}}
    inputs.plan["comparisons"] = [
        {
            "id": "versus",
            "candidate": "candidate",
            "baseline": "baseline",
            "group": "all",
            "measure": "mean",
            "family": "comparisons",
            "independence": {"primary": "rollout"},
            "pair_key": ["prompt_id", "repeat"],
            "diagnostic_methods": ["exact_mcnemar", "newcombe_paired"],
        }
    ]
    (unscored,) = [
        item
        for item in inputs.rollouts
        if (item["run_key"], item["item_id"], item["repeat"]) == ("r2", "p1", 1)
    ]
    unscored["score_harmful"] = None


def _newcombe_withheld(triggered: Outcome, control: Outcome) -> None:
    # Withheld: newcombe_paired is not reported; exact_mcnemar and the interval are as before.
    row, kept = _comparison_root(triggered), _comparison_root(control)
    assert "newcombe_paired" not in row.diagnostic_outputs
    assert "newcombe_paired" in kept.diagnostic_outputs
    assert row.diagnostic_outputs["exact_mcnemar"] == kept.diagnostic_outputs["exact_mcnemar"]
    assert (row.lower, row.upper) == (kept.lower, kept.upper)


# The kind of input, as safety_stats.registry.SOURCE_KINDS names them, that these
# cases' triggers break. A case whose trigger breaks another kind names it as its source.
SOURCE: SourceKind = "analysis"

CASES = [
    Case(
        # Nothing of task t2 was drawn into the sample.
        code="HIERARCHY_NODE_UNSAMPLED",
        shows="withheld",
        source="manifest",
        trigger=analysis(_sampled(p3_drawn=False, nodes=True)),
        control=analysis(_sampled(p3_drawn=True, nodes=True)),
        effect=_node_unsampled,
    ),
    Case(
        # Group b's only frame item was not drawn, so the group has no sampled block.
        code="DIRECT_MEAN_INTERVAL_INVALID",
        shows="invalid",
        trigger=analysis(_sampled(p3_drawn=False, domain_group=True)),
        control=analysis(_sampled(p3_drawn=True, domain_group=True)),
        effect=_no_direct_interval,
    ),
    Case(
        # Task t2 has weight zero, so the task level does not partition the group.
        code="DIRECT_MEAN_CALIBRATION_INVALID",
        shows="invalid",
        trigger=analysis(_frontier_direct_means(p3_weight=0)),
        control=analysis(_frontier_direct_means(p3_weight=1)),
        effect=_no_calibrated_interval,
    ),
    Case(
        # Task t2 has weight zero, so the task level does not partition the group.
        code="COMPARISON_CALIBRATION_INVALID",
        shows="invalid",
        trigger=analysis(_comparison(p3_weight=0)),
        control=analysis(_comparison(p3_weight=1)),
        effect=_no_comparison_interval,
    ),
    Case(
        # No input reaches it: a patch makes the comparison's interval engine refuse.
        code="COMPARISON_INTERVAL_INVALID",
        shows="invalid",
        trigger=patched(
            "safety_stats.analysis.comparisons._difference_rectangle_from_blocks",
            _refuse,
            analysis(_comparison(p3_weight=1)),
        ),
        control=analysis(_comparison(p3_weight=1)),
        effect=_no_comparison_interval,
    ),
    Case(
        # No input reaches it: a patch makes the direct interval of every node fail.
        code="HIERARCHY_MEAN_INTERVAL_INVALID",
        shows="invalid",
        trigger=patched(
            "safety_stats.analysis.risks._direct_intervals",
            _refuse_nodes,
            analysis(_nodes),
        ),
        control=analysis(_nodes),
        effect=_no_node_interval,
    ),
    Case(
        # No input reaches it: a patch makes the raw group and node intervals disjoint.
        code="HIERARCHY_CONFIDENCE_REGION_EMPTY",
        shows="withheld",
        trigger=patched(
            "safety_stats.analysis.regions.project_hierarchy_confidence_region",
            _empty_region("project_hierarchy_confidence_region"),
            analysis(_nodes),
        ),
        control=analysis(_nodes),
        effect=_hierarchy_raw,
    ),
    Case(
        # No input reaches it: a patch makes the raw candidate-arm intervals disjoint.
        code="COMPARISON_HIERARCHY_CONFIDENCE_REGION_EMPTY",
        shows="withheld",
        trigger=patched(
            "safety_stats.analysis.comparison_regions."
            "project_two_arm_difference_hierarchy_confidence_region",
            _empty_region("project_two_arm_difference_hierarchy_confidence_region"),
            analysis(_comparison(p3_weight=1)),
        ),
        control=analysis(_comparison(p3_weight=1)),
        effect=_comparison_raw,
    ),
    Case(
        # No input reaches it: a patch makes the raw intervals of an arm disjoint.
        code="MULTI_ARM_CONFIDENCE_REGION_EMPTY",
        shows="withheld",
        trigger=patched(
            "safety_stats.analysis.comparison_regions.project_hierarchy_confidence_region",
            _empty_region("project_hierarchy_confidence_region"),
            analysis(_comparison_set),
        ),
        control=analysis(_comparison_set),
        effect=_multi_arm_unprojected,
    ),
    Case(
        # A patch lowers the limit on the tables newcombe_paired evaluates to one, below the two
        # that the unscored rollout's completions produce. Without it, the case would need 668
        # pairs unscored in both arms.
        code="PAIR_NEWCOMBE_UNAVAILABLE",
        shows="withheld",
        trigger=patched(
            "safety_stats.stats.paired._PARTIAL_NEWCOMBE_COMPLETION_LIMIT",
            1,
            analysis(_paired_unscored),
        ),
        control=analysis(_paired_unscored),
        effect=_newcombe_withheld,
    ),
]
