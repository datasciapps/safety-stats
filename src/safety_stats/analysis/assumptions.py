"""The registry's assumptions an analysis rests on, with the plan settings each one took."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from safety_stats.config.loader import plan_defaults
from safety_stats.config.models import AnalysisPlan
from safety_stats.registry import ASSUMPTIONS

# Whether an assumption of the registry applies to a plan. Every registry entry has one, which
# tests/unit/test_assumptions_used.py checks, so a new entry cannot be left out silently.
APPLIES: dict[str, Callable[[AnalysisPlan], bool]] = {
    # A sampled design rests on probability_sample instead.
    "fixed_design": lambda plan: plan.design.type == "fixed",
    # A sampled design's intervals take the drawn frame item as the independent block, whatever
    # independence profile the plan declares, so they rest on the draw, not on the profile.
    "probability_sample": lambda plan: plan.design.type == "sampled",
    "run_stability": lambda plan: True,
    # Prompt bounds, when enabled, still take each rollout of a prompt as an independent draw.
    "rollout_independence": (
        lambda plan: plan.design.type == "fixed" or plan.outputs.prompt_bounds.enabled
    ),
    "arm_independence": lambda plan: bool(plan.comparisons or plan.comparison_sets),
    "scorer_validity": lambda plan: True,
    "judge_accuracy": lambda plan: True,
    "missing_scores": lambda plan: True,
    "sample_errors": lambda plan: True,
    "token_limit": lambda plan: True,
    "content_filter_stops": lambda plan: True,
    "complete_design": lambda plan: True,
    # An exploratory analysis does not claim its plan was fixed in advance, and cannot pass.
    "plan_fixed_in_advance": lambda plan: plan.analysis.intent == "confirmatory",
    # The adaptive-attacker bound needs no independence between runs, but rests on this.
    # rollout_independence still applies to such a plan: its risk bounds rest on it.
    "adaptive_attack_stream": lambda plan: bool(plan.adaptive_claims),
}


def assumptions_used(plan: AnalysisPlan) -> list[dict[str, Any]]:
    """The registry's assumptions that apply to the plan, in registry order.

    Each entry gives the assumption's ``id``, ``statement`` and ``if_wrong`` from the registry,
    and ``settings``: the value of each plan field that sets it, found by the entry's
    ``plan_field`` pattern, and whether that value is a default the plan left out. The
    statement and if-wrong text are the ones for the value the plan took, when the entry has
    its own for it.
    """

    defaulted = {entry["path"] for entry in plan_defaults(plan)}
    used = []
    for entry in ASSUMPTIONS.values():
        if not APPLIES[entry.id](plan):
            continue
        settings = (
            [
                {"path": path, "value": value, "default": _is_default(path, defaulted)}
                for path, value in plan_values(plan, entry.plan_field)
            ]
            if entry.plan_field is not None
            else []
        )
        # A field that matched one path is stated for its value; a pattern that matched
        # several paths, or none, keeps the entry's own statement.
        statement, if_wrong = (
            entry.stated_for(settings[0]["value"])
            if len(settings) == 1
            else (entry.statement, entry.if_wrong)
        )
        used.append(
            {
                "id": entry.id,
                "statement": statement,
                "if_wrong": if_wrong,
                "settings": settings,
            }
        )
    return used


def plan_values(plan: AnalysisPlan, pattern: str) -> list[tuple[str, Any]]:
    """Each concrete path a registry ``plan_field`` pattern matches in the plan, with its value.

    ``<id>`` stands for every key of a mapping and ``[]`` for every item of a list, as in the
    registry. Where a section on the way is None, the section's path is given with None.
    """

    found: list[tuple[str, Any]] = [("", plan)]
    for segment in pattern.split("."):
        name, items = (segment[:-2], True) if segment.endswith("[]") else (segment, False)
        following: list[tuple[str, Any]] = []
        for path, value in found:
            if value is None or not isinstance(value, BaseModel | dict):
                following.append((path, value))
            elif name == "<id>" and isinstance(value, dict):
                following.extend((f"{path}.{key}", item) for key, item in value.items())
            else:
                child = f"{path}.{name}" if path else name
                field = _field(value, name)
                if items:
                    following.extend(
                        (f"{child}[{index}]", item) for index, item in enumerate(field or [])
                    )
                else:
                    following.append((child, field))
        found = following
    return [(path, to_jsonable_python(value, by_alias=True)) for path, value in found]


def _field(value: BaseModel | dict[str, Any], name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    for attribute, info in type(value).model_fields.items():
        if name in (attribute, info.alias):
            return getattr(value, attribute)
    raise KeyError(f"{type(value).__name__} has no field {name!r}")


def _is_default(path: str, defaulted: set[str]) -> bool:
    """Whether the path, or a section it lies in, is one the plan left out."""

    return path in defaulted or any(
        path.startswith(f"{default}.") or path.startswith(f"{default}[") for default in defaulted
    )
