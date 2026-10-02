"""Upgrade analysis plans written for an earlier plan schema.

Plans are frozen and hashed, so the loader never rewrites one silently. A plan that
declares an older ``schema_version`` is refused with a pointer to
``inspect-safety-stats migrate-plan``, which writes the upgraded plan to a new file
for the author to review and re-freeze.
"""

from __future__ import annotations

import copy
from typing import Any

CURRENT_SCHEMA_VERSION = "2.0"
SUPPORTED_LEGACY_VERSIONS = frozenset({"1.0"})

_REMOVED_COMPARISON_FIELDS = (
    "pair_key",
    "diagnostic_methods",
    "cluster_key",
    "bootstrap_replicates",
    "bootstrap_seed",
)
_ROLLOUT_PROFILE = {"block_by": ["rollout"], "comparison_coupling": "independent"}
_PROMPT_PROFILE = {"block_by": ["prompt_id"], "comparison_coupling": "joint"}


class PlanSchemaError(ValueError):
    """A plan declares a schema version this release cannot load directly."""


def declared_schema_version(raw: dict[str, Any]) -> str:
    value = raw.get("schema_version", CURRENT_SCHEMA_VERSION)
    return str(value)


def check_schema_version(raw: dict[str, Any]) -> None:
    """Refuse a plan whose schema is not current, explaining how to proceed."""

    version = declared_schema_version(raw)
    if version == CURRENT_SCHEMA_VERSION:
        return
    if version in SUPPORTED_LEGACY_VERSIONS:
        raise PlanSchemaError(
            f"this plan declares schema_version {version!r}; this release reads "
            f"{CURRENT_SCHEMA_VERSION!r}. Upgrade it with "
            "`inspect-safety-stats migrate-plan --plan <old> --output <new>`, review the "
            "result, and re-freeze it before analysis."
        )
    raise PlanSchemaError(
        f"unsupported plan schema_version {version!r}; this release reads "
        f"{CURRENT_SCHEMA_VERSION!r} and can migrate "
        + ", ".join(repr(item) for item in sorted(SUPPORTED_LEGACY_VERSIONS))
    )


def _selection_from_legacy(
    section: dict[str, Any],
    location: str,
    used_profiles: set[str],
    notes: list[str],
) -> None:
    unit = section.pop("independence_unit", None)
    cluster_sensitivity = bool(section.pop("cluster_sensitivity", False))
    if unit is None and not cluster_sensitivity:
        return
    if "independence" in section:
        notes.append(
            f"{location}: kept the existing independence selection and dropped the legacy "
            "independence_unit/cluster_sensitivity fields"
        )
        return
    primary = "prompt" if unit == "prompt" else "rollout"
    sensitivities = ["prompt"] if cluster_sensitivity and primary != "prompt" else []
    section["independence"] = {"primary": primary, "sensitivities": sensitivities}
    used_profiles.add(primary)
    used_profiles.update(sensitivities)
    notes.append(
        f"{location}: independence_unit={unit or 'rollout'!r}"
        + (", cluster_sensitivity=true" if cluster_sensitivity else "")
        + f" became independence.primary={primary!r}"
        + (", sensitivities=['prompt']" if sensitivities else "")
    )


def migrate_plan(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return an upgraded copy of a legacy plan and a list of what changed."""

    version = declared_schema_version(raw)
    if version == CURRENT_SCHEMA_VERSION:
        return copy.deepcopy(raw), ["plan already uses the current schema; nothing to do"]
    if version not in SUPPORTED_LEGACY_VERSIONS:
        raise PlanSchemaError(f"cannot migrate plan schema_version {version!r}")

    plan = copy.deepcopy(raw)
    notes: list[str] = []
    used_profiles: set[str] = set()

    outputs = plan.get("outputs") or {}
    direct = outputs.get("direct_mean_intervals")
    if isinstance(direct, dict):
        _selection_from_legacy(direct, "outputs.direct_mean_intervals", used_profiles, notes)
        if direct.get("enabled") and "independence" not in direct:
            direct["independence"] = {"primary": "rollout", "sensitivities": []}
            used_profiles.add("rollout")
            notes.append(
                "outputs.direct_mean_intervals: no independence unit was declared, so the "
                "rollout profile became primary"
            )

    for collection in ("comparisons", "comparison_sets"):
        for index, item in enumerate(plan.get(collection) or []):
            if not isinstance(item, dict):
                continue
            location = f"{collection}[{item.get('id', index)}]"
            _selection_from_legacy(item, location, used_profiles, notes)
            if "independence" not in item:
                item["independence"] = {"primary": "rollout", "sensitivities": []}
                used_profiles.add("rollout")
                notes.append(f"{location}: the rollout profile became primary")
            removed = [field for field in _REMOVED_COMPARISON_FIELDS if field in item]
            for field in removed:
                item.pop(field)
            if removed:
                notes.append(
                    f"{location}: removed paired/cluster diagnostic fields "
                    + ", ".join(removed)
                    + " (no longer part of the package)"
                )

    design = plan.setdefault("design", {})
    profiles = design.setdefault("independence_profiles", {})
    for profile_id in sorted(used_profiles):
        if profile_id in profiles:
            continue
        profiles[profile_id] = dict(
            _ROLLOUT_PROFILE if profile_id == "rollout" else _PROMPT_PROFILE
        )
        notes.append(f"design.independence_profiles: added the {profile_id!r} profile")

    plan["schema_version"] = CURRENT_SCHEMA_VERSION
    notes.append(f"schema_version {version!r} -> {CURRENT_SCHEMA_VERSION!r}")
    return plan, notes
