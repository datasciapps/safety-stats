from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import AnalysisPlan, GroupSpec, WeightSpec
from inspect_safety_stats.schema.observations import observation_field
from inspect_safety_stats.stats.independence import _BLOCK_OBSERVATION_FIELDS

REQUIRED_COLUMNS = ("prompt_id", "question_id", "task_id")
# Every observation and prompt estimate carries a question and a task. A manifest that omits
# one of these levels, because its plan never names it, gives each prompt a level of its own.
_PROMPT_LEVEL_COLUMNS = ("question_id", "task_id")
# Columns that load_manifest converts to numbers or booleans itself; they are never labels.
_TYPED_COLUMNS = frozenset({"expected_rollouts", "mandatory", "zero_tolerance"})
# How many blank rows an error names before it counts the rest.
_SHOWN_ROWS = 5


@dataclass(frozen=True)
class ResolvedGroup:
    id: str
    title: str
    prompt_ids: tuple[str, ...]
    weights: dict[str, float]


def load_inclusion_probabilities(manifest: pd.DataFrame, column: str) -> np.ndarray:
    """Read declared inclusion probabilities for a sampled design frame."""

    if column not in manifest:
        raise ValueError(f"design manifest has no inclusion probability column {column!r}")
    values = pd.to_numeric(manifest[column], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"inclusion probabilities in {column!r} must be finite")
    if np.any(values <= 0) or np.any(values > 1):
        raise ValueError(
            f"every inclusion probability in {column!r} must lie in (0, 1]; an item that "
            "can never be drawn leaves its share of the estimand unidentified"
        )
    return values


def load_selection(manifest: pd.DataFrame, column: str) -> np.ndarray:
    """Read which frame items a sampled design's draw selected, from a true/false column."""

    if column not in manifest:
        raise ValueError(
            f"design manifest has no selection column {column!r}, which "
            "design.selection_column names; add it, marking each frame item the draw selected "
            "true and every other false"
        )
    values = manifest[column]
    if values.isna().any():
        raise ValueError(
            f"selection column {column!r} has missing values; mark every frame item true or "
            "false"
        )
    try:
        return np.asarray([_parse_bool(value) for value in values], dtype=bool)
    except ValueError as exc:
        raise ValueError(f"selection column {column!r}: {exc}") from exc


def plan_manifest_columns(plan: AnalysisPlan) -> tuple[str, ...]:
    """The columns of ``REQUIRED_COLUMNS`` the plan names, ``prompt_id`` always among them.

    A plan names a column through its hierarchy, a nested weight hierarchy, an explicit weight
    column, a group filter, an independence profile, a comparison's pair or cluster key, a
    sampling column, a threshold column or the zero-tolerance field. A level it never names takes
    the prompt's own identity.
    """

    named: set[str] = set(plan.design.hierarchy)
    for spec in [plan.design.weights, *(group.weights for group in plan.groups)]:
        if spec is None:
            continue
        if spec.strategy == "nested_equal":
            named.update(spec.hierarchy)
        if spec.column is not None:
            named.add(spec.column)
    for group in plan.groups:
        named.update(group.filter)
    for profile in plan.design.independence_profiles.values():
        named.update(profile.block_by)
    for comparison in plan.expanded_comparisons():
        # A pair or cluster key may name an identifier the observations carry.
        named.update(observation_field(name) for name in comparison.pair_key or [])
        if comparison.cluster_key:
            named.add(observation_field(comparison.cluster_key))
    for column in (
        plan.design.inclusion_probability_column,
        plan.design.stratum_column,
        plan.policy.zero_tolerance.manifest_field,
        *(measure.threshold_column for measure in plan.outputs.risk_measures),
    ):
        if column is not None:
            named.add(column)
    return tuple(column for column in REQUIRED_COLUMNS if column == "prompt_id" or column in named)


def plan_grouping_columns(plan: AnalysisPlan) -> tuple[str, ...]:
    """The manifest columns whose values the plan uses as labels, to group or identify prompts.

    They are the levels of ``design.hierarchy`` and of every ``nested_equal`` weight
    hierarchy, the columns of a group ``filter``, the manifest columns of an independence
    profile's ``block_by`` and ``design.stratum_column``. A column the plan or the loader reads
    as a number or a boolean is not one: an inclusion probability, a weight or threshold
    column, the zero-tolerance field, ``expected_rollouts`` or ``mandatory``. The identifier
    columns of ``REQUIRED_COLUMNS`` are labels whether or not the plan names them.
    """

    named: set[str] = set(plan.design.hierarchy)
    numeric: set[str | None] = {
        plan.design.inclusion_probability_column,
        plan.policy.zero_tolerance.manifest_field,
        *(measure.threshold_column for measure in plan.outputs.risk_measures),
    }
    for spec in [plan.design.weights, *(group.weights for group in plan.groups)]:
        if spec is None:
            continue
        if spec.strategy == "nested_equal":
            named.update(spec.hierarchy)
        numeric.add(spec.column)
    for group in plan.groups:
        named.update(group.filter)
    for profile in plan.design.independence_profiles.values():
        # The other fields come from the observations, as validate_profile_fields reads them.
        named.update(
            field
            for field in profile.block_by
            if field != "rollout" and observation_field(field) not in _BLOCK_OBSERVATION_FIELDS
        )
    if plan.design.stratum_column is not None:
        named.add(plan.design.stratum_column)
    return tuple(sorted(named - numeric - _TYPED_COLUMNS))


def check_zero_tolerance_field(plan: AnalysisPlan, manifest: pd.DataFrame) -> None:
    """Refuse a manifest that lacks the zero-tolerance column the plan names.

    The default field is optional: a manifest without it flags no prompt. A plan that sets
    ``policy.zero_tolerance.manifest_field`` itself asks for the gate, and a manifest that
    lacks the column would silently switch the gate off.
    """

    spec = plan.policy.zero_tolerance
    if spec.enabled and "manifest_field" in spec.model_fields_set:
        if spec.manifest_field not in manifest.columns:
            raise ValueError(
                f"design manifest has no column {spec.manifest_field!r}, which the plan names "
                "as policy.zero_tolerance.manifest_field; without it no prompt is held to zero "
                "tolerance. Add the column or set policy.zero_tolerance.enabled: false"
            )


def load_manifest(
    path: str | Path,
    required: Iterable[str] = REQUIRED_COLUMNS,
    grouping: Iterable[str] = (),
) -> pd.DataFrame:
    """Read a design manifest that has every ``required`` column.

    ``required`` is a subset of ``REQUIRED_COLUMNS`` and always holds ``prompt_id``; the
    analysis passes ``plan_manifest_columns(plan)``. An identifier column the manifest omits
    takes each prompt's ``prompt_id``.

    The identifier columns of ``REQUIRED_COLUMNS`` and the ``grouping`` columns present are
    read as text, exactly as written; the analysis passes ``plan_grouping_columns(plan)``. With
    pandas' type inference, labels such as ``1`` and ``01``, ``1`` and ``1.0``, or ``true`` and
    ``True`` would read as one value, and ``NA`` or ``None`` as a missing one, whose handling
    differs between pandas versions. A blank value in one of these columns raises
    ``ValueError`` naming the column and the rows. Every other column keeps pandas' reading, so
    numeric columns stay numbers.
    """

    required = tuple(required)
    unknown = sorted(set(required) - set(REQUIRED_COLUMNS))
    if unknown or "prompt_id" not in required:
        raise ValueError(
            f"required manifest columns must include prompt_id and come from {REQUIRED_COLUMNS}"
        )
    candidates = [
        column
        for column in dict.fromkeys([*REQUIRED_COLUMNS, *grouping])
        if column not in _TYPED_COLUMNS
    ]
    # The labels are not inferred even here. In a large file pandas infers a type chunk by
    # chunk, and warns of a column whose chunks disagree, as `1` and `01` with `x` can.
    manifest = pd.read_csv(path, dtype=dict.fromkeys(candidates, str))
    missing = [column for column in REQUIRED_COLUMNS if column in required]
    missing = [column for column in missing if column not in manifest.columns]
    if missing:
        raise ValueError(f"design manifest is missing required columns: {', '.join(missing)}")
    labels = [column for column in candidates if column in manifest.columns]
    # The file is read a second time, wholly as text and with no missing-value markers, so that
    # every other column keeps exactly the values pandas' inference gives it.
    text = pd.read_csv(path, dtype=str, keep_default_na=False)
    # ORDER MATTERS: prompt_id comes first in labels, so a blank in a later column can name
    # each of its rows by a prompt_id that is known not to be blank.
    for column in labels:
        cells = text[column]
        blank = cells.isna() | (cells.str.strip() == "")
        if blank.any():
            raise ValueError(_blank_message(text, column, blank.to_numpy(dtype=bool)))
        manifest[column] = cells
    for column in _PROMPT_LEVEL_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = manifest["prompt_id"]
    if manifest["prompt_id"].duplicated().any():
        duplicates = sorted(manifest.loc[manifest["prompt_id"].duplicated(), "prompt_id"].unique())
        raise ValueError(f"design manifest has duplicate prompt_id values: {duplicates[:10]}")
    if "expected_rollouts" in manifest:
        values = pd.to_numeric(manifest["expected_rollouts"], errors="raise")
        if np.any(values <= 0) or np.any(values % 1 != 0):
            raise ValueError("expected_rollouts must contain positive integers")
        manifest["expected_rollouts"] = values.astype(int)
    for column in ("mandatory", "zero_tolerance"):
        if column in manifest:
            manifest[column] = manifest[column].map(_parse_bool)
    return manifest


def _blank_message(text: pd.DataFrame, column: str, blank: np.ndarray) -> str:
    rows = np.flatnonzero(blank)
    shown = []
    for position in rows[:_SHOWN_ROWS]:
        # The header is row 1, as a spreadsheet numbers it.
        row = f"row {position + 2}"
        if column != "prompt_id" and "prompt_id" in text.columns:
            row += f" (prompt_id {text['prompt_id'].iat[position]!r})"
        shown.append(row)
    more = f" and {len(rows) - _SHOWN_ROWS} more" if len(rows) > _SHOWN_ROWS else ""
    return (
        f"design manifest column {column!r} is blank in {', '.join(shown)}{more}, counting "
        "the header as row 1. The analysis reads this column as labels that identify or group "
        "prompts, so give every row a value"
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"invalid boolean manifest value {value!r}")


def _filter_group(
    manifest: pd.DataFrame, group: GroupSpec, labels: Iterable[str] = ()
) -> pd.DataFrame:
    """The prompts of ``manifest`` that match every value of ``group.filter``.

    A column in ``labels`` holds text as written, and is matched only by a value written as
    text. A YAML value that is not text has already lost how it was written: ``01`` and ``1``
    both load as the number 1, and ``yes`` as true.
    """

    labels = set(labels)
    selected = manifest
    for column, expected in group.filter.items():
        if column not in selected:
            raise ValueError(f"group {group.id!r} references unknown manifest column {column!r}")
        if column in labels and not isinstance(expected, str):
            raise ValueError(
                f"group {group.id!r} filters manifest column {column!r} on {expected!r}, which "
                "is not text. The analysis reads the column's values as written, so write the "
                f"value in quotes, as the manifest writes it, for example {{{column}: '...'}}"
            )
        selected = selected.loc[selected[column] == expected]
    if selected.empty:
        raise ValueError(f"group {group.id!r} resolves to no prompts")
    return selected.copy()


def _nested_weights(frame: pd.DataFrame, hierarchy: list[str]) -> np.ndarray:
    for column in hierarchy:
        if column not in frame:
            raise ValueError(f"nested weights reference unknown manifest column {column!r}")
    if not hierarchy or hierarchy[-1] != "prompt_id":
        raise ValueError("nested_equal hierarchy must end with prompt_id")
    result = np.ones(len(frame), dtype=float)
    parent_columns: list[str] = []
    for column in hierarchy:
        if parent_columns:
            child_counts = frame.groupby(parent_columns, dropna=False)[column].transform("nunique")
        else:
            child_counts = pd.Series(frame[column].nunique(), index=frame.index)
        result /= child_counts.to_numpy(dtype=float)
        parent_columns.append(column)
    return np.asarray(result / np.sum(result), dtype=float)


def resolve_weights(frame: pd.DataFrame, spec: WeightSpec) -> np.ndarray:
    if spec.strategy == "equal_prompt":
        return np.full(len(frame), 1.0 / len(frame))
    if spec.strategy == "nested_equal":
        return _nested_weights(frame, spec.hierarchy)
    if spec.column is None:
        raise ValueError("explicit weights require a manifest column")
    if spec.column not in frame:
        raise ValueError(f"explicit weight column {spec.column!r} is absent")
    values = pd.to_numeric(frame[spec.column], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.sum(values) <= 0:
        raise ValueError("explicit weights must be finite, nonnegative, and have positive total")
    return np.asarray(values / np.sum(values), dtype=float)


def resolve_groups(plan: AnalysisPlan, manifest: pd.DataFrame) -> list[ResolvedGroup]:
    resolved: list[ResolvedGroup] = []
    seen: set[str] = set()
    labels = {*REQUIRED_COLUMNS, *plan_grouping_columns(plan)}
    for group in plan.groups:
        if group.id in seen:
            raise ValueError(f"duplicate group id {group.id!r}")
        seen.add(group.id)
        selected = _filter_group(manifest, group, labels).sort_values("prompt_id", kind="stable")
        weights = resolve_weights(selected, group.weights or plan.design.weights)
        resolved.append(
            ResolvedGroup(
                id=group.id,
                title=group.title or group.id,
                prompt_ids=tuple(selected["prompt_id"].tolist()),
                weights={
                    prompt_id: float(weight)
                    for prompt_id, weight in zip(selected["prompt_id"], weights, strict=True)
                },
            )
        )
    return resolved
