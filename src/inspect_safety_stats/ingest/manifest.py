from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import AnalysisPlan, GroupSpec, WeightSpec

REQUIRED_COLUMNS = ("prompt_id", "question_id", "task_id")


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


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"design manifest is missing required columns: {', '.join(missing)}")
    for column in REQUIRED_COLUMNS:
        if manifest[column].isna().any():
            raise ValueError(f"design manifest column {column!r} contains missing values")
        manifest[column] = manifest[column].astype(str)
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


def _filter_group(manifest: pd.DataFrame, group: GroupSpec) -> pd.DataFrame:
    selected = manifest
    for column, expected in group.filter.items():
        if column not in selected:
            raise ValueError(f"group {group.id!r} references unknown manifest column {column!r}")
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
    for group in plan.groups:
        if group.id in seen:
            raise ValueError(f"duplicate group id {group.id!r}")
        seen.add(group.id)
        selected = _filter_group(manifest, group).sort_values("prompt_id", kind="stable")
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
