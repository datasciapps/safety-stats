from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import IndependenceProfileSpec
from inspect_safety_stats.schema.observations import Observation

_BLOCK_OBSERVATION_FIELDS = frozenset(
    {
        "observation_id",
        "source_log",
        "source_log_sha256",
        "eval_id",
        "run_id",
        "sample_id",
        "inspect_id",
        "epoch",
        "prompt_id",
        "question_id",
        "task_id",
        "cohort_id",
        "model",
        "configuration_fingerprint",
    }
)


@dataclass(frozen=True)
class BoundedBlocks:
    """Observed lower/upper completions for a weighted sum of independent blocks."""

    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray
    block_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.block_ids)


def validate_profile_fields(
    profiles: dict[str, IndependenceProfileSpec], manifest: pd.DataFrame
) -> None:
    available = _BLOCK_OBSERVATION_FIELDS | frozenset(str(column) for column in manifest.columns)
    for profile_id, profile in profiles.items():
        unknown = sorted(set(profile.block_by) - available - {"rollout"})
        if unknown:
            raise ValueError(
                f"independence profile {profile_id!r} references unavailable fields: "
                + ", ".join(unknown)
            )
        # A missing manifest value has no block identity: depending on the column's inferred
        # dtype it either split every affected rollout into its own block or merged all of
        # them into one. Neither is a declared partition, so require a value everywhere.
        for field in profile.block_by:
            if field in _BLOCK_OBSERVATION_FIELDS or field == "rollout":
                continue
            missing = manifest[field].isna()
            if missing.any():
                prompts = sorted(manifest.loc[missing, "prompt_id"].astype(str))
                shown = ", ".join(prompts[:5]) + (", ..." if len(prompts) > 5 else "")
                raise ValueError(
                    f"independence profile {profile_id!r} blocks by manifest column {field!r}, "
                    f"which has no value for {len(prompts)} prompt(s): {shown}. Give every "
                    "prompt a value; a prompt that forms its own block can use its prompt_id."
                )


def _value(
    observation: Observation,
    field: str,
    manifest_by_prompt: pd.DataFrame,
) -> Any:
    if field in _BLOCK_OBSERVATION_FIELDS:
        return getattr(observation, field)
    return manifest_by_prompt.at[observation.prompt_id, field]


def block_key(
    observation: Observation,
    profile: IndependenceProfileSpec,
    manifest_by_prompt: pd.DataFrame,
    *,
    comparison: bool,
) -> tuple[Any, ...]:
    if profile.block_by == ["rollout"]:
        key: tuple[Any, ...] = (observation.observation_id,)
    else:
        key = tuple(_value(observation, field, manifest_by_prompt) for field in profile.block_by)
    if comparison and profile.comparison_coupling == "independent":
        return (observation.cohort_id, *key)
    return key


def _serialized_key(key: tuple[Any, ...]) -> str:
    return "\x1f".join("<null>" if value is None else str(value) for value in key)


def direct_bounded_blocks(
    observations: list[Observation],
    manifest: pd.DataFrame,
    profile: IndependenceProfileSpec,
    *,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    prompt_weights: np.ndarray,
) -> BoundedBlocks:
    """Aggregate rollout contributions according to a declared block partition."""

    prompt_list = list(prompt_ids)
    if len(set(prompt_list)) != len(prompt_list):
        raise ValueError("prompt identifiers must be unique")
    prompt_weight_array = np.asarray(prompt_weights, dtype=float)
    if prompt_weight_array.ndim != 1 or len(prompt_list) != len(prompt_weight_array):
        raise ValueError("prompt identifiers and weights must be one-dimensional and aligned")
    if (
        not np.all(np.isfinite(prompt_weight_array))
        or np.any(prompt_weight_array < 0)
        or not math.isclose(float(np.sum(prompt_weight_array)), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("prompt weights must be nonnegative, finite, and sum to one")
    weights_by_prompt = dict(zip(prompt_list, prompt_weight_array, strict=True))
    selected = [
        row
        for row in observations
        if row.cohort_id == cohort_id and row.prompt_id in weights_by_prompt
    ]
    counts: dict[str, int] = {}
    for row in selected:
        counts[row.prompt_id] = counts.get(row.prompt_id, 0) + 1
    missing = sorted(set(prompt_list) - set(counts))
    if missing:
        raise ValueError(
            f"independence blocks cannot be formed; prompts have no rows: {missing[:5]}"
        )
    manifest_by_prompt = manifest.set_index("prompt_id", drop=False)
    accumulators: dict[tuple[Any, ...], list[float]] = {}
    for row in selected:
        coefficient = weights_by_prompt[row.prompt_id] / counts[row.prompt_id]
        if coefficient == 0:
            continue
        key = block_key(row, profile, manifest_by_prompt, comparison=False)
        values = accumulators.setdefault(key, [0.0, 0.0, 0.0])
        values[0] += coefficient
        if row.failure is True:
            values[1] += coefficient
            values[2] += coefficient
        elif row.failure is None:
            values[2] += coefficient
    ordered = sorted(accumulators.items(), key=lambda item: _serialized_key(item[0]))
    masses = np.asarray([values[0] for _, values in ordered], dtype=float)
    lower = np.asarray([values[1] / values[0] for _, values in ordered], dtype=float)
    upper = np.asarray([values[2] / values[0] for _, values in ordered], dtype=float)
    masses /= np.sum(masses)
    return BoundedBlocks(
        lower=lower,
        upper=upper,
        weights=masses,
        block_ids=tuple(_serialized_key(key) for key, _ in ordered),
    )


def comparison_contrast_blocks(
    observations: list[Observation],
    manifest: pd.DataFrame,
    profile: IndependenceProfileSpec,
    *,
    candidate: str,
    baseline: str,
    prompt_ids: tuple[str, ...] | list[str],
    prompt_weights: np.ndarray,
) -> BoundedBlocks:
    """Encode Q=(1+candidate-baseline)/2 using declared cross-arm blocks."""

    prompt_list = list(prompt_ids)
    if len(set(prompt_list)) != len(prompt_list):
        raise ValueError("prompt identifiers must be unique")
    prompt_weight_array = np.asarray(prompt_weights, dtype=float)
    if prompt_weight_array.ndim != 1 or len(prompt_list) != len(prompt_weight_array):
        raise ValueError("prompt identifiers and weights must be one-dimensional and aligned")
    if (
        not np.all(np.isfinite(prompt_weight_array))
        or np.any(prompt_weight_array < 0)
        or not math.isclose(float(np.sum(prompt_weight_array)), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("prompt weights must be nonnegative, finite, and sum to one")
    weights_by_prompt = dict(zip(prompt_list, prompt_weight_array, strict=True))
    selected = [
        row
        for row in observations
        if row.cohort_id in {candidate, baseline} and row.prompt_id in weights_by_prompt
    ]
    counts: dict[tuple[str, str], int] = {}
    for row in selected:
        key = (row.cohort_id, row.prompt_id)
        counts[key] = counts.get(key, 0) + 1
    missing = [
        f"{cohort}:{prompt}"
        for cohort in (candidate, baseline)
        for prompt in prompt_list
        if (cohort, prompt) not in counts
    ]
    if missing:
        raise ValueError(f"comparison blocks cannot be formed; missing rows: {missing[:5]}")
    manifest_by_prompt = manifest.set_index("prompt_id", drop=False)
    # Each accumulator is candidate mass/lower/upper followed by baseline mass/lower/upper.
    accumulators: dict[tuple[Any, ...], list[float]] = {}
    for row in selected:
        coefficient = weights_by_prompt[row.prompt_id] / counts[(row.cohort_id, row.prompt_id)]
        if coefficient == 0:
            continue
        key = block_key(row, profile, manifest_by_prompt, comparison=True)
        values = accumulators.setdefault(key, [0.0] * 6)
        offset = 0 if row.cohort_id == candidate else 3
        values[offset] += coefficient
        if row.failure is True:
            values[offset + 1] += coefficient
            values[offset + 2] += coefficient
        elif row.failure is None:
            values[offset + 2] += coefficient
    ordered = sorted(accumulators.items(), key=lambda item: _serialized_key(item[0]))
    masses: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for _, (cm, cl, cu, bm, bl, bu) in ordered:
        mass = (cm + bm) / 2
        masses.append(mass)
        lowers.append((bm + cl - bu) / (2 * mass))
        uppers.append((bm + cu - bl) / (2 * mass))
    weights = np.asarray(masses, dtype=float)
    weights /= np.sum(weights)
    return BoundedBlocks(
        lower=np.asarray(lowers, dtype=float),
        upper=np.asarray(uppers, dtype=float),
        weights=weights,
        block_ids=tuple(_serialized_key(key) for key, _ in ordered),
    )
