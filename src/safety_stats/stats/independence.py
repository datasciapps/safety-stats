from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.models import IndependenceProfileSpec
from inspect_safety_stats.schema.observations import (
    LEGACY_FIELD_NAMES,
    Observation,
    observation_field,
)

_BLOCK_OBSERVATION_FIELDS = frozenset(
    {
        "observation_id",
        "source",
        "source_sha256",
        "run_key",
        "run_id",
        "rollout_id",
        "item_id",
        "repeat",
        "prompt_id",
        "question_id",
        "task_id",
        "cohort_id",
        "model",
        "configuration_fingerprint",
    }
)
# The observation fields report schema 3.0 renamed, by their new names, with their old ones.
_SCHEMA_2_NAMES = {new: old for old, new in LEGACY_FIELD_NAMES.items()}
_RENAMED_OBSERVATION_FIELDS = frozenset(_SCHEMA_2_NAMES)


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


@dataclass(frozen=True)
class BlockIndex:
    """Where each (cohort, prompt) sits among the observations, and manifest values by prompt.

    An analysis forms blocks for every hierarchy node, profile and comparison. Building this
    once spares each of those calls a scan of every observation and a manifest reindex.
    """

    observations: Sequence[Observation]
    manifest: pd.DataFrame
    positions: Mapping[tuple[str, str], tuple[int, ...]]
    manifest_values: Mapping[str, Mapping[str, Any]]

    def rows(self, cohorts: set[str], prompts: Mapping[str, Any]) -> list[Observation]:
        """Observations of the given cohorts and prompts, in their original order.

        ORDER MATTERS: block masses are float sums accumulated in this order, and blocks
        whose keys serialize alike keep it through a stable sort. Returning the rows in any
        other order would change the last bits of the bounds.
        """
        positions = sorted(
            position
            for cohort in cohorts
            for prompt in prompts
            for position in self.positions.get((cohort, prompt), ())
        )
        return [self.observations[position] for position in positions]


def block_index(observations: Sequence[Observation], manifest: pd.DataFrame) -> BlockIndex:
    positions: dict[tuple[str, str], list[int]] = {}
    for position, row in enumerate(observations):
        positions.setdefault((row.cohort_id, row.prompt_id), []).append(position)
    indexed = manifest.set_index("prompt_id", drop=False)
    # to_numpy() yields the same scalar types as DataFrame.at, which block keys used before:
    # numpy scalars for numeric and boolean columns, the stored objects otherwise.
    columns = {str(column): indexed[column].to_numpy() for column in indexed.columns}
    manifest_values = {
        str(prompt_id): {column: values[row] for column, values in columns.items()}
        for row, prompt_id in enumerate(indexed.index)
    }
    return BlockIndex(
        observations=observations,
        manifest=manifest,
        positions={key: tuple(value) for key, value in positions.items()},
        manifest_values=manifest_values,
    )


def _current_index(
    index: BlockIndex | None, observations: Sequence[Observation], manifest: pd.DataFrame
) -> BlockIndex:
    if index is not None and index.observations is observations and index.manifest is manifest:
        return index
    return block_index(observations, manifest)


def validate_profile_fields(
    profiles: dict[str, IndependenceProfileSpec], manifest: pd.DataFrame
) -> None:
    available = (
        _BLOCK_OBSERVATION_FIELDS
        | frozenset(LEGACY_FIELD_NAMES)
        | frozenset(str(column) for column in manifest.columns)
    )
    for profile_id, profile in profiles.items():
        unknown = sorted(set(profile.block_by) - available - {"rollout"})
        if unknown:
            raise ValueError(
                f"independence profile {profile_id!r} references unavailable fields: "
                + ", ".join(unknown)
            )
        # Report schema 3.0 gave six observation fields neutral names. A manifest column that
        # already had one of them was what block_by meant, and the observation field now takes
        # its place, which would change the blocks without a word. Refuse the ambiguous name.
        for field in profile.block_by:
            if field in _RENAMED_OBSERVATION_FIELDS and field in manifest.columns:
                raise ValueError(
                    f"independence profile {profile_id!r} blocks by {field!r}, which names both "
                    "a design manifest column and, since report schema 3.0, the observation "
                    f"field once called {_SCHEMA_2_NAMES[field]!r}. Rename the manifest column "
                    f"to block by it, or write {_SCHEMA_2_NAMES[field]!r} to block by the "
                    "observation field."
                )
        # A missing manifest value has no block identity: depending on the column's inferred
        # dtype it either split every affected rollout into its own block or merged all of
        # them into one. Neither is a declared partition, so require a value everywhere.
        for field in profile.block_by:
            if observation_field(field) in _BLOCK_OBSERVATION_FIELDS or field == "rollout":
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
    manifest_values: Mapping[str, Mapping[str, Any]],
) -> Any:
    if observation_field(field) in _BLOCK_OBSERVATION_FIELDS:
        return getattr(observation, observation_field(field))
    return manifest_values[observation.prompt_id][field]


def block_key(
    observation: Observation,
    profile: IndependenceProfileSpec,
    manifest_values: Mapping[str, Mapping[str, Any]],
    *,
    comparison: bool,
) -> tuple[Any, ...]:
    if profile.block_by == ["rollout"]:
        key: tuple[Any, ...] = (observation.observation_id,)
    else:
        key = tuple(_value(observation, field, manifest_values) for field in profile.block_by)
    if comparison and profile.comparison_coupling == "independent":
        return (observation.cohort_id, *key)
    return key


def _serialized_key(key: tuple[Any, ...]) -> str:
    return "\x1f".join("<null>" if value is None else str(value) for value in key)


# The observation fields that normalization copies from the prompt's manifest row, as text, so
# that every rollout of a prompt carries the same value.
_PROMPT_LEVEL_OBSERVATION_FIELDS = frozenset({"prompt_id", "question_id", "task_id"})


def prompt_level_block_ids(
    profile: IndependenceProfileSpec,
    manifest_values: Mapping[str, Mapping[str, Any]],
    *,
    prompt_ids: Sequence[str],
    prompt_weights: np.ndarray,
    comparison_arms: tuple[str, str] | None = None,
) -> frozenset[str] | None:
    """The blocks that any complete ledger of these prompts forms, when the prompts decide them.

    A profile that blocks only by manifest columns and by ``prompt_id``, ``question_id`` or
    ``task_id``, which normalization copies from the manifest, puts every rollout of a prompt in
    the block its manifest row names. The blocks of a ledger that has a row of every prompt are
    then known before any rollout is: they are the ``block_ids`` that ``direct_bounded_blocks``
    gives for one cohort or, with ``comparison_arms`` as (candidate, baseline), that
    ``comparison_contrast_blocks`` gives. Prompts of zero weight form no block, as there. Under
    any other profile a block depends on the rollouts themselves (their identity, run or
    repeat), and the result is None. ``manifest_values`` is ``BlockIndex.manifest_values``.
    """

    if profile.block_by == ["rollout"] or any(
        observation_field(field) in _BLOCK_OBSERVATION_FIELDS
        and observation_field(field) not in _PROMPT_LEVEL_OBSERVATION_FIELDS
        for field in profile.block_by
    ):
        return None
    prefixes: list[tuple[Any, ...]] = [()]
    if comparison_arms is not None and profile.comparison_coupling == "independent":
        prefixes = [(cohort,) for cohort in comparison_arms]
    keys: set[str] = set()
    weights = np.asarray(prompt_weights, dtype=float)
    for prompt_id, weight in zip(prompt_ids, weights, strict=True):
        if weight == 0:
            continue
        values = manifest_values[prompt_id]
        # As block_key reads them: a prompt-level observation field as the observation holds
        # it, the manifest's value as text, and any other field as the manifest holds it.
        key = tuple(
            str(values[observation_field(field)])
            if observation_field(field) in _PROMPT_LEVEL_OBSERVATION_FIELDS
            else values[field]
            for field in profile.block_by
        )
        keys.update(_serialized_key((*prefix, *key)) for prefix in prefixes)
    return frozenset(keys)


def direct_bounded_blocks(
    observations: list[Observation],
    manifest: pd.DataFrame,
    profile: IndependenceProfileSpec,
    *,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    prompt_weights: np.ndarray,
    index: BlockIndex | None = None,
) -> BoundedBlocks:
    """Aggregate rollout contributions according to a declared block partition.

    Pass an ``index`` built by :func:`block_index` from the same observations and manifest
    to avoid rebuilding it on every call.
    """

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
    index = _current_index(index, observations, manifest)
    selected = index.rows({cohort_id}, weights_by_prompt)
    counts: dict[str, int] = {}
    for row in selected:
        counts[row.prompt_id] = counts.get(row.prompt_id, 0) + 1
    missing = sorted(set(prompt_list) - set(counts))
    if missing:
        raise ValueError(
            f"independence blocks cannot be formed; prompts have no rows: {missing[:5]}"
        )
    accumulators: dict[tuple[Any, ...], list[float]] = {}
    for row in selected:
        coefficient = weights_by_prompt[row.prompt_id] / counts[row.prompt_id]
        if coefficient == 0:
            continue
        key = block_key(row, profile, index.manifest_values, comparison=False)
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


def prompt_outcome_counts(
    observations: list[Observation],
    manifest: pd.DataFrame,
    *,
    cohort_id: str,
    prompt_ids: tuple[str, ...] | list[str],
    index: BlockIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Failures, unclassified outcomes and attempted rollouts of each prompt, in order.

    Every row of the prompt in the cohort is an attempted rollout, as in
    :func:`direct_bounded_blocks`.
    """

    index = _current_index(index, observations, manifest)
    failures = np.zeros(len(prompt_ids), dtype=np.int64)
    unclassified = np.zeros(len(prompt_ids), dtype=np.int64)
    attempted = np.zeros(len(prompt_ids), dtype=np.int64)
    for position, prompt_id in enumerate(prompt_ids):
        for row_position in index.positions.get((cohort_id, prompt_id), ()):
            row = index.observations[row_position]
            attempted[position] += 1
            if row.failure is True:
                failures[position] += 1
            elif row.failure is None:
                unclassified[position] += 1
    return failures, unclassified, attempted


def comparison_contrast_blocks(
    observations: list[Observation],
    manifest: pd.DataFrame,
    profile: IndependenceProfileSpec,
    *,
    candidate: str,
    baseline: str,
    prompt_ids: tuple[str, ...] | list[str],
    prompt_weights: np.ndarray,
    index: BlockIndex | None = None,
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
    index = _current_index(index, observations, manifest)
    selected = index.rows({candidate, baseline}, weights_by_prompt)
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
    # Each accumulator is candidate mass/lower/upper followed by baseline mass/lower/upper.
    accumulators: dict[tuple[Any, ...], list[float]] = {}
    for row in selected:
        coefficient = weights_by_prompt[row.prompt_id] / counts[(row.cohort_id, row.prompt_id)]
        if coefficient == 0:
            continue
        key = block_key(row, profile, index.manifest_values, comparison=True)
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
