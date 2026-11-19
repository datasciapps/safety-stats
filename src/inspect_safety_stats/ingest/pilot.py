"""Random pilot selection within strata of a design manifest.

A pilot is run to estimate what the full study will cost and roughly what its rates will be.
Its means estimate those quantities without bias only if the pilot prompts are a random sample.
Taking the first prompts of each category is not one: identifiers are often assigned in an
order that tracks length, difficulty or source, so the first few can differ systematically
from the rest.

`draw_pilot` draws a fixed number of prompts from each stratum by simple random sampling
without replacement. Each prompt's rank is a hash of the seed and its `prompt_id`, so the draw:

- uses only the frame and the seed, never an outcome;
- does not depend on the order of the manifest's rows or on the platform;
- is nested: the pilot drawn with `per_stratum=k` contains the one drawn with any smaller `k`
  and the same seed, so a pilot can be extended without discarding the prompts already run.

The returned frame is the whole manifest with three added columns. It doubles as the frame of
a `design.type: sampled` plan with `sampling: fixed_size`, `stratum_column: pilot_stratum`
and `inclusion_probability_column: pilot_inclusion_probability`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pandas as pd

STRATUM_COLUMN = "pilot_stratum"
PROBABILITY_COLUMN = "pilot_inclusion_probability"
SELECTED_COLUMN = "pilot_selected"
ADDED_COLUMNS = (STRATUM_COLUMN, PROBABILITY_COLUMN, SELECTED_COLUMN)

_SEPARATOR = "/"


def _rank_key(seed: int, prompt_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{prompt_id}".encode()).digest()


def draw_pilot(
    manifest: pd.DataFrame,
    *,
    per_stratum: int,
    strata: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Draw `per_stratum` prompts at random from each stratum of `manifest`.

    Strata are the distinct combinations of the `strata` columns; with no columns the whole
    manifest is one stratum. Returns a copy of the manifest, rows in their original order, with
    `pilot_stratum` (the stratum's values joined by "/"), `pilot_inclusion_probability`
    (draws over stratum size) and `pilot_selected`.

    The draw hashes each `prompt_id` as written, so `prompt_id` must hold text: read the
    manifest with `pd.read_csv(path, dtype=str, keep_default_na=False)`, as `plan pilot`
    does. Read with pandas' defaults, an identifier such as `0001` would become the number 1
    and draw different prompts from the command's.

    Raises ValueError when a `prompt_id` is not text or is blank, when a stratum holds fewer
    than `per_stratum` prompts, when a strata column is absent or has missing values, or when
    the manifest already has an added column.
    """

    if isinstance(per_stratum, bool) or per_stratum < 1:
        raise ValueError("per_stratum must be a positive integer")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if "prompt_id" not in manifest:
        raise ValueError("design manifest has no prompt_id column")
    if manifest.empty:
        raise ValueError("design manifest has no prompts")
    prompt_ids = manifest["prompt_id"]
    not_text = [value for value in prompt_ids if not isinstance(value, str)]
    if not_text:
        raise ValueError(
            f"design manifest has prompt_id values that are not text: {not_text[:10]!r}; "
            "read it with pd.read_csv(path, dtype=str, keep_default_na=False), so that each "
            "identifier is hashed as written"
        )
    if (prompt_ids.str.strip() == "").any():
        raise ValueError("design manifest has missing prompt_id values")
    if prompt_ids.duplicated().any():
        duplicates = sorted(prompt_ids[prompt_ids.duplicated()].unique())
        raise ValueError(f"design manifest has duplicate prompt_id values: {duplicates[:10]}")
    clashes = [column for column in ADDED_COLUMNS if column in manifest]
    if clashes:
        raise ValueError(
            f"design manifest already has {', '.join(clashes)}; draw from the original "
            "manifest, not from a pilot frame"
        )
    columns = list(strata)
    if len(set(columns)) != len(columns):
        raise ValueError("strata columns must be distinct")
    for column in columns:
        if column not in manifest:
            raise ValueError(f"strata column {column!r} is not in the design manifest")
        values = manifest[column]
        # A missing value would form a stratum of its own, or join another, depending on how
        # the CSV was parsed. Reject it rather than let parsing decide the strata.
        if values.isna().any() or (values.astype(str).str.strip() == "").any():
            raise ValueError(f"strata column {column!r} has missing values")

    if columns:
        labels = manifest[columns].astype(str).agg(_SEPARATOR.join, axis=1)
    else:
        labels = pd.Series("all", index=manifest.index)
    if len(columns) > 1:
        # Joining could merge two strata, e.g. ("a/b", "c") and ("a", "b/c").
        distinct = manifest[columns].astype(str).drop_duplicates()
        if len(distinct) != labels.nunique():
            raise ValueError(
                f"strata values contain {_SEPARATOR!r}, so two strata would share a label"
            )

    selected = pd.Series(False, index=manifest.index)
    probability = pd.Series(0.0, index=manifest.index)
    small: list[str] = []
    for label in sorted(labels.unique()):
        members = labels.index[labels == label]
        size = len(members)
        if size < per_stratum:
            small.append(f"{label} ({size})")
            continue
        ranked = sorted(members, key=lambda row: _rank_key(seed, prompt_ids[row]))
        selected[ranked[:per_stratum]] = True
        probability[members] = per_stratum / size
    if small:
        raise ValueError(
            f"{len(small)} strata hold fewer than {per_stratum} prompts: "
            f"{', '.join(small[:10])}; draw fewer per stratum or merge those strata"
        )

    result = manifest.copy()
    result[STRATUM_COLUMN] = labels
    result[PROBABILITY_COLUMN] = probability
    result[SELECTED_COLUMN] = selected
    return result
