"""Random pilot selection within strata of a design manifest.

A pilot is run to estimate what the full study will cost and roughly what its rates will be.
It can do so only if the pilot prompts are a random sample. Taking the first prompts of each
category is not one: identifiers are often assigned in an order that tracks length,
difficulty or source, so the first few can differ systematically from the rest.

Equal draws from strata of unequal size give unequal inclusion probabilities, so the pilot's
plain mean over-represents small strata. Weighting each drawn prompt by
1 / `pilot_inclusion_probability`, as the sampled-design analysis does, makes it unbiased
for the manifest's mean.

`draw_pilot` draws a fixed number of prompts from each stratum by simple random sampling
without replacement. Each prompt's rank is a hash of the seed and its `prompt_id`, so the draw:

- uses only the frame and the seed, never an outcome;
- does not depend on the order of the manifest's rows or on the platform;
- is nested: the pilot drawn with `per_stratum=k` contains the one drawn with any smaller `k`
  and the same seed, so a pilot can be enlarged for planning without discarding the prompts
  already run.

The rank is a hash rather than a value from `numpy.random.default_rng(seed)`, which the
contributing guide asks for: a generator gives values in the order it meets the rows, so its
draw would depend on that order, and numpy does not promise the same values from its methods
across versions. Ranks from a generator would nest too. The derivation is in
docs/statistical-methods.md, section 5.8.

The returned frame is the whole manifest with five added columns, among them the seed and
the per-stratum count, so the frame alone records how to repeat the draw. It doubles as
the frame of a `design.type: sampled` plan with `sampling: fixed_size`,
`stratum_column: pilot_stratum`, `inclusion_probability_column:
pilot_inclusion_probability` and `selection_column: pilot_selected`.

Analyse a pilot as exploratory. A pilot enlarged after its outcomes were seen is not a
fixed-size confirmatory design: stopping at the first size whose bound passes raises the
false-pass rate above alpha. A confirmatory analysis uses fresh rollouts under a plan frozen
before they ran, of the whole manifest or of a new draw with a new seed.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import pandas as pd

STRATUM_COLUMN = "pilot_stratum"
PROBABILITY_COLUMN = "pilot_inclusion_probability"
SELECTED_COLUMN = "pilot_selected"
SEED_COLUMN = "pilot_seed"
PER_STRATUM_COLUMN = "pilot_per_stratum"
ADDED_COLUMNS = (
    STRATUM_COLUMN,
    PROBABILITY_COLUMN,
    SELECTED_COLUMN,
    SEED_COLUMN,
    PER_STRATUM_COLUMN,
)

_SEPARATOR = "/"


def _rank_key(seed: int, prompt_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{prompt_id}".encode()).digest()


def _check_strata_read_back(labels: pd.Series) -> None:
    """Refuse strata labels that analyze refuses, or that a reader of the frame would merge.

    The draw treats stratum labels as text, and so does analyze, which reads `pilot_stratum`
    exactly as written (`load_manifest`). Analyze refuses a label that pandas reads as a
    missing value, such as `N/A` or `None`, so the draw refuses it too, rather than write a
    frame that analyze will not take. It also refuses labels that pandas' default type
    inference reads as the same number or boolean, such as `1` and `01` or `true` and `True`.
    Analyze keeps them apart, but a spreadsheet or a pandas reader with its defaults, with
    which a person may check the draw, would show them as one stratum.
    """

    text = labels.rename(STRATUM_COLUMN).to_csv(index=False, lineterminator="\n")
    read = pd.read_csv(io.StringIO(text))[STRATUM_COLUMN]
    missing = sorted(set(labels.to_numpy()[read.isna().to_numpy()]))
    if missing:
        raise ValueError(
            "analyze refuses these strata labels, which pandas and analyze read as missing "
            f"values: {missing[:10]!r}. Recode those strata values, for example as 'none'"
        )
    pairs = pd.DataFrame({"drawn": labels.to_numpy(), "read": read.astype(str).to_numpy()})
    groups = pairs.drop_duplicates().groupby("read")["drawn"].agg(sorted)
    merged = [
        f"{', '.join(repr(label) for label in drawn)} (read as {read_as!r})"
        for read_as, drawn in groups.items()
        if len(drawn) > 1
    ]
    if merged:
        raise ValueError(
            "pandas' default type inference, which a spreadsheet or another reader of the "
            "pilot frame may apply, would merge strata whose labels read as the same number "
            f"or boolean: {'; '.join(merged[:10])}. Recode the strata values so that they "
            "differ as text, for example with a letter prefix"
        )


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
    (draws over stratum size), `pilot_selected`, and `pilot_seed` and `pilot_per_stratum`
    (the arguments, repeated on every row so that the frame records the draw).

    The draw hashes each `prompt_id` as written, so `prompt_id` must hold text: read the
    manifest with `pd.read_csv(path, dtype=str, keep_default_na=False)`, as `plan pilot`
    does. Read with pandas' defaults, an identifier such as `0001` would become the number 1
    and draw different prompts from the command's.

    Raises ValueError when a `prompt_id` is not text or is blank, when a stratum holds fewer
    than `per_stratum` prompts, when a strata column is absent or has missing values, when a
    stratum label marks a missing value (`N/A`, say), when two strata labels would read as
    one under pandas' default type inference (`1` and `01`, say), or when the manifest already
    has an added column.
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
    _check_strata_read_back(labels)

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
    result[SEED_COLUMN] = seed
    result[PER_STRATUM_COLUMN] = per_stratum
    return result
