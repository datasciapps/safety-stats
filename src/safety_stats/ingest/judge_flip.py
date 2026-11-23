"""Read the judge flip manifest: rescored verdicts of judge-positive completions (#122)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("observation_id", "wrapper", "verdict")
_VERDICTS = {"safe": False, "unsafe": True}


def load_judge_flip_manifest(path: str | Path) -> dict[str, dict[str, bool]]:
    """Load rescored verdicts keyed by observation identifier, then wrapper; True is unsafe.

    ``verdict`` accepts only ``safe`` and ``unsafe``. A rescoring that failed or gave no
    verdict must be rerun or resolved before it reaches this file: dropping it here would
    change which completions the rate is over. A file with a header and no rows is accepted,
    for a run without judge-positive completions; the analysis checks it against the run.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"judge flip manifest is missing required columns: {', '.join(missing)}")
    verdicts: dict[str, dict[str, bool]] = {}
    for observation_id, raw_wrapper, raw_verdict in zip(
        frame["observation_id"], frame["wrapper"], frame["verdict"], strict=True
    ):
        key = str(observation_id).strip()
        if not key:
            raise ValueError("judge flip manifest contains a blank observation_id")
        wrapper = str(raw_wrapper).strip()
        if not wrapper:
            raise ValueError(f"judge flip manifest has a blank wrapper for {key!r}")
        rows = verdicts.setdefault(key, {})
        if wrapper in rows:
            raise ValueError(
                f"judge flip manifest has more than one row for {key!r} under wrapper "
                f"{wrapper!r}"
            )
        normalized = str(raw_verdict).strip().lower()
        if normalized not in _VERDICTS:
            raise ValueError(
                f"judge flip manifest has invalid verdict {raw_verdict!r} for {key!r} under "
                f"wrapper {wrapper!r}; expected 'safe' or 'unsafe'"
            )
        rows[wrapper] = _VERDICTS[normalized]
    return verdicts
