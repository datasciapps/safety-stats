from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.stats.misclassification import JudgeStratumCounts

REQUIRED_COLUMNS = ("observation_id", "human_label")
_LABELS = {"safe": False, "unsafe": True}


def load_validation_manifest(path: str | Path) -> dict[str, bool]:
    """Load human adjudications keyed by normalized observation identifier.

    ``human_label`` accepts only ``safe`` and ``unsafe``. An item a labeller cannot
    resolve must be handled by the declared collection protocol before it reaches
    this file; silently dropping it here would break the sampling design that makes
    the resulting proportions binomial.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"validation manifest is missing required columns: {', '.join(missing)}")
    labels: dict[str, bool] = {}
    for observation_id, raw in zip(frame["observation_id"], frame["human_label"], strict=True):
        key = str(observation_id).strip()
        if not key:
            raise ValueError("validation manifest contains a blank observation_id")
        if key in labels:
            raise ValueError(f"validation manifest has duplicate observation_id {key!r}")
        normalized = str(raw).strip().lower()
        if normalized not in _LABELS:
            raise ValueError(
                f"validation manifest has invalid human_label {raw!r} for {key!r}; "
                "expected 'safe' or 'unsafe'"
            )
        labels[key] = _LABELS[normalized]
    if not labels:
        raise ValueError("validation manifest contains no rows")
    return labels


def judge_stratum_counts(
    observations: list[Observation],
    labels: dict[str, bool],
) -> tuple[dict[str, dict[str, JudgeStratumCounts]], list[str]]:
    """Cross human labels against judge verdicts, per cohort and judge stratum.

    Unclassified observations belong to neither judge stratum. They are already
    covered by the conservative missing-outcome completion in the underlying
    interval and are excluded here.
    """

    sizes: dict[tuple[str, bool], int] = defaultdict(int)
    labelled: dict[tuple[str, bool], int] = defaultdict(int)
    unsafe: dict[tuple[str, bool], int] = defaultdict(int)
    seen: set[str] = set()
    for row in observations:
        if row.failure is None:
            continue
        key = (row.cohort_id, bool(row.failure))
        sizes[key] += 1
        human = labels.get(row.observation_id)
        if human is None:
            continue
        seen.add(row.observation_id)
        labelled[key] += 1
        if human:
            unsafe[key] += 1

    unmatched = sorted(set(labels) - seen)
    counts: dict[str, dict[str, JudgeStratumCounts]] = {}
    for cohort_id, verdict in sizes:
        stratum = "judge_unsafe" if verdict else "judge_safe"
        counts.setdefault(cohort_id, {})[stratum] = JudgeStratumCounts(
            labelled=labelled[(cohort_id, verdict)],
            truly_unsafe=unsafe[(cohort_id, verdict)],
            stratum_size=sizes[(cohort_id, verdict)],
        )
    return counts, unmatched
