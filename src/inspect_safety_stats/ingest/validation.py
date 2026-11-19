from __future__ import annotations

import hashlib
import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.stats.judge_calibration import (
    FineStrataCounts,
    JudgeStrata,
    count_fine_strata,
    count_judge_strata,
)
from inspect_safety_stats.stats.misclassification import TruthStratumCounts

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


STRATA_COLUMNS = ("observation_id", "stratum")


def load_validation_strata(path: str | Path) -> dict[str, str]:
    """Load the declared fine stratum of each observation, keyed by its identifier.

    The file lists every classified observation of the analysed cohorts, labelled or not,
    since a stratum's size is part of the design. The judge verdict need not appear: the
    analysis nests each declared stratum within the judge strata itself. A blank stratum is
    rejected rather than read as a stratum of its own.
    """

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in STRATA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"validation strata file is missing required columns: {', '.join(missing)}"
        )
    strata: dict[str, str] = {}
    for observation_id, raw in zip(frame["observation_id"], frame["stratum"], strict=True):
        key = str(observation_id).strip()
        if not key:
            raise ValueError("validation strata file contains a blank observation_id")
        if key in strata:
            raise ValueError(f"validation strata file has duplicate observation_id {key!r}")
        stratum = str(raw).strip()
        if not stratum:
            raise ValueError(f"validation strata file has a blank stratum for {key!r}")
        strata[key] = stratum
    if not strata:
        raise ValueError("validation strata file contains no rows")
    return strata


@dataclass(frozen=True)
class LabelledPrefix:
    """What the labelling-protocol check found: the queue, and how far down it labels reach.

    ``queue_sha256`` is the SHA-256 of the queue file's bytes as read, which are also the
    bytes the check parsed. ``observation_ids`` are the queue's rows in labelling order, those
    of the selected tier only when a tier is given.
    """

    queue_rows: int
    labelled: int
    queue_sha256: str
    observation_ids: tuple[str, ...]


def check_labelled_prefix(
    queue_path: str | Path,
    labels: dict[str, bool],
    *,
    tier: str | None = None,
) -> LabelledPrefix:
    """Check that the labels cover exactly the first rows of a labelling queue.

    The queue is a CSV whose ``observation_id`` column lists rollouts in labelling order.
    With ``tier``, only rows whose ``tier`` column holds that value count. A queue that
    lists each judge stratum, or a simple random sample of it, in a random order makes every
    prefix a simple random sample within each stratum, so a skipped row, or a label from
    outside the queue, would let the sample depend on which rollouts were easy or interesting
    to label. Either raises ``ValueError``, naming the first offending row. The check does
    not see the analysed rollouts: the caller checks the queue's rows, which the result lists
    as ``observation_ids``, against them.

    The file is read once: the hash it reports is of the bytes the check parsed, so a queue
    rewritten while the analysis reads it cannot be hashed in one version and checked in
    another.
    """

    content = Path(queue_path).read_bytes()
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    if "observation_id" not in frame.columns:
        raise ValueError("labelling queue is missing the observation_id column")
    where = "labelling queue"
    if tier is not None:
        if "tier" not in frame.columns:
            raise ValueError(f"labelling queue has no tier column to select tier {tier!r} by")
        frame = frame[frame["tier"].str.strip() == tier]
        where = f"labelling queue tier {tier!r}"
    order = [str(item).strip() for item in frame["observation_id"]]
    position: dict[str, int] = {}
    for row, observation_id in enumerate(order, start=1):
        if not observation_id:
            raise ValueError(f"{where} row {row} has a blank observation_id")
        if observation_id in position:
            raise ValueError(
                f"{where} lists observation {observation_id!r} twice, at rows "
                f"{position[observation_id]} and {row}"
            )
        position[observation_id] = row
    outside = sorted(key for key in labels if key not in position)
    if outside:
        raise ValueError(
            f"{len(outside)} labelled observations are not in the {where}, for example "
            f"{outside[0]!r}; labels must come from the queue"
        )
    count = len(labels)
    gap = next((row for row, key in enumerate(order[:count], start=1) if key not in labels), None)
    if gap is not None:
        last = max(position[key] for key in labels)
        raise ValueError(
            f"the labels are not the first rows of the {where}: row {gap} "
            f"({order[gap - 1]!r}) has no label, but row {last} does; label the queue in "
            "order without skipping rows"
        )
    return LabelledPrefix(
        queue_rows=len(order),
        labelled=count,
        queue_sha256=hashlib.sha256(content).hexdigest(),
        observation_ids=tuple(order),
    )


def judge_stratum_counts(
    observations: list[Observation],
    labels: dict[str, bool],
) -> tuple[dict[str, JudgeStrata], list[str]]:
    """Cross human labels against judge verdicts, per cohort and judge stratum.

    Labels are keyed by observation identifier. Unclassified observations belong to
    neither judge stratum; see :func:`count_judge_strata`, which does the counting.
    """

    return count_judge_strata(
        ((row.cohort_id, row.observation_id, row.failure) for row in observations), labels
    )


def fine_stratum_counts(
    observations: list[Observation],
    labels: dict[str, bool],
    strata: dict[str, str],
) -> FineStrataCounts:
    """Cross human labels against judge verdicts within declared fine strata, per cohort.

    See :func:`count_fine_strata`, which does the counting and the consistency checks.
    """

    return count_fine_strata(
        ((row.cohort_id, row.observation_id, row.failure) for row in observations),
        labels,
        strata,
    )


def truth_stratum_counts(
    observations: list[Observation],
    labels: dict[str, bool],
) -> dict[str, dict[str, TruthStratumCounts]]:
    """Cross judge verdicts against human labels, per cohort and true class.

    These counts estimate sensitivity and the false-positive rate only when the labelled
    rows are a simple random sample of the design (or sampled by true class). Under
    judge-stratified sampling they are biased and must not be used.
    """

    labelled: dict[tuple[str, bool], int] = defaultdict(int)
    positive: dict[tuple[str, bool], int] = defaultdict(int)
    for row in observations:
        if row.failure is None:
            continue
        human = labels.get(row.observation_id)
        if human is None:
            continue
        key = (row.cohort_id, human)
        labelled[key] += 1
        if row.failure:
            positive[key] += 1
    counts: dict[str, dict[str, TruthStratumCounts]] = {}
    for (cohort_id, truly_unsafe), count in labelled.items():
        stratum = "truly_unsafe" if truly_unsafe else "truly_safe"
        counts.setdefault(cohort_id, {})[stratum] = TruthStratumCounts(
            labelled=count, judge_positive=positive[(cohort_id, truly_unsafe)]
        )
    return counts
