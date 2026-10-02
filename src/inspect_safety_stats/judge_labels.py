"""Draw the human-labelling sample that ``judge_calibration: validated`` consumes.

Validated judge correction is only as good as its sample. The labelled observations
must be a simple random sample within each judge stratum (judge said unsafe, judge
said safe) of each cohort, drawn without looking at their content. This module draws
exactly that from a completed report's ``observations.csv``, with a recorded seed, and
writes two files:

``labels.csv``
    The labelling sheet. Rows are shuffled across strata and cohorts and carry neither
    the judge's verdict nor the model, so a labeller cannot anchor on either. Fill in
    ``human_label`` with ``safe`` or ``unsafe`` for every row and point
    ``inputs.judge_calibration.validation_manifest`` at the file.
``labels.sampling.json``
    The audit record: seed, stratum sizes, draws and the selected identifiers, so a
    reviewer can confirm the sample was drawn as declared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}
_SHEET_COLUMNS = (
    "observation_id",
    "prompt_id",
    "epoch",
    "eval_id",
    "sample_id",
    "source_log",
)


@dataclass
class LabelSample:
    sheet_path: Path
    record_path: Path
    rows: int
    strata: dict[str, dict[str, dict[str, int]]]
    notes: list[str] = field(default_factory=list)


def _verdict(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def draw_label_sample(
    report: str | Path,
    output: str | Path,
    *,
    judge_unsafe: int,
    judge_safe: int,
    seed: int,
    overwrite: bool = False,
) -> LabelSample:
    """Draw ``judge_unsafe`` and ``judge_safe`` observations per cohort, uniformly at random."""

    if judge_unsafe < 1 or judge_safe < 1:
        raise ValueError(
            "both judge strata need at least one label; validated calibration estimates "
            "the judge's accuracy on each"
        )
    report_dir = Path(report)
    observations_path = report_dir / "observations.csv"
    if not observations_path.is_file():
        raise FileNotFoundError(
            f"{observations_path} not found; run analyze with report.include_observations "
            "enabled (the default) first"
        )
    frame = pd.read_csv(observations_path, dtype=str, keep_default_na=False)
    missing = [
        column for column in ("observation_id", "cohort_id", "failure") if column not in frame
    ]
    if missing:
        raise ValueError(f"observations.csv is missing columns: {', '.join(missing)}")
    frame["_verdict"] = frame["failure"].map(_verdict)
    unclassified = int(frame["_verdict"].isna().sum())
    frame = frame.loc[frame["_verdict"].notna()].sort_values("observation_id", kind="stable")

    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    strata: dict[str, dict[str, dict[str, int]]] = {}
    chosen_ids: dict[str, dict[str, list[str]]] = {}
    notes: list[str] = []
    if unclassified:
        notes.append(
            f"{unclassified} unclassified observations belong to neither judge stratum and "
            "were not eligible"
        )
    for cohort_id in sorted(frame["cohort_id"].unique()):
        cohort = frame.loc[frame["cohort_id"] == cohort_id]
        for verdict, name, requested in (
            (True, "judge_unsafe", judge_unsafe),
            (False, "judge_safe", judge_safe),
        ):
            stratum = cohort.loc[cohort["_verdict"] == verdict]
            size = len(stratum)
            draws = min(requested, size)
            if size == 0:
                notes.append(
                    f"cohort {cohort_id!r} has no {name.replace('_', '-')} observations; "
                    "validated correction will refuse this cohort"
                )
            elif draws < requested:
                notes.append(
                    f"cohort {cohort_id!r} {name.replace('_', '-')} stratum has only {size} "
                    "observations; all of them were taken (a census of the stratum)"
                )
            picks = (
                stratum.iloc[np.sort(rng.choice(size, size=draws, replace=False))]
                if draws
                else stratum.iloc[0:0]
            )
            selected.append(picks)
            strata.setdefault(str(cohort_id), {})[name] = {
                "stratum_size": size,
                "requested": requested,
                "drawn": draws,
            }
            chosen_ids.setdefault(str(cohort_id), {})[name] = sorted(
                picks["observation_id"].tolist()
            )

    sample = pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0]
    if sample.empty:
        raise ValueError("no classified observations were available to sample")
    order = rng.permutation(len(sample))
    sheet = sample.iloc[order].reset_index(drop=True)
    columns = [column for column in _SHEET_COLUMNS if column in sheet.columns]
    sheet = sheet[columns].copy()
    sheet["human_label"] = ""
    sheet["notes"] = ""

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    sheet_path = directory / "labels.csv"
    record_path = directory / "labels.sampling.json"
    existing = [path for path in (sheet_path, record_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite " + ", ".join(str(path) for path in existing)
            + "; pass --overwrite"
        )
    sheet.to_csv(sheet_path, index=False)

    plan_hash = None
    report_json = report_dir / "report.json"
    if report_json.is_file():
        try:
            plan_hash = json.loads(report_json.read_text(encoding="utf-8"))["analysis"].get(
                "plan_sha256"
            )
        except (KeyError, ValueError, AttributeError):
            plan_hash = None
    record = {
        "design": "simple_random_within_judge_stratum",
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(),
        "report": str(report_dir.resolve()),
        "plan_sha256": plan_hash,
        "strata": strata,
        "selected_observation_ids": chosen_ids,
        "blinding": "labels.csv omits the judge verdict and the model and is shuffled",
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return LabelSample(
        sheet_path=sheet_path,
        record_path=record_path,
        rows=len(sheet),
        strata=strata,
        notes=notes,
    )
