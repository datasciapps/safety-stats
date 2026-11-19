"""The two files around the rescoring: what to rescore, and the flip manifest it gives.

    python flip_manifest.py positives REPORT_DIR LOG_DIR positives.csv
    python flip_manifest.py manifest RESCORING_LOG_DIR flips.csv

``positives`` reads the judge-positive observations from a report written with
``report.include_observations: true``, finds each completion in the run's Inspect logs by its
run, item and epoch, and writes ``observation_id, item_id, repeat, request, completion``.
Judge-positive means the primary judge scored the completion unsafe: a failure that a plan
policy classified, such as a sample error counted as a failure, is left out, by the test the
analysis applies.
``manifest`` reads the ``judge_flip`` task's logs and writes ``observation_id, wrapper,
verdict``, the CSV that ``inputs.judge_flip.manifest`` names. Both fail rather than drop a row:
a missing completion, an errored or unparsed rescoring, or a completion rescored twice under
one wrapper.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

from inspect_safety_stats.ingest.outcomes import judge_positive
from inspect_safety_stats.schema.observations import Observation


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _observation(row: dict[str, str]) -> Observation:
    """An observation read back from observations.csv, which writes None as an empty cell."""

    fields = Observation.model_fields
    return Observation.model_validate(
        {
            key: None if value == "" and fields[key].default is None else value
            for key, value in row.items()
        }
    )


def export_positives(report: Path, logs: Path, output: Path) -> int:
    """Write the judge-positive completions of the run; returns how many.

    They are the completions the analysis rates. A failure that a plan policy classified was
    never judged, so rescoring it would say nothing about the judge, and the analysis rejects
    a manifest that rescores one.
    """

    with (report / "observations.csv").open(encoding="utf-8", newline="") as handle:
        positives = [row for row in csv.DictReader(handle) if judge_positive(_observation(row))]
    texts: dict[tuple[str, str, int], tuple[str, str]] = {}
    for info in list_eval_logs(str(logs)):
        log = read_eval_log(info)
        for sample in log.samples or []:
            request = sample.input if isinstance(sample.input, str) else str(sample.input)
            key = (log.eval.eval_id, str(sample.id), sample.epoch)
            texts[key] = (request, sample.output.completion)
    rows = []
    for row in positives:
        key = (row["run_key"], row["item_id"], int(row["repeat"]))
        if key not in texts:
            raise ValueError(f"no completion in {logs} for observation {row['observation_id']}")
        request, completion = texts[key]
        rows.append([row["observation_id"], row["item_id"], row["repeat"], request, completion])
    _write(output, ["observation_id", "item_id", "repeat", "request", "completion"], sorted(rows))
    return len(rows)


def write_manifest(logs: Path, output: Path) -> int:
    """Write the flip manifest from the rescoring logs; returns its row count."""

    verdicts: dict[tuple[str, str], str] = {}
    for info in list_eval_logs(str(logs)):
        log = read_eval_log(info)
        if log.status != "success":
            raise ValueError(f"rescoring log {info.name} did not finish: {log.status}")
        for sample in log.samples or []:
            key = (str(sample.metadata["observation_id"]), str(sample.metadata["wrapper"]))
            if sample.error is not None or sample.scores is None:
                raise ValueError(f"rescoring of {key} has no verdict; rerun it")
            value = str(sample.scores["verdict"].value)
            if value not in {"safe", "unsafe"}:
                raise ValueError(f"rescoring of {key} gave {value!r}; resolve it and rerun")
            if key in verdicts:
                raise ValueError(f"{key} was rescored more than once; keep one log per rescoring")
            verdicts[key] = value
    rows = [[observation, wrapper, value] for (observation, wrapper), value in verdicts.items()]
    _write(output, ["observation_id", "wrapper", "verdict"], sorted(rows))
    return len(rows)


def main(argv: list[str]) -> None:
    if len(argv) == 4 and argv[0] == "positives":
        count = export_positives(Path(argv[1]), Path(argv[2]), Path(argv[3]))
        print(f"{count} judge-positive completions written to {argv[3]}")
    elif len(argv) == 3 and argv[0] == "manifest":
        count = write_manifest(Path(argv[1]), Path(argv[2]))
        print(f"{count} rescored verdicts written to {argv[2]}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
