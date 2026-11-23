"""Run the judge flip example offline end to end, with simulated models.

    python examples/judge_flip/run_offline.py

1. Evaluate the simulated target (``target_run``), scored by the simulated primary judge.
2. Analyse it without the flip section, to read the judge-positive observations.
3. Export those completions, rescore them under the four wrappers (``judge_flip``), and write
   the flip manifest.
4. Analyse with ``plan.yaml``, whose ``inputs.judge_flip`` reads the manifest.

Logs, the completions and the manifest go to ``run/`` beside this file, the report to
``reports/judge-flip``, replacing an earlier run's. It needs no credentials.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flip_manifest import export_positives, write_manifest
from inspect_ai import eval as inspect_eval
from judge_flip import judge_flip, target_run

from safety_stats import analyze, load_plan

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
REPORT = Path("reports/judge-flip")


def main() -> None:
    shutil.rmtree(RUN, ignore_errors=True)
    target_logs = RUN / "target-logs"
    inspect_eval(target_run(), model="mockllm/target", log_dir=str(target_logs), display="none")

    plan = load_plan(HERE / "plan.yaml")
    unflipped = plan.model_copy(
        update={"inputs": plan.inputs.model_copy(update={"judge_flip": None})}
    )
    analyze(plan=unflipped, logs=target_logs).write(RUN / "unflipped-report")
    count = export_positives(RUN / "unflipped-report", target_logs, RUN / "positives.csv")

    rescoring_logs = RUN / "rescoring-logs"
    inspect_eval(
        judge_flip(positives=str(RUN / "positives.csv")),
        model="mockllm/judge",
        log_dir=str(rescoring_logs),
        display="none",
    )
    write_manifest(rescoring_logs, RUN / "flips.csv")

    report = analyze(plan=HERE / "plan.yaml", logs=target_logs)
    report.write(REPORT, overwrite=True)
    print(f"judge-positive completions rescored: {count}")
    for item in report.integrity.diagnostics:
        if item.code.startswith("JUDGE_FLIP"):
            print(f"{item.severity} {item.code}: {item.message}")
            for wrapper, rate in item.details.get("by_wrapper", {}).items():
                print(
                    f"  {wrapper:<16}{rate['flipped']:>4} of {rate['rescored']}"
                    f"  [{rate['lower']:.3f}, {rate['upper']:.3f}]"
                )
    for claim in report.claims:
        print(f"claim {claim.claim_id}: {claim.status} (upper bound {claim.upper})")
    print(f"report: {REPORT / 'report.html'}")


if __name__ == "__main__":
    main()
