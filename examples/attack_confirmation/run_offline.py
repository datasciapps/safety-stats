"""Run the attack example offline end to end: both simulated configurations, then the analysis.

    python examples/attack_confirmation/run_offline.py

It writes the Inspect logs to logs/attack-confirmation and the report to
reports/attack-confirmation, replacing the output of an earlier run, then prints the
confirmed rate beside the rate the attacker's own judge reported. It needs no credentials.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from attack_confirmation import attack_confirmation
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from inspect_safety_stats import analyze

HERE = Path(__file__).resolve().parent
LOGS = Path("logs/attack-confirmation")
REPORT = Path("reports/attack-confirmation")
CONFIGURATIONS = ("mockllm/baseline", "mockllm/candidate")


def failure_rate(log: EvalLog, scorer: str) -> float:
    """The unweighted share of runs that the named scorer marked FAIL."""

    values = [
        sample.scores[scorer].value for sample in log.samples or [] if sample.scores is not None
    ]
    return sum(value == "FAIL" for value in values) / len(values)


def main() -> None:
    # Old logs would join the new ones as extra attack runs, so the run starts from none.
    shutil.rmtree(LOGS, ignore_errors=True)
    logs = inspect_eval(
        attack_confirmation(), model=list(CONFIGURATIONS), log_dir=str(LOGS), display="none"
    )
    report = analyze(plan=HERE / "plan.yaml", logs=LOGS)
    report.write(REPORT, overwrite=True)

    print(f"{'configuration':<20}{'attacker-reported':>20}{'confirmed':>12}")
    for log in logs:
        print(
            f"{log.eval.model:<20}"
            f"{failure_rate(log, 'attacker_reported'):>20.2f}"
            f"{failure_rate(log, 'confirmed_jailbreak'):>12.2f}"
        )
    for claim in report.claims:
        print(f"claim {claim.claim_id}: {claim.status} (upper bound {claim.upper})")
    print(f"report: {REPORT / 'report.html'}")


if __name__ == "__main__":
    main()
