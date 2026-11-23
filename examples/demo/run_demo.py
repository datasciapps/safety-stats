"""Run the offline demo end to end: both simulated configurations, then the analysis.

    python examples/demo/run_demo.py

It writes the Inspect logs to logs/demo and the report to reports/demo, replacing the output
of any earlier demo run, then prints the headline numbers. It needs no provider credentials.
"""

import shutil
from pathlib import Path

from demo_eval import refusal_demo
from inspect_ai import eval as inspect_eval
from show_results import show

from safety_stats import analyze

HERE = Path(__file__).resolve().parent
LOGS = Path("logs/demo")
REPORT = Path("reports/demo")


def main() -> None:
    # Old logs would join the new ones as extra rollouts, so the demo starts from none.
    shutil.rmtree(LOGS, ignore_errors=True)
    # A task object, not a file path: Inspect reads an absolute Windows path as a glob.
    inspect_eval(
        refusal_demo(),
        model=["mockllm/baseline", "mockllm/candidate"],
        log_dir=str(LOGS),
    )
    report = analyze(plan=HERE / "plan.yaml", logs=LOGS)
    report.write(REPORT, overwrite=True)
    print()
    show(REPORT)


if __name__ == "__main__":
    main()
