"""Run the harness-versions example end to end: both harness versions, then the analysis.

    python examples/harness_versions/run_example.py

It writes the Inspect logs to logs/harness-versions and the report to reports/harness-versions,
replacing the output of any earlier run, then prints the headline numbers. It needs no provider
credentials.
"""

import shutil
import sys
from pathlib import Path

from harness_eval import harness_versions
from inspect_ai import eval as inspect_eval

from safety_stats import SafetyReport, analyze

HERE = Path(__file__).resolve().parent
LOGS = Path("logs/harness-versions")
REPORT = Path("reports/harness-versions")
VERSIONS = ("v1", "v2")


def _cell(risk) -> str:  # type: ignore[no-untyped-def]
    return f"{risk.estimate:.3f} [{risk.lower:.3f}, {risk.upper:.3f}]"


def show(directory: Path) -> None:
    """Print each version's mean failure rate and the v2-minus-v1 difference."""
    text = (directory / "report.json").read_text(encoding="utf-8")
    report = SafetyReport.model_validate_json(text)
    print("Mean failure rate over all prompts, 95% simultaneous intervals")
    for risk in report.group_risks:
        if risk.measure_id == "mean":
            print(f"  harness {risk.cohort_id}: {_cell(risk)}")
    for comparison in report.comparisons:
        if comparison.path or comparison.estimand != "risk_difference":
            continue
        print()
        print(
            f"v2 minus v1, all prompts: {comparison.estimate:+.3f} "
            f"[{comparison.lower:+.3f}, {comparison.upper:+.3f}]"
        )
        if comparison.lower is not None and comparison.lower > 0:
            print("The whole interval is above zero: version 2 fails more often.")
        elif comparison.upper is not None and comparison.upper < 0:
            print("The whole interval is below zero: version 2 fails less often.")
        else:
            print(
                "The interval includes zero: no difference is established. Its upper end is "
                "the largest increase in the failure rate that the runs leave open."
            )
    print()
    print(f"Interactive report: {(directory / 'report.html').as_posix()}")


def main() -> None:
    # Old logs would join the new ones as extra rollouts, so the example starts from none.
    shutil.rmtree(LOGS, ignore_errors=True)
    for version in VERSIONS:
        # A task object, not a file path: Inspect reads an absolute Windows path as a glob.
        logs = inspect_eval(
            harness_versions(harness=version), model="mockllm/model", log_dir=str(LOGS)
        )
        if any(log.status != "success" for log in logs):
            sys.exit(f"the run of harness {version} did not succeed")
    report = analyze(plan=HERE / "plan.yaml", logs=LOGS)
    report.write(REPORT, overwrite=True)
    print()
    show(REPORT)


if __name__ == "__main__":
    main()
