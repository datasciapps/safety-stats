"""Print the headline numbers of a demo report.

    python examples/demo/show_results.py reports/demo

It reads report.json through the public `SafetyReport` model, as any script can.
"""

import sys
from pathlib import Path

from inspect_safety_stats import SafetyReport

GROUPS = ["all", "cyber", "bio", "privacy"]


def show(directory: Path) -> None:
    text = (directory / "report.json").read_text(encoding="utf-8")
    report = SafetyReport.model_validate_json(text)
    means = {
        (risk.cohort_id, risk.group_id): risk
        for risk in report.group_risks
        if risk.measure_id == "mean"
    }
    print("Mean failure rate, 95% simultaneous intervals")
    print(f"  {'group':<9}{'baseline':<24}{'candidate':<24}")
    for group in GROUPS:
        cells = []
        for cohort in ("baseline", "candidate"):
            risk = means[(cohort, group)]
            cells.append(f"{risk.estimate:.3f} [{risk.lower:.3f}, {risk.upper:.3f}]")
        print(f"  {group:<9}{cells[0]:<24}{cells[1]:<24}")
    for comparison in report.comparisons:
        if comparison.path:
            continue
        print()
        print(
            f"Candidate minus baseline, all prompts: {comparison.estimate:+.3f} "
            f"[{comparison.lower:+.3f}, {comparison.upper:+.3f}]"
        )
        if comparison.upper is not None and comparison.upper < 0:
            print("The whole interval is below zero: the candidate fails less often.")
        else:
            print("The interval includes zero: no difference is established.")
    print()
    print(f"Interactive report: {(directory / 'report.html').as_posix()}")


if __name__ == "__main__":
    show(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/demo"))
