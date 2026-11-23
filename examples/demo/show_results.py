"""Print the headline numbers of a demo report.

    python examples/demo/show_results.py reports/demo

It reads report.json through the public `SafetyReport` model, as any script can. A saved
report is read as stored (`STORED_REPORT`), so that one written by an earlier version still
reads after the package's registry of diagnostic codes changes.
"""

import sys
import textwrap
from pathlib import Path

from safety_stats import SafetyReport
from safety_stats.schema.reports import STORED_REPORT

COHORTS = ("baseline", "candidate")


def _cell(risk) -> str:  # type: ignore[no-untyped-def]
    """A group or hierarchy risk as 'estimate [lower, upper]'."""
    return f"{risk.estimate:.3f} [{risk.lower:.3f}, {risk.upper:.3f}]"


def show(directory: Path) -> None:
    text = (directory / "report.json").read_text(encoding="utf-8")
    report = SafetyReport.model_validate_json(text, context={STORED_REPORT: True})
    # The plan's one group covers all prompts; the breakdown by risk domain is a level of its
    # hierarchy, so those rates are hierarchy risks.
    rows: dict[str, dict[str, str]] = {}
    for risk in report.group_risks:
        if risk.measure_id == "mean":
            rows.setdefault("all", {})[risk.cohort_id] = _cell(risk)
    for node in report.hierarchy_risks:
        if node.level == "risk_domain":
            domain = node.path["risk_domain"]
            rows.setdefault(domain, {})[node.cohort_id] = _cell(node)
    print("Mean failure rate, 95% simultaneous intervals")
    print(f"  {'group':<9}{'baseline':<24}{'candidate':<24}")
    for group in ["all", *sorted(name for name in rows if name != "all")]:
        print(f"  {group:<9}{rows[group][COHORTS[0]]:<24}{rows[group][COHORTS[1]]:<24}")
    for comparison in report.comparisons:
        if comparison.path or comparison.estimand != "risk_difference":
            continue
        print()
        print(
            f"Candidate minus baseline, all prompts: {comparison.estimate:+.3f} "
            f"[{comparison.lower:+.3f}, {comparison.upper:+.3f}]"
        )
        if comparison.upper is not None and comparison.upper < 0:
            print("The whole interval is below zero: the candidate fails less often.")
        elif comparison.lower is not None and comparison.lower > 0:
            print("The whole interval is above zero: the candidate fails more often.")
        else:
            print("The interval includes zero: no difference is established.")
    for diagnostic in report.integrity.diagnostics:
        print()
        print(f"{diagnostic.severity} {diagnostic.code}:")
        print(textwrap.indent(textwrap.fill(diagnostic.message, width=86), "  "))
    print()
    print(f"Interactive report: {(directory / 'report.html').as_posix()}")


if __name__ == "__main__":
    show(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/demo"))
