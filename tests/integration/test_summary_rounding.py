"""report.md, index.md and analyze never show a bound tighter than the analysis gives.

Every upper bound is rounded up and every lower bound down at the decimals the page shows, and
a claim's thresholds are shown exactly, so that a reader comparing a shown bound with a shown
threshold reaches the verdict the analysis reached.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from safety_stats.cli import app
from safety_stats.schema.reports import SafetyReport
from tests.cli_output import plain
from tests.summary_fixtures import Variant, analysed, claim_spec, section, table, write_variant

# The review's case: 20 prompts of 500 rollouts without a failure. The upper bound on each
# rate, about 0.00044, lies below the claim's 0.0005 and rounds to 0.0% at one decimal.
NO_FAILURES = Variant(
    "no_failures",
    rollouts=500,
    failures={"baseline": [0], "candidate": [0]},
    plan={"claims": [claim_spec("candidate", acceptable_upper=0.0005, unacceptable_lower=0.01)]},
    hierarchy_nodes=False,
)


def _decimal(text: str) -> Decimal:
    return Decimal(text.removesuffix("%").removesuffix(" pts").replace("−", "-"))


def _interval(text: str) -> tuple[Decimal, Decimal]:
    left, right = text.split(" †")[0].strip("[]").split(", ")
    return _decimal(left), _decimal(right)


def _outward(
    shown: tuple[Decimal, Decimal],
    lower: float | None,
    upper: float | None,
    step: Decimal,
    scale: int = 0,
) -> bool:
    """Whether ``shown`` holds [lower, upper] and each end lies within one step of it."""

    assert lower is not None and upper is not None
    low = Decimal(repr(lower)).scaleb(scale)
    high = Decimal(repr(upper)).scaleb(scale)
    return shown[0] <= low < shown[0] + step and shown[1] - step < high <= shown[1]


def test_analyze_prints_a_bound_near_zero_rounded_up(tmp_path: Path) -> None:
    plan, source = write_variant(NO_FAILURES, tmp_path / "inputs")
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--plan",
            str(plan),
            "--runs",
            str(source.runs),
            "--rollouts",
            str(source.rollouts),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    report = SafetyReport.model_validate_json(
        (output / "report.json").read_text(encoding="utf-8")
    )
    candidate = next(risk for risk in report.group_risks if risk.cohort_id == "candidate")
    assert candidate.upper is not None and 0 < candidate.upper < 0.0005
    # Plain typer.echo lines, uncoloured; plain() would join them into one.
    assert "  candidate / all: 0.0% [0.0%, 0.1%]" in result.stdout.splitlines()

    summary = (output / "report.md").read_text(encoding="utf-8")
    rates = {row["Model"]: row for row in table(section(summary, "Failure risk"))}
    assert rates["candidate"]["Interval"] == "[0.0%, 0.1%]"

    (claim,) = table(section(summary, "Claims"))
    assert claim["Status"] == "`PASS`"
    # Rounded up, the bound stays at or below the threshold it passed, which is shown exactly.
    assert Decimal(claim["Upper"]) >= Decimal(repr(candidate.upper))
    assert Decimal(claim["Upper"]) <= Decimal(claim["Acceptable below"]) == Decimal("0.0005")

    index = (output / "index.md").read_text(encoding="utf-8")
    headline = {
        (row["Kind"], row["Model"], row["Estimand"]): row
        for row in table(index.split("## Headline estimates", 1)[1])
    }
    assert headline[("group_risk", "candidate", "—")]["Upper"] == "0.0005"
    difference = headline[("comparison", "candidate", "risk_difference")]
    assert Decimal(difference["Lower"]) <= -Decimal("0.0004") < Decimal("0.0004") <= Decimal(
        difference["Upper"]
    )


@pytest.fixture(scope="module")
def confirmatory(tmp_path_factory: pytest.TempPathFactory) -> tuple[SafetyReport, str]:
    variant = Variant("rounding", measures=({"id": "worst", "type": "maximum"},))
    report, written = analysed(variant, tmp_path_factory.mktemp("rounding"))
    return report, (written / "report.md").read_text(encoding="utf-8")


def test_every_rate_interval_is_rounded_outward(confirmatory: tuple[SafetyReport, str]) -> None:
    report, summary = confirmatory
    rows = {row["Model"]: row for row in table(section(summary, "Failure risk"))}
    risks = [risk for risk in report.group_risks if risk.measure_id == "mean"]
    assert len(risks) == 2
    for risk in risks:
        shown = _interval(rows[risk.cohort_id]["Interval"])
        assert _outward(shown, risk.lower, risk.upper, Decimal("0.1"), scale=2), risk.cohort_id


def test_every_comparison_interval_is_rounded_outward(
    confirmatory: tuple[SafetyReport, str],
) -> None:
    report, summary = confirmatory
    (row,) = table(section(summary, "Comparisons"))
    by_estimand = {item.estimand: item for item in report.comparisons if item.level is None}
    difference = by_estimand["risk_difference"]
    ratio = by_estimand["risk_ratio"]
    assert _outward(
        _interval(row["Interval"]), difference.lower, difference.upper, Decimal("0.1"), scale=2
    )
    assert _outward(_interval(row["Interval 2"]), ratio.lower, ratio.upper, Decimal("0.01"))


def test_every_breakdown_interval_is_rounded_outward(
    confirmatory: tuple[SafetyReport, str],
) -> None:
    report, summary = confirmatory
    rows = {row["category"]: row for row in table(section(summary, "By category"))}
    nodes = [node for node in report.hierarchy_risks if node.level == "category"]
    assert nodes
    for node in nodes:
        cell = rows[" → ".join(node.path.values())][node.cohort_id]
        shown = _interval(cell.split(" ", 1)[1])
        assert _outward(shown, node.lower, node.upper, Decimal("0.1"), scale=2), cell


def test_every_tail_interval_is_rounded_outward(confirmatory: tuple[SafetyReport, str]) -> None:
    report, summary = confirmatory
    rows = {
        row["Model"]: row
        for row in table(section(summary, "Tail and threshold measures"))
        if row["Measure"] == "worst"
    }
    risks = [risk for risk in report.group_risks if risk.measure_id == "worst"]
    assert len(risks) == 2
    for risk in risks:
        shown = _interval(rows[risk.cohort_id]["Interval"])
        assert _outward(shown, risk.lower, risk.upper, Decimal("0.001")), risk.cohort_id


def test_every_claim_bound_is_rounded_outward_at_the_reports_decimals(
    confirmatory: tuple[SafetyReport, str],
) -> None:
    report, summary = confirmatory
    rows = {row["Claim"]: row for row in table(section(summary, "Claims"))}
    places = report.presentation.decimal_places
    assert places == 6
    step = Decimal(1).scaleb(-places)
    for claim in report.claims:
        row = rows[claim.title]
        shown = (Decimal(row["Lower"]), Decimal(row["Upper"]))
        assert _outward(shown, claim.lower, claim.upper, step), claim.claim_id
        assert Decimal(row["Acceptable below"]) == Decimal(repr(claim.acceptable_upper))
        assert Decimal(row["Unacceptable above"]) == Decimal(repr(claim.unacceptable_lower))


def test_the_median_rollouts_per_prompt_is_not_truncated(
    confirmatory: tuple[SafetyReport, str],
) -> None:
    """Half the prompts with 31 valid rollouts and half with 30 have a median of 30.5."""

    from safety_stats.report.summary import summary_markdown

    report, summary = confirmatory
    assert "· 30 rollouts per prompt (median) ·" in summary
    estimates = [
        item.model_copy(update={"n_valid": 30 + index % 2})
        for index, item in enumerate(report.prompt_estimates)
    ]
    halfway = summary_markdown(report.model_copy(update={"prompt_estimates": estimates}))
    assert "· 30.5 rollouts per prompt (median) ·" in halfway.splitlines()[2]


def test_the_tail_budget_says_it_is_for_a_prompt_with_no_failures(
    confirmatory: tuple[SafetyReport, str],
) -> None:
    """The per-prompt budget is computed at rate 0, which needs fewer rollouts than any other.

    The other budgets assume the observed rate holds; this one says what it assumes instead.
    """

    from safety_stats.report.summary import precision_target
    from safety_stats.stats.planning import rollouts_for_half_width

    report, summary = confirmatory
    target = precision_target(report)
    alpha = next(item.alpha_upper for item in report.prompt_estimates if item.alpha_upper)
    needed = rollouts_for_half_width(rate=0.0, half_width=target, alpha=alpha)
    assert needed is not None
    for rate in (0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.99):
        other = rollouts_for_half_width(rate=rate, half_width=target, alpha=alpha)
        assert other is not None and other >= needed, rate
    line = next(
        line for line in section(summary, "Precision").splitlines() if "tail or threshold" in line
    )
    assert line.endswith(
        f"per-prompt bounds that tight need about {needed} rollouts on every prompt, the count "
        "for a prompt with no failures and the fewest any prompt needs (currently 30 per "
        "prompt, median)"
    )
    assert "assume the observed rate holds, unless they say otherwise" in summary


def test_the_coverage_figure_is_rounded_down(confirmatory: tuple[SafetyReport, str]) -> None:
    from safety_stats.report.summary import summary_markdown

    report, summary = confirmatory
    assert "with 95.0% simultaneous coverage" in summary
    risks = [
        risk.model_copy(update={"simultaneous_coverage": 0.99996})
        if risk.simultaneous_coverage
        else risk
        for risk in report.group_risks
    ]
    page = summary_markdown(report.model_copy(update={"group_risks": risks}))
    # To the nearest tenth this would read 100.0%.
    assert "with 99.9% simultaneous coverage" in page
