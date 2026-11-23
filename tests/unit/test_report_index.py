"""index.json and index.md map a report directory: every file, and pointers that resolve."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from safety_stats.cli import app
from safety_stats.report.index import (
    AGENT_GUIDE_URL,
    DIAGNOSTICS_URL,
    INDEX_JSON,
    INDEX_MD,
    index_document,
    index_files,
)
from safety_stats.schema.reports import (
    STORED_REPORT,
    JudgeCalibrationPresentation,
    RiskIntervalResult,
    SafetyReport,
)
from tests.summary_fixtures import section, table

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "real_simpleqa" / "report"


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer.lstrip("/").split("/"):
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _comparison(level: str | None, estimate: float) -> dict[str, Any]:
    return {
        "comparison_id": "c",
        "candidate_cohort": "candidate",
        "baseline_cohort": "baseline",
        "group_id": "all-eight",
        "measure_id": "mean",
        "estimand": "risk_difference",
        "level": level,
        "path": {"domain": "d"} if level else {},
        "estimate": estimate,
        "lower": estimate - 0.1,
        "upper": estimate + 0.1,
        "method": "m",
        "eligibility": "certifying",
    }


def _report(*, include_decisions: bool = False) -> SafetyReport:
    document = json.loads((EXAMPLE / "report.json").read_text(encoding="utf-8"))
    document["comparisons"] = [_comparison("domain", 0.3), _comparison(None, 0.1)]
    document["presentation"]["include_decisions"] = include_decisions
    return SafetyReport.model_validate(document, context={STORED_REPORT: True})


def test_the_index_lists_every_file_written_and_checksums_cover_it(tmp_path: Path) -> None:
    directory = _report().write(tmp_path / "report")
    index = json.loads((directory / INDEX_JSON).read_text(encoding="utf-8"))
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))

    written = sorted(path.name for path in directory.iterdir())
    assert [item["name"] for item in index["artifacts"]] == written
    assert {INDEX_JSON, INDEX_MD} <= set(checksums)


def test_every_pointer_in_the_index_resolves_in_report_json(tmp_path: Path) -> None:
    directory = _report().write(tmp_path / "report")
    index = json.loads((directory / INDEX_JSON).read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))

    # The index copies report.json's summary and adds a CSV row to each headline.
    copied = index["summary"] | {
        "headline": [
            {key: value for key, value in item.items() if key != "csv"}
            for item in index["summary"]["headline"]
        ]
    }
    assert copied == report["summary"]
    for item in index["summary"]["headline"]:
        record = _resolve(report, item["item"])
        assert record["estimate"] == item["estimate"]
    for warning in index["summary"]["warnings"]:
        for occurrence in warning["occurrences"]:
            assert _resolve(report, occurrence["diagnostic"])["code"] == warning["code"]
            if occurrence["item"] is not None:
                _resolve(report, occurrence["item"])


def _judged_report(*, include_decisions: bool = False) -> SafetyReport:
    """The example with a validated correction of its headline rate and comparison."""

    report = _report(include_decisions=include_decisions)
    report.presentation.judge_calibration = JudgeCalibrationPresentation(mode="validated")
    for row in (report.group_risks[2], report.comparisons[1]):
        row.intervals.append(
            RiskIntervalResult(
                interval_id="judge_corrected_validated",
                lower=0.123456,
                upper=0.654321,
                method="validated_judge_misclassification_correction",
                role="reference",
                simultaneous_coverage=0.93,
                eligibility="certifying",
            )
        )
    return report


def test_a_judge_corrected_headline_gives_the_true_risk_in_both_files(tmp_path: Path) -> None:
    directory = _judged_report().write(tmp_path / "report")
    index = json.loads((directory / INDEX_JSON).read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    rows = table(section((directory / INDEX_MD).read_text(encoding="utf-8"), "Headline estimates"))

    headline = index["summary"]["headline"]
    assert len(rows) == len(headline) == 2
    for item, row in zip(headline, rows, strict=True):
        corrected = item["judge_corrected"]
        assert corrected["calibration"] == "validated"
        interval = _resolve(report, corrected["item"])
        assert interval["interval_id"] == "judge_corrected_validated"
        assert (corrected["lower"], corrected["upper"]) == (interval["lower"], interval["upper"])
        assert row["Pointer"] == f"`{item['item']}`"
        # Rounded outward, and the coverage down.
        assert (row["True risk lower"], row["True risk upper"]) == ("0.1234", "0.6544")
        assert row["True risk coverage"] == "0.9300"
        assert (row["True risk eligibility"], row["Judge calibration"]) == (
            "certifying",
            "validated",
        )
        assert row["True risk pointer"] == f"`{corrected['item']}`"


@pytest.mark.parametrize("include_decisions", [False, True])
def test_a_written_report_follows_the_report_schema(
    tmp_path: Path, include_decisions: bool
) -> None:
    """The CLI's schema accepts every part of the summary: decisions, claims, corrections."""

    result = CliRunner().invoke(app, ["schema", "report"])
    assert result.exit_code == 0, result.output
    directory = _judged_report(include_decisions=include_decisions).write(tmp_path / "report")
    payload = json.loads((directory / "report.json").read_text(encoding="utf-8"))

    assert ("claims" in payload["summary"]) is include_decisions
    assert all("judge_corrected" in item for item in payload["summary"]["headline"])
    errors = [
        f"{error.json_path}: {error.message}"
        for error in Draft202012Validator(json.loads(result.stdout)).iter_errors(payload)
    ]
    assert errors == []
    # The schema says summary is derived: optional, and read-only.
    schema = json.loads(result.stdout)
    assert "summary" not in schema["required"]
    assert schema["properties"]["summary"]["readOnly"] is True


def test_an_index_without_a_judge_correction_has_no_true_risk_columns() -> None:
    files = index_files(_report(), ["report.json", "comparisons.csv"])

    headline = json.loads(files[INDEX_JSON])["summary"]["headline"]
    assert headline
    assert all("judge_corrected" not in item for item in headline)
    assert "True risk" not in files[INDEX_MD]


def test_a_headline_comparison_gives_its_csv_row(tmp_path: Path) -> None:
    directory = _report().write(tmp_path / "report")
    index = json.loads((directory / INDEX_JSON).read_text(encoding="utf-8"))
    (comparison,) = [item for item in index["summary"]["headline"] if item["kind"] == "comparison"]

    assert comparison["item"] == "/comparisons/1"
    assert comparison["csv"] == {"file": "comparisons.csv", "row": 2}
    with (directory / "comparisons.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[comparison["csv"]["row"] - 1]["estimate"]) == comparison["estimate"]
    (group_risk,) = [item for item in index["summary"]["headline"] if item["kind"] == "group_risk"]
    assert group_risk["csv"] is None


@pytest.mark.parametrize("include_decisions", [False, True])
def test_the_index_hides_decisions_as_the_report_does(include_decisions: bool) -> None:
    files = index_files(_report(include_decisions=include_decisions), ["report.json"])
    summary = json.loads(files[INDEX_JSON])["summary"]

    assert ("decision" in summary) is include_decisions
    assert ("Overall decision" in files[INDEX_MD]) is include_decisions
    assert ("Decisions are hidden" in files[INDEX_MD]) is not include_decisions


def test_the_markdown_twin_shows_each_pointer_of_the_json(tmp_path: Path) -> None:
    files = index_files(_report(), ["report.json", "comparisons.csv"])
    document = json.loads(files[INDEX_JSON])
    markdown = files[INDEX_MD]

    for item in document["summary"]["headline"]:
        assert f"`{item['item']}`" in markdown
    for warning in document["summary"]["warnings"]:
        assert f"`{warning['code']}`" in markdown
        for occurrence in warning["occurrences"]:
            assert f"`{occurrence['diagnostic']}`" in markdown
    for artifact in document["artifacts"]:
        assert f"`{artifact['name']}`" in markdown


def test_a_file_the_index_does_not_describe_fails_loudly() -> None:
    with pytest.raises(ValueError, match="does not describe notes.txt"):
        index_document(_report(), ["report.json", "notes.txt"])


def test_a_report_read_back_from_report_json_writes_the_same_index(tmp_path: Path) -> None:
    directory = _report().write(tmp_path / "report")
    saved = SafetyReport.model_validate_json((directory / "report.json").read_text("utf-8"))
    names = [
        name
        for name in json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        if name not in {INDEX_JSON, INDEX_MD}
    ]

    for name, text in index_files(saved, names).items():
        assert (directory / name).read_bytes() == text.encode("utf-8")


def _checkout() -> Path | None:
    """The repository root when the tests run in its git checkout, else None."""

    root = Path(__file__).resolve().parents[2]
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # no git
        return None
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root:
        return None
    return root


@pytest.mark.parametrize("url", [AGENT_GUIDE_URL, DIAGNOSTICS_URL])
def test_the_guide_link_names_a_tracked_page_on_main(url: str) -> None:
    prefix = "https://github.com/datasciapps/safety-stats/blob/main/"
    assert url.startswith(prefix)
    page = url.removeprefix(prefix)
    root = _checkout()
    if root is None:
        pytest.skip("not run in a git checkout of the repository, so no file is tracked")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", page], cwd=root, capture_output=True, check=False
    )
    assert tracked.returncode == 0, f"{page} is not tracked"


def test_the_pages_link_the_diagnostics_reference_where_a_reader_can_open_it(
    tmp_path: Path,
) -> None:
    """A report is read away from the repository, where a docs/ path leads nowhere."""

    directory = _report().write(tmp_path / "report")
    for name in (INDEX_MD, "report.md"):
        page = (directory / name).read_text(encoding="utf-8")
        assert f"[the diagnostics reference]({DIAGNOSTICS_URL})" in page, name
        assert "`docs/diagnostics.md`" not in page, name
