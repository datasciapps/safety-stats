"""The design manifest's identifier and grouping columns are read as written, as text.

The analysis read the manifest with pandas' type inference. Labels that read as the same number
or boolean (`1` and `01`, `1` and `1.0`, `true` and `True`) merged into one stratum, block,
hierarchy node or group without a word, an identifier such as `0001` became `1`, and a blank
value or one pandas reads as missing (`NA`, `None`) behaved differently with the pandas version.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from inspect_safety_stats import analyze
from inspect_safety_stats.cli import app
from inspect_safety_stats.config.loader import load_plan
from inspect_safety_stats.ingest.normalize import read_inputs
from inspect_safety_stats.ingest.rollout_files import RolloutFiles
from inspect_safety_stats.schema.reports import SafetyReport
from tests.cli_output import plain

REPEATS = 4

_PLAN = """
schema_version: "3.0"
analysis: {id: labels, title: Manifest labels, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
design:
@DESIGN@
cohorts:
  candidate: {match: {model: m}}
groups:
@GROUPS@
outputs:
  prompt_bounds: {enabled: false}
  direct_mean_intervals:
    enabled: true
    method: weighted_hoeffding
    family: means
    simultaneous: true
    independence: {primary: @PRIMARY@}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
"""

_ROLLOUT = "    rollout: {block_by: [rollout], comparison_coupling: independent}"


def _write(
    tmp_path: Path,
    manifest: str,
    observed: list[str],
    *,
    design: str,
    groups: str = "  - {id: all}",
    primary: str = "rollout",
) -> tuple[Path, RolloutFiles]:
    (tmp_path / "design.csv").write_text(manifest, encoding="utf-8", newline="\n")
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        _PLAN.replace("@DESIGN@", design)
        .replace("@GROUPS@", groups)
        .replace("@PRIMARY@", primary),
        encoding="utf-8",
        newline="\n",
    )
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps({"run_key": "r1", "model": "m", "status": "success"}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps(
                {
                    "run_key": "r1",
                    "item_id": item,
                    "repeat": repeat,
                    "score_harmful": "FAIL" if repeat == 1 and index % 2 == 0 else "PASS",
                }
            )
            + "\n"
            for index, item in enumerate(observed)
            for repeat in range(1, REPEATS + 1)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return plan, RolloutFiles(runs=runs, rollouts=rollouts)


def _errors(report: SafetyReport) -> list[tuple[str, str]]:
    return sorted(
        (item.code, item.scope or "")
        for item in report.integrity.diagnostics
        if item.severity == "ERROR"
    )


def _ingest_message(report: SafetyReport) -> str:
    (failure,) = [item for item in report.integrity.diagnostics if item.code == "INGEST_FAILED"]
    return failure.message


# A frame of two strata of four items, each item drawn with probability 1/2, so the design
# draws two items from each stratum.
_SAMPLED = """  type: sampled
  sampling: fixed_size
  stratum_column: stratum
  inclusion_probability_column: pi
  hierarchy: [task_id, question_id, prompt_id]
  weights: {strategy: equal_prompt}
  independence_profiles:
""" + _ROLLOUT


def _frame(first: str, second: str) -> str:
    rows = [f"a{index},qa{index},t,{first},0.5,{REPEATS}" for index in range(1, 5)]
    rows += [f"b{index},qb{index},t,{second},0.5,{REPEATS}" for index in range(1, 5)]
    return "prompt_id,question_id,task_id,stratum,pi,expected_rollouts\n" + "\n".join(rows) + "\n"


@pytest.mark.parametrize(
    "labels", [("1", "01"), ("1", "1.0"), ("true", "True")], ids=["zero", "decimal", "case"]
)
def test_strata_whose_labels_read_as_one_number_stay_two_strata(
    tmp_path: Path, labels: tuple[str, str]
) -> None:
    """A run that drew one item from one stratum and three from the other misses its plan.

    Merged into one stratum of eight, the four draws matched a planned four, so the analysis
    certified a run that broke its fixed-size design.
    """

    first, second = labels
    plan, source = _write(
        tmp_path, _frame(first, second), ["a1", "b1", "b2", "b3"], design=_SAMPLED
    )
    report = analyze(plan=plan, source=source)
    assert _errors(report) == sorted(
        [
            ("SAMPLE_DRAWS_MISMATCH", f"cohorts/candidate/strata/{first}"),
            ("SAMPLE_DRAWS_MISMATCH", f"cohorts/candidate/strata/{second}"),
        ]
    )


def test_strata_stay_as_written_beyond_the_parser_s_first_chunk(tmp_path: Path) -> None:
    """pandas infers a column's type chunk by chunk in a large file. Here the first chunk holds
    only `1` and `01`, which it read as the number 1, so stratum `01` joined stratum `1`.

    The frame is read as the analysis reads it, but not analysed, as analysing its 135,000
    prompts is slow: the smaller tests above show what a merged stratum does to the analysis.
    """

    size = 2**16
    labels = ["1" if index % 2 == 0 else "01" for index in range(2 * size)] + ["x"] * 4096
    # Two draws from each stratum.
    counts = Counter(labels)
    rows = [
        f"p{index},q{index},t,{label},{2 / counts[label]!r},{REPEATS}"
        for index, label in enumerate(labels)
    ]
    manifest = "prompt_id,question_id,task_id,stratum,pi,expected_rollouts\n"
    plan, source = _write(tmp_path, manifest + "\n".join(rows) + "\n", ["p0"], design=_SAMPLED)
    # The premise: pandas' own reading of this frame is chunked, and the chunks disagree.
    with pytest.warns(pd.errors.DtypeWarning):
        pd.read_csv(tmp_path / "design.csv")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = read_inputs(load_plan(plan), source)
    assert raw.manifest["stratum"].tolist() == labels
    # Nor does the analysis warn of a column whose type it never infers.
    assert not [item for item in caught if issubclass(item.category, pd.errors.DtypeWarning)]


@pytest.mark.parametrize("label", ["NA", "None", "N/A", "null", "nan", "#N/A", "<NA>", " NULL "])
def test_a_stratum_label_that_marks_a_missing_value_is_an_error(
    tmp_path: Path, label: str
) -> None:
    """R, spreadsheets, databases and pandas write these for a missing value. pandas read them
    as missing, so the analysis failed with an unrelated message or formed one stratum of
    them, depending on its version; read as text, they would have been a stratum silently."""

    plan, source = _write(tmp_path, _frame(label, "b"), ["a1", "a2", "b1", "b2"], design=_SAMPLED)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    message = _ingest_message(report)
    assert (
        "design manifest column 'stratum' has no value in "
        f"row 2 (prompt_id 'a1', written {label!r}), row 3 (prompt_id 'a2', written {label!r})"
    ) in message
    assert "recode a label meant literally" in message


_FIXED = """  type: fixed
  require_expected_rollouts: false
  hierarchy: [task_id, question_id, prompt_id]
  weights: {strategy: equal_prompt}
  independence_profiles:
""" + _ROLLOUT


def test_identifiers_keep_their_leading_zeros(tmp_path: Path) -> None:
    """`0001` became the number 1, so no rollout of item `0001` joined its manifest row."""

    manifest = "prompt_id,question_id,task_id\n0001,007,01\n0002,007,01\n"
    plan, source = _write(tmp_path, manifest, ["0001", "0002"], design=_FIXED)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    assert [(row.prompt_id, row.question_id, row.task_id) for row in report.prompt_estimates] == [
        ("0001", "007", "01"),
        ("0002", "007", "01"),
    ]


def test_identifiers_padded_in_the_manifest_are_named_to_the_logs(tmp_path: Path) -> None:
    """The logs record `1` and the manifest `0001`: the messages said to add the prompt to the
    manifest and to run the missing one, when one identifier has to be written as the other."""

    manifest = "prompt_id,question_id,task_id\n0001,q,t\n0002,q,t\n0003,q,t\n"
    plan, source = _write(tmp_path, manifest, ["1", "2", "3", "x"], design=_FIXED)
    report = analyze(plan=plan, source=source)
    messages = {
        (item.code, (item.details or {}).get("prompt_id")): item.message
        for item in report.integrity.diagnostics
    }
    hint = "the same number written differently. Identifiers are compared as written"
    assert messages[("PROMPT_UNEXPECTED", "1")] == (
        f"observed prompt '1' is absent from the design manifest; the manifest has '0001', {hint}"
        ": write the manifest's prompt_id as the logs record it"
    )
    assert messages[("PROMPT_MISSING", "0001")] == (
        f"prompt '0001' has no observations in cohort 'candidate'; the logs have '1', {hint}: "
        "write the manifest's prompt_id as the logs record it"
    )
    assert messages[("PROMPT_UNEXPECTED", "x")] == (
        "observed prompt 'x' is absent from the design manifest"
    )


def test_identifiers_that_read_as_one_number_are_two_prompts(tmp_path: Path) -> None:
    """`1` and `01` were one duplicated prompt_id."""

    manifest = "prompt_id,question_id,task_id\n1,q1,t\n01,q2,t\n"
    plan, source = _write(tmp_path, manifest, ["1", "01"], design=_FIXED)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    assert sorted(row.prompt_id for row in report.prompt_estimates) == ["01", "1"]


_CLUSTERS = """  type: fixed
  require_expected_rollouts: false
  hierarchy: [task_id, question_id, prompt_id]
  weights: {strategy: equal_prompt}
  independence_profiles:
    cluster: {block_by: [cluster], comparison_coupling: joint}
""" + _ROLLOUT


def test_blocks_whose_labels_read_as_one_number_stay_two_blocks(tmp_path: Path) -> None:
    manifest = (
        "prompt_id,question_id,task_id,cluster\np1,q1,t,1\np2,q2,t,1\np3,q3,t,01\np4,q4,t,01\n"
    )
    plan, source = _write(
        tmp_path, manifest, ["p1", "p2", "p3", "p4"], design=_CLUSTERS, primary="cluster"
    )
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    primary = next(item for item in mean.intervals if item.role == "primary")
    assert primary.independent_block_count == 2


_CATEGORIES = """  type: fixed
  require_expected_rollouts: false
  hierarchy: [category, prompt_id]
  weights: {strategy: nested_equal, hierarchy: [category, prompt_id]}
  independence_profiles:
""" + _ROLLOUT


def test_hierarchy_levels_whose_labels_read_as_one_number_stay_two_nodes(
    tmp_path: Path,
) -> None:
    """Nested weights gave categories `1` and `01` one share between them."""

    manifest = "prompt_id,category\np1,1\np2,01\np3,01\np4,01\n"
    plan, source = _write(tmp_path, manifest, ["p1", "p2", "p3", "p4"], design=_CATEGORIES)
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    weights = {row["prompt_id"]: row["weight"] for row in report.design["prompts"]}
    expected = {"p1": 1 / 2, "p2": 1 / 6, "p3": 1 / 6, "p4": 1 / 6}
    assert weights == pytest.approx(expected, rel=1e-12, abs=0)
    assert sorted(row["category"] for row in report.design["prompts"]) == ["01", "01", "01", "1"]


def test_planning_weights_hierarchy_levels_as_analyze_reads_them(tmp_path: Path) -> None:
    """`plan fixed-design` gave categories `1` and `01` one share, as analyze did."""

    manifest = tmp_path / "design.csv"
    manifest.write_text(
        "prompt_id,category,expected_rollouts\np1,1,10\np2,01,10\np3,01,10\np4,01,10\n",
        encoding="utf-8",
        newline="\n",
    )
    result = CliRunner().invoke(
        app,
        [
            "plan",
            "fixed-design",
            "--manifest",
            str(manifest),
            "--threshold",
            "0.5",
            "--hierarchy",
            "category,prompt_id",
        ],
    )
    assert result.exit_code == 0, plain(result.output)
    # Weights 1/2, 1/6, 1/6 and 1/6 over ten rollouts each; one category of four gives 40.
    assert "Effective sample size: 30 " in plain(result.output)


_PLANNED = """
schema_version: "3.0"
analysis:
  id: planning
  title: Planning
  intent: confirmatory
  plan_frozen_at: 2025-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  weights: {strategy: nested_equal, hierarchy: [category, prompt_id]}
  independence_profiles:
""" + _ROLLOUT + """
cohorts:
  candidate: {match: {model: m}}
groups:
  - {id: all}
outputs:
  prompt_bounds: {enabled: false}
  direct_mean_intervals:
    enabled: true
    method: weighted_hoeffding
    family: means
    independence: {primary: rollout}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: all_mean, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.3, unacceptable_lower: 0.6}
"""


def test_planning_from_a_plan_weights_hierarchy_levels_as_analyze_reads_them(
    tmp_path: Path,
) -> None:
    """`plan fixed-design --plan` gave categories `1` and `01` one share, as analyze did."""

    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,category,expected_rollouts\n"
        "p1,q1,t,1,10\np2,q2,t,01,10\np3,q3,t,01,10\np4,q4,t,01,10\n",
        encoding="utf-8",
        newline="\n",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(_PLANNED, encoding="utf-8", newline="\n")
    result = CliRunner().invoke(app, ["plan", "fixed-design", "--plan", str(plan)])
    assert result.exit_code == 0, plain(result.output)
    # Weights 1/2, 1/6, 1/6 and 1/6 over ten rollouts each; one category of four gives 40.
    assert "Effective sample size: 30 " in plain(result.output)


_TIERS = "prompt_id,question_id,task_id,tier\np1,q1,t,1\np2,q2,t,1\np3,q3,t,01\np4,q4,t,01\n"


def test_a_group_filter_matches_the_label_as_written(tmp_path: Path) -> None:
    """A filter on `01` matched nothing, as the column held the number 1."""

    plan, source = _write(
        tmp_path,
        _TIERS,
        ["p1", "p2", "p3", "p4"],
        design=_FIXED,
        groups="  - {id: tier, filter: {tier: '01'}}",
    )
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "valid", report.integrity.diagnostics
    tier = sorted(row["prompt_id"] for row in report.design["prompts"] if row["group_id"] == "tier")
    assert tier == ["p3", "p4"]


@pytest.mark.parametrize("value", ["1", "true", "1.0", "null"])
def test_a_group_filter_value_that_is_not_text_is_refused(tmp_path: Path, value: str) -> None:
    """An unquoted 1 matched both `1` and `01`; YAML has already lost how it was written."""

    plan, source = _write(
        tmp_path,
        _TIERS,
        ["p1", "p2", "p3", "p4"],
        design=_FIXED,
        groups=f"  - {{id: tier, filter: {{tier: {value}}}}}",
    )
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    message = _ingest_message(report)
    assert "group 'tier' filters manifest column 'tier'" in message
    assert "in quotes" in message


# Each manifest holds one cell without a value, written @, and the row it is in.
_EMPTY = {
    "prompt_id": (
        "prompt_id,question_id,task_id\np1,q1,t\n@,q2,t\np3,q3,t\n",
        _FIXED,
        "  - {id: all}",
        "row 3",
    ),
    "question_id": (
        "prompt_id,question_id,task_id\np1,q1,t\np2,@,t\np3,q3,t\n",
        _FIXED,
        "  - {id: all}",
        "row 3 (prompt_id 'p2'",
    ),
    "stratum": (
        _frame("a", "b").replace("a3,qa3,t,a,", "a3,qa3,t,@,"),
        _SAMPLED,
        "  - {id: all}",
        "row 4 (prompt_id 'a3'",
    ),
    "cluster": (
        "prompt_id,question_id,task_id,cluster\np1,q1,t,c\np2,q2,t,@\np3,q3,t,c\n",
        _CLUSTERS,
        "  - {id: all}",
        "row 3 (prompt_id 'p2'",
    ),
    "category": (
        "prompt_id,category\np1,c\np2,c\np3,@\n",
        _CATEGORIES,
        "  - {id: all}",
        "row 4 (prompt_id 'p3'",
    ),
    "tier": (
        "prompt_id,question_id,task_id,tier\np1,q1,t,x\np2,q2,t,@\np3,q3,t,x\n",
        _FIXED,
        "  - {id: all}\n  - {id: x, filter: {tier: x}}",
        "row 3 (prompt_id 'p2'",
    ),
}


@pytest.mark.parametrize("cell", ["", " ", "NA", "None"], ids=["blank", "space", "NA", "None"])
@pytest.mark.parametrize("column", sorted(_EMPTY))
def test_a_label_without_a_value_is_an_error_naming_the_column_and_row(
    tmp_path: Path, column: str, cell: str
) -> None:
    """A blank value merged, split, dropped or crashed, depending on the column and the pandas
    version. A missing-value marker did the same at first, and was then read as a label: an
    `NA` cluster was one independence block, an `NA` question one question."""

    manifest, design, groups, row = _EMPTY[column]
    manifest = manifest.replace("@", cell)
    if cell.strip():
        row += f", written {cell!r})" if "(" in row else f" (written {cell!r})"
    elif "(" in row:
        row += ")"
    plan, source = _write(tmp_path, manifest, ["p1", "p3", "a1", "a2", "b1", "b2"], design=design)
    plan.write_text(
        plan.read_text(encoding="utf-8").replace("  - {id: all}", groups),
        encoding="utf-8",
        newline="\n",
    )
    report = analyze(plan=plan, source=source)
    assert report.integrity.status == "invalid"
    message = _ingest_message(report)
    assert f"design manifest column {column!r} has no value in {row}, counting" in message
