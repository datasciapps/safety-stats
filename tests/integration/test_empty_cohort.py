"""A cohort that no evaluation fills: analyze reports it as validate does, and goes on.

``validate`` reports such a cohort as COHORT_EMPTY, an ERROR. ``analyze`` used to raise instead
of writing a report whenever the plan calibrated direct means or comparisons over hierarchy
frontiers, as ``init``'s plan does, because the empty cohort's targets formed no blocks. It now
writes the report, with integrity invalid for the same errors, and calibrates every other
cohort as it would were the empty cohort's ledger complete.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from safety_stats import analyze, validate
from safety_stats.cli import app
from safety_stats.ingest.rollout_files import RolloutFiles
from safety_stats.scaffold import scaffold_plan
from tests.cli_output import plain
from tests.diagnostic_codes.harness import Inputs, Outcome, run_analysis, write_inputs
from tests.mock_inspect_logs import write_mock_logs

FAMILY = {"purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}
MODELS = {"candidate": "model/candidate", "baseline": "model/baseline", "third": "model/third"}
# Prompt, question, task and domain: two tasks of two questions each, one question of two
# prompts in each task, so that every hierarchy level and every profile has several blocks.
PROMPTS = (
    ("p1", "q1", "t1", "a"),
    ("p2", "q1", "t1", "a"),
    ("p3", "q2", "t1", "a"),
    ("p4", "q3", "t2", "b"),
    ("p5", "q3", "t2", "b"),
    ("p6", "q4", "t2", "b"),
)
REPEATS = 4
PROFILES = {
    "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"},
    "prompt": {"block_by": ["prompt_id"], "comparison_coupling": "joint"},
    "question": {"block_by": ["question_id"], "comparison_coupling": "independent"},
    "domain": {"block_by": ["domain"], "comparison_coupling": "joint"},
    "run": {"block_by": ["run_key"], "comparison_coupling": "independent"},
}
LEDGER_REQUIRED = "A complete valid attempted-rollout ledger is required."
BOTH_LEDGERS_REQUIRED = "Both arms require a complete valid attempted-rollout ledger."


def _inputs(cohorts: tuple[str, ...], primary: str = "rollout") -> Inputs:
    """Each cohort one run of every prompt; direct means, nodes and claims for each cohort."""

    runs: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    for number, cohort in enumerate(cohorts, start=1):
        run_key = f"r{number}"
        runs.append(
            {
                "run_key": run_key,
                "model": MODELS[cohort],
                "status": "success",
                "started_at": "2026-02-01T00:00:00Z",
            }
        )
        for position, (prompt, *_) in enumerate(PROMPTS):
            for repeat in range(1, REPEATS + 1):
                failing = (repeat + position + number) % (2 + number) == 0
                rollouts.append(
                    {
                        "run_key": run_key,
                        "item_id": prompt,
                        "repeat": repeat,
                        "score_harmful": "FAIL" if failing else "PASS",
                    }
                )
    independence = {
        "primary": primary,
        "sensitivities": [profile for profile in PROFILES if profile != primary],
    }
    plan: dict[str, Any] = {
        "schema_version": "3.0",
        "analysis": {"id": "empty_cohort", "title": "Empty cohort", "intent": "exploratory"},
        "inputs": {
            "design_manifest": "design.csv",
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        },
        "design": {
            "hierarchy": ["task_id", "question_id", "prompt_id"],
            "independence_profiles": copy.deepcopy(PROFILES),
        },
        "cohorts": {cohort: {"match": {"model": MODELS[cohort]}} for cohort in cohorts},
        "groups": [{"id": "all"}, {"id": "b", "filter": {"domain": "b"}}],
        "outputs": {
            "prompt_bounds": {"enabled": True, "family": "displayed"},
            "risk_measures": [
                {"id": "mean", "type": "weighted_mean"},
                {"id": "worst", "type": "maximum"},
            ],
            "direct_mean_intervals": {
                "enabled": True,
                "method": "weighted_kl_chernoff",
                "family": "direct",
                "independence": independence,
                "hierarchy_nodes": True,
                "hierarchy_calibration": "frontier_sidak",
                "hierarchy_region": "linear_program",
            },
        },
        "multiplicity": {
            "families": [
                {"id": "displayed", **FAMILY},
                {"id": "direct", **FAMILY},
                {"id": "comparisons", **FAMILY},
                {
                    "id": "claims",
                    "purpose": "all_mandatory",
                    "method": "bonferroni",
                    "alpha": 0.05,
                },
            ]
        },
        "claims": [
            {
                "id": f"low_{cohort}",
                "title": "Mean failure rate is low",
                "cohort": cohort,
                "group": "all",
                "measure": "mean",
                "family": "claims",
                "acceptable_upper": 0.9,
                "unacceptable_lower": 0.95,
            }
            for cohort in cohorts
        ],
    }
    manifest = [
        {
            "prompt_id": prompt,
            "question_id": question,
            "task_id": task,
            "domain": domain,
            "expected_rollouts": REPEATS,
        }
        for prompt, question, task, domain in PROMPTS
    ]
    return Inputs(plan=plan, manifest=manifest, runs=runs, rollouts=rollouts)


def _comparison(
    inputs: Inputs, candidate: str, baseline: str, calibration: str = "frontier_sidak"
) -> None:
    inputs.plan.setdefault("comparisons", []).append(
        {
            "id": f"{candidate}_vs_{baseline}",
            "candidate": candidate,
            "baseline": baseline,
            "group": "all",
            "measure": "mean",
            "estimand": "risk_difference",
            "method": "weighted_kl_chernoff",
            "family": "comparisons",
            "independence": {"primary": "rollout", "sensitivities": ["prompt", "question", "run"]},
            "hierarchy_nodes": True,
            "hierarchy_calibration": calibration,
        }
    )


def _emptied(inputs: Inputs, cohort: str) -> Inputs:
    """The inputs with ``cohort`` naming a model that no run records."""

    emptied = copy.deepcopy(inputs)
    emptied.plan["cohorts"][cohort]["match"]["model"] = "model/unevaluated"
    return emptied


def _errors(diagnostics: Any) -> set[str]:
    return {item.code for item in diagnostics if item.severity == "ERROR"}


def _validated_errors(inputs: Inputs, directory: Path) -> set[str]:
    plan = write_inputs(inputs, directory)
    summary = validate(
        plan=plan,
        source=RolloutFiles(runs=directory / "runs.jsonl", rollouts=directory / "rollouts.jsonl"),
    )
    return _errors(summary.diagnostics)


def _cohort_rows(outcome: Outcome, cohort: str) -> dict[str, Any]:
    """Everything the report computes for one cohort, as JSON, and its claims' bounds.

    A claim's status depends on the whole report's integrity, so only its numbers are kept.
    """

    report = outcome.report

    def rows(items: Any) -> list[Any]:
        return [item.model_dump(mode="json") for item in items if item.cohort_id == cohort]

    return {
        "prompt_estimates": rows(report.prompt_estimates),
        "group_risks": rows(report.group_risks),
        "hierarchy_risks": rows(report.hierarchy_risks),
        "claims": [
            (claim.claim_id, claim.estimate, claim.lower, claim.upper)
            for claim in report.claims
            if claim.claim_id == f"low_{cohort}"
        ],
    }


def _withheld(row: Any) -> bool:
    """Whether a result row has no bound: neither its own nor any of its intervals'."""

    return (row.lower, row.upper) == (None, None) and all(
        (interval.lower, interval.upper) == (None, None) for interval in row.intervals
    )


def _comparison_rows(outcome: Outcome, comparison_id: str) -> list[Any]:
    return [
        row.model_dump(mode="json")
        for row in outcome.report.comparisons
        if row.comparison_id == comparison_id
    ]


@pytest.fixture(scope="module")
def mock_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_mock_logs(tmp_path_factory.mktemp("mock") / "logs")


def test_inits_plan_with_a_cohort_that_matches_no_evaluation_is_reported_invalid(
    mock_logs: Path, tmp_path: Path
) -> None:
    complete = scaffold_plan(mock_logs, tmp_path / "complete", failure_values=["FAIL"],
                             group_by="category")
    result = scaffold_plan(mock_logs, tmp_path / "wrong", failure_values=["FAIL"],
                           group_by="category")
    text = result.plan_path.read_text(encoding="utf-8")
    assert "model: mockllm/beta" in text
    result.plan_path.write_text(
        text.replace("model: mockllm/beta", "model: mockllm/gamma"),
        encoding="utf-8",
        newline="\n",
    )

    summary = validate(plan=result.plan_path, logs=str(mock_logs))
    report = analyze(plan=result.plan_path, logs=str(mock_logs))
    assert report.integrity.status == summary.status == "invalid"
    assert "COHORT_EMPTY" in _errors(report.integrity.diagnostics)
    assert _errors(report.integrity.diagnostics) == _errors(summary.diagnostics)

    # The cohort the logs do fill is reported as with the plan that names both models.
    full = analyze(plan=complete.plan_path, logs=str(mock_logs))
    assert full.integrity.status == "valid"

    def alpha(rows: Any) -> list[Any]:
        return [row.model_dump(mode="json") for row in rows if row.cohort_id == "alpha"]

    assert alpha(report.group_risks) == alpha(full.group_risks)
    assert alpha(report.hierarchy_risks) == alpha(full.hierarchy_risks)
    assert any(row.intervals for row in report.hierarchy_risks if row.cohort_id == "alpha")
    for row in report.group_risks:
        if row.cohort_id == "beta":
            assert _withheld(row)
            assert LEDGER_REQUIRED in row.warnings

    output = tmp_path / "out"
    command = CliRunner().invoke(
        app,
        ["analyze", "--plan", str(result.plan_path), "--logs", str(mock_logs),
         "--output", str(output)],
    )
    assert command.exit_code == 3, plain(command.output)
    assert (output / "report.json").is_file()


@pytest.mark.parametrize("calibration", ["frontier_sidak", "flat_bonferroni"])
@pytest.mark.parametrize("empty", ["candidate", "third"])
def test_a_comparison_with_an_empty_arm_leaves_the_other_comparisons_as_they_were(
    empty: str, calibration: str, tmp_path: Path
) -> None:
    inputs = _inputs(("candidate", "baseline", "third"))
    # Direct means off, so that the comparisons' calibration is what the test sees.
    inputs.plan["outputs"]["direct_mean_intervals"] = {"enabled": False}
    _comparison(inputs, "candidate", "baseline", calibration)
    _comparison(inputs, "third", "baseline", calibration)
    emptied = _emptied(inputs, empty)

    full = run_analysis(inputs, tmp_path / "full")
    outcome = run_analysis(emptied, tmp_path / "empty")
    assert full.status == "valid"
    assert outcome.status == "invalid"
    assert "COHORT_EMPTY" in _errors(outcome.diagnostics)
    assert _errors(outcome.diagnostics) == _validated_errors(emptied, tmp_path / "validate")

    kept = "third_vs_baseline" if empty == "candidate" else "candidate_vs_baseline"
    withheld = f"{empty}_vs_baseline"
    expected, found = _comparison_rows(full, kept), _comparison_rows(outcome, kept)
    assert expected

    # Under the run profile a block is a run, which the empty arm does not have, so the blocks
    # its targets would form are unknown. Under frontier_sidak they are calibrated as if each
    # overlapped the others, which spends the most alpha: the other comparison's intervals
    # under that profile may be wider than with a complete ledger, and never narrower.
    def by_run(rows: list[Any]) -> list[Any]:
        return [
            interval
            for row in rows
            for interval in row["intervals"]
            if interval["independence_profile"] == "run"
        ]

    for complete, conservative in zip(by_run(expected), by_run(found), strict=True):
        assert conservative["alpha_lower"] <= complete["alpha_lower"]
        assert conservative["alpha_upper"] <= complete["alpha_upper"]
        assert conservative["lower"] <= complete["lower"]
        assert conservative["upper"] >= complete["upper"]
    for rows in (expected, found):
        for row in rows:
            row["intervals"] = [
                item for item in row["intervals"] if item["independence_profile"] != "run"
            ]
    # Every other row and interval is what a complete ledger gives.
    assert found == expected
    rows = [row for row in outcome.report.comparisons if row.comparison_id == withheld]
    assert rows
    for row in rows:
        assert _withheld(row)
        assert BOTH_LEDGERS_REQUIRED in row.warnings


@pytest.mark.parametrize(
    ("primary", "missing"),
    [
        ("rollout", None),
        ("question", None),
        ("domain", None),
        ("run", None),
        ("rollout", "p3"),
    ],
    ids=["rollout", "question", "domain", "run", "rollout-missing-prompt"],
)
def test_a_cohort_without_a_complete_ledger_leaves_the_other_cohort_as_it_was(
    primary: str, missing: str | None, tmp_path: Path
) -> None:
    inputs = _inputs(("candidate", "baseline"), primary)
    _comparison(inputs, "candidate", "baseline")
    if missing is None:
        changed = _emptied(inputs, "candidate")
    else:
        changed = copy.deepcopy(inputs)
        changed.rollouts = [
            row
            for row in changed.rollouts
            if not (row["run_key"] == "r1" and row["item_id"] == missing)
        ]

    full = run_analysis(inputs, tmp_path / "full")
    outcome = run_analysis(changed, tmp_path / "changed")
    assert full.status == "valid"
    assert outcome.status == "invalid"
    assert _errors(outcome.diagnostics) == _validated_errors(changed, tmp_path / "validate")
    if missing is None:
        assert "COHORT_EMPTY" in _errors(outcome.diagnostics)

    baseline = _cohort_rows(full, "baseline")
    assert any(row["intervals"] for row in baseline["hierarchy_risks"])
    assert _cohort_rows(outcome, "baseline") == baseline
    # The candidate's targets that hold a prompt it has no rows of are withheld, as are the
    # comparisons of those targets; the rest are reported.
    def lacks_rows(group: str, path: dict[str, str]) -> bool:
        if missing is None:
            return True
        return group == "all" and path.get("task_id", "t1") == "t1" and (
            path.get("question_id", "q2") == "q2"
        )

    rows = [
        *(row for row in outcome.report.group_risks if row.measure_id == "mean"),
        *outcome.report.hierarchy_risks,
    ]
    candidate = [row for row in rows if row.cohort_id == "candidate"]
    assert candidate
    for row in candidate:
        expected = lacks_rows(row.group_id, getattr(row, "path", {}))
        assert _withheld(row) is expected, (row.group_id, getattr(row, "path", {}))
        assert (LEDGER_REQUIRED in row.warnings) is expected
    assert outcome.report.comparisons
    for row in outcome.report.comparisons:
        expected = lacks_rows("all", row.path)
        assert _withheld(row) is expected, row.path
        assert (BOTH_LEDGERS_REQUIRED in row.warnings) is expected
