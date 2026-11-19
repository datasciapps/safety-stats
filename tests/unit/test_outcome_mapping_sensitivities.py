"""Experimental outcome mapping sensitivities: alternative readings of the same score."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from inspect_safety_stats import analyze
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.rollout_files import RolloutFiles
from inspect_safety_stats.schema.reports import MappingSensitivityResult, mapping_results
from tests.report_rendering import rendering_differences

PRIMARY = {"failure_values": ["COMPLY", "PARTIAL"], "safe_values": ["REFUSE"]}
LENIENT = {"failure_values": ["COMPLY"], "safe_values": ["PARTIAL", "REFUSE"]}

BASE: dict[str, Any] = {
    "schema_version": "3.0",
    "analysis": {"id": "mappings", "title": "Outcome mappings", "intent": "exploratory"},
    "inputs": {
        "design_manifest": "design.csv",
        "score": {"name": "verdict", "mapping": PRIMARY},
    },
    "cohorts": {"candidate": {"match": {"model": "m"}}},
    "groups": [{"id": "all"}],
    "outputs": {
        "prompt_bounds": {"enabled": True, "family": "displayed"},
        "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
    },
    "multiplicity": {
        "families": [
            {
                "id": "displayed",
                "purpose": "simultaneous_intervals",
                "method": "bonferroni",
                "alpha": 0.05,
            }
        ]
    },
    "claims": [],
}


def _plan(sensitivities: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    plan = copy.deepcopy(BASE)
    if sensitivities is not None:
        plan["inputs"]["score"]["mapping_sensitivities"] = sensitivities
    return plan


def test_a_plan_without_sensitivities_leaves_them_out_and_keeps_its_hash() -> None:
    plan = AnalysisPlan.model_validate(_plan())
    assert "mapping_sensitivities" not in plan.model_dump(mode="json")["inputs"]["score"]
    assert canonical_plan_hash(plan) == canonical_plan_hash(
        AnalysisPlan.model_validate(_plan([]))
    )


def test_declared_sensitivities_are_kept_in_the_plan_and_its_hash() -> None:
    declared = [{"id": "partial_safe", "title": "Partial refusal is safe", "mapping": LENIENT}]
    plan = AnalysisPlan.model_validate(_plan(declared))
    dumped = plan.model_dump(mode="json", by_alias=True)
    assert dumped["inputs"]["score"]["mapping_sensitivities"][0]["id"] == "partial_safe"
    assert AnalysisPlan.model_validate(dumped) == plan
    assert canonical_plan_hash(plan) != canonical_plan_hash(AnalysisPlan.model_validate(_plan()))


@pytest.mark.parametrize(
    ("sensitivities", "message"),
    [
        (
            [{"id": "a", "mapping": LENIENT}, {"id": "a", "mapping": LENIENT}],
            "identifiers must be unique",
        ),
        ([{"id": "primary", "mapping": LENIENT}], "names the primary mapping"),
        ([{"id": "same", "mapping": PRIMARY}], "repeats the primary mapping"),
        ([{"id": "has space", "mapping": LENIENT}], "id"),
        ([{"id": "a", "mapping": {"failure_values": ["COMPLY"]}}], "requires both"),
    ],
)
def test_invalid_sensitivities_are_rejected(
    sensitivities: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        AnalysisPlan.model_validate(_plan(sensitivities))


@pytest.mark.parametrize(
    "mapping",
    [
        {"failure_values": ["PARTIAL", "COMPLY"], "safe_values": ["REFUSE"]},
        {"failure_values": ["COMPLY", "PARTIAL", "COMPLY"], "safe_values": ["REFUSE", "REFUSE"]},
    ],
    ids=["reordered", "repeated"],
)
def test_an_alternative_that_classifies_as_the_primary_does_is_rejected(
    mapping: dict[str, Any],
) -> None:
    # What counts is how each value is classified, not the order or repeats of the values.
    with pytest.raises(ValidationError, match="repeats the primary mapping"):
        AnalysisPlan.model_validate(_plan([{"id": "same", "mapping": mapping}]))


def test_values_of_another_type_make_another_mapping() -> None:
    # Matching is type-sensitive, so a score of 1 is unmapped under 1.0 or true.
    plan = _plan(
        [
            {"id": "floats", "mapping": {"failure_values": [1.0], "safe_values": [0.0]}},
            {"id": "booleans", "mapping": {"failure_values": [True], "safe_values": [False]}},
        ]
    )
    plan["inputs"]["score"]["mapping"] = {"failure_values": [1], "safe_values": [0]}
    sensitivities = AnalysisPlan.model_validate(plan).inputs.score.mapping_sensitivities
    assert [item.id for item in sensitivities] == ["floats", "booleans"]


def test_a_list_among_the_primary_values_still_lets_the_plan_load() -> None:
    # No score value matches a list, and reading the scores fails on it; the plan loads.
    plan = _plan([{"id": "partial_safe", "mapping": LENIENT}])
    plan["inputs"]["score"]["mapping"] = {"failure_values": [["COMPLY"]], "safe_values": ["X"]}
    (sensitivity,) = AnalysisPlan.model_validate(plan).inputs.score.mapping_sensitivities
    assert sensitivity.id == "partial_safe"


def test_a_judge_score_cannot_declare_sensitivities() -> None:
    plan = _plan()
    plan["inputs"]["judges"] = [
        {
            "id": "j",
            "score": {
                "name": "j",
                "mapping": PRIMARY,
                "mapping_sensitivities": [{"id": "a", "mapping": LENIENT}],
            },
        }
    ]
    with pytest.raises(ValidationError, match="applies to inputs.score only"):
        AnalysisPlan.model_validate(plan)


# Four prompts, four repeats each: p1 always complies, p2 always partially refuses, p3 and p4
# always refuse. A partial refusal as a failure gives a mean of 0.5; as safe, 0.25.
VERDICTS = {"p1": "COMPLY", "p2": "PARTIAL", "p3": "REFUSE", "p4": "REFUSE"}
CLAIM = {
    "id": "low_risk",
    "title": "Failure risk is low",
    "cohort": "candidate",
    "group": "all",
    "measure": "mean",
    "family": "gate",
    "acceptable_upper": 0.9,
    "unacceptable_lower": 0.95,
    "mandatory": True,
}


class _CountingSource:
    def __init__(self, source: Any) -> None:
        self.source = source
        self.reads = 0

    def read(self) -> Any:
        self.reads += 1
        return self.source.read()


def _analysis(tmp_path: Path, sensitivities: list[dict[str, Any]] | None) -> tuple[Path, Any]:
    plan = _plan(sensitivities)
    plan["claims"] = [CLAIM]
    plan["multiplicity"]["families"].append(
        {"id": "gate", "purpose": "all_mandatory", "method": "intersection_union", "alpha": 0.05}
    )
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(f"{prompt},q{prompt},t,4\n" for prompt in VERDICTS),
        encoding="utf-8",
    )
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        json.dumps({"run_key": "r", "model": "m", "status": "success"}) + "\n", encoding="utf-8"
    )
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        "".join(
            json.dumps(
                {"run_key": "r", "item_id": prompt, "repeat": repeat, "score_verdict": verdict}
            )
            + "\n"
            for prompt, verdict in VERDICTS.items()
            for repeat in range(1, 5)
        ),
        encoding="utf-8",
    )
    return path, RolloutFiles(runs=runs, rollouts=rollouts)


def _mean(rows: list[MappingSensitivityResult]) -> MappingSensitivityResult:
    return next(row for row in rows if row.result_id == "candidate:all:mean")


def test_each_mapping_is_analysed_once_from_one_read_and_never_moves_the_decision(
    tmp_path: Path,
) -> None:
    plan, files = _analysis(
        tmp_path,
        [
            {"id": "partial_safe", "title": "Partial refusal is safe", "mapping": LENIENT},
            # Another mapping that reads every value the rollouts give as the primary does.
            {
                "id": "alike",
                "mapping": {
                    "failure_values": ["PARTIAL", "COMPLY"],
                    "safe_values": ["REFUSE", "UNSEEN"],
                },
            },
        ],
    )
    source = _CountingSource(files)
    report = analyze(plan=plan, source=source)
    assert source.reads == 1
    (tmp_path / "baseline").mkdir()
    baseline_plan, baseline_files = _analysis(tmp_path / "baseline", None)
    baseline = analyze(plan=baseline_plan, source=baseline_files)
    # The primary results are those of the same plan without sensitivities.
    assert report.group_risks == baseline.group_risks
    assert report.claims == baseline.claims
    assert report.decision == baseline.decision
    primary = _mean(mapping_results(report))
    assert primary.estimate == pytest.approx(0.5, abs=1e-12)
    lenient, alike = report.outcome_mapping_sensitivities
    assert lenient.id == "partial_safe"
    assert lenient.mapping == LENIENT
    assert lenient.integrity_status == "valid"
    assert _mean(lenient.results).estimate == pytest.approx(0.25, abs=1e-12)
    assert _mean(lenient.results).upper < primary.upper  # type: ignore[operator]
    assert alike.results == mapping_results(report)
    assert alike.decision == report.decision


def test_an_incomplete_alternative_mapping_is_invalid_on_its_own(tmp_path: Path) -> None:
    plan, files = _analysis(
        tmp_path,
        [
            {
                "id": "no_partial",
                "mapping": {"failure_values": ["COMPLY"], "safe_values": ["REFUSE"]},
            }
        ],
    )
    report = analyze(plan=plan, source=files)
    assert report.integrity.status == "valid"
    (sensitivity,) = report.outcome_mapping_sensitivities
    assert sensitivity.integrity_status == "invalid"
    assert sensitivity.diagnostic_counts["SCORE_UNMAPPED"] == 4


def _describing_outcomes(plan: Path, mapping: dict[str, Any] | None = None) -> None:
    """Add what the outcomes are described by: a tail measure at four rollouts per prompt,
    which TAIL_MEASURES_UNINFORMATIVE flags, and a profile that blocks the prompts by task,
    whose design effect CLUSTER_DESIGN_EFFECT reports. p1 and p2 are task a, p3 and p4 task b.
    """

    document = yaml.safe_load(plan.read_text(encoding="utf-8"))
    if mapping is not None:
        document["inputs"]["score"]["mapping"] = mapping
    document["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
    document["design"] = {
        "independence_profiles": {
            "item": {"block_by": ["prompt_id"]},
            "task": {"block_by": ["task_id"]},
        }
    }
    plan.write_text(yaml.safe_dump(document), encoding="utf-8")
    (plan.parent / "design.csv").write_text(
        "prompt_id,question_id,task_id,expected_rollouts\n"
        + "".join(
            f"{prompt},q{prompt},{task},4\n"
            for prompt, task in zip(VERDICTS, ["a", "a", "b", "b"], strict=True)
        ),
        encoding="utf-8",
    )


def test_an_alternative_counts_the_diagnostics_its_mapping_would_as_the_primary(
    tmp_path: Path,
) -> None:
    plan, files = _analysis(tmp_path, [{"id": "partial_safe", "mapping": LENIENT}])
    _describing_outcomes(plan)
    report = analyze(plan=plan, source=files)
    (tmp_path / "as_primary").mkdir()
    as_primary_plan, as_primary_files = _analysis(tmp_path / "as_primary", None)
    _describing_outcomes(as_primary_plan, mapping=LENIENT)
    as_primary = analyze(plan=as_primary_plan, source=as_primary_files)

    counts: dict[str, int] = {}
    for item in as_primary.integrity.diagnostics:
        counts[item.code] = counts.get(item.code, 0) + 1
    assert counts == {"CLUSTER_DESIGN_EFFECT": 1, "TAIL_MEASURES_UNINFORMATIVE": 1}
    (sensitivity,) = report.outcome_mapping_sensitivities
    assert sensitivity.diagnostic_counts == counts
    assert sensitivity.integrity_status == as_primary.integrity.status


def test_a_receipt_for_another_plan_makes_each_alternative_invalid_too(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from inspect_safety_stats.config.registration import write_receipt

    plan, files = _analysis(tmp_path, [{"id": "partial_safe", "mapping": LENIENT}])
    (sensitivity,) = analyze(plan=plan, source=files).outcome_mapping_sensitivities
    assert sensitivity.integrity_status == "valid"
    receipt = tmp_path / "plan.receipt.json"
    write_receipt(
        receipt, "0" * 64, package_version="0", now=datetime(2024, 6, 1, tzinfo=timezone.utc)
    )

    report = analyze(plan=plan, source=files, registration_receipt=receipt)

    # The receipt is for another plan, which voids every analysis of this one.
    assert report.integrity.status == "invalid"
    (sensitivity,) = report.outcome_mapping_sensitivities
    assert sensitivity.integrity_status == "invalid"
    assert sensitivity.diagnostic_counts["PLAN_RECEIPT_MISMATCH"] == 1


def test_the_report_shows_the_mappings_side_by_side(tmp_path: Path) -> None:
    plan, files = _analysis(tmp_path, [{"id": "partial_safe", "mapping": LENIENT}])
    directory = analyze(plan=plan, source=files).write(tmp_path / "report")
    document = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in document["outcome_mapping_sensitivities"]] == ["partial_safe"]
    table = pd.read_csv(directory / "outcome-mapping-sensitivity.csv")
    assert list(table.columns) == [
        "kind",
        "result_id",
        "cohort_id",
        "group_id",
        "measure_id",
        "mapping_id",
        "estimate",
        "lower",
        "upper",
        "status",
        "descriptive",
    ]
    mean = table[table["result_id"] == "candidate:all:mean"]
    assert list(mean["mapping_id"]) == ["primary", "partial_safe"]
    assert list(mean["estimate"]) == pytest.approx([0.5, 0.25], abs=1e-12)
    assert list(table["kind"].drop_duplicates()) == ["group_risk", "claim"]
    # An alternative's rows, its claim's status among them, never enter the decision.
    assert list(zip(table["mapping_id"], table["descriptive"], strict=True)) == [
        ("primary", False),
        ("partial_safe", True),
    ] * 2
    checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    assert "outcome-mapping-sensitivity.csv" in checksums
    markdown = (directory / "report-full.md").read_text(encoding="utf-8")
    assert "## Outcome mapping sensitivity (experimental)" in markdown
    assert "| partial_safe | — | COMPLY | PARTIAL, REFUSE | `valid` |" in markdown
    assert "| primary | — | COMPLY, PARTIAL | REFUSE | `valid` |" in markdown
    # The section, the CSV and the primary mapping all render from report.json alone (L38).
    assert rendering_differences(directory) == []


def test_a_report_without_sensitivities_has_none_of_their_outputs(tmp_path: Path) -> None:
    plan, files = _analysis(tmp_path, None)
    directory = analyze(plan=plan, source=files).write(tmp_path / "report")
    document = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert "outcome_mapping_sensitivities" not in document
    assert not (directory / "outcome-mapping-sensitivity.csv").exists()
    assert "Outcome mapping sensitivity" not in (directory / "report-full.md").read_text(
        encoding="utf-8"
    )


def test_shown_decisions_mark_a_sensitivity_decision_as_descriptive(tmp_path: Path) -> None:
    plan, files = _analysis(tmp_path, [{"id": "partial_safe", "mapping": LENIENT}])
    document = yaml.safe_load(plan.read_text(encoding="utf-8"))
    document["report"] = {"include_decisions": True}
    plan.write_text(yaml.safe_dump(document), encoding="utf-8")
    directory = analyze(plan=plan, source=files).write(tmp_path / "report")
    markdown = (directory / "report-full.md").read_text(encoding="utf-8")
    row = "| partial_safe | — | COMPLY | PARTIAL, REFUSE | `valid` | `INCONCLUSIVE` (descriptive) |"
    assert row in markdown
    claim_rows = [line for line in markdown.splitlines() if line.startswith("| claim low_risk |")]
    assert [line.split(" | ")[1] for line in claim_rows] == ["primary", "partial_safe"]
    assert claim_rows[0].endswith(" | INCONCLUSIVE |")
    assert claim_rows[1].endswith(" | INCONCLUSIVE (descriptive) |")
