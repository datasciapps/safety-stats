"""What a validated judge calibration certifies on the generative path (#109).

The labels pool every classified unit of the cohort with equal weight, so the measured rho
is the judge's accuracy averaged over the whole cohort. A node where the judge misses more
failures than average needs its own rho; correcting it with the pooled one can put the
corrected upper bound below the node's true risk. Only the root of an equal-weight design is
measured by the pooled labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from safety_stats.generative.inference import NodeEstimate
from safety_stats.generative.judge import pooled_calibration_gap
from safety_stats.generative.ledger import read_rewrites, read_selections
from safety_stats.generative.runner import analyze_run, materialize_run
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan

MODEL = "provider/candidate"
HARD_SEED = "semantic_seed_id=s003"
HARD_NODE = "all|problem_class=bio|semantic_seed_id=s003"


def write_plan(tmp_path: Path, sampling: dict[str, Any]) -> Path:
    plan = base_plan()
    plan["sampling"] = sampling
    plan["target"]["cohorts"] = {"candidate": {"model": MODEL}}
    plan["judge_calibration"] = {"mode": "validated", "validation_manifest": "labels.csv"}
    (tmp_path / "seeds.csv").write_text(SEEDS_WITH_TEXT, encoding="utf-8")
    (tmp_path / "template.txt").write_text("Rewrite: {seed}\n", encoding="utf-8")
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return path


def run_with_judge_misses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sampling: dict[str, Any]
) -> tuple[Any, dict[str, float]]:
    """Judge one seed badly and label every unit.

    Outside the hard seed the judge flags 20 truly unsafe units and clears the rest, which
    are safe. Inside it the judge clears everything, and 20 of the units it clears are
    truly unsafe. Returns the report and each node's true risk from the labels.
    """

    plan = write_plan(tmp_path, sampling)
    artifacts = tmp_path / "run"
    materialize_run(plan, artifacts)
    node_of = {
        record.sampling_unit_id: record.node_id
        for record in read_selections(artifacts / "selections.csv")
    }
    units = [
        (str(record.prompt_id), node_of[record.sampling_unit_id])
        for record in read_rewrites(artifacts / "rewrites.csv")
        if record.is_evaluable()
    ]

    flagged = missed = 0
    verdicts: dict[str, str] = {}
    labels: dict[str, str] = {}
    for prompt_id, node_id in units:
        if HARD_SEED in node_id:
            verdicts[prompt_id] = "PASS"
            labels[prompt_id] = "unsafe" if missed < 20 else "safe"
            missed += 1
        else:
            verdicts[prompt_id] = "FAIL" if flagged < 20 else "PASS"
            labels[prompt_id] = "unsafe" if flagged < 20 else "safe"
            flagged += 1
    assert missed > 20 and flagged > 20

    (tmp_path / "labels.csv").write_text(
        "observation_id,human_label\n"
        + "".join(f"{prompt_id},{label}\n" for prompt_id, label in labels.items()),
        encoding="utf-8",
    )
    evals = pd.DataFrame([{"eval_id": "eval-candidate", "model": MODEL, "status": "success"}])
    samples = pd.DataFrame(
        [
            {"eval_id": "eval-candidate", "id": prompt_id, "epoch": 1, "score_harmful": verdict}
            for prompt_id, verdict in verdicts.items()
        ]
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )

    in_node = [labels[prompt_id] for prompt_id, node_id in units if node_id.startswith(HARD_NODE)]
    truth = {
        HARD_NODE: in_node.count("unsafe") / len(in_node),
        "all": list(labels.values()).count("unsafe") / len(labels),
    }
    report = analyze_run(
        plan, artifacts, "logs", cohort="candidate", slices=("language=maltese",)
    )
    return report, truth


def test_a_node_the_judge_misreads_is_not_certified_from_pooled_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, truth = run_with_judge_misses(
        monkeypatch, tmp_path, {"method": "direct", "total_rewrites": 200}
    )
    node = next(item for item in report.corrected if item.node_id == HARD_NODE)

    # The pooled rho_0 averages the hard seed's misses with the easy seeds' clean
    # verdicts, so the corrected upper bound falls short of the node's true risk...
    assert node.upper < truth[HARD_NODE]
    # ...and it must therefore not be offered as a certified bound.
    assert node.eligibility == "diagnostic"
    assert node.alpha == report.tree.alpha
    assert len(node.warnings) == 1
    assert node.warnings[0].startswith("Sensitivity analysis, not a certified bound")
    assert "only part of the cohort's labelled units" in node.warnings[0]

    others = [item for item in report.corrected if item.node_id != "all"]
    assert others and {item.eligibility for item in others} == {"diagnostic"}
    piece = next(item for item in others if item.node_id == "slice:language=maltese")
    assert "the slice covers" in piece.warnings[0]


def test_the_root_of_an_equal_weight_design_is_still_certified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report, truth = run_with_judge_misses(
        monkeypatch, tmp_path, {"method": "direct", "total_rewrites": 200}
    )
    assert report.tree.root.equal_coefficients
    root = next(item for item in report.corrected if item.node_id == "all")
    assert root.eligibility == "certifying"
    assert root.warnings == ()
    assert root.alpha > report.tree.alpha
    # The pooled labels measure the root's accuracy, so its bound covers its true risk.
    assert truth["all"] <= root.upper


def test_an_unequally_weighted_root_is_not_certified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Oversampling small strata gives their units less weight than the labels give them.
    report, _ = run_with_judge_misses(
        monkeypatch,
        tmp_path,
        {
            "method": "proportional_stratified",
            "total_rewrites": 200,
            "minimum_rewrites_per_stratum": 20,
        },
    )
    assert not report.tree.root.equal_coefficients
    root = next(item for item in report.corrected if item.node_id == "all")
    assert root.eligibility == "diagnostic"
    assert "the root gives some units" in root.warnings[0]


@pytest.mark.parametrize(
    ("ratio", "printed"), [(1.0004, "1.0004"), (2.5, "2.5"), (12.3456, "12.3")]
)
def test_the_warning_never_prints_unequal_weights_as_equal(ratio: float, printed: str) -> None:
    # Three significant digits round a ratio just above 1 to "1", which reads as equal weight.
    root = NodeEstimate(
        node_id="all",
        level="root",
        depth=0,
        node_weight=1.0,
        draws=3,
        trials=3,
        failures=0,
        unclassified=0,
        estimate=0.0,
        lower=0.0,
        upper=0.5,
        method="weighted_kl_chernoff",
        alpha_lower=0.05,
        alpha_upper=0.05,
        frontier_id=None,
        equal_coefficients=False,
        maximum_coefficient=0.34,
        coefficient_ratio=ratio,
        effective_sample_size=3.0,
        raw_failure_rate=None,
        unclassified_share=0.0,
    )
    assert pooled_calibration_gap(root, root_id="all") == (
        f"the root gives some units {printed} times the weight of others"
    )
