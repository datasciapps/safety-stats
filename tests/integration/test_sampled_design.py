"""End-to-end coverage for `design.type: sampled`.

The manifest is the sampling *frame*: most of its items are never evaluated, and the
estimand remains the frame-wide weighted mean rather than the realized sample mean.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze
from inspect_safety_stats.ingest.pilot import draw_pilot
from inspect_safety_stats.report import visualization
from tests.template_functions import function_source

FRAME_SIZE = 200
INCLUSION = 0.3
ROLLOUTS = 20

_PLAN = """schema_version: "2.0"
analysis:
  id: sampled-frame
  title: Sampled frame
  intent: confirmatory
  plan_frozen_at: 2026-01-01T00:00:00Z
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {{failure_values: ["FAIL"], safe_values: ["PASS"]}}
design:
  type: sampled
{sampling}  inclusion_probability_column: inclusion_probability
  hierarchy: [task_id, question_id, prompt_id]
  weights: {{strategy: equal_prompt}}
  independence_profiles:
    rollout: {{block_by: [rollout], comparison_coupling: independent}}
cohorts:
  candidate:
    match: {{model: model/candidate}}
groups:
  - id: all
    filter: {{}}
    weights: {{strategy: equal_prompt}}
outputs:
  prompt_bounds: {{enabled: false, simultaneous: false, family: displayed}}
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    reference_methods: [weighted_hoeffding]
    family: direct_means
    simultaneous: true
    independence: {{primary: rollout}}
    hierarchy_nodes: true
  risk_measures:
{measures}
multiplicity:
  families:
    - {{id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
    - {{id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}}
claims: []
report: {{include_observations: false}}
"""


def _write_frame(
    tmp_path: Path,
    *,
    measures: str = "    - {id: mean, type: weighted_mean}",
    fixed_size: bool = False,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(4242)
    strata = ["alpha", "beta"]
    frame, drawn = [], []
    for index in range(FRAME_SIZE):
        prompt_id = f"item-{index:03d}"
        frame.append(
            {
                "prompt_id": prompt_id,
                "question_id": f"q-{index // 4:03d}",
                "task_id": f"task-{index % 5}",
                "sampling_stratum": strata[index % 2],
                "inclusion_probability": INCLUSION,
                "expected_rollouts": ROLLOUTS,
            }
        )
    if fixed_size:
        # Draw exactly INCLUSION * stratum size from each stratum, without replacement.
        for stratum in strata:
            members = [row["prompt_id"] for row in frame if row["sampling_stratum"] == stratum]
            for prompt_id in rng.sample(members, round(INCLUSION * len(members))):
                drawn.append((prompt_id, rng.betavariate(0.5, 12.0)))
    else:
        for row in frame:
            if rng.random() < INCLUSION:
                drawn.append((row["prompt_id"], rng.betavariate(0.5, 12.0)))
    pd.DataFrame(frame).to_csv(tmp_path / "design.csv", index=False)
    sampling = (
        "  sampling: fixed_size\n  stratum_column: sampling_stratum\n" if fixed_size else ""
    )
    (tmp_path / "plan.yaml").write_text(
        _PLAN.format(measures=measures, sampling=sampling), encoding="utf-8"
    )

    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/x.eval",
                "created": "2026-06-01T00:00:00Z",
                "status": "success",
                "task_name": "sampled",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "frame",
                "epochs": ROLLOUTS,
            }
        ]
    )
    rows = [
        {
            "sample_id": f"{prompt_id}-{epoch}",
            "eval_id": "eval-1",
            "log": "/x.eval",
            "id": prompt_id,
            "epoch": epoch,
            "score_harmful": "FAIL" if rng.random() < risk else "PASS",
            "error": None,
            "retries": 0,
        }
        for prompt_id, risk in drawn
        for epoch in range(1, ROLLOUTS + 1)
    ]
    return tmp_path / "plan.yaml", evals, pd.DataFrame(rows)


def _patch(monkeypatch: pytest.MonkeyPatch, evals: pd.DataFrame, samples: pd.DataFrame) -> None:
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def test_frame_mean_is_estimated_by_horvitz_thompson(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path)
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    assert report.integrity.status == "valid"
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    primary = next(item for item in mean.intervals if item.role == "primary")
    assert primary.interval_id == "horvitz_thompson_weighted_kl_chernoff"
    assert primary.eligibility == "certifying"
    assert primary.independence_profile == "frame_item"

    # The estimand is the frame, so the block count is the realized sample, not the frame.
    sampled = sum(1 for item in report.prompt_estimates if item.n_total > 0)
    assert len(report.prompt_estimates) == FRAME_SIZE
    assert 0 < sampled < FRAME_SIZE
    assert primary.independent_block_count == sampled

    # The inverse-probability cost is stated rather than buried.
    assert any("Poisson sampling" in text for text in primary.assumptions)
    assert any("widens every endpoint" in text for text in primary.assumptions)
    assert mean.lower is not None and mean.upper is not None and mean.lower < mean.upper


def test_the_report_scope_names_the_frame_rather_than_a_fixed_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path)
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    # The intervals are about the whole frame, so the scope statement must not say the
    # results apply to a fixed evaluation set.
    assert report.scope["design"] == "sampled"
    assert "fixed evaluation set" not in report.scope["disclaimer"]
    assert "declared item frame" in report.scope["disclaimer"]
    assert "inclusion probabilities" in report.scope["randomness"]
    assert "not strategically underperforming" in report.scope["disclaimer"]
    assert report.scope["what_results_establish"].endswith("docs/what-results-establish.md")


def test_the_report_text_speaks_of_the_frame_rather_than_a_fixed_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scope statement said the frame while the report around it still said fixed prompts."""
    plan_path, evals, samples = _write_frame(tmp_path)
    _patch(monkeypatch, evals, samples)

    output = analyze(plan=plan_path, logs="ignored").write(tmp_path / "report")

    summary = (output / "report.md").read_text(encoding="utf-8")
    assert f"\n{FRAME_SIZE} frame items · " in summary
    assert (
        "Intervals bound the risk over the declared item frame with 95.0% simultaneous "
        "coverage, allowing for both the draw of items from the frame and the rollouts of the "
        "drawn items. Each drawn item enters as one observation, so the rollouts of one item "
        "may depend on each other arbitrarily." in summary
    )
    assert "prompt-level sensitivity allows" not in summary
    full = (output / "report-full.md").read_text(encoding="utf-8")
    assert "\n## Sampled-design risk measures\n" in full
    assert "Node estimates use the declared design weights over the frame, normalized" in full
    assert "fixed prompts" not in summary
    assert "Fixed-design risk measures" not in full
    assert "fixed-design weights" not in full


_DESIGN_TEXT_DRIVER = """
const fs = require("fs");
const [functionsPath, reportPath] = process.argv.slice(2);
const api = new Function(fs.readFileSync(functionsPath, "utf8") + "\\nreturn {designText};")();
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
process.stdout.write(JSON.stringify(api.designText(report)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("design", ["sampled", "fixed"])
def test_the_html_report_names_the_design_its_values_are_about(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, design: str
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path)
    if design == "fixed":
        # The same inputs analysed as a fixed design (it reports the undrawn items missing).
        text = plan_path.read_text(encoding="utf-8").replace("  type: sampled\n", "  type: fixed\n")
        plan_path.write_text(
            text.replace("  inclusion_probability_column: inclusion_probability\n", ""),
            encoding="utf-8",
        )
    _patch(monkeypatch, evals, samples)
    report = analyze(plan=plan_path, logs="ignored")
    template = visualization._TEMPLATE.read_text(encoding="utf-8")
    (tmp_path / "functions.js").write_text(
        function_source(template, "designText"), encoding="utf-8"
    )
    (tmp_path / "report.json").write_text(
        json.dumps(report.model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "driver.js").write_text(_DESIGN_TEXT_DRIVER, encoding="utf-8")
    node = shutil.which("node")
    assert node is not None

    completed = subprocess.run(
        [
            node,
            str(tmp_path / "driver.js"),
            str(tmp_path / "functions.js"),
            str(tmp_path / "report.json"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    text = json.loads(completed.stdout)
    if design == "sampled":
        assert text == {
            "riskHeading": "Sampled-design risk measures",
            "prompts": "Frame items",
            "footer": "Values are design-based estimates for the declared item frame; "
            "hierarchy filters do not create claims beyond the frame.",
        }
    else:
        # A fixed design keeps its wording.
        assert text == {
            "riskHeading": "Fixed-design risk measures",
            "prompts": "Fixed prompts",
            "footer": "Values are fixed-design summaries; hierarchy filters do not create "
            "population-level claims.",
        }
        # The static heading, shown before the script runs, is the fixed design's.
        assert '<h2 id="risk-heading">Fixed-design risk measures</h2>' in template


def test_unsampled_hierarchy_nodes_warn_rather_than_invalidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path)
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code: item for item in report.integrity.diagnostics}
    assert "HIERARCHY_NODE_UNSAMPLED" in codes
    assert codes["HIERARCHY_NODE_UNSAMPLED"].severity == "WARNING"
    # A node with nothing drawn is unidentified, not broken.
    unestimated = [row for row in report.hierarchy_risks if row.lower is None]
    assert unestimated
    for row in unestimated:
        assert any("not identified" in text for text in row.warnings)
    assert any(row.lower is not None for row in report.hierarchy_risks)


def test_sampled_design_rejects_functionals_it_cannot_evaluate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the weighted mean is linear in the frame's risk vector."""
    plan_path, evals, samples = _write_frame(
        tmp_path,
        measures=(
            "    - {id: mean, type: weighted_mean}\n"
            "    - {id: worst, type: maximum}\n"
            "    - {id: tail, type: cvar, beta: 0.9}"
        ),
    )
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code for item in report.integrity.diagnostics}
    assert "PLAN_SAMPLED_MEASURE_UNSUPPORTED" in codes
    assert report.integrity.status == "invalid"


def test_a_truncated_run_is_caught_rather_than_silently_biasing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fewer realized items than the declared probabilities imply means the design is wrong."""
    plan_path, evals, samples = _write_frame(tmp_path)
    keep = sorted(samples["id"].unique())[:5]
    _patch(monkeypatch, evals, samples.loc[samples["id"].isin(keep)])

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code: item for item in report.integrity.diagnostics}
    assert "SAMPLE_SIZE_IMPLAUSIBLE" in codes
    assert codes["SAMPLE_SIZE_IMPLAUSIBLE"].severity == "ERROR"
    assert report.integrity.status == "invalid"


def test_a_transported_judge_calibration_bounds_the_frame_mean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transported mode runs on a sampled design, and bounds the frame mean.

    The frame mean is a weighted mean of every frame prompt's risk, so dividing its upper
    bound by the published sensitivity bounds its true risk when transport holds on every
    frame prompt.
    """
    from scipy.stats import beta

    plan_path, evals, samples = _write_frame(tmp_path)
    text = plan_path.read_text(encoding="utf-8")
    plan_path.write_text(
        text.replace(
            "inputs:\n",
            "inputs:\n"
            "  judge_calibration:\n"
            "    mode: transported\n"
            "    published_set:\n"
            "      name: Published judge validation set\n"
            "      source: https://example.org/judge-validation\n"
            "      judge: judge-model-v1\n"
            "      rubric: harmful-behaviour rubric v2\n"
            "      human_unsafe: 120\n"
            "      judge_flagged_unsafe: 100\n",
            1,
        ),
        encoding="utf-8",
    )
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    assert report.integrity.status == "valid"
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    primary = next(item for item in mean.intervals if item.role == "primary")
    assert primary.interval_id == "horvitz_thompson_weighted_kl_chernoff"
    (interval,) = [
        item for item in mean.intervals if item.interval_id == "judge_corrected_transported"
    ]
    sensitivity_lower = float(beta.ppf(0.05, 100, 120 - 100 + 1))
    assert mean.upper is not None
    assert (interval.lower, interval.eligibility) == (0.0, "diagnostic")
    assert interval.upper == pytest.approx(
        min(1.0, mean.upper / sensitivity_lower), rel=1e-12, abs=0
    )


def test_fixed_designs_are_unaffected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fixed design must still refuse a manifest prompt with no observations."""
    plan_path, evals, samples = _write_frame(tmp_path)
    text = (tmp_path / "plan.yaml").read_text(encoding="utf-8")
    text = text.replace("  type: sampled\n", "  type: fixed\n")
    text = text.replace("  inclusion_probability_column: inclusion_probability\n", "")
    (tmp_path / "plan.yaml").write_text(text, encoding="utf-8")
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code for item in report.integrity.diagnostics}
    assert "PROMPT_MISSING" in codes
    assert report.integrity.status == "invalid"


def test_stratified_fixed_size_draws_are_estimated_without_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path, fixed_size=True)
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    assert report.integrity.status == "valid"
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    primary = next(item for item in mean.intervals if item.role == "primary")
    assert primary.interval_id.startswith("stratified_without_replacement_")
    assert primary.eligibility == "certifying"
    assert any("without replacement" in text for text in primary.assumptions)
    assert any("Hoeffding's reduction" in text for text in primary.assumptions)

    # Equal design weights leave no inverse-probability inflation: the estimator is a
    # plain sample mean, so the drawn items carry the whole weight.
    assert primary.maximum_increment == pytest.approx(1 / primary.independent_block_count)
    assert mean.lower is not None and mean.upper is not None and mean.lower < mean.upper


def test_fixed_size_run_that_misses_its_plan_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fixed-size design must realize its plan exactly, or pi misstates the sample."""
    plan_path, evals, samples = _write_frame(tmp_path, fixed_size=True)
    dropped = sorted(samples["id"].unique())[:3]
    _patch(monkeypatch, evals, samples.loc[~samples["id"].isin(dropped)])

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code: item for item in report.integrity.diagnostics}
    assert "SAMPLE_DRAWS_MISMATCH" in codes
    assert codes["SAMPLE_DRAWS_MISMATCH"].severity == "ERROR"
    assert report.integrity.status == "invalid"


def test_fixed_size_requires_a_constant_probability_within_each_stratum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_path, evals, samples = _write_frame(tmp_path, fixed_size=True)
    frame = pd.read_csv(tmp_path / "design.csv")
    frame.loc[0, "inclusion_probability"] = 0.9
    frame.to_csv(tmp_path / "design.csv", index=False)
    _patch(monkeypatch, evals, samples)

    report = analyze(plan=plan_path, logs="ignored")

    codes = {item.code for item in report.integrity.diagnostics}
    assert "SAMPLE_STRATUM_INCONSISTENT" in codes
    assert report.integrity.status == "invalid"


def test_a_drawn_pilot_frame_is_analysed_as_a_fixed_size_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """draw_pilot's frame is the frame of a fixed-size design, with no columns to edit."""
    plan_path, evals, samples = _write_frame(tmp_path, fixed_size=True)
    frame = pd.read_csv(tmp_path / "design.csv").drop(columns="inclusion_probability")
    pilot = draw_pilot(frame, per_stratum=12, strata=["sampling_stratum"], seed=20261116)
    pilot.to_csv(tmp_path / "design.csv", index=False)
    plan_path.write_text(
        plan_path.read_text(encoding="utf-8")
        .replace("stratum_column: sampling_stratum", "stratum_column: pilot_stratum")
        .replace(
            "inclusion_probability_column: inclusion_probability",
            "inclusion_probability_column: pilot_inclusion_probability",
        ),
        encoding="utf-8",
    )
    template = samples.iloc[0].to_dict()
    drawn = pilot.loc[pilot["pilot_selected"], "prompt_id"].tolist()
    pilot_samples = pd.DataFrame(
        [
            {
                **template,
                "sample_id": f"{prompt_id}-{epoch}",
                "id": prompt_id,
                "epoch": epoch,
                "score_harmful": "FAIL" if epoch == 1 and index % 3 == 0 else "PASS",
            }
            for index, prompt_id in enumerate(drawn)
            for epoch in range(1, ROLLOUTS + 1)
        ]
    )
    _patch(monkeypatch, evals, pilot_samples)

    report = analyze(plan=plan_path, logs="ignored")

    assert report.integrity.status == "valid", report.integrity.diagnostics
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    primary = next(item for item in mean.intervals if item.role == "primary")
    assert primary.interval_id.startswith("stratified_without_replacement_")
    assert primary.independent_block_count == 24
