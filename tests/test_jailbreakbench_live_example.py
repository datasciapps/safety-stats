from __future__ import annotations

import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "real_jailbreakbench"
sys.path.insert(0, str(EXAMPLE_DIR))

import run_jailbreakbench as runner  # noqa: E402
from jailbreakbench_live import (  # noqa: E402
    DATA_SHA256,
    behavior_rows,
    harmful_compliance_failure,
    jailbreakbench_100,
    samples,
)
from run_jailbreakbench import write_design, write_plan  # noqa: E402


def test_pinned_jbb_dataset_is_complete_and_balanced() -> None:
    data = (EXAMPLE_DIR / "jbb_behaviors_harmful.csv").read_bytes()
    assert sha256(data).hexdigest() == DATA_SHA256

    rows = behavior_rows()
    inspect_samples = samples()
    assert len(rows) == len(inspect_samples) == 100
    assert len({str(sample.id) for sample in inspect_samples}) == 100
    assert set(Counter(row["Category"] for row in rows).values()) == {10}
    assert len({row["Category"] for row in rows}) == 10
    assert Counter(row["Source"] for row in rows) == {
        "Original": 55,
        "TDC/HarmBench": 27,
        "AdvBench": 18,
    }
    assert len(jailbreakbench_100(grader_model="mockllm/grader").dataset) == 100


def test_jbb_score_mapping_and_fixed_template() -> None:
    assert harmful_compliance_failure("C") is True
    assert harmful_compliance_failure("I") is False
    assert harmful_compliance_failure("") is None

    direct = samples()[0]
    transformed = samples("PREFIX\n{goal}\nSUFFIX", attack_name="fixed-test")[0]
    assert str(transformed.input).startswith("PREFIX\n")
    assert str(transformed.input).endswith("\nSUFFIX")
    assert str(direct.input) in str(transformed.input)
    with pytest.raises(ValueError, match="exactly one"):
        samples("missing placeholder")


def test_runner_writes_balanced_multi_arm_plan(tmp_path: Path) -> None:
    write_design(tmp_path / "design.csv", rollouts=8, attack_name="direct_request")
    write_plan(
        tmp_path / "analysis-plan.yaml",
        arms={
            "baseline": "model/baseline",
            "candidate": "model/candidate",
            "variant": "model/variant",
        },
        reference="baseline",
        run_id="jbb-test",
        frozen_at="2026-08-21T12:00:00+00:00",
        attack_name="direct_request",
    )
    manifest = pd.read_csv(tmp_path / "design.csv")
    plan = load_plan(tmp_path / "analysis-plan.yaml")

    assert len(manifest) == 100
    assert set(manifest["expected_rollouts"]) == {8}
    assert set(manifest.groupby("category_id").size()) == {10}
    assert plan.design.hierarchy == ["category_id", "prompt_id"]
    assert plan.design.weights.hierarchy == ["category_id", "prompt_id"]
    assert plan.cohorts["baseline"].match == {"model": "model/baseline"}
    assert plan.cohorts["candidate"].match == {"model": "model/candidate"}
    assert len(plan.comparison_sets) == 1
    assert plan.comparison_sets[0].contrasts.reference == "baseline"
    expanded = plan.expanded_comparisons()
    assert {item.estimand for item in expanded} == {
        "risk_difference",
        "risk_ratio",
        "relative_risk_reduction",
    }
    assert all(item.method == "weighted_kl_chernoff" for item in expanded)
    assert all(item.hierarchy_nodes for item in expanded)
    assert all(item.hierarchy_calibration == "frontier_sidak" for item in expanded)
    assert all(item.hierarchy_region == "linear_program" for item in expanded)
    assert plan.report.include_prompt_text is False
    assert plan.report.include_decisions is False


def test_analysis_selects_newest_successful_log_per_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_design(tmp_path / "design.csv", rollouts=2, attack_name="direct_request")
    write_plan(
        tmp_path / "analysis-plan.yaml",
        arms={
            "baseline": "model/baseline",
            "candidate": "model/candidate",
            "variant": "model/variant",
        },
        reference="baseline",
        run_id="jbb-log-selection",
        frozen_at="2026-08-21T12:00:00+00:00",
        attack_name="direct_request",
    )
    infos = [
        SimpleNamespace(name="baseline-old.eval", mtime=1.0),
        SimpleNamespace(name="baseline-new.eval", mtime=3.0),
        SimpleNamespace(name="candidate-failed.eval", mtime=4.0),
        SimpleNamespace(name="candidate-success.eval", mtime=2.0),
        SimpleNamespace(name="variant-success.eval", mtime=5.0),
    ]
    headers = {
        "baseline-old.eval": SimpleNamespace(
            status="success", eval=SimpleNamespace(model="model/baseline")
        ),
        "baseline-new.eval": SimpleNamespace(
            status="success", eval=SimpleNamespace(model="model/baseline")
        ),
        "candidate-failed.eval": SimpleNamespace(
            status="error", eval=SimpleNamespace(model="model/candidate")
        ),
        "candidate-success.eval": SimpleNamespace(
            status="success", eval=SimpleNamespace(model="model/candidate")
        ),
        "variant-success.eval": SimpleNamespace(
            status="success", eval=SimpleNamespace(model="model/variant")
        ),
    }
    monkeypatch.setattr(runner, "list_eval_logs", lambda *args, **kwargs: infos)
    monkeypatch.setattr(
        runner,
        "read_eval_log",
        lambda info, header_only=True: headers[info.name],
    )

    assert runner.latest_successful_logs(tmp_path) == [
        "baseline-new.eval",
        "candidate-success.eval",
        "variant-success.eval",
    ]


def _offline_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations = []
    sample_rows = []
    for arm, model in (
        ("baseline", "model/baseline"),
        ("candidate", "model/candidate"),
        ("variant", "model/variant"),
    ):
        eval_id = f"eval-{arm}"
        log_path = f"/not/a/local/{arm}.eval"
        evaluations.append(
            {
                "eval_id": eval_id,
                "run_id": f"run-{arm}",
                "log": log_path,
                "created": "2026-08-21T12:00:00Z",
                "status": "success",
                "task_name": "jailbreakbench_100",
                "task_version": "1.0",
                "model": model,
                "model_generate_config": {"temperature": 1.0, "max_tokens": 512},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "jbb-harmful",
                "epochs": 2,
            }
        )
        for index in range(100):
            prompt_id = f"jbb-{index:03d}"
            divisor = {"candidate": 10, "baseline": 4, "variant": 2}[arm]
            failure = index % divisor == 0
            for epoch in (1, 2):
                sample_rows.append(
                    {
                        "sample_id": f"{arm}-{prompt_id}-{epoch}",
                        "eval_id": eval_id,
                        "log": log_path,
                        "id": prompt_id,
                        "epoch": epoch,
                        "score_jbb_harmful_compliance": "FAIL" if failure else "PASS",
                        "error": None,
                        "retries": 0,
                    }
                )
    return pd.DataFrame(evaluations), pd.DataFrame(sample_rows)


def test_complete_offline_analysis_and_visualizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_design(tmp_path / "design.csv", rollouts=2, attack_name="direct_request")
    write_plan(
        tmp_path / "analysis-plan.yaml",
        arms={
            "baseline": "model/baseline",
            "candidate": "model/candidate",
            "variant": "model/variant",
        },
        reference="baseline",
        run_id="jbb-offline",
        frozen_at="2026-08-21T12:00:00+00:00",
        attack_name="direct_request",
    )
    eval_frame, sample_frame = _offline_frames()
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df",
        lambda logs: eval_frame,
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: sample_frame,
    )

    report = analyze(plan=tmp_path / "analysis-plan.yaml", logs="ignored")
    assert report.integrity.status == "valid"
    assert len(report.prompt_estimates) == 300
    assert len(report.group_risks) == 12
    assert len(report.hierarchy_risks) == 30
    assert len(report.comparisons) == 66
    assert len(report.comparison_sets) == 1

    difference = next(
        item
        for item in report.comparisons
        if item.comparison_id == "models_vs_reference::candidate_vs_baseline::risk_difference"
        and item.level is None
    )
    assert difference.estimate is not None and difference.estimate < 0
    assert difference.comparison_set_id == "models_vs_reference"
    assert difference.pair_count is None
    assert {interval.independence_profile for interval in difference.intervals} == {
        "rollout",
        "prompt",
        "category",
    }

    output = report.write(tmp_path / "report")
    assert (output / "report.html").exists()
    assert (output / "comparisons.csv").exists()
    assert (output / "hierarchy-risks.csv").exists()
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "models_vs_reference" in html
    assert "Configuration risk and comparison hierarchy" in html
    assert "Within configuration ·" in html
    assert "Across configurations ·" in html
    assert "Independence profile" in html
    assert "independence_profile" in html

    first_goal = behavior_rows()[0]["Goal"]
    for name in ("report.html", "report.md", "report.json"):
        assert first_goal not in (output / name).read_text(encoding="utf-8")
