import sys
from pathlib import Path

import yaml

EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "real_xstest"
sys.path.insert(0, str(EXAMPLE_DIR))

from run_openai import _write_design, _write_plan  # noqa: E402
from xstest_live import _samples, strict_policy_passes, xstest_openai  # noqa: E402


def test_live_xstest_uses_eight_unique_fixed_prompts() -> None:
    samples = _samples()
    assert len(samples) == 8
    assert len({str(sample.id) for sample in samples}) == 8
    assert len(xstest_openai(grader_model="mockllm/grader").dataset) == 8


def test_strict_refusal_policy_mapping() -> None:
    assert strict_policy_passes("comply", "C") is True
    assert strict_policy_passes("comply", "P") is False
    assert strict_policy_passes("comply", "I") is False
    assert strict_policy_passes("refuse", "I") is True
    assert strict_policy_passes("refuse", "P") is False
    assert strict_policy_passes("refuse", "C") is False
    assert strict_policy_passes("refuse", "X") is None


def test_runner_freezes_model_and_expected_rollouts(tmp_path: Path) -> None:
    design_path = tmp_path / "design.csv"
    plan_path = tmp_path / "safety-plan.yaml"
    _write_design(design_path, rollouts=3)
    _write_plan(
        plan_path,
        model="openai/test-model",
        run_id="test-run",
        frozen_at="2026-08-18T12:00:00+00:00",
    )

    assert all(
        line.endswith(",3,false,true") or line.endswith(",3,true,true")
        for line in design_path.read_text(encoding="utf-8").splitlines()[1:]
    )
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    assert plan["analysis"]["id"] == "test-run"
    assert plan["analysis"]["plan_frozen_at"] == "2026-08-18T12:00:00+00:00"
    assert plan["cohorts"]["candidate"]["match"]["model"] == "openai/test-model"
