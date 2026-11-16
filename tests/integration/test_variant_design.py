"""The prompt-variant level of fixed designs, on the template in examples/fixed_design.

Each question_id is a base prompt and each prompt_id beneath it one wording of it. The
template has unequal variant counts (three, one and two) so that a weighting which counted
variants instead of base prompts would show.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_groups
from inspect_safety_stats.schema.reports import SafetyReport

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "fixed_design"
PLAN = EXAMPLE / "variant-plan.yaml"
EPOCHS = 20

# nested_equal over [task_id, question_id, prompt_id]: each task half, each base prompt an
# equal share of its task, each variant an equal share of its base prompt.
EXPECTED_ALL_WEIGHTS = {
    "cyber-credentials-original": Fraction(1, 12),
    "cyber-credentials-paraphrase": Fraction(1, 12),
    "cyber-credentials-roleplay": Fraction(1, 12),
    "cyber-auth-bypass-original": Fraction(1, 4),
    "bio-pathogen-original": Fraction(1, 4),
    "bio-pathogen-paraphrase": Fraction(1, 4),
}


def _effective_size(weights: list[Fraction]) -> float:
    return float(1 / sum(weight * weight for weight in weights))


def test_variant_weights_split_each_base_prompt_not_each_variant() -> None:
    plan = load_plan(PLAN)
    manifest = load_manifest(plan.inputs.design_manifest)
    groups = {group.id: group for group in resolve_groups(plan, manifest)}

    weights = groups["all"].weights
    assert set(weights) == set(EXPECTED_ALL_WEIGHTS)
    for prompt, expected in EXPECTED_ALL_WEIGHTS.items():
        assert weights[prompt] == pytest.approx(float(expected), rel=1e-12, abs=0)

    by_question = manifest.set_index("prompt_id")["question_id"]
    question_weight: dict[str, float] = {}
    for prompt, weight in weights.items():
        question = by_question[prompt]
        question_weight[question] = question_weight.get(question, 0.0) + weight
    # Three, one and two variants, yet both cyber base prompts weigh the same.
    assert question_weight == pytest.approx(
        {"cyber-credentials": 0.25, "cyber-auth-bypass": 0.25, "bio-pathogen": 0.5},
        rel=1e-12,
        abs=0,
    )

    cyber = groups["cyber"].weights
    assert cyber["cyber-auth-bypass-original"] == pytest.approx(0.5, rel=1e-12, abs=0)
    for variant in ("original", "paraphrase", "roleplay"):
        assert cyber[f"cyber-credentials-{variant}"] == pytest.approx(1 / 6, rel=1e-12, abs=0)


def _analyze(monkeypatch: pytest.MonkeyPatch, failing_question: str) -> SafetyReport:
    manifest = pd.read_csv(EXAMPLE / "variant-design.csv")
    eval_frame = pd.DataFrame(
        [
            {
                "eval_id": "e",
                "run_id": "r",
                "log": "/logs/candidate.eval",
                "created": "2026-09-01T00:00:00Z",
                "status": "success",
                "task_name": "variant_safety",
                "task_version": "1",
                "model": "openai/gpt-4o-mini",
                "model_generate_config": {},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "variants",
                "epochs": EPOCHS,
            }
        ]
    )
    rows = []
    for prompt, question in zip(manifest["prompt_id"], manifest["question_id"], strict=True):
        for epoch in range(1, EPOCHS + 1):
            # Every wording of one base prompt fails on the same epochs: the dependence
            # between variants that the question profile exists for.
            failure = question == failing_question and epoch <= 5
            rows.append(
                {
                    "sample_id": f"{prompt}-{epoch}",
                    "eval_id": "e",
                    "log": "/logs/candidate.eval",
                    "id": prompt,
                    "epoch": epoch,
                    "score_safety_failure": "FAIL" if failure else "PASS",
                    "error": None,
                    "retries": 0,
                }
            )
    sample_frame = pd.DataFrame(rows)
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: eval_frame
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, **_: sample_frame,
    )
    return analyze(plan=PLAN, logs="ignored")


def test_question_profile_blocks_all_variants_of_a_base_prompt_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _analyze(monkeypatch, failing_question="cyber-credentials")
    assert report.integrity.status == "valid"

    weights = {"all": list(EXPECTED_ALL_WEIGHTS.values())}
    weights["cyber"] = [Fraction(1, 6)] * 3 + [Fraction(1, 2)]
    question_weights = {
        "all": [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)],
        "cyber": [Fraction(1, 2), Fraction(1, 2)],
    }
    expected = {
        group: {
            "rollout": (EPOCHS * len(weights[group]), EPOCHS * _effective_size(weights[group])),
            "prompt": (len(weights[group]), _effective_size(weights[group])),
            "question": (len(question_weights[group]), _effective_size(question_weights[group])),
        }
        for group in weights
    }

    seen: set[tuple[str, str]] = set()
    for group_risk in report.group_risks:
        if group_risk.measure_id != "mean":
            continue
        uppers: dict[str, float] = {}
        for interval in group_risk.intervals:
            if interval.method != "weighted_hoeffding":
                continue
            profile = interval.independence_profile
            assert profile is not None
            blocks, size = expected[group_risk.group_id][profile]
            assert interval.independent_block_count == blocks
            assert interval.effective_sample_size == pytest.approx(size, rel=1e-9, abs=0)
            assert interval.upper is not None
            uppers[profile] = interval.upper
            seen.add((group_risk.group_id, profile))
        # Merging blocks can only grow the sum of squared block weights, so coarser blocks
        # never narrow a Hoeffding bound (both may reach 1 with so few base prompts).
        assert uppers["rollout"] <= uppers["prompt"] <= uppers["question"]
    assert seen == {(group, profile) for group in weights for profile in expected[group]}
