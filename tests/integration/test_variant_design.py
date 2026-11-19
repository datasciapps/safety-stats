"""The prompt-variant level of fixed designs, on the template in examples/fixed_design.

Each question_id is a base prompt and each prompt_id beneath it one wording of it. The
template has unequal variant counts (three, one and two) so that a weighting which counted
variants instead of base prompts would show.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.ingest.manifest import load_manifest, resolve_groups
from inspect_safety_stats.schema.reports import RiskIntervalResult, SafetyReport

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "fixed_design"
PLAN = EXAMPLE / "variant-plan.yaml"
EPOCHS = 20
PROFILES = ("rollout", "prompt", "question_repeat", "question")
METHODS = ("weighted_empirical_bernstein", "weighted_hoeffding")

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
EXPECTED_WEIGHTS = {
    "all": EXPECTED_ALL_WEIGHTS,
    "cyber": {
        "cyber-credentials-original": Fraction(1, 6),
        "cyber-credentials-paraphrase": Fraction(1, 6),
        "cyber-credentials-roleplay": Fraction(1, 6),
        "cyber-auth-bypass-original": Fraction(1, 2),
    },
}


def _effective_size(weights: list[Fraction]) -> float:
    return float(1 / sum(weight * weight for weight in weights))


def _manifest() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE / "variant-design.csv")


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


def _analyze(monkeypatch: pytest.MonkeyPatch, *, coupled: bool) -> SafetyReport:
    """The template on synthetic logs in which each credential-theft wording fails 5 of 20 times.

    ``coupled``: every wording fails on epochs 1-5, so the wordings share each epoch's outcome,
    as they would through a seed, cache, sandbox or batch shared by the same epoch.
    Otherwise the i-th wording fails on epochs 5i+1 to 5i+5: the same failures per wording,
    on different epochs.
    """
    manifest = _manifest()
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
    failing = manifest.loc[manifest["question_id"] == "cyber-credentials", "prompt_id"].tolist()
    rows = []
    for prompt in manifest["prompt_id"]:
        first = 1 if coupled or prompt not in failing else 1 + 5 * failing.index(prompt)
        for epoch in range(1, EPOCHS + 1):
            failure = prompt in failing and first <= epoch < first + 5
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


Target = tuple[str, tuple[tuple[str, str], ...]]


def _targets(report: SafetyReport) -> dict[Target, list[RiskIntervalResult]]:
    """The direct intervals of every group mean and hierarchy node, by group and node path."""
    targets = {
        (risk.group_id, ()): risk.intervals
        for risk in report.group_risks
        if risk.measure_id == "mean"
    }
    for node in report.hierarchy_risks:
        targets[(node.group_id, tuple(node.path.items()))] = node.intervals
    return targets


def _target_weights(target: Target) -> dict[str, Fraction]:
    """The weights of a target's prompts: its group's, renormalized within a node."""
    group, path = target
    manifest = _manifest().set_index("prompt_id")
    inside = {
        prompt: weight
        for prompt, weight in EXPECTED_WEIGHTS[group].items()
        if all(manifest.at[prompt, level] == value for level, value in path)
    }
    total = sum(inside.values())
    return {prompt: weight / total for prompt, weight in inside.items()}


def _expected_blocks(weights: Mapping[str, Fraction]) -> dict[str, tuple[int, float]]:
    """Each profile's block count and effective sample size over prompts with these weights.

    The effective size is the reciprocal of the sum of squared block weights
    (docs/independence-profiles.md). A rollout carries its prompt's weight over EPOCHS, and a
    base prompt's epoch the base prompt's weight over EPOCHS.
    """
    question_of = _manifest().set_index("prompt_id")["question_id"]
    by_question: dict[str, Fraction] = {}
    for prompt, weight in weights.items():
        question = question_of[prompt]
        by_question[question] = by_question.get(question, Fraction(0)) + weight
    prompts, questions = list(weights.values()), list(by_question.values())
    return {
        "rollout": (EPOCHS * len(prompts), EPOCHS * _effective_size(prompts)),
        "prompt": (len(prompts), _effective_size(prompts)),
        "question_repeat": (EPOCHS * len(questions), EPOCHS * _effective_size(questions)),
        "question": (len(questions), _effective_size(questions)),
    }


def test_each_profile_blocks_every_group_and_node_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _analyze(monkeypatch, coupled=True)
    assert report.integrity.status == "valid"

    targets = _targets(report)
    manifest = _manifest()
    nodes = {
        (group, (("task_id", task),))
        for group, weights in EXPECTED_WEIGHTS.items()
        for task in manifest.loc[manifest["prompt_id"].isin(list(weights)), "task_id"]
    } | {
        (group, (("task_id", task), ("question_id", question)))
        for group, weights in EXPECTED_WEIGHTS.items()
        for task, question in manifest.loc[
            manifest["prompt_id"].isin(list(weights)), ["task_id", "question_id"]
        ].itertuples(index=False)
    }
    # Two group roots, and a node for each task and each base prompt of each group.
    assert set(targets) == {(group, ()) for group in EXPECTED_WEIGHTS} | nodes

    for target, intervals in targets.items():
        expected = _expected_blocks(_target_weights(target))
        seen = set()
        for interval in intervals:
            if interval.independence_profile is None:
                continue  # the propagated envelope and the judge correction have no blocks
            blocks, size = expected[interval.independence_profile]
            assert interval.independent_block_count == blocks, (target, interval.interval_id)
            assert interval.effective_sample_size == pytest.approx(size, rel=1e-9, abs=0)
            seen.add((interval.method, interval.independence_profile))
        assert seen == {(method, profile) for method in METHODS for profile in PROFILES}

    claim = next(claim for claim in report.claims if claim.claim_id == "mean_safe")
    assert [
        (row.independence_profile, row.role, row.independent_block_count)
        for row in claim.profile_verdicts
    ] == [
        ("rollout", "primary", 120),
        ("prompt", "sensitivity", 6),
        ("question_repeat", "sensitivity", 60),
        ("question", "sensitivity", 3),
    ]
    assert claim.status == claim.profile_verdicts[0].status


def test_per_epoch_blocks_keep_a_bound_where_per_base_prompt_blocks_reach_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With two or three base prompts, or one in a base-prompt node, `question` is vacuous.

    Hoeffding's width depends on the data only through the estimate, which every profile
    shares, so its uppers order as the profiles' effective sizes do.
    """
    report = _analyze(monkeypatch, coupled=True)
    for target, intervals in _targets(report).items():
        uppers = {
            interval.independence_profile: interval.upper
            for interval in intervals
            if interval.method == "weighted_hoeffding"
        }
        assert uppers["question"] == pytest.approx(1.0, rel=0, abs=1e-12), target
        question_repeat = uppers["question_repeat"]
        assert question_repeat is not None and question_repeat < 0.75, target
        if target[1] == ():
            # The roots' base prompts have several wordings, so one block per base prompt
            # and epoch holds fewer independent units than one block per rollout.
            rollout = uppers["rollout"]
            assert rollout is not None and rollout < question_repeat, target


def test_per_epoch_coupling_shows_only_in_the_design_effect_of_question_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wordings sharing each epoch's outcome change no bound of the template's profiles.

    Only a profile that declares the coupling accounts for it, which is why a plan whose
    wordings can be coupled needs such a profile as its primary. The coupling is visible in
    the CLUSTER_DESIGN_EFFECT of `question_repeat`, and not in that of `question`, whose
    blocks hold the same rollouts either way.
    """
    coupled = _analyze(monkeypatch, coupled=True)
    staggered = _analyze(monkeypatch, coupled=False)

    def effects(report: SafetyReport) -> dict[str, dict[str, Any]]:
        return {
            str(item.details["profile"]): {"severity": item.severity, **item.details}
            for item in report.integrity.diagnostics
            if item.code == "CLUSTER_DESIGN_EFFECT"
        }

    shared, apart = effects(coupled), effects(staggered)
    assert set(shared) == set(apart) == {"question_repeat", "question"}
    assert shared["question"] == apart["question"]

    # Blocks of three, one and two wordings, twenty of each: the mean block size an outcome
    # sits in is 20 * (9 + 1 + 4) / 120 = 7/3, and every block's outcomes agree.
    assert shared["question_repeat"]["severity"] == "WARNING"
    assert shared["question_repeat"]["primaries_splitting_blocks"] == ["rollout"]
    (cohort,) = shared["question_repeat"]["cohorts"]
    assert cohort["blocks"] == 60
    assert cohort["intracluster_correlation"] == pytest.approx(1.0, rel=0, abs=1e-12)
    assert cohort["design_effect"] == pytest.approx(7 / 3, rel=1e-12, abs=0)
    assert apart["question_repeat"]["severity"] == "INFO"
    (cohort,) = apart["question_repeat"]["cohorts"]
    assert cohort["intracluster_correlation"] < 0
    assert cohort["design_effect"] == 1.0

    shared_intervals, apart_intervals = _targets(coupled), _targets(staggered)
    for target, intervals in shared_intervals.items():
        by_key = {
            (interval.method, interval.independence_profile): interval
            for interval in apart_intervals[target]
        }
        for interval in intervals:
            if interval.independence_profile in (None, "question_repeat"):
                continue
            other = by_key[(interval.method, interval.independence_profile)]
            assert (interval.lower, interval.upper) == pytest.approx(
                (other.lower, other.upper), rel=1e-12, abs=1e-12
            ), (target, interval.interval_id)

