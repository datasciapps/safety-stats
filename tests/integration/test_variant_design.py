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

import numpy as np
import pandas as pd
import pytest

from safety_stats import analyze, load_plan
from safety_stats.config.models import (
    AnalysisPlan,
    IndependenceProfileSpec,
    RiskMeasureSpec,
)
from safety_stats.ingest.manifest import load_manifest, resolve_groups
from safety_stats.schema.observations import Observation
from safety_stats.schema.reports import RiskIntervalResult, SafetyReport
from safety_stats.stats.direct_mean import DIRECT_MEAN_ENGINES
from safety_stats.stats.independence import block_index, direct_bounded_blocks

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


def _analyze(
    monkeypatch: pytest.MonkeyPatch, *, coupled: bool, plan: AnalysisPlan | None = None
) -> SafetyReport:
    """The template on synthetic logs in which each credential-theft wording fails 5 of 20 times.

    ``coupled``: every wording fails on epochs 1-5, so the wordings share each epoch's outcome,
    as they would through a seed, cache, sandbox or batch shared by the same epoch.
    Otherwise the i-th wording fails on epochs 5i+1 to 5i+5: the same failures per wording,
    on different epochs. ``plan`` replaces the template plan.
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
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: eval_frame
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, **_: sample_frame,
    )
    return analyze(plan=PLAN if plan is None else plan, logs="ignored")


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


def test_hierarchy_nodes_divide_the_alpha_of_the_group_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under flat_bonferroni each node is one more target of the direct-mean family.

    The family's alpha of 0.05 is two-sided, 0.025 for each direction. The template's two
    group roots and eight nodes divide it by ten, the roots alone by two, as
    docs/plan-reference.md says.
    """
    plan = load_plan(PLAN)
    direct = plan.outputs.direct_mean_intervals
    assert direct.hierarchy_calibration == "flat_bonferroni"
    roots_spec = direct.model_copy(update={"hierarchy_nodes": False})
    without_nodes = plan.model_copy(
        update={"outputs": plan.outputs.model_copy(update={"direct_mean_intervals": roots_spec})}
    )
    with_nodes = _targets(_analyze(monkeypatch, coupled=True))
    roots_only = _targets(_analyze(monkeypatch, coupled=True, plan=without_nodes))
    assert len(with_nodes) == 10
    assert set(roots_only) == {(group, ()) for group in EXPECTED_WEIGHTS}
    for targets, alpha in ((with_nodes, 0.025 / 10), (roots_only, 0.025 / 2)):
        for target, intervals in targets.items():
            for interval in intervals:
                if interval.independence_profile is not None:
                    assert interval.alpha_upper == pytest.approx(alpha, rel=1e-12, abs=0), target
    # So the nodes widen the primary interval a claim is decided on.
    for root, intervals in roots_only.items():
        alone = next(interval for interval in intervals if interval.role == "primary")
        shared = next(interval for interval in with_nodes[root] if interval.role == "primary")
        assert alone.upper is not None and shared.upper is not None
        assert alone.upper < shared.upper, root


def test_maximum_and_top_k_mean_count_wordings_not_base_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two measures docs/plan-reference.md says count wordings.

    `maximum` rests on the prompt bounds, whose equal_bonferroni allocation divides the
    family's 0.025 per direction over all six wordings. `top_k_mean` needs equal weights,
    which nested_equal does not give the wordings of either group.
    """
    plan = load_plan(PLAN)
    top = RiskMeasureSpec(id="top2", type="top_k_mean", k=2)
    with_top = plan.model_copy(
        update={
            "outputs": plan.outputs.model_copy(
                update={"risk_measures": [*plan.outputs.risk_measures, top]}
            )
        }
    )
    report = _analyze(monkeypatch, coupled=True, plan=with_top)

    assert len(report.prompt_estimates) == 6
    for estimate in report.prompt_estimates:
        assert estimate.alpha_upper == pytest.approx(0.025 / 6, rel=1e-12, abs=0)
    worst = {risk.group_id: risk for risk in report.group_risks if risk.measure_id == "worst"}
    uppers = {estimate.prompt_id: estimate.upper for estimate in report.prompt_estimates}
    for group, weights in EXPECTED_WEIGHTS.items():
        largest = max(upper for prompt, upper in uppers.items() if prompt in weights and upper)
        assert worst[group].upper == pytest.approx(largest, rel=1e-12, abs=0)

    refused = sorted(
        item.scope for item in report.integrity.diagnostics if item.code == "RISK_MEASURE_INVALID"
    )
    assert refused == [
        f"cohorts/candidate/groups/{group}/measures/top2" for group in sorted(EXPECTED_WEIGHTS)
    ]
    assert report.integrity.status == "invalid"
    for risk in report.group_risks:
        if risk.measure_id == "top2":
            assert (risk.estimate, risk.lower, risk.upper) == (None, None, None)


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


def _rollout(
    prompt: str, question: str, epoch: int, failure: bool, task: str = "task"
) -> Observation:
    return Observation(
        observation_id=f"{prompt}-{epoch}",
        source="simulated.eval",
        run_key="run",
        rollout_id=f"{prompt}-{epoch}",
        item_id=prompt,
        repeat=epoch,
        prompt_id=prompt,
        question_id=question,
        task_id=task,
        cohort_id="candidate",
        score_name="safety",
        score_raw="FAIL" if failure else "PASS",
        failure=failure,
        model="model",
        configuration_fingerprint="configuration",
        collection_order=epoch,
    )


def test_wordings_sharing_each_epoch_need_question_repeat_blocks_for_coverage() -> None:
    """The simulation docs/plan-reference.md describes for coupled wordings.

    Four base prompts of ten wordings, twenty epochs, every wording at risk 0.12. One draw per
    base prompt and epoch decides every wording at that epoch. The blocks come from the
    analysis's own block construction. Upper bounds at one-sided level 0.025 under `rollout`
    miss the risk far more often than 2.5%; under `[question_id, repeat]` they do not. The
    margin is three simulation standard errors at the nominal rate.
    """
    bases, wordings, risk, alpha, repetitions = 4, 10, 0.12, 0.025, 1000
    manifest = pd.DataFrame(
        [
            {"prompt_id": f"q{base}-w{wording}", "question_id": f"q{base}", "task_id": "task"}
            for base in range(bases)
            for wording in range(wordings)
        ]
    )
    prompts = manifest["prompt_id"].tolist()
    weights = np.full(len(prompts), 1 / len(prompts))
    cells = [
        (prompt_index, prompt, question, epoch)
        for prompt_index, (prompt, question) in enumerate(
            zip(manifest["prompt_id"], manifest["question_id"], strict=True)
        )
        for epoch in range(1, EPOCHS + 1)
    ]
    rollouts = {
        failure: [
            _rollout(prompt, question, epoch, failure) for _, prompt, question, epoch in cells
        ]
        for failure in (False, True)
    }
    base_of = np.asarray([prompt_index // wordings for prompt_index, *_ in cells])
    epoch_of = np.asarray([epoch - 1 for *_, epoch in cells])
    profiles = {
        "rollout": IndependenceProfileSpec(block_by=["rollout"]),
        "question_repeat": IndependenceProfileSpec(block_by=["question_id", "repeat"]),
    }
    methods = ("weighted_hoeffding", "weighted_empirical_bernstein", "weighted_kl_chernoff")
    misses = dict.fromkeys(((profile, method) for profile in profiles for method in methods), 0)
    generator = np.random.default_rng(20261119)
    for _ in range(repetitions):
        shared = generator.random((bases, EPOCHS)) < risk
        failures = shared[base_of, epoch_of]
        observations = [rollouts[bool(failure)][row] for row, failure in enumerate(failures)]
        index = block_index(observations, manifest)
        for name, profile in profiles.items():
            blocks = direct_bounded_blocks(
                observations,
                manifest,
                profile,
                cohort_id="candidate",
                prompt_ids=prompts,
                prompt_weights=weights,
                index=index,
            )
            for method in methods:
                bounds = DIRECT_MEAN_ENGINES[method].bounded(
                    blocks.lower,
                    blocks.upper,
                    blocks.weights,
                    alpha_lower=alpha,
                    alpha_upper=alpha,
                    independence_profile=name,
                )
                misses[(name, method)] += bounds.upper < risk

    limit = alpha + 3 * (alpha * (1 - alpha) / repetitions) ** 0.5
    for method in methods:
        assert misses[("rollout", method)] / repetitions > limit, (method, misses)
        assert misses[("question_repeat", method)] / repetitions <= limit, (method, misses)


def test_a_block_key_is_the_raw_value_not_the_hierarchy_path() -> None:
    """Two tasks that both name a base prompt q1 share its block unless task_id is in the key."""
    manifest = pd.DataFrame(
        [
            {"prompt_id": f"{task}-q1-w{wording}", "question_id": "q1", "task_id": task}
            for task in ("t1", "t2")
            for wording in range(2)
        ]
    )
    observations = [
        _rollout(prompt, "q1", epoch, failure=False, task=task)
        for prompt, task in zip(manifest["prompt_id"], manifest["task_id"], strict=True)
        for epoch in (1, 2)
    ]
    counts = {
        tuple(block_by): direct_bounded_blocks(
            observations,
            manifest,
            IndependenceProfileSpec(block_by=block_by),
            cohort_id="candidate",
            prompt_ids=manifest["prompt_id"].tolist(),
            prompt_weights=np.full(4, 0.25),
        ).count
        for block_by in (
            ["question_id"],
            ["task_id", "question_id"],
            ["question_id", "repeat"],
            ["task_id", "question_id", "repeat"],
        )
    }
    assert counts == {
        ("question_id",): 1,
        ("task_id", "question_id"): 2,
        ("question_id", "repeat"): 2,
        ("task_id", "question_id", "repeat"): 4,
    }
