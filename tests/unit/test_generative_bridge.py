"""Rewrite execution, Inspect materialization, and the outcome join."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats.generative.execution import (
    execute_rewrites,
    load_seed_texts,
    write_prompt_store,
)
from inspect_safety_stats.generative.inference import estimate_tree
from inspect_safety_stats.generative.inspect_bridge import (
    EvaluatedSample,
    join_outcomes,
    materialize_dataset,
    materialize_for_cohorts,
    samples_from_observations,
)
from inspect_safety_stats.generative.kernels import (
    DeterministicFakeKernel,
    RewriteOutcome,
    RewriteRequest,
    resolve_kernel_fingerprint,
)
from inspect_safety_stats.generative.sampling import draw_selections, plan_design
from inspect_safety_stats.generative.schema import load_generative_plan
from inspect_safety_stats.generative.taxonomy import compile_taxonomy
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan, materialize


@dataclass
class RepeatingKernel:
    """Always returns the same prompt for a seed, so deduplication has to fire."""

    kernel_id: str = "repeating"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        return RewriteOutcome(status="valid", prompt=f"fixed rewrite of {request.semantic_seed_id}")


def prepare(tmp_path: Path, total: int = 60, **overrides: Any) -> tuple[Any, ...]:
    plan = base_plan()
    plan["sampling"] = {"method": "direct", "total_rewrites": total}
    plan["target"]["cohorts"] = {
        "baseline": {"model": "model-a"},
        "candidate": {"model": "model-b"},
    }
    for key, value in overrides.items():
        plan["rewrite"][key] = value
    parsed = load_generative_plan(materialize(tmp_path, plan, SEEDS_WITH_TEXT))
    taxonomy = compile_taxonomy(parsed, base_path=tmp_path)
    design = plan_design(taxonomy, parsed)
    selections = draw_selections(taxonomy, design)
    fingerprint = resolve_kernel_fingerprint(parsed, base_path=tmp_path)
    texts = load_seed_texts(parsed, base_path=tmp_path)
    return parsed, taxonomy, design, selections, fingerprint, texts


def test_seed_texts_load_from_the_manifest(tmp_path: Path) -> None:
    _, _, _, _, _, texts = prepare(tmp_path)
    assert texts["s001"].startswith("How do I synthesize compound A")
    assert set(texts) == {"s001", "s002", "s003"}


def test_execution_records_every_selection(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    assert len(result.rewrites) == len(selections)
    assert len(result.accepted) == len(selections)
    assert {record.sampling_unit_id for record in result.rewrites} == {
        record.sampling_unit_id for record in selections
    }
    assert all(record.rewrite_kernel_sha256 == fingerprint.sha256 for record in result.rewrites)


def test_execution_is_deterministic(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    first = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    again = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    assert first.accepted == again.accepted


def test_a_salt_makes_a_cohort_rewrites_independent(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    shared = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    salted = execute_rewrites(
        plan, selections, DeterministicFakeKernel(), fingerprint, texts, rewrite_salt="candidate"
    )
    assert [item.prompt for item in shared.accepted] != [item.prompt for item in salted.accepted]
    assert [item.prompt_id for item in shared.accepted] == [
        item.prompt_id for item in salted.accepted
    ]


def test_exhausted_retries_are_kept_as_incomplete_units(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    always_invalid = DeterministicFakeKernel(invalid_every=1)
    result = execute_rewrites(plan, selections, always_invalid, fingerprint, texts)
    assert not result.accepted
    assert len(result.exhausted) == len(selections)
    assert len(result.rewrites) == len(selections)
    assert all(record.validation_status == "invalid" for record in result.rewrites)
    assert all(record.rewrite_attempts == 5 for record in result.rewrites)
    assert all(record.prompt_id is None for record in result.rewrites)


def test_deduplication_rejects_repeats_within_its_scope(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, RepeatingKernel(), fingerprint, texts)
    # One accepted rewrite per seed; every later draw on that seed is a duplicate.
    assert len(result.accepted) == 3
    assert result.duplicates_rejected > 0
    assert len(result.exhausted) == len(selections) - 3


def test_deduplication_can_be_switched_off(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(
        tmp_path, deduplication={"scope": "none", "on_duplicate": "accept"}
    )
    result = execute_rewrites(plan, selections, RepeatingKernel(), fingerprint, texts)
    assert len(result.accepted) == len(selections)
    assert result.duplicates_rejected == 0


def test_the_ledger_holds_hashes_and_the_prompt_store_holds_text(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    for record in result.rewrites:
        serialized = json.dumps(record.model_dump(mode="json"))
        assert "How do I synthesize" not in serialized
        assert record.prompt_sha256 is not None

    store = tmp_path / "prompts.csv"
    write_prompt_store(result.accepted, store)
    assert "How do I synthesize" in store.read_text(encoding="utf-8")


def test_materialization_writes_one_sample_per_accepted_rewrite(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    path = tmp_path / "dataset.jsonl"
    written = materialize_dataset(selections, result, path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert written == len(rows) == len(result.accepted)
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["id"] for row in rows} == {item.prompt_id for item in result.accepted}
    sample = rows[0]
    assert sample["input"]
    assert sample["metadata"]["path_hash"]
    assert "prompt" not in sample["metadata"]


def test_shared_rewrites_give_paired_cohorts_the_same_file(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    datasets = materialize_for_cohorts(plan, selections, result, tmp_path / "datasets")
    assert set(datasets) == {"baseline", "candidate"}
    assert datasets["baseline"] == datasets["candidate"]


def test_independent_rewrites_are_refused_by_the_paired_helper(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    plan.target.shared_rewrites = False
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    with pytest.raises(ValueError, match="distinct rewrite_salt"):
        materialize_for_cohorts(plan, selections, result, tmp_path / "datasets")


def evaluate(result: Any, *, cohort: str = "baseline", failing: int = 0) -> list[EvaluatedSample]:
    return [
        EvaluatedSample(
            sample_id=entry.prompt_id,
            cohort_id=cohort,
            epoch=1,
            failure=index < failing,
        )
        for index, entry in enumerate(result.accepted)
    ]


def test_join_maps_samples_back_to_sampling_units(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    joined = join_outcomes(
        plan, selections, list(result.rewrites), evaluate(result, failing=7), cohort_id="baseline"
    )
    assert len(joined.outcomes) == len(selections)
    assert sum(outcome.failures for outcome in joined.outcomes) == 7
    assert sum(outcome.unclassified for outcome in joined.outcomes) == 0
    assert not joined.has_errors


def test_join_detects_a_duplicated_sample(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    samples = evaluate(result)
    joined = join_outcomes(
        plan, selections, list(result.rewrites), samples + samples[:3], cohort_id="baseline"
    )
    codes = [item.code for item in joined.diagnostics]
    assert codes.count("GENERATIVE_SAMPLE_DUPLICATE") == 3
    assert joined.has_errors
    # The repeat is discarded rather than counted twice.
    assert sum(outcome.trials for outcome in joined.outcomes) == len(selections)


def test_join_detects_a_sample_from_another_run(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    stray = EvaluatedSample(sample_id="p-not-ours", cohort_id="baseline", epoch=1, failure=True)
    joined = join_outcomes(
        plan, selections, list(result.rewrites), [*evaluate(result), stray], cohort_id="baseline"
    )
    assert any(item.code == "GENERATIVE_SAMPLE_UNMATCHED" for item in joined.diagnostics)
    assert joined.has_errors


def test_unevaluated_units_are_completed_as_unclassified(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    partial = evaluate(result)[:40]
    joined = join_outcomes(plan, selections, list(result.rewrites), partial, cohort_id="baseline")
    assert len(joined.outcomes) == len(selections)
    assert len(joined.unevaluated) == len(selections) - 40
    assert sum(outcome.unclassified for outcome in joined.outcomes) == len(selections) - 40
    assert any(item.code == "GENERATIVE_UNIT_UNEVALUATED" for item in joined.diagnostics)


def test_join_rejects_an_undeclared_cohort(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    with pytest.raises(ValueError, match="not declared in target.cohorts"):
        join_outcomes(plan, selections, list(result.rewrites), [], cohort_id="ghost")


def test_join_detects_more_rollouts_than_the_plan_declares(tmp_path: Path) -> None:
    plan, _, _, selections, fingerprint, texts = prepare(tmp_path)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    samples = evaluate(result)
    extra = [
        EvaluatedSample(sample_id=samples[0].sample_id, cohort_id="baseline", epoch=2, failure=True)
    ]
    joined = join_outcomes(
        plan, selections, list(result.rewrites), samples + extra, cohort_id="baseline"
    )
    assert any(item.code == "GENERATIVE_ROLLOUT_OVERRUN" for item in joined.diagnostics)


def test_end_to_end_from_plan_to_interval(tmp_path: Path) -> None:
    plan, taxonomy, design, selections, fingerprint, texts = prepare(tmp_path, total=400)
    result = execute_rewrites(plan, selections, DeterministicFakeKernel(), fingerprint, texts)
    materialize_for_cohorts(plan, selections, result, tmp_path / "datasets")
    joined = join_outcomes(
        plan, selections, list(result.rewrites), evaluate(result, failing=8), cohort_id="baseline"
    )
    assert not joined.has_errors

    tree = estimate_tree(
        taxonomy, design, selections, list(joined.outcomes), alpha=0.05, report_all_nodes=False
    )
    assert tree.root.draws == 400
    assert tree.root.failures == 8
    assert tree.root.estimate == pytest.approx(0.02)
    assert tree.root.lower == 0.0 < tree.root.upper < 0.05
    assert tree.root.method == "clopper_pearson_one_sided_missingness_robust"


def test_observations_adapt_to_the_join(tmp_path: Path) -> None:
    from inspect_safety_stats.schema.observations import Observation

    observation = Observation(
        observation_id="o1",
        source="log.eval",
        run_key="e1",
        rollout_id="p-u-000000",
        item_id="p-u-000000",
        repeat=1,
        prompt_id="",
        question_id="",
        task_id="",
        cohort_id="baseline",
        score_name="harmful",
        failure=True,
        model="model-a",
        configuration_fingerprint="f",
        collection_order=0,
    )
    adapted = samples_from_observations([observation])
    assert adapted == [
        EvaluatedSample(sample_id="p-u-000000", cohort_id="baseline", epoch=1, failure=True)
    ]
