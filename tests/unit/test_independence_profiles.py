from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import (
    DirectMeanIntervalsSpec,
    IndependenceProfileSpec,
)
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.stats.independence import (
    BoundedBlocks,
    block_index,
    comparison_contrast_blocks,
    direct_bounded_blocks,
    prompt_level_block_ids,
    validate_profile_fields,
)


def _observation(cohort: str, prompt: str, epoch: int, failure: bool) -> Observation:
    return Observation(
        observation_id=f"{cohort}-{prompt}-{epoch}",
        source="test.eval",
        run_key=f"eval-{cohort}",
        rollout_id=f"{cohort}-{prompt}-{epoch}",
        item_id=prompt,
        repeat=epoch,
        prompt_id=prompt,
        question_id=prompt,
        task_id="task",
        cohort_id=cohort,
        score_name="safety",
        score_raw="FAIL" if failure else "PASS",
        failure=failure,
        model=cohort,
        configuration_fingerprint=cohort,
        collection_order=epoch,
    )


def _data() -> tuple[list[Observation], pd.DataFrame]:
    observations = [
        _observation(cohort, prompt, epoch, failure=(prompt == "p1" and epoch == 1))
        for cohort in ("candidate", "baseline")
        for prompt in ("p1", "p2")
        for epoch in (1, 2)
    ]
    manifest = pd.DataFrame(
        [
            {"prompt_id": "p1", "question_id": "q1", "task_id": "task", "domain": "a"},
            {"prompt_id": "p2", "question_id": "q2", "task_id": "task", "domain": "a"},
        ]
    )
    return observations, manifest


def test_profiles_form_rollout_prompt_and_manifest_blocks() -> None:
    observations, manifest = _data()
    weights = np.asarray([0.5, 0.5])
    rollout = direct_bounded_blocks(
        observations,
        manifest,
        IndependenceProfileSpec(block_by=["rollout"]),
        cohort_id="candidate",
        prompt_ids=["p1", "p2"],
        prompt_weights=weights,
    )
    prompt = direct_bounded_blocks(
        observations,
        manifest,
        IndependenceProfileSpec(block_by=["prompt_id"]),
        cohort_id="candidate",
        prompt_ids=["p1", "p2"],
        prompt_weights=weights,
    )
    domain = direct_bounded_blocks(
        observations,
        manifest,
        IndependenceProfileSpec(block_by=["domain"]),
        cohort_id="candidate",
        prompt_ids=["p1", "p2"],
        prompt_weights=weights,
    )
    assert (rollout.count, prompt.count, domain.count) == (4, 2, 1)
    assert np.dot(rollout.weights, rollout.lower) == pytest.approx(0.25)
    assert np.dot(prompt.weights, prompt.lower) == pytest.approx(0.25)
    assert np.dot(domain.weights, domain.lower) == pytest.approx(0.25)


def test_comparison_coupling_controls_cross_arm_block_partition() -> None:
    observations, manifest = _data()
    arguments = dict(
        observations=observations,
        manifest=manifest,
        candidate="candidate",
        baseline="baseline",
        prompt_ids=["p1", "p2"],
        prompt_weights=np.asarray([0.5, 0.5]),
    )
    independent = comparison_contrast_blocks(
        **arguments,
        profile=IndependenceProfileSpec(block_by=["prompt_id"], comparison_coupling="independent"),
    )
    joint = comparison_contrast_blocks(
        **arguments,
        profile=IndependenceProfileSpec(block_by=["prompt_id"], comparison_coupling="joint"),
    )
    assert independent.count == 4
    assert joint.count == 2
    assert np.dot(independent.weights, independent.lower) == pytest.approx(0.5)
    assert np.dot(joint.weights, joint.lower) == pytest.approx(0.5)


def test_profile_fields_are_checked_against_observations_and_manifest() -> None:
    _, manifest = _data()
    validate_profile_fields(
        {
            "rollout": IndependenceProfileSpec(block_by=["rollout"]),
            "domain": IndependenceProfileSpec(block_by=["domain"]),
            "epoch": IndependenceProfileSpec(block_by=["epoch"]),
        },
        manifest,
    )
    with pytest.raises(ValueError, match="unavailable fields"):
        validate_profile_fields(
            {"bad": IndependenceProfileSpec(block_by=["not_a_field"])}, manifest
        )


@pytest.mark.parametrize("value", [1.0, "a"])
def test_missing_manifest_block_values_are_rejected(value: object) -> None:
    """A missing block value split or merged rollouts depending on the column's dtype."""
    _, manifest = _data()
    manifest["cluster"] = [value, None]
    manifest = manifest.astype({"cluster": "float64" if isinstance(value, float) else "object"})
    profiles = {"cluster": IndependenceProfileSpec(block_by=["cluster"])}
    with pytest.raises(ValueError, match=r"column 'cluster'.*1 prompt\(s\): p2"):
        validate_profile_fields(profiles, manifest)
    # Observation fields come from the logs, not the manifest, and are never missing.
    validate_profile_fields({"prompt": IndependenceProfileSpec(block_by=["prompt_id"])}, manifest)


def test_legacy_direct_mean_independence_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="independence_unit"):
        DirectMeanIntervalsSpec.model_validate(
            {
                "enabled": True,
                "family": "means",
                "independence_unit": "rollout",
            }
        )
    with pytest.raises(ValidationError, match="cluster_sensitivity"):
        DirectMeanIntervalsSpec.model_validate(
            {
                "enabled": True,
                "family": "means",
                "cluster_sensitivity": True,
            }
        )


def _same(left: BoundedBlocks, right: BoundedBlocks) -> bool:
    return (
        left.block_ids == right.block_ids
        and left.lower.tobytes() == right.lower.tobytes()
        and left.upper.tobytes() == right.upper.tobytes()
        and left.weights.tobytes() == right.weights.tobytes()
    )


def test_block_index_values_match_dataframe_at_for_every_dtype() -> None:
    """Block keys are built and serialized from these values, so even their types must match."""
    manifest = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2"],
            "question_id": ["q1", "q2"],
            "task_id": ["t", "t"],
            "integer": [1, 2],
            "real": [0.1, 0.25],
            "flag": [True, False],
            "text": ["a", "b"],
            "string": pd.array(["x", "y"], dtype="string"),
            "mixed": pd.Series([1, "b"], dtype="object"),
        }
    )
    values = block_index([], manifest).manifest_values
    indexed = manifest.set_index("prompt_id", drop=False)
    for prompt in ("p1", "p2"):
        for column in manifest.columns:
            expected = indexed.at[prompt, column]
            actual = values[prompt][column]
            assert type(actual) is type(expected), (prompt, column)
            assert actual == expected and str(actual) == str(expected), (prompt, column)


def test_a_shared_block_index_gives_identical_blocks() -> None:
    observations, manifest = _data()
    index = block_index(observations, manifest)
    weights = np.asarray([0.25, 0.75])
    for block_by in (["rollout"], ["prompt_id"], ["domain"], ["epoch"]):
        for coupling in ("independent", "joint"):
            if block_by == ["rollout"] and coupling == "joint":
                continue
            profile = IndependenceProfileSpec(block_by=block_by, comparison_coupling=coupling)
            inputs = (observations, manifest, profile)
            for cohort in ("candidate", "baseline"):
                arguments = dict(cohort_id=cohort, prompt_ids=["p1", "p2"], prompt_weights=weights)
                assert _same(
                    direct_bounded_blocks(*inputs, **arguments),
                    direct_bounded_blocks(*inputs, index=index, **arguments),
                )
            arguments = dict(
                candidate="candidate",
                baseline="baseline",
                prompt_ids=["p1", "p2"],
                prompt_weights=weights,
            )
            assert _same(
                comparison_contrast_blocks(*inputs, **arguments),
                comparison_contrast_blocks(*inputs, index=index, **arguments),
            )


def test_a_block_index_for_other_observations_is_not_used() -> None:
    observations, manifest = _data()
    stale = block_index(observations[:1], manifest)
    inputs = (observations, manifest, IndependenceProfileSpec(block_by=["prompt_id"]))
    arguments = dict(
        cohort_id="candidate", prompt_ids=["p1", "p2"], prompt_weights=np.asarray([0.5, 0.5])
    )
    assert _same(
        direct_bounded_blocks(*inputs, **arguments),
        direct_bounded_blocks(*inputs, index=stale, **arguments),
    )


@pytest.mark.parametrize(
    ("legacy", "neutral"),
    [("epoch", "repeat"), ("eval_id", "run_key"), ("sample_id", "rollout_id")],
)
def test_a_legacy_observation_field_blocks_like_its_neutral_name(
    legacy: str, neutral: str
) -> None:
    observations, manifest = _data()
    arguments = dict(
        cohort_id="candidate", prompt_ids=["p1", "p2"], prompt_weights=np.asarray([0.5, 0.5])
    )
    blocks = {
        name: direct_bounded_blocks(
            observations, manifest, IndependenceProfileSpec(block_by=[name]), **arguments
        )
        for name in (legacy, neutral)
    }
    assert _same(blocks[legacy], blocks[neutral])


def _ledger() -> tuple[list[Observation], pd.DataFrame]:
    """Two arms with two rollouts of four prompts, taking question and task as normalization does.

    The manifest's question ids are numbers here, which the observations hold as text.
    """

    manifest = pd.DataFrame(
        [
            {"prompt_id": "p1", "question_id": 1, "task_id": "t1", "domain": "a", "level": 1},
            {"prompt_id": "p2", "question_id": 1, "task_id": "t1", "domain": "b", "level": 1},
            {"prompt_id": "p3", "question_id": 2, "task_id": "t2", "domain": "a", "level": 2},
            {"prompt_id": "p4", "question_id": 3, "task_id": "t2", "domain": "b", "level": 2},
        ]
    )
    observations = []
    for cohort in ("candidate", "baseline"):
        for row in manifest.itertuples():
            for repeat in (1, 2):
                observation = _observation(cohort, str(row.prompt_id), repeat, failure=False)
                observations.append(
                    observation.model_copy(
                        update={"question_id": str(row.question_id), "task_id": row.task_id}
                    )
                )
    return observations, manifest


@pytest.mark.parametrize(
    "block_by",
    [["prompt_id"], ["question_id"], ["task_id"], ["domain"], ["level"], ["task_id", "domain"]],
)
@pytest.mark.parametrize("coupling", ["independent", "joint"])
def test_prompt_level_blocks_are_those_a_complete_ledger_forms(
    block_by: list[str], coupling: str
) -> None:
    observations, manifest = _ledger()
    index = block_index(observations, manifest)
    profile = IndependenceProfileSpec(block_by=block_by, comparison_coupling=coupling)
    # p2 has no weight, so it forms no block.
    prompts: dict[str, Any] = {
        "prompt_ids": ["p1", "p2", "p3", "p4"],
        "prompt_weights": np.asarray([0.5, 0, 0.25, 0.25]),
    }
    direct = direct_bounded_blocks(
        observations, manifest, profile, cohort_id="candidate", index=index, **prompts
    )
    contrast = comparison_contrast_blocks(
        observations,
        manifest,
        profile,
        candidate="candidate",
        baseline="baseline",
        index=index,
        **prompts,
    )
    assert prompt_level_block_ids(profile, index.manifest_values, **prompts) == frozenset(
        direct.block_ids
    )
    assert prompt_level_block_ids(
        profile, index.manifest_values, comparison_arms=("candidate", "baseline"), **prompts
    ) == frozenset(contrast.block_ids)


@pytest.mark.parametrize(
    "block_by",
    [["rollout"], ["run_key"], ["repeat"], ["epoch"], ["item_id"], ["question_id", "repeat"]],
)
def test_blocks_that_depend_on_the_rollouts_are_not_known_before_them(
    block_by: list[str],
) -> None:
    observations, manifest = _ledger()
    index = block_index(observations, manifest)
    assert (
        prompt_level_block_ids(
            IndependenceProfileSpec(block_by=block_by),
            index.manifest_values,
            prompt_ids=["p1", "p2"],
            prompt_weights=np.asarray([0.5, 0.5]),
        )
        is None
    )
