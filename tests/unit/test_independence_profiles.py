from __future__ import annotations

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
    comparison_contrast_blocks,
    direct_bounded_blocks,
    validate_profile_fields,
)


def _observation(cohort: str, prompt: str, epoch: int, failure: bool) -> Observation:
    return Observation(
        observation_id=f"{cohort}-{prompt}-{epoch}",
        source_log="test.eval",
        eval_id=f"eval-{cohort}",
        sample_id=f"{cohort}-{prompt}-{epoch}",
        inspect_id=prompt,
        epoch=epoch,
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
