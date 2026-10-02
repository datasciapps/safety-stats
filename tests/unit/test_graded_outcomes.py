from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from inspect_safety_stats.config.models import InputsSpec, OutcomeMappingSpec
from inspect_safety_stats.ingest.outcomes import map_harm
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds


def _scale(harmful: str = "high") -> OutcomeMappingSpec:
    return OutcomeMappingSpec.model_validate(
        {"scale": {"minimum": 1, "maximum": 5, "harmful": harmful}}
    )


def test_scale_maps_linearly_in_either_direction() -> None:
    assert map_harm(1, _scale()) == 0.0
    assert map_harm(5, _scale()) == 1.0
    assert map_harm(4, _scale()) == pytest.approx(0.75)
    assert map_harm(4, _scale("low")) == pytest.approx(0.25)
    assert map_harm(None, _scale()) is None


def test_scale_refuses_values_it_does_not_describe() -> None:
    with pytest.raises(ValueError, match="outside the declared scale"):
        map_harm(6, _scale())
    with pytest.raises(ValueError, match="nonnumeric"):
        map_harm("high", _scale())
    with pytest.raises(ValueError, match="nonnumeric"):
        map_harm(True, _scale())


def test_binary_mappings_give_zero_or_one() -> None:
    mapping = OutcomeMappingSpec(failure_values=["FAIL"], safe_values=["PASS"])
    assert map_harm("FAIL", mapping) == 1.0
    assert map_harm("PASS", mapping) == 0.0


def test_exactly_one_mapping_style() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        OutcomeMappingSpec.model_validate(
            {"failure_values": ["F"], "safe_values": ["P"], "scale": {"minimum": 0, "maximum": 1}}
        )
    with pytest.raises(ValidationError, match="maximum > minimum"):
        OutcomeMappingSpec.model_validate({"scale": {"minimum": 1, "maximum": 1}})


def test_judge_calibration_is_refused_for_graded_scores() -> None:
    with pytest.raises(ValidationError, match="binary judge verdict"):
        InputsSpec.model_validate(
            {
                "design_manifest": "design.csv",
                "score": {"name": "rubric", "mapping": {"scale": {"minimum": 1, "maximum": 5}}},
                "judge_calibration": {
                    "mode": "assumed",
                    "primary": "a",
                    "scenarios": [
                        {"id": "a", "rho_1": {"lower": 0.9, "upper": 1.0},
                         "rho_0": {"lower": 0.0, "upper": 0.01}}
                    ],
                },
            }
        )


def _observation(**values: object) -> Observation:
    base: dict[str, object] = {
        "observation_id": "o", "source_log": "l", "eval_id": "e", "sample_id": "s",
        "inspect_id": "p", "epoch": 1, "prompt_id": "p", "question_id": "p", "task_id": "t",
        "cohort_id": "c", "score_name": "s", "model": "m", "configuration_fingerprint": "f",
        "collection_order": 1,
    }
    base.update(values)
    return Observation.model_validate(base)


def test_observation_harm_defaults_from_binary_failure() -> None:
    assert _observation(failure=True).harm == 1.0
    assert _observation(failure=False).harm == 0.0
    assert _observation().harm is None
    assert _observation(failure=True, harm=0.4).harm == 0.4
    with pytest.raises(ValidationError, match="must be classified"):
        _observation(harm=0.4)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        _observation(failure=True, harm=1.5)


def test_kl_bounds_cover_graded_means_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(20261002)
    alpha = 0.05
    misses = 0
    replicates = 400
    for _ in range(replicates):
        means = rng.uniform(0, 0.3, size=20)
        samples = np.concatenate(
            [rng.beta(1.0, (1 - mean) / mean, size=10) for mean in means]
        )
        weights = np.full(len(samples), 1 / len(samples))
        bounds = weighted_bounded_kl_chernoff_mean_bounds(
            samples, samples, weights, alpha_lower=alpha / 2, alpha_upper=alpha / 2,
            independence_profile="rollout",
        )
        truth = float(np.mean(means))
        misses += not bounds.lower <= truth <= bounds.upper
    assert misses / replicates <= alpha
