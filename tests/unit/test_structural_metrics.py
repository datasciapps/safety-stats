from __future__ import annotations

import numpy as np
import pytest

from inspect_safety_stats.structural.audit_metrics import (
    concentration_summaries,
    hierarchy_marginal_risk_summaries,
    prompt_risk_summaries,
    vulnerable_mass_summaries,
    weighted_tail_risk_draws,
)


def test_weighted_tail_uses_fractional_boundary_weight() -> None:
    risks = np.asarray([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    weights = np.asarray([0.2, 0.3, 0.5])
    result = weighted_tail_risk_draws(risks, weights, prompt_mass=0.25)
    assert result[0] == pytest.approx(0.2 * 0.9 + 0.05 * 0.5)
    assert result[1] == pytest.approx(0.25 * 0.9)


def test_vulnerable_mass_is_computed_for_each_posterior_draw() -> None:
    risks = np.asarray([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    weights = np.asarray([0.2, 0.3, 0.5])
    result = vulnerable_mass_summaries(
        risks,
        weights,
        thresholds=[0.4],
        credible_mass=0.8,
    )[0]
    assert result.mass.mean == pytest.approx((0.5 + 0.8) / 2)
    assert result.expected_exceedance_mass == pytest.approx(0.65)


def test_concentration_reports_absolute_tail_and_near_zero_draws() -> None:
    risks = np.asarray([[0.9, 0.5, 0.1], [0.0, 0.0, 0.0]])
    weights = np.asarray([0.2, 0.3, 0.5])
    result = concentration_summaries(
        risks,
        weights,
        prompt_masses=[0.25],
        credible_mass=0.8,
        zero_risk_tolerance=1e-12,
    )[0]
    assert result.near_zero_total_risk_probability == pytest.approx(0.5)
    assert result.risk_share is not None
    assert result.risk_share.mean == pytest.approx((0.2 * 0.9 + 0.05 * 0.5) / 0.38)


def test_prompt_summary_contains_no_prompt_content() -> None:
    result = prompt_risk_summaries(
        np.asarray([[0.1], [0.2], [0.3]]),
        ["prompt-1"],
        np.asarray([1.0]),
        thresholds=[0.15],
        credible_mass=0.8,
    )[0]
    assert set(result.model_dump()) == {
        "prompt_id",
        "weight",
        "risk",
        "exceedance_probability",
    }
    assert result.exceedance_probability["0.15"] == pytest.approx(2 / 3)


def test_hierarchy_marginals_use_normalized_design_weights_within_node() -> None:
    result = hierarchy_marginal_risk_summaries(
        np.asarray([[0.1, 0.3, 0.9], [0.2, 0.4, 0.8]]),
        np.asarray([0.2, 0.3, 0.5]),
        hierarchy=("task", "domain"),
        hierarchy_values={
            "task": ["t", "t", "t"],
            "domain": ["a", "a", "b"],
        },
        credible_mass=0.8,
    )
    task = next(item for item in result if item.level == "task")
    domain_a = next(item for item in result if item.level == "domain" and item.category == "a")
    assert task.prompt_count == 3
    assert task.design_weight == pytest.approx(1.0)
    assert task.risk.mean == pytest.approx(0.56)
    assert domain_a.path == {"task": "t", "domain": "a"}
    assert domain_a.design_weight == pytest.approx(0.5)
    assert domain_a.risk.mean == pytest.approx((0.22 + 0.32) / 2)


def test_tail_metric_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        weighted_tail_risk_draws(
            np.asarray([[0.1, 0.2]]),
            np.asarray([0.2, 0.2]),
            prompt_mass=0.5,
        )
