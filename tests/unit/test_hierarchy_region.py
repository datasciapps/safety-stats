import itertools

import numpy as np
import pytest

from safety_stats.stats import (
    LinearIntervalConstraint,
    project_contrast_hierarchy_confidence_region,
    project_hierarchy_confidence_region,
    project_ratio_hierarchy_confidence_region,
    project_two_arm_difference_hierarchy_confidence_region,
    weighted_kl_chernoff_mean_bounds,
)


def test_projection_propagates_parent_information_to_child() -> None:
    constraints = [
        LinearIntervalConstraint("parent", np.array([0.5, 0.5]), 0.10, 0.20),
        LinearIntervalConstraint("child_a", np.array([1.0, 0.0]), 0.00, 0.10),
        LinearIntervalConstraint("child_b", np.array([0.0, 1.0]), 0.00, 1.00),
    ]
    result = project_hierarchy_confidence_region(
        constraints,
        {
            "parent": [0.5, 0.5],
            "child_a": [1.0, 0.0],
            "child_b": [0.0, 1.0],
        },
        variable_count=2,
    )
    assert result.feasible
    assert result.intervals["parent"] == pytest.approx((0.10, 0.20))
    assert result.intervals["child_a"] == pytest.approx((0.00, 0.10))
    assert result.intervals["child_b"] == pytest.approx((0.10, 0.40))


def test_projection_uses_declared_weights_not_observation_counts() -> None:
    result = project_hierarchy_confidence_region(
        [
            LinearIntervalConstraint("parent", np.array([0.8, 0.2]), 0.15, 0.15),
            LinearIntervalConstraint("child_a", np.array([1.0, 0.0]), 0.10, 0.10),
        ],
        {"child_b": [0.0, 1.0]},
        variable_count=2,
    )
    assert result.feasible
    assert result.intervals["child_b"] == pytest.approx((0.35, 0.35))


def test_projection_reports_an_empty_region() -> None:
    result = project_hierarchy_confidence_region(
        [
            LinearIntervalConstraint("parent", np.array([0.5, 0.5]), 0.10, 0.20),
            LinearIntervalConstraint("child_a", np.array([1.0, 0.0]), 0.80, 0.90),
            LinearIntervalConstraint("child_b", np.array([0.0, 1.0]), 0.80, 0.90),
        ],
        {"parent": [0.5, 0.5]},
        variable_count=2,
    )
    assert not result.feasible
    assert result.intervals == {}
    assert result.message


def test_projection_preserves_simultaneous_coverage_under_heterogeneity() -> None:
    probabilities = np.array([0.05, 0.25, 0.60, 0.80])
    true_targets = {
        "root": float(np.mean(probabilities)),
        "left": float(np.mean(probabilities[:2])),
        "right": float(np.mean(probabilities[2:])),
    }
    coefficients = {
        "root": np.array([0.25, 0.25, 0.25, 0.25]),
        "left": np.array([0.5, 0.5, 0.0, 0.0]),
        "right": np.array([0.0, 0.0, 0.5, 0.5]),
    }
    endpoint_alpha = 0.05 / 2 / 3
    coverage = 0.0
    for outcomes in itertools.product((0, 1), repeat=4):
        mass = float(
            np.prod(
                [
                    probability if outcome else 1 - probability
                    for outcome, probability in zip(outcomes, probabilities, strict=True)
                ]
            )
        )
        constraints: list[LinearIntervalConstraint] = []
        for target_id, indices in {
            "root": (0, 1, 2, 3),
            "left": (0, 1),
            "right": (2, 3),
        }.items():
            result = weighted_kl_chernoff_mean_bounds(
                failures=[outcomes[index] for index in indices],
                unclassified=[0] * len(indices),
                trials=[1] * len(indices),
                weights=[1 / len(indices)] * len(indices),
                alpha_lower=endpoint_alpha,
                alpha_upper=endpoint_alpha,
            )
            constraints.append(
                LinearIntervalConstraint(
                    target_id,
                    coefficients[target_id],
                    result.lower,
                    result.upper,
                )
            )
        projection = project_hierarchy_confidence_region(
            constraints,
            coefficients,
            variable_count=4,
        )
        if projection.feasible and all(
            projection.intervals[target_id][0] <= true_value <= projection.intervals[target_id][1]
            for target_id, true_value in true_targets.items()
        ):
            coverage += mass
    assert coverage >= 0.95 - 1e-12


def test_a_projected_upper_endpoint_can_miss_when_only_a_raw_lower_endpoint_misses() -> None:
    """Projection keeps simultaneous coverage, not one-sided coverage in each direction.

    That is why the analysis reports projected intervals at their simultaneous floor in each
    direction.
    """
    constraints = [
        LinearIntervalConstraint("parent", np.array([0.5, 0.5]), 0.20, 0.50),
        LinearIntervalConstraint("child_a", np.array([1.0, 0.0]), 0.05, 0.80),
        LinearIntervalConstraint("child_b", np.array([0.0, 1.0]), 0.90, 1.00),
    ]
    truth = np.array([0.5, 0.1])
    assert all(float(c.coefficients @ truth) <= c.upper for c in constraints)
    assert float(constraints[2].coefficients @ truth) < constraints[2].lower
    projection = project_hierarchy_confidence_region(
        constraints, {"child_a": [1.0, 0.0]}, variable_count=2
    )
    _, projected_upper = projection.intervals["child_a"]
    assert projected_upper == pytest.approx(0.1, abs=1e-9)
    assert projected_upper < truth[0]


def test_contrast_region_propagates_child_constraints_to_root() -> None:
    constraints = [
        LinearIntervalConstraint("root", np.array([0.5, 0.5]), -1.0, 1.0),
        LinearIntervalConstraint("left", np.array([1.0, 0.0]), -0.4, -0.2),
        LinearIntervalConstraint("right", np.array([0.0, 1.0]), 0.2, 0.6),
    ]

    result = project_contrast_hierarchy_confidence_region(
        constraints, {"root": np.array([0.5, 0.5])}, variable_count=2
    )

    assert result.feasible
    assert result.intervals["root"] == pytest.approx((-0.1, 0.2))


def test_ratio_region_uses_common_two_arm_hierarchy() -> None:
    root = np.array([0.5, 0.5])
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])
    candidate = [
        LinearIntervalConstraint("root", root, 0.0, 1.0),
        LinearIntervalConstraint("left", left, 0.1, 0.2),
        LinearIntervalConstraint("right", right, 0.3, 0.4),
    ]
    baseline = [
        LinearIntervalConstraint("root", root, 0.0, 1.0),
        LinearIntervalConstraint("left", left, 0.2, 0.3),
        LinearIntervalConstraint("right", right, 0.4, 0.5),
    ]

    result = project_ratio_hierarchy_confidence_region(
        candidate, baseline, {"root": root}, variable_count=2
    )

    assert result.feasible
    assert result.intervals["root"] == pytest.approx((0.5, 1.0))


def test_two_arm_difference_region_uses_common_hierarchy() -> None:
    root = np.array([0.5, 0.5])
    left = np.array([1.0, 0.0])
    right = np.array([0.0, 1.0])
    candidate = [
        LinearIntervalConstraint("root", root, 0.0, 1.0),
        LinearIntervalConstraint("left", left, 0.1, 0.2),
        LinearIntervalConstraint("right", right, 0.3, 0.4),
    ]
    baseline = [
        LinearIntervalConstraint("root", root, 0.0, 1.0),
        LinearIntervalConstraint("left", left, 0.2, 0.3),
        LinearIntervalConstraint("right", right, 0.4, 0.5),
    ]

    result = project_two_arm_difference_hierarchy_confidence_region(
        candidate, baseline, {"root": root}, variable_count=2
    )

    assert result.feasible
    assert result.intervals["root"] == pytest.approx((-0.2, 0.0))
