from __future__ import annotations

import itertools
import math
import random

import pytest

from inspect_safety_stats.stats.tree_calibration import (
    IndependentFrontier,
    any_overlap,
    calibrate_independent_frontiers,
    sidak_local_alpha,
)


def test_sidak_local_alpha_has_exact_frontier_error() -> None:
    frontier_alpha = 0.0125
    local_alpha = sidak_local_alpha(frontier_alpha, 17)

    assert 1 - (1 - local_alpha) ** 17 == pytest.approx(frontier_alpha)
    assert local_alpha > frontier_alpha / 17


def test_frontier_calibration_uses_aliases_without_spending_alpha_twice() -> None:
    frontiers = [
        IndependentFrontier(
            frontier_id="root+task",
            targets={"root": frozenset({"p1", "p2"})},
            aliases={"root": ("root", "only-task")},
        ),
        IndependentFrontier(
            frontier_id="question",
            targets={
                "q1": frozenset({"p1"}),
                "q2": frozenset({"p2"}),
            },
            aliases={"q1": ("q1",), "q2": ("q2",)},
        ),
    ]

    result = calibrate_independent_frontiers(frontiers, directional_family_alpha=0.025)

    assert result["root"].endpoint_alpha == pytest.approx(0.0125)
    assert result["only-task"].endpoint_alpha == pytest.approx(0.0125)
    expected_question_alpha = 1 - math.sqrt(1 - 0.0125)
    assert result["q1"].endpoint_alpha == pytest.approx(expected_question_alpha)
    assert result["q2"].endpoint_alpha == pytest.approx(expected_question_alpha)
    assert result["q1"].frontier_size == 2


def test_frontier_calibration_rejects_overlapping_targets() -> None:
    frontier = IndependentFrontier(
        frontier_id="invalid",
        targets={
            "left": frozenset({"p1", "p2"}),
            "right": frozenset({"p2", "p3"}),
        },
        aliases={},
    )

    with pytest.raises(ValueError, match="not disjoint"):
        calibrate_independent_frontiers([frontier], directional_family_alpha=0.025)


def test_frontier_calibration_accepts_prespecified_frontier_weights() -> None:
    frontiers = [
        IndependentFrontier("primary", {"a": frozenset({"p1"})}, {}),
        IndependentFrontier("secondary", {"b": frozenset({"p2"})}, {}),
    ]
    result = calibrate_independent_frontiers(
        frontiers,
        directional_family_alpha=0.03,
        frontier_weights={"primary": 2.0, "secondary": 1.0},
    )

    assert result["a"].frontier_alpha == pytest.approx(0.02)
    assert result["b"].frontier_alpha == pytest.approx(0.01)


def test_tree_wide_error_is_bounded_for_arbitrarily_dependent_frontiers() -> None:
    """Exhaust all cross-frontier couplings compatible with frontier error bounds.

    Each Boolean denotes whether a frontier has at least one failed endpoint. The
    union bound across the two (possibly dependent) frontiers is the only step
    needed after exact Sidak calibration inside each frontier.
    """

    directional_alpha = 0.025
    frontier_alpha = directional_alpha / 2
    for joint_mass in (0.0, frontier_alpha / 2, frontier_alpha):
        probabilities = {
            (False, False): 1 - 2 * frontier_alpha + joint_mass,
            (True, False): frontier_alpha - joint_mass,
            (False, True): frontier_alpha - joint_mass,
            (True, True): joint_mass,
        }
        assert all(
            probabilities[state] >= 0 for state in itertools.product([False, True], repeat=2)
        )
        any_failure = sum(mass for state, mass in probabilities.items() if any(state))
        assert any_failure <= directional_alpha + 1e-15


@pytest.mark.parametrize("frontier_alpha,target_count", [(0.0, 2), (1.0, 2), (0.1, 0)])
def test_sidak_local_alpha_rejects_invalid_arguments(
    frontier_alpha: float, target_count: int
) -> None:
    with pytest.raises(ValueError):
        sidak_local_alpha(frontier_alpha, target_count)


def test_any_overlap_agrees_with_the_pairwise_check() -> None:
    rng = random.Random(20261020)
    for _ in range(2000):
        sets = [
            frozenset(rng.sample(range(40), rng.randint(0, 6))) for _ in range(rng.randint(0, 9))
        ]
        pairwise = any(left & right for left, right in itertools.combinations(sets, 2))
        assert any_overlap(sets) is pairwise, sets
