from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inspect_safety_stats.structural.data import ModelFrame
from inspect_safety_stats.structural.pymc_backend import (
    _nested_centering_matrix,
    _prompt_centering_matrix,
)


def _frame() -> ModelFrame:
    frame = pd.DataFrame(
        {
            "prompt_id": ["p1", "p2", "p3", "p4"],
            "weight": [0.25] * 4,
            "task": ["a", "a", "b", "b"],
            "topic": ["a1", "a2", "b1", "b2"],
            "k_unsafe": [0, 1, 0, 1],
            "n": [2] * 4,
        }
    )
    return ModelFrame(
        frame=frame,
        cohort_id="candidate",
        group_id="all",
        declared_hierarchy=("task", "topic", "prompt_id"),
        hierarchy=("task", "topic"),
        prompt_frailty_levels=("prompt_id",),
        category_levels={"task": ("a", "b"), "topic": ("a1", "a2", "b1", "b2")},
        category_codes={
            "task": np.asarray([0, 0, 1, 1]),
            "topic": np.asarray([0, 1, 2, 3]),
        },
        empirical_report_path=Path("report.json"),
        empirical_report_sha256="0" * 64,
        empirical_weighted_mean=0.25,
    )


def test_top_level_effects_are_globally_centered() -> None:
    matrix = _nested_centering_matrix(_frame(), 0)
    transformed = matrix @ np.asarray([2.0, 4.0])
    assert transformed.tolist() == [-1.0, 1.0]


def test_child_effects_are_centered_within_parent() -> None:
    matrix = _nested_centering_matrix(_frame(), 1)
    transformed = matrix @ np.asarray([1.0, 3.0, 10.0, 14.0])
    assert transformed.tolist() == [-1.0, 1.0, -2.0, 2.0]


def test_prompt_frailties_are_centered_within_lowest_hierarchy_level() -> None:
    matrix = _prompt_centering_matrix(_frame())
    transformed = matrix @ np.asarray([1.0, 3.0, 10.0, 14.0])
    assert transformed.tolist() == [0.0, 0.0, 0.0, 0.0]
