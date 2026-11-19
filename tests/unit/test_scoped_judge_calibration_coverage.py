"""Coverage of a judge calibration scoped to a small group, from labels drawn over its cohort.

The labels are a simple random sample within each of the cohort's judge strata, of a size
fixed in advance, and only part of each stratum is labelled. A calibration scoped to a group
counts the labels that land on the group's rollouts. Given how many land in a judge stratum of
the group, those are a simple random sample of it, so the exact bounds hold for the group's
own rho_1 and rho_0. The simulation checks that through the analysis's own counting: a group
holding a tenth of the cohort, whose judge is worse than elsewhere, labelled again and again.

If the scoped counts took the cohort's labels, or another group's, the box would measure the
cohort's accuracy, which this group's is far from, and the test would fail.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspect_safety_stats.analysis.judge_calibration import _validated_judge_calibrations
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.manifest import ResolvedGroup
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import Diagnostic

PROMPTS, ROLLOUTS, SMALL = 100, 20, 10
FAMILY_ALPHA = 0.05
TRIALS = 400


def _population() -> list[tuple[str, str, bool, bool]]:
    """Each rollout's prompt, observation id, judge verdict and truth: a fixed population.

    The judge flags 30% of rollouts everywhere. In the small group 75% of its flags are true
    failures and 5% of its passes are missed failures; elsewhere 95% and 2%.
    """

    rng = np.random.default_rng(20261119)
    rows = []
    for prompt in range(PROMPTS):
        rho_1, rho_0 = (0.75, 0.05) if prompt < SMALL else (0.95, 0.02)
        for rollout in range(ROLLOUTS):
            verdict = bool(rng.random() < 0.3)
            truth = bool(rng.random() < (rho_1 if verdict else rho_0))
            rows.append((f"p{prompt:03d}", f"o{prompt:03d}-{rollout:02d}", verdict, truth))
    return rows


def _data(rows: list[tuple[str, str, bool, bool]]) -> NormalizedData:
    observations = [
        Observation(
            observation_id=observation_id,
            source="simulated",
            run_key="run",
            rollout_id=observation_id,
            item_id=prompt_id,
            repeat=1,
            prompt_id=prompt_id,
            question_id=prompt_id,
            task_id="task",
            cohort_id="candidate",
            score_name="harmful",
            score_raw="FAIL" if verdict else "PASS",
            failure=verdict,
            outcome_status="classified_failure" if verdict else "classified_safe",
            model="model",
            configuration_fingerprint="configuration",
            collection_order=order,
        )
        for order, (prompt_id, observation_id, verdict, _) in enumerate(rows)
    ]
    prompts = sorted({prompt_id for prompt_id, *_ in rows})
    groups = [
        ResolvedGroup(
            id=group_id,
            title=group_id,
            prompt_ids=tuple(members),
            weights={prompt_id: 1 / len(members) for prompt_id in members},
        )
        for group_id, members in (("all", prompts), ("small", prompts[:SMALL]))
    ]
    return NormalizedData(observations, pd.DataFrame(), groups, [], [], {})


def _plan(directory: Path) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "analysis": {"id": "scoped", "title": "Scoped", "intent": "exploratory"},
            "inputs": {
                "design_manifest": str(directory / "design.csv"),
                "score": {
                    "name": "harmful",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
                "judge_calibration": {
                    "mode": "validated",
                    "validation_manifest": str(directory / "labels.csv"),
                    "family": "judge_error",
                    "groups": ["all", "small"],
                },
            },
            "cohorts": {"candidate": {"match": {"model": "model"}}},
            "groups": [
                {"id": "all", "filter": {"suite": "s"}},
                {"id": "small", "filter": {"risk_domain": "small"}},
            ],
            "outputs": {"prompt_bounds": {"enabled": False}},
            "multiplicity": {
                "families": [
                    {
                        "id": "judge_error",
                        "purpose": "simultaneous_intervals",
                        "method": "bonferroni",
                        "alpha": FAMILY_ALPHA,
                    }
                ]
            },
            "claims": [],
        }
    )


def _rho(rows: list[tuple[str, str, bool, bool]], prompts: set[str]) -> tuple[float, float]:
    """The finite-population rho_1 and rho_0 of these prompts' rollouts."""

    def share(verdict: bool) -> float:
        truths = [
            truth for prompt, _, judged, truth in rows if judged == verdict and prompt in prompts
        ]
        return sum(truths) / len(truths)

    return share(True), share(False)


@pytest.mark.parametrize("labels_per_stratum", [30, 100, 400])
def test_a_small_group_s_calibration_covers_its_own_accuracy(
    tmp_path: Path, labels_per_stratum: int
) -> None:
    rows = _population()
    data = _data(rows)
    plan = _plan(tmp_path)
    prompts = sorted({prompt_id for prompt_id, *_ in rows})
    truth = {"all": _rho(rows, set(prompts)), "small": _rho(rows, set(prompts[:SMALL]))}
    # The small group's judge is far from the cohort's, so the cohort's box cannot pass for it.
    assert truth["all"][0] - truth["small"][0] > 0.2
    strata = {
        verdict: [observation_id for _, observation_id, judged, _ in rows if judged == verdict]
        for verdict in (True, False)
    }
    labelled_truth = {observation_id: truth for _, observation_id, _, truth in rows}
    # Every judge stratum is labelled in part only.
    assert all(labels_per_stratum < len(members) for members in strata.values())

    rng = np.random.default_rng(labels_per_stratum)
    reported = {"all": 0, "small": 0}
    misses = 0
    for _ in range(TRIALS):
        drawn = [
            observation_id
            for members in strata.values()
            for observation_id in rng.choice(members, size=labels_per_stratum, replace=False)
        ]
        (tmp_path / "labels.csv").write_text(
            "observation_id,human_label\n"
            + "".join(
                f"{item},{'unsafe' if labelled_truth[item] else 'safe'}\n" for item in drawn
            ),
            encoding="utf-8",
        )
        diagnostics: list[Diagnostic] = []
        calibrations = _validated_judge_calibrations(plan, data, diagnostics)[0]
        missed = False
        for group_id, (rho_1, rho_0) in truth.items():
            calibration = calibrations.get(("candidate", group_id))
            if calibration is None:
                # A refused calibration certifies nothing, so it cannot miss.
                continue
            reported[group_id] += 1
            # Two groups split the judge family's budget, so each box spends half of it.
            assert calibration.alpha == pytest.approx(FAMILY_ALPHA / 2, abs=0)
            missed |= not (
                calibration.rho_1_lower <= rho_1 <= calibration.rho_1_upper
                and calibration.rho_0_lower <= rho_0 <= calibration.rho_0_upper
            )
        misses += missed

    # Both boxes hold at once at the family's level. A binomial standard error above the level
    # still fails a box that undercovers.
    level = FAMILY_ALPHA
    assert misses / TRIALS <= level + 3 * math.sqrt(level * (1 - level) / TRIALS)
    # The cohort-wide box is always reported. The small group gets about a tenth of the labels,
    # so its box is often refused with few of them, but not once the cohort has many.
    assert reported["all"] == TRIALS
    if labels_per_stratum == 400:
        assert reported["small"] >= 0.95 * TRIALS
