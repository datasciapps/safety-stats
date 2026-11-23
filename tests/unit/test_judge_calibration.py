"""The judge-calibration core that both pipelines share."""

from __future__ import annotations

import pytest

from safety_stats.stats.judge_calibration import (
    NO_STRATA,
    JudgeStrata,
    calibrate_cohort,
    cohort_alpha_share,
    completion_diagnostic,
    count_judge_strata,
)
from safety_stats.stats.misclassification import (
    JudgeStratumCounts,
    estimate_judge_calibration,
)


def test_strata_are_counted_per_cohort_and_skip_unclassified_rows() -> None:
    rows = [
        ("candidate", "a", True),
        ("candidate", "b", True),
        ("candidate", "c", False),
        ("candidate", "d", None),
        ("candidate", None, False),
        ("baseline", "e", False),
    ]
    counts, unmatched = count_judge_strata(rows, {"a": True, "c": False, "d": True, "zz": False})

    # A label on an unclassified row and a label on no row are both reported.
    assert unmatched == ["d", "zz"]
    assert counts["candidate"] == JudgeStrata(
        judge_unsafe=JudgeStratumCounts(labelled=1, truly_unsafe=1, stratum_size=2),
        judge_safe=JudgeStratumCounts(labelled=1, truly_unsafe=0, stratum_size=2),
    )
    assert counts["baseline"].empty == ("judge_unsafe",)
    assert list(counts) == ["candidate", "baseline"]


def test_a_cohort_with_only_unclassified_rows_has_no_strata() -> None:
    counts, unmatched = count_judge_strata([("c", "a", None)], {"a": True})
    assert counts == {}
    assert unmatched == ["a"]
    assert NO_STRATA.empty == ("judge_unsafe", "judge_safe")


def test_the_summary_records_both_strata() -> None:
    strata = JudgeStrata(JudgeStratumCounts(3, 2, 10), JudgeStratumCounts(4, 0, 20))
    assert strata.summary() == {
        "judge_unsafe": {"labelled": 3, "truly_unsafe": 2, "stratum_size": 10},
        "judge_safe": {"labelled": 4, "truly_unsafe": 0, "stratum_size": 20},
    }


def test_every_cohort_gets_an_equal_share_of_the_budget() -> None:
    assert cohort_alpha_share(0.06, 3) == pytest.approx(0.02, abs=0)
    with pytest.raises(ValueError, match="at least one declared cohort"):
        cohort_alpha_share(0.05, 0)


def test_an_empty_stratum_is_refused_before_estimation() -> None:
    strata = JudgeStrata(JudgeStratumCounts(0, 0, 0), JudgeStratumCounts(5, 0, 9))
    outcome = calibrate_cohort(strata, scenario_id="s", alpha=0.05)
    assert outcome.calibration is None
    assert outcome.refusal is None
    assert outcome.empty_strata == ("judge_unsafe",)


def test_an_uninformative_sample_is_refused_with_the_estimators_reason() -> None:
    strata = JudgeStrata(JudgeStratumCounts(2, 1, 10), JudgeStratumCounts(2, 1, 10))
    outcome = calibrate_cohort(strata, scenario_id="s", alpha=0.05)
    assert outcome.calibration is None
    assert outcome.empty_strata == ()
    assert outcome.refusal is not None
    assert "strictly informative" in outcome.refusal


def test_a_calibration_is_the_estimators_and_completion_is_reported() -> None:
    unsafe = JudgeStratumCounts(40, 39, 100)
    safe = JudgeStratumCounts(0, 0, 900)
    outcome = calibrate_cohort(JudgeStrata(unsafe, safe), scenario_id="s", alpha=0.05)
    assert outcome.calibration == estimate_judge_calibration(
        scenario_id="s", unsafe_stratum=unsafe, safe_stratum=safe, alpha=0.05
    )
    diagnostic = completion_diagnostic(outcome.calibration, "cohorts/candidate")
    assert diagnostic is not None
    assert diagnostic.code == "JUDGE_STRATUM_UNLABELLED_COMPLETED"
    assert diagnostic.scope == "cohorts/candidate"
    assert diagnostic.details == {"completed_strata": ["judge_safe"]}


def test_a_measured_calibration_needs_no_completion_warning() -> None:
    strata = JudgeStrata(JudgeStratumCounts(40, 39, 100), JudgeStratumCounts(40, 0, 900))
    outcome = calibrate_cohort(strata, scenario_id="s", alpha=0.05)
    assert outcome.calibration is not None
    assert completion_diagnostic(outcome.calibration, "candidate") is None
