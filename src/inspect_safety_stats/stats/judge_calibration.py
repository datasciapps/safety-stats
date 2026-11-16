"""One judge-calibration core, shared by the fixed-design and the generative pipeline.

A validated judge calibration takes the same steps in both designs: cross human labels
against judge verdicts within each judge stratum, give every declared cohort an equal
share of the calibration budget, refuse a cohort with an empty stratum, estimate
``rho_1`` and ``rho_0`` from the labels (or refuse when they do not establish an
informative judge), and say when a stratum without labels was completed as ``[0, 1]``.
Those steps live here once. Each pipeline only supplies its rows -- an observation or a
generative sampling unit, reduced to a cohort, a label key and a verdict -- and renders
the outcome in its own report.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.misclassification import (
    JudgeCalibration,
    JudgeStratumCounts,
    completion_warning,
    estimate_judge_calibration,
)

JUDGE_STRATA = ("judge_unsafe", "judge_safe")


@dataclass(frozen=True)
class JudgeStrata:
    """Human-labelled counts in both judge strata of one cohort."""

    judge_unsafe: JudgeStratumCounts
    judge_safe: JudgeStratumCounts

    @property
    def empty(self) -> tuple[str, ...]:
        """The strata, in ``JUDGE_STRATA`` order, that hold no classified row at all."""

        return tuple(name for name in JUDGE_STRATA if getattr(self, name).stratum_size == 0)

    def summary(self) -> dict[str, dict[str, int]]:
        """The counts of both strata, as a report records them."""

        return {
            name: {
                "labelled": counts.labelled,
                "truly_unsafe": counts.truly_unsafe,
                "stratum_size": counts.stratum_size,
            }
            for name in JUDGE_STRATA
            for counts in (getattr(self, name),)
        }


NO_STRATA = JudgeStrata(JudgeStratumCounts(0, 0, 0), JudgeStratumCounts(0, 0, 0))


def count_judge_strata(
    rows: Iterable[tuple[str, str | None, bool | None]],
    labels: dict[str, bool],
) -> tuple[dict[str, JudgeStrata], list[str]]:
    """Cross human labels against judge verdicts, per cohort and judge stratum.

    Each row is ``(cohort_id, label_key, verdict)``. A row without a verdict is
    unclassified and belongs to neither stratum: the interval already completes it
    conservatively, and the correction keeps that completion (see
    ``correct_judge_risk``). A row without a label key counts towards its stratum's size
    but cannot be labelled. The label keys no counted row matched are returned sorted,
    so that a label for an unknown or unclassified row is reported, not absorbed.
    """

    sizes: dict[tuple[str, bool], int] = {}
    labelled: dict[tuple[str, bool], int] = {}
    unsafe: dict[tuple[str, bool], int] = {}
    cohorts: dict[str, None] = {}
    seen: set[str] = set()
    for cohort_id, label_key, verdict in rows:
        if verdict is None:
            continue
        cohorts.setdefault(cohort_id)
        key = (cohort_id, bool(verdict))
        sizes[key] = sizes.get(key, 0) + 1
        if label_key is None:
            continue
        human = labels.get(label_key)
        if human is None:
            continue
        seen.add(label_key)
        labelled[key] = labelled.get(key, 0) + 1
        if human:
            unsafe[key] = unsafe.get(key, 0) + 1

    def stratum(cohort_id: str, verdict: bool) -> JudgeStratumCounts:
        key = (cohort_id, verdict)
        return JudgeStratumCounts(
            labelled=labelled.get(key, 0),
            truly_unsafe=unsafe.get(key, 0),
            stratum_size=sizes.get(key, 0),
        )

    counts = {
        cohort_id: JudgeStrata(stratum(cohort_id, True), stratum(cohort_id, False))
        for cohort_id in cohorts
    }
    return counts, sorted(set(labels) - seen)


def cohort_alpha_share(alpha: float, cohorts: int) -> float:
    """The calibration budget of one cohort: an equal share of ``alpha``.

    ``rho`` is prevalence-dependent, so every declared cohort needs its own calibration.
    Giving each an equal share keeps a union over all of them within ``alpha``.
    """

    if cohorts < 1:
        raise ValueError("a judge calibration budget needs at least one declared cohort")
    return alpha / cohorts


@dataclass(frozen=True)
class CohortCalibration:
    """What a cohort's labels establish: a calibration, or why there is none.

    Exactly one of these holds: ``empty_strata`` names the strata without any classified
    row, ``refusal`` carries the estimator's reason for refusing the labels, or
    ``calibration`` is set.
    """

    strata: JudgeStrata
    calibration: JudgeCalibration | None = None
    empty_strata: tuple[str, ...] = ()
    refusal: str | None = None


def calibrate_cohort(strata: JudgeStrata, *, scenario_id: str, alpha: float) -> CohortCalibration:
    """Estimate one cohort's calibration from its strata within its share ``alpha``."""

    empty = strata.empty
    if empty:
        return CohortCalibration(strata=strata, empty_strata=empty)
    try:
        calibration = estimate_judge_calibration(
            scenario_id=scenario_id,
            unsafe_stratum=strata.judge_unsafe,
            safe_stratum=strata.judge_safe,
            alpha=alpha,
        )
    except ValueError as exc:
        return CohortCalibration(strata=strata, refusal=str(exc))
    return CohortCalibration(strata=strata, calibration=calibration)


def completion_diagnostic(calibration: JudgeCalibration, scope: str) -> Diagnostic | None:
    """The warning for strata the calibration completed as ``[0, 1]``, if it completed any."""

    warning = completion_warning(calibration)
    if warning is None:
        return None
    details: dict[str, Any] = {"completed_strata": list(calibration.completed_strata)}
    return Diagnostic(
        code="JUDGE_STRATUM_UNLABELLED_COMPLETED",
        severity="WARNING",
        message=warning,
        scope=scope,
        details=details,
    )
