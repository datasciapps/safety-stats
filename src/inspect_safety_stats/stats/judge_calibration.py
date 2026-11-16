"""One judge-calibration core, shared by the fixed-design and the generative pipeline.

A validated judge calibration takes the same steps in both designs: cross human labels
against judge verdicts within each judge stratum, give every declared cohort an equal
share of the calibration budget, refuse a cohort with an empty stratum, estimate
``rho_1`` and ``rho_0`` from the labels (or refuse when they do not establish an
informative judge), and say when a stratum without labels was completed as ``[0, 1]``.
Those steps live here once. Each pipeline only supplies its rows -- an observation or a
generative sampling unit, reduced to a cohort, a label key and a verdict -- and renders
the outcome in its own report.

The experimental stratified design (:func:`count_fine_strata`, and ``calibrate_cohort``
with ``fine``) refines each judge stratum into declared fine strata labelled without
replacement. Only the fixed-design pipeline offers it so far.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.stats.misclassification import (
    FineStratumBounds,
    JudgeCalibration,
    JudgeStratumCounts,
    completion_warning,
    estimate_judge_calibration,
    estimate_stratified_judge_calibration,
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


FineStrata = dict[str, dict[str, JudgeStratumCounts]]
"""One cohort's declared fine strata: judge stratum name, then fine stratum name."""


@dataclass(frozen=True)
class FineStrataCounts:
    """Labels crossed against judge verdicts within declared fine strata.

    ``by_cohort`` maps a cohort to its :data:`FineStrata`. ``unmatched_labels`` are label
    keys no classified row matched, as :func:`count_judge_strata` reports them.
    ``unstratified`` are classified rows the strata file does not place, and
    ``unknown_strata`` are strata-file keys that name no classified row; either means the
    file does not describe the design the labels were drawn from.
    """

    by_cohort: dict[str, FineStrata]
    unmatched_labels: list[str]
    unstratified: list[str]
    unknown_strata: list[str]

    def judge_strata(self, cohort_id: str) -> JudgeStrata:
        """The cohort's two judge strata, each the sum of its fine strata."""

        fine = self.by_cohort.get(cohort_id)
        if fine is None:
            return NO_STRATA

        def total(name: str) -> JudgeStratumCounts:
            items = fine.get(name, {}).values()
            return JudgeStratumCounts(
                labelled=sum(item.labelled for item in items),
                truly_unsafe=sum(item.truly_unsafe for item in items),
                stratum_size=sum(item.stratum_size for item in items),
            )

        return JudgeStrata(total("judge_unsafe"), total("judge_safe"))


def count_fine_strata(
    rows: Iterable[tuple[str, str | None, bool | None]],
    labels: dict[str, bool],
    strata: dict[str, str],
) -> FineStrataCounts:
    """Cross human labels against judge verdicts within declared fine strata, per cohort.

    Each row is ``(cohort_id, label_key, verdict)``, as for :func:`count_judge_strata`, and
    ``strata`` maps a label key to its declared stratum. A fine stratum is the declared
    stratum within one cohort and one judge verdict, so declared strata need not respect the
    judge's verdicts: the counting nests them. An unclassified row belongs to no stratum.
    """

    sizes: dict[tuple[str, str, str], int] = {}
    labelled: dict[tuple[str, str, str], int] = {}
    unsafe: dict[tuple[str, str, str], int] = {}
    seen_labels: set[str] = set()
    seen_strata: set[str] = set()
    unstratified: list[str] = []
    for cohort_id, label_key, verdict in rows:
        if verdict is None:
            continue
        if label_key is None or label_key not in strata:
            unstratified.append(f"{cohort_id}/{label_key}")
            continue
        declared = strata[label_key]
        seen_strata.add(label_key)
        key = (cohort_id, JUDGE_STRATA[0] if verdict else JUDGE_STRATA[1], declared)
        sizes[key] = sizes.get(key, 0) + 1
        human = labels.get(label_key)
        if human is None:
            continue
        seen_labels.add(label_key)
        labelled[key] = labelled.get(key, 0) + 1
        if human:
            unsafe[key] = unsafe.get(key, 0) + 1

    by_cohort: dict[str, FineStrata] = {}
    for key in sorted(sizes):
        cohort_id, judge_stratum, declared = key
        by_cohort.setdefault(cohort_id, {}).setdefault(judge_stratum, {})[declared] = (
            JudgeStratumCounts(
                labelled=labelled.get(key, 0),
                truly_unsafe=unsafe.get(key, 0),
                stratum_size=sizes[key],
            )
        )
    return FineStrataCounts(
        by_cohort=by_cohort,
        unmatched_labels=sorted(set(labels) - seen_labels),
        unstratified=sorted(unstratified),
        unknown_strata=sorted(set(strata) - seen_strata),
    )


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
    fine_strata: tuple[FineStratumBounds, ...] = ()


def calibrate_cohort(
    strata: JudgeStrata,
    *,
    scenario_id: str,
    alpha: float,
    fine: FineStrata | None = None,
) -> CohortCalibration:
    """Estimate one cohort's calibration from its strata within its share ``alpha``.

    With ``fine``, the declared fine strata whose sums ``strata`` holds, the labels are
    read as a stratified sample without replacement; without, as a simple random sample
    within each judge stratum.
    """

    empty = strata.empty
    if empty:
        return CohortCalibration(strata=strata, empty_strata=empty)
    try:
        if fine is None:
            calibration = estimate_judge_calibration(
                scenario_id=scenario_id,
                unsafe_stratum=strata.judge_unsafe,
                safe_stratum=strata.judge_safe,
                alpha=alpha,
            )
            return CohortCalibration(strata=strata, calibration=calibration)
        stratified = estimate_stratified_judge_calibration(
            scenario_id=scenario_id,
            unsafe_strata=fine.get("judge_unsafe", {}),
            safe_strata=fine.get("judge_safe", {}),
            alpha=alpha,
        )
    except ValueError as exc:
        return CohortCalibration(strata=strata, refusal=str(exc))
    return CohortCalibration(
        strata=strata,
        calibration=stratified.calibration,
        fine_strata=stratified.fine_strata,
    )


def fine_strata_summary(fine_strata: tuple[FineStratumBounds, ...]) -> list[dict[str, Any]]:
    """The fine strata behind a stratified calibration, as a report records them."""

    return [
        {
            "judge_stratum": item.judge_stratum,
            "stratum": item.stratum,
            "labelled": item.counts.labelled,
            "truly_unsafe": item.counts.truly_unsafe,
            "stratum_size": item.counts.stratum_size,
            "truly_unsafe_lower": item.unsafe_lower,
            "truly_unsafe_upper": item.unsafe_upper,
            "alpha": item.alpha,
        }
        for item in fine_strata
    ]


def unlabelled_fine_strata_diagnostic(
    fine_strata: tuple[FineStratumBounds, ...], scope: str
) -> Diagnostic | None:
    """The warning for fine strata completed as ``[0, N_h]``, if some judge stratum has any.

    A judge stratum with no label at all is reported by :func:`completion_diagnostic`.
    """

    labelled = {
        name: any(
            item.counts.labelled for item in fine_strata if item.judge_stratum == name
        )
        for name in JUDGE_STRATA
    }
    empty = [
        item
        for item in fine_strata
        if item.counts.labelled == 0 and labelled[item.judge_stratum]
    ]
    if not empty:
        return None
    return Diagnostic(
        code="JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED",
        severity="WARNING",
        message=(
            f"{len(empty)} declared fine strata have no human-labelled observation, so each "
            "is completed conservatively with its unsafe count anywhere from none to all of "
            "its rollouts; labelling them would narrow the correction"
        ),
        scope=scope,
        details={
            "fine_strata": [
                {
                    "judge_stratum": item.judge_stratum,
                    "stratum": item.stratum,
                    "stratum_size": item.counts.stratum_size,
                }
                for item in empty
            ]
        },
    )


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
