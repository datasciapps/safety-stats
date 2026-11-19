"""A plan's labelling protocol: the queue, its order and the stopping rule (roadmap F41, L50).

One prompt, 400 rollouts of one cohort: 80 judge-unsafe, of which 60 are truly unsafe, and
320 judge-safe, of which 5 are truly unsafe. The queue alternates the two judge strata, each
in a fixed shuffled order, and puts a row of an `oversample` tier after every fourth
calibration row. The labels are the queue's first calibration rows unless a test breaks that.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.config.loader import canonical_plan_hash
from inspect_safety_stats.schema.reports import Diagnostic, RiskIntervalResult, SafetyReport
from inspect_safety_stats.schema.scopes import cohort_scope
from inspect_safety_stats.stats import misclassification
from inspect_safety_stats.stats.misclassification import JudgeStratumCounts

ROLLOUTS = 400
JUDGE_UNSAFE = 80
JUDGE = """  judge_calibration:
    mode: validated
    validation_manifest: labels.csv
    family: judge_error
"""

PLAN = """
schema_version: "2.0"
analysis: {id: protocol, title: Labelling protocol, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
@JUDGE@
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: true
  require_expected_rollouts: true
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
cohorts:
  candidate: {match: {model: model/candidate}, require_uniform: [task_version]}
groups:
  - id: all
    filter: {suite: s}
outputs:
  prompt_bounds: {enabled: true, simultaneous: true, family: displayed,
                  alpha_allocation: equal_bonferroni}
  direct_mean_intervals:
    enabled: true
    method: weighted_kl_chernoff
    family: direct_means
    groups: [all]
    simultaneous: true
    coverage_mode: two_sided
    independence: {primary: rollout}
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: direct_means, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.25, unacceptable_lower: 0.50}
policy: {zero_tolerance: {enabled: true, manifest_field: zero_tolerance}}
report: {include_observations: true, decimal_places: 6}
"""


ABSENT = "not-a-rollout-of-the-run"


def _truly_unsafe(epoch: int) -> bool:
    return epoch <= 60 or JUDGE_UNSAFE < epoch <= JUDGE_UNSAFE + 5


def _patch_logs(monkeypatch: pytest.MonkeyPatch, unscored: int | None = None) -> None:
    """Serve the run's logs; the rollout of epoch ``unscored``, if any, has no score."""

    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-candidate",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "fixed_safety",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": ROLLOUTS,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"candidate-{epoch}",
                "eval_id": "eval-candidate",
                "log": "/not/a/local/log.eval",
                "id": "p1",
                "epoch": epoch,
                "score_harmful": (
                    None if epoch == unscored else "FAIL" if epoch <= JUDGE_UNSAFE else "PASS"
                ),
                "error": None,
                "retries": 0,
            }
            for epoch in range(1, ROLLOUTS + 1)
        ]
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, **_: samples,
    )


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_logs(monkeypatch)


def _observation_ids(directory: Path) -> dict[int, str]:
    """Each epoch's observation identifier, as an analysis without a protocol reports it."""

    plan = _write_plan(directory / "probe", None)
    report = analyze(plan=load_plan(plan), logs="ignored")
    return {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in report._observations
    }


def _queue(ids: dict[int, str], complete: bool = False) -> pd.DataFrame:
    """The queue: 30 rows of each judge stratum, or with ``complete`` every rollout of both."""

    rng = np.random.default_rng(41)
    unsafe = [ids[e] for e in rng.permutation(np.arange(1, JUDGE_UNSAFE + 1))]
    safe = [ids[e] for e in rng.permutation(np.arange(JUDGE_UNSAFE + 1, ROLLOUTS + 1))]
    rows: list[dict[str, str]] = []
    if complete:
        # The same two orders, alternating until the judge-unsafe stratum runs out.
        for slot in range(2 * JUDGE_UNSAFE):
            rows.append({"observation_id": (unsafe if slot % 2 else safe)[slot // 2],
                         "tier": "calibration"})
        rows += [{"observation_id": key, "tier": "calibration"} for key in safe[JUDGE_UNSAFE:]]
        return pd.DataFrame(rows)
    # Calibration rows take each stratum's order from the front, oversample rows from the
    # back, so the tiers never meet.
    for slot in range(60):
        rows.append({"observation_id": (unsafe if slot % 2 else safe)[slot // 2],
                     "tier": "calibration"})
        if slot % 4 == 3:
            rows.append({"observation_id": safe[-1 - slot // 4], "tier": "oversample"})
    return pd.DataFrame(rows)


def _write_plan(directory: Path, protocol: str | None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts,zero_tolerance\n"
        f"p1,q1,t1,s,{ROLLOUTS},false\n",
        encoding="utf-8",
    )
    # Without a protocol the plan has no calibration either: it only finds the identifiers.
    text = PLAN.replace("@JUDGE@\n", "" if protocol is None else JUDGE + protocol)
    plan = directory / "plan.yaml"
    plan.write_text(text.strip() + "\n", encoding="utf-8")
    return plan


def _labels(directory: Path, ids: dict[int, str], observation_ids: list[str]) -> None:
    truth = {value: _truly_unsafe(epoch) for epoch, value in ids.items()}
    # A key the run does not have is labelled safe.
    rows = ["observation_id,human_label"] + [
        f"{key},{'unsafe' if truth.get(key) else 'safe'}" for key in observation_ids
    ]
    (directory / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _protocol(stopping: str, *, queue: str = "queue.csv", **extra: object) -> str:
    lines = [
        "    protocol:",
        f"      queue: {queue}",
        "      tier: calibration",
        "      seed: 41",
        f"      stopping: {stopping}",
        *(f"      {key}: {value}" for key, value in extra.items()),
    ]
    return "\n".join(lines) + "\n"


def _run(
    tmp_path: Path,
    protocol: str,
    take: int = 40,
    edit: str = "none",
    queue_file: str = "queue.csv",
    given: pd.DataFrame | None = None,
) -> tuple[SafetyReport, pd.DataFrame]:
    """Analyse with the first ``take`` calibration rows of the queue labelled.

    The queue is ``given``, or else :func:`_queue`'s, changed as ``edit`` says.
    """

    ids = _observation_ids(tmp_path)
    queue = _queue(ids, complete=edit == "complete") if given is None else given
    if edit in ("absent", "absent_labelled"):
        # A rollout the run does not have, first in the tier and labelled, or right after
        # the labelled rows.
        at = 0
        if edit == "absent":
            at = int(queue.index[queue["tier"] == "calibration"][take - 1]) + 1
        stray = pd.DataFrame([{"observation_id": ABSENT, "tier": "calibration"}])
        queue = pd.concat([queue.iloc[:at], stray, queue.iloc[at:]], ignore_index=True)
    # pandas compresses the queue as its name says, as for queue.csv.gz.
    queue.to_csv(tmp_path / queue_file, index=False, lineterminator="\n")
    calibration = list(queue.loc[queue["tier"] == "calibration", "observation_id"])
    labelled = calibration[:take]
    if edit == "skip":
        labelled = calibration[: take - 2] + calibration[take - 1 : take + 1]
    elif edit == "oversample":
        labelled = labelled + [queue.loc[queue["tier"] == "oversample", "observation_id"].iloc[0]]
    _labels(tmp_path, ids, labelled)
    plan = _write_plan(tmp_path, protocol)
    return analyze(plan=load_plan(plan), logs="ignored"), queue


def _corrected(report: SafetyReport) -> RiskIntervalResult | None:
    mean = next(row for row in report.group_risks if row.measure_id == "mean")
    return next(
        (item for item in mean.intervals if item.interval_id == "judge_corrected_validated"),
        None,
    )


QUEUE_ASSUMPTION = (
    "The declared labelling queue lists, within each judge stratum, all of its classified "
    "rollouts or a simple random sample of them, in a uniformly random order fixed before "
    "labelling; the analysis checks that the labels are its first rows and that it lists only "
    "classified rollouts of the run, not how its rows were drawn."
)


def _partial(report: SafetyReport) -> list[Diagnostic]:
    return [
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_LABELLING_QUEUE_PARTIAL"
    ]


def _violations(report: SafetyReport) -> list[str]:
    return [
        item.message
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_LABELLING_PROTOCOL_VIOLATED"
    ]


@pytest.mark.usefixtures("logs")
@pytest.mark.parametrize("queue_file", ["queue.csv", "queue.csv.gz"])
def test_a_labelled_prefix_is_recorded_and_keeps_clopper_pearson(
    tmp_path: Path, queue_file: str
) -> None:
    # A compressed queue is read as pandas reads its path, and hashed as stored.
    report, _ = _run(
        tmp_path, _protocol("label_independent", queue=queue_file), queue_file=queue_file
    )

    assert _violations(report) == []
    record = report.provenance["judge_calibration"]["protocol"]
    queue_bytes = (tmp_path / queue_file).read_bytes()
    assert record == {
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "queue_rows": 60,
        "labelled_prefix": 40,
        "tier": "calibration",
        "seed": 41,
        "stopping": "label_independent",
        "size": None,
        "bound": "clopper_pearson",
    }
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    assert any("exact Clopper-Pearson" in text for text in corrected.assumptions)


@pytest.mark.usefixtures("logs")
@pytest.mark.parametrize("scoped", [False, True])
def test_a_queue_that_lists_part_of_a_stratum_is_a_warning(
    tmp_path: Path, scoped: bool
) -> None:
    # The calibration tier lists 30 of the 80 judge-unsafe and 30 of the 320 judge-safe
    # rollouts, so that those rows are a random sample of each stratum is only declared.
    groups = "    groups: [all]\n" if scoped else ""
    report, _ = _run(tmp_path, groups + _protocol("label_independent"))

    (found,) = _partial(report)
    population = "group 'all' in cohort 'candidate'" if scoped else "cohort 'candidate'"
    assert found.severity == "WARNING"
    assert found.message == (
        "the labelling queue tier 'calibration' lists 30 of the 80 classified rollouts of the "
        f"judge-unsafe stratum and 30 of the 320 of the judge-safe stratum of {population}; "
        "the labels are a random sample of such a stratum only if the queue's rows of it were "
        "drawn from all of its rollouts at random before labelling, which the analysis "
        "cannot check"
    )
    assert found.scope == (
        cohort_scope("candidate", "groups", "all") if scoped else cohort_scope("candidate")
    )
    assert found.details == {
        "judge_strata": [
            {"judge_stratum": "judge_unsafe", "queue_rows": 30, "labelled": 20, "stratum_size": 80},
            {"judge_stratum": "judge_safe", "queue_rows": 30, "labelled": 20, "stratum_size": 320},
        ]
    }
    # Recorded: the correction is still reported, and states what the queue must be.
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    assert QUEUE_ASSUMPTION in corrected.assumptions


@pytest.mark.usefixtures("logs")
def test_a_queue_of_every_classified_rollout_gives_no_warning(tmp_path: Path) -> None:
    report, queue = _run(tmp_path, _protocol("label_independent"), edit="complete")

    assert len(queue) == ROLLOUTS
    assert _partial(report) == []
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    assert QUEUE_ASSUMPTION in corrected.assumptions


@pytest.mark.usefixtures("logs")
def test_a_stratified_design_states_the_queue_within_its_fine_strata(tmp_path: Path) -> None:
    # Two fine strata within each judge stratum: the first and second half of its rollouts.
    ids = _observation_ids(tmp_path / "ids")
    rows = ["observation_id,stratum"] + [
        f"{key},{'u' if epoch <= JUDGE_UNSAFE else 's'}{int(epoch % 2 == 0)}"
        for epoch, key in sorted(ids.items())
    ]
    tmp_path.joinpath("strata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    design = (
        "    sampling: stratified_within_judge_stratum\n    validation_strata: strata.csv\n"
    )
    report, _ = _run(
        tmp_path, design + _protocol("label_independent"), take=200, edit="complete"
    )

    corrected = _corrected(report)
    assert corrected is not None
    assert QUEUE_ASSUMPTION.replace("each judge stratum", "each declared fine stratum") in (
        corrected.assumptions
    )
    assert QUEUE_ASSUMPTION not in corrected.assumptions
    assert _partial(report) == []


STRATIFIED = "    sampling: stratified_within_judge_stratum\n    validation_strata: strata.csv\n"
SMALL = 10


def _fine_stratum(epoch: int) -> str:
    """The judge-unsafe rollouts split by parity; the last ``SMALL`` judge-safe ones, all truly
    safe, apart from the others, so that leaving them unlabelled keeps the box informative."""

    if epoch <= JUDGE_UNSAFE:
        return f"u{epoch % 2}"
    return "s1" if epoch > ROLLOUTS - SMALL else "s0"


def _stratified_queue(directory: Path, listed: dict[str, int]) -> pd.DataFrame:
    """Write the strata file, and return a queue of ``listed[h]`` rollouts of each fine stratum
    ``h``, drawn at random from it, all in one random order."""

    ids = _observation_ids(directory / "ids")
    rows = ["observation_id,stratum"] + [
        f"{key},{_fine_stratum(epoch)}" for epoch, key in sorted(ids.items())
    ]
    directory.joinpath("strata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rng = np.random.default_rng(7)
    chosen: list[int] = []
    for stratum, count in sorted(listed.items()):
        members = [epoch for epoch in sorted(ids) if _fine_stratum(epoch) == stratum]
        chosen += [int(epoch) for epoch in rng.permutation(members)[:count]]
    order = [ids[int(epoch)] for epoch in rng.permutation(chosen)]
    return pd.DataFrame({"observation_id": order, "tier": "calibration"})


@pytest.mark.usefixtures("logs")
def test_a_stratified_design_ignores_the_queue_rows_of_an_unlabelled_fine_stratum(
    tmp_path: Path,
) -> None:
    # The queue lists every rollout of three fine strata and none of the fourth, s1, so it
    # lists 310 of the 320 judge-safe rollouts. s1 has no labels and is completed with
    # certainty, so it rests on no sample: the labels are a random sample of every fine
    # stratum that has any, and the judge stratum is not the design's sampling unit.
    queue = _stratified_queue(tmp_path, {"u0": 40, "u1": 40, "s0": 310})
    report, _ = _run(
        tmp_path, STRATIFIED + _protocol("label_independent"), take=120, given=queue
    )

    assert _violations(report) == []
    (unlabelled,) = [
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED"
    ]
    assert unlabelled.details["fine_strata"] == [
        {"judge_stratum": "judge_safe", "stratum": "s1", "stratum_size": SMALL}
    ]
    assert _partial(report) == []
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"


@pytest.mark.usefixtures("logs")
def test_a_stratified_design_warns_of_a_labelled_fine_stratum_the_queue_lists_in_part(
    tmp_path: Path,
) -> None:
    # Every rollout of u0, u1 and s1 is listed, and half of s0's.
    queue = _stratified_queue(tmp_path, {"u0": 40, "u1": 40, "s0": 155, "s1": SMALL})
    report, _ = _run(
        tmp_path, STRATIFIED + _protocol("label_independent"), take=120, given=queue
    )

    ids = _observation_ids(tmp_path / "ids")
    stratum_of = {key: _fine_stratum(epoch) for epoch, key in ids.items()}
    labelled = sum(stratum_of[key] == "s0" for key in queue["observation_id"][:120])
    (found,) = _partial(report)
    assert found.message == (
        "the labelling queue tier 'calibration' lists 155 of the 310 classified rollouts of "
        "declared stratum 's0' in the judge-safe stratum of cohort 'candidate'; the labels are "
        "a random sample of such a stratum only if the queue's rows of it were drawn from all "
        "of its rollouts at random before labelling, which the analysis cannot check"
    )
    assert found.details == {
        "fine_strata": [
            {
                "judge_stratum": "judge_safe",
                "stratum": "s0",
                "queue_rows": 155,
                "labelled": labelled,
                "stratum_size": ROLLOUTS - JUDGE_UNSAFE - SMALL,
            }
        ]
    }
    assert labelled > 0
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    assert QUEUE_ASSUMPTION.replace("each judge stratum", "each declared fine stratum") in (
        corrected.assumptions
    )


@pytest.mark.usefixtures("logs")
def test_a_skipped_queue_row_stops_the_calibration(tmp_path: Path) -> None:
    report, queue = _run(tmp_path, _protocol("label_independent"), edit="skip")

    (message,) = _violations(report)
    skipped = queue.loc[queue["tier"] == "calibration", "observation_id"].iloc[38]
    assert f"row 39 ({skipped!r}) has no label, but row 41 does" in message
    assert _corrected(report) is None
    assert "protocol" not in report.provenance.get("judge_calibration", {})


@pytest.mark.usefixtures("logs")
def test_a_label_outside_the_declared_tier_stops_the_calibration(tmp_path: Path) -> None:
    report, _ = _run(tmp_path, _protocol("label_independent"), edit="oversample")

    (message,) = _violations(report)
    assert "1 labelled observations are not in the labelling queue tier 'calibration'" in (
        message
    )
    assert _corrected(report) is None


@pytest.mark.usefixtures("logs")
def test_a_queue_row_the_run_does_not_have_stops_the_calibration(tmp_path: Path) -> None:
    # The labels are a prefix of the tier, but the tier lists a rollout of some other run,
    # so it is not a random order of this run's judge strata.
    report, _ = _run(tmp_path, _protocol("label_independent"), edit="absent")

    (message,) = _violations(report)
    assert message == (
        "1 rows of the labelling queue tier 'calibration' (0 of them labelled) name "
        "observations absent from the analysed logs or with unclassified outcomes, the first "
        f"at row 41 ({ABSENT!r}); the queue must order classified rollouts of the analysed "
        "run only"
    )
    (found,) = [
        item
        for item in report.integrity.diagnostics
        if item.code == "JUDGE_LABELLING_PROTOCOL_VIOLATED"
    ]
    assert found.details == {"observation_ids": [ABSENT]}
    assert report.integrity.status == "invalid"
    assert _corrected(report) is None
    assert "protocol" not in report.provenance.get("judge_calibration", {})


@pytest.mark.usefixtures("logs")
def test_a_labelled_queue_row_the_run_does_not_have_is_one_error(tmp_path: Path) -> None:
    # The label for the stray row is a protocol violation, not a second error for the same
    # row, and a fixed size never counts it.
    report, _ = _run(tmp_path, _protocol("fixed_size", size=40), edit="absent_labelled")

    errors = [item.code for item in report.integrity.diagnostics if item.severity == "ERROR"]
    assert errors == ["JUDGE_LABELLING_PROTOCOL_VIOLATED"]
    (message,) = _violations(report)
    assert "(1 of them labelled)" in message and f"the first at row 1 ({ABSENT!r})" in message
    assert _corrected(report) is None


def test_an_unclassified_queue_row_stops_the_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rollout after the labelled rows has no score, so it is in neither judge stratum.
    _patch_logs(monkeypatch)
    ids = _observation_ids(tmp_path / "ids")
    row = _queue(ids).query("tier == 'calibration'")["observation_id"].iloc[40]
    _patch_logs(monkeypatch, unscored=next(e for e, key in ids.items() if key == row))

    report, _ = _run(tmp_path, _protocol("label_independent"))

    (message,) = _violations(report)
    assert "(0 of them labelled)" in message and f"the first at row 41 ({row!r})" in message
    assert _corrected(report) is None


@pytest.mark.usefixtures("logs")
@pytest.mark.parametrize(("size", "accepted"), [(40, True), (50, False)])
def test_a_fixed_size_must_be_the_number_labelled(
    tmp_path: Path, size: int, accepted: bool
) -> None:
    report, _ = _run(tmp_path, _protocol("fixed_size", size=size))

    if accepted:
        assert _violations(report) == []
        assert _corrected(report) is not None
    else:
        assert _violations(report) == [
            "the labelling protocol fixes 50 labels in advance, but the validation manifest "
            "has 40"
        ]
        assert _corrected(report) is None


@pytest.mark.usefixtures("logs")
def test_data_dependent_stopping_bounds_each_stratum_with_the_confidence_sequence(
    tmp_path: Path,
) -> None:
    report, _ = _run(tmp_path, _protocol("data_dependent"), take=60)

    summary = report.provenance["judge_calibration"]
    assert summary["protocol"]["bound"] == "confidence_sequence"
    strata = summary["cohorts"]["candidate"]
    endpoint_alpha = summary["per_cohort_alpha"] / 4
    expected = [
        misclassification.judge_accuracy_confidence_sequence(
            JudgeStratumCounts(
                strata[name]["labelled"],
                strata[name]["truly_unsafe"],
                strata[name]["stratum_size"],
            ),
            alpha=endpoint_alpha,
        )
        for name in ("judge_unsafe", "judge_safe")
    ]
    corrected = _corrected(report)
    assert corrected is not None and corrected.eligibility == "certifying"
    text = "\n".join(corrected.assumptions)
    rho_1, rho_0 = expected
    assert f"[{rho_1[0]:.6g}, {rho_1[1]:.6g}]" in text
    assert f"[{rho_0[0]:.6g}, {rho_0[1]:.6g}]" in text
    assert "anytime-valid binomial confidence sequence" in text
    assert "labelling may have stopped because of the labels seen" in text
    # The sequence bounds the judge's accuracy, which the stratum's rollouts only sample, not
    # the share of them that is truly unsafe.
    assert "even a census leaves the probability uncertain" in text
    assert "hypergeometric" not in text
    assert "Clopper-Pearson" not in text


@pytest.mark.usefixtures("logs")
def test_the_queue_path_is_plan_relative_in_the_report_and_the_hash(tmp_path: Path) -> None:
    report, _ = _run(tmp_path / "a", _protocol("label_independent"))

    assert report._resolved_plan["inputs"]["judge_calibration"]["protocol"]["queue"] == "queue.csv"
    plan = tmp_path / "a" / "plan.yaml"
    moved = tmp_path / "b" / "c"
    moved.mkdir(parents=True)
    (moved / "plan.yaml").write_bytes(plan.read_bytes())
    assert canonical_plan_hash(load_plan(plan)) == canonical_plan_hash(
        load_plan(moved / "plan.yaml")
    )
