"""The JUDGE_FLIP_RATE diagnostic on the fixed path (#122), end to end through ``analyze``.

One prompt, 200 rollouts; the judge calls epochs 1-40 unsafe. A fake rescoring judge flips
a known set of those under each of four wrappers, and the manifest records its verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from inspect_safety_stats import analyze, load_plan
from inspect_safety_stats.config.loader import canonical_plan_hash, report_plan
from inspect_safety_stats.schema.reports import Diagnostic, SafetyReport

ROLLOUTS = 200
POSITIVES = 40
WRAPPERS = ("refusal_prefix", "benign_padding", "educational", "story")

PLAN = """
schema_version: "2.0"
analysis: {id: flips, title: Judge flips, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
@FLIP@
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
  risk_measures:
    - {id: mean, type: weighted_mean}
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.40, unacceptable_lower: 0.60}
report: {include_observations: true, decimal_places: 6}
"""


def _flip_section(warn_above: float, sampling: str = "all") -> str:
    return (
        "  judge_flip:\n"
        "    manifest: flips.csv\n"
        f"    wrappers: [{', '.join(WRAPPERS)}]\n"
        f"    sampling: {sampling}\n"
        f"    warn_above: {warn_above}\n"
        "    alpha: 0.05\n"
    )


def _flips(epoch: int) -> set[str]:
    """The wrappers the fake rescoring judge is talked out of an unsafe verdict by."""

    if epoch <= 10:
        return {"story"}
    if epoch <= 14:
        return {"educational", "story"}
    return set()


@pytest.fixture(autouse=True)
def fake_logs(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "score_harmful": "FAIL" if epoch <= POSITIVES else "PASS",
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
        lambda logs, parallel=False: samples,
    )


def _write_plan(directory: Path, flip: str, *, counting: bool = False) -> Path:
    """The plan; ``counting`` counts sample errors and missing scores as failures."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts\n"
        f"p1,q1,t1,s,{ROLLOUTS}\n",
        encoding="utf-8",
    )
    text = PLAN.replace("@FLIP@\n", flip)
    if counting:
        text = text.replace("missing: conservative_bounds", "missing: count_as_failure")
        text = text.replace("errors: block", "errors: count_as_failure")
    plan = directory / "plan.yaml"
    plan.write_text(text.strip() + "\n", encoding="utf-8")
    return plan


@pytest.fixture
def positives(tmp_path: Path) -> dict[int, str]:
    """Each judge-positive epoch's observation identifier, from an analysis without flips."""

    report = analyze(plan=load_plan(_write_plan(tmp_path / "plain", "")), logs="ignored")
    return {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in report._observations
        if item.failure is True
    }


def _write_flips(directory: Path, positives: dict[int, str], epochs: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "flips.csv").write_text(
        "observation_id,wrapper,verdict\n"
        + "".join(
            f"{positives[epoch]},{wrapper},{'safe' if wrapper in _flips(epoch) else 'unsafe'}\n"
            for epoch in epochs
            for wrapper in WRAPPERS
        ),
        encoding="utf-8",
    )


def _analyze(directory: Path, flip: str, *, counting: bool = False) -> SafetyReport:
    plan = _write_plan(directory, flip, counting=counting)
    return analyze(plan=load_plan(plan), logs="ignored")


def _only(report: SafetyReport, code: str) -> Diagnostic:
    found = [item for item in report.integrity.diagnostics if item.code == code]
    assert len(found) == 1
    return found[0]


def test_the_diagnostic_counts_flips_and_changes_no_claim(
    tmp_path: Path, positives: dict[int, str]
) -> None:
    assert len(positives) == POSITIVES
    plain = _analyze(tmp_path / "plain", "")
    _write_flips(tmp_path / "flips", positives, sorted(positives))

    report = _analyze(tmp_path / "flips", _flip_section(warn_above=0.5))

    diagnostic = _only(report, "JUDGE_FLIP_RATE")
    assert diagnostic.severity == "INFO"
    assert diagnostic.scope == "plan/inputs/judge_flip"
    details = diagnostic.details
    assert details["manifest"] == "flips.csv"
    assert details["judge_positive"] == POSITIVES
    assert details["any_wrapper"]["flipped"] == 14
    assert details["any_wrapper"]["rescored"] == POSITIVES
    assert details["any_wrapper"]["estimate"] == pytest.approx(14 / 40, rel=1e-12, abs=0)
    assert 0 < details["any_wrapper"]["lower"] < 14 / 40 < details["any_wrapper"]["upper"] < 1
    assert details["by_wrapper"]["story"]["flipped"] == 14
    assert details["by_wrapper"]["educational"]["flipped"] == 4
    assert details["by_wrapper"]["refusal_prefix"]["flipped"] == 0
    assert details["method"] == "clopper_pearson_two_sided"
    # Diagnostic only: integrity, every claim and the decision are as without the flip rate.
    assert report.integrity.status == plain.integrity.status
    assert [c.model_dump() for c in report.claims] == [c.model_dump() for c in plain.claims]
    assert report.decision == plain.decision


def test_a_rate_above_the_threshold_warns_without_degrading(
    tmp_path: Path, positives: dict[int, str]
) -> None:
    plain = _analyze(tmp_path / "plain", "")
    _write_flips(tmp_path / "flips", positives, sorted(positives))

    report = _analyze(tmp_path / "flips", _flip_section(warn_above=0.3))

    diagnostic = _only(report, "JUDGE_FLIP_RATE")
    assert diagnostic.severity == "WARNING"
    assert "above the declared warn_above 0.3" in diagnostic.message
    assert report.integrity.status == plain.integrity.status
    assert [c.status for c in report.claims] == [c.status for c in plain.claims]


def test_a_simple_random_sample_is_rated_over_the_rescored_completions(
    tmp_path: Path, positives: dict[int, str]
) -> None:
    _write_flips(tmp_path / "flips", positives, list(range(1, 21)))

    report = _analyze(tmp_path / "flips", _flip_section(0.5, sampling="simple_random"))

    details = _only(report, "JUDGE_FLIP_RATE").details
    assert details["judge_positive"] == POSITIVES
    assert (details["any_wrapper"]["flipped"], details["any_wrapper"]["rescored"]) == (14, 20)


def test_an_incomplete_census_is_an_error(tmp_path: Path, positives: dict[int, str]) -> None:
    _write_flips(tmp_path / "flips", positives, list(range(1, 21)))

    report = _analyze(tmp_path / "flips", _flip_section(0.5))

    error = _only(report, "JUDGE_FLIP_INVALID")
    assert error.severity == "ERROR"
    assert "20 judge-positive completions were not rescored" in error.message
    assert report.integrity.status == "invalid"
    assert not [item for item in report.integrity.diagnostics if item.code == "JUDGE_FLIP_RATE"]


def test_a_missing_manifest_is_an_error(tmp_path: Path) -> None:
    report = _analyze(tmp_path / "flips", _flip_section(0.5))

    assert _only(report, "JUDGE_FLIP_INVALID").severity == "ERROR"
    assert report.integrity.status == "invalid"


def test_the_written_report_carries_the_diagnostic(
    tmp_path: Path, positives: dict[int, str]
) -> None:
    _write_flips(tmp_path / "flips", positives, sorted(positives))
    report = _analyze(tmp_path / "flips", _flip_section(0.3))

    report.write(tmp_path / "out")

    written = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    codes = [item["code"] for item in written["integrity"]["diagnostics"]]
    assert "JUDGE_FLIP_RATE" in codes
    assert "JUDGE_FLIP_RATE" in (tmp_path / "out" / "report.html").read_text(encoding="utf-8")
    markdown = (tmp_path / "out" / "report-full.md").read_text(encoding="utf-8")
    assert "14 of 40 rescored judge-positive completions, from 1 prompt, flip" in markdown


def test_the_manifest_path_is_plan_relative_in_the_hash_and_the_report(tmp_path: Path) -> None:
    first = load_plan(_write_plan(tmp_path / "one", _flip_section(0.5)))
    second = load_plan(_write_plan(tmp_path / "two" / "deeper", _flip_section(0.5)))

    assert first.inputs.judge_flip is not None
    assert first.inputs.judge_flip.manifest.is_absolute()
    assert canonical_plan_hash(first) == canonical_plan_hash(second)
    assert report_plan(first)["inputs"]["judge_flip"]["manifest"] == "flips.csv"


def test_a_plan_without_the_section_leaves_it_out() -> None:
    from inspect_safety_stats.config.models import InputsSpec

    inputs = InputsSpec.model_validate(
        {
            "design_manifest": "design.csv",
            "score": {
                "name": "harmful",
                "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
            },
        }
    )

    assert "judge_flip" not in inputs.model_dump(mode="json")



# Epochs whose failure a plan policy, not the judge, classifies: a sample error and a missing
# score, each counted as a failure under ``counting``.
POLICY_FAILURES = {41: "error", 42: "error", 43: "missing", 44: "missing"}


@pytest.fixture
def failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[int, str]:
    """Logs with the policy failures added; each failing epoch's observation identifier."""

    import inspect_safety_stats.ingest.inspect_logs as inspect_logs

    samples = inspect_logs.samples_df("ignored").copy()
    for epoch, kind in POLICY_FAILURES.items():
        row = samples["epoch"] == epoch
        samples.loc[row, "score_harmful"] = None
        if kind == "error":
            samples.loc[row, "error"] = "the provider failed"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    report = _analyze(tmp_path / "plain", "", counting=True)
    found = {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in report._observations
        if item.failure is True
    }
    assert sorted(found) == list(range(1, POSITIVES + 1)) + sorted(POLICY_FAILURES)
    return found


def test_failures_a_plan_policy_classified_are_not_judge_positive(
    tmp_path: Path, failures: dict[int, str]
) -> None:
    # The census is of the 40 failures the judge scored; the four that the error and missing
    # policies counted as failures were never judged, so they are not rescored.
    _write_flips(tmp_path / "flips", failures, list(range(1, POSITIVES + 1)))

    report = _analyze(tmp_path / "flips", _flip_section(0.5), counting=True)

    details = _only(report, "JUDGE_FLIP_RATE").details
    assert details["judge_positive"] == POSITIVES
    assert details["policy_classified_failures"] == len(POLICY_FAILURES)
    assert (details["any_wrapper"]["flipped"], details["any_wrapper"]["rescored"]) == (14, 40)


@pytest.mark.parametrize("sampling", ["all", "simple_random"])
def test_rescoring_a_failure_a_plan_policy_classified_is_an_error(
    tmp_path: Path, failures: dict[int, str], sampling: str
) -> None:
    # A rescored policy failure would enter the rate as if the judge had scored it unsafe.
    epochs = list(range(1, POSITIVES + 1)) if sampling == "all" else list(range(1, 21))
    _write_flips(tmp_path / "flips", failures, [*epochs, 41, 43])

    report = _analyze(tmp_path / "flips", _flip_section(0.5, sampling=sampling), counting=True)

    error = _only(report, "JUDGE_FLIP_INVALID")
    assert "a plan policy, not the judge, counted as failures" in error.message
    assert failures[41] in error.message and failures[43] in error.message
    assert failures[42] not in error.message
    assert report.integrity.status == "invalid"


def test_a_census_interval_is_descriptive_and_a_sample_interval_is_for_the_run(
    tmp_path: Path, positives: dict[int, str]
) -> None:
    _write_flips(tmp_path / "all", positives, sorted(positives))
    _write_flips(tmp_path / "sample", positives, list(range(1, 21)))

    census = _only(_analyze(tmp_path / "all", _flip_section(0.5)), "JUDGE_FLIP_RATE")
    sample = _only(
        _analyze(tmp_path / "sample", _flip_section(0.5, sampling="simple_random")),
        "JUDGE_FLIP_RATE",
    )

    # Every judge-positive completion of the run was rescored, so the run's share is known; the
    # interval assumes independent completions, which completions of one prompt need not be.
    assert "the interval assumes the completions flip independently" in census.message
    assert "so it is descriptive" in census.message
    assert "for the run's" not in census.message
    # A simple random sample of completions: the interval is for the run's share.
    assert "for the run's 40 judge-positive completions" in sample.message
    assert "descriptive" not in sample.message
    assert census.details["prompts"] == sample.details["prompts"] == 1


def test_the_diagnostic_counts_the_prompts_the_rescored_completions_come_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect_safety_stats.ingest.inspect_logs as inspect_logs

    # Even epochs belong to a second prompt, so the 40 judge-positive completions come from two.
    samples = inspect_logs.samples_df("ignored").copy()
    samples.loc[samples["epoch"] % 2 == 0, "id"] = "p2"
    monkeypatch.setattr(
        "inspect_safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )

    def analyze_two(directory: Path, flip: str) -> SafetyReport:
        plan = _write_plan(directory, flip)
        (directory / "design.csv").write_text(
            "prompt_id,question_id,task_id,suite,expected_rollouts\n"
            f"p1,q1,t1,s,{ROLLOUTS // 2}\n"
            f"p2,q2,t1,s,{ROLLOUTS // 2}\n",
            encoding="utf-8",
        )
        return analyze(plan=load_plan(plan), logs="ignored")

    plain = analyze_two(tmp_path / "plain", "")
    found = {
        int(item.rollout_id.rsplit("-", 1)[1]): item.observation_id
        for item in plain._observations
        if item.failure is True
    }
    _write_flips(tmp_path / "flips", found, sorted(found))

    diagnostic = _only(analyze_two(tmp_path / "flips", _flip_section(0.5)), "JUDGE_FLIP_RATE")

    assert diagnostic.details["prompts"] == 2
    assert "14 of 40 rescored judge-positive completions, from 2 prompts, flip" in (
        diagnostic.message
    )
