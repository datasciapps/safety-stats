"""Tail and threshold measures warn when the rollouts per prompt cannot make them informative.

Their intervals are propagated from simultaneous per-prompt bounds, so a large total of
rollouts spread thinly over many prompts still leaves them wide. Each cohort and group is
judged on its own prompts, and each measure on its own interval: a worst-of-k measure needs
more rollouts than one prompt does, and a threshold measure fewer when its threshold lies
above a failure-free prompt's bound.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from safety_stats import analyze, load_plan
from safety_stats.ingest.rollout_files import RolloutFiles
from safety_stats.schema.reports import Diagnostic, SafetyReport
from safety_stats.stats.binomial import binomial_upper_bound

CODE = "TAIL_MEASURES_UNINFORMATIVE"

PLAN = """
schema_version: "2.0"
analysis: {id: tail, title: Tail, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
outputs:
  prompt_bounds: {family: displayed}
  risk_measures: [@MEASURES@]
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
@REPORT@
"""

ROLLOUTS = 10


def _plan(tmp_path: Path, measures: str, report: str = "") -> Path:
    (tmp_path / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts\n"
        f"p1,q1,t1,s,{ROLLOUTS}\np2,q2,t1,s,{ROLLOUTS}\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        PLAN.replace("@MEASURES@", measures).replace("@REPORT@", report).strip() + "\n",
        encoding="utf-8",
    )
    return plan


@pytest.fixture(autouse=True)
def _logs(monkeypatch: pytest.MonkeyPatch) -> None:
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-1",
                "run_id": "run-1",
                "log": "/not/a/local/log.eval",
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "tail",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {"temperature": 1},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "tail",
                "epochs": ROLLOUTS,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt}-{epoch}",
                "eval_id": "eval-1",
                "log": "/not/a/local/log.eval",
                "id": prompt,
                "epoch": epoch,
                "score_harmful": "FAIL" if (prompt, epoch) == ("p1", 1) else "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt in ("p1", "p2")
            for epoch in range(1, ROLLOUTS + 1)
        ]
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )


def _warnings(report: SafetyReport) -> list[Diagnostic]:
    return [item for item in report.integrity.diagnostics if item.code == CODE]


def test_tail_measures_warn_when_rollouts_cannot_support_them(tmp_path: Path) -> None:
    measures = "{id: mean, type: weighted_mean}, {id: cvar_90, type: cvar, beta: 0.9}"
    report = analyze(plan=_plan(tmp_path, measures), logs="ignored")

    (warning,) = _warnings(report)
    assert warning.severity == "WARNING"
    assert warning.scope == "cohorts/candidate/groups/all"
    assert (warning.details["cohort"], warning.details["group"]) == ("candidate", "all")
    # Only the tail measure is named. With direct_mean_intervals off, as here, the mean's
    # interval is propagated from the same per-prompt bounds, but the warning covers tail and
    # threshold measures only; the summary marks a wide interval of any measure with a dagger.
    assert warning.details["measures"] == ["cvar_90"]
    assert warning.details["median_rollouts_per_prompt"] == ROLLOUTS
    alpha = report.prompt_estimates[0].alpha_upper
    assert alpha is not None
    upper = binomial_upper_bound(0, ROLLOUTS, alpha=alpha)
    assert warning.details["zero_failure_prompt_upper"] == pytest.approx(upper, rel=1e-12, abs=0)
    needed = warning.details["rollouts_per_prompt_needed"]
    # The fewest rollouts at which a prompt with no failures meets the 0.05 target.
    assert binomial_upper_bound(0, needed, alpha=alpha) <= 0.05
    assert binomial_upper_bound(0, needed - 1, alpha=alpha) > 0.05
    assert warning.details["rollouts_per_prompt_needed_by_measure"] == {"cvar_90": needed}
    assert warning.message.startswith("In cohort 'candidate', group 'all', cvar_90 is built")
    assert warning.message.endswith(f"About {needed} rollouts per prompt would be needed.")
    # A warning only: the analysis is still valid.
    assert report.integrity.status == "valid"


def test_mean_only_plans_do_not_warn_about_tail_measures(tmp_path: Path) -> None:
    report = analyze(plan=_plan(tmp_path, "{id: mean, type: weighted_mean}"), logs="ignored")
    assert _warnings(report) == []


def test_worst_of_k_warns_unless_it_has_a_direct_interval(tmp_path: Path) -> None:
    propagated = analyze(
        plan=_plan(tmp_path, "{id: worst_3, type: worst_of_k, k: 3}"), logs="ignored"
    )
    assert [item.details["measures"] for item in _warnings(propagated)] == [["worst_3"]]
    direct = analyze(
        plan=_plan(tmp_path, "{id: worst_3, type: worst_of_k, k: 3, direct_interval: true}"),
        logs="ignored",
    )
    assert _warnings(direct) == []


def test_a_loose_enough_precision_target_does_not_warn(tmp_path: Path) -> None:
    plan = _plan(
        tmp_path,
        "{id: max, type: maximum}",
        report="report: {precision_target: 0.49}",
    )
    assert _warnings(analyze(plan=plan, logs="ignored")) == []


def test_the_default_precision_target_leaves_the_plan_hash_alone(tmp_path: Path) -> None:
    default = load_plan(_plan(tmp_path, "{id: max, type: maximum}"))
    assert "precision_target" not in default.model_dump(mode="json")["report"]
    stated = load_plan(
        _plan(tmp_path, "{id: max, type: maximum}", report="report: {precision_target: 0.1}")
    )
    assert stated.model_dump(mode="json")["report"]["precision_target"] == 0.1


def _fewest(meets: Callable[[int], bool]) -> int:
    """The fewest rollouts per prompt at which ``meets(rollouts)`` holds, counted one by one."""

    rollouts = 1
    while not meets(rollouts):
        rollouts += 1
    return rollouts


def test_threshold_measures_that_certify_zero_are_not_warned_about(tmp_path: Path) -> None:
    # Ten rollouts bound a failure-free prompt at about 0.355, below the threshold 0.5: a
    # prompt without failures counts as below it, so the threshold measures can certify 0.
    measures = (
        "{id: exceed, type: threshold_exceedance, threshold: 0.5}, "
        "{id: excess, type: threshold_excess, threshold: 0.5}"
    )
    report = analyze(plan=_plan(tmp_path, measures), logs="ignored")

    alpha = report.prompt_estimates[0].alpha_upper
    assert alpha is not None
    assert binomial_upper_bound(0, ROLLOUTS, alpha=alpha) < 0.5
    assert _warnings(report) == []


def test_a_worst_of_k_measure_needs_the_rollouts_its_own_bound_needs(tmp_path: Path) -> None:
    report = analyze(
        plan=_plan(tmp_path, "{id: worst_3, type: worst_of_k, k: 3}"), logs="ignored"
    )

    (warning,) = _warnings(report)
    alpha = report.prompt_estimates[0].alpha_upper
    assert alpha is not None

    def measure_upper(rollouts: int) -> float:
        # The measure's upper bound when no prompt fails: each prompt at its own bound.
        return -math.expm1(3 * math.log1p(-binomial_upper_bound(0, rollouts, alpha=alpha)))

    needed = warning.details["rollouts_per_prompt_needed"]
    assert needed == _fewest(lambda rollouts: measure_upper(rollouts) <= 0.05)
    # More than one prompt needs: at that count the measure would still be above the target.
    per_prompt = _fewest(lambda rollouts: binomial_upper_bound(0, rollouts, alpha=alpha) <= 0.05)
    assert measure_upper(per_prompt) > 0.05
    assert needed > per_prompt


ROLLOUT_PLAN = """
schema_version: "3.0"
analysis: {id: tail, title: Tail, intent: exploratory}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
cohorts:
@COHORTS@
groups:
@GROUPS@
outputs:
  prompt_bounds: {enabled: true, family: displayed}
  risk_measures: [@MEASURES@]
multiplicity:
  families:
    - {id: displayed, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
claims: []
@DESIGN@
"""


def _rollout_files(
    tmp_path: Path,
    measures: str,
    rollouts: dict[str, dict[str, int]],
    *,
    groups: str = "  - id: all",
    columns: dict[str, dict[str, object]] | None = None,
    design: str = "design: {require_expected_rollouts: false}",
) -> tuple[Path, RolloutFiles]:
    """A plan on rollout files with ``rollouts[cohort][prompt]`` safe rollouts of each prompt.

    Each cohort's run has the cohort's name as its model. ``columns[name][prompt]`` adds
    manifest columns.
    """

    prompts = list(dict.fromkeys(prompt for counts in rollouts.values() for prompt in counts))
    extra = columns or {}
    (tmp_path / "design.csv").write_text(
        ",".join(["prompt_id", "question_id", "task_id", *extra]) + "\n"
        + "".join(
            ",".join([prompt, f"q{prompt}", "t1", *(str(extra[name][prompt]) for name in extra)])
            + "\n"
            for prompt in prompts
        ),
        encoding="utf-8",
        newline="\n",
    )
    cohorts = "\n".join(f"  {cohort}: {{match: {{model: {cohort}}}}}" for cohort in rollouts)
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        ROLLOUT_PLAN.replace("@COHORTS@", cohorts)
        .replace("@GROUPS@", groups)
        .replace("@MEASURES@", measures)
        .replace("@DESIGN@", design)
        .strip()
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        "".join(
            json.dumps({"run_key": cohort, "model": cohort, "status": "success"}) + "\n"
            for cohort in rollouts
        ),
        encoding="utf-8",
        newline="\n",
    )
    records = tmp_path / "rollouts.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {"run_key": cohort, "item_id": prompt, "repeat": repeat, "score_harmful": "PASS"}
            )
            + "\n"
            for cohort, counts in rollouts.items()
            for prompt, count in counts.items()
            for repeat in range(1, count + 1)
        ),
        encoding="utf-8",
        newline="\n",
    )
    return plan, RolloutFiles(runs=runs, rollouts=records)


def _uninformative(plan: Path, source: RolloutFiles) -> list[Diagnostic]:
    return _warnings(analyze(plan=plan, source=source))


def test_a_threshold_below_the_zero_failure_bound_warns_with_the_rollouts_it_needs(
    tmp_path: Path,
) -> None:
    # At ten rollouts a failure-free prompt is bounded at about 0.355, above both thresholds:
    # exceedance counts every prompt whose bound is above its threshold, so p2's weight, and
    # excess rises by the bound minus each threshold. The maximum is wide as well.
    measures = (
        "{id: exceed, type: threshold_exceedance, threshold_column: limit}, "
        "{id: excess, type: threshold_excess, threshold: 0.2}, {id: worst, type: maximum}"
    )
    plan, source = _rollout_files(
        tmp_path,
        measures,
        {"candidate": {"p1": ROLLOUTS, "p2": ROLLOUTS}},
        columns={"limit": {"p1": 0.9, "p2": 0.3}},
    )
    report = analyze(plan=plan, source=source)

    (warning,) = _warnings(report)
    alpha = report.prompt_estimates[0].alpha_upper
    assert alpha is not None

    def bound(rollouts: int) -> float:
        return binomial_upper_bound(0, rollouts, alpha=alpha)

    needed = {
        # p2's bound must fall to its threshold; p1's, at 0.9, never counts.
        "exceed": _fewest(lambda rollouts: bound(rollouts) <= 0.3),
        "excess": _fewest(lambda rollouts: bound(rollouts) - 0.2 <= 0.05),
        "worst": _fewest(lambda rollouts: bound(rollouts) <= 0.05),
    }
    assert warning.details["measures"] == ["exceed", "excess", "worst"]
    assert warning.details["rollouts_per_prompt_needed_by_measure"] == needed
    assert warning.details["rollouts_per_prompt_needed"] == max(needed.values())
    assert warning.message.endswith(
        f"Rollouts per prompt needed: about {needed['exceed']} for exceed; "
        f"about {needed['excess']} for excess; about {needed['worst']} for worst."
    )


def test_a_threshold_no_bound_can_reach_is_warned_about_without_a_count(tmp_path: Path) -> None:
    # Every upper bound is above 0, so every prompt counts as exceeding a zero threshold.
    plan, source = _rollout_files(
        tmp_path,
        "{id: exceed, type: threshold_exceedance, threshold: 0}",
        {"candidate": {"p1": 1000, "p2": 1000}},
    )

    (warning,) = _uninformative(plan, source)
    assert warning.details["rollouts_per_prompt_needed"] is None
    assert warning.details["rollouts_per_prompt_needed_by_measure"] == {"exceed": None}
    assert warning.message.endswith("No number of rollouts per prompt up to 10^9 would be enough.")


def test_each_cohort_is_judged_on_its_own_rollouts(tmp_path: Path) -> None:
    # Pooled with bbb's, aaa's ten rollouts per prompt were hidden by bbb's three hundred.
    plan, source = _rollout_files(
        tmp_path,
        "{id: cvar_90, type: cvar, beta: 0.9}",
        {"aaa": {"p1": ROLLOUTS, "p2": ROLLOUTS}, "bbb": {"p1": 300, "p2": 300}},
    )

    (warning,) = _uninformative(plan, source)
    assert warning.scope == "cohorts/aaa/groups/all"
    assert warning.details["cohort"] == "aaa"
    assert warning.details["median_rollouts_per_prompt"] == ROLLOUTS
    assert warning.message.startswith("In cohort 'aaa', group 'all', cvar_90 is built")


def test_each_group_is_judged_on_its_own_prompts_at_their_lower_median(tmp_path: Path) -> None:
    plan, source = _rollout_files(
        tmp_path,
        "{id: worst, type: maximum}",
        {"candidate": {"p1": ROLLOUTS, "p2": ROLLOUTS, "p3": 300, "p4": 300}},
        groups=(
            "  - id: all\n"
            "  - {id: thin, filter: {depth: thin}}\n"
            "  - {id: thick, filter: {depth: thick}}"
        ),
        columns={"depth": {"p1": "thin", "p2": "thin", "p3": "thick", "p4": "thick"}},
    )

    warnings = _uninformative(plan, source)
    # Half of all's prompts have ten rollouts: the lower median, ten, decides, not 300.
    medians = [
        (item.details["group"], item.details["median_rollouts_per_prompt"]) for item in warnings
    ]
    assert medians == [("all", ROLLOUTS), ("thin", ROLLOUTS)]


def test_prompts_whose_bounds_are_withheld_do_not_count(tmp_path: Path) -> None:
    # The manifest expects five rollouts of p1 and p2, and each has ten: an error withholds
    # their bounds, so the maximum has no interval, and p3's rollouts decide nothing.
    plan, source = _rollout_files(
        tmp_path,
        "{id: worst, type: maximum}",
        {"candidate": {"p1": ROLLOUTS, "p2": ROLLOUTS, "p3": 300}},
        columns={"expected_rollouts": {"p1": 5, "p2": 5, "p3": 300}},
        design="",
    )
    report = analyze(plan=plan, source=source)

    withheld = {item.prompt_id for item in report.prompt_estimates if item.upper is None}
    assert withheld == {"p1", "p2"}
    assert _warnings(report) == []


def test_no_warning_when_a_withheld_bound_leaves_the_measure_without_an_interval(
    tmp_path: Path,
) -> None:
    # p3 and p4 certify at ten rollouts, which alone would warn; p1 and p2 are withheld.
    plan, source = _rollout_files(
        tmp_path,
        "{id: worst, type: maximum}",
        {"candidate": {"p1": ROLLOUTS, "p2": ROLLOUTS, "p3": ROLLOUTS, "p4": ROLLOUTS}},
        columns={"expected_rollouts": {"p1": 5, "p2": 5, "p3": ROLLOUTS, "p4": ROLLOUTS}},
        design="",
    )
    report = analyze(plan=plan, source=source)

    withheld = {item.prompt_id for item in report.prompt_estimates if item.upper is None}
    assert withheld == {"p1", "p2"}
    (risk,) = [item for item in report.group_risks if item.measure_id == "worst"]
    assert (risk.lower, risk.upper) == (None, None)
    assert _warnings(report) == []


def test_no_warning_when_every_prompt_bound_is_withheld(tmp_path: Path) -> None:
    # The manifest has no expected_rollouts column, which the default design requires.
    plan, source = _rollout_files(
        tmp_path,
        "{id: worst, type: maximum}",
        {"candidate": {"p1": ROLLOUTS, "p2": ROLLOUTS}},
        design="",
    )
    report = analyze(plan=plan, source=source)

    assert "PLAN_EXPECTED_ROLLOUTS_MISSING" in {
        item.code for item in report.integrity.diagnostics
    }
    assert all(item.upper is None for item in report.prompt_estimates)
    assert _warnings(report) == []
