"""Anytime-valid bounds for runs stopped early (#126), end to end through real Inspect logs.

A fixed design of 40 prompts runs on Inspect's mock model in a shuffled order with a recorded
seed, and an early-stopping manager, standing in for optstop, halts it after 25 samples. Four
prompts always fail, so theta is 0.1. Other runs of the same design leave out chosen samples
and epochs, are not shuffled, or are cut short by a limit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.util._early_stopping import EarlyStop
from typer.testing import CliRunner

from safety_stats import AnalysisPlan, analyze, load_plan, validate
from safety_stats.cli import app
from safety_stats.ingest.records import SampleOrder
from safety_stats.schema.reports import Diagnostic, SafetyReport
from safety_stats.stats.sequential import anytime_upper_bound
from safety_stats.validate.plan import validate_plan_references

PROMPTS = 40
STOP_AFTER = 25
SEED = 11

PLAN = """
schema_version: "2.0"
analysis: {id: stopped, title: Stopped early, intent: confirmatory,
           plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: verdict
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: conservative_bounds
    errors: block
design:
  type: fixed
  hierarchy: [task_id, question_id, prompt_id]
  rollout_key: [eval_id, id, epoch]
  require_complete_manifest: false
  require_expected_rollouts: false
  independence_profiles:
    rollout: {block_by: [rollout], comparison_coupling: independent}
    prompt: {block_by: [prompt_id], comparison_coupling: independent}
    question: {block_by: [question_id], comparison_coupling: independent}
cohorts:
  mock: {match: {model: mockllm/model}}
groups:
  - id: all
    filter: {suite: s}
outputs:
  prompt_bounds: {enabled: false}
  risk_measures:
    - {id: mean, type: weighted_mean}
@ANYTIME@
multiplicity:
  families:
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: loose, title: Loose, cohort: mock, group: all, measure: mean, family: gate,
     acceptable_upper: 0.8, unacceptable_lower: 0.9}
  - {id: tight, title: Tight, cohort: mock, group: all, measure: mean, family: gate,
     acceptable_upper: 0.05, unacceptable_lower: 0.5}
report: {include_observations: true, decimal_places: 6}
"""

SECTION = (
    "  anytime_bounds:\n"
    "    independence_profile: {profile}\n"
    "    family: {family}\n"
)


def _failing(prompt: str) -> bool:
    return int(prompt[1:]) % 10 == 0


@scorer(metrics=[])
def verdict():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value="FAIL" if _failing(str(state.sample_id)) else "PASS")

    return score


class SkipAt:
    """An early-stopping manager that skips chosen epochs of the samples at chosen places of
    the run's order, and runs everything else."""

    def __init__(self, places: list[tuple[int, int]]) -> None:
        self.places = places
        self.skipped: set[tuple[str | int, int]] = set()

    async def start_task(self, task: Any, samples: list[Sample], epochs: int) -> str:
        # Inspect passes the samples in the order it schedules them: the shuffled order.
        self.skipped = {(samples[place].id or "", epoch) for place, epoch in self.places}
        return "skip_at"

    async def schedule_sample(self, id: str | int, epoch: int) -> EarlyStop | None:
        if (id, epoch) in self.skipped:
            return EarlyStop(id=id, epoch=epoch, reason="skip_at")
        return None

    async def complete_sample(self, id: str | int, epoch: int, scores: Any) -> None:
        return None

    async def complete_task(self) -> dict[str, Any]:
        return {}


# The gapped run: two epochs, of which the sample 11th in the order misses both, the 21st its
# second, and the last its first.
GAPS = [(10, 1), (10, 2), (20, 2), (PROMPTS - 1, 1)]


class StopAfter:
    """An early-stopping manager that halts the run after a fixed number of samples."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.completed = 0

    async def start_task(self, task: Any, samples: list[Sample], epochs: int) -> str:
        return "stop_after"

    async def schedule_sample(self, id: str | int, epoch: int) -> EarlyStop | None:
        if self.completed >= self.count:
            return EarlyStop(id=id, epoch=epoch, reason="stop_after")
        return None

    async def complete_sample(self, id: str | int, epoch: int, scores: Any) -> None:
        self.completed += 1

    async def complete_task(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _run_from_the_drive_that_holds_the_logs(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work from the drive of pytest's temporary root, where these tests write real logs (#35)."""

    monkeypatch.chdir(tmp_path_factory.getbasetemp())


def _run(
    log_dir: Path,
    shuffle: bool | int | None,
    *,
    stopping: StopAfter | SkipAt | None = None,
    epochs: int = 1,
    limit: int | None = None,
) -> str:
    task = Task(
        dataset=MemoryDataset([Sample(id=f"p{i}", input=f"q{i}") for i in range(PROMPTS)]),
        solver=generate(),
        scorer=verdict(),
        early_stopping=stopping or StopAfter(STOP_AFTER),
        epochs=epochs,
    )
    (log,) = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(log_dir),
        sample_shuffle=shuffle,
        max_samples=1,
        limit=limit,
        display="none",
    )
    # Inspect gives the location relative to the working directory of the run.
    return str(Path(log.location).resolve())


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("anytime-logs")
    return {
        "seeded": _run(root / "seeded", SEED),
        "unseeded": _run(root / "unseeded", True),
        "unshuffled": _run(root / "unshuffled", None),
        # Inspect records the seed 0, and does not shuffle under it.
        "zero": _run(root / "zero", 0),
        "gapped": _run(root / "gapped", SEED, stopping=SkipAt(GAPS), epochs=2),
        "limited": _run(root / "limited", SEED, limit=30),
    }


def _analyze(
    directory: Path, log: str | list[str], section: str = "", edit: Callable[[str], str] = str
) -> SafetyReport:
    """Analyse ``log`` under the plan with ``section`` in its outputs, edited by ``edit``."""

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite\n"
        + "".join(f"p{i},q{i},t1,s\n" for i in range(PROMPTS)),
        encoding="utf-8",
    )
    plan = directory / "plan.yaml"
    plan.write_text(
        edit(PLAN.replace("@ANYTIME@\n", section)).strip() + "\n", encoding="utf-8"
    )
    return analyze(plan=load_plan(plan), logs=log)


def _found(report: SafetyReport, code: str) -> list[Diagnostic]:
    return [item for item in report.integrity.diagnostics if item.code == code]


def _order(log: str) -> list[str]:
    header = read_eval_log(log, header_only=True)
    return [str(item) for item in header.eval.dataset.sample_ids or []]


def _section(profile: str = "prompt", rollouts: int = 1, groups: str = "") -> str:
    return (
        SECTION.format(profile=profile, family="gate")
        + f"    rollouts_per_item: {rollouts}\n"
        + (f"    groups: {groups}\n" if groups else "")
    )


def _uniform_bound(
    prefix: list[str], shares: list[float], *, alpha: float = 0.05, independent: int = 1
) -> float:
    """The bound of the design's equal weights over ``prefix`` with these shares."""

    return anytime_upper_bound(
        shares,
        np.array([int(prompt[1:]) for prompt in prefix]),
        np.full(PROMPTS, 1 / PROMPTS),
        alpha=alpha,
        independent_rollouts=independent,
    ).upper


def test_a_stopped_shuffled_run_gets_the_bound_of_its_recorded_prefix(
    logs: dict[str, str], tmp_path: Path
) -> None:
    header = read_eval_log(logs["seeded"], header_only=True)
    order = [str(item) for item in header.eval.dataset.sample_ids or []]
    prefix = order[:STOP_AFTER]

    report = _analyze(
        tmp_path, logs["seeded"], SECTION.format(profile="prompt", family="gate")
    )

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    details = diagnostic.details
    assert diagnostic.severity == "INFO"
    assert diagnostic.message.startswith("experimental anytime-valid upper bound ")
    assert details["experimental"] is True
    assert details["sample_shuffle"] == SEED
    assert details["population"] == PROMPTS
    assert details["draws"] == details["complete_draws"] == STOP_AFTER
    assert details["rollouts_counted_as_failures"] == 0
    assert details["independent_rollouts"] == 1
    expected = anytime_upper_bound(
        [1.0 if _failing(prompt) else 0.0 for prompt in prefix],
        np.array([int(prompt[1:]) for prompt in prefix]),
        np.full(PROMPTS, 1 / PROMPTS),
        alpha=0.05,
    )
    assert details["upper"] == expected.upper
    assert details["alpha"] == 0.05
    assert {item["claim_id"]: item["certified"] for item in details["claims"]} == {
        "loose": expected.upper <= 0.8,
        "tight": False,
    }
    assert expected.upper <= 0.8


def test_prompts_and_epochs_missing_before_the_last_drawn_prompt_count_as_failures(
    logs: dict[str, str], tmp_path: Path
) -> None:
    """The 11th prompt in the order ran no epoch, so with one rollout per item it counts as a
    failure where the order places it, not where it finished. The last prompt ran only its
    second epoch, so the stream ends before it."""

    order = _order(logs["gapped"])
    report = _analyze(tmp_path, logs["gapped"], _section(rollouts=1))

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    details = diagnostic.details
    assert (details["draws"], details["complete_draws"]) == (PROMPTS - 1, PROMPTS - 2)
    assert details["rollouts_counted_as_failures"] == 1
    prefix = order[: PROMPTS - 1]
    shares = [
        1.0 if place == 10 else float(_failing(prompt)) for place, prompt in enumerate(prefix)
    ]
    assert details["upper"] == _uniform_bound(prefix, shares)
    assert details["upper"] > _uniform_bound(prefix, [float(_failing(p)) for p in prefix])


@pytest.mark.parametrize(("profile", "independent"), [("rollout", 2), ("prompt", 1)])
def test_two_rollouts_per_item_count_each_missing_epoch_as_a_failure(
    logs: dict[str, str], tmp_path: Path, profile: str, independent: int
) -> None:
    """Each prompt's share is over its first two epochs, a missing one counted as a failure.
    Only the rollout profile counts the two rollouts as independent in step 2."""

    order = _order(logs["gapped"])
    report = _analyze(tmp_path, logs["gapped"], _section(profile, rollouts=2))

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    details = diagnostic.details
    assert (details["draws"], details["complete_draws"]) == (PROMPTS, PROMPTS - 3)
    assert details["rollouts_counted_as_failures"] == 4
    assert details["independent_rollouts"] == independent
    missing = {place: sum(1 for gap, _ in GAPS if gap == place) for place, _ in GAPS}
    shares = [
        (float(_failing(prompt)) * (2 - missing.get(place, 0)) + missing.get(place, 0)) / 2
        for place, prompt in enumerate(order)
    ]
    assert details["upper"] == _uniform_bound(order, shares, independent=independent)


def test_the_familys_alpha_is_split_over_the_bounded_groups(
    logs: dict[str, str], tmp_path: Path
) -> None:
    def two_groups(plan: str) -> str:
        return plan.replace(
            "  - id: all\n    filter: {suite: s}\n",
            "  - id: all\n    filter: {suite: s}\n  - id: also\n    filter: {suite: s}\n",
        )

    prefix = _order(logs["seeded"])[:STOP_AFTER]
    shares = [float(_failing(prompt)) for prompt in prefix]
    both = _analyze(tmp_path / "both", logs["seeded"], _section(), two_groups)
    one = _analyze(tmp_path / "one", logs["seeded"], _section(groups="[also]"), two_groups)

    bounds = _found(both, "ANYTIME_UPPER_BOUND")
    assert sorted(item.details["group"] for item in bounds) == ["all", "also"]
    for item in bounds:
        assert item.details["alpha"] == 0.025
        assert item.details["upper"] == _uniform_bound(prefix, shares, alpha=0.025)
    (only,) = _found(one, "ANYTIME_UPPER_BOUND")
    assert (only.details["group"], only.details["alpha"]) == ("also", 0.05)
    assert only.details["upper"] == _uniform_bound(prefix, shares)


def test_a_bound_equal_to_acceptable_upper_certifies_the_claim(
    logs: dict[str, str], tmp_path: Path
) -> None:
    section = SECTION.format(profile="prompt", family="gate")
    (first,) = _found(_analyze(tmp_path / "first", logs["seeded"], section), "ANYTIME_UPPER_BOUND")
    upper = first.details["upper"]
    report = _analyze(
        tmp_path / "equal",
        logs["seeded"],
        section,
        lambda plan: plan.replace("acceptable_upper: 0.8,", f"acceptable_upper: {upper!r},"),
    )

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    assert diagnostic.details["upper"] == upper
    assert diagnostic.details["claims"][0] == {
        "claim_id": "loose",
        "acceptable_upper": upper,
        "certified": True,
    }


def test_the_bound_changes_no_claim(logs: dict[str, str], tmp_path: Path) -> None:
    plain = _analyze(tmp_path / "plain", logs["seeded"])
    with_bound = _analyze(
        tmp_path / "bound", logs["seeded"], SECTION.format(profile="prompt", family="gate")
    )
    assert [claim.model_dump() for claim in with_bound.claims] == [
        claim.model_dump() for claim in plain.claims
    ]
    assert with_bound.decision == plain.decision


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        (
            lambda plan: plan.replace("intent: confirmatory", "intent: exploratory"),
            "the analysis is exploratory",
        ),
        # The prompts after the stop have no rollouts, which a complete manifest forbids.
        (
            lambda plan: plan.replace(
                "require_complete_manifest: false", "require_complete_manifest: true"
            ),
            "data integrity is invalid",
        ),
    ],
    ids=["exploratory", "integrity-invalid"],
)
def test_the_bound_certifies_no_claim_where_no_claim_can_pass(
    logs: dict[str, str], tmp_path: Path, edit: Callable[[str], str], why: str
) -> None:
    """A claim passes only in a confirmatory analysis of valid integrity; the bound, which
    certifies the loose claim in one, certifies nothing in any other."""

    report = _analyze(
        tmp_path, logs["seeded"], SECTION.format(profile="prompt", family="gate"), edit
    )

    (diagnostic,) = _found(report, "ANYTIME_UPPER_BOUND")
    assert diagnostic.details["upper"] <= 0.8
    assert [item["certified"] for item in diagnostic.details["claims"]] == [None, None]
    assert diagnostic.details["certification_withheld"] == why
    assert f"it certifies no claim, because {why}" in diagnostic.message
    assert "PASS" not in {claim.status for claim in report.claims}


@pytest.mark.parametrize(
    ("run", "why"),
    [
        ("unshuffled", "Inspect does not shuffle by default"),
        ("unseeded", "without a recorded seed"),
        ("zero", "its samples were not shuffled, as Inspect does not shuffle under "
                 "sample_shuffle 0"),
    ],
)
def test_a_run_without_a_recorded_order_is_refused(
    logs: dict[str, str], tmp_path: Path, run: str, why: str
) -> None:
    report = _analyze(tmp_path, logs[run], SECTION.format(profile="prompt", family="gate"))

    (diagnostic,) = _found(report, "ANYTIME_ORDER_UNRECORDED")
    assert diagnostic.severity == "ERROR"
    assert why in diagnostic.message
    assert diagnostic.message.endswith("Run it with sample_shuffle set to a nonzero integer seed")
    assert not _found(report, "ANYTIME_UPPER_BOUND")
    assert report.integrity.status == "invalid"


def test_a_limit_that_leaves_prompts_out_of_the_order_is_refused(
    logs: dict[str, str], tmp_path: Path
) -> None:
    report = _analyze(tmp_path, logs["limited"], SECTION.format(profile="prompt", family="gate"))

    (diagnostic,) = _found(report, "ANYTIME_BOUND_INVALID")
    assert "does not place 10 of its prompts" in diagnostic.message
    assert not _found(report, "ANYTIME_UPPER_BOUND")
    assert report.integrity.status == "invalid"


def test_two_runs_in_one_cohort_are_refused(logs: dict[str, str], tmp_path: Path) -> None:
    report = _analyze(
        tmp_path, [logs["seeded"], logs["zero"]], SECTION.format(profile="prompt", family="gate")
    )

    (diagnostic,) = _found(report, "ANYTIME_BOUND_INVALID")
    assert "its observations come from 2 runs" in diagnostic.message
    assert not _found(report, "ANYTIME_UPPER_BOUND")


@pytest.mark.parametrize("run", ["seeded", "unseeded", "unshuffled", "zero"])
def test_validate_refuses_what_analyze_refuses(
    logs: dict[str, str], tmp_path: Path, run: str
) -> None:
    """CONTRIBUTING 5.2: validate accepts exactly what analyze accepts, at the CLI too."""

    report = _analyze(tmp_path, logs[run], SECTION.format(profile="prompt", family="gate"))
    plan = tmp_path / "plan.yaml"
    checked = validate(plan=load_plan(plan), logs=logs[run])
    command = CliRunner().invoke(app, ["validate", "--plan", str(plan), "--logs", logs[run]])

    def errors(diagnostics: list[Diagnostic]) -> list[tuple[str, str, str | None]]:
        return [
            (item.code, item.message, item.scope)
            for item in diagnostics
            if item.severity == "ERROR"
        ]

    assert errors(checked.diagnostics) == errors(report.integrity.diagnostics)
    assert checked.status == report.integrity.status
    assert (checked.status == "invalid") == (run != "seeded")
    assert command.exit_code == (3 if run != "seeded" else 0)


@pytest.mark.parametrize(
    ("profile", "family", "problem"),
    [
        ("question", "gate", "must block by rollout or by prompt_id"),
        ("absent", "gate", "is not declared in design.independence_profiles"),
        ("prompt", "absent", "is not declared in multiplicity.families"),
    ],
)
def test_a_plan_the_bound_cannot_serve_is_refused(
    logs: dict[str, str], tmp_path: Path, profile: str, family: str, problem: str
) -> None:
    report = _analyze(
        tmp_path, logs["seeded"], SECTION.format(profile=profile, family=family)
    )

    (diagnostic,) = _found(report, "PLAN_ANYTIME_BOUNDS_INVALID")
    assert problem in diagnostic.message
    assert not _found(report, "ANYTIME_UPPER_BOUND")


def test_a_sampled_design_is_refused(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        PLAN.replace("@ANYTIME@\n", SECTION.format(profile="prompt", family="gate")).replace(
            "  type: fixed\n",
            "  type: sampled\n  inclusion_probability_column: inclusion_probability\n",
        ),
        encoding="utf-8",
    )

    (refusal,) = _plan_refusals(load_plan(plan))
    assert "need design.type: fixed, not 'sampled'" in refusal.message


def test_plans_without_the_section_keep_their_dump(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text(PLAN.replace("@ANYTIME@\n", ""), encoding="utf-8")
    assert "anytime_bounds" not in load_plan(plan).model_dump(mode="json")["outputs"]


# --- judge calibration: the Inspect readers replaced by fixed frames -------------------------

JUDGE_EPOCHS = 100
JUDGE_PROMPTS = [f"j{i}" for i in range(20)]

JUDGE_PLAN = """
schema_version: "2.0"
analysis: {id: judged, title: Judged, intent: confirmatory, plan_frozen_at: 2025-01-01T00:00:00Z}
inputs:
  design_manifest: design.csv
  score:
    name: harmful
    mapping: {failure_values: ["FAIL"], safe_values: ["PASS"]}
    missing: block
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
  candidate: {match: {model: model/candidate}}
groups:
  - {id: all, filter: {suite: s}}
  - {id: also, filter: {suite: s}}
outputs:
  prompt_bounds: {enabled: false}
  risk_measures:
    - {id: mean, type: weighted_mean}
  anytime_bounds:
    independence_profile: rollout
    family: gate
    rollouts_per_item: 100
@GROUPS@
multiplicity:
  families:
    - {id: judge_error, purpose: simultaneous_intervals, method: bonferroni, alpha: 0.05}
    - {id: gate, purpose: all_mandatory, method: intersection_union, alpha: 0.05}
claims:
  - {id: mean_all, title: All, cohort: candidate, group: all, measure: mean, family: gate,
     acceptable_upper: 0.06, unacceptable_lower: 0.09}
report: {include_observations: true, decimal_places: 6}
"""

VALIDATED = (
    "  judge_calibration:\n"
    "    mode: validated\n"
    "    validation_manifest: labels.csv\n"
    "    family: judge_error\n"
)
ASSUMED = (
    "  judge_calibration:\n"
    "    mode: assumed\n"
    "    primary: s1\n"
    "    scenarios:\n"
    "      - {id: s1, rho_1: {lower: 0.9, upper: 1.0}, rho_0: {lower: 0.0, upper: 0.08}}\n"
)
TRANSPORTED = (
    "  judge_calibration:\n"
    "    mode: transported\n"
    "    published_set:\n"
    "      name: Published judge validation set\n"
    "      source: https://example.org/judge-validation\n"
    "      judge: judge-model-v1\n"
    "      rubric: harmful-behaviour rubric v2\n"
    "      human_unsafe: 120\n"
    "      judge_flagged_unsafe: 100\n"
)


def _judge_plan(directory: Path, judge: str = "", groups: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "design.csv").write_text(
        "prompt_id,question_id,task_id,suite,expected_rollouts\n"
        + "".join(f"{prompt},q{prompt},t,s,{JUDGE_EPOCHS}\n" for prompt in JUDGE_PROMPTS),
        encoding="utf-8",
    )
    plan = directory / "plan.yaml"
    text = JUDGE_PLAN.replace("@JUDGE@\n", judge).replace("@GROUPS@\n", groups)
    plan.write_text(text.strip() + "\n", encoding="utf-8")
    return plan


def _judge_frames(judge_unsafe: Callable[[int], bool]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One run of every prompt for ``JUDGE_EPOCHS`` epochs, judged unsafe on the given epochs."""

    log = "/not/a/local/log.eval"
    evals = pd.DataFrame(
        [
            {
                "eval_id": "eval-candidate",
                "run_id": "run-1",
                "log": log,
                "created": "2026-01-01T00:00:00Z",
                "status": "success",
                "task_name": "t",
                "task_version": "1",
                "model": "model/candidate",
                "model_generate_config": {},
                "solver": "generate",
                "solver_args": {},
                "sandbox_type": None,
                "sandbox_config": None,
                "dataset_name": "fixed",
                "epochs": JUDGE_EPOCHS,
            }
        ]
    )
    samples = pd.DataFrame(
        [
            {
                "sample_id": f"{prompt}-{epoch}",
                "eval_id": "eval-candidate",
                "log": log,
                "id": prompt,
                "epoch": epoch,
                "score_harmful": "FAIL" if judge_unsafe(epoch) else "PASS",
                "error": None,
                "retries": 0,
            }
            for prompt in JUDGE_PROMPTS
            for epoch in range(1, JUDGE_EPOCHS + 1)
        ]
    )
    return evals, samples


def _read_frames(
    monkeypatch: pytest.MonkeyPatch,
    frames: tuple[pd.DataFrame, pd.DataFrame],
    order: list[str],
    *,
    shuffled: bool = True,
) -> None:
    """Make ``analyze`` read ``frames`` as its Inspect logs, run in ``order`` with seed 7.

    ``shuffled`` is what the log says of whether Inspect shuffled the samples.
    """

    evals, samples = frames
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.evals_df", lambda logs, **_: evals
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.samples_df",
        lambda logs, parallel=False: samples,
    )
    monkeypatch.setattr(
        "safety_stats.ingest.inspect_logs.read_inspect_sample_orders",
        lambda sources, logs=None: {
            key: SampleOrder(
                run_key=key,
                source=source,
                shuffle=7,
                shuffled=shuffled,
                sample_ids=tuple(order),
            )
            for key, source in sources.items()
        },
    )


def _plan_refusals(plan: AnalysisPlan) -> list[Diagnostic]:
    return [
        item
        for item in validate_plan_references(plan)
        if item.code == "PLAN_ANYTIME_BOUNDS_INVALID"
    ]


def test_a_bound_on_the_judges_risk_never_certifies_a_claim_on_true_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every prompt is truly unsafe on 10 of its 100 epochs, so true risk is 0.10, and the
    judge flags 3 of them, so judge-reported risk is 0.03. Human labels on every rollout
    validate the judge, so the claim (acceptable_upper 0.06) is about true risk.

    The bound on the judge's risk lies near 0.04, at or below 0.06, and used to be reported as
    certifying the claim. The plan is now refused before any bound is computed.
    """

    _read_frames(
        monkeypatch,
        _judge_frames(lambda epoch: epoch <= 3),
        [str(item) for item in np.random.default_rng(7).permutation(JUDGE_PROMPTS)],
    )
    uncalibrated = analyze(plan=load_plan(_judge_plan(tmp_path / "plain")), logs="ignored")
    (bound,) = [
        item
        for item in _found(uncalibrated, "ANYTIME_UPPER_BOUND")
        if item.details["group"] == "all"
    ]
    assert bound.details["upper"] <= 0.06
    labels = tmp_path / "judged" / "labels.csv"
    labels.parent.mkdir()
    labels.write_text(
        "observation_id,human_label\n"
        + "".join(
            f"{row.observation_id},{'unsafe' if row.repeat <= 10 else 'safe'}\n"
            for row in uncalibrated._observations
        ),
        encoding="utf-8",
    )

    report = analyze(
        plan=load_plan(_judge_plan(tmp_path / "judged", VALIDATED)), logs="ignored"
    )

    assert not _found(report, "ANYTIME_UPPER_BOUND")
    (refusal,) = _found(report, "PLAN_ANYTIME_BOUNDS_INVALID")
    assert "validated judge calibration covers the bounded groups ['all', 'also']" in (
        refusal.message
    )
    assert report.integrity.status == "invalid"
    (claim,) = report.claims
    assert claim.status != "PASS"


@pytest.mark.parametrize(
    ("judge", "groups", "remedy"),
    [
        (VALIDATED, "", "remove outputs.anytime_bounds"),
        (ASSUMED, "", "remove outputs.anytime_bounds"),
        (TRANSPORTED, "", "remove outputs.anytime_bounds"),
        # Bounding the groups a validated calibration leaves out certifies nothing, since every
        # claim is decided on corrected risk, so it is not the remedy offered.
        (VALIDATED + "    groups: [also]\n", "", "a validated calibration decides every claim"),
        (ASSUMED + "    groups: [all]\n", "    groups: [all]\n", "bound only groups"),
    ],
    ids=["validated", "assumed", "transported", "validated-on-other-groups", "assumed-on-it"],
)
def test_every_judge_calibration_that_covers_a_bounded_group_is_refused(
    tmp_path: Path, judge: str, groups: str, remedy: str
) -> None:
    (refusal,) = _plan_refusals(load_plan(_judge_plan(tmp_path, judge, groups)))
    assert remedy in refusal.message


def test_a_judge_calibration_scoped_to_other_groups_leaves_the_bound(tmp_path: Path) -> None:
    plan = load_plan(
        _judge_plan(tmp_path, ASSUMED + "    groups: [also]\n", "    groups: [all]\n")
    )
    assert not _plan_refusals(plan)


@pytest.mark.parametrize(
    ("order", "shuffled", "refusal"),
    [
        (
            JUDGE_PROMPTS + JUDGE_PROMPTS[:1],
            True,
            ("ANYTIME_BOUND_INVALID", "lists a prompt of the group twice"),
        ),
        (
            JUDGE_PROMPTS,
            False,
            ("ANYTIME_ORDER_UNRECORDED", "records sample_shuffle 7 but not the shuffled"),
        ),
    ],
    ids=["prompt-listed-twice", "seed-without-shuffle"],
)
def test_an_order_inspect_does_not_write_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    order: list[str],
    shuffled: bool,
    refusal: tuple[str, str],
) -> None:
    """Logs that Inspect does not write, read through a replaced order reader: an order that
    lists a prompt twice, and a seed recorded for samples the log says were not shuffled."""

    _read_frames(monkeypatch, _judge_frames(lambda epoch: epoch <= 3), order, shuffled=shuffled)
    report = analyze(plan=load_plan(_judge_plan(tmp_path)), logs="ignored")

    code, message = refusal
    found = _found(report, code)
    assert [item.scope for item in found] == [
        f"cohorts/candidate/groups/{group}/outputs/anytime_bounds" for group in ("all", "also")
    ]
    assert all(message in item.message for item in found)
    assert not _found(report, "ANYTIME_UPPER_BOUND")


def test_a_validated_calibration_on_other_groups_leaves_the_bound_certifying_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run of the test above, with the calibration scoped to the other group and the bound
    on the group it leaves out. True risk is 0.10 and judge-reported risk 0.03. The claim on
    that group has no certifying correction, so it is INCONCLUSIVE; the bound, near 0.04 and at
    or below its acceptable_upper of 0.06, used to report it as certified."""

    _read_frames(
        monkeypatch,
        _judge_frames(lambda epoch: epoch <= 3),
        [str(item) for item in np.random.default_rng(7).permutation(JUDGE_PROMPTS)],
    )
    uncalibrated = analyze(plan=load_plan(_judge_plan(tmp_path / "plain")), logs="ignored")
    labels = tmp_path / "judged" / "labels.csv"
    labels.parent.mkdir()
    labels.write_text(
        "observation_id,human_label\n"
        + "".join(
            f"{row.observation_id},{'unsafe' if row.repeat <= 10 else 'safe'}\n"
            for row in uncalibrated._observations
        ),
        encoding="utf-8",
    )
    plan = load_plan(
        _judge_plan(
            tmp_path / "judged", VALIDATED + "    groups: [also]\n", "    groups: [all]\n"
        )
    )
    assert not _plan_refusals(plan)

    report = analyze(plan=plan, logs="ignored")

    (claim,) = report.claims
    assert claim.status == "INCONCLUSIVE"
    (bound,) = _found(report, "ANYTIME_UPPER_BOUND")
    assert bound.details["group"] == "all" and bound.details["upper"] <= 0.06
    assert [item["certified"] for item in bound.details["claims"]] == [None]
    why = bound.details["certification_withheld"]
    assert why.startswith("the validated judge calibration decides every claim")
    assert f"it certifies no claim, because {why}" in bound.message
