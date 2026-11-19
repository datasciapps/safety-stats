"""Cases for the codes whose source is the logs or rollout files, and the manifest they fill.

Most runs change the base rollout files. Content-filter and token-limit stops need a source
that reports how each output stopped, which rollout files do not: those runs read the rollout
files through a source that adds a ``stop_reason`` field's stop. The codes that need a real
Inspect log (an unreadable transcript, a recorded sample order) run a small evaluation on
Inspect's offline mock model.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from inspect_safety_stats.ingest.records import RolloutRecord, SourceData, StopContext
from tests.diagnostic_codes.harness import (
    BASE_RUN,
    Case,
    Inputs,
    Outcome,
    Run,
    analysis,
    base_inputs,
    claim,
    rollout,
    write_inputs,
)

Change = Callable[[Inputs], None]
Check = Callable[[Outcome, Outcome], None]


def _changes(*changes: Change) -> Change:
    def change(inputs: Inputs) -> None:
        for item in changes:
            item(inputs)

    return change


# --- what the cases look at ------------------------------------------------------------------


def _estimate(outcome: Outcome, prompt: str, cohort: str = "candidate") -> Any:
    (found,) = [
        item
        for item in outcome.report.prompt_estimates
        if item.prompt_id == prompt and item.cohort_id == cohort
    ]
    return found


def _withheld(outcome: Outcome, prompt: str, cohort: str = "candidate") -> bool:
    estimate = _estimate(outcome, prompt, cohort)
    return estimate.lower is None and estimate.upper is None


def _risk(outcome: Outcome, cohort: str = "candidate", group: str = "all") -> Any:
    (found,) = [
        item
        for item in outcome.report.group_risks
        if item.cohort_id == cohort and item.group_id == group
    ]
    return found


def _only_prompt_withheld(prompt: str) -> Check:
    """The prompt's bounds are withheld and the other prompt's are reported as before."""

    other = "p2" if prompt == "p1" else "p1"

    def check(triggered: Outcome, control: Outcome) -> None:
        assert _withheld(triggered, prompt) and not _withheld(control, prompt)
        assert _estimate(triggered, other) == _estimate(control, other)

    return check


def _bounds_withheld_everywhere(triggered: Outcome, control: Outcome) -> None:
    assert all(_withheld(triggered, prompt) for prompt in ("p1", "p2"))
    assert not any(_withheld(control, prompt) for prompt in ("p1", "p2"))


def _same_results(triggered: Outcome, control: Outcome) -> None:
    """Nothing is computed differently: the same estimates, risks and claims."""

    assert triggered.report.prompt_estimates == control.report.prompt_estimates
    assert triggered.report.group_risks == control.report.group_risks
    assert claim(triggered).model_dump() == claim(control).model_dump()


def _same_estimates(triggered: Outcome, control: Outcome) -> None:
    """The rollouts are analysed as before: the prompt estimates do not change."""

    assert triggered.report.prompt_estimates == control.report.prompt_estimates


# --- rollout-level changes ------------------------------------------------------------------


def _plan_score(**policy: str) -> Change:
    def change(inputs: Inputs) -> None:
        inputs.plan["inputs"]["score"].update(policy)

    return change


def _first(value: Any = "PASS", **fields: Any) -> Change:
    """Replace the first rollout of p1 with one of score ``value`` and ``fields``."""

    def change(inputs: Inputs) -> None:
        inputs.rollouts[0] = rollout("p1", 1, value, **fields)

    return change


def _unclassified_p1(triggered: Outcome, control: Outcome) -> None:
    # The rollout's outcome is unknown, so p1's bounds cover both outcomes: still reported,
    # and wider than with the rollout classified.
    estimate, before = _estimate(triggered, "p1"), _estimate(control, "p1")
    assert (estimate.n_unclassified, before.n_unclassified) == (1, 0)
    assert estimate.upper is not None and estimate.upper > before.upper


# --- cohorts and runs ------------------------------------------------------------------------


def _baseline_cohort(model: str, *, rollouts: bool = True) -> Change:
    """Add cohort ``baseline`` matching ``model``, and run r2 of model ``b``, a copy of r1."""

    def change(inputs: Inputs) -> None:
        inputs.plan["cohorts"]["baseline"] = {"match": {"model": model}}
        inputs.runs.append({**BASE_RUN, "run_key": "r2", "model": "b"})
        if rollouts:
            inputs.rollouts += [{**item, "run_key": "r2"} for item in inputs.rollouts]

    return change


def _empty_baseline(triggered: Outcome, control: Outcome) -> None:
    # The empty cohort's bounds are withheld; the other cohort's are reported as before.
    assert all(_withheld(triggered, prompt, "baseline") for prompt in ("p1", "p2"))
    assert not any(_withheld(control, prompt, "baseline") for prompt in ("p1", "p2"))
    for prompt in ("p1", "p2"):
        assert _estimate(triggered, prompt) == _estimate(control, prompt)


def _two_runs(temperatures: tuple[int, int]) -> Change:
    """Split the cohort over runs r1 and r2, which record a uniform-required temperature."""

    def change(inputs: Inputs) -> None:
        inputs.plan["cohorts"]["candidate"]["require_uniform"] = ["attributes.temperature"]
        inputs.runs[0]["temperature"] = temperatures[0]
        inputs.runs.append({**BASE_RUN, "run_key": "r2", "temperature": temperatures[1]})
        for item in inputs.rollouts:
            if item["repeat"] > 2:
                item["run_key"] = "r2"

    return change


def _drift_withholds(triggered: Outcome, control: Outcome) -> None:
    _bounds_withheld_everywhere(triggered, control)
    assert _risk(triggered).lower is None and _risk(control).lower is not None


def _match_by_run_key(inputs: Inputs) -> None:
    inputs.plan["cohorts"]["candidate"]["match"] = {"run_key": "r1"}


def _no_model(inputs: Inputs) -> None:
    del inputs.runs[0]["model"]


def _require_temperature(inputs: Inputs) -> None:
    inputs.plan["cohorts"]["candidate"]["require_uniform"] = ["attributes.temperature"]


def _record_temperature(inputs: Inputs) -> None:
    inputs.runs[0]["temperature"] = 0


def _run_without(field: str) -> Change:
    def change(inputs: Inputs) -> None:
        del inputs.runs[0][field]

    return change


def _status_error(inputs: Inputs) -> None:
    inputs.runs[0]["status"] = "error"


def _incomplete_run_still_analysed(triggered: Outcome, control: Outcome) -> None:
    # The run's rollouts enter the observations and no bound is withheld, but no claim passes.
    _same_estimates(triggered, control)
    assert (claim(triggered).status, claim(control).status) == ("INCONCLUSIVE", "PASS")


def _unreadable_rollout(inputs: Inputs) -> None:
    inputs.rollouts[0]["repeat"] = 0


def _nothing_reported(triggered: Outcome, control: Outcome) -> None:
    report = triggered.report
    assert (report.prompt_estimates, report.group_risks, report.comparisons) == ([], [], [])
    assert control.report.prompt_estimates != []
    assert claim(triggered).status == "INCONCLUSIVE"


# --- judges ----------------------------------------------------------------------------------

MAPPING = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}


def _judged(second_score: str, value: str = "PASS") -> Change:
    """Judges j1 (on ``harmful``) and j2 (on ``second_score``), and the primary score their
    ``any`` composite. Every rollout's ``other`` score is PASS, but the first one's ``value``."""

    def change(inputs: Inputs) -> None:
        inputs.plan["inputs"]["judges"] = [
            {"id": "j1", "score": {"name": "harmful", "mapping": MAPPING}},
            {"id": "j2", "score": {"name": second_score, "mapping": MAPPING}},
        ]
        inputs.plan["inputs"]["composites"] = [
            {"id": "either", "rule": "any", "judges": ["j1", "j2"]}
        ]
        # A composite's verdict is a boolean: True when it finds a failure.
        inputs.plan["inputs"]["score"] = {
            "name": "either",
            "mapping": {"failure_values": [True], "safe_values": [False]},
        }
        for item in inputs.rollouts:
            item["score_other"] = "PASS"
        inputs.rollouts[0]["score_other"] = value

    return change


def _missing_votes(count: int) -> Check:
    """The judge's verdict counts as missing, so the composite is unknown for ``count`` p1
    rollouts that j1 finds safe; p1's bounds are still reported."""

    def check(triggered: Outcome, control: Outcome) -> None:
        estimate = _estimate(triggered, "p1")
        assert (estimate.n_unclassified, _estimate(control, "p1").n_unclassified) == (count, 0)
        assert estimate.lower is not None and estimate.upper is not None

    return check


# --- manifest --------------------------------------------------------------------------------


def _no_expected_rollouts(inputs: Inputs) -> None:
    for row in inputs.manifest:
        del row["expected_rollouts"]


def _expected_rollouts_not_required(inputs: Inputs) -> None:
    inputs.plan["design"] = {"require_expected_rollouts": False}


def _drop_p2_rollouts(inputs: Inputs) -> None:
    inputs.rollouts = [item for item in inputs.rollouts if item["item_id"] != "p2"]


def _drop_p2_from_manifest(inputs: Inputs) -> None:
    inputs.manifest = [row for row in inputs.manifest if row["prompt_id"] != "p2"]


def _missing_prompt_withheld(triggered: Outcome, control: Outcome) -> None:
    # p2's bounds are withheld, and so is the group risk, which needs every prompt's ledger.
    assert _withheld(triggered, "p2")
    assert _risk(triggered).lower is None and _risk(control).lower is not None


def _p3_rollouts(inputs: Inputs) -> None:
    inputs.rollouts += [rollout("p3", repeat) for repeat in range(1, 5)]


def _p1_expects(count: int) -> Change:
    def change(inputs: Inputs) -> None:
        inputs.manifest[0]["expected_rollouts"] = count

    return change


def _p1_rollout(repeat: int) -> Change:
    def change(inputs: Inputs) -> None:
        inputs.rollouts.append(rollout("p1", repeat, rollout_id="extra"))

    return change


def _duplicate_withholds(triggered: Outcome, control: Outcome) -> None:
    _only_prompt_withheld("p1")(triggered, control)
    # The duplicate stays in the observations.
    assert _estimate(triggered, "p1").n_total == 5


def _drop_p1_repeat_4(inputs: Inputs) -> None:
    inputs.rollouts = [
        item for item in inputs.rollouts if (item["item_id"], item["repeat"]) != ("p1", 4)
    ]


# --- comparisons -----------------------------------------------------------------------------


def _paired(inputs: Inputs) -> None:
    """Cohort baseline (run r2, a copy of r1) compared with candidate on group t1, which
    holds only p1, with paired diagnostics on pair_key [prompt_id, repeat]."""

    _baseline_cohort("b")(inputs)
    inputs.manifest[1]["task_id"] = "t2"
    inputs.plan["groups"].append({"id": "t1", "filter": {"task_id": "t1"}})
    inputs.plan["design"] = {
        "independence_profiles": {
            "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"}
        }
    }
    inputs.plan["multiplicity"]["families"].append(
        {
            "id": "comparisons",
            "purpose": "simultaneous_intervals",
            "method": "bonferroni",
            "alpha": 0.05,
        }
    )
    inputs.plan["comparisons"] = [
        {
            "id": "cmp",
            "candidate": "candidate",
            "baseline": "baseline",
            "group": "t1",
            "measure": "mean",
            "family": "comparisons",
            "independence": {"primary": "rollout"},
            "pair_key": ["prompt_id", "repeat"],
            "diagnostic_methods": ["exact_mcnemar", "newcombe_paired"],
        }
    ]


def _baseline_rollout(prompt: str, repeat: int, fields: dict[str, Any]) -> Change:
    """Update the fields of the baseline's rollout of ``prompt`` and ``repeat``."""

    def change(inputs: Inputs) -> None:
        (found,) = [
            item
            for item in inputs.rollouts
            if (item["run_key"], item["item_id"], item["repeat"]) == ("r2", prompt, repeat)
        ]
        found.update(fields)

    return change


def _pairing_lost(triggered: Outcome, control: Outcome) -> None:
    (broken,), (kept,) = triggered.report.comparisons, control.report.comparisons
    assert (broken.pair_count, broken.diagnostic_outputs) == (None, {})
    assert kept.pair_count == 4 and "exact_mcnemar" in kept.diagnostic_outputs


def _pairs_bounded(triggered: Outcome, control: Outcome) -> None:
    # The paired diagnostics report estimate_lower and estimate_upper in place of an estimate.
    (bounded,), (scored,) = triggered.report.comparisons, control.report.comparisons
    for method in ("exact_mcnemar", "newcombe_paired"):
        output, before = bounded.diagnostic_outputs[method], scored.diagnostic_outputs[method]
        assert "estimate" not in output and "estimate" in before
        assert output["estimate_lower"] <= before["estimate"] <= output["estimate_upper"]
        assert output["estimate_lower"] < output["estimate_upper"]


# --- sampled designs -------------------------------------------------------------------------


def _sampled(probabilities: tuple[float, float], *, fixed_size: bool = False) -> Change:
    """A sampled design over the manifest frame with these inclusion probabilities, in which
    only p1 was drawn."""

    def change(inputs: Inputs) -> None:
        inputs.plan["design"] = {
            "type": "sampled",
            "inclusion_probability_column": "inclusion",
            "independence_profiles": {
                "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"}
            },
        }
        if fixed_size:
            inputs.plan["design"]["sampling"] = "fixed_size"
        inputs.plan["outputs"]["prompt_bounds"] = {"enabled": False}
        inputs.plan["outputs"]["direct_mean_intervals"] = {
            "enabled": True,
            "method": "weighted_kl_chernoff",
            "family": "displayed",
            "independence": {"primary": "rollout"},
        }
        for row, probability in zip(inputs.manifest, probabilities, strict=True):
            row["inclusion"] = probability
        inputs.rollouts = [item for item in inputs.rollouts if item["item_id"] == "p1"]

    return change


def _draw_p2_too(inputs: Inputs) -> None:
    inputs.rollouts += [rollout("p2", repeat) for repeat in range(1, 5)]


def _sampled_two_cohorts(*, empty: bool) -> Change:
    """A Poisson sample in cohorts candidate and baseline; with ``empty`` candidate's run r1
    has no rollouts."""

    def change(inputs: Inputs) -> None:
        _sampled((0.5, 0.5))(inputs)
        _baseline_cohort("b")(inputs)
        if empty:
            inputs.rollouts = [item for item in inputs.rollouts if item["run_key"] == "r2"]

    return change


def _cohort_risk_withheld(triggered: Outcome, control: Outcome) -> None:
    risk, before = _risk(triggered), _risk(control)
    assert (risk.lower, risk.upper) == (None, None)
    assert before.lower is not None and before.upper is not None


# --- stops: a source that reports how each output stopped ------------------------------------


@dataclass(frozen=True)
class StoppedRollouts:
    """Rollout files whose rollouts may carry a ``stop_reason``: ``content_filter`` for a
    provider block that delivered no content, ``max_tokens`` for the generation token limit."""

    runs: Path
    rollouts: Path

    def read(self) -> SourceData:
        from inspect_safety_stats.ingest.rollout_files import read_rollout_files

        def stop(item: RolloutRecord) -> StopContext | None:
            reason = item.attributes.get("stop_reason")
            if reason is None:
                return None
            return StopContext(
                content_delivered=False,
                block_type=None,
                block_category=None,
                block_error=False,
                stop_reason=reason,
            )

        return replace(read_rollout_files(self.runs, self.rollouts), stop_context=stop)


def _report_outcome(report: Any) -> Outcome:
    return Outcome(
        diagnostics=tuple(report.integrity.diagnostics),
        status=report.integrity.status,
        report=report,
        assumptions=frozenset(item.id for item in report.assumptions_used),
    )


def _stopped(change: Change) -> Run:
    """The main analysis on the changed base inputs, read with ``StoppedRollouts``."""

    def run(directory: Path) -> Outcome:
        from inspect_safety_stats import analyze

        inputs = base_inputs()
        change(inputs)
        plan = write_inputs(inputs, directory)
        source = StoppedRollouts(directory / "runs.jsonl", directory / "rollouts.jsonl")
        return _report_outcome(analyze(plan=plan, source=source))

    return run


def _blocked_counts_as_failure(triggered: Outcome, control: Outcome) -> None:
    # The PASS-scored rollout whose output the filter stopped counts as a failure.
    failures = (_estimate(triggered, "p1").n_failure, _estimate(control, "p1").n_failure)
    assert failures == (1, 0)


def _blocked_counts_as_safe(triggered: Outcome, control: Outcome) -> None:
    # The FAIL-scored rollout whose sample error was a policy block counts as safe.
    estimate, before = _estimate(triggered, "p1"), _estimate(control, "p1")
    assert (estimate.n_failure, before.n_failure) == (0, 1)
    assert (estimate.n_policy_blocked_safe, before.n_policy_blocked_safe) == (1, 0)


def _truncation_counted(triggered: Outcome, control: Outcome) -> None:
    # The unscored truncated answer counts as a failure, as the one scored FAIL does, and is
    # counted in n_token_limit.
    estimate, before = _estimate(triggered, "p1"), _estimate(control, "p1")
    assert (estimate.n_failure, estimate.n_valid) == (before.n_failure, before.n_valid) == (1, 4)
    assert (estimate.n_token_limit, before.n_token_limit) == (1, 0)


# --- real Inspect logs -----------------------------------------------------------------------

MOCK_MODEL = "mockllm/model"
SHUFFLE_SEED = 7


def _inspect_inputs(change: Change | None = None) -> Inputs:
    """The base inputs for a cohort of Inspect runs on the mock model."""

    inputs = base_inputs()
    inputs.plan["cohorts"]["candidate"]["match"] = {"model": MOCK_MODEL}
    if change is not None:
        change(inputs)
    return inputs


def _inspect_run(
    log_dir: Path, *, epochs: int = 4, shuffle: int | None = None, error: bool = False
) -> None:
    """Evaluate p1 and p2 on the mock model, scored as the base rollouts are: p2 fails its
    first epoch. With ``error`` the first epoch of p1 ends in a sample error."""

    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import Generate, TaskState, generate, solver

    @scorer(metrics=[], name="harmful")
    def harmful():  # type: ignore[no-untyped-def]
        async def score(state: TaskState, target: Target) -> Score:
            failed = state.sample_id == "p2" and state.epoch == 1
            return Score(value="FAIL" if failed else "PASS")

        return score

    @solver
    def flaky():  # type: ignore[no-untyped-def]
        async def solve(state: TaskState, generate_: Generate) -> TaskState:
            if error and state.sample_id == "p1" and state.epoch == 1:
                raise RuntimeError("the provider went away")
            return state

        return solve

    task = Task(
        dataset=[Sample(id=prompt, input=f"question {prompt}") for prompt in ("p1", "p2")],
        solver=[flaky(), generate()],
        scorer=harmful(),
    )
    (log,) = inspect_eval(
        task,
        model=MOCK_MODEL,
        epochs=epochs,
        log_dir=str(log_dir),
        sample_shuffle=shuffle,
        fail_on_error=False,
        display="none",
    )
    assert log.status == "success"


def on_inspect_logs(
    runs: list[dict[str, Any]],
    change: Change | None = None,
    source: Callable[[Path], Any] | None = None,
) -> Run:
    """Analyse the Inspect logs of ``runs``, each given as ``_inspect_run``'s options, read
    from the logs, or from ``source`` of their directory."""

    def run(directory: Path) -> Outcome:
        from inspect_safety_stats import analyze

        directory.mkdir(parents=True, exist_ok=True)
        # Inspect drops the drive letter of a log on another drive (#35): work from this one.
        with contextlib.chdir(directory):
            log_dir = directory / "logs"
            for options in runs:
                _inspect_run(log_dir, **options)
            plan = write_inputs(_inspect_inputs(change), directory)
            if source is None:
                report = analyze(plan=plan, logs=str(log_dir))
            else:
                report = analyze(plan=plan, source=source(log_dir))
        return _report_outcome(report)

    return run


@dataclass(frozen=True)
class VanishingLogs:
    """Inspect logs that, with ``vanish``, are removed once their runs and samples are read,
    before the transcripts of their sample errors are."""

    log_dir: Path
    vanish: bool

    def read(self) -> SourceData:
        from inspect_safety_stats.ingest.inspect_logs import InspectLogs

        data = InspectLogs(str(self.log_dir)).read()
        if self.vanish:
            for log in self.log_dir.iterdir():
                log.unlink()
        return data


def _vanishing(vanish: bool) -> Run:
    return on_inspect_logs(
        [{"error": True}],
        _plan_score(errors="count_as_failure"),
        lambda log_dir: VanishingLogs(log_dir, vanish),
    )


def anytime_bounds(inputs: Inputs) -> None:
    inputs.plan["design"] = {
        "independence_profiles": {
            "prompt": {"block_by": ["prompt_id"], "comparison_coupling": "independent"}
        }
    }
    inputs.plan["outputs"]["anytime_bounds"] = {
        "independence_profile": "prompt",
        "family": "claims",
    }


def no_anytime_bound(triggered: Outcome, control: Outcome) -> None:
    bounds = [len(item.with_code("ANYTIME_UPPER_BOUND")) for item in (triggered, control)]
    assert bounds == [0, 1]


CASES = [
    Case(
        code="COHORT_EMPTY",
        # Cohort baseline matches a model no run has; in the control run r2 has it.
        trigger=analysis(_baseline_cohort("absent", rollouts=False)),
        control=analysis(_baseline_cohort("b")),
        effect=_empty_baseline,
    ),
    Case(
        code="CONFIG_DRIFT",
        # The cohort's two runs record different values of a uniform-required field.
        trigger=analysis(_two_runs((0, 1))),
        control=analysis(_two_runs((0, 0))),
        effect=_drift_withholds,
    ),
    Case(
        code="CONFIG_UNOBSERVED",
        # The run, matched by its key, records no model, its only configuration field.
        trigger=analysis(_changes(_match_by_run_key, _no_model)),
        control=analysis(_match_by_run_key),
        effect=_same_estimates,
    ),
    Case(
        code="INGEST_FAILED",
        # A rollout's repeat of 0 makes the rollout files unreadable.
        trigger=analysis(_unreadable_rollout),
        control=analysis(),
        effect=_nothing_reported,
    ),
    Case(
        code="JUDGE_SCORE_MISSING",
        # Judge j2 names a score the rollouts do not carry.
        trigger=analysis(_judged("absent")),
        control=analysis(_judged("other")),
        effect=_missing_votes(4),
    ),
    Case(
        code="JUDGE_SCORE_UNMAPPED",
        # Judge j2's score takes a value its mapping does not cover.
        trigger=analysis(_judged("other", "MAYBE")),
        control=analysis(_judged("other")),
        effect=_missing_votes(1),
    ),
    Case(
        code="LOG_INCOMPLETE",
        # The run's status is error.
        trigger=analysis(_status_error),
        control=analysis(),
        effect=_incomplete_run_still_analysed,
    ),
    Case(
        code="PAIR_BROKEN",
        # The baseline's p1 repeat 4 is numbered 5: key (p1, 4) and (p1, 5) are in one arm.
        trigger=analysis(_changes(_paired, _baseline_rollout("p1", 4, {"repeat": 5}))),
        control=analysis(_paired),
        effect=_pairing_lost,
    ),
    Case(
        code="PAIR_UNSCORED_COMPLETED",
        # A baseline rollout of p1, in the compared group, is unscored; in the control the
        # unscored rollout is of p2, outside it.
        trigger=analysis(_changes(_paired, _baseline_rollout("p1", 1, {"score_harmful": None}))),
        control=analysis(_changes(_paired, _baseline_rollout("p2", 1, {"score_harmful": None}))),
        effect=_pairs_bounded,
    ),
    Case(
        code="PLAN_EXPECTED_ROLLOUTS_MISSING",
        # The manifest has no expected_rollouts column while the plan requires it.
        trigger=analysis(_no_expected_rollouts),
        control=analysis(_changes(_no_expected_rollouts, _expected_rollouts_not_required)),
        effect=_bounds_withheld_everywhere,
    ),
    Case(
        code="PROMPT_MISSING",
        # Manifest prompt p2 has no rollouts; the control leaves it out of the manifest too.
        trigger=analysis(_drop_p2_rollouts),
        control=analysis(_changes(_drop_p2_rollouts, _drop_p2_from_manifest)),
        effect=_missing_prompt_withheld,
    ),
    Case(
        code="PROMPT_UNEXPECTED",
        # The rollouts evaluate p3, which the manifest does not declare.
        trigger=analysis(_p3_rollouts),
        control=analysis(),
        # p3's rollouts are left out; the manifest prompts' bounds are not withheld.
        effect=_same_estimates,
    ),
    Case(
        code="PROVENANCE_UNOBSERVED",
        # The cohort requires a uniform temperature that its run does not record.
        trigger=analysis(_require_temperature),
        control=analysis(_changes(_require_temperature, _record_temperature)),
        effect=_same_results,
    ),
    Case(
        code="ROLLOUT_DUPLICATE",
        # A fifth rollout of p1 repeats the key (r1, p1, 1); in the control it is repeat 5.
        trigger=analysis(_changes(_p1_expects(5), _p1_rollout(1))),
        control=analysis(_changes(_p1_expects(5), _p1_rollout(5))),
        effect=_duplicate_withholds,
    ),
    Case(
        code="ROLLOUT_EXCESS",
        # p1 has a fifth rollout against 4 expected; the control expects 5.
        trigger=analysis(_p1_rollout(5)),
        control=analysis(_changes(_p1_rollout(5), _p1_expects(5))),
        effect=_only_prompt_withheld("p1"),
    ),
    Case(
        code="ROLLOUT_SHORTFALL",
        # p1 has 3 rollouts against 4 expected; the control expects 3.
        trigger=analysis(_drop_p1_repeat_4),
        control=analysis(_changes(_drop_p1_repeat_4, _p1_expects(3))),
        effect=_only_prompt_withheld("p1"),
    ),
    Case(
        code="RUN_STATUS_UNOBSERVED",
        # The run records no status.
        trigger=analysis(_run_without("status")),
        control=analysis(),
        effect=_same_estimates,
    ),
    Case(
        code="SAMPLE_DRAWS_MISMATCH",
        # A fixed-size design plans one draw from the frame, and both items were evaluated.
        trigger=analysis(_changes(_sampled((0.5, 0.5), fixed_size=True), _draw_p2_too)),
        control=analysis(_sampled((0.5, 0.5), fixed_size=True)),
    ),
    Case(
        code="SAMPLE_DRAWS_NOT_INTEGER",
        # Inclusion probability 0.25 for both frame items plans half a draw.
        trigger=analysis(_sampled((0.25, 0.25), fixed_size=True)),
        control=analysis(_sampled((0.5, 0.5), fixed_size=True)),
    ),
    Case(
        code="SAMPLE_EMPTY",
        # Cohort candidate's run has no rollouts of any frame item.
        trigger=analysis(_sampled_two_cohorts(empty=True)),
        control=analysis(_sampled_two_cohorts(empty=False)),
        effect=_cohort_risk_withheld,
    ),
    Case(
        code="SAMPLE_ERROR",
        # A rollout ends in a sample error under errors: block.
        trigger=analysis(_changes(_plan_score(errors="block"), _first(None, error="boom"))),
        control=analysis(_plan_score(errors="block")),
        effect=_only_prompt_withheld("p1"),
    ),
    Case(
        code="SAMPLE_ERROR_UNCLASSIFIED",
        # A rollout ends in a sample error of unknown stage under errors: end_to_end.
        trigger=analysis(_first(None, error="boom")),
        control=analysis(),
        effect=_unclassified_p1,
    ),
    Case(
        code="SAMPLE_SIZE_IMPLAUSIBLE",
        # One item drawn where inclusion probabilities of 0.01 expect 0.02 (sd 0.14).
        trigger=analysis(_sampled((0.01, 0.01))),
        control=analysis(_sampled((0.5, 0.5))),
        effect=_cohort_risk_withheld,
    ),
    Case(
        code="SAMPLE_STRATUM_INCONSISTENT",
        # The frame's one stratum has inclusion probabilities 0.5 and 0.6.
        trigger=analysis(_sampled((0.5, 0.6), fixed_size=True)),
        control=analysis(_sampled((0.5, 0.5), fixed_size=True)),
    ),
    Case(
        code="SCORE_MISSING",
        # A rollout has no score under missing: block.
        trigger=analysis(_changes(_plan_score(missing="block"), _first(None))),
        control=analysis(_plan_score(missing="block")),
        effect=_only_prompt_withheld("p1"),
    ),
    Case(
        code="SCORE_MISSING_BOUNDED",
        # A rollout has no score under missing: conservative_bounds.
        trigger=analysis(_first(None)),
        control=analysis(),
        effect=_unclassified_p1,
    ),
    Case(
        code="SCORE_UNMAPPED",
        # A rollout's score is a value the mapping does not classify.
        trigger=analysis(_first("MAYBE")),
        control=analysis(),
        effect=_only_prompt_withheld("p1"),
    ),
    Case(
        code="START_TIME_UNOBSERVED",
        # The run records no start time.
        trigger=analysis(_run_without("started_at")),
        control=analysis(),
        effect=_same_estimates,
    ),
    Case(
        code="TARGET_POLICY_BLOCKED_REFUSAL",
        # A PASS-scored output was stopped by a content filter before it delivered content,
        # under content_filter_stops: failure.
        trigger=_stopped(
            _changes(
                _plan_score(content_filter_stops="failure", errors="conservative_bounds"),
                _first("PASS", stop_reason="content_filter"),
            )
        ),
        control=_stopped(
            _plan_score(content_filter_stops="failure", errors="conservative_bounds")
        ),
        effect=_blocked_counts_as_failure,
    ),
    Case(
        code="TARGET_POLICY_BLOCKED_SAFE",
        # A FAIL-scored rollout ends in a target-side policy-block error that delivered
        # nothing, under errors: end_to_end; the control has no error.
        trigger=analysis(
            _first(
                "FAIL",
                error="blocked",
                error_stage="target_generation",
                content_delivered=False,
                provider_error_code="content_filter",
            )
        ),
        control=analysis(_first("FAIL")),
        effect=_blocked_counts_as_safe,
    ),
    Case(
        code="TOKEN_LIMIT_REACHED",
        # An unscored answer stopped at the token limit, under token_limit: failure; in the
        # control the truncated answer was scored FAIL.
        trigger=_stopped(
            _changes(_plan_score(token_limit="failure"), _first(None, stop_reason="max_tokens"))
        ),
        control=_stopped(
            _changes(_plan_score(token_limit="failure"), _first("FAIL", stop_reason="max_tokens"))
        ),
        effect=_truncation_counted,
    ),
    Case(
        code="TRANSCRIPT_UNREAD",
        # The log of a run with a sample error is gone when its transcript is read.
        trigger=_vanishing(True),
        control=_vanishing(False),
        effect=_same_estimates,
    ),
    Case(
        code="ANYTIME_ORDER_UNRECORDED",
        # The run did not shuffle its samples; the control shuffled them with a seed.
        trigger=on_inspect_logs([{}], anytime_bounds),
        control=on_inspect_logs([{"shuffle": SHUFFLE_SEED}], anytime_bounds),
        effect=no_anytime_bound,
    ),
    Case(
        code="ANYTIME_BOUND_INVALID",
        # The cohort's rollouts come from two seeded runs of two epochs each, so no one run's
        # order covers them; the control is one seeded run of four epochs.
        trigger=on_inspect_logs(
            [{"shuffle": SHUFFLE_SEED, "epochs": 2}, {"shuffle": SHUFFLE_SEED + 1, "epochs": 2}],
            anytime_bounds,
        ),
        control=on_inspect_logs([{"shuffle": SHUFFLE_SEED}], anytime_bounds),
        effect=no_anytime_bound,
    ),
]
