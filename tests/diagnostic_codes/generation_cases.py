"""Cases for the codes of the generative pipeline: its rewrites, its logs and their join.

Each run materializes a small generative run with a kernel and validator that need no
provider, stands in for the Inspect logs the way the generative tests do (by replacing
``evals_df`` and ``samples_df``), analyses it with ``analyze_run`` and writes the report. A
generative report has no integrity status: ``status`` is ``invalid`` exactly when
``generative analyze`` would exit with an error, which it does when ``report.has_errors``.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from inspect_safety_stats.generative.kernels import (
    RewriteOutcome,
    RewriteRequest,
    ValidationVerdict,
)
from inspect_safety_stats.registry import SourceKind
from tests.diagnostic_codes.harness import Case, Outcome, Run
from tests.generative_fixtures import SEEDS_WITH_TEXT, base_plan, materialize

MODULE = "tests.diagnostic_codes.generation_cases"
FAILURES = 6  # the first prompts of every evaluation fail, the rest pass

Rows = list[dict[str, Any]]
# Builds the evaluations and the samples of the logs from the accepted prompt ids.
Logs = Callable[[list[str]], tuple[Rows, Rows]]


@dataclass
class ArmKernel:
    """Writes one prompt per unit, and returns ``invalid`` for every rewrite of ``hard``."""

    hard: str | None = None
    kernel_id: str = "fake-rewriter"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        if request.path_values.get("language") == self.hard:
            return RewriteOutcome(status="invalid", detail="cannot hold the constraint")
        return RewriteOutcome(status="valid", prompt=f"rewrite of {request.sampling_unit_id}")


@dataclass
class ArmValidator:
    """Rejects the first attempt of every rewrite of ``hard``, and accepts the rest."""

    hard: str | None = None
    validator_id: str = "fake-validator"

    def validate(self, request: RewriteRequest, prompt: str) -> ValidationVerdict:
        if request.path_values.get("language") == self.hard and request.attempt == 1:
            return ValidationVerdict(status="invalid", detail="not equivalent")
        return ValidationVerdict(status="valid")


@dataclass
class Restyling:
    """A first attempt that repeats one wording per seed; retries reword per unit."""

    kernel_id: str = "fake-rewriter"

    def rewrite(self, request: RewriteRequest) -> RewriteOutcome:
        if request.attempt == 1:
            mark = "!" * (int(request.path_hash[:2], 16) % 5)
            text = f"EXPLAIN: {request.semantic_seed_id}{mark}"
            return RewriteOutcome(status="valid", prompt=text)
        return RewriteOutcome(
            status="valid",
            prompt=f"Explain {request.semantic_seed_id} for {request.sampling_unit_id}",
        )


# Resolved by materialize_run as "module:attribute", as a user's own kernel would be.
EXHAUSTS_MALTESE = ArmKernel(hard="maltese")
EXHAUSTS_NOTHING = ArmKernel()
REJECTS_MALTESE = ArmValidator(hard="maltese")
REJECTS_NOTHING = ArmValidator()
RESTYLING = Restyling()


def evaluation(
    eval_id: str = "eval-a",
    model: str = "model-a",
    *,
    status: str = "success",
    score: str = "score_harmful",
    epochs: tuple[int, ...] = (1,),
    value: Callable[[int], str] = lambda index: "FAIL" if index < FAILURES else "PASS",
) -> Callable[[list[str]], tuple[Rows, Rows]]:
    """One evaluation of every prompt, ``value(index)`` scoring the prompt at ``index``."""

    def build(prompts: list[str]) -> tuple[Rows, Rows]:
        samples = [
            {"eval_id": eval_id, "id": prompt, "epoch": epoch, score: value(index)}
            for index, prompt in enumerate(prompts)
            for epoch in epochs
        ]
        return [{"eval_id": eval_id, "model": model, "status": status}], samples

    return build


def together(*parts: Logs, change: Callable[[Rows], Rows] = lambda rows: rows) -> Logs:
    """The logs of several evaluations, their samples passed through ``change``."""

    def build(prompts: list[str]) -> tuple[Rows, Rows]:
        evals: Rows = []
        samples: Rows = []
        for part in parts:
            more_evals, more_samples = part(prompts)
            evals += more_evals
            samples += more_samples
        return evals, change(samples)

    return build


def plan() -> dict[str, Any]:
    """A plan of 60 units of one cohort that the fake kernel rewrites without a diagnostic."""

    built = base_plan()
    built["sampling"] = {"method": "direct", "total_rewrites": 60}
    built["rewrite"]["kernel_id"] = "fake-rewriter"
    return built


@dataclass(frozen=True)
class Analysis:
    """A case's report, and the directory it was written to."""

    report: Any
    output: Path


def generative(
    change: Callable[[dict[str, Any]], None] | None = None,
    *,
    logs: Logs | None = None,
    kernel: str = "fake",
    validator: str | None = None,
    cohort: str | None = None,
    slices: tuple[str, ...] = (),
) -> Run:
    """Materialize the plan, changed by ``change``, evaluate it with ``logs``, analyse it."""

    def run(directory: Path) -> Outcome:
        from inspect_safety_stats.generative.ledger import read_rewrites
        from inspect_safety_stats.generative.runner import analyze_run, materialize_run

        directory.mkdir(parents=True)
        edited = copy.deepcopy(plan())
        if change is not None:
            change(edited)
        plan_path = materialize(directory, edited, SEEDS_WITH_TEXT)
        artifacts = directory / "run"
        materialize_run(plan_path, artifacts, kernel=kernel, validator=validator)
        prompts = [
            str(record.prompt_id)
            for record in read_rewrites(artifacts / "rewrites.csv")
            if record.is_evaluable()
        ]
        evals, samples = (logs or evaluation())(prompts)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "inspect_safety_stats.ingest.inspect_logs.evals_df",
                lambda logs, **_: pd.DataFrame(evals),
            )
            patch.setattr(
                "inspect_safety_stats.ingest.inspect_logs.samples_df",
                lambda logs, **_: pd.DataFrame(samples),
            )
            report = analyze_run(plan_path, artifacts, "logs", cohort=cohort, slices=slices)
        output = report.write(directory / "report")
        return Outcome(
            diagnostics=tuple(report.diagnostics),
            # What generative analyze exits with an error on.
            status="invalid" if report.has_errors else "valid",
            report=Analysis(report, output),
        )

    return run


def cohorts(**models: str) -> Callable[[dict[str, Any]], None]:
    def change(edited: dict[str, Any]) -> None:
        edited["target"]["cohorts"] = {name: {"model": model} for name, model in models.items()}

    return change


def validated(validator_id: str = "fake-validator") -> Callable[[dict[str, Any]], None]:
    def change(edited: dict[str, Any]) -> None:
        edited["rewrite"]["validation"] = {"validator_id": validator_id, "invoke": True}

    return change


def deduplicated(**settings: str) -> Callable[[dict[str, Any]], None]:
    def change(edited: dict[str, Any]) -> None:
        edited["rewrite"]["deduplication"] = dict(settings)

    return change


def rollouts(count: int) -> Callable[[dict[str, Any]], None]:
    def change(edited: dict[str, Any]) -> None:
        edited["target"]["rollouts_per_rewrite"] = count

    return change


CALIBRATION = {
    "mode": "assumed",
    "primary": "declared",
    "scenarios": [
        {
            "id": "declared",
            "rho_1": {"lower": 0.85, "upper": 1.0},
            "rho_0": {"lower": 0.0, "upper": 0.02},
        }
    ],
}


def judged(family: bool) -> Callable[[dict[str, Any]], None]:
    def change(edited: dict[str, Any]) -> None:
        edited["judge_calibration"] = copy.deepcopy(CALIBRATION)
        if family:
            edited["inference"] = {
                "slice_family": {"slices": [{"level": "language", "value": "maltese"}]}
            }

    return change


# --- what the codes do to the results ------------------------------------------------------


def report(outcome: Outcome) -> Any:
    return outcome.report.report


def root(outcome: Outcome) -> dict[str, Any]:
    return asdict(report(outcome).tree.root)


def nodes(outcome: Outcome) -> list[dict[str, Any]]:
    return [asdict(node) for node in report(outcome).tree.nodes]


def only(outcome: Outcome, code: str, level: str | None = None) -> Any:
    """The one diagnostic of ``code``, of the hierarchy level ``level`` if given."""

    found = outcome.with_code(code)
    if level is not None:
        found = [item for item in found if item.scope == f"plan/taxonomy/hierarchy/{level}"]
    (item,) = found
    return item


def same_estimates(triggered: Outcome, control: Outcome) -> None:
    # Nothing is recomputed: every node's estimate and interval is the control's.
    assert nodes(triggered) == nodes(control)


def exhaustion_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the rates are in the details and acceptance.csv; exhausted units are
    # completed as unclassified in the join.
    details = only(triggered, "GENERATIVE_ACCEPTANCE_UNEVEN", "language").details
    assert (details["maltese"], details["english"]) == (1.0, 0.0)
    frame = pd.read_csv(triggered.report.output / "acceptance.csv")
    rates = dict(zip(frame["value"], frame["exhaustion_rate"], strict=True))
    assert (rates["maltese"], rates["english"]) == (1.0, 0.0)
    exhausted = int(frame.loc[frame["value"] == "maltese", "exhausted"].iloc[0])
    assert exhausted > 0
    assert root(triggered)["draws"] == root(control)["draws"]
    assert root(triggered)["unclassified"] == exhausted
    assert root(control)["unclassified"] == 0


def rejection_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the rejection rates are in the details and fidelity.csv; nothing else moves.
    details = only(triggered, "GENERATIVE_FIDELITY_UNEVEN", "language").details
    assert details["maltese"] == pytest.approx(0.5, abs=0)
    assert details["english"] == 0.0
    frame = pd.read_csv(triggered.report.output / "fidelity.csv")
    rates = dict(zip(frame["value"], frame["validator_rejection_rate"], strict=True))
    assert rates["maltese"] == pytest.approx(0.5, abs=0)
    assert rates["english"] == 0.0
    same_estimates(triggered, control)


def dependence_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the count of units retried after a duplicate, out of the evaluable units.
    details = only(triggered, "GENERATIVE_RETRY_DEPENDENCE").details
    fidelity = [row for row in report(triggered).fidelity if row.level == "language"]
    assert details["units"] == sum(row.retried_after_duplicate for row in fidelity) > 0
    assert details["evaluable_units"] == root(triggered)["draws"] == 60
    assert all(row.retried_after_duplicate == 0 for row in report(control).fidelity)
    same_estimates(triggered, control)


def validators_named(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the message names the declared validator and the one the ledger records.
    message = only(triggered, "GENERATIVE_VALIDATOR_MISMATCH").message
    assert "'semantic-equivalence-v1'" in message and "'fake-validator'" in message
    same_estimates(triggered, control)


def unmatched_ignored(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the other model's samples, all failures, change nothing.
    same_estimates(triggered, control)
    assert root(triggered)["failures"] == FAILURES


def all_unclassified(triggered: Outcome, control: Outcome) -> None:
    # Every unit is completed as unclassified, where the control classifies them all.
    assert root(triggered)["unclassified"] == root(triggered)["trials"] == 60
    assert root(triggered)["draws"] == root(control)["draws"]
    assert root(control)["unclassified"] == 0


def left_out(triggered: Outcome, control: Outcome) -> None:
    # The offending samples are dropped, so the estimates are the control's.
    same_estimates(triggered, control)


def held_samples_joined(triggered: Outcome, control: Outcome) -> None:
    # The samples the log holds are joined and the units it lacks completed as unclassified,
    # as for the finished log of the control.
    assert (root(triggered)["failures"], root(triggered)["unclassified"]) == (FAILURES, 30)
    same_estimates(triggered, control)


def all_rollouts_trials(triggered: Outcome, control: Outcome) -> None:
    # Each unit enters with both evaluated rollouts as trials, as if two were declared.
    assert root(triggered)["trials"] == root(control)["trials"] == 120
    assert root(triggered)["failures"] == root(control)["failures"]


def completed_and_widened(count: int) -> Callable[[Outcome, Outcome], None]:
    def check(triggered: Outcome, control: Outcome) -> None:
        # Adjusted: the samples become unclassified, completed at both extremes, so the
        # bound widens against the control, which classifies them as safe.
        assert root(triggered)["unclassified"] == count
        assert root(control)["unclassified"] == 0
        assert root(triggered)["draws"] == root(control)["draws"]
        assert root(triggered)["upper"] > root(control)["upper"]
        assert root(triggered)["lower"] <= root(control)["lower"]

    return check


def family_uncorrected(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the family is estimated, but nothing of it is corrected; the root, the nodes
    # and the --slice cut are corrected exactly as without the family.
    family = report(triggered).slice_family
    assert [item.member_id for item in family.slices] == ["language=maltese"]
    corrected = report(triggered).corrected
    assert corrected == report(control).corrected
    assert "slice:language=maltese" in {item.node_id for item in corrected}
    assert root(triggered)["node_id"] in {item.node_id for item in corrected}


def _first_half(rows: Rows) -> Rows:
    return rows[: len(rows) // 2]


def _safe(index: int) -> str:
    return "PASS"


def _unmapped(index: int) -> str:
    return "MAYBE" if index >= 55 else "PASS"


# The kind of input, as inspect_safety_stats.registry.SOURCE_KINDS names them, that these
# cases' triggers break. A case whose trigger breaks another kind names it as its source.
SOURCE: SourceKind = "generation"

CASES = [
    # The kernel exhausts every Maltese unit and no English one.
    Case(
        code="GENERATIVE_ACCEPTANCE_UNEVEN",
        shows="recorded",
        trigger=generative(kernel=f"{MODULE}:EXHAUSTS_MALTESE"),
        control=generative(kernel=f"{MODULE}:EXHAUSTS_NOTHING"),
        effect=exhaustion_recorded,
    ),
    # Two cohorts declare the model the only evaluation ran.
    Case(
        code="GENERATIVE_COHORT_AMBIGUOUS",
        shows="invalid",
        source="plan",
        trigger=generative(cohorts(baseline="model-a", candidate="model-a")),
        control=generative(cohorts(baseline="model-a", candidate="model-b")),
        effect=all_unclassified,
    ),
    # The analysed cohort's model has no evaluation among the logs.
    Case(
        code="GENERATIVE_COHORT_NOT_EVALUATED",
        shows="invalid",
        trigger=generative(
            cohorts(baseline="model-a", candidate="model-b"), cohort="candidate"
        ),
        control=generative(
            cohorts(baseline="model-a", candidate="model-b"),
            logs=together(evaluation(), evaluation("eval-b", "model-b")),
            cohort="candidate",
        ),
        effect=all_unclassified,
    ),
    # A second evaluation ran a model no cohort declares.
    Case(
        code="GENERATIVE_EVAL_UNMATCHED",
        shows="recorded",
        trigger=generative(
            logs=together(evaluation(), evaluation("eval-z", "model-z", value=lambda i: "FAIL"))
        ),
        control=generative(),
        effect=unmatched_ignored,
    ),
    # The validator rejects the first attempt of every Maltese rewrite and of nothing else.
    Case(
        code="GENERATIVE_FIDELITY_UNEVEN",
        shows="recorded",
        trigger=generative(validated(), validator=f"{MODULE}:REJECTS_MALTESE"),
        control=generative(validated(), validator=f"{MODULE}:REJECTS_NOTHING"),
        effect=rejection_recorded,
    ),
    # The evaluation stopped with status error; the control's same log finished.
    Case(
        code="GENERATIVE_LOG_INCOMPLETE",
        shows="invalid",
        trigger=generative(
            logs=together(evaluation(status="error"), change=_first_half)
        ),
        control=generative(logs=together(evaluation(), change=_first_half)),
        effect=held_samples_joined,
    ),
    # Near-duplicate first attempts are rejected and retried; the control accepts them.
    Case(
        code="GENERATIVE_RETRY_DEPENDENCE",
        shows="recorded",
        trigger=generative(
            deduplicated(near_duplicate="normalized_text"), kernel=f"{MODULE}:RESTYLING"
        ),
        control=generative(
            deduplicated(near_duplicate="normalized_text", on_duplicate="accept"),
            kernel=f"{MODULE}:RESTYLING",
        ),
        effect=dependence_recorded,
    ),
    # Two epochs were evaluated where the plan declares one rollout per rewrite.
    Case(
        code="GENERATIVE_ROLLOUT_OVERRUN",
        shows="invalid",
        trigger=generative(rollouts(1), logs=evaluation(epochs=(1, 2))),
        control=generative(rollouts(2), logs=evaluation(epochs=(1, 2))),
        effect=all_rollouts_trials,
    ),
    # A retried log of the same model holds every sample again, now failing.
    Case(
        code="GENERATIVE_SAMPLE_DUPLICATE",
        shows="invalid",
        trigger=generative(
            logs=together(evaluation(), evaluation("eval-retry", value=lambda i: "FAIL"))
        ),
        control=generative(),
        effect=left_out,
    ),
    # The cohort's log holds a sample no accepted rewrite wrote.
    Case(
        code="GENERATIVE_SAMPLE_UNMATCHED",
        shows="invalid",
        trigger=generative(
            logs=together(
                evaluation(),
                change=lambda rows: [
                    *rows,
                    {"eval_id": "eval-a", "id": "elsewhere", "epoch": 1, "score_harmful": "FAIL"},
                ],
            )
        ),
        control=generative(),
        effect=left_out,
    ),
    # The logs were scored by another scorer than target.score.name.
    Case(
        code="GENERATIVE_SCORE_ABSENT",
        shows="invalid",
        trigger=generative(logs=evaluation(score="score_other")),
        control=generative(),
        effect=all_unclassified,
    ),
    # Five samples carry a score value the mapping does not list.
    Case(
        code="GENERATIVE_SCORE_UNMAPPED",
        shows="adjusted",
        trigger=generative(logs=evaluation(value=_unmapped)),
        control=generative(logs=evaluation(value=_safe)),
        effect=completed_and_widened(5),
    ),
    # Five accepted rewrites were never evaluated.
    Case(
        code="GENERATIVE_UNIT_UNEVALUATED",
        shows="adjusted",
        trigger=generative(logs=together(evaluation(value=_safe), change=lambda rows: rows[5:])),
        control=generative(logs=evaluation(value=_safe)),
        effect=completed_and_widened(5),
    ),
    # The plan names another validator than the one that judged the rewrites.
    Case(
        code="GENERATIVE_VALIDATOR_MISMATCH",
        shows="recorded",
        trigger=generative(validated("semantic-equivalence-v1"), validator="fake"),
        control=generative(validated("fake-validator"), validator="fake"),
        effect=validators_named,
    ),
    # A judge calibration is declared together with a slice family.
    Case(
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_SLICE_FAMILY",
        shows="withheld",
        source="analysis",
        trigger=generative(judged(family=True), slices=("language=maltese",)),
        control=generative(judged(family=False), slices=("language=maltese",)),
        effect=family_uncorrected,
    ),
]
