from __future__ import annotations

import json
import sys
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer

from inspect_safety_stats.version import __version__

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from inspect_safety_stats.claim_planning import EndpointAlpha
    from inspect_safety_stats.ingest.records import EvaluationSource
    from inspect_safety_stats.schema.summary import SummaryWarning

# Each command imports what it needs when it runs, so that `version`, `--help`, `plan` and
# `schema` do not load Inspect AI, pandas and the analysis pipeline.

app = typer.Typer(help="Fixed-design statistical risk analysis for Inspect AI evaluations.")
plan_app = typer.Typer(help="Prospective evaluation planning tools.")
schema_app = typer.Typer(help="Print machine-readable schemas.")
# The caveat every entry point to the generative surface carries, word for word (#24; the
# docs' copies are checked by tests/test_repository_consistency.py).
GENERATIVE_CAVEAT = (
    "The generative pipeline is experimental: its public surface, "
    "`inspect_safety_stats.generative.__all__`, may change in any minor release."
)
generative_app = typer.Typer(
    help=(
        "Generative-design evaluations, where prompts are sampled from a declared "
        "taxonomy rather than fixed in advance. " + GENERATIVE_CAVEAT
    )
)
app.add_typer(plan_app, name="plan")
app.add_typer(schema_app, name="schema")
app.add_typer(generative_app, name="generative")


class ExitCode(IntEnum):
    """Every exit status a command chooses, in one table (docs/api-reference.md#exit-codes).

    ERROR is guaranteed only by analyze, hash-plan, init, visualize and the generative commands,
    and by validate for an invalid integrity. A plan that validate or plan fixed-design --plan
    cannot load is not routed here (a traceback exiting 1, or a usage error exiting 2).

    typer itself exits 2 on a usage error, such as a missing option, before any command runs.
    """

    OK = 0
    # analyze --decision-exit-codes: the decision is FAIL.
    FAIL = 1
    # plan fixed-design --plan: no claim of the plan could be planned.
    NOTHING_PLANNED = 1
    # analyze --decision-exit-codes: the decision is INCONCLUSIVE.
    INCONCLUSIVE = 2
    # The command could not run, or its result supports no claim: an invalid integrity, or a
    # generative analysis with ERROR diagnostics.
    ERROR = 3


_DECISION_EXIT_CODES = {
    "PASS": ExitCode.OK,
    "FAIL": ExitCode.FAIL,
    "INCONCLUSIVE": ExitCode.INCONCLUSIVE,
}


DecisionPolicyOption = Annotated[
    Path | None,
    typer.Option(
        "--decision-policy",
        exists=True,
        dir_okay=False,
        help=(
            "A decision policy whose thresholds replace the plan's for the claims it sets, "
            "without changing the plan's hash (experimental; docs/decision-policy.md)."
        ),
    ),
]
ReceiptOption = Annotated[
    Path | None,
    typer.Option(
        "--registration-receipt",
        exists=True,
        dir_okay=False,
        help=(
            "Receipt written by hash-plan --register: checked against the plan's hash and "
            "the earliest evaluation start. Experimental: see docs/plan-reference.md."
        ),
    ),
]
RunsOption = Annotated[
    Path | None,
    typer.Option(
        "--runs",
        exists=True,
        dir_okay=False,
        help=(
            "Runs file of a neutral rollout source (JSONL or CSV); requires "
            "--rollouts. Experimental: see docs/rollout-files.md."
        ),
    ),
]
RolloutsOption = Annotated[
    Path | None,
    typer.Option(
        "--rollouts",
        exists=True,
        dir_okay=False,
        help=(
            "Rollouts file of a neutral rollout source (JSONL or CSV); requires "
            "--runs. Experimental: see docs/rollout-files.md."
        ),
    ),
]


def _source_arguments(
    logs: list[Path] | None, runs: Path | None, rollouts: Path | None
) -> tuple[str | list[str] | None, EvaluationSource | None]:
    """What to analyse, as ``logs`` and ``source``: Inspect logs or rollout files."""

    if runs is not None or rollouts is not None:
        if runs is None or rollouts is None:
            raise typer.BadParameter("--runs and --rollouts are given together")
        if logs:
            raise typer.BadParameter("pass either --logs or --runs and --rollouts, not both")
        from inspect_safety_stats.ingest.rollout_files import RolloutFiles

        return None, RolloutFiles(runs=runs, rollouts=rollouts)
    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    return (str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]), None


@app.command("init")
def init_command(
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    failure_values: Annotated[
        list[str] | None,
        typer.Option("--failure-values", help="Score values that count as a safety failure."),
    ] = None,
    safe_values: Annotated[
        list[str] | None,
        typer.Option(
            "--safe-values",
            help=(
                "Score values that count as safe. Required when the score has more than two "
                "values; with two, the one --failure-values does not name is safe."
            ),
        ),
    ] = None,
    failure_threshold: Annotated[
        float | None,
        typer.Option("--failure-threshold", help="Numeric scores at or above this are failures."),
    ] = None,
    score: Annotated[str | None, typer.Option("--score", help="Scorer name to analyze.")] = None,
    group_by: Annotated[
        str | None,
        typer.Option("--group-by", help="Sample metadata field for a one-level breakdown."),
    ] = None,
    baseline: Annotated[
        str | None,
        typer.Option("--baseline", help="Model (or cohort id) to compare the others against."),
    ] = None,
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    nested_weights: Annotated[
        bool,
        typer.Option(
            "--nested-weights/--equal-prompt-weights",
            help="Weight each group equally rather than each prompt.",
        ),
    ] = False,
    output: Annotated[Path, typer.Option("--output", file_okay=False)] = Path("."),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    intent: Annotated[
        str,
        typer.Option(
            "--intent",
            help=(
                "The plan's analysis.intent. Only exploratory: a plan written after the logs "
                "exist cannot be confirmatory."
            ),
        ),
    ] = "exploratory",
) -> None:
    """Write a starter design manifest and plan from completed Inspect logs (experimental).

    The plan reports the root failure risk per model with a prompt-level sensitivity,
    an optional one-level breakdown, and a comparison when there are several models.
    Its header notes each assumption the plan rests on, rollout independence first,
    with what goes wrong if it does not hold and how to change it.
    Edit it to add claims, risk measures or judge calibration.
    """

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    from inspect_safety_stats.scaffold import scaffold_plan

    try:
        result = scaffold_plan(
            [str(path) for path in logs],
            output,
            score=score,
            failure_values=failure_values,
            safe_values=safe_values,
            failure_threshold=failure_threshold,
            group_by=group_by,
            baseline=baseline,
            alpha=alpha,
            weights="nested_equal" if nested_weights else "equal_prompt",
            overwrite=overwrite,
            intent=intent,
        )
    except Exception as exc:
        typer.echo(f"Init failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc
    typer.echo(f"Score: {result.score}")
    if result.failure_values is not None:
        typer.echo(f"  failure values: {result.failure_values!r}")
        typer.echo(f"  safe values:    {result.safe_values!r}")
    else:
        typer.echo(f"  failure if score >= {failure_threshold}")
    typer.echo(
        f"Prompts: {result.prompts}, rollouts per prompt: {result.rollouts_per_prompt}"
        + (f", grouped by {result.group_by!r}" if result.group_by else "")
    )
    for cohort, model in result.cohorts.items():
        marker = " (reference)" if cohort == result.reference else ""
        typer.echo(f"Cohort {cohort}: {model}{marker}")
    for note in result.notes:
        typer.echo(f"Note: {note}")
    typer.echo("Assumptions (the plan's header says what each means and how to change it):")
    for assumption in result.assumptions:
        typer.echo(f"  {assumption.assumption}: {assumption.choice}")
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Plan: {result.plan_path}")
    log_args = " ".join(f"--logs {path}" for path in result.logs)
    typer.echo(
        f"Next: inspect-safety-stats analyze --plan {result.plan_path} {log_args} "
        "--output <report-dir>"
    )


def _echo_warnings(warnings: list[SummaryWarning], index_md: Path) -> None:
    """Print the report's errors and warnings by code on stderr, as index.md lists them.

    Each code gets one line: its severity, how often it occurred, its effect on the results
    and the registry's consequence, which says what the analysis did about it, so that a
    reader of the command's output alone sees what qualifies the headline. The registry's
    assumption is phrased as the assumption, so printed alone it would state as fact what the
    code says may not hold; the header names the section of report.md that gives it beside
    the effect, and the section of index.md that lists each code's diagnostics.
    """

    from inspect_safety_stats.report.index import SUMMARY_PAGE, WARNINGS_HEADING
    from inspect_safety_stats.report.summary import WARNINGS_SECTION

    if not warnings:
        return
    kinds = (
        "Errors and warnings"
        if any(warning.severity == "ERROR" for warning in warnings)
        else "Warnings"
    )
    typer.echo(
        f'{kinds} by code, listed in {index_md} under "{WARNINGS_HEADING}"; '
        f"{index_md.with_name(SUMMARY_PAGE)} gives each one's assumption and effect under "
        f'"{WARNINGS_SECTION}":',
        err=True,
    )
    for warning in warnings:
        times = "occurrence" if warning.count == 1 else "occurrences"
        effect = f", {warning.effect}" if warning.effect else ""
        consequence = warning.consequence or "no registry entry gives its consequence"
        typer.echo(
            f"  {warning.severity} {warning.code} ({warning.count} {times}{effect}): "
            f"{consequence}",
            err=True,
        )


@app.command("analyze")
def analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    runs: RunsOption = None,
    rollouts: RolloutsOption = None,
    output: Annotated[Path, typer.Option("--output")] = Path("safety-stats-report"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    decision_exit_codes: Annotated[
        bool, typer.Option("--decision-exit-codes/--no-decision-exit-codes")
    ] = False,
    allow_invalid: Annotated[bool, typer.Option("--allow-invalid")] = False,
    decision_policy: DecisionPolicyOption = None,
    registration_receipt: ReceiptOption = None,
) -> None:
    """Analyze completed Inspect logs or rollout files against a plan and write the report."""

    log_input, source = _source_arguments(logs, runs, rollouts)
    from inspect_safety_stats.api import analyze as run_analysis

    try:
        report = run_analysis(
            plan=plan,
            logs=log_input,
            parallel=parallel or False,
            source=source,
            decision_policy=decision_policy,
            registration_receipt=registration_receipt,
        )
        path = report.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc
    typer.echo(f"Analysis: {report.analysis['id']}")
    typer.echo(f"Plan SHA-256: {report.analysis['plan_sha256']}")
    typer.echo(f"Integrity: {report.integrity.status}")
    if report.presentation.include_decisions:
        for claim in report.claims:
            if claim.mandatory:
                typer.echo(f"{claim.status}: {claim.title}")
        for adaptive in report.adaptive_claims:
            if adaptive.mandatory:
                typer.echo(f"{adaptive.status}: {adaptive.title} (adaptive attacker, FAIL only)")
        typer.echo(f"Decision: {report.decision}")
    from inspect_safety_stats.schema.summary import is_headline_risk

    # The headline is each weighted-mean failure rate, as in report.json's summary.
    headline = [risk for risk in report.group_risks if is_headline_risk(risk)]
    if headline:
        typer.echo("Failure rate (weighted mean):")
    for risk in headline:
        interval = (
            f"[{100 * risk.lower:.1f}%, {100 * risk.upper:.1f}%]"
            if risk.lower is not None and risk.upper is not None
            else "interval unavailable"
        )
        estimate = "—" if risk.estimate is None else f"{100 * risk.estimate:.1f}%"
        typer.echo(f"  {risk.cohort_id} / {risk.group_id}: {estimate} {interval}")
    _echo_warnings(report.summary.warnings, path / "index.md")
    typer.echo(f"Summary: {path / 'report.md'}")
    typer.echo(f"Index: {path / 'index.json'}")
    typer.echo(f"Report: {path}")
    # An invalid analysis supports no statistical claim, so it must not pass a CI gate by
    # exiting 0. The report is still written above so that its diagnostics can be audited.
    if report.integrity.status == "invalid" and not allow_invalid:
        typer.echo(
            "Integrity is invalid: see the report's diagnostics. "
            "Pass --allow-invalid to exit 0 anyway.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.ERROR)
    if decision_exit_codes:
        raise typer.Exit(code=_DECISION_EXIT_CODES[report.decision])


@app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    runs: RunsOption = None,
    rollouts: RolloutsOption = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    decision_policy: DecisionPolicyOption = None,
    registration_receipt: ReceiptOption = None,
) -> None:
    """Check logs or rollout files against a plan (manifest, cohorts, rollouts, scores)."""

    log_input, source = _source_arguments(logs, runs, rollouts)
    from inspect_safety_stats.api import validate as run_validation

    result = run_validation(
        plan=plan,
        logs=log_input,
        parallel=parallel or False,
        source=source,
        decision_policy=decision_policy,
        registration_receipt=registration_receipt,
    )
    for diagnostic in result.diagnostics:
        scope = f" [{diagnostic.scope}]" if diagnostic.scope else ""
        typer.echo(f"{diagnostic.severity} {diagnostic.code}{scope}: {diagnostic.message}")
    typer.echo(f"Integrity: {result.status}")
    raise typer.Exit(code=ExitCode.ERROR if result.status == "invalid" else ExitCode.OK)


@app.command("visualize")
def visualize_command(
    report: Annotated[Path, typer.Option("--report", exists=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build a self-contained interactive HTML visualization from a report."""

    from inspect_safety_stats.report.visualization import write_visualization

    try:
        path = write_visualization(report, output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Visualization failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc
    typer.echo(f"Visualization: {path}")


@plan_app.command("zero-failure")
def zero_failure_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Rollouts needed for zero observed failures to bound the rate below a threshold."""

    from inspect_safety_stats.stats.planning import zero_failure_sample_size

    local_alpha = alpha / family_size
    count = zero_failure_sample_size(threshold=threshold, alpha=local_alpha)
    typer.echo(f"Required rollouts: {count}")
    typer.echo(
        f"Decision rule: PASS only if all {count} rollouts are valid and zero failures occur"
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {local_alpha:.8g} (Bonferroni share of {alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided confidence level: {(1 - local_alpha) * 100:g}%")


@plan_app.command("certification")
def certification_planning_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    anticipated_rate: Annotated[float, typer.Option("--anticipated-rate", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Plan an exact one-sided binomial certification claim."""

    from inspect_safety_stats.stats.planning import (
        certification_power,
        certification_sample_size,
        maximum_certifiable_failure_rate,
        maximum_certifying_failures,
        minimum_detectable_failure_rate,
    )

    family_alpha = alpha
    alpha = alpha / family_size
    try:
        trials = certification_sample_size(
            true_rate=anticipated_rate,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        cutoff = maximum_certifying_failures(trials=trials, threshold=threshold, alpha=alpha)
        achieved = certification_power(
            trials=trials,
            true_rate=anticipated_rate,
            threshold=threshold,
            alpha=alpha,
        )
        maximum_certifiable = maximum_certifiable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        minimum_detectable = minimum_detectable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Required rollouts: {trials}")
    typer.echo(f"Maximum certifying failures: {cutoff}")
    typer.echo(f"Achieved power: {achieved:.6f}")
    typer.echo(
        "Maximum certifiable failure rate at target power: "
        + ("unattainable" if maximum_certifiable is None else f"{maximum_certifiable:.8g}")
    )
    typer.echo(
        "Minimum detectable failure rate at target power: "
        + ("unattainable" if minimum_detectable is None else f"{minimum_detectable:.8g}")
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {alpha:.8g} (Bonferroni share of {family_alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha: {alpha:g}")


@plan_app.command("detectable-rate")
def detectable_rate_command(
    trials: Annotated[int, typer.Option("--trials", min=1)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Invert exact one-sided tests at a fixed rollout horizon."""

    from inspect_safety_stats.stats.planning import (
        maximum_certifiable_failure_rate,
        minimum_detectable_failure_rate,
    )

    family_alpha = alpha
    alpha = alpha / family_size
    try:
        maximum_certifiable = maximum_certifiable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
        minimum_detectable = minimum_detectable_failure_rate(
            trials=trials,
            threshold=threshold,
            alpha=alpha,
            power=power,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Rollouts: {trials}")
    typer.echo(f"Threshold: {threshold:.8g}")
    typer.echo(f"Target power: {power:.8g}")
    typer.echo(
        "Maximum certifiable failure rate: "
        + ("unattainable" if maximum_certifiable is None else f"{maximum_certifiable:.8g}")
    )
    typer.echo(
        "Minimum detectable failure rate: "
        + ("unattainable" if minimum_detectable is None else f"{minimum_detectable:.8g}")
    )
    if maximum_certifiable is not None:
        typer.echo(f"Support-side detectable gap: {threshold - maximum_certifiable:.8g}")
    if minimum_detectable is not None:
        typer.echo(f"Rejection-side detectable gap: {minimum_detectable - threshold:.8g}")
    if family_size > 1:
        typer.echo(
            f"Local alpha: {alpha:.8g} (Bonferroni share of {family_alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha: {alpha:g}")


class WeightStrategyOption(str, Enum):
    equal_prompt = "equal_prompt"
    nested_equal = "nested_equal"
    explicit = "explicit"


class IndependenceUnitOption(str, Enum):
    rollout = "rollout"
    prompt_cluster = "prompt_cluster"


def _planning_design(
    manifest: Path,
    weights: WeightStrategyOption,
    hierarchy: str,
    weight_column: str | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read a design manifest and resolve its weights as `analyze` would."""

    from inspect_safety_stats.config.models import WeightSpec
    from inspect_safety_stats.ingest.manifest import (
        REQUIRED_COLUMNS,
        load_manifest,
        resolve_weights,
    )

    # resolve_weights only reads the column for explicit weights; ignoring it silently would
    # plan under a weighting the user did not ask for.
    if weight_column is not None and weights is not WeightStrategyOption.explicit:
        raise typer.BadParameter("--weight-column applies only with --weights explicit")
    try:
        spec = WeightSpec.model_validate(
            {
                "strategy": weights.value,
                "hierarchy": [level.strip() for level in hierarchy.split(",") if level.strip()],
                "column": weight_column,
            }
        )
        # Only the identifier levels the weighting names are required, as in analyze.
        named = {"prompt_id", *(spec.hierarchy if spec.strategy == "nested_equal" else [])}
        if spec.column is not None:
            named.add(spec.column)
        frame = load_manifest(manifest, [column for column in REQUIRED_COLUMNS if column in named])
        return frame, resolve_weights(frame, spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@plan_app.command("fixed-design")
def fixed_design_planning_command(
    ctx: typer.Context,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            exists=True,
            dir_okay=False,
            help="Design manifest; with --plan, overrides the plan's inputs.design_manifest.",
        ),
    ] = None,
    threshold: Annotated[float | None, typer.Option("--threshold", min=0.0, max=1.0)] = None,
    plan_path: Annotated[
        Path | None,
        typer.Option(
            "--plan",
            exists=True,
            dir_okay=False,
            help=(
                "Analysis plan: plan every claim with the plan's groups, weights, hierarchy, "
                "independence profile, multiplicity family and thresholds."
            ),
        ),
    ] = None,
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    rollouts: Annotated[int | None, typer.Option("--rollouts", min=1)] = None,
    weights: Annotated[
        WeightStrategyOption, typer.Option("--weights")
    ] = WeightStrategyOption.nested_equal,
    hierarchy: Annotated[str, typer.Option("--hierarchy")] = "task_id,question_id,prompt_id",
    weight_column: Annotated[str | None, typer.Option("--weight-column")] = None,
    independence_unit: Annotated[
        IndependenceUnitOption, typer.Option("--independence-unit")
    ] = IndependenceUnitOption.rollout,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Detectable rates for a declared weighted design.

    Reads the design manifest so planning sees the same prompts, weights, and
    expected rollout counts that `analyze` will resolve. Rollout counts come
    from the manifest's `expected_rollouts` column unless `--rollouts` supplies
    a uniform override. Without --plan the rates are weighted Hoeffding guarantees.

    With --plan, every claim is planned as `analyze` will decide it, and the options the
    plan supplies are refused rather than overridden.
    """

    if plan_path is not None:
        supplied = [
            f"--{name.replace('_', '-')}"
            for name in (
                "threshold",
                "alpha",
                "weights",
                "hierarchy",
                "weight_column",
                "independence_unit",
                "family_size",
            )
            # typer vendors its own click, so the source is compared by name.
            if getattr(ctx.get_parameter_source(name), "name", "DEFAULT") != "DEFAULT"
        ]
        if supplied:
            raise typer.BadParameter(
                f"{', '.join(supplied)} cannot be combined with --plan, which supplies them"
            )
        _plan_fixed_design_from_plan(plan_path, manifest, power=power, rollouts=rollouts)
        return
    if manifest is None or threshold is None:
        raise typer.BadParameter("give --manifest and --threshold, or --plan")

    from inspect_safety_stats.stats.planning import fixed_design_detectable_failure_rates

    local_alpha = alpha / family_size
    unit: Literal["rollout", "prompt_cluster"] = (
        "rollout" if independence_unit is IndependenceUnitOption.rollout else "prompt_cluster"
    )
    frame, weight_values = _planning_design(manifest, weights, hierarchy, weight_column)
    try:
        if rollouts is not None:
            trials = [rollouts] * len(frame)
        elif "expected_rollouts" in frame:
            trials = [int(value) for value in frame["expected_rollouts"]]
        else:
            raise ValueError(
                "the design manifest has no expected_rollouts column; "
                "pass --rollouts for a uniform count"
            )
        rates = fixed_design_detectable_failure_rates(
            trials,
            weight_values,
            threshold=threshold,
            support_alpha=local_alpha,
            rejection_alpha=local_alpha,
            power=power,
            independence_unit=unit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Prompts: {len(frame)}")
    typer.echo(f"Total rollouts: {sum(trials)}")
    typer.echo(f"Independence unit: {rates.independence_unit}")
    typer.echo(f"Effective sample size: {rates.effective_sample_size:.8g}")
    typer.echo(f"Threshold: {threshold:.8g}")
    typer.echo(f"Target power: {power:.8g}")
    typer.echo(
        "Maximum certifiable failure rate at target power: "
        + (
            "unattainable"
            if rates.maximum_certifiable_failure_rate is None
            else f"{rates.maximum_certifiable_failure_rate:.8g}"
        )
    )
    typer.echo(
        "Minimum detectable failure rate at target power: "
        + (
            "unattainable"
            if rates.minimum_detectable_failure_rate is None
            else f"{rates.minimum_detectable_failure_rate:.8g}"
        )
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {local_alpha:.8g} (Bonferroni share of {alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"One-sided alpha per direction: {local_alpha:g}")


def _rate(value: float | None) -> str:
    return "unattainable" if value is None else f"{value:.8g}"


def _alpha_share(alpha: EndpointAlpha) -> str:
    if alpha.calibration == "pointwise":
        return "pointwise, not simultaneous"
    if alpha.calibration == "frontier_sidak":
        return (
            f"Sidak share of {alpha.frontier_alpha:g} across the {alpha.frontier_size} "
            f"target(s) of frontier {alpha.frontier_id}"
        )
    return (
        f"Bonferroni share of {alpha.directional_family_alpha:g} across "
        f"{alpha.target_count} targets and {alpha.cohort_count} cohorts"
    )


def _plan_fixed_design_from_plan(
    plan_path: Path, manifest: Path | None, *, power: float, rollouts: int | None
) -> None:
    from inspect_safety_stats.claim_planning import PlannedClaim, plan_design
    from inspect_safety_stats.config.loader import load_plan

    try:
        plan = load_plan(plan_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    source, named_by = (
        (plan.inputs.design_manifest, "inputs.design_manifest")
        if manifest is None
        else (manifest, "--manifest")
    )
    try:
        design = plan_design(plan, power=power, manifest=source, rollouts=rollouts)
    except OSError as exc:
        # The design manifest is the only file plan_design reads.
        raise typer.BadParameter(
            f"cannot read the design manifest {source}, which {named_by} names: "
            f"{exc.strerror or exc}"
        ) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    claims = design.claims
    typer.echo(
        f"Design manifest: {source} "
        + (
            "(inputs.design_manifest)"
            if manifest is None
            else "(--manifest, in place of the plan's inputs.design_manifest)"
        )
    )
    typer.echo(f"Claims: {len(claims)}")
    typer.echo(f"Target power: {power:.8g}")
    if design.method is not None:
        typer.echo(
            f"Interval method: {design.method}, as outputs.direct_mean_intervals declares; "
            "Hoeffding is shown as the reference"
        )
    for item in claims:
        typer.echo("")
        if not isinstance(item, PlannedClaim):
            typer.echo(f"Claim {item.claim_id}: not planned: {item.reason}")
            continue
        typer.echo(
            f"Claim {item.claim_id}: cohort {item.cohort}, group {item.group}, "
            f"measure {item.measure}"
        )
        typer.echo(f"  Prompts: {item.prompts}")
        typer.echo(f"  Total rollouts: {item.total_rollouts}")
        typer.echo(
            f"  Independence profile: {item.independence_profile}"
            + (
                ""
                if item.independent_blocks is None
                else f" ({item.independent_blocks} independent blocks)"
            )
        )
        planned = item.planned
        typer.echo(f"  Effective sample size: {planned.effective_sample_size:.8g}")
        alpha = item.alpha
        typer.echo(
            f"  One-sided alpha per direction: {alpha.endpoint_alpha:.8g} "
            f"(family {alpha.family_id}, {_alpha_share(alpha)})"
        )
        support = item.hoeffding_support
        rejection = item.hoeffding_rejection
        typer.echo(
            f"  PASS at acceptable_upper {item.acceptable_upper:.8g}: maximum certifiable "
            f"failure rate at target power {_rate(planned.maximum_certifiable_failure_rate)} "
            f"({planned.method}); Hoeffding reference "
            + _rate(support.maximum_certifiable_failure_rate if support else None)
        )
        typer.echo(
            f"  FAIL at unacceptable_lower {item.unacceptable_lower:.8g}: minimum detectable "
            f"failure rate at target power {_rate(planned.minimum_detectable_failure_rate)} "
            f"({planned.method}); Hoeffding reference "
            + _rate(rejection.minimum_detectable_failure_rate if rejection else None)
        )
        typer.echo(
            f"  Certification floor: {item.certification_floor:.8g}, the upper bound analyze "
            "reports when every rollout is safe"
            + (
                f" ({item.raw_certification_floor:.8g} before the hierarchy-region projection; "
                "the rates above are planned before it, and the projection can only raise "
                "their power)"
                if item.certification_floor < item.raw_certification_floor
                else ""
            )
        )
        for reference in item.references:
            typer.echo(
                f"  Reference method {reference.method}: maximum certifiable "
                f"{_rate(reference.maximum_certifiable_failure_rate)}, minimum detectable "
                f"{_rate(reference.minimum_detectable_failure_rate)}"
            )
        for unplanned in item.unplanned_references:
            typer.echo(
                f"  Reference method {unplanned}: not planned; the planner cannot invert its "
                "interval"
            )
    if design.intervals:
        typer.echo("")
        typer.echo(
            f"Direct-mean intervals analyze will report ({design.method}); "
            "raw floor is the interval's upper bound when every rollout is safe, and floor "
            "the bound analyze reports then, after the hierarchy-region projection when the "
            "plan asks for it. A target resolves when its floor is at or below its group's "
            "smallest acceptable_upper:"
        )
        typer.echo(
            f"  {'profile':<14} {'cohort':<14} {'target':<36} {'blocks':>7} {'ESS':>9} "
            f"{'alpha':>11} {'calibration':<15} {'raw floor':>9} {'floor':>9} {'resolves':>8}"
        )
        for interval in design.intervals:
            target = (
                f"{interval.group}"
                if interval.level is None
                else f"{interval.group}:" + "/".join(interval.path.values())
            )
            resolves = {None: "-", True: "yes", False: "no"}[interval.resolves]
            typer.echo(
                f"  {interval.independence_profile:<14} {interval.cohort:<14} {target:<36} "
                f"{interval.independent_blocks:>7} {interval.effective_sample_size:>9.4g} "
                f"{interval.alpha.endpoint_alpha:>11.4g} {interval.alpha.calibration:<15} "
                f"{interval.raw_certification_floor:>9.4g} "
                f"{interval.certification_floor:>9.4g} {resolves:>8}"
            )
        unresolved: dict[tuple[str, str, str, str], list[int]] = {}
        for interval in design.intervals:
            if interval.resolves is None or interval.level is None:
                continue
            key = (
                interval.independence_profile,
                interval.cohort,
                interval.group,
                interval.level,
            )
            tally = unresolved.setdefault(key, [0, 0])
            tally[1] += 1
            if not interval.resolves:
                tally[0] += 1
        notes = [
            f"  level {level} of group {group} in cohort {cohort} under profile {profile}: "
            f"{failed} of {total} nodes cannot be certified at the group's acceptable_upper, "
            "even when every rollout is safe"
            for (profile, cohort, group, level), (failed, total) in unresolved.items()
            if failed
        ]
        if notes:
            typer.echo("")
            typer.echo("Levels the declared rollouts cannot resolve:")
            for note in notes:
                typer.echo(note)
    if not any(isinstance(item, PlannedClaim) for item in claims):
        typer.echo("")
        typer.echo("No claim of this plan could be planned.", err=True)
        raise typer.Exit(code=ExitCode.NOTHING_PLANNED)


@plan_app.command("pilot")
def pilot_command(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, help="Design manifest CSV."),
    ],
    per_stratum: Annotated[
        int, typer.Option("--per-stratum", min=1, help="Prompts to draw from each stratum.")
    ],
    seed: Annotated[
        int, typer.Option("--seed", help="Selection seed; the frame records it as pilot_seed.")
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Pilot frame CSV to write."),
    ],
    strata: Annotated[
        str,
        typer.Option(
            "--strata",
            help="Comma-separated manifest columns whose combinations form the strata; "
            "empty for one stratum.",
        ),
    ] = "task_id",
) -> None:
    """Draw a random pilot within strata of a design manifest (experimental).

    Writes the whole manifest with pilot_stratum, pilot_inclusion_probability,
    pilot_selected, pilot_seed and pilot_per_stratum added. Run the pilot on the rows where
    pilot_selected is true. The same file is the frame of a sampled design (sampling:
    fixed_size, stratum_column: pilot_stratum, inclusion_probability_column:
    pilot_inclusion_probability, selection_column: pilot_selected). Analyse a pilot as
    exploratory: a pilot enlarged after its outcomes were seen certifies nothing.
    """

    import pandas as pd

    from inspect_safety_stats.ingest.manifest import REQUIRED_COLUMNS, load_manifest
    from inspect_safety_stats.ingest.pilot import draw_pilot

    # samefile also catches a path that differs only in case on a case-insensitive
    # filesystem (macOS, Windows) and a hard link, which resolve() leaves distinct.
    if output.resolve() == manifest.resolve() or (output.exists() and output.samefile(manifest)):
        raise typer.BadParameter("--output must not overwrite the design manifest")
    columns = [column.strip() for column in strata.split(",") if column.strip()]
    try:
        # Check the manifest as analyze reads it, requiring only the identifier columns the
        # draw uses: analyze gives a plan that never names question_id or task_id each
        # prompt's own. draw_pilot checks strata columns outside REQUIRED_COLUMNS.
        load_manifest(
            manifest,
            required=("prompt_id", *(column for column in columns if column in REQUIRED_COLUMNS)),
        )
        # Read every value as written, so the pilot frame repeats the manifest's own text
        # rather than pandas' reading of it (an integer column with a gap would turn into
        # floats, for example).
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        pilot = draw_pilot(frame, per_stratum=per_stratum, strata=columns, seed=seed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(output, index=False, lineterminator="\n")
    selected = pilot.loc[pilot["pilot_selected"]]
    typer.echo(f"Strata: {pilot['pilot_stratum'].nunique()}")
    typer.echo(f"Pilot prompts: {len(selected)} of {len(pilot)}")
    typer.echo(f"Seed: {seed}")
    typer.echo(f"Pilot frame: {output}")

class ComparisonUnitOption(str, Enum):
    rollout = "rollout"
    pair = "pair"
    prompt_cluster = "prompt_cluster"


_COMPARISON_UNITS: dict[ComparisonUnitOption, Literal["rollout", "pair", "prompt_cluster"]] = {
    ComparisonUnitOption.rollout: "rollout",
    ComparisonUnitOption.pair: "pair",
    ComparisonUnitOption.prompt_cluster: "prompt_cluster",
}


class CoverageModeOption(str, Enum):
    two_sided = "two_sided"
    per_direction = "per_direction"


@plan_app.command("comparison")
def comparison_planning_command(
    manifest: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    margin: Annotated[float, typer.Option("--margin", min=-1.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    candidate_rollouts: Annotated[int | None, typer.Option("--candidate-rollouts", min=1)] = None,
    baseline_rollouts: Annotated[int | None, typer.Option("--baseline-rollouts", min=1)] = None,
    weights: Annotated[
        WeightStrategyOption, typer.Option("--weights")
    ] = WeightStrategyOption.nested_equal,
    hierarchy: Annotated[str, typer.Option("--hierarchy")] = "task_id,question_id,prompt_id",
    weight_column: Annotated[str | None, typer.Option("--weight-column")] = None,
    independence_unit: Annotated[
        ComparisonUnitOption, typer.Option("--independence-unit")
    ] = ComparisonUnitOption.rollout,
    coverage_mode: Annotated[
        CoverageModeOption, typer.Option("--coverage-mode")
    ] = CoverageModeOption.two_sided,
    family_size: Annotated[int, typer.Option("--family-size", min=1)] = 1,
) -> None:
    """Risk differences a two-arm design resolves at a margin (experimental).

    The difference is candidate minus baseline risk over the manifest's weighted design.
    Reports the largest true difference whose upper bound falls at or below --margin, and
    the smallest whose lower bound reaches it, each with --power, under the Hoeffding
    guarantee of a comparisons entry with method weighted_hoeffding. That is not the setup
    init writes, nor the one the guide recommends for two or more configurations: both use a
    comparison_sets entry with method weighted_kl_chernoff. To rely on the numbers, declare
    the contrast as a comparisons entry with method weighted_hoeffding. The numbers do not
    hold for a comparison_sets entry or another method: a comparison set bounds each arm
    separately, which is usually wider, and weighted_kl_chernoff and
    weighted_empirical_bernstein, often narrower on sparse failures when prompt weights are
    even, can be wider with uneven weights or at moderate failure rates. Rollout counts per
    arm come from the manifest's expected_rollouts column unless --candidate-rollouts or
    --baseline-rollouts override them. --independence-unit pair plans for a profile that
    blocks by prompt_id and repeat with joint coupling; prompt_cluster for one that blocks
    by prompt_id with joint coupling.
    """

    from inspect_safety_stats.stats.planning import comparison_resolvable_differences

    # As in analyze: two_sided splits each claim's share of the family alpha between the
    # lower and the upper bound; per_direction spends it on each.
    local_alpha = alpha / family_size
    direction_alpha = (
        local_alpha / 2 if coverage_mode is CoverageModeOption.two_sided else local_alpha
    )
    frame, weight_values = _planning_design(manifest, weights, hierarchy, weight_column)
    try:
        expected: list[int] | None = (
            [int(value) for value in frame["expected_rollouts"]]
            if "expected_rollouts" in frame
            else None
        )
        if expected is None and (candidate_rollouts is None or baseline_rollouts is None):
            raise ValueError(
                "the design manifest has no expected_rollouts column; pass "
                "--candidate-rollouts and --baseline-rollouts for uniform counts"
            )
        candidate_trials = (
            [candidate_rollouts] * len(frame) if candidate_rollouts is not None else expected
        )
        baseline_trials = (
            [baseline_rollouts] * len(frame) if baseline_rollouts is not None else expected
        )
        if candidate_trials is None or baseline_trials is None:
            raise ValueError("rollout counts could not be resolved for both arms")
        plan = comparison_resolvable_differences(
            candidate_trials,
            baseline_trials,
            weight_values,
            margin=margin,
            alpha_lower=direction_alpha,
            alpha_upper=direction_alpha,
            power=power,
            independence_unit=_COMPARISON_UNITS[independence_unit],
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Prompts: {len(frame)}")
    typer.echo(f"Candidate rollouts: {sum(candidate_trials)}")
    typer.echo(f"Baseline rollouts: {sum(baseline_trials)}")
    typer.echo(f"Independence unit: {plan.independence_unit}")
    typer.echo(f"Independent blocks: {plan.block_count}")
    typer.echo(f"Effective sample size: {plan.effective_sample_size:.8g}")
    typer.echo(f"Margin: {margin:.8g}")
    typer.echo(f"Target power: {power:.8g}")
    typer.echo(
        "Interval method: weighted_hoeffding in a comparisons entry only; not comparison_sets "
        "(as init writes) or other methods, where power can fall far below the target"
    )
    typer.echo(
        "Largest difference with upper bound at or below the margin: "
        + (
            "unattainable"
            if plan.maximum_supportable_difference is None
            else f"{plan.maximum_supportable_difference:.8g}"
        )
    )
    typer.echo(
        "Smallest difference with lower bound at or above the margin: "
        + (
            "unattainable"
            if plan.minimum_detectable_difference is None
            else f"{plan.minimum_detectable_difference:.8g}"
        )
    )
    if family_size > 1:
        typer.echo(
            f"Local alpha: {local_alpha:.8g} (Bonferroni share of {alpha:g} "
            f"across {family_size} simultaneous claims)"
        )
    typer.echo(f"Coverage mode: {coverage_mode.value}")
    typer.echo(f"One-sided alpha per direction: {direction_alpha:g}")


@schema_app.command("plan")
def plan_schema_command() -> None:
    """Print the JSON schema for analysis plans."""

    from inspect_safety_stats.config.models import AnalysisPlan

    typer.echo(json.dumps(AnalysisPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("report")
def report_schema_command() -> None:
    """Print the JSON schema for reports."""

    from inspect_safety_stats.schema.reports import SafetyReport

    typer.echo(json.dumps(SafetyReport.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("claims")
def claims_schema_command() -> None:
    """Print the JSON schema for claims.json (experimental)."""

    from inspect_safety_stats.schema.claims import ClaimsExport

    typer.echo(json.dumps(ClaimsExport.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("decision-policy")
def decision_policy_schema_command() -> None:
    """Print the JSON schema for decision policies (experimental)."""

    from inspect_safety_stats.config.decision_policy import DecisionPolicy

    typer.echo(json.dumps(DecisionPolicy.model_json_schema(), indent=2, sort_keys=True))


@app.command("version")
def version_command() -> None:
    """Print the package version."""

    typer.echo(__version__)


def _one_line_error(exc: Exception) -> str:
    """Render an error on one line, a pydantic error as ``field: message`` per failure."""

    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        text = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()
        )
    else:
        text = str(exc)
    # A validator's own message may hold line breaks; the line is collapsed either way.
    return " ".join(text.split())


@app.command("hash-plan")
def hash_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    inference: Annotated[
        bool,
        typer.Option(
            "--inference",
            help=(
                "Print the inference hash instead: the plan's hash without the thresholds a "
                "decision policy may set, which a decision policy names (experimental)."
            ),
        ),
    ] = False,
    register: Annotated[
        Path | None,
        typer.Option(
            "--register",
            dir_okay=False,
            help=(
                "Also write a registration receipt (the hash and the current UTC time) to "
                "this new file, for analyze --registration-receipt. Anchor it externally, "
                "for example in a signed commit. Experimental: see docs/plan-reference.md."
            ),
        ),
    ] = None,
) -> None:
    """Print a plan's canonical SHA-256, for registering a frozen plan."""

    from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan

    if inference and register is not None:
        # A receipt records the canonical hash, which analyze checks; registering the
        # inference hash would make a receipt no analysis can match.
        raise typer.BadParameter("--register records the canonical hash; omit --inference")
    import yaml

    try:
        resolved = load_plan(plan)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # pydantic's ValidationError is a ValueError; other exceptions are bugs and keep
        # their traceback.
        typer.echo(f"Plan invalid: {_one_line_error(exc)}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc
    if inference:
        from inspect_safety_stats.config.decision_policy import inference_plan_hash

        typer.echo(inference_plan_hash(resolved))
        return
    digest = canonical_plan_hash(resolved)
    # The hash stays the first line on its own: scripts read it.
    typer.echo(digest)
    if register is None:
        return
    from inspect_safety_stats.config.registration import write_receipt

    try:
        receipt = write_receipt(register, digest, package_version=__version__)
    except FileExistsError as exc:
        typer.echo(
            f"Registration failed: {register} exists; a receipt is never overwritten",
            err=True,
        )
        raise typer.Exit(code=ExitCode.ERROR) from exc
    except OSError as exc:
        # Such as a directory that does not exist or cannot be written: no receipt is written.
        typer.echo(
            f"Registration failed: cannot write {register}: {exc.strerror or exc}", err=True
        )
        raise typer.Exit(code=ExitCode.ERROR) from exc
    typer.echo(f"Registered at: {receipt.registered_at.isoformat()}")
    typer.echo(f"Receipt: {register}")
    typer.echo(
        "The receipt's time is the local clock's. Anchor the receipt externally, for "
        "example in a signed commit or with a timestamping service, before generation starts."
    )


@generative_app.command("compile", epilog=GENERATIVE_CAVEAT)
def generative_compile_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    """Compile a generative plan and report what the canonical taxonomy contains.

    A declared slice family is checked against the paths materialize will select, so a
    member the analysis could not estimate is refused before anything is generated.
    """

    from inspect_safety_stats.generative.sampling import draw_selections, plan_design
    from inspect_safety_stats.generative.schema import load_generative_plan
    from inspect_safety_stats.generative.taxonomy import compile_taxonomy

    try:
        parsed = load_generative_plan(plan)
        taxonomy = compile_taxonomy(parsed, base_path=plan.parent)
        design = plan_design(taxonomy, parsed)
        if parsed.inference.slice_family is not None:
            # The runner imports Inspect and pandas, which a plan without a family never needs.
            from inspect_safety_stats.generative.runner import check_declared_slice_family

            # The selected paths do not depend on the stage: these are the ones materialize
            # selects under --stage pilot or main.
            check_declared_slice_family(
                parsed, taxonomy, design, draw_selections(taxonomy, design)
            )
    except Exception as exc:
        typer.echo(f"Compilation failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc

    typer.echo(f"Evaluation: {parsed.evaluation.id}")
    typer.echo(f"Taxonomy fingerprint: {taxonomy.fingerprint}")
    typer.echo(f"Nodes: {len(taxonomy.nodes)}")
    typer.echo(f"Terminal paths: {len(taxonomy.terminal_paths)}")
    typer.echo(f"Pruned by constraint: {len(taxonomy.pruned_by_constraint)}")
    typer.echo(f"Pruned as zero weight: {len(taxonomy.pruned_zero_weight)}")
    typer.echo(f"Design: {design.method}, {design.total_rewrites} rewrites")
    if design.allocations:
        typer.echo(f"Strata: {len(design.allocations)}")
    typer.echo(f"Cohorts: {', '.join(parsed.target.cohorts)}")


@generative_app.command("materialize", epilog=GENERATIVE_CAVEAT)
def generative_materialize_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("generative-run"),
    kernel: Annotated[str, typer.Option("--kernel")] = "fake",
    validator: Annotated[
        str | None,
        typer.Option(
            "--validator",
            help=(
                "Experimental. The validator the plan's rewrite.validation.validator_id "
                "names, resolved like --kernel ('fake', 'fake:reject_every=N', "
                "'module:attribute' or 'path/to/file.py:attribute'). Required when the plan "
                "sets rewrite.validation.invoke, refused otherwise."
            ),
        ),
    ] = None,
    stage: Annotated[str, typer.Option("--stage")] = "main",
    salt: Annotated[str, typer.Option("--rewrite-salt")] = "",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    """Select paths, run the rewrite procedure, and write the dataset and ledgers.

    Ledgers are appended as each unit completes, so an interrupted run leaves a usable
    record. --resume continues from it, executing only the units still outstanding.
    """

    from inspect_safety_stats.generative.runner import materialize_run

    # A console script does not put the working directory on sys.path, as `python -m` does,
    # but someone naming a module on the command line means the one in their project. The
    # CLI owns the process, so it sets this; the library leaves sys.path to its caller.
    working = str(Path.cwd())
    if working not in sys.path:
        sys.path.insert(0, working)

    try:
        run = materialize_run(
            plan,
            output,
            kernel=kernel,
            stage=stage,
            overwrite=overwrite,
            resume=resume,
            rewrite_salt=salt,
            validator=validator,
        )
    except Exception as exc:
        typer.echo(f"Materialization failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc

    typer.echo(f"Taxonomy fingerprint: {run.taxonomy_fingerprint}")
    typer.echo(f"Kernel: {run.kernel.kernel_id} @ {run.kernel.sha256}")
    if run.validator_id is not None:
        typer.echo(f"Validator: {run.validator_id}")
    typer.echo(f"Selections: {run.selections}")
    if run.resumed:
        typer.echo(f"Resumed: {run.resumed} units carried over, "
                   f"{run.selections - run.resumed} executed")
    typer.echo(
        f"Rewrites: {run.accepted} accepted, {run.duplicates_rejected} duplicates rejected, "
        f"{run.exhausted} exhausted, {run.errored} errored"
    )
    if run.kernel_mismatch:
        typer.echo(f"Warning: {run.kernel_mismatch}", err=True)
    if run.validator_mismatch:
        typer.echo(f"Warning: {run.validator_mismatch}", err=True)
    typer.echo(f"Dataset: {run.dataset}")
    typer.echo(f"Artifacts: {run.output}")
    if run.accepted == 0:
        if run.errored >= run.exhausted:
            typer.echo(
                "Every rewrite failed with a provider error, so nothing was generated. "
                "Check credentials, connectivity and the model name in the plan.",
                err=True,
            )
            if run.first_error:
                typer.echo(f"First error: {run.first_error}", err=True)
        else:
            typer.echo(
                "No rewrite passed validation; there is nothing to evaluate. Raise "
                "rewrite.validation.maximum_attempts, loosen rewrite.deduplication, or "
                "check that the kernel returns usable text.",
                err=True,
            )
        raise typer.Exit(code=ExitCode.ERROR)


@generative_app.command("analyze", epilog=GENERATIVE_CAVEAT)
def generative_analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    artifacts: Annotated[Path, typer.Option("--artifacts", exists=True, file_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("generative-report"),
    cohort: Annotated[str | None, typer.Option("--cohort")] = None,
    slice_: Annotated[list[str] | None, typer.Option("--slice")] = None,
    compare: Annotated[list[str] | None, typer.Option("--compare")] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Join a completed Inspect run against the ledgers and estimate the design."""

    from inspect_safety_stats.generative.runner import analyze_run

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    try:
        report = analyze_run(
            plan,
            artifacts,
            log_input,
            cohort=cohort,
            slices=tuple(slice_ or ()),
            comparisons=tuple(compare or ()),
            parallel=parallel or False,
        )
        path = report.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc

    root = report.tree.root
    typer.echo(f"Evaluation: {report.evaluation['id']}")
    typer.echo(f"Taxonomy fingerprint: {report.evaluation['taxonomy_fingerprint']}")
    typer.echo(f"Primary cohort: {report.evaluation['primary_cohort']}")
    typer.echo(f"Draws: {root.draws}, failures: {root.failures}")
    typer.echo(f"Root risk upper bound: {root.upper:.5f} ({root.method})")
    for cut in report.slices:
        typer.echo(f"  {cut.node_id}: upper {cut.upper:.5f}")
    for comparison in report.comparisons:
        typer.echo(
            f"  {comparison.candidate_cohort} - {comparison.baseline_cohort}: "
            f"upper {comparison.upper:+.5f} ({comparison.method})"
        )
    if report.slice_family is not None:
        # The family is never judge-corrected, even where the plan declares a calibration.
        typer.echo("Slice family (experimental, simultaneous, judge-reported scale):")
        for member in report.slice_family.slices:
            typer.echo(f"  {member.member_id}: upper {member.arm.upper:.5f}")
        for contrast in report.slice_family.contrasts:
            typer.echo(
                f"  {contrast.id} ({contrast.candidate_value} - {contrast.baseline_value}): "
                f"upper {contrast.upper:+.5f}"
            )
    typer.echo(f"Nodes reported: {len(report.tree.nodes)}, unsampled: "
               f"{len(report.tree.unsampled_nodes)}")
    for diagnostic in report.diagnostics:
        if diagnostic.severity == "ERROR":
            typer.echo(f"[ERROR] {diagnostic.code}: {diagnostic.message}", err=True)
    typer.echo(f"Report: {path}")
    if report.has_errors:
        raise typer.Exit(code=ExitCode.ERROR)


@generative_app.command("budget", epilog=GENERATIVE_CAVEAT)
def generative_budget_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    assumed_rate: Annotated[float, typer.Option("--assumed-rate", min=0.0, max=0.999)] = 0.03,
    budget: Annotated[list[int] | None, typer.Option("--budget", min=1)] = None,
    threshold: Annotated[float | None, typer.Option("--threshold", min=0.0, max=1.0)] = None,
    acceptance_rate: Annotated[
        float, typer.Option("--acceptance-rate", min=0.01, max=1.0)
    ] = 1.0,
    target_bound: Annotated[
        float | None, typer.Option("--target-bound", min=0.0, max=1.0)
    ] = None,
) -> None:
    """Project what a budget will buy, before generating or evaluating anything.

    Everything reported comes from the compiled taxonomy and one assumed failure rate.
    The projection assumes risk is uniform across the design; real heterogeneity moves
    individual nodes either way, so node figures show what the geometry permits rather
    than what will happen.
    """

    from inspect_safety_stats.generative.planning import forecast, rewrites_for_bound
    from inspect_safety_stats.generative.schema import load_generative_plan
    from inspect_safety_stats.generative.taxonomy import compile_taxonomy

    try:
        parsed = load_generative_plan(plan)
        taxonomy = compile_taxonomy(parsed, base_path=plan.parent)
    except Exception as exc:
        typer.echo(f"Planning failed: {exc}", err=True)
        raise typer.Exit(code=ExitCode.ERROR) from exc

    declared = parsed.sampling.total_rewrites
    budgets = sorted(set(budget)) if budget else sorted(
        {max(1, int(declared * factor)) for factor in (0.25, 1, 4, 16)}
    )

    typer.echo(f"Evaluation: {parsed.evaluation.id}")
    typer.echo(
        f"Taxonomy: {len(taxonomy.terminal_paths)} terminal paths, "
        f"{len(taxonomy.levels)} levels, declared budget {declared}"
    )
    typer.echo(
        f"Assumed failure rate {assumed_rate}, alpha {parsed.inference.alpha}"
        + (f", acceptance rate {acceptance_rate}" if acceptance_rate < 1.0 else "")
    )
    typer.echo("")

    header = f"{'budget':>9} {'root bound':>11} {'paths hit':>10} {'unsampled':>10}"
    header += f"{'rewrite calls':>15} {'rollouts':>10}"
    if threshold is not None:
        header += f"{'certif. power':>14}"
    typer.echo(header)
    projected = {
        total: forecast(
            taxonomy,
            parsed,
            total_rewrites=total,
            assumed_rate=assumed_rate,
            acceptance_rate=acceptance_rate,
            threshold=threshold,
        )
        for total in budgets
    }
    for total, item in projected.items():
        row = (
            f"{total:>9} {item.root_upper:>11.4f} {item.paths_with_a_draw:>9.0%} "
            f"{item.expected_unsampled_nodes:>10.0f}{item.rewrite_calls:>15.0f} "
            f"{item.target_rollouts:>9}"
        )
        if threshold is not None and item.certification_power is not None:
            row += f"{item.certification_power:>14.2f}"
        if item.starved_strata:
            row += "   infeasible: the declared stratum floor needs more"
        typer.echo(row)

    # Detail the budget the caller actually asked about.
    detail = max(budgets) if budget else declared
    chosen = projected.get(detail) or forecast(
        taxonomy,
        parsed,
        total_rewrites=detail,
        assumed_rate=assumed_rate,
        acceptance_rate=acceptance_rate,
        threshold=threshold,
    )
    typer.echo(f"\nAt a budget of {detail}, by level:")
    typer.echo(
        f"  {'level':<20} {'nodes':>6} {'median draws':>13} {'median bound':>13} "
        f"{'worst bound':>12} {'unsampled':>10}"
    )
    for level in chosen.levels:
        typer.echo(
            f"  {level.level:<20} {level.nodes:>6} {level.median_draws:>13.1f} "
            f"{level.median_upper:>13.4f} {level.worst_upper:>12.4f} "
            f"{level.expected_unsampled:>10.1f}"
        )

    notes: list[str] = []
    for level in chosen.levels:
        if level.median_upper > 4 * chosen.root_upper:
            notes.append(
                f"level {level.level!r} cannot be resolved at this budget: its median node "
                f"would report {level.median_upper:.3f} against a root bound of "
                f"{chosen.root_upper:.4f}"
            )
    if chosen.exhausted_units >= 1:
        notes.append(
            f"about {chosen.exhausted_units:.0f} units would exhaust their retries and be "
            "completed as unclassified"
        )
    if chosen.starved_strata:
        notes.append(f"{chosen.starved_strata} strata would receive no draws")
    elif chosen.coefficient_ratio > 1.0 and chosen.coefficient_ratio != float("inf"):
        notes.append(
            f"allocation is not exactly proportional (coefficient ratio "
            f"{chosen.coefficient_ratio:.4f}), so the sharp exact kernel is unavailable"
        )
    if threshold is not None and chosen.certification_power is not None:
        verdict = "would" if chosen.certification_power >= 0.8 else "would not"
        notes.append(
            f"certifying root risk below {threshold} {verdict} be reliable at {detail} "
            f"rewrites (power {chosen.certification_power:.2f})"
        )
    if target_bound is not None:
        needed = rewrites_for_bound(
            assumed_rate=assumed_rate, target_upper=target_bound, alpha=parsed.inference.alpha
        )
        notes.append(
            f"reaching a root bound of {target_bound} needs about {needed} rewrites"
            if needed
            else f"a root bound of {target_bound} is unreachable at the assumed rate"
        )
    if notes:
        typer.echo("")
        for note in notes:
            typer.echo(f"! {note}")


@schema_app.command("generative-plan", epilog=GENERATIVE_CAVEAT)
def generative_plan_schema_command() -> None:
    """Print the JSON schema for a generative evaluation plan (experimental)."""

    from inspect_safety_stats.generative.schema import GenerativePlan

    typer.echo(json.dumps(GenerativePlan.model_json_schema(), indent=2))


if __name__ == "__main__":
    app()
