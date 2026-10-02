from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from inspect_safety_stats.api import analyze as run_analysis
from inspect_safety_stats.api import validate as run_validation
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan, read_plan_document
from inspect_safety_stats.config.migrate import migrate_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.judge_labels import draw_label_sample
from inspect_safety_stats.report.visualization import write_visualization
from inspect_safety_stats.scaffold import scaffold_plan
from inspect_safety_stats.schema.reports import SafetyReport
from inspect_safety_stats.stats.planning import (
    certification_power,
    certification_sample_size,
    judge_label_plan,
    maximum_certifiable_failure_rate,
    maximum_certifying_failures,
    minimum_detectable_failure_rate,
    zero_failure_sample_size,
)
from inspect_safety_stats.version import __version__

app = typer.Typer(help="Fixed-design statistical risk analysis for Inspect AI evaluations.")
plan_app = typer.Typer(help="Prospective evaluation planning tools.")
schema_app = typer.Typer(help="Print machine-readable schemas.")
judge_app = typer.Typer(help="Human labelling for validated judge-misclassification correction.")
app.add_typer(plan_app, name="plan")
app.add_typer(schema_app, name="schema")
app.add_typer(judge_app, name="judge")


@app.command("init")
def init_command(
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    failure_values: Annotated[
        list[str] | None,
        typer.Option("--failure-values", help="Score values that count as a safety failure."),
    ] = None,
    safe_values: Annotated[
        list[str] | None,
        typer.Option("--safe-values", help="Score values that count as safe (default: the rest)."),
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
) -> None:
    """Write a starter design manifest and plan from completed Inspect logs.

    The plan reports the root failure risk per model with a prompt-level sensitivity,
    an optional one-level breakdown, and a comparison when there are several models.
    Edit it to add claims, risk measures or judge calibration.
    """

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
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
        )
    except Exception as exc:
        typer.echo(f"Init failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
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
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Plan: {result.plan_path}")
    log_args = " ".join(f"--logs {path}" for path in logs)
    typer.echo(
        f"Next: inspect-safety-stats analyze --plan {result.plan_path} {log_args} "
        "--output <report-dir>"
    )


@app.command("analyze")
def analyze_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("safety-stats-report"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
    decision_exit_codes: Annotated[
        bool, typer.Option("--decision-exit-codes/--no-decision-exit-codes")
    ] = False,
) -> None:
    """Analyze completed Inspect logs against a plan and write the report directory."""

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    try:
        report = run_analysis(plan=plan, logs=log_input, parallel=parallel or False)
        path = report.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Analysis failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Analysis: {report.analysis['id']}")
    typer.echo(f"Plan SHA-256: {report.analysis['plan_sha256']}")
    typer.echo(f"Integrity: {report.integrity.status}")
    if report.presentation.get("include_decisions", False):
        for claim in report.claims:
            if claim.mandatory:
                typer.echo(f"{claim.status}: {claim.title}")
        typer.echo(f"Decision: {report.decision}")
    measures = {
        str(item.get("id")): str(item.get("type"))
        for item in report._resolved_plan.get("outputs", {}).get("risk_measures", []) or []
    }
    for risk in report.group_risks:
        if measures.get(risk.measure_id) != "weighted_mean":
            continue
        interval = (
            f"[{100 * risk.lower:.1f}%, {100 * risk.upper:.1f}%]"
            if risk.lower is not None and risk.upper is not None
            else "interval unavailable"
        )
        estimate = "—" if risk.estimate is None else f"{100 * risk.estimate:.1f}%"
        typer.echo(f"  {risk.cohort_id} / {risk.group_id}: {estimate} {interval}")
    typer.echo(f"Summary: {path / 'report.md'}")
    typer.echo(f"Report: {path}")
    if decision_exit_codes:
        raise typer.Exit(code={"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report.decision])


@app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
) -> None:
    """Check logs against a plan (manifest, cohorts, rollouts, scores) without analyzing."""

    if not logs:
        raise typer.BadParameter("at least one --logs path is required")
    log_input: str | list[str] = str(logs[0]) if len(logs) == 1 else [str(path) for path in logs]
    result = run_validation(plan=plan, logs=log_input, parallel=parallel or False)
    for diagnostic in result.diagnostics:
        scope = f" [{diagnostic.scope}]" if diagnostic.scope else ""
        typer.echo(f"{diagnostic.severity} {diagnostic.code}{scope}: {diagnostic.message}")
    typer.echo(f"Integrity: {result.status}")
    raise typer.Exit(code=3 if result.status == "invalid" else 0)


@app.command("visualize")
def visualize_command(
    report: Annotated[Path, typer.Option("--report", exists=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build a self-contained interactive HTML visualization from a report."""

    try:
        path = write_visualization(report, output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Visualization failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Visualization: {path}")


@plan_app.command("zero-failure")
def zero_failure_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
) -> None:
    """Rollouts needed for zero observed failures to bound the rate below a threshold."""

    count = zero_failure_sample_size(threshold=threshold, alpha=alpha)
    typer.echo(f"Required rollouts: {count}")
    typer.echo(
        f"Decision rule: PASS only if all {count} rollouts are valid and zero failures occur"
    )
    typer.echo(f"One-sided confidence level: {(1 - alpha) * 100:g}%")


@plan_app.command("certification")
def certification_planning_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    anticipated_rate: Annotated[float, typer.Option("--anticipated-rate", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
) -> None:
    """Plan an exact one-sided binomial certification claim."""

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
    typer.echo(f"One-sided alpha: {alpha:g}")


@plan_app.command("detectable-rate")
def detectable_rate_command(
    trials: Annotated[int, typer.Option("--trials", min=1)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
) -> None:
    """Invert exact one-sided tests at a fixed rollout horizon."""

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
    typer.echo(f"One-sided alpha: {alpha:g}")


@plan_app.command("judge-labels")
def judge_labels_planning_command(
    judge_risk_upper: Annotated[
        float,
        typer.Option(
            "--judge-risk-upper",
            min=0.0,
            max=1.0,
            help="Uncorrected (judge-reported) upper bound you expect, e.g. from a pilot.",
        ),
    ],
    target: Annotated[
        float,
        typer.Option("--target", min=0.0, max=1.0, help="True-risk upper bound to certify."),
    ],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
    cohorts: Annotated[int, typer.Option("--cohorts", min=1)] = 1,
    rho_1: Annotated[
        float,
        typer.Option(
            "--rho-1",
            min=0.0,
            max=1.0,
            help="Anticipated P(truly unsafe | judge said unsafe).",
        ),
    ] = 1.0,
    rho_0: Annotated[
        float,
        typer.Option(
            "--rho-0",
            min=0.0,
            max=1.0,
            help="Anticipated P(truly unsafe | judge said safe).",
        ),
    ] = 0.0,
) -> None:
    """Plan human labels per cohort for a validated judge-corrected bound."""

    try:
        result = judge_label_plan(
            judge_risk_upper=judge_risk_upper,
            target_upper=target,
            alpha=alpha,
            cohorts=cohorts,
            anticipated_rho_1=rho_1,
            anticipated_rho_0=rho_0,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    floor = rho_0 + judge_risk_upper * (rho_1 - rho_0)
    if result is None:
        typer.echo(
            f"Unreachable: with unlimited labels the corrected bound only falls to {floor:.4g}; "
            "a better judge, more rollouts, or a looser target is needed."
        )
        raise typer.Exit(code=1)
    typer.echo(f"Per cohort: label {result.judge_unsafe_labels} judge-unsafe and "
               f"{result.judge_safe_labels} judge-safe observations")
    typer.echo(f"Planned rho_1 bounds: [{result.rho_1_lower:.4g}, {result.rho_1_upper:.4g}]")
    typer.echo(f"Planned rho_0 upper bound: {result.rho_0_upper:.4g}")
    typer.echo(f"Planned corrected upper bound: {result.corrected_upper:.4g} (target {target:g})")
    typer.echo(f"Floor with unlimited labels: {result.floor:.4g}")
    typer.echo(f"Endpoint alpha: {result.endpoint_alpha:.4g} "
               f"(alpha {alpha:g} / {cohorts} cohorts / 4 endpoints)")
    typer.echo("Assumes the anticipated accuracies hold; the judge-safe labels usually dominate.")


@judge_app.command("sample")
def judge_sample_command(
    report: Annotated[Path, typer.Option("--report", exists=True, file_okay=False)],
    judge_unsafe: Annotated[
        int,
        typer.Option("--judge-unsafe", min=1, help="Labels per cohort, judge said unsafe."),
    ],
    judge_safe: Annotated[
        int,
        typer.Option("--judge-safe", min=1, help="Labels per cohort, judge said safe."),
    ],
    seed: Annotated[int, typer.Option("--seed", help="Recorded in the audit file.")],
    output: Annotated[Path, typer.Option("--output", file_okay=False)] = Path("."),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Draw a blinded random labelling sample within each judge stratum of each cohort."""

    try:
        result = draw_label_sample(
            report,
            output,
            judge_unsafe=judge_unsafe,
            judge_safe=judge_safe,
            seed=seed,
            overwrite=overwrite,
        )
    except Exception as exc:
        typer.echo(f"Sampling failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    for cohort, strata in result.strata.items():
        unsafe, safe = strata["judge_unsafe"], strata["judge_safe"]
        typer.echo(
            f"{cohort}: {unsafe['drawn']} of {unsafe['stratum_size']} judge-unsafe, "
            f"{safe['drawn']} of {safe['stratum_size']} judge-safe"
        )
    for note in result.notes:
        typer.echo(f"Note: {note}")
    typer.echo(f"Labelling sheet ({result.rows} rows): {result.sheet_path}")
    typer.echo(f"Sampling record: {result.record_path}")
    typer.echo(
        "Fill human_label with safe or unsafe for every row, then set "
        "inputs.judge_calibration to mode: validated with validation_manifest pointing at "
        "the sheet."
    )


@schema_app.command("plan")
def plan_schema_command() -> None:
    """Print the JSON schema for analysis plans."""
    typer.echo(json.dumps(AnalysisPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("report")
def report_schema_command() -> None:
    """Print the JSON schema for reports."""
    typer.echo(json.dumps(SafetyReport.model_json_schema(), indent=2, sort_keys=True))


@app.command("version")
def version_command() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("hash-plan")
def hash_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    """Print a plan's canonical SHA-256, for registering a frozen plan."""

    typer.echo(canonical_plan_hash(load_plan(plan)))


@app.command("migrate-plan")
def migrate_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Upgrade a plan written for an earlier plan schema to the current one.

    The upgraded plan is written to a new file. Review it and re-freeze it
    (analysis.plan_frozen_at) before using it for a confirmatory analysis.
    """

    if output.exists() and not overwrite:
        raise typer.BadParameter(f"{output} exists; pass --overwrite to replace it")
    try:
        upgraded, notes = migrate_plan(read_plan_document(plan))
        AnalysisPlan.model_validate(upgraded)
    except Exception as exc:
        typer.echo(f"Migration failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    output.write_text(yaml.safe_dump(upgraded, sort_keys=False), encoding="utf-8")
    for note in notes:
        typer.echo(f"- {note}")
    typer.echo(f"Upgraded plan: {output}")
    typer.echo("Review it and update analysis.plan_frozen_at before a confirmatory analysis.")


if __name__ == "__main__":
    app()

