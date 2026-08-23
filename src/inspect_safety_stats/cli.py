from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from inspect_safety_stats.api import analyze as run_analysis
from inspect_safety_stats.api import validate as run_validation
from inspect_safety_stats.certification import certify as run_certification
from inspect_safety_stats.certification.config import CertificationPlan
from inspect_safety_stats.certification.fixed_design_planner import (
    FixedDesignPlanningReport,
    plan_fixed_design,
)
from inspect_safety_stats.certification.planning import (
    certification_power,
    certification_sample_size,
    maximum_certifiable_failure_rate,
    maximum_certifying_failures,
    minimum_detectable_failure_rate,
)
from inspect_safety_stats.certification.schema import CertificationReport
from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.report.visualization import write_visualization
from inspect_safety_stats.schema.reports import SafetyReport
from inspect_safety_stats.schema.structural_reports import StructuralReport
from inspect_safety_stats.stats.planning import zero_failure_sample_size
from inspect_safety_stats.structural.artifacts import write_structural_visualization
from inspect_safety_stats.structural.fit import fit_structural_model, validate_structural_inputs
from inspect_safety_stats.version import __version__

app = typer.Typer(help="Fixed-design statistical risk analysis for Inspect AI evaluations.")
plan_app = typer.Typer(help="Prospective evaluation planning tools.")
schema_app = typer.Typer(help="Print machine-readable schemas.")
model_app = typer.Typer(help="Non-certifying hierarchical safety-landscape modelling.")
app.add_typer(plan_app, name="plan")
app.add_typer(schema_app, name="schema")
app.add_typer(model_app, name="model")


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
    else:
        typer.echo(f"Risk measures: {len(report.group_risks)}")
    typer.echo(f"Report: {path}")
    if decision_exit_codes:
        raise typer.Exit(code={"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report.decision])


@app.command("validate")
def validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    logs: Annotated[list[Path] | None, typer.Option("--logs", exists=True)] = None,
    parallel: Annotated[int, typer.Option("--parallel", min=0)] = 0,
) -> None:
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


@app.command("certify")
def certify_command(
    report: Annotated[Path, typer.Option("--report", exists=True)],
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("certification-report"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    evidence_exit_codes: Annotated[
        bool, typer.Option("--evidence-exit-codes/--no-evidence-exit-codes")
    ] = False,
) -> None:
    """Evaluate predeclared one-sided statistical certification claims."""

    try:
        certification = run_certification(empirical_report=report, plan=plan)
        destination = certification.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Certification failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Certification: {certification.analysis['id']}")
    typer.echo(f"Supported: {certification.summary['supported']}")
    typer.echo(f"Rejected: {certification.summary['rejected']}")
    typer.echo(f"Inconclusive: {certification.summary['inconclusive']}")
    typer.echo(f"Report: {destination}")
    if evidence_exit_codes:
        code = (
            1
            if certification.summary["rejected"]
            else 2
            if certification.summary["inconclusive"]
            else 0
        )
        raise typer.Exit(code=code)


@model_app.command("validate")
def model_validate_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    report: Annotated[Path, typer.Option("--report", exists=True)],
) -> None:
    """Validate the structural model frame without fitting a Bayesian model."""

    try:
        metadata = validate_structural_inputs(plan=plan, empirical_report=report)
    except Exception as exc:
        typer.echo(f"Structural validation failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo("Structural inputs: valid")
    typer.echo(f"Cohort: {metadata['cohort_id']}")
    typer.echo(f"Group: {metadata['group_id']}")
    typer.echo(f"Prompts: {metadata['prompt_count']}")
    typer.echo(f"Rollouts: {metadata['rollout_count']}")


@model_app.command("fit")
def model_fit_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    report: Annotated[Path, typer.Option("--report", exists=True)],
    output: Annotated[Path, typer.Option("--output")] = Path("structural-model-report"),
    cross_validation: Annotated[
        bool, typer.Option("--cross-validation/--no-cross-validation")
    ] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Fit the non-certifying hierarchical safety-landscape model."""

    try:
        structural = fit_structural_model(
            plan=plan,
            empirical_report=report,
            run_cross_validation=cross_validation,
        )
        destination = structural.write(output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Structural fit failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Structural model status: {structural.status}")
    typer.echo("Certification eligible: no")
    typer.echo(f"Report: {destination}")


@model_app.command("visualize")
def model_visualize_command(
    report: Annotated[Path, typer.Option("--report", exists=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Regenerate the standalone structural-model visualization."""

    try:
        path = write_structural_visualization(report, output, overwrite=overwrite)
    except Exception as exc:
        typer.echo(f"Structural visualization failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Structural visualization: {path}")


@plan_app.command("zero-failure")
def zero_failure_command(
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)],
    alpha: Annotated[float, typer.Option("--alpha", min=0.0, max=1.0)] = 0.05,
) -> None:
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


@plan_app.command("fixed-design")
def fixed_design_planning_command(
    analysis_plan: Annotated[Path, typer.Option("--analysis-plan", exists=True, dir_okay=False)],
    certification_plan: Annotated[
        Path, typer.Option("--certification-plan", exists=True, dir_okay=False)
    ],
    anticipated_rate: Annotated[float, typer.Option("--anticipated-rate", min=0.0, max=1.0)],
    power: Annotated[float, typer.Option("--power", min=0.0, max=1.0)] = 0.9,
    direction: Annotated[str, typer.Option("--direction")] = "support",
    minimum_rollouts: Annotated[int, typer.Option("--minimum-rollouts", min=1)] = 1,
    maximum_rollouts: Annotated[int | None, typer.Option("--maximum-rollouts", min=1)] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("fixed-design-plan"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Plan fixed-design rollout counts from analysis and certification plans."""

    try:
        if direction not in {"support", "rejection"}:
            raise ValueError("direction must be support or rejection")
        report = plan_fixed_design(
            analysis_plan=analysis_plan,
            certification_plan=certification_plan,
            anticipated_rate=anticipated_rate,
            power=power,
            direction=direction,  # type: ignore[arg-type]
            minimum_rollouts=minimum_rollouts,
            maximum_rollouts=maximum_rollouts,
        )
        destination = report.write(output, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        typer.echo(f"Fixed-design planning failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    typer.echo(f"Feasible: {str(report.summary['feasible']).lower()}")
    typer.echo(
        "Required common rollouts per prompt: "
        f"{report.summary['required_equal_rollouts_per_prompt']}"
    )
    typer.echo(f"Common total rollouts: {report.summary['common_total_rollouts']}")
    typer.echo(f"Adaptive total rollouts: {report.summary['adaptive_total_rollouts']}")
    typer.echo(
        "Adaptive rollout range: "
        f"{report.summary['adaptive_min_rollouts']}–"
        f"{report.summary['adaptive_max_rollouts']}"
    )
    typer.echo(f"Report: {destination}")


@schema_app.command("plan")
def plan_schema_command() -> None:
    typer.echo(json.dumps(AnalysisPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("report")
def report_schema_command() -> None:
    typer.echo(json.dumps(SafetyReport.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("structural-report")
def structural_report_schema_command() -> None:
    typer.echo(json.dumps(StructuralReport.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("certification-plan")
def certification_plan_schema_command() -> None:
    typer.echo(json.dumps(CertificationPlan.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("certification-report")
def certification_report_schema_command() -> None:
    typer.echo(json.dumps(CertificationReport.model_json_schema(), indent=2, sort_keys=True))


@schema_app.command("fixed-design-planning-report")
def fixed_design_planning_report_schema_command() -> None:
    typer.echo(json.dumps(FixedDesignPlanningReport.model_json_schema(), indent=2, sort_keys=True))


@app.command("version")
def version_command() -> None:
    typer.echo(__version__)


@app.command("hash-plan")
def hash_plan_command(
    plan: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
) -> None:
    typer.echo(canonical_plan_hash(load_plan(plan)))


if __name__ == "__main__":
    app()
