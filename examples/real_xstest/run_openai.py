"""Run a multi-rollout OpenAI XSTest evaluation and produce a safety report."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval
from xstest_live import xstest_openai

from inspect_safety_stats import analyze

HERE = Path(__file__).resolve().parent


def _inspect_model_name(value: str) -> str:
    return value if "/" in value else f"openai/{value}"


def _write_design(destination: Path, rollouts: int) -> None:
    with (HERE / "design.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["expected_rollouts"] = str(rollouts)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plan(destination: Path, model: str, run_id: str, frozen_at: str) -> None:
    with (HERE / "live-plan-template.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    plan["analysis"]["id"] = run_id
    plan["analysis"]["title"] = f"Live multi-rollout XSTest evaluation: {model}"
    plan["analysis"]["plan_frozen_at"] = frozen_at
    plan["cohorts"]["candidate"]["match"]["model"] = model
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gpt-5.6-sol",
        help="Evaluated model, with or without the openai/ Inspect provider prefix.",
    )
    parser.add_argument(
        "--grader-model",
        default=None,
        help="Refusal judge model. Defaults to the evaluated model.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Independent API calls per fixed prompt (default: 10).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum concurrent samples (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New run directory. Defaults to examples/real_xstest/live-runs/<UTC timestamp>.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; export it before running this script.")
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be at least 1")
    if args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1")

    model = _inspect_model_name(args.model)
    grader_model = _inspect_model_name(args.grader_model or args.model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"xstest-live-{timestamp.lower()}"
    run_dir = (args.output or HERE / "live-runs" / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"
    report_dir = run_dir / "report"
    plan_path = run_dir / "safety-plan.yaml"

    _write_design(run_dir / "design.csv", args.rollouts)
    _write_plan(
        plan_path,
        model=model,
        run_id=run_id,
        frozen_at=datetime.now(timezone.utc).isoformat(),
    )

    expected_calls = 2 * 8 * args.rollouts
    print(f"Run directory: {run_dir}")
    print(f"Evaluated model: {model}")
    print(f"Grader model: {grader_model}")
    print(f"Rollouts: {args.rollouts} per prompt (8 prompts)")
    print(f"Expected model calls: approximately {expected_calls} including grading")

    logs = inspect_eval(
        xstest_openai(grader_model=grader_model),
        model=model,
        epochs=args.rollouts,
        log_dir=str(log_dir),
        log_format="eval",
        max_samples=args.max_samples,
        display="plain",
        ctl_server=False,
    )
    if not logs or any(str(log.status) != "success" for log in logs):
        raise RuntimeError("Inspect evaluation did not complete successfully; inspect the logs.")

    report = analyze(plan=plan_path, logs=log_dir)
    report.write(report_dir)
    print(f"Decision: {report.decision}")
    print(f"Markdown report: {report_dir / 'report.md'}")
    print(f"Machine-readable report: {report_dir / 'report.json'}")


if __name__ == "__main__":
    main()
