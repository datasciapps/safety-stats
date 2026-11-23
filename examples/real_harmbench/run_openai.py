"""Run repeated OpenAI rollouts on all 400 fixed HarmBench text behaviors."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from harmbench_live import _behavior_rows, harmbench_direct_request
from inspect_ai import eval as inspect_eval

from safety_stats import analyze

HERE = Path(__file__).resolve().parent


def _inspect_model_name(value: str) -> str:
    return value if "/" in value else f"openai/{value}"


def _write_design(destination: Path, rollouts: int) -> None:
    rows = []
    for row in _behavior_rows():
        copyright = row["FunctionalCategory"] == "copyright"
        rows.append(
            {
                "prompt_id": row["BehaviorID"],
                "question_id": row["BehaviorID"],
                "task_id": (
                    "harmbench-copyright-reproduction"
                    if copyright
                    else "harmbench-harmful-compliance"
                ),
                "assessment_family": "copyright" if copyright else "harmful_behavior",
                "functional_category": row["FunctionalCategory"],
                "risk_domain": row["SemanticCategory"],
                "attack_method": "direct_request",
                "source_split": row["SourceSplit"],
                "expected_rollouts": str(rollouts),
                "zero_tolerance": "true",
                "mandatory": "true",
            }
        )
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plan(destination: Path, model: str, run_id: str, frozen_at: str) -> None:
    with (HERE / "live-plan-template.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    plan["analysis"]["id"] = run_id
    plan["analysis"]["title"] = f"HarmBench direct-request evaluation: {model}"
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
        help="Harm classifier model. Defaults to the evaluated model.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Independent API calls per fixed behavior (default: 1; set explicitly for repeats).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum concurrent samples (default: 4).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries for each model API request (default: 6).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New run directory. Defaults to examples/real_harmbench/live-runs/<UTC timestamp>.",
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
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be nonnegative")

    model = _inspect_model_name(args.model)
    grader_model = _inspect_model_name(args.grader_model or args.model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"harmbench-text-direct-{timestamp.lower()}"
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

    behavior_count = len(_behavior_rows())
    expected_calls = 2 * behavior_count * args.rollouts
    print(f"Run directory: {run_dir}")
    print(f"Evaluated model: {model}")
    print(f"Grader model: {grader_model}")
    print(f"Rollouts: {args.rollouts} per behavior ({behavior_count} fixed text behaviors)")
    print("Attack method: DirectRequest")
    print(f"Expected model calls: approximately {expected_calls} including grading")

    logs = inspect_eval(
        harmbench_direct_request(grader_model=grader_model),
        model=model,
        epochs=args.rollouts,
        log_dir=str(log_dir),
        log_format="eval",
        max_samples=args.max_samples,
        max_retries=args.max_retries,
        retry_on_error=2,
        fail_on_error=False,
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
