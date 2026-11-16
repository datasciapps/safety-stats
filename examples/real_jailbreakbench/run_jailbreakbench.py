"""Run and analyze a complete multi-model JailbreakBench-100 evaluation."""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log
from jailbreakbench_live import behavior_id, behavior_rows, category_id, jailbreakbench_100

from inspect_safety_stats import analyze

HERE = Path(__file__).resolve().parent


def inspect_model_name(value: str) -> str:
    return value if "/" in value else f"openai/{value}"


def _read_optional_text(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"text file is empty: {path}")
    return value


def write_design(destination: Path, rollouts: int, attack_name: str) -> None:
    rows = [
        {
            "prompt_id": behavior_id(row["Index"]),
            "question_id": behavior_id(row["Index"]),
            "task_id": "jailbreakbench-harmful-compliance",
            "category_id": category_id(row["Category"]),
            "category": row["Category"],
            "behavior_label": row["Behavior"],
            "source_dataset": row["Source"],
            "attack_method": attack_name,
            "expected_rollouts": str(rollouts),
            "zero_tolerance": "false",
            "mandatory": "true",
        }
        for row in behavior_rows()
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plan(
    destination: Path,
    *,
    arms: dict[str, str],
    reference: str,
    grader_model: str,
    run_id: str,
    frozen_at: str,
    attack_name: str,
) -> None:
    with (HERE / "live-comparison-plan-template.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    plan["analysis"]["id"] = run_id
    plan["analysis"]["title"] = f"JailbreakBench-100: {len(arms)}-arm model comparison"
    plan["analysis"]["description"] += (
        f" Prompt transformation: {attack_name}. Every arm is graded by {grader_model}."
    )
    plan["analysis"]["plan_frozen_at"] = frozen_at
    # Every cohort matches the same grader, so a difference between arms cannot be a
    # difference between judges: a log graded by another model joins no cohort.
    plan["cohorts"] = {
        arm_id: {
            "match": {"model": model, "task_arg_grader_model": grader_model},
            "require_uniform": ["task_version", "model_generate_config", "solver"],
        }
        for arm_id, model in arms.items()
    }
    comparison_set = plan["comparison_sets"][0]
    comparison_set["cohorts"] = list(arms)
    comparison_set["contrasts"]["reference"] = reference
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)


def analyze_run(run_dir: Path, *, overwrite: bool = False) -> Path:
    """Analyze completed logs and write the self-contained report and visualizer."""

    report = analyze(
        plan=run_dir / "analysis-plan.yaml",
        logs=latest_successful_logs(run_dir),
    )
    destination = report.write(run_dir / "report", overwrite=overwrite)
    return destination


def latest_successful_logs(run_dir: Path) -> list[str]:
    """Select the newest successful Inspect log for each declared model arm and its grader."""

    with (run_dir / "analysis-plan.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    expected = [
        (str(cohort["match"]["model"]), str(cohort["match"]["task_arg_grader_model"]))
        for cohort in plan["cohorts"].values()
    ]
    selected: dict[tuple[str, str], tuple[float, str]] = {}
    for info in list_eval_logs(str(run_dir / "logs"), formats=["eval"], recursive=True):
        log = read_eval_log(info, header_only=True)
        key = (str(log.eval.model), str((log.eval.task_args or {}).get("grader_model")))
        if str(log.status) != "success" or key not in expected:
            continue
        candidate = (info.mtime or 0.0, info.name)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    missing = [
        f"{model} graded by {grader}"
        for model, grader in expected
        if (model, grader) not in selected
    ]
    if missing:
        raise RuntimeError(
            "no successful Inspect log is available for model arm(s): " + ", ".join(missing)
        )
    return [selected[key][1] for key in expected]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="ID=MODEL",
        help="Cohort identifier and Inspect model name; repeat at least three times.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        metavar="ID",
        help="Cohort identifier used as the reference arm.",
    )
    parser.add_argument(
        "--grader-model",
        required=True,
        help="Inspect model used for the binary harmful-compliance classification.",
    )
    parser.add_argument("--rollouts", type=int, default=8, help="Rollouts per behavior and arm.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument(
        "--prompt-template-file",
        type=Path,
        default=None,
        help="Optional fixed template containing exactly one {goal} placeholder.",
    )
    parser.add_argument(
        "--attack-name",
        default="direct_request",
        help="Stable label recorded for the fixed prompt transformation.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional common system prompt used for every comparison arm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New run directory; defaults to live-runs/<UTC timestamp>.",
    )
    return parser.parse_args()


def _parse_arms(values: list[str]) -> dict[str, str]:
    arms: dict[str, str] = {}
    for value in values:
        arm_id, separator, model = value.partition("=")
        if not separator or not arm_id or not model:
            raise SystemExit("every --arm must have the form ID=MODEL")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", arm_id) is None:
            raise SystemExit(f"invalid arm identifier: {arm_id!r}")
        if arm_id in arms:
            raise SystemExit(f"duplicate arm identifier: {arm_id!r}")
        arms[arm_id] = inspect_model_name(model)
    if len(arms) < 3:
        raise SystemExit("provide at least three --arm values")
    if len(set(arms.values())) != len(arms):
        raise SystemExit("every arm must currently use a distinct Inspect model name")
    return arms


def main() -> None:
    args = _arguments()
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be at least 1")
    if args.max_samples < 1 or args.max_tokens < 1:
        raise SystemExit("--max-samples and --max-tokens must be at least 1")
    if args.max_retries < 0 or args.sample_retries < 0:
        raise SystemExit("retry counts must be nonnegative")
    if args.prompt_template_file is None and args.attack_name != "direct_request":
        raise SystemExit("a non-direct --attack-name requires --prompt-template-file")

    arms = _parse_arms(args.arm)
    if args.reference not in arms:
        raise SystemExit("--reference must name one of the declared --arm identifiers")
    grader_model = inspect_model_name(args.grader_model)
    if any(
        model.startswith("openai/")
        for model in (*arms.values(), grader_model)
    ) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; export it before running OpenAI models.")

    prompt_template = _read_optional_text(args.prompt_template_file)
    system_prompt = _read_optional_text(args.system_prompt_file)
    if prompt_template is not None and prompt_template.count("{goal}") != 1:
        raise SystemExit("--prompt-template-file must contain exactly one {goal} placeholder")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"jbb100-comparison-{timestamp.lower()}"
    run_dir = (args.output or HERE / "live-runs" / timestamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    log_dir = run_dir / "logs"

    write_design(run_dir / "design.csv", args.rollouts, args.attack_name)
    write_plan(
        run_dir / "analysis-plan.yaml",
        arms=arms,
        reference=args.reference,
        grader_model=grader_model,
        run_id=run_id,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        attack_name=args.attack_name,
    )

    target_calls = len(arms) * 100 * args.rollouts
    judge_calls = target_calls
    print(f"Run directory: {run_dir}")
    print(f"Reference arm: {args.reference} ({arms[args.reference]})")
    print("Arms: " + ", ".join(f"{arm}={model}" for arm, model in arms.items()))
    print(f"Grader: {grader_model}")
    print(f"Fixed design: 100 harmful behaviors, {args.rollouts} rollouts per arm")
    print(f"Expected calls: {target_calls} target generations + {judge_calls} grading calls")

    for arm, model in arms.items():
        print(f"Starting {arm} arm: {model}")
        logs = inspect_eval(
            jailbreakbench_100(
                grader_model=grader_model,
                prompt_template=prompt_template,
                attack_name=args.attack_name,
                system_prompt=system_prompt,
            ),
            model=model,
            epochs=args.rollouts,
            log_dir=str(log_dir),
            log_format="eval",
            max_samples=args.max_samples,
            max_retries=args.max_retries,
            retry_on_error=args.sample_retries,
            fail_on_error=False,
            display="plain",
            ctl_server=False,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if not logs or any(str(log.status) != "success" for log in logs):
            raise RuntimeError(
                f"{arm} evaluation did not complete successfully; logs were retained in {log_dir}"
            )

    report_dir = analyze_run(run_dir)
    print(f"Statistical report: {report_dir / 'report.md'}")
    print(f"Static visualizer: {report_dir / 'report.html'}")
    print(f"Comparison data: {report_dir / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
