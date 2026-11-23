"""Run and analyze English/French/German/Maltese JailbreakBench-100 conditions."""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log
from jailbreakbench_multilingual import (
    LANGUAGES,
    jailbreakbench_100_multilingual,
    load_translation_artifact,
    metadata_path,
    multilingual_task_name,
)
from run_jailbreakbench import inspect_model_name, write_design

from safety_stats import analyze

HERE = Path(__file__).resolve().parent


def write_plan(
    destination: Path,
    *,
    target_model: str,
    run_id: str,
    frozen_at: str,
    translation_artifact_sha256: str,
) -> None:
    with (HERE / "multilingual-comparison-plan-template.yaml").open(
        encoding="utf-8"
    ) as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    plan["analysis"]["id"] = run_id
    plan["analysis"]["title"] = f"JailbreakBench-100 language comparison: {target_model}"
    plan["analysis"]["description"] += (
        f" Frozen translation artifact SHA-256: {translation_artifact_sha256}."
    )
    plan["analysis"]["plan_frozen_at"] = frozen_at
    for language in LANGUAGES:
        plan["cohorts"][language.id]["match"]["model"] = target_model
    with destination.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)


def latest_successful_logs(run_dir: Path) -> list[str]:
    """Select the newest successful log for each planned model/task pair."""

    with (run_dir / "analysis-plan.yaml").open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    expected = [
        (str(cohort["match"]["model"]), str(cohort["match"]["task_name"]))
        for cohort in plan["cohorts"].values()
    ]
    selected: dict[tuple[str, str], tuple[float, str]] = {}
    for info in list_eval_logs(str(run_dir / "logs"), formats=["eval"], recursive=True):
        log = read_eval_log(info, header_only=True)
        key = (str(log.eval.model), str(log.eval.task))
        if str(log.status) != "success" or key not in expected:
            continue
        candidate = (info.mtime or 0.0, info.name)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate
    missing = [f"{model} / {task}" for model, task in expected if (model, task) not in selected]
    if missing:
        raise RuntimeError(
            "no successful Inspect log is available for language arm(s): "
            + ", ".join(missing)
        )
    return [selected[key][1] for key in expected]


def analyze_run(run_dir: Path, *, overwrite: bool = False) -> Path:
    """Analyze completed language logs and write the unified static report."""

    report = analyze(
        plan=run_dir / "analysis-plan.yaml",
        logs=latest_successful_logs(run_dir),
    )
    return report.write(run_dir / "report", overwrite=overwrite)


def _read_optional_text(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"text file is empty: {path}")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--grader-model", required=True)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional identical system prompt for all language conditions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="New run directory; defaults to multilingual-live-runs/<UTC timestamp>.",
    )
    return parser.parse_args()


def _copy_translation_inputs(source: Path, run_dir: Path) -> Path:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    destination = inputs_dir / "jbb100-en-fr-de-mt.csv"
    shutil.copy2(source, destination)
    shutil.copy2(metadata_path(source), metadata_path(destination))
    load_translation_artifact(destination)
    return destination


def main() -> None:
    args = _arguments()
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be at least 1")
    if args.max_samples < 1 or args.max_tokens < 1:
        raise SystemExit("--max-samples and --max-tokens must be at least 1")
    if args.max_retries < 0 or args.sample_retries < 0:
        raise SystemExit("retry counts must be nonnegative")

    source_translations = args.translations.resolve()
    _, translation_metadata = load_translation_artifact(source_translations)
    target_model = inspect_model_name(args.target_model)
    grader_model = inspect_model_name(args.grader_model)
    if any(
        model.startswith("openai/") for model in (target_model, grader_model)
    ) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; export it before running OpenAI models.")
    system_prompt = _read_optional_text(args.system_prompt_file)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"jbb100-multilingual-{timestamp.lower()}"
    run_dir = (
        args.output or HERE / "multilingual-live-runs" / timestamp
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    frozen_translations = _copy_translation_inputs(source_translations, run_dir)
    log_dir = run_dir / "logs"

    write_design(
        run_dir / "design.csv",
        rollouts=args.rollouts,
        attack_name="translated_direct_request",
    )
    write_plan(
        run_dir / "analysis-plan.yaml",
        target_model=target_model,
        run_id=run_id,
        frozen_at=datetime.now(timezone.utc).isoformat(),
        translation_artifact_sha256=str(translation_metadata["artifact_sha256"]),
    )

    target_calls = len(LANGUAGES) * 100 * args.rollouts
    print(f"Run directory: {run_dir}")
    print(f"Target model: {target_model}")
    print(f"Grader: {grader_model}")
    print("Language conditions: " + ", ".join(language.name for language in LANGUAGES))
    print(f"Fixed design: 100 matched behaviors, {args.rollouts} rollouts per language")
    print(f"Expected calls: {target_calls} target generations + {target_calls} grading calls")

    for language in LANGUAGES:
        print(f"Starting {language.name} condition: {multilingual_task_name(language.id)}")
        logs = inspect_eval(
            jailbreakbench_100_multilingual(
                grader_model=grader_model,
                translation_artifact=str(frozen_translations),
                language_id=language.id,
                system_prompt=system_prompt,
            ),
            model=target_model,
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
                f"{language.name} evaluation did not complete successfully; "
                f"logs were retained in {log_dir}"
            )

    report_dir = analyze_run(run_dir)
    print(f"Statistical report: {report_dir / 'report.md'}")
    print(f"Static visualizer: {report_dir / 'report.html'}")
    print(f"Language comparisons: {report_dir / 'comparisons.csv'}")


if __name__ == "__main__":
    main()
