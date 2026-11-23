"""Append one language arm to a completed multilingual JailbreakBench run."""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log
from jailbreakbench_multilingual import (
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_REVISION,
    LANGUAGE_BY_ID,
    LanguageSpec,
    extend_translation_artifact,
    jailbreakbench_100_multilingual,
    load_translation_artifact,
    metadata_path,
    multilingual_task_name,
)
from run_multilingual_jailbreakbench import latest_successful_logs
from translate_jailbreakbench import NllbTranslator

from safety_stats import analyze


@dataclass(frozen=True)
class ExistingRunSettings:
    target_model: str
    grader_model: str
    rollouts: int
    temperature: float | None
    max_tokens: int | None
    max_retries: int


def _successful_headers(run_dir: Path) -> list[Any]:
    return [read_eval_log(path, header_only=True) for path in latest_successful_logs(run_dir)]


def existing_run_settings(run_dir: Path) -> ExistingRunSettings:
    """Infer and validate settings shared by the already completed language arms."""

    headers = _successful_headers(run_dir)
    models = {str(log.eval.model) for log in headers}
    graders = {str(log.eval.metadata.get("grader_model")) for log in headers}
    rollouts = {int(log.eval.config.epochs or 0) for log in headers}
    generate_configs = [
        log.eval.model_generate_config.model_dump(mode="python", exclude_none=True)
        for log in headers
    ]
    if len(models) != 1 or len(graders) != 1 or len(rollouts) != 1:
        raise ValueError("existing language logs do not share one model, grader, and rollout count")
    if any(config != generate_configs[0] for config in generate_configs[1:]):
        raise ValueError("existing language logs do not share one generation configuration")
    config = generate_configs[0]
    unsupported = set(config) - {"max_retries", "max_tokens", "temperature"}
    if unsupported:
        raise ValueError(
            "incremental runner cannot safely reproduce generation setting(s): "
            + ", ".join(sorted(unsupported))
        )
    rollout_count = next(iter(rollouts))
    if rollout_count < 1:
        raise ValueError("existing language logs have no positive epoch count")
    return ExistingRunSettings(
        target_model=next(iter(models)),
        grader_model=next(iter(graders)),
        rollouts=rollout_count,
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
        max_retries=int(config.get("max_retries", 0)),
    )


def _source_translation_artifact(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "inputs").glob("*.csv"))
    valid: list[Path] = []
    for candidate in candidates:
        if not metadata_path(candidate).is_file():
            continue
        try:
            _, candidate_metadata = load_translation_artifact(candidate)
        except (FileNotFoundError, ValueError):
            continue
        if candidate_metadata.get("parent_artifact_sha256") is not None:
            continue
        valid.append(candidate)
    if len(valid) != 1:
        raise ValueError(
            f"expected exactly one source translation artifact in {run_dir / 'inputs'}, "
            f"found {len(valid)}"
        )
    return valid[0]


def _has_successful_language_log(
    run_dir: Path, *, target_model: str, language_id: str
) -> bool:
    expected_task = multilingual_task_name(language_id)
    for info in list_eval_logs(str(run_dir / "logs"), formats=["eval"], recursive=True):
        log = read_eval_log(info, header_only=True)
        if (
            str(log.status) == "success"
            and str(log.eval.model) == target_model
            and str(log.eval.task) == expected_task
        ):
            return True
    return False


def extend_analysis_plan(
    run_dir: Path,
    *,
    language: LanguageSpec,
    target_model: str,
    artifact_sha256: str,
) -> Path:
    """Add one cohort and include it in the existing all-pairs comparison set."""

    plan_path = run_dir / "analysis-plan.yaml"
    with plan_path.open(encoding="utf-8") as stream:
        plan: dict[str, Any] = yaml.safe_load(stream)
    if language.id in plan["cohorts"]:
        return plan_path
    backup = run_dir / f"analysis-plan.{len(plan['cohorts'])}-language.yaml"
    if not backup.exists():
        shutil.copy2(plan_path, backup)
    plan["cohorts"][language.id] = {
        "match": {
            "model": target_model,
            "task_name": multilingual_task_name(language.id),
        },
        "require_uniform": ["task_version", "model_generate_config", "solver"],
    }
    comparison_sets = [
        item for item in plan.get("comparison_sets", []) if item["id"] == "language_conditions"
    ]
    if len(comparison_sets) != 1:
        raise ValueError("analysis plan must contain one language_conditions comparison set")
    comparison_sets[0]["cohorts"].append(language.id)
    plan["analysis"]["title"] = (
        f"JailbreakBench-100 {len(plan['cohorts'])}-language comparison: {target_model}"
    )
    plan["analysis"]["description"] += (
        f" Incrementally added {language.name}; extended translation artifact SHA-256: "
        f"{artifact_sha256}."
    )
    with plan_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(plan, stream, sort_keys=False)
    return plan_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--language", default="maltese")
    parser.add_argument("--translation-model", default=DEFAULT_TRANSLATION_MODEL)
    parser.add_argument("--translation-revision", default=DEFAULT_TRANSLATION_REVISION)
    parser.add_argument("--translation-device", default="auto")
    parser.add_argument("--translation-batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--sample-retries", type=int, default=2)
    parser.add_argument(
        "--report-name",
        default="report-4-language",
        help="New report directory name; the original report is retained.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    run_dir = args.run_dir.resolve()
    if args.language not in LANGUAGE_BY_ID:
        raise SystemExit(f"unsupported --language: {args.language}")
    language = LANGUAGE_BY_ID[args.language]
    if args.max_samples < 1 or args.sample_retries < 0 or args.translation_batch_size < 1:
        raise SystemExit("concurrency and batch size must be positive; retries must be nonnegative")
    settings = existing_run_settings(run_dir)
    if any(
        model.startswith("openai/")
        for model in (settings.target_model, settings.grader_model)
    ) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    source_artifact = _source_translation_artifact(run_dir)
    _, source_metadata = load_translation_artifact(source_artifact)
    extended_artifact = run_dir / "inputs" / "jbb100-en-fr-de-mt.csv"
    if extended_artifact.exists():
        translations, extended_metadata = load_translation_artifact(extended_artifact)
        if language.id not in translations:
            raise ValueError("existing extended artifact does not contain the requested language")
        if extended_metadata.get("parent_artifact_sha256") != source_metadata["artifact_sha256"]:
            raise ValueError("existing extended artifact does not descend from the source artifact")
    else:
        print(f"Loading translation model for {language.name}: {args.translation_model}")
        translator = NllbTranslator(
            model_id=args.translation_model,
            revision=args.translation_revision,
            device=args.translation_device,
            max_source_tokens=512,
            max_new_tokens=512,
            num_beams=5,
            local_files_only=False,
        )
        extend_translation_artifact(
            source_artifact,
            extended_artifact,
            language=language,
            translate=translator.translate,
            translation_model=translator.model_id,
            requested_revision=translator.requested_revision,
            resolved_revision=translator.resolved_revision,
            generation_config=translator.generation_config,
            batch_size=args.translation_batch_size,
        )
        _, extended_metadata = load_translation_artifact(extended_artifact)
        print(f"Frozen {language.name} translation added without changing existing rows")

    if not _has_successful_language_log(
        run_dir, target_model=settings.target_model, language_id=language.id
    ):
        print(
            f"Starting {language.name}: 100 prompts x {settings.rollouts} rollouts; "
            f"target={settings.target_model}; grader={settings.grader_model}"
        )
        eval_kwargs: dict[str, Any] = {
            "model": settings.target_model,
            "epochs": settings.rollouts,
            "log_dir": str(run_dir / "logs"),
            "log_format": "eval",
            "max_samples": args.max_samples,
            "max_retries": settings.max_retries,
            "retry_on_error": args.sample_retries,
            "fail_on_error": False,
            "display": "plain",
            "ctl_server": False,
        }
        if settings.temperature is not None:
            eval_kwargs["temperature"] = settings.temperature
        if settings.max_tokens is not None:
            eval_kwargs["max_tokens"] = settings.max_tokens
        logs = inspect_eval(
            jailbreakbench_100_multilingual(
                grader_model=settings.grader_model,
                translation_artifact=str(extended_artifact),
                language_id=language.id,
            ),
            **eval_kwargs,
        )
        if not logs or any(str(log.status) != "success" for log in logs):
            raise RuntimeError(
                f"{language.name} evaluation did not complete successfully; logs were retained"
            )
    else:
        print(f"Reusing the existing successful {language.name} log")

    extend_analysis_plan(
        run_dir,
        language=language,
        target_model=settings.target_model,
        artifact_sha256=str(extended_metadata["artifact_sha256"]),
    )
    report = analyze(
        plan=run_dir / "analysis-plan.yaml",
        logs=latest_successful_logs(run_dir),
    )
    destination = report.write(run_dir / args.report_name, overwrite=True)
    print(f"Four-language report: {destination / 'report.md'}")
    print(f"Four-language visualizer: {destination / 'report.html'}")


if __name__ == "__main__":
    main()
