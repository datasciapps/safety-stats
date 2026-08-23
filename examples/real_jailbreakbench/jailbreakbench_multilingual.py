"""Frozen multilingual inputs and Inspect tasks for JailbreakBench-100."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message
from jailbreakbench_live import (
    DATA_SHA256,
    DATA_SOURCE,
    behavior_id,
    behavior_rows,
    category_id,
    jbb_harmful_compliance,
)

TRANSLATION_SCHEMA_VERSION = "1.0"
DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_TRANSLATION_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SOURCE_LANGUAGE_CODE = "eng_Latn"


@dataclass(frozen=True)
class LanguageSpec:
    """One predeclared language condition."""

    id: str
    name: str
    nllb_code: str


LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(id="english", name="English", nllb_code="eng_Latn"),
    LanguageSpec(id="french", name="French", nllb_code="fra_Latn"),
    LanguageSpec(id="german", name="German", nllb_code="deu_Latn"),
    LanguageSpec(id="maltese", name="Maltese", nllb_code="mlt_Latn"),
)
LANGUAGE_BY_ID = {language.id: language for language in LANGUAGES}

TRANSLATION_COLUMNS = (
    "prompt_id",
    "language_id",
    "language_name",
    "language_code",
    "translated_goal",
    "source_goal_sha256",
    "translated_goal_sha256",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def metadata_path(artifact_path: Path) -> Path:
    """Return the sidecar path associated with a translation CSV."""

    return artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRANSLATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_translation_artifact(
    destination: Path,
    *,
    translate: Callable[[Sequence[str], str], Sequence[str]],
    translation_model: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    generation_config: dict[str, Any],
    languages: Sequence[LanguageSpec] = LANGUAGES,
    batch_size: int = 8,
    overwrite: bool = False,
) -> Path:
    """Translate and freeze all language arms with content-free provenance metadata."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    sidecar = metadata_path(destination)
    if not overwrite and (destination.exists() or sidecar.exists()):
        raise FileExistsError(f"translation artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_rows = behavior_rows()
    goals = [row["Goal"] for row in source_rows]
    declared_languages = tuple(languages)
    if not declared_languages or declared_languages[0].id != "english":
        raise ValueError("translation artifacts must begin with the English source arm")
    if len({language.id for language in declared_languages}) != len(declared_languages):
        raise ValueError("translation artifact languages must be unique")
    if any(LANGUAGE_BY_ID.get(language.id) != language for language in declared_languages):
        raise ValueError("translation artifact contains an unsupported language")
    translated: dict[str, list[str]] = {"english": goals.copy()}
    for language in declared_languages:
        if language.id == "english":
            continue
        output: list[str] = []
        for start in range(0, len(goals), batch_size):
            batch = goals[start : start + batch_size]
            result = [str(value).strip() for value in translate(batch, language.nllb_code)]
            if len(result) != len(batch):
                raise ValueError(
                    f"translator returned {len(result)} rows for a batch of {len(batch)} "
                    f"in {language.id}"
                )
            if any(not value for value in result):
                raise ValueError(f"translator returned an empty {language.id} translation")
            output.extend(result)
        translated[language.id] = output

    artifact_rows: list[dict[str, str]] = []
    for language in declared_languages:
        for source, translated_goal in zip(
            source_rows, translated[language.id], strict=True
        ):
            artifact_rows.append(
                {
                    "prompt_id": behavior_id(source["Index"]),
                    "language_id": language.id,
                    "language_name": language.name,
                    "language_code": language.nllb_code,
                    "translated_goal": translated_goal,
                    "source_goal_sha256": sha256_text(source["Goal"]),
                    "translated_goal_sha256": sha256_text(translated_goal),
                }
            )

    _write_csv(destination, artifact_rows)
    artifact_sha256 = sha256_bytes(destination.read_bytes())
    metadata = {
        "schema_version": TRANSLATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "benchmark": "JailbreakBench JBB-Behaviors harmful split",
            "uri": DATA_SOURCE,
            "sha256": DATA_SHA256,
            "source_language_code": SOURCE_LANGUAGE_CODE,
            "selection_rule": "all 100 records in the pinned harmful split",
        },
        "languages": [asdict(language) for language in declared_languages],
        "translation": {
            "model": translation_model,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "generation_config": generation_config,
            "english_arm": "identity copy of the pinned source text",
        },
        "row_count": len(artifact_rows),
        "artifact_sha256": artifact_sha256,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_translation_artifact(destination)
    return destination


def extend_translation_artifact(
    source: Path,
    destination: Path,
    *,
    language: LanguageSpec,
    translate: Callable[[Sequence[str], str], Sequence[str]],
    translation_model: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    generation_config: dict[str, Any],
    batch_size: int = 8,
    overwrite: bool = False,
) -> Path:
    """Append one translated language without regenerating existing language rows."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if LANGUAGE_BY_ID.get(language.id) != language:
        raise ValueError(f"unsupported language declaration: {language}")
    existing, source_metadata = load_translation_artifact(source)
    if language.id in existing:
        raise ValueError(f"translation artifact already contains {language.id}")
    sidecar = metadata_path(destination)
    if not overwrite and (destination.exists() or sidecar.exists()):
        raise FileExistsError(f"translation artifact already exists: {destination}")

    source_rows = behavior_rows()
    goals = [row["Goal"] for row in source_rows]
    translated: list[str] = []
    for start in range(0, len(goals), batch_size):
        batch = goals[start : start + batch_size]
        result = [str(value).strip() for value in translate(batch, language.nllb_code)]
        if len(result) != len(batch):
            raise ValueError(
                f"translator returned {len(result)} rows for a batch of {len(batch)} "
                f"in {language.id}"
            )
        if any(not value for value in result):
            raise ValueError(f"translator returned an empty {language.id} translation")
        translated.extend(result)

    with source.open(newline="", encoding="utf-8") as stream:
        source_artifact_rows = list(csv.DictReader(stream))
    appended_rows = [
        {
            "prompt_id": behavior_id(row["Index"]),
            "language_id": language.id,
            "language_name": language.name,
            "language_code": language.nllb_code,
            "translated_goal": translated_goal,
            "source_goal_sha256": sha256_text(row["Goal"]),
            "translated_goal_sha256": sha256_text(translated_goal),
        }
        for row, translated_goal in zip(source_rows, translated, strict=True)
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(destination, source_artifact_rows + appended_rows)
    artifact_sha256 = sha256_bytes(destination.read_bytes())
    declared = [LanguageSpec(**item) for item in source_metadata["languages"]]
    translation_metadata = dict(source_metadata["translation"])
    extensions = list(translation_metadata.get("extensions", []))
    extensions.append(
        {
            "language": asdict(language),
            "model": translation_model,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "generation_config": generation_config,
        }
    )
    translation_metadata["extensions"] = extensions
    metadata = {
        **source_metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "languages": [asdict(item) for item in (*declared, language)],
        "translation": translation_metadata,
        "row_count": len(source_artifact_rows) + len(appended_rows),
        "artifact_sha256": artifact_sha256,
        "parent_artifact_sha256": source_metadata["artifact_sha256"],
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    load_translation_artifact(destination)
    return destination


def load_translation_artifact(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Load and fully validate a frozen translation artifact and its sidecar."""

    sidecar = metadata_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"translation artifact does not exist: {path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"translation metadata does not exist: {sidecar}")
    metadata: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != TRANSLATION_SCHEMA_VERSION:
        raise ValueError("unsupported translation artifact schema")
    actual_artifact_hash = sha256_bytes(path.read_bytes())
    if metadata.get("artifact_sha256") != actual_artifact_hash:
        raise ValueError("translation artifact checksum does not match its metadata")
    source = metadata.get("source", {})
    if source.get("sha256") != DATA_SHA256:
        raise ValueError("translation artifact was not produced from the pinned JBB source")
    declared_values = metadata.get("languages")
    if not isinstance(declared_values, list) or not declared_values:
        raise ValueError("translation artifact has no language declaration")
    try:
        declared_languages = [LanguageSpec(**value) for value in declared_values]
    except (TypeError, KeyError) as error:
        raise ValueError("translation artifact has an invalid language declaration") from error
    if len({language.id for language in declared_languages}) != len(declared_languages):
        raise ValueError("translation artifact declares a duplicate language")
    if any(LANGUAGE_BY_ID.get(language.id) != language for language in declared_languages):
        raise ValueError("translation artifact declares an unsupported language")

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRANSLATION_COLUMNS:
            raise ValueError("translation artifact has unexpected columns")
        rows = list(reader)
    expected_count = len(behavior_rows()) * len(declared_languages)
    if len(rows) != expected_count or metadata.get("row_count") != expected_count:
        raise ValueError(f"expected {expected_count} translation rows, found {len(rows)}")

    source_by_id = {
        behavior_id(row["Index"]): row["Goal"] for row in behavior_rows()
    }
    declared_by_id = {language.id: language for language in declared_languages}
    result: dict[str, dict[str, str]] = {language.id: {} for language in declared_languages}
    for row in rows:
        language_id = row["language_id"]
        prompt_id = row["prompt_id"]
        language = declared_by_id.get(language_id)
        if language is None:
            raise ValueError(f"unknown translation language: {language_id!r}")
        if row["language_name"] != language.name or row["language_code"] != language.nllb_code:
            raise ValueError(f"language metadata mismatch for {language_id}")
        source_goal = source_by_id.get(prompt_id)
        if source_goal is None:
            raise ValueError(f"unknown translated prompt identifier: {prompt_id!r}")
        translated_goal = row["translated_goal"]
        if not translated_goal:
            raise ValueError(f"empty translation for {prompt_id} in {language_id}")
        if row["source_goal_sha256"] != sha256_text(source_goal):
            raise ValueError(f"source checksum mismatch for {prompt_id} in {language_id}")
        if row["translated_goal_sha256"] != sha256_text(translated_goal):
            raise ValueError(f"translation checksum mismatch for {prompt_id} in {language_id}")
        if prompt_id in result[language_id]:
            raise ValueError(f"duplicate translation for {prompt_id} in {language_id}")
        result[language_id][prompt_id] = translated_goal

    expected_ids = set(source_by_id)
    for language_id, values in result.items():
        if set(values) != expected_ids:
            raise ValueError(f"translation coverage is incomplete for {language_id}")
    if result["english"] != source_by_id:
        raise ValueError("English arm must be an exact copy of the pinned source goals")
    return result, metadata


def multilingual_samples(
    translation_artifact: Path,
    language_id: str,
) -> list[Sample]:
    """Create samples for one frozen language arm while preserving semantic IDs."""

    if language_id not in LANGUAGE_BY_ID:
        raise ValueError(f"unsupported language: {language_id!r}")
    translations, metadata = load_translation_artifact(translation_artifact)
    if language_id not in translations:
        raise ValueError(f"translation artifact does not contain {language_id!r}")
    language = LANGUAGE_BY_ID[language_id]
    artifact_sha256 = str(metadata["artifact_sha256"])
    return [
        Sample(
            id=behavior_id(row["Index"]),
            input=translations[language_id][behavior_id(row["Index"])],
            metadata={
                "benchmark": "JBB-Behaviors",
                "behavior_id": behavior_id(row["Index"]),
                "behavior_label": row["Behavior"],
                "category": row["Category"],
                "category_id": category_id(row["Category"]),
                "source_dataset": row["Source"],
                "attack_method": "translated_direct_request",
                "language_id": language.id,
                "language_name": language.name,
                "language_code": language.nllb_code,
                "translation_artifact_sha256": artifact_sha256,
            },
        )
        for row in behavior_rows()
    ]


def multilingual_task_name(language_id: str) -> str:
    if language_id not in LANGUAGE_BY_ID:
        raise ValueError(f"unsupported language: {language_id!r}")
    return f"jailbreakbench_100_{language_id}"


@task
def jailbreakbench_100_multilingual(
    grader_model: str,
    translation_artifact: str,
    language_id: str,
    system_prompt: str | None = None,
) -> Task:
    """Evaluate one language condition from a validated frozen artifact."""

    path = Path(translation_artifact).resolve()
    _, metadata = load_translation_artifact(path)
    language = LANGUAGE_BY_ID.get(language_id)
    if language is None:
        raise ValueError(f"unsupported language: {language_id!r}")
    solvers = [generate()] if system_prompt is None else [system_message(system_prompt), generate()]
    return Task(
        name=multilingual_task_name(language_id),
        dataset=multilingual_samples(path, language_id),
        solver=solvers,
        scorer=jbb_harmful_compliance(grader_model=grader_model),
        version="1.0",
        metadata={
            "benchmark": "JailbreakBench JBB-Behaviors multilingual",
            "dataset_source": DATA_SOURCE,
            "dataset_sha256": DATA_SHA256,
            "selection_rule": "all 100 records in the pinned harmful split",
            "attack_method": "translated_direct_request",
            "language_id": language.id,
            "language_name": language.name,
            "language_code": language.nllb_code,
            "translation_artifact_sha256": metadata["artifact_sha256"],
            "translation_model": metadata["translation"]["model"],
            "translation_revision": metadata["translation"]["resolved_revision"],
            "grader_model": grader_model,
            "scorer": "portable Inspect model grader; not leaderboard-compatible",
        },
    )
