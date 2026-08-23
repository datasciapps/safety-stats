from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from inspect_safety_stats.schema.observations import Observation
from inspect_safety_stats.schema.reports import SafetyReport


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EmpiricalCertificationSource:
    directory: Path
    report_path: Path
    design_path: Path
    report: SafetyReport
    design: pd.DataFrame
    observations: list[Observation]
    report_sha256: str
    design_sha256: str


def load_empirical_source(source: str | Path) -> EmpiricalCertificationSource:
    candidate = Path(source).resolve()
    report_path = candidate / "report.json" if candidate.is_dir() else candidate
    directory = report_path.parent
    design_path = directory / "design.resolved.csv"
    observations_path = directory / "observations.csv"
    if not report_path.is_file():
        raise FileNotFoundError(f"empirical report JSON does not exist: {report_path}")
    if not design_path.is_file():
        raise FileNotFoundError(f"resolved design does not exist: {design_path}")

    checksums_path = directory / "checksums.json"
    report_digest = sha256_file(report_path)
    design_digest = sha256_file(design_path)
    observations_digest = sha256_file(observations_path) if observations_path.is_file() else None
    if checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        for name, digest in (
            ("report.json", report_digest),
            ("design.resolved.csv", design_digest),
            ("observations.csv", observations_digest),
        ):
            expected = checksums.get(name)
            if expected is not None and digest is None:
                raise ValueError(f"empirical report artifact is missing: {name}")
            if expected is not None and digest is not None and expected != digest:
                raise ValueError(f"empirical report checksum mismatch for {name}")

    report = SafetyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    design = pd.read_csv(design_path, dtype=str, keep_default_na=False)
    required = {"group_id", "prompt_id", "weight"}
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"resolved design is missing columns: {', '.join(missing)}")
    design["weight"] = pd.to_numeric(design["weight"], errors="raise")
    observations: list[Observation] = []
    if observations_path.is_file():
        observation_frame = pd.read_csv(observations_path, dtype=str, keep_default_na=False)
        optional_fields = {
            "source_log_sha256",
            "run_id",
            "error_stage",
            "provider_error_code",
        }
        for record in observation_frame.to_dict(orient="records"):
            values = dict(record)
            for field in optional_fields:
                if values.get(field) == "":
                    values[field] = None
            for field in ("failure", "content_delivered"):
                if values.get(field) == "":
                    values[field] = None
                elif field in values:
                    values[field] = str(values[field]).lower() == "true"
            for field in ("epoch", "retries", "collection_order"):
                values[field] = int(values[field])
            observations.append(Observation.model_validate(values))
    return EmpiricalCertificationSource(
        directory=directory,
        report_path=report_path,
        design_path=design_path,
        report=report,
        design=design,
        observations=observations,
        report_sha256=report_digest,
        design_sha256=design_digest,
    )
