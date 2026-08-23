from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
from inspect_safety_stats.config.models import AnalysisPlan, StructuralModelSpec
from inspect_safety_stats.schema.reports import SafetyReport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_failure(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid normalized failure value {value!r}")


@dataclass(frozen=True)
class ModelFrame:
    frame: pd.DataFrame
    cohort_id: str
    group_id: str
    declared_hierarchy: tuple[str, ...]
    hierarchy: tuple[str, ...]
    prompt_frailty_levels: tuple[str, ...]
    category_levels: dict[str, tuple[str, ...]]
    category_codes: dict[str, np.ndarray]
    empirical_report_path: Path
    empirical_report_sha256: str
    empirical_weighted_mean: float | None

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(self.frame["prompt_id"].astype(str))

    @property
    def failures(self) -> np.ndarray:
        return self.frame["k_unsafe"].to_numpy(dtype=int)

    @property
    def trials(self) -> np.ndarray:
        return self.frame["n"].to_numpy(dtype=int)

    @property
    def weights(self) -> np.ndarray:
        return self.frame["weight"].to_numpy(dtype=float)

    def records(self) -> list[dict[str, Any]]:
        return [
            {str(key): value for key, value in row.items()}
            for row in self.frame.to_dict(orient="records")
        ]


def _coerce_plan(plan: AnalysisPlan | str | Path) -> AnalysisPlan:
    return plan if isinstance(plan, AnalysisPlan) else load_plan(plan)


def _report_paths(source: str | Path) -> tuple[Path, Path, Path]:
    candidate = Path(source).resolve()
    report_path = candidate / "report.json" if candidate.is_dir() else candidate
    directory = report_path.parent
    if not report_path.is_file():
        raise FileNotFoundError(f"empirical report JSON does not exist: {report_path}")
    design_path = directory / "design.resolved.csv"
    observations_path = directory / "observations.csv"
    if not design_path.is_file():
        raise FileNotFoundError(f"resolved design does not exist: {design_path}")
    if not observations_path.is_file():
        raise FileNotFoundError(
            "structural modelling requires observations.csv; rerun empirical analysis with "
            "report.include_observations: true"
        )
    return report_path, design_path, observations_path


def _structural_spec(plan: AnalysisPlan) -> StructuralModelSpec:
    if plan.structural_model is None or not plan.structural_model.enabled:
        raise ValueError("the analysis plan does not enable structural_model")
    return plan.structural_model


def _validate_references(
    plan: AnalysisPlan, spec: StructuralModelSpec, report: SafetyReport
) -> None:
    assert spec.cohort is not None and spec.group is not None
    if spec.cohort not in plan.cohorts:
        raise ValueError(f"structural model references unknown cohort {spec.cohort!r}")
    if spec.group not in {group.id for group in plan.groups}:
        raise ValueError(f"structural model references unknown group {spec.group!r}")
    report_cohorts = {str(item.get("cohort_id", item.get("id"))) for item in report.cohorts}
    if spec.cohort not in report_cohorts:
        raise ValueError(f"cohort {spec.cohort!r} is absent from the empirical report")
    if spec.group not in {str(item.get("id")) for item in report.design.get("groups", [])}:
        raise ValueError(f"group {spec.group!r} is absent from the empirical report")


def _validate_hierarchy(
    frame: pd.DataFrame, hierarchy: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing = [column for column in hierarchy if column not in frame]
    if missing:
        raise ValueError(f"structural hierarchy columns are absent: {', '.join(missing)}")
    if frame[hierarchy].isna().any().any():
        raise ValueError("structural hierarchy contains missing values")
    for parent, child in zip(hierarchy, hierarchy[1:], strict=False):
        parent_counts = frame.groupby(child, dropna=False)[parent].nunique()
        if bool((parent_counts > 1).any()):
            raise ValueError(
                f"structural hierarchy is not nested: a {child!r} level has multiple "
                f"{parent!r} parents"
            )
    if not hierarchy or hierarchy[-1] != "prompt_id":
        raise ValueError("structural hierarchy must end with prompt_id")

    prompt_count = len(frame)
    effect_hierarchy: list[str] = []
    prompt_frailty_levels: list[str] = []
    terminal_alias_started = False
    for column in hierarchy[:-1]:
        uniquely_identifies_prompt = frame[column].astype(str).nunique() == prompt_count
        if uniquely_identifies_prompt:
            terminal_alias_started = True
            prompt_frailty_levels.append(column)
        elif terminal_alias_started:
            raise ValueError(
                "prompt-identifying hierarchy levels must form a terminal block immediately "
                "above prompt_id"
            )
        else:
            effect_hierarchy.append(column)
    prompt_frailty_levels.append("prompt_id")
    return tuple(effect_hierarchy), tuple(prompt_frailty_levels)


def build_model_frame(
    *,
    plan: AnalysisPlan | str | Path,
    empirical_report: str | Path,
) -> ModelFrame:
    resolved_plan = _coerce_plan(plan)
    spec = _structural_spec(resolved_plan)
    assert spec.cohort is not None and spec.group is not None
    assert spec.hierarchy is not None
    report_path, design_path, observations_path = _report_paths(empirical_report)
    report = SafetyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    _validate_references(resolved_plan, spec, report)
    empirical_plan_hash = report.analysis.get("plan_sha256")
    if empirical_plan_hash is not None and empirical_plan_hash != canonical_plan_hash(
        resolved_plan
    ):
        raise ValueError(
            "the structural plan does not match the plan embedded in the empirical report; "
            "rerun empirical analysis after adding structural_model to the plan"
        )

    design = pd.read_csv(design_path)
    if "group_id" not in design or "weight" not in design:
        raise ValueError("resolved design must contain group_id and weight")
    design = design.loc[design["group_id"].astype(str) == spec.group].copy()
    if design.empty:
        raise ValueError(f"resolved design contains no rows for group {spec.group!r}")
    if design["prompt_id"].astype(str).duplicated().any():
        raise ValueError("resolved structural group contains duplicate prompt IDs")
    design["prompt_id"] = design["prompt_id"].astype(str)
    design["weight"] = pd.to_numeric(design["weight"], errors="raise")
    if bool((design["weight"] < 0).any()) or not np.isclose(design["weight"].sum(), 1.0):
        raise ValueError("resolved structural weights must be nonnegative and sum to one")
    effect_hierarchy, prompt_frailty_levels = _validate_hierarchy(design, spec.hierarchy)

    observation_columns = ["prompt_id", "cohort_id", "failure", "configuration_fingerprint"]
    observations = pd.read_csv(observations_path, usecols=observation_columns)
    observations = observations.loc[
        (observations["cohort_id"].astype(str) == spec.cohort)
        & observations["prompt_id"].astype(str).isin(set(design["prompt_id"]))
    ].copy()
    if observations.empty:
        raise ValueError("no observations match the structural cohort and group")
    observations["prompt_id"] = observations["prompt_id"].astype(str)
    observations["_failure"] = observations["failure"].map(_as_failure)
    unclassified = int(observations["_failure"].isna().sum())
    if unclassified:
        raise ValueError(
            f"structural model requires complete classified outcomes; found {unclassified} "
            "unclassified attempted rollouts"
        )
    fingerprints = observations["configuration_fingerprint"].dropna().astype(str).unique()
    if len(fingerprints) != 1:
        raise ValueError(
            "structural model version 1 requires exactly one rollout configuration fingerprint"
        )

    counts = (
        observations.groupby("prompt_id", sort=False)
        .agg(n=("_failure", "size"), k_unsafe=("_failure", "sum"))
        .reset_index()
    )
    counts["k_unsafe"] = counts["k_unsafe"].astype(int)
    missing_prompts = sorted(set(design["prompt_id"]) - set(counts["prompt_id"]))
    if missing_prompts:
        raise ValueError(
            f"structural model group is missing observations for {len(missing_prompts)} prompts"
        )

    columns = [
        "prompt_id",
        "weight",
        *(column for column in spec.hierarchy if column != "prompt_id"),
    ]
    frame = design[columns].merge(counts, on="prompt_id", how="left", validate="one_to_one")
    frame = frame.sort_values("prompt_id", kind="stable").reset_index(drop=True)
    category_levels: dict[str, tuple[str, ...]] = {}
    category_codes: dict[str, np.ndarray] = {}
    for column in effect_hierarchy:
        values = frame[column].astype(str)
        levels = tuple(sorted(values.unique()))
        lookup = {level: index for index, level in enumerate(levels)}
        category_levels[column] = levels
        category_codes[column] = values.map(lookup).to_numpy(dtype=int)
        frame[column] = values

    empirical_mean = next(
        (
            item.estimate
            for item in report.group_risks
            if item.cohort_id == spec.cohort
            and item.group_id == spec.group
            and item.method == "weighted_mean"
        ),
        None,
    )
    if empirical_mean is None:
        empirical_mean = float(np.dot(frame["weight"], frame["k_unsafe"] / frame["n"]))

    return ModelFrame(
        frame=frame,
        cohort_id=spec.cohort,
        group_id=spec.group,
        declared_hierarchy=tuple(spec.hierarchy),
        hierarchy=effect_hierarchy,
        prompt_frailty_levels=prompt_frailty_levels,
        category_levels=category_levels,
        category_codes=category_codes,
        empirical_report_path=report_path,
        empirical_report_sha256=_sha256(report_path),
        empirical_weighted_mean=float(empirical_mean),
    )


def model_frame_metadata(model_frame: ModelFrame) -> dict[str, Any]:
    return {
        "cohort_id": model_frame.cohort_id,
        "group_id": model_frame.group_id,
        "prompt_count": len(model_frame.frame),
        "rollout_count": int(model_frame.trials.sum()),
        "unsafe_count": int(model_frame.failures.sum()),
        "hierarchy": list(model_frame.declared_hierarchy),
        "effect_hierarchy": list(model_frame.hierarchy),
        "prompt_frailty_levels": list(model_frame.prompt_frailty_levels),
        "hierarchy_level_counts": {
            column: int(model_frame.frame[column].astype(str).nunique())
            for column in model_frame.declared_hierarchy
        },
        "weight_min": float(model_frame.weights.min()),
        "weight_max": float(model_frame.weights.max()),
        "empirical_weighted_mean": model_frame.empirical_weighted_mean,
    }
