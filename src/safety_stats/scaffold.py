"""Write a starter design manifest and analysis plan from existing Inspect logs.

``inspect-safety-stats init`` is the shortest route from a finished Inspect run to a
report. It reads only the structure of the logs -- models, tasks, sample identifiers,
epochs, scorer names, the set of score values and sample metadata keys -- and writes a
deliberately small plan: one cohort per model, the root weighted-mean failure risk with
a prompt-level sensitivity, an optional one-level breakdown, and a comparison between
models when there is more than one.

Two things are never guessed. The score mapping is always explicit: the caller names
the failure values (or a numeric threshold), because ``C``/``I`` or ``1``/``0`` mean
different things for different scorers. And the generated plan is ``exploratory``,
because it is written after the logs exist; a confirmatory plan has to be frozen before
the evaluation runs.

Every choice the plan rests on comes with a note, drawn from the registry of reportable
assumptions (``inspect_safety_stats.registry.ASSUMPTIONS``): what init chose, what that
assumes, what follows if the assumption is wrong, and what to change. Independence comes
first. The notes go to the caller and into the plan file's header.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from inspect_ai.analysis import evals_df, samples_df

from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.outcomes import is_missing
from inspect_safety_stats.registry import ASSUMPTIONS, AssumptionEntry

_IDENTIFIER = re.compile(r"[^A-Za-z0-9_.-]+")
_UNIFORM_FIELDS = ("task_version", "model_generate_config", "solver")
_SCHEMA_VERSION = "2.0"
_ZERO_TOLERANCE_FIELD = "zero_tolerance"


class _PlainDumper(yaml.SafeDumper):
    """Write repeated values out in full rather than as YAML anchors and aliases."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass(frozen=True)
class AssumptionNote:
    """One choice init made, and the registry's assumption that the choice rests on."""

    assumption: str
    choice: str

    @property
    def entry(self) -> AssumptionEntry:
        return ASSUMPTIONS[self.assumption]


@dataclass
class ScaffoldResult:
    plan_path: Path
    manifest_path: Path
    cohorts: dict[str, str]
    reference: str | None
    score: str
    failure_values: list[Any] | None
    safe_values: list[Any] | None
    prompts: int
    rollouts_per_prompt: int
    group_by: str | None
    notes: list[str] = field(default_factory=list)
    assumptions: list[AssumptionNote] = field(default_factory=list)
    # The logs to analyze the plan with: those given, or, when init ignored some, only the
    # evaluation logs it used, since analyze would report the ignored ones incomplete.
    logs: list[str] = field(default_factory=list)


def _native(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _cohort_id(model: str, taken: set[str]) -> str:
    base = _IDENTIFIER.sub("-", model.split("/")[-1]).strip("-.") or "model"
    if not base[0].isalnum():
        base = f"m{base}"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _matches(argument: str, value: Any) -> bool:
    if str(value) == argument:
        return True
    if isinstance(value, bool):
        return argument.strip().lower() == str(value).lower()
    if isinstance(value, (int, float)):
        try:
            return float(argument) == float(value)
        except ValueError:
            return False
    return False


def _resolve_mapping(
    observed: list[Any],
    *,
    score: str,
    failure_values: Sequence[str] | None,
    safe_values: Sequence[str] | None,
    failure_threshold: float | None,
) -> tuple[dict[str, Any], list[Any] | None, list[Any] | None, list[str]]:
    notes: list[str] = []
    if failure_threshold is not None:
        if failure_values or safe_values:
            raise ValueError("use either --failure-values or --failure-threshold, not both")
        nonnumeric = [
            value
            for value in observed
            if isinstance(value, bool) or not isinstance(value, (int, float))
        ]
        if nonnumeric:
            raise ValueError(
                f"score {score!r} has nonnumeric values {nonnumeric[:5]!r}; a threshold "
                "mapping needs numeric scores"
            )
        threshold_mapping: dict[str, Any] = {
            "failure_if": {"operator": "greater_than_or_equal", "value": failure_threshold}
        }
        return threshold_mapping, None, None, notes
    if not failure_values:
        shown = ", ".join(repr(value) for value in observed[:12])
        raise ValueError(
            f"say which values of score {score!r} are safety failures, with "
            f"--failure-values (observed values: {shown}). The package never infers "
            "whether a value such as C, I, 1 or 0 means a failure."
        )
    failures: list[Any] = []
    for argument in failure_values:
        matched = [value for value in observed if _matches(argument, value)]
        if not matched:
            raise ValueError(
                f"failure value {argument!r} never occurs in score {score!r}; observed "
                + ", ".join(repr(value) for value in observed[:12])
            )
        failures.extend(value for value in matched if value not in failures)
    if safe_values:
        safes: list[Any] = []
        for argument in safe_values:
            matched = [value for value in observed if _matches(argument, value)]
            if not matched:
                raise ValueError(
                    f"safe value {argument!r} never occurs in score {score!r}"
                )
            safes.extend(value for value in matched if value not in safes)
        unmapped = [value for value in observed if value not in failures + safes]
        if unmapped:
            raise ValueError(
                f"score {score!r} has values that are neither failure nor safe: "
                f"{unmapped[:12]!r}; list them with --safe-values or --failure-values"
            )
    else:
        safes = [value for value in observed if value not in failures]
        # A value nobody named is never taken as safe, except the other value of a two-valued
        # score: there the failure value leaves no doubt about what the rest means.
        if len(observed) > 2 and safes:
            raise ValueError(
                f"score {score!r} has {len(observed)} values, and --failure-values names "
                f"only {failures!r}; init does not take the others as safe unless you say "
                "so. Name the safe ones with --safe-values and add any other failures to "
                "--failure-values: " + ", ".join(repr(value) for value in safes)
            )
        if safes:
            notes.append(
                f"score {score!r} has two values, so {safes[0]!r}, the one --failure-values "
                "does not name, was taken as safe; pass --safe-values to name it yourself"
            )
    if set(map(repr, failures)) & set(map(repr, safes)):
        raise ValueError("a score value cannot be both a failure and safe")
    if not safes:
        notes.append(
            "every observed score value is mapped to failure; check the mapping before "
            "relying on the analysis"
        )
    mapping: dict[str, Any] = {"failure_values": failures, "safe_values": safes}
    return mapping, failures, safes, notes


def _left_at_default(assumption: str) -> AssumptionNote:
    """The note for a plan field init leaves out, so that the registry's default applies."""

    entry = ASSUMPTIONS[assumption]
    return AssumptionNote(
        assumption,
        f"{entry.plan_field} is left out, so it takes its default, {entry.default}.",
    )


def _assumption_notes(
    *,
    score: str,
    mapping: dict[str, Any],
    prompts: int,
    rollouts: int,
    comparisons: bool,
) -> list[AssumptionNote]:
    """A note for each choice of the plan, drawn from the registry: independence first."""

    if "failure_if" in mapping:
        rule = f"failure when the score is at least {mapping['failure_if']['value']!r}"
    else:
        rule = (
            f"failure values {mapping['failure_values']!r}, "
            f"safe values {mapping['safe_values']!r}"
        )
    notes = [
        AssumptionNote(
            "rollout_independence",
            "The primary independence profile is rollout, with prompt as a sensitivity: every "
            "rollout is its own block, and the prompt profile, which blocks the rollouts of a "
            "prompt together, is reported beside it.",
        )
    ]
    if comparisons:
        notes.append(
            AssumptionNote(
                "arm_independence",
                "Under the rollout profile the models are compared as independent arms "
                "(comparison_coupling: independent); the prompt sensitivity couples them "
                "(comparison_coupling: joint).",
            )
        )
    notes += [
        AssumptionNote(
            "plan_fixed_in_advance",
            "analysis.intent is exploratory, because init writes the plan after the logs "
            "exist.",
        ),
        AssumptionNote(
            "fixed_design",
            f"design.type is fixed: the bounds cover the {prompts} prompts of design.csv.",
        ),
        AssumptionNote(
            "run_stability",
            "Each model is one cohort, which must be uniform in "
            + ", ".join(_UNIFORM_FIELDS)
            + " (require_uniform).",
        ),
        AssumptionNote(
            "complete_design",
            f"Every prompt expects {rollouts} rollouts per model (expected_rollouts), and "
            "design.require_expected_rollouts is left out, so it takes its default, "
            f"{ASSUMPTIONS['complete_design'].default}.",
        ),
        AssumptionNote(
            "scorer_validity",
            f"The score {score!r} is taken as the safety score, with {rule}.",
        ),
        AssumptionNote(
            "judge_accuracy",
            "inputs.judge_calibration is left out, so the score's verdicts are taken as "
            "correct, as they are under inputs.judge_calibration.mode: "
            f"{ASSUMPTIONS['judge_accuracy'].default}, its default.",
        ),
        _left_at_default("missing_scores"),
        _left_at_default("sample_errors"),
        _left_at_default("token_limit"),
        _left_at_default("content_filter_stops"),
    ]
    return notes


def _alpha_text(alpha: float, families: Sequence[str]) -> str:
    """How the plan spends alpha: once in each multiplicity family, not once in all."""

    joint = max(0.0, 1 - len(families) * alpha)
    return (
        f"each multiplicity family ({', '.join(families)}) spends alpha {alpha:g} on its own: "
        f"the intervals of one family hold together with probability at least {1 - alpha:g}, "
        "and all "
        f"of the plan's intervals hold together with probability at least {joint:g}. To make "
        "them all hold together with a stated probability, split alpha between the families "
        "(multiplicity.families)"
    )


def _header(
    stamp: datetime, assumptions: Sequence[AssumptionNote], alpha_text: str
) -> str:
    """The comment that opens the plan: how it was made, and the assumptions it rests on."""

    lines = [
        f"Generated by `inspect-safety-stats init` on {stamp:%Y-%m-%d %H:%M} UTC.",
        "Review the score mapping, cohorts and weights before running `analyze`.",
        "Exploratory by construction: a confirmatory plan must be frozen before the",
        "evaluation runs (set analysis.intent and analysis.plan_frozen_at).",
        "",
        *textwrap.wrap(
            "Zero tolerance is off (policy.zero_tolerance.enabled: false): design.csv has no "
            f"{_ZERO_TOLERANCE_FIELD} column, so the gate would hold no prompt. To hold prompts "
            f"to zero tolerance, add a {_ZERO_TOLERANCE_FIELD} column, true for those prompts, "
            "and set enabled: true.",
            width=88,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        "",
        *textwrap.wrap(
            alpha_text[0].upper() + alpha_text[1:] + ".",
            width=88,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        "",
        "The choices init made, and the assumptions they rest on (from",
        "inspect_safety_stats.registry):",
    ]
    for note in assumptions:
        entry = note.entry
        lines.append("")
        for first, text in (
            ("- ", f"{note.assumption}: {note.choice}"),
            ("  ", f"Assumes: {entry.statement}"),
            ("  ", f"If wrong: {entry.if_wrong}"),
            ("  ", f"Remedy: {entry.remedy}"),
        ):
            lines += textwrap.wrap(
                text,
                width=94,
                initial_indent=first,
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
    return "".join(f"# {line}".rstrip() + "\n" for line in lines)


def _refuse_confirmatory(intent: str) -> None:
    if intent == "exploratory":
        return
    if intent != "confirmatory":
        raise ValueError(f"intent must be exploratory, not {intent!r}")
    raise ValueError(
        "init writes its plan after the logs exist, so the plan cannot be confirmatory: a "
        "confirmatory analysis rests on the assumption plan_fixed_in_advance. Use --intent "
        "exploratory, the default. For a confirmatory analysis, write the plan before the "
        "runs. " + ASSUMPTIONS["plan_fixed_in_advance"].remedy
    )


def scaffold_plan(
    logs: str | Path | Sequence[str | Path],
    output: str | Path,
    *,
    score: str | None = None,
    failure_values: Sequence[str] | None = None,
    safe_values: Sequence[str] | None = None,
    failure_threshold: float | None = None,
    group_by: str | None = None,
    baseline: str | None = None,
    alpha: float = 0.05,
    weights: str = "equal_prompt",
    overwrite: bool = False,
    intent: str = "exploratory",
) -> ScaffoldResult:
    """Write ``design.csv`` and ``safety-plan.yaml`` for the given logs into ``output``.

    ``intent`` other than ``exploratory`` is refused: a plan written after the logs exist
    cannot be confirmatory.
    """

    _refuse_confirmatory(intent)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if weights not in {"equal_prompt", "nested_equal"}:
        raise ValueError("weights must be equal_prompt or nested_equal")
    log_input: Any = (
        [str(item) for item in logs]
        if isinstance(logs, Sequence) and not isinstance(logs, (str, bytes))
        else str(logs)
    )
    evaluations = evals_df(log_input)
    if evaluations.empty:
        raise ValueError("no Inspect evaluation logs were found")
    notes: list[str] = []

    status = evaluations.get("status", pd.Series(["success"] * len(evaluations)))
    finished = (status.astype(str) == "success").to_numpy()
    successful = evaluations.loc[finished].copy()
    if successful.empty:
        raise ValueError("none of the evaluation logs finished successfully")
    given = log_input if isinstance(log_input, list) else [log_input]
    analyze_logs = given
    if not finished.all():
        ignored = sorted(evaluations.loc[~finished, "log"].astype(str))
        notes.append(
            f"ignored {len(ignored)} evaluation log(s) that did not finish successfully: "
            + ", ".join(ignored)
            + ". analyze would report them incomplete, so give it only the logs init used, "
            "as Next: does"
        )
        analyze_logs = sorted(successful["log"].astype(str).unique())

    # analyze takes a run's task as part of its configuration, and a cohort whose runs differ
    # in configuration is invalid (CONFIG_DRIFT). A cohort here is one model, so it can span
    # only one task.
    tasks = sorted(
        set(successful["task_name"].astype(str)) if "task_name" in successful else {"task"}
    )
    if len(tasks) > 1:
        raise ValueError(
            f"the logs hold evaluations of {len(tasks)} tasks ({', '.join(tasks)}); init writes "
            "a plan for one task, because analyze requires the evaluations of each model to "
            "share one configuration, task included. Pass the logs of one task with --logs, and "
            "run init once for each task."
        )
    task = tasks[0]

    per_model = successful.groupby(successful["model"].astype(str))["eval_id"].nunique()
    repeated = sorted(str(model) for model, count in per_model.items() if count > 1)
    if repeated:
        raise ValueError(
            "more than one successful evaluation per model would double-count rollouts: "
            + ", ".join(repeated)
            + ". Pass the specific .eval files to use with --logs."
        )

    # Read the samples of only the logs init uses: samples of an ignored log that lack a
    # metadata field would turn the field's integers into floats, 1 into 1.0, for every log.
    samples = samples_df(analyze_logs)
    samples = samples.loc[samples["eval_id"].astype(str).isin(successful["eval_id"].astype(str))]
    if samples.empty:
        raise ValueError("the successful evaluations contain no samples")

    score_columns = sorted(column for column in samples.columns if column.startswith("score_"))
    available = [column.removeprefix("score_") for column in score_columns]
    if score is None:
        if len(available) != 1:
            raise ValueError(
                "choose the safety score with --score; available scores: "
                + (", ".join(available) or "none")
            )
        score = available[0]
    if f"score_{score}" not in samples:
        raise ValueError(
            f"score {score!r} is absent from the logs; available: " + ", ".join(available)
        )
    observed: list[Any] = []
    for value in samples[f"score_{score}"].tolist():
        native = _native(value)
        if is_missing(native) or native in observed:
            continue
        observed.append(native)
    if not observed:
        raise ValueError(f"score {score!r} has no recorded values")
    mapping, failures, safes, mapping_notes = _resolve_mapping(
        observed,
        score=score,
        failure_values=failure_values,
        safe_values=safe_values,
        failure_threshold=failure_threshold,
    )
    notes.extend(mapping_notes)

    samples = samples.assign(_prompt_id=samples["id"].astype(str))

    group_column: str | None = None
    if group_by is not None:
        metadata = f"metadata_{group_by}"
        if metadata not in samples:
            keys = sorted(
                column.removeprefix("metadata_")
                for column in samples.columns
                if column.startswith("metadata_")
            )
            raise ValueError(
                f"sample metadata has no field {group_by!r}; available: "
                + (", ".join(keys) or "none")
            )
        group_column = _IDENTIFIER.sub("_", group_by)
        if group_column in {"prompt_id", "question_id", "task_id", "expected_rollouts"}:
            group_column = f"{group_column}_group"
        # A sample without the field is refused, not grouped under a made-up value that could
        # merge with a real one. An empty value counts as missing: design.csv would read back
        # an empty cell as missing.
        absent = samples[metadata].map(lambda value: is_missing(value) or str(value) == "")
        if absent.any():
            named = sorted(samples.loc[absent, "_prompt_id"].unique())
            raise ValueError(
                f"metadata field {group_by!r} is missing or empty for {len(named)} sample(s): "
                + ", ".join(named[:10])
                + (", ..." if len(named) > 10 else "")
                + ". init does not invent a group for them: record the field for every sample, "
                "or leave out --group-by"
            )
        conflicting = samples.groupby("_prompt_id")[metadata].nunique(dropna=False)
        if (conflicting > 1).any():
            raise ValueError(
                f"metadata field {group_by!r} differs between rollouts of one sample, so it "
                "cannot define a fixed grouping"
            )

    epochs = pd.to_numeric(
        successful["epochs"] if "epochs" in successful else pd.Series(dtype=float),
        errors="coerce",
    )
    planned = int(epochs.max()) if epochs.notna().any() else None
    observed_rollouts = int(
        samples.groupby(["eval_id", "_prompt_id"]).size().max()
    )
    rollouts = planned or observed_rollouts
    if epochs.notna().any() and epochs.nunique() > 1:
        notes.append(
            "models were run with different epoch counts; expected_rollouts uses the "
            f"largest ({rollouts}), so shorter runs will be reported as incomplete"
        )

    rows: list[dict[str, Any]] = []
    first = samples.drop_duplicates("_prompt_id").sort_values("_prompt_id", kind="stable")
    for _, sample in first.iterrows():
        row: dict[str, Any] = {
            "prompt_id": sample["_prompt_id"],
            "question_id": sample["_prompt_id"],
            "task_id": task,
        }
        if group_column is not None:
            row[group_column] = str(sample[f"metadata_{group_by}"])
        row["expected_rollouts"] = rollouts
        rows.append(row)
    manifest = pd.DataFrame(rows)

    taken: set[str] = set()
    cohorts: dict[str, str] = {}
    for model in sorted(successful["model"].astype(str).unique()):
        cohorts[_cohort_id(model, taken)] = model
    reference: str | None = None
    if baseline is not None:
        by_model = {model: cohort for cohort, model in cohorts.items()}
        reference = baseline if baseline in cohorts else by_model.get(baseline)
        if reference is None:
            raise ValueError(
                f"baseline {baseline!r} is neither a model nor a cohort in these logs; models: "
                + ", ".join(cohorts.values())
            )
    elif len(cohorts) == 2:
        reference = next(iter(cohorts))
        notes.append(
            f"two models and no --baseline: {cohorts[reference]!r} was taken as the "
            "reference, so differences read as the other model minus it; pass --baseline "
            "to choose"
        )
    elif len(cohorts) > 2:
        notes.append(
            "several models and no --baseline: every pair is compared; pass --baseline to "
            "compare each model against one reference instead"
        )

    hierarchy = [
        *([group_column] if group_column else []),
        "prompt_id",
    ]
    has_nodes = len(hierarchy) > 1
    independence = {"primary": "rollout", "sensitivities": ["prompt"]}
    families = [
        {"id": "prompt_bounds", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": alpha},
        {"id": "risk", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": alpha},
    ]
    stamp = datetime.now(timezone.utc)
    plan: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "analysis": {
            "id": _IDENTIFIER.sub("-", f"{task}-{stamp:%Y%m%dT%H%M%SZ}"),
            # Cohort ids, not model names: a model name can be a path on the machine that
            # served it, and the title heads every report.
            "title": f"{task}: " + ", ".join(cohorts),
            "description": (
                "Starter plan generated by inspect-safety-stats init. Exploratory: it was "
                "written after the logs existed."
            ),
            "intent": "exploratory",
        },
        "inputs": {
            "design_manifest": "design.csv",
            "score": {"name": score, "mapping": mapping},
        },
        "design": {
            "type": "fixed",
            "hierarchy": hierarchy,
            "weights": (
                {"strategy": "nested_equal", "hierarchy": hierarchy}
                if weights == "nested_equal"
                else {"strategy": "equal_prompt", "hierarchy": hierarchy}
            ),
            "independence_profiles": {
                "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"},
                "prompt": {"block_by": ["prompt_id"], "comparison_coupling": "joint"},
            },
        },
        "cohorts": {
            cohort: {"match": {"model": model}, "require_uniform": list(_UNIFORM_FIELDS)}
            for cohort, model in cohorts.items()
        },
        "groups": [{"id": "all", "title": "All prompts", "filter": {}}],
        "outputs": {
            "prompt_bounds": {
                "enabled": True,
                "simultaneous": True,
                "family": "prompt_bounds",
            },
            "direct_mean_intervals": {
                "enabled": True,
                "method": "weighted_kl_chernoff",
                "family": "risk",
                "independence": independence,
                "hierarchy_nodes": has_nodes,
                "hierarchy_calibration": "frontier_sidak" if has_nodes else "flat_bonferroni",
            },
            "risk_measures": [{"id": "mean", "type": "weighted_mean"}],
        },
        "multiplicity": {"families": families},
        "claims": [],
    }
    if len(cohorts) >= 2:
        families.append(
            {"id": "comparisons", "purpose": "simultaneous_intervals", "method": "bonferroni",
             "alpha": alpha}
        )
        plan["comparison_sets"] = [
            {
                "id": "models",
                "cohorts": (
                    [reference, *[cohort for cohort in cohorts if cohort != reference]]
                    if reference
                    else list(cohorts)
                ),
                "contrasts": (
                    {"type": "versus_reference", "reference": reference}
                    if reference
                    else {"type": "all_pairs"}
                ),
                "estimands": ["risk_difference", "risk_ratio"],
                "group": "all",
                "measure": "mean",
                "method": "weighted_kl_chernoff",
                "family": "comparisons",
                "independence": independence,
                "hierarchy_nodes": has_nodes,
                "hierarchy_calibration": "frontier_sidak" if has_nodes else "flat_bonferroni",
            }
        ]
    # The manifest has no zero-tolerance column, so an enabled gate could flag nothing, and a
    # report would list the default enabled: true as if it held prompts to zero tolerance.
    plan["policy"] = {
        "zero_tolerance": {"enabled": False, "manifest_field": _ZERO_TOLERANCE_FIELD}
    }
    AnalysisPlan.model_validate(plan)
    alpha_text = _alpha_text(alpha, [str(family["id"]) for family in families])
    notes.append(alpha_text)
    assumptions = _assumption_notes(
        score=score,
        mapping=mapping,
        prompts=len(manifest),
        rollouts=rollouts,
        comparisons=len(cohorts) >= 2,
    )

    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "safety-plan.yaml"
    manifest_path = directory / "design.csv"
    existing = [path for path in (plan_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite " + ", ".join(str(path) for path in existing)
            + "; pass --overwrite"
        )
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    plan_path.write_text(
        _header(stamp, assumptions, alpha_text)
        + "\n"
        + yaml.dump(plan, Dumper=_PlainDumper, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return ScaffoldResult(
        plan_path=plan_path,
        manifest_path=manifest_path,
        cohorts=cohorts,
        reference=reference,
        score=score,
        failure_values=failures,
        safe_values=safes,
        prompts=len(manifest),
        rollouts_per_prompt=rollouts,
        group_by=group_by,
        notes=notes,
        assumptions=assumptions,
        logs=analyze_logs,
    )
