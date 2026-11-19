from __future__ import annotations

import copy
import dataclasses
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_safety_stats.analysis.core import (
    _add_judge_claims,
    _add_judge_configurations,
    _analysis_core,
)
from inspect_safety_stats.analysis.inputs import _coerce_plan, _JudgeInputs, _sorted_integrity
from inspect_safety_stats.analysis.judge_panel import _judge_panel_summary
from inspect_safety_stats.analysis.policy import _apply_decision_policy, _decision_policy_record
from inspect_safety_stats.analysis.report import (
    _design_frame,
    _policy_events,
    _report_frame,
    _resolved_design,
    _unread_input_report,
)
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.decision_policy import DecisionPolicy
from inspect_safety_stats.config.loader import report_path, report_plan
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.decision.engine import roll_up
from inspect_safety_stats.ingest.normalize import normalize, read_inputs
from inspect_safety_stats.ingest.records import EvaluationSource, NormalizedData
from inspect_safety_stats.schema.reports import Diagnostic, IntegritySummary, SafetyReport
from inspect_safety_stats.stats.independence import validate_profile_fields
from inspect_safety_stats.validate.plan import validate_plan_references

if TYPE_CHECKING:
    from inspect_safety_stats.ingest.inspect_logs import LogInput


def _source(
    logs: LogInput, parallel: bool | int, source: EvaluationSource | None
) -> EvaluationSource:
    """The source ``analyze`` and ``validate`` read: ``source``, or else the Inspect logs."""

    if source is not None:
        return source
    # Imported here, so that importing this module does not load Inspect AI.
    from inspect_safety_stats.ingest.inspect_logs import InspectLogs

    return InspectLogs(logs, parallel=parallel)


def _named_paths(value: object) -> Iterator[str | os.PathLike[str]]:
    """The paths an input names: a path, a sequence of paths, or a dataclass source's fields."""

    if isinstance(value, str | os.PathLike):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _named_paths(getattr(value, field.name))
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _named_paths(item)


def _report_message(message: str, plan: AnalysisPlan, inputs: tuple[object, ...]) -> str:
    """An input error's message with each absolute input path in it as a report records it.

    An exception names the file it failed on as it opened it, often by its absolute path. A
    report names its inputs relative to the plan's directory (``report_path``), so that the
    same analysis writes the same bytes in any checkout and a shared report does not reveal
    the local directory layout; the message follows suit. A directory's form also replaces the
    start of the paths of the files in it. Paths as given relative to the working directory,
    and URIs, are left as they are.
    """

    forms: dict[str, str] = {}
    for path in _named_paths(inputs):
        text = os.fspath(path)
        if "://" in text:
            continue
        form = report_path(text, plan)
        for absolute in (text, str(Path(text).resolve())):
            if Path(absolute).is_absolute():
                forms[absolute] = form
                # OSError messages show the path's repr, which doubles Windows separators.
                forms[absolute.replace("\\", "\\\\")] = form
    if not forms:
        return message
    pattern = "|".join(re.escape(text) for text in sorted(forms, key=len, reverse=True))
    return re.sub(pattern, lambda match: forms[match.group(0)], message)


def _read_input(
    plan: AnalysisPlan,
    logs: LogInput,
    parallel: bool | int,
    source: EvaluationSource | None,
    diagnostics: list[Diagnostic],
) -> tuple[NormalizedData, _JudgeInputs | None] | None:
    """Read and normalize the analysis input, or record why it cannot be read.

    ``validate`` and ``analyze`` both read their input here, so that bad input gives both the
    same ``INGEST_FAILED`` error, its paths relative to the plan's directory. None when the
    input cannot be read. Passing both ``logs`` and ``source`` is a mistake in the call, not in
    the input, and raises ``ValueError``. Beside the normalized data comes what the per-judge
    runs need, when the plan asks for them with ``outputs.judge_claims``.
    """

    if source is not None and logs is not None:
        raise ValueError("pass either logs or source, not both")
    try:
        manifest, groups, source_data = read_inputs(plan, _source(logs, parallel, source))
        # ORDER MATTERS: the per-judge runs take copies made before the primary run
        # normalizes the manifest and groups, so nothing that run does can reach them.
        judge_inputs = (
            _JudgeInputs(manifest.copy(), copy.deepcopy(groups), source_data)
            if plan.outputs.judge_claims
            else None
        )
        data = normalize(plan, manifest, groups, source_data)
        validate_profile_fields(plan.design.independence_profiles, data.manifest)
    except Exception as exc:
        calibration = plan.inputs.judge_calibration
        inputs = (
            plan.inputs.design_manifest,
            None if calibration is None else calibration.validation_manifest,
            logs,
            source,
        )
        message = _report_message(str(exc), plan, inputs)
        diagnostics.append(Diagnostic(code="INGEST_FAILED", severity="ERROR", message=message))
        return None
    diagnostics.extend(data.diagnostics)
    return data, judge_inputs


def validate(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
    source: EvaluationSource | None = None,
    decision_policy: DecisionPolicy | str | Path | None = None,
) -> IntegritySummary:
    """Check a plan and its evaluation data without computing any bound or decision.

    ``plan`` is an ``AnalysisPlan`` or the path of a plan file, read with ``load_plan``. The
    data comes from ``source`` when given, or else from the Inspect logs in ``logs``, read with
    ``parallel`` passed to Inspect's log reader. Passing both ``logs`` and ``source`` is an
    error. As in ``analyze``, a failure to read the data does not raise: it is reported as an
    ``INGEST_FAILED`` diagnostic. ``decision_policy`` is checked against the plan as
    ``analyze`` checks it. Returns the integrity status (``valid``, ``degraded`` or
    ``invalid``) and the sorted diagnostics.
    """

    resolved = _coerce_plan(plan)
    diagnostics = validate_plan_references(resolved)
    _apply_decision_policy(resolved, decision_policy, diagnostics)
    _read_input(resolved, logs, parallel, source, diagnostics)
    return _sorted_integrity(diagnostics)


def analyze(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
    source: EvaluationSource | None = None,
    decision_policy: DecisionPolicy | str | Path | None = None,
) -> SafetyReport:
    """Run the analysis a plan declares on its evaluation data and return the safety report.

    ``plan`` is an ``AnalysisPlan`` or the path of a plan file, read with ``load_plan``. The
    data comes from ``source`` when given, or else from the Inspect logs in ``logs``, read with
    ``parallel`` passed to Inspect's log reader. Passing both ``logs`` and ``source`` is an
    error. The report holds the prompt-level estimates, group and hierarchy risk bounds,
    comparisons, claims with their pass, fail or inconclusive status, and the overall
    decision. Its results apply to the declared fixed evaluation set, not to a wider prompt
    population. Write it to disk with ``SafetyReport.write``. Data that cannot be read does not
    raise: the report then carries an ``INGEST_FAILED`` diagnostic, its integrity is invalid,
    and every claim is INCONCLUSIVE.

    ``decision_policy`` (experimental) is a ``DecisionPolicy`` or the path of a policy file,
    read with ``load_decision_policy``. Its thresholds replace the plan's for the claims it
    sets, and the report records every version of it, while the plan and its hash stay as
    frozen (docs/decision-policy.md).
    """

    frozen = _coerce_plan(plan)
    diagnostics = validate_plan_references(frozen)
    application = _apply_decision_policy(frozen, decision_policy, diagnostics)
    # Every stage decides with the policy's thresholds; the report records the frozen plan.
    resolved = frozen if application is None else application.effective
    read = _read_input(resolved, logs, parallel, source, diagnostics)
    if read is None:
        return _unread_input_report(
            resolved, diagnostics, frozen, _decision_policy_record(application, None, diagnostics)
        )
    data, judge_inputs = read
    # ORDER MATTERS: the record adds its diagnostic before the analysis reads integrity.
    policy_record = _decision_policy_record(application, data.first_started_at, diagnostics)
    core = _analysis_core(resolved, data, diagnostics)
    estimates = core.estimates
    risks = core.risks
    hierarchy_risks = core.hierarchy_risks
    comparisons = core.comparisons
    judge_calibration_summary = core.judge_calibration_summary
    comparison_sets = core.comparison_sets
    integrity_certifiable = core.integrity_status != "invalid"
    claims = core.claims
    events = _policy_events(resolved, data)
    decision = roll_up(
        [claim.status for claim in claims if claim.mandatory],
        integrity_valid=integrity_certifiable,
        zero_tolerance_failure=bool(events),
        confirmatory=resolved.analysis.intent == "confirmatory",
    )
    integrity = _sorted_integrity(diagnostics)
    judge_panel = _judge_panel_summary(resolved, data)
    if judge_panel is not None and core.panel_correction is not None:
        judge_panel["derived_correction"] = core.panel_correction
    if judge_panel is not None and judge_inputs is not None:
        _add_judge_claims(judge_panel, resolved, judge_inputs)
        _add_judge_configurations(judge_panel, resolved, data, judge_inputs.source)
    report = SafetyReport(
        **_report_frame(frozen),
        provenance={
            "source_format": data.source_format,
            "sources": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(
                    (report_path(path, resolved), digest)
                    for path, digest in data.source_hashes.items()
                )
            ],
            **(
                {"judge_calibration": judge_calibration_summary}
                if judge_calibration_summary
                else {}
            ),
            **({"judge_panel": judge_panel} if judge_panel else {}),
        },
        integrity=integrity,
        design={
            **_design_frame(resolved),
            "manifest_sha256": sha256_value(data.manifest.to_dict(orient="records")),
            "prompt_count": len(data.manifest),
            "groups": [
                {"id": group.id, "title": group.title, "prompt_count": len(group.prompt_ids)}
                for group in data.groups
            ],
        },
        cohorts=data.cohorts,
        prompt_estimates=sorted(estimates, key=lambda item: (item.cohort_id, item.prompt_id)),
        group_risks=sorted(
            risks, key=lambda item: (item.cohort_id, item.group_id, item.measure_id)
        ),
        hierarchy_risks=sorted(
            hierarchy_risks,
            key=lambda item: (
                item.cohort_id,
                item.group_id,
                resolved.design.hierarchy.index(item.level),
                tuple(item.path.values()),
            ),
        ),
        comparisons=sorted(comparisons, key=lambda item: item.comparison_id),
        comparison_sets=comparison_sets,
        claims=sorted(claims, key=lambda item: item.claim_id),
        policy_events=events,
        decision=decision,
        decision_policy=policy_record,
    )
    report._observations = data.observations
    report._resolved_design = _resolved_design(data, resolved.design.hierarchy)
    report._resolved_plan = report_plan(frozen)
    return report
