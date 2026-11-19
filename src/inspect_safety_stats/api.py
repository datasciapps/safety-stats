from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from inspect_safety_stats.analysis.anytime import OrderReader, anytime_bound_diagnostics
from inspect_safety_stats.analysis.core import (
    _add_judge_claims,
    _add_judge_configurations,
    _analysis_core,
)
from inspect_safety_stats.analysis.design_effect import cluster_design_effects
from inspect_safety_stats.analysis.inputs import _coerce_plan, _JudgeInputs, _sorted_integrity
from inspect_safety_stats.analysis.judge_flip import judge_flip_diagnostic
from inspect_safety_stats.analysis.judge_panel import _judge_panel_summary
from inspect_safety_stats.analysis.policy import (
    _apply_decision_policy,
    _decision_policy_record,
    _policy_roll_up,
    _policy_set_claims,
    _PolicyApplication,
)
from inspect_safety_stats.analysis.report import (
    _design_frame,
    _policy_events,
    _report_frame,
    _resolved_design,
    _unread_input_report,
)
from inspect_safety_stats.analysis.tail_precision import (
    single_rollout_tail_claims,
    tail_measure_precision,
)
from inspect_safety_stats.config.canonicalize import canonical_value, sha256_value
from inspect_safety_stats.config.decision_policy import DecisionPolicy
from inspect_safety_stats.config.loader import canonical_plan_hash, report_path, report_plan
from inspect_safety_stats.config.models import AnalysisPlan, OutcomeMappingSensitivitySpec
from inspect_safety_stats.config.registration import RegistrationReceipt, load_receipt
from inspect_safety_stats.decision.engine import roll_up
from inspect_safety_stats.ingest.normalize import normalize, read_inputs
from inspect_safety_stats.ingest.records import EvaluationSource, NormalizedData, SampleOrder
from inspect_safety_stats.schema.reports import (
    Diagnostic,
    IntegritySummary,
    OutcomeMappingSensitivity,
    SafetyReport,
    result_readings,
)
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


def _matched_receipt(
    frozen: AnalysisPlan, receipt: RegistrationReceipt | None, diagnostics: list[Diagnostic]
) -> RegistrationReceipt | None:
    """The receipt when it records the frozen plan's hash; otherwise an error and None.

    The hash is the frozen plan's, not the one a decision policy makes effective: a policy
    leaves the plan as registered.
    """

    if receipt is None:
        return None
    digest = canonical_plan_hash(frozen)
    if receipt.plan_sha256 == digest:
        return receipt
    diagnostics.append(
        Diagnostic(
            code="PLAN_RECEIPT_MISMATCH",
            severity="ERROR",
            message=(
                "the registration receipt records another plan: its hash is "
                f"{receipt.plan_sha256}, this plan's is {digest}"
            ),
        )
    )
    # A receipt for another plan says nothing about this one's timing.
    return None


def _read_input(
    plan: AnalysisPlan,
    logs: LogInput,
    parallel: bool | int,
    source: EvaluationSource | None,
    diagnostics: list[Diagnostic],
    receipt: RegistrationReceipt | None = None,
) -> tuple[NormalizedData, _JudgeInputs | None] | None:
    """Read and normalize the analysis input, or record why it cannot be read.

    ``validate`` and ``analyze`` both read their input here, so that bad input gives both the
    same ``INGEST_FAILED`` error. None when the input cannot be read. Passing both ``logs``
    and ``source`` is a mistake in the call, not in the input, and raises ``ValueError``.
    Beside the normalized data comes what the per-judge runs need, when the plan asks for
    them with ``outputs.judge_claims``, or the outcome mapping sensitivities, when it declares
    ``inputs.score.mapping_sensitivities``. ``receipt`` is a registration receipt already matched
    to the plan, for the timing check.
    """

    if source is not None and logs is not None:
        raise ValueError("pass either logs or source, not both")
    try:
        manifest, groups, source_data = read_inputs(plan, _source(logs, parallel, source))
        # ORDER MATTERS: the per-judge runs and the outcome mapping sensitivities take copies
        # made before the primary run normalizes the manifest and groups, so nothing that run
        # does can reach them.
        judge_inputs = (
            _JudgeInputs(manifest.copy(), copy.deepcopy(groups), source_data, receipt)
            if plan.outputs.judge_claims or plan.inputs.score.mapping_sensitivities
            else None
        )
        data = normalize(plan, manifest, groups, source_data, receipt)
        validate_profile_fields(plan.design.independence_profiles, data.manifest)
    except Exception as exc:
        diagnostics.append(Diagnostic(code="INGEST_FAILED", severity="ERROR", message=str(exc)))
        return None
    diagnostics.extend(data.diagnostics)
    diagnostics.extend(single_rollout_tail_claims(plan, data))
    cluster_design_effects(plan, data, diagnostics)
    return data, judge_inputs


def _mapping_plan(plan: AnalysisPlan, spec: OutcomeMappingSensitivitySpec) -> AnalysisPlan:
    """The plan as it would read with this alternative as its primary outcome mapping."""

    score = plan.inputs.score.model_copy(
        update={"mapping": spec.mapping.model_copy(deep=True), "mapping_sensitivities": []}
    )
    inputs = plan.inputs.model_copy(update={"score": score})
    outputs = plan.outputs.model_copy(update={"judge_claims": False})
    return plan.model_copy(update={"inputs": inputs, "outputs": outputs})


def _outcome_mapping_sensitivity(
    plan: AnalysisPlan,
    inputs: _JudgeInputs,
    spec: OutcomeMappingSensitivitySpec,
    application: _PolicyApplication | None = None,
) -> OutcomeMappingSensitivity:
    """The analysis repeated under one alternative outcome mapping, on the same rollouts.

    ``application`` is the decision policy the primary analysis applied, if any: ``plan``
    already carries its thresholds, and its overall decision is held to the same roll-up.
    """

    alternative = _mapping_plan(plan, spec)
    diagnostics = validate_plan_references(alternative)
    try:
        data = normalize(
            alternative,
            inputs.manifest.copy(),
            copy.deepcopy(inputs.groups),
            inputs.source,
            inputs.receipt,
        )
    except ValueError as exc:
        raise ValueError(f"outcome mapping sensitivity {spec.id!r}: {exc}") from exc
    diagnostics.extend(data.diagnostics)
    diagnostics.extend(single_rollout_tail_claims(alternative, data))
    core = _analysis_core(alternative, data, diagnostics, _policy_set_claims(application))
    decision = roll_up(
        core.mandatory_statuses(),
        integrity_valid=core.integrity_status != "invalid",
        zero_tolerance_failure=bool(_policy_events(alternative, data)),
        confirmatory=alternative.analysis.intent == "confirmatory",
    )
    # ORDER MATTERS: before the integrity is summarised, which counts its diagnostic.
    decision = _policy_roll_up(application, core.coverage_events, decision, diagnostics)
    integrity = _sorted_integrity(diagnostics)
    counts: dict[str, int] = {}
    for item in integrity.diagnostics:
        counts[item.code] = counts.get(item.code, 0) + 1
    return OutcomeMappingSensitivity(
        id=spec.id,
        title=spec.title,
        mapping=spec.mapping.model_dump(mode="json", exclude_none=True),
        integrity_status=integrity.status,
        diagnostic_counts=dict(sorted(counts.items())),
        decision=decision,
        results=result_readings(core.risks, core.comparisons, core.claims),
    )


def validate(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
    source: EvaluationSource | None = None,
    decision_policy: DecisionPolicy | str | Path | None = None,
    registration_receipt: str | Path | None = None,
) -> IntegritySummary:
    """Check a plan and its evaluation data without computing any bound or decision.

    ``plan`` is an ``AnalysisPlan`` or the path of a plan file, read with ``load_plan``. The
    data comes from ``source`` when given, or else from the Inspect logs in ``logs``, read with
    ``parallel`` passed to Inspect's log reader. Passing both ``logs`` and ``source`` is an
    error. Unlike ``analyze``, a failure to read the data does not raise: it is reported as an
    ``INGEST_FAILED`` diagnostic. ``decision_policy`` is checked against the plan as
    ``analyze`` checks it, and so is ``registration_receipt``; a malformed receipt is reported
    as ``INGEST_FAILED``. Returns the integrity status (``valid``, ``degraded`` or
    ``invalid``) and the sorted diagnostics.
    """

    resolved = _coerce_plan(plan)
    diagnostics = validate_plan_references(resolved)
    _apply_decision_policy(resolved, decision_policy, diagnostics)
    try:
        receipt = None if registration_receipt is None else load_receipt(registration_receipt)
    except ValueError as exc:
        diagnostics.append(Diagnostic(code="INGEST_FAILED", severity="ERROR", message=str(exc)))
        return _sorted_integrity(diagnostics)
    receipt = _matched_receipt(resolved, receipt, diagnostics)
    _read_input(resolved, logs, parallel, source, diagnostics, receipt)
    return _sorted_integrity(diagnostics)


def _order_reader(logs: LogInput, source: EvaluationSource | None) -> OrderReader | None:
    """How to read the runs' recorded sample orders: from Inspect logs, which record them."""

    if source is not None:
        return None

    def read(sources: Mapping[str, str]) -> Mapping[str, SampleOrder]:
        from inspect_safety_stats.ingest.inspect_logs import read_inspect_sample_orders

        return read_inspect_sample_orders(sources, logs)

    return read


def analyze(
    *,
    plan: AnalysisPlan | str | Path,
    logs: LogInput = None,
    parallel: bool | int = False,
    source: EvaluationSource | None = None,
    decision_policy: DecisionPolicy | str | Path | None = None,
    registration_receipt: str | Path | None = None,
) -> SafetyReport:
    """Run the analysis a plan declares on its evaluation data and return the safety report.

    ``plan`` is an ``AnalysisPlan`` or the path of a plan file, read with ``load_plan``. The
    data comes from ``source`` when given, or else from the Inspect logs in ``logs``, read with
    ``parallel`` passed to Inspect's log reader. Passing both ``logs`` and ``source`` is an
    error. The report holds the prompt-level estimates, group and hierarchy risk bounds,
    comparisons, claims with their pass, fail or inconclusive status, and the overall
    decision. Its results apply to the declared fixed evaluation set, not to a wider prompt
    population. Write it to disk with ``SafetyReport.write``. Raises when the data cannot be read;
    ``validate`` reports that as a diagnostic instead.

    ``decision_policy`` (experimental) is a ``DecisionPolicy`` or the path of a policy file,
    read with ``load_decision_policy``. Its thresholds replace the plan's for the claims it
    sets, and the report records every version of it, while the plan and its hash stay as
    frozen. When it moves any mandatory claim's acceptable_upper and the mandatory claims rest
    on separate coverage events, an overall PASS becomes INCONCLUSIVE (docs/decision-policy.md).

    ``registration_receipt`` (experimental) is the path of a receipt written by
    ``hash-plan --register``. It is checked against the frozen plan's hash and the earliest
    evaluation start; a malformed receipt raises ``ValueError``.
    """

    frozen = _coerce_plan(plan)
    receipt = None if registration_receipt is None else load_receipt(registration_receipt)
    diagnostics = validate_plan_references(frozen)
    application = _apply_decision_policy(frozen, decision_policy, diagnostics)
    receipt = _matched_receipt(frozen, receipt, diagnostics)
    # Every stage decides with the policy's thresholds; the report records the frozen plan.
    resolved = frozen if application is None else application.effective
    read = _read_input(resolved, logs, parallel, source, diagnostics, receipt)
    if read is None:
        return _unread_input_report(
            resolved, diagnostics, frozen, _decision_policy_record(application, None, diagnostics)
        )
    data, judge_inputs = read
    # ORDER MATTERS: the record adds its diagnostic before the analysis reads integrity.
    policy_record = _decision_policy_record(
        application, data.first_started_at, diagnostics, start_unrecorded=data.start_unrecorded
    )
    # ORDER MATTERS too: an invalid flip manifest must be an error before integrity is read.
    judge_flip_diagnostic(resolved, data, diagnostics)
    # ORDER MATTERS as well: a run without a recorded order must be an error before integrity
    # is read.
    anytime_bound_diagnostics(resolved, data, diagnostics, _order_reader(logs, source))
    core = _analysis_core(resolved, data, diagnostics, _policy_set_claims(application))
    estimates = core.estimates
    tail_measure_precision(resolved, estimates, diagnostics)
    risks = core.risks
    hierarchy_risks = core.hierarchy_risks
    comparisons = core.comparisons
    judge_calibration_summary = core.judge_calibration_summary
    comparison_sets = core.comparison_sets
    integrity_certifiable = core.integrity_status != "invalid"
    claims = core.claims
    events = _policy_events(resolved, data)
    decision = roll_up(
        core.mandatory_statuses(),
        integrity_valid=integrity_certifiable,
        zero_tolerance_failure=bool(events),
        confirmatory=resolved.analysis.intent == "confirmatory",
    )
    # ORDER MATTERS: before the integrity is summarised, so that its diagnostic is reported.
    decision = _policy_roll_up(application, core.coverage_events, decision, diagnostics)
    integrity = _sorted_integrity(diagnostics)
    judge_panel = _judge_panel_summary(resolved, data)
    if judge_panel is not None and core.panel_correction is not None:
        judge_panel["derived_correction"] = core.panel_correction
    if judge_panel is not None and judge_inputs is not None and resolved.outputs.judge_claims:
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
            "prompts": canonical_value(_resolved_design(data, resolved.design.hierarchy)),
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
        adaptive_claims=core.adaptive_claims,
        policy_events=events,
        decision=decision,
        decision_policy=policy_record,
    )
    if judge_inputs is not None:
        report.outcome_mapping_sensitivities = [
            _outcome_mapping_sensitivity(resolved, judge_inputs, spec, application)
            for spec in resolved.inputs.score.mapping_sensitivities
        ]
    report._observations = data.observations
    report._resolved_plan = report_plan(frozen)
    return report
