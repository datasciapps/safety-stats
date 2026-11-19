"""The analysis core over one plan and one normalized source, and per-judge runs."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from inspect_safety_stats.analysis.adaptive_claims import _adaptive_claims
from inspect_safety_stats.analysis.claims import _claims
from inspect_safety_stats.analysis.comparisons import _comparison_set_results, _comparisons
from inspect_safety_stats.analysis.direct_mean import _direct_mean_context, _hierarchy_nodes
from inspect_safety_stats.analysis.estimates import _prompt_estimates
from inspect_safety_stats.analysis.inputs import (
    _derived_run_diagnostics,
    _integrity_status,
    _JudgeInputs,
)
from inspect_safety_stats.analysis.judge_correction import apply_judge_correction
from inspect_safety_stats.analysis.judge_panel import apply_panel_derived_correction
from inspect_safety_stats.analysis.regions import project_hierarchy_regions
from inspect_safety_stats.analysis.results import RiskRows
from inspect_safety_stats.analysis.risks import _group_risks, _hierarchy_risks
from inspect_safety_stats.analysis.tail_precision import single_rollout_tail_claims
from inspect_safety_stats.config.canonicalize import sha256_value
from inspect_safety_stats.config.models import AnalysisPlan, JudgeSpec
from inspect_safety_stats.decision.engine import mandatory_statuses
from inspect_safety_stats.ingest.normalize import normalize
from inspect_safety_stats.ingest.outcomes import is_missing
from inspect_safety_stats.ingest.records import NormalizedData, SourceData
from inspect_safety_stats.schema.reports import (
    AdaptiveClaimResult,
    ClaimResult,
    ComparisonResult,
    ComparisonSetResult,
    Decision,
    Diagnostic,
    GroupRiskResult,
    HierarchyRiskResult,
    PromptEstimate,
)
from inspect_safety_stats.schema.scopes import scope_path
from inspect_safety_stats.validate.plan import noisy_valid_test_supported


@dataclass
class _AnalysisCore:
    """What one plan decides over one normalized source, before the report is assembled."""

    estimates: list[PromptEstimate]
    risks: list[GroupRiskResult]
    hierarchy_risks: list[HierarchyRiskResult]
    comparisons: list[ComparisonResult]
    judge_calibration_summary: dict[str, Any] | None
    panel_correction: dict[str, Any] | None
    comparison_sets: list[ComparisonSetResult]
    integrity_status: Literal["valid", "degraded", "invalid"]
    claims: list[ClaimResult]
    adaptive_claims: list[AdaptiveClaimResult]

    def mandatory_statuses(self) -> list[Decision]:
        """The statuses the overall decision rolls up: every mandatory claim's, of both kinds."""

        return mandatory_statuses(self.claims, self.adaptive_claims)


def _analysis_core(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> _AnalysisCore:
    estimates = _prompt_estimates(plan, data, diagnostics)
    hierarchy_nodes = _hierarchy_nodes(plan, data)
    direct_context = _direct_mean_context(
        plan,
        data,
        nodes=hierarchy_nodes,
        diagnostics=diagnostics,
    )
    risks = _group_risks(
        plan,
        data,
        estimates,
        diagnostics,
        direct_context=direct_context,
    )
    hierarchy_risks = _hierarchy_risks(
        plan,
        data,
        hierarchy_nodes,
        estimates,
        diagnostics,
        direct_context,
    )
    # Comparisons read the prompt estimates only, and project their own regions.
    comparisons = _comparisons(plan, data, estimates, diagnostics)
    computed = RiskRows(risks=risks, hierarchy_risks=hierarchy_risks, comparisons=comparisons)
    # Each adjusting stage returns new rows and passes them on, so the judge correction
    # corrects the bounds that are displayed: the projected ones.
    projected = project_hierarchy_regions(plan, data, hierarchy_nodes, computed, diagnostics)
    judge_correction = apply_judge_correction(
        plan, data, hierarchy_nodes, projected, diagnostics
    )
    panel_correction = apply_panel_derived_correction(plan, judge_correction.rows, diagnostics)
    final = panel_correction.rows
    risks, hierarchy_risks, comparisons = final.risks, final.hierarchy_risks, final.comparisons
    judge_calibration_summary = judge_correction.summary
    comparison_sets = _comparison_set_results(plan, comparisons, risks)
    integrity_status = _integrity_status(diagnostics)
    claims = _claims(
        plan,
        data,
        risks,
        comparisons,
        integrity_status != "invalid",
        diagnostics,
        judge_calibration_summary,
    )
    return _AnalysisCore(
        estimates=estimates,
        risks=risks,
        hierarchy_risks=hierarchy_risks,
        comparisons=comparisons,
        judge_calibration_summary=judge_calibration_summary,
        panel_correction=panel_correction.summary,
        comparison_sets=comparison_sets,
        integrity_status=integrity_status,
        claims=claims,
        adaptive_claims=_adaptive_claims(plan, data, integrity_status != "invalid"),
    )


def _corrected_interval_claims(plan: AnalysisPlan) -> set[str]:
    """The ``noisy_valid_test`` claims a judge's run bounds as ``corrected_interval`` claims.

    The test needs the plan's calibration labels, which validate the primary outcome, not a
    judge. Where the plan can run the test, a judge's run bounds such a claim on its interval
    instead; where it cannot, there are none, and the claim stays a plan error in every run.
    """

    if not noisy_valid_test_supported(plan):
        return set()
    return {claim.id for claim in plan.claims if claim.method == "noisy_valid_test"}


def _judge_plan(plan: AnalysisPlan, judge: JudgeSpec) -> AnalysisPlan:
    """The plan as it would read with this judge as its only, primary outcome.

    The plan's judge calibration validates its primary outcome, not this judge, so it is
    left out, and so is the panel, which a single judge's run has no use for. A
    ``noisy_valid_test`` claim's test needs that calibration's labels, so where the plan can
    run the test the judge's run bounds the claim as a ``corrected_interval`` claim instead
    (its row decides nothing under the validated calibration the test requires). Where the
    plan cannot, the claim is kept, so the judge's run reports the primary's plan error.
    """

    remapped = _corrected_interval_claims(plan)
    claims = [
        claim.model_copy(update={"method": "corrected_interval"})
        if claim.id in remapped
        else claim
        for claim in plan.claims
    ]
    inputs = plan.inputs.model_copy(
        update={
            "score": judge.score,
            "judges": [],
            "composites": [],
            "judge_calibration": None,
            "panel_groups": None,
        }
    )
    outputs = plan.outputs.model_copy(update={"judge_claims": False})
    return plan.model_copy(update={"inputs": inputs, "outputs": outputs, "claims": claims})


_UNCALIBRATED_JUDGE_REASON = (
    "The plan declares a validated judge calibration, so its thresholds are about true risk. "
    "The calibration's labels validate the primary outcome, not this judge, so this judge's "
    "bounds are on its reported risk only and the claim is not decided on them."
)
_INVALID_REPORT_REASON = (
    "The report's integrity is invalid, so no judge's row is decided either, whatever this "
    "judge's own run found; the report's ERROR diagnostics say why."
)
_NOISY_VALID_TEST_REASON = (
    "The claim's noisy_valid_test needs labels of this judge's verdicts, which the plan does "
    "not have, so this judge's run bounds it as a corrected_interval claim."
)


def _judge_claim_row(
    claim: ClaimResult,
    *,
    validated_calibration: bool,
    noisy_valid_test: bool,
    report_invalid: bool,
) -> dict[str, Any]:
    """One claim of a judge's run, as its row in the judge panel.

    A judge's run has no calibration, so its bounds are on judge-reported risk. Under a
    validated calibration the plan's thresholds are about true risk, where the primary refuses
    to decide on judge-reported bounds; so does the row, which stays INCONCLUSIVE, and so does
    its descriptive status under an exploratory intent, as the primary's would be. When the
    report's integrity is invalid and the judge's run is not, the row decides nothing either,
    with no descriptive status, as a claim of an invalid analysis has none.
    """

    status, descriptive = claim.status, claim.descriptive_status
    reasons = list(claim.reasons)
    if validated_calibration:
        status, reasons = "INCONCLUSIVE", [_UNCALIBRATED_JUDGE_REASON]
        descriptive = None if descriptive is None else "INCONCLUSIVE"
        if noisy_valid_test:
            reasons.append(_NOISY_VALID_TEST_REASON)
    if report_invalid:
        status, descriptive = "INCONCLUSIVE", None
        reasons.append(_INVALID_REPORT_REASON)
    return {
        "claim_id": claim.claim_id,
        "status": status,
        "descriptive_status": descriptive,
        "estimate": claim.estimate,
        "lower": claim.lower,
        "upper": claim.upper,
        "method": claim.method,
        "reasons": reasons,
    }


def _unread_judge_claim_rows(plan: AnalysisPlan, why: str) -> list[dict[str, Any]]:
    """The rows of a judge's run that could not read its input: every claim undecided, unbounded."""

    return [
        {
            "claim_id": claim.id,
            "status": "INCONCLUSIVE",
            "descriptive_status": None,
            "estimate": None,
            "lower": None,
            "upper": None,
            "method": "unavailable",
            "reasons": [f"This judge's run could not read its input: {why}"],
        }
        for claim in sorted(plan.claims, key=lambda item: item.id)
    ]


def _add_judge_claims(
    panel: dict[str, Any],
    plan: AnalysisPlan,
    inputs: _JudgeInputs,
    plan_diagnostics: Sequence[Diagnostic],
    *,
    report_valid: bool,
) -> None:
    """Decide every claim again on each declared judge's verdicts, beside the primary's.

    Each judge's run is the analysis the plan would give with that judge as its primary
    outcome, over the same rollouts, so comparing judge configurations is one analysis. It
    carries the primary run's ``plan_diagnostics`` (``_derived_run_diagnostics``), so a
    registration receipt for another plan or a failed decision policy makes it invalid too.
    An invalid report never shows a judge's run as anything but ``invalid``, nor a row as
    decided, whatever made it invalid: ``report_valid`` is false when the primary analysis
    is invalid, for an error of its own outcome too. Its
    bounds are on that judge's reported risk; under a validated calibration its rows decide
    nothing (``_judge_claim_row``). Beside its integrity status, each judge records how often
    each diagnostic code occurred in its run, so the report says why a run is not valid. A
    judge whose own ``score`` asks for what the source cannot give, such as the stop reasons
    its ``token_limit`` needs, gets an ``invalid`` run with ``INGEST_FAILED``, as the primary
    would, and rows that decide nothing; the rest of the report stands.
    """

    validated_calibration = (
        plan.inputs.judge_calibration is not None
        and plan.inputs.judge_calibration.mode == "validated"
    )
    # Only where the judge's run does bound them so: where the plan cannot run the test, each
    # run reports the plan error instead, and its rows must not say otherwise.
    noisy_claims = _corrected_interval_claims(plan)
    by_id = {judge.id: judge for judge in plan.inputs.judges}
    for entry in panel["judges"]:
        judge_plan = _judge_plan(plan, by_id[entry["id"]])
        diagnostics = _derived_run_diagnostics(judge_plan, plan_diagnostics)
        try:
            data = normalize(
                judge_plan,
                inputs.manifest.copy(),
                copy.deepcopy(inputs.groups),
                inputs.source,
                inputs.receipt,
            )
        except ValueError as exc:
            # The primary run read the same source, so only the judge's own score can ask for
            # more than it gives.
            diagnostics.append(
                Diagnostic(code="INGEST_FAILED", severity="ERROR", message=str(exc))
            )
            entry["integrity"] = "invalid"
            entry["diagnostic_counts"] = _code_counts(diagnostics)
            entry["claims"] = _unread_judge_claim_rows(plan, str(exc))
            continue
        diagnostics.extend(data.diagnostics)
        diagnostics.extend(single_rollout_tail_claims(judge_plan, data))
        core = _analysis_core(judge_plan, data, diagnostics)
        # The counts stay the judge's own run's; the report's errors are in the report.
        entry["integrity"] = core.integrity_status if report_valid else "invalid"
        entry["diagnostic_counts"] = _code_counts(diagnostics)
        entry["claims"] = [
            _judge_claim_row(
                claim,
                validated_calibration=validated_calibration,
                noisy_valid_test=claim.claim_id in noisy_claims,
                report_invalid=not report_valid and core.integrity_status != "invalid",
            )
            for claim in sorted(core.claims, key=lambda item: item.claim_id)
        ]


def _code_counts(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """How often each diagnostic code occurs, by code."""

    counts: dict[str, int] = {}
    for item in diagnostics:
        counts[item.code] = counts.get(item.code, 0) + 1
    return dict(sorted(counts.items()))


class UnreadableField(ValueError):
    """A run field the judge configurations cannot read. The message says why without quoting
    the value, which can hold credentials."""


def _recorded_field(value: Any) -> Any:
    """A run field as recorded: None when it is absent, decoded when it is a JSON string."""

    if is_missing(value):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise UnreadableField("it is not JSON") from None


def scorer_configurations(value: Any, score_name: str) -> list[dict[str, str]] | None:
    """Fingerprints of the configurations a run records for the scorer named like a judge's score.

    Each is ``{"name": <score name>, "sha256": <digest>}``, the digest being the SHA-256 of
    the scorer's whole recorded entry (name, options, metrics and metadata) as canonical JSON.
    The entry itself is never copied: scorer options and metadata can hold credentials,
    endpoints and local paths, such as a request header, a token or a template file, and
    Inspect redacts only some of them. Equal fingerprints mean equal recorded configurations.
    The digest is unsalted, so reports stay comparable and byte-identical; it therefore hides
    a value only as well as the value is hard to guess. Someone who knows the rest of the
    entry can hash candidates for one unknown value, which recovers a value with few
    possibilities, such as an internal IP address or a user name in a path, in seconds; a long
    random credential, such as an API key or a bearer token, cannot be recovered so.
    None when the run records no scorers at all, an empty list when none has that name.
    Raises ``UnreadableField`` when the value is not a JSON list of scorers, or the scorer's
    entry is not JSON data.
    """

    scorers = _recorded_field(value)
    if scorers is None:
        return None
    if not isinstance(scorers, list):
        raise UnreadableField("it is not a list of scorers")
    try:
        return [
            {"name": score_name, "sha256": sha256_value(item)}
            for item in scorers
            if isinstance(item, dict) and item.get("name") == score_name
        ]
    except (TypeError, ValueError):
        raise UnreadableField(f"the entry of scorer {score_name!r} is not JSON data") from None


def _role_models(value: Any) -> dict[str, Any] | None:
    """Each model role's name and model id, without the rest of its configuration.

    A run's roles also record ``base_url``, ``args`` and the generate config with its
    ``extra_headers``, which can hold endpoints and credentials; none of it reaches a report.
    None when the run records no roles. Raises ``UnreadableField`` when the value is not a
    JSON mapping of roles, or a role has no model id.
    """

    roles = _recorded_field(value)
    if roles is None:
        return None
    if not isinstance(roles, dict):
        raise UnreadableField("it is not a mapping of model roles")
    named: dict[str, Any] = {}
    for name in sorted(roles, key=str):
        role = roles[name]
        model = role.get("model") if isinstance(role, dict) else None
        if not isinstance(model, str):
            raise UnreadableField(
                f"model role {str(name)!r} has no model id; each role must map to a "
                "configuration with a 'model'"
            )
        named[name] = {"model": model}
    return named


def _unreadable_configuration(run_key: str, field: str, why: str) -> Diagnostic:
    """The warning for a run field that cannot be read. It names the field, never its value."""

    return Diagnostic(
        code="JUDGE_CONFIGURATION_UNREADABLE",
        severity="WARNING",
        message=(
            f"run {run_key} records {field} that cannot be read: {why}. The judges' recorded "
            f"configurations leave out this run's {field}"
        ),
        scope=scope_path("runs", run_key),
        details={"run_key": run_key, "field": field},
    )


def _add_judge_configurations(
    panel: dict[str, Any],
    plan: AnalysisPlan,
    data: NormalizedData,
    source_data: SourceData,
    diagnostics: list[Diagnostic],
) -> None:
    """Record, per judge and cohort, the scorer fingerprints and model roles of its runs.

    The values are the distinct ones the cohort's runs record, sorted by their canonical
    form; a cohort whose runs record none says so with null. Neither copies a configuration
    value other than a role's model id, so no credential, endpoint or path reaches a report.
    A run whose ``scorers`` or ``model_roles`` cannot be read contributes nothing to it, with
    a ``JUDGE_CONFIGURATION_UNREADABLE`` warning, once per run and field.
    """

    runs_by_key = {run.run_key: run for run in source_data.runs}
    by_id = {judge.id: judge for judge in plan.inputs.judges}
    warned: set[tuple[str, str]] = set()

    def warn(run_key: str, field: str, exc: UnreadableField) -> None:
        if (run_key, field) not in warned:
            warned.add((run_key, field))
            diagnostics.append(_unreadable_configuration(run_key, field, str(exc)))

    for entry in panel["judges"]:
        score_name = by_id[entry["id"]].score.name
        configuration: dict[str, Any] = {}
        for summary in sorted(data.cohorts, key=lambda item: item["cohort_id"]):
            scorers: dict[str, Any] = {}
            roles: dict[str, Any] = {}
            for run_key in summary["run_keys"]:
                attributes = runs_by_key[run_key].attributes
                try:
                    found = scorer_configurations(attributes.get("scorers"), score_name)
                except UnreadableField as exc:
                    warn(run_key, "scorers", exc)
                    found = None
                for item in found or []:
                    scorers[item["sha256"]] = item
                try:
                    named = _role_models(attributes.get("model_roles"))
                except UnreadableField as exc:
                    warn(run_key, "model_roles", exc)
                    named = None
                if named is not None:
                    roles[sha256_value(named)] = named
            configuration[summary["cohort_id"]] = {
                "scorer": [scorers[key] for key in sorted(scorers)] or None,
                "model_roles": [roles[key] for key in sorted(roles)] or None,
            }
        entry["configuration"] = configuration
