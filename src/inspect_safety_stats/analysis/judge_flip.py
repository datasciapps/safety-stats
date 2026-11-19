"""The ``JUDGE_FLIP_RATE`` diagnostic of the fixed path (#122, experimental)."""

from __future__ import annotations

from typing import Any

from inspect_safety_stats.config.loader import report_message, report_path
from inspect_safety_stats.config.models import AnalysisPlan
from inspect_safety_stats.ingest.judge_flip import load_judge_flip_manifest
from inspect_safety_stats.ingest.records import NormalizedData
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.schema.scopes import plan_scope
from inspect_safety_stats.stats.judge_flip import FlipRate, judge_flip_rate

SCOPE = plan_scope("inputs", "judge_flip")
METHOD = "clopper_pearson_two_sided"


def _rate(rate: FlipRate) -> dict[str, Any]:
    return {
        "flipped": rate.flipped,
        "rescored": rate.rescored,
        "estimate": rate.estimate,
        "lower": rate.lower,
        "upper": rate.upper,
    }


def judge_flip_diagnostic(
    plan: AnalysisPlan, data: NormalizedData, diagnostics: list[Diagnostic]
) -> None:
    """Add the flip-rate diagnostic the plan's ``inputs.judge_flip`` asks for, if any.

    The rate is over the primary judge's judge-positive observations, pooled over the
    analysed cohorts. It is INFO, or a WARNING above ``warn_above``, and degrades nothing, so
    it never changes a claim status. A manifest that cannot be read, or that does not fit the
    run and the declared design, is a ``JUDGE_FLIP_INVALID`` error, as an invalid validation
    manifest is.
    """

    spec = plan.inputs.judge_flip
    if spec is None:
        return
    judge_positive = [row.observation_id for row in data.observations if row.failure is True]
    try:
        verdicts = load_judge_flip_manifest(spec.manifest)
        result = judge_flip_rate(
            judge_positive,
            verdicts,
            spec.wrappers,
            sampling=spec.sampling,
            alpha=spec.alpha,
        )
    except (OSError, ValueError) as exc:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_FLIP_INVALID",
                severity="ERROR",
                message=report_message(str(exc), plan, (spec.manifest,)),
                scope=SCOPE,
            )
        )
        return
    details: dict[str, Any] = {
        "manifest": report_path(spec.manifest, plan),
        "sampling": spec.sampling,
        "wrappers": list(spec.wrappers),
        "judge_positive": result.judge_positive,
        "alpha": spec.alpha,
        "method": METHOD,
        "warn_above": spec.warn_above,
    }
    if result.any_wrapper is None:
        diagnostics.append(
            Diagnostic(
                code="JUDGE_FLIP_RATE",
                severity="INFO",
                message="the run has no judge-positive completion, so no flip rate",
                scope=SCOPE,
                details={**details, "any_wrapper": None, "by_wrapper": {}},
            )
        )
        return
    rate = result.any_wrapper
    above = rate.estimate > spec.warn_above
    diagnostics.append(
        Diagnostic(
            code="JUDGE_FLIP_RATE",
            severity="WARNING" if above else "INFO",
            message=(
                f"{rate.flipped} of {rate.rescored} rescored judge-positive completions flip "
                f"to safe under at least one wrapper (rate {rate.estimate:.4g}, "
                f"{1 - spec.alpha:.4g} interval [{rate.lower:.4g}, {rate.upper:.4g}])"
                + (
                    f", above the declared warn_above {spec.warn_above:.4g}: the judge misses "
                    "unsafe content it is shown in another frame, so its PASS verdicts may "
                    "understate risk"
                    if above
                    else ""
                )
            ),
            scope=SCOPE,
            details={
                **details,
                "any_wrapper": _rate(rate),
                "by_wrapper": {
                    wrapper: _rate(item) for wrapper, item in result.by_wrapper.items()
                },
            },
        )
    )
