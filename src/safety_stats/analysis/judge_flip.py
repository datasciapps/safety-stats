"""The ``JUDGE_FLIP_RATE`` diagnostic of the fixed path (#122, experimental)."""

from __future__ import annotations

from typing import Any

from safety_stats.config.loader import report_message, report_path
from safety_stats.config.models import AnalysisPlan
from safety_stats.ingest.judge_flip import load_judge_flip_manifest
from safety_stats.ingest.outcomes import classified_by_policy, judge_positive
from safety_stats.ingest.records import NormalizedData
from safety_stats.schema.reports import Diagnostic
from safety_stats.schema.scopes import plan_scope
from safety_stats.stats.judge_flip import FlipRate, judge_flip_rate

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

    The rate is over the observations the primary judge scored unsafe, pooled over the
    analysed cohorts. A failure that a plan policy classified, such as a sample error counted
    as a failure, is not among them: no judge scored it, so its rescoring would say nothing
    about the judge, and a manifest that rescores one is rejected. The diagnostic is INFO, or
    a WARNING above ``warn_above``, and degrades nothing, so it never changes a claim status.
    A manifest that cannot be read, or that does not fit the run and the declared design, is
    a ``JUDGE_FLIP_INVALID`` error, as an invalid validation manifest is.

    The message says what the interval is about. Under ``simple_random`` it is for the run's
    share, since the sample is drawn completion by completion. Under ``all`` the share is
    known, and the interval is descriptive: it assumes the completions flip independently,
    which completions of one prompt need not, so the details count the prompts the rescored
    completions come from.
    """

    spec = plan.inputs.judge_flip
    if spec is None:
        return
    positive = [row.observation_id for row in data.observations if judge_positive(row)]
    policy_failures = {
        row.observation_id
        for row in data.observations
        if row.failure is True and classified_by_policy(row)
    }
    try:
        verdicts = load_judge_flip_manifest(spec.manifest)
        by_policy = sorted(policy_failures.intersection(verdicts))
        if by_policy:
            raise ValueError(
                "the judge flip manifest rescores completions that a plan policy, not the "
                "judge, counted as failures (a sample error, a content-filter stop, an "
                "unscored truncated answer or a missing score): "
                + ", ".join(by_policy[:5])
                + (f" and {len(by_policy) - 5} more" if len(by_policy) > 5 else "")
                + "; leave them out of the rescoring"
            )
        result = judge_flip_rate(
            positive,
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
    prompt_of = {row.observation_id: row.prompt_id for row in data.observations}
    prompts = len({prompt_of[key] for key in verdicts})
    details: dict[str, Any] = {
        "manifest": report_path(spec.manifest, plan),
        "sampling": spec.sampling,
        "wrappers": list(spec.wrappers),
        "judge_positive": result.judge_positive,
        "policy_classified_failures": len(policy_failures),
        "prompts": prompts,
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
    reading = (
        f" for the run's {result.judge_positive} judge-positive completions"
        if spec.sampling == "simple_random"
        else (
            "; the interval assumes the completions flip independently, which completions of "
            "one prompt need not, so it is descriptive"
        )
    )
    diagnostics.append(
        Diagnostic(
            code="JUDGE_FLIP_RATE",
            severity="WARNING" if above else "INFO",
            message=(
                f"{rate.flipped} of {rate.rescored} rescored judge-positive completions, from "
                f"{prompts} prompt{'' if prompts == 1 else 's'}, flip to safe under at least "
                f"one wrapper (rate {rate.estimate:.4g}, {1 - spec.alpha:.4g} interval "
                f"[{rate.lower:.4g}, {rate.upper:.4g}]{reading})"
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
