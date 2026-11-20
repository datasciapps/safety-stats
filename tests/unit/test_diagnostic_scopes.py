"""A diagnostic's scope is a path of collection/id pairs with percent-encoded ids.

Every ``Diagnostic`` an analysis builds validates its scope, so every test that runs an
analysis also checks that each scope it emits follows the convention.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inspect_safety_stats.analysis.estimates import prompt_blocked
from inspect_safety_stats.schema.reports import STORED_REPORT, Diagnostic, IntegritySummary
from inspect_safety_stats.schema.scopes import (
    cohort_of,
    cohort_scope,
    node_id,
    plan_scope,
    prompt_scope,
    scope_path,
    scope_segments,
)


@pytest.mark.parametrize(
    ("scope", "segments"),
    [
        (plan_scope("outputs", "direct_mean_intervals"), ["plan", "outputs", "direct_mean_intervals"]),  # noqa: E501
        (plan_scope("claims", "mean.safe"), ["plan", "claims", "mean.safe"]),
        (cohort_scope("candidate"), ["cohorts", "candidate"]),
        (prompt_scope("candidate", "p1"), ["cohorts", "candidate", "prompts", "p1"]),
        (scope_path("comparisons"), ["comparisons"]),
        (scope_path("runs", "r1", "prompts", "p1", "repeats", 2), ["runs", "r1", "prompts", "p1", "repeats", "2"]),  # noqa: E501
        # A hierarchy node is one segment, its level=value pairs root level first.
        (cohort_scope("c", "groups", "g", "levels", "q", "nodes", node_id({"t": "t1", "q": "a/b"})), ["cohorts", "c", "groups", "g", "levels", "q", "nodes", "t=t1/q=a/b"]),  # noqa: E501
    ],
)
def test_a_scope_splits_into_the_segments_it_was_built_from(
    scope: str, segments: list[str]
) -> None:
    assert scope_segments(scope) == segments


@pytest.mark.parametrize("prompt_id", ["a/b", "50%", "%2F", "a%2Fb/c", "x%25", ":", "é ü"])
def test_any_id_round_trips_through_its_encoding(prompt_id: str) -> None:
    scope = prompt_scope("candidate", prompt_id)
    assert scope.count("/") == 3
    assert scope_segments(scope)[-1] == prompt_id


def test_only_percent_and_slash_are_encoded() -> None:
    assert prompt_scope("c", "a/b%c d:e.f") == "cohorts/c/prompts/a%2Fb%25c d:e.f"


@pytest.mark.parametrize(
    "scope",
    [
        "candidate",  # a bare id: the free text scopes were before
        "candidate:p1",
        "inputs.judge_calibration",
        "plan",  # a plan root names no location
        "cohorts/",  # an empty segment
        "cohorts//prompts/p1",
        "cohorts/c/prompts",  # a collection without its id
        "cohorts/c/prompts/50%",  # `%` not encoded
        "cohorts/c/prompts/%41",  # encoded, but not for `%` or `/` alone
        "outputs/direct_mean_intervals",  # no root
    ],
)
def test_a_scope_off_the_convention_is_rejected(scope: str) -> None:
    with pytest.raises(ValueError):
        scope_segments(scope)
    with pytest.raises(ValidationError):
        Diagnostic(code="COHORT_EMPTY", severity="ERROR", message="empty", scope=scope)


def test_a_stored_report_with_free_text_scopes_still_reads() -> None:
    old = '{"status": "invalid", "diagnostics": [' + (
        '{"code": "COHORT_EMPTY", "severity": "ERROR", "message": "empty", "scope": "candidate"}'
    ) + "]}"
    summary = IntegritySummary.model_validate_json(old, context={STORED_REPORT: True})
    assert summary.diagnostics[0].scope == "candidate"


def test_cohort_of_names_the_cohort_a_scope_lies_under() -> None:
    assert cohort_of(prompt_scope("a/b", "p1")) == "a/b"
    assert cohort_of(plan_scope("claims", "a")) is None


@pytest.mark.parametrize(
    "scope",
    [
        scope_path("strata", "candidate"),
        plan_scope("claims", "candidate"),
        scope_path("prompts", "candidate"),
        cohort_scope("baseline", "strata", "candidate"),
    ],
)
def test_an_error_about_something_named_like_a_cohort_does_not_block_that_cohort(
    scope: str,
) -> None:
    # Scopes were bare ids, and an error whose scope equalled a cohort's id blocked it.
    error = Diagnostic(code="SAMPLE_DRAWS_MISMATCH", severity="ERROR", message="", scope=scope)
    assert prompt_blocked([error], "candidate", "p1") is False


def test_an_error_about_the_cohort_blocks_it() -> None:
    error = Diagnostic(
        code="GENERATIVE_SAMPLE_UNMATCHED",
        severity="ERROR",
        message="",
        scope=cohort_scope("candidate"),
    )
    assert prompt_blocked([error], "candidate", "p1") is True
    assert prompt_blocked([error], "baseline", "p1") is False
