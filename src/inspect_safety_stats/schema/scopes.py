"""The path a diagnostic's ``scope`` holds: where in the plan, the data or the results it sits.

A scope is a ``/``-separated path. Its first segment names what it points into:

- ``plan``: a location in the analysis plan (or the generative plan), field names then ids,
  as in ``plan/outputs/direct_mean_intervals``, ``plan/claims/mean_safe`` or
  ``plan/inputs/judges/j1``. ``decision_policy`` is the same for a decision policy.
- ``cohorts``, ``runs``, ``evals``, ``logs``, ``prompts``, ``strata``: the data, id then
  collection/id pairs, as in ``cohorts/candidate/prompts/p1``,
  ``runs/<run key>/prompts/p1/repeats/2`` or ``logs/<log name>``. ``prompts`` and ``strata``
  are the design manifest's, before any cohort; a cohort's own sit under it.
- ``comparisons``, ``comparison_sets``, ``families``: results that belong to no one cohort,
  as in ``comparisons/c1/methods/m/units/prompt``. Results of one cohort sit under it:
  ``cohorts/candidate/groups/g1/measures/worst_of_k``. A hierarchy node is
  ``levels/<level>/nodes/<node>`` below its group or comparison, its id written by
  :func:`node_id`.

A scope names the item a finding concerns down to the result row, method and profile it is
about, so that two findings of one code have two scopes: a report sorts its diagnostics by
severity, code and scope, and two that tie keep the order they were emitted in, which no
output order may depend on.

Below every root but ``plan`` and ``decision_policy`` the segments are an id, then
collection/id pairs; the root alone means every item of it (``comparisons``: every
comparison). Every segment is percent-encoded for ``%``
(``%25``) and ``/`` (``%2F``) only, so any id, a prompt id from data included, splits back
unambiguously and the rest stays readable.

Build a scope with :func:`scope_path` (or the shorthands below it) and read one with
:func:`scope_segments`. ``Diagnostic`` rejects a scope that does not parse.
"""

from __future__ import annotations

from collections.abc import Mapping

PLAN_ROOT = "plan"
DECISION_POLICY_ROOT = "decision_policy"
PAIRED_ROOTS = frozenset(
    {
        "cohorts",
        "runs",
        "evals",
        "logs",
        "prompts",
        "strata",
        "comparisons",
        "comparison_sets",
        "families",
    }
)
ROOTS = PAIRED_ROOTS | {PLAN_ROOT, DECISION_POLICY_ROOT}


def encode_segment(segment: str) -> str:
    # ORDER MATTERS: `%` first, or the `%` of `%2F` would be encoded again.
    return segment.replace("%", "%25").replace("/", "%2F")


def decode_segment(segment: str) -> str:
    # ORDER MATTERS: `%2F` first, or `%252F` (an id holding `%2F`) would decode to `/`.
    return segment.replace("%2F", "/").replace("%25", "%")


def scope_path(root: str, *segments: object) -> str:
    """The scope of ``root`` followed by ``segments``, each encoded."""
    path = "/".join([root, *(encode_segment(str(segment)) for segment in segments)])
    scope_segments(path)
    return path


def scope_segments(scope: str) -> list[str]:
    """The decoded segments of ``scope``, root first; ``ValueError`` if it breaks the convention."""
    raw = scope.split("/")
    root = raw[0]
    if root not in ROOTS:
        raise ValueError(
            f"scope {scope!r} starts with {root!r}, not one of {', '.join(sorted(ROOTS))}"
        )
    if any(segment == "" for segment in raw):
        raise ValueError(f"scope {scope!r} has an empty segment")
    if root not in PAIRED_ROOTS and len(raw) < 2:
        raise ValueError(f"scope {scope!r} names no location after its root")
    segments = [decode_segment(segment) for segment in raw[1:]]
    for segment, decoded in zip(raw[1:], segments, strict=True):
        if encode_segment(decoded) != segment:
            raise ValueError(
                f"scope {scope!r} has segment {segment!r}, which is not percent-encoded "
                "for % and / alone"
            )
    if root in PAIRED_ROOTS and segments and len(segments) % 2 != 1:
        raise ValueError(
            f"scope {scope!r} under {root!r} is not an id followed by collection/id pairs"
        )
    return [root, *segments]


def plan_scope(*location: object) -> str:
    """A location in the plan: ``plan/<field>/.../<id>``."""
    return scope_path(PLAN_ROOT, *location)


def cohort_scope(cohort_id: object, *pairs: object) -> str:
    """A cohort, or an item of it: ``cohorts/<id>/<collection>/<id>...``."""
    return scope_path("cohorts", cohort_id, *pairs)


def prompt_scope(cohort_id: object, prompt_id: object) -> str:
    return cohort_scope(cohort_id, "prompts", prompt_id)


def node_id(path: Mapping[str, object]) -> str:
    """A hierarchy node's id in a scope: its ``level=value`` pairs, root level first.

    The pairs are joined by ``/``, which the scope encodes, so the node is one segment:
    ``.../levels/question_id/nodes/task_id=t1%2Fquestion_id=q1``.
    """
    return "/".join(f"{key}={value}" for key, value in path.items())


def cohort_of(scope: str) -> str | None:
    """The cohort id a scope lies under, or ``None`` when it lies under no cohort."""
    segments = scope_segments(scope)
    return segments[1] if segments[0] == "cohorts" else None
