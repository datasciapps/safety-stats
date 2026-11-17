"""Where in ``report.json`` the item a scope names is: a JSON pointer (RFC 6901).

A diagnostic's ``scope`` says where a finding sits in the plan, the data or the results
(:mod:`inspect_safety_stats.schema.scopes`). When the report holds a record of that item, its
``item`` is the pointer to that record, so a reader goes from the finding to the result it
concerns without searching:

==========================================  ==========================================
scope                                       item
==========================================  ==========================================
``plan/claims/<id>``                        ``/claims/<i>``
``plan/adaptive_claims/<id>``               ``/adaptive_claims/<i>``
``plan/comparisons/<id>``,                  ``/comparisons/<i>``, the comparison's
``comparisons/<id>`` and below it           overall result (no hierarchy level)
``comparison_sets/<id>`` and below it       ``/comparison_sets/<i>``
``plan/multiplicity/families/<id>``,        ``/multiplicity/<i>``
``families/<id>``
``cohorts/<id>``                            ``/cohorts/<i>``
``cohorts/<id>/prompts/<id>``               ``/prompt_estimates/<i>``
``cohorts/<id>/groups/<id>/measures/<id>``  ``/group_risks/<i>``
``prompts/<id>``                            ``/design/prompts/<i>``
``plan/claims``, ``plan/comparisons``,      the whole list: ``/claims``,
``comparisons``, ``comparison_sets``        ``/comparisons``, ``/comparison_sets``
``decision_policy`` and below it            ``/decision_policy``, when the report has one
==========================================  ==========================================

Every other scope, and a scope whose item the report does not hold, has no item: plan settings
such as ``plan/outputs/risk_measures``, and data the report keeps no record of (runs, logs,
evaluations, strata). The pointer names one record exactly; it never falls back to an
enclosing one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_safety_stats.schema.scopes import scope_segments

if TYPE_CHECKING:
    from inspect_safety_stats.schema.reports import SafetyReport

_WHOLE_LISTS = {
    ("plan", "claims"): "/claims",
    ("plan", "adaptive_claims"): "/adaptive_claims",
    ("plan", "comparisons"): "/comparisons",
    ("comparisons",): "/comparisons",
    ("comparison_sets",): "/comparison_sets",
}


def _first(pairs: list[tuple[object, int]]) -> dict[object, int]:
    found: dict[object, int] = {}
    for key, index in pairs:
        found.setdefault(key, index)
    return found


class ItemPointers:
    """The pointers of one report's items, looked up by scope."""

    def __init__(self, report: SafetyReport) -> None:
        self._has_policy = report.decision_policy is not None
        self._claims = _first([(item.claim_id, i) for i, item in enumerate(report.claims)])
        self._adaptive = _first(
            [(item.claim_id, i) for i, item in enumerate(report.adaptive_claims)]
        )
        self._comparisons = _first(
            [
                (item.comparison_id, i)
                for i, item in enumerate(report.comparisons)
                if item.level is None
            ]
        )
        self._sets = _first(
            [(item.comparison_set_id, i) for i, item in enumerate(report.comparison_sets)]
        )
        self._families = _first(
            [(item.get("id"), i) for i, item in enumerate(report.multiplicity)]
        )
        self._cohorts = _first(
            [(item.get("cohort_id"), i) for i, item in enumerate(report.cohorts)]
        )
        self._prompts = _first(
            [
                ((item.cohort_id, item.prompt_id), i)
                for i, item in enumerate(report.prompt_estimates)
            ]
        )
        self._risks = _first(
            [
                ((item.cohort_id, item.group_id, item.measure_id), i)
                for i, item in enumerate(report.group_risks)
            ]
        )
        design_prompts = report.design.get("prompts")
        self._design_prompts = _first(
            [
                (item.get("prompt_id"), i)
                for i, item in enumerate(design_prompts if isinstance(design_prompts, list) else [])
                if isinstance(item, dict)
            ]
        )

    def pointer(self, scope: str | None) -> str | None:
        """The pointer to the item ``scope`` names; None when the report holds no record of it."""

        if scope is None:
            return None
        try:
            segments = scope_segments(scope)
        except ValueError:
            # A report written before scopes were paths has free-text scopes.
            return None
        whole = _WHOLE_LISTS.get(tuple(segments))
        if whole is not None:
            return whole
        root, rest = segments[0], segments[1:]
        if root == "decision_policy":
            return "/decision_policy" if self._has_policy else None
        if root == "plan":
            return self._plan_pointer(rest)
        if root == "comparisons" and rest:
            return _at("/comparisons", self._comparisons.get(rest[0]))
        if root == "comparison_sets" and rest:
            return _at("/comparison_sets", self._sets.get(rest[0]))
        if root == "families" and len(rest) == 1:
            return _at("/multiplicity", self._families.get(rest[0]))
        if root == "prompts" and len(rest) == 1:
            return _at("/design/prompts", self._design_prompts.get(rest[0]))
        if root == "cohorts":
            return self._cohort_pointer(rest)
        return None

    def _plan_pointer(self, rest: list[str]) -> str | None:
        if len(rest) == 2 and rest[0] == "claims":
            return _at("/claims", self._claims.get(rest[1]))
        if len(rest) == 2 and rest[0] == "adaptive_claims":
            return _at("/adaptive_claims", self._adaptive.get(rest[1]))
        if len(rest) == 2 and rest[0] == "comparisons":
            return _at("/comparisons", self._comparisons.get(rest[1]))
        if len(rest) == 3 and rest[:2] == ["multiplicity", "families"]:
            return _at("/multiplicity", self._families.get(rest[2]))
        return None

    def _cohort_pointer(self, rest: list[str]) -> str | None:
        if len(rest) == 1:
            return _at("/cohorts", self._cohorts.get(rest[0]))
        if len(rest) == 3 and rest[1] == "prompts":
            return _at("/prompt_estimates", self._prompts.get((rest[0], rest[2])))
        if len(rest) == 5 and rest[1] == "groups" and rest[3] == "measures":
            return _at("/group_risks", self._risks.get((rest[0], rest[2], rest[4])))
        return None


def _at(collection: str, index: int | None) -> str | None:
    return None if index is None else f"{collection}/{index}"
