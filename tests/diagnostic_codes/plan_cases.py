"""Cases for the codes whose source is the plan or its registration."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.diagnostic_codes.harness import (
    BASE_RUN,
    Case,
    Inputs,
    Outcome,
    Run,
    analysis,
    claim,
)
from tests.diagnostic_codes.judge_analysis_cases import census, judged, validated
from tests.diagnostic_codes.logs_cases import (
    SHUFFLE_SEED,
    anytime_bounds,
    no_anytime_bound,
    on_inspect_logs,
)

Change = Callable[[Inputs], None]


def _changes(*changes: Change) -> Change:
    def change(inputs: Inputs) -> None:
        for item in changes:
            item(inputs)

    return change


def _outputs(inputs: Inputs) -> dict[str, Any]:
    outputs: dict[str, Any] = inputs.plan["outputs"]
    return outputs


def _family(family_id: str, method: str = "bonferroni") -> dict[str, Any]:
    return {"id": family_id, "purpose": "simultaneous_intervals", "method": method, "alpha": 0.05}


def _add_family(family_id: str, method: str = "bonferroni") -> Change:
    def change(inputs: Inputs) -> None:
        inputs.plan["multiplicity"]["families"].append(_family(family_id, method))

    return change


def _measures(*measures: dict[str, Any]) -> Change:
    def change(inputs: Inputs) -> None:
        _outputs(inputs)["risk_measures"] = [dict(measure) for measure in measures]

    return change


def _claim(**fields: Any) -> Change:
    def change(inputs: Inputs) -> None:
        inputs.plan["claims"][0].update(fields)

    return change


def _rollout_profile(inputs: Inputs) -> None:
    inputs.plan.setdefault("design", {})["independence_profiles"] = {
        "rollout": {"block_by": ["rollout"], "comparison_coupling": "independent"}
    }


def _direct_means(**fields: Any) -> Change:
    """Direct mean intervals in their own bonferroni family, independent rollouts."""

    def change(inputs: Inputs) -> None:
        _rollout_profile(inputs)
        inputs.plan["multiplicity"]["families"].append(_family("direct_means"))
        _outputs(inputs)["direct_mean_intervals"] = {
            "enabled": True,
            "family": "direct_means",
            "independence": {"primary": "rollout"},
            **fields,
        }

    return change


def _baseline(match: dict[str, Any] | None = None, **run: Any) -> Change:
    """A second run, r2, of model n, and a cohort ``baseline`` that matches it."""

    def change(inputs: Inputs) -> None:
        inputs.plan["cohorts"]["baseline"] = {"match": match or {"model": "n"}}
        inputs.runs.append({**BASE_RUN, "run_key": "r2", "model": "n", **run})
        inputs.rollouts.extend(
            {**record, "run_key": "r2"} for record in list(inputs.rollouts)
        )

    return change


def _red_r1(inputs: Inputs) -> None:
    inputs.runs[0]["team"] = "red"


def _comparison(candidate: str = "candidate", **fields: Any) -> Change:
    def change(inputs: Inputs) -> None:
        _rollout_profile(inputs)
        inputs.plan["multiplicity"]["families"].append(_family("comparisons"))
        inputs.plan["comparisons"] = [
            {
                "id": "versus",
                "candidate": candidate,
                "baseline": "baseline",
                "group": "all",
                "measure": "mean",
                "family": "comparisons",
                "independence": {"primary": "rollout"},
                **fields,
            }
        ]

    return change


def _sampled(inputs: Inputs) -> None:
    """A sampled design: a frame of three prompts, of which p3 was not drawn."""

    design = inputs.plan.setdefault("design", {})
    design.update(type="sampled", inclusion_probability_column="inclusion_probability")
    inputs.manifest.append(
        {"prompt_id": "p3", "question_id": "q3", "task_id": "t1", "expected_rollouts": 4}
    )
    for row in inputs.manifest:
        row["inclusion_probability"] = 0.75
    _outputs(inputs)["prompt_bounds"]["simultaneous"] = False
    _direct_means()(inputs)


def _sampled_prompt_sensitivity(inputs: Inputs) -> None:
    """A sampled design that declares a prompt-level sensitivity profile."""

    _sampled(inputs)
    inputs.plan["design"]["independence_profiles"]["prompt"] = {
        "block_by": ["prompt_id"],
        "comparison_coupling": "joint",
    }
    _outputs(inputs)["direct_mean_intervals"]["independence"]["sensitivities"] = ["prompt"]


def _sampled_without_direct_means(inputs: Inputs) -> None:
    _sampled(inputs)
    _outputs(inputs)["direct_mean_intervals"]["enabled"] = False


def _prompt_bounds_reported(outcome: Outcome) -> bool:
    return all(item.upper is not None for item in outcome.report.prompt_estimates)


def _no_prompt_bounds(triggered: Outcome, control: Outcome) -> None:
    assert _prompt_bounds_reported(control)
    assert not any(item.upper is not None for item in triggered.report.prompt_estimates)


def _claim_undecided(triggered: Outcome, control: Outcome) -> None:
    assert claim(control).status == "PASS"
    assert claim(triggered).status == "INCONCLUSIVE"


def _claim_unchanged(triggered: Outcome, control: Outcome) -> None:
    assert claim(triggered) == claim(control)


def _results_unchanged(triggered: Outcome, control: Outcome) -> None:
    assert triggered.report.prompt_estimates == control.report.prompt_estimates
    assert triggered.report.group_risks == control.report.group_risks
    assert triggered.report.claims == control.report.claims


# --- Decision policies and registration receipts -------------------------------------------


def _policy(
    *, set_at: str = "2026-01-15T00:00:00Z", claim_id: str = "low", sha: str | None = None
) -> Change:
    """A decision policy with one version that tightens a claim's thresholds."""

    def text(plan: Path) -> str:
        from inspect_safety_stats.config.decision_policy import inference_plan_hash
        from inspect_safety_stats.config.loader import load_plan

        version = {
            "id": "v1",
            "set_at": set_at,
            "reason": "tighter",
            "thresholds": {
                claim_id: {"acceptable_upper": 0.01, "unacceptable_lower": 0.95},
            },
        }
        return json.dumps(
            {
                "inference_plan_sha256": sha or inference_plan_hash(load_plan(plan)),
                "versions": [version],
            }
        )

    def change(inputs: Inputs) -> None:
        inputs.files["policy.json"] = text
        inputs.decision_policy = "policy.json"

    return change


def _two_gates_loosened(*claim_ids: str) -> Change:
    """Claims ``low`` on direct means and ``worst_low`` on prompt bounds, and a policy.

    The two claims' bounds rest on two coverage events, the direct-mean family and the
    prompt family. The policy loosens acceptable_upper of the claims in ``claim_ids`` and
    restates the plan's for the others, for both claims when ``claim_ids`` is empty.
    """

    def text(plan: Path) -> str:
        from inspect_safety_stats.config.decision_policy import inference_plan_hash
        from inspect_safety_stats.config.loader import load_plan

        thresholds = {
            claim_id: {
                "acceptable_upper": 0.91 if claim_id in claim_ids else 0.9,
                "unacceptable_lower": 0.95,
            }
            for claim_id in ("low", "worst_low")
        }
        version = {"id": "v1", "set_at": "2026-01-15T00:00:00Z", "thresholds": thresholds}
        return json.dumps(
            {"inference_plan_sha256": inference_plan_hash(load_plan(plan)), "versions": [version]}
        )

    def change(inputs: Inputs) -> None:
        _measures(MEAN, WORST)(inputs)
        _direct_means()(inputs)
        inputs.plan["claims"].append(
            {**inputs.plan["claims"][0], "id": "worst_low", "measure": "worst"}
        )
        inputs.files["policy.json"] = text
        inputs.decision_policy = "policy.json"

    return change


def _overall_pass_withheld(triggered: Outcome, control: Outcome) -> None:
    # Every claim keeps its bounds and passes in both; only the overall decision differs.
    def verdicts(outcome: Outcome) -> list[tuple[str, str, float | None]]:
        return [(item.claim_id, item.status, item.upper) for item in outcome.report.claims]

    assert verdicts(triggered) == verdicts(control)
    assert {item.status for item in control.report.claims} == {"PASS"}
    assert (triggered.report.decision, control.report.decision) == ("INCONCLUSIVE", "PASS")


def _fixed_claim(method: str) -> Callable[[Inputs, list[Any]], None]:
    """A validated judge calibration, claim ``low`` tested by ``method``, and a policy on it."""

    def change(inputs: Inputs, observations: list[Any]) -> None:
        validated(inputs, census(observations), sampling="simple_random")
        _claim(method=method)(inputs)
        _policy()(inputs)

    return change


def _anytime_family(family_id: str) -> Change:
    def change(inputs: Inputs) -> None:
        _outputs(inputs)["anytime_bounds"]["family"] = family_id

    return change


def _anytime_logs(family_id: str | None) -> Run:
    change = anytime_bounds
    if family_id is not None:
        change = _changes(change, _anytime_family(family_id))
    return on_inspect_logs([{"shuffle": SHUFFLE_SEED}], change)


def _policy_applied(applied: bool) -> Callable[[Outcome, Outcome], None]:
    def check(triggered: Outcome, control: Outcome) -> None:
        assert control.report.decision_policy.applied
        assert triggered.report.decision_policy.applied is applied

    return check


def _receipt(*, registered_at: str = "2026-01-15T00:00:00+00:00", sha: str | None = None
             ) -> Change:
    """A registration receipt for the plan, written at ``registered_at``."""

    def text(plan: Path) -> str:
        from inspect_safety_stats.config.loader import canonical_plan_hash, load_plan
        from inspect_safety_stats.config.registration import write_receipt

        written = plan.parent / "receipt.tmp"
        write_receipt(
            written,
            sha or canonical_plan_hash(load_plan(plan)),
            package_version="0",
            now=datetime.fromisoformat(registered_at),
        )
        content = written.read_text(encoding="utf-8")
        written.unlink()
        return content

    def change(inputs: Inputs) -> None:
        inputs.files["receipt.json"] = text
        inputs.registration_receipt = "receipt.json"

    return change


def _self_declared(inputs: Inputs) -> None:
    del inputs.plan["analysis"]["registration"]


def _frozen_at(when: str) -> Change:
    def change(inputs: Inputs) -> None:
        inputs.plan["analysis"]["plan_frozen_at"] = when

    return change


def _registered_timing_recorded(triggered: Outcome, control: Outcome) -> None:
    # The receipt takes the place of PLAN_SELF_DECLARED, and nothing else changes.
    (registered,) = triggered.with_code("PLAN_REGISTERED")
    assert registered.details["registered_at"] == "2026-01-15T00:00:00+00:00"
    assert triggered.with_code("PLAN_SELF_DECLARED") == []
    assert control.with_code("PLAN_SELF_DECLARED") != []
    _results_unchanged(triggered, control)


# --- Measures --------------------------------------------------------------------------------

MEAN = {"id": "mean", "type": "weighted_mean"}
WORST = {"id": "worst", "type": "maximum"}
WORST_OF_2 = {"id": "worst_of_2", "type": "worst_of_k", "k": 2, "direct_interval": True}


def _group_risk(outcome: Outcome, measure: str) -> Any:
    (found,) = [item for item in outcome.report.group_risks if item.measure_id == measure]
    return found


def _no_risk_estimate(measure: str) -> Callable[[Outcome, Outcome], None]:
    def check(triggered: Outcome, control: Outcome) -> None:
        assert _group_risk(control, measure).estimate is not None
        assert _group_risk(triggered, measure).estimate is None

    return check


def _no_group_risks(triggered: Outcome, control: Outcome) -> None:
    # The plan asked for no group risk, and none is reported; prompt bounds are as before.
    assert triggered.report.group_risks == []
    assert control.report.group_risks != []
    assert triggered.report.prompt_estimates == control.report.prompt_estimates


def _no_direct_means(triggered: Outcome, control: Outcome) -> None:
    # Direct mean intervals bound weighted means only: without one, none is computed, and
    # the measure both analyses have is reported as before.
    def direct(outcome: Outcome) -> list[str]:
        return [
            interval.interval_id
            for risk in outcome.report.group_risks
            for interval in risk.intervals
            if interval.interval_id.startswith("direct_")
        ]

    assert direct(control) != []
    assert direct(triggered) == []
    assert _group_risk(triggered, "worst") == _group_risk(control, "worst")


def _sampled_prompt_bounds_recorded(triggered: Outcome, control: Outcome) -> None:
    # The simultaneous bounds are reported as computed, for the sampled items only.
    def bounded(outcome: Outcome) -> list[str]:
        return sorted(
            item.prompt_id for item in outcome.report.prompt_estimates if item.upper is not None
        )

    assert bounded(triggered) == bounded(control) == ["p1", "p2"]


def _sampled_sensitivities_withheld(triggered: Outcome, control: Outcome) -> None:
    # Only the frame intervals are reported, as without the sensitivity profile.
    def profiles(outcome: Outcome) -> list[str | None]:
        return sorted(
            {
                interval.independence_profile
                for row in outcome.report.group_risks
                for interval in row.intervals
                if interval.independence_profile is not None
            }
        )

    assert profiles(triggered) == profiles(control) == ["frame_item"]


CASES = [
    Case(
        code="ALPHA_ALLOCATION_INVALID",
        shows="invalid",
        # Explicit alpha must name every manifest prompt; the trigger leaves out p2.
        trigger=analysis(
            lambda inputs: _outputs(inputs)["prompt_bounds"].update(
                alpha_allocation="explicit", explicit_alpha={"p1": 0.01}
            )
        ),
        control=analysis(
            lambda inputs: _outputs(inputs)["prompt_bounds"].update(
                alpha_allocation="explicit", explicit_alpha={"p1": 0.01, "p2": 0.01}
            )
        ),
        effect=_no_prompt_bounds,
    ),
    Case(
        code="COHORT_OVERLAP",
        shows="invalid",
        # Run r1 is red like r2, so a cohort of red runs takes it as well as candidate does.
        trigger=analysis(_changes(_red_r1, _baseline({"attributes.team": "red"}, team="red"))),
        control=analysis(
            _changes(_red_r1, _baseline({"attributes.team": "blue"}, team="blue"))
        ),
        effect=_claim_undecided,
    ),
    Case(
        code="DECISION_POLICY_AFTER_GENERATION",
        shows="recorded",
        # The policy's version is set after r1 started, on 2026-02-01.
        trigger=analysis(_policy(set_at="2026-02-02T00:00:00Z")),
        control=analysis(_policy(set_at="2026-01-15T00:00:00Z")),
        effect=lambda triggered, control: (
            _policy_applied(True)(triggered, control),
            _claim_unchanged(triggered, control),
        ),
    ),
    Case(
        code="DECISION_POLICY_OVERALL_UNCERTIFIED",
        shows="adjusted",
        # The policy moves one claim's acceptable_upper, and the two mandatory claims rest on
        # two coverage events; the control's policy restates the plan's thresholds.
        trigger=analysis(_two_gates_loosened("low")),
        control=analysis(_two_gates_loosened()),
        effect=_overall_pass_withheld,
    ),
    Case(
        code="DECISION_POLICY_SELF_DECLARED",
        shows="recorded",
        # The policy's version states a time before r1 started, on 2026-02-01, which nothing
        # checks; the control's states a time after it.
        trigger=analysis(_policy(set_at="2026-01-15T00:00:00Z")),
        control=analysis(_policy(set_at="2026-02-02T00:00:00Z")),
        effect=lambda triggered, control: (
            _policy_applied(True)(triggered, control),
            _claim_unchanged(triggered, control),
        ),
    ),
    Case(
        code="DECISION_POLICY_MISMATCH",
        shows="invalid",
        # The policy names another plan's inference hash.
        trigger=analysis(_policy(sha="0" * 64)),
        control=analysis(_policy()),
        effect=_policy_applied(False),
    ),
    Case(
        code="DECISION_POLICY_UNKNOWN_CLAIM",
        shows="invalid",
        trigger=analysis(_policy(claim_id="missing")),
        control=analysis(_policy()),
    ),
    Case(
        code="DECISION_POLICY_FIXED_CLAIM",
        shows="invalid",
        # The policy sets thresholds of a noisy_valid_test claim; the control's claim is a
        # corrected interval, whose thresholds a policy may set.
        trigger=judged(_fixed_claim("noisy_valid_test")),
        control=judged(_fixed_claim("corrected_interval")),
        effect=_policy_applied(False),
    ),
    Case(
        code="PLAN_ANYTIME_BOUNDS_INVALID",
        shows="invalid",
        # The anytime bounds name a family the plan does not declare. Only Inspect logs of a
        # shuffled run record the order the bounds need, so both run on such logs.
        trigger=_anytime_logs("undeclared"),
        control=_anytime_logs(None),
        effect=no_anytime_bound,
    ),
    Case(
        code="PLAN_DIAGNOSTIC_GATE",
        shows="invalid",
        # The claim gates on a measure the plan declares diagnostic only.
        trigger=analysis(
            _changes(_measures(MEAN, {**WORST, "diagnostic_only": True}), _claim(measure="worst"))
        ),
        control=analysis(_changes(_measures(MEAN, WORST), _claim(measure="worst"))),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_DUPLICATE_ID",
        shows="invalid",
        trigger=analysis(lambda inputs: inputs.plan["groups"].append({"id": "all"})),
        control=analysis(lambda inputs: inputs.plan["groups"].append({"id": "every"})),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_INVALID_CLAIM",
        shows="invalid",
        # noisy_valid_test needs validated judge calibration, which the plan does not have.
        trigger=analysis(_claim(method="noisy_valid_test")),
        control=analysis(_claim(method="corrected_interval")),
    ),
    Case(
        code="PLAN_INVALID_COMPARISON",
        shows="invalid",
        # The comparison puts the baseline cohort in both arms.
        trigger=analysis(_changes(_baseline(), _comparison(candidate="baseline"))),
        control=analysis(_changes(_baseline(), _comparison())),
    ),
    Case(
        code="PLAN_INVALID_MULTIPLICITY",
        shows="invalid",
        # Simultaneous prompt bounds take their alpha from an intersection-union family.
        trigger=analysis(
            lambda inputs: inputs.plan["multiplicity"]["families"][0].update(
                method="intersection_union"
            )
        ),
        control=analysis(
            lambda inputs: inputs.plan["multiplicity"]["families"][0].update(
                method="weighted_bonferroni"
            )
        ),
    ),
    Case(
        code="PLAN_METHOD_UNUSED",
        shows="recorded",
        # Direct mean intervals are on, but the plan asks for no weighted mean.
        trigger=analysis(
            _changes(_direct_means(), _measures(WORST), _claim(measure="worst"))
        ),
        control=analysis(
            _changes(_direct_means(), _measures(WORST, MEAN), _claim(measure="worst"))
        ),
        effect=_no_direct_means,
    ),
    Case(
        code="PLAN_NO_RISK_MEASURES",
        shows="recorded",
        trigger=analysis(_changes(_measures(), lambda inputs: inputs.plan.update(claims=[]))),
        control=analysis(lambda inputs: inputs.plan.update(claims=[])),
        effect=_no_group_risks,
    ),
    Case(
        code="PLAN_POSTHOC",
        shows="invalid",
        # The plan was frozen after r1 started.
        trigger=analysis(_frozen_at("2026-03-01T00:00:00Z")),
        control=analysis(_frozen_at("2026-01-31T00:00:00Z")),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_RECEIPT_MISMATCH",
        shows="invalid",
        trigger=analysis(_receipt(sha="0" * 64)),
        control=analysis(_receipt()),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_RECEIPT_POSTHOC",
        shows="invalid",
        # The receipt was written after r1 started.
        trigger=analysis(_receipt(registered_at="2026-02-01T00:00:01+00:00")),
        control=analysis(_receipt()),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_REGISTERED",
        shows="recorded",
        trigger=analysis(_changes(_self_declared, _receipt())),
        control=analysis(_self_declared),
        effect=_registered_timing_recorded,
    ),
    Case(
        code="PLAN_SAMPLED_COMPARISON_UNSUPPORTED",
        shows="invalid",
        trigger=analysis(_changes(_sampled, _baseline(), _comparison())),
        control=analysis(_changes(_sampled, _baseline())),
    ),
    Case(
        code="PLAN_SAMPLED_MEASURE_UNSUPPORTED",
        shows="invalid",
        trigger=analysis(_changes(_sampled, _measures(MEAN, WORST))),
        control=analysis(_changes(_sampled, _measures(MEAN))),
    ),
    Case(
        code="PLAN_SAMPLED_PROMPT_BOUNDS",
        shows="recorded",
        trigger=analysis(
            _changes(
                _sampled, lambda inputs: _outputs(inputs)["prompt_bounds"].update(simultaneous=True)
            )
        ),
        control=analysis(_sampled),
        effect=_sampled_prompt_bounds_recorded,
    ),
    Case(
        code="PLAN_SAMPLED_SENSITIVITY_PROFILES",
        shows="withheld",
        trigger=analysis(_sampled_prompt_sensitivity),
        control=analysis(_sampled),
        effect=_sampled_sensitivities_withheld,
    ),
    Case(
        code="PLAN_SAMPLED_REQUIRES_DIRECT_MEAN",
        shows="invalid",
        trigger=analysis(_sampled_without_direct_means),
        control=analysis(_sampled),
    ),
    Case(
        code="PLAN_SELF_DECLARED",
        shows="recorded",
        # Without an external registration, the freeze time is the plan's own word.
        trigger=analysis(_self_declared),
        control=analysis(),
        effect=_results_unchanged,
    ),
    Case(
        code="PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED",
        shows="invalid",
        trigger=analysis(
            _changes(
                _sampled,
                lambda inputs: _outputs(inputs)["direct_mean_intervals"].update(
                    single_valued_bound=True
                ),
            )
        ),
        control=analysis(_sampled),
    ),
    Case(
        code="PLAN_UNKNOWN_REFERENCE",
        shows="invalid",
        trigger=analysis(_claim(group="nowhere")),
        control=analysis(_claim(group="all")),
        effect=_claim_undecided,
    ),
    Case(
        code="PLAN_WORST_OF_K_INTERVAL_UNSUPPORTED",
        shows="invalid",
        # The direct worst-of-k interval is calibrated in the direct-mean family, which is off.
        trigger=analysis(_measures(MEAN, WORST_OF_2)),
        control=analysis(_changes(_direct_means(), _measures(MEAN, WORST_OF_2))),
    ),
    Case(
        code="RISK_MEASURE_INVALID",
        shows="invalid",
        # The mean of the top three prompts, of two.
        trigger=analysis(_measures(MEAN, {"id": "top", "type": "top_k_mean", "k": 3})),
        control=analysis(_measures(MEAN, {"id": "top", "type": "top_k_mean", "k": 2})),
        effect=_no_risk_estimate("top"),
    ),
]
