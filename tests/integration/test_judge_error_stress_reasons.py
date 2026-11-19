"""Where the judge-error stress does not apply, and that it never moves a claim, end to end.

Each case runs ``analyze`` through the diagnostic-code harness with the stress switched on.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from inspect_safety_stats.schema.observations import Observation
from tests.diagnostic_codes.harness import (
    Inputs,
    Outcome,
    base_inputs,
    claim,
    rollout,
    run_analysis,
)
from tests.diagnostic_codes.judge_analysis_cases import (
    census,
    comparison,
    judged,
    panel,
    stressed_judge,
    validated,
)


def _stressed(inputs: Inputs) -> None:
    inputs.plan.setdefault("report", {})["judge_error_stress"] = True


def _stress_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch the stress on in the plans ``panel()`` writes, which start from base_inputs."""

    def stressed() -> Inputs:
        inputs = base_inputs()
        _stressed(inputs)
        return inputs

    monkeypatch.setattr("tests.diagnostic_codes.judge_analysis_cases.base_inputs", stressed)


def _stress(outcome: Outcome, claim_id: str = "low") -> Any:
    record = claim(outcome, claim_id).judge_error_stress
    assert record is not None
    return record


def _run(directory: Path, *changes: Callable[[Inputs], None]) -> Outcome:
    inputs = base_inputs()
    _stressed(inputs)
    for change in changes:
        change(inputs)
    return run_analysis(inputs, directory)


def _judged_and_stressed(
    directory: Path,
    change: Callable[[Inputs, list[Any]], None],
    data: Callable[[Inputs], None] | None = None,
) -> Outcome:
    def stressed_data(inputs: Inputs) -> None:
        _stressed(inputs)
        if data is not None:
            data(inputs)

    return judged(change, stressed_data)(directory)


def _comparison_claim(inputs: Inputs) -> None:
    comparison("weighted_kl_chernoff")(inputs)
    inputs.plan["claims"].append(
        {
            "id": "versus",
            "title": "Candidate is no riskier than baseline",
            "comparison": "versus",
            "family": "claims",
            "acceptable_upper": 0.5,
            "unacceptable_lower": 0.9,
        }
    )


def test_a_comparison_claim_is_not_stressed(tmp_path: Path) -> None:
    outcome = _run(tmp_path, _comparison_claim)

    stress = _stress(outcome, "versus")
    assert not stress.applicable
    assert str(stress.reason).startswith("comparison claims are not stressed")
    # The claim on the candidate's own row still is.
    assert _stress(outcome).applicable


def test_a_claim_decided_on_a_validated_correction_is_not_stressed(tmp_path: Path) -> None:
    outcome = _judged_and_stressed(
        tmp_path, lambda inputs, observations: validated(inputs, census(observations))
    )

    decided = claim(outcome)
    assert decided.method == "validated_judge_misclassification_correction"
    stress = _stress(outcome)
    assert not stress.applicable
    assert "validated judge correction" in str(stress.reason)


def test_a_claim_a_validated_calibration_leaves_inconclusive_is_not_stressed(
    tmp_path: Path,
) -> None:
    # The maximum has only an uncertified correction, so the claim on it is INCONCLUSIVE
    # rather than decided on judge-reported risk.
    def worst(inputs: Inputs) -> None:
        inputs.plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
        inputs.plan["claims"][0]["measure"] = "worst"

    outcome = _judged_and_stressed(
        tmp_path, lambda inputs, observations: validated(inputs, census(observations)), worst
    )

    assert claim(outcome).status == "INCONCLUSIVE"
    stress = _stress(outcome)
    assert not stress.applicable
    assert "validated calibration leaves the claim INCONCLUSIVE" in str(stress.reason)


def test_a_noisy_valid_claim_is_not_stressed(tmp_path: Path) -> None:
    def noisy_valid(inputs: Inputs, observations: list[Any]) -> None:
        validated(inputs, census(observations), sampling="simple_random")
        inputs.plan["claims"][0]["method"] = "noisy_valid_test"

    outcome = _judged_and_stressed(tmp_path, noisy_valid)

    assert claim(outcome).method == "noisy_valid_certification_test"
    stress = _stress(outcome)
    assert not stress.applicable
    assert "noisy-but-valid test" in str(stress.reason)


def test_a_claim_of_an_invalid_analysis_is_not_stressed(tmp_path: Path) -> None:
    # A rollout of a prompt the design manifest does not list makes data integrity invalid;
    # the listed prompts are still bounded.
    def unexpected(inputs: Inputs) -> None:
        inputs.rollouts.append(rollout("p3", 1))

    outcome = _run(tmp_path, unexpected)

    assert outcome.status == "invalid"
    assert claim(outcome).upper is not None
    stress = _stress(outcome)
    assert not stress.applicable
    assert "data integrity is invalid" in str(stress.reason)


@pytest.mark.parametrize(("rule", "source"), [("any", "composite:panel"), ("majority", None)])
def test_a_majority_composite_has_no_declared_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rule: str, source: str | None
) -> None:
    _stress_panels(monkeypatch)
    outcome = panel(rule, None, None)(tmp_path)

    stress = _stress(outcome)
    assert stress.applicable
    # The tipping lines need no box, so the record has them either way.
    assert stress.pass_sensitivity_at_perfect_specificity is not None
    assert stress.accuracy_source == source
    assert (stress.stressed_status is None) == (source is None)


@pytest.mark.parametrize(
    ("sensitivity", "specificity", "acceptable_upper", "status", "stressed"),
    [
        # Upper bound 0.757 passes 0.9; over Se = 0.5 its true-risk upper bound is 1.
        (0.5, 1.0, 0.9, "PASS", "INCONCLUSIVE"),
        # 0.757 does not pass 0.25; at Se = 0.95, Sp = 0.3 its true-risk upper bound is
        # (0.757 - 0.7) / 0.25 = 0.227, which does.
        (0.95, 0.3, 0.25, "INCONCLUSIVE", "PASS"),
    ],
)
def test_the_stress_never_moves_a_claim(
    tmp_path: Path,
    sensitivity: float,
    specificity: float,
    acceptable_upper: float,
    status: str,
    stressed: str,
) -> None:
    off_inputs = base_inputs()
    off_inputs.plan["inputs"]["judges"] = [
        {
            "id": "primary_judge",
            "score": copy.deepcopy(off_inputs.plan["inputs"]["score"]),
            "accuracy": {
                "sensitivity": {"lower": sensitivity, "upper": sensitivity},
                "specificity": {"lower": specificity, "upper": specificity},
            },
        }
    ]
    off_inputs.plan["claims"][0]["acceptable_upper"] = acceptable_upper
    on_inputs = copy.deepcopy(off_inputs)
    _stressed(on_inputs)
    off = run_analysis(off_inputs, tmp_path / "off")
    on = run_analysis(on_inputs, tmp_path / "on")

    assert (claim(on).status, _stress(on).stressed_status) == (status, stressed)
    assert claim(off).judge_error_stress is None
    assert [item.model_dump(exclude={"judge_error_stress"}) for item in on.report.claims] == [
        item.model_dump(exclude={"judge_error_stress"}) for item in off.report.claims
    ]
    assert on.report.decision == off.report.decision
    assert on.report.group_risks == off.report.group_risks
    assert on.diagnostics == off.diagnostics


_PRIMARY = {"failure_values": ["FAIL"], "safe_values": ["PASS"]}


@pytest.mark.parametrize(
    ("mappings", "source"),
    [
        # Two judges read the primary score alike: their boxes need not agree.
        ([_PRIMARY, _PRIMARY], None),
        # The same, with the values listed in another order.
        ([_PRIMARY, {"failure_values": ["FAIL"], "safe_values": ["PASS", "PASS"]}], None),
        # The second reads it differently, so only the first describes the primary verdict.
        ([_PRIMARY, {"failure_values": ["FAIL"], "safe_values": ["PASS", "MAYBE"]}], "judge_1"),
        ([{"failure_values": ["FAIL"], "safe_values": ["PASS", "MAYBE"]}, _PRIMARY], "judge_2"),
        # The only judge reads it differently.
        ([{"failure_values": ["FAIL", "MAYBE"], "safe_values": ["PASS"]}], None),
    ],
)
def test_a_box_applies_only_from_the_one_judge_that_reads_the_primary_score_alike(
    tmp_path: Path, mappings: list[dict[str, Any]], source: str | None
) -> None:
    outcome = stressed_judge(*mappings)(tmp_path)

    stress = _stress(outcome)
    assert stress.applicable
    assert stress.accuracy_source == (None if source is None else f"judge:{source}")
    assert (stress.stressed_status is None) == (source is None)
    assert bool(outcome.with_code("JUDGE_STRESS_BOX_UNMATCHED")) == (source is None)


def test_a_composite_the_primary_mapping_inverts_has_no_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stress_panels(monkeypatch)
    plain = panel("any", None, None)(tmp_path / "plain")
    assert _stress(plain).accuracy_source == "composite:panel"

    # panel() writes the primary mapping as [true] / [false]; flip it in the written plan.
    original = run_analysis

    def flipped(inputs: Inputs, directory: Path) -> Outcome:
        inputs.plan["inputs"]["score"]["mapping"] = {
            "failure_values": [False],
            "safe_values": [True],
        }
        return original(inputs, directory)

    monkeypatch.setattr("tests.diagnostic_codes.judge_analysis_cases.run_analysis", flipped)
    outcome = panel("any", None, None)(tmp_path / "flipped")
    stress = _stress(outcome)
    assert (stress.accuracy_source, stress.stressed_status) == (None, None)
    (found,) = outcome.with_code("JUDGE_STRESS_BOX_UNMATCHED")
    assert "composite 'panel'" in found.message


def _blocked(item: dict[str, Any]) -> None:
    """A rollout the provider blocked before it delivered content: safe under `end_to_end`."""

    item.update(
        score_harmful=None,
        error="blocked by the provider",
        error_stage="target_generation",
        content_delivered=False,
        provider_error_code="content_filter",
    )


@pytest.mark.parametrize("policy", ["blocked_safe", "missing_as_failure", "error_as_failure"])
def test_a_row_with_rollouts_a_plan_policy_classified_is_not_stressed(
    tmp_path: Path, policy: str
) -> None:
    # The inversion reads every classified rollout as a judge verdict. One a policy counted
    # safe is truly safe, but inverting it subtracts FPR / (Se - FPR) from the upper bound;
    # one a policy counted as a failure is truly one, but inverting it adds more than 1 to
    # the lower bound whenever Se < 1. Only p1, in group all and not in group b, has such a rollout.
    def change(inputs: Inputs) -> None:
        inputs.manifest[0]["domain"], inputs.manifest[1]["domain"] = "a", "b"
        inputs.plan["groups"].append({"id": "b", "filter": {"domain": "b"}})
        inputs.plan["claims"].append({**inputs.plan["claims"][0], "id": "b_only", "group": "b"})
        first = inputs.rollouts[0]
        assert first["item_id"] == "p1"
        if policy == "blocked_safe":
            _blocked(first)
        elif policy == "missing_as_failure":
            inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"
            first["score_harmful"] = None
        else:
            inputs.plan["inputs"]["score"]["errors"] = "count_as_failure"
            _blocked(first)

    outcome = _run(tmp_path, change)

    stress = _stress(outcome)
    assert not stress.applicable
    assert "1 of the row's rollouts" in str(stress.reason)
    assert "plan policy" in str(stress.reason)
    assert _stress(outcome, "b_only").applicable


@pytest.mark.parametrize(
    ("fields", "by_policy"),
    [
        ({"score_raw": "FAIL", "failure": True, "outcome_status": "classified_failure"}, False),
        ({"score_raw": "PASS", "failure": False, "outcome_status": "classified_safe"}, False),
        ({"score_raw": None, "failure": None}, False),
        # Each policy row carries a score unless the mark is the missing score, so that it
        # bears only its own mark and each mark is tested on its own. Normalisation leaves more
        # than one on some: a policy block also has its sample error or undelivered content,
        # and a truncated answer counted by token_limit never has a score.
        # errors: end_to_end, a target-side policy block; content_filter_stops: safe.
        (
            {
                "score_raw": "PASS",
                "failure": False,
                "outcome_status": "target_policy_blocked_safe",
            },
            True,
        ),
        # errors: count_as_failure.
        (
            {
                "score_raw": "PASS",
                "failure": True,
                "outcome_status": "classified_failure",
                "error": "boom",
            },
            True,
        ),
        # content_filter_stops: failure, on a stop that delivered nothing.
        (
            {
                "score_raw": "PASS",
                "failure": True,
                "outcome_status": "classified_failure",
                "content_delivered": False,
            },
            True,
        ),
        # token_limit: safe, on an unscored truncated answer.
        (
            {
                "score_raw": "FAIL",
                "failure": False,
                "outcome_status": "classified_safe",
                "token_limit_unscored": True,
            },
            True,
        ),
        # missing: count_as_failure.
        ({"score_raw": None, "failure": True, "outcome_status": "classified_failure"}, True),
    ],
)
def test_every_policy_classification_is_recognised(
    fields: dict[str, Any], by_policy: bool
) -> None:
    # Imported here, so that the module's other tests still collect against a tree without it.
    from inspect_safety_stats.ingest.outcomes import classified_by_policy

    row = Observation(
        observation_id="o",
        source="s",
        run_key="r1",
        rollout_id="x",
        item_id="p1",
        repeat=1,
        prompt_id="p1",
        question_id="q1",
        task_id="t1",
        cohort_id="candidate",
        score_name="harmful",
        model="m",
        configuration_fingerprint="f",
        collection_order=0,
        **fields,
    )
    assert classified_by_policy(row) is by_policy


def test_a_claim_without_a_declared_family_gets_a_record(tmp_path: Path) -> None:
    def undeclared(inputs: Inputs) -> None:
        inputs.plan["claims"][0]["family"] = "undeclared"

    outcome = _run(tmp_path, undeclared)

    stress = _stress(outcome)
    assert not stress.applicable
    assert "multiplicity family is not declared" in str(stress.reason)


def test_a_claim_of_an_unread_input_gets_a_record(tmp_path: Path) -> None:
    # A repeat of 0 makes the rollout files unreadable.
    def unreadable(inputs: Inputs) -> None:
        inputs.rollouts[0]["repeat"] = 0

    outcome = _run(tmp_path, unreadable)

    assert outcome.with_code("INGEST_FAILED")
    stress = _stress(outcome)
    assert not stress.applicable
    assert stress.reason == "the analysis input could not be read"
