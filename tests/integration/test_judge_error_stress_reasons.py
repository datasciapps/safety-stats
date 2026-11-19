"""Where the judge-error stress does not apply, and that it never moves a claim, end to end.

Each case runs ``analyze`` through the diagnostic-code harness with the stress switched on.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

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
