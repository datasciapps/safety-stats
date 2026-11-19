"""Judge corrections of rows that hold rollouts a plan policy classified, end to end.

A plan policy, not the judge, classifies a provider's policy block counted safe, a sample error
or missing score counted as a failure, and a content-filter stop or unscored truncated answer
the plan classifies. Its classification is the truth the estimand defines, and the row's bounds
do not carry the share of such rollouts, so the correction of such a row credits the judge with
no false alarm at the upper bound and no missed failure at the lower.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from inspect_safety_stats.stats.judges import JudgeAccuracyBox, identified_risk_set
from tests.diagnostic_codes.harness import Inputs, Outcome, claim
from tests.diagnostic_codes.judge_analysis_cases import (
    DECLARED,
    RHO_0,
    RHO_1,
    assumed,
    census,
    comparison,
    judge_verdicts,
    judged,
    panel,
    policy_blocked,
    validated,
    with_labels,
)


def _domains(inputs: Inputs) -> None:
    """p1 in domain a and p2 in domain b, with a group b of p2 only."""

    inputs.manifest[0]["domain"], inputs.manifest[1]["domain"] = "a", "b"
    inputs.plan["groups"].append({"id": "b", "filter": {"domain": "b"}})


def _classify_first_by_policy(policy: str) -> Callable[[Inputs], None]:
    """Have a plan policy classify p1's first rollout: safe, or as a failure."""

    def data(inputs: Inputs) -> None:
        _domains(inputs)
        first = inputs.rollouts[0]
        assert first["item_id"] == "p1"
        if policy == "blocked_safe":
            policy_blocked(inputs)
        elif policy == "missing_as_failure":
            inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"
            first["score_harmful"] = None
        else:
            inputs.plan["inputs"]["score"]["errors"] = "count_as_failure"
            policy_blocked(inputs)

    return data


ABSENT = "None of the row's rollouts was classified by a plan policy"


def _counts_policy_rollouts(interval: Any) -> bool:
    """Whether the interval states the policy-classified rollouts of its row."""

    return any("rollouts were classified by a plan policy" in item for item in interval.assumptions)


def _assumes_none(interval: Any) -> None:
    """The interval was corrected over the box as given, and says what that assumes."""

    assert not _counts_policy_rollouts(interval)
    assert any(item.startswith(ABSENT) for item in interval.assumptions)


def _row(outcome: Outcome, group: str = "all", measure: str = "mean") -> Any:
    (found,) = [
        item
        for item in outcome.report.group_risks
        if (item.cohort_id, item.group_id, item.measure_id) == ("candidate", group, measure)
    ]
    return found


def _interval(row: Any, interval_id: str) -> Any:
    found = [item for item in row.intervals if item.interval_id == interval_id]
    return found[0] if found else None


@pytest.mark.parametrize("policy", ["blocked_safe", "missing_as_failure", "error_as_failure"])
def test_an_assumed_correction_credits_no_judge_error_on_a_row_a_policy_classified(
    tmp_path: Any, policy: str
) -> None:
    outcome = judged(assumed, _classify_first_by_policy(policy))(tmp_path)

    # Group all holds p1's policy-classified rollout: rho_1 may reach 1 at the upper bound and
    # rho_0 fall to 0 at the lower.
    row = _row(outcome)
    widened = _interval(row, DECLARED)
    assert widened.lower == pytest.approx(row.lower * RHO_1[0], abs=1e-12)
    assert widened.upper == pytest.approx(row.upper + (1 - row.upper) * RHO_0[1], abs=1e-12)
    safe, failures = (1, 0) if policy == "blocked_safe" else (0, 1)
    assert any(
        item.startswith(
            f"1 of the row's rollouts were classified by a plan policy, not by the judge "
            f"({safe} counted safe, {failures} counted as failures)"
        )
        for item in widened.assumptions
    )
    # Group b does not, so it keeps the declared box.
    row = _row(outcome, "b")
    plain = _interval(row, DECLARED)
    assert plain.lower == pytest.approx(
        row.lower * RHO_1[0] + (1 - row.lower) * RHO_0[0], abs=1e-12
    )
    assert plain.upper == pytest.approx(
        row.upper * RHO_1[1] + (1 - row.upper) * RHO_0[1], abs=1e-12
    )
    _assumes_none(plain)
    (found,) = outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")
    assert found.details == {"groups": ["all"], "policy_safe": safe, "policy_failures": failures}


def _box(interval: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """The rho_1 and rho_0 ranges a corrected interval states, to six significant digits."""

    def ranges(given: str) -> tuple[float, float]:
        (text,) = [item for item in interval.assumptions if item.startswith(given)]
        low, high = re.findall(r"\[([^,]+), ([^\]]+)\]", text)[0]
        return float(low), float(high)

    return (
        ranges("P(truly unsafe | judge said unsafe)"),
        ranges("P(truly unsafe | judge said safe)"),
    )


def test_a_validated_correction_of_such_a_row_still_certifies_and_has_no_reference(
    tmp_path: Any,
) -> None:
    labels = with_labels(judge_verdicts, sampling="simple_random")
    outcome = judged(labels, _classify_first_by_policy("blocked_safe"))(tmp_path)

    row = _row(outcome)
    corrected = _interval(row, "judge_corrected_validated")
    (rho_1, rho_0) = _box(corrected)
    assert corrected.lower == pytest.approx(row.lower * rho_1[0], rel=1e-5, abs=0)
    assert corrected.upper == pytest.approx(
        row.upper + (1 - row.upper) * rho_0[1], rel=1e-5, abs=0
    )
    # The widened box holds whatever the policy's share, so the correction still certifies,
    # and the claim is decided on it.
    assert (corrected.role, corrected.eligibility) == ("reference", "certifying")
    decided = claim(outcome)
    assert (decided.lower, decided.upper) == (corrected.lower, corrected.upper)
    # The asymptotic references would read the blocked rollout as a judge verdict.
    assert _interval(row, "judge_corrected_validated_delta") is None
    assert _interval(row, "judge_corrected_validated_lang_reiczigel") is None
    reasons = [
        item.message
        for item in outcome.with_code("JUDGE_LANG_REICZIGEL_UNAVAILABLE")
        if item.details.get("group_id") == "all"
    ]
    assert reasons and "classified by a plan policy" in reasons[0]
    # Group b holds no such rollout and keeps both.
    other = _row(outcome, "b")
    assert _interval(other, "judge_corrected_validated_delta") is not None
    plain = _interval(other, corrected.interval_id)
    _assumes_none(plain)


def test_each_arm_of_a_comparison_is_corrected_by_its_own_rollouts(tmp_path: Any) -> None:
    def data(inputs: Inputs) -> None:
        comparison("weighted_kl_chernoff")(inputs)
        # Only the candidate's run r1 has the blocked rollout; the baseline's r2 does not.
        policy_blocked(inputs)

    outcome = judged(assumed, data)(tmp_path)

    (result,) = [item for item in outcome.report.comparisons if item.level is None]
    primary = next(item for item in result.intervals if item.role == "primary")
    corrected = next(item for item in result.intervals if item.interval_id == DECLARED)
    assert corrected.candidate_lower == pytest.approx(
        primary.candidate_lower * RHO_1[0], abs=1e-12
    )
    assert corrected.candidate_upper == pytest.approx(
        primary.candidate_upper + (1 - primary.candidate_upper) * RHO_0[1], abs=1e-12
    )
    assert corrected.baseline_lower == pytest.approx(
        primary.baseline_lower * RHO_1[0] + (1 - primary.baseline_lower) * RHO_0[0], abs=1e-12
    )
    assert corrected.baseline_upper == pytest.approx(
        primary.baseline_upper * RHO_1[1] + (1 - primary.baseline_upper) * RHO_0[1], abs=1e-12
    )
    assert any(
        item.startswith("Candidate arm: 1 of the row's rollouts were classified by a plan policy")
        for item in corrected.assumptions
    )
    # The baseline's cohort holds no such rollout, and the plan sets no policy that classifies
    # rollouts, so its arm states nothing of them.
    assert not any(item.startswith("Baseline arm:") for item in corrected.assumptions)
    (found,) = outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")
    assert found.details["groups"] == ["all"]


def test_an_arm_without_such_rollouts_states_what_its_correction_assumes(tmp_path: Any) -> None:
    def data(inputs: Inputs) -> None:
        comparison("weighted_kl_chernoff")(inputs)
        policy_blocked(inputs)
        # A policy the plan sets: every cohort's rows may hold rollouts it classifies.
        inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"

    outcome = judged(assumed, data)(tmp_path)

    (result,) = [item for item in outcome.report.comparisons if item.level is None]
    corrected = next(item for item in result.intervals if item.interval_id == DECLARED)
    notes = [item for item in corrected.assumptions if item.startswith(("Candidate", "Baseline"))]
    assert notes[0].startswith("Candidate arm: 1 of the row's rollouts were classified")
    assert notes[1].startswith(f"Baseline arm: {ABSENT}")
    assert len(notes) == 2


# --- The judge strata leave such rollouts out -----------------------------------------------


def _strata(inputs: Inputs, observations: list[Any], policy_row: bool) -> None:
    """A stratified census of the classified rollouts, each one's stratum its prompt.

    With ``policy_row``, the labels and the strata file also take in the rollout the plan
    counts safe, as files drawn from every classified row of observations.csv do.
    """

    validated(
        inputs,
        census(observations) if policy_row else judge_verdicts(observations),
        sampling="stratified_within_judge_stratum",
        validation_strata="strata.csv",
    )
    rows = [
        f"{item.observation_id},{item.prompt_id}\n"
        for item in observations
        if item.failure is not None and (policy_row or item.error is None)
    ]
    inputs.files["strata.csv"] = "observation_id,stratum\n" + "".join(rows)


@pytest.mark.parametrize("sampling", ["simple_random", "stratified_within_judge_stratum"])
def test_the_judge_strata_hold_the_judge_verdicts_only(tmp_path: Any, sampling: str) -> None:
    def labels(inputs: Inputs, observations: list[Any]) -> None:
        if sampling == "simple_random":
            validated(inputs, census(observations), sampling="simple_random")
        else:
            _strata(inputs, observations, policy_row=True)

    outcome = judged(labels, policy_blocked)(tmp_path)

    # p1's 19 other rollouts and p2's 10 judge-safe ones; the blocked rollout's label is left
    # out with the rollout, so every judge-safe label agrees with the judge.
    cohort = outcome.report.provenance["judge_calibration"]["cohorts"]["candidate"]
    assert cohort["judge_safe"] == {"labelled": 29, "truly_unsafe": 0, "stratum_size": 29}
    assert cohort["judge_unsafe"] == {"labelled": 10, "truly_unsafe": 10, "stratum_size": 10}
    (found,) = outcome.with_code("JUDGE_VALIDATION_POLICY_CLASSIFIED")
    blocked = outcome.report._observations[0].observation_id
    assert found.details == {
        "labels": 1,
        "strata_rows": int(sampling != "simple_random"),
        "observation_ids": [blocked],
    }
    if sampling == "simple_random":
        # The truth strata the noisy-but-valid test reads leave it out too.
        truth = outcome.report.provenance["judge_calibration"]["truth_strata"]["candidate"]
        assert truth["truly_safe"] == {"labelled": 29, "judge_positive": 0}
    assert outcome.status != "invalid"


def test_a_strata_file_need_not_place_rollouts_a_policy_classified(tmp_path: Any) -> None:
    def labels(inputs: Inputs, observations: list[Any]) -> None:
        _strata(inputs, observations, policy_row=False)

    outcome = judged(labels, policy_blocked)(tmp_path)

    assert not outcome.with_code("JUDGE_VALIDATION_STRATA_INCOMPLETE")
    assert not outcome.with_code("JUDGE_VALIDATION_POLICY_CLASSIFIED")
    corrected = _interval(_row(outcome), "judge_corrected_validated")
    assert corrected is not None and corrected.eligibility == "certifying"


# --- The noisy-but-valid test refuses such a row --------------------------------------------


@pytest.mark.parametrize("blocked", [True, False])
def test_the_noisy_valid_test_does_not_read_policy_classifications_as_verdicts(
    tmp_path: Any, blocked: bool
) -> None:
    def noisy_valid(inputs: Inputs, observations: list[Any]) -> None:
        validated(inputs, judge_verdicts(observations), sampling="simple_random")
        inputs.plan["claims"][0]["method"] = "noisy_valid_test"
        # The test needs a certifying direct-mean bound with its own alpha.
        inputs.plan["multiplicity"]["families"].append(
            {"id": "direct", "purpose": "simultaneous_intervals", "method": "bonferroni",
             "alpha": 0.05}
        )
        inputs.plan["outputs"]["direct_mean_intervals"] = {
            "enabled": True,
            "method": "weighted_kl_chernoff",
            "family": "direct",
            "independence": {"primary": "rollout"},
        }
        inputs.plan.setdefault("design", {}).setdefault(
            "independence_profiles", {"rollout": {"block_by": ["rollout"]}}
        )

    outcome = judged(noisy_valid, policy_blocked if blocked else None)(tmp_path)

    decided = claim(outcome)
    assert decided.method == "noisy_valid_certification_test"
    refused = any("classified by a plan policy" in item for item in decided.reasons)
    assert refused is blocked
    if blocked:
        assert decided.status == "INCONCLUSIVE"
        assert (decided.lower, decided.upper) == (None, None)
    else:
        assert decided.upper is not None


# --- The panel-derived correction ----------------------------------------------------------


def _derived_bounds(interval: Any) -> tuple[float, float, float, float]:
    """The derived sensitivity and specificity box an interval states."""

    (text,) = [item for item in interval.assumptions if item.startswith("Derived sensitivity")]
    (se_low, se_high), (sp_low, sp_high) = re.findall(r"\[([^,]+), ([^\]]+)\]", text)
    return float(se_low), float(se_high), float(sp_low), float(sp_high)


def test_the_panel_derived_correction_credits_no_panel_error_on_a_row_a_policy_classified(
    tmp_path: Any,
) -> None:
    outcome = panel("any", None, None, policy_blocked)(tmp_path)

    # Group all holds the blocked rollout: the inversion takes specificity 1 at the upper
    # bound and sensitivity 1 at the lower.
    row = _row(outcome)
    widened = _interval(row, "judge_corrected_panel_derived")
    sensitivity, _, specificity, _ = _derived_bounds(widened)
    false_positive = 1 - specificity
    assert widened.upper == pytest.approx(min(1.0, row.upper / sensitivity), abs=1e-12)
    assert widened.lower == pytest.approx(
        max(0.0, (row.lower - false_positive) / specificity), abs=1e-12
    )
    assert any("not by the panel" in item for item in widened.assumptions)
    assert not widened.warnings
    # Group b, p2 only, is inverted over the derived box itself.
    row = _row(outcome, "b")
    plain = _interval(row, "judge_corrected_panel_derived")
    box = JudgeAccuracyBox(*_derived_bounds(plain))
    expected = identified_risk_set(row.lower, row.upper, box)
    assert (plain.lower, plain.upper) == pytest.approx((expected.lower, expected.upper), abs=0)
    _assumes_none(plain)
    (found,) = outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")
    assert found.details == {"groups": ["all"], "policy_safe": 1, "policy_failures": 0}



# --- A prompt of weight zero ---------------------------------------------------------------


def _weightless_p1_with_missing_scores(inputs: Inputs) -> None:
    """p1 weighted 0 and p2 weighted 1, with 15 of p1's 20 scores missing as failures.

    The maximum ignores weights, so p1, nearly all failures, sets the row's upper bound.
    """

    for row, weight in zip(inputs.manifest, (0, 1), strict=True):
        row["weight"] = weight
    inputs.plan["design"] = {"weights": {"strategy": "explicit", "column": "weight"}}
    inputs.plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
    inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"
    for item in inputs.rollouts[:15]:
        assert item["item_id"] == "p1"
        item["score_harmful"] = None


def test_a_prompt_of_weight_zero_that_a_policy_classified_still_widens_the_correction(
    tmp_path: Any,
) -> None:
    outcome = judged(assumed, _weightless_p1_with_missing_scores)(tmp_path)

    row = _row(outcome, measure="worst")
    widened = _interval(row, DECLARED)
    assert widened.upper == pytest.approx(row.upper + (1 - row.upper) * RHO_0[1], abs=1e-12)
    assert widened.lower == pytest.approx(row.lower * RHO_1[0], abs=1e-12)
    assert any(
        item.startswith("15 of the row's rollouts were classified by a plan policy")
        for item in widened.assumptions
    )


# --- A row whose sample holds none ---------------------------------------------------------


def _missing_as_failure(inputs: Inputs) -> None:
    _domains(inputs)
    inputs.plan["inputs"]["score"]["missing"] = "count_as_failure"


@pytest.mark.parametrize(
    ("data", "stated"), [(_domains, False), (_missing_as_failure, True)], ids=["default", "chosen"]
)
def test_a_row_whose_sample_holds_no_such_rollout_states_what_its_correction_assumes(
    tmp_path: Any, data: Callable[[Inputs], None], stated: bool
) -> None:
    # No rollout is classified by a policy in either run. The correction uses the box as given,
    # which assumes the population holds none either. A plan that sets a policy classifying
    # rollouts says so on every row; at the defaults, a cohort whose sample holds none does not.
    outcome = judged(assumed, data)(tmp_path)

    for group in ("all", "b"):
        interval = _interval(_row(outcome, group), DECLARED)
        assert any(item.startswith(ABSENT) for item in interval.assumptions) is stated
        assert not _counts_policy_rollouts(interval)
    assert not outcome.with_code("JUDGE_CORRECTION_POLICY_CLASSIFIED")

