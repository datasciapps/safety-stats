"""Cases for the codes of judge calibration, judge corrections and the analysis's intervals."""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.diagnostic_codes.harness import (
    Case,
    Inputs,
    Outcome,
    Run,
    analysis,
    base_inputs,
    claim,
    rollout,
    run_analysis,
)

# --- Judge calibration -------------------------------------------------------------------
#
# A judged run: two prompts of twenty rollouts, the judge flagging half of p2's. Human labels
# name observations by their identifier, a hash of the rollout, so a judged run first analyses
# its inputs without the judge sections to learn the identifiers, then writes the label files.

ROLLOUTS = 20
Labels = list[tuple[str, str]]


def _judged_rollouts(inputs: Inputs) -> None:
    for row in inputs.manifest:
        row["expected_rollouts"] = ROLLOUTS
    inputs.rollouts = [rollout("p1", repeat) for repeat in range(1, ROLLOUTS + 1)] + [
        rollout("p2", repeat, "FAIL" if repeat <= ROLLOUTS // 2 else "PASS")
        for repeat in range(1, ROLLOUTS + 1)
    ]


def judged(
    change: Callable[[Inputs, list[Any]], None],
    data: Callable[[Inputs], None] | None = None,
) -> Run:
    """A run on the judged rollouts, changed by ``change`` given the probed observations."""

    def run(directory: Path) -> Outcome:
        inputs = base_inputs()
        _judged_rollouts(inputs)
        if data is not None:
            data(inputs)
        probe = run_analysis(copy.deepcopy(inputs), directory / "probe")
        change(inputs, list(probe.report._observations))
        return run_analysis(inputs, directory)

    return run


def _truth(observation: Any) -> str:
    """A label that agrees with the judge."""

    return "unsafe" if observation.failure else "safe"


def census(observations: list[Any]) -> Labels:
    """Every classified rollout labelled, in agreement with the judge."""

    return [
        (item.observation_id, _truth(item)) for item in observations if item.failure is not None
    ]


def _csv(header: str, rows: Sequence[tuple[str, ...]]) -> str:
    return header + "\n" + "".join(",".join(row) + "\n" for row in rows)


def validated(inputs: Inputs, labels: Labels, **spec: Any) -> None:
    """Declare a validated calibration on ``labels``, in a judge family of its own."""

    inputs.plan["inputs"]["judge_calibration"] = {
        "mode": "validated",
        "validation_manifest": "labels.csv",
        "family": "judge",
        **spec,
    }
    inputs.plan["multiplicity"]["families"].append(
        {"id": "judge", "purpose": "simultaneous_intervals", "method": "bonferroni", "alpha": 0.05}
    )
    inputs.files["labels.csv"] = _csv("observation_id,human_label", labels)


def with_labels(labels: Callable[[list[Any]], Labels], **spec: Any) -> Callable[..., None]:
    def change(inputs: Inputs, observations: list[Any]) -> None:
        validated(inputs, labels(observations), **spec)

    return change


def _row(outcome: Outcome, measure: str = "mean", group: str = "all") -> Any:
    (found,) = [
        item
        for item in outcome.report.group_risks
        if item.measure_id == measure and item.group_id == group and item.cohort_id == "candidate"
    ]
    return found


def _interval(outcome: Outcome, interval_id: str, measure: str = "mean") -> Any:
    found = [item for item in _row(outcome, measure).intervals if item.interval_id == interval_id]
    return found[0] if found else None


CORRECTED = "judge_corrected_validated"


def _some(observations: list[Any], judge_unsafe: int, judge_safe: int) -> Labels:
    """The first few rollouts of each judge stratum, labelled as the judge says."""

    unsafe = [item for item in observations if item.failure is True][:judge_unsafe]
    safe = [item for item in observations if item.failure is False][:judge_safe]
    return [(item.observation_id, _truth(item)) for item in unsafe + safe]


def _uninformative(triggered: Outcome, control: Outcome) -> None:
    # Withheld: no validated correction, so the claim is INCONCLUSIVE rather than decided.
    assert _interval(triggered, CORRECTED) is None
    assert _interval(control, CORRECTED) is not None
    assert claim(triggered).status == "INCONCLUSIVE"
    assert claim(control).status == "PASS"


def _corrected_claims(triggered: Outcome, control: Outcome) -> None:
    # Adjusted: the claim is decided on the corrected bounds, at the union of the claim's and
    # the calibration's budgets, with no point estimate.
    decided, plain = claim(triggered), claim(control)
    assert decided.method == "validated_judge_misclassification_correction"
    assert plain.method != decided.method
    assert decided.estimate is None and plain.estimate is not None
    assert decided.alpha == pytest.approx(0.1, abs=1e-12)
    assert plain.alpha == pytest.approx(0.05, abs=1e-12)
    corrected = _interval(triggered, CORRECTED)
    assert (decided.lower, decided.upper) == (corrected.lower, corrected.upper)
    assert decided.status == "PASS"


def _worst_claim(inputs: Inputs) -> None:
    inputs.plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
    inputs.plan["claims"][0]["measure"] = "worst"


def _with_worst(inputs: Inputs) -> None:
    inputs.plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})


def _correction_unavailable(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the claim on the maximum has only an uncertified correction, so it is
    # INCONCLUSIVE; the same claim on the mean is decided on its certified one.
    (found,) = triggered.with_code("JUDGE_CORRECTION_UNAVAILABLE")
    assert found.details["claim_ids"] == ["low"]
    assert claim(triggered).status == "INCONCLUSIVE"
    assert _interval(triggered, CORRECTED, "worst").eligibility == "diagnostic"
    assert claim(control).status == "PASS"


def _only_judge_safe(observations: list[Any]) -> Labels:
    return [item for item in census(observations) if item[1] == "safe"]


def _completed(triggered: Outcome, control: Outcome) -> None:
    # Adjusted: the unlabelled judge-unsafe stratum is completed as [0, 1], which widens the
    # correction.
    wide, narrow = _interval(triggered, CORRECTED), _interval(control, CORRECTED)
    assert wide.lower < narrow.lower
    assert wide.upper >= narrow.upper
    assert any("has no human label, so its range is [0, 1]" in item for item in wide.assumptions)


def _no_judge_failures(inputs: Inputs) -> None:
    inputs.rollouts = [{**item, "score_harmful": "PASS"} for item in inputs.rollouts]


def _stratum_empty(triggered: Outcome, control: Outcome) -> None:
    # Withheld: no correction for the cohort, so its claim is INCONCLUSIVE.
    assert _interval(triggered, CORRECTED) is None
    assert claim(triggered).status == "INCONCLUSIVE"
    assert _interval(control, CORRECTED) is not None


def _unmatched(observations: list[Any]) -> Labels:
    return [*census(observations), ("not-a-rollout", "safe")]


def _bad_label(observations: list[Any]) -> Labels:
    labels = census(observations)
    return [(labels[0][0], "maybe"), *labels[1:]]


def _cannot_pass(triggered: Outcome, control: Outcome) -> None:
    # Invalid: no claim can pass.
    assert claim(triggered).status != "PASS"
    assert claim(control).status == "PASS"


def _no_correction(triggered: Outcome, control: Outcome) -> None:
    # Invalid: no validated correction is reported, and the claim cannot pass.
    assert _interval(triggered, CORRECTED) is None
    assert _interval(control, CORRECTED) is not None
    assert claim(triggered).status != "PASS"


def _domains(inputs: Inputs) -> None:
    """p1 and p2 in domains a and b, and a group of domain b."""

    inputs.manifest[0]["domain"], inputs.manifest[1]["domain"] = "a", "b"
    inputs.plan["groups"].append({"id": "b", "filter": {"domain": "b"}})


def _all(observations: list[Any]) -> Labels:
    return [(item.observation_id, _truth(item)) for item in observations]


def _p2_only(observations: list[Any]) -> Labels:
    return [
        label
        for item, label in zip(observations, _all(observations), strict=True)
        if item.prompt_id == "p2"
    ]


def _results_unchanged(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the labels outside the scoped group calibrate nothing, so every row and
    # claim is what it is without them.
    assert triggered.report.group_risks == control.report.group_risks
    assert triggered.report.claims == control.report.claims
    assert _interval(triggered, CORRECTED) is None  # group all is out of scope


PUBLISHED = {
    "name": "published",
    "source": "https://example.org/published",
    "judge": "judge-model",
    "rubric": "rubric-v1",
    "human_unsafe": 100,
    "judge_flagged_unsafe": 95,
}


def _transported(inputs: Inputs, _: list[Any]) -> None:
    inputs.plan["inputs"]["judge_calibration"] = {"mode": "transported", "published_set": PUBLISHED}


def _transported_effect(triggered: Outcome, control: Outcome) -> None:
    # Adjusted: the row gains a transported sensitivity interval with a lower end of zero,
    # which decides no claim.
    interval = _interval(triggered, "judge_corrected_transported")
    assert interval.lower == 0.0
    assert (interval.role, interval.eligibility) == ("sensitivity", "diagnostic")
    assert interval.upper > _row(triggered).upper
    assert _interval(control, "judge_corrected_transported") is None
    assert claim(triggered) == claim(control)


def _strata(observations: list[Any], drop: int = 0, extra: bool = False) -> str:
    """Every classified rollout's declared stratum, its prompt."""

    rows = [
        (item.observation_id, item.prompt_id) for item in observations if item.failure is not None
    ][drop:]
    if extra:
        rows.append(("not-a-rollout", "p1"))
    return _csv("observation_id,stratum", rows)


def stratified(
    labels: Callable[[list[Any]], Labels] = census, drop: int = 0, extra: bool = False
) -> Callable[..., None]:
    def change(inputs: Inputs, observations: list[Any]) -> None:
        validated(
            inputs,
            labels(observations),
            sampling="stratified_without_replacement",
            validation_strata="strata.csv",
        )
        inputs.files["strata.csv"] = _strata(observations, drop, extra)

    return change


def _stratified_effect(triggered: Outcome, control: Outcome) -> None:
    # Adjusted: exact hypergeometric bounds over the fine strata, and no delta-method reference.
    corrected = _interval(triggered, CORRECTED)
    assert any("hypergeometric" in item for item in corrected.assumptions)
    assert _interval(triggered, CORRECTED + "_delta") is None
    assert _interval(control, CORRECTED + "_delta") is not None


def _unlabelled_p1(observations: list[Any]) -> Labels:
    """Every label but those of p1, whose rollouts are all judge-safe."""

    return [
        label
        for item, label in zip(observations, _all(observations), strict=True)
        if item.prompt_id != "p1"
    ]


def _fine_completed(triggered: Outcome, control: Outcome) -> None:
    # Adjusted: p1's unlabelled fine stratum is completed from none to all of its rollouts,
    # which widens the correction.
    wide, narrow = _interval(triggered, CORRECTED), _interval(control, CORRECTED)
    assert wide.upper > narrow.upper
    (found,) = triggered.with_code("JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED")
    assert found.details["fine_strata"] == [
        {"judge_stratum": "judge_safe", "stratum": "p1", "stratum_size": ROLLOUTS}
    ]


def protocol(skip_first: bool) -> Callable[..., None]:
    """Labels drawn from a queue of every classified rollout, skipping its first row or not."""

    def change(inputs: Inputs, observations: list[Any]) -> None:
        queue = census(observations)
        validated(
            inputs,
            queue[1:] if skip_first else queue,
            protocol={"queue": "queue.csv", "stopping": "label_independent"},
        )
        inputs.files["queue.csv"] = _csv("observation_id", [(key,) for key, _ in queue])

    return change


def _p3(classified: int) -> Callable[[Inputs], None]:
    """A third prompt whose rollouts have no score, but for the first ``classified``."""

    def data(inputs: Inputs) -> None:
        inputs.manifest.append(
            {"prompt_id": "p3", "question_id": "q3", "task_id": "t1", "expected_rollouts": ROLLOUTS}
        )
        inputs.rollouts += [
            rollout("p3", repeat, "PASS" if repeat <= classified else None)
            for repeat in range(1, ROLLOUTS + 1)
        ]

    return data


def _no_lang_reiczigel(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the row has no Lang-Reiczigel limits, but its exact correction is reported.
    limits = CORRECTED + "_lang_reiczigel"
    assert _interval(triggered, limits) is None
    assert _interval(control, limits) is not None
    assert _interval(triggered, CORRECTED) is not None


def flips(manifest: Callable[[list[Any]], str] | None) -> Callable[..., None]:
    """Declare a flip manifest of the judge-positive rollouts rescored in wrapper w."""

    def change(inputs: Inputs, observations: list[Any]) -> None:
        if manifest is None:
            return
        inputs.plan["inputs"]["judge_flip"] = {
            "manifest": "flips.csv",
            "wrappers": ["w"],
            "warn_above": 0.2,
        }
        inputs.files["flips.csv"] = manifest(observations)

    return change


def _flip_rows(observations: list[Any], flipped: int, verdict: str = "unsafe") -> str:
    positive = [item.observation_id for item in observations if item.failure is True]
    return _csv(
        "observation_id,wrapper,verdict",
        [(key, "w", "safe" if index < flipped else verdict) for index, key in enumerate(positive)],
    )


def _flip_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the rate and its interval are reported, and no claim or row changes.
    (found,) = triggered.with_code("JUDGE_FLIP_RATE")
    assert found.severity == "WARNING"
    rate = found.details["any_wrapper"]
    assert (rate["flipped"], rate["rescored"]) == (5, ROLLOUTS // 2)
    assert rate["lower"] < 0.5 < rate["upper"]
    assert triggered.report.claims == control.report.claims
    assert triggered.report.group_risks == control.report.group_risks


def _baseline(inputs: Inputs) -> None:
    """A second cohort, baseline, whose run r2 repeats the candidate's rollouts."""

    inputs.plan["cohorts"]["baseline"] = {"match": {"model": "b"}}
    inputs.runs.append({**inputs.runs[0], "run_key": "r2", "model": "b"})
    inputs.rollouts += [{**item, "run_key": "r2"} for item in inputs.rollouts]


def comparison(method: str) -> Callable[[Inputs], None]:
    def data(inputs: Inputs) -> None:
        _baseline(inputs)
        inputs.plan["multiplicity"]["families"].append(
            {
                "id": "comparisons",
                "purpose": "simultaneous_intervals",
                "method": "bonferroni",
                "alpha": 0.05,
            }
        )
        inputs.plan["comparisons"] = [
            {
                "id": "versus",
                "candidate": "candidate",
                "baseline": "baseline",
                "group": "all",
                "measure": "mean",
                "estimand": "risk_difference",
                "method": method,
                "family": "comparisons",
                "independence": {"primary": "rollout"},
            }
        ]
        inputs.plan.setdefault("design", {})["independence_profiles"] = {
            "rollout": {"block_by": ["rollout"]}
        }

    return data


def _comparison_uncorrected(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the comparison stays on the judge-reported scale; the group rows are corrected.
    def corrected(outcome: Outcome) -> list[Any]:
        (row,) = [item for item in outcome.report.comparisons if item.level is None]
        return [item for item in row.intervals if item.interval_id == CORRECTED]

    assert corrected(triggered) == []
    assert len(corrected(control)) == 1
    assert _interval(triggered, CORRECTED) is not None


def _tail_target(target: float | None) -> Callable[[Inputs], None]:
    def data(inputs: Inputs) -> None:
        inputs.plan["outputs"]["risk_measures"].append({"id": "worst", "type": "maximum"})
        if target is not None:
            inputs.plan["report"] = {"precision_target": target}

    return data


def _tail_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the tail measure is reported as computed, the same as under a looser target,
    # and even a prompt with no failures is bounded above the 0.05 target.
    (found,) = triggered.with_code("TAIL_MEASURES_UNINFORMATIVE")
    assert found.details["measures"] == ["worst"]
    assert found.details["median_rollouts_per_prompt"] == ROLLOUTS
    assert found.details["zero_failure_prompt_upper"] > 0.05
    assert triggered.report.group_risks == control.report.group_risks
    assert triggered.report.prompt_estimates == control.report.prompt_estimates
    assert triggered.report.claims == control.report.claims


def panel(rule: str, groups_a: list[str] | None, groups_b: list[str] | None) -> Run:
    """A composite primary of judges a and b, each with declared, optionally scoped, accuracy."""

    def change(inputs: Inputs) -> None:
        _domains(inputs)

        def judge(name: str, groups: list[str] | None) -> dict[str, Any]:
            accuracy: dict[str, Any] = {
                "sensitivity": {"lower": 0.8, "upper": 0.95},
                "specificity": {"lower": 0.9, "upper": 0.99},
            }
            if groups is not None:
                accuracy["groups"] = groups
            return {
                "id": f"judge_{name}",
                "score": {
                    "name": f"harmful_{name}",
                    "mapping": {"failure_values": ["FAIL"], "safe_values": ["PASS"]},
                },
                "accuracy": accuracy,
            }

        spec = inputs.plan["inputs"]
        spec["score"] = {
            "name": "panel",
            "mapping": {"failure_values": [True], "safe_values": [False]},
        }
        spec["judges"] = [judge("a", groups_a), judge("b", groups_b)]
        spec["composites"] = [{"id": "panel", "rule": rule, "judges": ["judge_a", "judge_b"]}]
        for item in inputs.rollouts:
            item["score_harmful_a"] = item["score_harmful_b"] = item.pop("score_harmful")

    def run(directory: Path) -> Outcome:
        inputs = base_inputs()
        change(inputs)
        return run_analysis(inputs, directory)

    return run


def stressed_judge(*mappings: dict[str, Any]) -> Run:
    """The judge-error stress, with one judge per mapping reading the primary score.

    Every judge declares the same accuracy box.
    """

    def change(inputs: Inputs) -> None:
        inputs.plan.setdefault("report", {})["judge_error_stress"] = True
        inputs.plan["inputs"]["judges"] = [
            {
                "id": f"judge_{index}",
                "score": {"name": "harmful", "mapping": mapping},
                "accuracy": {
                    "sensitivity": {"lower": 0.8, "upper": 0.95},
                    "specificity": {"lower": 0.9, "upper": 0.99},
                },
            }
            for index, mapping in enumerate(mappings, start=1)
        ]

    return analysis(change)


def _stress_box_withheld(triggered: Outcome, control: Outcome) -> None:
    # Withheld: the claim keeps its tipping lines but gets no verdict over the box.
    stressed, plain = claim(triggered).judge_error_stress, claim(control).judge_error_stress
    assert stressed.applicable and stressed.pass_sensitivity_at_perfect_specificity is not None
    assert (stressed.accuracy_source, stressed.stressed_status) == (None, None)
    assert plain.accuracy_source == "judge:judge_1" and plain.stressed_status is not None
    assert claim(triggered).status == claim(control).status


PANEL = "judge_corrected_panel_derived"


def _panel_rows(outcome: Outcome) -> list[str]:
    return sorted(
        item.group_id
        for item in outcome.report.group_risks
        if any(interval.interval_id == PANEL for interval in item.intervals)
    )


def _panel_withheld(triggered: Outcome, control: Outcome) -> None:
    # Withheld: no row gets a panel-derived correction.
    assert _panel_rows(triggered) == []
    assert _panel_rows(control) != []


CASES = [
    Case(
        # Two labels per judge stratum cannot show that rho_1 exceeds rho_0.
        code="JUDGE_CALIBRATION_UNINFORMATIVE",
        trigger=judged(with_labels(lambda obs: _some(obs, 2, 2))),
        control=judged(with_labels(census)),
        effect=_uninformative,
    ),
    Case(
        # A certifying validated correction retargets the claim to true risk.
        code="JUDGE_CORRECTED_CLAIMS",
        trigger=judged(with_labels(census)),
        control=judged(lambda inputs, _: None),
        effect=_corrected_claims,
    ),
    Case(
        # A claim on a maximum has no certifying validated correction.
        code="JUDGE_CORRECTION_UNAVAILABLE",
        trigger=judged(with_labels(census), _worst_claim),
        control=judged(with_labels(census), _with_worst),
        effect=_correction_unavailable,
    ),
    Case(
        # Only the judge-safe stratum is labelled.
        code="JUDGE_STRATUM_UNLABELLED_COMPLETED",
        trigger=judged(with_labels(_only_judge_safe)),
        control=judged(with_labels(census)),
        effect=_completed,
    ),
    Case(
        # The judge never says unsafe, so its unsafe stratum is empty.
        code="JUDGE_VALIDATION_STRATUM_EMPTY",
        trigger=judged(with_labels(census), _no_judge_failures),
        control=judged(with_labels(census)),
        effect=_stratum_empty,
    ),
    Case(
        # A label names no rollout of the run.
        code="JUDGE_VALIDATION_UNMATCHED",
        trigger=judged(with_labels(_unmatched)),
        control=judged(with_labels(census)),
        effect=_cannot_pass,
    ),
    Case(
        # A label is neither safe nor unsafe.
        code="JUDGE_VALIDATION_INVALID",
        trigger=judged(with_labels(_bad_label)),
        control=judged(with_labels(census)),
        effect=_no_correction,
    ),
    Case(
        # p1's labels lie outside group b, the calibration's only group.
        code="JUDGE_VALIDATION_OUT_OF_SCOPE",
        trigger=judged(with_labels(_all, groups=["b"]), _domains),
        control=judged(with_labels(_p2_only, groups=["b"]), _domains),
        effect=_results_unchanged,
    ),
    Case(
        # The sensitivity is carried over from a published set, not measured on this run.
        code="JUDGE_CALIBRATION_TRANSPORTED",
        trigger=judged(_transported),
        control=judged(lambda inputs, _: None),
        effect=_transported_effect,
    ),
    Case(
        # The labels are declared a stratified sample within fine strata.
        code="JUDGE_CALIBRATION_STRATIFIED",
        trigger=judged(stratified()),
        control=judged(with_labels(census)),
        effect=_stratified_effect,
    ),
    Case(
        # p1's fine stratum of the judge-safe stratum has no label.
        code="JUDGE_FINE_STRATUM_UNLABELLED_COMPLETED",
        trigger=judged(stratified(_unlabelled_p1)),
        control=judged(stratified()),
        effect=_fine_completed,
    ),
    Case(
        # The strata file leaves one classified rollout out.
        code="JUDGE_VALIDATION_STRATA_INCOMPLETE",
        trigger=judged(stratified(drop=1)),
        control=judged(stratified()),
        effect=_no_correction,
    ),
    Case(
        # The strata file names a rollout the run does not have.
        code="JUDGE_VALIDATION_STRATA_UNMATCHED",
        trigger=judged(stratified(extra=True)),
        control=judged(stratified()),
        effect=_no_correction,
    ),
    Case(
        # The labels skip the first row of the declared labelling queue.
        code="JUDGE_LABELLING_PROTOCOL_VIOLATED",
        trigger=judged(protocol(skip_first=True)),
        control=judged(protocol(skip_first=False)),
        effect=_no_correction,
    ),
    Case(
        # Prompt p3 of the row has no classified rollout.
        code="JUDGE_LANG_REICZIGEL_UNAVAILABLE",
        trigger=judged(with_labels(census, sampling="simple_random"), _p3(0)),
        control=judged(with_labels(census, sampling="simple_random"), _p3(1)),
        effect=_no_lang_reiczigel,
    ),
    Case(
        # Half the rescored judge-positive completions flip to safe in a wrapper. The rate is
        # reported whenever a flip manifest is declared, so the control declares none.
        code="JUDGE_FLIP_RATE",
        trigger=judged(flips(lambda obs: _flip_rows(obs, 5))),
        control=judged(flips(None)),
        effect=_flip_recorded,
    ),
    Case(
        # A rescored verdict is neither safe nor unsafe.
        code="JUDGE_FLIP_INVALID",
        trigger=judged(flips(lambda obs: _flip_rows(obs, 0, verdict="maybe"))),
        control=judged(flips(lambda obs: _flip_rows(obs, 0))),
        effect=_cannot_pass,
    ),
]
CASES += [
    Case(
        # The comparison's primary interval is a contrast bound without arm bounds.
        code="JUDGE_CORRECTION_UNAVAILABLE_FOR_COMPARISON",
        trigger=judged(with_labels(census), comparison("weighted_empirical_bernstein")),
        control=judged(with_labels(census), comparison("weighted_kl_chernoff")),
        effect=_comparison_uncorrected,
    ),
    Case(
        # Twenty rollouts per prompt cannot bound a failure-free prompt below 0.05.
        code="TAIL_MEASURES_UNINFORMATIVE",
        trigger=judged(lambda inputs, _: None, _tail_target(None)),
        control=judged(lambda inputs, _: None, _tail_target(0.25)),
        effect=_tail_recorded,
    ),
    Case(
        # A majority composite has no closed-form accuracy box.
        code="PANEL_CORRECTION_UNSUPPORTED_RULE",
        trigger=panel("majority", None, None),
        control=panel("any", None, None),
        effect=_panel_withheld,
    ),
    Case(
        # Judge a vouches for group all only, judge b for group b only.
        code="PANEL_CORRECTION_NO_COMMON_GROUP",
        trigger=panel("any", ["all"], ["b"]),
        control=panel("any", ["all", "b"], ["b"]),
        effect=_panel_withheld,
    ),
    Case(
        # The judge with declared accuracy reads the primary score, but counts a third value
        # as safe that the primary mapping does not know.
        code="JUDGE_STRESS_BOX_UNMATCHED",
        trigger=stressed_judge({"failure_values": ["FAIL"], "safe_values": ["PASS", "MAYBE"]}),
        control=stressed_judge({"failure_values": ["FAIL"], "safe_values": ["PASS"]}),
        effect=_stress_box_withheld,
    ),
]


# --- Anytime-valid bounds ----------------------------------------------------------------
#
# The bound reads the random order Inspect recorded for the run, which rollout files do not
# carry, so these runs write a real Inspect log of the mock model in a seeded shuffled order.

SHUFFLED = 10


def _anytime(bounds: bool) -> Run:
    def run(directory: Path) -> Outcome:
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.scorer import Score, Target, scorer
        from inspect_ai.solver import TaskState, generate

        from inspect_safety_stats import analyze
        from tests.diagnostic_codes.harness import write_inputs

        @scorer(metrics=[])
        def harmful():  # type: ignore[no-untyped-def]
            async def score(state: TaskState, target: Target) -> Score:
                return Score(value="FAIL" if state.sample_id == "p0" else "PASS")

            return score

        inputs = base_inputs()
        inputs.manifest = [
            {"prompt_id": f"p{i}", "question_id": f"q{i}", "task_id": "t1", "expected_rollouts": 1}
            for i in range(SHUFFLED)
        ]
        inputs.plan["cohorts"] = {"candidate": {"match": {"model": "mockllm/model"}}}
        inputs.plan["design"] = {"independence_profiles": {"prompt": {"block_by": ["prompt_id"]}}}
        if bounds:
            inputs.plan["outputs"]["anytime_bounds"] = {
                "independence_profile": "prompt",
                "family": "claims",
            }
        directory.mkdir(parents=True)
        # Inspect drops the drive of a log on another drive than the working directory (#35).
        with contextlib.chdir(directory):
            task = Task(
                dataset=MemoryDataset(
                    [Sample(id=f"p{i}", input=f"q{i}") for i in range(SHUFFLED)]
                ),
                solver=generate(),
                scorer=harmful(),
            )
            (log,) = inspect_eval(
                task,
                model="mockllm/model",
                log_dir=str(directory / "logs"),
                sample_shuffle=7,
                display="none",
            )
            plan = write_inputs(inputs, directory)
            report = analyze(plan=plan, logs=str(Path(log.location).resolve()))
        return Outcome(
            diagnostics=tuple(report.integrity.diagnostics),
            status=report.integrity.status,
            report=report,
            assumptions=frozenset(item.id for item in report.assumptions_used),
        )

    return run


def _anytime_recorded(triggered: Outcome, control: Outcome) -> None:
    # Recorded: the bound, and whether it certifies the mean claim, sit beside the claims,
    # whose status does not change.
    (found,) = triggered.with_code("ANYTIME_UPPER_BOUND")
    details = found.details
    assert (details["sample_shuffle"], details["population"]) == (7, SHUFFLED)
    assert details["claims"] == [
        {"claim_id": "low", "acceptable_upper": 0.9, "certified": details["upper"] <= 0.9}
    ]
    assert triggered.report.claims == control.report.claims


CASES += [
    Case(
        # The plan asks for the anytime-valid bound of a run in a recorded random order. The
        # bound is reported whenever the plan asks for it, so the control does not.
        code="ANYTIME_UPPER_BOUND",
        trigger=_anytime(bounds=True),
        control=_anytime(bounds=False),
        effect=_anytime_recorded,
    ),
]
