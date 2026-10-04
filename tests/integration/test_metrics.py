from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Epochs, Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from inspect_safety_stats import analyze
from inspect_safety_stats.metrics import (
    failure_rate,
    failure_rate_lower,
    failure_rate_upper,
)
from inspect_safety_stats.scaffold import scaffold_plan
from inspect_safety_stats.stats import binomial_upper_bound

MODEL = "mockllm/model"


def _draw(sample_id: Any, epoch: int) -> float:
    digest = hashlib.sha256(f"{sample_id}:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@scorer(metrics=[accuracy()])
def varying_scorer(rates: dict[str, float], unknown: frozenset[str] = frozenset()):  # type: ignore[no-untyped-def]
    """FAIL on a deterministic, epoch-dependent share of each prompt's rollouts."""

    async def score(state: TaskState, target: Target) -> Score:
        sample = str(state.sample_id)
        if sample in unknown and state.epoch == 1:
            return Score(value="ERROR")
        failed = _draw(sample, state.epoch) < rates.get(sample, 0.0)
        return Score(value="FAIL" if failed else "PASS")

    return score


@scorer(metrics=[accuracy()])
def graded_scorer():  # type: ignore[no-untyped-def]
    """A 1-5 rubric grade that varies by prompt and epoch."""

    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1 + int(_draw(state.sample_id, state.epoch) * 5) % 5)

    return score


def _run(
    tmp_path: Path,
    scorer_instance: Any,
    metrics: list[Any],
    *,
    prompts: int = 10,
    epochs: int | Epochs = 12,
) -> tuple[dict[str, float], Path]:
    task = Task(
        name="metric_task",
        dataset=[Sample(input=f"q{index}", id=f"p{index}") for index in range(prompts)],
        solver=generate(),
        scorer=scorer_instance,
        metrics=metrics,
    )
    (log,) = inspect_eval(
        task, model=MODEL, epochs=epochs, log_dir=str(tmp_path / "logs"), display="none"
    )
    assert log.status == "success" and log.results is not None
    values = {
        name: float(item.value)
        for score in log.results.scores
        for name, item in score.metrics.items()
    }
    return values, tmp_path / "logs"


def _bounds(**mapping: Any) -> list[Any]:
    return [
        failure_rate(**mapping),
        failure_rate_lower(**mapping),
        failure_rate_upper(**mapping),
    ]


RATES = {"p0": 0.5, "p1": 0.25, "p2": 0.8, "p3": 0.1}


def test_metrics_match_the_analysis_of_the_same_log(tmp_path: Path) -> None:
    values, logs = _run(tmp_path, varying_scorer(RATES), _bounds(failure_values=["FAIL"]))
    plan = scaffold_plan(logs, tmp_path / "plan", failure_values=["FAIL"]).plan_path
    report = analyze(plan=plan, logs=str(logs))
    (risk,) = [
        item
        for item in report.group_risks
        if item.group_id == "all" and item.measure_id == "mean"
    ]
    # One model, one group: the analysis spends the 0.05 family on one two-sided interval,
    # exactly what the metrics report.
    assert values["failure_rate"] == pytest.approx(risk.estimate)
    assert values["failure_rate_lower"] == pytest.approx(risk.lower, abs=1e-9)
    assert values["failure_rate_upper"] == pytest.approx(risk.upper, abs=1e-9)
    assert values["failure_rate_lower"] < values["failure_rate"] < values["failure_rate_upper"]


def test_epoch_reducers_do_not_change_the_bound(tmp_path: Path) -> None:
    plain, _ = _run(tmp_path / "plain", varying_scorer(RATES), _bounds(failure_values=["FAIL"]))
    reduced, _ = _run(
        tmp_path / "reduced",
        varying_scorer(RATES),
        _bounds(failure_values=["FAIL"]),
        epochs=Epochs(12, "max"),
    )
    # A max reducer would turn every prompt with any failure into a failure; the metrics
    # still see each rollout.
    for name in ("failure_rate", "failure_rate_lower", "failure_rate_upper"):
        assert reduced[name] == pytest.approx(plain[name])


def test_zero_failures_give_the_exact_zero_failure_bound(tmp_path: Path) -> None:
    values, _ = _run(
        tmp_path, varying_scorer({}), _bounds(failure_values=["FAIL"]), prompts=10, epochs=30
    )
    assert values["failure_rate"] == 0.0
    assert values["failure_rate_lower"] == 0.0
    assert values["failure_rate_upper"] == pytest.approx(1 - 0.025 ** (1 / 300))
    assert values["failure_rate_upper"] == pytest.approx(
        binomial_upper_bound(0, 300, alpha=0.025), rel=1e-6
    )


def test_unclassified_values_widen_the_interval(tmp_path: Path) -> None:
    unknown = frozenset({"p5", "p6"})
    strict, _ = _run(
        tmp_path / "strict",
        varying_scorer(RATES, unknown),
        _bounds(failure_values=["FAIL"], safe_values=["PASS"]),
    )
    lenient, _ = _run(
        tmp_path / "lenient",
        varying_scorer(RATES, unknown),
        _bounds(failure_values=["FAIL"]),
    )
    # Without safe_values, ERROR counts as safe; with them it is unknown and widens the
    # upper bound but leaves the estimate and lower bound where completing it as safe puts
    # them.
    assert strict["failure_rate"] == pytest.approx(lenient["failure_rate"])
    assert strict["failure_rate_lower"] == pytest.approx(lenient["failure_rate_lower"])
    assert strict["failure_rate_upper"] > lenient["failure_rate_upper"]


def test_graded_scale_bounds_mean_harm(tmp_path: Path) -> None:
    high, _ = _run(tmp_path / "high", graded_scorer(), _bounds(harm_scale=[1, 5]))
    low, _ = _run(
        tmp_path / "low", graded_scorer(), _bounds(harm_scale=[1, 5], lower_is_harmful=True)
    )
    assert 0 < high["failure_rate_lower"] < high["failure_rate"] < high["failure_rate_upper"] < 1
    assert low["failure_rate"] == pytest.approx(1 - high["failure_rate"])
    # Reversing the scale mirrors the interval: the Bernoulli KL is symmetric.
    assert low["failure_rate_lower"] == pytest.approx(1 - high["failure_rate_upper"], abs=1e-9)
    assert low["failure_rate_upper"] == pytest.approx(1 - high["failure_rate_lower"], abs=1e-9)


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"failure_values": ["FAIL"], "harm_scale": [0, 1]},
        {"failure_values": []},
        {"failure_values": ["FAIL"], "safe_values": ["FAIL"]},
        {"harm_scale": [5, 1]},
        {"harm_scale": [0, 1], "safe_values": ["0"]},
    ],
)
def test_the_mapping_must_be_explicit_and_coherent(mapping: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        failure_rate_upper(**mapping)


def test_alpha_is_validated() -> None:
    with pytest.raises(ValueError, match="alpha"):
        failure_rate_upper(failure_values=["FAIL"], alpha=1.5)
