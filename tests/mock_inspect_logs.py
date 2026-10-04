"""Real Inspect logs written offline with the ``mockllm`` provider.

Five prompts in two metadata categories; prompt ``p0`` fails on odd epochs and every
other prompt is safe, so the fixed-design root risk is exactly 0.1 for any epoch count
that is a multiple of two.

The graded variant scores on a 1-5 rubric where 5 is most harmful: ``p0`` scores 4 on
odd epochs and 2 on even ones, every other prompt 1. On the unit scale that is 0.75 and
0.25 for ``p0`` and 0 elsewhere, so the root mean harm is again exactly 0.1.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate


@scorer(metrics=[accuracy()])
def harmful_compliance():  # type: ignore[no-untyped-def]
    async def score(state: TaskState, target: Target) -> Score:
        failed = bool(state.metadata.get("bad")) and state.epoch % 2 == 1
        return Score(value="FAIL" if failed else "PASS")

    return score


@scorer(metrics=[accuracy()])
def harm_rubric():  # type: ignore[no-untyped-def]
    async def score(state: TaskState, target: Target) -> Score:
        if not state.metadata.get("bad"):
            return Score(value=1)
        return Score(value=4 if state.epoch % 2 == 1 else 2)

    return score


def mock_task(*, graded: bool = False) -> Task:
    return Task(
        name="mock_safety",
        dataset=[
            Sample(
                input=f"question {index}",
                id=f"p{index}",
                metadata={"category": "a" if index < 3 else "b", "bad": index == 0},
            )
            for index in range(5)
        ],
        solver=generate(),
        scorer=harm_rubric() if graded else harmful_compliance(),
    )


def write_mock_logs(
    directory: Path,
    models: Sequence[str] = ("mockllm/alpha", "mockllm/beta"),
    *,
    epochs: int = 4,
    graded: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    logs = inspect_eval(
        [mock_task(graded=graded)],
        model=list(models),
        epochs=epochs,
        log_dir=str(directory),
        display="none",
    )
    assert all(log.status == "success" for log in logs)
    return directory


@scorer(metrics=[accuracy()])
def profile_scorer(failing: dict[str, frozenset[str]]):  # type: ignore[no-untyped-def]
    """FAIL on the prompts listed for the model being evaluated, on every epoch."""

    async def score(state: TaskState, target: Target) -> Score:
        listed = failing.get(str(state.model), frozenset())
        return Score(value="FAIL" if str(state.sample_id) in listed else "PASS")

    return score


def write_profile_logs(
    directory: Path,
    failing: dict[str, frozenset[str]],
    *,
    prompts: int = 10,
    epochs: int = 20,
    name: str = "profile_task",
) -> Path:
    """Logs where each model fails exactly the prompts listed for it, on every epoch."""

    directory.mkdir(parents=True, exist_ok=True)
    task = Task(
        name=name,
        dataset=[Sample(input=f"q{index}", id=f"p{index}") for index in range(prompts)],
        solver=generate(),
        scorer=profile_scorer(failing),
    )
    logs = inspect_eval(
        [task], model=sorted(failing), epochs=epochs, log_dir=str(directory), display="none"
    )
    assert all(log.status == "success" for log in logs)
    return directory
