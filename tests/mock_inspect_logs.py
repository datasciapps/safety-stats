"""Real Inspect logs written offline with the ``mockllm`` provider.

Five prompts in two metadata categories; prompt ``p0`` fails on odd epochs and every
other prompt is safe, so the fixed-design root risk is exactly 0.1 for any epoch count
that is a multiple of two.

``mock_task`` takes other names, sample identifier prefixes and sample metadata, for logs of
several tasks, of a score with more values, or of metadata a sample lacks: a sample whose
metadata lists ``verdicts`` gets them in turn, one per epoch, and one with ``error`` set makes
its evaluation fail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, generate, solver

DEFAULT_METADATA: tuple[dict[str, Any], ...] = tuple(
    {"category": "a" if index < 3 else "b", "bad": index == 0} for index in range(5)
)


@scorer(metrics=[accuracy()])
def harmful_compliance():  # type: ignore[no-untyped-def]
    async def score(state: TaskState, target: Target) -> Score:
        verdicts = state.metadata.get("verdicts")
        if verdicts:
            return Score(value=verdicts[(state.epoch - 1) % len(verdicts)])
        failed = bool(state.metadata.get("bad")) and state.epoch % 2 == 1
        return Score(value="FAIL" if failed else "PASS")

    return score


@solver
def fail_where_asked():  # type: ignore[no-untyped-def]
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.metadata.get("error"):
            raise RuntimeError("the sample asked to fail")
        return state

    return solve


def mock_task(
    name: str = "mock_safety",
    metadata: Sequence[Mapping[str, Any]] = DEFAULT_METADATA,
    *,
    prefix: str = "p",
) -> Task:
    return Task(
        name=name,
        dataset=[
            Sample(input=f"question {index}", id=f"{prefix}{index}", metadata=dict(item))
            for index, item in enumerate(metadata)
        ],
        # The default task keeps its single solver, so the logs it writes stay as they were.
        solver=(
            [fail_where_asked(), generate()]
            if any(item.get("error") for item in metadata)
            else generate()
        ),
        scorer=harmful_compliance(),
    )


def write_mock_logs(
    directory: Path,
    models: Sequence[str] = ("mockllm/alpha", "mockllm/beta"),
    *,
    epochs: int = 4,
    tasks: Sequence[Task] | None = None,
    succeed: bool = True,
) -> Path:
    """Run ``tasks`` (by default ``mock_task()``) on each model into ``directory``.

    Every evaluation must succeed, or with ``succeed=False`` every one must fail.
    """

    directory.mkdir(parents=True, exist_ok=True)
    logs = inspect_eval(
        list(tasks) if tasks is not None else [mock_task()],
        model=list(models),
        epochs=epochs,
        log_dir=str(directory),
        display="none",
    )
    expected = "success" if succeed else "error"
    if any(log.status != expected for log in logs):
        raise RuntimeError(f"an evaluation did not end in {expected}")
    return directory
