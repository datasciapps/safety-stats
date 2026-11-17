"""Real Inspect logs written offline with the ``mockllm`` provider.

Five prompts in two metadata categories; prompt ``p0`` fails on odd epochs and every
other prompt is safe, so the fixed-design root risk is exactly 0.1 for any epoch count
that is a multiple of two.
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


def mock_task() -> Task:
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
        scorer=harmful_compliance(),
    )


def write_mock_logs(
    directory: Path,
    models: Sequence[str] = ("mockllm/alpha", "mockllm/beta"),
    *,
    epochs: int = 4,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    logs = inspect_eval(
        [mock_task()],
        model=list(models),
        epochs=epochs,
        log_dir=str(directory),
        display="none",
    )
    assert all(log.status == "success" for log in logs)
    return directory
