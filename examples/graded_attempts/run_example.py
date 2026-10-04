"""Run the graded-score, repeated-attempt example end to end, offline.

    python examples/graded_attempts/run_example.py [--output DIR]

1. Evaluates the rubric task for two mock models, 8 epochs each, with Inspect.
2. Writes a starter plan with ``init --harm-scale 1 5`` (5 is the most harmful grade).
3. Adds a worst-of-4 attempt measure in its own multiplicity family.
4. Analyzes and writes the report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from inspect_ai import eval as inspect_eval

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rubric_task import rubric_safety  # noqa: E402

from inspect_safety_stats import analyze  # noqa: E402
from inspect_safety_stats.scaffold import scaffold_plan  # noqa: E402


def run(output: Path, *, epochs: int = 8) -> Path:
    logs = output / "logs"
    results = inspect_eval(
        rubric_safety(),
        model=["mockllm/model-a", "mockllm/model-b"],
        epochs=epochs,
        log_dir=str(logs),
        display="none",
    )
    if any(log.status != "success" for log in results):
        raise RuntimeError("the example evaluation did not finish")

    plan_result = scaffold_plan(
        logs,
        output / "analysis",
        harm_scale=(1.0, 5.0),
        group_by="category",
        baseline="mockllm/model-a",
        overwrite=True,
    )
    plan = yaml.safe_load(plan_result.plan_path.read_text(encoding="utf-8"))
    plan["outputs"]["risk_measures"].append(
        {"id": "worst_of_4", "type": "k_attempt_risk", "k": 4, "family": "attempts"}
    )
    plan["multiplicity"]["families"].append(
        {"id": "attempts", "purpose": "simultaneous_intervals", "method": "bonferroni",
         "alpha": 0.05}
    )
    plan_result.plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    report = analyze(plan=plan_result.plan_path, logs=str(logs))
    return report.write(output / "report", overwrite=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "run")
    args = parser.parse_args()
    destination = run(args.output)
    print(f"Summary: {destination / 'report.md'}")
    print(f"Interactive report: {destination / 'report.html'}")


if __name__ == "__main__":
    main()
