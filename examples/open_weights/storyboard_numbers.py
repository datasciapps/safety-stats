"""The storyboard's numbers, from one analysis report folder.

Reads only the content-free files `analyze` writes (`report.json`, `observations.csv`,
`hierarchy-risks.csv`, `comparisons.csv`, `plan.resolved.yaml`) and prints Markdown:

- what each epoch (one rollout per prompt) would have reported as a single run, beside the
  study's interval;
- the chance of at least one failure in k attempts, for k from 1 to 20, with the package's
  estimator (`worst_of_k_risk`), and the intervals the plan's worst-of-k measures carry;
- the same chance estimated without bias, with an interval for each k;
- every category against its claim's threshold, and for the open ones, how many rollouts per
  prompt the exact binomial planner needs at the observed rate;
- the comparisons, overall and per category.

    python storyboard_numbers.py REPORT_DIR [--output FILE]

The single-run and worst-of-k rates weight prompts as the plan does (categories equal, prompts
equal within a category); the benchmark's usual unweighted share is printed beside them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import comb

from inspect_safety_stats.stats.direct_mean import weighted_bounded_kl_chernoff_mean_bounds
from inspect_safety_stats.stats.planning import certification_sample_size
from inspect_safety_stats.stats.risk import worst_of_k_risk

TASK_GROUPS = {"harmful-compliance": "harm", "over-refusal": "over-refusal"}
KS = (1, 2, 3, 5, 10, 15, 20)
POWER = 0.9
# Two-sided level of the unbiased worst-of-k intervals, each k on its own.
ALPHA = 0.05


def _missing(value: float | None) -> bool:
    return value is None or math.isnan(value)


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{100 * value:.2f}%"


def _range(lower: float | None, upper: float | None) -> str:
    if _missing(lower) or _missing(upper):
        return "n/a"
    return f"{_pct(lower)} to {_pct(upper)}"


def _points(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100 * value:+.2f}"


def prompt_weights(design: pd.DataFrame) -> dict[str, float]:
    """Within each task, categories weigh equally and prompts equally within a category."""

    weights: dict[str, float] = {}
    for _, task_rows in design.groupby("task_id"):
        categories = task_rows["category"].nunique()
        for _, rows in task_rows.groupby("category"):
            for prompt_id in rows["prompt_id"]:
                weights[str(prompt_id)] = 1 / categories / len(rows)
    return weights


def thresholds(plan: dict[str, Any]) -> dict[tuple[str, str], float]:
    """The acceptable upper limit of each mean claim, by cohort and group."""

    return {
        (claim["cohort"], claim["group"]): float(claim["acceptable_upper"])
        for claim in plan.get("claims", [])
        if claim.get("measure") == "mean" and "cohort" in claim and "acceptable_upper" in claim
    }


def single_runs(observations: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Each epoch as one run: its weighted and unweighted failure rates per cohort and task.

    Unclassified rollouts are left out of the run they belong to, as a single run would.
    """

    classified = observations[
        observations["outcome_status"].isin(["classified_failure", "classified_safe"])
    ].copy()
    classified["fail"] = classified["outcome_status"] == "classified_failure"
    classified["weight"] = classified["prompt_id"].astype(str).map(weights)
    rows = []
    for (cohort, task, epoch), run in classified.groupby(["cohort_id", "task_id", "repeat"]):
        rows.append(
            {
                "cohort_id": cohort,
                "task_id": task,
                "epoch": int(epoch),
                "weighted": float(np.average(run["fail"], weights=run["weight"])),
                "plain": float(run["fail"].mean()),
                "failures": int(run["fail"].sum()),
                "rollouts": len(run),
            }
        )
    return pd.DataFrame(rows)


def worst_of_k_curve(
    prompt_estimates: list[dict[str, Any]], weights: dict[str, float], cohort: str, task: str
) -> dict[int, float] | None:
    """The package's worst-of-k estimate at every k, from the per-prompt rates.

    None when a prompt has no classified rollout, as the report then gives no estimate either.
    """

    rows = [p for p in prompt_estimates if p["cohort_id"] == cohort and p["task_id"] == task]
    if any(p["p_hat"] is None for p in rows):
        return None
    rates = np.array([float(p["p_hat"]) for p in rows])
    w = np.array([weights[str(p["prompt_id"])] for p in rows])
    w = w / w.sum()
    return {k: worst_of_k_risk(rates, w, k=k) for k in range(1, 21)}


def prompt_counts(observations: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Failures, safe answers and rollouts of every prompt, by cohort and task, with its weight."""

    status = observations["outcome_status"]
    frame = observations.assign(
        failures=status == "classified_failure", safe=status == "classified_safe"
    )
    counts = (
        frame.groupby(["cohort_id", "task_id", "prompt_id"])
        .agg(failures=("failures", "sum"), safe=("safe", "sum"), rollouts=("repeat", "size"))
        .reset_index()
    )
    counts["weight"] = counts["prompt_id"].astype(str).map(weights)
    return counts


def _none_fail(safe: np.ndarray, rollouts: np.ndarray, k: int) -> np.ndarray:
    """The share of each prompt's k-subsets of rollouts that hold no failure."""

    return np.asarray(comb(safe, k) / comb(rollouts, k), dtype=float)


def unbiased_worst_of_k(
    counts: pd.DataFrame, k: int, alpha: float = ALPHA
) -> tuple[float | None, float, float] | None:
    """The chance of at least one failure in k attempts: an unbiased estimate and an interval.

    For a prompt with n rollouts and x failures, 1 - C(n - x, k)/C(n, k), the share of its
    k-subsets of rollouts with a failure, has expectation exactly 1 - (1 - p)^k when k <= n.
    Putting x/n into that curve instead runs low, because the curve is concave.

    The interval treats rollouts as independent, as the study's primary intervals do. The
    share is a U-statistic of order k, and Hoeffding (1963, section 5) shows its moment
    generating function is at most that of the mean of floor(n/k) independent [0, 1] terms with
    the same expectation, one per disjoint block of k rollouts. The package's KL-Chernoff bounds
    for bounded blocks rest on that function alone, so each prompt enters as floor(n/k) blocks
    at its share, each weighted 1/floor(n/k) of the prompt's weight; two-sided at `alpha`. At
    k = 1 this is the rollout-level interval; from k > n/2 on, one block per prompt. Unclassified
    rollouts count as safe for the lower bound and as failures for the upper one; the share
    only grows with failures, so this is conservative.

    None when a prompt has fewer than k rollouts. The estimate alone is None when a prompt has
    fewer than k classified rollouts.
    """

    rollouts = counts["rollouts"].to_numpy()
    if (rollouts < k).any():
        return None
    failures = counts["failures"].to_numpy()
    safe = counts["safe"].to_numpy()
    classified = failures + safe
    weights = counts["weight"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    estimate = (
        None
        if (classified < k).any()
        else float(weights @ (1 - _none_fail(safe, classified, k)))
    )
    blocks = rollouts // k
    bounds = weighted_bounded_kl_chernoff_mean_bounds(
        np.repeat(1 - _none_fail(rollouts - failures, rollouts, k), blocks),
        np.repeat(1 - _none_fail(safe, rollouts, k), blocks),
        np.repeat(weights / blocks, blocks),
        alpha_lower=alpha / 2,
        alpha_upper=alpha / 2,
        independence_profile="rollout",
    )
    return estimate, bounds.lower, bounds.upper


def rollouts_to_certify(rate: float, threshold: float, alpha: float) -> int | None:
    """Rollouts the exact one-sided binomial planner needs at `rate`, or None if it cannot."""

    if rate >= threshold:
        return None
    return int(
        certification_sample_size(true_rate=rate, threshold=threshold, alpha=alpha, power=POWER)
    )


def build(report_dir: Path) -> str:
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    plan = yaml.safe_load((report_dir / "plan.resolved.yaml").read_text(encoding="utf-8"))
    observations = pd.read_csv(
        report_dir / "observations.csv", dtype={"prompt_id": str}, low_memory=False
    )
    design = pd.read_csv(report_dir / "design.resolved.csv", dtype={"prompt_id": str})
    hierarchy = pd.read_csv(report_dir / "hierarchy-risks.csv")
    comparisons = pd.read_csv(report_dir / "comparisons.csv")
    weights = prompt_weights(design)
    limits = thresholds(plan)
    risks = {(g["cohort_id"], g["group_id"], g["measure_id"]): g for g in report["group_risks"]}
    analysis = report["analysis"]
    classified = int(observations["outcome_status"].str.startswith("classified").sum())
    epochs = int(observations["repeat"].max())

    lines = [
        f"# Storyboard numbers: {analysis['id']}",
        "",
        f"- Plan SHA-256 `{analysis['plan_sha256']}`, frozen {analysis['plan_frozen_at']}, "
        f"intent {analysis['intent']}.",
        f"- Decision {report['decision']}, integrity {report['integrity']['status']}.",
        f"- {len(observations):,} rollouts, {classified:,} classified.",
        "",
        "## One run against the study",
        "",
        "| Cohort | Group | Study estimate | Study interval | Epoch 1 "
        f"| {epochs} single runs | Unweighted, {epochs} runs | Runs within the threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    runs = single_runs(observations, weights)
    for (cohort, task), rows in runs.groupby(["cohort_id", "task_id"]):
        group = TASK_GROUPS[str(task)]
        mean = risks[(cohort, group, "mean")]
        curve = worst_of_k_curve(report["prompt_estimates"], weights, str(cohort), str(task))
        if (curve is None) != (mean["estimate"] is None) or (
            curve is not None and not math.isclose(curve[1], mean["estimate"], abs_tol=1e-9)
        ):
            raise ValueError(
                f"{cohort}/{group}: weighted per-prompt rates give "
                f"{None if curve is None else curve[1]}, the report {mean['estimate']}; "
                "the plan weights prompts differently"
            )
        first = rows[rows["epoch"] == 1].iloc[0]
        limit = limits.get((str(cohort), group))
        within = (
            ""
            if limit is None
            else f"{int((rows['weighted'] <= limit).sum())} of {len(rows)} (≤ {_pct(limit)})"
        )
        lines.append(
            f"| {cohort} | {group} | {_pct(mean['estimate'])} "
            f"| {_range(mean['lower'], mean['upper'])} "
            f"| {_pct(first['weighted'])} ({first['failures']} of {first['rollouts']}) "
            f"| {_range(rows['weighted'].min(), rows['weighted'].max())} "
            f"| {_range(rows['plain'].min(), rows['plain'].max())} | {within} |"
        )

    lines += [
        "",
        "## Asking again: at least one failure in k attempts",
        "",
        "| Cohort | Group | " + " | ".join(f"k = {k}" for k in KS) + " | Report intervals |",
        "|---|---|" + "---|" * len(KS) + "---|",
    ]
    for (cohort, task), _ in runs.groupby(["cohort_id", "task_id"]):
        group = TASK_GROUPS[str(task)]
        curve = worst_of_k_curve(report["prompt_estimates"], weights, str(cohort), str(task))
        measures = sorted(
            (int(measure.split("_")[-1]), row)
            for (c, g, measure), row in risks.items()
            if c == cohort and g == group and measure.startswith("worst_of_")
        )
        intervals = [f"k = {k}: {_range(row['lower'], row['upper'])}" for k, row in measures]
        lines.append(
            f"| {cohort} | {group} | "
            + " | ".join(_pct(None if curve is None else curve[k]) for k in KS)
            + f" | {'; '.join(intervals)} |"
        )

    counts = prompt_counts(observations, weights)
    lines += [
        "",
        "## Asking again, unbiased: at least one failure in k attempts",
        "",
        "Each prompt's share of k-subsets of its rollouts with a failure, weighted as the plan "
        f"weights prompts; {100 * (1 - ALPHA):.0f}% intervals with rollouts independent, each k "
        "on its own.",
        "",
        "| Cohort | Group | " + " | ".join(f"k = {k}" for k in KS) + " |",
        "|---|---|" + "---|" * len(KS),
    ]
    for (cohort, task), rows in counts.groupby(["cohort_id", "task_id"]):
        cells = []
        for k in KS:
            result = unbiased_worst_of_k(rows, k)
            cells.append(
                "n/a" if result is None else f"{_pct(result[0])} ({_range(result[1], result[2])})"
            )
        lines.append(f"| {cohort} | {TASK_GROUPS[str(task)]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Categories against the claim's threshold (simultaneous intervals)",
        "",
    ]
    lines += (
        [
            "| Cohort | Group | Category | Estimate | Interval | Status "
            "| Rollouts per prompt to certify at this rate |",
            "|---|---|---|---|---|---|---|",
        ]
        if limits
        else ["The plan makes no claims, so no category has a threshold."]
    )
    categories = hierarchy[hierarchy["level"] == "category"]
    for _, row in categories.sort_values(["group_id", "cohort_id", "estimate"]).iterrows():
        limit = limits.get((row["cohort_id"], row["group_id"]))
        if limit is None:
            continue
        category = json.loads(row["path"])["category"]
        if _missing(row["lower"]) or _missing(row["upper"]) or _missing(row["estimate"]):
            status, needed = "no interval", ""
        elif row["upper"] <= limit:
            status, needed = f"below {_pct(limit)}", ""
        elif row["lower"] > limit:
            status, needed = f"above {_pct(limit)}", ""
        else:
            status = "open"
            alpha = float(row["endpoint_alpha"])
            trials = rollouts_to_certify(float(row["estimate"]), limit, alpha)
            needed = (
                "not at this rate (estimate at or above the threshold)"
                if trials is None
                else f"{math.ceil(trials / int(row['prompt_count']))} (now {epochs})"
            )
        lines.append(
            f"| {row['cohort_id']} | {row['group_id']} | {category} | {_pct(row['estimate'])} "
            f"| {_range(row['lower'], row['upper'])} | {status} | {needed} |"
        )

    lines += [
        "",
        "## Comparisons",
        "",
        "| Comparison | Level | Category | Difference | Interval | Independence |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in comparisons[comparisons["level"].isin(["task_id", "category"])].iterrows():
        category = json.loads(row["path"]).get("category", "all")
        lines.append(
            f"| {row['comparison_id']} | {row['level']} | {category} "
            f"| {_points(row['estimate'])} points "
            f"| {_points(row['lower'])} to {_points(row['upper'])} "
            f"| {row['independence_profile']} |"
        )
    return "\n".join(lines) + "\n"


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="A folder `analyze` wrote.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown file to write.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    text = build(args.report)
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
