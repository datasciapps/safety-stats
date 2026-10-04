"""Harm and over-refusal in one report.

A model that refuses everything is perfectly safe on harmful prompts, so a harm number
on its own rewards over-refusal. Safety comparisons therefore report two rates side by
side: harmful compliance on prompts that should be refused, and over-refusal on prompts
that should be answered. They are usually separate Inspect evaluations with their own
prompts, manifests and scorers, so this module combines two finished reports rather
than asking one plan to hold both.

Coverage
--------
Each report's intervals hold simultaneously with probability at least one minus the
error budget of the families they come from. By the union bound, every interval in the
combined report holds simultaneously with probability at least one minus the sum of the
distinct families' budgets. No independence between the two evaluations is assumed. The
report states that combined floor for the rate table, for the difference table, and for
everything shown together.

Models are matched across the two reports by the model each cohort evaluated.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.schema.reports import ComparisonResult, GroupRiskResult, SafetyReport


@dataclass(frozen=True)
class RateCell:
    estimate: float | None
    lower: float | None
    upper: float | None
    family: str | None
    family_alpha: float | None
    source: str | None = None


@dataclass(frozen=True)
class ModelRow:
    model: str
    harm_cohort: str
    refusal_cohort: str
    harm: RateCell
    over_refusal: RateCell


@dataclass(frozen=True)
class PairRow:
    candidate: str
    baseline: str
    harm: RateCell
    over_refusal: RateCell
    reading: str


@dataclass
class TradeoffResult:
    harm_source: str
    refusal_source: str
    harm_group: str
    refusal_group: str
    measure_label: str
    models: list[ModelRow]
    pairs: list[PairRow]
    rate_coverage: float | None
    difference_coverage: float | None
    overall_coverage: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class _Loaded:
    report: SafetyReport
    plan: dict[str, Any]
    source: str


def _load(source: str | Path) -> _Loaded:
    path = Path(source)
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise FileNotFoundError(f"no report.json at {report_path}")
    report = SafetyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    plan_path = report_path.parent / "plan.resolved.yaml"
    plan = (
        yaml.safe_load(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    )
    return _Loaded(report=report, plan=plan or {}, source=str(report_path.parent))


def _mean_measure(loaded: _Loaded, requested: str | None) -> str:
    measures = loaded.plan.get("outputs", {}).get("risk_measures", []) or []
    means = [str(item["id"]) for item in measures if item.get("type") == "weighted_mean"]
    if requested is not None:
        if requested not in {str(item.get("id")) for item in measures}:
            raise ValueError(f"{loaded.source} has no risk measure {requested!r}")
        return requested
    if not means:
        raise ValueError(f"{loaded.source} reports no weighted_mean measure")
    return "mean" if "mean" in means else means[0]


def _group(loaded: _Loaded, requested: str | None) -> str:
    groups = [str(item["id"]) for item in loaded.report.design.get("groups", [])]
    if requested is not None:
        if requested not in groups:
            raise ValueError(f"{loaded.source} has no group {requested!r}; groups: {groups}")
        return requested
    if "all" in groups:
        return "all"
    if not groups:
        raise ValueError(f"{loaded.source} declares no groups")
    return groups[0]


def _models(loaded: _Loaded) -> dict[str, str]:
    """Model -> cohort, for cohorts that evaluated exactly one model."""

    mapping: dict[str, str] = {}
    for cohort in loaded.report.cohorts:
        models = [str(item) for item in cohort.get("models", [])]
        if len(models) == 1:
            mapping[models[0]] = str(cohort["cohort_id"])
    return mapping


def _cell(risk: GroupRiskResult | None, source: str) -> RateCell:
    if risk is None:
        return RateCell(None, None, None, None, None, source)
    primary = next((item for item in risk.intervals if item.role == "primary"), None)
    return RateCell(
        estimate=risk.estimate,
        lower=risk.lower,
        upper=risk.upper,
        family=primary.family if primary is not None else None,
        family_alpha=(primary.family_alpha if primary is not None else risk.family_alpha),
        source=source,
    )


def _difference(
    loaded: _Loaded, candidate: str, baseline: str, group: str, measure: str
) -> RateCell:
    for item in loaded.report.comparisons:
        if (
            item.level is not None
            or item.estimand != "risk_difference"
            or item.group_id != group
            or item.measure_id != measure
        ):
            continue
        if (item.candidate_cohort, item.baseline_cohort) == (candidate, baseline):
            return _comparison_cell(item, flip=False, source=loaded.source)
        if (item.candidate_cohort, item.baseline_cohort) == (baseline, candidate):
            return _comparison_cell(item, flip=True, source=loaded.source)
    return RateCell(None, None, None, None, None, loaded.source)


def _declared_orientation(
    loaded: _Loaded,
    models: dict[str, str],
    group: str,
    measure: str,
    first: str,
    second: str,
) -> tuple[str, str] | None:
    """The (candidate, baseline) models in the order the report's comparison declared."""

    cohorts = {models[first]: first, models[second]: second}
    for item in loaded.report.comparisons:
        if (
            item.level is None
            and item.estimand == "risk_difference"
            and item.group_id == group
            and item.measure_id == measure
            and {item.candidate_cohort, item.baseline_cohort} == set(cohorts)
        ):
            return cohorts[item.candidate_cohort], cohorts[item.baseline_cohort]
    return None


def _comparison_cell(item: ComparisonResult, *, flip: bool, source: str) -> RateCell:
    sign = -1.0 if flip else 1.0
    lower, upper = item.lower, item.upper
    if flip:
        lower, upper = (
            None if item.upper is None else -item.upper,
            None if item.lower is None else -item.lower,
        )
    return RateCell(
        estimate=None if item.estimate is None else sign * item.estimate,
        lower=lower,
        upper=upper,
        family=item.family,
        family_alpha=item.family_alpha,
        source=source,
    )


def _direction(cell: RateCell) -> int | None:
    if cell.lower is None or cell.upper is None:
        return None
    if cell.upper < 0:
        return -1
    if cell.lower > 0:
        return 1
    return 0


def _reading(candidate: str, harm: RateCell, refusal: RateCell) -> str:
    h, r = _direction(harm), _direction(refusal)
    if h is None or r is None:
        return "not available in both reports"
    harm_text = {-1: "less harmful", 1: "more harmful", 0: "harm difference unresolved"}[h]
    refusal_text = {
        -1: "refuses less",
        1: "refuses more",
        0: "over-refusal difference unresolved",
    }[r]
    if h == -1 and r == -1:
        return f"{candidate} is better on both"
    if h == 1 and r == 1:
        return f"{candidate} is worse on both"
    if h == -1 and r == 1:
        return f"trade-off: {candidate} is less harmful but refuses more"
    if h == 1 and r == -1:
        return f"trade-off: {candidate} is more harmful but refuses less"
    if h == 0 and r == 0:
        return "no difference resolved"
    return f"{candidate}: {harm_text}; {refusal_text}"


def _coverage(cells: list[RateCell]) -> float | None:
    """Union-bound floor: one minus the summed budgets of the distinct families used.

    A family is identified by its report and its id; two reports' families that share a
    name are still separate error budgets.
    """

    budgets: dict[tuple[str, str, str], float] = {}
    for index, cell in enumerate(cells):
        if cell.family_alpha is None:
            return None
        key = (
            ("cell", str(cell.source), str(index))
            if cell.family is None
            else ("family", str(cell.source), cell.family)
        )
        budgets[key] = cell.family_alpha
    return max(0.0, 1.0 - sum(budgets.values())) if budgets else None


def combine_tradeoff(
    harm: str | Path,
    over_refusal: str | Path,
    *,
    harm_group: str | None = None,
    refusal_group: str | None = None,
    measure: str | None = None,
) -> TradeoffResult:
    """Combine a harmful-compliance report and an over-refusal report."""

    harm_loaded = _load(harm)
    refusal_loaded = _load(over_refusal)
    harm_measure = _mean_measure(harm_loaded, measure)
    refusal_measure = _mean_measure(refusal_loaded, measure)
    harm_group_id = _group(harm_loaded, harm_group)
    refusal_group_id = _group(refusal_loaded, refusal_group)
    harm_models = _models(harm_loaded)
    refusal_models = _models(refusal_loaded)
    common = sorted(set(harm_models) & set(refusal_models))
    if not common:
        raise ValueError(
            "the two reports share no model: harm has "
            + ", ".join(sorted(harm_models))
            + "; over-refusal has "
            + ", ".join(sorted(refusal_models))
        )
    notes: list[str] = []
    only_harm = sorted(set(harm_models) - set(refusal_models))
    only_refusal = sorted(set(refusal_models) - set(harm_models))
    if only_harm or only_refusal:
        notes.append(
            "models in only one report are left out: " + ", ".join(only_harm + only_refusal)
        )

    def risk(loaded: _Loaded, cohort: str, group: str, measure_id: str) -> GroupRiskResult | None:
        return next(
            (
                item
                for item in loaded.report.group_risks
                if item.cohort_id == cohort
                and item.group_id == group
                and item.measure_id == measure_id
            ),
            None,
        )

    rows = [
        ModelRow(
            model=model,
            harm_cohort=harm_models[model],
            refusal_cohort=refusal_models[model],
            harm=_cell(
                risk(harm_loaded, harm_models[model], harm_group_id, harm_measure),
                harm_loaded.source,
            ),
            over_refusal=_cell(
                risk(refusal_loaded, refusal_models[model], refusal_group_id, refusal_measure),
                refusal_loaded.source,
            ),
        )
        for model in common
    ]
    pairs: list[PairRow] = []
    for index, first_model in enumerate(common):
        for second_model in common[:index]:
            orientation = _declared_orientation(
                harm_loaded, harm_models, harm_group_id, harm_measure, first_model, second_model
            ) or _declared_orientation(
                refusal_loaded, refusal_models, refusal_group_id, refusal_measure,
                first_model, second_model,
            )
            if orientation is None:
                continue
            candidate, baseline = orientation
            harm_cell = _difference(
                harm_loaded, harm_models[candidate], harm_models[baseline],
                harm_group_id, harm_measure,
            )
            refusal_cell = _difference(
                refusal_loaded, refusal_models[candidate], refusal_models[baseline],
                refusal_group_id, refusal_measure,
            )
            pairs.append(
                PairRow(
                    candidate=candidate,
                    baseline=baseline,
                    harm=harm_cell,
                    over_refusal=refusal_cell,
                    reading=_reading(candidate, harm_cell, refusal_cell),
                )
            )
    for pair in pairs:
        if pair.harm.lower is None or pair.over_refusal.lower is None:
            notes.append(
                f"{pair.candidate} vs {pair.baseline} is compared in only one report; declare "
                "the comparison in both plans to read the trade-off"
            )

    graded = any(
        bool(loaded.plan.get("inputs", {}).get("score", {}).get("mapping", {}).get("scale"))
        for loaded in (harm_loaded, refusal_loaded)
    )
    rate_cells = [cell for row in rows for cell in (row.harm, row.over_refusal)]
    difference_cells = [
        cell for pair in pairs for cell in (pair.harm, pair.over_refusal) if cell.lower is not None
    ]
    return TradeoffResult(
        harm_source=harm_loaded.source,
        refusal_source=refusal_loaded.source,
        harm_group=harm_group_id,
        refusal_group=refusal_group_id,
        measure_label="mean harm" if graded else "rate",
        models=rows,
        pairs=pairs,
        rate_coverage=_coverage(rate_cells),
        difference_coverage=_coverage(difference_cells) if difference_cells else None,
        overall_coverage=_coverage(rate_cells + difference_cells),
        notes=notes,
    )


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _pts(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:+.1f} pts"


def _range(cell: RateCell, *, signed: bool = False) -> str:
    if cell.lower is None or cell.upper is None:
        return "unavailable"
    form = _pts if signed else _pct
    return f"[{form(cell.lower)}, {form(cell.upper)}]"


def tradeoff_markdown(result: TradeoffResult) -> str:
    lines = [
        "# Harm and over-refusal",
        "",
        f"Harm: `{result.harm_source}` (group `{result.harm_group}`). "
        f"Over-refusal: `{result.refusal_source}` (group `{result.refusal_group}`).",
        "",
        "## Per model",
        "",
        "| Model | Harmful compliance | Interval | Over-refusal | Interval |",
        "|---|---:|---|---:|---|",
    ]
    for row in result.models:
        lines.append(
            f"| {row.model} | {_pct(row.harm.estimate)} | {_range(row.harm)} | "
            f"{_pct(row.over_refusal.estimate)} | {_range(row.over_refusal)} |"
        )
    if result.rate_coverage is not None:
        lines.extend(
            ["", f"All rates hold simultaneously with probability at least "
             f"{_pct(result.rate_coverage)} (union bound over both reports' families)."]
        )
    if result.pairs:
        lines.extend(
            [
                "",
                "## Comparisons",
                "",
                "| Candidate vs baseline | Harm difference | Interval | Over-refusal difference "
                "| Interval | Reading |",
                "|---|---:|---|---:|---|---|",
            ]
        )
        for pair in result.pairs:
            lines.append(
                f"| {pair.candidate} vs {pair.baseline} | {_pts(pair.harm.estimate)} | "
                f"{_range(pair.harm, signed=True)} | {_pts(pair.over_refusal.estimate)} | "
                f"{_range(pair.over_refusal, signed=True)} | {pair.reading} |"
            )
        if result.difference_coverage is not None:
            lines.extend(
                ["", f"All differences hold simultaneously with probability at least "
                 f"{_pct(result.difference_coverage)}. A difference is resolved when its "
                 "interval excludes zero; negative is better on both axes."]
            )
    if result.overall_coverage is not None:
        lines.extend(
            ["", f"Everything on this page holds simultaneously with probability at least "
             f"{_pct(result.overall_coverage)}."]
        )
    if result.notes:
        lines.extend(["", "## Notes", "", *[f"- {note}" for note in result.notes]])
    lines.extend(
        [
            "",
            "A model that refuses everything has no harmful compliance and maximal "
            "over-refusal; read the two columns together. Both rates are properties of their "
            "own fixed prompt sets.",
        ]
    )
    return "\n".join(lines) + "\n"


def _nice_max(value: float) -> float:
    """Round an axis maximum up to a value whose ticks read cleanly."""

    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if value <= step:
            return step
    return 1.0


def _svg(result: TradeoffResult) -> str:
    width, height = 640, 440
    left, right, top, bottom = 84, 24, 20, 60
    plotted = [
        row for row in result.models
        if None not in (row.harm.lower, row.harm.upper, row.over_refusal.lower,
                        row.over_refusal.upper, row.harm.estimate, row.over_refusal.estimate)
    ]
    x_max = _nice_max(max([0.0, *[float(row.harm.upper or 0) for row in plotted]]))
    y_max = _nice_max(max([0.0, *[float(row.over_refusal.upper or 0) for row in plotted]]))

    def x(value: float) -> float:
        return left + (width - left - right) * value / x_max

    def y(value: float) -> float:
        return height - bottom - (height - top - bottom) * value / y_max

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Harmful compliance against over-refusal with confidence intervals">',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx, gy = x(fraction * x_max), y(fraction * y_max)
        parts.extend(
            [
                f'<line class="grid" x1="{gx}" y1="{top}" x2="{gx}" y2="{height - bottom}"/>',
                f'<line class="grid" x1="{left}" y1="{gy}" x2="{width - right}" y2="{gy}"/>',
                f'<text class="tick" x="{gx}" y="{height - bottom + 18}" '
                f'text-anchor="middle">{_pct(fraction * x_max)}</text>',
                f'<text class="tick" x="{left - 8}" y="{gy + 4}" '
                f'text-anchor="end">{_pct(fraction * y_max)}</text>',
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" '
            f'y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<text class="label" x="{(left + width - right) / 2}" y="{height - 14}" '
            'text-anchor="middle">Harmful compliance</text>',
            f'<text class="label" x="16" y="{(top + height - bottom) / 2}" '
            f'text-anchor="middle" transform="rotate(-90 16 {(top + height - bottom) / 2})">'
            "Over-refusal</text>",
        ]
    )
    for row in plotted:
        hx, ry = float(row.harm.estimate or 0), float(row.over_refusal.estimate or 0)
        parts.extend(
            [
                f'<line class="bar" x1="{x(float(row.harm.lower or 0))}" y1="{y(ry)}" '
                f'x2="{x(float(row.harm.upper or 0))}" y2="{y(ry)}"/>',
                f'<line class="bar" x1="{x(hx)}" y1="{y(float(row.over_refusal.lower or 0))}" '
                f'x2="{x(hx)}" y2="{y(float(row.over_refusal.upper or 0))}"/>',
                f'<circle class="point" cx="{x(hx)}" cy="{y(ry)}" r="5"/>',
                f'<text class="name" x="{x(hx) + (-8 if x(hx) > 0.7 * width else 8)}" '
                f'y="{y(ry) - 8}" text-anchor="{"end" if x(hx) > 0.7 * width else "start"}">'
                f"{html.escape(row.model)}</text>",
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def tradeoff_html(result: TradeoffResult) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(row.model)}</td><td>{_pct(row.harm.estimate)}</td>"
        f"<td>{_range(row.harm)}</td><td>{_pct(row.over_refusal.estimate)}</td>"
        f"<td>{_range(row.over_refusal)}</td></tr>"
        for row in result.models
    )
    pairs = "".join(
        f"<tr><td>{html.escape(pair.candidate)} vs {html.escape(pair.baseline)}</td>"
        f"<td>{_range(pair.harm, signed=True)}</td>"
        f"<td>{_range(pair.over_refusal, signed=True)}</td>"
        f"<td>{html.escape(pair.reading)}</td></tr>"
        for pair in result.pairs
    )
    coverage = (
        f"<p>Everything shown holds simultaneously with probability at least "
        f"{_pct(result.overall_coverage)}.</p>"
        if result.overall_coverage is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harm and over-refusal</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#d8d6cf; --accent:#2f5d8a; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161614; --fg:#ecebe6; --muted:#a3a29b; --line:#3a3935; --accent:#8db4dc; }}
}}
body {{ margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:880px; margin:0 auto; }}
h1 {{ font-size:1.4rem; margin:0 0 .25rem; }} h2 {{ font-size:1.05rem; margin:2rem 0 .5rem; }}
p {{ color:var(--muted); }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:.4rem .5rem; border-bottom:1px solid var(--line); }}
svg {{ width:100%; max-width:640px; height:auto; }}
.axis {{ stroke:var(--muted); }} .grid {{ stroke:var(--line); }}
.bar {{ stroke:var(--accent); stroke-width:2; }}
.point {{ fill:var(--accent); }} .label, .tick, .name {{ fill:var(--fg); font-size:12px; }}
.tick {{ fill:var(--muted); }}
.wrap {{ overflow-x:auto; }}
</style></head>
<body><main>
<h1>Harm and over-refusal</h1>
<p>Lower-left is better on both. Bars are each model's simultaneous intervals.</p>
{_svg(result)}
<h2>Per model</h2>
<div class="wrap"><table><thead><tr><th>Model</th><th>Harmful compliance</th><th>Interval</th>
<th>Over-refusal</th><th>Interval</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Comparisons</h2>
<div class="wrap"><table><thead><tr><th>Candidate vs baseline</th><th>Harm difference</th>
<th>Over-refusal difference</th><th>Reading</th></tr></thead><tbody>{pairs}</tbody></table></div>
{coverage}
</main></body></html>
"""


def write_tradeoff(
    result: TradeoffResult, output: str | Path, *, overwrite: bool = False
) -> Path:
    directory = Path(output)
    targets = [directory / name for name in ("tradeoff.md", "tradeoff.json", "tradeoff.html")]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite " + ", ".join(str(path) for path in existing)
            + "; pass --overwrite"
        )
    directory.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(tradeoff_markdown(result), encoding="utf-8")
    targets[1].write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    targets[2].write_text(tradeoff_html(result), encoding="utf-8")
    return directory
