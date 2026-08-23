# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.certification.fixed_design_planner import (
    FixedDesignPlanningReport,
)
from inspect_safety_stats.config.canonicalize import canonical_value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _claim_rows(report: FixedDesignPlanningReport) -> list[dict[str, Any]]:
    return [
        {
            **claim.model_dump(mode="json", exclude={"target"}),
            "target": json.dumps(claim.target, sort_keys=True),
        }
        for claim in report.claims
    ]


def _allocation_rows(report: FixedDesignPlanningReport) -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": item.prompt_id,
            **item.hierarchy,
            "rollouts": item.rollouts,
            "existing_expected_rollouts": item.existing_expected_rollouts,
            "maximum_claim_weight": item.maximum_claim_weight,
        }
        for item in report.allocation
    ]


def _markdown(report: FixedDesignPlanningReport) -> str:
    summary = report.summary
    lines = [
        f"# {report.certification.get('title', 'Fixed-design rollout plan')}",
        "",
        f"- Direction: `{report.settings['direction']}`",
        f"- Anticipated failure rate: {_percent(report.settings['anticipated_rate'])}",
        f"- Target power: {_percent(report.settings['power'])}",
        f"- Feasible within configured rollout limit: `{str(summary['feasible']).lower()}`",
        f"- Minimum common rollouts per prompt: {summary['required_equal_rollouts_per_prompt']}",
        f"- Common-allocation total rollouts: {summary['common_total_rollouts']}",
        f"- Weight-adaptive total rollouts: {summary['adaptive_total_rollouts']}",
        f"- Weight-adaptive range: {summary['adaptive_min_rollouts']}–{summary['adaptive_max_rollouts']} per prompt",
        "",
        "## Claims",
        "",
        "| Claim | Prompts | Threshold | Local alpha | Equal rollouts | Planned range | Effective size | Detectable rate | Binding |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for claim in report.claims:
        title = claim.title.replace("|", "\\|")
        lines.append(
            f"| {title} | {claim.prompt_count} | "
            f"{_percent(claim.threshold)} | {claim.local_alpha:.6g} | "
            f"{claim.required_equal_rollouts_per_prompt} | "
            f"{claim.planned_min_rollouts}–{claim.planned_max_rollouts} | "
            f"{claim.planned_effective_size:.1f} | "
            f"{_percent(claim.planned_detectable_rate)} | "
            f"{'yes' if claim.binding_equal_allocation else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The weighted calculations are conservative finite-sample Hoeffding power guarantees for the declared fixed design under independent rollouts. They permit heterogeneous prompt failure probabilities.",
            "",
        ]
    )
    lines.extend(f"- Warning: {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def _html(report: FixedDesignPlanningReport) -> str:
    summary = report.summary
    claim_rows = "".join(
        "<tr>"
        f"<td>{html.escape(claim.title)}</td>"
        f"<td class='num'>{claim.prompt_count}</td>"
        f"<td class='num'>{_percent(claim.threshold)}</td>"
        f"<td class='num'>{claim.local_alpha:.6g}</td>"
        f"<td class='num'>{claim.required_equal_rollouts_per_prompt}</td>"
        f"<td class='num'>{claim.planned_min_rollouts}–{claim.planned_max_rollouts}</td>"
        f"<td class='num'>{claim.planned_effective_size:.1f}</td>"
        f"<td class='num'>{_percent(claim.planned_detectable_rate)}</td>"
        f"<td>{'yes' if claim.binding_equal_allocation else 'no'}</td>"
        "</tr>"
        for claim in report.claims
    )
    warning = "".join(f"<li>{html.escape(item)}</li>" for item in report.warnings)
    feasible = "Feasible" if summary["feasible"] else "Exceeds limit"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report.certification.get("title", "Fixed-design rollout plan")))}</title>
<style>
:root{{--bg:#0e131b;--panel:#171e29;--line:#303b4d;--text:#e9eef7;--muted:#aab5c7;--blue:#8fb2ff;--green:#5fd49a;--red:#ff858f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1450px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px}}h2{{margin-top:32px}}p{{color:var(--muted)}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));gap:12px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700}}.label{{color:var(--muted)}}.ok{{color:var(--green)}}.bad{{color:var(--red)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}.num{{text-align:right;font-variant-numeric:tabular-nums}}@media(max-width:800px){{.summary{{grid-template-columns:1fr 1fr}}main{{padding:16px}}}}
</style></head><body><main>
<h1>{html.escape(str(report.certification.get("title", "Fixed-design rollout plan")))}</h1>
<p>{html.escape(str(report.analysis.get("title", "")))} · {report.settings["direction"]} planning at {_percent(report.settings["power"])} power and {_percent(report.settings["anticipated_rate"])} anticipated failure rate.</p>
<div class="summary"><div class="card"><div class="label">Status</div><div class="value {"ok" if summary["feasible"] else "bad"}">{feasible}</div></div><div class="card"><div class="label">Common rollouts / prompt</div><div class="value">{summary["required_equal_rollouts_per_prompt"]}</div></div><div class="card"><div class="label">Common total rollouts</div><div class="value">{summary["common_total_rollouts"]}</div></div><div class="card"><div class="label">Adaptive total rollouts</div><div class="value">{summary["adaptive_total_rollouts"]}</div></div></div>
<h2>Claim requirements</h2><div class="table"><table><thead><tr><th>Claim</th><th class="num">Prompts</th><th class="num">Threshold</th><th class="num">Local alpha</th><th class="num">Equal rollouts</th><th class="num">Planned range</th><th class="num">Effective size</th><th class="num">Detectable rate</th><th>Binding</th></tr></thead><tbody>{claim_rows}</tbody></table></div>
<h2>Interpretation</h2><p>The common-rollout result is the smallest single rollout count satisfying every claim. The adaptive allocation gives higher-weight prompts more rollouts and is integer-reduced while preserving every Hoeffding power constraint. Calculations assume independent rollouts and allow heterogeneous prompt failure probabilities.</p>
{f"<h2>Warnings</h2><ul>{warning}</ul>" if warning else ""}
</main></body></html>"""


def write_fixed_design_planning_report(
    report: FixedDesignPlanningReport,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    directory = Path(output).resolve()
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"planning output exists and is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"planning output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "planning-report.json").write_text(
        json.dumps(
            canonical_value(report.model_dump(mode="json")),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "planning-report.md").write_text(_markdown(report), encoding="utf-8")
    (directory / "planning-report.html").write_text(_html(report), encoding="utf-8")
    _write_csv(directory / "planning-claims.csv", _claim_rows(report))
    _write_csv(directory / "rollout-allocation.csv", _allocation_rows(report))
    (directory / "analysis-plan.resolved.yaml").write_text(
        yaml.safe_dump(
            canonical_value(report._resolved_analysis_plan),
            sort_keys=True,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (directory / "certification-plan.resolved.yaml").write_text(
        yaml.safe_dump(
            canonical_value(report._resolved_certification_plan),
            sort_keys=True,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    generated = [
        "analysis-plan.resolved.yaml",
        "certification-plan.resolved.yaml",
        "planning-claims.csv",
        "planning-report.html",
        "planning-report.json",
        "planning-report.md",
        "rollout-allocation.csv",
    ]
    checksums = {name: _sha256(directory / name) for name in generated}
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory
