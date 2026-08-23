# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.certification.schema import (
    CertificationClaimResult,
    CertificationReport,
)
from inspect_safety_stats.config.canonicalize import canonical_value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: float | None, places: int) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _percent(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{places}f}%"


def _target_label(claim: CertificationClaimResult) -> str:
    target = claim.target
    if target["type"] == "prompt_failure_rate":
        return f"prompt {target['prompt_id']}"
    if target["type"] == "group_mean":
        return f"group {target['group']}"
    path = " → ".join(str(value) for value in target.get("path", {}).values())
    return f"{target['level']} {path}"


def _markdown(report: CertificationReport, places: int) -> str:
    lines = [
        f"# {report.analysis.get('title', 'Statistical certification evidence')}",
        "",
        f"**Supported:** {report.summary['supported']}  ",
        f"**Rejected:** {report.summary['rejected']}  ",
        f"**Inconclusive:** {report.summary['inconclusive']}",
        "",
        str(report.scope["disclaimer"]),
        "",
        "## Claims",
        "",
        (
            "Support means the simultaneous one-sided upper bound is at or below the "
            "specified threshold. Rejection means the corresponding lower bound is strictly "
            "above it. Otherwise the evidence is inconclusive."
        ),
        "",
        (
            "| Claim | Target | Threshold | Estimate | Support upper | Support adjusted p | "
            "Rejection lower | Rejection adjusted p | Evidence | Robustness |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for claim in report.claims:
        rejection = claim.rejection
        lines.append(
            "| "
            + " | ".join(
                [
                    claim.title.replace("|", "\\|"),
                    _target_label(claim).replace("|", "\\|"),
                    _number(claim.threshold, places),
                    _number(claim.estimate, places),
                    _number(claim.support.bound, places),
                    _number(claim.support.adjusted_p_value, places),
                    _number(rejection.bound if rejection else None, places),
                    _number(rejection.adjusted_p_value if rejection else None, places),
                    claim.evidence,
                    claim.robustness,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Directional evidence", ""])
    for claim in report.claims:
        lines.extend(
            [
                f"### {claim.title}",
                "",
                f"- Claim: risk ≤ {_number(claim.threshold, places)}.",
                (
                    f"- Support: upper bound {_number(claim.support.bound, places)}, "
                    f"raw p={_number(claim.support.raw_p_value, places)}, adjusted "
                    f"p={_number(claim.support.adjusted_p_value, places)}, local "
                    f"alpha={_number(claim.support.local_alpha, places)}."
                ),
            ]
        )
        if claim.rejection:
            lines.append(
                f"- Rejection: lower bound {_number(claim.rejection.bound, places)}, "
                f"raw p={_number(claim.rejection.raw_p_value, places)}, adjusted "
                f"p={_number(claim.rejection.adjusted_p_value, places)}, local "
                f"alpha={_number(claim.rejection.local_alpha, places)}."
            )
        lines.append(
            f"- Data: {claim.prompt_count} fixed prompts, {claim.attempted_rollouts} attempted "
            f"rollouts, {claim.unclassified_outcomes} unclassified."
        )
        for sensitivity in claim.sensitivities:
            lines.append(
                f"- Sensitivity `{sensitivity.method}` / `{sensitivity.independence_profile}`: "
                f"{_number(sensitivity.lower, places)}–"
                f"{_number(sensitivity.upper, places)}; `{sensitivity.evidence}`."
            )
        for diagnostic in claim.detectability:
            lines.append(
                f"- Detectability `{diagnostic.method}` / "
                f"`{diagnostic.independence_profile or 'exact prompt'}` at "
                f"{_percent(diagnostic.target_power, 1)} power: maximum certifiable rate "
                f"{_percent(diagnostic.maximum_certifiable_failure_rate)}, minimum detectable "
                f"rate {_percent(diagnostic.minimum_detectable_failure_rate)}, effective size "
                f"{_number(diagnostic.effective_sample_size, 1)} ({diagnostic.guarantee})."
            )
        lines.extend(f"- Warning: {warning}" for warning in claim.warnings)
        lines.append("")
    lines.extend(
        [
            "## Error control",
            "",
            (
                "Support and rejection use separate one-sided error families. Adjusted p-values "
                "are Bonferroni values for their declared family; they are not posterior "
                "probabilities that a claim is true or false."
            ),
            "",
        ]
    )
    for family in report.families:
        lines.append(
            f"- `{family['id']}`: {family['purpose']}, {family['method']}, alpha={family['alpha']}."
        )
    lines.extend(["", "## Provenance", ""])
    lines.append(f"- Empirical analysis: `{report.source['empirical_analysis_id']}`")
    lines.append(f"- Empirical report SHA-256: `{report.source['empirical_report_sha256']}`")
    lines.append(f"- Certification plan SHA-256: `{report.analysis['plan_sha256']}`")
    return "\n".join(lines) + "\n"


def _html(report: CertificationReport, places: int) -> str:
    rows: list[str] = []
    detectability_rows: list[str] = []
    for claim in report.claims:
        rejection = claim.rejection
        status_class = claim.evidence.lower()
        rows.append(
            "<tr>"
            f"<td>{html.escape(claim.title)}</td>"
            f"<td>{html.escape(_target_label(claim))}</td>"
            f"<td class='num'>{_percent(claim.threshold)}</td>"
            f"<td class='num'>{_percent(claim.estimate)}</td>"
            f"<td class='num'>{_percent(claim.support.bound)}</td>"
            f"<td class='num'>{_number(claim.support.adjusted_p_value, places)}</td>"
            f"<td class='num'>{_percent(rejection.bound if rejection else None)}</td>"
            f"<td class='num'>{_number(rejection.adjusted_p_value if rejection else None, places)}</td>"
            f"<td><span class='badge {status_class}'>{claim.evidence}</span></td>"
            "</tr>"
        )
        for diagnostic in claim.detectability:
            detectability_rows.append(
                "<tr>"
                f"<td>{html.escape(claim.title)}</td>"
                f"<td>{html.escape(diagnostic.independence_profile or 'exact prompt')}</td>"
                f"<td class='num'>{_percent(diagnostic.target_power, 1)}</td>"
                f"<td class='num'>{_number(diagnostic.effective_sample_size, 1)}</td>"
                f"<td class='num'>{_percent(diagnostic.maximum_certifiable_failure_rate)}</td>"
                f"<td class='num'>{_percent(diagnostic.minimum_detectable_failure_rate)}</td>"
                "</tr>"
            )
    warnings = "".join(
        f"<li>{html.escape(warning)}</li>"
        for warning in [*report.warnings, *(w for c in report.claims for w in c.warnings)]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report.analysis.get("title", "Certification evidence")))}</title>
<style>
:root{{--bg:#0e131b;--panel:#171e29;--line:#303b4d;--text:#e9eef7;--muted:#aab5c7;--blue:#8fb2ff;--green:#5fd49a;--red:#ff858f;--amber:#e6bd63}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px}}h2{{margin-top:34px}}p{{color:var(--muted)}}.summary{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:24px 0}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}.value{{font-size:25px;font-weight:700}}.label{{color:var(--muted)}}.table{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:11px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-size:12px;text-transform:uppercase}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.badge{{font-size:12px;font-weight:700;padding:4px 8px;border-radius:999px}}.supported{{color:var(--green);background:#17382c}}.rejected{{color:var(--red);background:#421e25}}.inconclusive{{color:var(--amber);background:#3d321c}}code{{color:var(--blue)}}@media(max-width:800px){{.summary{{grid-template-columns:1fr 1fr}}main{{padding:16px}}}}
</style></head><body><main>
<h1>{html.escape(str(report.analysis.get("title", "Statistical certification evidence")))}</h1>
<p>{html.escape(str(report.scope["disclaimer"]))}</p>
<div class="summary"><div class="card"><div class="label">Claims</div><div class="value">{report.summary["total"]}</div></div><div class="card"><div class="label">Supported</div><div class="value">{report.summary["supported"]}</div></div><div class="card"><div class="label">Rejected</div><div class="value">{report.summary["rejected"]}</div></div><div class="card"><div class="label">Inconclusive</div><div class="value">{report.summary["inconclusive"]}</div></div></div>
<h2>Threshold claims</h2><p>Support uses a one-sided upper bound; rejection uses a separate one-sided lower bound. Error budgets are controlled independently.</p>
<div class="table"><table><thead><tr><th>Claim</th><th>Target</th><th class="num">Threshold</th><th class="num">Estimate</th><th class="num">Support upper</th><th class="num">Support adjusted p</th><th class="num">Rejection lower</th><th class="num">Rejection adjusted p</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
{f'''<h2>Detectability at the evaluated horizon</h2><p>Results use each claim's selected primary independence profile. Individual prompts use exact binomial inversion; weighted means use conservative fixed-design Hoeffding power guarantees.</p><div class="table"><table><thead><tr><th>Claim</th><th>Independence profile</th><th class="num">Power</th><th class="num">Effective size</th><th class="num">Max certifiable rate</th><th class="num">Min detectable rate</th></tr></thead><tbody>{''.join(detectability_rows)}</tbody></table></div>''' if detectability_rows else ""}
<h2>Interpretation</h2><p><strong>SUPPORTED</strong> means the adjusted one-sided evidence places the upper bound at or below the threshold. <strong>REJECTED</strong> means the lower bound is strictly above it. These statements apply only under the recorded assumptions.</p>
{f"<h2>Warnings</h2><ul>{warnings}</ul>" if warnings else ""}
<h2>Provenance</h2><p>Empirical report <code>{html.escape(str(report.source["empirical_report_sha256"]))}</code><br>Certification plan <code>{html.escape(str(report.analysis["plan_sha256"]))}</code></p>
</main></body></html>"""


def _claim_rows(report: CertificationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim in report.claims:
        rejection = claim.rejection
        rows.append(
            {
                "claim_id": claim.claim_id,
                "title": claim.title,
                "target": json.dumps(claim.target, sort_keys=True),
                "threshold": claim.threshold,
                "estimate": claim.estimate,
                "support_upper": claim.support.bound,
                "support_raw_p_value": claim.support.raw_p_value,
                "support_adjusted_p_value": claim.support.adjusted_p_value,
                "support_local_alpha": claim.support.local_alpha,
                "rejection_lower": rejection.bound if rejection else None,
                "rejection_raw_p_value": rejection.raw_p_value if rejection else None,
                "rejection_adjusted_p_value": (rejection.adjusted_p_value if rejection else None),
                "rejection_local_alpha": rejection.local_alpha if rejection else None,
                "evidence": claim.evidence,
                "robustness": claim.robustness,
                "prompt_count": claim.prompt_count,
                "attempted_rollouts": claim.attempted_rollouts,
                "unclassified_outcomes": claim.unclassified_outcomes,
                "sensitivities": json.dumps(
                    [item.model_dump(mode="json") for item in claim.sensitivities],
                    sort_keys=True,
                ),
                "detectability": json.dumps(
                    [item.model_dump(mode="json") for item in claim.detectability],
                    sort_keys=True,
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_certification_report(
    report: CertificationReport, output: str | Path, *, overwrite: bool = False
) -> Path:
    directory = Path(output).resolve()
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"certification output exists and is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"certification output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    places = int(report._resolved_plan.get("report", {}).get("decimal_places", 6))
    (directory / "certification-report.json").write_text(
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
    (directory / "certification-report.md").write_text(_markdown(report, places), encoding="utf-8")
    (directory / "certification-report.html").write_text(_html(report, places), encoding="utf-8")
    _write_csv(directory / "certification-claims.csv", _claim_rows(report))
    (directory / "certification-plan.resolved.yaml").write_text(
        yaml.safe_dump(canonical_value(report._resolved_plan), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    generated = [
        "certification-claims.csv",
        "certification-plan.resolved.yaml",
        "certification-report.html",
        "certification-report.json",
        "certification-report.md",
    ]
    checksums = {name: _sha256(directory / name) for name in generated}
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory
