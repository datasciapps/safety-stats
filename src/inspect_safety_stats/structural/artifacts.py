# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from inspect_safety_stats.config.canonicalize import canonical_value
from inspect_safety_stats.schema.structural_reports import StructuralReport


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _interval(summary: Any) -> str:
    return f"{_number(summary.lower)}–{_number(summary.upper)}"


def _path_label(path: dict[str, str]) -> str:
    return " → ".join(path.values())


def _markdown(report: StructuralReport) -> str:
    fit = report.primary_fit
    lines = [
        f"# {report.analysis.get('title', 'Structural safety landscape')}",
        "",
        "**Certification eligible:** `false`  ",
        f"**Model status:** `{report.status}`  ",
        f"**Primary model:** `{fit.link}` link with `{fit.prompt_frailty}` prompt frailty",
        "",
        "## Interpretation",
        "",
        str(report.scope["interpretation"]),
        "",
        (
            "All ranges in this report are posterior credible intervals conditional on the "
            "hierarchical model and priors. They are not frequentist confidence bounds and do "
            "not participate in the empirical safety decision."
        ),
        "",
        "## Declared hierarchy",
        "",
        f"- Full design and model hierarchy: `{' → '.join(report.data['hierarchy'])}`",
        f"- Shared-effect hierarchy: `{' → '.join(report.data['effect_hierarchy'])}`",
        f"- Levels represented by terminal prompt frailty: `{' → '.join(report.data['prompt_frailty_levels'])}`",
        "",
        (
            "A terminal level that is one-to-one with prompts cannot have a separately "
            "identified variance component; it remains in the declared hierarchy and is "
            "represented by the same terminal frailty."
        ),
        "",
        "## Weighted design risk",
        "",
        "| Quantity | Posterior mean | Posterior median | Credible interval |",
        "|---|---:|---:|---:|",
        (
            f"| Model weighted mean | {_number(fit.weighted_mean_risk.mean)} | "
            f"{_number(fit.weighted_mean_risk.median)} | "
            f"{_interval(fit.weighted_mean_risk)} |"
        ),
        (
            f"| Prompt frailty scale | {_number(fit.prompt_frailty_scale.mean)} | "
            f"{_number(fit.prompt_frailty_scale.median)} | "
            f"{_interval(fit.prompt_frailty_scale)} |"
        ),
        "",
        "## Vulnerable-prompt mass",
        "",
        "| Risk threshold | Posterior median mass | Credible interval | Expected mass |",
        "|---:|---:|---:|---:|",
    ]
    for vulnerable in fit.vulnerable_mass:
        lines.append(
            f"| {_number(vulnerable.threshold)} | {_number(vulnerable.mass.median)} | "
            f"{_interval(vulnerable.mass)} | "
            f"{_number(vulnerable.expected_exceedance_mass)} |"
        )
    lines.extend(
        [
            "",
            "## Risk concentration",
            "",
            "| Highest-risk prompt mass | Median risk share | Share interval | Absolute tail risk | Near-zero denominator |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for concentration in fit.concentration:
        lines.append(
            f"| {_number(concentration.prompt_mass)} | "
            f"{_number(concentration.risk_share.median if concentration.risk_share else None)} | "
            f"{_interval(concentration.risk_share) if concentration.risk_share else 'undefined'} | "
            f"{_interval(concentration.absolute_tail_risk)} | "
            f"{_number(concentration.near_zero_total_risk_probability)} |"
        )
    diagnostics = fit.diagnostics
    lines.extend(
        [
            "",
            "## Marginal posterior risk by hierarchy node",
            "",
            (
                "Each posterior draw averages prompt risks within the node using the resolved "
                "fixed-design weights, normalized within that node."
            ),
            "",
            "| Level | Node path | Design weight | Prompts | Posterior mean | Posterior median | Credible interval |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for marginal in fit.hierarchy_marginal_risks:
        lines.append(
            f"| {marginal.level} | {_path_label(marginal.path)} | "
            f"{_number(marginal.design_weight)} | {marginal.prompt_count} | "
            f"{_number(marginal.risk.mean)} | {_number(marginal.risk.median)} | "
            f"{_interval(marginal.risk)} |"
        )
    lines.extend(
        [
            "",
            "## Hierarchy effects",
            "",
            "| Level | Category | Median intensity multiplier | Credible interval |",
            "|---|---|---:|---:|",
        ]
    )
    for effect in fit.hierarchy_effects:
        lines.append(
            f"| {effect.level} | {effect.category} | "
            f"{_number(effect.intensity_multiplier.median)} | "
            f"{_interval(effect.intensity_multiplier)} |"
        )
    lines.extend(
        [
            "",
            "## Model diagnostics",
            "",
            f"- Sampler status: `{diagnostics.status}`",
            f"- Divergences: {diagnostics.divergences}",
            f"- Maximum R-hat: {_number(diagnostics.max_rhat)}",
            f"- Minimum bulk ESS: {_number(diagnostics.min_bulk_ess)}",
            f"- Minimum tail ESS: {_number(diagnostics.min_tail_ess)}",
            f"- Prompt-level cross-validation: `{'enabled' if fit.cross_validation.enabled else 'disabled'}`",
            f"- Mean held-out log predictive density: {_number(fit.cross_validation.mean_log_predictive_density)}",
            f"- Mean held-out Brier score: {_number(fit.cross_validation.mean_brier_score)}",
            "",
            "## Prompt posterior table",
            "",
            "The content-free prompt summary is in `prompt-posterior.csv`. Posterior draws are in `posterior.nc`.",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = [
        *report.warnings,
        *diagnostics.warnings,
        *fit.cross_validation.warnings,
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["No warnings."])
    return "\n".join(lines) + "\n"


def _safe_json(value: Any) -> str:
    return (
        json.dumps(canonical_value(value), ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_structural_visualization(report: StructuralReport) -> str:
    payload = _safe_json(report.model_dump(mode="json"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structural safety landscape</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#9da7b3;--blue:#79a7ff;--amber:#e3b341}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1300px;margin:auto;padding:28px}} h1{{margin:0 0 8px;font-size:30px}} h2{{margin-top:34px}}
.notice{{border:1px solid #6e5a21;background:#2b2412;padding:14px 16px;border-radius:10px;color:#f2cc60}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.label{{color:var(--muted);font-size:13px}} .value{{font-size:24px;font-weight:700;margin:5px 0}} .interval{{color:var(--blue)}}
table{{border-collapse:collapse;width:100%;background:var(--panel)}} th,td{{border-bottom:1px solid var(--line);padding:9px;text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{position:sticky;top:0;background:#1d2430}}
.table-wrap{{max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:10px}} input{{width:100%;max-width:420px;background:var(--panel);border:1px solid var(--line);color:var(--text);padding:10px;border-radius:8px;margin-bottom:10px}}
.bar{{height:10px;background:#273244;border-radius:6px;overflow:hidden}} .bar span{{display:block;height:100%;background:var(--blue)}} code{{color:#c9d1d9}}
</style></head><body><main>
<h1 id="title"></h1><p id="scope" class="label"></p>
<div class="notice"><strong>Non-certifying structural analysis.</strong> Credible intervals are conditional on the model and priors. They are not confidence bounds and do not affect the empirical safety decision.</div>
<h2>Overview</h2><div class="grid" id="overview"></div>
<h2>Declared hierarchy</h2><div class="grid" id="hierarchy"></div>
<h2>Vulnerable-prompt mass</h2><div class="grid" id="vulnerable"></div>
<h2>Risk concentration</h2><div class="grid" id="concentration"></div>
<h2>Marginal posterior risk by hierarchy node</h2><p class="label">Joint posterior prompt risks averaged with fixed-design weights normalized within each node.</p><div class="table-wrap"><table><thead><tr><th>Level</th><th>Node path</th><th>Design weight</th><th>Prompts</th><th>Mean risk</th><th>Credible interval</th></tr></thead><tbody id="marginals"></tbody></table></div>
<h2>Prompt posterior risks</h2><input id="search" placeholder="Filter by prompt identifier" aria-label="Filter prompts"><div class="table-wrap"><table><thead><tr><th>Prompt</th><th>Weight</th><th>Mean</th><th>Median</th><th>Credible interval</th></tr></thead><tbody id="prompts"></tbody></table></div>
<h2>Diagnostics</h2><div class="grid" id="diagnostics"></div>
</main><script>
const report={payload}; const fit=report.primary_fit;
const pct=x=>x==null?'—':(100*x).toFixed(2)+'%'; const num=x=>x==null?'—':Number(x).toFixed(4);
const interval=s=>s?pct(s.lower)+'–'+pct(s.upper):'undefined';
const esc=s=>String(s).replace(/[&<>"']/g,c=>c==='&'?'&amp;':c==='<'?'&lt;':c==='>'?'&gt;':c==='"'?'&quot;':'&#39;');
document.getElementById('title').textContent=report.analysis.title;
document.getElementById('scope').textContent=report.scope.interpretation;
const card=(label,value,detail='')=>`<div class="card"><div class="label">${{label}}</div><div class="value">${{value}}</div><div class="interval">${{detail}}</div></div>`;
document.getElementById('overview').innerHTML=card('Model status',report.status)+card('Weighted mean risk',pct(fit.weighted_mean_risk.mean),interval(fit.weighted_mean_risk))+card('Prompt frailty scale',num(fit.prompt_frailty_scale.mean),num(fit.prompt_frailty_scale.lower)+'–'+num(fit.prompt_frailty_scale.upper))+card('Prompts',report.data.prompt_count,'Fixed weighted design');
document.getElementById('hierarchy').innerHTML=card('Full design and model hierarchy',report.data.hierarchy.join(' → '),'Single source of truth')+card('Shared-effect hierarchy',report.data.effect_hierarchy.join(' → '),'Levels eligible for partial pooling')+card('Terminal prompt frailty',report.data.prompt_frailty_levels.join(' → '),'One-to-one terminal levels share one identifiable effect');
document.getElementById('vulnerable').innerHTML=fit.vulnerable_mass.map(x=>card('Risk above '+pct(x.threshold),pct(x.mass.median),interval(x.mass))).join('');
document.getElementById('concentration').innerHTML=fit.concentration.map(x=>card('Highest-risk '+pct(x.prompt_mass)+' of prompt mass',x.risk_share?pct(x.risk_share.median):'undefined',x.risk_share?interval(x.risk_share):'Total risk near zero')).join('');
document.getElementById('marginals').innerHTML=fit.hierarchy_marginal_risks.map(x=>`<tr><td>${{esc(x.level)}}</td><td>${{esc(Object.values(x.path).join(' → '))}}</td><td>${{pct(x.design_weight)}}</td><td>${{x.prompt_count}}</td><td>${{pct(x.risk.mean)}}</td><td>${{interval(x.risk)}}</td></tr>`).join('');
const rows=fit.prompt_risks; const body=document.getElementById('prompts');
function renderRows(q=''){{const query=q.toLowerCase();body.innerHTML=rows.filter(x=>x.prompt_id.toLowerCase().includes(query)).map(x=>`<tr><td><code>${{esc(x.prompt_id)}}</code></td><td>${{pct(x.weight)}}</td><td>${{pct(x.risk.mean)}}</td><td>${{pct(x.risk.median)}}</td><td>${{interval(x.risk)}}</td></tr>`).join('')}}
renderRows(); document.getElementById('search').addEventListener('input',e=>renderRows(e.target.value));
const d=fit.diagnostics,cv=fit.cross_validation;document.getElementById('diagnostics').innerHTML=card('Sampler',d.status,d.divergences+' divergences')+card('Maximum R-hat',num(d.max_rhat),'Target < 1.01')+card('Minimum bulk ESS',num(d.min_bulk_ess))+card('Held-out Brier score',num(cv.mean_brier_score),cv.enabled?cv.folds+' prompt-level folds':'Disabled');
</script></body></html>"""


def _prompt_rows(report: StructuralReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report.primary_fit.prompt_risks:
        row = {
            "prompt_id": item.prompt_id,
            "weight": item.weight,
            "risk_mean": item.risk.mean,
            "risk_median": item.risk.median,
            "risk_sd": item.risk.sd,
            "risk_lower": item.risk.lower,
            "risk_upper": item.risk.upper,
        }
        row.update(
            {
                f"p_above_{threshold}": value
                for threshold, value in item.exceedance_probability.items()
            }
        )
        rows.append(row)
    return rows


def _audit_rows(report: StructuralReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vulnerable in report.primary_fit.vulnerable_mass:
        rows.append(
            {
                "metric": "vulnerable_prompt_mass",
                "parameter": vulnerable.threshold,
                "mean": vulnerable.mass.mean,
                "median": vulnerable.mass.median,
                "lower": vulnerable.mass.lower,
                "upper": vulnerable.mass.upper,
            }
        )
    for concentration in report.primary_fit.concentration:
        rows.append(
            {
                "metric": "risk_concentration_share",
                "parameter": concentration.prompt_mass,
                "mean": concentration.risk_share.mean if concentration.risk_share else None,
                "median": concentration.risk_share.median if concentration.risk_share else None,
                "lower": concentration.risk_share.lower if concentration.risk_share else None,
                "upper": concentration.risk_share.upper if concentration.risk_share else None,
            }
        )
    return rows


def _hierarchy_rows(report: StructuralReport) -> list[dict[str, Any]]:
    return [
        {
            "level": effect.level,
            "category": effect.category,
            "log_effect_mean": effect.log_intensity_effect.mean,
            "log_effect_median": effect.log_intensity_effect.median,
            "log_effect_lower": effect.log_intensity_effect.lower,
            "log_effect_upper": effect.log_intensity_effect.upper,
            "multiplier_mean": effect.intensity_multiplier.mean,
            "multiplier_median": effect.intensity_multiplier.median,
            "multiplier_lower": effect.intensity_multiplier.lower,
            "multiplier_upper": effect.intensity_multiplier.upper,
        }
        for effect in report.primary_fit.hierarchy_effects
    ]


def _hierarchy_marginal_rows(report: StructuralReport) -> list[dict[str, Any]]:
    return [
        {
            "level": marginal.level,
            "node_path": _path_label(marginal.path),
            "design_weight": marginal.design_weight,
            "prompt_count": marginal.prompt_count,
            "risk_mean": marginal.risk.mean,
            "risk_median": marginal.risk.median,
            "risk_sd": marginal.risk.sd,
            "risk_lower": marginal.risk.lower,
            "risk_upper": marginal.risk.upper,
        }
        for marginal in report.primary_fit.hierarchy_marginal_risks
    ]


def write_structural_report(
    report: StructuralReport, output: str | Path, *, overwrite: bool = False
) -> Path:
    directory = Path(output).resolve()
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"structural output exists and is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"structural report directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "structural-report.json").write_text(
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
    (directory / "structural-report.md").write_text(_markdown(report), encoding="utf-8")
    (directory / "structural-report.html").write_text(
        render_structural_visualization(report), encoding="utf-8"
    )
    _write_csv(directory / "model-frame.csv", report._model_frame)
    _write_csv(directory / "prompt-posterior.csv", _prompt_rows(report))
    _write_csv(directory / "audit-metrics.csv", _audit_rows(report))
    _write_csv(directory / "hierarchy-effects.csv", _hierarchy_rows(report))
    _write_csv(
        directory / "hierarchy-marginal-risks.csv",
        _hierarchy_marginal_rows(report),
    )
    (directory / "model.resolved.yaml").write_text(
        yaml.safe_dump(canonical_value(report._resolved_model), sort_keys=True),
        encoding="utf-8",
    )
    (directory / "diagnostics.json").write_text(
        json.dumps(
            canonical_value(
                {
                    "primary": {
                        "sampler": report.primary_fit.diagnostics.model_dump(mode="json"),
                        "prior_predictive": report.primary_fit.prior_predictive.model_dump(
                            mode="json"
                        ),
                        "posterior_predictive": report.primary_fit.posterior_predictive.model_dump(
                            mode="json"
                        ),
                        "cross_validation": report.primary_fit.cross_validation.model_dump(
                            mode="json"
                        ),
                    },
                    "sensitivities": [
                        {
                            "fit_id": fit.fit_id,
                            "sampler": fit.diagnostics.model_dump(mode="json"),
                        }
                        for fit in report.sensitivity_fits
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated = [
        "audit-metrics.csv",
        "diagnostics.json",
        "hierarchy-effects.csv",
        "hierarchy-marginal-risks.csv",
        "model-frame.csv",
        "model.resolved.yaml",
        "prompt-posterior.csv",
        "structural-report.html",
        "structural-report.json",
        "structural-report.md",
    ]
    if report._posterior is not None:
        report._posterior.to_netcdf(directory / "posterior.nc")
        generated.append("posterior.nc")
    for fit_id, posterior in sorted(report._sensitivity_posteriors.items()):
        name = f"posterior-{fit_id}.nc"
        posterior.to_netcdf(directory / name)
        generated.append(name)
    checksums = {name: _sha256(directory / name) for name in sorted(generated)}
    (directory / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def write_structural_visualization(
    source: str | Path,
    output: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    candidate = Path(source).resolve()
    report_path = candidate / "structural-report.json" if candidate.is_dir() else candidate
    if not report_path.is_file():
        raise FileNotFoundError(f"structural report JSON does not exist: {report_path}")
    report = StructuralReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    destination = (
        Path(output).resolve()
        if output is not None
        else report_path.parent / "structural-report.html"
    )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"structural visualization already exists: {destination}")
    destination.write_text(render_structural_visualization(report), encoding="utf-8")
    checksums_path = report_path.parent / "checksums.json"
    if destination.parent == report_path.parent and checksums_path.is_file():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        checksums[destination.name] = _sha256(destination)
        checksums_path.write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return destination
