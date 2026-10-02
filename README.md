# Inspect Safety Stats

[![CI](https://github.com/aduncan001/safety_stats/actions/workflows/ci.yml/badge.svg)](https://github.com/aduncan001/safety_stats/actions/workflows/ci.yml)

`inspect-safety-stats` provides fixed-design statistical risk analysis for binary safety outcomes produced by [Inspect AI](https://inspect.aisi.org.uk/).

It is designed for hand-curated safety evaluations in which prompts, questions, tasks, groups, weights, thresholds, and rollout counts are fixed in advance. It quantifies stochastic rollout uncertainty without pretending that benchmark prompts are IID samples from a population.

## Current capabilities

- Exact one-sided Clopper–Pearson failure-rate bounds per fixed prompt.
- Direct finite-sample weighted KL/Chernoff, variance-adaptive empirical-Bernstein, and
  weighted-Hoeffding intervals for fixed-design mean risk, with conservative completion of
  unclassified outcomes.
- Simultaneous mean-risk intervals at every declared hierarchy node, with hierarchy-aware
  frontier Šidák calibration, a common linear confidence region, and creator-declared named
  independence profiles over arbitrary observation or manifest fields.
- Explicit two-sided versus per-direction family coverage semantics, with 95% simultaneous
  two-sided coverage as the reporting default.
- Missingness-robust exact bounds for attempted rollouts with unavailable classifications.
- End-to-end system safety as the default estimand, with stage-aware provider policy blocks.
- Exact zero-failure rollout planning.
- Exact fixed-horizon power and detectable-failure-rate planning.
- Pre-evaluation fixed-design rollout planning with hierarchy and policy weights.
- Creator-declared task → question → prompt hierarchy.
- Optional sampled-frame designs, with Horvitz–Thompson frame-mean estimation from
  declared inclusion probabilities.
- Evaluation-specific independence profiles kept separate from the aggregation hierarchy, with
  configurable cross-model coupling and block-overlap-aware tree calibration.
- Equal-prompt, nested-equal, and explicit policy weights.
- Weighted mean, worst-item, CVaR, top-k, entropic, threshold-exceedance, and threshold-excess risk.
- First-class fixed-design model/configuration comparison sets for three or more arms, with joint
  arm-risk regions, risk differences, risk ratios, relative reductions, simultaneous rank sets,
  hierarchy-frontier calibration, and named dependence sensitivities.
- Exact McNemar, Newcombe paired score-interval, and fixed-cluster bootstrap diagnostics.
- Bonferroni and weighted Bonferroni alpha allocation, with an intersection-union gate.
- Metrics-first reports with estimates, uncertainty bounds, hierarchy drill-down, and optional predeclared decision analysis.
- Judge-misclassification correction, mapping judge-reported risk onto true risk either
  under declared conditional accuracies or from human labels drawn within each judge stratum.
- Deterministic JSON, Markdown, CSV, plan, and checksum sidecars.

Anytime-valid confidence sequences, adaptive rollout allocation, live hooks, and early stopping are intentionally deferred.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

See [Installation](docs/installation.md) for supported versions and troubleshooting.
The hierarchy calibration and coverage proof are documented in
[Hierarchy-aware simultaneous inference](docs/hierarchy-simultaneous-inference.md).
Configuration and block mathematics are documented in
[Configuring hierarchy and independence](docs/independence-profiles.md).
Comparison estimands, common contrast/two-arm regions, and usage are documented in
[Fixed-design model and configuration comparisons](docs/model-comparisons.md).

## Quick start

Plan a zero-failure study:

```bash
inspect-safety-stats plan zero-failure --threshold 0.01 --alpha 0.05
```

Calculate exact detectable rates at a fixed horizon:

```bash
inspect-safety-stats plan detectable-rate \
  --trials 500 --threshold 0.05 --alpha 0.05 --power 0.90
```

Validate logs against a frozen design:

```bash
inspect-safety-stats validate \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/
```

Run the analysis:

```bash
inspect-safety-stats analyze \
  --plan examples/fixed_design/safety-plan.yaml \
  --logs logs/ \
  --output reports/example
```

The command writes:

```text
reports/example/
  report.json
  report.md
  report.html
  plan.resolved.yaml
  design.resolved.csv
  hierarchy-risks.csv
  comparisons.csv
  comparison-sets.csv
  observations.csv
  checksums.json
```

`report.html` is a self-contained interactive visualization. It needs no server or external
JavaScript and supports cohort/measure/independence-profile selection, group-risk intervals, joint
multi-arm risk tables, simultaneous rank sets, pairwise matrices, one collapsible hierarchy
selector for within-configuration risks and across-configuration effects, prompt search, and audit
details.

Regenerate or relocate a visualization from any existing report directory:

```bash
inspect-safety-stats visualize \
  --report reports/example \
  --output reports/example-dashboard.html
```

The CLI exits with code `0` after a completed analysis and `3` when analysis cannot run. Use
`--decision-exit-codes` to opt into decision-sensitive exit codes (`0` pass, `1` fail, and `2`
inconclusive).

## Python API

```python
from inspect_safety_stats import analyze, write_visualization

report = analyze(plan="safety-plan.yaml", logs="logs/")
report.write("reports/candidate")
write_visualization(report, "reports/candidate-dashboard.html")

for risk in report.group_risks:
    print(risk.group_id, risk.measure_id, risk.estimate, risk.lower, risk.upper)
```

Low-level functions are also public:

```python
from inspect_safety_stats.stats import (
    binomial_upper_bound,
    entropic_risk,
    weighted_cvar,
    zero_failure_sample_size,
)

n = zero_failure_sample_size(threshold=0.01, alpha=0.05)
upper = binomial_upper_bound(0, n, alpha=0.05)
cvar = weighted_cvar([0.001, 0.02, 0.20], [0.5, 0.3, 0.2], beta=0.8)
entropic = entropic_risk([0.001, 0.02, 0.20], [0.5, 0.3, 0.2], lambda_=10)
```

## Required inputs

An analysis uses:

1. completed Inspect `.eval` logs;
2. a CSV manifest with one row per fixed prompt;
3. a strict YAML or JSON analysis plan.

The score mapping is always explicit. The package never assumes whether `C`, `I`, `PASS`, `FAIL`, `0`, or `1` means a safety failure.

The default estimand is end-to-end system safety. An explicit provider policy block counts as a
safe outcome only when it occurs during target generation and no content was delivered. Grader
errors and unparseable scores remain unknown and widen conservative bounds. Set
`inputs.estimand: underlying_model` when provider blocks should remain unknown rather than count
as system-level safe outcomes.

## Safety scope

Every report states:

> Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts or establish safety for omitted hazards or deployment conditions.

The package distinguishes simultaneous uncertainty guarantees from diagnostic outputs. Diagnostic
resampling and post hoc slices should be interpreted with their stated assumptions. Optional threshold decisions can be enabled with
`report.include_decisions: true`; they are omitted from human-facing reports by default.

## Documentation

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Analysis plan reference](docs/plan-reference.md)
- [Python and command-line API reference](docs/api-reference.md)
- [Interpreting results](docs/interpretation.md)
- [Judge misclassification](docs/judge-misclassification.md)
- [Design-based inference for sampled item frames](docs/design-based-sampling.md)
- [Version 1 engineering design](docs/archive/v1-design.md) (archived; superseded by the shipped package)
- [Mathematical methods and justification](docs/statistical-methods.md)
- [Multi-model safety comparison: complete JailbreakBench-100 example](examples/real_jailbreakbench/README.md)
- [Multilingual configuration comparison: JailbreakBench-100 in English, French, German, and Maltese](examples/real_jailbreakbench/MULTILINGUAL.md)
- [Primary safety example: HarmBench direct-request multi-rollout evaluation](examples/real_harmbench/README.md)
- [Sampled-frame example: HarmBench from a probability sample](examples/sampled_harmbench/README.md)
- [Secondary calibration example: XSTest over-refusal](examples/real_xstest/README.md)

## License

MIT
