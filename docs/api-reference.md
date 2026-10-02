# API reference

The stable version 1 surface consists of the high-level analyzer, strict plan and report models,
the command-line interface, and pure statistical functions. Objects not documented here should
be treated as implementation details.

## High-level Python API

### `analyze`

```python
from inspect_safety_stats import analyze

report = analyze(
    plan="safety-plan.yaml",
    logs="logs/",
    parallel=False,
)
```

`plan` accepts an `AnalysisPlan` or a YAML/JSON path. `logs` accepts any local path, sequence of
paths, or loaded `EvalLog` object accepted by Inspect's dataframe readers. The function returns a
validated `SafetyReport`. It does not write files until `report.write(...)` is called.

### `validate`

```python
from inspect_safety_stats import validate

integrity = validate(plan="safety-plan.yaml", logs="logs/")
```

Runs plan-reference, manifest, provenance, cohort, rollout, and score-mapping checks without
making a deployment decision. `integrity.status` is `valid`, `degraded`, or `invalid`; individual diagnostics
carry a severity, stable code, message, and optional scope.

### `load_plan`

```python
from inspect_safety_stats import load_plan

plan = load_plan("safety-plan.yaml")
```

Parses a strict plan. Unknown keys and invalid method parameters are rejected. A relative manifest
path is resolved relative to the plan file.

### `SafetyReport.write`

```python
output = report.write("reports/run-001")
```

Writes deterministic JSON, a one-page Markdown summary (`report.md`), the full Markdown report
(`report-full.md`), a self-contained interactive HTML visualization, resolved-plan YAML,
resolved-design CSV, optional normalized observations, and a checksum manifest. A non-empty destination is rejected unless
`overwrite=True`.

### Exact threshold planning

Fixed-horizon planning functions are importable from `inspect_safety_stats.stats`:

```python
zero_failure_sample_size(*, threshold, alpha)
certification_sample_size(*, true_rate, threshold, alpha, power, max_trials=1_000_000)
certification_power(*, trials, true_rate, threshold, alpha)
maximum_certifying_failures(*, trials, threshold, alpha)
maximum_certifiable_failure_rate(*, trials, threshold, alpha, power)
minimum_detectable_failure_rate(*, trials, threshold, alpha, power)
minimum_rejecting_failures(*, trials, threshold, alpha)
rejection_power(*, trials, true_rate, threshold, alpha)
rejection_sample_size(*, true_rate, threshold, alpha, power, max_trials=1_000_000)
fixed_design_detectable_failure_rates(trials, weights, *, threshold, support_alpha,
                                      rejection_alpha, power, independence_unit="rollout")
```

The detectable-rate functions exactly invert the corresponding one-sided binomial test. They
return `None` when the requested direction is unattainable at the supplied horizon and alpha.
`fixed_design_detectable_failure_rates` gives the weighted-design analogue under an
independent-rollout Hoeffding guarantee, permitting heterogeneous prompt probabilities.

Use the multiplicity-adjusted local alpha when planning a claim that belongs to a family.

### `write_visualization`

```python
from inspect_safety_stats import write_visualization

write_visualization("reports/run-001", "reports/run-001-dashboard.html")
```

Accepts a `SafetyReport`, `report.json`, or report directory and writes one portable HTML file.
The report data, resolved hierarchy, and declared group weights are embedded locally; no server,
network access, or external JavaScript is required. When the source is a `SafetyReport`, `output`
is required. For a report path, omitting `output` writes `report.html` beside `report.json`.

Group-risk records expose `lower_one_sided_coverage` and `upper_one_sided_coverage` separately.
`simultaneous_coverage` is the Bonferroni lower bound for displaying both directions jointly; it
must not be read as the coverage of either individual direction. `coverage_mode`, `family_alpha`,
`directional_family_alpha`, `coverage_scope`, and `interval_method` make the construction
machine-readable. The default `two_sided` mode splits family alpha across the two directions.

## Statistical functions

All functions below are importable from `inspect_safety_stats.stats`. They are pure and do not read
logs or plans.

### Exact binomial bounds

```python
binomial_upper_bound(failures, trials, *, alpha)
binomial_lower_bound(failures, trials, *, alpha)
binomial_bounds(failures, trials, *, alpha_lower, alpha_upper)
```

These are one-sided Clopper–Pearson bounds. Counts must be integers, `trials` must be positive,
and alpha must lie strictly between zero and one.

### Direct fixed-design mean bounds

The plan-level pipeline uses named independence profiles and the generic bounded-block functions:

```python
weighted_bounded_hoeffding_mean_bounds(
    completion_lower, completion_upper, block_weights,
    *, alpha_lower, alpha_upper, independence_profile,
)
weighted_bounded_empirical_bernstein_mean_bounds(...)
weighted_bounded_kl_chernoff_mean_bounds(...)
```

Each aligned block value lies in `[0,1]`, block weights are nonnegative and sum to one, and blocks
are assumed mutually independent. See [Configuring hierarchy and independence](independence-profiles.md).
The count-based wrappers below remain useful as low-level mathematical conveniences for the two
canonical partitions; they are not the analysis-plan configuration interface.

```python
weighted_hoeffding_mean_bounds(
    failures,
    unclassified,
    trials,
    weights,
    *,
    alpha_lower,
    alpha_upper,
    independence_unit="rollout",
)
weighted_empirical_bernstein_mean_bounds(
    failures,
    unclassified,
    trials,
    weights,
    *,
    alpha_lower,
    alpha_upper,
    independence_unit="rollout",
)
weighted_kl_chernoff_mean_bounds(
    failures,
    unclassified,
    trials,
    weights,
    *,
    alpha_lower,
    alpha_upper,
    independence_unit="rollout",
)
```

The result includes the lower and upper risk bounds, all-safe and all-failure completion
estimates, directional half-widths, concentration variance
`sum(weights**2 / trials)`, and its reciprocal effective concentration size. The function assumes
independent weighted rollout contributions and fixed weights and rollout counts.

The empirical-Bernstein result additionally records certified upper variance proxies and the
maximum weighted increment. With `independence_unit="prompt_cluster"`, each within-prompt batch
mean is treated as one bounded independent unit, allowing arbitrary dependence within the batch.
This is a sensitivity analysis over the fixed prompts, not population inference.

`weighted_kl_chernoff_mean_bounds` uses Bernoulli-KL inversion when every atomic contribution has
equal weight and a rigorous maximum-increment weighted Poisson-Chernoff inversion otherwise. Its
`chernoff_kernel` result field records which construction was used. It is especially useful for
sparse failures and does not require equal prompt failure probabilities.

When hierarchy-region projection is enabled, each reported direct interval also records
`raw_lower`, `raw_upper`, `projection_method`, and `projection_constraint_count`. The displayed
endpoints are projections of the common fixed-design linear confidence region; the raw fields are
the concentration endpoints before hierarchy constraints are imposed.

With `hierarchy_calibration: frontier_sidak`, interval results additionally record
`calibration_method`, `calibration_frontier`, `calibration_frontier_alpha`, and
`calibration_frontier_size`. The public calibration helpers are:

```python
sidak_local_alpha(frontier_alpha, target_count)
calibrate_independent_frontiers(frontiers, *, directional_family_alpha)
```

An `IndependentFrontier` maps target identifiers to disjoint sets of atomic identifiers. The
calibrator divides directional family alpha across possibly dependent frontiers and computes the
equal Šidák endpoint alpha within each frontier. `aliases` may identify multiple report nodes that
represent the same weighted statistic; aliases receive the same calibration without another alpha
charge. Overlapping targets within one frontier are rejected.

### Fixed-design risk measures

```python
weighted_mean(values, weights)
maximum_risk(values, weights=None)
weighted_cvar(values, weights, *, beta)
top_k_mean(values, *, k)
entropic_risk(values, weights, *, lambda_)
threshold_exceedance(values, weights, *, threshold)
threshold_excess(values, weights, *, threshold)
```

Weights must be nonnegative and sum to one. Scalar or item-specific thresholds are accepted.
CVaR uses fractional boundary mass. The implementation evaluates entropic risk in a numerically
stable form. Any measure may be marked `diagnostic_only` in the plan to bar it from backing a claim.

### Fixed-design comparisons

The primary fixed-design comparison functions are:

```python
direct_risk_difference_bounds(
    candidate_failures,
    candidate_unclassified,
    candidate_trials,
    baseline_failures,
    baseline_unclassified,
    baseline_trials,
    weights,
    *,
    alpha_lower,
    alpha_upper,
    method="weighted_kl_chernoff",
    independence_unit="rollout",
)

direct_ratio_bounds(
    ...,
    estimand="risk_ratio",  # or "relative_risk_reduction"
)
```

The prompts and weights are aligned and fixed, while arm rollout counts may differ. Weighted
empirical-Bernstein and Hoeffding difference inference directly bound a transformed contrast.
Weighted KL/Chernoff uses a jointly calibrated two-arm rectangle, which retains sparse-failure
information and is shared with ratio inference. Unclassified outcomes receive adverse arm-specific
completions. The default rollout construction assumes independent attempted generations within
each arm. The direct transformed contrast also requires independence across arms; the two-arm
rectangle does not. `prompt_cluster` permits arbitrary dependence within a prompt batch, including
across the two arms, while requiring prompt batches to be independent.

Ratio intervals use a simultaneous rectangular confidence set for the two arm means. When the
baseline lower bound is zero, no finite upper risk-ratio bound exists; the API returns `None`
rather than an artificial finite value.

All three direct engines—`weighted_kl_chernoff`, `weighted_empirical_bernstein`, and
`weighted_hoeffding`—support rollout and prompt-cluster comparison inference. The plan-driven
hierarchy implementation additionally exposes the projection primitives:

```python
project_contrast_hierarchy_confidence_region(
    constraints,
    targets,
    *,
    variable_count,
)

project_ratio_hierarchy_confidence_region(
    candidate_constraints,
    baseline_constraints,
    targets,
    *,
    variable_count,
)
```

The first projects a common region over prompt contrasts in `[-1, 1]`. The package also exports
`project_two_arm_difference_hierarchy_confidence_region` for a linear difference projection from
arm constraints. The ratio function projects from the same kind of two-arm region with a
Charnes--Cooper linear-fractional transformation. Returned objects report feasibility, projected
intervals, constraint count, and any solver message.

The plan-driven `comparison_sets` compiler applies these primitives to two or more arms. It
intersects every calibrated constraint for a shared arm, projects that arm hierarchy once, and
derives all requested differences and ratios from the common arm projections. This orchestration
is intentionally exposed through `analyze()` rather than a separate low-level convenience
function because its coverage depends on plan-wide multiplicity calibration.

For plan configuration and report-field semantics, see
[Fixed-design model and configuration comparisons](model-comparisons.md).

### Judge misclassification

```python
JudgeCalibration(scenario_id, rho_1_lower, rho_1_upper, rho_0_lower, rho_0_upper, source)
JudgeStratumCounts(labelled, truly_unsafe, stratum_size)
estimate_judge_calibration(*, scenario_id, unsafe_stratum, safe_stratum, alpha)
correct_judge_risk(*, observed_lower, observed_upper, observed_estimate, calibration)
```

Maps a judge-system risk interval onto the true-risk scale via
`R* = R * rho_1 + (1 - R) * rho_0`. The identity is multilinear, so the returned bounds are
the exact extrema over the declared box. It applies unchanged to any positively homogeneous,
translation-equivariant risk measure. `estimate_judge_calibration` derives the same box from
human labels by exact Clopper-Pearson bounds within each judge stratum. See
[Judge misclassification](judge-misclassification.md).

### Multiplicity

```python
bonferroni_allocate(alpha, count, weights=None)
```

Allocation supports equal or positive prospective weights. Simultaneous interval families use
Bonferroni across dependent frontiers and exact Sidak product calibration among disjoint siblings;
see [Hierarchy-aware simultaneous inference](hierarchy-simultaneous-inference.md).

## Command-line interface

```text
inspect-safety-stats init --logs LOGS --failure-values VALUE [--group-by FIELD] [--baseline MODEL] [--output DIR]
inspect-safety-stats analyze --plan PLAN --logs LOGS --output DIRECTORY
inspect-safety-stats validate --plan PLAN --logs LOGS
inspect-safety-stats visualize --report REPORT_OR_DIRECTORY [--output REPORT.html]
inspect-safety-stats plan zero-failure --threshold P --alpha A
inspect-safety-stats plan certification --threshold P --anticipated-rate Q --alpha A --power B
inspect-safety-stats plan detectable-rate --trials N --threshold P --alpha A --power B
inspect-safety-stats plan judge-labels --judge-risk-upper U --target T [--cohorts K] [--rho-1 R1] [--rho-0 R0]
inspect-safety-stats judge sample --report DIRECTORY --judge-unsafe N1 --judge-safe N0 --seed S [--output DIR]
inspect-safety-stats migrate-plan --plan OLD --output NEW
inspect-safety-stats hash-plan --plan PLAN
inspect-safety-stats schema plan
inspect-safety-stats schema report
inspect-safety-stats version
```

`init` writes `design.csv` and a minimal `safety-plan.yaml` from completed logs. It never infers
the score mapping: pass `--failure-values` (or `--failure-threshold` for numeric scores), and
the remaining observed values are taken as safe unless `--safe-values` lists them. The plan is
`exploratory` because it is written after the logs exist.

Repeat `--logs` to select multiple files or directories. By default, `analyze` exits `0` after a
completed analysis regardless of any internally computed threshold decision; `validate` exits `3`
for invalid input. Pass `--decision-exit-codes` to opt into `0` for `PASS`, `1` for `FAIL`, and
`2` for `INCONCLUSIVE`; see [Confirmatory mode](confirmatory-mode.md).

## Starter plans and labelling samples

```python
from inspect_safety_stats.scaffold import scaffold_plan
from inspect_safety_stats.judge_labels import draw_label_sample

result = scaffold_plan("logs/", "analysis/", failure_values=["FAIL"], group_by="category")
sample = draw_label_sample("report/", "labels/", judge_unsafe=40, judge_safe=200, seed=7)
```

`scaffold_plan` is the function behind `init` and returns the paths it wrote, the cohorts and
the resolved score mapping. `draw_label_sample` is the function behind `judge sample`; it writes
a blinded, shuffled `labels.csv` and a `labels.sampling.json` audit record.

## Precision and labelling budgets

```python
from inspect_safety_stats.stats.planning import (
    rollouts_for_half_width,
    rollout_scale_for_half_width,
    judge_label_plan,
)

rollouts_for_half_width(rate=0.05, half_width=0.02, alpha=0.025)
rollout_scale_for_half_width(rate=0.1, observed_half_width=0.19, observed_trials=80,
                             target_half_width=0.05, alpha=0.0025)
judge_label_plan(judge_risk_upper=0.05, target_upper=0.08, alpha=0.05, cohorts=2)
```

`rollouts_for_half_width` returns the smallest number of equally weighted rollouts at which both
exact one-sided bounds lie within `half_width` of an assumed rate; at rate zero it reproduces
zero-failure planning. `rollout_scale_for_half_width` anchors on an interval actually observed,
whatever bound produced it, and returns the factor by which rollouts must grow to reach a target
width. `judge_label_plan` returns the cheapest judge-unsafe and judge-safe label counts per cohort
for which a validated correction would bound true risk below a target, or `None` when no number
of labels can.

## Compatibility and stability

The runtime supports Python 3.10 or newer and Inspect AI `>=0.3.259,<0.4`. Plans declare
`schema_version: "2.0"`; a `1.0` plan is refused with a pointer to `migrate-plan`. Reports are
strict models. New fields or methods may be added compatibly within the package's `0.x`
development series, but consumers should validate `schema_version` before automated use.
