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

Results from another evaluation system are passed as `source` instead of `logs`
(experimental). `RolloutFiles` reads a runs file and a rollouts file in the neutral format of
[Rollout files](rollout-files.md):

```python
from inspect_safety_stats import analyze
from inspect_safety_stats.ingest import RolloutFiles

report = analyze(
    plan="safety-plan.yaml",
    source=RolloutFiles(runs="runs.jsonl", rollouts="rollouts.jsonl"),
)
```

Any object with a `read()` method that returns the package's `SourceData` records can serve as a
source; `inspect_safety_stats.ingest.EvaluationSource` is the protocol. Passing both `logs` and
`source` is an error. `validate` takes the same `source` keyword.

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

Writes deterministic JSON, Markdown, a self-contained interactive HTML visualization,
resolved-plan YAML, resolved-design CSV, optional normalized observations, and a checksum
manifest. A non-empty destination is rejected unless
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
worst_of_k_risk(values, weights, *, k)
worst_of_k_estimate(failures, rollouts, weights, *, k)
entropic_risk(values, weights, *, lambda_)
threshold_exceedance(values, weights, *, threshold)
threshold_excess(values, weights, *, threshold)
```

Weights must be nonnegative and sum to one. Scalar or item-specific thresholds are accepted.
CVaR uses fractional boundary mass. The implementation evaluates entropic risk in a numerically
stable form. `worst_of_k_risk` requires probability-valued inputs in `[0, 1]` and reports the
weighted chance that at least one of `k` independent attempts fails. Bounds use it on prompt
bounds, which it preserves because it is monotone. `worst_of_k_estimate` estimates the same
quantity from integer failure and rollout counts. It uses the unbiased U-statistic
`1 - C(n - f, k) / C(n, k)` per prompt, where plugging `f / n` into `worst_of_k_risk` would
understate the risk. Every prompt needs at least `k` rollouts. Any measure may be marked
`diagnostic_only` in the plan to bar it from backing a claim.

### Paired and clustered comparisons

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

Paired rollout diagnostics remain available:

```python
exact_mcnemar(candidate, baseline)
newcombe_paired_interval(
    candidate,
    baseline,
    *,
    alpha_lower=0.05,
    alpha_upper=0.05,
)
```

Inputs are equal-length Boolean sequences in which `True` means failure. The orientation is
candidate minus baseline. `exact_mcnemar` returns the full 2×2 table and three exact conditional
p-values. `newcombe_paired_interval` implements Newcombe method 10 with separate one-sided error
levels.

`paired_cluster_bootstrap` is available from
`inspect_safety_stats.stats.clusters`. It resamples whole creator-declared clusters while
preserving pairs. It is a fixed-design sensitivity analysis, not a population interval.

### Design-based sampling

```python
InclusionDesign(design_weights, inclusion_probabilities, sampled)
horvitz_thompson_mean_bounds(design, completion_lower, completion_upper, *,
                             alpha_lower, alpha_upper, method="weighted_empirical_bernstein")

FixedSizeDesign(stratum_index, frame_weights, draws, sampled)
fixed_size_mean_bounds(design, completion_lower, completion_upper, *,
                       alpha_lower, alpha_upper, method="weighted_empirical_bernstein")
```

Bounds a frame-wide weighted mean from an item sample with known inclusion probabilities.
The Horvitz-Thompson estimator is rescaled into an ordinary bounded weighted mean, so the
existing concentration engines apply unchanged. `horvitz_thompson_mean_bounds` assumes
independent inclusion; `fixed_size_mean_bounds` covers stratified sampling without
replacement via Hoeffding's reduction. Both report
`inflation` and `maximum_weight`, which govern the precision cost. Not yet called by the
analysis pipeline; see
[Design-based inference for sampled item frames](design-based-sampling.md).

### Judge misclassification

```python
JudgeCalibration(scenario_id, rho_1_lower, rho_1_upper, rho_0_lower, rho_0_upper, source,
                 alpha=None, completed_strata=())   # strata completed as [0, 1]
JudgeStratumCounts(labelled, truly_unsafe, stratum_size)
estimate_judge_calibration(*, scenario_id, unsafe_stratum, safe_stratum, alpha)
completion_warning(calibration)   # message naming completed strata, or None
correct_judge_risk(*, observed_lower, observed_upper, observed_estimate, calibration)
delta_method_corrected_risk(*, observed_estimate, observed_variance, unsafe_stratum,
                            safe_stratum, alpha)   # asymptotic reference, diagnostic
TruthStratumCounts(labelled, judge_positive)
noisy_valid_certification(*, judge_reported_lower, judge_reported_upper, judge_reported_alpha,
                          truly_unsafe, truly_safe, calibration_alpha, acceptable_upper,
                          unacceptable_lower)   # finite-sample threshold test
```

### Judge panels

```python
compose_verdicts(verdicts, rule)          # any | all | majority; None when undecidable
effective_votes(blocks)                   # Kish count over declared dependence blocks
pairwise_agreement(first, second)         # AgreementSummary(compared, disagreements, agreement, kappa)
JudgeAccuracyBox(sensitivity_lower, sensitivity_upper, specificity_lower, specificity_upper)
composite_accuracy_box(judges, blocks, rule)   # any | all; Frechet within blocks, product across
identified_risk_set(observed_lower, observed_upper, box)  # IdentifiedRiskSet(lower, upper, identified)
```

Bookkeeping for declared judge panels; see the `inputs.judges` section of the plan reference.
Composites are undecided when a missing vote could still change them or on a `majority` tie, and
agreement statistics are descriptive diagnostics only.

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
inspect-safety-stats analyze --plan PLAN --logs LOGS --output DIRECTORY
inspect-safety-stats validate --plan PLAN --logs LOGS
inspect-safety-stats analyze --plan PLAN --runs RUNS --rollouts ROLLOUTS --output DIRECTORY
inspect-safety-stats validate --plan PLAN --runs RUNS --rollouts ROLLOUTS
inspect-safety-stats visualize --report REPORT_OR_DIRECTORY [--output REPORT.html]
inspect-safety-stats plan zero-failure --threshold P --alpha A [--family-size M]
inspect-safety-stats plan certification --threshold P --anticipated-rate Q --alpha A --power B [--family-size M]
inspect-safety-stats plan detectable-rate --trials N --threshold P --alpha A --power B [--family-size M]
inspect-safety-stats plan fixed-design --manifest DESIGN.csv --threshold P [--alpha A] [--power B]
    [--rollouts N] [--weights STRATEGY] [--hierarchy LEVELS] [--weight-column COLUMN]
    [--independence-unit UNIT] [--family-size M]
inspect-safety-stats hash-plan --plan PLAN
inspect-safety-stats schema plan
inspect-safety-stats schema report
inspect-safety-stats version
```

Every `plan` subcommand accepts `--family-size M`, which plans at the Bonferroni share
`alpha / M`. Use it to match the `equal_bonferroni` allocation that `analyze` applies to a
family of `M` simultaneous claims, so a planned rollout count is sufficient at analysis time
rather than only in isolation. `plan fixed-design` reads the design manifest itself: weights
follow the declared strategy (`nested_equal` by default, `equal_prompt`, or `explicit` with
`--weight-column`, which is rejected for any other strategy), and rollout counts come from the `expected_rollouts` column unless
`--rollouts` supplies a uniform override. Its guarantees are the independent-rollout weighted
Hoeffding construction of `fixed_design_detectable_failure_rates`; pass
`--independence-unit prompt_cluster` to plan under prompt-level independence instead.

Repeat `--logs` to select multiple files or directories. `--runs` and `--rollouts` name the two
files of a neutral rollout source instead of Inspect logs, and are given together
(experimental; see [Rollout files](rollout-files.md)).

Exit codes of `analyze`:

| Situation | Default | With `--decision-exit-codes` |
|---|---|---|
| Integrity `valid` or `degraded` | `0` | `0` PASS, `1` FAIL, `2` INCONCLUSIVE |
| Integrity `invalid` (the report is still written) | `3` | `3` |
| Integrity `invalid` with `--allow-invalid` | `0` | `2`, because an invalid analysis is never PASS |
| The analysis cannot run, for example an unreadable plan or log | `3` | `3` |

`validate` exits `3` for invalid input and `0` otherwise.

## Compatibility and stability

The runtime supports Python 3.10 or newer. Reading Inspect logs needs Inspect AI
`>=0.3.259,<0.4`, which the `inspect` extra installs. This release reads analysis plans of
schema version `2.0` and `3.0` (see [the plan reference](plan-reference.md#schema-versions)) and
writes reports of schema version `3.0`. Plans and reports are strict: unknown fields are errors,
and a plan declaring any other version, `1.0` included, is refused. New fields or methods may
be added compatibly within the package's
`0.x` development series, but consumers should validate `schema_version` before automated use.

## Generative evaluation

`inspect_safety_stats.generative` carries its own public surface, listed in its `__all__`
and guarded by a snapshot test. Names outside that list are internal and may change.

```python
from inspect_safety_stats.generative import (
    load_generative_plan, compile_taxonomy,   # the plan and the tree it compiles to
    forecast, rewrites_for_bound,             # deciding a budget before generating anything
    RewriteKernel, RewriteRequest, RewriteOutcome,   # the piece you supply
    materialize_run, analyze_run,             # the two-phase workflow
    GenerativeReport,
)
```

**The workflow.** Two phases, because an Inspect run happens between them:

```python
run = materialize_run("plan.yaml", "artifacts/", kernel="mypkg:build_kernel", resume=False)
# evaluate run.dataset with Inspect, then
report = analyze_run("plan.yaml", "artifacts/", "logs/", slices=("language=maltese",))
report.write("report/")
```

**Deciding a budget** costs nothing and needs no data:

```python
plan = load_generative_plan("plan.yaml")
taxonomy = compile_taxonomy(plan, base_path=".")
projection = forecast(taxonomy, plan, total_rewrites=1200, assumed_rate=0.03)
projection.root_upper, projection.levels, projection.expected_unsampled_nodes
```

**Supplying a kernel** requires only the protocol: any object with `kernel_id` and a
`rewrite(request) -> RewriteOutcome` method. `DeterministicFakeKernel` stands in for it in
tests; [`examples/generative_live/rewriter.py`](../examples/generative_live/rewriter.py) is a
provider-backed reference implementation.

**Conversations (experimental).** Under `rewrite.output: conversation` a kernel returns
`RewriteOutcome(status="valid", messages=...)`, a tuple of `ChatTurn(role, content)` with
the roles `user` and `assistant` and, unless the plan attaches system prompts, `system`.
`conversation_problem(turns)` says why a conversation cannot be evaluated, or returns
`None`, so a kernel parsing one out of model output can report a malformed reply as
`invalid`. `RewriteRequest.system_prompt` and `RewriteRequest.conversation_turns` carry the
plan's system-prompt and turns arms for the request's path. See
[conversations and system prompts](generative-evaluation.md#conversations-and-system-prompts-experimental).

**Composing a flow by hand** — when the two-phase runner does not fit — is supported through
`plan_design`, `draw_selections`, `execute_rewrites`, `join_outcomes`, `estimate_tree`,
`estimate_slice`, `compare_cohorts` and `estimate_reweighted`.

Full behaviour is documented in [generative evaluation](generative-evaluation.md).
