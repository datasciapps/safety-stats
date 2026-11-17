---
name: inspect-safety-stats
description: >
  Compute finite-sample statistical bounds, rollout sample sizes, LLM-judge misclassification
  corrections, paired or multi-model comparisons, and Inspect AI (.eval) evaluation reports. Use
  when planning rollout counts, certifying zero-failure thresholds, bounding multi-epoch benchmark
  failure rates, correcting imperfect LLM judges, or analyzing fixed-design binary LLM evaluations
  across safety, reliability, and capability benchmarks.
---

# Using `inspect-safety-stats`

## 1. Pick the right entry point first

- **Counts, per-item failure rates, paired pass/fail lists, or judge calibration numbers:**
  Call `inspect_safety_stats.stats` directly in Python—no log files, YAML plans, or [Inspect AI](https://inspect.aisi.org.uk/) harness required.
- **Completed Inspect AI `.eval` logs:**
  Run `inspect-safety-stats validate` then `inspect-safety-stats analyze` with a `design.csv` manifest and a strict `schema_version: "2.0"` YAML plan.
- **Custom JSONL/CSV evaluation logs needing full hierarchy reports and `report.html`:**
  Convert records into in-memory `inspect_ai.log.EvalLog` objects and pass `logs=[...]` to `inspect_safety_stats.validate()` and `inspect_safety_stats.analyze()`, reading `design.csv` and `plan.yaml` from disk.

## 2. Input requirements in `inspect_safety_stats.stats`

1. **Weights sum to 1.0:** Every weighted estimator (`weighted_mean`, `weighted_cvar`, `entropic_risk`, `worst_of_k_estimate`, `worst_of_k_risk`, `weighted_*_mean_bounds`, `direct_*_bounds`, `fixed_design_detectable_failure_rates`) requires nonnegative 1D weights summing to `1.0`.
2. **One-sided tail budgets:** Interval functions take `alpha_lower` and `alpha_upper` separately. For a 95% two-sided interval, pass `alpha_lower=0.025, alpha_upper=0.025`.
3. **Paired tests require Python `bool` lists:** `newcombe_paired_interval(candidate, baseline, ...)` and `exact_mcnemar(candidate, baseline)` reject integers (`0`/`1`) and `numpy.bool_`. Convert with `[bool(x) for x in ...]` where `True` denotes a failure; the estimand is `candidate - baseline` (negative means `candidate` failed less often). For multi-rollout contrasts, use `direct_risk_difference_bounds` and `direct_ratio_bounds`.
4. **Single-judge `rho_1` and `rho_0` are predictive values, not sensitivity/specificity:**
   - `rho_1 = P(true failure | judge flagged failure)`
   - `rho_0 = P(true failure | judge cleared as pass)`
   - Requires `rho_1_lower > rho_0_upper` (`JudgeCalibration` in `inspect_safety_stats.stats`; human-audit `JudgeStratumCounts` and `estimate_judge_calibration` are in `inspect_safety_stats.stats.misclassification`, where `truly_unsafe` counts true failures). Both depend on the evaluated output distribution and prompt weights; pooled cohort calibration does not generally certify subgroups or tail measures, and estimated calibration spends error budget alongside the rollout interval.
   - Prevalence-free `sensitivity` and `specificity` belong only to multi-judge panels (`JudgeAccuracyBox` in `inspect_safety_stats.stats`).
5. **Exact import locations for planning:** `zero_failure_sample_size` is in `inspect_safety_stats.stats`; `certification_sample_size`, `maximum_certifying_failures`, `maximum_certifiable_failure_rate`, `minimum_detectable_failure_rate`, `rejection_sample_size`, and `fixed_design_detectable_failure_rates` are in `inspect_safety_stats.stats.planning`.

```python
from inspect_safety_stats.stats import (
    JudgeCalibration,
    binomial_bounds,
    correct_judge_risk,
    newcombe_paired_interval,
    weighted_kl_chernoff_mean_bounds,
    zero_failure_sample_size,
)

n = zero_failure_sample_size(threshold=0.01, alpha=0.05)  # 299
prompt_ci = binomial_bounds(0, 50, alpha_lower=0.025, alpha_upper=0.025)
mean_ci = weighted_kl_chernoff_mean_bounds(
    failures=[0, 2],
    unclassified=[0, 0],
    trials=[25, 25],
    weights=[0.5, 0.5],
    alpha_lower=0.025,
    alpha_upper=0.025,
)
corrected = correct_judge_risk(
    observed_lower=mean_ci.lower,
    observed_upper=mean_ci.upper,
    observed_estimate=mean_ci.completion_estimate_lower,
    calibration=JudgeCalibration("declared", 0.90, 1.0, 0.0, 0.02),
)
paired = newcombe_paired_interval(
    [False, False, True, False],
    [True, False, True, True],
    alpha_lower=0.025,
    alpha_upper=0.025,
)
```

## 3. Workflow for `schema_version: "2.0"` plans and logs

Never guess YAML field names—every plan model sets Pydantic `extra="forbid"` and rejects booleans (`true`/`false`) in integer and float fields.

1. **Inspect the schema first** when using unfamiliar options:
   ```bash
   inspect-safety-stats schema plan
   ```
2. **Check the seven mandatory plan rules before running:**
   - Top-level keys `schema_version: "2.0"`, `analysis`, `inputs`, `cohorts`, `groups`, `outputs`, `multiplicity`, and `claims` are all present (`claims: []` if no threshold gates are declared). Optional `comparisons` and `comparison_sets` are also top-level keys, not nested inside `outputs`.
   - `design.csv` has one row per prompt with `prompt_id`, `question_id`, `task_id` (plus integer `expected_rollouts` = planned repetitions/epochs per prompt, unless `design.require_expected_rollouts: false`).
   - Either set `outputs.prompt_bounds.family` to an ID in `multiplicity.families`, or set `outputs.prompt_bounds: {enabled: false}` (leaving `enabled: true` with `family: null` fails validation with `PLAN_UNKNOWN_REFERENCE`).
   - Set `policy: {zero_tolerance: {enabled: false}}` unless `design.csv` has a boolean `zero_tolerance` column.
   - Set `inputs.estimand: underlying_model` on capability, factuality, or QA benchmarks so provider content-filter blocks are not counted as passes (the default `end_to_end_system` treats provider blocks as non-failures for safety red-teaming). Use `analysis.intent: exploratory` when the plan was not frozen prior to the evaluation run.
   - Declare at least one entry in `outputs.risk_measures` (for example `{id: mean, type: weighted_mean}`).
   - Each populated item in `claims` requires `id`, `title`, `family` (e.g. `purpose: all_mandatory, method: intersection_union, alpha: 0.05`), and `acceptable_upper < unacceptable_lower`, plus **either** `{cohort, group, measure}` (aggregate claim) **or** `{comparison}` alone (comparison claim).
3. **Validate before analyzing:**
   ```bash
   inspect-safety-stats validate --plan plan.yaml --logs logs/
   inspect-safety-stats analyze --plan plan.yaml --logs logs/ --output reports/run-001
   ```
   - Exit `0`: `valid` or `degraded` integrity (inspect warnings in `report.json`).
   - Exit `2`: invalid or missing CLI arguments.
   - Exit `3`: `invalid` integrity (`analyze` writes `report.json` first so diagnostics can be audited), or an execution error before report generation.
   - With `--decision-exit-codes`: `0` = `PASS`, `1` = `FAIL`, `2` = `INCONCLUSIVE`, `3` = `invalid` or execution error.

## 4. Reporting discipline

Always preserve the fixed-design scope boundary and declared independence assumptions when summarizing outputs: rollout-level intervals assume generations are independent conditional on the prompt, cover stochastic rollout uncertainty on the declared prompts and weights (unless `design.type: sampled` with valid inclusion probabilities was used), and do not estimate failure prevalence over unseen prompts or deployment traffic.

## 5. Deeper references

- Complete single-file API map, minimal `v2.0` YAML, and external `EvalLog` adapter: [llms-full.txt](../../llms-full.txt)
- Plan schema guide: [docs/plan-reference.md](../../docs/plan-reference.md)
- API and CLI contract: [docs/api-reference.md](../../docs/api-reference.md)
- Judge misclassification math and config: [docs/judge-misclassification.md](../../docs/judge-misclassification.md)
- Multi-model comparisons: [docs/model-comparisons.md](../../docs/model-comparisons.md)
