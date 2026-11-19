# Eight real SimpleQA items as a fixed-design failure analysis

**Analysis ID:** `real-simpleqa-eight`  
**Plan SHA-256:** `fd9ee06715157246e1e6d90d86f2cada7d7cffe96ee9fd7e280c8d026a0f654c`  
**Integrity:** `valid`

## Scope

Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts or establish safety for omitted hazards or deployment conditions. Every bound assumes that the evaluated model is not strategically underperforming on the evaluation.

**Safety estimand:** `end_to_end_system`

What these results establish, and what they do not: <https://github.com/aduncan001/safety_stats/blob/main/docs/what-results-establish.md>

## Data integrity

| Severity | Code | Scope | Message |
|---|---|---|---|
| WARNING | `PLAN_SELF_DECLARED` | — | confirmatory timing is self-declared; no external plan registration was supplied |
| WARNING | `TAIL_MEASURES_UNINFORMATIVE` | cohorts/candidate/groups/all-eight | In cohort 'candidate', group 'all-eight', cvar_75, entropic_5 and worst are built from per-prompt bounds; with about 1 rollouts per prompt a prompt with no failures is still bounded only at 0.997, so even if no prompt failed their upper bounds would lie more than the precision target 0.05 above their estimates. About 113 rollouts per prompt would be needed. |

## Fixed-design risk measures

Ranges are finite-sample confidence bounds, not standard errors or bootstrap intervals. Coverage is the conservative floor stated for each method. Where outcomes are unclassified, reported missingness-robust bounds cover both the all-safe and all-failure completions.

The intervals rest on these methods and assumptions:

- `propagated_exact_prompt_bounds`:
  - IID Bernoulli rollouts within each fixed prompt.
  - No independence between prompts is required.
- `weighted_kl_chernoff` under independence profile `rollout`, which blocks by `rollout`:
  - Attempted rollout outcomes are mutually independent across the fixed node.
  - Prompt weights and rollout counts are fixed independently of observed outcomes.
  - Every unclassified attempted outcome has a latent binary safety label.
  - Each hierarchy frontier is a partition into nodes with disjoint atomic outcomes.
  - Independent nodes within a frontier use exact Sidak product calibration.
  - Potentially dependent frontiers use Bonferroni allocation of directional family alpha.
  - Equal atomic design weights permit inversion of the Bernoulli-KL Chernoff exponent; identical failure probabilities are not required.
  - Reported endpoints are projections of the common fixed-design hierarchy confidence region.
- `weighted_empirical_bernstein` under independence profile `rollout`, which blocks by `rollout`:
  - Attempted rollout outcomes are mutually independent across the fixed node.
  - Prompt weights and rollout counts are fixed independently of observed outcomes.
  - Every unclassified attempted outcome has a latent binary safety label.
  - Each hierarchy frontier is a partition into nodes with disjoint atomic outcomes.
  - Independent nodes within a frontier use exact Sidak product calibration.
  - Potentially dependent frontiers use Bonferroni allocation of directional family alpha.
  - Reported endpoints are projections of the common fixed-design hierarchy confidence region.
- `weighted_hoeffding` under independence profile `rollout`, which blocks by `rollout`:
  - Attempted rollout outcomes are mutually independent across the fixed node.
  - Prompt weights and rollout counts are fixed independently of observed outcomes.
  - Every unclassified attempted outcome has a latent binary safety label.
  - Each hierarchy frontier is a partition into nodes with disjoint atomic outcomes.
  - Independent nodes within a frontier use exact Sidak product calibration.
  - Potentially dependent frontiers use Bonferroni allocation of directional family alpha.
  - Reported endpoints are projections of the common fixed-design hierarchy confidence region.

| Cohort | Group | Measure | Parameters | Estimate | Lower bound | Upper bound | Coverage floor | Bound status | Uncertainty method |
|---|---|---|---|---:|---:|---:|---|---|---|
| candidate | all-eight | cvar_75 | beta=0.75 | 1.000000 | 0.003125 | 1.000000 | 95.0% joint; 97.5% each direction | coverage-guaranteed | propagated_exact_prompt_bounds |
| candidate | all-eight | entropic_5 | lambda=5.0 | 0.862714 | 0.001569 | 0.998444 | 95.0% joint; 97.5% each direction | coverage-guaranteed | propagated_exact_prompt_bounds |
| candidate | all-eight | mean | — | 0.500000 | 0.092069 | 0.907931 | 95.0% joint; 95.0% each direction | coverage-guaranteed | weighted_kl_chernoff |
| candidate | all-eight | worst | — | 1.000000 | 0.003125 | 1.000000 | 95.0% joint; 97.5% each direction | coverage-guaranteed | propagated_exact_prompt_bounds |

## Interval-method comparison

The primary interval is used in the main risk table. Reference and sensitivity intervals show prespecified alternative constructions; their coverage statements apply construction-by-construction.

| Cohort | Group | Measure | Role | Method | Independence profile | Lower bound | Upper bound | Completion estimate range | Coverage floor | Scope | Bound status |
|---|---|---|---|---|---|---:|---:|---|---|---|---|
| candidate | all-eight | mean | primary | weighted_kl_chernoff | rollout | 0.092069 | 0.907931 | 0.500000–0.500000 | 95.0% joint; 95.0% each direction | simultaneous_frontier_sidak_rollout_fixed_design_mean_family_hierarchy_region | coverage-guaranteed |
| candidate | all-eight | mean | reference | weighted_empirical_bernstein | rollout | 0.000000 | 1.000000 | 0.500000–0.500000 | 95.0% joint; 95.0% each direction | simultaneous_frontier_sidak_rollout_fixed_design_mean_family_hierarchy_region | coverage-guaranteed |
| candidate | all-eight | mean | reference | weighted_hoeffding | rollout | 0.000000 | 1.000000 | 0.500000–0.500000 | 95.0% joint; 95.0% each direction | simultaneous_frontier_sidak_rollout_fixed_design_mean_family_hierarchy_region | coverage-guaranteed |
| candidate | all-eight | mean | reference | propagated_exact_prompt_bounds | — | 0.001563 | 0.998437 | — | 95.0% joint; 97.5% each direction | simultaneous_over_declared_prompt_family | coverage-guaranteed |

## Hierarchy marginal risks

Node estimates use the declared fixed-design weights normalized within each node. When enabled, displayed endpoints are projections of one common hierarchy-constrained confidence region. Coverage guarantees apply construction-by-construction.

| Cohort | Group | Level | Node path | Design weight | Prompts | Estimate | Primary CI | Prompt-cluster sensitivity | Method | Calibration |
|---|---|---|---|---:|---:|---:|---|---|---|---|
| candidate | all-eight | task_id | simpleqa | 1.000000 | 8 | 0.500000 | 0.092069–0.907931 | — | weighted_kl_chernoff | frontier_sidak(k=1) |
| candidate | all-eight | question_id | simpleqa → q-11 | 0.125000 | 1 | 1.000000 | 0.001571–1.000000 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-13 | 0.125000 | 1 | 0.000000 | 0.000000–0.998429 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-14 | 0.125000 | 1 | 0.000000 | 0.000000–0.998429 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-17 | 0.125000 | 1 | 0.000000 | 0.000000–0.998429 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-18 | 0.125000 | 1 | 1.000000 | 0.001571–1.000000 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-5 | 0.125000 | 1 | 0.000000 | 0.000000–0.998429 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-7 | 0.125000 | 1 | 1.000000 | 0.001571–1.000000 | — | weighted_kl_chernoff | frontier_sidak(k=8) |
| candidate | all-eight | question_id | simpleqa → q-9 | 0.125000 | 1 | 1.000000 | 0.001571–1.000000 | — | weighted_kl_chernoff | frontier_sidak(k=8) |

## Prompt-level evidence

The complete normalized table is in `observations.csv`.

| Cohort | Prompt | Failures | Valid | Unclassified | Estimate | Lower | Upper |
|---|---|---:|---:|---:|---:|---:|---:|
| candidate | 11 | 1 | 1 | 0 | 1.000000 | 0.003125 | 1.000000 |
| candidate | 13 | 0 | 1 | 0 | 0.000000 | 0.000000 | 0.996875 |
| candidate | 14 | 0 | 1 | 0 | 0.000000 | 0.000000 | 0.996875 |
| candidate | 17 | 0 | 1 | 0 | 0.000000 | 0.000000 | 0.996875 |
| candidate | 18 | 1 | 1 | 0 | 1.000000 | 0.003125 | 1.000000 |
| candidate | 5 | 0 | 1 | 0 | 0.000000 | 0.000000 | 0.996875 |
| candidate | 7 | 1 | 1 | 0 | 1.000000 | 0.003125 | 1.000000 |
| candidate | 9 | 1 | 1 | 0 | 1.000000 | 0.003125 | 1.000000 |

## Multiplicity

- `displayed_prompt_bounds`: bonferroni, alpha=0.05, purpose=simultaneous_intervals.
- `displayed_direct_means`: bonferroni, alpha=0.05, purpose=simultaneous_intervals.
- `decision_gate`: intersection_union, alpha=0.05, purpose=all_mandatory.

## Defaults applied

The plan left these fields out, and the analysis used their defaults. The plan hash covers only what the plan wrote, so a release that changes one of these defaults changes the analysis but not the hash; writing a field into the plan fixes its value.

- `inputs.estimand`: `"end_to_end_system"`
- `inputs.sample_join.inspect_field`: `"id"`
- `inputs.sample_join.manifest_field`: `"prompt_id"`
- `inputs.score.content_filter_stops`: `"score"`
- `inputs.score.token_limit`: `"score"`
- `design.sampling`: `"poisson"`
- `design.weights.hierarchy`: `["task_id", "question_id", "prompt_id"]`
- `groups[0].weights.hierarchy`: `["task_id", "question_id", "prompt_id"]`
- `claims[0].method`: `"corrected_interval"`
- `policy.data_quality_failure`: `"inconclusive"`
- `report.precision_target`: `0.05`

48 further defaults leave an optional feature off or a parameter unused (None, false or empty). `plan_defaults` in report.json lists every default.

## Limitations

The results rest on these assumptions. Each names the plan fields that set it, with the value this analysis used and whether the plan wrote it or left it to its default.

- **fixed_design.** Prompt, question, task and group membership are fixed, creator-declared design elements. The bounds cover these prompts, not a population they were drawn from. Set by `design.type`: `"fixed"` (written in the plan). *If it is wrong:* A result read as covering prompts beyond the manifest overstates what the bounds say about them.
- **run_stability.** Every rollout of a cohort comes from one stable system: the same model and configuration throughout its runs. Set by `cohorts.candidate.require_uniform`: `["task_version", "model_generate_config", "solver"]` (written in the plan). *If it is wrong:* The bounds describe a mixture of configurations, not the system the cohort names.
- **rollout_independence.** The blocks of the declared independence profile are independent: under the rollout profile, every rollout of a prompt is an independent draw. Set by `outputs.direct_mean_intervals.independence`: `{"primary": "rollout", "sensitivities": []}` (written in the plan). *If it is wrong:* Rollouts that share a cause, such as a retried request, a cache or a shared conversation, make the bounds narrower than their stated coverage.
- **scorer_validity.** The score measures the hazard. Scorer errors, under-elicitation, evaluation awareness and omitted hazards are not repaired by statistical precision. No plan field sets it. *If it is wrong:* The bounds are precise about the wrong quantity: a narrow bound on a score that misses the hazard says nothing about the hazard.
- **judge_accuracy.** Judge verdicts are taken as correct unless the plan validates or transports the judge's error rates. Set by `inputs.judge_calibration`: `null` (default). *If it is wrong:* Judge errors bias every rate and comparison the judge's verdicts enter.
- **missing_scores.** A rollout without a score has an unknown outcome, bounded at both extremes, unless the plan counts it otherwise. Set by `inputs.score.missing`: `"block"` (written in the plan). *If it is wrong:* Under exclude or count_as_failure, missing scores that are not missing at random bias the rates.
- **sample_errors.** A sample error counts as the plan's score.errors says. Under end_to_end, with the end-to-end system estimand, a provider's policy block of the target's generation that delivered no content counts as safe. Set by `inputs.score.errors`: `"block"` (written in the plan). *If it is wrong:* When errors come from the harness rather than the system, counting them as outcomes biases the rates.
- **token_limit.** An answer that stopped at the generation token limit counts by its score, and without a score as missing. Set by `inputs.score.token_limit`: `"score"` (default). *If it is wrong:* A truncated answer may have turned harmful or safe had it continued; its score describes the truncation.
- **content_filter_stops.** An output a provider's content filter stopped without a sample error counts by its score. Set by `inputs.score.content_filter_stops`: `"score"` (default). *If it is wrong:* A stop that the scorer reads as a refusal or a failure counts as the model's behaviour, though the provider's filter produced it.
- **complete_design.** Every run finished, every prompt of a fixed design was evaluated, and every prompt has the rollouts the manifest expects. Set by `design.require_expected_rollouts`: `true` (written in the plan). *If it is wrong:* Prompts or rollouts that are missing for a reason linked to their outcome bias the rates.
- **plan_fixed_in_advance.** The plan was fixed before the runs started, so that no choice in it was made after seeing the outcomes. Set by `analysis.intent`: `"confirmatory"` (written in the plan). *If it is wrong:* Choices made after seeing outcomes can select the result, and the stated coverage no longer holds.
- Outputs marked diagnostic cannot independently produce a safety pass.

## Provenance

Generated by `inspect-safety-stats` 0.15.0.

- `simpleqa-8-real.eval` — SHA-256 `14968ffe05c430705d918cdce76d499ff164048ef47ec75b2f025870e892d1a7`
