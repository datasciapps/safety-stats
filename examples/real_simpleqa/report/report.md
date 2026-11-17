# Eight real SimpleQA items as a fixed-design failure analysis

8 fixed prompts · 1 rollout per prompt (median) · 1 model · data integrity **valid** (2 warnings)

## Warnings

| Code | Where | Assumption | Effect |
|---|---|---|---|
| `PLAN_SELF_DECLARED` (warning) | — | The confirmatory plan's freeze time is backed by an external registration, not only by the time its author wrote in analysis.plan_frozen_at. | **recorded**. Nothing is computed differently and integrity is unchanged; the report records that the plan's timing rests on its author's word. |
| `TAIL_MEASURES_UNINFORMATIVE` (warning) | `plan/outputs/risk_measures` | Each prompt has enough rollouts that its simultaneous bound can be tighter than report.precision_target, since tail and threshold measures are built from those per-prompt bounds and not from the total rollout count. | **recorded**. The tail and threshold measures are reported as computed, but at the median rollout count even a prompt with no failures is bounded above the precision target, so their intervals can only be wide. |

Each code's message for this analysis is under Data integrity in `report-full.md`; `docs/diagnostics.md` gives the remedy for each.

## Failure risk

| Model | Group | Failure rate | Interval | Prompt-level sensitivity |
|---|---|---:|---|---|
| candidate | all-eight | 50.0% | [9.2%, 90.8%] † | — |

Intervals bound the rollout uncertainty on these fixed prompts with 95.0% simultaneous coverage. The prompt-level sensitivity allows arbitrary dependence among the rollouts of one prompt.

## By task_id

| task_id | Prompts | candidate |
|---|---:|---|
| simpleqa | 8 | 50.0% [9.2%, 90.8%] † |

## Tail and threshold measures

| Model | Group | Measure | Value | Interval |
|---|---|---|---:|---|
| candidate | all-eight | cvar_75 | 1.000 | [0.003, 1.000] † |
| candidate | all-eight | entropic_5 | 0.863 | [0.002, 0.998] † |
| candidate | all-eight | worst | 1.000 | [0.003, 1.000] † |

These are built from simultaneous per-prompt bounds, so their precision is set by the rollouts on each prompt rather than the total.

## Precision

† marks an interval with an endpoint more than ±5.0% from its estimate (`report.precision_target`). The budgets below assume the observed rate holds; they are planning figures, not guarantees.

- candidate · all-eight: 50.0% [9.2%, 90.8%]: needs about 78.2× the current rollouts, about 79 per prompt
- 1 of 1 task_id intervals exceed ±5.0%; the widest (candidate · simpleqa: 50.0% [9.2%, 90.8%]) needs about 78.2× the current rollouts, about 79 per prompt
- 3 tail or threshold intervals exceed ±0.05; per-prompt bounds that tight need about 113 rollouts on every prompt (currently 1)

## Assumptions used

- **fixed_design.** Prompt, question, task and group membership are fixed, creator-declared design elements. The bounds cover these prompts, not a population they were drawn from.
- **run_stability.** Every rollout of a cohort comes from one stable system: the same model and configuration throughout its runs.
- **rollout_independence.** The blocks of the declared independence profile are independent: under the rollout profile, every rollout of a prompt is an independent draw.
- **scorer_validity.** The score measures the hazard. Scorer errors, under-elicitation, evaluation awareness and omitted hazards are not repaired by statistical precision.
- **judge_accuracy.** Judge verdicts are taken as correct unless the plan validates or transports the judge's error rates. Left to the default: `inputs.judge_calibration`.
- **missing_scores.** A rollout without a score has an unknown outcome, bounded at both extremes, unless the plan counts it otherwise.
- **sample_errors.** A sample error counts as the plan's score.errors says. Under end_to_end, with the end-to-end system estimand, a provider's policy block of the target's generation that delivered no content counts as safe.
- **token_limit.** An answer that stopped at the generation token limit counts by its score, and without a score as missing. Left to the default: `inputs.score.token_limit`.
- **content_filter_stops.** An output a provider's content filter stopped without a sample error counts by its score. Left to the default: `inputs.score.content_filter_stops`.
- **complete_design.** Every run finished, every prompt of a fixed design was evaluated, and every prompt has the rollouts the manifest expects.
- **plan_fixed_in_advance.** The plan was fixed before the runs started, so that no choice in it was made after seeing the outcomes.

What follows if one is wrong, and the value of each setting, is under Limitations in `report-full.md`.

## Scope

Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts or establish safety for omitted hazards or deployment conditions.

Every interval, sensitivity, prompt-level bound and diagnostic message is in `report-full.md`, the interactive `report.html` and the CSV files beside this page. Plan SHA-256 `fd9ee06715157246e1e6d90d86f2cada7d7cffe96ee9fd7e280c8d026a0f654c`; inspect-safety-stats 0.15.0.
