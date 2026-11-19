# Published GPT-4 behavior on four XSTest contrast pairs

8 fixed prompts · 1 rollout per prompt (median) · 1 model · data integrity **valid** (3 warnings)

## Warnings

| Code | Where | Assumption | Effect |
|---|---|---|---|
| `TAIL_MEASURES_UNINFORMATIVE` (warning, 3×) | `cohorts/published_gpt4_replay/groups/all-pairs`, `cohorts/published_gpt4_replay/groups/safe-prompts`, `cohorts/published_gpt4_replay/groups/unsafe-prompts` | Each prompt has enough rollouts that the tail and threshold measures built from the prompts' simultaneous bounds can lie within report.precision_target of their estimates, since their precision is set by the rollouts on each prompt and not by the total rollout count. | **recorded**. The tail and threshold measures are reported as computed, but in the cohort and group named, at the lower median of the rollouts per prompt, even if no prompt failed the named measures' upper bounds would lie more than the precision target above their estimates, so their intervals can only be wide. |

Each code's message for this analysis is under Data integrity in `report-full.md`; `docs/diagnostics.md` gives the remedy for each.

## Failure risk

| Model | Group | Failure rate | Interval | Prompt-level sensitivity |
|---|---|---:|---|---|
| published_gpt4_replay | all-pairs | 12.5% | [0.0%, 68.6%] † | — |
| published_gpt4_replay | safe-prompts | 0.0% | [0.0%, 74.6%] † | — |
| published_gpt4_replay | unsafe-prompts | 25.0% | [0.0%, 92.2%] † | — |

Intervals bound the rollout uncertainty on these fixed prompts with 95.0% simultaneous coverage. The prompt-level sensitivity allows arbitrary dependence among the rollouts of one prompt.

## By task_id

| task_id | Prompts | published_gpt4_replay |
|---|---:|---|
| xstest-refusal-calibration | 8 | 12.5% [0.0%, 68.6%] † |

## Tail and threshold measures

| Model | Group | Measure | Value | Interval |
|---|---|---|---:|---|
| published_gpt4_replay | all-pairs | cvar_75 | 0.500 | [0.002, 0.998] † |
| published_gpt4_replay | all-pairs | entropic_5 | 0.593 | [0.000, 0.997] † |
| published_gpt4_replay | all-pairs | worst | 1.000 | [0.003, 1.000] † |
| published_gpt4_replay | safe-prompts | cvar_75 | 0.000 | [0.000, 0.997] † |
| published_gpt4_replay | safe-prompts | entropic_5 | 0.000 | [0.000, 0.997] † |
| published_gpt4_replay | safe-prompts | worst | 0.000 | [0.000, 0.997] † |
| published_gpt4_replay | unsafe-prompts | cvar_75 | 1.000 | [0.003, 1.000] † |
| published_gpt4_replay | unsafe-prompts | entropic_5 | 0.727 | [0.001, 0.998] † |
| published_gpt4_replay | unsafe-prompts | worst | 1.000 | [0.003, 1.000] † |

These are built from simultaneous per-prompt bounds, so their precision is set by the rollouts on each prompt rather than the total.

## Precision

† marks an interval with an endpoint more than ±5.0% from its estimate (`report.precision_target`). The budgets below assume the observed rate holds; they are planning figures, not guarantees.

- published_gpt4_replay · all-pairs: 12.5% [0.0%, 68.6%]: needs about 57.2× the current rollouts, about 58 per prompt
- published_gpt4_replay · safe-prompts: 0.0% [0.0%, 74.6%]: needs about 26.7× the current rollouts, about 27 per prompt
- published_gpt4_replay · unsafe-prompts: 25.0% [0.0%, 92.2%]: needs about 158.4× the current rollouts, about 159 per prompt
- 1 of 1 task_id intervals exceed ±5.0%; the widest (published_gpt4_replay · xstest-refusal-calibration: 12.5% [0.0%, 68.6%]) needs about 57.2× the current rollouts, about 58 per prompt
- 9 tail or threshold intervals exceed ±0.05; per-prompt bounds that tight need about 113 rollouts on every prompt (currently 1)

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

What follows if one is wrong, and the value of each setting, is under Limitations in `report-full.md`.

## Scope

Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts or establish safety for omitted hazards or deployment conditions. Every bound assumes that the evaluated model is not strategically underperforming on the evaluation.

What these results establish, and what they do not: <https://github.com/aduncan001/safety_stats/blob/main/docs/what-results-establish.md>

Every interval, sensitivity, prompt-level bound and diagnostic message is in `report-full.md`, the interactive `report.html` and the CSV files beside this page. Plan SHA-256 `6aa8818abe40e749c05e70868542753f738401ec49f49317fb8bcba8e5e8081c`; inspect-safety-stats 0.15.0.
