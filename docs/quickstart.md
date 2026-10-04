# Quickstart

From a finished Inspect evaluation to a report in three commands. You need completed `.eval`
logs and the name of the score that marks a safety failure.

## 1. Write a starter plan from the logs

```bash
inspect-safety-stats init \
  --logs logs/ \
  --failure-values FAIL \
  --group-by category \
  --baseline openai/gpt-4o \
  --output analysis/
```

`init` reads the structure of the logs (models, tasks, sample ids, epochs, scorers, metadata)
and writes two files into `analysis/`:

- `design.csv`, one row per fixed prompt with its expected rollout count and, with
  `--group-by`, the sample metadata field used for the breakdown;
- `safety-plan.yaml`, a minimal plan: one cohort per model, the weighted-mean failure risk
  with a weighted KL/Chernoff interval and a prompt-level sensitivity, the breakdown, and a
  comparison between models when there is more than one.

It never guesses which score values are failures. `--failure-values` names them (repeat it for
several) and every other observed value is taken as safe unless `--safe-values` lists them;
`--failure-threshold` handles numeric scores. With several scorers in the logs, choose one with
`--score`. Without `--baseline`, two models are compared against the first alphabetically and
several models are compared pairwise.

The plan is `exploratory`: it was written after the logs existed. For a confirmatory analysis,
freeze a plan before running the evaluation; see [Confirmatory mode](confirmatory-mode.md).

## 2. Analyze

```bash
inspect-safety-stats analyze \
  --plan analysis/safety-plan.yaml \
  --logs logs/ \
  --output report/
```

The command prints the headline failure rate for each model and writes the report directory.

## 3. Read `report/report.md`

A one-page summary: the failure rate per model with its interval, the comparison, the
breakdown, and a **Precision** section. An interval marked † has an endpoint more than
`report.precision_target` (5 percentage points by default) from its estimate, and the
Precision section says roughly how many more rollouts would resolve it. This is a JailbreakBench
run of 100 prompts with 8 rollouts each:

```text
| Model       | Group | Failure rate | Interval      | Prompt-level sensitivity |
| gpt-4o      | all   | 4.8%         | [2.7%, 7.5%]  | [0.7%, 14.6%]            |
| gpt-4o-mini | all   | 6.5%         | [4.1%, 9.6%]  | [1.4%, 17.2%]            |

| Candidate vs baseline | Difference | Interval             | Ratio | Interval     | Reading                  |
| gpt-4o-mini vs gpt-4o | +1.8 pts   | [-3.4 pts, +6.9 pts] | 1.37  | [0.54, 3.54] | no difference resolved † |
```

The overall rates are well determined; the difference between the models is not, and the
per-category rates (10 prompts each) need roughly ten times the rollouts before they say much.

Everything else is alongside: `report-full.md` (every interval construction, sensitivity and
prompt-level bound), `report.html` (an interactive explorer that needs no server), and CSVs.

## Before the next run: plan the budget

The planners answer "how many rollouts do I need?" before any are spent:

```bash
inspect-safety-stats plan zero-failure --threshold 0.01 --alpha 0.05
inspect-safety-stats plan certification --threshold 0.05 --anticipated-rate 0.02 --power 0.9
inspect-safety-stats plan detectable-rate --trials 800 --threshold 0.05 --power 0.9
inspect-safety-stats plan comparison --baseline-rate 0.0475 --candidate-rate 0.065 --prompts 100
```

The last line asks what the JailbreakBench run above would have needed to resolve the
difference between the two models: about 88 rollouts per prompt rather than 8. Add `--margin`
to plan a non-inferiority claim instead.

## Graded scores and repeated attempts

If the scorer returns a grade or a probability rather than a verdict, describe its scale
instead of naming failure values: `init --harm-scale 1 5` (add `--lower-is-harmful` when a low
grade is the worse outcome). Every rate in the report is then a mean harm on a 0–100% scale.

To report the chance that an adversary with several tries at each prompt succeeds, add a
`k_attempt_risk` measure to the plan; see the [plan reference](plan-reference.md#risk-over-repeated-attempts).

## A quick view inside Inspect

To see the failure rate and its interval in the eval log as you iterate, add the
[Inspect metrics](inspect-metrics.md) `failure_rate`, `failure_rate_lower` and
`failure_rate_upper` to the task. They cover one model on one task; use `analyze` for anything
you will act on.

## Harm and over-refusal together

Analyze a harmful-compliance evaluation and an over-refusal evaluation (refusals of benign
prompts as the failure) separately, then put them side by side:

```bash
inspect-safety-stats tradeoff --harm harm/report/ --over-refusal refusal/report/
```

See [Harm and over-refusal in one report](harm-and-over-refusal.md).

## When an LLM judge scores the outcomes

Every interval above bounds how often the **judge reports** a failure. To bound the true rate,
add judge calibration: declared accuracy ranges as a sensitivity analysis, or human labels for a
certifying correction. `inspect-safety-stats plan judge-labels` sizes the labelling and
`inspect-safety-stats judge sample` draws it. See [Judge misclassification](judge-misclassification.md).

## Going further

The generated plan is a starting point. Edit it to add tail measures (CVaR, worst prompt),
further independence profiles, explicit weights, judge calibration, or threshold claims; the
[analysis plan reference](plan-reference.md) documents every field, and the
[fixed-design example](../examples/fixed_design/README.md) shows a hand-written plan.
