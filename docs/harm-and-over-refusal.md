# Harm and over-refusal in one report

A model that refuses everything never complies with a harmful request. Harmful-compliance rates
mean little without the matching over-refusal rate, so `tradeoff` reads one analysis of each and
puts them side by side.

## Run it

Analyze the two evaluations separately, each with its own plan, then combine the reports:

```bash
inspect-safety-stats analyze --plan harm/safety-plan.yaml --logs harm/logs/ --output harm/report/
inspect-safety-stats analyze --plan refusal/safety-plan.yaml --logs refusal/logs/ --output refusal/report/

inspect-safety-stats tradeoff --harm harm/report/ --over-refusal refusal/report/ --output tradeoff/
```

In the over-refusal plan the failure value is the refusal of a benign prompt (on XSTest's safe
prompts, for example, a refusal). `tradeoff` writes `tradeoff.md`, `tradeoff.json` and
`tradeoff.html` (a scatter of the two rates with their intervals) and prints the same tables:

```text
| Model           | Harmful compliance | Interval       | Over-refusal | Interval       |
| mockllm/careful | 0.0%               | [0.0%, 2.2%]   | 60.0%        | [49.6%, 69.9%] |
| mockllm/eager   | 60.0%              | [49.6%, 69.9%] | 10.0%        | [4.9%, 17.4%]  |

| Candidate vs baseline            | Harm difference        | Over-refusal difference | Reading                                                   |
| mockllm/eager vs mockllm/careful | [+46.3 pts, +70.6 pts] | [-66.0 pts, -30.7 pts]  | trade-off: mockllm/eager is more harmful but refuses less |
```

## What it reads

From each report it takes the mean measure (the first weighted mean if a plan has several, or
`--measure`), the `all` group unless `--harm-group` or `--refusal-group` names another, and the
models whose cohorts each contain a single model. Models found in only one report are listed
and left out. Each pair of models takes its difference from the comparison declared in each
report, in the declared candidate–baseline order. A pair that only one report compared is shown
with its other difference missing ("not available in both reports"); a pair that neither
report compared is left out.

Nothing is recomputed. Every number is the one in the source report, with the interval method
and family that produced it.

## Readings

A difference is resolved when its interval excludes zero; lower is better on both axes.

| Harm difference | Over-refusal difference | Reading |
|---|---|---|
| below zero | below zero | better on both |
| above zero | above zero | worse on both |
| below zero | above zero | trade-off: less harmful but refuses more |
| above zero | below zero | trade-off: more harmful but refuses less |
| resolved on one axis only | | that axis's direction; the other unresolved |
| neither resolved | | no difference resolved |

## Coverage

The two reports were calibrated separately, so their guarantees are combined by the union bound.
Each distinct (report, family) budget counts once; families that share a name in different
reports are still separate budgets. With two reports that each spend 0.05 on rates and 0.05 on
comparisons:

- all per-model rates hold simultaneously with probability at least 90%;
- all differences hold simultaneously with probability at least 90%;
- everything on the page holds simultaneously with probability at least 80%.

Spend less per family in each plan if the combined guarantee needs to be stronger.

## Boundaries

Each rate belongs to its own fixed prompt set: harmful compliance on the harmful prompts you
ran, over-refusal on the benign prompts you ran. The report does not weigh one against the other;
how much over-refusal is an acceptable price for less harm is a judgement it leaves to you.
