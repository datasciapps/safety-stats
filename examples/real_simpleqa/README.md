# Worked example: eight real SimpleQA results

This example uses eight genuine samples from a completed Inspect evaluation of
`codelion/SimpleQA-Verified` with `openai/gpt-4o`. Four received `C` and four received `I` from
Inspect's `model_graded_qa` scorer. For this worked failure analysis, `I` is explicitly mapped to
failure and `C` to safe.

This is a real-model, real-dataset integration example. It demonstrates the statistical and audit
workflow; factual-question accuracy is not itself a deployment-safety claim.

## Files

- `simpleqa-8-real.eval`: derived eight-sample Inspect log;
- `design.csv`: fixed creator-declared hierarchy and rollout requirements;
- `safety-plan.yaml`: score mapping, risk measures, multiplicity, and decision thresholds;
- `report/`: written by the command below (not tracked in the repository);
- `make_subset.py`: reproducible extraction from the original completed log.

## Run it

From the repository root:

```bash
inspect-safety-stats validate \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval

inspect-safety-stats analyze \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval \
  --output examples/real_simpleqa/report \
  --overwrite \
  --no-decision-exit-codes
```

## What happened

The observed mean is 0.5: four failures among eight equally weighted fixed items. That is
descriptively concerning, but each item has only one stochastic rollout. The prompt-bound family
spends its 0.05 budget two-sided over all eight prompts, giving each endpoint alpha `0.003125`.

With one rollout per item:

- a zero-failure item has an exact upper bound of `0.996875`;
- a one-failure item has an exact lower bound of `0.003125` and upper bound `1`.

The report gives (version 0.16.0):

| Quantity | Estimate | Lower | Upper | Construction |
|---|---:|---:|---:|---|
| Mean failure risk | 0.500000 | 0.092069 | 0.907931 | direct weighted KL/Chernoff (primary) |
| Mean failure risk | 0.500000 | 0.001563 | 0.998437 | propagated prompt envelope (reference) |
| CVaR, worst 25% | 1.000000 | 0.003125 | 1.000000 | propagated prompt envelope |
| Entropic risk, lambda 5 | 0.862714 | 0.001569 | 0.998444 | propagated prompt envelope |
| Worst item | 1.000000 | 0.003125 | 1.000000 | propagated prompt envelope |

The direct interval treats the eight rollouts as independent and is much tighter than the
envelope, which needs no independence between prompts. Even so, it spans most of [0, 1].

The mandatory mean-risk claim declares:

- `PASS` only if the upper bound is at most `0.25`;
- `FAIL` only if the lower bound is at least `0.60`;
- `INCONCLUSIVE` otherwise.

Here the upper bound is far above `0.25`, while the lower bound is far below `0.60`. The correct
decision is therefore `INCONCLUSIVE`, not pass and not fail. More independent rollouts of these
same eight fixed items could narrow rollout uncertainty. They would not add new benchmark items or
justify generalization to unseen questions.

## Read the outputs

Start with `report/report.md`, the one-page summary with the decision, and `report/report-full.md`
for every construction. Use `report/report.json` for every numeric result and eligibility label. `observations.csv` gives one auditable row per rollout,
`design.resolved.csv` records the effective policy weights, and `checksums.json` protects the
generated artifact set against unnoticed edits.
