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
- `report/`: the generated report and audit sidecars;
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

The observed plug-in mean is 0.5: four failures among eight equally weighted fixed items. That is
descriptively concerning, but each item has only one stochastic rollout. The plan allocates the
0.05 one-sided family error budget over all eight prompt endpoints, giving per-item alpha
`0.00625`.

With one rollout per item:

- a zero-failure item has an exact upper bound of `0.99375`;
- a one-failure item has an exact lower bound of `0.00625` and upper bound `1`.

After monotone propagation through the equal-weight mean, the report gives approximately:

| Quantity | Estimate | Lower | Upper |
|---|---:|---:|---:|
| Mean failure risk | 0.500000 | 0.003125 | 0.996875 |
| CVaR, worst 25% | 1.000000 | 0.006250 | 1.000000 |
| Entropic risk, lambda 5 | 0.862714 | 0.005005 | 0.996880 |
| Worst item | 1.000000 | 0.006250 | 1.000000 |

The mandatory mean-risk claim declares:

- `PASS` only if the upper bound is at most `0.25`;
- `FAIL` only if the lower bound is at least `0.60`;
- `INCONCLUSIVE` otherwise.

Here the upper bound is far above `0.25`, while the lower bound is far below `0.60`. The correct
decision is therefore `INCONCLUSIVE`, not pass and not fail. More independent rollouts of these
same eight fixed items could narrow rollout uncertainty. They would not add new benchmark items or
justify generalization to unseen questions.

## Read the outputs

Start with `report/report.md` for the human-readable decision. Use `report/report.json` for every
numeric result and eligibility label. `observations.csv` gives one auditable row per rollout,
`design.resolved.csv` records the effective policy weights, and `checksums.json` protects the
generated artifact set against unnoticed edits.
