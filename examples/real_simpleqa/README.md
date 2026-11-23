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
safety-stats validate \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval

safety-stats analyze \
  --plan examples/real_simpleqa/safety-plan.yaml \
  --logs examples/real_simpleqa/simpleqa-8-real.eval \
  --output examples/real_simpleqa/report \
  --overwrite \
  --no-decision-exit-codes
```

## What happened

The observed plug-in mean is 0.5: four failures among eight equally weighted fixed items. That is
descriptively concerning, but each item has only one stochastic rollout. The prompt bounds are
two-sided: the 0.05 family error budget is split over both directions and all eight prompt
endpoints, giving each item alpha `0.003125` per direction.

With one rollout per item:

- a zero-failure item has an exact upper bound of `0.996875`;
- a one-failure item has an exact lower bound of `0.003125` and upper bound `1`.

The mean uses a direct weighted KL-Chernoff interval over the rollouts. The other measures
propagate the exact prompt bounds monotonically. The report gives approximately:

| Quantity | Estimate | Lower | Upper | Method |
|---|---:|---:|---:|---|
| Mean failure risk | 0.500000 | 0.092069 | 0.907931 | `weighted_kl_chernoff` |
| CVaR, worst 25% | 1.000000 | 0.003125 | 1.000000 | propagated prompt bounds |
| Entropic risk, lambda 5 | 0.862714 | 0.001569 | 0.998444 | propagated prompt bounds |
| Worst item | 1.000000 | 0.003125 | 1.000000 | propagated prompt bounds |

The mandatory mean-risk claim declares:

- `PASS` only if the upper bound is at most `0.25`;
- `FAIL` only if the lower bound is at least `0.60`;
- `INCONCLUSIVE` otherwise.

Here the upper bound is far above `0.25`, while the lower bound is far below `0.60`. The correct
decision is therefore `INCONCLUSIVE`, not pass and not fail. More independent rollouts of these
same eight fixed items could narrow rollout uncertainty. They would not add new benchmark items or
justify generalization to unseen questions.

## Read the outputs

Start with `report/report.md`, the one-page summary, or `report/report.html` in a browser.
`report/report-full.md` is the full human-readable report. Use `report/report.json` for every numeric result, including the claim statuses and
the overall decision. The Markdown and HTML reports show decisions only when the plan sets
`report.include_decisions: true`, which this plan does not. `observations.csv` gives one
auditable row per rollout, `design.resolved.csv` records the effective policy weights, and
`checksums.json` protects the generated artifact set against unnoticed edits.

`tests/test_example_reports.py` reruns the command above and requires every file in `report/`
to match byte for byte, so the committed report is always the one the current package writes.
