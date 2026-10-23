# Offline demo

This example runs the whole workflow on one machine in about half a minute, with no API key. It
runs a 12-prompt refusal evaluation in Inspect for two configurations, then analyzes their logs
under a frozen plan and compares the two.

The model is simulated. Each configuration fails on each prompt with a fixed probability, set in
[demo_eval.py](demo_eval.py), and every draw is seeded by configuration, prompt and epoch. The same
commands therefore write the same logs and the same report on every machine. The numbers show
how the analysis works; they say nothing about any real model.

## Run it

From the repository root, after `python -m pip install -e .`:

```bash
python examples/demo/run_demo.py
```

The script writes the logs to `logs/demo` and the report to `reports/demo`, replacing the output
of an earlier demo run. It then prints the mean failure rate of each configuration, overall and
per risk domain, and the candidate-minus-baseline difference. Open `reports/demo/report.html` for
the interactive report.

## The same steps, one at a time

```bash
inspect eval examples/demo/demo_eval.py \
  --model mockllm/baseline,mockllm/candidate --log-dir logs/demo

inspect-safety-stats validate --plan examples/demo/plan.yaml --logs logs/demo

inspect-safety-stats analyze --plan examples/demo/plan.yaml \
  --logs logs/demo --output reports/demo

python examples/demo/show_results.py reports/demo
```

The part of `--model` after `mockllm/` picks the simulated configuration. `analyze` refuses to
write into a report directory that is not empty, so remove `reports/demo` first or choose a new
directory. Run the eval into an empty `logs/demo`: earlier logs there would be analyzed too.

## What to look at

- **Intervals, not just rates.** `show_results.py` prints each estimate with a 95% simultaneous
  interval. The baseline fails on 5% of its bio rollouts, yet the interval runs from 0 to about
  0.31: 80 rollouts cannot pin the rate down further while coverage holds jointly across every
  displayed interval.
- **Zero failures is not zero risk.** The candidate never fails on the cyber prompts, and its
  upper bound there is still about 0.26.
- **The comparison.** The candidate-minus-baseline interval over all prompts lies wholly below
  zero, so the candidate fails less often on this fixed design.
- **The audit trail.** `report.md` and `report.json` record the plan's SHA-256, the data
  integrity checks and every method used. `checksums.json` holds the SHA-256 of every other file in the report.

## Make it real

To evaluate real models, replace `simulated_model()` with `generate()` in
[demo_eval.py](demo_eval.py), and pass real model names to `--model`. Then change the two cohort
filters in [plan.yaml](plan.yaml) to match. The marker-based scorer is a stand-in, not a
validated safety scorer. The [fixed-design example](../fixed_design/README.md) and the
[HarmBench example](../real_harmbench/README.md) show real evaluations.
