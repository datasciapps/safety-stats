# Offline demo

The demo runs a 12-prompt refusal evaluation in Inspect for two configurations, a baseline and a
candidate, then analyzes the logs under the plan that `safety-stats init` writes for
them and compares the two. It needs no API key and takes about half a minute.

The model is simulated. [demo_eval.py](demo_eval.py) gives each configuration a fixed failure
probability for each prompt and seeds every draw with the configuration, the prompt and the
epoch. The scores, and so the numbers in the report, come out the same on every run and every
machine. They show how the analysis works and say nothing about any real model.

## Run it

From the repository root, after `python -m pip install -e '.[inspect]'`:

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

safety-stats validate --plan examples/demo/plan.yaml --logs logs/demo

safety-stats analyze --plan examples/demo/plan.yaml \
  --logs logs/demo --output reports/demo

python examples/demo/show_results.py reports/demo
```

The name after `mockllm/` selects the simulated configuration. Start from an empty `logs/demo`:
`validate` reports logs left there by an earlier run as `ROLLOUT_EXCESS`, and the data integrity
becomes invalid. `analyze` refuses to write into a
directory that is not empty, so delete `reports/demo` first or choose another output directory.

## Where the plan comes from

[plan.yaml](plan.yaml) and [design.csv](design.csv) are what `init` writes for the demo logs:

```bash
safety-stats init --logs logs/demo --failure-values FAIL \
  --group-by risk_domain --baseline mockllm/baseline
```

`init` writes `safety-plan.yaml`, which was renamed to `plan.yaml` here. Each sample of the demo
task carries its risk domain (cyber, bio or privacy) in its metadata, and `--group-by` breaks
the rates down by it. With two models, `init` compares the candidate with the baseline. The
header of the plan lists each choice `init` made and the assumption behind it, rollout
independence first.

One edit was made after `init`: two risk measures, `cvar_90` (the mean of the worst tenth of
prompts) and `worst` (the worst prompt). They are what makes the report emit a warning, described
below.

## What to look at

`show_results.py` prints each mean failure rate with its 95% simultaneous interval, over all
prompts and per risk domain. The baseline fails on 5% of its bio rollouts, and the interval runs
from 0.004 to 0.182. Eighty rollouts do not pin the rate down when the 95% has to hold for every
displayed interval at once.

The candidate never fails on a cyber prompt, and its upper bound there is still 0.074. Clean
rollouts only bound the rate from above.

Over all prompts, the candidate-minus-baseline difference is -0.196, with an interval from -0.294
to -0.088. The whole interval is below zero, so on this design the candidate fails less often.
The report also holds the ratio of the two rates and the difference within each risk domain.

The script ends with the report's warnings, `TAIL_MEASURES_UNINFORMATIVE`, one for each model.
`cvar_90` and `worst` are built from a bound on each prompt, so their precision depends on the
rollouts per prompt, not on the total. With 20 rollouts, a prompt that never fails is still
bounded only at 0.291, so even if no prompt failed, the upper bounds of both measures would lie
more than the plan's precision target of 0.05 above their estimates. About 134 rollouts per
prompt would be needed. The warnings leave the report valid. They say that these two measures
cannot be narrow on this design, whatever the models do.

`report.md` is the one-page summary: the warnings come first, with the assumption they check,
and the page ends with the assumptions used. `report-full.md` and `report.json` also record the
plan's SHA-256, the integrity checks, the methods used and the assumptions the analysis rests
on. `checksums.json` holds the SHA-256 of every
other file in the report.

## With a real model

Replace `simulated_model()` with `generate()` in [demo_eval.py](demo_eval.py), pass real model
names to `--model`, and run `init` on the new logs for a plan and design that match them. Replace
the scorer as well. It only looks for the marker that the simulated
model writes, so a real model would pass every rollout. The
[fixed-design example](../fixed_design/README.md) and the
[HarmBench example](../real_harmbench/README.md) are real evaluations.
