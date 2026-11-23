# Two versions of a harness, one model

A harness is everything around the model: the system prompt, the tools, the filters, the agent
loop. This example asks the question a harness change raises: with the model unchanged, does
version 2 of the harness fail the safety evaluation more or less often than version 1? It runs
the same 12-prompt refusal evaluation in Inspect once inside each version, then compares the two
failure rates. The plan declares no threshold: the question is whether the versions differ, not
whether either passes.

It needs no API key and takes about half a minute. The model and the harness are simulated.
[harness_eval.py](harness_eval.py) gives each harness version a fixed failure probability for
each prompt and seeds every draw with the version, the prompt and the epoch, so the numbers come
out the same on every run and every machine. They show how the analysis works and say nothing
about any real harness.

## Run it

From the repository root, after `python -m pip install -e '.[inspect]'`:

```bash
python examples/harness_versions/run_example.py
```

The script writes the logs to `logs/harness-versions` and the report to
`reports/harness-versions`, replacing the output of an earlier run, and prints:

```text
Mean failure rate over all prompts, 95% simultaneous intervals
  harness v1: 0.050 [0.017, 0.108]
  harness v2: 0.117 [0.062, 0.193]

v2 minus v1, all prompts: +0.067 [-0.046, +0.176]
The interval includes zero: no difference is established. Its upper end is the largest increase in the failure rate that the runs leave open.
```

## The same steps, one at a time

```bash
inspect eval examples/harness_versions/harness_eval.py -T harness=v1 \
  --model mockllm/model --log-dir logs/harness-versions
inspect eval examples/harness_versions/harness_eval.py -T harness=v2 \
  --model mockllm/model --log-dir logs/harness-versions

safety-stats analyze --plan examples/harness_versions/plan.yaml \
  --logs logs/harness-versions --output reports/harness-versions
```

Start from an empty `logs/harness-versions`, since logs left there by an earlier run would be
analyzed too.

## How the plan tells the versions apart

Both runs use the same model, so the model name cannot separate them. Inspect records each task
argument in the log, and the two cohorts of [plan.yaml](plan.yaml) select the runs by it:
`task_arg_harness: v1` and `task_arg_harness: v2`. The comparison set `harness` takes v1 as the
reference and reports v2 minus v1 and v2 over v1, overall and per risk domain.

## What to look at

Version 2 failed on 28 of 240 rollouts and version 1 on 12, so its observed rate is 0.067 higher.
The interval for the difference, from -0.046 to +0.176, includes zero: on these 240 rollouts per
version the runs do not establish that version 2 is less safe. They do not establish that it is
as safe either. The upper end, +0.176, is the largest increase in the mean failure rate that the
runs leave open at the 95% level. If an increase of that size would matter, run more rollouts
before shipping version 2. The [demo](../demo/README.md) shows the other outcome, an interval
wholly below zero.

The bounds cover these 12 prompts, not prompts beyond them, and they assume the rollouts are
independent draws; the `prompt` independence profile, reported beside the primary `rollout`
profile, shows how much wider they get when the rollouts of a prompt share a cause.

## With a real harness

Replace `simulated_harness()` in [harness_eval.py](harness_eval.py) with the solver chain of each
version, for example a different `system_message()` or tool set chosen by the `harness`
argument, pass a real model name to `--model`, and change the `model` match of both cohorts in
the plan to that name. Replace the scorer as well: it only looks for the marker the simulated
harness writes.
