# Open-weight safety demo

One open-weight model, `Qwen/Qwen3.8-27B` in BF16, is evaluated in two configurations (thinking
off and thinking on) and judged by itself. This is a real-data case: the example was used for a
full study, served locally with vLLM, and the limitations that study exposed are recorded in the
[limitations log](../../docs/limitations.md). The study's run log, its presentation and its
registered plans are kept on the `demo/open-weights` branch, not here. Another
open-weight model runs the same way once its card's sampling is in the task (see
[Another open-weight model](#another-open-weight-model)), and a hosted Claude model can be run
through the same task, judged by the local model (see [A hosted model](#a-hosted-model)).

| File | Purpose |
|---|---|
| `open_weights_task.py` | One Inspect task: HarmBench's 300 standard and contextual text behaviors (DirectRequest, harmful compliance) and XSTest's 250 safe prompts (over-refusal). The `thinking` argument selects the configuration and the model card's sampling for it. The judge uses HarmBench's classifier rubric and XSTest's classification prompt. |
| `run_open_weights.py` | Runs both configurations in one Inspect call against a running vLLM server, writes `design.csv` and `plan.yaml` next to the logs, then analyzes them. Runs a hosted model in one configuration. |
| `provider_guard.py` | Halts a hosted-model run at the provider's first error, warning, unexpected stop or classifier decline beyond the allowance, and cancels its batches in progress. |
| `plan-template.yaml` | The analysis plan the runner fills in: cohorts by `task_arg_thinking`, the hierarchy `task_id -> category -> prompt_id`, unpaired comparisons of the two configurations, and six claims with their thresholds, frozen before the main run. |
| `judge_check.py` | Judges the published human-labelled sets with the main run's prompts and settings, and writes agreement counts. |
| `storyboard_numbers.py` | The storyboard's numbers from one report folder (see [Storyboard numbers](#storyboard-numbers)). |
| `chart_cards.py` | The storyboard's chart cards, as SVG, from one report folder (see [Chart cards](#chart-cards)). |
| `label_queue.py` | Orders rollouts for human labelling, writes the labelling sheet, and turns the labels into validation manifests (see [Human labels](#human-labels)). |

## Guards

- Models must be `vllm/...` or `mockllm/...`, and the scripts refuse to start without
  `VLLM_BASE_URL`, so Inspect never starts a server of its own. The API key comes from
  `VLLM_API_KEY`, never from the command line. The model under test may also be
  `anthropic/...`; the judge stays local.
- Logs hold prompts and completions. Output folders must be outside the repository and are
  created with mode 700.
- The XSTest prompts and both labelled sets are not in the repository. The scripts read them from
  a folder you name and check each file's SHA-256 against the published version.

## Running

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY="$(cat path/to/key)"
python examples/open_weights/judge_check.py --labels DATA --max-connections 128 --output RUN/judge
python examples/open_weights/run_open_weights.py --xstest-csv DATA/xstest_prompts.csv \
  --epochs 20 --max-connections 128 --output RUN/main
```

`--max-connections` is the number of requests in flight across both configurations and the judge.
It turns off Inspect's adaptive connection control, which would otherwise start at 20 requests.
`--per-category N` runs the first N prompts of every category, for a pilot.

## Storyboard numbers

```bash
python examples/open_weights/storyboard_numbers.py RUN/main/report --output NUMBERS.md
```

It reads only the content-free files `analyze` writes and prints five tables: what each epoch
would have reported as a single run, beside the study's interval; the chance of at least one
failure in k attempts for k up to 20, from the package's `worst_of_k_risk` at the per-prompt
rates, beside the intervals of the plan's worst-of-k measures; the same chance estimated without
bias, with a 95% interval for each k; every category against its claim's threshold, with, for
the open ones, the rollouts per prompt the exact one-sided binomial planner
(`certification_sample_size`, power 0.9) needs at the observed rate; and the comparisons. Rates
weight prompts as the plan does. Where the report has no estimate, because a prompt has no
classified rollout, the table says n/a.

Putting a prompt's observed rate x/n into 1 − (1 − p)^k runs low, because the curve is concave.
The unbiased table takes instead each prompt's share of k-subsets of its rollouts that hold a
failure, 1 − C(n − x, k)/C(n, k), whose expectation is exactly 1 − (1 − p)^k when k ≤ n. The
interval treats rollouts as independent, as the study's primary interval does. Hoeffding (1963)
showed that such a share is an average of means of ⌊n/k⌋ independent terms, one for each
disjoint group of k rollouts, so a Chernoff bound for ⌊n/k⌋ independent terms between 0 and 1
holds for it. Each prompt therefore enters the package's KL-Chernoff bounds for bounded terms
(`weighted_bounded_kl_chernoff_mean_bounds`) as ⌊n/k⌋ terms, with unclassified rollouts counted
safe for the lower end and failures for the upper one. At k = 1 this is the package's
rollout-level bound for one rate on its own; the interval widens in steps where ⌊n/k⌋ drops, and
beyond k = n/2 each prompt is a single term. The estimate needs k classified rollouts of every
prompt and the interval k rollouts; short of that the cell says n/a (limitations log, L51). The
package now offers the same interval for a plan's `worst_of_k` measures, as an experimental opt-in
(`direct_interval: true`, see the [plan reference](../../docs/plan-reference.md)).

## Chart cards

```bash
python examples/open_weights/chart_cards.py RUN/main/report --output-dir CARDS
```

It reads the same content-free files and writes three of the storyboard's cards as SVG, 1600 by
900, each in a light and a dark version (`NAME.svg` and `NAME-dark.svg`):

- `single-runs` (1:10): each epoch's score as a single run would report it, one dot per epoch,
  above the study's interval and the claim's threshold.
- `asking-again` (1:45): the unbiased chance of at least one failure in k attempts for k from 1 to
  20, with one bar per k for its interval, so the steps where ⌊n/k⌋ drops stay visible.
- `two-sides` (3:40): one box per cohort spanning its harm and over-refusal intervals, against
  both thresholds.

`--cohort` picks the cohort of the first two cards (default `thinking_on`). The SVG is written by
hand, so the example needs no plotting library. Since nothing renders the cards in the tests, the
script checks the layout itself: every text, with an estimate of its width, against the card's
edges, every other text, and every mark it must not cover. A label tries a few places and takes
the first clear one; when none is, the script lists the collisions and writes nothing.

## Human labels

The judge's verdicts are the data, so the question is how often they are wrong. `label_queue.py`
picks the rollouts people label, and their order, so that one set of labels serves two uses:

- **Calibration.** A simple random sample of fixed size within each judge stratum (a cohort and
  task's rollouts the judge called safe, or unsafe), which the package's validated judge
  calibration needs. The sizes are set before any label: a few rows for every stratum first
  (`--calibration-floor`; the package refuses a stratum without labels), then greedily, by how
  much each extra label narrows the stratum's exact error-rate interval, times the stratum's size
  and how close its cell is to its threshold.
- **Oversampling.** The remaining rows go to fine strata, which split each judge stratum by the
  other judges' verdicts and by whether the prompt's other rollouts drew the opposite verdict.
  Every fine stratum gets a few rows first (`--floor`), then rows go where the expected error
  variance is largest. Disagreements are labelled far more often than a random sample would label
  them, and since every row's chance of selection is known, the summary weights them back.

Each judge stratum is shuffled once with the seed, the calibration rows come from the front of
the shuffle and the oversampling rows from beyond it, and nothing about the order depends on a
label. Any prefix of the queue is therefore a valid design: label in order, without skipping,
settle every "unsure" before making manifests, and stop when the time runs out, never because of
what the labels show.

```bash
python examples/open_weights/label_queue.py queue --primary RUN/main/report \
  --judge other=RUN/other-judge/report --seed N --size 1200 --output LABELS/queue
python examples/open_weights/label_queue.py sheet --queue LABELS/queue --output PRIVATE/sheet.csv
python examples/open_weights/label_queue.py manifest --queue LABELS/queue \
  --labels PRIVATE/sheet.csv --output LABELS/manifests
```

`queue` and `manifest` write only identifiers, verdicts, counts, and the paths of the logs and
reports relative to the queue folder. `sheet` writes the prompts and
answers to be labelled, with one yes-or-no question per row and the judge's rules in a file beside
it, but not the judges' verdicts. Its folder must be outside the repository and closed to everyone
else (mode 700); the sheet is a new file with mode 600. `manifest` writes the calibration rows as
one `judge-labels-<task>.csv` per task, and `summary.md` with each cell's corrected share and exact
bounds and each judge's error rates.

- The manifests are per task: the package pools a cohort's tasks into one judge stratum and
  rejects labels for rollouts its plan does not hold, so per-benchmark error rates need one plan
  per task. The calibration tier is a random order within each cohort, task and verdict, not
  within a cohort's verdict over all tasks, so it is a valid labelling queue
  (`inputs.judge_calibration.protocol` with `tier: calibration`) only for such a plan per task,
  and that plan's queue may list only its task's rows: the package refuses a queue row its run
  did not classify.
- The demo's plans were frozen without a judge calibration, and adding one afterwards is post
  hoc, so corrections from labels on these runs are exploratory. A plan that is to use them
  declares `inputs.judge_calibration` with `mode: validated` before its rollouts are generated.
- Certifying one cell takes far more labels in its judge-safe stratum than a queue spread over
  every cell gives it. `--cell COHORT/TASK` keeps a queue to the cells named.

## Another open-weight model

Each configuration samples as the card of the model under test says, from `MODEL_CARDS` in
`open_weights_task.py`; the runner refuses a `vllm/` model with no entry there, rather than give
it another model's sampling. `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` has one: its chat
template switches thinking with the same `enable_thinking` argument, it samples at temperature
1.0 and top_p 0.95 in both configurations, and it takes no reasoning effort, so the runner refuses
`--reasoning-effort` for it. Serve it under its own name and run it as above, judged by itself:

```bash
python examples/open_weights/run_open_weights.py \
  --model vllm/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --xstest-csv DATA/xstest_prompts.csv --epochs 20 --max-connections 128 --output RUN/nemotron
```

## A hosted model

`--model anthropic/...` runs a hosted Claude model in one configuration, judged by the local
model on the vLLM server. Claude 5-series models always think, so the run takes `--arms on`; only
`--reasoning-effort` is set, the model samples as any API caller gets it (the current API rejects
temperature, top_p and top_k), and `--max-tokens-on` defaults to 32000 for thinking and answer
together. Requests go through the Message Batches API at half price (`--batch-size 0` for none).
The key is read from a file into `ANTHROPIC_API_KEY`; a run refuses `ANTHROPIC_AUTH_TOKEN` and
`ANTHROPIC_BASE_URL`, so requests go to the account the key names. A key that is not scoped to a
workspace needs `--workspace-id wrkspc_...`, which is sent as `anthropic-workspace-id` on every
request, batch cancellation included. The `anthropic` package must be installed.

When the provider's safety classifier declines before delivering anything (stop reason `refusal`,
recorded by Inspect as `content_filter`), the rollout is not judged: it counts as safe on
HarmBench and as a refusal on XSTest, and its category is kept in the score's metadata. Content
delivered before a decline mid-answer is judged as usual. The plan's description says so. The
example's scorer does this itself; for other tasks, the package's experimental plan field
`inputs.score.content_filter_stops` makes the same choice at analysis time (see the
[plan reference](../../docs/plan-reference.md)).

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY="$(cat path/to/key)"
python examples/open_weights/run_open_weights.py --model anthropic/claude-opus-5-5 --arms on \
  --reasoning-effort medium --api-key-file ~/.anthropic-api.key --max-provider-blocks 0 \
  --xstest-csv DATA/xstest_prompts.csv --per-category 7 --epochs 1 \
  --max-connections 32 --judge-max-connections 64 --output RUN/hosted-pilot
```

For the main run, drop `--per-category` and set `--epochs`. `--max-connections` is the number of
hosted requests outstanding at once, and so caps the size of a batch; `--judge-max-connections`
is the judge's share of the local server.

### Failing closed

A hosted run spends the key's credit under its account, so it stops at the first sign of trouble
instead of retrying. `provider_guard.py` lists what halts it: any exception from a request (every
HTTP error the API returns, a timeout, a network failure, an errored, expired or canceled batch
item), an output carrying an error, a stop reason other than an answer or the token cap, a
classifier decline beyond `--max-provider-blocks`, and any warning from the Anthropic SDK or
Inspect's Anthropic provider and batcher. No request, batch creation or batch poll is retried.
The local judge keeps Inspect's retries, and a judge error that outlasts them also stops the run.
Before any request, the request parameters are built offline and the run refuses a deprecated
model, sampling parameters, extra fields, refusal fallbacks, a warning, or an effort that would
not be sent.

On a halt the runner exits with the reason, writes it to `halt.json` in the output folder (with
the classifier declines by category, the warnings and the cancelled batches, but no prompt or
answer text), and cancels the run's batches still in progress, during the halt and again after
Inspect returns. What the guard cannot do:

- It sees only what the API returns. A notice about the account that arrives another way, such
  as by email or in the Console, does not stop a run.
- Requests already sent when it halts are processed and billed: up to `--max-connections` of
  them. With batches a decline is seen only when its batch ends, after the rest of that batch was
  processed.
- Cancelling covers every batch in progress in the key's workspace created from five minutes
  before the run started. Do not run other batch jobs with the same key at the same time.
