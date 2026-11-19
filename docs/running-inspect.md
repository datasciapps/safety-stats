# Running Inspect for analysis

inspect-safety-stats reads the logs of an evaluation you run yourself in
[Inspect](https://inspect.aisi.org.uk/). How you run it decides what the analysis can use: how
many rollouts each prompt has, which runs form a cohort, and whether long answers survive. This
page covers four settings that matter for that and whose defaults can get in the way: epochs,
cohorts by task argument, the request timeout, and the connection limit that a target model and
an LLM judge share.

The Inspect details were checked against Inspect 0.3.290. Inspect changes quickly, so check them
against the version you run.

## Epochs: one scored rollout per epoch

Run every prompt several times with `--epochs N`, and keep every epoch's score with
`--no-epochs-reducer`:

```bash
inspect eval my_task.py --model provider/model --epochs 20 --no-epochs-reducer --log-dir logs/mine
```

The package reads the unreduced score of every sample epoch, so each epoch is one rollout. The
manifest's `expected_rollouts` must equal the number of finished epochs per prompt and cohort.
Retries of a failed sample do not add rollouts.

## Cohorts by task argument

A cohort is the set of runs that the plan's `cohorts.*.match` selects, and all of its runs must
share one configuration. The package fingerprints each run's task name, task version, model,
generation config, solver and its arguments, sandbox, dataset name and epochs, plus the fields
the cohort's `require_uniform` names. When the runs of a cohort differ in any of these,
`validate` and `analyze` report `CONFIG_DRIFT` as an error.

So one cohort cannot span two Inspect tasks. To evaluate several prompt sets together, such as
harmful requests and safe prompts for over-refusal, put them in one task, and tell the prompt
sets apart with a manifest column that groups filter on.

To compare configurations of one model, such as thinking on and off, write one task with a task
argument and run it once per value:

```bash
inspect eval my_task.py -T thinking=false --model provider/model --epochs 20 --no-epochs-reducer --log-dir logs/mine
inspect eval my_task.py -T thinking=true --model provider/model --epochs 20 --no-epochs-reducer --log-dir logs/mine
```

Every task argument becomes a run field named `task_arg_<name>`, which a cohort can match. In a
schema 3.0 plan:

```yaml
cohorts:
  thinking_off:
    match: {model: provider/model, attributes.task_arg_thinking: false}
    require_uniform: [attributes.task_version, attributes.model_generate_config]
  thinking_on:
    match: {model: provider/model, attributes.task_arg_thinking: true}
    require_uniform: [attributes.task_version, attributes.model_generate_config]
```

A schema 2.0 plan writes `task_arg_thinking` without the `attributes.` prefix (see
[Schema versions](plan-reference.md#schema-versions)).

To split one cohort's run across machines, give every machine the same task, arguments and
`--epochs`, and divide the prompts between them, for example with `--sample-id`.

### What a report records of a configuration

The report records the configuration fingerprints, not the values: `report.json` lists each
cohort's under `cohorts[].configuration_fingerprints`, and `observations.csv` gives the
fingerprint of each rollout's run in `configuration_fingerprint`. A fingerprint is the SHA-256
of the fields and their values, as Inspect's `evals_df` reads them, written as JSON with sorted
keys and no whitespace. Equal fingerprints mean equal configurations. The values themselves can
hold credentials and endpoints, such as request headers in the generation config's
`extra_headers`, a base URL or the path of a sandbox's configuration, so no report or
diagnostic copies them from the logs, with one exception: when `outputs.judge_claims` is on,
`report.json`, `report-full.md` and `report.html` record each scorer's options and metadata as
the logs give them. Of the rest, only the model is named.

A fingerprint hides a value only as well as the value is hard to guess. It has no salt, so the
same configuration always gives the same fingerprint, which keeps reports byte-identical and
lets two of them be compared. Someone who knows the rest of a run's configuration can hash
candidate values for the one they do not know and compare the result with the report. A value
with few possibilities, such as an internal IP address, a port or a user name in a path, can be
confirmed or recovered that way in seconds. A long random credential, such as an API key or a
bearer token, cannot. If a run's configuration holds a guessable value that must stay private,
share the report only with readers who may see the logs.

## Request timeout

Inspect gives each HTTP request to a model 600 seconds by default. Requests are not streamed, so
the client sees nothing until the whole answer is ready. A local server under full load can be
slower than that. When many requests share a GPU, each gets a few tokens per second, and an
answer of a thousand tokens or more can take longer than 600 seconds. Inspect then drops the
request, loses its work, and retries it after an exponential backoff. In one run with 160
requests in flight on a single GPU server, 22 requests timed out this way, and the run took
34 minutes instead of 24. With a longer timeout, the longest request took 1,414 seconds.

Set the limit in the environment before Inspect starts:

```bash
export INSPECT_HTTP_REQUEST_TIMEOUT=14400   # seconds; 4 hours
```

This applies to the providers whose HTTP client Inspect builds with its own defaults, the
OpenAI-compatible ones and vLLM among them. A provider with its own client may have its own
limit.

## Connections: the target and the judge

By default Inspect controls concurrency adaptively. It starts at 20 requests to a model, stays
between 10 and 100, and backs off when the provider signals a rate limit. Setting
`max_connections` (`--max-connections`) turns this off. The limit is then fixed, and the number
of samples Inspect runs at once (`max_samples`) defaults to the same number. A local server that
returns no rate-limit signals is usually better served by a fixed limit chosen for it than by
the adaptive start of 20.

Choose the limit for what the server can hold. Past the number of sequences its KV cache keeps
for answers of the real length, a vLLM server preempts and recomputes requests, and throughput
falls.

When the scorer calls an LLM judge, the judge is a second model with its own connection limit,
and the two may share one pool:

- Inspect shares a pool between models of one provider that return the same connection key.
  vLLM's key is the server's endpoint. A target and a judge served by one vLLM server share one
  pool, and the limit counts the requests of both.
- The limits of a shared pool are set by whichever model creates the pool first. Give the
  judge's `GenerateConfig` the same `max_connections` as the evaluation, so that the outcome
  does not depend on which model creates the pool first.
- OpenAI-compatible providers key the pool by API key and model, so a judge that is a different
  model keeps a pool of its own. Its `max_connections` then caps the judge calls alone.

Several tasks run in one `inspect_ai.eval(...)` call with `max_tasks` set to their number share
the server too. While one configuration finishes its last long answers, the other keeps the
server busy.

## Checklist

- `--epochs N --no-epochs-reducer`, and `expected_rollouts: N` in the manifest.
- One task per cohort. Configurations differ by task argument, matched as
  `attributes.task_arg_<name>`.
- Runs split across machines share every setting and divide the prompts.
- `INSPECT_HTTP_REQUEST_TIMEOUT` set well above the slowest answer under full load.
- An explicit `--max-connections` for a local server, below what its cache holds. The same limit
  goes on a judge that shares the server.
