# Rollout files

**Experimental.** The format and its options may change in a minor release.

`inspect-safety-stats` reads Inspect AI logs by default. Results from any other evaluation
system can be analysed by writing them as two plain files, a runs file and a rollouts file, and
passing them in place of the logs:

```bash
inspect-safety-stats analyze --plan plan.yaml --runs runs.jsonl --rollouts rollouts.jsonl \
  --output report/
inspect-safety-stats validate --plan plan.yaml --runs runs.jsonl --rollouts rollouts.jsonl
```

```python
from inspect_safety_stats import analyze
from inspect_safety_stats.ingest import RolloutFiles

report = analyze(
    plan="plan.yaml",
    source=RolloutFiles(runs="runs.jsonl", rollouts="rollouts.jsonl"),
)
```

The plan, the design manifest and every rule of the analysis are the same as for Inspect logs.

## File formats

A file whose name ends in `.csv` is read as CSV with a header row. Any other file is read as
JSON Lines: one JSON object per line, with blank lines ignored. Both files must be UTF-8.

JSON Lines keeps the type of every value, so prefer it whenever a score mapping depends on
types. In CSV, every cell is text, read this way:

| Cell | Value |
|---|---|
| empty | missing |
| `true` or `false` | a boolean |
| a JSON number, such as `1`, `0.25` or `-3e-2` | that number |
| anything else | the text itself |

A score written as the text `"1"` can therefore only be expressed in JSON Lines.

## The runs file

One record per evaluation run: one model, under one configuration, over some items.

| Field | Required | Meaning |
|---|---|---|
| `run_key` | yes | Identifies the run. Unique in the file. |
| `model` | no | The model evaluated. Part of the run's configuration. |
| `status` | no | The run counts as complete only when this is `success`. A run without a status counts as complete. |
| `started_at` | no | ISO 8601 time the run started, used to check that a confirmatory plan was frozen before it. A time without an offset is read as UTC. |
| `run_id` | no | A label grouping runs, reported with each observation. |

Every other field is an attribute of the run. Cohort `match` and `require_uniform` may name any
field of the run, including `model` and `status`. A run's configuration fingerprint covers
`model` and the fields its cohort lists in `require_uniform`, so list there every field that
must not drift between the runs of a cohort.

## The rollouts file

One record per rollout: one response of one run to one item.

| Field | Required | Meaning |
|---|---|---|
| `run_key` | yes | The run the rollout belongs to; must name a run in the runs file. |
| `item_id` | yes | The item, matched to `prompt_id` in the design manifest. |
| `repeat` | no | The repeat number, a positive integer. Defaults to `1`. |
| `rollout_id` | no | Identifies the rollout in the source system. Defaults to `run_key:item_id:repeat`. |
| `scores` | no | JSON Lines only: an object from score name to value. |
| `score_<name>` | no | The value of score `<name>`. A score is given either here or in `scores`, not both. |
| `error` | no | The rollout's terminal error, if it ended in one. |
| `error_stage` | no | `target_generation`, `post_generation` or `unknown` (the default). |
| `content_delivered` | no | `true` when the target delivered content before the error. |
| `provider_error_code` | no | The provider's error code, such as `content_filter`. |
| `retries` | no | How many times the rollout was retried. Defaults to `0`. |

Every other field is an attribute of the rollout, which `design.rollout_key` may name. Name the
neutral fields there:

```yaml
design:
  rollout_key: [run_key, item_id, repeat]
```

The plan's `inputs.score.name` and each judge's `score.name` select a score by name, and the
score mapping applies to its value as written.

An error with `error_stage: target_generation`, `content_delivered: false` and one of the policy
codes `bio_policy`, `content_filter`, `content_policy_violation`, `guardrail_blocked`,
`policy_violation`, `safety_filter` or `safety_violation` as its `provider_error_code` counts as
a safe outcome under `errors: end_to_end` with the `end_to_end_system` estimand, as for Inspect
logs. Any other error follows the plan's error policy.

## What is reported

Observations carry the run key in `eval_id`, the rollout id in `sample_id`, the item in
`inspect_id` and the repeat in `epoch`. The report's provenance lists the rollouts file with
its SHA-256. Each observation's id is a hash of the run key, rollout id, item and repeat, with
the score name, so it stays the same when the files are read again.

Rollouts are collected in the order of run key, item and repeat, whatever their order in the
file, so the order of the file does not change any estimate. A diagnostic about one rollout's
judge score names the rollout by its position among the records of the rollouts file, counting
from 1.
