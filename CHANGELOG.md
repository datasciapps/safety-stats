# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Added

- **Rollout files (experimental).** Results from any evaluation system can be analysed from a
  runs file and a rollouts file in JSON Lines or CSV: `analyze` and `validate` take
  `--runs` and `--rollouts` in place of `--logs`, and `source=RolloutFiles(...)` in place of
  `logs=` in Python. See [Rollout files](docs/rollout-files.md).

- **Plan schema 3.0.** Plans name run and rollout fields neutrally, so one plan works with
  any source: `run_key`, `model`, `status` and `run_id` for runs; `run_key`, `item_id`,
  `repeat` and `rollout_id` for rollouts; any other source field as `attributes.<name>`.
  `design.rollout_key` defaults to `[run_key, item_id, repeat]`, and `inputs.sample_join` is
  removed. Schema 2.0 plans keep their meaning and their hash. See
  [Schema versions](docs/plan-reference.md#schema-versions) for the migration.

### Changed

- **Inspect AI is an optional extra (breaking for installation).** `inspect-ai` and `pyarrow`
  are no longer installed by default. Install `inspect-safety-stats[inspect]` to read Inspect
  logs with `--logs` or `logs=`; without the extra, reading them fails with an error that names
  it. Rollout files need no extra. The `dev` extra includes it.

- **Report schema 3.0: neutral observation and provenance fields (breaking).** Reports name
  observation fields after no evaluation tool: `source_log`, `source_log_sha256`, `eval_id`,
  `sample_id`, `inspect_id` and `epoch` in `observations.csv` and `report.json` are now `source`,
  `source_sha256`, `run_key`, `rollout_id`, `item_id` and `repeat`. The cohort fields `eval_ids`
  and `evaluation_count` are now `run_keys` and `run_count`. Provenance replaces `source_logs` and
  `inspect_version_constraint` with `sources` and `source_format`, which names the format read
  and, for Inspect logs, the supported Inspect versions. Observation ids, and so every estimate,
  are unchanged. Plans may still name the old observation fields in `block_by`, `pair_key` and
  `cluster_key`. Update any script that reads these columns or keys.

- **Unchecked safeguards are reported.** A run without a status was silently counted as
  complete, and a confirmatory plan whose runs had no start time skipped the post hoc check
  without a word. Each now gives a warning that degrades integrity: `RUN_STATUS_UNOBSERVED`,
  `START_TIME_UNOBSERVED`, and `CONFIG_UNOBSERVED` for a run that reports no configuration. A
  run without a status still counts as complete. Analyses of Inspect logs, which always carry
  these fields, are unchanged.
