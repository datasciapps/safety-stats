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

- **Two-arm comparison sets.** `comparison_sets` accept two cohorts, where they needed at least
  three. A two-arm set gives the same estimates and bounds as the plain comparison written
  with the set's settings.

- **Help text for every command.** `analyze`, `validate`, `version`, `hash-plan`,
  `plan zero-failure`, `schema plan` and `schema report` showed an empty line in `--help`; each
  now has a one-line description.

### Changed

- **An unlabelled judge stratum is completed, not refused (changes results).** A validated
  judge calibration needed a human label in both judge strata. A stratum without one is now
  completed conservatively: its error rate is bounded by [0, 1], the correction is still
  reported, and a `JUDGE_STRATUM_UNLABELLED_COMPLETED` warning names the stratum. With both
  strata unlabelled the calibration is still refused. `JudgeCalibration` gains
  `completed_strata`. See
  [Judge misclassification](docs/judge-misclassification.md#estimation-and-error-budget).

- **Unbiased `worst_of_k` estimates (changes results).** The row estimate plugged each prompt's
  failure rate into `1 - (1 - p)^k`, which understates the risk. It is now the unbiased
  U-statistic, `1 - C(n - f, k) / C(n, k)` per prompt, weighted over prompts, so estimates
  rise. When a prompt has fewer than `k` classified rollouts the row reports no estimate, with
  a warning. The bounds are unchanged. `worst_of_k_estimate` is exported from `stats`.

- **Reports name inputs relative to the plan's directory (changes report bytes).** The design
  manifest, the judge validation manifest, the provenance `sources` and each observation's
  `source` were recorded as absolute local paths, so the same analysis wrote different bytes
  in different checkouts and revealed the local directory layout (#63). They are now relative
  to the plan's directory, in POSIX form. URIs, and paths with no relative form, keep their
  form apart from using forward slashes. Observation ids and validation manifests are
  unchanged.

- **A minimal plan is told what it is missing (#57).** Plan validation requires
  `outputs.prompt_bounds.family` whenever prompt bounds are enabled, pointwise ones included,
  where pointwise bounds without a family were silently absent; with prompt bounds disabled a
  family is no longer needed. The messages for a missing family and for a missing
  `expected_rollouts` column say what to add or which switch turns the requirement off, and a
  plan without risk measures gets a `PLAN_NO_RISK_MEASURES` warning.

- **Unquoted schema versions.** `schema_version: 2.0` and `3.0` written without quotes are
  accepted, and hash the same as their quoted forms. A schema 1.0 plan is refused with what
  schema 2.0 changed, and any other version with the supported ones.


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
