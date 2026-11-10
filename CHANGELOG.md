# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Added

- **Rollout files (experimental).** Results from any evaluation system can be analysed from a
  runs file and a rollouts file in JSON Lines or CSV: `analyze` and `validate` take
  `--runs` and `--rollouts` in place of `--logs`, and `source=RolloutFiles(...)` in place of
  `logs=` in Python. See [Rollout files](docs/rollout-files.md).

### Changed

- **Unchecked safeguards are reported.** A run without a status was silently counted as
  complete, and a confirmatory plan whose runs had no start time skipped the post hoc check
  without a word. Each now gives a warning that degrades integrity: `RUN_STATUS_UNOBSERVED`,
  `START_TIME_UNOBSERVED`, and `CONFIG_UNOBSERVED` for a run that reports no configuration. A
  run without a status still counts as complete. Analyses of Inspect logs, which always carry
  these fields, are unchanged.
