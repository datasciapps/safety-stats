# Changelog

Changes that a user can observe, newest first. Releases before this file was started are
described in their pull requests.

## Unreleased

### Added

- **Rollout files (experimental).** Results from any evaluation system can be analysed from a
  runs file and a rollouts file in JSON Lines or CSV: `analyze` and `validate` take
  `--runs` and `--rollouts` in place of `--logs`, and `source=RolloutFiles(...)` in place of
  `logs=` in Python. See [Rollout files](docs/rollout-files.md).
