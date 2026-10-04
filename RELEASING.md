# Releasing

Releases are published to PyPI by `.github/workflows/release.yml` when a version tag is pushed.

## One-time setup

On PyPI, add a trusted publisher for the `inspect-safety-stats` project: owner `aduncan001`,
repository `safety_stats`, workflow `release.yml`, environment `pypi`. In the GitHub repository,
create an environment named `pypi` (optionally with required reviewers). No API token is stored.

## Each release

1. Update the version in `pyproject.toml`, `src/inspect_safety_stats/version.py` and the
   "remains current in" line of `docs/independence-profiles.md`; the repository-consistency
   test fails if they disagree.
2. Move the release notes in `CHANGELOG.md` under the new version heading.
3. Merge to `main` with CI green.
4. Tag and push: `git tag v0.16.0 && git push origin v0.16.0`. The workflow refuses a tag that
   does not match the package version, builds, checks and smoke-tests the wheel, then publishes.
