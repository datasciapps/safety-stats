# Releasing

`.github/workflows/release.yml` publishes the package when a version tag is pushed:

- `v<version>`, such as `v0.16.0`, publishes that version to PyPI. The tag must equal the
  package version.
- `v<version>rc<N>`, such as `v0.16.0rc1`, publishes a release candidate to TestPyPI. It is built
  from the tagged commit as version `0.16.0rc1`, so the candidate and the release are the same
  commit and no commit changes the version back.

Either tag must point at a commit on `main`. Before it publishes, the workflow checks the tag
against the version (`scripts/release_version.py`). It then builds the sdist and wheel, runs
`twine check`, and installs the wheel into a clean environment with the checks every pull
request's build job runs (`scripts/check_wheel.sh`).

## One-time setup

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/). No API token is
stored in the repository.

1. On PyPI, add a trusted publisher for the project `inspect-safety-stats` with this repository's
   owner and name, the workflow `release.yml` and the environment `pypi`. Until the first upload
   the project does not exist, so add it as a pending publisher.
2. On TestPyPI, do the same with the environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi`. Give `pypi`
   required reviewers, so that a release waits for a maintainer's approval after its checks pass.

## Each release

1. In one pull request, set the version in `pyproject.toml`, `src/inspect_safety_stats/version.py`,
   `CITATION.cff` and the "remains current in" line of `docs/independence-profiles.md`. The
   repository-consistency test fails if they disagree. In the same pull request, move the
   `Unreleased` notes in `CHANGELOG.md` under a heading with the version and the date.
2. Rehearse the release locally on that branch: `scripts/release-rehearsal.sh <branch>`. It
   builds the wheel, installs it on every supported Python version and runs the tests against the
   installed package.
3. Merge the pull request with CI green.
4. Tag the merge commit as a candidate and push the tag:
   `git tag v0.16.0rc1 <commit> && git push origin v0.16.0rc1`. Install it from TestPyPI into a
   fresh environment and run the offline example:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ 'inspect-safety-stats[inspect]==0.16.0rc1'
   ```

   A candidate that needs a fix gets the fix through a pull request and a new candidate tag
   (`rc2`) on the new commit.
5. Tag the same commit with the version and push it: `git tag v0.16.0 <commit> &&
   git push origin v0.16.0`. After the `pypi` environment's reviewers approve, the workflow
   publishes to PyPI.
6. Create the GitHub release from the tag, with the version's `CHANGELOG.md` section as its notes.

A version on PyPI cannot be replaced. A release that turns out wrong is fixed by the next
version, and the broken one can be yanked on PyPI.
