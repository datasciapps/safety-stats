# Releasing

`.github/workflows/release.yml` publishes the package when a version tag is pushed:

- `v<version>`, such as `v0.16.0`, publishes that version to PyPI. The tag must equal the
  package version.
- `v<version>rc<N>`, such as `v0.16.0rc1`, publishes a release candidate to TestPyPI. It is built
  from the tagged commit as version `0.16.0rc1`, so the candidate and the release are the same
  commit and no commit changes the version back. `N` counts from 1, without leading zeros.

Either tag must point at a commit on `main`. Before it publishes, the workflow checks the tag
against the version (`scripts/release_version.py`). It then builds the sdist and wheel, runs
`twine check`, and installs the wheel into a clean environment with the checks every pull
request's build job runs (`scripts/check_wheel.sh`).

The workflow does not check that the tagged commit's CI passed, or that a final release was
published as a candidate first. The steps below do both.

## One-time setup

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/). No API token is
stored in the repository. A trusted publisher is bound to the repository, the workflow file and
the environment, not to a tag, so the GitHub settings below decide who can publish.

1. On PyPI, add a trusted publisher for the project `inspect-safety-stats`. Until the first upload
   the project does not exist, so add it as a pending publisher. The form asks for:
   - **PyPI project name:** `inspect-safety-stats`
   - **Owner** and **Repository name:** this repository's
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi`. Fill it in: a publisher without an environment accepts uploads
     from any job of the workflow.
2. On TestPyPI, do the same with the environment `testpypi`.
3. In the GitHub repository settings, create the environments `pypi` and `testpypi`:
   - Under **Deployment branches and tags**, allow only tags matching `v*`.
   - Give `pypi` **required reviewers**, so that a release waits for a maintainer's approval
     after its checks pass. Required reviewers are not available on every GitHub plan for a
     private repository. Without them, only who may push a `v*` tag gates a release.
4. Add a tag ruleset for `v*` that restricts who may create tags and blocks updating and deleting
   them, so a published tag cannot be moved.

The publish action makes attestations by default. If the first candidate's upload fails on
them, set `attestations: false` on both publish steps in `release.yml` through a pull request.

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
   `git tag v0.16.0rc1 <commit> && git push origin v0.16.0rc1`. When the workflow has published
   it, install it from TestPyPI into a fresh environment and run the offline demo from a checkout
   of the same commit:

   ```bash
   python -m venv /tmp/rc && . /tmp/rc/bin/activate
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ 'inspect-safety-stats[inspect]==0.16.0rc1'
   inspect-safety-stats version
   python examples/demo/run_demo.py
   ```

   A candidate that needs a fix gets the fix through a pull request and a new candidate tag
   (`rc2`) on the new commit.
5. Tag the same commit as the candidate with the version and push it: `git tag v0.16.0 <commit>
   && git push origin v0.16.0`. After the `pypi` environment's reviewers approve, the workflow
   publishes to PyPI.
6. Create the GitHub release from the tag, with the version's `CHANGELOG.md` section as its notes.

## When something fails

- **A check fails before publishing** (the tag is not on `main`, the tag and the version
  disagree, a build or wheel check fails): nothing was uploaded. Fix the cause through a pull
  request and push the next candidate tag on the fixed commit. A final tag that failed this way
  is deleted and pushed again on the fixed commit, by someone the tag ruleset lets bypass it.
- **The publish job fails**, for example because a trusted publisher names another workflow or
  environment: fix the setting and re-run the failed job. It reuses the distributions the build
  job made.
- **A re-pushed tag** starts a new run, and the index refuses a file name it already has, so a
  published version is never replaced.

A version on PyPI cannot be replaced. A release that turns out wrong is fixed by the next
version, and the broken one can be yanked on PyPI.
