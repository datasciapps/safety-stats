<!--
CONTRIBUTING.md is the contract for every change; AGENTS.md is the short version.
Delete the comments as you fill this in.
-->

## What and why

<!-- One concern per pull request. -->

Closes #

## Change class

<!-- CONTRIBUTING section 3. Anything that changes an output byte is B, a bug fix included. -->

- [ ] A. Behaviour-neutral
- [ ] B. Behaviour change (issue agreed by a maintainer, changelog entry, regenerated outputs in a separate commit)
- [ ] C. New capability (everything in B, plus CONTRIBUTING section 8)

## Gate

<!-- Run locally; say on which operating system. Run `ruff check` only, never `ruff format`. -->

- [ ] `ruff check src tests examples research scripts`
- [ ] `mypy src/safety_stats`
- [ ] `pytest`

Operating system:

## Evidence

<!--
Paste the verdict:
- bug fix: `python scripts/prove_red.py <tests>` reports PROVEN;
- class A: `python scripts/compare_artifacts.py` reports IDENTICAL.
Once the change is committed, pass `--base <commit>`.
-->

## Golden outputs and docs

<!-- Which expected outputs, fixtures, docs and CHANGELOG.md entries change, and why. -->
