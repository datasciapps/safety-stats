# Contributing to inspect-safety-stats

This package produces statistical claims that people use to make safety decisions about AI
systems, and institutions already consume its reports. That shapes everything below:
**correctness, conservativeness, and stable, reproducible outputs come before speed or
convenience.** None of this is meant to keep contributors out. It is meant to make it safe for
many people to contribute.

Items marked **[proposed]** are not yet agreed by the maintainers. Treat them as the intended
direction, and argue with them in an issue. Items marked **[gap]** describe a rule the code does
not meet yet, with the issue that tracks it.

Contents:
[setup](#1-development-setup) · [before you code](#2-before-you-write-code) ·
[change classes](#3-change-classes) · [protected interfaces](#4-protected-interfaces) ·
[engineering](#5-engineering-practices) · [cross-platform](#6-linux-macos-and-windows) ·
[tests](#7-tests) · [statistics](#8-statistical-contributions) ·
[data and security](#9-data-secrets-and-sensitive-content) · [review](#10-pull-requests-and-review) ·
[commits](#11-commits-and-branches) · [releases](#12-releases-and-deprecation)

---

## 1. Development setup

You need Python 3.11–3.13 on Linux, macOS, or Windows, and nothing else. No compiler, Rust
toolchain, GPU, or provider API key is needed to develop, test, or run any part of the package.

```bash
python -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The gate that CI runs, and that every pull request must pass:

```bash
ruff check src tests examples research scripts
mypy src/inspect_safety_stats        # strict
pytest
```

- **Lint:** run `ruff check` only. Never run `ruff format` over the tree: it would rewrite
  hundreds of untouched lines and bury the real change in review.
- **Optional extras:** `[openai]` and `[translation]` are only for the live examples. Nothing in
  the test suite may need them.
- **`uv`:** works too (`uv venv`, then `uv pip install -e ".[dev]"`). Do not commit `uv.lock`;
  CI installs from `pyproject.toml`.

## 2. Before you write code

- **Open or claim an issue first** for anything beyond a typo or a docs fix. Say which change
  class (section 3) you expect. Two implementations of the same estimator on two branches are
  expensive to reconcile.
- **Large or scope-changing work needs a short design note first.** That covers a new subsystem,
  removing a feature, or changing a default method. Open an issue that states the problem, the
  options, and the user impact. Scope decisions are made there, not by letting a branch drift.
- **One concern per pull request.** A refactor and a behaviour change never share a pull request.

## 3. Change classes

Every pull request declares one of these.

| Class | What it means | What it requires |
|---|---|---|
| **A. Behaviour-neutral** | Refactor, performance, typing, internal docs | Every report artifact byte-identical before and after. Show it in the pull request with the output of `python scripts/compare_artifacts.py`. |
| **B. Behaviour change** | Anything a user can observe: a number, label, default, diagnostic, exit code, file name, file content, or schema field | An issue agreed by a maintainer *before* merge; a changelog entry; regenerated golden outputs in a **separate commit** that explains every difference; a schema-version decision (4.2, 4.3) |
| **C. New capability** | A new interval method, risk measure, judge mode, report artifact, or CLI command | Everything in B, plus section 8 |

- **Bug fixes:** a bug fix that changes output is class **B**, even when the old output was
  wrong.
- **Less-conservative changes:** a change that makes bounds less conservative (narrower, more
  certifying, fewer blocked prompts) needs a second approving reviewer.
- **New capabilities land as experimental first:**
  - they are opt-in in the plan;
  - the docs and CLI help say they are experimental;
  - they graduate after a release cycle with docs, goldens, and no open correctness issue.

## 4. Protected interfaces

Downstream users and institutions rely on these. Changing one is a class **B** change and needs
maintainer agreement in an issue. That includes renaming, reordering, changing a default,
removing, and tightening validation so that previously accepted input is rejected.

### 4.1 Python API

- **`inspect_safety_stats.__all__`:** `analyze`, `validate`, `load_plan`, `AnalysisPlan`,
  `SafetyReport`, `render_visualization`, `write_visualization`, `__version__`. These are the same
  objects as in their source modules. The package root loads them on first use (see 5.1).
- **`inspect_safety_stats.stats.__all__`:** for every exported function, its name, parameters,
  defaults, return type, and the fields of the dataclasses it returns.
- **`SafetyReport.write(...)`** and the set of files it writes.
- **Experimental:** `inspect_safety_stats.generative.__all__` may change in any minor release.
- **Internal:** anything not in an `__all__`, or with a leading `_`, is internal. Class A
  pull requests may change it freely, provided report artifacts stay byte-identical.

### 4.2 Analysis plan (YAML/JSON), `schema_version: "2.0"` or `"3.0"`

Plans are strict (`extra="forbid"`).

**Compatible within a schema version:**

- a new **optional** field whose default reproduces the previous behaviour exactly;
- a new opt-in enum value, such as a new method or measure.

**Breaking, needing a schema-version bump and a migration path:**

- removing or renaming a field or an enum value;
- changing a default;
- changing what an existing value means.

**Plan hash:** `inspect-safety-stats hash-plan` must give an unchanged plan file the same hash
from any directory (#42).

- **[gap]** It does not yet survive package upgrades, because defaults are hashed (#56).

### 4.3 Report outputs

- **`report.json`:** field names, types, and meanings. `inspect-safety-stats schema report` is
  the contract. The report carries its own `schema_version`.
- **Output files:** the names of the files in the report directory, and the CSV column names
  **and their order**.
- **`checksums.json`:** what it covers, and its semantics.
- **Determinism:** the same inputs, plan, package version, and pinned NumPy/SciPy must produce
  byte-identical artifacts on every supported Python version and operating system.
  - Text artifacts are written with LF line endings (#31).
  - Never let a `set` or emission order decide an output order; sort with an explicit, total key.
  - Input paths are recorded relative to the plan's directory, in POSIX form (#63).

### 4.4 Numerical behaviour

- Default methods, alpha allocation, coverage semantics (two-sided versus per-direction), the
  estimand, and the conservative completion of unknown outcomes.
- Report precision and rounding. A change in the last printed digit can flip a reader's
  conclusion.
- The scope statement printed in every report.

### 4.5 Command line

- **Commands and flags:** command and subcommand names, flags, and their defaults.
- **Exit codes of `analyze`:**
  - `0`: integrity is valid or degraded;
  - `3`: integrity is invalid (the report is still written), or the analysis could not run;
  - with `--decision-exit-codes`: `0` PASS, `1` FAIL, `2` INCONCLUSIVE.
- **Exit codes of `validate`:** `3` for invalid input.
- These codes gate CI pipelines. The table in `docs/api-reference.md` is the contract.
- **Machine-read output:** stdout lines that scripts parse, such as `Required rollouts: N`
  from `plan zero-failure`.

## 5. Engineering practices

### 5.1 Architecture and layering

- **`stats/` is the numeric core.** It takes arrays and plain values and returns frozen
  dataclasses. Adapt richer objects at the boundary (ingestion or the analysis pipeline), not
  inside `stats/`.
  - **[gap]** `stats/independence.py` still reads pydantic models and a pandas manifest.
- **Importing the package is cheap.**
  - The package root serves its public names on first use (PEP 562).
  - CLI commands import what they need inside the command.
  - Do not add module-level imports of the analysis pipeline, Inspect, or pandas to
    `inspect_safety_stats/__init__.py` or `cli.py`.
  - `tests/unit/test_package_imports.py` checks this in a fresh interpreter.
- **Presentation never computes statistics.** `report/` formats results that the analysis
  already computed. If the HTML or Markdown needs a number or a sentence about a method, add it
  to the report model first.
- **Method and measure names.** Dispatch on a method or measure name in one place, and define
  each `Literal[...]` of names once. A new method must work, or fail with a clear diagnostic,
  everywhere methods are accepted.
- **Typed data between stages.** Pipeline stages pass typed objects, not untyped
  `dict[str, Any]` and not tuples with more than two elements. Report provenance is not an
  internal data channel.
- **Order-dependent code.** Where reordering seemingly independent statements would change
  results, leave an `ORDER MATTERS:` comment that says what must come first and what breaks
  otherwise. Typical cases:
  - import side effects;
  - stages that mutate shared results;
  - lists that later stages read;
  - float accumulation order;
  - sorts that fall back on emission order.

  #66 tracks the existing sites.
- **No hidden state or printing.** Library code keeps no module-level mutable state and does not
  `print`. The CLI prints; the library reports through diagnostics and return values.

### 5.2 Errors, diagnostics, and validation

- **User input errors are diagnostics, never crashes.**
  - An unknown reference in a plan yields `Diagnostic(severity="ERROR", code=…)`, the report is
    written, and `analyze` exits 3.
- **Blocking must target the right prompt.** A diagnostic that should withhold a prompt's bounds
  names the cohort and prompt in `details`, as rollout-level ERRORs do (#40). Do not rely on
  prefix-matching free-form `scope` strings.
- **No `assert` for runtime validation.** Python strips asserts under `-O`; raise a specific
  exception instead. `test_the_package_has_no_runtime_asserts` checks the whole package.
- **Actionable messages.** Error messages name the plan path or manifest column at fault, and say
  how to fix it. For example, a missing `block_by` value names the column and the prompts (#38).
- **`validate` and `analyze` agree.** `validate` should accept exactly what `analyze` accepts. If
  you add a check to one, add it to the other.

### 5.3 Numerics

- **Conservative by default.** Unknown outcomes widen bounds. Never silently drop or impute them.
- **Accumulation order is behaviour.** Changing the order of a float sum is class B, because it
  can move the last bit.
- **Comparisons.** Compare computed floats with explicit tolerances. Use exact equality only
  where it is intended and commented.
- **Missing values.** Declared grouping columns must not depend on pandas dtype inference. Reject
  missing values or map them explicitly.
- **Randomness.** Use `numpy.random.default_rng(seed)`, with the seed recorded in the plan or the
  report. Never use the global random state.
- **Output formatting.** Write numbers through `canonical_value`, not ad-hoc `str()` or f-strings.

### 5.4 Performance and the Rust policy

- **Measure first.** Profile before optimising. A performance pull request is class A: attach the
  timings and an A/B showing byte-identical artifacts on realistic and stress-sized designs.
  `python scripts/compare_artifacts.py --prompts 2000` runs the bundled examples, a synthetic
  design of that size, and offline cases for the paths the examples do not reach (two cohorts and
  comparisons, every error stage, judge panels and calibration, sampled designs, an incomplete
  run, the generative design) against the committed code. Its timings are single runs, so
  measure performance separately.
- **Prefer algorithmic fixes in Python and NumPy:**
  - index once;
  - memoise pure functions;
  - group in one pass instead of masking per item;
  - hand SciPy the sparse form it builds internally anyway.
- **Compiled accelerators.** A compiled accelerator, for example Rust via PyO3, is acceptable
  only for a **stable** kernel: one that has not changed materially in recent releases and is
  not being redesigned on an open branch. It must meet all of the following:
  1. A pure-Python reference implementation stays the default fallback and serves as the oracle
     in differential tests.
  2. It ships as prebuilt wheels for Linux, macOS, and Windows, so users and other contributors
     never need a Rust toolchain.
  3. It matches the reference bit for bit, or within a tolerance agreed in an issue and pinned by
     tests. Goldens are always produced by the reference.
  4. CI runs both the pure-Python path and the accelerated path.
- **Code under active development stays in Python.** Contributor ergonomics outweigh marginal
  speedups.

### 5.5 Dependencies

- **Runtime dependencies.** A new runtime dependency needs maintainer agreement. Heavy or
  optional functionality goes in an extra.
- **The `inspect-ai` range.** `inspect-ai` is pinned to a tested range. The weekly CI run
  installs the newest allowed release. Widen the range only with a passing run.

### 5.6 Typing, style, and docs

- **Typing.** `mypy --strict` must pass. Do not add new `Any` to the core data flow, and prefer
  `Mapping` and `Sequence` in signatures. When third-party stubs are narrower than the runtime,
  annotate the value as `Any` with a comment saying why.
- **Docstrings.** Public functions have docstrings that state the assumptions, the guarantee
  (coverage and direction), and a reference.
- **Docs ship with the change.** A user-facing change updates the relevant guide in the same
  pull request, and `docs/plan-reference.md` for new plan fields.
- **Version.** The version lives in four places: `pyproject.toml`,
  `src/inspect_safety_stats/version.py`, `docs/independence-profiles.md`, and `CITATION.cff`.
  `tests/test_repository_consistency.py` fails if they disagree.

## 6. Linux, macOS, and Windows

All three are first-class platforms.

- **Line endings in outputs.** Write text files with `encoding="utf-8", newline="\n"`, and write
  pandas CSVs with `lineterminator="\n"`. Without them, Windows writes CRLF and the bytes and
  checksums differ by platform.
- **Line endings in the repository.** `.gitattributes` keeps text LF in every checkout and
  never converts CSV fixtures, whose bytes tests pin. A Windows checkout made before it was added
  may need `git checkout -- '*.csv'`.
- **Paths.**
  - Use `pathlib`; never build paths with string concatenation or a hard-coded separator.
  - Paths written into outputs are relative to a documented base, in POSIX form. Reports name
    their inputs relative to the plan's directory (`report_path` in `config/loader.py`).
- **Case-insensitive filesystems.** macOS and Windows filesystems are case-insensitive, so no two
  generated file names may differ only in case.
- **Path length.** Keep generated directory trees shallow: Windows' default path limit is 260
  characters.
- **Files in use.** Close files before renaming or deleting them, and use `tmp_path` in tests.
- **Console output.** CLI output must survive a non-UTF-8 console when redirected on Windows.
- **Processes.** Library code must not use fork-only multiprocessing, POSIX signals, or
  shell-specific subprocess calls.
- **Known upstream issue.** On Windows, inspect-ai cannot read logs that live on a different
  drive from the working directory (#35).
- **CI coverage:**
  - Linux runs on Python 3.11–3.13 for every pull request.
  - Windows runs on 3.12 for every pull request.
  - A push to `main` runs lint, types, the build and one Linux test job: the pull request
    already ran the full matrix on the merged tree.
  - macOS, and the full matrix, run weekly and on demand. For a platform-sensitive branch, run
    `gh workflow run CI --ref <branch>`.
  - Runner minutes on this private repository are billed to its owner. Push a branch when it is
    ready for review, not after every commit.

## 7. Tests

| Layer | Location | Purpose |
|---|---|---|
| Unit | `tests/unit/` | Single functions and their edge cases: n = 0, all failures, zero weights, extreme alpha, one prompt, one rollout |
| Property | `tests/property/` | Hypothesis invariants: monotonicity, bound ordering, coverage direction |
| Integration | `tests/integration/` | `analyze` and `validate` end to end on synthetic logs |
| Examples | `tests/test_*_example.py` | The bundled examples still run offline |
| Repository | `tests/test_repository_consistency.py` | Example plans validate, documentation links resolve, the version agrees in all three places, CI tests every advertised Python version, and only expected files sit at the repository root |
| Scripts | `tests/test_dev_scripts.py` | The developer scripts in `scripts/` do what they claim, on every platform |
| Golden **[proposed]** | `tests/golden/` | Byte-for-byte expected outputs for representative plans (#27) |

- **Bug fixes.** Every bug fix comes with a test that fails without it. Check it with
  `python scripts/prove_red.py <tests>`, which runs the tests against the committed source and
  then against your working tree, and paste its verdict in the pull request. Pass
  `--base <commit>` once the fix is committed. It never modifies the working tree.
- **Bounds.** A new bound or estimator needs a seeded Monte Carlo coverage test showing empirical
  coverage at or above nominal, with the simulation error stated. It also needs exact-value
  tests against an independent computation.
- **Golden outputs.** Regenerate them only with the regeneration script, in a dedicated commit.
  Class A pull requests never touch them.
- **Network.** Tests never call a model provider or the network. Live examples skip cleanly
  without credentials.

## 8. Statistical contributions

1. **Issue.** Open an issue that states:
   - the estimand;
   - the assumptions;
   - the guarantee in one sentence, for example "under independent blocks,
     P(true risk ≤ U) ≥ 1 − α simultaneously over all declared nodes";
   - a citation or proof sketch.
2. **Docs.** Add the derivation to `docs/statistical-methods.md`, and a user-level explanation
   to the relevant guide.
3. **Coverage of methods.** Register the method or measure wherever methods are accepted:
   direct means, comparisons, hierarchy projection, judge correction, planning. It must work, or
   fail with a clear diagnostic, in each of them.
4. **Tests.** Write the tests described in section 7.
5. **Wording.** The report must state *this* method's assumptions. Falling back to another
   method's text is a bug (#36).
6. **Review.** At least one reviewer with the relevant statistical background approves.

## 9. Data, secrets, and sensitive content

- **Harmful content.** Safety benchmarks contain harmful prompts and completions. Commit only
  what is already public in the source benchmark and needed for an example or a test. Never
  commit raw completions from live runs; `run/`, `live-runs/` and `logs/` are ignored for this
  reason.
- **Third-party data.** Third-party data needs a license that allows redistribution. Record its
  source, version, and license in the example's README.
- **Secrets.** API keys come only from environment variables or an ignored `.env`. Check `.eval`
  fixtures for leaked headers or keys before committing them.
- **Bug reports.** Do not attach full logs or reports from private evaluations. Reduce them to a
  minimal reproduction, and redact local paths and usernames.

## 10. Pull requests and review

- **Description.** A pull request names its change class and linked issue (`Closes #N`), how it
  was verified (commands, and on which operating system), and its effect on golden outputs and
  docs.
- **Review.** Changes to `stats/`, the plan or report schema, defaults, or exit codes need review
  by a maintainer.
- **CI.** CI is green before merge.
- **Merging.** Maintainers merge. A maintainer merges a contributor's pull request after review
  and green CI; contributors do not merge their own pull requests, whatever their repository
  permissions. Do not push directly to `main`.
- **AI-assisted contributions** are welcome under the same bar. The human author is accountable
  for every line, can explain it in review, and keeps the pull request small enough to review.
  Point coding agents at [AGENTS.md](AGENTS.md). It lists the checks to run and the mistakes this
  repository has already made.
- **[proposed]** Sign off your commits (`git commit -s`, Developer Certificate of Origin) with the
  identity you contribute under.

## 11. Commits and branches

- **Subjects.** Write commit subjects in the imperative and in sentence case, saying what changes
  and why it matters. For example: "Correct comparisons per cohort, and fix the judge alpha
  allocation". Use the body for the reasoning and the evidence.
- **One logical change per commit.** Do not mix moves and edits in one commit, so reviewers can
  confirm a move is unchanged text.
- **Branches.** `main` is the only long-lived branch. Feature branches are short-lived and kept
  close to `main`.

## 12. Releases and deprecation [proposed]

- **Versioning.** Bump the minor version for class B and C changes, and the patch version for
  class A changes and fixes that do not change output.
- **What a release includes.** A git tag, a changelog section, and a GitHub release. It is
  published to PyPI through trusted publishing once that exists (#20).
- **Deprecation.** Removing or renaming a protected interface takes at least one minor release.
  During it, the old interface keeps working and warns with a `DeprecationWarning` or a plan
  diagnostic that names the replacement. The migration is documented in the changelog.
- **Changed results.** A release never silently changes the result of an unchanged plan. When it
  does change results, the changelog says so under "Changed results".
