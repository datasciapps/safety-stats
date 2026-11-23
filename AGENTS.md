# Notes for coding agents

[CONTRIBUTING.md](CONTRIBUTING.md) is the contract for every change, whoever or whatever writes
it. This file is the short version to read first: the checks to run before you finish, and the
mistakes that have already broken this repository once, each with what now catches it. It
applies to people too.

## Before you finish

- **Change class.** Decide whether the change is class A, B or C (CONTRIBUTING section 3)
  before writing code. Anything that changes an output byte is class B, a bug fix included.
- **Gate.** Run all three:

  ```bash
  ruff check src tests examples research scripts
  mypy src/safety_stats
  pytest
  ```

  Run `ruff check` only. `ruff format` rewrites hundreds of untouched lines.
- **Evidence.** Paste the verdict into the pull request.
  - Bug fix: `python scripts/prove_red.py <tests>` must report `PROVEN`. A test that passes
    without the fix proves nothing. For a fix outside `src/`, such as in `examples/` or
    `scripts/`, add `--tree`.
  - Class A: `python scripts/compare_artifacts.py` must report `IDENTICAL`.
  - Both scripts compare against `HEAD` by default. Once the change is committed, pass
    `--base <commit>`.

## Mistakes that have happened here

| Mistake | What catches it now | Rule |
|---|---|---|
| `pytest.approx(3e-17, rel=1e-9)` accepted 0.0, so a precision test passed against the bug it was written for | `test_approx_tolerances_cannot_make_small_comparisons_vacuous` | `approx` adds `abs=1e-12` unless told otherwise. Pass `abs=` whenever you pass `rel=` or compare small values, and `abs=0` for a purely relative check. |
| A CLI test passed locally and failed on CI, where typer colours `--weights` into `-` and `-weights` | `tests/conftest.py`, which renders the CI way in every run | Assert on `tests.cli_output.plain(result.output)`, not on the raw output. |
| Report bytes and checksums differed on Windows, because text was written with CRLF | `test_text_artifacts_have_the_same_bytes_on_every_platform`, on Windows CI | `newline="\n"` for text files, `lineterminator="\n"` for pandas CSVs (CONTRIBUTING section 6). |
| Tests reading real logs failed on Windows when the temporary directory and the checkout were on different drives | the autouse fixture in `tests/integration/test_generative_real_logs.py` | Inspect drops the drive letter of such logs (#35). Run from the drive that holds them. |
| `k: true` in a plan was accepted as `k = 1` | `test_plans_reject_a_boolean_k` | Pydantic's lax mode turns booleans into integers. Reject them in a `mode="before"` validator. |
| `1 - (1 - p) ** k` returned 0 for tiny `p` | `test_worst_of_k_stays_accurate_for_tiny_probabilities` | Use `-expm1(k * log1p(-p))` and similar forms, and test tiny inputs with `abs=0`. |
| A missing `block_by` value changed how prompts were blocked, depending on pandas' dtype inference | `test_missing_manifest_block_values_are_rejected` | Reject missing values in declared grouping columns. Never let dtype inference decide. |
| `import safety_stats` took seconds | `tests/unit/test_package_imports.py` | No module-level imports of pandas, Inspect or the pipeline in `__init__.py` or `cli.py` (CONTRIBUTING section 5.1). |

## Working in the tree

- **Order-dependent code.** Search for `ORDER MATTERS` before reordering statements that look
  independent. When you create such a dependency, leave a note that says what must come first
  and what breaks otherwise.
- **Expected outputs and fixtures.** Never edit expected outputs or CSV fixtures to make a test
  pass.
  - Class A changes never touch them.
  - Class B changes regenerate them in a separate commit that explains every difference.
- **The committed version of a file.** Never run `git checkout -- <path>` or `git stash` just to
  see or test the committed code. Either can discard uncommitted work, and once lost a fix that
  had to be redone. Both scripts above export committed source to a temporary directory and
  leave the working tree alone.
- **Test helpers.** Shared helpers live in modules without a `test_` prefix, such as
  `tests/cli_output.py` and `tests/generative_fixtures.py`. Do not import one test module from
  another.
- **Network.** Tests never call a model provider or the network.
- **Platforms.** Linux, macOS and Windows are all first-class (CONTRIBUTING section 6). Pull
  requests run on Linux and Windows. macOS runs weekly and on demand.
- **CI minutes.** Runner minutes on this private repository are billed to its owner. Run the
  gate locally, and push a branch when it is ready for review, not after every commit.

## Commits and pull requests

- **Commits.** One logical change per commit. Write the subject in the imperative and in
  sentence case, and put the reasoning and evidence in the body. Do not mix moves and edits.
- **One concern per pull request.** Class B and C changes need an issue that a maintainer
  agreed to before merge (CONTRIBUTING section 3).
- **Merging.** Contributors do not merge their own pull requests. Do not push directly to
  `main`.
- **Accountability.** AI-assisted contributions meet the same bar. The human author is
  accountable for every line (CONTRIBUTING section 10).
